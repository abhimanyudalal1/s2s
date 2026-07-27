"""
Three-line comparison per window: ECMWF forecast vs UNet vs observed IMD.

Plots the India-mean anomaly time series over the test years, one line each:
    observed IMD, raw ECMWF, UNet.

Anomalies (each minus its own seasonal climatology), because raw ECMWF tp and
IMD rain are in different units -- anomalies put all three on one axis.

Reads the same cache + checkpoint. No retraining.
  python s2s_threeline.py
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DTYPE = np.float32
CLIM_WINDOW_DAYS = 7
RAW_PRECIP_VAR = "total_precipitation"


def _clim(values, doys, window=CLIM_WINDOW_DAYS, n_doy=366):
    shp = values.shape[1:]
    v2 = values.reshape(len(values), -1)
    centers = np.arange(1, n_doy + 1)
    d = np.abs(doys[None, :].astype(int) - centers[:, None])
    M = (np.minimum(d, n_doy - d) <= window).astype(DTYPE)
    counts = M @ np.isfinite(v2).astype(DTYPE)
    sums = M @ np.nan_to_num(v2).astype(DTYPE)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return c.reshape((n_doy,) + shp).astype(DTYPE)


def unet_predict(args, z, X, y, doy, wid, fit_i, clat, clon, flat_lat, flat_lon, mask):
    import torch, torch.nn as nn, torch.nn.functional as F
    H, W = len(flat_lat), len(flat_lon)
    n_var, n_win = X.shape[1], len(z["window_names"])
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")

    clim_X = _clim(X[fit_i], doy[fit_i])
    Xa = X - clim_X[doy - 1]
    xm = np.nanmean(Xa[fit_i], (0, 2, 3), keepdims=True)
    xs = np.nanstd(Xa[fit_i], (0, 2, 3), keepdims=True)
    Xa = np.nan_to_num((Xa - xm) / np.where(xs < 1e-8, 1.0, xs)).astype(DTYPE)

    gy = 2 * (flat_lat - clat[0]) / (clat[-1] - clat[0]) - 1
    gx = 2 * (flat_lon - clon[0]) / (clon[-1] - clon[0]) - 1
    gyy, gxx = np.meshgrid(gy, gx, indexing="ij")
    samp = torch.tensor(np.stack([gxx, gyy], -1).astype(np.float32)[None]).to(dev)
    lat2 = (flat_lat[:, None] - flat_lat.mean()) / flat_lat.std()
    lon2 = (flat_lon[None, :] - flat_lon.mean()) / flat_lon.std()
    static = torch.tensor(np.stack([mask.astype(DTYPE),
        np.broadcast_to(lat2, (H, W)).astype(DTYPE),
        np.broadcast_to(lon2, (H, W)).astype(DTYPE)])[None]).to(dev)

    ckpt = torch.load(args.ckpt, map_location=dev)
    base, drop = ckpt["args"].get("base", 24), ckpt["args"].get("drop", 0.2)

    def gn(c):
        for g in (8, 4, 2, 1):
            if c % g == 0:
                return nn.GroupNorm(g, c)

    class Block(nn.Module):
        def __init__(self, ci, co, drop=0.0):
            super().__init__()
            L = [nn.Conv2d(ci, co, 3, padding=1), gn(co), nn.SiLU(),
                 nn.Conv2d(co, co, 3, padding=1), gn(co), nn.SiLU()]
            if drop > 0:
                L.append(nn.Dropout2d(drop))
            self.f = nn.Sequential(*L)
        def forward(self, x):
            return self.f(x)

    class MWUNet(nn.Module):
        def __init__(self, n_var, n_win, base=24, drop=0.2, emb=4):
            super().__init__()
            self.emb = nn.Embedding(n_win, emb)
            self.enc_c1 = Block(n_var + emb, base * 2)
            self.enc_c2 = Block(base * 2, base * 2)
            self.inp = Block(base * 2 + 3, base)
            self.d1 = Block(base, base * 2, drop)
            self.d2 = Block(base * 2, base * 4, drop)
            self.bott = Block(base * 4, base * 4, drop)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.du2 = Block(base * 4, base * 2, drop)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.du1 = Block(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)
            self.pool = nn.MaxPool2d(2)
        def forward(self, xc, wid, samp, static):
            b, _, h, w = xc.shape
            e = self.emb(wid)[:, :, None, None].expand(-1, -1, h, w)
            c = self.enc_c2(self.enc_c1(torch.cat([xc, e], 1)))
            f = F.grid_sample(c, samp.expand(b, -1, -1, -1),
                              mode="bilinear", align_corners=True)
            f = torch.cat([f, static.expand(b, -1, -1, -1)], 1)
            H0, W0 = f.shape[-2:]
            f = F.pad(f, (0, (-W0) % 4, 0, (-H0) % 4), mode="replicate")
            e0 = self.inp(f); e1 = self.d1(self.pool(e0)); e2 = self.d2(self.pool(e1))
            u = self.du2(torch.cat([self.u2(self.bott(e2)), e1], 1))
            u = self.du1(torch.cat([self.u1(u), e0], 1))
            return self.head(u)[:, :, :H0, :W0].squeeze(1)

    model = MWUNet(n_var, n_win, base=base, drop=drop).to(dev)
    model.load_state_dict(ckpt["state"]); model.eval()
    Xt, widt = torch.tensor(Xa), torch.tensor(wid)
    out = []
    with torch.no_grad():
        for a in range(0, len(X), args.batch):
            j = slice(a, a + args.batch)
            out.append(model(Xt[j].to(dev), widt[j].to(dev), samp, static).cpu().numpy())
    return np.concatenate(out)


def main():
    class Args:
        cache = "data/cache/unet_cache_mw.npz"
        ckpt = "results/models/unet_mw_mae.pt"   # Pointing to the new model checkpoint
        batch = 16
        out = "results/figures/threeline_mae.png"
    args = Args()


    z = np.load(args.cache, allow_pickle=False)
    X, y, doy, wid = z["X"], z["y"], z["doy"], z["wid"]
    mask, is_test, year = z["mask"], z["is_test"], z["year"]
    clat, clon = z["clat"], z["clon"]
    flat_lat, flat_lon = z["flat_lat"], z["flat_lon"]
    feature_vars = [str(v) for v in z["feature_vars"]]
    window_names = [str(w) for w in z["window_names"]]
    n_win = len(window_names)
    tp_idx = feature_vars.index(RAW_PRECIP_VAR)
    fit_i = ~is_test

    # raw ECMWF tp on the fine grid
    import xarray as xr
    tp_fine = xr.DataArray(X[:, tp_idx], dims=("s", "lat", "lon"),
                           coords={"lat": clat, "lon": clon}
                           ).interp(lat=flat_lat, lon=flat_lon).values.astype(DTYPE)

    # anomalies, each vs its own train climatology
    obs_a = y - _clim(y[fit_i], doy[fit_i])[doy - 1]
    ecm_a = tp_fine - _clim(tp_fine[fit_i], doy[fit_i])[doy - 1]
    unet_a = unet_predict(args, z, X, y, doy, wid, fit_i,
                          clat, clon, flat_lat, flat_lon, mask)

    # India-mean over valid cells, test samples only
    m3 = mask[None]
    def india_mean(a):
        return np.nansum(np.where(m3, a, np.nan) * m3, axis=(1, 2)) / m3.sum()

    fig, axes = plt.subplots(n_win, 1, figsize=(13, 3.2 * n_win), sharex=False)
    axes = np.atleast_1d(axes)
    for w in range(n_win):
        sel = (wid == w) & is_test
        order = np.argsort(np.arange(len(sel))[sel])   # keep chronological-ish
        idx = np.where(sel)[0]
        ax = axes[w]
        t = np.arange(len(idx))
        ax.plot(t, india_mean(obs_a)[idx], "-", color="black", lw=2, label="observed IMD")
        ax.plot(t, india_mean(ecm_a)[idx], "-", color="#888", lw=1.5, label="raw ECMWF")
        ax.plot(t, india_mean(unet_a)[idx], "-", color="#E45756", lw=1.5, label="UNet")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        r_u = np.corrcoef(india_mean(unet_a)[idx], india_mean(obs_a)[idx])[0, 1]
        r_e = np.corrcoef(india_mean(ecm_a)[idx], india_mean(obs_a)[idx])[0, 1]
        ax.set_title(f"{window_names[w]}  (India-mean anomaly; "
                     f"corr vs obs: UNet {r_u:.2f}, ECMWF {r_e:.2f})")
        ax.set_ylabel("anomaly (mm/day)")
        if w == 0:
            ax.legend(ncol=3, loc="upper right", fontsize=9)
    axes[-1].set_xlabel("test-set forecast index (chronological within window)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
