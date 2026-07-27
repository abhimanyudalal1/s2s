"""
SPATIAL correlation maps: ECMWF vs UNet, both against IMD, per cell, per window.

The India-mean threeline plot averages away all spatial structure -- it tells
you nothing about WHERE each model works. This computes per-cell anomaly
correlation over the test forecasts and maps it.

Panels per window:
    1. ECMWF vs IMD      -- raw forecast skill, per cell
    2. UNet  vs IMD      -- your model, per cell
    3. UNet - ECMWF      -- red where UNet wins, blue where ECMWF wins
    4. IMD anomaly std   -- observed variability, for context (not a corr map)

NOTE: IMD has no correlation panel of its own -- it IS the truth the other
two are scored against. Panel 4 shows its variability instead, which is what
you actually want alongside: high-corr cells in a low-variance region mean
something different from high-corr in the monsoon core.

Writes results/models/corr_maps.nc (open in Panoply) and
results/figures/corr_maps.png. Run from the repo root.

  python scripts/s2s_corr_maps.py
  python scripts/s2s_corr_maps.py --ckpt results/models/unet_mw_mae.pt --out corr_maps_mae
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


def corr_map(p, t, fin):
    """Per-cell Pearson correlation over the sample axis. p,t: (n,H,W)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = np.nanmean(np.where(fin, p, np.nan), axis=0)
        tm = np.nanmean(np.where(fin, t, np.nan), axis=0)
        num = np.nansum(np.where(fin, (p - pm) * (t - tm), np.nan), axis=0)
        den = np.sqrt(np.nansum(np.where(fin, (p - pm) ** 2, np.nan), axis=0)
                      * np.nansum(np.where(fin, (t - tm) ** 2, np.nan), axis=0))
        return np.where(den > 0, num / den, np.nan)


def unet_predict(args, z, X, y, doy, wid, fit_i, clat, clon, flat_lat, flat_lon, mask):
    import torch, torch.nn as nn, torch.nn.functional as F
    H, W = len(flat_lat), len(flat_lon)
    n_var, n_win = X.shape[1], len(z["window_names"])
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")

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
    a = ckpt.get("args", {}) or {}
    base, drop = a.get("base", 24), a.get("drop", 0.2)
    print(f"    checkpoint: base={base} drop={drop} loss={a.get('loss','?')}")

    def gn(c):
        for g in (8, 4, 2, 1):
            if c % g == 0:
                return nn.GroupNorm(g, c)

    class Block(nn.Module):
        def __init__(s, ci, co, dr=0.0):
            super().__init__()
            L = [nn.Conv2d(ci, co, 3, padding=1), gn(co), nn.SiLU(),
                 nn.Conv2d(co, co, 3, padding=1), gn(co), nn.SiLU()]
            if dr > 0:
                L.append(nn.Dropout2d(dr))
            s.f = nn.Sequential(*L)
        def forward(s, x): return s.f(x)

    class MWUNet(nn.Module):
        def __init__(s, n_var, n_win, base=24, drop=0.2, emb=4):
            super().__init__()
            s.emb = nn.Embedding(n_win, emb)
            s.enc_c1 = Block(n_var + emb, base*2); s.enc_c2 = Block(base*2, base*2)
            s.inp = Block(base*2+3, base)
            s.d1 = Block(base, base*2, drop); s.d2 = Block(base*2, base*4, drop)
            s.bott = Block(base*4, base*4, drop)
            s.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2); s.du2 = Block(base*4, base*2, drop)
            s.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2); s.du1 = Block(base*2, base)
            s.head = nn.Conv2d(base, 1, 1); s.pool = nn.MaxPool2d(2)
        def forward(s, xc, wid, samp, static):
            b, _, h, w = xc.shape
            e = s.emb(wid)[:, :, None, None].expand(-1, -1, h, w)
            c = s.enc_c2(s.enc_c1(torch.cat([xc, e], 1)))
            f = F.grid_sample(c, samp.expand(b, -1, -1, -1), mode="bilinear", align_corners=True)
            f = torch.cat([f, static.expand(b, -1, -1, -1)], 1)
            H0, W0 = f.shape[-2:]
            f = F.pad(f, (0, (-W0) % 4, 0, (-H0) % 4), mode="replicate")
            e0 = s.inp(f); e1 = s.d1(s.pool(e0)); e2 = s.d2(s.pool(e1))
            u = s.du2(torch.cat([s.u2(s.bott(e2)), e1], 1))
            u = s.du1(torch.cat([s.u1(u), e0], 1))
            return s.head(u)[:, :, :H0, :W0].squeeze(1)

    model = MWUNet(n_var, n_win, base=base, drop=drop).to(dev)
    model.load_state_dict(ckpt["state"]); model.eval()
    Xt, widt = torch.tensor(Xa), torch.tensor(wid)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            j = slice(i, i + args.batch)
            out.append(model(Xt[j].to(dev), widt[j].to(dev), samp, static).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/unet_cache_mw.npz")
    ap.add_argument("--ckpt", default="results/models/grad/unet_mw_grad.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="corr_maps")
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=False)
    X, y, doy, wid = z["X"], z["y"], z["doy"], z["wid"]
    mask, is_test = z["mask"], z["is_test"]
    clat, clon = z["clat"], z["clon"]
    flat_lat, flat_lon = z["flat_lat"], z["flat_lon"]
    feature_vars = [str(v) for v in z["feature_vars"]]
    window_names = [str(w) for w in z["window_names"]]
    n_win = len(window_names)
    tp_idx = feature_vars.index(RAW_PRECIP_VAR)
    fit_i = ~is_test

    # raw ECMWF precip interpolated to the IMD grid
    import xarray as xr
    tp_fine = xr.DataArray(X[:, tp_idx], dims=("s", "lat", "lon"),
                           coords={"lat": clat, "lon": clon}
                           ).interp(lat=flat_lat, lon=flat_lon).values.astype(DTYPE)

    # anomalies: each series minus its OWN train climatology (units cancel)
    obs_a = y - _clim(y[fit_i], doy[fit_i])[doy - 1]
    ecm_a = tp_fine - _clim(tp_fine[fit_i], doy[fit_i])[doy - 1]
    unet_a = unet_predict(args, z, X, y, doy, wid, fit_i,
                          clat, clon, flat_lat, flat_lon, mask)

    m3 = mask[None]
    dv = {}
    stats = []
    for w in range(n_win):
        sel = (wid == w) & is_test
        fin = np.isfinite(obs_a[sel]) & m3
        c_e = corr_map(ecm_a[sel], obs_a[sel], fin)
        c_u = corr_map(unet_a[sel], obs_a[sel], fin)
        obs_sd = np.nanstd(np.where(fin, obs_a[sel], np.nan), axis=0)
        c_e[~mask] = np.nan; c_u[~mask] = np.nan; obs_sd[~mask] = np.nan
        wn = window_names[w]
        dv[f"{wn}_corr_ecmwf"] = (("lat", "lon"), c_e)
        dv[f"{wn}_corr_unet"] = (("lat", "lon"), c_u)
        dv[f"{wn}_corr_diff"] = (("lat", "lon"), c_u - c_e)
        dv[f"{wn}_obs_std"] = (("lat", "lon"), obs_sd)
        stats.append((wn, np.nanmean(c_e[mask]), np.nanmean(c_u[mask]),
                      100 * np.nanmean((c_u - c_e)[mask] > 0)))

    out = xr.Dataset(dv, coords={"lat": flat_lat, "lon": flat_lon})
    out.attrs["note"] = ("per-cell anomaly correlation over test forecasts; "
                         "corr_diff = unet - ecmwf (>0 means UNet better)")
    out.to_netcdf(f"results/models/{args.out}.nc")

    print(f"\n{'window':>8} {'ECMWF r':>9} {'UNet r':>9} {'UNet better':>12}")
    for wn, ce, cu, pct in stats:
        print(f"{wn:>8} {ce:>9.3f} {cu:>9.3f} {pct:>11.0f}%")

    # ---- figure: rows = windows, cols = ecmwf / unet / diff / obs std ----
    fig, axes = plt.subplots(n_win, 4, figsize=(19, 4.4 * n_win))
    axes = np.atleast_2d(axes)
    ext = [flat_lon[0], flat_lon[-1], flat_lat[0], flat_lat[-1]]
    for w in range(n_win):
        wn = window_names[w]
        panels = [
            (f"{wn}_corr_ecmwf", "ECMWF vs IMD", "RdBu_r", -0.6, 0.6),
            (f"{wn}_corr_unet", "UNet vs IMD", "RdBu_r", -0.6, 0.6),
            (f"{wn}_corr_diff", "UNet - ECMWF", "PuOr_r", -0.4, 0.4),
            (f"{wn}_obs_std", "IMD anomaly std (mm/day)", "viridis", None, None),
        ]
        for c, (key, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[w, c]
            arr = out[key].values
            kw = dict(cmap=cmap, origin="lower", extent=ext, aspect="auto")
            if vmin is not None:
                kw.update(vmin=vmin, vmax=vmax)
            im = ax.imshow(arr, **kw)
            fig.colorbar(im, ax=ax, fraction=0.046)
            mean_v = np.nanmean(arr[mask])
            ax.set_title(f"{wn}  {title}\n(mean {mean_v:+.3f})", fontsize=10)
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(f"results/figures/{args.out}.png", dpi=130)
    print(f"\n-> results/models/{args.out}.nc (Panoply), results/figures/{args.out}.png")
    print("   corr_diff: orange = UNet better, purple = ECMWF better")


if __name__ == "__main__":
    main()
