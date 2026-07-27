"""
Three-way skill comparison for the multi-window UNet:

    (1) raw ECMWF        -- the coarse tp forecast, regridded, NO learning
    (2) quantile mapping -- classic statistical downscaling (the D-b baseline)
    (3) UNet             -- your trained model

all scored against IMD observed, per window, in anomaly space, over India's
valid cells. Produces the skill-vs-window curve and RMSE-vs-window curve that
answer the question climatology-skill cannot: DID DOWNSCALING BEAT THE RAW
FORECAST YOU STARTED WITH?

Why this matters more than skill-vs-climatology
-----------------------------------------------
Beating climatology only proves you have *some* signal. Beating raw ECMWF
proves your model added value over the forecast it was handed. If the UNet
curve sits on top of the raw-ECMWF curve, the network is just reproducing the
forecast and earns nothing -- you want to know that. This is the D-a/D-b
comparison from the original plan; you need it regardless.

Everything is scored in ANOMALY space (each series minus its own train-year
climatology). This is not optional here: raw ECMWF tp and IMD rain are in
different units / have different means, and anomalies cancel that offset so
the three are compared on equal footing. Skill = 1 - rmse/rmse_clim, where
rmse_clim is the error of predicting zero anomaly (i.e. climatology). So a
curve above 0 beats climatology; the GAP between curves is what downscaling
and the UNet each add.

Reads the SAME cache and the .pt checkpoint the training wrote. Run AFTER
`train`. Does not retrain.

USAGE
  Run from the repo root.
  python scripts/only_mse_script.py             # uses default cache + checkpoint
  python scripts/only_mse_script.py --cache data/cache/unet_cache_mw.npz \
      --ckpt results/models/unet_skill_maps_mw.pt
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DTYPE = np.float32
CLIM_WINDOW_DAYS = 7
RAW_PRECIP_VAR = "total_precipitation"   # ECMWF predictor used as the raw forecast


# ---- climatology (same matmul trick as training) ----

def _doy_matrix(doys, window, n_doy=366):
    centers = np.arange(1, n_doy + 1)
    d = np.abs(doys[None, :].astype(int) - centers[:, None])
    return (np.minimum(d, n_doy - d) <= window).astype(DTYPE)


def _clim(values, doys, window):
    """(n, ...) -> (366, ...) NaN-aware DOY climatology."""
    shp = values.shape[1:]
    v2 = values.reshape(len(values), -1)
    M = _doy_matrix(doys, window)
    counts = M @ np.isfinite(v2).astype(DTYPE)
    sums = M @ np.nan_to_num(v2).astype(DTYPE)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return c.reshape((366,) + shp).astype(DTYPE)


def skill_acc(p, t, fin):
    """Per-cell skill vs zero-anomaly climatology, and ACC. p,t: (n,H,W)."""
    se_m = np.where(fin, (t - p) ** 2, np.nan)
    se_c = np.where(fin, t ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        rmse_m = np.sqrt(np.nanmean(se_m, axis=0))
        rmse_c = np.sqrt(np.nanmean(se_c, axis=0))
        ok = rmse_c > 1e-6
        skill = np.where(ok, 1 - rmse_m / np.where(ok, rmse_c, 1), np.nan)
        tm = np.nanmean(np.where(fin, t, np.nan), axis=0)
        pm = np.nanmean(np.where(fin, p, np.nan), axis=0)
        num = np.nansum(np.where(fin, (t - tm) * (p - pm), np.nan), axis=0)
        den = np.sqrt(np.nansum(np.where(fin, (t - tm) ** 2, np.nan), axis=0)
                      * np.nansum(np.where(fin, (p - pm) ** 2, np.nan), axis=0))
        acc = np.where(den > 0, num / den, np.nan)
    return skill, acc, rmse_m


def quantile_map(raw_tr, obs_tr, raw_ev, nq=100):
    """Per-cell empirical quantile mapping (the classic D-b downscaler).

    Map raw ECMWF tp onto the IMD distribution cell by cell, trained on train
    years. Operates on the fine-grid, coarsely-interpolated raw field vs IMD.
    Returns mapped raw_ev in observation units.
    """
    n, H, W = raw_ev.shape
    out = np.empty_like(raw_ev)
    qs = np.linspace(0, 100, nq)
    rt = raw_tr.reshape(len(raw_tr), -1)
    ot = obs_tr.reshape(len(obs_tr), -1)
    re = raw_ev.reshape(n, -1)
    om = out.reshape(n, -1)
    for c in range(rt.shape[1]):
        rc, oc = rt[:, c], ot[:, c]
        rc = rc[np.isfinite(rc)]
        oc = oc[np.isfinite(oc)]
        if len(rc) < 10 or len(oc) < 10:
            om[:, c] = re[:, c]
            continue
        rq = np.percentile(rc, qs)
        oq = np.percentile(oc, qs)
        om[:, c] = np.interp(re[:, c], rq, oq)
    return out


def main():
    # Jupyter Notebook equivalent of command-line args
    class Args:
        cache = "data/cache/unet_cache_mw.npz"
        ckpt = "results/models/unet_skill_maps_mw.pt"
        batch = 16
        out = "results/figures/baseline_comparison"
    args = Args()

    z = np.load(args.cache, allow_pickle=False)
    X, y, doy, wid = z["X"], z["y"], z["doy"], z["wid"]
    year, mask, is_test = z["year"], z["mask"], z["is_test"]
    clat, clon = z["clat"], z["clon"]
    flat_lat, flat_lon = z["flat_lat"], z["flat_lon"]
    feature_vars = [str(v) for v in z["feature_vars"]]
    window_names = [str(w) for w in z["window_names"]]
    H, W = len(flat_lat), len(flat_lon)
    n_win = len(window_names)

    if RAW_PRECIP_VAR not in feature_vars:
        raise SystemExit(f"'{RAW_PRECIP_VAR}' not in features {feature_vars}; "
                         "set RAW_PRECIP_VAR to your ECMWF precip variable name")
    tp_idx = feature_vars.index(RAW_PRECIP_VAR)

    # same train/test logic as training: test years fixed, rest = train for
    # the FINAL model. We evaluate all three on the TEST years.
    fit_i = ~is_test
    te_i = is_test
    print(f"train {fit_i.sum()} | test {te_i.sum()} samples "
          f"({len(np.unique(year[te_i]))} test years)")

    # --- raw ECMWF precip on the FINE grid ---
    # X is coarse (n, V, h, w). Interpolate the tp channel to the fine grid so
    # it is comparable to IMD. Coordinate-aware bilinear via the same mapping
    # the UNet's grid_sample uses.
    import xarray as xr
    tp_coarse = xr.DataArray(
        X[:, tp_idx], dims=("s", "lat", "lon"),
        coords={"lat": clat, "lon": clon})
    tp_fine = tp_coarse.interp(lat=flat_lat, lon=flat_lon).values.astype(DTYPE)
    # (n, H, W) raw forecast in ECMWF units, on IMD grid

    # --- climatologies on TRAIN years only ---
    clim_obs = _clim(y[fit_i], doy[fit_i], CLIM_WINDOW_DAYS)     # (366,H,W)
    clim_raw = _clim(tp_fine[fit_i], doy[fit_i], CLIM_WINDOW_DAYS)

    y_te = y[te_i]
    fin = np.isfinite(y_te) & mask[None]
    obs_anom = y_te - clim_obs[doy[te_i] - 1]

    # === (1) RAW ECMWF: anomaly = raw - raw_clim. units cancel. ===
    raw_anom = tp_fine[te_i] - clim_raw[doy[te_i] - 1]

    # === (2) QUANTILE MAPPING: map raw->obs dist on train, then anomalise ===
    qm_ev = quantile_map(tp_fine[fit_i], y[fit_i], tp_fine[te_i])
    qm_anom = qm_ev - clim_obs[doy[te_i] - 1]     # mapped into obs units already

    # === (3) UNet: reload checkpoint, redo fold-local anomalisation, predict ===
    unet_anom = run_unet(args, z, X, y, doy, wid, fit_i, te_i,
                         clat, clon, flat_lat, flat_lon, mask)

    # --- score all three per window ---
    print(f"\n{'window':>8} | {'raw ECMWF':>20} | {'quantile map':>20} | "
          f"{'UNet':>20}")
    print(f"{'':>8} | {'skill':>9} {'ACC':>9} | {'skill':>9} {'ACC':>9} | "
          f"{'skill':>9} {'ACC':>9}")
    curves = {"raw": [], "qm": [], "unet": []}
    accs = {"raw": [], "qm": [], "unet": []}
    wid_te = wid[te_i]
    for w in range(n_win):
        m = wid_te == w
        if m.sum() == 0:
            continue
        row = []
        for name, arr in (("raw", raw_anom), ("qm", qm_anom), ("unet", unet_anom)):
            sk, ac, _ = skill_acc(arr[m], obs_anom[m], fin[m])
            ms, ma = np.nanmean(sk[mask]), np.nanmean(ac[mask])
            curves[name].append(ms)
            accs[name].append(ma)
            row += [ms, ma]
        print(f"{window_names[w]:>8} | {row[0]:>+9.3f} {row[1]:>9.3f} | "
              f"{row[2]:>+9.3f} {row[3]:>9.3f} | {row[4]:>+9.3f} {row[5]:>9.3f}")

    plot_curves(window_names, curves, accs, args.out)
    print(f"\n-> {args.out}_skill.png, {args.out}_acc.png")
    interpret(window_names, curves)


def run_unet(args, z, X, y, doy, wid, fit_i, te_i,
             clat, clon, flat_lat, flat_lon, mask):
    """Reload the trained UNet and predict test-set anomalies.

    Repeats the training's fold-local anomalisation using the FINAL model's
    fit years (all non-test), so the standardisation matches what the
    checkpoint was trained with.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    H, W = len(flat_lat), len(flat_lon)
    n_var = X.shape[1]
    n_win = len(z["window_names"])
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")

    # fold-local anomalise + standardise on fit years (mirrors anomalise_fold)
    clim_y = _clim(y[fit_i], doy[fit_i], CLIM_WINDOW_DAYS)
    clim_X = _clim(X[fit_i], doy[fit_i], CLIM_WINDOW_DAYS)
    Xa = X - clim_X[doy - 1]
    xm = np.nanmean(Xa[fit_i], axis=(0, 2, 3), keepdims=True)
    xs = np.nanstd(Xa[fit_i], axis=(0, 2, 3), keepdims=True)
    xs = np.where(xs < 1e-8, 1.0, xs)
    Xa = np.nan_to_num((Xa - xm) / xs).astype(DTYPE)

    # rebuild grid + static exactly as training did
    gy = 2 * (flat_lat - clat[0]) / (clat[-1] - clat[0]) - 1
    gx = 2 * (flat_lon - clon[0]) / (clon[-1] - clon[0]) - 1
    gyy, gxx = np.meshgrid(gy, gx, indexing="ij")
    samp = torch.tensor(np.stack([gxx, gyy], -1).astype(np.float32)[None]).to(dev)
    lat2 = (flat_lat[:, None] - flat_lat.mean()) / flat_lat.std()
    lon2 = (flat_lon[None, :] - flat_lon.mean()) / flat_lon.std()
    static = torch.tensor(np.stack([
        mask.astype(DTYPE),
        np.broadcast_to(lat2, (H, W)).astype(DTYPE),
        np.broadcast_to(lon2, (H, W)).astype(DTYPE)])[None]).to(dev)

    ckpt = torch.load(args.ckpt, map_location=dev)
    base = ckpt["args"].get("base", 24)
    drop = ckpt["args"].get("drop", 0.2)

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

        def forward(s, x):
            return s.f(x)

    class MWUNet(nn.Module):
        def __init__(s, n_var, n_win, base=24, drop=0.2, emb=4):
            super().__init__()
            s.emb = nn.Embedding(n_win, emb)
            s.enc_c1 = Block(n_var + emb, base * 2); s.enc_c2 = Block(base * 2, base * 2)
            s.inp = Block(base * 2 + 3, base)
            s.d1 = Block(base, base * 2, drop); s.d2 = Block(base * 2, base * 4, drop)
            s.bott = Block(base * 4, base * 4, drop)
            s.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2); s.du2 = Block(base * 4, base * 2, drop)
            s.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2); s.du1 = Block(base * 2, base)
            s.head = nn.Conv2d(base, 1, 1); s.pool = nn.MaxPool2d(2)

        def forward(s, xc, wid, samp, static):
            b, _, h, w = xc.shape
            e = s.emb(wid)[:, :, None, None].expand(-1, -1, h, w)
            xc = torch.cat([xc, e], 1)
            c = s.enc_c2(s.enc_c1(xc))
            f = F.grid_sample(c, samp.expand(b, -1, -1, -1), mode="bilinear", align_corners=True)
            f = torch.cat([f, static.expand(b, -1, -1, -1)], 1)
            H0, W0 = f.shape[-2:]
            f = F.pad(f, (0, (-W0) % 4, 0, (-H0) % 4), mode="replicate")
            e0 = s.inp(f); e1 = s.d1(s.pool(e0)); e2 = s.d2(s.pool(e1))
            u = s.du2(torch.cat([s.u2(s.bott(e2)), e1], 1))
            u = s.du1(torch.cat([s.u1(u), e0], 1))
            return s.head(u)[:, :, :H0, :W0].squeeze(1)

    model = MWUNet(n_var, n_win, base=base, drop=drop).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()

    Xt = torch.tensor(Xa)
    widt = torch.tensor(wid)
    idx = np.where(te_i)[0]
    out = []
    with torch.no_grad():
        for a in range(0, len(idx), args.batch):
            j = idx[a:a + args.batch]
            out.append(model(Xt[j].to(dev), widt[j].to(dev), samp, static).cpu().numpy())
    return np.concatenate(out)   # UNet already predicts anomalies


def plot_curves(window_names, curves, accs, out):
    x = np.arange(len(window_names))
    style = {"raw": ("o-", "#888888", "raw ECMWF"),
             "qm": ("s--", "#4C78A8", "quantile mapping"),
             "unet": ("D-", "#E45756", "UNet (ours)")}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in ("raw", "qm", "unet"):
        m, c, lab = style[k]
        ax.plot(x, curves[k], m, color=c, lw=2, ms=7, label=lab)
    ax.axhline(0, color="k", lw=0.9, ls=":")
    ax.text(0.02, 0.02, "0 = climatology", transform=ax.transAxes,
            fontsize=8, color="k", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(window_names)
    ax.set_ylabel("skill vs climatology  (1 - RMSE/RMSE_clim)")
    ax.set_xlabel("forecast window")
    ax.set_title("Downscaling skill by window: does the UNet beat the raw forecast?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}_skill.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in ("raw", "qm", "unet"):
        m, c, lab = style[k]
        ax.plot(x, accs[k], m, color=c, lw=2, ms=7, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(window_names)
    ax.set_ylabel("anomaly correlation (ACC)")
    ax.set_xlabel("forecast window")
    ax.set_title("Anomaly correlation by window")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}_acc.png", dpi=140)
    plt.close(fig)


def interpret(window_names, curves):
    print("\nRead the skill plot like this:")
    for i, w in enumerate(window_names):
        r, q, u = curves["raw"][i], curves["qm"][i], curves["unet"][i]
        gain = u - r
        verdict = ("UNet adds real value over raw forecast" if gain > 0.01
                   else "UNet ~ raw forecast (little downscaling gain)"
                   if gain > -0.01 else "UNet WORSE than raw -- investigate")
        print(f"  {w:>8}: UNet {u:+.3f} vs raw {r:+.3f} "
              f"(gain {gain:+.3f}) -- {verdict}")


# Jupyter Notebook equivalent of command-line args
class Args:
    cmd = 'train' # Set to 'prepare' to generate cache, or 'train' to train the model
    epochs = 60
    batch = 16
    base = 24
    drop = 0.2
    wd = 1e-3
    lr = 2e-4
    patience = 10
    folds = 5

args = Args()

if args.cmd == 'prepare':
    ecmwf_ds = ds_ecmv   # noqa: F821  <- replace with your open datasets
    imd_ds = ds_imd      # noqa: F821
    prepare(ecmwf_ds, imd_ds, CACHE)
else:
    if not os.path.exists(CACHE):
        raise SystemExit(f'run `prepare` first ({CACHE} missing)')
    build_and_run(args, CACHE, OUT_MAPS)
