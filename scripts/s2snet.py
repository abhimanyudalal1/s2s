"""
UNet downscaling: ECMWF S2S (coarse, windowed, anomaly space) -> IMD 0.25
rain anomaly over India, for one target window (default weeks 3-4).

This pulls together every element settled in the diagnostics phase:

  WINDOWED TARGET     week 3-4 mean, not daily leads. The correlation
                      diagnostics showed daily signal dies by lead ~22 but
                      windowed r roughly quadruples; days 15-28 is where the
                      contribution lives.
  ANOMALY SPACE       obs minus DOY climatology; predictors minus their own
                      window climatology (the drift correction). Train-year
                      climatology only -- no leakage.
  MASKED LOSS         supervision only on IMD's valid cells. Strict mask
                      (.all, 4885 cells) so every cell has a full record.
  MASK AS CHANNEL     plus normalised lat/lon, so the net knows where
                      supervision lives and where it is.
  COARSE INPUT        predictors stay at 1.5 deg; the coarse->fine expansion
                      happens INSIDE the model (precomputed grid_sample =
                      exact coordinate-aware bilinear, then conv refinement).
                      Nothing is pre-interpolated to 0.25 on disk.
  GROUPNORM           not BatchNorm: batch stats over a 72%-ocean domain
                      would be dominated by unsupervised pixels.

MEMORY BUDGET (M4 Air, 16 GB)
-----------------------------
Windowing is what makes this fit. One sample per init (not per lead):
    X:  3720 x 21 x ~26 x ~26  float32  ~ 0.2 GB   (coarse, in RAM)
    y:  3720 x 129 x 135       float32  ~ 0.26 GB  (fine, in RAM)
Whole dataset lives in memory; the DataLoader just indexes tensors.
Model ~2-6M params; batch 16 at 129x135 is comfortably under 2 GB peak.
The one expensive pass (window aggregation over the archive) runs once in
`prepare` and is cached to .npz -- afterwards every training run starts in
seconds.

USAGE
-----
  python s2s_unet.py prepare      # archive -> cached arrays (slow, once)
  python s2s_unet.py train        # train + evaluate + write skill maps
  python s2s_unet.py train --epochs 80 --batch 8 --base 32
"""

import argparse
import os
import time
from contextlib import contextmanager

import numpy as np

# ---- CONFIG ----
IMD_TARGET_VAR = "rain"
WINDOW = (15, 28)               # week 3-4; (29, 42) for weeks 5-6
CLIM_WINDOW_DAYS = 7
COARSE_PAD = 3.0                # deg beyond IMD box kept in the input
MONTHS = None                   # e.g. (6, 7, 8, 9) for JJAS inits
VAL_YEARS_N, TEST_YEARS_N = 3, 3
CACHE = "data/cache/unet_cache_w{lo}_{hi}.npz"
OUT_MAPS = "results/models/unet_skill_maps_w{lo}_{hi}.nc"
DTYPE = np.float32


@contextmanager
def stage(name):
    print(f"[ ] {name} ...", flush=True)
    t0 = time.perf_counter()
    yield
    print(f"[x] {name}  ({time.perf_counter() - t0:.1f}s)", flush=True)


# ======================================================================
# PREPARE: archive -> cached arrays (numpy/xarray only, no torch needed)
# ======================================================================

def normalize_step(ds, verbose=True):
    v = ds["step"].values
    if np.issubdtype(v.dtype, np.timedelta64):
        return ds
    mx = int(np.nanmax(v))
    u = "D" if mx <= 60 else ("h" if mx <= 24 * 60 else "s")
    if verbose:
        print(f"    step is {v.dtype} (max {mx}) -> units='{u}'")
    td = v.astype(np.int64).astype(f"timedelta64[{u}]").astype("timedelta64[ns]")
    return ds.assign_coords(step=("step", td))


def ensure_valid_time(ds):
    ds = normalize_step(ds)
    expected = ds["time"] + ds["step"]
    if "valid_time" in ds.coords:
        got = ds["valid_time"]
        if got.shape == expected.shape and (got.values == expected.values).all():
            return ds
        print("    WARNING: existing valid_time != time + step -> rebuilding")
        ds = ds.drop_vars("valid_time")
    return ds.assign_coords(valid_time=expected)


def _doy_matrix(doys, window, n_doy=366):
    centers = np.arange(1, n_doy + 1)
    d = np.abs(doys[None, :].astype(int) - centers[:, None])
    return (np.minimum(d, n_doy - d) <= window).astype(DTYPE)


def _clim_grid(values, doys, window):
    """(n, ...) -> (366, ...). NaN-aware DOY climatology via matmul."""
    shp = values.shape[1:]
    v2 = values.reshape(len(values), -1)
    M = _doy_matrix(doys, window)
    counts = M @ np.isfinite(v2).astype(DTYPE)
    sums = M @ np.nan_to_num(v2).astype(DTYPE)
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return clim.reshape((366,) + shp).astype(DTYPE)


def prepare(ecmwf_ds, imd_ds, lo, hi, cache_path):
    import xarray as xr
    from dask.diagnostics import ProgressBar

    with stage("Coarse subset over padded India box"):
        ecmwf_ds = ensure_valid_time(ecmwf_ds)
        for c in ("lat", "lon"):
            if ecmwf_ds[c].values[0] > ecmwf_ds[c].values[-1]:
                ecmwf_ds = ecmwf_ds.sortby(c)
        la0, la1 = float(imd_ds.lat.min()), float(imd_ds.lat.max())
        lo0, lo1 = float(imd_ds.lon.min()), float(imd_ds.lon.max())
        sub = ecmwf_ds.sel(lat=slice(la0 - COARSE_PAD, la1 + COARSE_PAD),
                           lon=slice(lo0 - COARSE_PAD, lo1 + COARSE_PAD))
        if MONTHS is not None:
            sub = sub.sel(time=sub["time"].dt.month.isin(list(MONTHS)))
        feature_vars = list(sub.data_vars)
        print(f"    {dict(sub.sizes)} x {len(feature_vars)} vars")

    with stage(f"Window aggregation days {lo}-{hi} (the one slow pass)"):
        leads = (sub["step"].values / np.timedelta64(1, "D")).astype(int)
        sel = np.where((leads >= lo) & (leads <= hi))[0]
        if len(sel) == 0:
            raise ValueError(f"no leads in [{lo},{hi}]")
        with ProgressBar():
            Xw = sub.isel(step=sel).mean(dim="step").compute()
        X = np.stack([Xw[v].values for v in feature_vars], axis=1).astype(DTYPE)
        # X: (n_init, n_var, n_clat, n_clon)
        clat, clon = Xw["lat"].values, Xw["lon"].values
        print(f"    X {X.shape}  {X.nbytes/1e9:.2f} GB")

    with stage("IMD windowed target on the fine grid"):
        vt = sub["valid_time"].isel(step=sel).dt.floor("D").values
        imd = imd_ds[IMD_TARGET_VAR]
        imd = imd.assign_coords(time=imd["time"].dt.floor("D"))
        flat = vt.ravel()
        with ProgressBar():
            y_all = imd.reindex(time=flat).astype(DTYPE).compute().values
        y = np.nanmean(y_all.reshape(vt.shape + y_all.shape[1:]), axis=1)
        flat_lat, flat_lon = imd["lat"].values, imd["lon"].values
        print(f"    y {y.shape}  {y.nbytes/1e9:.2f} GB")

    with stage("Mask (strict), splits, climatologies (train only)"):
        # strict mask: cell valid on EVERY day => uniform record length
        mask = np.isfinite(y).all(axis=0)
        print(f"    strict mask: {int(mask.sum())} cells "
              f"(loose would be {int(np.isfinite(y).any(axis=0).sum())})")

        init = sub["time"].values
        years = init.astype("datetime64[Y]").astype(int) + 1970
        uy = np.unique(years)
        test_y, val_y = set(uy[-TEST_YEARS_N:]), \
            set(uy[-(TEST_YEARS_N + VAL_YEARS_N):-TEST_YEARS_N])
        tr = ~np.isin(years, list(test_y | val_y))
        va = np.isin(years, list(val_y))
        te = np.isin(years, list(test_y))
        print(f"    train {tr.sum()} | val {va.sum()} | test {te.sum()} inits")

        centre = init + np.timedelta64((lo + hi) // 2, "D")
        import xarray as xr
        doy = xr.DataArray(centre, dims="t").dt.dayofyear.values

        clim_y = _clim_grid(y[tr], doy[tr], CLIM_WINDOW_DAYS)     # (366, H, W)
        clim_X = _clim_grid(X[tr], doy[tr], CLIM_WINDOW_DAYS)     # (366, V, h, w)

        y_anom = y - clim_y[doy - 1]
        X_anom = X - clim_X[doy - 1]

        # per-variable standardisation of predictors (train stats)
        xm = np.nanmean(X_anom[tr], axis=(0, 2, 3), keepdims=True)
        xs = np.nanstd(X_anom[tr], axis=(0, 2, 3), keepdims=True)
        xs = np.where(xs < 1e-8, 1.0, xs)
        X_anom = np.nan_to_num((X_anom - xm) / xs)

        # target scale (single scalar; masked cells only) for loss conditioning
        ys = float(np.nanstd(y_anom[tr][:, mask]))
        print(f"    target anomaly std (train, masked): {ys:.3f} mm/day")

    with stage(f"Caching -> {cache_path}"):
        np.savez_compressed(
            cache_path,
            X=X_anom, y=y_anom, mask=mask, doy=doy,
            tr=tr, va=va, te=te, y_std=ys,
            clim_y=clim_y, clat=clat, clon=clon,
            flat_lat=flat_lat, flat_lon=flat_lon,
            feature_vars=np.array(feature_vars),
            init=init.astype("datetime64[s]").astype(np.int64),
        )
        print(f"    {os.path.getsize(cache_path)/1e9:.2f} GB on disk")


# ======================================================================
# MODEL (torch imported lazily so `prepare` works without it)
# ======================================================================

def build_model_and_train(args, cache_path, out_maps):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"    device: {dev}")

    z = np.load(cache_path, allow_pickle=False)
    X, y, mask = z["X"], z["y"], z["mask"]
    tr, va, te = z["tr"], z["va"], z["te"]
    clat, clon = z["clat"], z["clon"]
    flat_lat, flat_lon = z["flat_lat"], z["flat_lon"]
    H, W = len(flat_lat), len(flat_lon)

    # ---- exact coarse->fine alignment as a precomputed grid_sample grid ----
    # grid_sample wants normalised coords in [-1, 1] over the COARSE extent.
    # This is coordinate-aware bilinear: no assumption that the grids nest.
    gy = 2 * (flat_lat - clat[0]) / (clat[-1] - clat[0]) - 1
    gx = 2 * (flat_lon - clon[0]) / (clon[-1] - clon[0]) - 1
    gyy, gxx = np.meshgrid(gy, gx, indexing="ij")
    samp_grid = torch.tensor(
        np.stack([gxx, gyy], axis=-1), dtype=torch.float32
    ).unsqueeze(0)                                    # (1, H, W, 2)

    # static channels: mask + normalised lat/lon
    lat2 = (flat_lat[:, None] - flat_lat.mean()) / flat_lat.std()
    lon2 = (flat_lon[None, :] - flat_lon.mean()) / flat_lon.std()
    static = np.stack([mask.astype(DTYPE),
                       np.broadcast_to(lat2, (H, W)).astype(DTYPE),
                       np.broadcast_to(lon2, (H, W)).astype(DTYPE)])
    static_t = torch.tensor(static).unsqueeze(0)      # (1, 3, H, W)

    n_var = X.shape[1]

    def gn(c):
        return nn.GroupNorm(min(8, c), c)

    class Block(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), gn(co), nn.SiLU(),
                nn.Conv2d(co, co, 3, padding=1), gn(co), nn.SiLU())

        def forward(self, x):
            return self.f(x)

    class DownscaleUNet(nn.Module):
        """Coarse predictors in; fine anomaly field out.

        Coarse branch encodes synoptic context at native 1.5 deg. Its
        features are lifted to the fine grid by the precomputed grid_sample
        (exact coordinate mapping), concatenated with static channels, then
        refined by a small UNet at 0.25 deg. The upsampling is therefore
        learned-refined, never a stored interpolation.
        """
        def __init__(self, n_var, base=24):
            super().__init__()
            self.enc_c1 = Block(n_var, base * 2)
            self.enc_c2 = Block(base * 2, base * 2)          # coarse context
            self.inp = Block(base * 2 + 3, base)             # + static
            self.d1 = Block(base, base * 2)
            self.d2 = Block(base * 2, base * 4)
            self.bott = Block(base * 4, base * 4)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.du2 = Block(base * 4, base * 2)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.du1 = Block(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, xc, samp, static):
            b = xc.shape[0]
            c = self.enc_c2(self.enc_c1(xc))                 # (b, 2base, h, w)
            f = F.grid_sample(c, samp.expand(b, -1, -1, -1),
                              mode="bilinear", align_corners=True)
            f = torch.cat([f, static.expand(b, -1, -1, -1)], dim=1)

            # pad fine grid to a multiple of 4 for two pool/unpool levels
            H0, W0 = f.shape[-2:]
            ph, pw = (-H0) % 4, (-W0) % 4
            f = F.pad(f, (0, pw, 0, ph), mode="replicate")

            e0 = self.inp(f)
            e1 = self.d1(self.pool(e0))
            e2 = self.d2(self.pool(e1))
            btm = self.bott(e2)
            u = self.du2(torch.cat([self.u2(btm), e1], dim=1))
            u = self.du1(torch.cat([self.u1(u), e0], dim=1))
            out = self.head(u)[:, :, :H0, :W0]
            return out.squeeze(1)

    # ---- data ----
    Xt = torch.tensor(X)
    yt = torch.tensor(np.nan_to_num(y))
    fin = torch.tensor(np.isfinite(y) & mask[None])          # per-sample mask
    mk_t = fin.float()

    def loader(idx, shuffle):
        ds = TensorDataset(Xt[idx], yt[idx], mk_t[idx])
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=0)

    tr_dl = loader(np.where(tr)[0], True)
    va_dl = loader(np.where(va)[0], False)

    model = DownscaleUNet(n_var, base=args.base).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"    params: {n_par/1e6:.2f} M")
    samp = samp_grid.to(dev)
    stat = static_t.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def masked_mse(pred, target, m):
        # fill-then-mask-then-normalise-by-mask; never by numel
        se = (pred - target) ** 2 * m
        return se.sum() / m.sum().clamp(min=1.0)

    def run_epoch(dl, train):
        model.train(train)
        tot, cnt = 0.0, 0
        with torch.set_grad_enabled(train):
            for xb, yb, mb in dl:
                xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
                pred = model(xb, samp, stat)
                loss = masked_mse(pred, yb, mb)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                tot += float(loss) * len(xb)
                cnt += len(xb)
        return tot / cnt

    with stage(f"Training {args.epochs} epochs, batch {args.batch}"):
        best, best_state, patience = np.inf, None, 0
        for ep in range(args.epochs):
            tl = run_epoch(tr_dl, True)
            vl = run_epoch(va_dl, False)
            sched.step()
            star = ""
            if vl < best - 1e-6:
                best, patience = vl, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                star = "  *"
            else:
                patience += 1
            print(f"    ep {ep:>3}  train {tl:.4f}  val {vl:.4f}{star}", flush=True)
            if patience >= args.patience:
                print(f"    early stop (no val improvement in {args.patience})")
                break
        model.load_state_dict(best_state)

    # ---- evaluate: skill vs climatology (predict-zero-anomaly), per cell ----
    with stage("Evaluating on validation + test"):
        import xarray as xr
        results = {}
        for name, idx in (("val", np.where(va)[0]), ("test", np.where(te)[0])):
            preds = []
            model.eval()
            with torch.no_grad():
                for a in range(0, len(idx), args.batch):
                    xb = Xt[idx[a:a + args.batch]].to(dev)
                    preds.append(model(xb, samp, stat).cpu().numpy())
            p = np.concatenate(preds)                        # (n, H, W) anomaly
            t = y[idx]                                       # NaN outside mask

            fin_e = np.isfinite(t) & mask[None]
            se_m = np.where(fin_e, (t - p) ** 2, np.nan)
            se_c = np.where(fin_e, t ** 2, np.nan)           # clim = 0 anomaly
            with np.errstate(invalid="ignore"):
                rmse_m = np.sqrt(np.nanmean(se_m, axis=0))
                rmse_c = np.sqrt(np.nanmean(se_c, axis=0))
                skill = 1 - rmse_m / rmse_c
                # per-cell ACC
                tm = np.nanmean(np.where(fin_e, t, np.nan), axis=0)
                pm = np.nanmean(np.where(fin_e, p, np.nan), axis=0)
                num = np.nansum(np.where(fin_e, (t - tm) * (p - pm), np.nan), axis=0)
                den = np.sqrt(np.nansum(np.where(fin_e, (t - tm) ** 2, np.nan), axis=0)
                              * np.nansum(np.where(fin_e, (p - pm) ** 2, np.nan), axis=0))
                acc = np.where(den > 0, num / den, np.nan)
            results[name] = (skill, acc)
            ms, ma = np.nanmean(skill[mask]), np.nanmean(acc[mask])
            pos = 100 * np.nanmean(skill[mask] > 0)
            print(f"    {name}: mean skill {ms:+.3f} | mean ACC {ma:.3f} "
                  f"| {pos:.0f}% cells positive")

        out = xr.Dataset(
            {f"{n}_{k}": (("lat", "lon"), arr)
             for n, (s, a) in results.items()
             for k, arr in (("skill", s), ("acc", a))},
            coords={"lat": flat_lat, "lon": flat_lon},
        )
        out.attrs["window_days"] = f"{WINDOW[0]}-{WINDOW[1]}"
        out.attrs["note"] = ("skill = 1 - rmse/rmse_clim in anomaly space; "
                             "clim = predict zero anomaly. NaN outside IMD mask.")
        out.to_netcdf(out_maps)
        torch.save({"state": best_state, "args": vars(args)},
                   out_maps.replace(".nc", ".pt"))
        print(f"    -> {out_maps} (+ .pt checkpoint). Plot in Panoply directly.")


# ======================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    lo, hi = WINDOW
    cache_path = CACHE.format(lo=lo, hi=hi)
    out_maps = OUT_MAPS.format(lo=lo, hi=hi)

    if args.cmd == "prepare":
        ecmwf_ds = ds_ecmv   # noqa: F821  <- replace with your open datasets
        imd_ds = ds_imd      # noqa: F821
        prepare(ecmwf_ds, imd_ds, lo, hi, cache_path)
    else:
        if not os.path.exists(cache_path):
            raise SystemExit(f"run `prepare` first ({cache_path} missing)")
        build_model_and_train(args, cache_path, out_maps)