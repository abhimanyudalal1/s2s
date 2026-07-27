# %% [markdown]
# # Lagged dual-encoder S2S downscaling — decomposed
#
# Cells run top-to-bottom once. After that, re-run only the cell you changed
# plus Cell 10 (the runner). State lives in notebook globals on purpose, so
# you can inspect anything at any point.
#
# | cell | owns | re-run when |
# |---|---|---|
# | 1 | config | every experiment |
# | 2 | utils | ~never |
# | 3 | climatology / anomalies | rarely |
# | 4 | prepare (archive → cache) | new predictors |
# | 5 | cache load, grids, dataset | perf tuning |
# | 6 | model | architecture experiments |
# | 7 | loss | loss experiments |
# | 8 | metrics | ~never |
# | 9 | train / predict | rarely |
# | 10 | CV runner | every experiment |
# | 11 | plots | ~never |


# %%
# ============ CELL 1: CONFIG ============
# The only cell you edit between experiments.

import gc, json, os, time
from contextlib import contextmanager
import numpy as np

DTYPE = np.float32
IMD_TARGET_VAR = "rain"

# --- data / windows ---
WINDOWS   = [("week2", 8, 14), ("week3_4", 15, 28), ("week5_6", 29, 42)]
LAG_DAYS  = [0, 7, 14, 21, 28, 35]   # [0] = no lags (the control run)
LAG_TOL_DAYS = 1
MONTHS    = None                     # None = full year; (6,7,8,9) = JJAS
COARSE_PAD = 3.0
CLIM_WINDOW_DAYS = 7
TEST_YEARS_N = 3

BIG_VARS = ["top_net_thermal_radiation", "geopotential_height_200",
            "geopotential_height_500", "geopotential_height_850",
            "geopotential_height_1000"]

# --- model / training ---
BASE      = 24
DROP      = 0.2
BATCH     = 8
EPOCHS    = 60
LR        = 2e-4
WD        = 1e-3
PATIENCE  = 10
FOLDS     = 5
USE_LAG_MIXER = True    # False -> feed lag channels straight to the 3x3 conv.
                        # The 1x1 mixer is an ARCHITECTURAL change vs the
                        # no-lag baseline; set False to keep the comparison
                        # clean, True to keep params down.

# --- loss ---
LOSS_NAME = "mse"       # "mse" | "regime"
REGIME_W  = (0.2, 0.2, 0.6)   # light / moderate / heavy
W_BASE, W_REGIME, W_AGG = 1.0, 1.0, 0.3

# --- paths (tag them so runs never collide or resume each other) ---
TAG      = f"lag{len(LAG_DAYS)}_{LOSS_NAME}"
CACHE    = "unet_cache_lag"          # a DIRECTORY of .npy files
OUT_MAPS = f"unet_{TAG}.nc"

print(f"TAG={TAG} | lags={LAG_DAYS} | months={MONTHS} | loss={LOSS_NAME}")


# %%
# ============ CELL 2: UTILS ============
# Never changes.

@contextmanager
def stage(name):
    print(f"[ ] {name} ...", flush=True)
    t0 = time.perf_counter()
    yield
    print(f"[x] {name}  ({time.perf_counter() - t0:.1f}s)", flush=True)


def normalize_step(ds, verbose=True):
    """zarr round-trips lose timedelta encoding; step comes back as bare int."""
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
    """Verify rather than trust: a wrong valid_time silently pairs forecasts
    with the wrong day's rain."""
    ds = normalize_step(ds)
    expected = ds["time"] + ds["step"]
    if "valid_time" in ds.coords:
        got = ds["valid_time"]
        if got.shape == expected.shape and (got.values == expected.values).all():
            return ds
        print("    WARNING: existing valid_time != time + step -> rebuilding")
        ds = ds.drop_vars("valid_time")
    return ds.assign_coords(valid_time=expected)


# %%
# ============ CELL 3: CLIMATOLOGY & ANOMALIES ============
# Leakage-critical: everything here takes `rows` = the TRAIN subset only.
#
# NOTE on lag channels: the climatology is computed PER CHANNEL, and channel
# block L always holds the field from exactly lag_L days before the target.
# So clim[doy, L] is already the climatology of (doy - lag_L). Indexing it
# with the target doy is correct, not a bug.

def _clim_grid(values, doys, window, rows=None, cell_chunk=20000, row_chunk=512):
    """(n, ...) -> (366, ...) DOY climatology.

    Chunked over cells AND rows so `values` can be a memmap and the train
    subset is never materialised. Not bit-identical to a single matmul
    (float32 addition reassociates over row chunks); differences ~1e-7.
    """
    shp = values.shape[1:]
    v2 = values.reshape(len(values), -1)
    n_cells = v2.shape[1]
    if rows is None:
        rows = np.arange(len(values))
    elif np.asarray(rows).dtype == bool:
        rows = np.where(rows)[0]
    rows = np.asarray(rows)

    centers = np.arange(1, 367)
    d = np.abs(doys[rows][None, :].astype(np.int32) - centers[:, None])
    M = (np.minimum(d, 366 - d) <= window).astype(DTYPE)

    out = np.empty((366, n_cells), DTYPE)
    for a in range(0, n_cells, cell_chunk):
        b = min(a + cell_chunk, n_cells)
        counts = np.zeros((366, b - a), DTYPE)
        sums = np.zeros((366, b - a), DTYPE)
        for r0 in range(0, len(rows), row_chunk):
            r1 = min(r0 + row_chunk, len(rows))
            blk = np.asarray(v2[rows[r0:r1], a:b])
            fin = np.isfinite(blk)
            counts += M[:, r0:r1] @ fin.astype(DTYPE)
            sums += M[:, r0:r1] @ np.where(fin, blk, 0).astype(DTYPE)
            del blk, fin
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, a:b] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        del counts, sums
    return out.reshape((366,) + shp)


def anomalise_lagged(X, doy, tr, chunk=256):
    """X: (N, n_lag, V, h, w) -> (N, n_lag*V, h, w) anomalised + standardised.
    Lags are folded into the channel axis. In-place chunked: no full-size
    clim[doy-1] broadcast, no triple copy."""
    N, nl, V, h, w = X.shape
    flat = X.reshape(N, nl * V, h, w)
    clim = _clim_grid(flat, doy, CLIM_WINDOW_DAYS, rows=tr)

    out = np.empty((N, nl * V, h, w), DTYPE)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        out[a:b] = flat[a:b] - clim[doy[a:b] - 1]
    del clim

    idx = np.where(tr)[0]
    s1 = np.zeros(nl * V, np.float64); s2 = np.zeros(nl * V, np.float64); cnt = 0
    for a in range(0, len(idx), chunk):
        blk = out[idx[a:a + chunk]]
        s1 += np.nansum(blk, axis=(0, 2, 3))
        s2 += np.nansum(blk.astype(np.float64) ** 2, axis=(0, 2, 3))
        cnt += blk.shape[0] * blk.shape[2] * blk.shape[3]
    m = (s1 / cnt).astype(DTYPE)[None, :, None, None]
    sd = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0)).astype(DTYPE)
    sd = np.where(sd < 1e-8, 1.0, sd)[None, :, None, None]

    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        np.subtract(out[a:b], m, out=out[a:b])
        np.divide(out[a:b], sd, out=out[a:b])
        np.nan_to_num(out[a:b], copy=False)
    return out


def anomalise_target(y, doy, tr, chunk=256):
    """Keeps NaN — the mask depends on it."""
    clim = _clim_grid(y, doy, CLIM_WINDOW_DAYS, rows=tr)
    out = np.empty(y.shape, DTYPE)
    for a in range(0, len(y), chunk):
        b = min(a + chunk, len(y))
        out[a:b] = y[a:b] - clim[doy[a:b] - 1]
    del clim
    return out


# %%
# ============ CELL 4: PREPARE (archive -> cache) ============
# Run once per predictor set. Writes a DIRECTORY of .npy files, not an .npz,
# because mmap_mode is silently ignored for npz members.

def _subset_box(ds, imd, pad):
    ds = ensure_valid_time(ds)
    for c in ("lat", "lon"):
        if ds[c].values[0] > ds[c].values[-1]:
            ds = ds.sortby(c)
    if pad is not None:
        la0, la1 = float(imd.lat.min()), float(imd.lat.max())
        lo0, lo1 = float(imd.lon.min()), float(imd.lon.max())
        ds = ds.sel(lat=slice(la0 - pad, la1 + pad), lon=slice(lo0 - pad, lo1 + pad))
    return ds


def _window_stack(sub, feature_vars, leads, lo, hi):
    from dask.diagnostics import ProgressBar
    sel = np.where((leads >= lo) & (leads <= hi))[0]
    if len(sel) == 0:
        raise ValueError(f"no leads in [{lo},{hi}]")
    with ProgressBar():
        Xw = sub.isel(step=sel).mean(dim="step").compute()
    return np.stack([Xw[v].values for v in feature_vars], axis=1).astype(DTYPE), sel


def _build_lag_index(all_inits, keep_mask, lag_days, tol):
    """idx[i, L] = row of the init nearest to keep_init[i] - lag_days[L].
    ok[i] False if ANY lag missing -> row dropped, never padded (zero in
    anomaly space means 'exactly climatological', a false claim)."""
    day = np.timedelta64(1, "D")
    kept = np.where(keep_mask)[0]
    idx = np.zeros((len(kept), len(lag_days)), np.int64)
    ok = np.ones(len(kept), bool)
    for L, lag in enumerate(lag_days):
        target = all_inits[kept] - lag * day
        pos = np.clip(np.searchsorted(all_inits, target), 1, len(all_inits) - 1)
        lo_ = all_inits[pos - 1]
        hi_ = all_inits[np.minimum(pos, len(all_inits) - 1)]
        pick_hi = np.abs(hi_ - target) < np.abs(target - lo_)
        chosen = np.where(pick_hi, np.minimum(pos, len(all_inits) - 1), pos - 1)
        ok &= np.abs(all_inits[chosen] - target) / day <= tol
        idx[:, L] = chosen
    return kept, idx, ok


def prepare(ecmwf_ds, big_ds, imd_ds, cache_path):
    import xarray as xr
    from dask.diagnostics import ProgressBar
    from numpy.lib.format import open_memmap

    imd = imd_ds[IMD_TARGET_VAR].assign_coords(
        time=imd_ds[IMD_TARGET_VAR]["time"].dt.floor("D"))
    flat_lat, flat_lon = imd["lat"].values, imd["lon"].values

    with stage("Subset both boxes (full year — lags need pre-season history)"):
        subA = _subset_box(ecmwf_ds, imd, COARSE_PAD)
        subB = _subset_box(big_ds, imd, pad=None)
        varsA = list(subA.data_vars)
        missing = [v for v in BIG_VARS if v not in subB.data_vars]
        if missing:
            raise KeyError(f"BIG_VARS missing: {missing}")
        varsB = BIG_VARS
        clatA, clonA = subA["lat"].values, subA["lon"].values
        clatB, clonB = subB["lat"].values, subB["lon"].values
        leadsA = (subA["step"].values / np.timedelta64(1, "D")).astype(int)
        leadsB = (subB["step"].values / np.timedelta64(1, "D")).astype(int)
        initA, initB = subA["time"].values, subB["time"].values
        print(f"    A: {len(varsA)} vars {len(clatA)}x{len(clonA)} | "
              f"B: {len(varsB)} vars {len(clatB)}x{len(clonB)}")

    # guards
    if initA.shape != initB.shape or not (initA == initB).all():
        raise ValueError("A and B have different init dates")
    for cla, clo, nm in [(clatA, clonA, "A"), (clatB, clonB, "B")]:
        if not (cla.min() <= flat_lat.min() and cla.max() >= flat_lat.max()
                and clo.min() <= flat_lon.min() and clo.max() >= flat_lon.max()):
            raise ValueError(f"IMD grid not inside box {nm} — grid_sample would clamp")

    with stage("Lag index"):
        month = initA.astype("datetime64[M]").astype(int) % 12 + 1
        keep = np.isin(month, MONTHS) if MONTHS else np.ones(len(initA), bool)
        kept, lag_idx, ok = _build_lag_index(initA, keep, LAG_DAYS, LAG_TOL_DAYS)
        kept, lag_idx = kept[ok], lag_idx[ok]
        print(f"    {len(kept)} usable inits ({int((~ok).sum())} dropped for "
              f"incomplete {max(LAG_DAYS)}d history)")
        if len(kept) == 0:
            raise ValueError("no inits have complete history")

    os.makedirs(cache_path, exist_ok=True)
    n_kept, nl, nw = len(kept), len(LAG_DAYS), len(WINDOWS)
    XA_mm = XB_mm = None
    y_l, doy_l, wid_l = [], [], []

    for wi, (wname, lo, hi) in enumerate(WINDOWS):
        with stage(f"Window {wname} (days {lo}-{hi}) x {nl} lags"):
            fullA, sel = _window_stack(subA, varsA, leadsA, lo, hi)
            fullB, _ = _window_stack(subB, varsB, leadsB, lo, hi)
            if XA_mm is None:
                XA_mm = open_memmap(f"{cache_path}/XA.npy", mode="w+", dtype=DTYPE,
                                    shape=(n_kept * nw, nl) + fullA.shape[1:])
                XB_mm = open_memmap(f"{cache_path}/XB.npy", mode="w+", dtype=DTYPE,
                                    shape=(n_kept * nw, nl) + fullB.shape[1:])
            XA_mm[wi * n_kept:(wi + 1) * n_kept] = fullA[lag_idx]
            XB_mm[wi * n_kept:(wi + 1) * n_kept] = fullB[lag_idx]
            del fullA, fullB; gc.collect()

            vt = subA["valid_time"].isel(time=kept, step=sel).dt.floor("D").values
            with ProgressBar():
                y_all = imd.reindex(time=vt.ravel()).astype(DTYPE).compute().values
            y_l.append(np.nanmean(y_all.reshape(vt.shape + y_all.shape[1:]), axis=1))
            del y_all
            centre = initA[kept] + np.timedelta64((lo + hi) // 2, "D")
            doy_l.append(xr.DataArray(centre, dims="t").dt.dayofyear.values)
            wid_l.append(np.full(len(kept), wi, np.int64))

    XA_mm.flush(); XB_mm.flush()
    y = np.concatenate(y_l); doy = np.concatenate(doy_l); wid = np.concatenate(wid_l)
    year = np.tile(initA[kept].astype("datetime64[Y]").astype(int) + 1970, nw)

    with stage("Mask + test holdout + cache"):
        mask = np.isfinite(y).all(axis=0)      # strict: valid on every sample
        uy = np.unique(year)
        is_test = np.isin(year, list(uy[-TEST_YEARS_N:]))
        print(f"    strict mask {int(mask.sum())} cells | test {sorted(uy[-TEST_YEARS_N:])}")
        np.save(f"{cache_path}/y.npy", y)
        np.savez(f"{cache_path}/meta.npz", doy=doy, wid=wid, year=year,
                 mask=mask, is_test=is_test, clatA=clatA, clonA=clonA,
                 clatB=clatB, clonB=clonB, flat_lat=flat_lat, flat_lon=flat_lon,
                 varsA=np.array(varsA), varsB=np.array(varsB),
                 lag_days=np.array(LAG_DAYS),
                 window_names=np.array([w[0] for w in WINDOWS]))
        tot = sum(os.path.getsize(f"{cache_path}/{f}") for f in os.listdir(cache_path))
        print(f"    {tot/1e9:.2f} GB -> {cache_path}/")


# RUN ONCE (uncomment):
# prepare(ds_ecmv, ds_big, ds_imd, CACHE)
