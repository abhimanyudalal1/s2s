import pandas as pd
import xarray as xr
import numpy as np

# from cell 1
ds_big = xr.open_zarr("/Users/abhimanyu/Downloads/IFS_reforecast_download-main/s2s_new_vars_sorted.zarr")
ds_imd = xr.open_zarr("data/raw/IMD_rainfall_0p25.zarr")
ds_imd = ds_imd.where(ds_imd != -999)
ds_ecm = xr.open_zarr("data/processed/s2s_reforecast_sorted.zarr")

step_td = pd.to_timedelta(ds_ecm.step.values, unit="D").to_numpy()
ds_ecmv = ds_ecm.assign_coords(
    valid_time=(("time", "step"),
                ds_ecm.time.values[:, None] + step_td[None, :])
)

# from cell 2 preparation logic
IMD_TARGET_VAR = "rain"
WINDOWS = [("week2", 8, 14), ("week3_4", 15, 28), ("week5_6", 29, 42)]
COARSE_PAD = 3.0
DTYPE = np.float32

BIG_VARS = [
    "top_net_thermal_radiation", 
    "geopotential_height_200",
    "geopotential_height_500", 
    "geopotential_height_850",
    "geopotential_height_1000"
]

def _subset_box(ds, imd, pad):
    for c in ("lat", "lon"):
        if ds[c].values[0] > ds[c].values[-1]:
            ds = ds.sortby(c)
    if pad is not None:
        la0, la1 = float(imd.lat.min()), float(imd.lat.max())
        lo0, lo1 = float(imd.lon.min()), float(imd.lon.max())
        ds = ds.sel(lat=slice(la0 - pad, la1 + pad), lon=slice(lo0 - pad, lo1 + pad))
    return ds

def _window_mean(sub, feature_vars, leads, lo, hi):
    sel = np.where((leads >= lo) & (leads <= hi))[0]
    Xw = sub.isel(step=sel).mean(dim="step").compute()
    return np.stack([Xw[v].values for v in feature_vars], axis=1).astype(DTYPE), sel

def _to_days(step_arr):
    if np.issubdtype(step_arr.dtype, np.timedelta64):
        return (step_arr / np.timedelta64(1, "D")).astype(int)
    return step_arr.astype(int)

imd = ds_imd[IMD_TARGET_VAR].assign_coords(time=ds_imd["time"].dt.floor("D"))
subA = _subset_box(ds_ecmv, imd, COARSE_PAD)
subB = _subset_box(ds_big, imd, pad=None)

leadsA = _to_days(subA["step"].values)
leadsB = _to_days(subB["step"].values)

wname, lo, hi = WINDOWS[0]
print(f"Testing {wname}...")
try:
    xa, sel = _window_mean(subA, list(subA.data_vars), leadsA, lo, hi)
    print("subA window_mean success. xa shape:", xa.shape)
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    xb, _ = _window_mean(subB, BIG_VARS, leadsB, lo, hi)
    print("subB window_mean success. xb shape:", xb.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
