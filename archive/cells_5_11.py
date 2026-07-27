# %%
# ============ CELL 5: LOAD CACHE + GRIDS + DATASET ============
# Re-run after prepare, or when tuning dataloader performance.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")

# XA/XB stay on disk: they are only read SEQUENTIALLY (anomalise walks them in
# row chunks), so paging is cheap. The random-access arrays the DataLoader hits
# are XAa/XBa, which cell 10 allocates in RAM.
XA = np.load(f"{CACHE}/XA.npy", mmap_mode="r")
XB = np.load(f"{CACHE}/XB.npy", mmap_mode="r")
y  = np.load(f"{CACHE}/y.npy")
_m = np.load(f"{CACHE}/meta.npz")

doy, wid = np.asarray(_m["doy"]), np.asarray(_m["wid"])
year, mask, is_test = np.asarray(_m["year"]), np.asarray(_m["mask"]), np.asarray(_m["is_test"])
flat_lat, flat_lon = np.asarray(_m["flat_lat"]), np.asarray(_m["flat_lon"])
clatA, clonA, clatB, clonB = _m["clatA"], _m["clonA"], _m["clatB"], _m["clonB"]
window_names = [str(w) for w in _m["window_names"]]

H, W = len(flat_lat), len(flat_lon)
N_WIN = len(window_names)
N_LAG = XA.shape[1]
C_A, C_B = XA.shape[2] * N_LAG, XB.shape[2] * N_LAG   # lags folded to channels


def _make_samp(clat, clon):
    """Coordinate-aware bilinear grid: maps the fine IMD grid into a source
    box's normalised [-1,1] frame. Asserting |g|<=1 catches an extent bug
    that would otherwise silently edge-clamp."""
    gy = 2 * (flat_lat - clat[0]) / (clat[-1] - clat[0]) - 1
    gx = 2 * (flat_lon - clon[0]) / (clon[-1] - clon[0]) - 1
    gyy, gxx = np.meshgrid(gy, gx, indexing="ij")
    s = np.stack([gxx, gyy], -1).astype(np.float32)[None]
    assert np.abs(s).max() <= 1.0, "fine grid outside source box"
    return torch.tensor(s).to(DEV)


SAMP_A, SAMP_B = _make_samp(clatA, clonA), _make_samp(clatB, clonB)

_lat2 = (flat_lat[:, None] - flat_lat.mean()) / flat_lat.std()
_lon2 = (flat_lon[None, :] - flat_lon.mean()) / flat_lon.std()
STATIC = torch.tensor(np.stack([
    mask.astype(DTYPE),
    np.broadcast_to(_lat2, (H, W)).astype(DTYPE),
    np.broadcast_to(_lon2, (H, W)).astype(DTYPE)])[None]).to(DEV)

# strict mask means isfinite(y[i]) & mask == mask for every i, so this is
# constant — recomputing it per __getitem__ was pure waste
FIN_STATIC = mask.astype(DTYPE)


class LagDS(Dataset):
    """from_numpy shares memory (torch.tensor copies). yb built per item so a
    full-size yt array never exists."""
    def __init__(self, XAa, XBa, ya, idx):
        self.XAa, self.XBa, self.ya, self.idx = XAa, XBa, ya, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(np.ascontiguousarray(self.XAa[j])),
                torch.from_numpy(np.ascontiguousarray(self.XBa[j])),
                int(wid[j]),
                torch.from_numpy(np.nan_to_num(self.ya[j])),
                torch.from_numpy(FIN_STATIC))


print(f"device {DEV} | {len(XA)} samples | encA {C_A}ch encB {C_B}ch | "
      f"grid {H}x{W} | {int(mask.sum())} valid cells")


# %%
# ============ CELL 6: MODEL ============
# Edit for architecture experiments.

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


class LaggedDualUNet(nn.Module):
    """Two coarse encoders on DIFFERENT boxes, each grid_sampled to the shared
    fine grid with its own sampling grid, concatenated there, then one decoder.

    Merging at the fine grid (not the bottleneck) is what lets the two extents
    coexist: each encoder reasons spatially in its own frame first.

    Lags enter as channels (n_lag x V) so the first conv can form differences
    across lags — that is the tendency signal. USE_LAG_MIXER=True inserts a 1x1
    to compress them first (fewer params); False feeds them straight to the 3x3
    (matches the no-lag baseline's structure, so the lag ablation is clean).
    """
    def __init__(self, cA, cB, n_win, base=24, drop=0.2, emb=4, mixer=True):
        super().__init__()
        self.emb = nn.Embedding(n_win, emb)
        self.mixer = mixer
        if mixer:
            self.mixA = nn.Conv2d(cA, base * 2, 1)
            self.mixB = nn.Conv2d(cB, base * 2, 1)
            inA, inB = base * 2 + emb, base * 2
        else:
            inA, inB = cA + emb, cB
        self.encA1 = Block(inA, base * 2); self.encA2 = Block(base * 2, base * 2)
        self.encB1 = Block(inB, base * 2); self.encB2 = Block(base * 2, base * 2)
        self.inp = Block(base * 4 + 3, base)
        self.d1 = Block(base, base * 2, drop)
        self.d2 = Block(base * 2, base * 4, drop)
        self.bott = Block(base * 4, base * 4, drop)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.du2 = Block(base * 4, base * 2, drop)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.du1 = Block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, xa, xb, w, sampA, sampB, static):
        b = xa.shape[0]
        if self.mixer:
            xa, xb = self.mixA(xa), self.mixB(xb)
        e = self.emb(w)[:, :, None, None].expand(-1, -1, xa.shape[2], xa.shape[3])
        ca = self.encA2(self.encA1(torch.cat([xa, e], 1)))
        cb = self.encB2(self.encB1(xb))
        fa = F.grid_sample(ca, sampA.expand(b, -1, -1, -1),
                           mode="bilinear", align_corners=True)
        fb = F.grid_sample(cb, sampB.expand(b, -1, -1, -1),
                           mode="bilinear", align_corners=True)
        f = torch.cat([fa, fb, static.expand(b, -1, -1, -1)], 1)
        H0, W0 = f.shape[-2:]
        f = F.pad(f, (0, (-W0) % 4, 0, (-H0) % 4), mode="replicate")
        e0 = self.inp(f)
        e1 = self.d1(self.pool(e0))
        e2 = self.d2(self.pool(e1))
        u = self.du2(torch.cat([self.u2(self.bott(e2)), e1], 1))
        u = self.du1(torch.cat([self.u1(u), e0], 1))
        return self.head(u)[:, :, :H0, :W0].squeeze(1)


def new_model():
    m = LaggedDualUNet(C_A, C_B, N_WIN, base=BASE, drop=DROP,
                       mixer=USE_LAG_MIXER).to(DEV)
    return m


print(f"params: {sum(p.numel() for p in new_model().parameters())/1e6:.2f}M "
      f"(mixer={USE_LAG_MIXER})")


# %%
# ============ CELL 7: LOSS ============
# Edit for loss experiments. Every loss returns (scalar, stats_dict).

def masked_mse(pred, target, m):
    """Fill-then-mask-then-normalise-BY-MASK. Never by numel, or the loss
    depends on how much ocean is in the domain."""
    se = (pred - target) ** 2 * m
    return se.sum() / m.sum().clamp(min=1.0), {}


def compute_pixel_thresholds(ya, tr, mask, pct=(10.0, 90.0), chunk=20000):
    """Per-pixel R10/R90 of the target ANOMALY, TRAIN samples only.

    Per-pixel, not global: 'heavy' in the Thar and 'heavy' in the Ghats are
    different numbers, and a global threshold would park the whole arid
    northwest permanently in the 'light' bucket.
    """
    flat = ya[tr].reshape(int(np.sum(tr)), -1)
    n_cells = flat.shape[1]
    r_lo = np.full(n_cells, np.nan, DTYPE)
    r_hi = np.full(n_cells, np.nan, DTYPE)
    mf = mask.reshape(-1)
    for a in range(0, n_cells, chunk):
        b = min(a + chunk, n_cells)
        valid = mf[a:b]
        if not valid.any():
            continue
        sub = flat[:, a:b][:, valid]
        with np.errstate(invalid="ignore"):
            r_lo[np.where(valid)[0] + a] = np.nanpercentile(sub, pct[0], axis=0)
            r_hi[np.where(valid)[0] + a] = np.nanpercentile(sub, pct[1], axis=0)
    return r_lo.reshape(mask.shape), r_hi.reshape(mask.shape)


class RegimeLoss(nn.Module):
    """base MSE + regime-partitioned MSE (per-pixel percentiles) + a
    spatial-aggregate term on the domain-mean anomaly.

    The aggregate term exists because per-cell MSE and domain-total error are
    different failures: errors that all lean one way cancel badly in the total.
    Your two reported metrics (per-cell ACC, India-mean corr) measure exactly
    these two things; plain MSE only optimises the first.
    """
    def __init__(self, r_lo, r_hi):
        super().__init__()
        self.register_buffer("r_lo", torch.as_tensor(np.nan_to_num(r_lo, nan=-1e9)))
        self.register_buffer("r_hi", torch.as_tensor(np.nan_to_num(r_hi, nan=+1e9)))
        self.wl, self.wm, self.wh = REGIME_W

    def forward(self, pred, target, m):
        se = (pred - target) ** 2
        base = (se * m).sum() / m.sum().clamp(min=1.0)

        lo, hi = self.r_lo.unsqueeze(0), self.r_hi.unsqueeze(0)
        m_low = m * (target < lo).float()
        m_high = m * (target > hi).float()
        m_mid = m * ((target >= lo) & (target <= hi)).float()
        # each regime normalised by ITS OWN count, so a rare regime is not
        # automatically negligible
        l_low = (se * m_low).sum() / m_low.sum().clamp(min=1.0)
        l_mid = (se * m_mid).sum() / m_mid.sum().clamp(min=1.0)
        l_high = (se * m_high).sum() / m_high.sum().clamp(min=1.0)
        regime = self.wl * l_low + self.wm * l_mid + self.wh * l_high

        wsum = m.sum((1, 2)).clamp(min=1.0)
        agg = (((pred * m).sum((1, 2)) / wsum
                - (target * m).sum((1, 2)) / wsum) ** 2).mean()

        total = W_BASE * base + W_REGIME * regime + W_AGG * agg
        return total, {"base": float(base.detach()), "high": float(l_high.detach()),
                       "agg": float(agg.detach())}


def make_criterion(ya, tr):
    """Built per fold — thresholds must come from that fold's train years."""
    if LOSS_NAME == "mse":
        return masked_mse
    if LOSS_NAME == "regime":
        r_lo, r_hi = compute_pixel_thresholds(ya, tr, mask)
        return RegimeLoss(r_lo, r_hi).to(DEV)
    raise ValueError(LOSS_NAME)


# %%
# ============ CELL 8: METRICS ============
# Never changes.

def skill_acc(p, t, fin):
    """Per-cell skill vs zero-anomaly climatology, and per-cell ACC.
    rc>1e-6 guard: a degenerate near-constant cell would give skill = -inf
    and poison the whole nanmean."""
    se_m = np.where(fin, (t - p) ** 2, np.nan)
    se_c = np.where(fin, t ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        rm = np.sqrt(np.nanmean(se_m, axis=0))
        rc = np.sqrt(np.nanmean(se_c, axis=0))
        ok = rc > 1e-6
        skill = np.where(ok, 1 - rm / np.where(ok, rc, 1), np.nan)
        tm = np.nanmean(np.where(fin, t, np.nan), axis=0)
        pm = np.nanmean(np.where(fin, p, np.nan), axis=0)
        num = np.nansum(np.where(fin, (t - tm) * (p - pm), np.nan), axis=0)
        den = np.sqrt(np.nansum(np.where(fin, (t - tm) ** 2, np.nan), axis=0)
                      * np.nansum(np.where(fin, (p - pm) ** 2, np.nan), axis=0))
        acc = np.where(den > 0, num / den, np.nan)
    return skill, acc


def std_ratio(p, t, fin):
    """Diagnostic only — reported, never optimised. <~0.8 means the model is
    hedging amplitude (MSE regressing toward the mean)."""
    pv, tv = p[fin], t[fin]
    return float(pv.std() / tv.std()) if tv.std() > 0 else np.nan


# %%
# ============ CELL 9: TRAIN / PREDICT ============
# Rarely changes.

def train_one(tr_idx, va_idx, XAa, XBa, ya, crit, max_epochs=EPOCHS, verbose=False):
    # num_workers=0: LagDS closes over notebook globals, and spawn would copy
    # XAa/XBa (~GBs) into every worker — worse than the problem it solves.
    tr_dl = DataLoader(LagDS(XAa, XBa, ya, tr_idx), batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(LagDS(XAa, XBa, ya, va_idx), batch_size=BATCH, shuffle=False)

    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best, best_state, wait = np.inf, None, 0

    for ep in range(max_epochs):
        model.train()
        for xa, xb, w, yb, mb in tr_dl:
            xa, xb, w, yb, mb = [t.to(DEV) for t in (xa, xb, w, yb, mb)]
            loss, _ = crit(model(xa, xb, w, SAMP_A, SAMP_B, STATIC), yb, mb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        vl, n, st_last = 0.0, 0, {}
        with torch.no_grad():
            for xa, xb, w, yb, mb in va_dl:
                xa, xb, w, yb, mb = [t.to(DEV) for t in (xa, xb, w, yb, mb)]
                l, st_last = crit(model(xa, xb, w, SAMP_A, SAMP_B, STATIC), yb, mb)
                vl += float(l) * len(xa); n += len(xa)
        vl /= n

        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose:
            extra = " ".join(f"{k} {v:.3f}" for k, v in st_last.items())
            print(f"      ep {ep:>2} vloss {vl:.4f} {extra}", flush=True)
        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best, ep + 1


def predict(model, idx, XAa, XBa, ya):
    dl = DataLoader(LagDS(XAa, XBa, ya, idx), batch_size=BATCH, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for xa, xb, w, _, _ in dl:
            out.append(model(xa.to(DEV), xb.to(DEV), w.to(DEV),
                             SAMP_A, SAMP_B, STATIC).cpu().numpy())
    return np.concatenate(out)


# %%
# ============ CELL 10: CV RUNNER ============
# The cell you re-run for every experiment. Resumes from *_folds.json.

def run_fold(tr_i, va_i):
    """Anomalise on THIS fold's train years only, train, score."""
    XAa = anomalise_lagged(XA, doy, tr_i)
    XBa = anomalise_lagged(XB, doy, tr_i)
    ya = anomalise_target(y, doy, tr_i)
    crit = make_criterion(ya, tr_i)
    model, vloss, eps = train_one(np.where(tr_i)[0], np.where(va_i)[0], XAa, XBa, ya, crit)
    p = predict(model, np.where(va_i)[0], XAa, XBa, ya)
    t = ya[va_i]
    fin = np.isfinite(np.asarray(y)[va_i]) & mask[None]
    sk, ac = skill_acc(p, t, fin)
    res = dict(skill=float(np.nanmean(sk[mask])), acc=float(np.nanmean(ac[mask])),
               std_ratio=std_ratio(p, t, fin), epochs=int(eps), vloss=float(vloss))
    del XAa, XBa, ya, model, p, t; gc.collect()
    return res


resume_path = OUT_MAPS.replace(".nc", "_folds.json")
done = {}
if os.path.exists(resume_path):
    done = {int(k): v for k, v in json.load(open(resume_path)).items()}
    print(f"resuming: folds {sorted(done)} cached")

nontest_years = sorted(set(year[~is_test].tolist()))
blocks = np.array_split(nontest_years, FOLDS)

with stage(f"[{TAG}] CV: {FOLDS} folds over {len(nontest_years)} years"):
    for fi, vy_arr in enumerate(blocks):
        if fi in done:
            r = done[fi]
            print(f"  fold {fi} (cached): skill {r['skill']:+.3f} | ACC {r['acc']:.3f}")
            continue
        vy = set(vy_arr.tolist())
        va_i = np.isin(year, list(vy)) & ~is_test
        tr_i = ~np.isin(year, list(vy)) & ~is_test
        r = run_fold(tr_i, va_i)
        r["val_years"] = sorted(vy)
        done[fi] = r
        json.dump({str(k): v for k, v in done.items()}, open(resume_path, "w"), indent=2)
        print(f"  fold {fi} val {sorted(vy)}: skill {r['skill']:+.3f} | "
              f"ACC {r['acc']:.3f} | std {r['std_ratio']:.3f} | {r['epochs']} ep  [saved]",
              flush=True)

    ks = [i for i in range(FOLDS) if i in done]
    fs = [done[i]["skill"] for i in ks]; fa = [done[i]["acc"] for i in ks]
    print(f"\n  [{TAG}] CV skill {np.mean(fs):+.4f} +/- {np.std(fs):.4f} | "
          f"CV ACC {np.mean(fa):.4f} +/- {np.std(fa):.4f}")


# %%
# ============ CELL 10b: FINAL MODEL -> TEST ============

with stage("Final model on all non-test years -> test"):
    es_years = set(nontest_years[-2:])          # early-stopping slice
    va_i = np.isin(year, list(es_years)) & ~is_test
    fit_i = (~is_test) & ~va_i

    XAa = anomalise_lagged(XA, doy, fit_i)
    XBa = anomalise_lagged(XB, doy, fit_i)
    ya = anomalise_target(y, doy, fit_i)
    crit = make_criterion(ya, fit_i)
    model, _, eps = train_one(np.where(fit_i)[0], np.where(va_i)[0], XAa, XBa, ya, crit)

    te_i = np.where(is_test)[0]
    p = predict(model, te_i, XAa, XBa, ya)
    t = ya[is_test]
    fin = np.isfinite(np.asarray(y)[is_test]) & mask[None]

    import xarray as xr
    wid_te = wid[is_test]
    dvars, summary = {}, {}
    print(f"  trained {eps} ep")
    for w in range(N_WIN):
        sm = wid_te == w
        if sm.sum() == 0:
            continue
        sk, ac = skill_acc(p[sm], t[sm], fin[sm])
        wn = window_names[w]
        dvars[f"{wn}_skill"] = (("lat", "lon"), sk)
        dvars[f"{wn}_acc"] = (("lat", "lon"), ac)
        summary[wn] = dict(skill=float(np.nanmean(sk[mask])),
                           acc=float(np.nanmean(ac[mask])),
                           std_ratio=std_ratio(p[sm], t[sm], fin[sm]))
        print(f"  test {wn:>8}: skill {summary[wn]['skill']:+.4f} | "
              f"ACC {summary[wn]['acc']:.4f} | std {summary[wn]['std_ratio']:.3f} | "
              f"{100*np.nanmean(sk[mask]>0):.0f}% cells+")

    xr.Dataset(dvars, coords={"lat": flat_lat, "lon": flat_lon}).to_netcdf(OUT_MAPS)
    np.savez_compressed(OUT_MAPS.replace(".nc", "_pred.npz"),
                        pred=p, obs=t, wid=wid_te)
    torch.save({"state": model.state_dict(), "tag": TAG},
               OUT_MAPS.replace(".nc", ".pt"))

    # running comparison across every experiment you've run
    tbl = "experiments.json"
    allr = json.load(open(tbl)) if os.path.exists(tbl) else {}
    allr[TAG] = dict(cv_skill=float(np.mean(fs)), cv_acc=float(np.mean(fa)),
                     cv_acc_std=float(np.std(fa)), test=summary,
                     lags=LAG_DAYS, loss=LOSS_NAME, months=str(MONTHS),
                     mixer=USE_LAG_MIXER, base=BASE)
    json.dump(allr, open(tbl, "w"), indent=2)
    print(f"  -> {OUT_MAPS} (+ _pred.npz, .pt); appended to {tbl}")

print(f"\n{'experiment':>22} {'CV ACC':>9} {'w3_4 ACC':>9} {'w5_6 ACC':>9}")
for k, v in sorted(json.load(open("experiments.json")).items()):
    w34 = v["test"].get("week3_4", {}); w56 = v["test"].get("week5_6", {})
    print(f"{k:>22} {v['cv_acc']:>9.4f} {w34.get('acc', float('nan')):>9.4f} "
          f"{w56.get('acc', float('nan')):>9.4f}")


# %%
# ============ CELL 11: PLOTS ============
# Reads the saved _pred.npz — no model rebuild, no inference.

def plot_all(pred_npz=None, tag=None):
    import matplotlib.pyplot as plt
    import xarray as xr
    pred_npz = pred_npz or OUT_MAPS.replace(".nc", "_pred.npz")
    tag = tag or TAG

    pr = np.load(pred_npz)
    unet_a, obs_a, wid_te = pr["pred"], pr["obs"], pr["wid"]

    # raw ECMWF tp -> fine grid -> anomaly (its own climatology; units cancel)
    varsA = [str(v) for v in _m["varsA"]]
    tp_i = varsA.index("total_precipitation")
    fit_i = ~is_test
    tp_c = np.asarray(XA[:, 0, tp_i])          # lag 0
    tp_f = xr.DataArray(tp_c, dims=("s", "lat", "lon"),
                        coords={"lat": clatA, "lon": clonA}
                        ).interp(lat=flat_lat, lon=flat_lon).values.astype(DTYPE)
    ecm_a = (tp_f - _clim_grid(tp_f, doy, CLIM_WINDOW_DAYS, rows=fit_i)[doy - 1])[is_test]

    m3 = mask[None]
    im = lambda a: np.nansum(np.where(m3, a, np.nan), axis=(1, 2)) / m3.sum()
    io, ie, iu = im(obs_a), im(ecm_a), im(unet_a)

    fig, axes = plt.subplots(N_WIN, 1, figsize=(13, 3.2 * N_WIN))
    for w, ax in enumerate(np.atleast_1d(axes)):
        s = wid_te == w
        tt = np.arange(s.sum())
        ax.plot(tt, io[s], "-", color="k", lw=2, label="observed IMD")
        ax.plot(tt, ie[s], "-", color="#888", lw=1.5, label="raw ECMWF")
        ax.plot(tt, iu[s], "-", color="#E45756", lw=1.5, label="UNet")
        ax.axhline(0, color="gray", lw=.5, ls=":")
        ru = np.corrcoef(iu[s], io[s])[0, 1]; re = np.corrcoef(ie[s], io[s])[0, 1]
        ax.set_title(f"{window_names[w]}  (India-mean; UNet {ru:.2f}, ECMWF {re:.2f})")
        ax.set_ylabel("anomaly (mm/day)")
        if w == 0:
            ax.legend(ncol=3, fontsize=9)
    fig.tight_layout(); fig.savefig(f"threeline_{tag}.png", dpi=140); plt.show()

    # per-cell correlation maps
    def cmap_(p_, t_, f_):
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = np.nanmean(np.where(f_, p_, np.nan), 0)
            tm = np.nanmean(np.where(f_, t_, np.nan), 0)
            num = np.nansum(np.where(f_, (p_ - pm) * (t_ - tm), np.nan), 0)
            den = np.sqrt(np.nansum(np.where(f_, (p_ - pm) ** 2, np.nan), 0)
                          * np.nansum(np.where(f_, (t_ - tm) ** 2, np.nan), 0))
            return np.where(den > 0, num / den, np.nan)

    fin_all = np.isfinite(np.asarray(y)[is_test]) & m3
    maps = {}
    fig, axes = plt.subplots(N_WIN, 3, figsize=(15, 4.4 * N_WIN))
    ext = [flat_lon[0], flat_lon[-1], flat_lat[0], flat_lat[-1]]
    for w in range(N_WIN):
        s = wid_te == w
        ce = cmap_(ecm_a[s], obs_a[s], fin_all[s]); ce[~mask] = np.nan
        cu = cmap_(unet_a[s], obs_a[s], fin_all[s]); cu[~mask] = np.nan
        wn = window_names[w]
        maps[f"{wn}_corr_ecmwf"] = ce; maps[f"{wn}_corr_unet"] = cu
        maps[f"{wn}_corr_diff"] = cu - ce
        for c, (arr, ttl, cm, vr) in enumerate([
                (ce, "ECMWF vs IMD", "RdBu_r", .6), (cu, "UNet vs IMD", "RdBu_r", .6),
                (cu - ce, "UNet - ECMWF", "PuOr_r", .4)]):
            ax = np.atleast_2d(axes)[w, c]
            i_ = ax.imshow(arr, cmap=cm, origin="lower", extent=ext,
                           aspect="auto", vmin=-vr, vmax=vr)
            fig.colorbar(i_, ax=ax, fraction=.046)
            ax.set_title(f"{wn} {ttl}\n(mean {np.nanmean(arr[mask]):+.3f})", fontsize=9)
    fig.tight_layout(); fig.savefig(f"corr_maps_{tag}.png", dpi=130); plt.show()

    xr.Dataset({k: (("lat", "lon"), v) for k, v in maps.items()},
               coords={"lat": flat_lat, "lon": flat_lon}).to_netcdf(f"corr_maps_{tag}.nc")
    np.savez_compressed(f"corr_maps_{tag}.npz", lat=flat_lat, lon=flat_lon,
                        mask=mask, **maps)
    print(f"-> threeline_{tag}.png, corr_maps_{tag}.png/.nc/.npz")


# plot_all()
