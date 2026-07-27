# ======================================================================
# MODEL + TRAIN
# ======================================================================
args.cmd = "train"

def build_and_run(args, cache_path, out_maps, loss_name, loss_kw):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"    device: {dev} | loss: {loss_name} {loss_kw}")

    z = np.load(cache_path, allow_pickle=False)
    X, y, doy, wid = z["X"], z["y"], z["doy"], z["wid"]
    year, mask, is_test = z["year"], z["mask"], z["is_test"]
    clat, clon = z["clat"], z["clon"]
    flat_lat, flat_lon = z["flat_lat"], z["flat_lon"]
    n_win = len(z["window_names"]); H, W = len(flat_lat), len(flat_lon)
    n_var = X.shape[1]

    gy = 2 * (flat_lat - clat[0]) / (clat[-1] - clat[0]) - 1
    gx = 2 * (flat_lon - clon[0]) / (clon[-1] - clon[0]) - 1
    gyy, gxx = np.meshgrid(gy, gx, indexing="ij")
    samp_np = np.stack([gxx, gyy], -1).astype(np.float32)[None]
    assert np.abs(samp_np).max() <= 1.0, "fine grid outside coarse box"

    lat2 = (flat_lat[:, None] - flat_lat.mean()) / flat_lat.std()
    lon2 = (flat_lon[None, :] - flat_lon.mean()) / flat_lon.std()
    static_np = np.stack([mask.astype(DTYPE),
                          np.broadcast_to(lat2, (H, W)).astype(DTYPE),
                          np.broadcast_to(lon2, (H, W)).astype(DTYPE)])[None]

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

    samp = torch.tensor(samp_np).to(dev)
    stat = torch.tensor(static_np).to(dev)

    def skill_acc(p, t, fin):
        se_m = np.where(fin, (t - p) ** 2, np.nan)
        se_c = np.where(fin, t ** 2, np.nan)
        with np.errstate(invalid="ignore"):
            rmse_m = np.sqrt(np.namean(se_m, axis=0))
            rmse_c = np.sqrt(np.nanmean(se_c, axis=0))
            ok = rmse_c > 1e-6
            skill = np.where(ok, 1 - rmse_m / np.where(ok, rmse_c, 1), np.nan)
            tm = np.nanmean(np.where(fin, t, np.nan), axis=0)
            pm = np.nanmean(np.where(fin, p, np.nan), axis=0)
            num = np.nansum(np.where(fin, (t - tm) * (p - pm), np.nan), axis=0)
            den = np.sqrt(np.nansum(np.where(fin, (t - tm) ** 2, np.nan), axis=0)
                          * np.nansum(np.where(fin, (p - pm) ** 2, np.nan), axis=0))
            acc = np.where(den > 0, num / den, np.nan)
        return skill, acc

    def std_ratio(p, t, fin):
        """Diagnostic only -- reported, never optimised."""
        pv, tv = p[fin], t[fin]
        return float(pv.std() / tv.std()) if tv.std() > 0 else np.nan

    def train_one(tr_idx, va_idx, Xa, ya, max_epochs):
        Xt = torch.tensor(Xa)
        yt = torch.tensor(np.nan_to_num(ya))
        widt = torch.tensor(wid)
        fin = torch.tensor((np.isfinite(ya) & mask[None]).astype(DTYPE))

        tstd = float(np.nanstd(ya[tr_idx][np.isfinite(ya[tr_idx])]))
        crit = make_loss(loss_name, tstd, **loss_kw)

        def dl(idx, sh):
            return DataLoader(TensorDataset(Xt[idx], widt[idx], yt[idx], fin[idx]),
                              batch_size=args.batch, shuffle=sh)

        tr_dl, va_dl = dl(tr_idx, True), dl(va_idx, False)
        model = MWUNet(n_var, n_win, base=args.base, drop=args.drop).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

        best, best_state, wait = np.inf, None, 0
        for ep in range(max_epochs):
            model.train()
            for xb, wb, yb, mb in tr_dl:
                xb, wb, yb, mb = [t.to(dev) for t in (xb, wb, yb, mb)]
                loss = crit(model(xb, wb, samp, stat), yb, mb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            vl, n = 0.0, 0
            with torch.no_grad():
                for xb, wb, yb, mb in va_dl:
                    xb, wb, yb, mb = [t.to(dev) for t in (xb, wb, yb, mb)]
                    vl += float(crit(model(xb, wb, samp, stat), yb, mb)) * len(xb)
                    n += len(xb)
            vl /= n
            if vl < best - 1e-6:
                best, wait = vl, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= args.patience:
                break
        model.load_state_dict(best_state)
        return model, best, ep + 1

    def predict(model, idx, Xa):
        Xt = torch.tensor(Xa); widt = torch.tensor(wid)
        out = []
        model.eval()
        with torch.no_grad():
            for a in range(0, len(idx), args.batch):
                j = idx[a:a + args.batch]
                out.append(model(Xt[j].to(dev), widt[j].to(dev), samp, stat).cpu().numpy())
        return np.concatenate(out)

    # ---------- rotating-year CV ----------
    nontest_years = sorted(set(year[~is_test].tolist()))
    blocks = np.array_split(nontest_years, args.folds)
    resume_path = out_maps.replace(".nc", "_folds.json")
    done = {}
    if os.path.exists(resume_path):
        with open(resume_path) as fh:
            done = {int(k): v for k, v in json.load(fh).items()}
        print(f"    resuming: folds {sorted(done)} cached ({resume_path})")

    with stage(f"Rotating-year CV: {args.folds} folds over {len(nontest_years)} years"):
        for fi, val_years in enumerate(blocks):
            if fi in done:
                r = done[fi]
                print(f"    fold {fi} (cached): skill {r['skill']:+.3f} | ACC {r['acc']:.3f}")
                continue
            val_years = set(val_years.tolist())
            va_i = np.isin(year, list(val_years)) & ~is_test
            tr_i = ~np.isin(year, list(val_years)) & ~is_test
            Xa, ya, _ = anomalise_fold(X, y, doy, tr_i)
            model, vloss, eps = train_one(np.where(tr_i)[0], np.where(va_i)[0],
                                          Xa, ya, args.epochs)
            p = predict(model, np.where(va_i)[0], Xa)
            t = ya[va_i]
            fin = np.isfinite(y[va_i]) & mask[None]
            sk, ac = skill_acc(p, t, fin)
            ms, ma = float(np.nanmean(sk[mask])), float(np.nanmean(ac[mask]))
            sr = std_ratio(p, t, fin)
            done[fi] = {"skill": ms, "acc": ma, "std_ratio": sr,
                        "val_years": sorted(val_years), "epochs": int(eps),
                        "vloss": float(vloss)}
            with open(resume_path, "w") as fh:
                json.dump({str(k): v for k, v in done.items()}, fh, indent=2)
            print(f"    fold {fi} val {sorted(val_years)}: skill {ms:+.3f} | "
                  f"ACC {ma:.3f} | std_ratio {sr:.3f} | {eps} ep   [saved]", flush=True)

        ks = [i for i in range(args.folds) if i in done]
        fs = [done[i]["skill"] for i in ks]; fa = [done[i]["acc"] for i in ks]
        fr = [done[i].get("std_ratio", np.nan) for i in ks]
        print(f"\n    [{loss_name}]  CV skill {np.mean(fs):+.4f} +/- {np.std(fs):.4f}"
              f" | CV ACC {np.mean(fa):.4f} +/- {np.std(fa):.4f}"
              f" | std_ratio {np.nanmean(fr):.3f}")

    # ---------- final model -> test ----------
    with stage("Final model on all non-test years -> test"):
        es_years = set(nontest_years[-2:])
        va_i = np.isin(year, list(es_years)) & ~is_test
        fit_i = (~is_test) & ~va_i
        Xa, ya, _ = anomalise_fold(X, y, doy, fit_i)
        model, _, eps = train_one(np.where(fit_i)[0], np.where(va_i)[0], Xa, ya, args.epochs)

        te_i = np.where(is_test)[0]
        p = predict(model, te_i, Xa)
        t = ya[is_test]
        fin = np.isfinite(y[is_test]) & mask[None]

        import xarray as xr
        wid_te = wid[is_test]; data_vars = {}
        print(f"    trained {eps} ep")
        summary = {}
        for w in range(n_win):
            sm = wid_te == w
            if sm.sum() == 0:
                continue
            sk, ac = skill_acc(p[sm], t[sm], fin[sm])
            sr = std_ratio(p[sm], t[sm], fin[sm])
            wn = str(z["window_names"][w])
            data_vars[f"{wn}_skill"] = (("lat", "lon"), sk)
            data_vars[f"{wn}_acc"] = (("lat", "lon"), ac)
            summary[wn] = {"skill": float(np.nanmean(sk[mask])),
                           "acc": float(np.nanmean(ac[mask])),
                           "std_ratio": sr,
                           "pct_pos": float(100 * np.nanmean(sk[mask] > 0))}
            print(f"    test {wn:>8}: skill {summary[wn]['skill']:+.4f} | "
                  f"ACC {summary[wn]['acc']:.4f} | std_ratio {sr:.3f} | "
                  f"{summary[wn]['pct_pos']:.0f}% cells+")

        out = xr.Dataset(data_vars, coords={"lat": flat_lat, "lon": flat_lon})
        out.attrs["loss"] = f"{loss_name} {loss_kw}"
        out.attrs["windows"] = ", ".join(f"{n}:{lo}-{hi}" for n, lo, hi in WINDOWS)
        out.to_netcdf(out_maps)
        torch.save({"state": model.state_dict(), "args": ARG_DICT}, out_maps.replace(".nc", ".pt"))

        # append to a cross-loss comparison table
        tbl = "results/models/loss_comparison.json"
        allr = json.load(open(tbl)) if os.path.exists(tbl) else {}
        allr[tag] = {"cv_skill": float(np.mean(fs)), "cv_acc": float(np.mean(fa)),
                     "cv_std_ratio": float(np.nanmean(fr)), "test": summary}
        json.dump(allr, open(tbl, "w"), indent=2)
        print(f"    -> {out_maps} (+ .pt); comparison appended to {tbl}")


# ======================================================================
if args.cmd == "prepare":
    prepare(ds_ecmv, ds_imd, CACHE)          # noqa: F821
else:
    if not os.path.exists(CACHE):
        raise SystemExit(f"run prepare first ({CACHE} missing)")
    build_and_run(args, CACHE, OUT_MAPS, LOSS_NAME, LOSS_KW)

    # print the running cross-loss comparison
    tbl = "results/models/loss_comparison.json"
    if os.path.exists(tbl):
        allr = json.load(open(tbl))
        print(f"\n{'loss':>22} {'CV skill':>10} {'CV ACC':>8} {'std':>6} "
              f"{'w3_4 skill':>11} {'w3_4 ACC':>9}")
        for k, v in sorted(allr.items()):
            w = v["test"].get("week3_4", {})
            print(f"{k:>22} {v['cv_skill']:>+10.4f} {v['cv_acc']:>8.4f} "
                  f"{v['cv_std_ratio']:>6.3f} {w.get('skill', float('nan')):>+11.4f} "
                  f"{w.get('acc', float('nan')):>9.4f}")