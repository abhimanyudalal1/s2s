# S2S Downscaling of ECMWF IFS Cycle 50r1 Reforecasts to IMD 0.25° Rainfall over India
## Full Technical Pipeline, Decision Register, and Research-Paper Plan

**Version 1.0 — July 2026**
**Intended use:** This document is the master specification. It is written so that both the researcher and any AI assistant (Claude Opus) can execute each step without ambiguity. Every open decision is explicitly flagged as `[DECISION-n]` and collected in Section 12.

---

## 0. Executive Summary

- **Task:** Statistically downscale ECMWF extended-range (S2S) reforecasts (IFS Cycle 50r1, ~2005/06–2024/25, leads to day 42/46) from **1° input** to **IMD 0.25° gauge-based daily rainfall** over India using deep learning (UNet family), with a five-stage predictor/architecture ladder culminating in a **causality-guided (CCM-attention) UNet**.
- **Primary target lead:** week-2/week-3 weekly accumulated rainfall (see `[DECISION-1]`), evaluated for weeks 1–6.
- **Primary season:** JJAS (Indian summer monsoon), with all-season data downloaded.
- **Resolution decision (settled):** predictors at 1° (native extended-range resolution is ~36 km; effective resolution ~150–250 km, so 1° retains virtually all real information; 4× SR factor to 0.25° is a well-posed downscaling problem). IMD 0.25° is the predictand; IMERG only as secondary verification.
- **Paper novelty claims:** (a) first DL downscaling of the newest-cycle ECMWF sub-seasonal reforecasts to IMD over India at S2S leads; (b) systematic predictor-hierarchy ablation (precip-only → surface → multi-level → physics-derived VIMT/shear); (c) causality-guided attention (CCM maps) improving skill and interpretability at S2S leads.

---

## 1. Scientific Framing (what the papers/book tell us to respect)

From the S2S book (Robertson & Vitart) and the two S2S papers, the non-negotiable considerations for any S2S ML work:

1. **Weekly aggregation is the unit of S2S skill.** Standard practice defines week 1 = days 1–7, week 2 = days 8–14, week 3 = days 15–21, week 4 = days 22–28 (Li & Robertson convention). Daily grid-point precipitation skill beyond ~day 10 over India is near zero; weekly accumulations are where measurable skill lives. **Train and verify primarily on weekly accumulated precipitation.**
2. **Lead-dependent bias correction is mandatory.** Model drift grows with lead; climatologies (model and obs) must be computed per lead time and per initialization date, from training years only. Anomalies are always defined relative to lead-dependent climatology.
3. **Reforecast configuration constrains everything.** The extended-range reforecast is produced "on the fly": for each real-time Monday and Thursday run, 11-member reforecasts are produced for the same day/month over the past 20 years. Cycle 50r1 (operational 12 May 2026) did not change resolution or reforecast configuration. All reforecasts within one cycle are from a *fixed* model version — this homogeneity is exactly why we use 50r1 reforecasts rather than mixing cycles.
4. **Verify against climatological probabilistic baselines** (RPSS, CRPSS, BSS) and against simple statistical post-processing (MOS/quantile mapping) — the book's Chapter 15 point that "it is hard to beat the recalibrated (multi)model mean" is the bar the DL model must clear to be publishable.
5. **Windows of opportunity:** S2S skill is state-dependent (MJO phase, ENSO, active/break cycle). Stratified verification is expected by reviewers and is a cheap source of insight.
6. **Ensemble information matters:** ensemble mean for deterministic scores; member spread for probabilistic scores. Do not throw members away.
7. **Small-sample statistics:** ~20 years × ~35 warm-season inits is a small dataset by DL standards. Guard against overfitting (leave-year-out splits, no leakage through climatology/normalization/causality maps, block bootstrap for significance).

---

## 2. Data Specification

### 2.1 Predictors — ECMWF extended-range reforecasts (Cycle 50r1)

| Item | Specification |
|---|---|
| System | ECMWF IFS Cycle 50r1 extended-range (sub-seasonal) ensemble, operational from 12 May 2026 |
| Reforecast structure | On-the-fly: for each real-time Mon/Thu 00 UTC run, 11 members × 20 past years (same calendar day). First 50r1 sub-seasonal reforecasts produced 15 May 2026 |
| Reforecast years | 20 years preceding the real-time date (≈2006–2025 for a May-2026 run; the user's 2005–2024 window is fine — pin exact years after first retrieval and record them). `[DECISION-2]` |
| Lead time | Daily steps 0–42 days (retrieve to 46 if available at no extra cost) |
| Grid | Retrieve at **1.0° regular lat-lon** (interpolated by MARS from ~36 km native). Additionally retrieve **precipitation only at 0.4°** for the raw-forecast baseline |
| Access | ECMWF MARS (institutional license) — stream `enfh`/`eefh` (check current stream names in the 50r1 implementation page), type `cf` + `pf`, or the WMO S2S database (1.5°, 24 h fields) as fallback. `[DECISION-3: MARS vs S2S-database access route — confirm what your institution has]` |
| Area subset at retrieval | **30°E–120°E, 15°S–45°N** (predictor domain, see 2.4). Never download global fields — cuts volume ~15× |
| Temporal resolution | 24-hourly (instantaneous fields at 00 UTC valid times; accumulations as step differences). If you want exact IMD-day alignment (03–03 UTC), retrieve 6-hourly `tp` (see 3.2). `[DECISION-4]` |

**Variables to retrieve:**

| Level | Variables | Notes |
|---|---|---|
| Surface / single-level | `tp` (total precip, **accumulated from init — must de-accumulate**), `10u`, `10v`, `2t`, `msl`, `2d` (2 m dewpoint) | Surface specific/relative humidity are not archived; **derive** q2m and RH2m from `2d` + `msl` + `2t` (Magnus formula). Flag this in code comments. |
| 850 hPa | `u, v, t, q, r`, `gh` (geopotential height, optional) | Core monsoon flow level |
| 700 hPa | `u, v, q` | **Added beyond the original plan** — required for a credible vertically integrated moisture transport (moisture is concentrated below 700 hPa; integrating 850→500→250 alone badly under-resolves the integral). `[DECISION-5: add 925 hPa too if retrieval cost allows]` |
| 500 hPa | `u, v, t, q, r`, `gh` | Mid-troposphere |
| 250 hPa | `u, v, t`, (`q` negligible) | Upper-level jet / shear |

**Derived predictors (computed offline, Experiment 4):**
- **VIMT (u,v components):** `VIMT_u = (1/g) ∫ q·u dp`, `VIMT_v = (1/g) ∫ q·v dp`, trapezoidal integration over available levels (surface→250 hPa using q2m at surface pressure ≈ msl; document the level set used).
- **Vertical wind shear:** monsoon shear = `u850 − u250` (zonal; Webster–Yang style uses 850–200, 250 is acceptable — state it), plus vector shear magnitude `|V850 − V250|`.
- Optional cheap extras: 850 hPa relative vorticity and 850–500 hPa thickness (from `gh`), MSE at 850. `[DECISION-6: include or not]`

**Static/auxiliary channels (crucial for downscaling):**
- 0.25° orography (from ETOPO/GTOPO or IMD-grid-matched SRTM aggregate), land–sea mask, sin/cos of day-of-year, normalized lat/lon channels, lead-time scalar. These give the decoder the high-resolution spatial prior (Western Ghats, Himalayan foothills) that the 1° input cannot supply.

### 2.2 Predictand — IMD gridded rainfall

- IMD 0.25° × 0.25° daily gridded rainfall (Pai et al. 2014), 1901–present, from IMD Pune (imdlib Python package or direct download). Grid: 66.5°E–100°E (135 lons) × 6.5°N–38.5°N (129 lats). Land-only; ocean/outside-India cells are missing → build a **binary verification mask** once and use it in the loss and all metrics.
- **Day convention:** IMD "day D" rainfall = gauge accumulation 0830 IST(D−1) → 0830 IST(D) = **03 UTC(D−1) → 03 UTC(D)**. See 3.2 for alignment with ECMWF.
- Secondary verification only: GPM IMERG-Final V07 daily 0.1° (regridded conservatively to 0.25°) for a robustness subsection. `[DECISION-7: include IMERG section or drop]`

### 2.3 Optional large-scale indices (for stratified verification and causality)
- MJO RMM1/RMM2 (BoM), Niño3.4, IOD/DMI, monsoon active/break classification (Rajeevan et al. core-zone index). Used for *evaluation stratification* and optional scalar conditioning — not required for the core pipeline.

### 2.4 Domains
- **Predictor domain:** 30°E–120°E, 15°S–45°N at 1° → 91 × 61 grid. Rationale: captures Somali jet, Arabian Sea and Bay of Bengal moisture sources, equatorial Indian Ocean (MJO), and mid-latitude intrusions.
- **Target domain:** IMD grid, 135 × 129 at 0.25°. Pad/crop to 144 × 132 (divisible by 16) for UNet; keep the mask exact.
- `[DECISION-8: predictor domain box — the above is the recommendation; a smaller 50–110°E, 5°S–40°N variant is a valid ablation]`

---

## 3. Preprocessing Pipeline (ordered, with data contracts)

Store everything as **Zarr** with dims `(init_date, member, lead_day, lat, lon)` for predictors and `(time, lat, lon)` for IMD. Write a `DATA_SPEC.md` documenting every array's dims, units, dtype, and grid — Opus should read it before touching any array.

**Step P1 — Retrieval.** MARS requests per variable-group per year; area/level subset in the request. De-accumulate `tp` to daily totals (`tp[d] − tp[d−1]`, clip ≥ 0). Convert units at ingest: precip → mm/day; q → g/kg; document all conversions.

**Step P2 — Calendar alignment.** For each (init, lead d), the forecast valid day is `init + d`. Match to IMD day using the 03 UTC convention: ECMWF daily precip for valid day D from 00 UTC steps is offset 3 h from the IMD day. Options: (a) accept the 3 h offset (standard for weekly aggregates — the offset washes out; document it); (b) retrieve 6-hourly `tp` and build exact 03–03 UTC sums. **Recommendation: (a) for weekly targets, (b) only if you later do daily-scale week-1 work.** `[DECISION-4 resolved-by-default]`

**Step P3 — Weekly aggregation.** Build weekly accumulated precip (target and raw-forecast baseline) and weekly-mean predictors for weeks 1–6 (d1–7, 8–14, 15–21, 22–28, 29–35, 36–42). Keep daily arrays on disk for the optional ConvLSTM variant.

**Step P4 — Lead- and date-dependent climatology.**
- Model climatology: for each (calendar init-date, lead-week, variable, grid point), mean over **training-years only**, over all 11 members, smoothed across neighboring init dates (±2 inits window or first 3 annual harmonics — pick one, `[DECISION-9]`).
- Observed climatology: same valid-week windows from IMD, training years only.
- Persist both; anomalies = value − climatology.

**Step P5 — Transformations & normalization.**
- Precipitation (input and target): `log1p(x)` (or fourth-root; `[DECISION-10]` — log1p recommended, invert at evaluation).
- All other predictors: standardize per channel per lead-week using training-set mean/std (per grid point OR domain-wide — per-grid-point recommended; `[DECISION-11]`).
- **Leakage rule (hard):** every statistic (climatology, mean/std, quantile maps, CCM maps) is computed from training years only, inside each cross-validation fold.

**Step P6 — Input tensor assembly.** For each sample (init, member, target-week): stack predictor channels (weekly means, anomaly-standardized) bilinearly **upsampled to the 0.25° target grid**, concatenate static channels → tensor `(C, 132, 144)`. Target: `(1, 132, 144)` transformed weekly precip + mask. Rationale for upsample-then-UNet rather than SR-decoder: simpler, robust with mixed static channels, standard in climate downscaling; an SR-style decoder (pixel-shuffle from 1° latent) is a legitimate ablation. `[DECISION-12]`

**Step P7 — Splits.**
- **Recommended:** test = last 5 reforecast years (e.g., 2020–2024), validation = 3 years (2017–2019), train = remainder (~12 years). This mimics operational use (train past → predict future).
- **For the paper's robustness:** additionally run 4-fold leave-5-years-out cross-validation on the final model only (Experiments run once on the fixed split; the headline model re-run across folds).
- `[DECISION-13: fixed chronological split vs full LYO-CV for everything — chronological recommended for cost]`
- **Sample count sanity check (JJAS, weekly targets):** inits with valid weeks in JJAS ≈ 40/yr × 20 yr × 11 members ≈ 8,800 member-samples per target week → adequate for a mid-size UNet, marginal for very deep/transformer models. This is why we stay in the UNet family.

**Step P8 — Ensemble handling.** `[DECISION-14 — important]`
- **Training:** treat each member as an independent sample paired with the same observation ("member-as-sample"). This is 11× data augmentation and implicitly trains toward the conditional mean.
- **Inference:** run every member through the trained network → an 11-member *downscaled ensemble*. Deterministic scores on the downscaled-ensemble mean; probabilistic scores from the member spread.
- Ablation: input = ensemble mean (+ ensemble std as an extra channel). Cheap and often competitive; report both.

---

## 4. Dimensionality Reduction — scrutinized

Original plan says "some kind of PCA will be required." **Scrutiny: for a convolutional model it is not.** CNN/UNets *are* the dimensionality reduction — the encoder learns a compressed representation, and ~25–40 input channels at 132×144 is trivially small for modern GPUs. Flattening fields into PCs would destroy the spatial structure the UNet exploits.

Where PCA/EOF **is** genuinely useful here:
1. **Causality stage (Section 8):** CCM/PCMCI on full grids is intractable and statistically fragile; run causal analysis on (a) per-grid-point predictor↔local-rainfall pairs (as in your `causality.py`), and/or (b) leading EOF PCs of large-scale fields (e.g., first 5 PCs of 850 hPa winds) vs. zone-mean rainfall.
2. **Linear/MOS baselines:** PC regression (predictor PCs → rainfall) is a classic S2S baseline worth one row in the results table.
3. **Scalar conditioning:** MJO RMMs are themselves EOF projections.

`[DECISION-15: adopt "no PCA in the DL pipeline; EOFs only for causality + linear baseline" — recommended yes]`

---

## 5. Experiment Ladder

All experiments share the preprocessing, splits, loss, and evaluation protocol. Change **one thing at a time**; log everything.

**E0 — Baselines (build these FIRST; the paper is unpublishable without them).**
- B1: Climatology (obs weekly climatology as forecast) — the RPSS/ACC reference.
- B2: Raw ECMWF `tp`, bilinear to 0.25°, lead-dependent mean-bias-corrected.
- B3: Raw ECMWF `tp` at 0.4° retrieval, bilinear to 0.25°, bias-corrected (answers "would finer raw input help?").
- B4: Empirical quantile mapping (EQM) per grid point, per lead-week, per ±15-day calendar window, training years only.
- B5: Per-grid-point ridge regression (MOS) on the E2 predictor set (grid-point values + PCs).

**E1 — Precip-only DL downscaling.** Input: ECMWF weekly `tp` (log1p anomaly) + static channels. Architecture: UNet (Section 6). Question answered: how much does spatial ML post-processing alone add over EQM?

**E2 — + Surface predictors.** Add `10u, 10v, 2t, msl, q2m/rh2m` weekly-mean standardized anomalies.

**E3 — + Pressure-level predictors.** Add 850/700/500/250 hPa `u, v, t, q, r(, gh)`. Now ~30–40 channels. Run a channel-importance analysis (permutation importance or integrated gradients) — feeds the paper's interpretability section and motivates E5.

**E4 — + Physics-derived predictors.** Add VIMT_u, VIMT_v, shear (and optional vorticity/thickness). Two sub-experiments: (a) added on top of E3; (b) replacing the raw level winds/q they were derived from (tests whether the physics-informed compression beats raw fields — a nice paper point).

**E5 — Causality-guided model.** CCM (or PCMCI) predictor→rainfall strength maps injected as attention/gating into the UNet, following your `Causality_into_UNET.py` template (CCM maps concatenated at input + `CausalityAttention` gating at the bottleneck). Details in Section 8. Compare against E3/E4 with identical channel sets: the claim is better skill and/or equal skill with fewer channels + interpretability.

**Optional E6 (stretch, only if E1–E5 finish early):** ConvLSTM over daily lead sequence (days 1–14 → week-2 field), or a probabilistic head (CRPS loss / diffusion decoder) for sharp ensemble downscaling. `[DECISION-16: keep as stretch goals, not core]`

**Lead-time treatment across all experiments:** single model conditioned on lead (FiLM embedding of lead-week index) vs. separate model per target week. **Recommendation: separate models for week 2 and week 3 first (simplest, no conditioning bugs), then one lead-conditioned model for weeks 1–6 as the "unified" result.** `[DECISION-17]`

---

## 6. Architecture & Training Specification

**Backbone (all of E1–E5):** UNet
- Encoder: 4 stages, base width 64 (64→128→256→512), two 3×3 convs per stage, GroupNorm(8) + GELU, 2× max-pool.
- Bottleneck: 512 channels (+ CausalityAttention in E5).
- Decoder: transposed-conv or bilinear-upsample+conv, skip connections; final 1×1 conv → 1 channel.
- Lead/date conditioning (if used): FiLM — small MLP on [lead-week one-hot, sin/cos doy] producing per-stage scale/shift.
- Parameters ≈ 8–31 M depending on width — appropriate for ~9k samples.

**Loss:** masked MSE on log1p precip over IMD land cells; ablate (a) MAE, (b) intensity-weighted MSE `w = 1 + α·y_raw/ȳ` to counter blurring of heavy rain (α≈0.2–0.5), (c) MSE + FSS-based spatial term. `[DECISION-18: headline loss — start plain masked MSE, upgrade to weighted if extremes verify poorly]`

**Training protocol:** AdamW (lr 3e-4, wd 1e-4), cosine decay, batch 16–32, ≤150 epochs, early stopping on val CRPS-proxy or masked RMSE (patience 15), light augmentation only if needed (small random crops; NO flips/rotations — orography breaks the symmetry). Mixed precision. 3 random seeds for the headline models; report mean ± range.

**Compute estimate:** at 132×144 with ≤40 channels, one epoch over ~9k samples ≈ 1–3 min on a single A100/RTX-4090-class GPU; a full experiment ≈ 1–4 GPU-hours. The whole ladder including ablations fits comfortably in ~150–300 GPU-hours. Data: 1° subset, all variables/leads/members/inits ≈ 40–120 GB as float32 Zarr (verify after first year retrieved; GRIB originals larger — delete after conversion).

---

## 7. Evaluation Protocol (fixed before any model sees test data)

All metrics: per target week (1–6), JJAS valid weeks, IMD land mask, test years only, computed on **untransformed mm/week**.

**Deterministic (ensemble-mean downscaled forecast):** RMSE, MAE, mean bias, grid-point temporal ACC (anomaly correlation) maps + domain-median ACC, pattern correlation per forecast. Skill vs. lead-week curves (the money figure).

**Categorical:** tercile categories of weekly rainfall (terciles from training-year obs climatology) → RPSS vs. climatology; ≥85th and ≥95th percentile exceedance → BSS, ETS/HSS.

**Spatial realism:** FSS at 1°, 2°, 4° scales for heavy-rain thresholds; radially averaged power spectra of predicted vs. observed fields (demonstrates the DL output isn't over-smoothed — reviewers will ask).

**Probabilistic (downscaled 11-member ensemble):** CRPSS vs. climatology, reliability diagrams for tercile probabilities, spread–error ratio.

**Stratified verification:** (a) active vs. break spells; (b) MJO phases 1–8 (amplitude >1); (c) strong/normal/weak monsoon test years; (d) homogeneous rainfall zones (NW, Central, NE, Peninsular, hilly). Plus 1–2 named case studies from test years (pick the largest week-2 heavy-rain events in 2020–2024).

**Significance:** block bootstrap over years (1000 resamples) for all skill-score differences vs. B4 (EQM); report 95% CIs. A DL model that does not significantly beat EQM at week 2 is a negative result — still publishable but framed differently.

---

## 8. Causality Stage (E5) — adapted from your scripts

Your existing template: `causality.py` computes per-grid-point **Convergent Cross Mapping** (pyEDM, E=3) between each predictor series and the target series, saving one 2-D CCM-strength map per predictor; `Causality_into_UNET.py` (i) concatenates these static maps as extra input channels and (ii) applies a `CausalityAttention` gate at the UNet bottleneck. Keep this architecture — it's simple and the paper can cite it as causality-guided feature weighting.

**Adaptations needed for this project (each is a code change Opus must make):**
1. **Series definition:** use daily JJAS *anomalies*, training years only. Two variants — `[DECISION-19]`:
   - (a) *Observation-side causality:* ERA5 predictors ↔ IMD rainfall (physical causal structure of the real atmosphere), or
   - (b) *Forecast-side causality:* ECMWF hindcast predictors at lead d ↔ IMD rainfall at same valid time (what is causally usable *within the forecast*, per lead).
   - Recommendation: (b) at the target lead — it directly measures exploitable signal in the predictor the network actually sees; (a) as a supplementary comparison figure.
2. **Grid handling:** predictors at 1° vs target at 0.25° — compute CCM on the 1° predictor cell containing each 0.25° target cell (or predictor regridded bilinearly), output maps on the 0.25° UNet grid.
3. **Lags:** run CCM at lags 0, −1 pentad, −2 pentads (or use PCMCI+ with tigramite, ParCorr, τ_max = 3 pentads, as a methodological upgrade; `[DECISION-20: CCM (matches your scripts, nonlinear, no significance test) vs PCMCI+ (lag-resolved, significance-tested, linear-ParCorr default)]`. Pragmatic answer: CCM for the model, PCMCI+ for one analysis figure.)
4. **Leakage:** CCM maps computed inside each fold from training years only.
5. **Normalization of maps:** rescale each CCM map to [0,1] before use as gates.
6. **Ablations that make E5 a real result:** (i) E3/E4 channels + CCM attention vs. without; (ii) channel pruning — drop channels whose domain-mean CCM < threshold, show skill retained with ~half the channels; (iii) random-map placebo (replace CCM maps with shuffled maps — skill gain must vanish, otherwise the attention is just extra capacity).

---

## 9. Engineering Plan (how the work is actually organized)

Repository layout (config-driven; every script runnable on a 1-year mini-subset for testing):
```
s2s-downscaling/
├── DATA_SPEC.md              # dims/units/grids contract — keep current
├── configs/ (yaml per experiment: channels, split, loss, arch)
├── src/data/ (mars_retrieve.py, deaccumulate.py, imd_ingest.py,
│              regrid.py, climatology.py, build_tensors.py)
├── src/models/ (unet.py, film.py, ccm_attention.py, convlstm.py)
├── src/train/ (dataset.py, train.py, infer.py)
├── src/causality/ (ccm_maps.py, pcmci_analysis.py)
├── src/eval/ (metrics.py, stratify.py, bootstrap.py, figures.py)
└── tests/ (shape/unit tests per module; run on mini-subset)
```
Practices: xarray + dask + zarr end-to-end; conda-lock environment; fixed seeds; experiment tracking (wandb or a results CSV with config hash); never overwrite raw data; figures scripted (no manual edits).

**Working with Claude Opus (since final execution is on a model weaker than Fable 5):**
- Feed Opus this document + `DATA_SPEC.md` as project knowledge; work **one module per conversation** with explicit input/output contracts ("write `climatology.py`: input zarr with dims (init_date, member, lead_day, lat, lon), output …").
- Always have Opus write a test on the 1-year mini-subset before running full-scale; verify shapes, units (print min/mean/max after every transform — precip in mm/day should look like precip in mm/day), and NaN masks explicitly.
- Keep every module <300 lines; avoid asking Opus for the whole pipeline in one shot; paste back error tracebacks verbatim.
- Non-negotiable checks Opus must implement as assertions: (1) no test-year data touches any fitted statistic; (2) tp de-accumulation non-negative; (3) IMD mask applied in loss AND metrics; (4) climatology subtraction verified by checking domain-mean anomaly ≈ 0 over training years.

---

## 10. Timeline (suggested, ~6–8 months to submission)

1. **Weeks 1–3:** MARS access sorted; retrieve 2 test years; IMD ingest; DATA_SPEC frozen; alignment verified visually (plot ECMWF vs IMD for a known heavy-rain week).
2. **Weeks 3–6:** full retrieval + preprocessing to Zarr; baselines B1–B5 complete with full evaluation — this alone is a checkpoint (you now know the skill bar).
3. **Weeks 6–10:** E1, E2 + evaluation.
4. **Weeks 10–14:** E3, E4 + channel-importance analysis.
5. **Weeks 14–18:** CCM/PCMCI maps, E5 + placebo/pruning ablations.
6. **Weeks 18–22:** stratified verification, case studies, bootstrap significance, seed-robustness runs, (optional) LYO-CV of headline model, (optional) IMERG robustness.
7. **Weeks 22–28:** figures + manuscript.

---

## 11. Paper Plan

- **Working title:** "Causality-guided deep learning downscaling of ECMWF sub-seasonal reforecasts for Indian monsoon rainfall."
- **Target journals (in order):** *npj Climate and Atmospheric Science*; *Artificial Intelligence for the Earth Systems (AIES)*; *QJRMS*; *Weather and Forecasting*; *Climate Dynamics*. (GRL only if you compress to the E5 headline.)
- **Core figures (≈9):** (1) schematic of pipeline + reforecast structure; (2) skill (ACC, RPSS) vs. lead-week: B2/B4 vs E1–E5; (3) spatial ACC/RPSS maps week 2 & 3, best model vs EQM; (4) example case-study maps (obs / raw / EQM / DL); (5) power spectra + FSS (sharpness); (6) predictor-ladder bar chart (E1→E4) with bootstrap CIs; (7) CCM maps for key predictors + placebo ablation; (8) stratified skill (MJO phase, active/break); (9) reliability diagram + CRPSS of downscaled ensemble.
- **Framing guardrails:** never claim to beat ECMWF at forecasting — the claim is better *calibrated, downscaled, higher-resolution* products from the same dynamical information; be explicit that daily grid-point skill at week 2+ is limited and weekly aggregates are the product.

---

## 12. Decision Register (all flags in one place)

| ID | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | Primary target lead ("2-week lead") | week 2 (d8–14) vs week 3 (d15–21) | Report both as headline; evaluate weeks 1–6 |
| 2 | Exact reforecast year span | 2005–2024 vs 2006–2025 | Whatever 50r1 on-the-fly gives for chosen inits; pin & record |
| 3 | Data access route | MARS direct vs WMO S2S DB (1.5°) | MARS at 1° if licensed; else redesign for 1.5° |
| 4 | Temporal alignment of tp | 24 h steps (3 h offset) vs 6 h exact 03–03 UTC | 24 h for weekly targets |
| 5 | Extra humidity levels for VIMT | +700 only vs +925 & 700 | At least +700; +925 if cheap |
| 6 | Extra derived predictors | vorticity/thickness/MSE | Optional ablation only |
| 7 | IMERG secondary verification | include vs drop | Include if time permits |
| 8 | Predictor domain box | 30–120E,15S–45N vs smaller | Large box; smaller as ablation |
| 9 | Climatology smoothing | ±2-init window vs harmonics | Either; document |
| 10 | Precip transform | log1p vs fourth-root | log1p |
| 11 | Standardization granularity | per-gridpoint vs domain-wide | Per-gridpoint |
| 12 | Input handling | upsample-to-target + UNet vs SR decoder | Upsample + UNet; SR as ablation |
| 13 | Split strategy | fixed chronological vs full LYO-CV | Chronological; LYO-CV on final model |
| 14 | Ensemble handling | member-as-sample vs ens-mean input | Member-as-sample; ens-mean ablation |
| 15 | PCA in DL pipeline | yes vs no | No (EOFs only for causality/linear baseline) |
| 16 | ConvLSTM / probabilistic head | core vs stretch | Stretch |
| 17 | Lead conditioning | per-week models vs FiLM-conditioned | Per-week first, then unified |
| 18 | Loss | MSE vs weighted vs +FSS | Masked MSE first |
| 19 | Causality series | obs-side (ERA5↔IMD) vs forecast-side | Forecast-side at target lead |
| 20 | Causality method | CCM vs PCMCI+ | CCM in model, PCMCI+ for analysis |

---

## 13. Risk Register

- **MARS access/quotas:** retrieval is the long pole; start immediately, retrieve area/level subsets, queue jobs per year.
- **DL fails to beat EQM at week 2:** plausible outcome; mitigations = weekly targets, member augmentation, weighted loss; fallback framing = "value concentrated in windows of opportunity" (stratified results become the story).
- **Blurriness of DL output:** monitor spectra early; escalate loss (weighted → FSS-term → probabilistic head) only if needed.
- **Leakage:** the single most common fatal flaw in this literature; the Section 9 assertions exist to prevent it.
- **Small test sample for extremes:** 5 test years of weekly JJAS extremes is thin; report CIs honestly and lean on the LYO-CV of the headline model.
