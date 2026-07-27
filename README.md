# s2s

Directory layout:

```
notebooks/            Analysis and modelling notebooks
scripts/              Standalone Python scripts and shared utilities
data/
  raw/                Source datasets (IMD rainfall, ECMWF S2S reforecast)
  processed/          Derived datasets (sorted/final reforecast, residuals, feature series)
  cache/              Training caches built by `prepare` (.npz files and .npy cache dirs)
results/
  models/             Checkpoints (.pt), skill/correlation maps (.nc), fold and loss JSON
  figures/            Standalone figures
  diagnostics/        Per-run correlation-diagnostic output directories
archive/              Scratch and superseded files, kept for reference
```

Paths are relative: **notebooks** reference data as `../data/...` (Jupyter's working
directory is `notebooks/`), **scripts** are run from the repo root and reference
`data/...` directly, e.g.

```bash
python scripts/s2s_corr_maps.py
```
