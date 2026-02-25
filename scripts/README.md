# Scripts/module map

This folder contains the CLI entrypoint and the small, separated modules it uses.

## High-level flow

1. `compute_scorecard.py` reads a *spec* JSON and `sites.json`.
2. It loads model + observation datasets via `scorecard_core.py`.
3. It builds a metric list via `scorecard_metric_registry.py`.
4. Each metric function (in `scorecard_metrics.py`) computes score numbers using helpers from `scorecard_core.py`.
5. Optionally, plots are written via `scorecard_plots.py`.
6. Optionally, the score table is written to disk via the `out` spec key.

## Visual schematic (module dependencies)

```mermaid
flowchart TD
  Spec["Spec JSON"] --> A["compute_scorecard.py (CLI)"]
  Sites["sites.json"] --> A

  A --> Core["scorecard_core.py (loaders + helpers)"]
  A --> Reg["scorecard_metric_registry.py (default metric list)"]
  Reg --> Metrics["scorecard_metrics.py (metric functions)"]
  Metrics --> Core

  A --> Plots["scorecard_plots.py (PNG plots, optional)"]
  Plots --> Core

  Model[("Model NetCDF files")] --> Core
  ObsBLH[("Obs BLH NetCDF")] --> Core
  ObsCO2[("Obs CO2 CSV")] --> Core

  A --> Term[("Terminal table")]
  A --> Out[("CSV/JSON table file (optional)")]
  Plots --> PNG[("PNG figures")]
```

Notes (actual filename patterns used by loaders):
- Model BLH: `{data_root}/{run_id}_*_srf_t0_{site}.nc`
- Model CO2: `{data_root}/{run_id}_*_z_t0_{site}.nc`
- Obs BLH: `{dir_obs}/{obs_subdir}/{obs_name}.nc`
- Obs CO2: `{dir_obs}/{obs_subdir}/*CO2*.csv`

## What lives where

- `compute_scorecard.py`
  - Argument parsing (`--list-metrics`, `--metrics`, `--no-figures`)
  - Loads datasets, runs metrics, prints the table, writes figures and/or `out`

- `scorecard_core.py`
  - Data loading: `load_model_blh`, `load_model_co2`, `load_obs_blh`, `load_obs_co2`
  - Time handling and selection: `align_hourly`, `sel_time_of_day`, `get_hours`
  - Shared math: `compute_bias_rmse_delta`, `daily_range`, `vertical_linear_gradient`
  - Lightweight container types: `Context`, `Metric`

- `scorecard_metric_registry.py`
  - The single place defining the default metric set (`build_default_metrics`)
  - Selection by name (`select_metrics`)

- `scorecard_metrics.py`
  - Metric functions only (`metric_*`) that return the dict consumed by the CLI

- `scorecard_plots.py`
  - Plotting functions writing `.png` (called only when figures are enabled)

