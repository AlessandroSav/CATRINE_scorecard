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
  A[compute_scorecard.py\nCLI entrypoint] -->|read_json(spec)| B[scorecard_core.py\ncore helpers + loaders]
  A -->|read_json(sites.json)| B
  A --> C[scorecard_metric_registry.py\ndefault metric list + selection]
  C --> D[scorecard_metrics.py\nmetric_* implementations]
  D --> B
  A -->|--no-figures disables| E[scorecard_plots.py\nPNG plotting]
  E --> B

  F[Spec JSON\n(e.g. scorecard_spec.example.json)] --> A
  G[sites.json] --> A

  %% Data sources (paths come from the spec + sites.json)
  H[(Model NetCDF files\n{data_root}/{run_id}_*_{site}.nc)] --> B
  I[(Obs BLH NetCDF\n{dir_obs}/{obs_subdir}/{obs_name}.nc)] --> B
  J[(Obs CO2 tower CSV\n{dir_obs}/{obs_subdir}/*CO2*.csv)] --> B

  %% Outputs
  A --> K[(Terminal table)]
  A -->|spec.out| L[(CSV/JSON table file)]
  E --> M[(PNG figures in fig_dir)]
```

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

