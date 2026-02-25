"""Core utilities for the CATRINE scorecard.

This module intentionally contains only reusable building blocks:
- time-series preparation + alignment
- common bias/RMSE delta computation
- data loading helpers
- small data structures (Context, Metric)

Metric definitions and metric lists live elsewhere.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
import xarray as xr


BiasMode = Literal["abs", "signed"]


def get_hours(spec: dict, key: str) -> tuple[int, int]:
	"""Return (start_hour, end_hour) for a time-of-day window.

	Uses sensible defaults when the spec doesn't define the key:
	- daytime: (7, 20)
	- nighttime: (20, 7)
	"""
	if key == "nighttime":
		default = (20, 7)
	else:
		default = (7, 20)
	val = spec.get(key, default)
	try:
		if val is None or len(val) != 2:
			return default
		return (int(val[0]), int(val[1]))
	except Exception:
		return default


def read_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _to_datetime_index(da: xr.DataArray) -> xr.DataArray:
	"""Ensure `time` coord is a datetime64 index if present."""
	if "time" not in da.coords and "time" not in da.dims:
		return da
	da = da.assign_coords(time=pd.to_datetime(da["time"].values, errors="coerce"))
	return da.dropna(dim="time")


def prep_time_series(da: xr.DataArray) -> xr.DataArray:
	"""Sort time, drop singleton dims, and average duplicate timestamps."""
	if "time" not in da.dims:
		return da

	for dim in list(da.dims):
		if dim != "time" and da.sizes.get(dim, 0) == 1:
			da = da.squeeze(dim, drop=True)

	da = _to_datetime_index(da).sortby("time")

	if not da.get_index("time").is_unique:
		da = da.groupby("time").mean()

	return da


def sel_time_of_day(
	obj: xr.Dataset | xr.DataArray,
	hours: tuple[int, int] | list[int],
) -> xr.Dataset | xr.DataArray:
	"""Select times-of-day by hour window.

	Selection is start-inclusive and end-exclusive.
	If end_hour < start_hour, selection wraps over midnight.
	"""
	if "time" not in obj.dims and "time" not in obj.coords:
		return obj

	if hours is None or len(hours) != 2:
		raise ValueError(f"hours must be a (start_hour, end_hour) pair, got {hours!r}")

	start_hour, end_hour = (int(hours[0]), int(hours[1]))
	hour = obj["time"].dt.hour

	if start_hour == end_hour:
		mask = xr.full_like(hour, True, dtype=bool)
	elif start_hour < end_hour:
		mask = (hour >= start_hour) & (hour < end_hour)
	else:
		mask = (hour >= start_hour) | (hour < end_hour)

	return obj.sel(time=mask)


def align_hourly(*series: xr.DataArray, freq: str = "1h") -> tuple[xr.DataArray, ...]:
	"""Prepare, resample, and align multiple time series on a common grid."""
	prepped = [
		prep_time_series(s).resample(time=freq).mean() if "time" in s.dims else prep_time_series(s)
		for s in series
	]
	aligned = xr.align(*prepped, join="inner")

	valid = None
	for s in aligned:
		m = np.isfinite(s)
		valid = m if valid is None else (valid & m)

	return tuple(s.where(valid, drop=True) for s in aligned)


def _pct_change(new: xr.DataArray, old: xr.DataArray) -> float:
	old_f = float(np.asarray(old).item())
	new_f = float(np.asarray(new).item())
	if old_f == 0.0 or not np.isfinite(old_f):
		return float("nan")
	return float((new_f - old_f) / old_f * 100.0)


def compute_bias_rmse_delta(
	obs: xr.DataArray,
	ctrl: xr.DataArray,
	exp: xr.DataArray,
	*,
	hours: tuple[int, int] | None = None,
	bias_mode: BiasMode = "abs",
) -> dict[str, float]:
	"""Compute control bias mean and % deltas (ctrl->exp) for bias and RMSE."""
	if hours is not None:
		obs = sel_time_of_day(obs, hours)
		ctrl = sel_time_of_day(ctrl, hours)
		exp = sel_time_of_day(exp, hours)

	bias_ctrl = (ctrl - obs).mean("time")
	bias_exp = (exp - obs).mean("time")

	if bias_mode == "abs":
		bias_delta_pct = _pct_change(abs(bias_exp), abs(bias_ctrl))
	elif bias_mode == "signed":
		bias_delta_pct = _pct_change(bias_exp, bias_ctrl)
	else:
		raise ValueError(f"Unknown bias_mode: {bias_mode!r}")

	rmse_ctrl = np.sqrt(((ctrl - obs) ** 2).mean("time"))
	rmse_exp = np.sqrt(((exp - obs) ** 2).mean("time"))
	rmse_delta_pct = _pct_change(rmse_exp, rmse_ctrl)

	return {
		"ctrl_bias_mean": float(np.asarray(bias_ctrl).item()),
		"bias": float(bias_delta_pct),
		"rmse": float(rmse_delta_pct),
	}


def daily_range(da: xr.DataArray) -> xr.DataArray:
	"""Daily max-min range."""
	return da.resample(time="1D").max() - da.resample(time="1D").min()


def _linear_slope_1d(y: np.ndarray, x: np.ndarray, *, min_points: int = 3) -> float:
	y = np.asarray(y)
	x = np.asarray(x)
	mask = np.isfinite(y) & np.isfinite(x)
	if int(mask.sum()) < int(min_points):
		return float("nan")
	return float(np.polyfit(x[mask], y[mask], 1)[0])


def vertical_linear_gradient(
	da: xr.DataArray,
	*,
	height_dim: str = "height",
	min_points: int = 3,
) -> xr.DataArray:
	"""Compute d(var)/d(height) per time via a linear fit across heights."""
	if height_dim not in da.dims:
		raise ValueError(f"Expected dimension '{height_dim}' in {da.dims}")
	x = da[height_dim]
	out = xr.apply_ufunc(
		_linear_slope_1d,
		da,
		x,
		input_core_dims=[[height_dim], [height_dim]],
		output_core_dims=[[]],
		vectorize=True,
		dask="parallelized",
		output_dtypes=[float],
		kwargs={"min_points": min_points},
	)
	name = da.name or "var"
	return out.rename(f"d{name}/d{height_dim}")


def _require_files(pattern: str) -> list[str]:
	files = sorted(glob.glob(pattern))
	if not files:
		raise FileNotFoundError(f"No files matched pattern: {pattern}")
	return files


def _open_by_coords(files: list[str]) -> xr.Dataset:
	if len(files) == 1:
		return xr.open_dataset(files[0]).sortby("time")	
	return xr.open_mfdataset(files, combine="by_coords").sortby("time")


def load_obs_blh(dir_obs: str, site_cfg: dict, start: str, end: str) -> xr.DataArray:
	obs_subdir = site_cfg.get("obs_subdir")
	obs_name = site_cfg.get("obs_name")
	if not obs_subdir or not obs_name:
		raise ValueError("Site config must define obs_subdir and obs_name for BLH observations")
	path = os.path.join(dir_obs, obs_subdir, f"{obs_name}.nc")
	ds = xr.open_dataset(path).sel(time=slice(start, end))
	if "MLH" not in ds:
		raise KeyError(f"Expected variable 'MLH' in {path}")
	return prep_time_series(ds["MLH"].rename("blh_obs"))


def load_model_blh(data_root: str, run_id: str, site: str, start: str, end: str) -> xr.DataArray:
	files = _require_files(os.path.join(data_root, f"{run_id}_*_srf_t0_{site}.nc"))
	ds = _open_by_coords(files).sel(time=slice(start, end))
	if "blh" not in ds:
		raise KeyError(f"Expected variable 'blh' in {files[0]}")
	return prep_time_series(ds["blh"].rename(f"blh_{run_id}"))


def load_obs_co2(dir_obs: str, site_cfg: dict, start: str, end: str) -> xr.DataArray:
	obs_subdir = site_cfg.get("obs_subdir")
	levels = site_cfg.get("levels")
	if not obs_subdir or not levels:
		raise ValueError("Site config must define obs_subdir and levels for CO2 observations")

	folder = os.path.join(dir_obs, obs_subdir)
	csvs = [os.path.join(folder, f) for f in os.listdir(folder) if "CO2" in f and f.endswith(".csv")]
	if not csvs:
		raise FileNotFoundError(f"No CO2 observation CSV found in {folder}")

	df = pd.read_csv(csvs[0])
	df.columns.values[0] = "time"
	df = df.set_index("time")
	obs_ds = df.to_xarray().sortby("time")
	obs_ds = obs_ds.assign_coords(time=pd.to_datetime(obs_ds["time"].values, errors="coerce")).dropna(dim="time")
	obs_ds = obs_ds.sel(time=slice(start, end))

	levels_i = [int(h) for h in levels]
	series: list[xr.DataArray] = []
	for h in levels_i:
		v = f"co2_{h}m"
		if v not in obs_ds:
			raise KeyError(f"Expected variable '{v}' in {csvs[0]}")
		series.append(obs_ds[v])

	da = xr.concat(series, dim="height").assign_coords(height=("height", levels_i)).rename("co2")
	return prep_time_series(da)


def load_model_co2(data_root: str, run_id: str, site: str, levels: list[float], start: str, end: str) -> xr.DataArray:
	files = _require_files(os.path.join(data_root, f"{run_id}_*_z_t0_{site}.nc"))
	ds = _open_by_coords(files).sel(time=slice(start, end))
	if "co2" not in ds:
		raise KeyError(f"Expected variable 'co2' in {files[0]}")
	da = ds["co2"].interp(height=np.asarray(levels, dtype=float)).rename(f"co2_{run_id}")
	return prep_time_series(da)


@dataclass(frozen=True)
class Metric:
	name: str
	unit: str | None
	fn: Callable[["Context"], dict[str, float]]


@dataclass(frozen=True)
class Context:
	spec: dict
	site_cfg: dict
	obs_blh: xr.DataArray
	ctrl_blh: xr.DataArray
	exp_blh: xr.DataArray
	obs_co2: xr.DataArray
	ctrl_co2: xr.DataArray
	exp_co2: xr.DataArray
