"""Metric definitions for the CATRINE scorecard.

This file should contain only `metric_*` functions.
The *selection/list* of which metrics to run lives in `scorecard_metric_registry.py`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from scorecard_core import (
	Context,
	align_hourly,
	compute_bias_rmse_delta,
	daily_range,
	get_hours,
	sel_time_of_day,
	vertical_linear_gradient,
)


def metric_blh_day(ctx: Context) -> dict[str, float]:
	obs, ctrl, exp = align_hourly(ctx.obs_blh, ctx.ctrl_blh, ctx.exp_blh)
	hours = get_hours(ctx.spec, "daytime")
	return compute_bias_rmse_delta(obs, ctrl, exp, hours=hours, bias_mode=ctx.spec.get("bias_mode", "abs"))


def metric_blh_night(ctx: Context) -> dict[str, float]:
	obs, ctrl, exp = align_hourly(ctx.obs_blh, ctx.ctrl_blh, ctx.exp_blh)
	hours = get_hours(ctx.spec, "nighttime")
	return compute_bias_rmse_delta(obs, ctrl, exp, hours=hours, bias_mode=ctx.spec.get("bias_mode", "abs"))


def metric_blh_amplitude(ctx: Context) -> dict[str, float]:
	obs, ctrl, exp = align_hourly(ctx.obs_blh, ctx.ctrl_blh, ctx.exp_blh)
	obs_amp = daily_range(obs)
	ctrl_amp = daily_range(ctrl)
	exp_amp = daily_range(exp)

	obs_amp, ctrl_amp, exp_amp = xr.align(obs_amp, ctrl_amp, exp_amp, join="inner")
	valid = np.isfinite(obs_amp) & np.isfinite(ctrl_amp) & np.isfinite(exp_amp)
	obs_amp = obs_amp.where(valid, drop=True)
	ctrl_amp = ctrl_amp.where(valid, drop=True)
	exp_amp = exp_amp.where(valid, drop=True)

	return compute_bias_rmse_delta(obs_amp, ctrl_amp, exp_amp, bias_mode=ctx.spec.get("bias_mode", "abs"))


def metric_co2_level(ctx: Context, *, level_m: float, hours_key: str = "daytime") -> dict[str, float]:
	obs, ctrl, exp = align_hourly(ctx.obs_co2, ctx.ctrl_co2, ctx.exp_co2)
	hours = get_hours(ctx.spec, hours_key)
	obs_l = obs.sel(height=level_m)
	ctrl_l = ctrl.sel(height=level_m)
	exp_l = exp.sel(height=level_m)
	return compute_bias_rmse_delta(obs_l, ctrl_l, exp_l, hours=hours, bias_mode=ctx.spec.get("bias_mode", "abs"))


def metric_co2_diurnal_range(ctx: Context, *, level_m: float) -> dict[str, float]:
	obs, ctrl, exp = align_hourly(ctx.obs_co2, ctx.ctrl_co2, ctx.exp_co2)
	obs_dr = daily_range(obs.sel(height=level_m))
	ctrl_dr = daily_range(ctrl.sel(height=level_m))
	exp_dr = daily_range(exp.sel(height=level_m))

	obs_dr, ctrl_dr, exp_dr = xr.align(obs_dr, ctrl_dr, exp_dr, join="inner")
	valid = np.isfinite(obs_dr) & np.isfinite(ctrl_dr) & np.isfinite(exp_dr)
	obs_dr = obs_dr.where(valid, drop=True)
	ctrl_dr = ctrl_dr.where(valid, drop=True)
	exp_dr = exp_dr.where(valid, drop=True)

	return compute_bias_rmse_delta(obs_dr, ctrl_dr, exp_dr, bias_mode=ctx.spec.get("bias_mode", "abs"))


def metric_co2_vertical_gradient(ctx: Context, *, hours_key: str) -> dict[str, float]:
	obs, ctrl, exp = align_hourly(ctx.obs_co2, ctx.ctrl_co2, ctx.exp_co2)
	hours = get_hours(ctx.spec, hours_key)

	obs_g = vertical_linear_gradient(sel_time_of_day(obs, hours))
	ctrl_g = vertical_linear_gradient(sel_time_of_day(ctrl, hours))
	exp_g = vertical_linear_gradient(sel_time_of_day(exp, hours))

	obs_g, ctrl_g, exp_g = xr.align(obs_g, ctrl_g, exp_g, join="inner")
	valid = np.isfinite(obs_g) & np.isfinite(ctrl_g) & np.isfinite(exp_g)
	obs_g = obs_g.where(valid, drop=True)
	ctrl_g = ctrl_g.where(valid, drop=True)
	exp_g = exp_g.where(valid, drop=True)

	return compute_bias_rmse_delta(obs_g, ctrl_g, exp_g, bias_mode=ctx.spec.get("bias_mode", "abs"))

def metric_co2_surface_flux(ctx: Context) -> dict[str, float]:
	if ctx.obs_co2_srf_flx is None or ctx.ctrl_co2_srf_flx is None or ctx.exp_co2_srf_flx is None:
		raise ValueError("CO2 surface flux metric requires obs/model surface flux datasets to be loaded")
	obs, ctrl, exp = align_hourly(ctx.obs_co2_srf_flx, ctx.ctrl_co2_srf_flx, ctx.exp_co2_srf_flx)
	obs_surf, ctrl_surf, exp_surf = xr.align(obs, ctrl, exp, join="inner")
	valid = np.isfinite(obs_surf) & np.isfinite(ctrl_surf) & np.isfinite(exp_surf)
	obs_surf = obs_surf.where(valid, drop=True)
	ctrl_surf = ctrl_surf.where(valid, drop=True)
	exp_surf = exp_surf.where(valid, drop=True)

	return compute_bias_rmse_delta(obs_surf, ctrl_surf, exp_surf, bias_mode=ctx.spec.get("bias_mode", "abs"))

def metric_co2_flux(ctx: Context, *, level_m: float) -> dict[str, float]:
	if ctx.obs_co2_flx is None or ctx.ctrl_co2_flx is None or ctx.exp_co2_flx is None:
		raise ValueError("CO2 flux metric requires obs/model flux datasets to be loaded")
	obs, ctrl, exp = align_hourly(ctx.obs_co2_flx, ctx.ctrl_co2_flx, ctx.exp_co2_flx, label="right")

	def _select_or_interp_height(da: xr.DataArray, target_m: float) -> xr.DataArray:
		if "height" not in da.dims:
			return da
		# Prefer exact selection if the target is present. This avoids xarray/scipy
		# interpolation propagating NaNs from other (possibly missing) height levels.
		heights = np.asarray(da["height"].values, dtype=float)
		target_m = float(target_m)
		if heights.size:
			idx = int(np.argmin(np.abs(heights - target_m)))
			if np.isclose(heights[idx], target_m, atol=1e-6, rtol=0.0):
				return da.sel(height=float(heights[idx]))
		# If interpolation is needed, drop any levels that are entirely missing.
		da2 = da.dropna(dim="height", how="all")
		return da2.interp(height=target_m)

	obs_l = _select_or_interp_height(obs, level_m)
	ctrl_l = _select_or_interp_height(ctrl, level_m)
	exp_l = _select_or_interp_height(exp, level_m)
	return compute_bias_rmse_delta(obs_l, ctrl_l, exp_l, bias_mode=ctx.spec.get("bias_mode", "abs"))