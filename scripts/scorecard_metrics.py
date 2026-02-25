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
