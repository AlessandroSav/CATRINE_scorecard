"""Plotting utilities for the CATRINE scorecard.

Keeps figure generation separate from metric computation.
All plots are saved as PNGs.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from scorecard_core import Context, align_hourly, sel_time_of_day


def _start_date_tag(spec: dict) -> str:
	dt = pd.to_datetime(spec["start"])
	return dt.strftime("%Y%m%d")


def _fig_prefix(spec: dict) -> str:
	site = spec["site"]
	ctrl = spec["control"]
	exp = spec["experiment"]	
	date_tag = _start_date_tag(spec)
	return f"{site}_{ctrl}_{exp}_{date_tag}"


def ensure_fig_dir(fig_dir: str) -> str:
	os.makedirs(fig_dir, exist_ok=True)
	return fig_dir


def save_scorecard_png(df, *, title: str, out_path: str) -> None:
	"""Render a DataFrame as a PNG image using matplotlib."""
	rows = len(df) + 1
	cols = len(df.columns)

	# Size tuned for readability; scales with number of rows.
	fig_w = max(8.0, cols * 2.2)
	fig_h = max(2.0, rows * 0.6)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))
	ax.axis("off")

	# Format numbers similarly to the notebook.
	cell_text = []
	for _, r in df.iterrows():
		row = []
		for c in df.columns:
			v = r[c]
			if c == "Metric":
				row.append(str(v))
			else:
				try:
					n = float(v)
					if np.isnan(n):
						row.append("nan")
					elif c == "Ctrl bias":
						row.append(f"{n:+.2f}")
					else:
						row.append(f"{n:+.2f}%")
				except Exception:
					row.append(str(v))
		cell_text.append(row)

	# Prepare background colors for delta columns, similar to:
	# .background_gradient(cmap="PiYG_r", vmin=-70, vmax=70, subset=["Δ bias %", "Δ RMSE %"]) in the notebook.
	delta_cols = ["Δ bias %", "Δ RMSE %"]
	delta_idx = [i for i, c in enumerate(df.columns) if c in delta_cols]
	# Create a full cell-colors matrix (rows x cols), excluding header row.
	default_color = (1, 1, 1, 1)
	cell_colors = [[default_color for _ in range(cols)] for _ in range(len(df))]

	cmap = cm.get_cmap("PiYG_r")
	norm = TwoSlopeNorm(vmin=-70.0, vcenter=0.0, vmax=70.0)
	for ridx in range(len(df)):
		for cidx in delta_idx:
			val = df.iloc[ridx, cidx]
			try:
				n = float(val)
				if np.isfinite(n):
					n = float(np.clip(n, -70.0, 70.0))
					cell_colors[ridx][cidx] = cmap(norm(n))
			except Exception:
				# leave default
				pass

	table = ax.table(
		cellText=cell_text,
		colLabels=list(df.columns),
		cellLoc="center",
		loc="center",
	)

	# Apply header styling
	for c in range(cols):
		header_cell = table[(0, c)]
		header_cell.set_facecolor((0.92, 0.92, 0.92, 1))
		header_cell.set_text_props(weight="bold")

	# Apply per-cell background colors (data rows start at row index 1 in the table)
	for r in range(1, rows):
		for c in range(cols):
			table[(r, c)].set_facecolor(cell_colors[r - 1][c])

	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.scale(1.0, 1.2)

	ax.set_title(title, fontsize=12, loc="left", pad=12)
	fig.tight_layout()
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def save_blh_plots(ctx: Context, *, fig_dir: str) -> list[str]:
	"""Save BLH time series + diurnal cycle."""
	prefix = _fig_prefix(ctx.spec)
	daytime = tuple(ctx.spec.get("daytime", [7, 20]))
	_ = daytime  # keeps parity with other plot functions (may be used later)

	obs, ctrl, exp = align_hourly(ctx.obs_blh, ctx.ctrl_blh, ctx.exp_blh)

	paths: list[str] = []

	# Time series
	fig, ax = plt.subplots(figsize=(10, 4))
	obs.plot(ax=ax, label="obs", marker="*", linestyle="-")
	ctrl.plot(ax=ax, label="ctrl", marker="*", linestyle="-")
	exp.plot(ax=ax, label="exp", marker="*", linestyle="-")
	ax.set_xlabel("Time")
	ax.set_ylabel("Boundary Layer Height (m)")
	ax.legend()
	out_path = os.path.join(fig_dir, f"blh_timeseries_{prefix}.png")
	fig.tight_layout()
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	paths.append(out_path)

	# Diurnal cycle
	fig, ax = plt.subplots(figsize=(7, 4))
	obs.groupby("time.hour").mean("time").plot(ax=ax, x="hour", label="obs")
	ctrl.groupby("time.hour").mean("time").plot(ax=ax, x="hour", label="ctrl")
	exp.groupby("time.hour").mean("time").plot(ax=ax, x="hour", label="exp")
	ax.set_title("Diurnal cycle of BLH")
	ax.set_xlabel("Hour")
	ax.set_ylabel("Boundary Layer Height (m)")
	ax.legend()
	out_path = os.path.join(fig_dir, f"blh_diurnal_{prefix}.png")
	fig.tight_layout()
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	paths.append(out_path)

	return paths


def save_co2_timeseries_and_diurnal(ctx: Context, *, level_m: float, fig_dir: str) -> list[str]:
	"""Save CO2 time series + diurnal cycle at a given tower level."""
	prefix = _fig_prefix(ctx.spec)
	obs, ctrl, exp = align_hourly(ctx.obs_co2, ctx.ctrl_co2, ctx.exp_co2)

	paths: list[str] = []

	fig, ax = plt.subplots(figsize=(10, 4))
	obs.sel(height=level_m).plot(ax=ax, label="obs", marker="*", linestyle="-")
	ctrl.sel(height=level_m).plot(ax=ax, label="ctrl", marker="*", linestyle="-")
	exp.sel(height=level_m).plot(ax=ax, label="exp", marker="*", linestyle="-")
	ax.set_title(f"CO2 at {int(level_m)} m")
	ax.set_xlabel("Time")
	ax.set_ylabel("CO2 Concentration (PPM)")
	ax.legend()
	out_path = os.path.join(fig_dir, f"co2_timeseries_{int(level_m)}m_{prefix}.png")
	fig.tight_layout()
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	paths.append(out_path)

	fig, ax = plt.subplots(figsize=(7, 4))
	obs.sel(height=level_m).groupby("time.hour").mean("time").plot(ax=ax, x="hour", label="obs")
	ctrl.sel(height=level_m).groupby("time.hour").mean("time").plot(ax=ax, x="hour", label="ctrl")
	exp.sel(height=level_m).groupby("time.hour").mean("time").plot(ax=ax, x="hour", label="exp")
	ax.set_title(f"Diurnal cycle of CO2 at {int(level_m)} m")
	ax.set_xlabel("Hour")
	ax.set_ylabel("CO2 Concentration (PPM)")
	ax.legend()
	out_path = os.path.join(fig_dir, f"co2_diurnal_{int(level_m)}m_{prefix}.png")
	fig.tight_layout()
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	paths.append(out_path)

	return paths


def save_co2_mean_profiles(ctx: Context, *, fig_dir: str) -> list[str]:
	"""Save mean daytime/nighttime CO2 profile plot with simple line fits."""
	prefix = _fig_prefix(ctx.spec)
	daytime = tuple(ctx.spec.get("daytime", [7, 20]))
	nighttime = tuple(ctx.spec.get("nighttime", [20, 7]))

	# Align on hourly to avoid time-mismatch issues.
	obs, ctrl, exp = align_hourly(ctx.obs_co2, ctx.ctrl_co2, ctx.exp_co2)

	obs_day = sel_time_of_day(obs, daytime)
	ctrl_day = sel_time_of_day(ctrl, daytime)
	exp_day = sel_time_of_day(exp, daytime)
	obs_night = sel_time_of_day(obs, nighttime)
	ctrl_night = sel_time_of_day(ctrl, nighttime)
	exp_night = sel_time_of_day(exp, nighttime)

	fig, ax = plt.subplots(figsize=(6.5, 6))

	def _plot_profile(da, label, style, color):
		m = da.mean("time")
		ax.plot(m.values, m["height"].values, style, label=label, color=color)

	_plot_profile(obs_day, "obs day", "-*", "blue")
	_plot_profile(ctrl_day, "ctrl day", "-*", "green")
	_plot_profile(exp_day, "exp day", "-*", "red")
	_plot_profile(obs_night, "obs night", "-o", "blue")
	_plot_profile(ctrl_night, "ctrl night", "-o", "green")
	_plot_profile(exp_night, "exp night", "-o", "red")

	def _fit_line(mean_profile, label, color):
		x = np.asarray(mean_profile.values, dtype=float)
		y = np.asarray(mean_profile["height"].values, dtype=float)
		mask = np.isfinite(x) & np.isfinite(y)
		if int(mask.sum()) < 2:
			return
		slope, intercept = np.polyfit(y[mask], x[mask], 1)
		line_x = slope * y + intercept
		ax.plot(line_x, y, "--", color=color, label=f"{label} fit")

	_fit_line(obs_day.mean("time"), "obs day", "blue")
	_fit_line(ctrl_day.mean("time"), "ctrl day", "green")
	_fit_line(exp_day.mean("time"), "exp day", "red")
	_fit_line(obs_night.mean("time"), "obs night", "blue")
	_fit_line(ctrl_night.mean("time"), "ctrl night", "green")
	_fit_line(exp_night.mean("time"), "exp night", "red")

	ax.set_ylabel("Height (m)")
	ax.set_xlabel("CO2 Concentration (PPM)")
	ax.set_title("Mean daytime and nighttime CO2 profile")
	ax.legend(fontsize=8)
	fig.tight_layout()
	out_path = os.path.join(fig_dir, f"co2_mean_profiles_{prefix}.png")
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)

	return [out_path]
