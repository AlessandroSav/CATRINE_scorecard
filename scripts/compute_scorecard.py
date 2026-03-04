"""Compute CATRINE scorecard metrics.

This is a thin CLI wrapper.

- Core utilities + data loading: `scorecard_core.py`
- Metric implementations: `scorecard_metrics.py`
- Default metric list + selection: `scorecard_metric_registry.py`

Typical usage:

	python scripts/compute_scorecard.py scorecard_spec_cabauw_2022.json
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from scorecard_core import (
	Context,
	load_model_blh,
	load_model_co2,
	load_model_co2_flx,
	load_model_co2_srf_flx,
	load_obs_blh,
	load_obs_co2,
	load_obs_co2_flx,
	load_obs_co2_srf_flx,
	read_json,
)
from scorecard_metric_registry import build_default_metrics, list_metric_names, select_metrics


def main() -> None:
	parser = argparse.ArgumentParser(description="Compute CATRINE scorecard metrics")
	parser.add_argument(
		"spec",
		nargs="?",
		default="scorecard_spec_cabauw_2022.json",
		help="Path to scorecard spec JSON (default: scorecard_spec_cabauw_2022.json)",
	)
	parser.add_argument(
		"--sites",
		default=os.path.join(os.path.dirname(__file__), "sites.json"),
		help="Path to sites.json (default: scripts/sites.json)",
	)
	parser.add_argument(
		"--metrics",
		default=None,
		help="Comma-separated metric names to run (default: run registry default set)",
	)
	parser.add_argument(
		"--list-metrics",
		action="store_true",
		help="List available metric names and exit",
	)
	parser.add_argument(
		"--no-figures",
		action="store_true",
		help="Skip writing PNG figures to figures/",
	)
	args = parser.parse_args()

	spec = read_json(args.spec)
	sites = read_json(args.sites)
	site = spec["site"]
	if site not in sites:
		raise KeyError(f"Site {site!r} not found in {args.sites}")
	site_cfg = sites[site]

	# Metric definitions can be listed without touching any data files.
	metrics = build_default_metrics(site_cfg)
	if args.list_metrics:
		for n in list_metric_names(metrics):
			print(n)
		return

	selected_names = None
	if args.metrics:
		selected_names = [p.strip() for p in args.metrics.split(",")]
	metrics = select_metrics(metrics, names=selected_names)
	metric_names = {m.name for m in metrics}
	need_surface_flux = "co2_surface_flux" in metric_names
	need_flux_profile = any(n.startswith("co2_flux") for n in metric_names)

	start = spec["start"]
	end = spec["end"]
	data_root = spec["data_root"]
	dir_obs = spec["dir_obs"]
	control = spec["control"]
	experiment = spec["experiment"]

	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	fig_dir = spec.get("fig_dir")
	if not fig_dir:
		fig_dir = os.path.join(repo_root, "figures")
	elif not os.path.isabs(fig_dir):
		fig_dir = os.path.join(repo_root, fig_dir)

	obs_blh = load_obs_blh(dir_obs, site_cfg, start, end)
	ctrl_blh = load_model_blh(data_root, control, site, start, end)
	exp_blh = load_model_blh(data_root, experiment, site, start, end)

	obs_co2_srf_flx = None
	ctrl_co2_srf_flx = None
	exp_co2_srf_flx = None
	if need_surface_flux:
		obs_co2_srf_flx = load_obs_co2_srf_flx(dir_obs, site_cfg, start, end)
		ctrl_co2_srf_flx = load_model_co2_srf_flx(data_root, control, site, start, end)
		exp_co2_srf_flx = load_model_co2_srf_flx(data_root, experiment, site, start, end)

	levels = [float(x) for x in (site_cfg.get("levels") or [])]
	obs_co2 = load_obs_co2(dir_obs, site_cfg, start, end)
	ctrl_co2 = load_model_co2(data_root, control, site, levels, start, end)
	exp_co2 = load_model_co2(data_root, experiment, site, levels, start, end)

	obs_co2_flx = None
	ctrl_co2_flx = None
	exp_co2_flx = None
	if need_flux_profile:
		flux_levels = site_cfg.get("obs_co2_flx_levels") or site_cfg.get("flx_levels") or levels
		flux_levels = [float(x) for x in (flux_levels or [])]
		obs_co2_flx = load_obs_co2_flx(dir_obs, site_cfg, start, end)
		ctrl_co2_flx = load_model_co2_flx(data_root, control, site, flux_levels, start, end)
		exp_co2_flx = load_model_co2_flx(data_root, experiment, site, flux_levels, start, end)

	ctx = Context(
		spec=spec,
		site_cfg=site_cfg,
		obs_blh=obs_blh,
		ctrl_blh=ctrl_blh,
		exp_blh=exp_blh,
		obs_co2=obs_co2,
		ctrl_co2=ctrl_co2,
		exp_co2=exp_co2,
		obs_co2_srf_flx=obs_co2_srf_flx,
		ctrl_co2_srf_flx=ctrl_co2_srf_flx,
		exp_co2_srf_flx=exp_co2_srf_flx,
		obs_co2_flx=obs_co2_flx,
		ctrl_co2_flx=ctrl_co2_flx,
		exp_co2_flx=exp_co2_flx,
	)

	rows: list[dict[str, object]] = []
	for m in metrics:
		d = m.fn(ctx)
		label = m.name if not m.unit else f"{m.name} [{m.unit}]"
		rows.append(
			{
				"Metric": label,
				"Ctrl bias": float(d.get("ctrl_bias_mean", np.nan)),
				"Δ bias %": float(d.get("bias", np.nan)),
				"Δ RMSE %": float(d.get("rmse", np.nan)),
			}
		)
	df = pd.DataFrame(rows, columns=["Metric", "Ctrl bias", "Δ bias %", "Δ RMSE %"])

	title = f"Site: {site} | {start} → {end} | control: {control} | experiment: {experiment}"
	print(title)
	print(df.to_string(index=False, justify="left"))

	# Figures
	if not args.no_figures:
		from scorecard_plots import (
			ensure_fig_dir,
			save_blh_plots,
			save_co2_flux_timeseries,
			save_co2_mean_profiles,
			save_co2_surface_flux_timeseries,
			save_co2_timeseries_and_diurnal,
			save_scorecard_png,
		)

		fig_dir = ensure_fig_dir(fig_dir)
		start_date = pd.to_datetime(start).strftime("%Y%m%d")
		scorecard_path = os.path.join(fig_dir, f"scorecard_{site}_{control}_{experiment}_{start_date}.png")
		save_scorecard_png(df, title=title, out_path=scorecard_path)

		# Save a small set of diagnostic plots (mirrors the notebook's core plots)
		_ = save_blh_plots(ctx, fig_dir=fig_dir)
		levels = site_cfg.get("levels") or []
		levels = [float(x) for x in levels]
		level_plot = levels[1] if len(levels) > 1 else (levels[0] if levels else 67.0)
		_ = save_co2_timeseries_and_diurnal(ctx, level_m=level_plot, fig_dir=fig_dir)
		_ = save_co2_mean_profiles(ctx, fig_dir=fig_dir)

		# Additional diagnostics (mirrors the notebook snippets); will be skipped
		# automatically when the corresponding datasets weren't loaded.
		_ = save_co2_flux_timeseries(ctx, level_m=180.0, fig_dir=fig_dir)
		_ = save_co2_flux_timeseries(ctx, level_m=5.0, fig_dir=fig_dir)

		_ = save_co2_surface_flux_timeseries(ctx, fig_dir=fig_dir)

	out = spec.get("out")
	if out:
		out = os.path.expanduser(out)
		os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
		if out.endswith(".csv"):
			df.to_csv(out, index=False)
		elif out.endswith(".json"):
			df.to_json(out, orient="records", indent=2)
		else:
			# default to CSV if user gives a bare path
			df.to_csv(out, index=False)


if __name__ == "__main__":
	main()

