"""Metric registry and selection.

This is meant to be the *single obvious place* where you can see:
- which metrics exist
- which ones are included by default
- how to select a subset by name

Metric implementations live in `scorecard_metrics.py`.
"""

from __future__ import annotations

from collections.abc import Iterable

from scorecard_core import Metric
from scorecard_metrics import (
	metric_blh_amplitude,
	metric_blh_day,
	metric_blh_night,
	metric_co2_diurnal_range,
	metric_co2_level,
	metric_co2_vertical_gradient,
)


def build_default_metrics(site_cfg: dict) -> list[Metric]:
	levels = site_cfg.get("levels") or []
	levels = [float(x) for x in levels]

	level_low = levels[0] if levels else 27.0
	level_high = levels[-1] if levels else 207.0
	level_mid = levels[1] if len(levels) > 1 else level_low

	# This explicit list is the scorecard “definition”.
	return [
		Metric("blh_day", "m", metric_blh_day),
		Metric("blh_night", "m", metric_blh_night),
		Metric("blh_amplitude", "m", metric_blh_amplitude),
		Metric(f"co2_DR_{int(level_mid)}m", "PPM", lambda ctx: metric_co2_diurnal_range(ctx, level_m=level_mid)),
		Metric(f"co2_{int(level_low)}m", "PPM", lambda ctx: metric_co2_level(ctx, level_m=level_low, hours_key="daytime")),
		Metric(f"co2_{int(level_high)}m", "PPM", lambda ctx: metric_co2_level(ctx, level_m=level_high, hours_key="daytime")),
		Metric("co2_gradient_day", "PPM/m", lambda ctx: metric_co2_vertical_gradient(ctx, hours_key="daytime")),
		Metric("co2_gradient_night", "PPM/m", lambda ctx: metric_co2_vertical_gradient(ctx, hours_key="nighttime")),
	]


def list_metric_names(metrics: Iterable[Metric]) -> list[str]:
	return [m.name for m in metrics]


def select_metrics(metrics: list[Metric], *, names: list[str] | None) -> list[Metric]:
	"""Select a subset of metrics by `Metric.name`.

	If `names` is None, returns the original list.
	"""
	if names is None:
		return metrics

	wanted = [n.strip() for n in names if n and n.strip()]
	wanted_set = set(wanted)
	selected = [m for m in metrics if m.name in wanted_set]

	missing = [n for n in wanted if n not in {m.name for m in metrics}]
	if missing:
		raise KeyError(f"Unknown metric name(s): {missing}. Available: {list_metric_names(metrics)}")

	return selected
