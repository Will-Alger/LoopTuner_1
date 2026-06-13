"""Explain what LoopTuner saw and why it did (or didn't) suggest changes.

Produces a structured report covering:

* how many raw Nightscout records were parsed (CGM, boluses, carbs, temp basals)
* the analysis-window dataset: count, date span, coverage per hour, how many
  windows actually contained carb absorption
* model convergence
* a per-setting decision breakdown: for each of ISF / CR / basal, how many of
  the 24 hours were left unchanged (consistent with current), changed
  (data-supported), or pooled because the hour was too sparse to decide.

This is the answer to "why did it barely suggest anything?" -- usually either
the settings already agree with the data, or there isn't enough signal per hour
to clear the confidence bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .preprocess import (
    parse_boluses,
    parse_carbs,
    parse_entries,
    parse_temp_basals,
)

log = logging.getLogger("looptuner")


@dataclass
class DataReport:
    n_cgm: int
    n_cgm_used: int
    n_boluses: int
    n_carb_entries: int
    n_temp_basals: int
    n_windows: int
    span_days: float
    date_start: str
    date_end: str
    windows_per_hour_min: int
    windows_per_hour_med: float
    windows_per_hour_max: int
    n_carb_windows: int
    decisions: dict          # setting -> {changed, unchanged, sparse}
    convergence: dict
    obs_sd: float


def _decision_counts(recs_list) -> dict:
    changed = sum(1 for r in recs_list if r.confident)
    sparse = sum(1 for r in recs_list if not r.confident and "sparse" in r.note)
    unchanged = len(recs_list) - changed - sparse
    return {"changed": changed, "unchanged": unchanged, "sparse": sparse}


def build_report(entries, treatments, dataset, recommendations, fit) -> DataReport:
    n_cgm = len(parse_entries(entries))
    boluses = parse_boluses(treatments)
    carbs = parse_carbs(treatments, 180.0)
    temps = parse_temp_basals(treatments)

    counts = np.bincount(dataset.hour, minlength=24)
    span = (dataset.window_start.max() - dataset.window_start.min())
    span_days = span / np.timedelta64(1, "D") if len(dataset) else 0.0

    report = DataReport(
        n_cgm=len(entries),
        n_cgm_used=int(n_cgm),
        n_boluses=len(boluses),
        n_carb_entries=len(carbs),
        n_temp_basals=len(temps),
        n_windows=len(dataset),
        span_days=round(float(span_days), 1),
        date_start=str(dataset.window_start.min())[:16] if len(dataset) else "n/a",
        date_end=str(dataset.window_start.max())[:16] if len(dataset) else "n/a",
        windows_per_hour_min=int(counts.min()),
        windows_per_hour_med=float(np.median(counts)),
        windows_per_hour_max=int(counts.max()),
        n_carb_windows=int((dataset.carb_act > 0).sum()),
        decisions={
            "ISF": _decision_counts(recommendations.isf),
            "Carb ratio": _decision_counts(recommendations.carb_ratio),
            "Basal": _decision_counts(recommendations.basal),
        },
        convergence=fit.diagnostics,
        obs_sd=round(fit.obs_sd, 1),
    )
    _log_report(report)
    return report


def _log_report(r: DataReport) -> None:
    log.info("Parsed %d CGM records (%d valid), %d boluses, %d carb entries, "
             "%d temp basals", r.n_cgm, r.n_cgm_used, r.n_boluses,
             r.n_carb_entries, r.n_temp_basals)
    log.info("Built %d analysis windows spanning %.1f days (%s -> %s)",
             r.n_windows, r.span_days, r.date_start, r.date_end)
    log.info("Windows per hour: min=%d median=%.0f max=%d; %d windows had carbs",
             r.windows_per_hour_min, r.windows_per_hour_med,
             r.windows_per_hour_max, r.n_carb_windows)
    for name, d in r.decisions.items():
        log.info("%s decisions: %d changed, %d unchanged, %d sparse-pooled",
                 name, d["changed"], d["unchanged"], d["sparse"])
    log.info("Model residual %.1f mg/dL; convergence %s", r.obs_sd, r.convergence)


def render_text(r: DataReport) -> str:
    out = []
    out.append("Data & decision summary")
    out.append("-" * 72)
    out.append(f"  CGM records parsed:    {r.n_cgm_used} valid of {r.n_cgm}")
    out.append(f"  Boluses:               {r.n_boluses}")
    out.append(f"  Carb entries:          {r.n_carb_entries}")
    out.append(f"  Temp basals:           {r.n_temp_basals}")
    out.append(f"  Analysis windows:      {r.n_windows} over {r.span_days} days")
    out.append(f"  Date range:            {r.date_start} -> {r.date_end}")
    out.append(f"  Windows/hour:          min {r.windows_per_hour_min}, "
               f"median {r.windows_per_hour_med:.0f}, max {r.windows_per_hour_max}")
    out.append(f"  Windows with carbs:    {r.n_carb_windows}")
    out.append("")
    out.append("  Per-setting decisions (of 24 hours):")
    for name, d in r.decisions.items():
        out.append(f"    {name:<12} {d['changed']:>2} changed   "
                   f"{d['unchanged']:>2} already-consistent   "
                   f"{d['sparse']:>2} too-sparse")
    out.append("")
    out.append("  Why so few changes? A setting is only changed when your current")
    out.append("  value falls outside the 94% credible interval AND the hour has")
    out.append("  enough data. 'already-consistent' means the data agrees with your")
    out.append("  current setting; 'too-sparse' means not enough data that hour.")
    return "\n".join(out)
