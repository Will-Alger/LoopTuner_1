"""Translate the fitted posterior into reviewable setting recommendations.

Every recommendation is guard-railed:

* a change is only proposed when the posterior is confident enough (the
  current value sits outside a central credible band) **and** there is enough
  data for that hour;
* the per-step change is capped (``max_relative_change``) so a single run can
  never swing a setting dramatically;
* values are rounded to clinician-friendly increments.

Nothing here is medical advice. Output is a *suggestion* to discuss with a
licensed clinician before any pump setting is changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import SafetyConfig
from .model import FitResult, N_HOURS
from .profile import Profile, schedule_value_at


@dataclass
class HourlyRecommendation:
    hour: int
    current: Optional[float]
    estimated: float
    hdi_low: float
    hdi_high: float
    recommended: float
    n_data: int
    confident: bool
    note: str = ""


@dataclass
class PopulationRecommendation:
    """A whole-day (pooled) recommendation, usable even from limited data."""

    name: str                  # "ISF", "Carb ratio", "Basal (overall)"
    unit: str
    current: Optional[float]
    estimated: float
    hdi_low: float
    hdi_high: float
    recommended: float
    confident: bool
    note: str
    detail: str = ""           # human-readable extra (e.g. "+12% basal")


@dataclass
class Recommendations:
    isf: list[HourlyRecommendation]
    carb_ratio: list[HourlyRecommendation]
    basal: list[HourlyRecommendation]
    population: list[PopulationRecommendation]
    max_basal: float
    max_bolus: float
    suspend_threshold: Optional[float]
    summary: dict = field(default_factory=dict)


def _round_to(x: float, step: float) -> float:
    # round(x/step)*step reintroduces binary-float noise (e.g. 1.2000000000002);
    # round again to kill it, to more decimals than any sane step needs.
    return round(round(x / step) * step, 6)


def _current_at_hour(schedule, hour: int) -> Optional[float]:
    if not schedule:
        return None
    return schedule_value_at(schedule, hour * 3600 + 1800)  # mid-hour


def _guardrail(
    current: Optional[float],
    estimated: float,
    hdi: np.ndarray,
    n_data: int,
    cfg: SafetyConfig,
    round_step: float,
    min_data: int,
) -> tuple[float, bool, str]:
    """Return (recommended_value, confident, note)."""
    low, high = float(hdi[0]), float(hdi[1])

    if n_data < min_data:
        # Too little local data: fall back to the (pooled) estimate but don't
        # claim confidence; partial pooling already kept this sane.
        target = current if current is not None else estimated
        return _round_to(target, round_step), False, "sparse hour (pooled)"

    if current is None:
        return _round_to(estimated, round_step), True, "no current value"

    # Confident only if the current value lies outside the credible interval,
    # i.e. the data disagrees with the current setting.
    confident = not (low <= current <= high)
    if not confident:
        return _round_to(current, round_step), False, "consistent with current"

    # Cap the move so a single run can't swing things too far.
    max_delta = abs(current) * cfg.max_relative_change
    delta = np.clip(estimated - current, -max_delta, max_delta)
    capped = bool(abs(estimated - current) > max_delta)
    value = _round_to(current + delta, round_step)
    note = "capped to max change" if capped else "adjusted"
    return value, True, note


def _schedule_average(schedule) -> Optional[float]:
    """Time-average of a daily schedule (mean over 24 hourly samples)."""
    if not schedule:
        return None
    return float(np.mean([schedule_value_at(schedule, h * 3600 + 1800) for h in range(24)]))


def _population_guardrail(current, estimated, hdi, cfg, round_step, enough_data, sparse_note):
    low, high = float(hdi[0]), float(hdi[1])
    if not enough_data:
        return (round(estimated, 2), False, sparse_note)
    if current is None:
        return (_round_to(estimated, round_step), True, "no current value")
    confident = not (low <= current <= high)
    if not confident:
        return (_round_to(current, round_step), False, "consistent with current")
    max_delta = abs(current) * cfg.max_relative_change
    delta = np.clip(estimated - current, -max_delta, max_delta)
    capped = abs(estimated - current) > max_delta
    return (_round_to(current + delta, round_step), True,
            "capped to max change" if capped else "adjusted")


def _build_population(fit, profile, cfg, total_windows, total_carb_windows):
    recs: list[PopulationRecommendation] = []

    # ISF (whole-day): pooled over all windows; needs reasonable insulin signal.
    cur_isf = _schedule_average(profile.isf)
    enough = total_windows >= 40
    val, conf, note = _population_guardrail(
        cur_isf, fit.isf_population, fit.isf_population_hdi, cfg,
        cfg.isf_round, enough, "not enough data yet",
    )
    recs.append(PopulationRecommendation(
        "Insulin sensitivity (ISF)", "mg/dL/U", cur_isf, fit.isf_population,
        float(fit.isf_population_hdi[0]), float(fit.isf_population_hdi[1]),
        val, conf, note,
    ))

    # Carb ratio (whole-day): needs enough windows that actually contained carbs.
    cur_cr = _schedule_average(profile.carb_ratio)
    enough_carb = total_carb_windows >= 30
    val, conf, note = _population_guardrail(
        cur_cr, fit.carb_ratio_population, fit.carb_ratio_population_hdi, cfg,
        cfg.carb_ratio_round, enough_carb,
        f"insufficient carb data ({total_carb_windows} windows; need ~30)",
    )
    recs.append(PopulationRecommendation(
        "Carb ratio (CR)", "g/U", cur_cr, fit.carb_ratio_population,
        float(fit.carb_ratio_population_hdi[0]), float(fit.carb_ratio_population_hdi[1]),
        val, conf, note,
    ))

    # Basal (overall scale): the single most data-efficient signal -- does the
    # whole basal schedule need to move up or down? current scale == 1.0.
    scale_hdi = fit.basal_scale_hdi
    enough = total_windows >= 40
    confident = enough and not (scale_hdi[0] <= 1.0 <= scale_hdi[1])
    cur_total = None
    if profile.basal:
        cur_total = sum(
            schedule_value_at(profile.basal, h * 3600 + 1800) for h in range(24)
        )
    if not enough:
        note, scale_val, detail = "not enough data yet", 1.0, ""
    elif not confident:
        note, scale_val, detail = "consistent with current", 1.0, "basal schedule looks right"
    else:
        capped_scale = float(np.clip(
            fit.basal_scale, 1 - cfg.max_relative_change, 1 + cfg.max_relative_change
        ))
        scale_val = capped_scale
        pct = (capped_scale - 1) * 100
        if cur_total is not None:
            detail = f"scale all basal rates by {pct:+.0f}% (daily basal {cur_total:.1f} → {cur_total * capped_scale:.1f} U)"
        else:
            detail = f"scale all basal rates by {pct:+.0f}%"
        note = "adjusted"
    recs.append(PopulationRecommendation(
        "Basal (overall)", "x", 1.0, float(fit.basal_scale),
        float(scale_hdi[0]), float(scale_hdi[1]),
        round(scale_val, 3), confident, note, detail,
    ))
    return recs


def build_recommendations(
    fit: FitResult,
    profile: Profile,
    cfg: SafetyConfig,
    min_data_per_hour: Optional[int] = None,
) -> Recommendations:
    if min_data_per_hour is None:
        min_data_per_hour = getattr(cfg, "min_data_per_hour", 20)
    isf_recs: list[HourlyRecommendation] = []
    cr_recs: list[HourlyRecommendation] = []
    basal_recs: list[HourlyRecommendation] = []

    # Posterior basal need per hour comes straight from the model (U/hr).
    basal_est = fit.basal
    basal_hdi = fit.basal_hdi

    for h in range(N_HOURS):
        n = int(fit.n_per_hour[h])

        # ISF -------------------------------------------------------------
        cur_isf = _current_at_hour(profile.isf, h)
        val, conf, note = _guardrail(
            cur_isf, float(fit.isf[h]), fit.isf_hdi[h], n, cfg,
            cfg.isf_round, min_data_per_hour,
        )
        isf_recs.append(HourlyRecommendation(
            h, cur_isf, float(fit.isf[h]), float(fit.isf_hdi[h][0]),
            float(fit.isf_hdi[h][1]), val, n, conf, note,
        ))

        # Carb ratio ------------------------------------------------------
        cur_cr = _current_at_hour(profile.carb_ratio, h)
        # Carb data is concentrated around meals; require carb signal.
        cr_min_data = min_data_per_hour
        val, conf, note = _guardrail(
            cur_cr, float(fit.carb_ratio[h]), fit.carb_ratio_hdi[h], n, cfg,
            cfg.carb_ratio_round, cr_min_data,
        )
        cr_recs.append(HourlyRecommendation(
            h, cur_cr, float(fit.carb_ratio[h]), float(fit.carb_ratio_hdi[h][0]),
            float(fit.carb_ratio_hdi[h][1]), val, n, conf, note,
        ))

        # Basal -----------------------------------------------------------
        cur_basal = _current_at_hour(profile.basal, h)
        val, conf, note = _guardrail(
            cur_basal, float(basal_est[h]), basal_hdi[h], n, cfg,
            cfg.basal_round, min_data_per_hour,
        )
        basal_recs.append(HourlyRecommendation(
            h, cur_basal, float(basal_est[h]), float(basal_hdi[h][0]),
            float(basal_hdi[h][1]), val, n, conf, note,
        ))

    # Safety delivery limits ------------------------------------------------
    rec_basal_values = [r.recommended for r in basal_recs]
    max_rec_basal = max(rec_basal_values) if rec_basal_values else 0.0
    max_basal = _round_to(max_rec_basal * cfg.max_basal_multiplier, cfg.basal_round)

    # Max bolus: cover the largest plausible single meal at the tightest
    # (smallest) carb ratio, i.e. most insulin-hungry time of day.
    cr_values = [r.recommended for r in cr_recs if r.recommended > 0]
    tightest_cr = min(cr_values) if cr_values else float(fit.carb_ratio.min())
    # Assume a large meal of ~100 g as the design point, rounded to 0.5 U.
    max_bolus = _round_to(100.0 / max(tightest_cr, 1e-6), 0.5)

    suspend_threshold = profile.target_low

    total_windows = int(fit.n_per_hour.sum())
    total_carb_windows = int(fit.n_carb_per_hour.sum())
    population = _build_population(
        fit, profile, cfg, total_windows, total_carb_windows
    )

    summary = {
        "population_isf": round(fit.isf_population, 1),
        "population_carb_ratio": round(fit.carb_ratio_population, 1),
        "mean_recommended_basal": round(float(np.mean(rec_basal_values)), 3),
        "total_daily_basal": round(float(np.sum(rec_basal_values)), 2),
        "basal_scale": round(fit.basal_scale, 3),
        "obs_sd_mgdl": round(fit.obs_sd, 1),
        "diagnostics": fit.diagnostics,
    }

    return Recommendations(
        isf=isf_recs,
        carb_ratio=cr_recs,
        basal=basal_recs,
        population=population,
        max_basal=max_basal,
        max_bolus=max_bolus,
        suspend_threshold=suspend_threshold,
        summary=summary,
    )
