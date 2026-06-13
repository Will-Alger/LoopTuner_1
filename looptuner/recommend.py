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
class Recommendations:
    isf: list[HourlyRecommendation]
    carb_ratio: list[HourlyRecommendation]
    basal: list[HourlyRecommendation]
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


def build_recommendations(
    fit: FitResult,
    profile: Profile,
    cfg: SafetyConfig,
    min_data_per_hour: int = 20,
) -> Recommendations:
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

    summary = {
        "population_isf": round(fit.isf_population, 1),
        "population_carb_ratio": round(fit.isf_population / fit.csf_population, 1),
        "mean_recommended_basal": round(float(np.mean(rec_basal_values)), 3),
        "total_daily_basal": round(float(np.sum(rec_basal_values)), 2),
        "obs_sd_mgdl": round(fit.obs_sd, 1),
        "diagnostics": fit.diagnostics,
    }

    return Recommendations(
        isf=isf_recs,
        carb_ratio=cr_recs,
        basal=basal_recs,
        max_basal=max_basal,
        max_bolus=max_bolus,
        suspend_threshold=suspend_threshold,
        summary=summary,
    )
