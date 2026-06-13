"""Human-readable rendering of recommendations."""

from __future__ import annotations

from .recommend import HourlyRecommendation, Recommendations

DISCLAIMER = (
    "DISCLAIMER: LoopTuner is an analysis aid, not a medical device. Every value "
    "below is a SUGGESTION derived from historical data and a simplified model. "
    "Do NOT change any pump setting without review by your licensed clinician."
)


def _condense(recs: list[HourlyRecommendation], unit: str) -> list[str]:
    """Collapse 24 hourly values into contiguous blocks for readability."""
    lines = []
    start = 0
    for h in range(1, 25):
        prev = recs[h - 1]
        change = h == 24 or recs[h].recommended != prev.recommended
        if change:
            end = h
            label = f"{start:02d}:00-{end % 24:02d}:00"
            cur = "n/a" if prev.current is None else f"{prev.current:g}"
            flag = "  *" if any(r.confident for r in recs[start:end]) else ""
            lines.append(
                f"  {label}  current={cur:>6}  ->  {prev.recommended:g} {unit}{flag}"
            )
            start = h
    return lines


def render(recs: Recommendations) -> str:
    out: list[str] = []
    out.append("=" * 72)
    out.append("LoopTuner — settings recommendations")
    out.append("=" * 72)
    out.append("")
    out.append(DISCLAIMER)
    out.append("")

    s = recs.summary
    out.append("Population summary")
    out.append("-" * 72)
    out.append(f"  Insulin sensitivity (typical):  {s['population_isf']} mg/dL/U")
    out.append(f"  Carb ratio (typical):           {s['population_carb_ratio']} g/U")
    out.append(f"  Recommended total daily basal:  {s['total_daily_basal']} U")
    out.append(f"  Model residual (1 SD):          {s['obs_sd_mgdl']} mg/dL")
    diag = s.get("diagnostics") or {}
    if diag:
        out.append(
            f"  Convergence: max R-hat={diag.get('max_r_hat')}, "
            f"min ESS={diag.get('min_ess_bulk')}, "
            f"divergences={diag.get('n_divergences')}"
        )
    out.append("")

    out.append("Insulin sensitivity factor (ISF)   [* = data-supported change]")
    out.append("-" * 72)
    out.extend(_condense(recs.isf, "mg/dL/U"))
    out.append("")

    out.append("Carb ratio (CR)")
    out.append("-" * 72)
    out.extend(_condense(recs.carb_ratio, "g/U"))
    out.append("")

    out.append("Basal rate")
    out.append("-" * 72)
    out.extend(_condense(recs.basal, "U/hr"))
    out.append("")

    out.append("Safety delivery limits (suggested)")
    out.append("-" * 72)
    out.append(f"  Maximum basal rate:  {recs.max_basal:g} U/hr")
    out.append(f"  Maximum bolus:       {recs.max_bolus:g} U")
    if recs.suspend_threshold is not None:
        out.append(f"  Suspend threshold:   {recs.suspend_threshold:g} mg/dL")
    out.append("  Minimum delivery:    0 U/hr (suspend on low)")
    out.append("")
    out.append("=" * 72)
    return "\n".join(out)
