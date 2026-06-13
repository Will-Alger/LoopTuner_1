"""Human-readable rendering of recommendations."""

from __future__ import annotations

from .recommend import HourlyRecommendation, Recommendations

DISCLAIMER = (
    "DISCLAIMER: LoopTuner is an analysis aid, not a medical device. Every value "
    "below is a SUGGESTION derived from historical data and a simplified model. "
    "Do NOT change any pump setting without review by your licensed clinician."
)


def condense_blocks(recs: list[HourlyRecommendation]) -> list[dict]:
    """Collapse 24 hourly recommendations into contiguous-value blocks.

    Returns a list of ``{start, end, label, current, recommended, confident}``.
    Shared by the text report and the web UI.
    """
    blocks = []
    start = 0
    for h in range(1, 25):
        prev = recs[h - 1]
        if h == 24 or recs[h].recommended != prev.recommended:
            end = h
            blocks.append({
                "start": start,
                "end": end % 24,
                "label": f"{start:02d}:00-{end % 24:02d}:00",
                "current": prev.current,
                "recommended": prev.recommended,
                "confident": any(r.confident for r in recs[start:end]),
            })
            start = h
    return blocks


def _condense(recs: list[HourlyRecommendation], unit: str) -> list[str]:
    """Collapse 24 hourly values into contiguous blocks for readability."""
    lines = []
    for b in condense_blocks(recs):
        cur = "n/a" if b["current"] is None else f"{b['current']:g}"
        flag = "  *" if b["confident"] else ""
        lines.append(
            f"  {b['label']}  current={cur:>6}  ->  {b['recommended']:g} {unit}{flag}"
        )
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

    out.append("Whole-day recommendation (works with limited data)")
    out.append("-" * 72)
    for p in recs.population:
        cur = "n/a" if p.current is None else f"{p.current:g}"
        if p.confident:
            if p.detail:
                out.append(f"  {p.name}: CHANGE — {p.detail}")
            else:
                out.append(f"  {p.name}: {cur} -> {p.recommended:g} {p.unit}  (CHANGE)")
        else:
            out.append(f"  {p.name}: keep {cur} {p.unit if p.unit!='x' else ''}".rstrip()
                       + f"  ({p.note})")
        out.append(f"      estimate {p.estimated:.2f} (94% CI {p.hdi_low:.2f}–{p.hdi_high:.2f})")
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
