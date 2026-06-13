"""Visualize how LoopTuner adjusted each setting.

Produces a 3-panel figure (ISF, carb ratio, basal) over hour-of-day showing,
for each setting:

* your **current** schedule (dashed grey),
* the **model estimate** with its 94% credible band (blue) -- wide bands are
  hours the data can't pin down, which is exactly why no change is proposed
  there,
* the **recommended** schedule (green),
* red markers on hours where the change is *data-supported* (your current value
  fell outside the credible interval).

Rendered headlessly to PNG bytes so it works on a server / WSL with no display.
"""

from __future__ import annotations

import io

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .profile import schedule_value_at


def _current_series(schedule):
    if not schedule:
        return None
    return np.array([schedule_value_at(schedule, h * 3600 + 1800) for h in range(24)])


def render_png(fit, profile, recs, dpi: int = 110) -> bytes:
    """Return a PNG (bytes) comparing current / estimate / recommended."""
    hours = np.arange(24)
    panels = [
        ("Insulin sensitivity (ISF)", "mg/dL/U",
         fit.isf, fit.isf_hdi, _current_series(profile.isf), recs.isf),
        ("Carb ratio (CR)", "g/U",
         fit.carb_ratio, fit.carb_ratio_hdi, _current_series(profile.carb_ratio), recs.carb_ratio),
        ("Basal rate", "U/hr",
         fit.basal, fit.basal_hdi, _current_series(profile.basal), recs.basal),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, (title, unit, mean, hdi, current, rec) in zip(axes, panels):
        ax.fill_between(
            hours, hdi[:, 0], hdi[:, 1], color="#2a5bd7", alpha=0.15,
            step="mid", label="94% credible interval",
        )
        ax.plot(hours, mean, color="#2a5bd7", drawstyle="steps-mid", lw=1.8,
                label="model estimate")
        if current is not None:
            ax.plot(hours, current, color="#888", drawstyle="steps-mid", ls="--",
                    lw=1.5, label="current")
        recv = np.array([r.recommended for r in rec], dtype=float)
        ax.plot(hours, recv, color="#0a7a2f", drawstyle="steps-mid", lw=1.8,
                label="recommended")
        conf = [i for i, r in enumerate(rec) if r.confident]
        if conf:
            ax.scatter(conf, recv[conf], color="#b3261e", s=30, zorder=5,
                       label="data-supported change")
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.margins(x=0.01)

    axes[0].legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)
    axes[-1].set_xlabel("hour of day")
    axes[-1].set_xticks(range(0, 24, 2))
    fig.suptitle("LoopTuner — current vs. recommended (suggestions only)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def save_png(fit, profile, recs, path: str, dpi: int = 110) -> None:
    with open(path, "wb") as fh:
        fh.write(render_png(fit, profile, recs, dpi=dpi))
