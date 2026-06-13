"""Generate synthetic Nightscout-shaped data with known ground truth.

Used to validate the whole pipeline offline: we pick per-hour ISF / carb-ratio
/ basal-need, simulate glucose under the same glucodynamic model the regression
assumes, emit ``entries`` + ``treatments`` + ``profile`` in Nightscout's JSON
shape, fit, and check the recovered parameters match the truth.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from .pharmacology import insulin_on_board_fraction


def _diurnal(base, amplitude, phase, hours):
    return base + amplitude * np.cos((hours - phase) / 24 * 2 * np.pi)


def generate(
    days: int = 21,
    seed: int = 0,
    tp: float = 75.0,
    td: float = 360.0,
) -> dict:
    """Return ``dict`` with keys ``entries``, ``treatments``, ``profile`` and
    the ground-truth schedules under ``truth``.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(24)

    # --- ground-truth per-hour parameters ---
    isf_true = _diurnal(45.0, 12.0, 4, hours)          # more resistant pre-dawn
    cr_true = _diurnal(10.0, 2.5, 7, hours)
    basal_true = _diurnal(0.85, 0.35, 3, hours)        # U/hr the body needs
    csf_true = isf_true / cr_true                        # mg/dL per g
    # EGP consistent with basal need: EGP = ISF * basal (steady-state identity).
    egp_true = isf_true * basal_true                     # mg/dL per hour

    step_min = 5
    n_steps = days * 24 * 60 // step_min
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=step_min * i) for i in range(n_steps)]

    # --- carbohydrate / bolus events (~3 meals/day) ---
    carb_events = []  # (step_index, grams, absorption_min)
    bolus_events = []  # (step_index, units)
    steps_per_day = 24 * 60 // step_min
    for d in range(days):
        for meal_hour, grams in [(7, 45), (12, 60), (19, 70)]:
            jitter = rng.integers(-6, 6)
            idx = d * steps_per_day + (meal_hour * 60 // step_min) + jitter
            if idx >= n_steps:
                continue
            grams = float(grams + rng.normal(0, 8))
            carb_events.append((idx, grams, 180.0))
            # Bolus using the true carb ratio at that hour (well-managed user),
            # with realistic dosing error.
            hour = times[idx].hour
            dose = grams / cr_true[hour] * rng.normal(1.0, 0.08)
            bolus_events.append((idx, float(max(dose, 0))))
            # Occasional correction bolus.
            if rng.random() < 0.2:
                bolus_events.append((idx + 6, float(rng.uniform(0.3, 1.2))))

    # --- build insulin delivery: scheduled basal + boluses ---
    dose_times_min = []
    dose_units = []
    for i in range(n_steps):
        hour = times[i].hour
        dose_times_min.append(i * step_min + step_min / 2)
        dose_units.append(basal_true[hour] * step_min / 60.0)
    for idx, units in bolus_events:
        dose_times_min.append(idx * step_min + step_min / 2)
        dose_units.append(units)
    dose_times_min = np.array(dose_times_min)
    dose_units = np.array(dose_units)

    # --- simulate glucose forward, 5-min steps ---
    bg = np.zeros(n_steps)
    bg[0] = 120.0
    grid_min = np.arange(n_steps) * step_min
    # Precompute carb absorption rate (g per step) per step.
    carb_rate = np.zeros(n_steps)
    for idx, grams, absorption in carb_events:
        a_steps = int(absorption // step_min)
        delay = 10 // step_min
        lo = idx + delay
        hi = min(lo + a_steps, n_steps)
        if hi > lo:
            carb_rate[lo:hi] += grams / (hi - lo)

    # Insulin activity (units absorbed) per step via IOB differences.
    iob_start = insulin_on_board_fraction(
        grid_min[:, None] - dose_times_min[None, :], tp, td
    )
    iob_end = insulin_on_board_fraction(
        (grid_min[:, None] + step_min) - dose_times_min[None, :], tp, td
    )
    insulin_act_step = ((iob_start - iob_end) * dose_units[None, :]).sum(axis=1)

    dt_hours = step_min / 60.0
    for i in range(n_steps - 1):
        hour = times[i].hour
        dbg = (
            -isf_true[hour] * insulin_act_step[i]
            + csf_true[hour] * carb_rate[i]
            + egp_true[hour] * dt_hours
        )
        bg[i + 1] = np.clip(bg[i] + dbg + rng.normal(0, 2.0), 40, 400)

    # --- emit Nightscout-shaped records ---
    entries = [
        {"date": int(t.timestamp() * 1000), "sgv": float(round(g))}
        for t, g in zip(times, bg)
    ]

    treatments = []
    for idx, grams, absorption in carb_events:
        treatments.append({
            "eventType": "Meal Bolus",
            "created_at": times[idx].isoformat(),
            "carbs": round(grams, 1),
            "absorptionTime": absorption,
        })
    for idx, units in bolus_events:
        treatments.append({
            "eventType": "Bolus",
            "created_at": times[idx].isoformat(),
            "insulin": round(units, 2),
        })

    profile = [{
        "defaultProfile": "Default",
        "store": {
            "Default": {
                "dia": td / 60.0,
                "units": "mg/dl",
                "basal": [
                    {"time": f"{h:02d}:00", "value": round(basal_true[h], 3),
                     "timeAsSeconds": h * 3600}
                    for h in range(24)
                ],
                "sens": [
                    {"time": f"{h:02d}:00", "value": round(isf_true[h], 1),
                     "timeAsSeconds": h * 3600}
                    for h in range(24)
                ],
                "carbratio": [
                    {"time": f"{h:02d}:00", "value": round(cr_true[h], 1),
                     "timeAsSeconds": h * 3600}
                    for h in range(24)
                ],
                "target_low": [{"time": "00:00", "value": 100}],
                "target_high": [{"time": "00:00", "value": 120}],
            }
        },
    }]

    return {
        "entries": entries,
        "treatments": treatments,
        "profile": profile,
        "truth": {
            "isf": isf_true,
            "carb_ratio": cr_true,
            "basal": basal_true,
            "csf": csf_true,
            "egp": egp_true,
        },
    }
