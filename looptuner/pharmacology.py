"""Insulin-action and carb-absorption models.

These convert discrete insulin doses and carb entries into the *activity*
features that drive the glucodynamic regression:

* ``insulin_on_board_fraction`` -- the fraction of a unit dose still active
  ``t`` minutes after delivery (the Loop/LoopKit exponential model).
* ``insulin_absorbed_in_interval`` -- the units of insulin that became active
  (left IOB) during a time interval; this is what actually moves glucose.
* ``carbs_absorbed_in_interval`` -- grams of carbohydrate absorbed during an
  interval under a simple piecewise-linear absorption model.

Everything here is deterministic and unit-tested independently of Nightscout
or PyMC so the feature pipeline can be validated in isolation.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Insulin: exponential ("Loop") model
# ---------------------------------------------------------------------------
def _exponential_params(tp: float, td: float) -> tuple[float, float, float]:
    """Return (tau, a, S) for the exponential insulin model.

    Parameters
    ----------
    tp : peak activity time (minutes)
    td : total duration of insulin action / DIA (minutes)
    """
    if not (0 < tp < td):
        raise ValueError(f"require 0 < tp < td, got tp={tp}, td={td}")
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    S = 1 / (1 - a + (1 + a) * np.exp(-td / tau))
    return tau, a, S


def insulin_on_board_fraction(
    t_minutes: np.ndarray | float, tp: float = 75.0, td: float = 360.0
) -> np.ndarray:
    """Fraction of a unit dose still on board ``t`` minutes after delivery.

    This is the LoopKit exponential insulin model. ``IOB(0) == 1`` and
    ``IOB(td) == 0``. Values before the dose (t < 0) are 1 (none absorbed yet);
    after ``td`` they are 0.
    """
    t = np.asarray(t_minutes, dtype=float)
    tau, a, S = _exponential_params(tp, td)

    iob = 1 - S * (1 - a) * (
        (t**2 / (tau * td * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1
    )
    iob = np.where(t <= 0, 1.0, iob)
    iob = np.where(t >= td, 0.0, iob)
    return np.clip(iob, 0.0, 1.0)


def insulin_absorbed_in_interval(
    dose_times_min: np.ndarray,
    dose_units: np.ndarray,
    interval_start_min: np.ndarray,
    interval_end_min: np.ndarray,
    tp: float = 75.0,
    td: float = 360.0,
) -> np.ndarray:
    """Units of insulin that became active during each interval.

    For every interval ``[start, end)`` this returns the sum over all doses of
    ``dose * (IOB(start - dose_time) - IOB(end - dose_time))`` -- i.e. the
    insulin that left IOB (acted on glucose) during the interval.

    All time arguments are in minutes on a common clock.

    Shapes: ``dose_*`` are ``(D,)``, ``interval_*`` are ``(N,)``; returns
    ``(N,)``.
    """
    dose_times_min = np.asarray(dose_times_min, dtype=float)
    dose_units = np.asarray(dose_units, dtype=float)
    interval_start_min = np.asarray(interval_start_min, dtype=float)
    interval_end_min = np.asarray(interval_end_min, dtype=float)

    if dose_times_min.size == 0:
        return np.zeros_like(interval_start_min)

    # (N, D) elapsed times.
    elapsed_start = interval_start_min[:, None] - dose_times_min[None, :]
    elapsed_end = interval_end_min[:, None] - dose_times_min[None, :]

    iob_start = insulin_on_board_fraction(elapsed_start, tp, td)
    iob_end = insulin_on_board_fraction(elapsed_end, tp, td)

    absorbed = (iob_start - iob_end) * dose_units[None, :]
    return absorbed.sum(axis=1)


# ---------------------------------------------------------------------------
# Carbohydrate absorption: piecewise-linear model
# ---------------------------------------------------------------------------
def carbs_absorbed_in_interval(
    carb_times_min: np.ndarray,
    carb_grams: np.ndarray,
    carb_absorption_min: np.ndarray,
    interval_start_min: np.ndarray,
    interval_end_min: np.ndarray,
    delay_min: float = 10.0,
) -> np.ndarray:
    """Grams of carbohydrate absorbed during each interval.

    Each carb entry is absorbed linearly between ``delay_min`` and
    ``delay_min + absorption_time`` after it is eaten. This is a deliberately
    simple, robust model -- it does not try to reproduce Loop's dynamic carb
    absorption, only to provide a stable regressor for the carb-sensitivity
    estimate.

    Shapes: ``carb_*`` are ``(C,)``, ``interval_*`` are ``(N,)``; returns
    ``(N,)``.
    """
    carb_times_min = np.asarray(carb_times_min, dtype=float)
    carb_grams = np.asarray(carb_grams, dtype=float)
    carb_absorption_min = np.asarray(carb_absorption_min, dtype=float)
    interval_start_min = np.asarray(interval_start_min, dtype=float)
    interval_end_min = np.asarray(interval_end_min, dtype=float)

    if carb_times_min.size == 0:
        return np.zeros_like(interval_start_min)

    start0 = carb_times_min + delay_min          # absorption begins
    end0 = start0 + carb_absorption_min          # absorption complete

    # Fraction absorbed by a given clock time for each entry: ramp from 0 to 1.
    def absorbed_fraction(t):  # t: (N, C)
        frac = (t - start0[None, :]) / np.maximum(carb_absorption_min[None, :], 1e-9)
        return np.clip(frac, 0.0, 1.0)

    f_end = absorbed_fraction(interval_end_min[:, None])
    f_start = absorbed_fraction(interval_start_min[:, None])
    grams = (f_end - f_start) * carb_grams[None, :]
    return grams.sum(axis=1)
