"""Turn raw Nightscout records into the regression dataset.

The output is one row per analysis *window* (default 30 min) with:

* ``delta_bg``       -- change in glucose across the window (mg/dL)
* ``insulin_act``    -- units of insulin that became active in the window
                        (boluses + scheduled basal overridden by temp basals)
* ``carb_act``       -- grams of carbohydrate absorbed in the window
* ``dt_hours``       -- window length in hours (for the EGP drift term)
* ``hour``           -- hour-of-day index 0..23 (the hierarchical grouping)

The insulin-activity feature deliberately includes *scheduled basal* delivery.
That makes the fitted endogenous-glucose term (EGP) equal to ``ISF * basal``
in fasting steady state, so ``basal = EGP / ISF`` recovers the basal the body
actually needs -- the basis for the basal-rate recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GridConfig, PharmacologyConfig
from .pharmacology import (
    carbs_absorbed_in_interval,
    insulin_absorbed_in_interval,
)
from .profile import Profile, schedule_value_at


@dataclass
class Dataset:
    delta_bg: np.ndarray
    insulin_act: np.ndarray
    carb_act: np.ndarray
    dt_hours: np.ndarray
    hour: np.ndarray
    window_start: pd.DatetimeIndex  # for diagnostics

    def __len__(self) -> int:
        return len(self.delta_bg)


# ---------------------------------------------------------------------------
# Parsing raw records into tidy frames
# ---------------------------------------------------------------------------
def parse_entries(entries: list) -> pd.DataFrame:
    rows = []
    for e in entries:
        sgv = e.get("sgv")
        ts = e.get("date")
        if sgv is None or ts is None:
            continue
        rows.append((pd.to_datetime(int(ts), unit="ms", utc=True), float(sgv)))
    df = pd.DataFrame(rows, columns=["time", "sgv"])
    return df.dropna().sort_values("time").reset_index(drop=True)


def parse_boluses(treatments: list) -> pd.DataFrame:
    rows = []
    for t in treatments:
        insulin = t.get("insulin")
        if not insulin:
            continue
        ts = t.get("created_at") or t.get("timestamp")
        if ts is None:
            continue
        rows.append((pd.to_datetime(ts, utc=True), float(insulin)))
    return pd.DataFrame(rows, columns=["time", "units"]).sort_values("time")


def parse_carbs(treatments: list, default_absorption_min: float) -> pd.DataFrame:
    rows = []
    for t in treatments:
        carbs = t.get("carbs")
        if not carbs:
            continue
        ts = t.get("created_at") or t.get("timestamp")
        if ts is None:
            continue
        absorption = t.get("absorptionTime") or default_absorption_min
        rows.append(
            (pd.to_datetime(ts, utc=True), float(carbs), float(absorption))
        )
    return pd.DataFrame(
        rows, columns=["time", "grams", "absorption_min"]
    ).sort_values("time")


def parse_temp_basals(treatments: list) -> pd.DataFrame:
    """Temp-basal events as (time, rate U/hr, duration_min)."""
    rows = []
    for t in treatments:
        et = (t.get("eventType") or "").lower()
        if "temp basal" not in et and et != "tempbasal":
            continue
        ts = t.get("created_at") or t.get("timestamp")
        if ts is None:
            continue
        # 'absolute' (Loop) or 'rate'; duration in minutes.
        rate = t.get("absolute")
        if rate is None:
            rate = t.get("rate")
        if rate is None:
            continue
        duration = t.get("duration") or 0
        rows.append(
            (pd.to_datetime(ts, utc=True), float(rate), float(duration))
        )
    return pd.DataFrame(
        rows, columns=["time", "rate", "duration_min"]
    ).sort_values("time")


# ---------------------------------------------------------------------------
# Reconstructing actual insulin delivery (scheduled basal + temp + boluses)
# ---------------------------------------------------------------------------
def basal_delivery_doses(
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    profile: Profile,
    temp_basals: pd.DataFrame,
    step_min: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise actual basal delivery into small pseudo-boluses.

    Returns ``(times_min, units)`` where each entry is the basal insulin
    delivered in one ``step_min`` slice, valued at the temp-basal rate when one
    is active, otherwise the scheduled basal rate. Time is minutes since
    ``grid_start``.
    """
    n_steps = int(np.ceil((grid_end - grid_start) / pd.Timedelta(minutes=step_min)))
    step_starts = grid_start + pd.to_timedelta(np.arange(n_steps) * step_min, unit="m")

    # Build an array of active temp-basal rate per step (NaN where none).
    temp_rate = np.full(n_steps, np.nan)
    for _, tb in temp_basals.iterrows():
        if tb["duration_min"] <= 0:
            continue
        start_idx = int((tb["time"] - grid_start) / pd.Timedelta(minutes=step_min))
        end_time = tb["time"] + pd.Timedelta(minutes=tb["duration_min"])
        end_idx = int(np.ceil((end_time - grid_start) / pd.Timedelta(minutes=step_min)))
        lo, hi = max(start_idx, 0), min(end_idx, n_steps)
        if lo < hi:
            temp_rate[lo:hi] = tb["rate"]

    seconds_of_day = (
        step_starts.hour * 3600 + step_starts.minute * 60 + step_starts.second
    ).to_numpy()
    scheduled = np.array(
        [schedule_value_at(profile.basal, s) for s in seconds_of_day]
    )
    rate = np.where(np.isnan(temp_rate), scheduled, temp_rate)  # U/hr
    units = rate * (step_min / 60.0)
    times_min = np.arange(n_steps) * step_min + step_min / 2.0  # mid-step
    return times_min, units


# ---------------------------------------------------------------------------
# Building the windowed dataset
# ---------------------------------------------------------------------------
def build_dataset(
    entries: list,
    treatments: list,
    profile: Profile,
    grid_cfg: GridConfig,
    pharm_cfg: PharmacologyConfig,
) -> Dataset:
    bg = parse_entries(entries)
    if len(bg) < 10:
        raise ValueError("not enough CGM data to build a dataset")

    # Clip sensor errors, then build a regular interval grid by averaging CGM
    # into 5-min bins and interpolating across short gaps only.
    bg = bg[(bg["sgv"] >= grid_cfg.bg_min) & (bg["sgv"] <= grid_cfg.bg_max)]
    bg = bg.set_index("time")
    grid_start = bg.index.min().floor(f"{grid_cfg.window_min}min")
    grid_end = bg.index.max().ceil(f"{grid_cfg.window_min}min")

    step = grid_cfg.interval_min
    fine = bg["sgv"].resample(f"{step}min").mean().reindex(
        pd.date_range(grid_start, grid_end, freq=f"{step}min")
    )
    # Fill gaps no longer than ~15 min; leave larger gaps as NaN so we never
    # difference glucose across a sensor dropout.
    max_fill = max(int(round(15 / step)), 1)
    bg_grid = fine.interpolate(method="time", limit=max_fill, limit_area="inside")

    # Window boundaries are every window_min along the fine grid; glucose at a
    # boundary is the instantaneous (interpolated) value there, so ΔBG and the
    # insulin/carb activity are integrated over the *same* interval.
    stride = grid_cfg.window_min // step
    boundary_vals = bg_grid.to_numpy()[::stride]
    boundary_times = bg_grid.index[::stride]
    window_starts = boundary_times[:-1]            # one window per gap
    bg_start = boundary_vals[:-1]
    bg_end = boundary_vals[1:]

    window_start_min = (
        (window_starts - grid_start) / pd.Timedelta(minutes=1)
    ).to_numpy(dtype=float)
    window_end_min = window_start_min + grid_cfg.window_min

    # --- insulin doses (boluses + reconstructed basal) ---
    boluses = parse_boluses(treatments)
    temp_basals = parse_temp_basals(treatments)
    basal_times, basal_units = basal_delivery_doses(
        grid_start, grid_end, profile, temp_basals, step_min=grid_cfg.interval_min
    )
    if len(boluses):
        bolus_times = (
            (boluses["time"] - grid_start) / pd.Timedelta(minutes=1)
        ).to_numpy(dtype=float)
        bolus_units = boluses["units"].to_numpy(dtype=float)
    else:
        bolus_times = np.array([])
        bolus_units = np.array([])

    dose_times = np.concatenate([basal_times, bolus_times])
    dose_units = np.concatenate([basal_units, bolus_units])

    insulin_act = insulin_absorbed_in_interval(
        dose_times,
        dose_units,
        window_start_min,
        window_end_min,
        tp=pharm_cfg.insulin_peak_min,
        td=pharm_cfg.insulin_duration_min,
    )

    # --- carbs ---
    carbs = parse_carbs(treatments, pharm_cfg.carb_default_absorption_min)
    if len(carbs):
        carb_times = (
            (carbs["time"] - grid_start) / pd.Timedelta(minutes=1)
        ).to_numpy(dtype=float)
        carb_act = carbs_absorbed_in_interval(
            carb_times,
            carbs["grams"].to_numpy(dtype=float),
            carbs["absorption_min"].to_numpy(dtype=float),
            window_start_min,
            window_end_min,
            delay_min=pharm_cfg.carb_delay_min,
        )
    else:
        carb_act = np.zeros_like(window_start_min)

    # --- ΔBG across each window (boundary to boundary) ---
    delta_bg = bg_end - bg_start

    hour = window_starts.hour.to_numpy()
    dt_hours = np.full_like(window_start_min, grid_cfg.window_min / 60.0)

    # --- quality mask: both boundaries present, no implausible jumps ---
    valid = (
        np.isfinite(delta_bg)
        & np.isfinite(bg_start)
        & np.isfinite(bg_end)
        & (np.abs(delta_bg) <= grid_cfg.max_abs_delta)
    )

    return Dataset(
        delta_bg=delta_bg[valid],
        insulin_act=insulin_act[valid],
        carb_act=carb_act[valid],
        dt_hours=dt_hours[valid],
        hour=hour[valid].astype(int),
        window_start=window_starts[valid],
    )
