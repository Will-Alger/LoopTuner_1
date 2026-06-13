"""Unit tests for the deterministic pharmacology + preprocessing pieces."""

import numpy as np
import pytest

from looptuner.pharmacology import (
    carbs_absorbed_in_interval,
    insulin_absorbed_in_interval,
    insulin_on_board_fraction,
)


def test_iob_boundaries():
    assert insulin_on_board_fraction(0.0) == pytest.approx(1.0)
    assert insulin_on_board_fraction(360.0, td=360.0) == pytest.approx(0.0)
    assert insulin_on_board_fraction(-10.0) == pytest.approx(1.0)
    assert insulin_on_board_fraction(1000.0, td=360.0) == pytest.approx(0.0)


def test_iob_monotonic_decreasing():
    t = np.linspace(0, 360, 200)
    iob = insulin_on_board_fraction(t)
    assert np.all(np.diff(iob) <= 1e-9)


def test_insulin_absorbed_sums_to_dose():
    # All insulin from a single dose should be absorbed across its full DIA.
    dose_times = np.array([0.0])
    dose_units = np.array([2.5])
    starts = np.arange(0, 360, 5.0)
    ends = starts + 5.0
    absorbed = insulin_absorbed_in_interval(dose_times, dose_units, starts, ends)
    assert absorbed.sum() == pytest.approx(2.5, rel=1e-3)
    assert np.all(absorbed >= 0)


def test_insulin_absorbed_empty():
    out = insulin_absorbed_in_interval(
        np.array([]), np.array([]), np.array([0.0, 5.0]), np.array([5.0, 10.0])
    )
    assert np.all(out == 0)


def test_carbs_absorbed_sums_to_total():
    carb_times = np.array([0.0])
    grams = np.array([60.0])
    absorption = np.array([180.0])
    starts = np.arange(0, 400, 5.0)
    ends = starts + 5.0
    absorbed = carbs_absorbed_in_interval(
        carb_times, grams, absorption, starts, ends, delay_min=10.0
    )
    assert absorbed.sum() == pytest.approx(60.0, rel=1e-6)
    # Nothing absorbed during the delay window.
    assert absorbed[0] == pytest.approx(0.0)


def test_carbs_linear_rate():
    # Over a 100-min absorption with no delay, half should be gone by 50 min.
    absorbed = carbs_absorbed_in_interval(
        np.array([0.0]), np.array([100.0]), np.array([100.0]),
        np.array([0.0]), np.array([50.0]), delay_min=0.0,
    )
    assert absorbed[0] == pytest.approx(50.0)
