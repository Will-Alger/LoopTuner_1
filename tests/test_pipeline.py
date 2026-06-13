"""Tests for profile parsing and dataset construction (no sampling)."""

import numpy as np

from looptuner.config import GridConfig, PharmacologyConfig
from looptuner.preprocess import build_dataset
from looptuner.profile import Profile, schedule_value_at
from looptuner.synthetic import generate


def test_schedule_lookup():
    sched = [(0, 0.8), (3 * 3600, 1.0), (12 * 3600, 0.9)]
    assert schedule_value_at(sched, 0) == 0.8
    assert schedule_value_at(sched, 3 * 3600) == 1.0
    assert schedule_value_at(sched, 6 * 3600) == 1.0
    assert schedule_value_at(sched, 23 * 3600) == 0.9


def test_profile_from_nightscout():
    data = generate(days=2, seed=1)
    profile = Profile.from_nightscout(data["profile"])
    assert len(profile.basal) == 24
    assert len(profile.isf) == 24
    assert len(profile.carb_ratio) == 24
    assert profile.dia_hours == 6.0


def test_build_dataset_shapes_and_finiteness():
    data = generate(days=4, seed=2)
    profile = Profile.from_nightscout(data["profile"])
    ds = build_dataset(
        data["entries"], data["treatments"], profile,
        GridConfig(), PharmacologyConfig(),
    )
    n = len(ds)
    assert n > 100
    for arr in (ds.delta_bg, ds.insulin_act, ds.carb_act, ds.dt_hours):
        assert len(arr) == n
        assert np.all(np.isfinite(arr))
    assert ds.hour.min() >= 0 and ds.hour.max() <= 23
    # Insulin activity should be strictly positive (basal always delivered).
    assert np.all(ds.insulin_act > 0)
    # Some windows must contain carb absorption.
    assert np.any(ds.carb_act > 0)
