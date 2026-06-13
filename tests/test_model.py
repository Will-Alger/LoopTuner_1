"""End-to-end recovery test: fit synthetic data with known truth.

Kept intentionally small (few days / draws) so it runs in CI in well under a
minute, but still exercises the full fetch->features->fit->recommend path and
checks the posterior recovers the ground-truth parameters where the data can
identify them.
"""

import numpy as np
import pytest

from looptuner.config import LoopTunerConfig
from looptuner.pipeline import run_from_records
from looptuner.synthetic import generate


@pytest.fixture(scope="module")
def fitted():
    data = generate(days=14, seed=0)
    cfg = LoopTunerConfig()
    cfg.sampler.draws = 400
    cfg.sampler.tune = 400
    cfg.sampler.chains = 2
    cfg.sampler.seed = 0
    result = run_from_records(
        cfg, data["entries"], data["treatments"], data["profile"],
        progressbar=False,
    )
    return result, data["truth"], data


def _mape(est, truth, mask=None):
    est, truth = np.asarray(est), np.asarray(truth)
    if mask is not None:
        est, truth = est[mask], truth[mask]
    return float(np.mean(np.abs((est - truth) / truth)) * 100)


def test_convergence(fitted):
    result, _, _ = fitted
    diag = result.fit.diagnostics
    assert diag["max_r_hat"] < 1.1
    assert diag["n_divergences"] == 0


def test_isf_recovery(fitted):
    result, truth, _ = fitted
    # Includes fundamentally-unidentifiable fasting hours, so allow slack.
    assert _mape(result.fit.isf, truth["isf"]) < 30


def test_carb_ratio_recovery(fitted):
    result, truth, _ = fitted
    # Only meaningful where carbs were actually eaten.
    ca, hr = result.dataset.carb_act, result.dataset.hour
    carb_n = np.array([(ca[hr == h] > 0).sum() for h in range(24)])
    mask = carb_n >= 10
    assert mask.sum() >= 6
    assert _mape(result.fit.carb_ratio, truth["carb_ratio"], mask) < 20


def test_basal_recovery(fitted):
    result, truth, _ = fitted
    assert _mape(result.fit.basal, truth["basal"]) < 25


def test_recommendations_are_sane(fitted):
    result, _, _ = fitted
    recs = result.recommendations
    assert len(recs.isf) == 24
    assert len(recs.basal) == 24
    assert recs.max_basal > 0
    assert recs.max_bolus > 0
    # Recommended basal should be positive and physiologically bounded.
    for r in recs.basal:
        assert 0 < r.recommended < 5
