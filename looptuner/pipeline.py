"""End-to-end orchestration: fetch -> features -> fit -> recommend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LoopTunerConfig
from .model import FitResult, fit as fit_model
from .nightscout import NightscoutClient
from .preprocess import Dataset, build_dataset
from .profile import Profile, schedule_value_at
from .recommend import Recommendations, build_recommendations


@dataclass
class PipelineResult:
    dataset: Dataset
    fit: FitResult
    recommendations: Recommendations
    profile: Profile


def _resolve_profile(cfg: LoopTunerConfig, profile_docs) -> Profile:
    if profile_docs:
        try:
            return Profile.from_nightscout(profile_docs)
        except (ValueError, KeyError):
            pass
    if cfg.fallback_basal_schedule:
        return Profile.from_schedule_list(cfg.fallback_basal_schedule)
    raise ValueError(
        "No usable Nightscout profile and no fallback_basal_schedule configured. "
        "A basal schedule is required to reconstruct insulin delivery."
    )


def run_from_records(
    cfg: LoopTunerConfig,
    entries: list,
    treatments: list,
    profile_docs: list,
    progressbar: bool = True,
) -> PipelineResult:
    """Run the pipeline on already-loaded records (used by tests/synthetic)."""
    profile = _resolve_profile(cfg, profile_docs)
    dataset = build_dataset(
        entries, treatments, profile, cfg.grid, cfg.pharmacology
    )
    scheduled_basal = np.array(
        [schedule_value_at(profile.basal, h * 3600 + 1800) for h in range(24)]
    )
    fit = fit_model(
        dataset, cfg.priors, cfg.sampler, scheduled_basal, progressbar=progressbar
    )
    recs = build_recommendations(fit, profile, cfg.safety)
    return PipelineResult(dataset, fit, recs, profile)


def run(cfg: LoopTunerConfig, use_cache: bool = True, progressbar: bool = True) -> PipelineResult:
    """Fetch from Nightscout and run the full pipeline."""
    client = NightscoutClient(cfg.nightscout)
    entries = client.entries(use_cache=use_cache)
    treatments = client.treatments(use_cache=use_cache)
    profile_docs = client.profile(use_cache=use_cache)
    return run_from_records(cfg, entries, treatments, profile_docs, progressbar)
