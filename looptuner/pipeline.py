"""End-to-end orchestration: fetch -> features -> fit -> recommend."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import LoopTunerConfig
from .diagnostics import DataReport, build_report
from .model import FitResult, fit as fit_model
from .nightscout import NightscoutClient
from .preprocess import Dataset, build_dataset
from .profile import Profile, schedule_value_at
from .recommend import Recommendations, build_recommendations

log = logging.getLogger("looptuner")


@dataclass
class PipelineResult:
    dataset: Dataset
    fit: FitResult
    recommendations: Recommendations
    profile: Profile
    report: DataReport


def _resolve_profile(cfg: LoopTunerConfig, profile_docs) -> Profile:
    if profile_docs:
        # An explicitly requested profile name should error if missing rather
        # than silently falling back, so the user knows their choice was wrong.
        if cfg.nightscout.profile_name:
            return Profile.from_nightscout(
                profile_docs, name=cfg.nightscout.profile_name
            )
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
    log.info("Using profile with %d basal, %d ISF, %d carb-ratio breakpoints",
             len(profile.basal), len(profile.isf), len(profile.carb_ratio))
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
    report = build_report(entries, treatments, dataset, recs, fit)
    return PipelineResult(dataset, fit, recs, profile, report)


def run(cfg: LoopTunerConfig, use_cache: bool = True, progressbar: bool = True) -> PipelineResult:
    """Fetch from Nightscout and run the full pipeline."""
    client = NightscoutClient(cfg.nightscout)
    log.info("Fetching up to %d days from %s", cfg.nightscout.days, cfg.nightscout.url)
    entries = client.entries(use_cache=use_cache)
    treatments = client.treatments(use_cache=use_cache)
    profile_docs = client.profile(use_cache=use_cache)
    log.info("Fetched %d entries, %d treatments, %d profile doc(s)",
             len(entries), len(treatments), len(profile_docs))
    return run_from_records(cfg, entries, treatments, profile_docs, progressbar)
