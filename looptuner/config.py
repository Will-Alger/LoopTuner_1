"""Configuration objects for LoopTuner.

All tunables live here so the pipeline (fetch -> features -> model ->
recommend) can be driven from a single config, loaded from a YAML/JSON file or
constructed in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class NightscoutConfig:
    """How to reach a Nightscout site."""

    url: str = ""                      # e.g. https://my-site.up.railway.app
    api_secret: Optional[str] = None   # plain secret; hashed to SHA1 by client
    token: Optional[str] = None        # access token (alternative to secret)
    days: int = 30                     # how much history to pull
    cache_dir: str = ".looptuner_cache"


@dataclass
class PharmacologyConfig:
    """Insulin and carb action parameters used to build features."""

    insulin_peak_min: float = 75.0      # exponential model peak
    insulin_duration_min: float = 360.0  # DIA
    carb_default_absorption_min: float = 180.0
    carb_delay_min: float = 10.0


@dataclass
class GridConfig:
    """Time-grid / dataset construction."""

    interval_min: int = 5               # CGM bin width
    # Aggregate raw 5-min bins into windows of this many minutes for the
    # regression (reduces sensor noise, aligns ICR/ISF effects).
    window_min: int = 30
    # Drop windows whose CGM coverage is below this fraction (gaps/calibration).
    min_coverage: float = 0.6
    # Glucose readings outside this range are treated as sensor errors.
    bg_min: float = 39.0
    bg_max: float = 401.0
    # Exclude windows where |ΔBG| exceeds this (compression lows, etc.).
    max_abs_delta: float = 80.0


@dataclass
class PriorConfig:
    """Priors for the hierarchical model (population-level guesses).

    These centre the shared means; per-hour values are pulled toward them, so
    sparse hours (e.g. 3am) borrow strength from the rest of the day.
    """

    isf_mean: float = 50.0          # mg/dL per unit
    isf_sd_log: float = 0.5         # spread of the *population* mean (log scale)
    isf_pool_sd: float = 0.3        # how far hours may drift from the mean (log)

    csf_mean: float = 4.0           # mg/dL per gram (== isf/carb_ratio)
    csf_sd_log: float = 0.5
    csf_pool_sd: float = 0.3

    # Basal need is modelled (not a vague EGP): each hour is anchored on the
    # *scheduled* basal from the profile, allowed to drift per hour, with a
    # single global scale so the whole schedule can move up/down if the data
    # says so. The endogenous-glucose term is then EGP[h] = ISF[h]*basal[h].
    basal_scale_sd_log: float = 0.4   # prior sd on a whole-schedule multiplier
    basal_pool_sd: float = 0.3        # per-hour drift from scheduled (log)
    basal_floor: float = 0.02         # U/hr floor so log() is well-defined

    obs_sd: float = 12.0            # expected per-window residual (mg/dL)
    student_t_nu: float = 4.0       # robustness to outliers / unannounced meals


@dataclass
class SamplerConfig:
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    seed: int = 42


@dataclass
class SafetyConfig:
    """How conservative the delivered recommendations are."""

    # Max basal recommended = multiplier x largest recommended hourly basal.
    max_basal_multiplier: float = 3.5
    # Round basal/CR/ISF to clinician-friendly increments.
    basal_round: float = 0.05
    isf_round: float = 1.0
    carb_ratio_round: float = 0.5
    # Only move a setting if the posterior is at least this confident
    # (fraction of credible interval on one side of current value).
    min_confidence: float = 0.75
    # Never recommend changing a setting by more than this fraction at once.
    max_relative_change: float = 0.30


@dataclass
class LoopTunerConfig:
    nightscout: NightscoutConfig = field(default_factory=NightscoutConfig)
    pharmacology: PharmacologyConfig = field(default_factory=PharmacologyConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    # Optional fallback basal schedule if no Nightscout profile is available:
    # list of {"start": "HH:MM", "rate": units_per_hour}
    fallback_basal_schedule: Optional[list] = None

    @classmethod
    def from_file(cls, path: str) -> "LoopTunerConfig":
        with open(path) as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "LoopTunerConfig":
        return cls(
            nightscout=NightscoutConfig(**raw.get("nightscout", {})),
            pharmacology=PharmacologyConfig(**raw.get("pharmacology", {})),
            grid=GridConfig(**raw.get("grid", {})),
            priors=PriorConfig(**raw.get("priors", {})),
            sampler=SamplerConfig(**raw.get("sampler", {})),
            safety=SafetyConfig(**raw.get("safety", {})),
            fallback_basal_schedule=raw.get("fallback_basal_schedule"),
        )

    def to_dict(self) -> dict:
        return asdict(self)
