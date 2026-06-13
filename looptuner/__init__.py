"""LoopTuner — Bayesian tuning suggestions for iOS closed-loop T1D settings.

NOT A MEDICAL DEVICE. All output is a suggestion to review with a licensed
clinician before changing any pump or loop setting.
"""

from .config import LoopTunerConfig
from .pipeline import PipelineResult, run, run_from_records
from .report import render

__version__ = "0.1.0"

__all__ = [
    "LoopTunerConfig",
    "PipelineResult",
    "run",
    "run_from_records",
    "render",
]
