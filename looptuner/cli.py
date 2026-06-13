"""Command-line interface for LoopTuner.

Examples
--------
Run against a live Nightscout site described by a config file::

    python -m looptuner.cli run --config myconfig.json

Validate the whole pipeline offline on synthetic data with known truth::

    python -m looptuner.cli demo
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import LoopTunerConfig
from .pipeline import run, run_from_records
from .report import render


def _cmd_run(args) -> int:
    cfg = LoopTunerConfig.from_file(args.config)
    if args.days:
        cfg.nightscout.days = args.days
    if args.profile:
        cfg.nightscout.profile_name = args.profile
    result = run(cfg, use_cache=not args.no_cache, progressbar=not args.quiet)
    print(render(result.recommendations))
    if args.json_out:
        _write_json(result, args.json_out)
    return 0


def _cmd_profiles(args) -> int:
    from .nightscout import NightscoutClient
    from .profile import list_profiles

    cfg = LoopTunerConfig.from_file(args.config)
    docs = NightscoutClient(cfg.nightscout).profile(use_cache=not args.no_cache)
    names, default = list_profiles(docs)
    if not names:
        print("No named profiles found in Nightscout.")
        return 0
    print("Available Nightscout profiles:")
    for n in names:
        print(f"  {'* ' if n == default else '  '}{n}")
    print("\n(* = site default. Use:  run --profile <name>)")
    return 0


def _cmd_ui(args) -> int:
    from .webui import serve
    serve(host=args.host, port=args.port)
    return 0


def _cmd_demo(args) -> int:
    from .synthetic import generate
    import numpy as np

    data = generate(days=args.days, seed=args.seed)
    cfg = LoopTunerConfig()
    cfg.sampler.draws = args.draws
    cfg.sampler.tune = args.draws
    cfg.sampler.chains = args.chains
    result = run_from_records(
        cfg, data["entries"], data["treatments"], data["profile"],
        progressbar=not args.quiet,
    )
    print(render(result.recommendations))

    truth = data["truth"]
    est_isf = result.fit.isf
    est_cr = result.fit.carb_ratio
    est_basal = result.fit.basal
    print("\nRecovery vs. ground truth (mean abs % error across 24 hours)")
    print("-" * 60)
    print(f"  ISF:        {_mape(est_isf, truth['isf']):.1f}%")
    print(f"  Carb ratio: {_mape(est_cr, truth['carb_ratio']):.1f}%")
    print(f"  Basal:      {_mape(est_basal, truth['basal']):.1f}%")
    return 0


def _mape(est, truth) -> float:
    import numpy as np
    return float(np.mean(np.abs((est - truth) / truth)) * 100)


def _write_json(result, path) -> None:
    recs = result.recommendations
    payload = {
        "summary": recs.summary,
        "max_basal": recs.max_basal,
        "max_bolus": recs.max_bolus,
        "suspend_threshold": recs.suspend_threshold,
        "isf": [vars(r) for r in recs.isf],
        "carb_ratio": [vars(r) for r in recs.carb_ratio],
        "basal": [vars(r) for r in recs.basal],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\nWrote machine-readable recommendations to {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="looptuner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="fetch Nightscout data and recommend")
    p_run.add_argument("--config", required=True, help="path to JSON config")
    p_run.add_argument("--days", type=int, help="override history window")
    p_run.add_argument("--profile", help="Nightscout profile name (default: site default)")
    p_run.add_argument("--no-cache", action="store_true")
    p_run.add_argument("--quiet", action="store_true")
    p_run.add_argument("--json-out", help="also write recommendations as JSON")
    p_run.set_defaults(func=_cmd_run)

    p_profiles = sub.add_parser("profiles", help="list available Nightscout profiles")
    p_profiles.add_argument("--config", required=True, help="path to JSON config")
    p_profiles.add_argument("--no-cache", action="store_true")
    p_profiles.set_defaults(func=_cmd_profiles)

    p_ui = sub.add_parser("ui", help="launch the local web UI (no config file)")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.set_defaults(func=_cmd_ui)

    p_demo = sub.add_parser("demo", help="run on synthetic data and report recovery")
    p_demo.add_argument("--days", type=int, default=21)
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--draws", type=int, default=500)
    p_demo.add_argument("--chains", type=int, default=2)
    p_demo.add_argument("--quiet", action="store_true")
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
