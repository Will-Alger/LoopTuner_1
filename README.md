# LoopTuner

**Bayesian tuning suggestions for iOS closed-loop (Loop / Trio / iAPS) Type-1
diabetes settings, learned from your own Nightscout history.**

> ⚠️ **NOT A MEDICAL DEVICE — NOT MEDICAL ADVICE.** LoopTuner is a personal
> analysis aid. Every number it prints is a *suggestion* derived from
> historical data and a deliberately simplified physiological model. Insulin
> dosing errors can be dangerous. **Do not change any pump, basal, ISF, carb
> ratio, or delivery-limit setting without review and approval from your
> licensed clinician.** Use entirely at your own risk.

---

## What it does

LoopTuner pulls your CGM, insulin, and carb history from Nightscout and fits a
**hierarchical Bayesian glucodynamic regression** to estimate, for each hour of
the day:

- **Insulin Sensitivity Factor (ISF)** — mg/dL drop per unit of insulin
- **Carb Ratio (CR)** — grams covered per unit of insulin
- **Basal rate** — units/hour your body actually needs
- **Endogenous glucose** drift used internally to derive basal

…and from those, suggested **safety delivery limits** (max basal, max bolus,
suspend threshold).

### The model

For each ~30-minute analysis window *t*, grouped by its hour-of-day *h*:

```
ΔBG[t] ~ StudentT(ν,
    mean = -ISF[h] · insulin_activity[t]      # insulin lowers glucose
           + CSF[h] · carb_activity[t]        # carbs raise glucose
           + EGP[h] · Δt,                      # endogenous drift
    σ)
```

where the per-hour parameters are **partially pooled** toward a shared
population mean:

```
log ISF[h] ~ Normal(log ISF_population, pool_sd)
log CSF[h] ~ Normal(log CSF_population, pool_sd)     CR[h] = ISF[h] / CSF[h]
EGP[h]      = ISF[h] · basal_need[h]
log basal_need[h] ~ Normal(log scheduled_basal[h] + global_scale, pool_sd)
```

**Why hierarchical / partial pooling?** Hours with sparse or uninformative data
(e.g. 3 am, when you're fasting and rarely bolus) cannot identify their own ISF
in isolation. The pooling priors pull those hours toward the population mean
and toward your existing schedule, so they "borrow strength" from
well-sampled hours instead of overfitting noise. Well-sampled hours (around
meals) move freely to fit the data.

Key modelling choices:

- **`insulin_activity`** includes *both boluses and reconstructed basal
  delivery* (scheduled basal overridden by temp basals), convolved with the
  LoopKit exponential insulin-action curve. Folding basal in is what makes the
  endogenous term `EGP = ISF · basal_need` interpretable, so the basal you
  *need* is read straight off the posterior.
- **`carb_activity`** uses a simple piecewise-linear absorption model
  (per-entry absorption time, default 3 h).
- **StudentT** observation noise makes the fit robust to unannounced meals,
  compression lows, and other fat-tailed artefacts.

### Honest about uncertainty

Each suggestion carries a 94% credible interval. A change is only flagged as
*data-supported* (`*` in the report) when (a) there is enough data for that
hour and (b) your current setting falls **outside** the credible interval.
Changes are additionally **capped** (default ≤30% per run) and rounded to
clinician-friendly increments. Fasting-hour ISF, which is mathematically
unidentifiable from this data, will simply track the population mean and never
be flagged as a confident change.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

PyMC compiles a small C/NUTS sampler on first run.

## Validate it offline (no Nightscout needed)

`demo` generates synthetic data from *known* per-hour ISF/CR/basal, runs the
full pipeline, and reports how well the posterior recovers the truth:

```bash
python -m looptuner.cli demo --days 21
```

Typical recovery on the synthetic benchmark: **ISF ≈ 14%**, **carb ratio ≈ 11%**
(≈4% at hours where meals actually occur), **basal ≈ 8%** mean absolute error —
with the residual ISF error concentrated entirely in fasting hours, exactly
where the data cannot identify it.

## Run against your Nightscout

1. Copy `examples/config.example.json` to `config.json` and fill in your
   Nightscout `url` and `api_secret` (or `token`).
2. Run:

```bash
python -m looptuner.cli run --config config.json --json-out recommendations.json
```

The first run caches the raw Nightscout pull under `.looptuner_cache/`; re-runs
are instant and offline. Add `--no-cache` to refresh.

### Example output (abridged)

```
Insulin sensitivity factor (ISF)   [* = data-supported change]
------------------------------------------------------------------------
  00:00-07:00  current=    50  ->  50 mg/dL/U
  07:00-11:00  current=    50  ->  43 mg/dL/U  *
  ...
Basal rate
------------------------------------------------------------------------
  00:00-03:00  current=  0.85  ->  0.95 U/hr  *
  ...
Safety delivery limits (suggested)
  Maximum basal rate:  3.5 U/hr
  Maximum bolus:       16.5 U
  Suspend threshold:   100 mg/dL
```

---

## Configuration

All knobs live in the JSON config (see `examples/config.example.json`) and map
to the dataclasses in `looptuner/config.py`:

| Section | Key settings |
|---|---|
| `nightscout` | `url`, `api_secret`/`token`, `days` of history |
| `pharmacology` | insulin `peak`/`duration` (DIA), carb absorption time |
| `grid` | window length, coverage and outlier thresholds |
| `priors` | population ISF/CR means, pooling strength |
| `sampler` | NUTS draws / tune / chains |
| `safety` | max change per run, rounding, delivery-limit multipliers |

If Nightscout has no usable profile, set `fallback_basal_schedule` (a basal
schedule is required to reconstruct insulin delivery).

## Project layout

```
looptuner/
  pharmacology.py   insulin-action & carb-absorption models (pure, tested)
  nightscout.py     Nightscout REST client (paginated, cached)
  profile.py        parse basal/ISF/CR schedules from a profile
  preprocess.py     build the windowed regression dataset
  model.py          PyMC hierarchical Bayesian model
  recommend.py      posterior -> guard-railed setting suggestions
  report.py         human-readable rendering (+ disclaimers)
  synthetic.py      ground-truth data generator for validation
  pipeline.py       fetch -> features -> fit -> recommend
  cli.py            `run` and `demo` commands
tests/              unit + end-to-end recovery tests
```

## Limitations

- Estimates are only as good as the logged data: unannounced meals, missed
  boluses, bad CGM calibration, and exercise all add noise (the StudentT
  likelihood absorbs some, not all).
- Fasting-hour ISF is not identifiable from glucose dynamics alone; treat those
  values as "consistent with population", not measurements.
- The carb and insulin models are intentionally simple and fixed (not Loop's
  dynamic carb absorption). They are good enough to *tune ratios*, not to
  replicate a closed loop.
- This tool does not model exercise, illness, hormones, or site failures.

---

*LoopTuner is independent software and is not affiliated with or endorsed by
Loop, Tidepool, Nightscout, or any device manufacturer.*
