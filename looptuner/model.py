"""Hierarchical Bayesian glucodynamic regression.

Model (per analysis window ``t``, grouped by hour-of-day ``h[t]``)::

    ΔBG[t] ~ StudentT(nu,
                      mu = -ISF[h] * insulin_act[t]
                           + CSF[h] * carb_act[t]
                           + EGP[h] * dt_hours[t],
                      sigma)

where the per-hour sensitivities are partially pooled toward a shared
population mean (hierarchical priors):

    log ISF[h] ~ Normal(log ISF_mean, isf_pool_sd)
    log CSF[h] ~ Normal(log CSF_mean, csf_pool_sd)

The partial pooling is what lets sparse hours (e.g. 3am) borrow strength from
neighbouring well-sampled hours: with little local data the posterior for that
hour is pulled toward the population mean rather than overfitting noise.

The endogenous-glucose term is not a vague free parameter -- that would be
confounded with ISF at fasting hours and bias it badly. Instead the *basal the
body needs* is modelled directly, anchored on the currently-scheduled basal
(known from the profile) with one global scale and a per-hour drift::

    log basal[h] ~ Normal(log scheduled_basal[h] + global_scale, basal_pool_sd)
    EGP[h]        = ISF[h] * basal[h]

so the basal recommendation is read straight off the posterior and the
endogenous term carries the correct physiological scale.

Interpretation of the parameters:

* ``ISF[h]``  -- insulin sensitivity factor, mg/dL drop per unit of insulin.
* ``CSF[h]``  -- carb sensitivity factor, mg/dL rise per gram. The carb ratio
  is recovered as ``CR[h] = ISF[h] / CSF[h]`` (grams covered per unit).
* ``EGP[h]``  -- net endogenous glucose appearance, mg/dL per hour, *given the
  basal that was actually delivered*. Because delivered basal is folded into
  ``insulin_act``, a positive ``EGP`` means the body needed more basal than it
  got (glucose drifted up); the basal correction follows from ISF and EGP.

StudentT observation noise makes the fit robust to unannounced meals,
compression lows and other fat-tailed artefacts that survive preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PriorConfig, SamplerConfig
from .preprocess import Dataset

N_HOURS = 24


@dataclass
class FitResult:
    """Posterior summaries, per hour-of-day where applicable."""

    isf: np.ndarray            # (N_HOURS,) posterior mean, mg/dL per U
    isf_hdi: np.ndarray        # (N_HOURS, 2) 94% HDI
    csf: np.ndarray            # (N_HOURS,) mg/dL per g
    csf_hdi: np.ndarray
    carb_ratio: np.ndarray     # (N_HOURS,) g per U = isf/csf
    carb_ratio_hdi: np.ndarray
    basal: np.ndarray          # (N_HOURS,) U/hr the body needs
    basal_hdi: np.ndarray
    egp: np.ndarray            # (N_HOURS,) mg/dL per hr = isf*basal
    egp_hdi: np.ndarray
    isf_population: float       # shared mean
    isf_population_hdi: np.ndarray          # (2,) 94% CI
    csf_population: float
    carb_ratio_population: float            # isf_pop / csf_pop
    carb_ratio_population_hdi: np.ndarray   # (2,)
    basal_scale: float         # global multiplier vs scheduled basal
    basal_scale_hdi: np.ndarray             # (2,)
    obs_sd: float
    n_per_hour: np.ndarray     # data count per hour (drives confidence)
    n_carb_per_hour: np.ndarray             # windows with carb absorption per hour
    idata: object = None       # arviz InferenceData (full posterior)
    diagnostics: dict = None


def fit(
    data: Dataset,
    priors: PriorConfig,
    sampler: SamplerConfig,
    scheduled_basal: np.ndarray,
    progressbar: bool = True,
) -> FitResult:
    """Sample the posterior and return per-hour summaries.

    ``scheduled_basal`` is the (24,) currently-scheduled basal rate (U/hr); it
    anchors the basal-need prior so the endogenous-glucose scale is correct.
    """
    import pymc as pm
    import arviz as az

    hour_idx = data.hour
    counts = np.bincount(hour_idx, minlength=N_HOURS)
    log_sched_basal = np.log(np.maximum(scheduled_basal, priors.basal_floor))

    with pm.Model() as model:
        # --- population (shared) means for the pooled sensitivities ---
        log_isf_mu = pm.Normal(
            "log_isf_mu", mu=np.log(priors.isf_mean), sigma=priors.isf_sd_log
        )
        log_csf_mu = pm.Normal(
            "log_csf_mu", mu=np.log(priors.csf_mean), sigma=priors.csf_sd_log
        )

        # --- per-hour deviations (non-centered for sampling stability) ---
        isf_offset = pm.Normal("isf_offset", 0.0, 1.0, shape=N_HOURS)
        csf_offset = pm.Normal("csf_offset", 0.0, 1.0, shape=N_HOURS)
        basal_offset = pm.Normal("basal_offset", 0.0, 1.0, shape=N_HOURS)
        # One global knob to move the whole basal schedule up/down.
        log_basal_scale = pm.Normal("log_basal_scale", 0.0, priors.basal_scale_sd_log)

        log_isf = log_isf_mu + isf_offset * priors.isf_pool_sd
        log_csf = log_csf_mu + csf_offset * priors.csf_pool_sd
        isf = pm.Deterministic("isf", pm.math.exp(log_isf))
        csf = pm.Deterministic("csf", pm.math.exp(log_csf))
        pm.Deterministic("carb_ratio", isf / csf)

        # Basal need anchored on the scheduled schedule (informative, correct
        # scale), with a global scale + per-hour drift.
        log_basal = (
            log_sched_basal + log_basal_scale + basal_offset * priors.basal_pool_sd
        )
        basal = pm.Deterministic("basal", pm.math.exp(log_basal))
        # Endogenous glucose appearance implied by that basal need.
        egp = pm.Deterministic("egp", isf * basal)

        # --- likelihood ---
        mu = (
            -isf[hour_idx] * data.insulin_act
            + csf[hour_idx] * data.carb_act
            + egp[hour_idx] * data.dt_hours
        )
        sigma = pm.HalfNormal("obs_sd", sigma=priors.obs_sd)
        pm.StudentT(
            "delta_bg",
            nu=priors.student_t_nu,
            mu=mu,
            sigma=sigma,
            observed=data.delta_bg,
        )

        sample_kwargs = dict(
            draws=sampler.draws,
            tune=sampler.tune,
            chains=sampler.chains,
            target_accept=sampler.target_accept,
            random_seed=sampler.seed,
            compute_convergence_checks=True,
        )
        try:
            idata = pm.sample(progressbar=progressbar, **sample_kwargs)
        except ImportError:
            # PyMC's rich progress bar optionally imports matplotlib; if that
            # (or any progress-bar dependency) is missing, sample without it
            # rather than failing the whole run.
            idata = pm.sample(progressbar=False, **sample_kwargs)

    post = idata.posterior
    isf_mean = post["isf"].mean(dim=("chain", "draw")).to_numpy()
    csf_mean = post["csf"].mean(dim=("chain", "draw")).to_numpy()
    cr_mean = post["carb_ratio"].mean(dim=("chain", "draw")).to_numpy()
    basal_mean = post["basal"].mean(dim=("chain", "draw")).to_numpy()
    egp_mean = post["egp"].mean(dim=("chain", "draw")).to_numpy()

    diagnostics = _diagnostics(az, idata)

    # Whole-day (population) parameters, pooled over all windows. These stay
    # well-identified even when no single hour has enough data, so they drive
    # the whole-day recommendation. Equal-tailed 94% CI straight from draws.
    isf_pop_draws = np.exp(post["log_isf_mu"].to_numpy().ravel())
    csf_pop_draws = np.exp(post["log_csf_mu"].to_numpy().ravel())
    cr_pop_draws = isf_pop_draws / csf_pop_draws
    scale_draws = np.exp(post["log_basal_scale"].to_numpy().ravel())
    n_carb_per_hour = np.array(
        [int((data.carb_act[hour_idx == h] > 0).sum()) for h in range(N_HOURS)]
    )

    return FitResult(
        isf=isf_mean,
        isf_hdi=_hdi(az, idata, "isf"),
        csf=csf_mean,
        csf_hdi=_hdi(az, idata, "csf"),
        carb_ratio=cr_mean,
        carb_ratio_hdi=_hdi(az, idata, "carb_ratio"),
        basal=basal_mean,
        basal_hdi=_hdi(az, idata, "basal"),
        egp=egp_mean,
        egp_hdi=_hdi(az, idata, "egp"),
        isf_population=float(np.mean(isf_pop_draws)),
        isf_population_hdi=_ci(isf_pop_draws),
        csf_population=float(np.mean(csf_pop_draws)),
        carb_ratio_population=float(np.mean(cr_pop_draws)),
        carb_ratio_population_hdi=_ci(cr_pop_draws),
        basal_scale=float(np.mean(scale_draws)),
        basal_scale_hdi=_ci(scale_draws),
        obs_sd=float(post["obs_sd"].mean()),
        n_per_hour=counts,
        n_carb_per_hour=n_carb_per_hour,
        idata=idata,
        diagnostics=diagnostics,
    )


def _ci(draws: np.ndarray, prob: float = 0.94) -> np.ndarray:
    lo = (1 - prob) / 2
    return np.quantile(draws, [lo, 1 - lo])


def _hdi(az, idata, var: str, prob: float = 0.94) -> np.ndarray:
    """Return the (n, 2) credible interval for ``var``.

    arviz renamed the probability kwarg (``hdi_prob`` -> ``prob``) across
    versions, so try both. If ``az.hdi`` is unavailable or incompatible, fall
    back to an equal-tailed quantile interval computed directly from the
    posterior draws.
    """
    for kwargs in ({"hdi_prob": prob}, {"prob": prob}):
        try:
            hdi = np.asarray(az.hdi(idata, var_names=[var], **kwargs)[var].to_numpy())
            # Normalize to (n, 2) regardless of arviz's axis ordering.
            if hdi.ndim == 2 and hdi.shape[0] == 2 and hdi.shape[1] != 2:
                hdi = hdi.T
            return hdi
        except TypeError:
            continue
        except Exception:
            break

    # Manual fallback: equal-tailed interval over chain+draw samples. Posterior
    # dims are (chain, draw, var_dim); collapse the sampling dims.
    arr = idata.posterior[var].to_numpy()
    arr = arr.reshape(-1, arr.shape[-1])  # (chain*draw, n)
    lo = (1 - prob) / 2
    bounds = np.quantile(arr, [lo, 1 - lo], axis=0)  # (2, n)
    return bounds.T


def _diagnostics(az, idata) -> dict:
    summary = az.summary(
        idata, var_names=["isf", "csf", "basal", "obs_sd"], kind="diagnostics"
    )
    return {
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "n_divergences": int(idata.sample_stats["diverging"].sum())
        if "diverging" in idata.sample_stats
        else 0,
    }
