"""
Bayesian A/B testing using PyMC.
Models dock availability rate with Beta-distributed priors.
Computes posterior on uplift and P(treatment > control).
"""
import numpy as np
import pandas as pd


def run_bayesian_ab(
    outcomes: pd.DataFrame,
    n_samples: int = 1000,
    random_seed: int = 42,
):
    """
    Fit a Bayesian A/B model using PyMC.

    Aggregates to daily group means before sampling (31 obs per group)
    to keep sampling fast while remaining statistically valid.

    Model:
        p_control   ~ Beta(1, 1)      # uninformative prior
        p_treatment ~ Beta(1, 1)
        obs_control   ~ Normal(p_control,   sigma)
        obs_treatment ~ Normal(p_treatment, sigma)
        uplift = p_treatment - p_control

    Returns (trace, summary_dict).
    """
    import pymc as pm

    # Aggregate to daily means — reduces likelihood size from ~35k to ~31 points
    daily = outcomes.groupby(['day', 'group'])['availability_rate'].mean().reset_index()
    ctrl = daily[daily['group'] == 'control']['availability_rate'].values
    trt  = daily[daily['group'] == 'treatment']['availability_rate'].values

    with pm.Model() as model:
        # Priors
        mu_ctrl = pm.Beta('mu_control',   alpha=1, beta=1)
        mu_trt  = pm.Beta('mu_treatment', alpha=1, beta=1)
        sigma   = pm.HalfNormal('sigma', sigma=0.1)

        # Likelihood
        pm.Normal('obs_control',   mu=mu_ctrl, sigma=sigma, observed=ctrl)
        pm.Normal('obs_treatment', mu=mu_trt,  sigma=sigma, observed=trt)

        # Derived quantity
        uplift = pm.Deterministic('uplift', mu_trt - mu_ctrl)

        trace = pm.sample(
            n_samples,
            tune=200,
            chains=1,       # single chain avoids Windows multiprocessing overhead
            progressbar=True,
            random_seed=random_seed,
            target_accept=0.9,
        )

    uplift_samples = trace.posterior['uplift'].values.flatten()
    prob_better    = float((uplift_samples > 0).mean())
    hdi            = _hdi(uplift_samples, credible_mass=0.95)

    summary = {
        'mean_uplift':      round(float(uplift_samples.mean()), 4),
        'median_uplift':    round(float(np.median(uplift_samples)), 4),
        'hdi_95_low':       round(hdi[0], 4),
        'hdi_95_high':      round(hdi[1], 4),
        'prob_treatment_better': round(prob_better, 4),
    }
    return trace, summary


def _hdi(samples: np.ndarray, credible_mass: float = 0.95) -> tuple:
    """Compute highest density interval for a 1-D sample array."""
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_width = int(np.floor(credible_mass * n))
    widths = sorted_samples[interval_width:] - sorted_samples[:n - interval_width]
    min_idx = int(np.argmin(widths))
    return (sorted_samples[min_idx], sorted_samples[min_idx + interval_width])
