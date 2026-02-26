"""
Statistical power analysis.
Calculates minimum detectable effect and required sample size
for the rebalancing A/B test.
"""
import numpy as np
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower


def compute_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Required sample size per group (station-days) for a two-proportion z-test.

    Parameters
    ----------
    baseline_rate : float  Baseline dock availability rate (e.g. 0.75)
    mde           : float  Minimum detectable effect in percentage points (e.g. 0.05)
    alpha         : float  Type I error rate
    power         : float  1 - Type II error rate

    Returns
    -------
    n : int  Required observations per group
    """
    treatment_rate = baseline_rate + mde
    effect_size = proportion_effectsize(treatment_rate, baseline_rate)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
    return int(np.ceil(n))


def power_curve(
    baseline_rate: float,
    mde_range: np.ndarray,
    n_per_group: int,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Compute achieved power for a range of MDEs given a fixed sample size.
    Returns array of power values matching mde_range.
    """
    analysis = NormalIndPower()
    powers = []
    for mde in mde_range:
        effect_size = proportion_effectsize(baseline_rate + mde, baseline_rate)
        p = analysis.solve_power(effect_size=effect_size, alpha=alpha, nobs1=n_per_group, alternative='two-sided')
        powers.append(p)
    return np.array(powers)


def sample_size_table(
    baseline_rate: float,
    mde_values: list,
    alpha_values: list = [0.05, 0.01],
    power: float = 0.80,
) -> dict:
    """Return a dict of {(mde, alpha): required_n} for a quick comparison table."""
    table = {}
    for mde in mde_values:
        for alpha in alpha_values:
            table[(mde, alpha)] = compute_sample_size(baseline_rate, mde, alpha, power)
    return table
