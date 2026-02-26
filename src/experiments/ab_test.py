"""
A/B test design and simulation for rebalancing strategies.
- Unit of randomization: station cluster
- Treatment: ML-optimized rebalancing schedule
- Control: current heuristic schedule
- Primary metric: dock availability rate (proportion of hours with ≥1 bike)
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Cluster assignment
# ---------------------------------------------------------------------------

def assign_clusters(cluster_ids: list, random_state: int = 42) -> dict:
    """
    Randomly assign clusters to treatment / control (50/50 split).
    Returns {cluster_id: 'treatment' | 'control'}.
    """
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(cluster_ids)
    mid = len(shuffled) // 2
    return {
        **{c: 'control'   for c in shuffled[:mid]},
        **{c: 'treatment' for c in shuffled[mid:]},
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_outcomes(
    station_clusters: pd.DataFrame,
    cluster_assignments: dict,
    n_days: int = 31,
    baseline_availability: float = 0.75,
    true_lift: float = 0.08,
    noise_std: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate daily dock availability rate per station for a test period.

    For each station-day:
      - Control: availability ~ N(baseline, noise_std), clipped [0,1]
      - Treatment: availability ~ N(baseline + true_lift, noise_std), clipped [0,1]

    Returns DataFrame with columns:
      station_id, cluster_id, group, day, availability_rate
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for _, row in station_clusters.iterrows():
        sid = row['station_id']
        cluster_id = int(row['kmeans_cluster'])
        group = cluster_assignments.get(cluster_id, 'control')
        mean = baseline_availability + (true_lift if group == 'treatment' else 0.0)
        rates = rng.normal(loc=mean, scale=noise_std, size=n_days).clip(0, 1)
        for day, rate in enumerate(rates):
            rows.append({
                'station_id': sid,
                'cluster_id': cluster_id,
                'group': group,
                'day': day,
                'availability_rate': rate,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Frequentist tests
# ---------------------------------------------------------------------------

def run_ttest(outcomes: pd.DataFrame) -> dict:
    """
    Two-sample independent t-test on mean availability_rate.
    Returns dict with statistic, p_value, mean_control, mean_treatment, observed_lift.
    """
    control   = outcomes[outcomes['group'] == 'control']['availability_rate']
    treatment = outcomes[outcomes['group'] == 'treatment']['availability_rate']
    stat, pval = stats.ttest_ind(treatment, control, equal_var=False)
    return {
        'test': 'Welch t-test',
        'statistic': round(stat, 4),
        'p_value': round(pval, 4),
        'mean_control': round(control.mean(), 4),
        'mean_treatment': round(treatment.mean(), 4),
        'observed_lift': round(treatment.mean() - control.mean(), 4),
        'significant_at_05': pval < 0.05,
    }


def run_mannwhitney(outcomes: pd.DataFrame) -> dict:
    """Mann-Whitney U test (non-parametric alternative to t-test)."""
    control   = outcomes[outcomes['group'] == 'control']['availability_rate']
    treatment = outcomes[outcomes['group'] == 'treatment']['availability_rate']
    stat, pval = stats.mannwhitneyu(treatment, control, alternative='two-sided')
    return {
        'test': 'Mann-Whitney U',
        'statistic': round(stat, 4),
        'p_value': round(pval, 4),
        'significant_at_05': pval < 0.05,
    }


# ---------------------------------------------------------------------------
# Sequential testing (O'Brien-Fleming alpha spending)
# ---------------------------------------------------------------------------

def obrien_fleming_boundary(t: float, alpha: float = 0.05) -> float:
    """
    O'Brien-Fleming alpha spending boundary at information fraction t ∈ (0,1].
    Returns the nominal alpha to use at this interim look.
    """
    from scipy.stats import norm
    z_final = norm.ppf(1 - alpha / 2)
    z_boundary = z_final / np.sqrt(t)
    return 2 * (1 - norm.cdf(z_boundary))


def run_sequential_test(
    outcomes: pd.DataFrame,
    n_looks: int = 4,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Simulate sequential interim analyses at equal information fractions.
    At each look, test using the O'Brien-Fleming boundary.

    Returns DataFrame with columns:
      look, day_cutoff, info_fraction, boundary_alpha, p_value, reject
    """
    max_day = outcomes['day'].max()
    looks = np.linspace(max_day // n_looks, max_day, n_looks, dtype=int)
    results = []
    for look_num, day_cut in enumerate(looks, start=1):
        t = (look_num) / n_looks
        boundary = obrien_fleming_boundary(t, alpha)
        subset = outcomes[outcomes['day'] <= day_cut]
        ctrl = subset[subset['group'] == 'control']['availability_rate']
        trt  = subset[subset['group'] == 'treatment']['availability_rate']
        _, pval = stats.ttest_ind(trt, ctrl, equal_var=False)
        results.append({
            'look': look_num,
            'day_cutoff': int(day_cut),
            'info_fraction': round(t, 2),
            'boundary_alpha': round(boundary, 4),
            'p_value': round(pval, 4),
            'reject': pval < boundary,
        })
    return pd.DataFrame(results)
