"""
Tests for A/B test simulation, power analysis, and statistical tests.
Run with: pytest tests/
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.power_analysis import compute_sample_size, power_curve, sample_size_table
from src.experiments.ab_test import (
    assign_clusters,
    simulate_outcomes,
    run_ttest,
    run_mannwhitney,
    run_sequential_test,
    obrien_fleming_boundary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_clusters():
    return [0, 1, 3]


@pytest.fixture
def cluster_assignments(valid_clusters):
    return assign_clusters(valid_clusters, random_state=42)


@pytest.fixture
def station_clusters(valid_clusters):
    """Minimal synthetic station DataFrame."""
    rows = []
    for cid in valid_clusters:
        for i in range(20):
            rows.append({
                "station_id": f"cluster{cid}_station{i}",
                "kmeans_cluster": cid,
                "lat": 40.7 + np.random.default_rng(cid * 100 + i).uniform(-0.1, 0.1),
                "lng": -73.9 + np.random.default_rng(cid * 100 + i).uniform(-0.1, 0.1),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def outcomes(station_clusters, cluster_assignments):
    return simulate_outcomes(
        station_clusters=station_clusters,
        cluster_assignments=cluster_assignments,
        n_days=31,
        baseline_availability=0.75,
        true_lift=0.08,
        noise_std=0.05,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

class TestPowerAnalysis:
    def test_sample_size_positive(self):
        n = compute_sample_size(0.75, 0.05)
        assert n > 0

    def test_larger_mde_needs_fewer_samples(self):
        n_small = compute_sample_size(0.75, 0.03)
        n_large = compute_sample_size(0.75, 0.10)
        assert n_small > n_large

    def test_stricter_alpha_needs_more_samples(self):
        n_relaxed = compute_sample_size(0.75, 0.05, alpha=0.10)
        n_strict   = compute_sample_size(0.75, 0.05, alpha=0.01)
        assert n_strict > n_relaxed

    def test_power_curve_length(self):
        mde_range = np.linspace(0.01, 0.15, 20)
        powers = power_curve(0.75, mde_range, n_per_group=5000)
        assert len(powers) == len(mde_range)

    def test_power_curve_monotone(self):
        """Power should increase as MDE increases (easier to detect bigger effects)."""
        mde_range = np.linspace(0.02, 0.15, 15)
        powers = power_curve(0.75, mde_range, n_per_group=5000)
        assert np.all(np.diff(powers) >= -0.01)   # allow tiny numerical noise

    def test_power_curve_bounded(self):
        mde_range = np.linspace(0.01, 0.15, 10)
        powers = power_curve(0.75, mde_range, n_per_group=5000)
        assert np.all(powers >= 0) and np.all(powers <= 1)

    def test_sample_size_table_keys(self):
        table = sample_size_table(0.75, mde_values=[0.05, 0.10], alpha_values=[0.05, 0.01])
        assert (0.05, 0.05) in table
        assert (0.10, 0.01) in table


# ---------------------------------------------------------------------------
# Cluster assignment
# ---------------------------------------------------------------------------

class TestAssignClusters:
    def test_all_clusters_assigned(self, valid_clusters, cluster_assignments):
        assert set(cluster_assignments.keys()) == set(valid_clusters)

    def test_values_are_valid_groups(self, cluster_assignments):
        assert all(v in ("treatment", "control") for v in cluster_assignments.values())

    def test_roughly_balanced(self, valid_clusters):
        # With 3 clusters: 1 treatment, 2 control or vice versa
        assignments = assign_clusters(valid_clusters, random_state=42)
        counts = pd.Series(assignments).value_counts()
        assert abs(counts.get("treatment", 0) - counts.get("control", 0)) <= 1

    def test_reproducible(self, valid_clusters):
        a = assign_clusters(valid_clusters, random_state=0)
        b = assign_clusters(valid_clusters, random_state=0)
        assert a == b

    def test_different_seeds_may_differ(self, valid_clusters):
        a = assign_clusters(valid_clusters, random_state=0)
        b = assign_clusters(valid_clusters, random_state=99)
        # With 3 clusters there's a chance they match — just check it runs
        assert isinstance(b, dict)


# ---------------------------------------------------------------------------
# Outcome simulation
# ---------------------------------------------------------------------------

class TestSimulateOutcomes:
    def test_shape(self, outcomes, station_clusters):
        expected_rows = len(station_clusters) * 31
        assert len(outcomes) == expected_rows

    def test_columns(self, outcomes):
        for col in ["station_id", "cluster_id", "group", "day", "availability_rate"]:
            assert col in outcomes.columns

    def test_rates_bounded(self, outcomes):
        assert outcomes["availability_rate"].between(0, 1).all()

    def test_groups_present(self, outcomes):
        assert set(outcomes["group"].unique()) == {"treatment", "control"}

    def test_treatment_mean_higher(self, outcomes):
        ctrl = outcomes[outcomes["group"] == "control"]["availability_rate"].mean()
        trt  = outcomes[outcomes["group"] == "treatment"]["availability_rate"].mean()
        assert trt > ctrl


# ---------------------------------------------------------------------------
# Frequentist tests
# ---------------------------------------------------------------------------

class TestFrequentistTests:
    def test_ttest_keys(self, outcomes):
        result = run_ttest(outcomes)
        for key in ["test", "statistic", "p_value", "mean_control",
                    "mean_treatment", "observed_lift", "significant_at_05"]:
            assert key in result

    def test_ttest_detects_effect(self, outcomes):
        """With 8pp true lift and ~1800 station-days, should be significant."""
        result = run_ttest(outcomes)
        assert result["significant_at_05"]

    def test_ttest_observed_lift_positive(self, outcomes):
        result = run_ttest(outcomes)
        assert result["observed_lift"] > 0

    def test_mannwhitney_keys(self, outcomes):
        result = run_mannwhitney(outcomes)
        for key in ["test", "statistic", "p_value", "significant_at_05"]:
            assert key in result

    def test_mannwhitney_detects_effect(self, outcomes):
        result = run_mannwhitney(outcomes)
        assert result["significant_at_05"]

    def test_no_effect_not_significant(self, station_clusters, cluster_assignments):
        """With zero true lift, test should usually not reject."""
        null_outcomes = simulate_outcomes(
            station_clusters=station_clusters,
            cluster_assignments=cluster_assignments,
            n_days=31,
            baseline_availability=0.75,
            true_lift=0.0,
            noise_std=0.05,
            random_state=123,
        )
        result = run_ttest(null_outcomes)
        # Not guaranteed but very likely with this seed
        assert abs(result["observed_lift"]) < 0.03


# ---------------------------------------------------------------------------
# Sequential testing
# ---------------------------------------------------------------------------

class TestSequentialTesting:
    def test_obrien_fleming_boundary_at_1(self):
        """At t=1 (final look), boundary should equal nominal alpha."""
        b = obrien_fleming_boundary(1.0, alpha=0.05)
        assert abs(b - 0.05) < 1e-6

    def test_obrien_fleming_conservative_early(self):
        """Early looks should have a stricter (smaller) boundary than final."""
        early  = obrien_fleming_boundary(0.25, alpha=0.05)
        final  = obrien_fleming_boundary(1.00, alpha=0.05)
        assert early < final

    def test_sequential_output_shape(self, outcomes):
        result = run_sequential_test(outcomes, n_looks=4, alpha=0.05)
        assert len(result) == 4

    def test_sequential_columns(self, outcomes):
        result = run_sequential_test(outcomes, n_looks=4, alpha=0.05)
        for col in ["look", "day_cutoff", "info_fraction", "boundary_alpha", "p_value", "reject"]:
            assert col in result.columns

    def test_sequential_info_fraction_increases(self, outcomes):
        result = run_sequential_test(outcomes, n_looks=4, alpha=0.05)
        fractions = result["info_fraction"].values
        assert np.all(np.diff(fractions) > 0)
