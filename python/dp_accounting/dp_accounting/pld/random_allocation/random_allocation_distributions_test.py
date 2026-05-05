"""
Core unit tests for random-allocation internals.

Includes distribution discretization coverage and external interface conversion checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from .random_allocation_distributions import (
    DenseDiscreteDist,
    Domain,
    PLDRealization,
    _compute_discrete_prob as compute_discrete_PMF,
    _zero_mass,
    compute_bin_log_ratio,
    compute_bin_width,
    compute_truncation,
    discretize_aligned_grid,
    discretize_aligned_range,
    discretize_continuous_distribution,
    enforce_mass_conservation,
    rediscretize_dist,
    rediscretize_prob as pmf_remap_to_grid_kernel,
)
from .random_allocation_core import linear_dist_to_dp_accounting_pmf
from .random_allocation_types import BoundType, RegularGrid, SpacingType
from .random_allocation_utils import exp_linear_to_geometric, log_geometric_to_linear
from scipy import stats


def _make_realization() -> PLDRealization:
    return PLDRealization(
        x_min=-0.5,
        step=0.5,
        prob_arr=np.array([0.2, 0.3, 0.25, 0.15], dtype=np.float64),
        p_max=0.1,
    )


def test_linear_dist_to_dp_accounting_preserves_shape_and_infinity_mass():
    original = _make_realization()
    pmf = linear_dist_to_dp_accounting_pmf(dist=original, pessimistic_estimate=True)

    assert pmf._probs.shape == original.prob_arr.shape
    assert pmf._infinity_mass == original.p_max
    assert pmf._discretization == original.step


def test_linear_dist_to_dp_accounting_handles_zero_finite_mass():
    realization = DenseDiscreteDist(
        x_min=0.0,
        step=1.0,
        prob_arr=np.array([0.0, 0.0], dtype=np.float64),
        p_max=1.0,
    )
    pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
    assert pmf._infinity_mass == 1.0
    assert np.allclose(pmf._probs, np.array([0.0, 0.0]))


def test_exp_log_round_trip_preserves_linear_step():
    original = DenseDiscreteDist(
        x_min=0.0,
        step=1e-4,
        prob_arr=np.array([0.4, 0.6], dtype=np.float64),
    )

    round_tripped = log_geometric_to_linear(exp_linear_to_geometric(original))

    assert round_tripped.step == original.step


def test_dense_dist_keeps_regular_grid_as_source_of_truth():
    grid = RegularGrid(
        x_min=1.0,
        step=1e-4,
        size=2,
        spacing_type=SpacingType.GEOMETRIC,
    )
    dist = DenseDiscreteDist.from_grid(
        grid=grid,
        prob_arr=np.array([0.4, 0.6], dtype=np.float64),
        domain=Domain.POSITIVES,
    )

    assert dist.grid == grid
    assert dist.step == 1e-4


class TestDiscritizeRange:
    """Test discretize_aligned_range function."""

    def test_linear_spacing(self):
        n_grid = 100
        x = discretize_aligned_range(
            x_min=0.0,
            x_max=10.0,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=(10.0 - 0.0) / (n_grid - 1),
        )
        assert len(x) >= n_grid
        assert x[0] <= 0.0
        assert x[-1] >= 10.0
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    def test_geometric_spacing(self):
        n_grid = 100
        x = discretize_aligned_range(
            x_min=1.0,
            x_max=100.0,
            spacing_type=SpacingType.GEOMETRIC,
            align_to_multiples=True,
            discretization=np.log(100.0 / 1.0) / (n_grid - 1),
        )
        assert len(x) >= n_grid
        assert x[0] <= 1.0
        assert x[-1] >= 100.0
        ratios = x[1:] / x[:-1]
        assert np.allclose(ratios, ratios[0])

    def test_nonpositive_discretization_rejected(self):
        with pytest.raises(ValueError, match="discretization must be positive"):
            discretize_aligned_range(
                x_min=0.0,
                x_max=10.0,
                spacing_type=SpacingType.LINEAR,
                align_to_multiples=True,
                discretization=0.0,
            )

    def test_two_points_linear(self):
        n_grid = 100
        x = discretize_aligned_range(
            x_min=1.0,
            x_max=3.0,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=(3.0 - 1.0) / (n_grid - 1),
        )
        assert len(x) >= n_grid
        assert x[0] <= 1.0
        assert x[-1] >= 3.0
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    def test_linear_spacing_covers_endpoint_after_alignment_rounding(self):
        x_min = -1.411426541779732
        x_max = 1.4160541697856062
        discretization = 0.041648652052517825

        x = discretize_aligned_range(
            x_min=x_min,
            x_max=x_max,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=discretization,
        )

        assert x[0] <= x_min
        assert x[-1] >= x_max
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    def test_linear_aligned_spacing_matches_requested_step(self):
        discretization = 0.25
        x = discretize_aligned_range(
            x_min=-1.12,
            x_max=2.18,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=discretization,
        )

        assert np.isclose(compute_bin_width(x), discretization)
        assert np.allclose(x / discretization, np.round(x / discretization))

    def test_continuous_discretization_uses_requested_linear_step(self):
        result = discretize_continuous_distribution(
            dist=stats.norm(loc=0.0, scale=1.0),
            tail_truncation=1e-3,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
            step=0.1,
            align_to_multiples=True,
        )

        assert np.isclose(compute_bin_width(result.x_array), 0.1)

    def test_continuous_discretization_uses_requested_geometric_step(self):
        step = np.log(1.05)
        result = discretize_continuous_distribution(
            dist=stats.lognorm(s=0.5, scale=1.0),
            tail_truncation=1e-3,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.GEOMETRIC,
            step=step,
            align_to_multiples=True,
        )

        assert result.step == step
        assert np.isclose(np.exp(result.step), 1.05)

    def test_generated_geometric_grid_preserves_pld_default_step(self):
        grid = discretize_aligned_grid(
            x_min=0.1,
            x_max=10.0,
            spacing_type=SpacingType.GEOMETRIC,
            align_to_multiples=True,
            discretization=1e-4,
        )
        x = grid.x_array

        dist = DenseDiscreteDist(
            x_min=grid.x_min,
            step=grid.step,
            prob_arr=np.full(x.size, 1.0 / x.size),
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

        assert dist.step == 1e-4
        assert np.allclose(dist.x_array, x)


class TestComputeBinWidth:
    def test_uniform_grid(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        width = compute_bin_width(x)
        assert np.isclose(width, 1.0)

    def test_nonuniform_grid_raises(self):
        x = np.array([1.0, 2.0, 3.5, 6.0])
        with pytest.raises(ValueError, match="non-uniform bin widths"):
            compute_bin_width(x)

    def test_single_point_raises(self):
        x = np.array([1.0])
        with pytest.raises(ValueError, match="less than 2 bins"):
            compute_bin_width(x)


class TestComputeBinLogRatio:
    def test_geometric_grid(self):
        x = np.array([1.0, 2.0, 4.0, 8.0])
        step = compute_bin_log_ratio(x)
        assert np.isclose(step, np.log(2.0))

    def test_nonuniform_grid_raises(self):
        x = np.array([1.0, 3.0, 6.0, 18.0])
        with pytest.raises(ValueError, match="non-uniform bin widths"):
            compute_bin_log_ratio(x)

    def test_single_point_raises(self):
        x = np.array([1.0])
        with pytest.raises(ValueError, match="less than 2 bins"):
            compute_bin_log_ratio(x)


class TestComputeDiscretePMF:
    def test_uniform_distribution(self):
        dist = stats.uniform(loc=0.0, scale=1.0)
        x_array = np.linspace(0.0, 1.0, 11)
        bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, PMF_min_increment=0.0
        )

        assert len(bin_prob) == 10
        assert np.all(bin_prob >= 0)
        assert np.allclose(bin_prob, 0.1, atol=0.01)
        assert p_left < 0.01
        assert p_right < 0.01

    def test_normal_distribution(self):
        dist = stats.norm(loc=0.0, scale=1.0)
        x_array = np.linspace(-3.0, 3.0, 1001)
        bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, PMF_min_increment=0.0
        )

        total = math.fsum([*map(float, bin_prob), p_left, p_right])
        assert np.isclose(total, 1.0, atol=1e-10)

    def test_exponential_distribution(self):
        dist = stats.expon(scale=1.0)
        x_array = np.linspace(0.0, 5.0, 51)
        _bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, PMF_min_increment=0.0
        )

        assert p_left < 0.01
        assert p_right > 0.0


class TestPMFRemapToGrid:
    def test_exact_alignment(self):
        x_in = np.array([1.0, 2.0, 3.0])
        pmf_in = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        x_out = x_in.copy()

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        assert np.allclose(pmf_out, pmf_in)

    def test_dominates_rounding(self):
        x_in = np.array([1.0, 2.5, 4.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0, 4.0])

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        assert pmf_out[2] >= 0.4

    def test_is_dominated_rounding(self):
        x_in = np.array([1.0, 2.5, 4.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0, 4.0])

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=False)
        assert pmf_out[1] >= 0.4

    def test_overflow_to_infinity(self):
        x_in = np.array([1.0, 2.0, 5.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0])

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        _, _, ppos = enforce_mass_conservation(
            prob_arr=pmf_out, expected_p_min=0.0, expected_p_max=0.0, bound_type=BoundType.DOMINATES
        )
        assert ppos >= 0.3

    def test_mass_conservation_in_remap(self):
        x_in = np.array([0.5, 1.5, 2.5, 3.5])
        pmf_in = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0])

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        total_in = math.fsum(map(float, pmf_in))
        pmf_out, pneg, ppos = enforce_mass_conservation(
            prob_arr=pmf_out, expected_p_min=0.0, expected_p_max=0.0, bound_type=BoundType.DOMINATES
        )
        total_out = math.fsum([*map(float, pmf_out), pneg, ppos])
        assert np.isclose(total_in, total_out, atol=1e-10)


class TestEnforceMassConservation:
    def test_dominates_can_consume_soft_p_min(self):
        prob_arr = np.array([0.4, 0.1], dtype=np.float64)
        prob_out, p_min, p_max = enforce_mass_conservation(
            prob_arr=prob_arr,
            expected_p_min=0.3,
            expected_p_max=0.4,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(prob_out, np.array([0.4, 0.1]))
        assert np.isclose(p_min, 0.1)
        assert np.isclose(p_max, 0.4)
        assert np.isclose(math.fsum([*map(float, prob_out), p_min, p_max]), 1.0)

    def test_is_dominated_can_consume_soft_p_max(self):
        prob_arr = np.array([0.1, 0.4], dtype=np.float64)
        prob_out, p_min, p_max = enforce_mass_conservation(
            prob_arr=prob_arr,
            expected_p_min=0.4,
            expected_p_max=0.3,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert np.allclose(prob_out, np.array([0.1, 0.4]))
        assert np.isclose(p_min, 0.4)
        assert np.isclose(p_max, 0.1)
        assert np.isclose(math.fsum([*map(float, prob_out), p_min, p_max]), 1.0)


class TestComputeTruncation:
    def test_strips_zero_edges_before_tail_truncation(self):
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.0, 0.8], dtype=np.float64),
            p_min=0.0,
            p_max=0.2,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_prob_arr, np.array([0.8], dtype=np.float64))
        assert np.isclose(new_p_min, 0.0)
        assert np.isclose(new_p_max, 0.2)
        assert (min_ind, max_ind) == (1, 1)

    def test_keeps_boundary_when_it_is_the_first_remaining_element(self):
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.0, 0.2, 0.5], dtype=np.float64),
            p_min=0.3,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_prob_arr, np.array([0.2, 0.5], dtype=np.float64))
        assert np.isclose(new_p_min, 0.3)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (1, 2)

    def test_truncation_folds_consumed_boundary_into_first_finite_bin(self):
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.2, 0.75], dtype=np.float64),
            p_min=0.05,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_prob_arr, np.array([0.25, 0.75], dtype=np.float64))
        assert np.isclose(new_p_min, 0.0)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (0, 1)

    def test_strips_zero_edges_for_is_dominated_right_tail(self):
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.8, 0.0], dtype=np.float64),
            p_min=0.2,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert np.allclose(new_prob_arr, np.array([0.8], dtype=np.float64))
        assert np.isclose(new_p_min, 0.2)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (0, 0)

    def test_dense_truncate_edges_updates_x_min_after_zero_edge_removal(self):
        dist = DenseDiscreteDist(
            x_min=0.0,
            step=1.0,
            prob_arr=np.array([0.0, 0.8], dtype=np.float64),
            p_max=0.2,
        )

        result = dist.truncate_edges(0.1, BoundType.DOMINATES)

        assert np.allclose(result.x_array, np.array([1.0], dtype=np.float64))
        assert np.allclose(result.prob_arr, np.array([0.8], dtype=np.float64))
        assert np.isclose(result.p_min, 0.0)
        assert np.isclose(result.p_max, 0.2)

    def test_dense_truncate_edges_removes_right_tail_under_truncation_budget(self):
        dist = DenseDiscreteDist(
            x_min=1.0,
            step=1.0,
            prob_arr=np.array([0.8, 0.1, 0.1], dtype=np.float64),
        )

        result = dist.truncate_edges(0.15, BoundType.DOMINATES)

        assert np.allclose(result.x_array, np.array([1.0, 2.0], dtype=np.float64))
        assert np.allclose(result.prob_arr, np.array([0.8, 0.1], dtype=np.float64))
        assert np.isclose(result.p_min, 0.0)
        assert np.isclose(result.p_max, 0.1)


class TestZeroMass:
    def test_raises_when_mass_is_at_least_total(self):
        with pytest.raises(ValueError, match="mass must be smaller than total array mass"):
            _zero_mass(
                values=np.array([0.2, 0.8], dtype=np.float64),
                mass=1.0,
                from_left=True,
                exact=True,
            )


class TestRediscretizeBoundaryFolding:
    def test_rediscretize_near_point_mass_distribution(self):
        dist = DenseDiscreteDist(
            x_min=0.5,
            step=0.5,
            prob_arr=np.array([1.0 - 1e-6, 1e-6], dtype=np.float64),
        )

        result = rediscretize_dist(
            dist=dist,
            tail_truncation=1e-8,
            loss_discretization=1e-2,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.DOMINATES,
        )

        assert np.isclose(result.step, 1e-2)
        assert np.isclose(result.x_array[0], 0.5)
        assert np.isclose(result.x_array[-1], 1.0)
        total = math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max])
        assert np.isclose(total, 1.0)

    def test_is_dominated_moves_p_max_into_last_finite_cell(self):
        dist = DenseDiscreteDist.from_x_array(
            x_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_max=0.1,
        )

        result = rediscretize_dist(
            dist=dist,
            tail_truncation=0.0,
            loss_discretization=1.0,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert np.isclose(result.p_max, 0.0)
        assert np.isclose(result.prob_arr[-1], 0.5)
        assert np.isclose(
            math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max]), 1.0
        )

    def test_dominates_linear_moves_p_min_into_first_finite_cell(self):
        dist = DenseDiscreteDist.from_x_array(
            x_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_min=0.1,
        )

        result = rediscretize_dist(
            dist=dist,
            tail_truncation=0.0,
            loss_discretization=1.0,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.DOMINATES,
        )

        assert np.isclose(result.p_min, 0.0)
        assert np.isclose(result.prob_arr[0], 0.3)
        assert np.isclose(
            math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max]), 1.0
        )

    def test_dominates_geometric_keeps_zero_atom(self):
        dist = DenseDiscreteDist.from_x_array(
            x_array=np.array([1.0, 2.0, 4.0], dtype=np.float64),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_min=0.1,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

        result = rediscretize_dist(
            dist=dist,
            tail_truncation=0.0,
            loss_discretization=np.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            bound_type=BoundType.DOMINATES,
        )

        assert np.isclose(result.p_min, 0.1)
        assert np.isclose(
            math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max]), 1.0
        )
