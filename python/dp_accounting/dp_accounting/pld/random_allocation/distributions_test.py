"""Core unit tests for random-allocation internals.

Includes distribution discretization coverage and external interface
conversion checks.
"""

from __future__ import annotations

import math

import numpy as np
from absl.testing import absltest
from scipy import stats

from dp_accounting.pld.random_allocation import core
from dp_accounting.pld.random_allocation import distributions
from dp_accounting.pld.random_allocation import definitions
from dp_accounting.pld.random_allocation import utils


def _make_realization() -> distributions.PLDRealization:
  return distributions.PLDRealization(
      x_min=-0.5,
      step=0.5,
      prob_arr=np.array([0.2, 0.3, 0.25, 0.15], dtype=np.float64),
      p_max=0.1,
  )


class LinearDistToDpAccountingTest(absltest.TestCase):

  def test_preserves_shape_and_infinity_mass(self):
    original = _make_realization()
    pmf = core._linear_dist_to_dp_accounting_pmf(
        dist=original, pessimistic_estimate=True
    )

    self.assertEqual(pmf._probs.shape, original.prob_arr.shape)
    self.assertEqual(pmf._infinity_mass, original.p_max)
    self.assertEqual(pmf._discretization, original.step)

  def test_handles_zero_finite_mass(self):
    realization = distributions.DenseDiscreteDist(
        x_min=0.0,
        step=1.0,
        prob_arr=np.array([0.0, 0.0], dtype=np.float64),
        p_max=1.0,
    )
    pmf = core._linear_dist_to_dp_accounting_pmf(
        dist=realization, pessimistic_estimate=True
    )
    self.assertEqual(pmf._infinity_mass, 1.0)
    np.testing.assert_allclose(pmf._probs, np.array([0.0, 0.0]))


class DenseDiscreteDistTest(absltest.TestCase):

  def test_exp_log_round_trip_preserves_linear_step(self):
    original = distributions.DenseDiscreteDist(
        x_min=0.0,
        step=1e-4,
        prob_arr=np.array([0.4, 0.6], dtype=np.float64),
    )

    round_tripped = utils.log_geometric_to_linear(
        utils.exp_linear_to_geometric(original)
    )

    self.assertEqual(round_tripped.step, original.step)

  def test_keeps_regular_grid_as_source_of_truth(self):
    grid = definitions.RegularGrid(
        x_min=1.0,
        step=1e-4,
        size=2,
        spacing_type=definitions.SpacingType.GEOMETRIC,
    )
    dist = distributions.DenseDiscreteDist.from_grid(
        grid=grid,
        prob_arr=np.array([0.4, 0.6], dtype=np.float64),
        domain=distributions.Domain.POSITIVES,
    )

    self.assertEqual(dist.grid, grid)
    self.assertEqual(dist.step, 1e-4)


class DiscretizeRangeTest(absltest.TestCase):
  """Tests for _discretize_aligned_range."""

  def test_linear_spacing(self):
    n_grid = 100
    x = distributions._discretize_aligned_range(
        x_min=0.0,
        x_max=10.0,
        spacing_type=definitions.SpacingType.LINEAR,
        align_to_multiples=True,
        discretization=(10.0 - 0.0) / (n_grid - 1),
    )
    self.assertGreaterEqual(len(x), n_grid)
    self.assertLessEqual(x[0], 0.0)
    self.assertGreaterEqual(x[-1], 10.0)
    diffs = np.diff(x)
    np.testing.assert_allclose(diffs, diffs[0])

  def test_geometric_spacing(self):
    n_grid = 100
    x = distributions._discretize_aligned_range(
        x_min=1.0,
        x_max=100.0,
        spacing_type=definitions.SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=np.log(100.0 / 1.0) / (n_grid - 1),
    )
    self.assertGreaterEqual(len(x), n_grid)
    self.assertLessEqual(x[0], 1.0)
    self.assertGreaterEqual(x[-1], 100.0)
    ratios = x[1:] / x[:-1]
    np.testing.assert_allclose(ratios, ratios[0])

  def test_nonpositive_discretization_rejected(self):
    with self.assertRaisesRegex(ValueError, "discretization must be positive"):
      distributions._discretize_aligned_range(
          x_min=0.0,
          x_max=10.0,
          spacing_type=definitions.SpacingType.LINEAR,
          align_to_multiples=True,
          discretization=0.0,
      )

  def test_two_points_linear(self):
    n_grid = 100
    x = distributions._discretize_aligned_range(
        x_min=1.0,
        x_max=3.0,
        spacing_type=definitions.SpacingType.LINEAR,
        align_to_multiples=True,
        discretization=(3.0 - 1.0) / (n_grid - 1),
    )
    self.assertGreaterEqual(len(x), n_grid)
    self.assertLessEqual(x[0], 1.0)
    self.assertGreaterEqual(x[-1], 3.0)
    diffs = np.diff(x)
    np.testing.assert_allclose(diffs, diffs[0])

  def test_linear_spacing_covers_endpoint_after_alignment_rounding(self):
    x_min = -1.411426541779732
    x_max = 1.4160541697856062
    discretization = 0.041648652052517825

    x = distributions._discretize_aligned_range(
        x_min=x_min,
        x_max=x_max,
        spacing_type=definitions.SpacingType.LINEAR,
        align_to_multiples=True,
        discretization=discretization,
    )

    self.assertLessEqual(x[0], x_min)
    self.assertGreaterEqual(x[-1], x_max)
    diffs = np.diff(x)
    np.testing.assert_allclose(diffs, diffs[0])

  def test_linear_aligned_spacing_matches_requested_step(self):
    discretization = 0.25
    x = distributions._discretize_aligned_range(
        x_min=-1.12,
        x_max=2.18,
        spacing_type=definitions.SpacingType.LINEAR,
        align_to_multiples=True,
        discretization=discretization,
    )

    np.testing.assert_allclose(
        distributions._compute_bin_width(x), discretization
    )
    np.testing.assert_allclose(x / discretization, np.round(x / discretization))

  def test_continuous_discretization_uses_requested_linear_step(self):
    result = distributions._discretize_continuous_distribution(
        dist=stats.norm(loc=0.0, scale=1.0),
        tail_truncation=1e-3,
        bound_type=definitions.BoundType.DOMINATES,
        spacing_type=definitions.SpacingType.LINEAR,
        step=0.1,
        align_to_multiples=True,
    )

    np.testing.assert_allclose(
        distributions._compute_bin_width(result.x_array), 0.1
    )

  def test_continuous_discretization_uses_requested_geometric_step(self):
    step = np.log(1.05)
    result = distributions._discretize_continuous_distribution(
        dist=stats.lognorm(s=0.5, scale=1.0),
        tail_truncation=1e-3,
        bound_type=definitions.BoundType.DOMINATES,
        spacing_type=definitions.SpacingType.GEOMETRIC,
        step=step,
        align_to_multiples=True,
    )

    self.assertEqual(result.step, step)
    np.testing.assert_allclose(np.exp(result.step), 1.05)

  def test_generated_geometric_grid_preserves_pld_default_step(self):
    grid = distributions.discretize_aligned_grid(
        x_min=0.1,
        x_max=10.0,
        spacing_type=definitions.SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=1e-4,
    )
    x = grid.x_array

    dist = distributions.DenseDiscreteDist(
        x_min=grid.x_min,
        step=grid.step,
        prob_arr=np.full(x.size, 1.0 / x.size),
        spacing_type=definitions.SpacingType.GEOMETRIC,
        domain=distributions.Domain.POSITIVES,
    )

    self.assertEqual(dist.step, 1e-4)
    np.testing.assert_allclose(dist.x_array, x)


class ComputeBinWidthTest(absltest.TestCase):

  def test_uniform_grid(self):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    width = distributions._compute_bin_width(x)
    np.testing.assert_allclose(width, 1.0)

  def test_nonuniform_grid_raises(self):
    x = np.array([1.0, 2.0, 3.5, 6.0])
    with self.assertRaisesRegex(ValueError, "non-uniform bin widths"):
      distributions._compute_bin_width(x)

  def test_single_point_raises(self):
    x = np.array([1.0])
    with self.assertRaisesRegex(ValueError, "less than 2 bins"):
      distributions._compute_bin_width(x)


class ComputeBinLogRatioTest(absltest.TestCase):

  def test_geometric_grid(self):
    x = np.array([1.0, 2.0, 4.0, 8.0])
    step = distributions._compute_bin_log_ratio(x)
    np.testing.assert_allclose(step, np.log(2.0))

  def test_nonuniform_grid_raises(self):
    x = np.array([1.0, 3.0, 6.0, 18.0])
    with self.assertRaisesRegex(ValueError, "non-uniform bin widths"):
      distributions._compute_bin_log_ratio(x)

  def test_single_point_raises(self):
    x = np.array([1.0])
    with self.assertRaisesRegex(ValueError, "less than 2 bins"):
      distributions._compute_bin_log_ratio(x)


class ComputeDiscretePmfTest(absltest.TestCase):

  def test_uniform_distribution(self):
    dist = stats.uniform(loc=0.0, scale=1.0)
    x_array = np.linspace(0.0, 1.0, 11)
    bin_prob, p_left, p_right = distributions._compute_discrete_prob(
        dist=dist,
        x_array=x_array,
        bound_type=definitions.BoundType.DOMINATES,
        pmf_min_increment=0.0,
    )

    self.assertEqual(len(bin_prob), 10)
    self.assertTrue(np.all(bin_prob >= 0))
    np.testing.assert_allclose(bin_prob, 0.1, atol=0.01, rtol=0)
    self.assertLess(p_left, 0.01)
    self.assertLess(p_right, 0.01)

  def test_normal_distribution(self):
    dist = stats.norm(loc=0.0, scale=1.0)
    x_array = np.linspace(-3.0, 3.0, 1001)
    bin_prob, p_left, p_right = distributions._compute_discrete_prob(
        dist=dist,
        x_array=x_array,
        bound_type=definitions.BoundType.DOMINATES,
        pmf_min_increment=0.0,
    )

    total = math.fsum([*map(float, bin_prob), p_left, p_right])
    np.testing.assert_allclose(total, 1.0, atol=1e-10, rtol=0)

  def test_exponential_distribution(self):
    dist = stats.expon(scale=1.0)
    x_array = np.linspace(0.0, 5.0, 51)
    _bin_prob, p_left, p_right = distributions._compute_discrete_prob(
        dist=dist,
        x_array=x_array,
        bound_type=definitions.BoundType.DOMINATES,
        pmf_min_increment=0.0,
    )

    self.assertLess(p_left, 0.01)
    self.assertGreater(p_right, 0.0)


class PmfRemapToGridTest(absltest.TestCase):

  def test_exact_alignment(self):
    x_in = np.array([1.0, 2.0, 3.0])
    pmf_in = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    x_out = x_in.copy()

    pmf_out = distributions._rediscretize_prob(
        x_in, pmf_in, x_out, dominates=True
    )
    np.testing.assert_allclose(pmf_out, pmf_in)

  def test_dominates_rounding(self):
    x_in = np.array([1.0, 2.5, 4.0])
    pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
    x_out = np.array([1.0, 2.0, 3.0, 4.0])

    pmf_out = distributions._rediscretize_prob(
        x_in, pmf_in, x_out, dominates=True
    )
    self.assertGreaterEqual(pmf_out[2], 0.4)

  def test_is_dominated_rounding(self):
    x_in = np.array([1.0, 2.5, 4.0])
    pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
    x_out = np.array([1.0, 2.0, 3.0, 4.0])

    pmf_out = distributions._rediscretize_prob(
        x_in, pmf_in, x_out, dominates=False
    )
    self.assertGreaterEqual(pmf_out[1], 0.4)

  def test_overflow_to_infinity(self):
    x_in = np.array([1.0, 2.0, 5.0])
    pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
    x_out = np.array([1.0, 2.0, 3.0])

    pmf_out = distributions._rediscretize_prob(
        x_in, pmf_in, x_out, dominates=True
    )
    _, _, ppos = distributions.enforce_mass_conservation(
        prob_arr=pmf_out,
        expected_p_min=0.0,
        expected_p_max=0.0,
        bound_type=definitions.BoundType.DOMINATES,
    )
    self.assertGreaterEqual(ppos, 0.3)

  def test_mass_conservation_in_remap(self):
    x_in = np.array([0.5, 1.5, 2.5, 3.5])
    pmf_in = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float64)
    x_out = np.array([1.0, 2.0, 3.0])

    pmf_out = distributions._rediscretize_prob(
        x_in, pmf_in, x_out, dominates=True
    )
    total_in = math.fsum(map(float, pmf_in))
    pmf_out, pneg, ppos = distributions.enforce_mass_conservation(
        prob_arr=pmf_out,
        expected_p_min=0.0,
        expected_p_max=0.0,
        bound_type=definitions.BoundType.DOMINATES,
    )
    total_out = math.fsum([*map(float, pmf_out), pneg, ppos])
    np.testing.assert_allclose(total_in, total_out, atol=1e-10, rtol=0)


class EnforceMassConservationTest(absltest.TestCase):

  def test_dominates_can_consume_soft_p_min(self):
    prob_arr = np.array([0.4, 0.1], dtype=np.float64)
    prob_out, p_min, p_max = distributions.enforce_mass_conservation(
        prob_arr=prob_arr,
        expected_p_min=0.3,
        expected_p_max=0.4,
        bound_type=definitions.BoundType.DOMINATES,
    )

    np.testing.assert_allclose(prob_out, np.array([0.4, 0.1]))
    np.testing.assert_allclose(p_min, 0.1)
    np.testing.assert_allclose(p_max, 0.4)
    total = math.fsum([*map(float, prob_out), p_min, p_max])
    np.testing.assert_allclose(total, 1.0)

  def test_is_dominated_can_consume_soft_p_max(self):
    prob_arr = np.array([0.1, 0.4], dtype=np.float64)
    prob_out, p_min, p_max = distributions.enforce_mass_conservation(
        prob_arr=prob_arr,
        expected_p_min=0.4,
        expected_p_max=0.3,
        bound_type=definitions.BoundType.IS_DOMINATED,
    )

    np.testing.assert_allclose(prob_out, np.array([0.1, 0.4]))
    np.testing.assert_allclose(p_min, 0.4)
    np.testing.assert_allclose(p_max, 0.1)
    total = math.fsum([*map(float, prob_out), p_min, p_max])
    np.testing.assert_allclose(total, 1.0)


class ComputeTruncationTest(absltest.TestCase):

  def test_strips_zero_edges_before_tail_truncation(self):
    new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = (
        distributions._compute_truncation(
            prob_arr=np.array([0.0, 0.8], dtype=np.float64),
            p_min=0.0,
            p_max=0.2,
            tail_truncation=0.1,
            bound_type=definitions.BoundType.DOMINATES,
        )
    )

    np.testing.assert_allclose(new_prob_arr, np.array([0.8], dtype=np.float64))
    np.testing.assert_allclose(new_p_min, 0.0)
    np.testing.assert_allclose(new_p_max, 0.2)
    self.assertEqual((min_ind, max_ind), (1, 1))

  def test_keeps_boundary_when_it_is_the_first_remaining_element(self):
    new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = (
        distributions._compute_truncation(
            prob_arr=np.array([0.0, 0.2, 0.5], dtype=np.float64),
            p_min=0.3,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=definitions.BoundType.DOMINATES,
        )
    )

    np.testing.assert_allclose(
        new_prob_arr, np.array([0.2, 0.5], dtype=np.float64)
    )
    np.testing.assert_allclose(new_p_min, 0.3)
    np.testing.assert_allclose(new_p_max, 0.0)
    self.assertEqual((min_ind, max_ind), (1, 2))

  def test_truncation_folds_consumed_boundary_into_first_finite_bin(self):
    new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = (
        distributions._compute_truncation(
            prob_arr=np.array([0.2, 0.75], dtype=np.float64),
            p_min=0.05,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=definitions.BoundType.DOMINATES,
        )
    )

    np.testing.assert_allclose(
        new_prob_arr, np.array([0.25, 0.75], dtype=np.float64)
    )
    np.testing.assert_allclose(new_p_min, 0.0)
    np.testing.assert_allclose(new_p_max, 0.0)
    self.assertEqual((min_ind, max_ind), (0, 1))

  def test_strips_zero_edges_for_is_dominated_right_tail(self):
    new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = (
        distributions._compute_truncation(
            prob_arr=np.array([0.8, 0.0], dtype=np.float64),
            p_min=0.2,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=definitions.BoundType.IS_DOMINATED,
        )
    )

    np.testing.assert_allclose(new_prob_arr, np.array([0.8], dtype=np.float64))
    np.testing.assert_allclose(new_p_min, 0.2)
    np.testing.assert_allclose(new_p_max, 0.0)
    self.assertEqual((min_ind, max_ind), (0, 0))

  def test_dense_truncate_edges_updates_x_min_after_zero_edge_removal(self):
    dist = distributions.DenseDiscreteDist(
        x_min=0.0,
        step=1.0,
        prob_arr=np.array([0.0, 0.8], dtype=np.float64),
        p_max=0.2,
    )

    result = dist.truncate_edges(0.1, definitions.BoundType.DOMINATES)

    np.testing.assert_allclose(
        result.x_array, np.array([1.0], dtype=np.float64)
    )
    np.testing.assert_allclose(
        result.prob_arr, np.array([0.8], dtype=np.float64)
    )
    np.testing.assert_allclose(result.p_min, 0.0)
    np.testing.assert_allclose(result.p_max, 0.2)

  def test_dense_truncate_edges_removes_right_tail_under_truncation_budget(
      self,
  ):
    dist = distributions.DenseDiscreteDist(
        x_min=1.0,
        step=1.0,
        prob_arr=np.array([0.8, 0.1, 0.1], dtype=np.float64),
    )

    result = dist.truncate_edges(0.15, definitions.BoundType.DOMINATES)

    np.testing.assert_allclose(
        result.x_array, np.array([1.0, 2.0], dtype=np.float64)
    )
    np.testing.assert_allclose(
        result.prob_arr, np.array([0.8, 0.1], dtype=np.float64)
    )
    np.testing.assert_allclose(result.p_min, 0.0)
    np.testing.assert_allclose(result.p_max, 0.1)


class ZeroMassTest(absltest.TestCase):

  def test_raises_when_mass_is_at_least_total(self):
    with self.assertRaisesRegex(
        ValueError, "mass must be smaller than total array mass"
    ):
      distributions._zero_mass(
          values=np.array([0.2, 0.8], dtype=np.float64),
          mass=1.0,
          from_left=True,
          exact=True,
      )


class RediscretizeBoundaryFoldingTest(absltest.TestCase):

  def test_rediscretize_near_point_mass_distribution(self):
    dist = distributions.DenseDiscreteDist(
        x_min=0.5,
        step=0.5,
        prob_arr=np.array([1.0 - 1e-6, 1e-6], dtype=np.float64),
    )

    result = distributions.rediscretize_dist(
        dist=dist,
        tail_truncation=1e-8,
        loss_discretization=1e-2,
        spacing_type=definitions.SpacingType.LINEAR,
        bound_type=definitions.BoundType.DOMINATES,
    )

    np.testing.assert_allclose(result.step, 1e-2)
    np.testing.assert_allclose(result.x_array[0], 0.5)
    np.testing.assert_allclose(result.x_array[-1], 1.0)
    total = math.fsum(
        [*map(float, result.prob_arr), result.p_min, result.p_max]
    )
    np.testing.assert_allclose(total, 1.0)

  def test_is_dominated_moves_p_max_into_last_finite_cell(self):
    dist = distributions.DenseDiscreteDist.from_x_array(
        x_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
        p_max=0.1,
    )

    result = distributions.rediscretize_dist(
        dist=dist,
        tail_truncation=0.0,
        loss_discretization=1.0,
        spacing_type=definitions.SpacingType.LINEAR,
        bound_type=definitions.BoundType.IS_DOMINATED,
    )

    np.testing.assert_allclose(result.p_max, 0.0)
    np.testing.assert_allclose(result.prob_arr[-1], 0.5)
    total = math.fsum(
        [*map(float, result.prob_arr), result.p_min, result.p_max]
    )
    np.testing.assert_allclose(total, 1.0)

  def test_dominates_linear_moves_p_min_into_first_finite_cell(self):
    dist = distributions.DenseDiscreteDist.from_x_array(
        x_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
        p_min=0.1,
    )

    result = distributions.rediscretize_dist(
        dist=dist,
        tail_truncation=0.0,
        loss_discretization=1.0,
        spacing_type=definitions.SpacingType.LINEAR,
        bound_type=definitions.BoundType.DOMINATES,
    )

    np.testing.assert_allclose(result.p_min, 0.0)
    np.testing.assert_allclose(result.prob_arr[0], 0.3)
    total = math.fsum(
        [*map(float, result.prob_arr), result.p_min, result.p_max]
    )
    np.testing.assert_allclose(total, 1.0)

  def test_dominates_geometric_keeps_zero_atom(self):
    dist = distributions.DenseDiscreteDist.from_x_array(
        x_array=np.array([1.0, 2.0, 4.0], dtype=np.float64),
        prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
        p_min=0.1,
        spacing_type=definitions.SpacingType.GEOMETRIC,
        domain=distributions.Domain.POSITIVES,
    )

    result = distributions.rediscretize_dist(
        dist=dist,
        tail_truncation=0.0,
        loss_discretization=np.log(2.0),
        spacing_type=definitions.SpacingType.GEOMETRIC,
        bound_type=definitions.BoundType.DOMINATES,
    )

    np.testing.assert_allclose(result.p_min, 0.1)
    total = math.fsum(
        [*map(float, result.prob_arr), result.p_min, result.p_max]
    )
    np.testing.assert_allclose(total, 1.0)


class RediscretizeProbNumbaTest(absltest.TestCase):

  def _check_numpy_matches_numba(self, dominates: bool) -> None:
    x_in = np.array([0.2, 0.8, 1.0, 1.7, 2.9, 4.2], dtype=np.float64)
    pmf_in = np.array([0.2, 0.0, 0.15, 0.25, 0.1, 0.3], dtype=np.float64)
    x_out = np.array([0.5, 1.0, 1.5, 3.0], dtype=np.float64)

    expected = distributions._numba_rediscretize_prob(
        x_in, pmf_in, x_out, dominates
    )
    actual = distributions._numpy_rediscretize_prob(
        x_in, pmf_in, x_out, dominates
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

  def test_numpy_matches_numba_dominates(self):
    self._check_numpy_matches_numba(True)

  def test_numpy_matches_numba_is_dominated(self):
    self._check_numpy_matches_numba(False)

  def test_dispatch_uses_numpy_fallback_without_numba(self):
    x_in = np.array([0.2, 1.0, 1.7], dtype=np.float64)
    pmf_in = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    x_out = np.array([0.5, 1.0, 1.5], dtype=np.float64)

    original_has_numba = definitions._HAS_NUMBA
    try:
      definitions._HAS_NUMBA = False
      expected = distributions._numpy_rediscretize_prob(
          x_in, pmf_in, x_out, True
      )
      actual = distributions._rediscretize_prob(x_in, pmf_in, x_out, True)
    finally:
      definitions._HAS_NUMBA = original_has_numba

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
  absltest.main()
