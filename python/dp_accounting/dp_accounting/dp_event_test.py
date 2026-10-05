# Copyright 2023, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
import attrs
import tree

from dp_accounting import dp_event


def assert_not_contains_attrs(structure):
  def _fn(structure):
    if attrs.has(type(structure)):
      raise AssertionError(
          'Expected structure to not contain `attrs` decorated classes, '
          f'found {structure}.'
      )
    return None

  tree.traverse(_fn, structure)


def assert_not_contains_named_tuples(structure):
  def _fn(structure):
    if isinstance(structure, dp_event.DpEventNamedTuple):
      raise AssertionError(
          'Expected structure to not contain `dp_event.DpEventNamedTuple`s, '
          f'found {structure}.'
      )
    return None

  tree.traverse(_fn, structure)


class DpEventTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('base_class', dp_event.DpEvent()),
      ('no_op', dp_event.NoOpDpEvent()),
      ('non_private', dp_event.NonPrivateDpEvent()),
      ('unsupported', dp_event.UnsupportedDpEvent()),
      ('epsilon_delta', dp_event.EpsilonDeltaDpEvent(1.0, 0.1)),
      ('gaussian', dp_event.GaussianDpEvent(1.0)),
      ('laplace', dp_event.LaplaceDpEvent(1.0)),
      ('dlaplace', dp_event.DiscreteLaplaceDpEvent(0.5, 1)),
      ('dgaussian', dp_event.DiscreteGaussianDpEvent(sigma=10.0)),
      (
          'dgaussian_dim',
          dp_event.DiscreteGaussianDpEvent(
              sigma=10.0, sensitivity=2.0, dimension=1
          ),
      ),
      (
          'self_composed',
          dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(1.0), 10),
      ),
      (
          'composed',
          dp_event.ComposedDpEvent(
              [dp_event.GaussianDpEvent(1.0), dp_event.LaplaceDpEvent(1.0)]
          ),
      ),
      (
          'poisson',
          dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)),
      ),
      (
          'sampled_with_replacement',
          dp_event.SampledWithReplacementDpEvent(
              1000, 10, dp_event.GaussianDpEvent(1.0)
          ),
      ),
      (
          'sampled_without_replacement',
          dp_event.SampledWithoutReplacementDpEvent(
              1000, 10, dp_event.GaussianDpEvent(1.0)
          ),
      ),
      ('tree_int', dp_event.SingleEpochTreeAggregationDpEvent(1.0, 5)),
      ('tree_list', dp_event.SingleEpochTreeAggregationDpEvent(1.0, [5, 10])),
      (
          'repeat_and_select',
          dp_event.RepeatAndSelectDpEvent(
              dp_event.GaussianDpEvent(1.0), 30.0, 1.0
          ),
      ),
      (
          'complex',
          dp_event.ComposedDpEvent([
              dp_event.SingleEpochTreeAggregationDpEvent(1.0, 5),
              dp_event.PoissonSampledDpEvent(0.1, dp_event.LaplaceDpEvent(1.0)),
              dp_event.SelfComposedDpEvent(
                  dp_event.SampledWithReplacementDpEvent(
                      1000, 10, dp_event.GaussianDpEvent(1.0)
                  ),
                  50,
              ),
          ]),
      ),
      (
          'mixture_gaussian',
          dp_event.MixtureOfGaussiansDpEvent(1.0, [0, 1, 2], [0.25, 0.5, 0.25]),
      ),
      (
          'exponential_mechanism',
          dp_event.ExponentialMechanismDpEvent(1.0),
      ),
      (
          'permute_and_flip',
          dp_event.PermuteAndFlipDpEvent(1.0),
      ),
      (
          'zcdp',
          dp_event.ZCDpEvent(1.0, 2.0),
      ),
      (
          'truncated_subsampled_gaussian',
          dp_event.TruncatedSubsampledGaussianDpEvent(
              10000,
              0.1,
              1000,
              1.0,
          ),
      ),
      (
          'random_allocation',
          dp_event.RandomAllocationDpEvent(
              event=dp_event.GaussianDpEvent(2.0),
              num_selected=10,
              num_steps=100,
          ),
      ),
  )
  def test_to_from_named_tuple(self, event):
    named_tuple = event.to_named_tuple()
    self.assertIsInstance(named_tuple, tuple)
    self.assertIsInstance(named_tuple, dp_event.DpEventNamedTuple)
    assert_not_contains_attrs(named_tuple)

    reconstructed = dp_event.DpEvent.from_named_tuple(named_tuple)
    assert_not_contains_named_tuples(reconstructed)
    self.assertEqual(event, reconstructed)

  @parameterized.named_parameters(
      ('rr_zero_noise', dp_event.RandomizedResponseDpEvent, (0.0, 1)),
      ('rr_one_noise', dp_event.RandomizedResponseDpEvent, (1.0, 1)),
      ('eps_delta_zeros', dp_event.EpsilonDeltaDpEvent, (0.0, 0.0)),
      ('eps_delta_ones', dp_event.EpsilonDeltaDpEvent, (1.0, 1.0)),
      ('gaussian_zero', dp_event.GaussianDpEvent, (0.0,)),
      ('laplace_zero', dp_event.LaplaceDpEvent, (0.0,)),
      ('dlaplace_zeros', dp_event.DiscreteLaplaceDpEvent, (0.0, 0)),
      ('dlaplace_zero_noise', dp_event.DiscreteLaplaceDpEvent, (0.0, 1)),
      ('dlaplace_zero_sens', dp_event.DiscreteLaplaceDpEvent, (1.0, 0)),
      ('dgaussian_zeros', dp_event.DiscreteGaussianDpEvent, (0.0, 0.0)),
      ('dgaussian_zero_sigma', dp_event.DiscreteGaussianDpEvent, (0.0, 1.0)),
      ('dgaussian_zero_sens', dp_event.DiscreteGaussianDpEvent, (1.0, 0.0)),
      (
          'self_composed_one',
          dp_event.SelfComposedDpEvent,
          (dp_event.NoOpDpEvent(), 1),
      ),
      (
          'poisson_zero',
          dp_event.PoissonSampledDpEvent,
          (0.0, dp_event.NoOpDpEvent()),
      ),
      (
          'poisson_one',
          dp_event.PoissonSampledDpEvent,
          (1.0, dp_event.NoOpDpEvent()),
      ),
      (
          'swr_zero_sample',
          dp_event.SampledWithReplacementDpEvent,
          (1, 0, dp_event.NoOpDpEvent()),
      ),
      (
          'swr_large_sample',
          dp_event.SampledWithReplacementDpEvent,
          (1, 5, dp_event.NoOpDpEvent()),
      ),
      (
          'swor_zero_sample',
          dp_event.SampledWithoutReplacementDpEvent,
          (1, 0, dp_event.NoOpDpEvent()),
      ),
      (
          'swor_full_sample',
          dp_event.SampledWithoutReplacementDpEvent,
          (1, 1, dp_event.NoOpDpEvent()),
      ),
      (
          'tree_zeros_scalar',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (0.0, 0),
      ),
      (
          'tree_zeros_list',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (0.0, [0]),
      ),
      (
          'repeat_and_select_boundary',
          dp_event.RepeatAndSelectDpEvent,
          (dp_event.NoOpDpEvent(), 1.0, 0.0),
      ),
      (
          'repeat_and_select_inf_shape',
          dp_event.RepeatAndSelectDpEvent,
          (dp_event.NoOpDpEvent(), 1.0, float('inf')),
      ),
      (
          'mog_zeros',
          dp_event.MixtureOfGaussiansDpEvent,
          (0.0, [0.0], [1.0]),
      ),
      ('zcdp_zeros', dp_event.ZCDpEvent, (0.0, 0.0)),
      ('exp_mech_zero', dp_event.ExponentialMechanismDpEvent, (0.0,)),
      ('permute_and_flip_zero', dp_event.PermuteAndFlipDpEvent, (0.0,)),
      (
          'truncated_subsampled_gaussian_zeros',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (0, 0.0, 0, 0.0),
      ),
      (
          'random_allocation_zero_selected',
          dp_event.RandomAllocationDpEvent,
          (dp_event.NoOpDpEvent(), 0, 1),
      ),
      (
          'random_allocation_all_selected',
          dp_event.RandomAllocationDpEvent,
          (dp_event.NoOpDpEvent(), 1, 1),
      ),
  )
  def test_valid_boundary_parameters(self, event_cls, args):
    event = event_cls(*args)
    self.assertIsInstance(event, event_cls)

  @parameterized.named_parameters(
      ('rr_neg_noise', dp_event.RandomizedResponseDpEvent, (-0.1, 2)),
      ('rr_large_noise', dp_event.RandomizedResponseDpEvent, (1.1, 2)),
      ('rr_nan_noise', dp_event.RandomizedResponseDpEvent, (float('nan'), 2)),
      ('rr_zero_buckets', dp_event.RandomizedResponseDpEvent, (0.5, 0)),
      ('rr_neg_buckets', dp_event.RandomizedResponseDpEvent, (0.5, -1)),
      (
          'rr_nan_buckets',
          dp_event.RandomizedResponseDpEvent,
          (0.5, float('nan')),
      ),
      ('eps_delta_neg_eps', dp_event.EpsilonDeltaDpEvent, (-0.1, 0.1)),
      ('eps_delta_nan_eps', dp_event.EpsilonDeltaDpEvent, (float('nan'), 0.1)),
      ('eps_delta_neg_delta', dp_event.EpsilonDeltaDpEvent, (1.0, -0.1)),
      ('eps_delta_large_delta', dp_event.EpsilonDeltaDpEvent, (1.0, 1.1)),
      (
          'eps_delta_nan_delta',
          dp_event.EpsilonDeltaDpEvent,
          (1.0, float('nan')),
      ),
      ('gaussian_neg_noise', dp_event.GaussianDpEvent, (-0.1,)),
      ('gaussian_nan_noise', dp_event.GaussianDpEvent, (float('nan'),)),
      ('laplace_neg_noise', dp_event.LaplaceDpEvent, (-0.1,)),
      ('laplace_nan_noise', dp_event.LaplaceDpEvent, (float('nan'),)),
      ('dlaplace_neg_noise', dp_event.DiscreteLaplaceDpEvent, (-0.1, 1)),
      (
          'dlaplace_nan_noise',
          dp_event.DiscreteLaplaceDpEvent,
          (float('nan'), 1),
      ),
      ('dlaplace_neg_sens', dp_event.DiscreteLaplaceDpEvent, (0.5, -1)),
      (
          'dlaplace_nan_sens',
          dp_event.DiscreteLaplaceDpEvent,
          (0.5, float('nan')),
      ),
      ('dgaussian_neg_sigma', dp_event.DiscreteGaussianDpEvent, (-0.1, 1.0)),
      (
          'dgaussian_nan_sigma',
          dp_event.DiscreteGaussianDpEvent,
          (float('nan'), 1.0),
      ),
      ('dgaussian_neg_sens', dp_event.DiscreteGaussianDpEvent, (1.0, -0.1)),
      (
          'dgaussian_nan_sens',
          dp_event.DiscreteGaussianDpEvent,
          (1.0, float('nan')),
      ),
      ('dgaussian_zero_dim', dp_event.DiscreteGaussianDpEvent, (1.0, 1.0, 0)),
      ('dgaussian_neg_dim', dp_event.DiscreteGaussianDpEvent, (1.0, 1.0, -1)),
      (
          'dgaussian_nan_dim',
          dp_event.DiscreteGaussianDpEvent,
          (1.0, 1.0, float('nan')),
      ),
      (
          'self_composed_zero_count',
          dp_event.SelfComposedDpEvent,
          (dp_event.NoOpDpEvent(), 0),
      ),
      (
          'self_composed_neg_count',
          dp_event.SelfComposedDpEvent,
          (dp_event.NoOpDpEvent(), -1),
      ),
      (
          'self_composed_nan_count',
          dp_event.SelfComposedDpEvent,
          (dp_event.NoOpDpEvent(), float('nan')),
      ),
      (
          'poisson_neg_prob',
          dp_event.PoissonSampledDpEvent,
          (-0.1, dp_event.NoOpDpEvent()),
      ),
      (
          'poisson_large_prob',
          dp_event.PoissonSampledDpEvent,
          (1.1, dp_event.NoOpDpEvent()),
      ),
      (
          'poisson_nan_prob',
          dp_event.PoissonSampledDpEvent,
          (float('nan'), dp_event.NoOpDpEvent()),
      ),
      (
          'swr_zero_dataset',
          dp_event.SampledWithReplacementDpEvent,
          (0, 1, dp_event.NoOpDpEvent()),
      ),
      (
          'swr_neg_dataset',
          dp_event.SampledWithReplacementDpEvent,
          (-1, 1, dp_event.NoOpDpEvent()),
      ),
      (
          'swr_nan_dataset',
          dp_event.SampledWithReplacementDpEvent,
          (float('nan'), 1, dp_event.NoOpDpEvent()),
      ),
      (
          'swr_neg_sample',
          dp_event.SampledWithReplacementDpEvent,
          (10, -1, dp_event.NoOpDpEvent()),
      ),
      (
          'swr_nan_sample',
          dp_event.SampledWithReplacementDpEvent,
          (10, float('nan'), dp_event.NoOpDpEvent()),
      ),
      (
          'swor_zero_dataset',
          dp_event.SampledWithoutReplacementDpEvent,
          (0, 0, dp_event.NoOpDpEvent()),
      ),
      (
          'swor_nan_dataset',
          dp_event.SampledWithoutReplacementDpEvent,
          (float('nan'), 0, dp_event.NoOpDpEvent()),
      ),
      (
          'swor_neg_sample',
          dp_event.SampledWithoutReplacementDpEvent,
          (10, -1, dp_event.NoOpDpEvent()),
      ),
      (
          'swor_nan_sample',
          dp_event.SampledWithoutReplacementDpEvent,
          (10, float('nan'), dp_event.NoOpDpEvent()),
      ),
      (
          'swor_sample_exceeds_dataset',
          dp_event.SampledWithoutReplacementDpEvent,
          (10, 11, dp_event.NoOpDpEvent()),
      ),
      (
          'tree_neg_noise',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (-0.1, 5),
      ),
      (
          'tree_nan_noise',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (float('nan'), 5),
      ),
      (
          'tree_neg_steps_scalar',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (1.0, -1),
      ),
      (
          'tree_nan_steps_scalar',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (1.0, float('nan')),
      ),
      (
          'tree_empty_steps',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (1.0, []),
      ),
      (
          'tree_neg_steps_list',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (1.0, [5, -1]),
      ),
      (
          'tree_nan_steps_list',
          dp_event.SingleEpochTreeAggregationDpEvent,
          (1.0, [5, float('nan')]),
      ),
      (
          'repeat_and_select_small_mean',
          dp_event.RepeatAndSelectDpEvent,
          (dp_event.NoOpDpEvent(), 0.9, 1.0),
      ),
      (
          'repeat_and_select_nan_mean',
          dp_event.RepeatAndSelectDpEvent,
          (dp_event.NoOpDpEvent(), float('nan'), 1.0),
      ),
      (
          'repeat_and_select_neg_shape',
          dp_event.RepeatAndSelectDpEvent,
          (dp_event.NoOpDpEvent(), 2.0, -0.1),
      ),
      (
          'repeat_and_select_nan_shape',
          dp_event.RepeatAndSelectDpEvent,
          (dp_event.NoOpDpEvent(), 2.0, float('nan')),
      ),
      (
          'mog_neg_std',
          dp_event.MixtureOfGaussiansDpEvent,
          (-0.1, [1.0], [1.0]),
      ),
      (
          'mog_nan_std',
          dp_event.MixtureOfGaussiansDpEvent,
          (float('nan'), [1.0], [1.0]),
      ),
      (
          'mog_len_mismatch',
          dp_event.MixtureOfGaussiansDpEvent,
          (1.0, [1.0], [0.5, 0.5]),
      ),
      (
          'mog_empty',
          dp_event.MixtureOfGaussiansDpEvent,
          (1.0, [], []),
      ),
      (
          'mog_neg_sens',
          dp_event.MixtureOfGaussiansDpEvent,
          (1.0, [-1.0, 1.0], [0.5, 0.5]),
      ),
      (
          'mog_nan_sens',
          dp_event.MixtureOfGaussiansDpEvent,
          (1.0, [float('nan')], [1.0]),
      ),
      (
          'mog_neg_prob',
          dp_event.MixtureOfGaussiansDpEvent,
          (1.0, [1.0, 2.0], [-0.1, 1.1]),
      ),
      (
          'mog_nan_prob',
          dp_event.MixtureOfGaussiansDpEvent,
          (1.0, [1.0, 2.0], [float('nan'), 1.0]),
      ),
      (
          'mog_probs_not_sum_to_1',
          dp_event.MixtureOfGaussiansDpEvent,
          (1.0, [1.0, 2.0], [0.3, 0.3]),
      ),
      ('zcdp_neg_rho', dp_event.ZCDpEvent, (-0.1, 0.0)),
      ('zcdp_nan_rho', dp_event.ZCDpEvent, (float('nan'), 0.0)),
      ('zcdp_neg_xi', dp_event.ZCDpEvent, (1.0, -0.1)),
      ('zcdp_nan_xi', dp_event.ZCDpEvent, (1.0, float('nan'))),
      ('exp_mech_neg_eps', dp_event.ExponentialMechanismDpEvent, (-0.1,)),
      (
          'exp_mech_nan_eps',
          dp_event.ExponentialMechanismDpEvent,
          (float('nan'),),
      ),
      ('permute_and_flip_neg_eps', dp_event.PermuteAndFlipDpEvent, (-0.1,)),
      (
          'permute_and_flip_nan_eps',
          dp_event.PermuteAndFlipDpEvent,
          (float('nan'),),
      ),
      (
          'truncated_subsampled_gaussian_neg_dataset',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (-1, 0.5, 5, 1.0),
      ),
      (
          'truncated_subsampled_gaussian_nan_dataset',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (float('nan'), 0.5, 5, 1.0),
      ),
      (
          'truncated_subsampled_gaussian_neg_prob',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (10, -0.1, 5, 1.0),
      ),
      (
          'truncated_subsampled_gaussian_large_prob',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (10, 1.1, 5, 1.0),
      ),
      (
          'truncated_subsampled_gaussian_nan_prob',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (10, float('nan'), 5, 1.0),
      ),
      (
          'truncated_subsampled_gaussian_neg_batch',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (10, 0.5, -1, 1.0),
      ),
      (
          'truncated_subsampled_gaussian_nan_batch',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (10, 0.5, float('nan'), 1.0),
      ),
      (
          'truncated_subsampled_gaussian_neg_noise',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (10, 0.5, 5, -0.1),
      ),
      (
          'truncated_subsampled_gaussian_nan_noise',
          dp_event.TruncatedSubsampledGaussianDpEvent,
          (10, 0.5, 5, float('nan')),
      ),
      (
          'random_allocation_zero_steps',
          dp_event.RandomAllocationDpEvent,
          (dp_event.NoOpDpEvent(), 0, 0),
      ),
      (
          'random_allocation_nan_steps',
          dp_event.RandomAllocationDpEvent,
          (dp_event.NoOpDpEvent(), 0, float('nan')),
      ),
      (
          'random_allocation_neg_selected',
          dp_event.RandomAllocationDpEvent,
          (dp_event.NoOpDpEvent(), -1, 10),
      ),
      (
          'random_allocation_nan_selected',
          dp_event.RandomAllocationDpEvent,
          (dp_event.NoOpDpEvent(), float('nan'), 10),
      ),
      (
          'random_allocation_selected_exceeds_steps',
          dp_event.RandomAllocationDpEvent,
          (dp_event.NoOpDpEvent(), 11, 10),
      ),
  )
  def test_invalid_parameters_raise_value_error(self, event_cls, args):
    with self.assertRaises(ValueError):
      event_cls(*args)


if __name__ == '__main__':
  absltest.main()
