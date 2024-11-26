# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pld_privacy_accountant."""

import math

from absl.testing import absltest
from absl.testing import parameterized

from dp_accounting import dp_event
from dp_accounting import privacy_accountant
from dp_accounting import privacy_accountant_test
from dp_accounting.pld import pld_privacy_accountant


class PldPrivacyAccountantTest(privacy_accountant_test.PrivacyAccountantTest,
                               parameterized.TestCase):

  def _make_test_accountants(self):
    return [
        pld_privacy_accountant.PLDAccountant(
            pld_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE),
        pld_privacy_accountant.PLDAccountant(
            pld_privacy_accountant.NeighborRel.REPLACE_ONE),
        pld_privacy_accountant.PLDAccountant(
            pld_privacy_accountant.NeighborRel.REPLACE_SPECIAL),
    ]

  @parameterized.parameters(
      (dp_event.RandomizedResponseDpEvent(0.1, 3),
       pld_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE,
       'neighboring_relation must be `REPLACE_ONE` or `REPLACE_SPECIAL`'),
      (dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)),
       pld_privacy_accountant.NeighborRel.REPLACE_ONE,
       'neighboring_relation must be `ADD_OR_REMOVE_ONE` or `REPLACE_SPECIAL`'),
  )
  def test_composition_errors_for_adjacency(
      self, event, neighboring_relation, error_msg):
    pld_accountant = pld_privacy_accountant.PLDAccountant(neighboring_relation)
    with self.assertRaisesRegex(privacy_accountant.UnsupportedEventError,
                                error_msg):
      pld_accountant.compose(event)

  @parameterized.named_parameters(
      ('replace_one',
       pld_privacy_accountant.NeighborRel.REPLACE_ONE, True),
      ('replace_special',
       pld_privacy_accountant.NeighborRel.REPLACE_SPECIAL, True),
      ('add_or_remove_one',
       pld_privacy_accountant.NeighborRel.ADD_OR_REMOVE_ONE, False),
  )
  def test_supports_randomized_response(
      self, neighboring_relation, supports_composition):
    pld_accountant = pld_privacy_accountant.PLDAccountant(neighboring_relation)
    event = dp_event.RandomizedResponseDpEvent(noise_parameter=0.1,
                                               num_buckets=3)
    self.assertEqual(pld_accountant.supports(event), supports_composition)

  @parameterized.named_parameters(
      ('replace_one', pld_privacy_accountant.NeighborRel.REPLACE_ONE),
      ('replace_special', pld_privacy_accountant.NeighborRel.REPLACE_SPECIAL),
  )
  def test_randomized_response_with_single_bucket(self, neighboring_relation):
    accountant = pld_privacy_accountant.PLDAccountant(neighboring_relation)
    accountant.compose(dp_event.RandomizedResponseDpEvent(0.0, 1))
    self.assertEqual(accountant.get_delta(0), 0)
    self.assertEqual(accountant.get_epsilon(0), 0)

  def test_randomized_response_with_zero_noise_parameter(self):
    accountant = pld_privacy_accountant.PLDAccountant(
        pld_privacy_accountant.NeighborRel.REPLACE_SPECIAL)
    accountant.compose(dp_event.RandomizedResponseDpEvent(0.0, 3))
    self.assertEqual(accountant.get_delta(1.0), 1)
    self.assertEqual(accountant.get_epsilon(0.01), math.inf)

  @parameterized.parameters(
      (pld_privacy_accountant.NeighborRel.REPLACE_ONE,
       4 / (3 + math.exp(1)), 4, 1.0),
      (pld_privacy_accountant.NeighborRel.REPLACE_SPECIAL,
       (4 - math.exp(1)) / 3, 4, 1.0),
      (pld_privacy_accountant.NeighborRel.REPLACE_SPECIAL,
       math.exp(-1), 2, 1.0),
  )
  def test_randomized_response(
      self, neighboring_relation,
      noise_parameter, num_buckets, expected_epsilon):
    accountant = pld_privacy_accountant.PLDAccountant(neighboring_relation)
    accountant.compose(
        dp_event.RandomizedResponseDpEvent(noise_parameter, num_buckets))
    self.assertAlmostEqual(accountant.get_delta(expected_epsilon), 0.0)

  @parameterized.parameters(
      dp_event.GaussianDpEvent(1.0),
      dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(1.0), 6),
      dp_event.ComposedDpEvent(
          [dp_event.GaussianDpEvent(1.0),
           dp_event.GaussianDpEvent(2.0)]),
      dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)),
      dp_event.ComposedDpEvent([
          dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)),
          dp_event.GaussianDpEvent(2.0)
      ]))
  def test_supports_gaussian(self, event):
    pld_accountant = pld_privacy_accountant.PLDAccountant()
    self.assertTrue(pld_accountant.supports(event))

  def test_poisson_subsampling_not_supported_for_replace_one(self):
    pld_accountant = pld_privacy_accountant.PLDAccountant(
        pld_privacy_accountant.NeighborRel.REPLACE_ONE)
    event = dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0))
    self.assertFalse(pld_accountant.supports(event))

  def test_supports_subsampled_gaussian_and_rr_composition(self):
    pld_accountant = pld_privacy_accountant.PLDAccountant(
        pld_privacy_accountant.NeighborRel.REPLACE_SPECIAL)
    event = dp_event.ComposedDpEvent([
        dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)),
        dp_event.RandomizedResponseDpEvent(noise_parameter=0.1, num_buckets=3)
    ])
    self.assertTrue(pld_accountant.supports(event))

  @parameterized.parameters(0, -1)
  def test_non_positive_composition_value_error(self, count):
    event = dp_event.GaussianDpEvent(1.0)
    accountant = pld_privacy_accountant.PLDAccountant()
    with self.assertRaises(ValueError):
      accountant.compose(event, count)

  @parameterized.parameters(
      dp_event.GaussianDpEvent(0),
      dp_event.LaplaceDpEvent(0),
      dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(0)),
      dp_event.PoissonSampledDpEvent(0.1, dp_event.LaplaceDpEvent(0)),
      dp_event.MixtureOfGaussiansDpEvent(0, [1, 2], [0.5, 0.5]),
  )
  def test_additive_noise_mechanisms_with_zero_noise_multiplier(self, event):
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(event)
    self.assertEqual(accountant.get_delta(1.0), 1)
    self.assertEqual(accountant.get_epsilon(0.01), math.inf)

  @parameterized.parameters(
      dp_event.PoissonSampledDpEvent(0, dp_event.GaussianDpEvent(1)),
      dp_event.PoissonSampledDpEvent(0, dp_event.LaplaceDpEvent(1)),
      dp_event.PoissonSampledDpEvent(0, dp_event.GaussianDpEvent(0)),
      dp_event.PoissonSampledDpEvent(0, dp_event.LaplaceDpEvent(0)),
      dp_event.MixtureOfGaussiansDpEvent(1, [0], [1.0]),
      dp_event.MixtureOfGaussiansDpEvent(0, [0], [1.0]),
  )
  def test_poisson_subsampling_with_zero_probability(self, event):
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(event)
    self.assertEqual(accountant.get_delta(0), 0)
    self.assertEqual(accountant.get_epsilon(0), 0)

  def test_gaussian_basic(self):
    gaussian_event = dp_event.GaussianDpEvent(noise_multiplier=math.sqrt(3))
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(gaussian_event, 1)
    accountant.compose(gaussian_event, 2)

    exact_epsilon = 1
    exact_delta = 0.126936
    self.assertAlmostEqual(
        accountant.get_delta(exact_epsilon), exact_delta, delta=1e-3)
    self.assertAlmostEqual(
        accountant.get_epsilon(exact_delta), exact_epsilon, delta=1e-3)

  def test_poisson_subsampled_gaussian(self):
    subsampled_gaussian_event = dp_event.PoissonSampledDpEvent(
        0.2, dp_event.GaussianDpEvent(noise_multiplier=0.5))
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(subsampled_gaussian_event, 1)
    accountant.compose(subsampled_gaussian_event, 2)

    exact_epsilon = 1
    expected_delta = 0.15594
    self.assertAlmostEqual(
        accountant.get_delta(exact_epsilon), expected_delta, delta=1e-3)
    self.assertAlmostEqual(
        accountant.get_epsilon(expected_delta), exact_epsilon, delta=1e-3)

  def test_self_composed_subsampled_gaussian(self):
    event = dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(0.2, dp_event.GaussianDpEvent(0.5)), 3)
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(event)

    exact_epsilon = 1
    expected_delta = 0.15594
    self.assertAlmostEqual(
        accountant.get_delta(exact_epsilon), expected_delta, delta=1e-3)
    self.assertAlmostEqual(
        accountant.get_epsilon(expected_delta), exact_epsilon, delta=1e-3)

  def test_laplace_basic(self):
    first_laplace_event = dp_event.LaplaceDpEvent(noise_multiplier=1)
    second_laplace_event = dp_event.LaplaceDpEvent(noise_multiplier=2)
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(first_laplace_event, 3)
    accountant.compose(second_laplace_event, 2)

    expected_epsilon = 4
    expected_delta = 1e-14  # expected delta is not 0 due to truncation in
    # self composition
    self.assertAlmostEqual(
        accountant.get_delta(expected_epsilon), expected_delta, delta=1e-6)
    self.assertAlmostEqual(
        accountant.get_epsilon(expected_delta), expected_epsilon, delta=1e-6)

  def test_poisson_subsampled_laplace(self):
    subsampled_laplace_event = dp_event.PoissonSampledDpEvent(
        0.2, dp_event.LaplaceDpEvent(noise_multiplier=0.5))
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(subsampled_laplace_event, 1)
    accountant.compose(subsampled_laplace_event, 2)

    exact_epsilon = 2.46964
    expected_delta = 1e-14  # expected delta is not 0 due to truncation in
    # self composition
    self.assertAlmostEqual(
        accountant.get_delta(exact_epsilon), expected_delta, delta=1e-6)
    self.assertAlmostEqual(
        accountant.get_epsilon(expected_delta), exact_epsilon, delta=1e-3)

  def test_mixture_of_gaussians_basic(self):
    first_mog_event = dp_event.MixtureOfGaussiansDpEvent(
        standard_deviation=1.0,
        sensitivities=[0.0, 1.0, 2.0],
        sampling_probs=[0.2, 0.6, 0.2],
    )
    second_mog_event = dp_event.MixtureOfGaussiansDpEvent(
        standard_deviation=2.0,
        sensitivities=[0.0, 2.0, 3.0],
        sampling_probs=[0.33, 0.34, 0.33],
    )
    accountant = pld_privacy_accountant.PLDAccountant()
    accountant.compose(first_mog_event, 3)
    accountant.compose(second_mog_event, 2)

    expected_epsilon = 17.193115
    expected_delta = 1e-6
    self.assertAlmostEqual(
        accountant.get_delta(expected_epsilon), expected_delta, delta=1e-12
    )
    self.assertAlmostEqual(
        accountant.get_epsilon(expected_delta), expected_epsilon, delta=1e-6
    )


if __name__ == '__main__':
  absltest.main()
