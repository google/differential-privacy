# Copyright 2024 Google LLC.
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
"""Tests for Gilbert and McMillan (2018) approximate DP tester."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config
from dp_auditorium.testers import histogram_tester


class HistogramPropertyTesterTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((8.77, 0.5, 2, 11, 0.1), (6.62, 0.5, 10, 37, 0.05))
  def test_get_error_tolerance(
      self,
      expected_error_tolerance,
      epsilon,
      histogram_size,
      num_samples,
      failure_probability,
  ):
    config = property_tester_config.HistogramPropertyTesterConfig(
        approximate_dp=privacy_property.ApproximateDp(
            epsilon=epsilon, delta=0.01
        ),
        max_value=1,
        min_value=0,
        test_discrete_mechanism=False,
        histogram_size=histogram_size,
    )
    probabilities1 = np.random.dirichlet(alpha=np.ones(histogram_size))
    probabilities2 = np.random.dirichlet(alpha=np.ones(histogram_size))
    with self.subTest(use_original_tester=True):
      config.use_original_tester = True
      tester = histogram_tester.HistogramTester(config)
      result = tester._get_error_tolerance(num_samples,
                                           probabilities1,
                                           probabilities2,
                                           failure_probability)
      self.assertAllClose(result, expected_error_tolerance, rtol=1e-2)
    with self.subTest(use_original_tester=False):
      config.use_original_tester = False
      tester = histogram_tester.HistogramTester(config)
      result = tester._get_error_tolerance(num_samples,
                                           probabilities1,
                                           probabilities2,
                                           failure_probability)
      # New tester always has smaller error tolerance than original tester.
      self.assertLess(result, expected_error_tolerance)

  @parameterized.parameters(True, False)
  def test_estimate_lower_bound(self, use_original_tester):
    """Verifies estimate of delta lower bound.

    Verifies that correct delta is calculated for a mechanism that
    deterministically outputs 0 for one dataset and 1 for another dataset.
    """
    # Simulate samples.
    num_samples = 100
    samples1 = np.zeros(num_samples)
    samples2 = np.ones(num_samples)
    probabilities1 = np.array([1, 0])
    probabilities2 = np.array([0, 1])

    # Initialize tester.
    config = property_tester_config.HistogramPropertyTesterConfig(
        approximate_dp=privacy_property.ApproximateDp(epsilon=3.0, delta=1e-6),
        max_value=1.0,
        min_value=0.0,
        test_discrete_mechanism=False,
        histogram_size=2,
    )
    config.use_original_tester = use_original_tester
    tester = histogram_tester.HistogramTester(config)

    # Estimate delta.
    failure_probability = 0.1
    expected_delta = 1.0 - tester._get_error_tolerance(
        num_samples, probabilities1, probabilities2, failure_probability
    )
    estimated_delta = tester.estimate_lower_bound(
        samples1, samples2, failure_probability
    )
    self.assertEqual(estimated_delta, expected_delta)

  def test_reject_property(self):
    """Verifies that privacy violation is correctly determined."""
    delta = 0.1
    config = property_tester_config.HistogramPropertyTesterConfig(
        approximate_dp=privacy_property.ApproximateDp(epsilon=3.0, delta=delta),
        max_value=1,
        min_value=0,
        test_discrete_mechanism=False,
        use_original_tester=True,
        histogram_size=2,
    )
    tester = histogram_tester.HistogramTester(config)
    rejects_property = tester.reject_property(delta + 0.1)
    with self.subTest('rejects_property'):
      self.assertTrue(rejects_property)
    does_not_reject_property = tester.reject_property(delta - 0.1)
    with self.subTest('does_not_reject_property'):
      self.assertFalse(does_not_reject_property)

  def test_estimate_discrete_distribution_with_empty_array(self):
    with self.assertRaisesRegex(
        ValueError, 'Cannot estimate probability mass from empty array.'
    ):
      _ = histogram_tester._estimate_discrete_distribution(
          np.array([]), universe_size=1
      )

  @parameterized.parameters(([-1, 3, 4],), ([0, 5],))
  def test_estimate_discrete_distribution_with_wrong_universe(self, samples):
    universe_size = 5
    with self.assertRaisesRegex(
        ValueError,
        'The samples from the mechanism have a different range than'
        f' {{0,..., {universe_size-1}}}.',
    ):
      _ = histogram_tester._estimate_discrete_distribution(
          np.array(samples), universe_size=universe_size
      )

  def test_estimate_continuous_distribution_with_empty_array(self):
    with self.assertRaises(ValueError):
      _ = histogram_tester._estimate_continuous_distribution(
          np.array([]),
          universe_size=1,
          min_value=0,
          max_value=1,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='OneElement',
          samples=np.array([0]),
          universe_size=1,
          expected_probability_mass=[1],
      ),
      dict(
          testcase_name='TwoElements1Repeated',
          samples=np.array([1, 1]),
          universe_size=2,
          expected_probability_mass=[0, 1],
      ),
      dict(
          testcase_name='TwoElements1MoreLikely',
          samples=np.array([1, 1, 0, 1]),
          universe_size=2,
          expected_probability_mass=[0.25, 0.75],
      ),
      dict(
          testcase_name='TwoElements0MoreLikely',
          samples=np.array([0, 0, 1, 0]),
          universe_size=2,
          expected_probability_mass=[0.75, 0.25],
      ),
      dict(
          testcase_name='TwoElements0Repeated',
          samples=np.array([0, 0]),
          universe_size=2,
          expected_probability_mass=[1, 0],
      ),
  )
  def test_estimate_discrete_distribution(
      self, samples, universe_size, expected_probability_mass
  ):
    result_probability_mass = histogram_tester._estimate_discrete_distribution(
        samples=samples, universe_size=universe_size
    )
    self.assertAllEqual(result_probability_mass, expected_probability_mass)

  @parameterized.named_parameters(
      dict(
          testcase_name='OneInterval',
          samples=np.array([0.01, -0.01]),
          universe_size=1,
          expected_probability_mass=[1],
      ),
      dict(
          testcase_name='TwoIntervalsOnlyLowerIntervalAppears',
          samples=np.array([0.01, -0.01]),
          universe_size=2,
          expected_probability_mass=[1, 0],
      ),
      dict(
          testcase_name='TwoIntervalsRoundsCorrectlyToUpper',
          samples=np.array([1, 1.5, 0, 1.5]),
          universe_size=2,
          expected_probability_mass=[0.25, 0.75],
      ),
      dict(
          testcase_name='TwoIntervalsRoundsCorrectlyToLower',
          samples=np.array([0, -1, 0.1, 1]),
          universe_size=2,
          expected_probability_mass=[0.75, 0.25],
      ),
      dict(
          testcase_name='TwoIntervalsOnlyUpperIntervalAppears',
          samples=np.array([2.0, 0.9]),
          universe_size=2,
          expected_probability_mass=[0, 1],
      ),
  )
  def test_estimate_continuous_distribution(
      self, samples, universe_size, expected_probability_mass
  ):
    """Verifies that different input samples produce the expected probabilities."""
    result_probability_mass = (
        histogram_tester._estimate_continuous_distribution(
            samples=samples,
            universe_size=universe_size,
            min_value=0.1,
            max_value=1.1,
        )
    )
    self.assertAllEqual(result_probability_mass, expected_probability_mass)


if __name__ == '__main__':
  absltest.main()
