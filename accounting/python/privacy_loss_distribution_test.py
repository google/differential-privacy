# Copyright 2020 Google LLC.
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

"""Tests for google3.third_party.differential_privacy.accounting.python.privacy_loss_distribution."""

import math
import unittest
from absl.testing import parameterized
from scipy import stats

import privacy_loss_distribution


def dictionary_almost_equal(testcase, dictionary1, dictionary2):
  """Check two dictionaries have almost equal values."""
  for i in dictionary1.keys():
    testcase.assertAlmostEqual(dictionary1[i], dictionary2.get(i, 0))
  for i in dictionary2.keys():
    testcase.assertAlmostEqual(dictionary1.get(i, 0), dictionary2[i])


class ConvolveTest(unittest.TestCase):

  def test_convolve_dictionary(self):
    dictionary1 = {1: 2, 3: 4}
    dictionary2 = {2: 3, 4: 6}
    expected_result = {3: 6, 5: 24, 7: 24}
    result = privacy_loss_distribution.convolve_dictionary(
        dictionary1, dictionary2)
    dictionary_almost_equal(self, expected_result, result)

  def test_self_convolve_dictionary(self):
    inp_dictionary = {1: 2, 3: 5, 4: 6}
    expected_result = {
        3: 8,
        5: 60,
        6: 72,
        7: 150,
        8: 360,
        9: 341,
        10: 450,
        11: 540,
        12: 216
    }
    result = privacy_loss_distribution.self_convolve_dictionary(
        inp_dictionary, 3)
    dictionary_almost_equal(self, expected_result, result)


class PrivacyLossDistributionTest(unittest.TestCase):

  def test_hockey_stick_basic(self):
    # Basic hockey stick divergence computation test
    probability_mass_function_lower = {1: math.log(0.5), 2: math.log(0.5)}
    probability_mass_function_upper = {1: math.log(0.6), 2: math.log(0.4)}
    privacy_loss_distribution_pessimistic = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower, probability_mass_function_upper)
    privacy_loss_distribution_optimistic = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower,
        probability_mass_function_upper,
        pessimistic_estimate=False)

    # The true 0-hockey stick divergence is 0.1
    # When using pessimistic estimate, the output should be in [0.1, 0.1+1e-4]
    self.assertLessEqual(
        0.1, privacy_loss_distribution_pessimistic.hockey_stick_divergence(0.0))
    self.assertGreaterEqual(
        0.1 + 1e-4,
        privacy_loss_distribution_pessimistic.hockey_stick_divergence(0.0))

    # When using optimistic estimate, the output should be in [0.1 - 1e-4, 0.1]
    self.assertGreaterEqual(
        0.1, privacy_loss_distribution_optimistic.hockey_stick_divergence(0.0))
    self.assertLessEqual(
        0.1 - 1e-4,
        privacy_loss_distribution_optimistic.hockey_stick_divergence(0.0))

    # The true math.log(1.1)-hockey stick divergence is 0.05
    # When using pessimistic estimate, the output should be in [0.05, 0.05+1e-4]
    self.assertLessEqual(
        0.05,
        privacy_loss_distribution_pessimistic.hockey_stick_divergence(
            math.log(1.1)))
    self.assertGreaterEqual(
        0.05 + 1e-4,
        privacy_loss_distribution_pessimistic.hockey_stick_divergence(
            math.log(1.1)))

    # When using optimistic estimate, the output should be in [0.05-1e-4, 0.05]
    self.assertGreaterEqual(
        0.05,
        privacy_loss_distribution_optimistic.hockey_stick_divergence(
            math.log(1.1)))
    self.assertLessEqual(
        0.05 - 1e-4,
        privacy_loss_distribution_pessimistic.hockey_stick_divergence(
            math.log(1.1)))

  def test_hockey_stick_unequal_support(self):
    # Hockey stick divergence computation test when the two distributions have
    # differenet supports
    probability_mass_function_lower = {
        1: math.log(0.2),
        2: math.log(0.2),
        3: math.log(0.6)
    }
    probability_mass_function_upper = {
        1: math.log(0.5),
        2: math.log(0.4),
        4: math.log(0.1)
    }
    privacy_loss_distribution_pessimistic = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower, probability_mass_function_upper)
    privacy_loss_distribution_optimistic = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower,
        probability_mass_function_upper,
        pessimistic_estimate=False)

    # Here 4 appears as an outcome of only mu_upper and hence should be included
    # in the infinity_mass variable.
    self.assertAlmostEqual(
        privacy_loss_distribution_pessimistic.infinity_mass, 0.1)
    self.assertAlmostEqual(
        privacy_loss_distribution_optimistic.infinity_mass, 0.1)

    # The true 0-hockey stick divergence is 0.6
    # When using pessimistic estimate, the output should be in [0.6, 0.6+1e-4]
    self.assertLessEqual(
        0.6, privacy_loss_distribution_pessimistic.hockey_stick_divergence(0.0))
    self.assertGreaterEqual(
        0.6 + 1e-4,
        privacy_loss_distribution_pessimistic.hockey_stick_divergence(0.0))

    # When using optimistic estimate, the output should lie in [0.6 - 1e-4, 0.6]
    self.assertGreaterEqual(
        0.6, privacy_loss_distribution_optimistic.hockey_stick_divergence(0.0))
    self.assertLessEqual(
        0.6 - 1e-4,
        privacy_loss_distribution_optimistic.hockey_stick_divergence(0.0))

    # The true 0.5-hockey stick divergence is 0.34051149172
    # When using pessimistic estimate, the output should be in
    # [0.3405, 0.3405 + 1e-4]
    self.assertLessEqual(
        0.3405,
        privacy_loss_distribution_pessimistic.hockey_stick_divergence(0.5))
    self.assertGreaterEqual(
        0.3405 + 1e-4,
        privacy_loss_distribution_pessimistic.hockey_stick_divergence(0.5))

    # When using optimistic estimate, the output should lie in
    # [0.3405 - 1e-4, 0.3405]
    self.assertGreaterEqual(
        0.3405,
        privacy_loss_distribution_optimistic.hockey_stick_divergence(0.5))
    self.assertLessEqual(
        0.3405 - 1e-4,
        privacy_loss_distribution_optimistic.hockey_stick_divergence(0.5))

  def test_truncation(self):
    # Test for truncation
    probability_mass_function_lower = {
        1: math.log(0.2),
        2: math.log(0.2),
        3: math.log(0.6)
    }
    probability_mass_function_upper = {
        1: math.log(0.55),
        2: math.log(0.02),
        3: math.log(0.03)
    }
    # Set the truncation threshold to be 0.1
    privacy_loss_distribution_pessimistic = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower,
        probability_mass_function_upper,
        log_mass_truncation_bound=math.log(0.1))
    privacy_loss_distribution_optimistic = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower,
        probability_mass_function_upper,
        log_mass_truncation_bound=math.log(0.1),
        pessimistic_estimate=False)

    # For the outcomes 2 and 3, the probability mass in mu_upper are below the
    # threshold. Hence, they should be discarded, resulting in the total
    # infinity_mass of 0.05 in the pessimistic case
    self.assertAlmostEqual(
        privacy_loss_distribution_pessimistic.infinity_mass, 0.05)

    # In the optimistic case, the infinity_mass should be zero.
    self.assertAlmostEqual(
        privacy_loss_distribution_optimistic.infinity_mass, 0)

    # The 10-hockey stick should be zero, but due to the mass truncation, the
    # output will be 0.05 in the pessimistic case.
    self.assertAlmostEqual(
        privacy_loss_distribution_pessimistic.hockey_stick_divergence(10), 0.05)
    self.assertAlmostEqual(
        privacy_loss_distribution_optimistic.hockey_stick_divergence(10), 0)

  def test_composition(self):
    # Test for composition of privacy loss distribution
    probability_mass_function_lower1 = {
        1: math.log(0.2),
        2: math.log(0.2),
        3: math.log(0.6)
    }
    probability_mass_function_upper1 = {
        1: math.log(0.5),
        2: math.log(0.2),
        4: math.log(0.3)
    }
    privacy_loss_distribution1 = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower1, probability_mass_function_upper1)
    probability_mass_function_lower2 = {
        1: math.log(0.4),
        2: math.log(0.6),
    }
    probability_mass_function_upper2 = {2: math.log(0.7), 3: math.log(0.3)}
    privacy_loss_distribution2 = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower2, probability_mass_function_upper2)

    # Result from composing the above two privacy loss distributions
    result = privacy_loss_distribution1.compose(privacy_loss_distribution2)

    # The correct result
    probability_mass_function_lower_composed = {
        (1, 1): math.log(0.08),
        (1, 2): math.log(0.12),
        (2, 1): math.log(0.08),
        (2, 2): math.log(0.12),
        (3, 1): math.log(0.24),
        (3, 2): math.log(0.36)
    }
    probability_mass_function_upper_composed = {
        (1, 2): math.log(0.35),
        (1, 3): math.log(0.15),
        (2, 2): math.log(0.14),
        (2, 3): math.log(0.06),
        (4, 2): math.log(0.21),
        (4, 3): math.log(0.09)
    }
    expected_result = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower_composed,
        probability_mass_function_upper_composed)

    # Check that the result is as expected. Note that we cannot check that the
    # rounded_down_probability_mass_function and
    # rounded_up_probability_mass_function of the two distributions are equal
    # directly because the rounding might cause off-by-one error in index.
    self.assertAlmostEqual(expected_result.value_discretization_interval,
                           result.value_discretization_interval)
    self.assertAlmostEqual(expected_result.infinity_mass,
                           result.infinity_mass)
    self.assertAlmostEqual(
        expected_result.hockey_stick_divergence(0),
        result.hockey_stick_divergence(0))
    self.assertAlmostEqual(
        expected_result.hockey_stick_divergence(0.5),
        result.hockey_stick_divergence(0.5))

  def test_self_composition(self):
    # Test for self composition of privacy loss distribution
    probability_mass_function_lower = {
        1: math.log(0.2),
        2: math.log(0.2),
        3: math.log(0.6)
    }
    probability_mass_function_upper = {
        1: math.log(0.5),
        2: math.log(0.2),
        4: math.log(0.3)
    }
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        probability_mass_function_lower, probability_mass_function_upper)
    result = pld.self_compose(3)

    expected_probability_mass_lower = {}
    for i, vi in probability_mass_function_lower.items():
      for j, vj in probability_mass_function_lower.items():
        for k, vk in probability_mass_function_lower.items():
          expected_probability_mass_lower[(i, j, k)] = vi + vj + vk
    expected_probability_mass_upper = {}
    for i, vi in probability_mass_function_upper.items():
      for j, vj in probability_mass_function_upper.items():
        for k, vk in probability_mass_function_upper.items():
          expected_probability_mass_upper[(i, j, k)] = vi + vj + vk
    expected_result = privacy_loss_distribution.PrivacyLossDistribution.from_two_probability_mass_functions(
        expected_probability_mass_lower, expected_probability_mass_upper)

    self.assertAlmostEqual(expected_result.value_discretization_interval,
                           result.value_discretization_interval)
    self.assertAlmostEqual(expected_result.infinity_mass,
                           result.infinity_mass)
    self.assertAlmostEqual(
        expected_result.hockey_stick_divergence(0),
        result.hockey_stick_divergence(0))
    self.assertAlmostEqual(
        expected_result.hockey_stick_divergence(0.5),
        result.hockey_stick_divergence(0.5))


class LaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, -0.1, 1), (1, 1, 2, -1), (1, 1, 0.3, 0.4),
                            (4, 4, -0.4, 1), (5, 5, 7, -1), (7, 7, 2.1, 0.4))
  def test_laplace_privacy_loss(self, parameter, sensitivity, x,
                                expected_privacy_loss):
    pld = privacy_loss_distribution.LaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    self.assertAlmostEqual(expected_privacy_loss, pld.privacy_loss(x))

  @parameterized.parameters(
      (1, 1, 1, 0), (1, 1, -1, math.inf), (1, 1, 0.4, 0.3), (4, 4, 1, 0),
      (5, 5, -1, math.inf), (7, 7, 0.4, 2.1), (1, 1, 2, -math.inf),
      (3, 1, 3.1, -math.inf), (4, 4, 1.1, -math.inf))
  def test_laplace_inverse_privacy_loss(self, parameter, sensitivity,
                                        privacy_loss, expected_x):
    pld = privacy_loss_distribution.LaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    self.assertAlmostEqual(expected_x, pld.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters((1, 1, 0, 1, {
      1: 0.5,
      -1: 0.18393972
  }), (3, 3, 0, 3, {
      1: 0.5,
      -1: 0.18393972
  }), (1, 2, 0, 2, {
      2: 0.5,
      -2: 0.06766764
  }), (4, 8, 0, 8, {
      2: 0.5,
      -2: 0.06766764
  }))
  def test_laplace_privacy_loss_tail(self, parameter, sensitivity,
                                     expected_lower_x_truncation,
                                     expected_upper_x_truncation,
                                     expected_tail_probability_mass_function):
    pld = privacy_loss_distribution.LaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    tail_pld = pld.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    dictionary_almost_equal(self, expected_tail_probability_mass_function,
                            tail_pld.tail_probability_mass_function)

  @parameterized.parameters(
      (1, 1, {1: 0.69673467, 0: 0.11932561, -1: 0.18393972}),
      (3, 3, {1: 0.69673467, 0: 0.11932561, -1: 0.18393972}),
      (1, 2, {2: 0.69673467, 1: 0.11932561, 0: 0.07237464, -1: 0.04389744,
              -2: 0.06766764}),
      (2, 4, {2: 0.69673467, 1: 0.11932561, 0: 0.07237464, -1: 0.04389744,
              -2: 0.06766764}))
  def test_laplace_varying_parameter_and_sensitivity(
      self, parameter, sensitivity, expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.LaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters(
      (0.5, {2: 0.61059961, 1: 0.08613506, 0: 0.06708205, -1: 0.05224356,
             -2: 0.18393972}),
      (0.3, {4: 0.52438529, 3: 0.06624934, 2: 0.05702133, 1: 0.04907872,
             0: 0.04224244, -1: 0.03635841, -2: 0.03129397, -3: 0.19337051}))
  def test_laplace_discretization(self, value_discretization_interval,
                                  expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.LaplacePrivacyLossDistribution(
        1, value_discretization_interval=value_discretization_interval)
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters(
      (1, {1: 0.5, 0: 0.19673467, -1: 0.30326533}),
      (2, {2: 0.5, 1: 0.19673467, 0: 0.11932561, -1: 0.07237464,
           -2: 0.11156508}))
  def test_laplace_optimistic(self, sensitivity,
                              expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.LaplacePrivacyLossDistribution(
        1,
        sensitivity=sensitivity,
        pessimistic_estimate=False,
        value_discretization_interval=1)
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((1, 1, 1, 0), (3, 3, 1, 0), (2, 4, 2, 0),
                            (2, 4, 0.5, 0.52763345), (1, 1, 0, 0.39346934),
                            (2, 2, 0, 0.39346934), (1, 1, -2, 0.86466472))
  def test_laplace_hockey_stick_divergence(
      self, parameter, sensitivity, epsilon, expected_divergence):
    pld = privacy_loss_distribution.LaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    self.assertAlmostEqual(expected_divergence,
                           pld.hockey_stick_divergence(epsilon))

  @parameterized.parameters((1, 1, 0, 1), (1, 1, 0.1, 1), (2, 1, 0.01, 2),
                            (1, 3, 0.01, 0.33333333))
  def test_laplace_from_privacy_parameters(self, sensitivity, epsilon, delta,
                                           expected_parameter):
    pld = privacy_loss_distribution.LaplacePrivacyLossDistribution.from_privacy_guarantee(
        privacy_loss_distribution.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity,
        value_discretization_interval=1)
    self.assertAlmostEqual(expected_parameter, pld._parameter)


class GaussianPrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, 5, -4.5), (1, 1, -3, 3.5), (1, 2, 3, -4),
                            (4, 4, 20, -4.5), (5, 5, -15, 3.5), (7, 14, 21, -4))
  def test_gaussian_privacy_loss(self, standard_deviation, sensitivity, x,
                                 expected_privacy_loss):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution(
        standard_deviation,
        sensitivity=sensitivity,
        value_discretization_interval=1)
    self.assertAlmostEqual(expected_privacy_loss, pld.privacy_loss(x))

  @parameterized.parameters((1, 1, -4.5, 5), (1, 1, 3.5, -3), (1, 2, -4, 3),
                            (4, 4, -4.5, 20), (5, 5, 3.5, -15), (7, 14, -4, 21))
  def test_gaussian_inverse_privacy_loss(self, standard_deviation, sensitivity,
                                         privacy_loss, expected_x):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution(
        standard_deviation,
        sensitivity=sensitivity,
        value_discretization_interval=1)
    self.assertAlmostEqual(expected_x, pld.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters((1, 1, -1, 1, True, {
      math.inf: 0.15865525,
      -0.5: 0.15865525
  }), (3, 3, -3, 3, True, {
      math.inf: 0.15865525,
      -0.5: 0.15865525
  }), (1, 2, -1, 1, True, {
      math.inf: 0.15865525,
      0: 0.15865525
  }), (4, 8, -4, 4, True, {
      math.inf: 0.15865525,
      0: 0.15865525
  }), (1, 1, -1, 1, False, {
      1.5: 0.15865525,
  }), (3, 3, -3, 3, False, {
      1.5: 0.15865525,
  }), (1, 2, -1, 1, False, {
      4.0: 0.15865525,
  }), (4, 8, -4, 4, False, {
      4.0: 0.15865525,
  }))
  def test_gaussian_privacy_loss_tail(self, standard_deviation, sensitivity,
                                      expected_lower_x_truncation,
                                      expected_upper_x_truncation,
                                      pessimistic_estimate,
                                      expected_tail_probability_mass_function):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution(
        standard_deviation,
        sensitivity=sensitivity,
        value_discretization_interval=1,
        pessimistic_estimate=pessimistic_estimate,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-1))
    tail_pld = pld.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    dictionary_almost_equal(self, expected_tail_probability_mass_function,
                            tail_pld.tail_probability_mass_function)

  @parameterized.parameters((1, 1, {
      2: 0.12447741,
      1: 0.38292492,
      0: 0.30853754
  }), (5, 5, {
      2: 0.12447741,
      1: 0.38292492,
      0: 0.30853754
  }), (1, 2, {
      1: 0.30853754,
      2: 0.19146246,
      3: 0.19146246,
      4: 0.12447741
  }), (3, 6, {
      1: 0.30853754,
      2: 0.19146246,
      3: 0.19146246,
      4: 0.12447741
  }))
  def test_gaussian_varying_standard_deviation_and_sensitivity(
      self, standard_deviation, sensitivity,
      expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution(
        standard_deviation,
        sensitivity=sensitivity,
        value_discretization_interval=1,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9))
    self.assertAlmostEqual(stats.norm.cdf(-0.9), pld.infinity_mass)
    dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((0.5, {
      3: 0.12447741,
      2: 0.19146246,
      1: 0.19146246,
      0: 0.30853754,
  }), (0.3, {
      5: 0.05790353,
      4: 0.10261461,
      3: 0.11559390,
      2: 0.11908755,
      1: 0.11220275,
      0: 0.09668214,
      -1: 0.21185540
  }))
  def test_gaussian_discretization(self, value_discretization_interval,
                                   expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution(
        1,
        value_discretization_interval=value_discretization_interval,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9))
    self.assertAlmostEqual(stats.norm.cdf(-0.9), pld.infinity_mass)
    dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((1, {
      1: 0.30853754,
      0: 0.38292492,
      -1: 0.12447741
  }), (2, {
      0: 0.12447741,
      1: 0.19146246,
      2: 0.19146246,
      3: 0.30853754,
  }))
  def test_gaussian_optimistic(self, sensitivity,
                               expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution(
        1,
        sensitivity=sensitivity,
        pessimistic_estimate=False,
        value_discretization_interval=1,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9))
    self.assertAlmostEqual(0, pld.infinity_mass)
    dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((0, 1), (-10, 2), (4, 0), (2, -1))
  def test_gaussian_value_errors(self, standard_deviation, sensitivity):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.GaussianPrivacyLossDistribution(
          standard_deviation, sensitivity=sensitivity)

  @parameterized.parameters((1, 1, 1, 0.12693674), (2, 2, 1, 0.12693674),
                            (1, 3, 1, 0.78760074), (2, 6, 1, 0.78760074),
                            (1, 1, 2, 0.02092364), (5, 5, 2, 0.02092364))
  def test_gaussian_hockey_stick_divergence(
      self, standard_deviation, sensitivity, epsilon, expected_divergence):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution(
        standard_deviation,
        sensitivity=sensitivity,
        value_discretization_interval=1)
    self.assertAlmostEqual(expected_divergence,
                           pld.hockey_stick_divergence(epsilon))

  @parameterized.parameters((1, 1, 0.12693674, 1), (2, 1, 0.12693674, 2),
                            (3, 1, 0.78760074, 1), (6, 1, 0.78760074, 2),
                            (1, 2, 0.02092364, 1), (5, 2, 0.02092364, 5),
                            (1, 16, 1e-5, 0.344), (2, 16, 1e-5, 0.688))
  def test_gaussian_from_privacy_parameters(self, sensitivity, epsilon, delta,
                                            expected_standard_deviation):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution.from_privacy_guarantee(
        privacy_loss_distribution.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity,
        value_discretization_interval=1)
    self.assertAlmostEqual(expected_standard_deviation, pld._standard_deviation,
                           3)

  @parameterized.parameters((1, 1, 4, 1, 2), (2, 1, 9, 2, 3))
  def test_gaussian_self_composition(self, standard_deviation, sensitivity,
                                     num_times, expected_standard_deviation,
                                     expected_sensitivity):
    pld = privacy_loss_distribution.GaussianPrivacyLossDistribution(
        standard_deviation,
        sensitivity=sensitivity,
        value_discretization_interval=1)
    composed_pld = pld.self_compose(num_times)
    self.assertAlmostEqual(expected_standard_deviation,
                           composed_pld._standard_deviation)
    self.assertAlmostEqual(expected_sensitivity, composed_pld._sensitivity)


class RandomizedResponsePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters(
      (0.5, 2, {2: 0.75, -1: 0.25}),
      (0.2, 4, {3: 0.85, -2: 0.05, 0: 0.1}))
  def test_randomized_response_basic(
      self, noise_parameter, num_buckets,
      expected_rounded_probability_mass_function):
    # Set value_discretization_interval = 1 here.
    pld = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_randomized_response(
            noise_parameter, num_buckets, value_discretization_interval=1))
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters(
      (0.7, {5: 0.85, -4: 0.05, 0: 0.1}),
      (2, {2: 0.85, -1: 0.05, 0: 0.1}))
  def test_randomized_response_discretization(
      self, value_discretization_interval,
      expected_rounded_probability_mass_function):
    # Set noise_parameter = 0.2, num_buckets = 4 here.
    # The true (non-discretized) PLD is
    # {2.83321334: 0.85, -2.83321334: 0.05, 0: 0.1}.
    pld = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_randomized_response(
            0.2, 4,
            value_discretization_interval=value_discretization_interval))
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters(
      (0.5, 2, {1: 0.75, -2: 0.25}),
      (0.2, 4, {2: 0.85, -3: 0.05, 0: 0.1}))
  def test_randomized_response_optimistic(
      self, noise_parameter, num_buckets,
      expected_rounded_probability_mass_function):
    # Set value_discretization_interval = 1 here.
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_randomized_response(
        noise_parameter,
        num_buckets,
        pessimistic_estimate=False,
        value_discretization_interval=1)
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((0.0, 10), (1.1, 4), (0.5, 1))
  def test_randomized_response_value_errors(self, noise_parameter, num_buckets):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.PrivacyLossDistribution.from_randomized_response(
          noise_parameter, num_buckets)


class DiscreteLaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, 0, 1), (1, 1, 1, -1), (0.3, 2, 0, 0.6),
                            (0.3, 2, 1, 0), (0.3, 2, 2, -0.6))
  def test_discrete_laplace_privacy_loss(self, parameter, sensitivity, x,
                                         expected_privacy_loss):
    pld = privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    self.assertAlmostEqual(expected_privacy_loss, pld.privacy_loss(x))

  @parameterized.parameters((1, 1, 0.4), (2, 7, -1.1))
  def test_discrete_laplace_privacy_loss_value_errors(
      self, parameter, sensitivity, x):
    pld = privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    with self.assertRaises(ValueError):
      pld.privacy_loss(x)

  @parameterized.parameters((1, 1, 1.1, -math.inf), (1, 1, 0.9, 0),
                            (1, 1, -1, math.inf), (0.3, 2, 0.7, -math.inf),
                            (0.3, 2, 0.2, 0), (0.3, 2, 0, 1),
                            (0.3, 2, -0.6, math.inf))
  def test_discrete_laplace_inverse_privacy_loss(self, parameter, sensitivity,
                                                 privacy_loss, expected_x):
    pld = privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    self.assertAlmostEqual(expected_x, pld.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters((1, 1, 1, 0, {
      1: 0.73105858,
      -1: 0.26894142
  }), (0.3, 2, 1, 1, {
      0.6: 0.57444252,
      -0.6: 0.31526074
  }))
  def test_discrete_laplace_privacy_loss_tail(
      self, parameter, sensitivity, expected_lower_x_truncation,
      expected_upper_x_truncation, expected_tail_probability_mass_function):
    pld = privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    tail_pld = pld.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    dictionary_almost_equal(self, expected_tail_probability_mass_function,
                            tail_pld.tail_probability_mass_function)

  @parameterized.parameters(
      (1.0, 1, {1: 0.73105858, -1: 0.26894142}),
      (1.0, 2, {2: 0.73105858, 0: 0.17000340, -2: 0.09893802}),
      (0.8, 2, {2: 0.68997448, 0: 0.17072207, -1: 0.13930345}),
      (0.8, 3, {3: 0.68997448, 1: 0.17072207, 0: 0.07671037, -2: 0.06259307}))
  def test_discrete_laplace_varying_standard_deviation_and_sensitivity(
      self, parameter, sensitivity, expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters(
      (0.7, {3: 0.73105858, 0: 0.17000340, -2: 0.09893802}),
      (2.2, {1: 0.73105858, 0: 0.26894142}))
  def test_discrete_laplace_discretization(
      self, value_discretization_interval,
      expected_rounded_probability_mass_function):
    # Set parameter = 1, sensitivity = 2 here.
    pld = privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
        1,
        sensitivity=2,
        value_discretization_interval=value_discretization_interval)
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters(
      (1.0, 1, {1: 0.73105858, -1: 0.26894142}),
      (1.0, 2, {2: 0.73105858, 0: 0.17000340, -2: 0.09893802}),
      (0.8, 2, {1: 0.68997448, 0: 0.17072207, -2: 0.13930345}),
      (0.8, 3, {2: 0.68997448, 0: 0.17072207, -1: 0.07671037, -3: 0.06259307}))
  def test_discrete_laplace_optimistic(
      self, parameter, sensitivity, expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity,
        pessimistic_estimate=False, value_discretization_interval=1)
    dictionary_almost_equal(
        self,
        expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((-3, 1), (2, 0.5), (2.0, -1))
  def test_discrete_laplace_value_errors(self, parameter, sensitivity):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
          parameter,
          sensitivity=sensitivity)

  @parameterized.parameters((1, 1, 1, 0), (0.333333, 3, 1, 0), (0.5, 4, 2, 0),
                            (0.5, 4, 0.5, 0.54202002), (0.5, 4, 1, 0.39346934),
                            (0.5, 4, -0.5, 0.72222110))
  def test_laplace_hockey_stick_divergence(
      self, parameter, sensitivity, epsilon, expected_divergence):
    pld = privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    self.assertAlmostEqual(expected_divergence,
                           pld.hockey_stick_divergence(epsilon))

  @parameterized.parameters((1, 1, 0, 1), (1, 1, 0.1, 1), (2, 1, 0.01, 0.5),
                            (1, 3, 0.01, 3))
  def test_discrete_laplace_from_privacy_parameters(self, sensitivity, epsilon,
                                                    delta, expected_parameter):
    pld = (
        privacy_loss_distribution.DiscreteLaplacePrivacyLossDistribution
        .from_privacy_guarantee(
            privacy_loss_distribution.DifferentialPrivacyParameters(
                epsilon, delta),
            sensitivity,
            value_discretization_interval=1))
    self.assertAlmostEqual(expected_parameter, pld._parameter)


if __name__ == '__main__':
  unittest.main()
