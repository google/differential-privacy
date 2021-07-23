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
"""Tests for privacy_loss_distribution.py."""
import math
import unittest
from absl.testing import parameterized
from scipy import stats

from dp_accounting import common
from dp_accounting import privacy_loss_distribution
from dp_accounting import test_util


class PrivacyLossDistributionTest(parameterized.TestCase):

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
        0.1, privacy_loss_distribution_pessimistic.get_delta_for_epsilon(0.0))
    self.assertGreaterEqual(
        0.1 + 1e-4,
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(0.0))

    # When using optimistic estimate, the output should be in [0.1 - 1e-4, 0.1]
    self.assertGreaterEqual(
        0.1, privacy_loss_distribution_optimistic.get_delta_for_epsilon(0.0))
    self.assertLessEqual(
        0.1 - 1e-4,
        privacy_loss_distribution_optimistic.get_delta_for_epsilon(0.0))

    # The true math.log(1.1)-hockey stick divergence is 0.05
    # When using pessimistic estimate, the output should be in [0.05, 0.05+1e-4]
    self.assertLessEqual(
        0.05,
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(
            math.log(1.1)))
    self.assertGreaterEqual(
        0.05 + 1e-4,
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(
            math.log(1.1)))

    # When using optimistic estimate, the output should be in [0.05-1e-4, 0.05]
    self.assertGreaterEqual(
        0.05,
        privacy_loss_distribution_optimistic.get_delta_for_epsilon(
            math.log(1.1)))
    self.assertLessEqual(
        0.05 - 1e-4,
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(
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
    self.assertAlmostEqual(privacy_loss_distribution_pessimistic.infinity_mass,
                           0.1)
    self.assertAlmostEqual(privacy_loss_distribution_optimistic.infinity_mass,
                           0.1)

    # The true 0-hockey stick divergence is 0.6
    # When using pessimistic estimate, the output should be in [0.6, 0.6+1e-4]
    self.assertLessEqual(
        0.6, privacy_loss_distribution_pessimistic.get_delta_for_epsilon(0.0))
    self.assertGreaterEqual(
        0.6 + 1e-4,
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(0.0))

    # When using optimistic estimate, the output should lie in [0.6 - 1e-4, 0.6]
    self.assertGreaterEqual(
        0.6, privacy_loss_distribution_optimistic.get_delta_for_epsilon(0.0))
    self.assertLessEqual(
        0.6 - 1e-4,
        privacy_loss_distribution_optimistic.get_delta_for_epsilon(0.0))

    # The true 0.5-hockey stick divergence is 0.34051149172
    # When using pessimistic estimate, the output should be in
    # [0.3405, 0.3405 + 1e-4]
    self.assertLessEqual(
        0.3405,
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(0.5))
    self.assertGreaterEqual(
        0.3405 + 1e-4,
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(0.5))

    # When using optimistic estimate, the output should lie in
    # [0.3405 - 1e-4, 0.3405]
    self.assertGreaterEqual(
        0.3405, privacy_loss_distribution_optimistic.get_delta_for_epsilon(0.5))
    self.assertLessEqual(
        0.3405 - 1e-4,
        privacy_loss_distribution_optimistic.get_delta_for_epsilon(0.5))

  @parameterized.parameters(
      ({
          4: 0.2,
          2: 0.7
      }, 0.5, 0.1, 0.5, 0.56358432),
      ({
          4: 0.2,
          2: 0.7
      }, 0.5, 0.1, 0.2, 1.30685282),
      ({
          1: 0.2,
          -1: 0.7
      }, 1, 0.1, 0.4, 0),
      ({
          1: 0.6
      }, 1, 0.5, 0.4, math.inf),
      ({
          -1: 0.1
      }, 1, 0, 0, 0),
      # Test resilience against overflow
      ({
          5000: 1
      }, 1, 0, 0.1, 5000),
      ({
          5000: 0.2,
          4000: 0.1,
          3000: 0.7
      }, 1, 0.1, 0.4, 4000))
  def test_get_epsilon_for_delta(self, rounded_probability_mass_function,
                                 value_discretization_interval, infinity_mass,
                                 delta, expected_epsilon):
    pld = privacy_loss_distribution.PrivacyLossDistribution(
        rounded_probability_mass_function, value_discretization_interval,
        infinity_mass)
    self.assertAlmostEqual(expected_epsilon, pld.get_epsilon_for_delta(delta))

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
    self.assertAlmostEqual(privacy_loss_distribution_pessimistic.infinity_mass,
                           0.05)

    # In the optimistic case, the infinity_mass should be zero.
    self.assertAlmostEqual(privacy_loss_distribution_optimistic.infinity_mass,
                           0)

    # The 10-hockey stick should be zero, but due to the mass truncation, the
    # output will be 0.05 in the pessimistic case.
    self.assertAlmostEqual(
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(10), 0.05)
    self.assertAlmostEqual(
        privacy_loss_distribution_optimistic.get_delta_for_epsilon(10), 0)

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
    self.assertAlmostEqual(expected_result.infinity_mass, result.infinity_mass)
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0),
        result.get_delta_for_epsilon(0))
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0.5),
        result.get_delta_for_epsilon(0.5))

  def test_composition_with_truncation(self):
    pld1 = privacy_loss_distribution.PrivacyLossDistribution(
        {
            0: 0.1,
            1: 0.7,
            2: 0.1
        }, 1, 0.1)
    pld2 = privacy_loss_distribution.PrivacyLossDistribution(
        {
            0: 0.1,
            1: 0.6,
            2: 0.2
        }, 1, 0.1)
    pld_composed = pld1.compose(pld2, tail_mass_truncation=0.021)
    self.assertAlmostEqual(pld_composed.infinity_mass, 0.211)
    test_util.dictionary_almost_equal(
        self, {
            1: 0.13,
            2: 0.45,
            3: 0.20,
            4: 0.02,
        }, pld_composed.rounded_probability_mass_function)

  def test_composition_different_estimate_types(self):
    pld1 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_laplace_mechanism(1, pessimistic_estimate=True))
    pld2 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_laplace_mechanism(1, pessimistic_estimate=False))
    with self.assertRaises(ValueError):
      pld1.validate_composable(pld2)
    with self.assertRaises(ValueError):
      pld1.compose(pld2)
    with self.assertRaises(ValueError):
      pld1.get_delta_for_epsilon_for_composed_pld(pld2, 0.1)

  def test_compose_and_get_epsilon_for_delta(self):
    pld1 = privacy_loss_distribution.PrivacyLossDistribution(
        {
            0: 0.1,
            1: 0.7,
            2: 0.1
        }, 0.4, 0.1)
    pld2 = privacy_loss_distribution.PrivacyLossDistribution(
        {
            1: 0.1,
            2: 0.6,
            3: 0.25
        }, 0.4, 0.05)
    self.assertAlmostEqual(
        0.29560003,
        pld1.get_delta_for_epsilon_for_composed_pld(pld2, 1.1))

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
    self.assertAlmostEqual(expected_result.infinity_mass, result.infinity_mass)
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0),
        result.get_delta_for_epsilon(0))
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0.5),
        result.get_delta_for_epsilon(0.5))

  def test_self_composition_truncation(self):
    # Use Gaussian mechanism because it has closed form formula even afer
    # composition. For the setting of parameter below, the privacy loss after
    # composition should be the same as privacy loss with standard_deviation =
    # sensitivity.
    standard_deviation = 20
    num_composition = standard_deviation * standard_deviation
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_gaussian_mechanism(
        standard_deviation,
        value_discretization_interval=1e-5)
    pld = pld.self_compose(num_composition)
    expected_delta = 0.00153
    self.assertAlmostEqual(expected_delta, pld.get_delta_for_epsilon(3), 4)

  def test_self_composition_truncation_account_for_truncated_mass(self):
    num_composition = 2
    tail_mass_truncation = 0.5
    epsilon_initial = 1
    pld = privacy_loss_distribution.PrivacyLossDistribution(
        {
            1: 0.7,
            -1: 0.3
        }, epsilon_initial, 0)
    pld = pld.self_compose(
        num_composition, tail_mass_truncation=tail_mass_truncation)
    self.assertGreater(
        pld.get_delta_for_epsilon(num_composition * epsilon_initial), 0)

  def test_self_composition_no_truncation_optimistic(self):
    num_composition = 2
    tail_mass_truncation = 0.5
    epsilon_initial = 1
    pld = privacy_loss_distribution.PrivacyLossDistribution(
        {
            1: 0.7,
            -1: 0.3
        },
        epsilon_initial,
        0,
        pessimistic_estimate=False)
    pld = pld.self_compose(
        num_composition, tail_mass_truncation=tail_mass_truncation)
    self.assertAlmostEqual(
        0, pld.get_delta_for_epsilon(num_composition * epsilon_initial))

  @parameterized.parameters((1, 0, 1, {
      1: 0.73105858,
      -1: 0.26894142
  }, 0), (1, 0, 0.3, {
      4: 0.73105858,
      -3: 0.26894142
  }, 0), (0.5, 0.2, 0.5, {
      1: 0.49796746,
      -1: 0.30203254
  }, 0.2), (0.5, 0.2, 0.07, {
      8: 0.49796746,
      -7: 0.30203254
  }, 0.2))
  def test_from_privacy_parameters(self, epsilon, delta,
                                   value_discretization_interval,
                                   expected_rounded_probability_mass_function,
                                   expected_infinity_mass):
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_privacy_parameters(
        common.DifferentialPrivacyParameters(epsilon, delta),
        value_discretization_interval=value_discretization_interval)
    self.assertAlmostEqual(expected_infinity_mass, pld.infinity_mass)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)


class LaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, {
      1: 0.69673467,
      0: 0.11932561,
      -1: 0.18393972
  }), (3, 3, {
      1: 0.69673467,
      0: 0.11932561,
      -1: 0.18393972
  }), (1, 2, {
      2: 0.69673467,
      1: 0.11932561,
      0: 0.07237464,
      -1: 0.04389744,
      -2: 0.06766764
  }), (2, 4, {
      2: 0.69673467,
      1: 0.11932561,
      0: 0.07237464,
      -1: 0.04389744,
      -2: 0.06766764
  }))
  def test_laplace_varying_parameter_and_sensitivity(
      self, parameter, sensitivity, expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_laplace_mechanism(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((0.5, {
      2: 0.61059961,
      1: 0.08613506,
      0: 0.06708205,
      -1: 0.05224356,
      -2: 0.18393972
  }), (0.3, {
      4: 0.52438529,
      3: 0.06624934,
      2: 0.05702133,
      1: 0.04907872,
      0: 0.04224244,
      -1: 0.03635841,
      -2: 0.03129397,
      -3: 0.19337051
  }))
  def test_laplace_discretization(self, value_discretization_interval,
                                  expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_laplace_mechanism(
        1, value_discretization_interval=value_discretization_interval)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((1, {
      1: 0.5,
      0: 0.19673467,
      -1: 0.30326533
  }), (2, {
      2: 0.5,
      1: 0.19673467,
      0: 0.11932561,
      -1: 0.07237464,
      -2: 0.11156508
  }))
  def test_laplace_optimistic(self, sensitivity,
                              expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_laplace_mechanism(
        1,
        sensitivity=sensitivity,
        pessimistic_estimate=False,
        value_discretization_interval=1)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)


class GaussianPrivacyLossDistributionTest(parameterized.TestCase):

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
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_gaussian_mechanism(
        standard_deviation,
        sensitivity=sensitivity,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=1)
    self.assertAlmostEqual(stats.norm.cdf(-0.9), pld.infinity_mass)
    test_util.dictionary_almost_equal(
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
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_gaussian_mechanism(
        1,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=value_discretization_interval)
    self.assertAlmostEqual(stats.norm.cdf(-0.9), pld.infinity_mass)
    test_util.dictionary_almost_equal(
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
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_gaussian_mechanism(
        1,
        sensitivity=sensitivity,
        pessimistic_estimate=False,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=1)
    self.assertAlmostEqual(0, pld.infinity_mass)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)


class DiscreteLaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1.0, 1, {
      1: 0.73105858,
      -1: 0.26894142
  }), (1.0, 2, {
      2: 0.73105858,
      0: 0.17000340,
      -2: 0.09893802
  }), (0.8, 2, {
      2: 0.68997448,
      0: 0.17072207,
      -1: 0.13930345
  }), (0.8, 3, {
      3: 0.68997448,
      1: 0.17072207,
      0: 0.07671037,
      -2: 0.06259307
  }))
  def test_discrete_laplace_varying_parameter_and_sensitivity(
      self, parameter, sensitivity, expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_discrete_laplace_mechanism(
        parameter, sensitivity=sensitivity, value_discretization_interval=1)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((0.1, {
      10: 0.73105858,
      -10: 0.26894142
  }), (0.03, {
      34: 0.73105858,
      -33: 0.26894142
  }))
  def test_discrete_laplace_discretization(
      self, value_discretization_interval,
      expected_rounded_probability_mass_function):
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_discrete_laplace_mechanism(
        1, value_discretization_interval=value_discretization_interval)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)


class DiscreteGaussianPrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, {
      5000: 0.45186276,
      -5000: 0.27406862
  }, 0.27406862), (1, 2, {
      0: 0.27406862
  }, 0.72593138), (3, 1, {
      556: 0.34579116,
      -555: 0.32710442
  }, 0.32710442))
  def test_discrete_gaussian_varying_sigma_and_sensitivity(
      self, sigma, sensitivity, expected_rounded_probability_mass_function,
      expected_infinity_mass):
    pld = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_discrete_gaussian_mechanism(
            sigma, sensitivity=sensitivity, truncation_bound=1))
    self.assertAlmostEqual(pld.infinity_mass, expected_infinity_mass)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((2, {
      15000: 0.24420134,
      5000: 0.40261995,
      -5000: 0.24420134,
      -15000: 0.05448868
  }, 0.05448868), (3, {
      25000: 0.05400558,
      15000: 0.24203622,
      5000: 0.39905027,
      -5000: 0.24203623,
      -15000: 0.05400558,
      -25000: 0.00443305
  }, 0.00443305))
  def test_discrete_gaussian_truncation(
      self, truncation_bound, expected_rounded_probability_mass_function,
      expected_infinity_mass):
    pld = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_discrete_gaussian_mechanism(1, truncation_bound=truncation_bound))
    self.assertAlmostEqual(pld.infinity_mass, expected_infinity_mass)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)


class RandomizedResponsePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((0.5, 2, {
      2: 0.75,
      -1: 0.25
  }), (0.2, 4, {
      3: 0.85,
      -2: 0.05,
      0: 0.1
  }))
  def test_randomized_response_basic(
      self, noise_parameter, num_buckets,
      expected_rounded_probability_mass_function):
    # Set value_discretization_interval = 1 here.
    pld = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_randomized_response(
            noise_parameter, num_buckets, value_discretization_interval=1))
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((0.7, {
      5: 0.85,
      -4: 0.05,
      0: 0.1
  }), (2, {
      2: 0.85,
      -1: 0.05,
      0: 0.1
  }))
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
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((0.5, 2, {
      1: 0.75,
      -2: 0.25
  }), (0.2, 4, {
      2: 0.85,
      -3: 0.05,
      0: 0.1
  }))
  def test_randomized_response_optimistic(
      self, noise_parameter, num_buckets,
      expected_rounded_probability_mass_function):
    # Set value_discretization_interval = 1 here.
    pld = privacy_loss_distribution.PrivacyLossDistribution.from_randomized_response(
        noise_parameter,
        num_buckets,
        pessimistic_estimate=False,
        value_discretization_interval=1)
    test_util.dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld.rounded_probability_mass_function)

  @parameterized.parameters((0.0, 10), (1.1, 4), (0.5, 1))
  def test_randomized_response_value_errors(self, noise_parameter, num_buckets):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.PrivacyLossDistribution.from_randomized_response(
          noise_parameter, num_buckets)


class IdentityPrivacyLossDistributionTest(parameterized.TestCase):

  def test_identity(self):
    pld = privacy_loss_distribution.PrivacyLossDistribution.identity()
    test_util.dictionary_almost_equal(self,
                                      pld.rounded_probability_mass_function,
                                      {0: 1})
    self.assertAlmostEqual(pld.infinity_mass, 0)

    pld = pld.compose(
        privacy_loss_distribution.PrivacyLossDistribution({
            1: 0.5,
            -1: 0.5
        }, 1e-4, 0))
    test_util.dictionary_almost_equal(self,
                                      pld.rounded_probability_mass_function, {
                                          1: 0.5,
                                          -1: 0.5
                                      })
    self.assertAlmostEqual(pld.infinity_mass, 0)


if __name__ == '__main__':
  unittest.main()
