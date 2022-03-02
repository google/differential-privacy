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


class BasicPrivacyLossDistributionTest(parameterized.TestCase):

  def test_hockey_stick_basic(self):
    # Basic hockey stick divergence computation test
    log_pmf_lower = {1: math.log(0.5), 2: math.log(0.5)}
    log_pmf_upper = {1: math.log(0.6), 2: math.log(0.4)}
    privacy_loss_distribution_pessimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper, pessimistic_estimate=True))
    privacy_loss_distribution_optimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper, pessimistic_estimate=False))

    # The true 0-hockey stick divergence is 0.1
    # When using pessimistic estimate, the output should be in [0.1, 0.1+1e-4]
    self.assertTrue(0.1 <= privacy_loss_distribution_pessimistic
                    .get_delta_for_epsilon(0.0) <= 0.1 + 1e-4)
    # When using optimistic estimate, the output should be in [0.1 - 1e-4, 0.1]
    self.assertTrue(0.1 - 1e-4 <= privacy_loss_distribution_optimistic
                    .get_delta_for_epsilon(0.0) <= 0.1)

    # The true math.log(1.1)-hockey stick divergence is 0.05
    # When using pessimistic estimate, the output should be in [0.05, 0.05+1e-4]
    self.assertTrue(0.05 <= privacy_loss_distribution_pessimistic
                    .get_delta_for_epsilon(math.log(1.1)) <= 0.05 + 1e-4)
    # When using optimistic estimate, the output should be in [0.05-1e-4, 0.05]
    self.assertTrue(0.05 - 1e-4 <= privacy_loss_distribution_optimistic
                    .get_delta_for_epsilon(math.log(1.1)) <= 0.05)

    # The true math.log(0.9)-hockey stick divergence is 0.15
    # When using pessimistic estimate, the output should be in [0.15, 0.15+1e-4]
    self.assertTrue(0.15 <= privacy_loss_distribution_pessimistic
                    .get_delta_for_epsilon(math.log(0.9)) <= 0.15 + 1e-4)
    # When using optimistic estimate, the output should be in [0.2-1e-4, 0.2]
    self.assertTrue(0.15 - 1e-4 <= privacy_loss_distribution_optimistic
                    .get_delta_for_epsilon(math.log(0.9)) <= 0.15)

    self.assertTrue(privacy_loss_distribution_pessimistic._symmetric)
    self.assertTrue(privacy_loss_distribution_optimistic._symmetric)

  def test_hockey_stick_unequal_support(self):
    # Hockey stick divergence computation test when the two distributions have
    # differenet supports
    log_pmf_lower = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper = {1: math.log(0.5), 2: math.log(0.4), 4: math.log(0.1)}
    privacy_loss_distribution_pessimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper, pessimistic_estimate=True))
    privacy_loss_distribution_optimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper, pessimistic_estimate=False))

    # Here 4 appears as an outcome of only mu_upper and hence should be included
    # in the infinity_mass variable.
    self.assertAlmostEqual(
        privacy_loss_distribution_pessimistic._basic_pld.infinity_mass, 0.1)
    self.assertAlmostEqual(
        privacy_loss_distribution_optimistic._basic_pld.infinity_mass, 0.1)

    # The true 0-hockey stick divergence is 0.6
    # When using pessimistic estimate, the output should be in [0.6, 0.6+1e-4]
    self.assertTrue(0.6 <= privacy_loss_distribution_pessimistic
                    .get_delta_for_epsilon(0.0) <= 0.6 + 1e-4)
    # When using optimistic estimate, the output should lie in [0.6 - 1e-4, 0.6]
    self.assertTrue(0.6 - 1e-4 <= privacy_loss_distribution_optimistic
                    .get_delta_for_epsilon(0.0) <= 0.6)

    # The true math.log(1.1)-hockey stick divergence is 0.56
    # When using pessimistic estimate, the output should be in
    # [0.56, 0.56 + 1e-4]
    self.assertTrue(0.56 <= privacy_loss_distribution_pessimistic
                    .get_delta_for_epsilon(math.log(1.1)) <= 0.56 + 1e-4)
    # When using optimistic estimate, the output should lie in
    # [0.56 - 1e-4, 0.56]
    self.assertTrue(0.56 - 1e-4 <= privacy_loss_distribution_optimistic
                    .get_delta_for_epsilon(math.log(1.1)) <= 0.56)

    # The true math.log(0.9)-hockey stick divergence is 0.64.
    # When using pessimistic estimate, the output should be in
    # [0.64, 0.64 + 1e-4]
    self.assertTrue(0.64 <= privacy_loss_distribution_pessimistic
                    .get_delta_for_epsilon(math.log(0.9)) <= 0.64 + 1e-4)
    # When using optimistic estimate, the output should lie in
    # [0.64 - 1e-4, 0.64]
    self.assertTrue(0.64 - 1e-4 <= privacy_loss_distribution_optimistic
                    .get_delta_for_epsilon(math.log(0.9)) <= 0.64)

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
    log_pmf_lower = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper = {1: math.log(0.55), 2: math.log(0.02), 3: math.log(0.03)}
    # Set the truncation threshold to be 0.1
    privacy_loss_distribution_pessimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower,
            log_pmf_upper,
            log_mass_truncation_bound=math.log(0.1)))
    privacy_loss_distribution_optimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower,
            log_pmf_upper,
            log_mass_truncation_bound=math.log(0.1),
            pessimistic_estimate=False))

    # For the outcomes 2 and 3, the probability mass in mu_upper are below the
    # threshold. Hence, they should be discarded, resulting in the total
    # infinity_mass of 0.05 in the pessimistic case
    self.assertAlmostEqual(
        privacy_loss_distribution_pessimistic._basic_pld.infinity_mass, 0.05)

    # In the optimistic case, the infinity_mass should be zero.
    self.assertAlmostEqual(
        privacy_loss_distribution_optimistic._basic_pld.infinity_mass, 0)

    # The 10-hockey stick should be zero, but due to the mass truncation, the
    # output will be 0.05 in the pessimistic case.
    self.assertAlmostEqual(
        privacy_loss_distribution_pessimistic.get_delta_for_epsilon(10), 0.05)
    self.assertAlmostEqual(
        privacy_loss_distribution_optimistic.get_delta_for_epsilon(10), 0)

  def test_composition(self):
    # Test for composition of privacy loss distribution
    log_pmf_lower1 = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper1 = {1: math.log(0.5), 2: math.log(0.2), 4: math.log(0.3)}
    privacy_loss_distribution1 = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower1, log_pmf_upper1))
    log_pmf_lower2 = {1: math.log(0.4), 2: math.log(0.6)}
    log_pmf_upper2 = {2: math.log(0.7), 3: math.log(0.3)}
    privacy_loss_distribution2 = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower2, log_pmf_upper2))

    # Result from composing the above two privacy loss distributions
    result = privacy_loss_distribution1.compose(privacy_loss_distribution2)

    # The correct result
    log_pmf_lower_composed = {
        (1, 1): math.log(0.08),
        (1, 2): math.log(0.12),
        (2, 1): math.log(0.08),
        (2, 2): math.log(0.12),
        (3, 1): math.log(0.24),
        (3, 2): math.log(0.36)
    }
    log_pmf_upper_composed = {
        (1, 2): math.log(0.35),
        (1, 3): math.log(0.15),
        (2, 2): math.log(0.14),
        (2, 3): math.log(0.06),
        (4, 2): math.log(0.21),
        (4, 3): math.log(0.09)
    }
    expected_result = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower_composed, log_pmf_upper_composed))

    # Check that the result is as expected. Note that we cannot check that the
    # rounded_down_probability_mass_function and
    # rounded_up_probability_mass_function of the two distributions are equal
    # directly because the rounding might cause off-by-one error in index.
    self.assertAlmostEqual(
        expected_result._basic_pld.value_discretization_interval,
        result._basic_pld.value_discretization_interval)
    self.assertAlmostEqual(
        expected_result._basic_pld.infinity_mass,
        result._basic_pld.infinity_mass)
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
    self.assertAlmostEqual(pld_composed._basic_pld.infinity_mass, 0.211)
    test_util.assert_dictionary_almost_equal(
        self, {
            1: 0.13,
            2: 0.45,
            3: 0.20,
            4: 0.02,
        }, pld_composed._basic_pld.rounded_probability_mass_function)

  def test_composition_different_estimate_types(self):
    pld1 = privacy_loss_distribution.from_laplace_mechanism(
        1, pessimistic_estimate=True)
    pld2 = privacy_loss_distribution.from_laplace_mechanism(
        1, pessimistic_estimate=False)
    with self.assertRaises(ValueError):
      pld1.validate_composable(pld2)
    with self.assertRaises(ValueError):
      pld1.compose(pld2)
    with self.assertRaises(ValueError):
      pld1.get_delta_for_epsilon_for_composed_pld(pld2, 0.1)

  def test_composition_different_discretization(self):
    pld1 = privacy_loss_distribution.PrivacyLossDistribution(
        {1: 0.5, -1: 0.5}, 1, 0, True)
    pld2 = privacy_loss_distribution.PrivacyLossDistribution(
        {2: 0.5, -2: 0.5}, 0.5, 0, True)

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
    log_pmf_lower = {
        1: math.log(0.2),
        2: math.log(0.2),
        3: math.log(0.6)
    }
    log_pmf_upper = {
        1: math.log(0.5),
        2: math.log(0.2),
        4: math.log(0.3)
    }
    pld = privacy_loss_distribution.from_two_probability_mass_functions(
        log_pmf_lower, log_pmf_upper)
    result = pld.self_compose(3)

    expected_log_pmf_lower = {}
    for i, vi in log_pmf_lower.items():
      for j, vj in log_pmf_lower.items():
        for k, vk in log_pmf_lower.items():
          expected_log_pmf_lower[(i, j, k)] = vi + vj + vk
    expected_log_pmf_upper = {}
    for i, vi in log_pmf_upper.items():
      for j, vj in log_pmf_upper.items():
        for k, vk in log_pmf_upper.items():
          expected_log_pmf_upper[(i, j, k)] = vi + vj + vk
    expected_result = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            expected_log_pmf_lower, expected_log_pmf_upper))

    self.assertAlmostEqual(
        expected_result._basic_pld.value_discretization_interval,
        result._basic_pld.value_discretization_interval)
    self.assertAlmostEqual(expected_result._basic_pld.infinity_mass,
                           result._basic_pld.infinity_mass)
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0),
        result.get_delta_for_epsilon(0))
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0.5),
        result.get_delta_for_epsilon(0.5))
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(-0.2),
        result.get_delta_for_epsilon(-0.2))

  def test_self_composition_truncation(self):
    # Use Gaussian mechanism because it has closed form formula even afer
    # composition. For the setting of parameter below, the privacy loss after
    # composition should be the same as privacy loss with standard_deviation =
    # sensitivity.
    standard_deviation = 20
    num_composition = standard_deviation * standard_deviation
    pld = privacy_loss_distribution.from_gaussian_mechanism(
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
  }, 0.2), (0, 0.1, 1, {
      0: 0.9
  }, 0.1))
  def test_from_privacy_parameters(self, epsilon, delta,
                                   value_discretization_interval,
                                   expected_rounded_probability_mass_function,
                                   expected_infinity_mass):
    pld = privacy_loss_distribution.from_privacy_parameters(
        common.DifferentialPrivacyParameters(epsilon, delta),
        value_discretization_interval=value_discretization_interval)
    self.assertAlmostEqual(expected_infinity_mass, pld._basic_pld.infinity_mass)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld._basic_pld.rounded_probability_mass_function)

  @parameterized.parameters((stats.norm.cdf, True, 1, 1e-2, {
      1: 0.34134474,
      2: 0.13590512,
      3: 0.02140023,
      0: 0.34134474,
      -1: 0.13590512,
      -2: 0.02140023
  }, 1e-2), (stats.norm.cdf, False, 1, 1e-2, {
      0: 0.34134474,
      1: 0.13590512,
      2: 0.02140023,
      -1: 0.34134474,
      -2: 0.13590512,
      -3: 0.02140023
  }, 0), (lambda x: stats.norm.cdf(x, scale=2), True, 2, 1e-2, {
      1: 0.34134474,
      2: 0.13590512,
      3: 0.02140023,
      0: 0.34134474,
      -1: 0.13590512,
      -2: 0.02140023
  }, 1e-2), (stats.norm.cdf, True, 2, 1e-4, {
      1: 0.47724986,
      2: 0.02271846,
      0: 0.47724986,
      -1: 0.02271846
  }, 1e-4))
  def test_create_from_cdf(self, cdf, pessimistic_estimate,
                           value_discretization_interval, tail_mass_truncation,
                           expected_rounded_probability_mass_function,
                           expected_infinity_mass):
    pld = privacy_loss_distribution.create_from_cdf(
        cdf,
        pessimistic_estimate=pessimistic_estimate,
        value_discretization_interval=value_discretization_interval,
        tail_mass_truncation=tail_mass_truncation)
    self.assertEqual(pld._basic_pld.pessimistic_estimate, pessimistic_estimate)
    self.assertAlmostEqual(pld._basic_pld.value_discretization_interval,
                           value_discretization_interval)
    self.assertAlmostEqual(pld._basic_pld.infinity_mass, expected_infinity_mass)
    test_util.assert_dictionary_almost_equal(
        self, pld._basic_pld.rounded_probability_mass_function,
        expected_rounded_probability_mass_function)


class AddRemovePrivacyLossDistributionTest(parameterized.TestCase):

  def test_init_errors(self):
    rounded_pmf = {1: 0.5, -1: 0.5}
    value_discretization_interval = 1
    infinity_mass = 0
    pessimistic_estimate = True
    with self.assertRaises(ValueError):
      privacy_loss_distribution.PrivacyLossDistribution(
          rounded_probability_mass_function=rounded_pmf,
          value_discretization_interval=value_discretization_interval,
          infinity_mass=infinity_mass,
          pessimistic_estimate=pessimistic_estimate,
          rounded_probability_mass_function_add=rounded_pmf,
          infinity_mass_add=infinity_mass,
          symmetric=True)
      privacy_loss_distribution.PrivacyLossDistribution(
          rounded_probability_mass_function=rounded_pmf,
          value_discretization_interval=value_discretization_interval,
          infinity_mass=infinity_mass,
          pessimistic_estimate=pessimistic_estimate,
          rounded_probability_mass_function_add=None,
          infinity_mass_add=None,
          symmetric=False)

  def test_value_errors(self):
    # Test ValueError for different estimation type
    basic_pld_remove = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            1: 0.5,
            -1: 0.5
        }, 1, 0, True)
    basic_pld_add = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            1: 0.5,
            -1: 0.5
        }, 1, 0, False)
    with self.assertRaises(ValueError):
      (privacy_loss_distribution.PrivacyLossDistribution
       .from_add_remove_basic_plds(basic_pld_remove, basic_pld_add))

    # Test ValueError for different value_discretization_interval
    basic_pld_remove = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            2: 0.5,
            -2: 0.5
        }, 0.5, 0, True)
    basic_pld_add = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            1: 0.5,
            -1: 0.5
        }, 1, 0, True)
    with self.assertRaises(ValueError):
      (privacy_loss_distribution.PrivacyLossDistribution
       .from_add_remove_basic_plds(basic_pld_remove, basic_pld_add))

  def test_hockey_stick_basic(self):
    # Basic hockey stick divergence computation test
    log_pmf_lower = {1: math.log(0.5), 2: math.log(0.5)}
    log_pmf_upper = {1: math.log(0.6), 2: math.log(0.4)}
    basic_pld_remove_pessimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper, pessimistic_estimate=True)._basic_pld)
    basic_pld_add_pessimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper, log_pmf_lower, pessimistic_estimate=True)._basic_pld)
    pld_pessimistic = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove_pessimistic,
                                    basic_pld_add_pessimistic))

    basic_pld_remove_optimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper,
            pessimistic_estimate=False)._basic_pld)
    basic_pld_add_optimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper, log_pmf_lower,
            pessimistic_estimate=False)._basic_pld)
    pld_optimistic = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove_optimistic,
                                    basic_pld_add_optimistic))

    # 0-hockey stick divergence is 0.1 (for basic_pld_remove & basic_pld_add)
    # When using pessimistic estimate, the output should be in [0.1, 0.1+1e-4]
    self.assertTrue(
        0.1 <= pld_pessimistic.get_delta_for_epsilon(0.0) <= 0.1 + 1e-4)
    # When using optimistic estimate, the output should be in [0.1 - 1e-4, 0.1]
    self.assertTrue(
        0.1 - 1e-4 <= pld_optimistic.get_delta_for_epsilon(0.0) <= 0.1)

    # math.log(1.1)-hockey stick divergence is 0.06 (for basic_pld_add)
    # When using pessimistic estimate, the output should be in [0.06, 0.06+1e-4]
    self.assertTrue(0.06 <= pld_pessimistic
                    .get_delta_for_epsilon(math.log(1.1)) <= 0.06 + 1e-4)
    # When using optimistic estimate, the output should be in [0.06-1e-4, 0.06]
    self.assertTrue(0.06 - 1e-4 <= pld_optimistic
                    .get_delta_for_epsilon(math.log(1.1)) <= 0.06)

    # math.log(0.9)-hockey stick divergence is 0.15 (for basic_pld_remove)
    # When using pessimistic estimate, the output should be in [0.15, 0.15+1e-4]
    self.assertTrue(0.15 <= pld_pessimistic
                    .get_delta_for_epsilon(math.log(0.9)) <= 0.15 + 1e-4)
    # When using optimistic estimate, the output should be in [0.15-1e-4, 0.15]
    self.assertTrue(0.15 - 1e-4 <= pld_optimistic
                    .get_delta_for_epsilon(math.log(0.9)) <= 0.15)

    self.assertFalse(pld_pessimistic._symmetric)
    self.assertFalse(pld_optimistic._symmetric)

  def test_hockey_stick_unequal_support(self):
    # Hockey stick divergence computation test when the two distributions have
    # differenet supports
    log_pmf_lower = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper = {1: math.log(0.5), 2: math.log(0.4), 4: math.log(0.1)}
    basic_pld_remove_pessimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper, pessimistic_estimate=True)._basic_pld)
    basic_pld_add_pessimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper, log_pmf_lower, pessimistic_estimate=True)._basic_pld)
    pld_pessimistic = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove_pessimistic,
                                    basic_pld_add_pessimistic))

    basic_pld_remove_optimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper,
            pessimistic_estimate=False)._basic_pld)
    basic_pld_add_optimistic = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper, log_pmf_lower,
            pessimistic_estimate=False)._basic_pld)
    pld_optimistic = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove_optimistic,
                                    basic_pld_add_optimistic))

    # Here 4 appears as an outcome of only mu_upper and hence should be included
    # in the infinity_mass variable of basic_pld_remove.
    self.assertAlmostEqual(pld_pessimistic._basic_pld_remove.infinity_mass,
                           0.1)
    self.assertAlmostEqual(pld_optimistic._basic_pld_remove.infinity_mass,
                           0.1)
    # Here 3 appears as an outcome of only mu_lower and hence should be included
    # in the infinity_mass variable of basic_pld_add.
    self.assertAlmostEqual(pld_pessimistic._basic_pld_add.infinity_mass,
                           0.6)
    self.assertAlmostEqual(pld_optimistic._basic_pld_add.infinity_mass,
                           0.6)

    # 0-hockey stick divergence is 0.6 (for basic_pld_remove & basic_pld_add)
    # When using pessimistic estimate, the output should be in [0.6, 0.6+1e-4]
    self.assertTrue(0.6 <= pld_pessimistic
                    .get_delta_for_epsilon(0.0) <= 0.6 + 1e-4)
    # When using optimistic estimate, the output should lie in [0.6 - 1e-4, 0.6]
    self.assertTrue(0.6 - 1e-4 <= pld_optimistic
                    .get_delta_for_epsilon(0.0) <= 0.6)

    # math.log(1.1)-hockey stick divergence is 0.6 (for basic_pld_add)
    # When using pessimistic estimate, the output should be in [0.6, 0.6 + 1e-4]
    self.assertTrue(0.6 <= pld_pessimistic
                    .get_delta_for_epsilon(math.log(1.1)) <= 0.6 + 1e-4)
    # When using optimistic estimate, the output should lie in [0.6 - 1e-4, 0.6]
    self.assertTrue(0.6 - 1e-4 <= pld_optimistic
                    .get_delta_for_epsilon(math.log(1.1)) <= 0.6)

    # math.log(0.9)-hockey stick divergence is 0.64 (for basic_pld_remove)
    # When using pessimistic estimate, the output should be
    # in [0.64, 0.64 + 1e-4]
    self.assertTrue(0.64 <= pld_pessimistic
                    .get_delta_for_epsilon(math.log(0.9)) <= 0.64 + 1e-4)
    # When using optimistic estimate, the output should lie in
    # [0.64 - 1e-4, 0.64]
    self.assertTrue(0.64 - 1e-4 <= pld_optimistic
                    .get_delta_for_epsilon(math.log(0.9)) <= 0.64)

  @parameterized.parameters(
      ({4: 0.2, 2: 0.7}, 0.1,
       {4: 0.1, 2: 0.8}, 0.1,
       0.5, 0.5, 0.56358432),
      ({4: 0.1, 2: 0.8}, 0.1,
       {4: 0.2, 2: 0.7}, 0.1,
       0.5, 0.2, 1.30685282),
      ({1: 0.3, -1: 0.8}, 0.1,
       {1: 0.2, -1: 0.7}, 0.1,
       1, 0.4, 0),
      ({1: 0.5}, 0.5,
       {1: 0.8}, 0.2,
       1, 0.4, math.inf),
      ({-1: 0.1}, 0,
       {-1: 0.2}, 0,
       1, 0, 0),
      # Test resilience against overflow
      ({5000: 1}, 0,
       {5000: 0.9}, 0.1,
       1, 0.1, 5000),
      ({5000: 0.2, 4000: 0.1, 3000: 0.7}, 0.1,
       {5000: 0.1, 4000: 0.1, 3000: 0.9}, 0.0,
       1, 0.4, 4000))
  def test_get_epsilon_for_delta(self, remove_rounded_pmf, remove_infinity_mass,
                                 add_rounded_pmf, add_infinity_mass,
                                 value_discretization_interval,
                                 delta, expected_epsilon):
    basic_pld_remove = privacy_loss_distribution.BasicPrivacyLossDistribution(
        remove_rounded_pmf, value_discretization_interval, remove_infinity_mass)
    basic_pld_add = privacy_loss_distribution.BasicPrivacyLossDistribution(
        add_rounded_pmf, value_discretization_interval, add_infinity_mass)
    pld = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove, basic_pld_add))
    self.assertAlmostEqual(expected_epsilon, pld.get_epsilon_for_delta(delta))

  def test_composition_different_estimate_types_error(self):
    basic_pld1 = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            1: 0.5,
            -1: 0.5
        }, 1, 0, True)
    pld1 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld1, basic_pld1))
    basic_pld2 = (
        privacy_loss_distribution.BasicPrivacyLossDistribution({
            1: 0.5,
            -1: 0.5
        }, 1, 0, False))
    pld2 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld2, basic_pld2))

    with self.assertRaises(ValueError):
      pld1.validate_composable(pld2)
    with self.assertRaises(ValueError):
      pld1.compose(pld2)
    with self.assertRaises(ValueError):
      pld1.get_delta_for_epsilon_for_composed_pld(pld2, 0.1)

  def test_composition_different_discretization_error(self):
    basic_pld1 = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            1: 0.5,
            -1: 0.5
        }, 1, 0, True)
    pld1 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld1, basic_pld1))
    basic_pld2 = (
        privacy_loss_distribution.BasicPrivacyLossDistribution({
            2: 0.5,
            -2: 0.5
        }, 0.5, 0, True))
    pld2 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld2, basic_pld2))

    with self.assertRaises(ValueError):
      pld1.validate_composable(pld2)
    with self.assertRaises(ValueError):
      pld1.compose(pld2)
    with self.assertRaises(ValueError):
      pld1.get_delta_for_epsilon_for_composed_pld(pld2, 0.1)

  def test_composition(self):
    # Test for composition of privacy loss distribution
    log_pmf_lower1 = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper1 = {1: math.log(0.5), 2: math.log(0.2), 4: math.log(0.3)}
    basic_pld_remove1 = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower1, log_pmf_upper1)._basic_pld)
    basic_pld_add1 = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper1, log_pmf_lower1)._basic_pld)
    pld1 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove1, basic_pld_add1))

    log_pmf_lower2 = {1: math.log(0.4), 2: math.log(0.6)}
    log_pmf_upper2 = {2: math.log(0.7), 3: math.log(0.3)}
    basic_pld_remove2 = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower2, log_pmf_upper2)._basic_pld)
    basic_pld_add2 = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper2, log_pmf_lower2)._basic_pld)
    pld2 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove2, basic_pld_add2))

    # Result from composing the above two privacy loss distributions
    result = pld1.compose(pld2)

    # The correct result
    log_pmf_lower_composed = {
        (1, 1): math.log(0.08),
        (1, 2): math.log(0.12),
        (2, 1): math.log(0.08),
        (2, 2): math.log(0.12),
        (3, 1): math.log(0.24),
        (3, 2): math.log(0.36)
    }
    log_pmf_upper_composed = {
        (1, 2): math.log(0.35),
        (1, 3): math.log(0.15),
        (2, 2): math.log(0.14),
        (2, 3): math.log(0.06),
        (4, 2): math.log(0.21),
        (4, 3): math.log(0.09)
    }
    basic_pld_remove_expected = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower_composed, log_pmf_upper_composed)._basic_pld)
    basic_pld_add_expected = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper_composed, log_pmf_lower_composed)._basic_pld)
    expected_result = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove_expected,
                                    basic_pld_add_expected))

    # Check that the result is as expected. Note that we cannot check that the
    # rounded_down_probability_mass_function and
    # rounded_up_probability_mass_function of the two distributions are equal
    # directly because the rounding might cause off-by-one error in index.
    self.assertAlmostEqual(
        expected_result._basic_pld_remove.value_discretization_interval,
        result._basic_pld_remove.value_discretization_interval)
    self.assertAlmostEqual(
        expected_result._basic_pld_add.value_discretization_interval,
        result._basic_pld_add.value_discretization_interval)
    self.assertAlmostEqual(expected_result._basic_pld_remove.infinity_mass,
                           result._basic_pld_remove.infinity_mass)
    self.assertAlmostEqual(expected_result._basic_pld_add.infinity_mass,
                           result._basic_pld_add.infinity_mass)
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0),
        result.get_delta_for_epsilon(0))
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0.5),
        result.get_delta_for_epsilon(0.5))
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(-0.5),
        result.get_delta_for_epsilon(-0.5))

  def test_composition_asymmetric_with_symmetric(self):
    # Test for composition of privacy loss distribution
    log_pmf_lower1 = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper1 = {1: math.log(0.5), 2: math.log(0.2), 4: math.log(0.3)}
    basic_pld_remove1 = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower1, log_pmf_upper1)._basic_pld)
    basic_pld_add1 = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper1, log_pmf_lower1)._basic_pld)
    pld1 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove1, basic_pld_add1))

    log_pmf_lower2 = {1: math.log(0.4), 2: math.log(0.6)}
    log_pmf_upper2 = {2: math.log(0.7), 3: math.log(0.3)}
    pld2 = privacy_loss_distribution.from_two_probability_mass_functions(
        log_pmf_lower2, log_pmf_upper2)

    # Result from composing the above two privacy loss distributions
    result12 = pld1.compose(pld2)
    result21 = pld2.compose(pld1)

    # The correct result
    log_pmf_lower1_lower2_composed = {
        (1, 1): math.log(0.08),
        (1, 2): math.log(0.12),
        (2, 1): math.log(0.08),
        (2, 2): math.log(0.12),
        (3, 1): math.log(0.24),
        (3, 2): math.log(0.36)
    }
    log_pmf_upper1_lower2_composed = {
        (1, 1): math.log(0.20),
        (1, 2): math.log(0.30),
        (2, 1): math.log(0.08),
        (2, 2): math.log(0.12),
        (4, 1): math.log(0.12),
        (4, 2): math.log(0.18)
    }
    log_pmf_lower1_upper2_composed = {
        (1, 2): math.log(0.14),
        (1, 3): math.log(0.06),
        (2, 2): math.log(0.14),
        (2, 3): math.log(0.06),
        (3, 2): math.log(0.42),
        (3, 3): math.log(0.18)
    }
    log_pmf_upper1_upper2_composed = {
        (1, 2): math.log(0.35),
        (1, 3): math.log(0.15),
        (2, 2): math.log(0.14),
        (2, 3): math.log(0.06),
        (4, 2): math.log(0.21),
        (4, 3): math.log(0.09)
    }
    basic_pld_remove_expected = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower1_lower2_composed,
            log_pmf_upper1_upper2_composed)._basic_pld)
    basic_pld_add_expected = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper1_lower2_composed,
            log_pmf_lower1_upper2_composed)._basic_pld)
    expected_result = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove_expected,
                                    basic_pld_add_expected))

    # Check that the result is as expected. Note that we cannot check that the
    # rounded_down_probability_mass_function and
    # rounded_up_probability_mass_function of the two distributions are equal
    # directly because the rounding might cause off-by-one error in index.
    for result in [result12, result21]:
      self.assertAlmostEqual(
          expected_result._basic_pld_remove.value_discretization_interval,
          result._basic_pld_remove.value_discretization_interval)
      self.assertAlmostEqual(
          expected_result._basic_pld_add.value_discretization_interval,
          result._basic_pld_add.value_discretization_interval)
      self.assertAlmostEqual(expected_result._basic_pld_remove.infinity_mass,
                             result._basic_pld_remove.infinity_mass)
      self.assertAlmostEqual(expected_result._basic_pld_add.infinity_mass,
                             result._basic_pld_add.infinity_mass)
      self.assertAlmostEqual(
          expected_result.get_delta_for_epsilon(0),
          result.get_delta_for_epsilon(0))
      self.assertAlmostEqual(
          expected_result.get_delta_for_epsilon(0.5),
          result.get_delta_for_epsilon(0.5))
      self.assertAlmostEqual(
          expected_result.get_delta_for_epsilon(-0.5),
          result.get_delta_for_epsilon(-0.5))

  def test_compose_and_get_epsilon_for_delta(self):
    basic_pld_remove1 = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {0: 0.1, 1: 0.7, 2: 0.1}, 0.4, 0.1)
    basic_pld_add1 = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            0: 0.2,
            1: 0.6,
            2: 0.1
        }, 0.4, 0.1)
    pld1 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove1, basic_pld_add1))
    basic_pld_remove2 = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {1: 0.1, 2: 0.6, 3: 0.25}, 0.4, 0.05)
    basic_pld_add2 = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            1: 0.2,
            2: 0.5,
            3: 0.25
        }, 0.4, 0.05)
    pld2 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove2, basic_pld_add2))
    self.assertAlmostEqual(
        0.29560003, pld1.get_delta_for_epsilon_for_composed_pld(pld2, 1.1))

  def test_compose_asymmetric_with_symmetric_and_get_delta_for_epsilon(self):
    basic_pld_remove1 = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {0: 0.1, 1: 0.7, 2: 0.1}, 0.4, 0.1)
    basic_pld_add1 = privacy_loss_distribution.BasicPrivacyLossDistribution(
        {
            0: 0.2,
            1: 0.6,
            2: 0.1
        }, 0.4, 0.1)
    pld1 = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove1, basic_pld_add1))
    pld2 = privacy_loss_distribution.PrivacyLossDistribution(
        {1: 0.1, 2: 0.6, 3: 0.25}, 0.4, 0.05)
    self.assertAlmostEqual(
        0.29560003,
        pld1.get_delta_for_epsilon_for_composed_pld(pld2, 1.1))
    self.assertAlmostEqual(
        0.29560003,
        pld2.get_delta_for_epsilon_for_composed_pld(pld1, 1.1))

  def test_self_composition(self):
    log_pmf_lower = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper = {1: math.log(0.5), 2: math.log(0.2), 4: math.log(0.3)}
    basic_pld_remove = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper)._basic_pld)
    basic_pld_add = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper, log_pmf_lower)._basic_pld)
    pld = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(basic_pld_remove, basic_pld_add))
    result = pld.self_compose(3)

    expected_log_pmf_lower = {}
    for i, vi in log_pmf_lower.items():
      for j, vj in log_pmf_lower.items():
        for k, vk in log_pmf_lower.items():
          expected_log_pmf_lower[(i, j, k)] = vi + vj + vk
    expected_log_pmf_upper = {}
    for i, vi in log_pmf_upper.items():
      for j, vj in log_pmf_upper.items():
        for k, vk in log_pmf_upper.items():
          expected_log_pmf_upper[(i, j, k)] = vi + vj + vk
    expected_basic_pld_remove = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            expected_log_pmf_lower, expected_log_pmf_upper)._basic_pld)
    expected_basic_pld_add = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            expected_log_pmf_upper, expected_log_pmf_lower)._basic_pld)
    expected_result = (
        privacy_loss_distribution.PrivacyLossDistribution
        .from_add_remove_basic_plds(expected_basic_pld_remove,
                                    expected_basic_pld_add))

    self.assertAlmostEqual(
        expected_result._basic_pld_remove.value_discretization_interval,
        result._basic_pld_remove.value_discretization_interval)
    self.assertAlmostEqual(expected_result._basic_pld_remove.infinity_mass,
                           result._basic_pld_remove.infinity_mass)
    self.assertAlmostEqual(
        expected_result._basic_pld_add.value_discretization_interval,
        result._basic_pld_add.value_discretization_interval)
    self.assertAlmostEqual(expected_result._basic_pld_add.infinity_mass,
                           result._basic_pld_add.infinity_mass)
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0),
        result.get_delta_for_epsilon(0))
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(0.5),
        result.get_delta_for_epsilon(0.5))
    self.assertAlmostEqual(
        expected_result.get_delta_for_epsilon(-0.2),
        result.get_delta_for_epsilon(-0.2))


class LaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1.0, 1.0, -0.1), (1.0, 1.0, 1.1), (1.0, 1.0, 0.0),
                            (-0.1, 1.0, 1.0), (0.0, 1.0, 1.0), (1.0, -1.0, 1.0),
                            (1.0, 0.0, 1.0))
  def test_laplace_value_errors(self, parameter, sensitivity, sampling_prob):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.from_laplace_mechanism(
          parameter, sensitivity=sensitivity, value_discretization_interval=1,
          sampling_prob=sampling_prob)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1.0, 1.0, {
          1: 0.69673467,
          0: 0.11932561,
          -1: 0.18393972
      }),
      (3.0, 3.0, 1.0, {
          1: 0.69673467,
          0: 0.11932561,
          -1: 0.18393972
      }),
      (1.0, 2.0, 1.0, {
          2: 0.69673467,
          1: 0.11932561,
          0: 0.07237464,
          -1: 0.04389744,
          -2: 0.06766764
      }),
      (2.0, 4.0, 1.0, {
          2: 0.69673467,
          1: 0.11932561,
          0: 0.07237464,
          -1: 0.04389744,
          -2: 0.06766764
      }),
      # Tests with sampling_prob < 1
      (1.0, 1.0, 0.8, {
          1: 0.69673467,
          0: 0.30326533
      }, {
          1: 0.6180408,
          0: 0.3819592
      }),
      (3.0, 3.0, 0.5, {
          1: 0.69673467,
          0: 0.30326533
      }, {
          1: 0.5,
          0: 0.5
      }),
      (1.0, 2.0, 0.7, {
          1: 0.81606028,
          0: 0.08497712,
          -1: 0.09896260
      }, {
          2: 0.49036933,
          1: 0.13605478,
          0: 0.37357589
      }),
      (2.0, 4.0, 0.3, {
          1: 0.81606028,
          0: 0.11302356,
          -1: 0.07091617
      }, {
          2: 0.20651251,
          1: 0.16706338,
          0: 0.62642411
      }))
  def test_laplace_varying_parameter_and_sensitivity(
      self, parameter, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_rounded_pmf_remove=None):
    """Verifies correctness of pessimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_laplace_mechanism(
        parameter, sensitivity=sensitivity, value_discretization_interval=1,
        sampling_prob=sampling_prob)

    if expected_rounded_pmf_remove is None:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld.rounded_probability_mass_function)
      self.assertTrue(pld._symmetric)
    else:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_remove,
          pld._basic_pld_remove.rounded_probability_mass_function)
      self.assertFalse(pld._symmetric)

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
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_laplace_mechanism(
        1, value_discretization_interval=value_discretization_interval)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld._basic_pld.rounded_probability_mass_function)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1.0, 1.0, {
          1: 0.5,
          0: 0.19673467,
          -1: 0.30326533
      }),
      (1.0, 2.0, 1.0, {
          2: 0.5,
          1: 0.19673467,
          0: 0.11932561,
          -1: 0.07237464,
          -2: 0.11156508
      }),
      # Tests with sampling_prob < 1
      (1.0, 1.0, 0.8, {
          0: 0.69673467,
          -1: 0.30326533
      }, {
          0: 0.6180408,
          -1: 0.3819592
      }),
      (3.0, 3.0, 0.5, {
          0: 0.69673467,
          -1: 0.30326533
      }, {
          0: 0.5,
          -1: 0.5
      }),
      (1.0, 2.0, 0.7, {
          0: 0.81606028,
          -1: 0.08497712,
          -2: 0.09896260
      }, {
          1: 0.49036933,
          0: 0.13605478,
          -1: 0.37357589
      }),
      (2.0, 4.0, 0.3, {
          0: 0.81606028,
          -1: 0.11302356,
          -2: 0.07091617
      }, {
          1: 0.20651251,
          0: 0.16706338,
          -1: 0.62642411
      }))
  def test_laplace_varying_parameter_and_sensitivity_optimistic(
      self, parameter, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_rounded_pmf_remove=None):
    """Verifies correctness of optimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_laplace_mechanism(
        parameter=parameter,
        sensitivity=sensitivity,
        pessimistic_estimate=False,
        value_discretization_interval=1,
        sampling_prob=sampling_prob)

    if expected_rounded_pmf_remove is None:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      self.assertTrue(pld._symmetric)
    else:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_remove,
          pld._basic_pld_remove.rounded_probability_mass_function)
      self.assertFalse(pld._symmetric)


class GaussianPrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1.0, 1.0, -0.1), (1.0, 1.0, 1.1), (1.0, 1.0, 0.0),
                            (-0.1, 1.0, 1.0), (0.0, 1.0, 1.0), (1.0, -1.0, 1.0),
                            (1.0, 0.0, 1.0))
  def test_gaussian_value_errors(self, standard_deviation, sensitivity,
                                 sampling_prob):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.from_gaussian_mechanism(
          standard_deviation,
          sensitivity=sensitivity,
          value_discretization_interval=1,
          sampling_prob=sampling_prob)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1.0, 1.0, {
          2: 0.12447741,
          1: 0.38292492,
          0: 0.24173034,
          -1: 0.0668072
      }),
      (5.0, 5.0, 1.0, {
          2: 0.12447741,
          1: 0.38292492,
          0: 0.24173034,
          -1: 0.0668072
      }),
      (1.0, 2.0, 1.0, {
          -3: 0.00620967,
          -2: 0.01654047,
          -1: 0.04405707,
          0: 0.09184805,
          1: 0.14988228,
          2: 0.19146246,
          3: 0.19146246,
          4: 0.12447741
      }),
      (3.0, 6.0, 1.0, {
          -3: 0.00620967,
          -2: 0.01654047,
          -1: 0.04405707,
          0: 0.09184805,
          1: 0.14988228,
          2: 0.19146246,
          3: 0.19146246,
          4: 0.12447741
      }),
      # Tests with sampling_prob < 1
      (1.0, 1.0, 0.8, {
          1: 0.50740234,
          0: 0.25872977,
          -1: 0.04980776
      }, {
          2: 0.06409531,
          1: 0.39779076,
          0: 0.38512252
      }),
      (5.0, 5.0, 0.6, {
          1: 0.50740234,
          0: 0.27649963,
          -1: 0.03203791
      }, {
          2: 0.00921465,
          1: 0.40715514,
          0: 0.46170751
      }),
      (1.0, 2.0, 0.4, {
          1: 0.65728462,
          0: 0.12528727,
          -1: 0.02551767,
          -2: 0.00785031
      }, {
          3: 0.06547773,
          2: 0.10625501,
          1: 0.18525477,
          0: 0.56826895
      }),
      (3.0, 6.0, 0.2, {
          1: 0.65728462,
          0: 0.14208735,
          -1: 0.01356463,
          -2: 0.00300327
      }, {
          3: 0.00957871,
          2: 0.05499325,
          1: 0.19231652,
          0: 0.70480685
      }))
  def test_gaussian_varying_standard_deviation_and_sensitivity(
      self, standard_deviation, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_rounded_pmf_remove=None):
    """Verifies correctness of pessimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation,
        sensitivity=sensitivity,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=1,
        sampling_prob=sampling_prob)

    if expected_rounded_pmf_remove is None:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld.rounded_probability_mass_function)
      test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                            pld._basic_pld.infinity_mass)
      self.assertTrue(pld._symmetric)
    else:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                            pld._basic_pld_add.infinity_mass)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_remove,
          pld._basic_pld_remove.rounded_probability_mass_function)
      test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                            pld._basic_pld_remove.infinity_mass)
      self.assertFalse(pld._symmetric)

  @parameterized.parameters((0.5, {
      3: 0.12447741,
      2: 0.19146246,
      1: 0.19146246,
      0: 0.14988228,
      -1: 0.09184805,
      -2: 0.06680720
  }), (0.3, {
      5: 0.05790353,
      4: 0.10261461,
      3: 0.11559390,
      2: 0.11908755,
      1: 0.11220275,
      0: 0.09668214,
      -1: 0.07618934,
      -2: 0.0549094,
      -3: 0.0361912,
      -4: 0.04456546
  }))
  def test_gaussian_discretization(self, value_discretization_interval,
                                   expected_rounded_probability_mass_function):
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        1,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=value_discretization_interval)
    test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                          pld._basic_pld.infinity_mass)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld._basic_pld.rounded_probability_mass_function)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1.0, 1.0, {
          1: 0.30853754,
          0: 0.38292492,
          -1: 0.24173034,
          -2: 0.03809064
      }),
      (5.0, 5.0, 1.0, {
          1: 0.30853754,
          0: 0.38292492,
          -1: 0.24173034,
          -2: 0.03809064
      }),
      (1.0, 2.0, 1.0, {
          3: 0.30853754,
          2: 0.19146246,
          1: 0.19146246,
          0: 0.14988228,
          -1: 0.09184805,
          -2: 0.04405707,
          -3: 0.01654047,
          -4: 0.00434385
      }),
      (3.0, 6.0, 1.0, {
          3: 0.30853754,
          2: 0.19146246,
          1: 0.19146246,
          0: 0.14988228,
          -1: 0.09184805,
          -2: 0.04405707,
          -3: 0.01654047,
          -4: 0.00434385
      }),
      # Tests with sampling_prob < 1
      (1.0, 1.0, 0.8, {
          0: 0.69146246,
          -1: 0.25872977,
          -2: 0.0210912
      }, {
          1: 0.21708672,
          0: 0.39779076,
          -1: 0.32533725
      }),
      (5.0, 5.0, 0.6, {
          0: 0.69146246,
          -1: 0.27649963,
          -2: 0.00332135
      }, {
          1: 0.13113735,
          0: 0.40715514,
          -1: 0.37085352
      }),
      (1.0, 2.0, 0.4, {
          0: 0.84134475,
          -1: 0.12528727,
          -2: 0.02551767,
          -3: 0.0059845
      }, {
          2: 0.14022127,
          1: 0.10625501,
          0: 0.18525477,
          -1: 0.45708655
      }),
      (3.0, 6.0, 0.2, {
          0: 0.84134475,
          -1: 0.14208735,
          -2: 0.01356463,
          -3: 0.00113746
      }, {
          2: 0.04788338,
          1: 0.05499325,
          0: 0.19231652,
          -1: 0.55718558
      }))
  def test_gaussian_varying_standard_deviation_and_sensitivity_optimistic(
      self, standard_deviation, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_rounded_pmf_remove=None):
    """Verifies correctness of optimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation,
        sensitivity=sensitivity,
        pessimistic_estimate=False,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=1,
        sampling_prob=sampling_prob)

    if expected_rounded_pmf_remove is None:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld.rounded_probability_mass_function)
      test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                            pld._basic_pld.infinity_mass)
      self.assertTrue(pld._symmetric)
    else:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                            pld._basic_pld_add.infinity_mass)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_remove,
          pld._basic_pld_remove.rounded_probability_mass_function)
      test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                            pld._basic_pld_remove.infinity_mass)
      self.assertFalse(pld._symmetric)


class DiscreteLaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1.0, 1, -0.1), (1.0, 1, 1.1), (1.0, 1, 0.0),
                            (-0.1, 1, 1.0), (0.0, 1, 1.0), (1.0, -1, 1.0),
                            (1.0, 0, 1.0), (1.0, 0.5, 1.0), (1.0, 1.0, 1.0))
  def test_discrete_laplace_value_errors(self, parameter, sensitivity,
                                         sampling_prob):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.from_discrete_laplace_mechanism(
          parameter, sensitivity=sensitivity, value_discretization_interval=1,
          sampling_prob=sampling_prob)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1, 1, {
          1: 0.73105858,
          -1: 0.26894142
      }),
      (1.0, 2, 1, {
          2: 0.73105858,
          0: 0.17000340,
          -2: 0.09893802
      }),
      (0.8, 2, 1, {
          2: 0.68997448,
          0: 0.17072207,
          -1: 0.13930345
      }),
      (0.8, 3, 1, {
          3: 0.68997448,
          1: 0.17072207,
          0: 0.07671037,
          -2: 0.06259307
      }),
      # Tests with sampling_prob < 1
      (1.0, 1, 0.8, {
          1: 0.7310585786300049,
          0: 0.2689414213699951
      }, {
          1: 0.63863515,
          0: 0.36136485
      }),
      (1.0, 2, 0.5, {
          1.0: 0.7310585786300049,
          0.0: 0.17000340156854787,
          -1.0: 0.09893801980144723
      }, {
          2.0: 0.41499829921572606,
          1.0: 0.0,
          0.0: 0.5850017007842739
      }),
      (0.8, 2, 0.3, {
          1: 0.6899744811276125,
          0: 0.3100255188723875
      }, {
          1: 0.30450475600966753,
          0: 0.6954952439903325
      }),
      (0.8, 3, 0.2, {
          1: 0.8606965547551659,
          0: 0.07671037249501267,
          -1: 0.0625930727498214
      }, {
          2: 0.1880693544253796,
          1: 0.09551271272152079,
          0: 0.7164179328530996,
      }))
  def test_discrete_laplace_varying_parameter_and_sensitivity(
      self, parameter, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_rounded_pmf_remove=None):
    """Verifies correctness of pessimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_discrete_laplace_mechanism(
        parameter, sensitivity=sensitivity, value_discretization_interval=1,
        sampling_prob=sampling_prob)
    if expected_rounded_pmf_remove is None:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld.rounded_probability_mass_function)
      self.assertTrue(pld._symmetric)
    else:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_remove,
          pld._basic_pld_remove.rounded_probability_mass_function)
      self.assertFalse(pld._symmetric)

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
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_discrete_laplace_mechanism(
        1, value_discretization_interval=value_discretization_interval)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld._basic_pld.rounded_probability_mass_function)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1, 1, {
          1: 0.73105858,
          -1: 0.26894142
      }),
      (1.0, 2, 1, {
          2: 0.73105858,
          0: 0.17000340,
          -2: 0.09893802
      }),
      (0.8, 2, 1, {
          1: 0.68997448,
          0: 0.17072207,
          -2: 0.13930345
      }),
      (0.8, 3, 1, {
          2: 0.68997448,
          0: 0.17072207,
          -1: 0.07671037,
          -3: 0.06259307
      }),
      # Tests with sampling_prob < 1
      (1.0, 1, 0.8, {
          0: 0.7310585786300049,
          -1: 0.2689414213699951
      }, {
          0: 0.63863515,
          -1: 0.36136485
      }),
      (1.0, 2, 0.5, {
          0: 0.9010619801985528,
          -2: 0.09893801980144723,
      }, {
          1: 0.41499829921572606,
          0: 0.17000340156854787,
          -1: 0.41499829921572606
      }),
      (0.8, 2, 0.3, {
          0: 0.8606965547551659,
          -1: 0.13930344524483407
      }, {
          0: 0.47522682963722107,
          -1: 0.5247731703627789
      }),
      (0.8, 3, 0.2, {
          0: 0.8606965547551659,
          -1: 0.07671037249501267,
          -2: 0.0625930727498214
      }, {
          1: 0.1880693544253796,
          0: 0.09551271272152079,
          -1: 0.7164179328530996
      }))
  def test_discrete_laplace_varying_parameter_and_sensitivity_optimistic(
      self, parameter, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_rounded_pmf_remove=None):
    """Verifies correctness of optimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_discrete_laplace_mechanism(
        parameter, sensitivity=sensitivity, value_discretization_interval=1,
        pessimistic_estimate=False,
        sampling_prob=sampling_prob)
    if expected_rounded_pmf_remove is None:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld.rounded_probability_mass_function)
      self.assertTrue(pld._symmetric)
    else:
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_remove,
          pld._basic_pld_remove.rounded_probability_mass_function)
      self.assertFalse(pld._symmetric)


class DiscreteGaussianPrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1.0, 1, -0.1), (1.0, 1, 1.1), (1.0, 1, 0.0),
                            (-0.1, 1, 1.0), (0.0, 1, 1.0), (1.0, -1, 1.0),
                            (1.0, 0, 1.0), (1.0, 0.5, 1.0), (1.0, 1.0, 1.0))
  def test_discrete_gaussian_value_errors(self, sigma, sensitivity,
                                          sampling_prob):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.from_discrete_gaussian_mechanism(
          sigma, sensitivity=sensitivity, truncation_bound=1,
          sampling_prob=sampling_prob)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1, 1.0, {
          5000: 0.45186276,
          -5000: 0.27406862
      }, 0.27406862),
      (1.0, 2, 1.0, {
          0: 0.27406862
      }, 0.72593138),
      (3.0, 1, 1.0, {
          556: 0.34579116,
          -555: 0.32710442
      }, 0.32710442),
      # Tests with sampling_prob < 1
      (1.0, 1, 0.6, {
          -3287: 0.27406862,
          2693: 0.45186276,
          9163: 0.27406862
      }, 0.0, {
          3288: 0.3807451,
          -2692: 0.34518628,
          -9162: 0.10962745
      }, 0.16444117),
      (1.0, 2, 0.3, {
          0: 0.27406862,
          3567: 0.7259314,
      }, 0.0, {
          0: 0.27406862,
          -3566: 0.50815197,
      }, 0.2177794),
      (3.0, 1, 0.1, {
          -56: 0.32710442,
          55: 0.34579116,
          1054: 0.32710442
      }, 0.0, {
          57: 0.32897309,
          -54: 0.34392248,
          -1053: 0.29439398
      }, 0.03271044))
  def test_discrete_gaussian_varying_sigma_and_sensitivity(
      self, sigma, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_infinity_mass_add,
      expected_rounded_pmf_remove=None, expected_infinity_mass_remove=None):
    """Verifies correctness of pessimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_discrete_gaussian_mechanism(
        sigma, sensitivity=sensitivity, truncation_bound=1,
        sampling_prob=sampling_prob)
    if expected_rounded_pmf_remove is None:
      self.assertAlmostEqual(pld._basic_pld.infinity_mass,
                             expected_infinity_mass_add)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld.rounded_probability_mass_function)
      self.assertTrue(pld._symmetric)
    else:
      self.assertAlmostEqual(pld._basic_pld_add.infinity_mass,
                             expected_infinity_mass_add)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      self.assertAlmostEqual(pld._basic_pld_remove.infinity_mass,
                             expected_infinity_mass_remove)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_remove,
          pld._basic_pld_remove.rounded_probability_mass_function)
      self.assertFalse(pld._symmetric)

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
    """Verifies correctness of pessimistic PLD for varying truncation bound."""
    pld = privacy_loss_distribution.from_discrete_gaussian_mechanism(
        1, truncation_bound=truncation_bound)
    self.assertAlmostEqual(pld._basic_pld.infinity_mass, expected_infinity_mass)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld._basic_pld.rounded_probability_mass_function)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1, 1.0, {
          5000: 0.45186276,
          -5000: 0.27406862
      }, 0.27406862),
      (1.0, 2, 1.0, {
          0: 0.27406862
      }, 0.72593138),
      (3.0, 1, 1.0, {
          555: 0.34579116,
          -556: 0.32710442
      }, 0.32710442),
      # Tests with sampling_prob < 1
      (1.0, 1, 0.6, {
          -3288: 0.27406862,
          2692: 0.45186276,
          9162: 0.27406862
      }, 0.0, {
          3287: 0.3807451,
          -2693: 0.34518628,
          -9163: 0.10962745
      }, 0.16444117),
      (1.0, 2, 0.3, {
          0: 0.27406862,
          3566: 0.7259314,
      }, 0.0, {
          0: 0.27406862,
          -3567: 0.50815197,
      }, 0.2177794),
      (3.0, 1, 0.1, {
          -57: 0.32710442,
          54: 0.34579116,
          1053: 0.32710442
      }, 0.0, {
          56: 0.32897309,
          -55: 0.34392248,
          -1054: 0.29439398
      }, 0.03271044))
  def test_discrete_gaussian_varying_sigma_and_sensitivity_optimistic(
      self, sigma, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_infinity_mass_add,
      expected_rounded_pmf_remove=None, expected_infinity_mass_remove=None):
    """Verifies correctness of optimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_discrete_gaussian_mechanism(
        sigma, sensitivity=sensitivity, truncation_bound=1,
        pessimistic_estimate=False,
        sampling_prob=sampling_prob)
    if expected_rounded_pmf_remove is None:
      self.assertAlmostEqual(pld._basic_pld.infinity_mass,
                             expected_infinity_mass_add)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld.rounded_probability_mass_function)
      self.assertTrue(pld._symmetric)
    else:
      self.assertAlmostEqual(pld._basic_pld_add.infinity_mass,
                             expected_infinity_mass_add)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_add,
          pld._basic_pld_add.rounded_probability_mass_function)
      self.assertAlmostEqual(pld._basic_pld_remove.infinity_mass,
                             expected_infinity_mass_remove)
      test_util.assert_dictionary_almost_equal(
          self, expected_rounded_pmf_remove,
          pld._basic_pld_remove.rounded_probability_mass_function)
      self.assertFalse(pld._symmetric)


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
    pld = privacy_loss_distribution.from_randomized_response(
        noise_parameter, num_buckets, value_discretization_interval=1)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld._basic_pld.rounded_probability_mass_function)

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
    pld = privacy_loss_distribution.from_randomized_response(
        0.2, 4, value_discretization_interval=value_discretization_interval)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld._basic_pld.rounded_probability_mass_function)

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
    pld = privacy_loss_distribution.from_randomized_response(
        noise_parameter,
        num_buckets,
        pessimistic_estimate=False,
        value_discretization_interval=1)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_probability_mass_function,
        pld._basic_pld.rounded_probability_mass_function)

  @parameterized.parameters((0.0, 10), (1.1, 4), (0.5, 1))
  def test_randomized_response_value_errors(self, noise_parameter, num_buckets):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.from_randomized_response(
          noise_parameter, num_buckets)


class IdentityPrivacyLossDistributionTest(parameterized.TestCase):

  def test_identity(self):
    pld = privacy_loss_distribution.identity()
    test_util.assert_dictionary_almost_equal(
        self, pld._basic_pld.rounded_probability_mass_function, {0: 1})
    self.assertAlmostEqual(pld._basic_pld.infinity_mass, 0)

    pld = pld.compose(
        privacy_loss_distribution.PrivacyLossDistribution({
            1: 0.5,
            -1: 0.5
        }, 1e-4, 0))
    test_util.assert_dictionary_almost_equal(
        self, pld._basic_pld.rounded_probability_mass_function, {
            1: 0.5,
            -1: 0.5
        })
    self.assertAlmostEqual(pld._basic_pld.infinity_mass, 0)


if __name__ == '__main__':
  unittest.main()
