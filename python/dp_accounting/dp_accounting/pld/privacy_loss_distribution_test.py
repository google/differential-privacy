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
from typing import Any, Mapping, Optional
import unittest

from absl.testing import parameterized
from scipy import stats

from dp_accounting import privacy_accountant
from dp_accounting.pld import common
from dp_accounting.pld import pld_pmf
from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.pld import test_util


def _assert_pld_pmf_equal(
    testcase: unittest.TestCase,
    pld: privacy_loss_distribution.PrivacyLossDistribution,
    expected_rounded_pmf_add: Mapping[int, float],
    expected_infinity_mass_add: float,
    expected_rounded_pmf_remove: Optional[Mapping[int, float]] = None,
    expected_infinity_mass_remove: Optional[float] = None):
  """Asserts equality of PLD with expected values."""
  def sparse_loss_probs(pmf: pld_pmf.PLDPmf) -> Mapping[int, float]:
    if isinstance(pmf, pld_pmf.SparsePLDPmf):
      return pmf._loss_probs
    elif isinstance(pmf, pld_pmf.DensePLDPmf):
      return common.list_to_dictionary(pmf._probs, pmf._lower_loss)
    return {}

  test_util.assert_dictionary_almost_equal(
      testcase, expected_rounded_pmf_add, sparse_loss_probs(pld._pmf_add))
  testcase.assertAlmostEqual(expected_infinity_mass_add,
                             pld._pmf_add._infinity_mass)
  if expected_rounded_pmf_remove is None:
    testcase.assertTrue(pld._symmetric)
  else:
    test_util.assert_dictionary_almost_equal(
        testcase,
        expected_rounded_pmf_remove,
        sparse_loss_probs(pld._pmf_remove))
    testcase.assertAlmostEqual(expected_infinity_mass_remove,
                               pld._pmf_remove._infinity_mass)
    testcase.assertFalse(pld._symmetric)


class PrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='basic_symmetric',
          log_pmf_lower={
              0: math.log(0.8),
              1: math.log(0.2),
          },
          log_pmf_upper={
              0: math.log(0.2),
              1: math.log(0.8),
          },
          pessimistic_estimate=True,
          value_discretization_interval=1,
          log_mass_truncation_bound=-math.inf,
          symmetric=True,
          expected_pmf={
              math.ceil(math.log(4)): 0.8,
              math.ceil(-math.log(4)): 0.2,
          },
          infinity_mass=0,
      ),
      dict(
          testcase_name='symmetric_with_inf_mass',
          log_pmf_lower={
              0: math.log(0.8),
              1: math.log(0.1),
              2: math.log(0.1),
          },
          log_pmf_upper={
              0: math.log(0.1),
              1: math.log(0.8),
              3: math.log(0.1),
          },
          pessimistic_estimate=True,
          value_discretization_interval=1,
          log_mass_truncation_bound=-math.inf,
          symmetric=True,
          expected_pmf={
              math.ceil(math.log(8)): 0.8,
              math.ceil(-math.log(8)): 0.1,
          },
          infinity_mass=0.1,
      ),
      dict(
          testcase_name='basic_asymmetric',
          log_pmf_lower={
              0: math.log(0.5),
              1: math.log(0.5),
          },
          log_pmf_upper={
              0: math.log(0.8),
              1: math.log(0.2),
          },
          pessimistic_estimate=True,
          value_discretization_interval=1,
          log_mass_truncation_bound=-math.inf,
          symmetric=False,
          expected_pmf={
              math.ceil(math.log(8/5)): 0.8,
              math.ceil(math.log(2/5)): 0.2,
          },
          infinity_mass=0,
          expected_pmf_add={
              math.ceil(math.log(5/8)): 0.5,
              math.ceil(math.log(5/2)): 0.5,
          },
          infinity_mass_add=0,
      ),
      dict(
          testcase_name='asymmetric_with_inf_mass',
          log_pmf_lower={
              0: math.log(0.4),
              1: math.log(0.4),
              2: math.log(0.2),
          },
          log_pmf_upper={
              0: math.log(0.7),
              1: math.log(0.2),
              3: math.log(0.1),
          },
          pessimistic_estimate=True,
          value_discretization_interval=1,
          log_mass_truncation_bound=-math.inf,
          symmetric=False,
          expected_pmf={
              math.ceil(math.log(7/4)): 0.7,
              math.ceil(math.log(2/4)): 0.2,
          },
          infinity_mass=0.1,
          expected_pmf_add={
              math.ceil(math.log(4/7)): 0.4,
              math.ceil(math.log(4/2)): 0.4,
          },
          infinity_mass_add=0.2,
      ),
  )
  def test_from_two_probability_mass_functions(
      self, log_pmf_lower, log_pmf_upper, pessimistic_estimate,
      value_discretization_interval, log_mass_truncation_bound, symmetric,
      expected_pmf, infinity_mass,
      expected_pmf_add=None, infinity_mass_add=None):
    pld = privacy_loss_distribution.from_two_probability_mass_functions(
        log_pmf_lower, log_pmf_upper, pessimistic_estimate,
        value_discretization_interval, log_mass_truncation_bound, symmetric)
    if symmetric:
      _assert_pld_pmf_equal(self, pld, expected_pmf, infinity_mass)
    else:
      _assert_pld_pmf_equal(self, pld, expected_pmf_add, infinity_mass_add,
                            expected_pmf, infinity_mass)


class AddRemovePrivacyLossDistributionTest(parameterized.TestCase):

  def _create_pld(
      self,
      log_pmf_lower: Mapping[Any, float],
      log_pmf_upper: Mapping[Any, float],
      pessimistic: bool = True
  ) -> privacy_loss_distribution.PrivacyLossDistribution:
    pmf_remove = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_lower, log_pmf_upper,
            pessimistic_estimate=pessimistic)._pmf_remove)
    pmf_add = (
        privacy_loss_distribution.from_two_probability_mass_functions(
            log_pmf_upper, log_pmf_lower,
            pessimistic_estimate=pessimistic)._pmf_remove)
    return privacy_loss_distribution.PrivacyLossDistribution(
        pmf_remove, pmf_add)

  def test_init_errors(self):
    rounded_pmf = {1: 0.5, -1: 0.5}
    value_discretization_interval = 1
    infinity_mass = 0
    pessimistic_estimate = True
    pld = privacy_loss_distribution.PrivacyLossDistribution
    with self.assertRaises(ValueError):
      pld.create_from_rounded_probability(
          rounded_probability_mass_function=rounded_pmf,
          infinity_mass=infinity_mass,
          value_discretization_interval=value_discretization_interval,
          pessimistic_estimate=pessimistic_estimate,
          rounded_probability_mass_function_add=rounded_pmf,
          infinity_mass_add=infinity_mass,
          symmetric=True)
    with self.assertRaises(ValueError):
      pld.create_from_rounded_probability(
          rounded_probability_mass_function=rounded_pmf,
          infinity_mass=infinity_mass,
          value_discretization_interval=value_discretization_interval,
          pessimistic_estimate=pessimistic_estimate,
          rounded_probability_mass_function_add=None,
          infinity_mass_add=None,
          symmetric=False)

  def test_hockey_stick_basic(self):
    # Basic hockey stick divergence computation test
    log_pmf_lower = {1: math.log(0.5), 2: math.log(0.5)}
    log_pmf_upper = {1: math.log(0.6), 2: math.log(0.4)}
    pld_pessimistic = self._create_pld(
        log_pmf_lower, log_pmf_upper, pessimistic=True)
    pld_optimistic = self._create_pld(
        log_pmf_lower, log_pmf_upper, pessimistic=False)

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
    pld_pessimistic = self._create_pld(
        log_pmf_lower, log_pmf_upper, pessimistic=True)
    pld_optimistic = self._create_pld(
        log_pmf_lower, log_pmf_upper, pessimistic=False)

    # Here 4 appears as an outcome of only mu_upper and hence should be included
    # in the infinity_mass variable of _pmf_remove.
    self.assertAlmostEqual(pld_pessimistic._pmf_remove._infinity_mass, 0.1)
    self.assertAlmostEqual(pld_optimistic._pmf_remove._infinity_mass, 0.1)

    # Here 3 appears as an outcome of only mu_lower and hence should be included
    # in the infinity_mass variable of basic_pld_add.
    self.assertAlmostEqual(pld_pessimistic._pmf_add._infinity_mass, 0.6)
    self.assertAlmostEqual(pld_optimistic._pmf_add._infinity_mass, 0.6)

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

  def test_composition(self):
    # Test for composition of privacy loss distribution
    log_pmf_lower1 = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper1 = {1: math.log(0.5), 2: math.log(0.2), 4: math.log(0.3)}
    pld1 = self._create_pld(log_pmf_lower1, log_pmf_upper1, pessimistic=True)

    log_pmf_lower2 = {1: math.log(0.4), 2: math.log(0.6)}
    log_pmf_upper2 = {2: math.log(0.7), 3: math.log(0.3)}
    pld2 = self._create_pld(log_pmf_lower2, log_pmf_upper2, pessimistic=True)

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
    expected_result = self._create_pld(log_pmf_lower_composed,
                                       log_pmf_upper_composed)

    # Check that the result is as expected. Note that we cannot check that the
    # rounded_down_probability_mass_function and
    # rounded_up_probability_mass_function of the two distributions are equal
    # directly because the rounding might cause off-by-one error in index.
    self.assertAlmostEqual(expected_result._pmf_remove._discretization,
                           result._pmf_remove._discretization)
    self.assertAlmostEqual(expected_result._pmf_add._discretization,
                           result._pmf_add._discretization)
    self.assertAlmostEqual(expected_result._pmf_remove._infinity_mass,
                           result._pmf_remove._infinity_mass)
    self.assertAlmostEqual(expected_result._pmf_add._infinity_mass,
                           result._pmf_add._infinity_mass)
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
    pld1 = self._create_pld(log_pmf_lower1, log_pmf_upper1)

    log_pmf_lower2 = {1: math.log(0.4), 2: math.log(0.6)}
    log_pmf_upper2 = {2: math.log(0.7), 3: math.log(0.3)}
    pld2 = self._create_pld(log_pmf_lower2, log_pmf_upper2)

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
    log_pmf_upper1_upper2_composed = {
        (1, 2): math.log(0.35),
        (1, 3): math.log(0.15),
        (2, 2): math.log(0.14),
        (2, 3): math.log(0.06),
        (4, 2): math.log(0.21),
        (4, 3): math.log(0.09)
    }

    expected_result = self._create_pld(log_pmf_lower1_lower2_composed,
                                       log_pmf_upper1_upper2_composed)
    # Check that the result is as expected. Note that we cannot check that the
    # rounded_down_probability_mass_function and
    # rounded_up_probability_mass_function of the two distributions are equal
    # directly because the rounding might cause off-by-one error in index.
    for result in [result12, result21]:
      self.assertAlmostEqual(expected_result._pmf_remove._discretization,
                             result._pmf_remove._discretization)
      self.assertAlmostEqual(expected_result._pmf_add._discretization,
                             result._pmf_add._discretization)
      self.assertAlmostEqual(expected_result._pmf_remove._infinity_mass,
                             result._pmf_remove._infinity_mass)
      self.assertAlmostEqual(expected_result._pmf_add._infinity_mass,
                             result._pmf_add._infinity_mass)
      self.assertAlmostEqual(
          expected_result.get_delta_for_epsilon(0),
          result.get_delta_for_epsilon(0))
      self.assertAlmostEqual(
          expected_result.get_delta_for_epsilon(0.5),
          result.get_delta_for_epsilon(0.5))
      self.assertAlmostEqual(
          expected_result.get_delta_for_epsilon(-0.5),
          result.get_delta_for_epsilon(-0.5))

  def test_self_composition(self):
    log_pmf_lower = {1: math.log(0.2), 2: math.log(0.2), 3: math.log(0.6)}
    log_pmf_upper = {1: math.log(0.5), 2: math.log(0.2), 4: math.log(0.3)}

    pld = self._create_pld(log_pmf_lower, log_pmf_upper)
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

    expected_result = self._create_pld(expected_log_pmf_lower,
                                       expected_log_pmf_upper)

    self.assertAlmostEqual(expected_result._pmf_remove._discretization,
                           result._pmf_remove._discretization)
    self.assertAlmostEqual(expected_result._pmf_remove._infinity_mass,
                           result._pmf_remove._infinity_mass)
    self.assertAlmostEqual(expected_result._pmf_add._discretization,
                           result._pmf_add._discretization)
    self.assertAlmostEqual(expected_result._pmf_add._infinity_mass,
                           result._pmf_add._infinity_mass)
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
        sampling_prob=sampling_prob, use_connect_dots=False)

    _assert_pld_pmf_equal(self, pld,
                          expected_rounded_pmf_add, 0.0,
                          expected_rounded_pmf_remove, 0.0)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1.0, 1.0, {
          1: 0.62245933,
          0: 0.14855068,
          -1: 0.22898999
      }),
      (3.0, 3.0, 1.0, {
          1: 0.62245933,
          0: 0.14855068,
          -1: 0.22898999
      }),
      (1.0, 2.0, 1.0, {
          2: 0.62245933,
          1: 0.14855068,
          0: 0.09010054,
          -1: 0.05464874,
          -2: 0.08424071
      }),
      (2.0, 4.0, 1.0, {
          2: 0.62245933,
          1: 0.14855068,
          0: 0.09010054,
          -1: 0.05464874,
          -2: 0.08424071
      }),
      # Tests with sampling_prob < 1
      (1.0, 1.0, 0.8, {
          1: 0.49796746,
          0: 0.31884054,
          -1: 0.18319199
      }, {
          1: 0.49796746,
          0: 0.31884054,
          -1: 0.18319199
      }),
      (3.0, 3.0, 0.5, {
          1: 0.31122967,
          0: 0.57427534,
          -1: 0.11449500,
      }, {
          1: 0.31122967,
          0: 0.57427534,
          -1: 0.11449500,
      }),
      (1.0, 2.0, 0.7, {
          1: 0.70000000,
          0: 0.17131139,
          -1: 0.08129580,
          -2: 0.04739281,
      }, {
          2: 0.35018810,
          1: 0.22098490,
          0: 0.17131139,
          -1: 0.25751561,
      }),
      (2.0, 4.0, 0.3, {
          1: 0.30000000,
          0: 0.59763391,
          -1: 0.09942388,
          -2: 0.00294221,
      }, {
          2: 0.02174013,
          1: 0.27026212,
          0: 0.59763391,
          -1: 0.11036383,
      }))
  def test_laplace_varying_parameter_and_sensitivity_connect_dots(
      self, parameter, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_rounded_pmf_remove=None):
    """Verifies correctness of connect_dots PLD for various parameter values."""
    pld = privacy_loss_distribution.from_laplace_mechanism(
        parameter, sensitivity=sensitivity, value_discretization_interval=1,
        sampling_prob=sampling_prob, use_connect_dots=True)

    _assert_pld_pmf_equal(self, pld,
                          expected_rounded_pmf_add, 0.0,
                          expected_rounded_pmf_remove, 0.0)

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
                                  expected_rounded_pmf):
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_laplace_mechanism(
        1, value_discretization_interval=value_discretization_interval,
        use_connect_dots=False)

    _assert_pld_pmf_equal(self, pld, expected_rounded_pmf, 0.0)

  @parameterized.parameters((0.5, {
      2: 0.56217650,
      1: 0.09684622,
      0: 0.07542391,
      -1: 0.05874020,
      -2: 0.20681318,
  }), (0.3, {
      4: 0.18817131,
      3: 0.37181835,
      2: 0.06128993,
      1: 0.05275273,
      0: 0.04540470,
      -1: 0.03908019,
      -2: 0.03363663,
      -3: 0.15117006,
      -4: 0.05667611,
  }))
  def test_laplace_discretization_connect_dots(
      self, value_discretization_interval, expected_rounded_pmf):
    """Verifies correctness of connect_dots PLD for varying discretization."""
    pld = privacy_loss_distribution.from_laplace_mechanism(
        1, value_discretization_interval=value_discretization_interval,
        use_connect_dots=True)
    _assert_pld_pmf_equal(self, pld, expected_rounded_pmf, 0.0)

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
        parameter=parameter, sensitivity=sensitivity,
        pessimistic_estimate=False, value_discretization_interval=1,
        sampling_prob=sampling_prob, use_connect_dots=False)

    _assert_pld_pmf_equal(self, pld,
                          expected_rounded_pmf_add, 0.0,
                          expected_rounded_pmf_remove, 0.0)


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
        sampling_prob=sampling_prob,
        use_connect_dots=False)

    test_util.assert_dictionary_almost_equal(self, expected_rounded_pmf_add,
                                             pld._pmf_add._loss_probs)  # pytype: disable=attribute-error
    test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                          pld._pmf_add._infinity_mass)
    if expected_rounded_pmf_remove is None:
      self.assertTrue(pld._symmetric)
    else:
      test_util.assert_dictionary_almost_equal(self,
                                               expected_rounded_pmf_remove,
                                               pld._pmf_remove._loss_probs)  # pytype: disable=attribute-error
      test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                            pld._pmf_remove._infinity_mass)
      self.assertFalse(pld._symmetric)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1.0, 1.0, {
          2: 0.167710257,
          1: 0.343270190,
          0: 0.319116756,
          -1: 0.126282046,
          -2: 0.022697115,
      }, 0.020923636),
      (5.0, 5.0, 1.0, {
          2: 0.167710257,
          1: 0.343270190,
          0: 0.319116756,
          -1: 0.126282046,
          -2: 0.022697115,
      }, 0.020923636),
      (1.0, 2.0, 1.0, {
          4: 0.156393834,
          3: 0.176732821,
          2: 0.195352391,
          1: 0.169838899,
          0: 0.116146963,
          -1: 0.062480239,
          -2: 0.026438071,
          -3: 0.008799009,
          -4: 0.002864453,
      }, 0.084953319),
      (3.0, 6.0, 1.0, {
          4: 0.156393834,
          3: 0.176732821,
          2: 0.195352391,
          1: 0.169838899,
          0: 0.116146963,
          -1: 0.062480239,
          -2: 0.026438071,
          -3: 0.008799009,
          -4: 0.002864453,
      }, 0.084953319),
      # Tests with sampling_prob < 1
      (1.0, 1.0, 0.8, {
          1: 0.448021700,
          0: 0.398104072,
          -1: 0.115544309,
          -2: 0.015193708,
      }, 0.023136210, {
          2: 0.112267164,
          1: 0.314081995,
          0: 0.398104072,
          -1: 0.164817973,
      }, 0.010728796),
      (5.0, 5.0, 0.6, {
          1: 0.363466985,
          0: 0.528456643,
          -1: 0.099555157,
          -2: 0.008521215,
      }, 0.000000000, {
          2: 0.062963737,
          1: 0.270618975,
          0: 0.528456643,
          -1: 0.133712031,
      }, 0.004248614),
      (1.0, 2.0, 0.4, {
          1: 0.431999550,
          0: 0.499732772,
          -1: 0.052519150,
          -2: 0.012224135,
          -3: 0.003524394,
      }, 0.000000000, {
          3: 0.070789342,
          2: 0.090324817,
          1: 0.142761851,
          0: 0.499732772,
          -1: 0.158923753,
      }, 0.037467466),
      (3.0, 6.0, 0.2, {
          1: 0.215999775,
          0: 0.738200118,
          -1: 0.038917231,
          -2: 0.005650510,
          -3: 0.001232367,
      }, 0.000000000, {
          3: 0.024752755,
          2: 0.041751932,
          1: 0.105788001,
          0: 0.738200118,
          -1: 0.079461876,
      }, 0.010045317))
  def test_gaussian_varying_standard_deviation_and_sensitivity_connect_dots(
      self, standard_deviation, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_infinity_mass_add,
      expected_rounded_pmf_remove=None, expected_infinity_mass_remove=None):
    """Verifies correctness of connect_dots PLD for various parameter values."""
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation,
        sensitivity=sensitivity,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=1,
        sampling_prob=sampling_prob,
        use_connect_dots=True)

    _assert_pld_pmf_equal(
        self, pld,
        expected_rounded_pmf_add, expected_infinity_mass_add,
        expected_rounded_pmf_remove, expected_infinity_mass_remove)

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
                                   expected_rounded_pmf):
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        1,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=value_discretization_interval,
        use_connect_dots=False)
    test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                          pld._pmf_remove._infinity_mass)
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_pmf,
        pld._pmf_remove._loss_probs)  # pytype: disable=attribute-error

  @parameterized.parameters((0.5, 0.056696236, {
      3: 0.178515818,
      2: 0.175063076,
      1: 0.195400642,
      0: 0.171573378,
      -1: 0.118516480,
      -2: 0.064402107,
      -3: 0.039832263,
  }), (0.3, 0.056696236, {
      5: 0.143933223,
      4: 0.093801719,
      3: 0.110114456,
      2: 0.118295665,
      1: 0.116301523,
      0: 0.104639278,
      -1: 0.086158287,
      -2: 0.064922037,
      -3: 0.044769197,
      -4: 0.028252535,
      -5: 0.032115843,
  }))
  def test_gaussian_discretization_connect_dots(
      self, value_discretization_interval, expected_infinity_mass,
      expected_rounded_pmf):
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        1,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=value_discretization_interval,
        use_connect_dots=True)
    _assert_pld_pmf_equal(
        self, pld, expected_rounded_pmf, expected_infinity_mass)

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
        sampling_prob=sampling_prob,
        use_connect_dots=False)

    test_util.assert_dictionary_almost_equal(self, expected_rounded_pmf_add,
                                             pld._pmf_add._loss_probs)  # pytype: disable=attribute-error
    test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                          pld._pmf_add._infinity_mass)
    if expected_rounded_pmf_remove is None:
      self.assertTrue(pld._symmetric)
    else:
      test_util.assert_dictionary_almost_equal(self,
                                               expected_rounded_pmf_remove,
                                               pld._pmf_remove._loss_probs)  # pytype: disable=attribute-error
      test_util.assert_almost_greater_equal(self, stats.norm.cdf(-0.9),
                                            pld._pmf_remove._infinity_mass)
      self.assertFalse(pld._symmetric)

  def test_subsampled_gaussian_does_not_overflow(self):
    """Verifies that creating subsampled Gaussian PLD does not result in overflow."""
    privacy_loss_distribution.from_gaussian_mechanism(
        0.02,
        value_discretization_interval=1,
        sampling_prob=0.1,
        use_connect_dots=False)


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
        sampling_prob=sampling_prob, use_connect_dots=False)

    _assert_pld_pmf_equal(self, pld,
                          expected_rounded_pmf_add, 0.0,
                          expected_rounded_pmf_remove, 0.0)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1, 1, {
          1: 0.731058579,
          -1: 0.268941421,
      }),
      (1.0, 2, 1, {
          2: 0.731058579,
          0: 0.170003402,
          -2: 0.098938020
      }),
      (0.8, 2, 1, {
          2: 0.492482728,
          1: 0.197491753,
          0: 0.170722074,
          -1: 0.072653156,
          -2: 0.066650289,
      }),
      (0.8, 3, 1, {
          3: 0.359853436,
          2: 0.330121045,
          1: 0.148724321,
          0: 0.043995505,
          -1: 0.054712620,
          -2: 0.044677025,
          -3: 0.017916048,
      }),
      # # Tests with sampling_prob < 1
      (1.0, 1, 0.8, {
          1: 0.584846863,
          0: 0.200000000,
          -1: 0.215153137,
      }, {
          1: 0.584846863,
          0: 0.200000000,
          -1: 0.215153137,
      }),
      (1.0, 2, 0.5, {
          1: 0.500000000,
          0: 0.401061980,
          -1: 0.067667642,
          -2: 0.031270378,
      }, {
          2: 0.231058579,
          1: 0.183939721,
          0: 0.401061980,
          -1: 0.183939721,
      }),
      (0.8, 2, 0.3, {
          1: 0.261344626,
          0: 0.642512060,
          -1: 0.096143315,
      }, {
          1: 0.261344626,
          0: 0.642512060,
          -1: 0.096143315,
      }),
      (0.8, 3, 0.2, {
          1: 0.228245419,
          0: 0.698218984,
          -1: 0.069698173,
          -2: 0.003837424,
      }, {
          2: 0.028354943,
          1: 0.189459276,
          0: 0.698218984,
          -1: 0.083966797,
      }))
  def test_discrete_laplace_varying_parameter_and_sensitivity_connect_dots(
      self, parameter, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_rounded_pmf_remove=None):
    """Verifies correctness of pessimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_discrete_laplace_mechanism(
        parameter, sensitivity=sensitivity, value_discretization_interval=1,
        sampling_prob=sampling_prob, use_connect_dots=True)

    _assert_pld_pmf_equal(self, pld,
                          expected_rounded_pmf_add, 0.0,
                          expected_rounded_pmf_remove, 0.0)

  @parameterized.parameters((0.1, {
      10: 0.73105858,
      -10: 0.26894142
  }), (0.03, {
      34: 0.73105858,
      -33: 0.26894142
  }))
  def test_discrete_laplace_discretization(
      self, value_discretization_interval,
      expected_rounded_pmf):
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_discrete_laplace_mechanism(
        1, value_discretization_interval=value_discretization_interval,
        use_connect_dots=False)

    _assert_pld_pmf_equal(self, pld, expected_rounded_pmf, 0.0)

  @parameterized.parameters((0.1, {
      10: 0.731058579,
      -10: 0.268941421
  }), (0.03, {
      34: 0.246127076,
      33: 0.484931503,
      -33: 0.180189243,
      -34: 0.088752178,
  }))
  def test_discrete_laplace_discretization_connect_dots(
      self, value_discretization_interval,
      expected_rounded_pmf):
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_discrete_laplace_mechanism(
        1, value_discretization_interval=value_discretization_interval,
        use_connect_dots=True)
    _assert_pld_pmf_equal(self, pld, expected_rounded_pmf, 0.0)

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
        sampling_prob=sampling_prob, use_connect_dots=False)

    _assert_pld_pmf_equal(self, pld,
                          expected_rounded_pmf_add, 0.0,
                          expected_rounded_pmf_remove, 0.0)


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
        sampling_prob=sampling_prob, use_connect_dots=False)

    _assert_pld_pmf_equal(
        self, pld,
        expected_rounded_pmf_add, expected_infinity_mass_add,
        expected_rounded_pmf_remove, expected_infinity_mass_remove)

  @parameterized.parameters(
      # Tests with sampling_prob = 1
      (1.0, 1, 1.0, {
          -5000: 0.274068619,
          5000: 0.451862762
      }, 0.274068619),
      (1.0, 2, 1.0, {
          0: 0.27406862
      }, 0.72593138),
      (3.0, 1, 1.0, {
          -556: 0.181720640,
          -555: 0.145383781,
          555: 0.153680690,
          556: 0.192110468,
      }, 0.327104421),
      # Tests with sampling_prob < 1
      (1.0, 1, 0.6, {
          -3288: 0.141485241,
          -3287: 0.132583378,
          2692: 0.025722139,
          2693: 0.426140623,
          9162: 0.025399872,
          9163: 0.248668747,
      }, 0.0, {
          3288: 0.196565440,
          3287: 0.184179664,
          -2692: 0.019651469,
          -2693: 0.325534808,
          -9162: 0.010160871,
          -9163: 0.099466577,
      }, 0.164441171),
      (1.0, 2, 0.3, {
          0: 0.274068619,
          3566: 0.181882996,
          3567: 0.544048385,
      }, 0.0, {
          -3567: 0.380824327,
          -3566: 0.127327639,
          0: 0.274068619,
      }, 0.217779414),
      (3.0, 1, 0.1, {
          -57: 0.315715606,
          -56: 0.011388815,
          54: 0.281098525,
          55: 0.064692633,
          1053: 0.129151121,
          1054: 0.197953300,
      }, 0.0, {
          -1054: 0.178150936,
          -1053: 0.116243043,
          -55: 0.064337801,
          -54: 0.279584684,
          56: 0.011452771,
          57: 0.317520323,
      }, 0.032710442))
  def test_discrete_gaussian_varying_sigma_and_sensitivity_connect_dots(
      self, sigma, sensitivity, sampling_prob,
      expected_rounded_pmf_add, expected_infinity_mass_add,
      expected_rounded_pmf_remove=None, expected_infinity_mass_remove=None):
    """Verifies correctness of pessimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_discrete_gaussian_mechanism(
        sigma, sensitivity=sensitivity, truncation_bound=1,
        sampling_prob=sampling_prob, use_connect_dots=True)

    _assert_pld_pmf_equal(
        self, pld,
        expected_rounded_pmf_add, expected_infinity_mass_add,
        expected_rounded_pmf_remove, expected_infinity_mass_remove)

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
      self, truncation_bound, expected_rounded_pmf, expected_infinity_mass):
    """Verifies correctness of pessimistic PLD for varying truncation bound."""
    pld = privacy_loss_distribution.from_discrete_gaussian_mechanism(
        1, truncation_bound=truncation_bound, use_connect_dots=False)

    _assert_pld_pmf_equal(
        self, pld, expected_rounded_pmf, expected_infinity_mass)

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
        pessimistic_estimate=False, sampling_prob=sampling_prob,
        use_connect_dots=False)

    _assert_pld_pmf_equal(
        self, pld,
        expected_rounded_pmf_add, expected_infinity_mass_add,
        expected_rounded_pmf_remove, expected_infinity_mass_remove)


class MixtureGaussianPrivacyLossDistributionTest(parameterized.TestCase):
  """Tests for from_mixture_gaussian_mechanism.

  We reuse some test cases from GaussianPrivacyLoss since
  MixtureGaussianPrivacyLoss generalizes that class. However, since
  MixtureGaussianPrivacyLoss uses tighter cutoffs on the PLD than
  GaussianPrivacyLoss, their expected values are different for the same test
  case.
  """

  @parameterized.named_parameters(
      # Gaussian mechanisms
      {
          'testcase_name': 'gaussian_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'expected_rounded_pmf_add': {
              2: 0.12447741,
              1: 0.38292492,
              0: 0.30853754,
          },
          'expected_rounded_pmf_remove': {
              2: 0.12447741,
              1: 0.38292492,
              0: 0.24173034,
              -1: 0.0668072
          },
      },
      {
          'testcase_name': 'gaussian_2',
          'standard_deviation': 1.0,
          'sensitivities': [2.0],
          'sampling_probs': [1.0],
          'expected_rounded_pmf_add': {
              1: 0.30853754,
              2: 0.19146246,
              3: 0.19146246,
              4: 0.12447741
          },
          'expected_rounded_pmf_remove': {
              -3: 0.00620967,
              -2: 0.01654047,
              -1: 0.04405707,
              0: 0.09184805,
              1: 0.14988228,
              2: 0.19146246,
              3: 0.19146246,
              4: 0.12447741,
          },
      },
      # Subsampled Gaussian mechanisms
      {
          'testcase_name': 'subsampled_gaussian_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.2, 0.8],
          'expected_rounded_pmf_add': {
              1: 0.50740234,
              0: 0.30853754,
          },
          'expected_rounded_pmf_remove': {
              2: 0.03303238,
              1: 0.39779076,
              0: 0.38512252
          },
      },
      {
          'testcase_name': 'subsampled_gaussian_2',
          'standard_deviation': 5.0,
          'sensitivities': [0.0, 5.0],
          'sampling_probs': [0.4, 0.6],
          'expected_rounded_pmf_add': {
              1: 0.50740234,
              0: 0.30853754,
          },
          'expected_rounded_pmf_remove': {
              1: 0.35423381,
              0: 0.46170751
          },
      },
      # Mixture Gaussian mechanisms
      {
          'testcase_name': 'mixture_gaussian_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'expected_rounded_pmf_add': {
              2: 0.315939874,
              1: 0.0,
              0: 0.5
          },
          'expected_rounded_pmf_remove': {
              2: 0.315949035,
              1: 0.300256821,
              0: 0.0,
              -1: 0.199743179
          },
      },
  )
  def test_mixture_gaussian_varying_standard_deviation_and_sensitivity(
      self,
      standard_deviation,
      sensitivities,
      sampling_probs,
      expected_rounded_pmf_add,
      expected_rounded_pmf_remove,
  ):
    """Verifies correctness of pessimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_mixture_gaussian_mechanism(
        standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=1,
        use_connect_dots=False,
    )

    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_pmf_add, pld._pmf_add._loss_probs  # pytype: disable=attribute-error
    )
    test_util.assert_almost_greater_equal(
        self, stats.norm.cdf(-0.9), pld._pmf_add._infinity_mass
    )
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_pmf_remove, pld._pmf_remove._loss_probs  # pytype: disable=attribute-error
    )
    test_util.assert_almost_greater_equal(
        self, stats.norm.cdf(-0.9), pld._pmf_remove._infinity_mass
    )
    self.assertFalse(pld._symmetric)

  @parameterized.named_parameters(
      # Gaussian mechanisms
      {
          'testcase_name': 'gaussian_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'expected_rounded_pmf_add': {
              -1: 0.158655254,
              0: 0.341344746,
              1: 0.391270256,
              2: 0.108729744
          },
          'expected_rounded_pmf_remove': {
              -2: 0.016737493,
              -1: 0.141917761,
              0: 0.341344746,
              1: 0.391270256,
              2: 0.108729744,
          },
      },
      {
          'testcase_name': 'gaussian_2',
          'standard_deviation': 1.0,
          'sensitivities': [2.0],
          'sampling_probs': [1.0],
          'expected_rounded_pmf_add': {
              0: 0.1749993,
              1: 0.3250007,
              2: 0.02438595,
              3: 0.390660732,
              4: 0.0,
          },
          'expected_rounded_pmf_remove': {
              -4: 0.0,
              -3: 0.023964237,
              -2: 0.0,
              -1: 0.119561076,
              0: 0.016344046,
              1: 0.38577247,
              2: 0.0,
              3: 0.390660732,
              4: 0.0,
          },
      },
      # Subsampled Gaussian mechanisms
      {
          'testcase_name': 'subsampled_gaussian_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.2, 0.8],
          'expected_rounded_pmf_add': {
              -1: 0.158655254,
              0: 0.422688932,
              1: 0.395727515
          },
          'expected_rounded_pmf_remove': {
              -1: 0.145580017,
              0: 0.422688932,
              1: 0.431001195,
              2: 0.000729856
          },
      },
      {
          'testcase_name': 'subsampled_gaussian_2',
          'standard_deviation': 5.0,
          'sensitivities': [0.0, 5.0],
          'sampling_probs': [0.4, 0.6],
          'expected_rounded_pmf_add': {
              -1: 0.107598496,
              0: 0.530601573,
              1: 0.361799931
          },
          'expected_rounded_pmf_remove': {
              -1: 0.133098756,
              0: 0.530601573,
              1: 0.292483037
          },
      },
      # Mixture Gaussian mechanisms
      {
          'testcase_name': 'mixture_gaussian_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'expected_rounded_pmf_add': {
              -1: 0.134743039,
              0: 0.390257664,
              1: 0.474999297,
              2: 0.0,
          },
          'expected_rounded_pmf_remove': {
              -2: 0.0,
              -1: 0.208940423,
              0: 0.365256961,
              1: 0.344684545,
              2: 0.058674138,
          },
      },
  )
  def test_mixture_gaussian_varying_standard_deviation_and_sensitivity_connect_dots(
      self,
      standard_deviation,
      sensitivities,
      sampling_probs,
      expected_rounded_pmf_add,
      expected_rounded_pmf_remove,
  ):
    """Verifies correctness of pessimistic PLD for various parameter values."""
    pld = privacy_loss_distribution.from_mixture_gaussian_mechanism(
        standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=1,
        use_connect_dots=True,
    )

    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_pmf_add, pld._pmf_add._loss_probs  # pytype: disable=attribute-error
    )
    test_util.assert_almost_greater_equal(
        self, stats.norm.cdf(-0.9), pld._pmf_add._infinity_mass
    )
    test_util.assert_dictionary_almost_equal(
        self, expected_rounded_pmf_remove, pld._pmf_remove._loss_probs  # pytype: disable=attribute-error
    )
    test_util.assert_almost_greater_equal(
        self, stats.norm.cdf(-0.9), pld._pmf_remove._infinity_mass
    )
    self.assertFalse(pld._symmetric)

  @parameterized.named_parameters(
      {
          'testcase_name': 'discretization_0.5',
          'value_discretization_interval': 0.5,
          'expected_add_pmf': {
              3: 0.02595983,
              2: 0.3045079,
              1: 0.22891269,
              0: 0.25655945
          },
          'expected_remove_pmf': {
              4: 0.01100095,
              3: 0.10672757,
              2: 0.1421942,
              1: 0.17070797,
              0: 0.17687728,
              -1: 0.144677,
              -2: 0.06376406,
          },
      },
      {
          'testcase_name': 'discretization_0.3',
          'value_discretization_interval': 0.3,
          'expected_add_pmf': {
              4: 0.08817633,
              3: 0.18512511,
              2: 0.16071804,
              1: 0.12536095,
              0: 0.25655945,
          },
          'expected_remove_pmf': {
              6: 0.01100095,
              5: 0.05979597,
              4: 0.07257058,
              3: 0.08541692,
              2: 0.09686874,
              1: 0.10497753,
              0: 0.10751992,
              -1: 0.10233776,
              -2: 0.08761942,
              -3: 0.08784134,
          },
      },
  )
  def test_mixture_gaussian_discretization(
      self,
      value_discretization_interval,
      expected_add_pmf,
      expected_remove_pmf,
  ):
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_mixture_gaussian_mechanism(
        standard_deviation=1,
        sensitivities=[0.0, 1.0, 2.0],
        sampling_probs=[0.2, 0.6, 0.2],
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=value_discretization_interval,
        use_connect_dots=False,
    )
    test_util.assert_almost_greater_equal(
        self, stats.norm.cdf(-0.9), pld._pmf_add._infinity_mass
    )
    test_util.assert_almost_greater_equal(
        self, stats.norm.cdf(-0.9), pld._pmf_remove._infinity_mass
    )
    test_util.assert_dictionary_almost_equal(
        self, expected_add_pmf, pld._pmf_add._loss_probs  # pytype: disable=attribute-error
    )
    test_util.assert_dictionary_almost_equal(
        self, expected_remove_pmf, pld._pmf_remove._loss_probs  # pytype: disable=attribute-error
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'discretization_0.5',
          'value_discretization_interval': 0.5,
          'expected_add_infinity_mass': 7.02736093e-05,
          'expected_remove_infinity_mass': 0.058450414,
          'expected_add_pmf': {
              -1: 0.1766054,
              0: 0.17661904,
              1: 0.26907836,
              2: 0.2845549,
              3: 0.09307203,
          },
          'expected_remove_pmf': {
              -3: 0.02076718,
              -2: 0.1046819,
              -1: 0.16320427,
              0: 0.17661904,
              1: 0.15880237,
              2: 0.12600393,
              3: 0.09134778,
              4: 0.10012311,
          },
      },
      {
          'testcase_name': 'discretization_0.3',
          'value_discretization_interval': 0.3,
          'expected_add_infinity_mass': 0.010838515,
          'expected_remove_infinity_mass': 0.072142753,
          'expected_add_pmf': {
              -1: 0.20621578,
              0: 0.10686974,
              1: 0.14244404,
              2: 0.17395599,
              3: 0.18472755,
              4: 0.17494839,
          },
          'expected_remove_pmf': {
              -4: 0.05269344,
              -3: 0.07510462,
              -2: 0.09546907,
              -1: 0.10552514,
              0: 0.10686974,
              1: 0.10149509,
              2: 0.09162041,
              3: 0.07936186,
              4: 0.06644975,
              5: 0.05409823,
              6: 0.09916991,
          },
      },
  )
  def test_mixture_gaussian_discretization_connect_dots(
      self,
      value_discretization_interval,
      expected_add_infinity_mass,
      expected_remove_infinity_mass,
      expected_add_pmf,
      expected_remove_pmf,
  ):
    """Verifies correctness of pessimistic PLD for varying discretization."""
    pld = privacy_loss_distribution.from_mixture_gaussian_mechanism(
        standard_deviation=1,
        sensitivities=[0.0, 1.0, 2.0],
        sampling_probs=[0.2, 0.6, 0.2],
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-0.9),
        value_discretization_interval=value_discretization_interval,
        use_connect_dots=True,
    )
    self.assertAlmostEqual(
        expected_add_infinity_mass, pld._pmf_add._infinity_mass
    )
    self.assertAlmostEqual(
        expected_remove_infinity_mass, pld._pmf_remove._infinity_mass
    )
    test_util.assert_dictionary_almost_equal(
        self, expected_add_pmf, pld._pmf_add._loss_probs  # pytype: disable=attribute-error
    )
    test_util.assert_dictionary_almost_equal(
        self, expected_remove_pmf, pld._pmf_remove._loss_probs  # pytype: disable=attribute-error
    )

  @parameterized.named_parameters(
      (
          'negative_stdev',
          -1.0,
          [1.0],
          [1.0],
      ),
      (
          'negative_sensitivity',
          1.0,
          [-1.0, 1.0],
          [0.5, 0.5],
      ),
      (
          'negative_probability',
          1.0,
          [0.0, 1.0, 2.0],
          [0.75, 0.75, -0.5],
      ),
      (
          'probability_greater_than_one',
          1.0,
          [0.0, 1.0, 2.0],
          [1.5, 0.5, 0.5],
      ),
      (
          'probabilities_dont_add_up_to_one',
          1.0,
          [0.0, 1.0, 2.0],
          [0.2, 0.2, 0.2],
      ),
      (
          'list_lengths_differ_1',
          1.0,
          [1.0],
          [0.5, 0.5],
      ),
      (
          'list_lengths_differ_2',
          1.0,
          [1.0, 2.0, 3.0],
          [0.5, 0.5],
      ),
  )
  def test_mixture_gaussian_value_errors(
      self, standard_deviation, sensitivities, sampling_probs
  ):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.from_mixture_gaussian_mechanism(
          standard_deviation,
          sensitivities=sensitivities,
          sampling_probs=sampling_probs,
          value_discretization_interval=1,
      )

  def test_mixture_gaussian_does_not_overflow(self):
    """Verifies that mixture Gaussian PLD does not result in overflow."""
    privacy_loss_distribution.from_mixture_gaussian_mechanism(
        standard_deviation=1,
        sensitivities=[0.0, 1.0, 10.0],
        sampling_probs=[0.98, 0.01, 0.01],
        value_discretization_interval=1,
        use_connect_dots=False,
    )


class RandomizedResponsePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          noise_parameter=0.5,
          num_buckets=2,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_ONE,
          expected_rounded_pmf={
              2: 0.75,  # ceil(log(3)) = ceil(1.098).
              -1: 0.25,  # ceil(-log(3)) = ceil(-1.098).
          },
      ),
      dict(
          noise_parameter=0.5,
          num_buckets=2,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
          expected_rounded_pmf={
              1: 0.75,  # ceil(log(3/2)) = ceil(0.405).
              0: 0.25,  # ceil(log(1/2))) = ceil(-0.693).
          },
          expected_rounded_pmf_add={
              1: 0.5,  # ceil(log(2)) = ceil(0.693).
              0: 0.5,  # ceil(log(2/3)) = ceil(-0.405).
          }
      ),
      dict(
          noise_parameter=0.2,
          num_buckets=4,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_ONE,
          expected_rounded_pmf={
              3: 0.85,  # ceil(log(17)) = ceil(2.833)
              -2: 0.05,  # ceil(-log(17)) = ceil(-2.833)
              0: 0.1,
          }
      ),
      dict(
          noise_parameter=0.2,
          num_buckets=4,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
          expected_rounded_pmf={
              2: 0.85,  # ceil(log(0.85 / 0.25)) = ceil(1.224)
              -1: 0.15,  # ceil(log(0.05 / 0.25)) = ceil(-1.609).
          },
          expected_rounded_pmf_add={
              2: 0.75,  # ceil(log(0.25 / 0.05)) = ceil(1.609).
              -1: 0.25,  # ceil(log(0.25 / 0.85)) = ceil(-1.224).
          },
      ),
  )
  def test_randomized_response_pessimistic(
      self, noise_parameter, num_buckets, neighbor_rel,
      expected_rounded_pmf, expected_rounded_pmf_add=None):
    # Set value_discretization_interval = 1 here.
    pld = privacy_loss_distribution.from_randomized_response(
        noise_parameter, num_buckets,
        value_discretization_interval=1,
        neighboring_relation=neighbor_rel)
    if neighbor_rel == privacy_accountant.NeighboringRelation.REPLACE_ONE:
      _assert_pld_pmf_equal(self, pld, expected_rounded_pmf, 0.0)
    else:  # Case of REPLACE_SPECIAL.
      _assert_pld_pmf_equal(
          self, pld, expected_rounded_pmf_add, 0.0, expected_rounded_pmf, 0.0
      )

  @parameterized.parameters(
      dict(
          value_discretization_interval=0.7,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_ONE,
          expected_rounded_pmf={
              5: 0.85,
              -4: 0.05,
              0: 0.1,
          },
      ),
      dict(
          value_discretization_interval=0.5,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
          expected_rounded_pmf={
              3: 0.85,
              -3: 0.15,
          },
          expected_rounded_pmf_add={
              4: 0.75,
              -2: 0.25,
          },
      ),
      dict(
          value_discretization_interval=2,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_ONE,
          expected_rounded_pmf={
              2: 0.85,
              -1: 0.05,
              0: 0.1,
          },
      ),
      dict(
          value_discretization_interval=2,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
          expected_rounded_pmf={
              1: 0.85,
              0: 0.15,
          },
          expected_rounded_pmf_add={
              1: 0.75,
              0: 0.25,
          },
      ),
  )
  def test_randomized_response_discretization(
      self, value_discretization_interval, neighbor_rel,
      expected_rounded_pmf, expected_rounded_pmf_add=None):
    # Set noise_parameter = 0.2, num_buckets = 4 here.
    pld = privacy_loss_distribution.from_randomized_response(
        0.2, 4, value_discretization_interval=value_discretization_interval,
        neighboring_relation=neighbor_rel)
    if neighbor_rel == privacy_accountant.NeighboringRelation.REPLACE_ONE:
      # The true (non-discretized) PLD is
      # {2.83321334: 0.85, -2.83321334: 0.05, 0: 0.1}.
      _assert_pld_pmf_equal(self, pld, expected_rounded_pmf, 0.0)
    else:  # Case of REPLACE_SPECIAL
      # The true (non-discretized) PLD is
      # REMOVE: {1.224: 0.85, -1.609: 0.15}.
      # ADD: {1.609: 0.75, -1.224: 0.25}.
      _assert_pld_pmf_equal(
          self, pld, expected_rounded_pmf_add, 0.0, expected_rounded_pmf, 0.0)

  @parameterized.parameters(
      dict(
          noise_parameter=0.5,
          num_buckets=2,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_ONE,
          expected_rounded_pmf={
              1: 0.75,  # floor(log(3)) = floor(1.098).
              -2: 0.25,  # floor(-log(3)) = floor(-1.098).
          },
      ),
      dict(
          noise_parameter=0.5,
          num_buckets=2,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
          expected_rounded_pmf={
              0: 0.75,  # floor(log(3/2)) = floor(0.405).
              -1: 0.25,  # floor(log(1/2))) = floor(-0.693).
          },
          expected_rounded_pmf_add={
              0: 0.5,  # floor(log(2)) = floor(0.693).
              -1: 0.5,  # floor(log(2/3)) = floor(-0.405).
          }
      ),
      dict(
          noise_parameter=0.2,
          num_buckets=4,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_ONE,
          expected_rounded_pmf={
              2: 0.85,  # floor(log(17)) = floor(2.833)
              -3: 0.05,  # floor(-log(17)) = floor(-2.833)
              0: 0.1,
          }
      ),
      dict(
          noise_parameter=0.2,
          num_buckets=4,
          neighbor_rel=privacy_accountant.NeighboringRelation.REPLACE_SPECIAL,
          expected_rounded_pmf={
              1: 0.85,  # floor(log(0.85 / 0.25)) = floor(1.224)
              -2: 0.15,  # floor(log(0.05 / 0.25)) = floor(-1.609).
          },
          expected_rounded_pmf_add={
              1: 0.75,  # floor(log(0.25 / 0.05)) = floor(1.609).
              -2: 0.25,  # floor(log(0.25 / 0.85)) = floor(-1.224).
          },
      ),
  )
  def test_randomized_response_optimistic(
      self, noise_parameter, num_buckets, neighbor_rel,
      expected_rounded_pmf, expected_rounded_pmf_add=None):
    # Set value_discretization_interval = 1 here.
    pld = privacy_loss_distribution.from_randomized_response(
        noise_parameter,
        num_buckets,
        pessimistic_estimate=False,
        value_discretization_interval=1,
        neighboring_relation=neighbor_rel)
    if neighbor_rel == privacy_accountant.NeighboringRelation.REPLACE_ONE:
      _assert_pld_pmf_equal(self, pld, expected_rounded_pmf, 0.0)
    else:  # Case of REPLACE_SPECIAL.
      _assert_pld_pmf_equal(
          self, pld, expected_rounded_pmf_add, 0.0, expected_rounded_pmf, 0.0
      )

  @parameterized.parameters((0.0, 10), (1.1, 4), (0.5, 1))
  def test_randomized_response_value_errors(self, noise_parameter, num_buckets):
    with self.assertRaises(ValueError):
      privacy_loss_distribution.from_randomized_response(
          noise_parameter, num_buckets)


class IdentityPrivacyLossDistributionTest(parameterized.TestCase):

  def test_identity(self):
    pld = privacy_loss_distribution.identity()
    _assert_pld_pmf_equal(self, pld, {0: 1}, 0.0)

    pld = pld.compose(
        privacy_loss_distribution.PrivacyLossDistribution
        .create_from_rounded_probability({
            1: 0.5,
            -1: 0.5
        }, 0, 1e-4))
    _assert_pld_pmf_equal(self, pld, {1: 0.5, -1: 0.5}, 0.0)


if __name__ == '__main__':
  unittest.main()
