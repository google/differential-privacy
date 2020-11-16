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

"""Tests for accountant."""

import unittest
from absl.testing import parameterized

from dp_accounting import accountant
from dp_accounting import common


class AccountantTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'basic_composition',
          'sensitivity': 21,
          'epsilon': 3,
          'delta': 0,
          'num_queries': 10,
          'expected_parameter': 70,
      },
      {
          'testcase_name': 'positive_delta',
          'sensitivity': 1,
          'epsilon': 1,
          'delta': 0.0001,
          'num_queries': 20,
          'expected_parameter': 13.6,
      },
      {
          'testcase_name': 'positive_delta_varying_sensitivity',
          'sensitivity': 0.5,
          'epsilon': 1,
          'delta': 0.0001,
          'num_queries': 20,
          'expected_parameter': 6.8,
      },)
  def test_get_smallest_laplace_noise(self, epsilon, delta, num_queries,
                                      sensitivity, expected_parameter):
    privacy_parameters = common.DifferentialPrivacyParameters(
        epsilon, delta)
    self.assertAlmostEqual(
        expected_parameter,
        accountant.get_smallest_laplace_noise(
            privacy_parameters, num_queries, sensitivity=sensitivity),
        delta=0.1)

  @parameterized.named_parameters(
      {
          'testcase_name': 'basic_composition',
          'sensitivity': 2,
          'epsilon': 3,
          'delta': 0,
          'num_queries': 5,
          'expected_parameter': 0.3,
      },
      {
          'testcase_name': 'positive_delta',
          'sensitivity': 1,
          'epsilon': 1,
          'delta': 0.0001,
          'num_queries': 20,
          'expected_parameter': 0.073,
      },
      {
          'testcase_name': 'positive_delta_varying_sensitivity',
          'sensitivity': 5,
          'epsilon': 1,
          'delta': 0.0001,
          'num_queries': 20,
          'expected_parameter': 0.014,
      },)
  def test_get_smallest_discrete_laplace_noise(self, epsilon, delta,
                                               num_queries, sensitivity,
                                               expected_parameter):
    privacy_parameters = common.DifferentialPrivacyParameters(
        epsilon, delta)
    self.assertAlmostEqual(
        expected_parameter,
        accountant.get_smallest_discrete_laplace_noise(
            privacy_parameters, num_queries, sensitivity=sensitivity),
        delta=1e-3)

  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'sensitivity': 1,
          'epsilon': 1,
          'delta': 0.78760074,
          'num_queries': 1,
          'expected_std': 1/3,
      },
      {
          'testcase_name': 'varying_sensitivity_and_num_queries',
          'sensitivity': 6,
          'epsilon': 1,
          'delta': 0.78760074,
          'num_queries': 25,
          'expected_std': 10,
      })
  def test_get_smallest_gaussian_noise(self, epsilon, delta, num_queries,
                                       sensitivity, expected_std):
    privacy_parameters = common.DifferentialPrivacyParameters(
        epsilon, delta)
    self.assertAlmostEqual(
        expected_std,
        accountant.get_smallest_gaussian_noise(
            privacy_parameters, num_queries, sensitivity=sensitivity))


if __name__ == '__main__':
  unittest.main()
