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

"""Tests for common."""

import math
import unittest
from absl.testing import parameterized

from dp_accounting import common


class CommonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_initial_guess',
          'func': (lambda x: -x),
          'value': -5,
          'lower_x': 0,
          'upper_x': 10,
          'initial_guess_x': None,
          'expected_x': 5,
      },
      {
          'testcase_name': 'with_initial_guess',
          'func': (lambda x: -x),
          'value': -5,
          'lower_x': 0,
          'upper_x': 10,
          'initial_guess_x': 2,
          'expected_x': 5,
      },
      {
          'testcase_name': 'out_of_range',
          'func': (lambda x: -x),
          'value': -5,
          'lower_x': 0,
          'upper_x': 4,
          'initial_guess_x': None,
          'expected_x': None,
      },
      {
          'testcase_name': 'infinite_upper_bound',
          'func': (lambda x: -1/(1/x)),
          'value': -5,
          'lower_x': 0,
          'upper_x': math.inf,
          'initial_guess_x': 2,
          'expected_x': 5,
      })
  def test_inverse_monotone_function(self, func, value, lower_x, upper_x,
                                     initial_guess_x, expected_x):
    search_parameters = common.BinarySearchParameters(
        lower_x, upper_x, initial_guess=initial_guess_x)
    self.assertAlmostEqual(
        expected_x,
        common.inverse_monotone_function(
            func,
            value,
            search_parameters))


if __name__ == '__main__':
  unittest.main()
