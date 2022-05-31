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
import numpy as np

from dp_accounting import common
from dp_accounting import test_util


class DifferentialPrivacyParametersTest(parameterized.TestCase):

  @parameterized.parameters((-0.1, 0.1), (1, -0.1), (1, 1.1))
  def test_epsilon_delta_value_errors(self, epsilon, delta):
    with self.assertRaises(ValueError):
      common.DifferentialPrivacyParameters(epsilon, delta)


class CommonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_initial_guess',
          'func': (lambda x: -x),
          'value': -4.5,
          'lower_x': 0,
          'upper_x': 10,
          'initial_guess_x': None,
          'expected_x': 4.5,
          'increasing': False,
      }, {
          'testcase_name': 'with_initial_guess',
          'func': (lambda x: -x),
          'value': -5,
          'lower_x': 0,
          'upper_x': 10,
          'initial_guess_x': 2,
          'expected_x': 5,
          'increasing': False,
      }, {
          'testcase_name': 'out_of_range',
          'func': (lambda x: -x),
          'value': -5,
          'lower_x': 0,
          'upper_x': 4,
          'initial_guess_x': None,
          'expected_x': None,
          'increasing': False,
      }, {
          'testcase_name': 'infinite_upper_bound',
          'func': (lambda x: -1 / (1 / x)),
          'value': -5,
          'lower_x': 0,
          'upper_x': math.inf,
          'initial_guess_x': 2,
          'expected_x': 5,
          'increasing': False,
      }, {
          'testcase_name': 'increasing_no_initial_guess',
          'func': (lambda x: x**2),
          'value': 25,
          'lower_x': 0,
          'upper_x': 10,
          'initial_guess_x': None,
          'expected_x': 5,
          'increasing': True,
      }, {
          'testcase_name': 'increasing_with_initial_guess',
          'func': (lambda x: x**2),
          'value': 25,
          'lower_x': 0,
          'upper_x': 10,
          'initial_guess_x': 2,
          'expected_x': 5,
          'increasing': True,
      }, {
          'testcase_name': 'increasing_out_of_range',
          'func': (lambda x: x**2),
          'value': 5,
          'lower_x': 6,
          'upper_x': 10,
          'initial_guess_x': None,
          'expected_x': None,
          'increasing': True,
      }, {
          'testcase_name': 'discrete',
          'func': (lambda x: -x),
          'value': -4.5,
          'lower_x': 0,
          'upper_x': 10,
          'initial_guess_x': None,
          'expected_x': 5,
          'increasing': False,
          'discrete': True,
      })
  def test_inverse_monotone_function(self,
                                     func,
                                     value,
                                     lower_x,
                                     upper_x,
                                     initial_guess_x,
                                     expected_x,
                                     increasing,
                                     discrete=False):
    search_parameters = common.BinarySearchParameters(
        lower_x, upper_x, initial_guess=initial_guess_x, discrete=discrete)
    x = common.inverse_monotone_function(
        func, value, search_parameters, increasing=increasing)
    if expected_x is None:
      self.assertIsNone(x)
    else:
      self.assertAlmostEqual(expected_x, x)


class DictListConversionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'truncate_both_sides',
          'input_list': [0.2, 0.5, 0.3],
          'offset': 1,
          'tail_mass_truncation': 0.6,
          'expected_result': {
              2: 0.5
          },
      }, {
          'testcase_name': 'truncate_lower_only',
          'input_list': [0.2, 0.5, 0.3],
          'offset': 1,
          'tail_mass_truncation': 0.4,
          'expected_result': {
              2: 0.5,
              3: 0.3
          },
      }, {
          'testcase_name': 'truncate_upper_only',
          'input_list': [0.4, 0.5, 0.1],
          'offset': 1,
          'tail_mass_truncation': 0.3,
          'expected_result': {
              1: 0.4,
              2: 0.5
          },
      }, {
          'testcase_name': 'truncate_all',
          'input_list': [0.4, 0.5, 0.1],
          'offset': 1,
          'tail_mass_truncation': 3,
          'expected_result': {},
      })
  def test_list_to_dict_truncation(self, input_list, offset,
                                   tail_mass_truncation, expected_result):
    result = common.list_to_dictionary(
        input_list, offset, tail_mass_truncation=tail_mass_truncation)
    test_util.assert_dictionary_almost_equal(self, expected_result, result)


class ConvolveTest(parameterized.TestCase):

  def test_convolve_dictionary(self):
    dictionary1 = {1: 2, 3: 4}
    dictionary2 = {2: 3, 4: 6}
    expected_result = {3: 6, 5: 24, 7: 24}
    result = common.convolve_dictionary(dictionary1, dictionary2)
    test_util.assert_dictionary_almost_equal(self, expected_result, result)

  def test_convolve_dictionary_with_truncation(self):
    dictionary1 = {1: 0.4, 2: 0.6}
    dictionary2 = {1: 0.7, 3: 0.3}
    expected_result = {3: 0.42, 4: 0.12}
    result = common.convolve_dictionary(dictionary1, dictionary2, 0.57)
    test_util.assert_dictionary_almost_equal(self, expected_result, result)

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
    result = common.self_convolve_dictionary(inp_dictionary, 3)
    test_util.assert_dictionary_almost_equal(self, expected_result, result)

  @parameterized.parameters(([3, 5, 7], 2, [9, 30, 67, 70, 49]),
                            ([1, 3, 4], 3, [1, 9, 39, 99, 156, 144, 64]))
  def test_self_convolve_basic(self, input_list, num_times, expected_result):
    min_val, result_list = common.self_convolve(input_list, num_times)
    self.assertEqual(0, min_val)
    self.assertSequenceAlmostEqual(expected_result, result_list)

  @parameterized.parameters(([0.1, 0.4, 0.5], 3, [-1], 0.5, 2, 6),
                            ([0.2, 0.6, 0.2], 3, [1], 0.7, 0, 5))
  def test_compute_self_convolve_bounds(self, input_list, num_times, orders,
                                        tail_mass_truncation,
                                        expected_lower_bound,
                                        expected_upper_bound):
    lower_bound, upper_bound = common.compute_self_convolve_bounds(
        input_list, num_times, tail_mass_truncation, orders=orders)
    self.assertEqual(expected_lower_bound, lower_bound)
    self.assertEqual(expected_upper_bound, upper_bound)

  @parameterized.parameters(
      ([0.1, 0.4, 0.5], 3, 0.5, 2, [0.063, 0.184, 0.315, 0.301, 0.137]),
      ([0.2, 0.6, 0.2], 3, 0.7, 1, [0.08, 0.24, 0.36, 0.24, 0.08]))
  def test_compute_self_convolve_with_truncation(self, input_list, num_times,
                                                 tail_mass_truncation,
                                                 expected_min_val,
                                                 expected_result_list):
    min_val, result_list = common.self_convolve(
        input_list, num_times, tail_mass_truncation=tail_mass_truncation)
    self.assertEqual(min_val, expected_min_val)
    self.assertSequenceAlmostEqual(expected_result_list, result_list)


class PLDPmfTest(parameterized.TestCase):

  def _create_pmf(self,
                  discretization: float,
                  dense: bool,
                  infinity_mass: float = 0.0,
                  lower_loss: int = 0,
                  probs: np.ndarray = np.array([1.0]),
                  pessimistic_estimate: bool = True) -> common.PLDPmf:
    """Helper function for creating PLD for testing."""
    if dense:
      return common.DensePLDPmf(discretization, lower_loss, probs,
                                infinity_mass, pessimistic_estimate)

    loss_probs = common.list_to_dictionary(probs, lower_loss)
    return common.SparsePLDPmf(loss_probs, discretization, infinity_mass,
                               pessimistic_estimate)

  def _check_dense_probs(self, dense_pmf: common.DensePLDPmf,
                         expected_lower_loss: int, expected_probs: np.ndarray):
    """Checks that resulting dense pmf satisfies expectations."""
    self.assertEqual(expected_lower_loss, dense_pmf._lower_loss)
    self.assertSequenceAlmostEqual(expected_probs, dense_pmf._probs)

  def _check_sparse_probs(self, sparse_pmf: common.SparsePLDPmf,
                          expected_lower_loss: int, expected_probs: np.ndarray):
    """Checks that resulting sparse pmf satisfies expectations."""
    expected_loss_probs = common.list_to_dictionary(expected_probs,
                                                    expected_lower_loss)
    test_util.assert_dictionary_almost_equal(self, expected_loss_probs,
                                             sparse_pmf._loss_probs)

  @parameterized.parameters(False, True)
  def test_delta_for_epsilon(self, dense):
    discretization = 0.1
    infinity_mass = 0.1
    lower_loss = -1
    probs = np.array([0.2, 0.3, 0, 0.4])
    pmf = self._create_pmf(discretization, dense, infinity_mass, lower_loss,
                           probs)
    self.assertAlmostEqual(0.1, pmf.get_delta_for_epsilon(3))  # infinity_mass
    self.assertAlmostEqual(0.1 + 0.4 * (1 - np.exp(-0.1)),
                           pmf.get_delta_for_epsilon(0.1))
    self.assertAlmostEqual(1, pmf.get_delta_for_epsilon(-20))
    self.assertEqual(infinity_mass, pmf.get_delta_for_epsilon(np.inf))
    self.assertAlmostEqual(1, pmf.get_delta_for_epsilon(-np.inf))

  @parameterized.parameters(False, True)
  def test_epsilon_for_delta(self, dense):
    discretization = 0.1
    lower_loss = -1  # loss_value
    probs = np.array([0.2, 0.3, 0, 0.4])  # probs for losses -0.1, 0, 0.1, 0.2
    infinity_mass = 0.1
    pmf = self._create_pmf(discretization, dense, infinity_mass, lower_loss,
                           probs)
    self.assertEqual(np.inf, pmf.get_epsilon_for_delta(0.05))  # <infinity_mass
    self.assertAlmostEqual(0.2, pmf.get_epsilon_for_delta(0.1))  # infinity_mass

    epsilon = 0.15
    delta = pmf.get_delta_for_epsilon(epsilon)
    self.assertAlmostEqual(epsilon, pmf.get_epsilon_for_delta(delta))

    self.assertAlmostEqual(np.inf, pmf.get_epsilon_for_delta(0))

  @parameterized.product(
      (
          {
              'tail_mass_truncation': 0,
              'expected_lower_loss': -3,
              'expected_probs': np.array([0.06, 0.31, 0.35]),
              'expected_truncated_to_inf_mass': 0
          },
          {
              'tail_mass_truncation':
                  0.1,  # no truncation 0.1/2 < min(0.06, 0.35)
              'expected_lower_loss': -3,
              'expected_probs': np.array([0.06, 0.31, 0.35]),
              'expected_truncated_to_inf_mass': 0
          },
          {
              'tail_mass_truncation': 0.15,  # truncation from left.
              'expected_lower_loss': -2,
              'expected_probs': np.array([0.37, 0.35]),
              'expected_truncated_to_inf_mass': 0
          },
          {
              'tail_mass_truncation': 0.72,  # truncation from both sides.
              'expected_lower_loss': -2,
              'expected_probs': np.array([0.37]),
              'expected_truncated_to_inf_mass':
                  0.35  # last element goes to inf.
          },
      ),
      dense=(False, True),
  )
  def test_compose_pessimistic(self, tail_mass_truncation, expected_lower_loss,
                               expected_probs, expected_truncated_to_inf_mass,
                               dense):
    discretization = 0.1
    pmf1 = self._create_pmf(
        discretization,
        lower_loss=-1,
        probs=np.array([0.2, 0.7]),
        infinity_mass=0.1,
        dense=dense)
    pmf2 = self._create_pmf(
        discretization,
        lower_loss=-2,
        probs=np.array([0.3, 0.5]),
        infinity_mass=0.2,
        dense=dense)
    pmf = pmf1.compose(pmf2, tail_mass_truncation)

    self.assertEqual(discretization, pmf._discretization)
    if dense:
      self._check_dense_probs(pmf, expected_lower_loss, expected_probs)
    else:
      self._check_sparse_probs(pmf, expected_lower_loss, expected_probs)

    expected_inf_mass = 1 - (1 - 0.1) * (1 -
                                         0.2) + expected_truncated_to_inf_mass
    self.assertAlmostEqual(expected_inf_mass, pmf._infinity_mass)

  @parameterized.product(
      (
          {
              'tail_mass_truncation': 0,
              'expected_lower_loss': -3,
              'expected_probs': np.array([0.06, 0.31, 0.35]),
              'expected_truncated_to_inf_mass': 0,
              'pessimistic_estimate': True
          },
          {
              'tail_mass_truncation': 0,
              'expected_lower_loss': -3,
              'expected_probs': np.array([0.06, 0.31, 0.35]),
              'expected_truncated_to_inf_mass': 0,
              'pessimistic_estimate': False
          },
          {
              'tail_mass_truncation':
                  0.1,  # no truncation 0.1/2 < min(0.06, 0.35)
              'expected_lower_loss': -3,
              'expected_probs': np.array([0.06, 0.31, 0.35]),
              'expected_truncated_to_inf_mass': 0,
              'pessimistic_estimate': True
          },
          {
              'tail_mass_truncation':
                  0.1,  # no truncation 0.1/2 < min(0.06, 0.35)
              'expected_lower_loss': -3,
              'expected_probs': np.array([0.06, 0.31, 0.35]),
              'expected_truncated_to_inf_mass': 0,
              'pessimistic_estimate': False
          },
          {
              'tail_mass_truncation': 0.15,  # truncation the left tail.
              'expected_lower_loss': -2,
              'expected_probs': np.array([0.37, 0.35]),
              'expected_truncated_to_inf_mass': 0,
              'pessimistic_estimate': True
          },
          {
              'tail_mass_truncation': 0.15,  # truncation the left tail.
              'expected_lower_loss': -2,
              'expected_probs': np.array([0.31, 0.35]),
              'expected_truncated_to_inf_mass': 0,
              'pessimistic_estimate': False
          },
          {
              'tail_mass_truncation': 0.72,  # truncation both tails.
              'expected_lower_loss': -2,
              'expected_probs': np.array([0.37]),
              'expected_truncated_to_inf_mass': 0.35,
              'pessimistic_estimate': True
          },
          {
              'tail_mass_truncation': 0.72,  # truncation both tails.
              'expected_lower_loss': -2,
              'expected_probs': np.array([0.66]),
              'expected_truncated_to_inf_mass': 0,
              'pessimistic_estimate': False
          },
      ),
      dense=(False, True))
  def test_compose(self, tail_mass_truncation, expected_lower_loss,
                   expected_probs, expected_truncated_to_inf_mass,
                   pessimistic_estimate, dense):
    discretization = 0.1
    pmf1 = self._create_pmf(
        discretization,
        lower_loss=-1,
        probs=np.array([0.2, 0.7]),
        infinity_mass=0.1,
        dense=dense,
        pessimistic_estimate=pessimistic_estimate)
    pmf2 = self._create_pmf(
        discretization,
        lower_loss=-2,
        probs=np.array([0.3, 0.5]),
        infinity_mass=0.2,
        dense=dense,
        pessimistic_estimate=pessimistic_estimate)
    pmf = pmf1.compose(pmf2, tail_mass_truncation)

    self.assertEqual(discretization, pmf._discretization)
    if dense:
      self._check_dense_probs(pmf, expected_lower_loss, expected_probs)
    else:
      self._check_sparse_probs(pmf, expected_lower_loss, expected_probs)

    expected_inf_mass = 1 - (1 - 0.1) * (1 -
                                         0.2) + expected_truncated_to_inf_mass
    self.assertAlmostEqual(expected_inf_mass, pmf._infinity_mass)

  @parameterized.parameters(False, True)
  def test_compose_different_discretization(self, dense):
    pmf1 = self._create_pmf(discretization=0.1, dense=dense)
    pmf2 = self._create_pmf(discretization=0.2, dense=dense)

    with self.assertRaisesRegex(
        ValueError, 'Discretization intervals are different: 0.1 != 0.2'):
      pmf1.compose(pmf2)

  @parameterized.parameters(False, True)
  def test_compose_different_estimation(self, dense):
    pmf1 = self._create_pmf(
        discretization=0.1, pessimistic_estimate=True, dense=dense)
    pmf2 = self._create_pmf(
        discretization=0.1, pessimistic_estimate=False, dense=dense)

    with self.assertRaisesRegex(ValueError, 'Estimation types are different'):
      pmf1.compose(pmf2)

  @parameterized.product(
      (
          {
              'num_times': 2,
              'tail_mass_truncation': 0,
              'expected_lower_loss': -2,
              'expected_probs': np.array([0.04, 0.28, 0.49]),
              'expected_truncated_to_inf_mass': 0
          },
          {
              'num_times':
                  5,
              'tail_mass_truncation':
                  0,
              'expected_lower_loss':
                  -5,
              'expected_probs':
                  np.array([0.00032, 0.0056, 0.0392, 0.1372, 0.2401, 0.16807]),
              'expected_truncated_to_inf_mass':
                  0
          },
          {
              'num_times': 2,
              'tail_mass_truncation': 0.1,
              'expected_lower_loss': -1,
              'expected_probs': np.array([0.32, 0.49]),
              'expected_truncated_to_inf_mass': 0
          },
          {
              'num_times':
                  5,
              'tail_mass_truncation':
                  0.01,  # truncation left tail.
              'expected_lower_loss':
                  -4,
              'expected_probs':
                  np.array([0.00032 + 0.0056, 0.0392, 0.1372, 0.2401, 0.16807]),
              'expected_truncated_to_inf_mass':
                  0
          },
      ),
      dense=(False, True),
  )
  def test_self_compose(self, num_times, tail_mass_truncation,
                        expected_lower_loss, expected_probs,
                        expected_truncated_to_inf_mass, dense):
    discretization = 0.1
    pmf_input = self._create_pmf(
        discretization,
        lower_loss=-1,
        probs=np.array([0.2, 0.7]),
        infinity_mass=0.1,
        dense=dense)
    pmf_result = pmf_input.self_compose(num_times, tail_mass_truncation)

    self.assertEqual(discretization, pmf_result._discretization)
    expected_inf_mass = 1 - (1 -
                             0.1)**num_times + expected_truncated_to_inf_mass
    self.assertAlmostEqual(expected_inf_mass, pmf_result._infinity_mass)
    if dense:
      self._check_dense_probs(pmf_result, expected_lower_loss, expected_probs)
    else:
      self._check_sparse_probs(pmf_result, expected_lower_loss, expected_probs)

  def test_self_compose_many_times_dense(self):
    discretization = 0.1
    num_times = 50
    tail_mass_truncation = 1e-2
    expected_lower_loss = -24
    expected_probs = np.array([
        0.00550859, 0.00668396, 0.01281092, 0.02267679, 0.03703876, 0.05575728,
        0.07724706, 0.09831444, 0.11470018, 0.12234686, 0.11894834, 0.10501745,
        0.08382972, 0.06018545, 0.03861902, 0.02197841, 0.01098969, 0.00477262
    ])
    expected_truncated_to_inf_mass = 0.00235534610580374
    pmf_input = self._create_pmf(
        discretization,
        lower_loss=-1,
        probs=np.array([0.3, 0.7]),
        infinity_mass=0,
        dense=True)
    pmf_result = pmf_input.self_compose(num_times, tail_mass_truncation)

    self.assertEqual(discretization, pmf_result._discretization)
    self.assertAlmostEqual(expected_truncated_to_inf_mass,
                           pmf_result._infinity_mass)
    self._check_dense_probs(pmf_result, expected_lower_loss, expected_probs)


if __name__ == '__main__':
  unittest.main()
