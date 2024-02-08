# Copyright 2022 Google LLC.
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
"""Tests for PLDPmf."""

import unittest

from absl.testing import parameterized
import numpy as np

from dp_accounting.pld import common
from dp_accounting.pld import pld_pmf
from dp_accounting.pld import test_util


class PLDPmfTest(parameterized.TestCase):

  def _create_pmf(self,
                  discretization: float,
                  dense: bool,
                  infinity_mass: float = 0.0,
                  lower_loss: int = 0,
                  probs: np.ndarray = np.array([1.0]),
                  pessimistic_estimate: bool = True) -> pld_pmf.PLDPmf:
    """Helper function for creating PLD for testing."""
    if dense:
      return pld_pmf.DensePLDPmf(discretization, lower_loss, probs,
                                 infinity_mass, pessimistic_estimate)

    loss_probs = common.list_to_dictionary(probs, lower_loss)
    return pld_pmf.SparsePLDPmf(loss_probs, discretization, infinity_mass,
                                pessimistic_estimate)

  def _check_dense_probs(self, dense_pmf: pld_pmf.DensePLDPmf,
                         expected_lower_loss: int, expected_probs: np.ndarray):
    """Checks that resulting dense pmf satisfies expectations."""
    self.assertEqual(expected_lower_loss, dense_pmf._lower_loss)
    self.assertSequenceAlmostEqual(expected_probs, dense_pmf._probs)

  def _check_sparse_probs(self, sparse_pmf: pld_pmf.SparsePLDPmf,
                          expected_lower_loss: int, expected_probs: np.ndarray):
    """Checks that resulting sparse pmf satisfies expectations."""
    expected_loss_probs = common.list_to_dictionary(expected_probs,
                                                    expected_lower_loss)
    test_util.assert_dictionary_almost_equal(self, expected_loss_probs,
                                             sparse_pmf._loss_probs)

  def _check_sparse_pld_pmf_equal(self, sparse_pmf: pld_pmf.SparsePLDPmf,
                                  discretization: float, infinity_mass: float,
                                  loss_probs: dict[int, float],
                                  pessimistic_estimate: bool):
    self.assertEqual(discretization, sparse_pmf._discretization)
    self.assertAlmostEqual(infinity_mass, sparse_pmf._infinity_mass)
    self.assertEqual(pessimistic_estimate, sparse_pmf._pessimistic_estimate)
    test_util.assert_dictionary_almost_equal(self, loss_probs,
                                             sparse_pmf._loss_probs)

  @parameterized.parameters(False, True)
  def test_delta_for_epsilon(self, dense: bool):
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
  def test_delta_for_epsilon_vectorized(self, dense: bool):
    discretization = 0.1
    infinity_mass = 0.1
    lower_loss = -1
    probs = np.array([0.2, 0.3, 0, 0.4])
    pmf = self._create_pmf(discretization, dense, infinity_mass, lower_loss,
                           probs)
    epsilon = [-np.inf, -20, 0.1, np.inf]
    expected_delta = [
        1, 1, infinity_mass + 0.4 * (1 - np.exp(-0.1)), infinity_mass
    ]

    self.assertSequenceAlmostEqual(expected_delta,
                                   pmf.get_delta_for_epsilon(epsilon))

  def test_delta_for_epsilon_not_sorted(self):
    pmf = self._create_pmf(discretization=0.1, dense=True, infinity_mass=0)
    epsilon = [2.0, 3.0, 1.0]  # not sorted

    with self.assertRaisesRegex(
        ValueError,
        'Epsilons in get_delta_for_epsilon must be sorted in ascending order'):
      pmf.get_delta_for_epsilon(epsilon)

  @parameterized.parameters(False, True)
  def test_get_delta_for_epsilon_for_composed_pld(self, dense):
    discretization = 0.1
    infinity_mass1, lower_loss1, probs1 = 0.1, -1, np.array(
        [0.2, 0.3, 0, 0.1, 0.3])
    infinity_mass2, lower_loss2, probs2 = 0.2, -2, np.array([0.1, 0, 0.4, 0.3])
    pmf1 = self._create_pmf(discretization, dense, infinity_mass1, lower_loss1,
                            probs1)
    pmf2 = self._create_pmf(discretization, dense, infinity_mass2, lower_loss2,
                            probs2)
    pmf_composed = pmf1.compose(pmf2)
    for epsilon in np.linspace(-10, 10, num=100):
      delta1 = pmf1.get_delta_for_epsilon_for_composed_pld(pmf2, epsilon)
      delta2 = pmf_composed.get_delta_for_epsilon(epsilon)
      self.assertAlmostEqual(delta1, delta2, msg=f'{epsilon}')

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
  def test_compose(self, tail_mass_truncation: float, expected_lower_loss: int,
                   expected_probs: np.ndarray,
                   expected_truncated_to_inf_mass: float,
                   pessimistic_estimate: bool, dense: bool):
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
  def test_compose_different_discretization(self, dense: bool):
    pmf1 = self._create_pmf(discretization=0.1, dense=dense)
    pmf2 = self._create_pmf(discretization=0.2, dense=dense)

    with self.assertRaisesRegex(
        ValueError, 'Discretization intervals are different: 0.1 != 0.2'):
      pmf1.compose(pmf2)
    with self.assertRaisesRegex(
        ValueError, 'Discretization intervals are different: 0.1 != 0.2'):
      pmf1.get_delta_for_epsilon_for_composed_pld(pmf2, 1)

  @parameterized.parameters(False, True)
  def test_compose_different_estimation(self, dense: bool):
    pmf1 = self._create_pmf(
        discretization=0.1, pessimistic_estimate=True, dense=dense)
    pmf2 = self._create_pmf(
        discretization=0.1, pessimistic_estimate=False, dense=dense)

    with self.assertRaisesRegex(ValueError, 'Estimation types are different'):
      pmf1.compose(pmf2)
    with self.assertRaisesRegex(ValueError, 'Estimation types are different'):
      pmf1.get_delta_for_epsilon_for_composed_pld(pmf2, 1)

  @parameterized.parameters(
      {
          'num_times': 2,
          'tail_mass_truncation': 0,
          'expected_lower_loss': -2,
          'expected_probs': np.array([0.04, 0.28, 0.49]),
      }, {
          'num_times': 5,
          'tail_mass_truncation': 0,
          'expected_lower_loss': -5,
          'expected_probs':
              np.array([0.00032, 0.0056, 0.0392, 0.1372, 0.2401, 0.16807]),
      }, {
          'num_times': 2,
          'tail_mass_truncation': 0.1,
          'expected_lower_loss': -1,
          'expected_probs': np.array([0.32, 0.49]),
      }, {
          'num_times': 5,
          'tail_mass_truncation': 0.01,  # truncation left tail.
          'expected_lower_loss': -4,
          'expected_probs':
              np.array([0.00032 + 0.0056, 0.0392, 0.1372, 0.2401, 0.16807]),
      },
  )
  def test_self_compose_sparse(self, num_times, tail_mass_truncation,
                               expected_lower_loss, expected_probs):
    discretization = 0.1
    infinity_mass = 0.1
    pmf_input = self._create_pmf(
        discretization,
        lower_loss=-1,
        probs=np.array([0.2, 0.7]),
        infinity_mass=infinity_mass,
        dense=False)
    pmf_result = pmf_input.self_compose(num_times, tail_mass_truncation)

    self.assertEqual(discretization, pmf_result._discretization)
    self.assertAlmostEqual(1 - (1 - infinity_mass)**num_times,
                           pmf_result._infinity_mass)
    self._check_sparse_probs(pmf_result, expected_lower_loss, expected_probs)

  @parameterized.parameters(
      {
          'num_times': 2,
          'tail_mass_truncation': 0,
          'expected_lower_loss': -2,
          'expected_probs': np.array([0.04, 0.28, 0.49]),
      }, {
          'num_times': 5,
          'tail_mass_truncation': 0,
          'expected_lower_loss': -5,
          'expected_probs':
              np.array([0.00032, 0.0056, 0.0392, 0.1372, 0.2401, 0.16807]),
      },
  )
  def test_self_compose_dense_probs(self, num_times, tail_mass_truncation,
                                    expected_lower_loss, expected_probs):
    discretization = 0.1
    infinity_mass = 0.1
    pmf_input = self._create_pmf(
        discretization,
        lower_loss=-1,
        probs=np.array([0.2, 0.7]),
        infinity_mass=infinity_mass,
        dense=True)
    pmf_result = pmf_input.self_compose(num_times, tail_mass_truncation)
    self.assertEqual(discretization, pmf_result._discretization)
    self.assertAlmostEqual(1 - (1 - infinity_mass)**num_times,
                           pmf_result._infinity_mass)
    self._check_dense_probs(pmf_result, expected_lower_loss, expected_probs)

  def test_self_compose_dense_truncation(self):
    discretization = 0.1
    infinity_mass = 0.0001
    num_times = 50
    tail_mass_truncation = 1e-2
    pmf_input = self._create_pmf(
        discretization,
        lower_loss=-1,
        probs=np.array([0.2, 0.7]),
        infinity_mass=infinity_mass,
        dense=True)
    pmf_result = pmf_input.self_compose(num_times, tail_mass_truncation)

    self.assertEqual(discretization, pmf_result._discretization)
    self.assertLessEqual(
        pmf_result._infinity_mass,
        tail_mass_truncation - np.expm1(num_times * np.log1p(-infinity_mass)))

  @parameterized.parameters((1, True), (100, True), (1000, True), (1001, False))
  def test_pmf_creation(self, num_points: int, is_sparse: bool):
    probs = np.ones(num_points) / num_points
    lower_loss = -1
    loss_probs = common.list_to_dictionary(probs, lower_loss)
    discretization = 0.01
    infinity_mass = 0
    pessimistic_estimate = True
    pmf = pld_pmf.create_pmf(loss_probs, discretization, infinity_mass,
                             pessimistic_estimate)

    if is_sparse:
      self.assertIsInstance(pmf, pld_pmf.SparsePLDPmf)
    else:
      self.assertIsInstance(pmf, pld_pmf.DensePLDPmf)

    self.assertEqual(num_points, pmf.size)

  @parameterized.named_parameters(
      dict(testcase_name='empty',
           discretization=0.1,
           rounded_epsilons=np.array([]),
           deltas=np.array([]),
           error_message='unequal or zero'),
      dict(testcase_name='unequal_length',
           discretization=0.1,
           rounded_epsilons=np.array([-1, 1]),
           deltas=np.array([0.5]),
           error_message='unequal or zero'),
      dict(testcase_name='non_sorted_epsilons',
           discretization=0.1,
           rounded_epsilons=np.array([1, -1]),
           deltas=np.array([0, 0.5]),
           error_message='not in strictly increasing order'),
      dict(testcase_name='non_monotone_deltas',
           discretization=0.1,
           rounded_epsilons=np.array([-1, 0, 1]),
           deltas=np.array([0.5, 0, 0.1]),
           error_message='not in non-increasing order'),
      dict(testcase_name='out_of_range_deltas_1',
           discretization=0.1,
           rounded_epsilons=np.array([-1, 0, 1]),
           deltas=np.array([1.1, 0.5, 0.1]),
           error_message='deltas are not between 0 and 1'),
      dict(testcase_name='out_of_range_deltas_2',
           discretization=0.1,
           rounded_epsilons=np.array([-1, 0, 1]),
           deltas=np.array([0.9, 0.5, -0.1]),
           error_message='deltas are not between 0 and 1'),
      )
  def test_connect_dots_pmf_value_errors(
      self, discretization, rounded_epsilons, deltas, error_message):
    with self.assertRaisesRegex(ValueError, error_message):
      pld_pmf.create_pmf_pessimistic_connect_dots(discretization,
                                                  rounded_epsilons,
                                                  deltas)

  @parameterized.named_parameters(
      dict(testcase_name='example1',
           discretization=0.1,
           rounded_epsilons=np.array([0, 1]),
           deltas=np.array([0.4, 0.1])))
  def test_connect_dots_pmf_non_negative(self, discretization,
                                         rounded_epsilons, deltas):
    pmf = pld_pmf.create_pmf_pessimistic_connect_dots(
        discretization, rounded_epsilons, deltas)
    for k, v in pmf._loss_probs.items():
      self.assertGreaterEqual(v, 0, msg=f'Probability for loss={k} is {v} < 0')
    self.assertGreaterEqual(
        pmf._infinity_mass, 0,
        msg=f'Probability for +inf is {pmf._infinity_mass} < 0')

  @parameterized.named_parameters(
      dict(testcase_name='trivial',
           # mu_upper = [1]
           # mu_lower = [1]
           discretization=0.1,
           rounded_epsilons=np.array([0]),
           deltas=np.array([0.0]),
           expected_loss_probs={0: 1.0},
           expected_infinity_mass=0.0),
      dict(testcase_name='rr_eps=0.1',
           # mu_upper = [1/(1+e^{-0.1}), 1/(1+e^{0.1})]
           # mu_lower = [1/(1+e^{0.1}), 1/(1+e^{-0.1})]
           discretization=0.1,
           rounded_epsilons=np.array([-1, 1]),
           deltas=np.array([1 - np.exp(-0.1), 0.0]),
           expected_loss_probs={-1: 1/(1 + np.exp(0.1)),
                                1: 1/(1 + np.exp(-0.1))},
           expected_infinity_mass=0.0),
      dict(testcase_name='rr_eps=0.1_abort_w_p_0.5',
           # mu_upper = [0.5/(1+e^{-0.1}), 0.5, 0.5/(1+e^{0.1})]
           # mu_lower = [0.5/(1+e^{0.1}), 0.5, 0.5/(1+e^{-0.1})]
           discretization=0.1,
           rounded_epsilons=np.array([-1, 0, 1]),
           deltas=np.array([1 - np.exp(-0.1),
                            0.5*(np.exp(0.1)-1)/(np.exp(0.1)+1),
                            0.0]),
           expected_loss_probs={-1: 0.5/(1 + np.exp(0.1)),
                                0: 0.5,
                                1: 0.5/(1 + np.exp(-0.1))},
           expected_infinity_mass=0.0),
      dict(testcase_name='rr_eps=0.1_delta=0.5',
           # mu_upper = [0.5, 0.5/(1+e^{-0.1}), 0.5/(1+e^{0.1}), 0.0]
           # mu_lower = [0.0, 0.5/(1+e^{0.1}), 0.5/(1+e^{-0.1}), 0.5]
           discretization=0.1,
           rounded_epsilons=np.array([-1, 0, 1]),
           deltas=np.array([1 - 0.5*np.exp(-0.1),
                            0.5 + 0.5*(np.exp(0.1)-1)/(np.exp(0.1)+1),
                            0.5]),
           expected_loss_probs={-1: 0.5/(1 + np.exp(0.1)),
                                0: 0.0,
                                1: 0.5/(1 + np.exp(-0.1))},
           expected_infinity_mass=0.5),
      )
  def test_connect_dots_pmf_creation(
      self, discretization, rounded_epsilons, deltas,
      expected_loss_probs, expected_infinity_mass):
    pmf = pld_pmf.create_pmf_pessimistic_connect_dots(discretization,
                                                      rounded_epsilons,
                                                      deltas)
    self._check_sparse_pld_pmf_equal(pmf,
                                     discretization,
                                     expected_infinity_mass,
                                     expected_loss_probs,
                                     pessimistic_estimate=True)

  @parameterized.named_parameters(
      dict(testcase_name='epsilon_empty',
           discretization=0.1,
           rounded_epsilon_lower=1,
           rounded_epsilon_upper=0,
           deltas=np.array([]),
           error_message='is smaller than'),
      dict(testcase_name='unequal_length',
           discretization=0.1,
           rounded_epsilon_lower=0,
           rounded_epsilon_upper=1,
           deltas=np.array([0.5]),
           error_message='Length of deltas is not equal to number of epsilons'),
      dict(testcase_name='non_monotone_deltas',
           discretization=0.1,
           rounded_epsilon_lower=-1,
           rounded_epsilon_upper=1,
           deltas=np.array([0.5, 0, 0.1]),
           error_message='not in non-increasing order'),
      dict(testcase_name='out_of_range_deltas_1',
           discretization=0.1,
           rounded_epsilon_lower=-1,
           rounded_epsilon_upper=1,
           deltas=np.array([1.1, 0.5, 0.1]),
           error_message='deltas are not between 0 and 1'),
      dict(testcase_name='out_of_range_deltas_2',
           discretization=0.1,
           rounded_epsilon_lower=-1,
           rounded_epsilon_upper=1,
           deltas=np.array([0.9, 0.5, -0.1]),
           error_message='deltas are not between 0 and 1'),
      )
  def test_connect_dots_pmf_fixed_gap_value_errors(
      self, discretization, rounded_epsilon_lower, rounded_epsilon_upper,
      deltas, error_message):
    with self.assertRaisesRegex(ValueError, error_message):
      pld_pmf.create_pmf_pessimistic_connect_dots_fixed_gap(
          discretization, rounded_epsilon_lower, rounded_epsilon_upper, deltas)

  @parameterized.named_parameters(
      dict(testcase_name='example1',
           discretization=0.1,
           rounded_epsilon_lower=0,
           rounded_epsilon_upper=1,
           deltas=np.array([0.4, 0.1])))
  def test_connect_dots_pmf_fixed_gap_non_negative(
      self, discretization, rounded_epsilon_lower, rounded_epsilon_upper,
      deltas):
    pmf = pld_pmf.create_pmf_pessimistic_connect_dots_fixed_gap(
        discretization, rounded_epsilon_lower, rounded_epsilon_upper, deltas)
    for k, v in pmf._loss_probs.items():
      self.assertGreaterEqual(v, 0, msg=f'Probability for loss={k} is {v} < 0')
    self.assertGreaterEqual(
        pmf._infinity_mass, 0,
        msg=f'Probability for +inf is {pmf._infinity_mass} < 0')

  @parameterized.named_parameters(
      dict(testcase_name='trivial',
           # mu_upper = [1]
           # mu_lower = [1]
           discretization=0.1,
           rounded_epsilon_lower=0,
           rounded_epsilon_upper=0,
           deltas=np.array([0.0]),
           expected_loss_probs={0: 1.0},
           expected_infinity_mass=0.0),
      dict(testcase_name='rr_eps=0.1',
           # mu_upper = [1/(1+e^{-0.1}), 1/(1+e^{0.1})]
           # mu_lower = [1/(1+e^{0.1}), 1/(1+e^{-0.1})]
           discretization=0.1,
           rounded_epsilon_lower=-1,
           rounded_epsilon_upper=1,
           deltas=np.array([1 - np.exp(-0.1),
                            (1 - np.exp(-0.1))/(1 + np.exp(-0.1)),
                            0.0]),
           expected_loss_probs={-1: 1/(1 + np.exp(0.1)),
                                1: 1/(1 + np.exp(-0.1))},
           expected_infinity_mass=0.0),
      dict(testcase_name='rr_eps=0.1_abort_w_p_0.5',
           # mu_upper = [0.5/(1+e^{-0.1}), 0.5, 0.5/(1+e^{0.1})]
           # mu_lower = [0.5/(1+e^{0.1}), 0.5, 0.5/(1+e^{-0.1})]
           discretization=0.1,
           rounded_epsilon_lower=-1,
           rounded_epsilon_upper=1,
           deltas=np.array([1 - np.exp(-0.1),
                            0.5*(np.exp(0.1)-1)/(np.exp(0.1)+1),
                            0.0]),
           expected_loss_probs={-1: 0.5/(1 + np.exp(0.1)),
                                0: 0.5,
                                1: 0.5/(1 + np.exp(-0.1))},
           expected_infinity_mass=0.0),
      dict(testcase_name='rr_eps=0.1_delta=0.5',
           # mu_upper = [0.5, 0.5/(1+e^{-0.1}), 0.5/(1+e^{0.1}), 0.0]
           # mu_lower = [0.0, 0.5/(1+e^{0.1}), 0.5/(1+e^{-0.1}), 0.5]
           discretization=0.1,
           rounded_epsilon_lower=-1,
           rounded_epsilon_upper=1,
           deltas=np.array([1 - 0.5*np.exp(-0.1),
                            0.5 + 0.5*(np.exp(0.1)-1)/(np.exp(0.1)+1),
                            0.5]),
           expected_loss_probs={-1: 0.5/(1 + np.exp(0.1)),
                                0: 0.0,
                                1: 0.5/(1 + np.exp(-0.1))},
           expected_infinity_mass=0.5),
      )
  def test_connect_dots_fixed_gap_pmf_creation(
      self, discretization, rounded_epsilon_lower, rounded_epsilon_upper,
      deltas, expected_loss_probs, expected_infinity_mass):
    pmf = pld_pmf.create_pmf_pessimistic_connect_dots_fixed_gap(
        discretization, rounded_epsilon_lower, rounded_epsilon_upper, deltas)
    self._check_sparse_pld_pmf_equal(pmf,
                                     discretization,
                                     expected_infinity_mass,
                                     expected_loss_probs,
                                     pessimistic_estimate=True)

  @parameterized.parameters((1, 100, True), (10, 100, True), (1, 1001, False),
                            (10, 101, False), (1001, 1, False),
                            (1001, 1001, False))
  def test_compose_pmfs(self, num_points1, num_points2, is_sparse):
    lower_loss = -1
    discretization = 0.01
    infinity_mass = 0
    pessimistic_estimate = True
    probs1 = np.ones(num_points1) / num_points1
    loss_probs1 = common.list_to_dictionary(probs1, lower_loss)
    pmf1 = pld_pmf.create_pmf(loss_probs1, discretization, infinity_mass,
                              pessimistic_estimate)

    probs2 = np.ones(num_points2) / num_points2
    loss_probs2 = common.list_to_dictionary(probs2, lower_loss)
    pmf2 = pld_pmf.create_pmf(loss_probs2, discretization, infinity_mass,
                              pessimistic_estimate)

    pmf = pld_pmf.compose_pmfs(pmf1, pmf2)

    if is_sparse:
      self.assertIsInstance(pmf, pld_pmf.SparsePLDPmf)
    else:
      self.assertIsInstance(pmf, pld_pmf.DensePLDPmf)


if __name__ == '__main__':
  unittest.main()
