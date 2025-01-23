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

import math
import random
from typing import Optional
import unittest

from absl.testing import parameterized
import numpy as np
from scipy import stats

from dp_accounting.pld import common
from dp_accounting.pld import privacy_loss_mechanism
from dp_accounting.pld import test_util


ADD = privacy_loss_mechanism.AdjacencyType.ADD
REM = privacy_loss_mechanism.AdjacencyType.REMOVE


def _assert_connect_bounds_equal(
    testcase: parameterized.TestCase,
    connect_dots_bounds: privacy_loss_mechanism.ConnectDotsBounds,
    epsilon_upper: Optional[float],
    epsilon_lower: Optional[float],
    lower_x: Optional[int],
    upper_x: Optional[int]) -> None:
  testcase.assertAlmostEqual(connect_dots_bounds.epsilon_upper, epsilon_upper)
  testcase.assertAlmostEqual(connect_dots_bounds.epsilon_lower, epsilon_lower)
  testcase.assertEqual(connect_dots_bounds.lower_x, lower_x)
  testcase.assertEqual(connect_dots_bounds.upper_x, upper_x)


class LaplacePrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, [-10, 0, 1, 10],
       [2.2699965e-05, 5.0000000e-01, 8.1606028e-01, 9.9997730e-01]),
      (2.0, 1.0, 1.0, ADD, [-10, 0, 1, 10],
       [3.3689735e-03, 5.0000000e-01, 6.9673467e-01, 9.9663103e-01]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, [-10, 0, 1, 10],
       [2.2699965e-05, 5.0000000e-01, 8.1606028e-01, 9.9997730e-01]),
      (2.0, 1.0, 0.2, ADD, [-10, 0, 1, 10],
       [3.3689735e-03, 5.0000000e-01, 6.9673467e-01, 9.9663103e-01]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, [-10, 0, 1, 10],
       [6.1704902e-05, 8.1606028e-01, 9.3233236e-01, 9.9999165e-01]),
      (2.0, 1.0, 1.0, REM, [-10, 0, 1, 10],
       [5.5544983e-03, 6.9673467e-01, 8.1606028e-01, 9.9795661e-01]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, [-10, 0, 1, 10],
       [5.3903915e-05, 7.5284822e-01, 9.0907794e-01, 9.9998878e-01]),
      (2.0, 1.0, 0.2, REM, [-10, 0, 1, 10],
       [3.8060785e-03, 5.3934693e-01, 7.2059979e-01, 9.9689614e-01]),
  )
  def test_mu_upper_cdf(
      self, parameter, sensitivity, sampling_prob, adjacency_type,
      x, expected_mu_upper_cdf):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertSequenceAlmostEqual(pl.mu_upper_cdf(x), expected_mu_upper_cdf)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, [0, 1, 2, 30],
       [-1.69314718, -0.69314718, -0.20326705, -1.27231559e-13]),
      (2.0, 1.0, 1.0, ADD, [-1, 0, 1, 30],
       [-1.69314718, -1.19314718, -0.69314718, -2.52173863e-07]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, [0, 1, 2, 30],
       [-1.39775265, -0.57409907, -0.17516956, -1.11161080e-13]),
      (2.0, 1.0, 0.2, ADD, [-1, 0, 1, 30],
       [-1.27511009, -0.77511009, -0.41948127, -1.72795709e-07]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, [-1, 0, 1, 30],
       [-1.69314718, -0.69314718, -0.20326705, -4.67403893e-14]),
      (2.0, 1.0, 1.0, REM, [-2, -1, 0, 30],
       [-1.69314718, -1.19314718, -0.69314718, -1.52951172e-07]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, [-1, 0, 1, 30],
       [-1.69314718, -0.69314718, -0.20326705, -4.67403893e-14]),
      (2.0, 1.0, 0.2, REM, [-2, -1, 0, 30],
       [-1.69314718, -1.19314718, -0.69314718, -1.52951172e-07]),
  )
  def test_mu_lower_log_cdf(
      self, parameter, sensitivity, sampling_prob, adjacency_type,
      x, expected_mu_lower_log_cdf):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertSequenceAlmostEqual(pl.mu_lower_log_cdf(x),
                                   expected_mu_lower_log_cdf)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, -0.1, 1.0), (1.0, 1.0, 1.0, ADD, 2.0, -1.0),
      (1.0, 1.0, 1.0, ADD, 0.3, 0.4), (4.0, 4.0, 1.0, ADD, -0.4, 1.0),
      (5.0, 5.0, 1.0, ADD, 7.0, -1.0), (7.0, 7.0, 1.0, ADD, 2.1, 0.4),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 1.1, -0.86483972516319),
      (2.0, 1.0, 0.2, ADD, -0.2, 0.0819629071393439),
      (1.0, 1.0, 0.5, ADD, 0.5, 0.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, -1.1, 1.0), (1.0, 1.0, 1.0, REM, 1.0, -1.0),
      (1.0, 1.0, 1.0, REM, -0.7, 0.4), (4.0, 4.0, 1.0, REM, -4.4, 1.0),
      (5.0, 5.0, 1.0, REM, 2.0, -1.0), (7.0, 7.0, 1.0, REM, -4.9, 0.4),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, -1.1, 0.86483972516319),
      (2.0, 1.0, 0.2, REM, 0.2, -0.0819629071393439),
      (1.0, 1.0, 0.5, REM, -0.5, 0.0))
  def test_laplace_privacy_loss(self, parameter, sensitivity, sampling_prob,
                                adjacency_type, x, expected_privacy_loss):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 1.0, 0.0), (1.0, 1.0, 1.0, ADD, -1.0, math.inf),
      (1.0, 1.0, 1.0, ADD, 0.4, 0.3), (4.0, 4.0, 1.0, ADD, 1.0, 0.0),
      (5.0, 5.0, 1.0, ADD, -1.0, math.inf), (7.0, 7.0, 1.0, ADD, 0.4, 2.1),
      (1.0, 1.0, 1.0, ADD, 2.0, -math.inf), (3, 1, 1, ADD, 3.1, -math.inf),
      (4.0, 4.0, 1.0, ADD, 1.1, -math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, -0.8649, math.inf),
      (1.0, 1.0, 0.7, ADD, 1.0, -math.inf),
      (2.0, 1.0, 0.2, ADD, 0.0819629071393439, 0),
      (1.0, 1.0, 0.5, ADD, 0.0, 0.5),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 1.0, -1.0), (1.0, 1.0, 1.0, REM, -1.0, math.inf),
      (1.0, 1.0, 1.0, REM, 0.4, -0.7), (4.0, 4.0, 1.0, REM, 1.0, -4.0),
      (5.0, 5.0, 1.0, REM, -1.0, math.inf), (7.0, 7.0, 1.0, REM, 0.4, -4.9),
      (1.0, 1.0, 1.0, REM, 2.0, -math.inf),
      (3.0, 1.0, 1.0, REM, 3.1, -math.inf),
      (4.0, 4.0, 1.0, REM, 1.1, -math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, 0.86483972516319, -1.0),
      (2.0, 1.0, 0.2, REM, -0.082, math.inf),
      (1.0, 1.0, 0.7, REM, 1.0, -math.inf),
      (1.0, 1.0, 0.5, REM, 0.0, -0.5))
  def test_laplace_inverse_privacy_loss(self, parameter, sensitivity,
                                        sampling_prob, adjacency_type,
                                        privacy_loss, expected_x):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 0.0, 1.0, {1: 0.5, -1: 0.18393972}),
      (3.0, 3.0, 1.0, ADD, 0.0, 3.0, {1: 0.5, -1: 0.18393972}),
      (1.0, 2.0, 1.0, ADD, 0.0, 2.0, {2: 0.5, -2: 0.06766764}),
      (4.0, 8.0, 1.0, ADD, 0.0, 8.0, {2: 0.5, -2: 0.06766764}),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 0.0, 1.0, {
          0.7046054708796524: 0.5,
          -0.864839725163191: 0.18393972
      }),
      (3.0, 3.0, 0.6, ADD, 0.0, 3.0, {
          0.4768628363884146: 0.5,
          -0.7085130668623151: 0.18393972
      }),
      (1.0, 2.0, 0.7, ADD, 0.0, 2.0, {
          0.929541389699331: 0.5,
          -1.699706179357965: 0.06766764
      }),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, -1.0, 0.0, {1: 0.5, -1: 0.18393972}),
      (3.0, 3.0, 1.0, REM, -3.0, 0.0, {1: 0.5, -1: 0.18393972}),
      (1.0, 2.0, 1.0, REM, -2.0, 0.0, {2: 0.5, -2: 0.06766764}),
      (4.0, 8.0, 1.0, REM, -8.0, 0.0, {2: 0.5, -2: 0.06766764}),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, -1.0, 0.0, {
          0.864839725163191: 0.4367879441171443,
          -0.7046054708796524: 0.2471517764685769
      }),
      (3.0, 3.0, 0.6, REM, -3.0, 0.0, {
          0.7085130668623151: 0.3735758882342885,
          -0.4768628363884146: 0.3103638323514328
      }),
      (1.0, 2.0, 0.7, REM, -2.0, 0.0, {
          1.699706179357965: 0.3703002924854919,
          -0.929541389699331: 0.1973673491328145
      }))
  def test_laplace_privacy_loss_tail(self, parameter, sensitivity,
                                     sampling_prob, adjacency_type,
                                     expected_lower_x_truncation,
                                     expected_upper_x_truncation,
                                     expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.assert_dictionary_almost_equal(
        self, expected_tail_probability_mass_function,
        tail_pld.tail_probability_mass_function)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 1.0, -1.0),
      (1.0, 2.0, 1.0, ADD, 2.0, -2.0),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 0.704605471, -0.864839725),
      (3.0, 3.0, 0.6, ADD, 0.476862836, -0.708513067),
      (1.0, 2.0, 0.7, ADD, 0.929541390, -1.699706179),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 1.0, -1.0),
      (1.0, 2.0, 1.0, REM, 2.0, -2.0),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, 0.864839725, -0.704605471),
      (3.0, 3.0, 0.6, REM, 0.708513067, -0.476862836),
      (1.0, 2.0, 0.7, REM, 1.699706179, -0.929541390))
  def test_laplace_connect_dots_bounds(self, parameter, sensitivity,
                                       sampling_prob, adjacency_type,
                                       expected_epsilon_upper,
                                       expected_epsilon_lower):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    _assert_connect_bounds_equal(self, pl.connect_dots_bounds(),
                                 expected_epsilon_upper, expected_epsilon_lower,
                                 None, None)

  @parameterized.parameters((-3.0, 1.0, 1.0, ADD), (0.0, 1.0, 1.0, ADD),
                            (1.0, 0.0, 1.0, REM), (2.0, -1.0, 1.0, REM),
                            (2.0, 1.0, 0.0, ADD), (1.0, 1.0, 1.2, REM),
                            (2.0, 1.0, -0.1, REM))
  def test_laplace_value_errors(self,
                                parameter,
                                sensitivity,
                                sampling_prob=1.0,
                                adjacency_type=ADD):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.LaplacePrivacyLoss(
          parameter,
          sensitivity=sensitivity,
          sampling_prob=sampling_prob,
          adjacency_type=adjacency_type)

  @parameterized.parameters((1.0, 1.0, 1.0, 1.1), (1.0, 1.0, 1.0, -0.1),
                            (1.0, 0.0, 1.0, 0.1), (1.0, -0.2, 1.0, 0.1),
                            (1.0, 1.1, 1.0, 0.2))
  def test_laplace_from_privacy_parameters_value_errors(
      self, sensitivity, sampling_prob, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity, sampling_prob=sampling_prob)

  @parameterized.parameters((1.0, 1.0, ADD, 1.0, 0.0, 1.0),
                            (1.0, 1.0, ADD, 1.0, 0.1, 1.0),
                            (2.0, 1.0, REM, 1.0, 0.01, 2.0),
                            (1.0, 1.0, REM, 3.0, 0.01, 0.33333333),
                            (1.0, 0.8, ADD, 1.0, 0.0, 0.8720521537764049),
                            (1.0, 0.5, REM, 1.0, 0.1, 0.671194938966816),
                            (2.0, 0.9, ADD, 1.0, 0.01, 1.8728716669259162),
                            (1.0, 0.7, REM, 3.0, 0.01, 0.2992554981396725))
  def test_laplace_from_privacy_parameters(self, sensitivity, sampling_prob,
                                           adjacency_type,
                                           epsilon, delta,
                                           expected_parameter):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss.from_privacy_guarantee(
        common.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_parameter, pl.parameter)
    self.assertEqual(adjacency_type, pl.adjacency_type)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 1.0, 0.0), (3.0, 3.0, 1.0, ADD, 1.0, 0.0),
      (2.0, 4.0, 1.0, ADD, 2.0, 0.0),
      (2.0, 4.0, 1.0, ADD, 0.5, 0.52763345),
      (1.0, 1.0, 1.0, ADD, 0.0, 0.39346934),
      (2.0, 2.0, 1.0, ADD, 0.0, 0.39346934),
      (1.0, 1.0, 1.0, ADD, -2.0, 0.86466472),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 1.0, 0.0),
      (2.0, 4.0, 0.8, ADD, 0.5, 0.3243606497234246),
      (1.0, 1.0, 0.6, ADD, 0.2, 0.1401134521354217),
      (2.0, 2.0, 0.3, ADD, 0.0, 0.1180408020862099),
      (5.0, 5.0, 0.2, ADD, 2.0, 0.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 1.0, 0.0), (3.0, 3.0, 1.0, REM, 1.0, 0.0),
      (2.0, 4.0, 1.0, REM, 2.0, 0.0), (2.0, 4.0, 1.0, REM, 0.5, 0.52763345),
      (1.0, 1.0, 1.0, REM, 0.0, 0.39346934),
      (2.0, 2.0, 1.0, REM, 0.0, 0.39346934),
      (1.0, 1.0, 1.0, REM, -2.0, 0.86466472),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE)
      (1.0, 1.0, 0.8, REM, 1.0, 0.0),
      (2.0, 4.0, 0.8, REM, 0.5, 0.4039564635032081),
      (1.0, 1.0, 0.6, REM, 0.2, 0.1741992102060086),
      (2.0, 2.0, 0.3, REM, 0.0, 0.1180408020862099),
      (5.0, 5.0, 0.2, REM, -0.25, 0.2211992169285951))
  def test_laplace_get_delta_for_epsilon(self, parameter, sensitivity,
                                         sampling_prob, adjacency_type, epsilon,
                                         expected_delta):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, [-2.0, -1.0, 0.0, 1.0],
       [0.86466472, 0.63212056, 0.39346934, 0.0]),
      (2.0, 4.0, 1.0, ADD, [-0.5, 0.0, 0.5, 1.0, 2.0],
       [0.7134952031, 0.6321205588, 0.5276334473, 0.3934693403, 0.0]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.2, ADD, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
       [0.2591817793, 0.2008511573, 0.1405456165, 0.07869386806,
        0.01879989609, 0.0, 0.0]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, [-2.0, -1.0, 0.0, 1.0],
       [0.86466472, 0.63212056, 0.39346934, 0.0]),
      (2.0, 4.0, 1.0, REM, [-0.5, 0.0, 0.5, 1.0, 2.0],
       [0.7134952031, 0.6321205588, 0.5276334473, 0.3934693403, 0.0]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE)
      (1.0, 1.0, 0.2, REM, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
       [0.2591817793, 0.1812692469, 0.1121734314, 0.07869386806,
        0.05015600993, 0.02391739939, 0.0]),
      )
  def test_laplace_get_delta_for_epsilon_vectorized(
      self, parameter, sensitivity, sampling_prob, adjacency_type,
      epsilon_values, expected_delta_values):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    for expected_delta, delta in zip(expected_delta_values,
                                     pl.get_delta_for_epsilon(epsilon_values)):
      self.assertAlmostEqual(expected_delta, delta)


class GaussianPrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, [-10, 0, 1, 10],
       [7.61985302416047e-24, 0.5, 0.8413447460685429, 1.0]),
      (2.0, 1.0, 1.0, ADD, [-10, 0, 1, 10],
       [2.866515718791933e-07, 0.5, 0.6914624612740131, 0.9999997133484281]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, [-10, 0, 1, 10],
       [7.61985302416047e-24, 0.5, 0.8413447460685429, 1.0]),
      (2.0, 1.0, 0.2, ADD, [-10, 0, 1, 10],
       [2.866515718791933e-07, 0.5, 0.6914624612740131, 0.9999997133484281]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, [-10, 0, 1, 10],
       [1.1285884059538324e-19, 0.8413447460685429, 0.9772498680518208, 1.0]),
      (2.0, 1.0, 1.0, REM, [-10, 0, 1, 10],
       [3.39767312473e-06, 0.6914624612, 0.8413447460, 0.9999999810]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, [-10, 0, 1, 10],
       [9.028859644691143e-20, 0.7730757968548344, 0.9500688436551652, 1.0]),
      (2.0, 1.0, 0.2, REM, [-10, 0, 1, 10],
       [9.08855882449e-07, 0.53829249225, 0.7214389182, 0.99999976688083]),
  )
  def test_mu_upper_cdf(
      self, standard_deviation, sensitivity, sampling_prob, adjacency_type,
      x, expected_mu_upper_cdf):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertSequenceAlmostEqual(pl.mu_upper_cdf(x), expected_mu_upper_cdf)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, [-10, 0, 1, 10],
       [-6.38249341e+01, -1.84102165e+00, -6.93147181e-01, -1.12858841e-19]),
      (2.0, 1.0, 1.0, ADD, [-10, 0, 1, 10],
       [-1.77793764e+01, -1.17591176e+00, -6.93147181e-01, -3.39767890e-06]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, [-10, 0, 1, 10],
       [-54.84062277, -1.48313922, -0.56516047, 0.]),
      (2.0, 1.0, 0.2, ADD, [-10, 0, 1, 10],
       [-1.52717161e+01, -7.72823689e-01, -4.25917894e-01, -9.08856295e-07]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, [-10, 0, 1, 10],
       [-5.32312852e+01, -6.93147181e-01, -0.172753779, -7.61985302e-24]),
      (2.0, 1.0, 1.0, REM, [-10, 0, 1, 10],
       [-1.50649984e+01, -6.93147181e-01, -3.68946415e-01, -2.86651613e-07]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, [-10, 0, 1, 10],
       [-5.32312852e+01, -6.93147181e-01, -1.72753779e-01, -7.61985302e-24]),
      (2.0, 1.0, 0.2, REM, [-10, 0, 1, 10],
       [-1.50649984e+01, -6.93147181e-01, -3.68946415e-01, -2.86651613e-07]),
  )
  def test_mu_lower_log_cdf(
      self, standard_deviation, sensitivity, sampling_prob, adjacency_type,
      x, expected_mu_lower_log_cdf):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertSequenceAlmostEqual(pl.mu_lower_log_cdf(x),
                                   expected_mu_lower_log_cdf)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 5.0, -4.5), (1.0, 1.0, 1.0, ADD, -3.0, 3.5),
      (1.0, 2.0, 1.0, ADD, 3.0, -4.0),
      (4.0, 4.0, 1.0, ADD, 20.0, -4.5), (5.0, 5.0, 1.0, ADD, -15.0, 3.5),
      (7.0, 14.0, 1.0, ADD, 21.0, -4.0),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 0.5, 0.0),
      (1.0, 1.0, 0.5, ADD, -4, 0.6820994357113515),
      (1.0, 2.0, 0.7, ADD, 0, 0.929541389699331),
      (4.0, 4.0, 0.3, ADD, -16, 0.3519252431310541),
      (5.0, 5.0, 0.45, ADD, 20, -2.737735427805667),
      (7.0, 14.0, 0.9, ADD, -7, 2.150000710600199),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 4.0, -4.5), (1.0, 1.0, 1.0, REM, -4.0, 3.5),
      (1.0, 2.0, 1.0, REM, 1.0, -4.0), (4.0, 4.0, 1.0, REM, 16.0, -4.5),
      (5.0, 5.0, 1.0, REM, -20.0, 3.5), (7.0, 14.0, 1.0, REM, 7.0, -4.0),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, -0.5, 0.0),
      (1.0, 1.0, 0.5, REM, 4.0, -0.6820994357113515),
      (1.0, 2.0, 0.7, REM, 0.0, -0.929541389699331),
      (4.0, 4.0, 0.3, REM, 16.0, -0.3519252431310541),
      (5.0, 5.0, 0.45, REM, -20.0, 2.737735427805667),
      (7.0, 14.0, 0.9, REM, 7.0, -2.150000710600199))
  def test_gaussian_privacy_loss(self, standard_deviation, sensitivity,
                                 sampling_prob, adjacency_type, x,
                                 expected_privacy_loss):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, -4.5, 5.0), (1.0, 1.0, 1.0, ADD, 3.5, -3.0),
      (1.0, 2.0, 1.0, ADD, -4.0, 3.0),
      (4.0, 4.0, 1.0, ADD, -4.5, 20.0), (5.0, 5.0, 1.0, ADD, 3.5, -15.0),
      (7.0, 14.0, 1.0, ADD, -4.0, 21.0),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 0.0, 0.5),
      (1.0, 1.0, 0.5, ADD, 0.6820994357113515, -4.0),
      (1.0, 2.0, 0.7, ADD, 0.929541389699331, 0.0),
      (4.0, 4.0, 0.3, ADD, 0.3519252431310541, -16.0),
      (5.0, 5.0, 0.45, ADD, -2.737735427805667, 20.0),
      (7.0, 14.0, 0.9, ADD, 2.150000710600199, -7.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, -4.5, 4.0), (1.0, 1.0, 1.0, REM, 3.5, -4.0),
      (1.0, 2.0, 1.0, REM, -4.0, 1.0), (4.0, 4.0, 1.0, REM, -4.5, 16.0),
      (5.0, 5.0, 1.0, REM, 3.5, -20.0), (7.0, 14.0, 1.0, REM, -4.0, 7.0),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, 0.0, -0.5),
      (1.0, 1.0, 0.5, REM, -0.6820994357113515, 4.0),
      (1.0, 2.0, 0.7, REM, -0.929541389699331, 0.0),
      (4.0, 4.0, 0.3, REM, -0.3519252431310541, 16.0),
      (5.0, 5.0, 0.45, REM, 2.737735427805667, -20.0),
      (7.0, 14.0, 0.9, REM, -2.150000710600199, 7.0))
  def test_gaussian_inverse_privacy_loss(self, standard_deviation, sensitivity,
                                         sampling_prob, adjacency_type,
                                         privacy_loss, expected_x):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, -1.0, 2.0, True, {
          math.inf: 0.15865525,
          -1.5: 0.02275013
      }),
      (3.0, 3.0, 1.0, ADD, -3.0, 6.0, True, {
          math.inf: 0.15865525,
          -1.5: 0.02275013
      }),
      (1.0, 2.0, 1.0, ADD, -1.0, 3.0, True, {
          math.inf: 0.15865525,
          -4.0: 0.00134989
      }),
      (4.0, 8.0, 1.0, ADD, -4.0, 12.0, True, {
          math.inf: 0.15865525,
          -4.0: 0.00134989
      }),
      (1.0, 1.0, 1.0, ADD, -1.0, 2.0, False, {
          1.5: 0.15865525,
      }),
      (3.0, 3.0, 1.0, ADD, -3.0, 6.0, False, {
          1.5: 0.15865525,
      }),
      (1.0, 2.0, 1.0, ADD, -1.0, 3.0, False, {
          4.0: 0.15865525,
      }),
      (4.0, 8.0, 1.0, ADD, -4.0, 12.0, False, {
          4.0: 0.15865525,
      }),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, -1.0, 2.0, True, {
          math.inf: 0.15865525,
          -1.331139: 0.02275013
      }),
      (3.0, 3.0, 0.8, ADD, -3.0, 6.0, True, {
          math.inf: 0.15865525,
          -1.331139: 0.02275013
      }),
      (1.0, 2.0, 0.5, ADD, -1.0, 3.0, True, {
          math.inf: 0.15865525,
          -3.325003: 0.00134990
      }),
      (4.0, 8.0, 0.6, ADD, -4.0, 12.0, True, {
          math.inf: 0.15865525,
          -3.501311: 0.00134990
      }),
      (1.0, 1.0, 0.9, ADD, -1.0, 2.0, False, {
          1.20125: 0.15865525,
      }),
      (3.0, 3.0, 0.7, ADD, -3.0, 6.0, False, {
          0.784843: 0.15865525,
      }),
      (1.0, 2.0, 0.4, ADD, -1.0, 3.0, False, {
          0.498689: 0.15865525,
      }),
      (4.0, 8.0, 0.2, ADD, -4.0, 12.0, False, {
          0.218575: 0.15865525,
      }),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, -2.0, 1.0, True, {
          math.inf: 0.15865525,
          -1.5: 0.02275013
      }),
      (3.0, 3.0, 1.0, REM, -6.0, 3.0, True, {
          math.inf: 0.15865525,
          -1.5: 0.02275013
      }),
      (1.0, 2.0, 1.0, REM, -3.0, 1.0, True, {
          math.inf: 0.15865525,
          -4.0: 0.00134989
      }),
      (4.0, 8.0, 1.0, REM, -12.0, 4.0, True, {
          math.inf: 0.15865525,
          -4.0: 0.00134989
      }),
      (1.0, 1.0, 1.0, REM, -2.0, 1.0, False, {
          1.5: 0.15865525,
      }),
      (3.0, 3.0, 1.0, REM, -6.0, 3.0, False, {
          1.5: 0.15865525,
      }),
      (1.0, 2.0, 1.0, REM, -3.0, 1.0, False, {
          4.0: 0.15865525,
      }),
      (4.0, 8.0, 1.0, REM, -12.0, 4.0, False, {
          4.0: 0.15865525,
      }),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, -2.0, 1.0, True, {
          math.inf: 0.1314742295348015,
          -0.971528299641668: 0.0499311563448348
      }),
      (3.0, 3.0, 0.8, REM, -6.0, 3.0, True, {
          math.inf: 0.1314742295348015,
          -0.971528299641668: 0.0499311563448348
      }),
      (1.0, 2.0, 0.5, REM, -3.0, 1.0, True, {
          math.inf: 0.0800025759815436,
          -0.6749972526421355: 0.0800025759815436,
      }),
      (4.0, 8.0, 0.6, REM, -12.0, 4.0, True, {
          math.inf: 0.0957331115715263,
          -0.88918789612552: 0.06427204039156087
      }),
      (1.0, 1.0, 0.9, REM, -2.0, 1.0, False, {
          1.419129383720773: 0.1450647417331293,
      }),
      (3.0, 3.0, 0.7, REM, -6.0, 3.0, False, {
          1.23465205122806: 0.1178837173364737,
      }),
      (1.0, 2.0, 0.4, REM, -3.0, 1.0, False, {
          3.110812103874479: 0.06427204039156088,
      }),
      (4.0, 8, 0.2, REM, -12.0, 4.0, False, {
          2.461265214250274: 0.03281096921159548,
      }))
  def test_gaussian_privacy_loss_tail(self, standard_deviation, sensitivity,
                                      sampling_prob, adjacency_type,
                                      expected_lower_x_truncation,
                                      expected_upper_x_truncation,
                                      pessimistic_estimate,
                                      expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        pessimistic_estimate=pessimistic_estimate,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-1),
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.assert_dictionary_almost_equal(
        self, expected_tail_probability_mass_function,
        tail_pld.tail_probability_mass_function)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 1.5, -1.5),
      (3.0, 3.0, 1.0, ADD, 1.5, -1.5),
      (1.0, 2.0, 1.0, ADD, 4.0, -4.0),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 0.971528300, -1.331138685),
      (3.0, 3.0, 0.8, ADD, 0.971528300, -1.331138685),
      (1.0, 2.0, 0.5, ADD, 0.674997253, -3.325002747),
      (4.0, 8.0, 0.6, ADD, 0.889187896, -3.501310856),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 1.5, -1.5),
      (3.0, 3.0, 1.0, REM, 1.5, -1.5),
      (1.0, 2.0, 1.0, REM, 4.0, -4.0),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, 1.331138685, -0.971528300),
      (3.0, 3.0, 0.8, REM, 1.331138685, -0.971528300),
      (1.0, 2.0, 0.5, REM, 3.325002747, -0.674997253),
      (4.0, 8.0, 0.6, REM, 3.501310856, -0.889187896))
  def test_gaussian_connect_dots_bounds(self, standard_deviation, sensitivity,
                                        sampling_prob, adjacency_type,
                                        expected_epsilon_upper,
                                        expected_epsilon_lower):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        pessimistic_estimate=True,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-1),
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    _assert_connect_bounds_equal(self, pl.connect_dots_bounds(),
                                 expected_epsilon_upper, expected_epsilon_lower,
                                 None, None)

  @parameterized.parameters((0.0, 1.0), (-10.0, 2.0), (4.0, 0.0), (2.0, -1.0),
                            (1.0, 1.0, 1.0, ADD, 1), (2.0, 1.0, 0.0, REM),
                            (1.0, 1.0, 1.2, ADD), (2.0, 1.0, -0.1, REM))
  def test_gaussian_value_errors(self, standard_deviation, sensitivity,
                                 sampling_prob=1.0, adjacency_type=ADD,
                                 log_mass_truncation_bound=-50):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.GaussianPrivacyLoss(
          standard_deviation,
          sensitivity=sensitivity,
          log_mass_truncation_bound=log_mass_truncation_bound,
          sampling_prob=sampling_prob,
          adjacency_type=adjacency_type)

  @parameterized.parameters((1.0, 1.0, 1.0, 0), (1.0, 1.0, 1.0, 1.1),
                            (1.0, 1.0, 1.0, -0.1), (1.0, 0, 1.0, 0.1),
                            (1.0, -0.2, 1.0, 0.1), (1.0, 1.1, 1.0, 0.2))
  def test_gaussian_from_privacy_parameters_value_errors(
      self, sensitivity, sampling_prob, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity,
          sampling_prob=sampling_prob)

  @parameterized.parameters((1.0, 1.0, ADD, 1.0, 0.12693674, 1.0),
                            (2.0, 1.0, REM, 1.0, 0.12693674, 2.0),
                            (3.0, 1.0, ADD, 1.0, 0.78760074, 1.0),
                            (6.0, 1.0, REM, 1.0, 0.78760074, 2.0),
                            (1.0, 1.0, ADD, 2.0, 0.02092364, 1.0),
                            (5.0, 1.0, REM, 2.0, 0.02092364, 5.0),
                            (1.0, 1.0, ADD, 16.0, 1e-5, 0.344),
                            (2.0, 1.0, REM, 16.0, 1e-5, 0.688),
                            (1.0, 0.8, ADD, 1.0, 0.081695179, 1.0),
                            (2.0, 0.7, ADD, 1.0, 0.143886147, 1.5),
                            (3.0, 0.5, ADD, 1.0, 0.267379199, 1.3),
                            (6.0, 0.01, ADD, 1.0, 0.0030216468, 2.0),
                            (1.0, 0.1, REM, 2.0, 2.355186318853955e-6, 1.0),
                            (5.0, 0.75, REM, 2.0, 0.0087720149, 5.0),
                            (1.0, 0.3, REM, 16, 0.0000329405, 0.3),
                            (2.0, 0.2, REM, 16, 0.0230238234, 0.4))
  def test_gaussian_from_privacy_parameters(self, sensitivity, sampling_prob,
                                            adjacency_type, epsilon, delta,
                                            expected_standard_deviation):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
        common.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_standard_deviation, pl.standard_deviation,
                           3)
    self.assertEqual(adjacency_type, pl.adjacency_type)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 1.0, 0.12693674),
      (2.0, 2.0, 1.0, ADD, 1.0, 0.12693674),
      (1.0, 3.0, 1.0, ADD, 1.0, 0.78760074),
      (2.0, 6.0, 1.0, ADD, 1.0, 0.78760074),
      (1.0, 1.0, 1.0, ADD, 2.0, 0.02092364),
      (5.0, 5.0, 1.0, ADD, 2.0, 0.02092364),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 1.0, 0.0231362104090899),
      (2.0, 2.0, 0.8, ADD, 1.0, 0.0231362104090899),
      (1.0, 3.0, 0.7, ADD, 1.0, 0.1195051215523554),
      (2.0, 6.0, 0.4, ADD, 1.0, 0.0),
      (1.0, 1.0, 0.3, ADD, 2.0, 0.0),
      (5.0, 5.0, 0.2, ADD, 2.0, 0.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 1.0, 0.12693674),
      (2.0, 2.0, 1.0, REM, 1.0, 0.12693674),
      (1.0, 3.0, 1.0, REM, 1.0, 0.78760074),
      (2.0, 6.0, 1.0, REM, 1.0, 0.78760074),
      (1.0, 1.0, 1.0, REM, 2.0, 0.02092364),
      (5.0, 5.0, 1.0, REM, 2.0, 0.02092364),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, 1.0, 0.0816951786585355),
      (2.0, 2.0, 0.8, REM, 1.0, 0.0816951786585355),
      (1.0, 3.0, 0.7, REM, 1.0, 0.5356298793262404),
      (2.0, 6.0, 0.4, REM, 1.0, 0.2888308005139968),
      (1.0, 1.0, 0.3, REM, 2.0, 0.0003341102928869332),
      (5.0, 5.0, 0.2, REM, -0.25, 0.2211992169285951))
  def test_gaussian_get_delta_for_epsilon(
      self, standard_deviation, sensitivity, sampling_prob, adjacency_type,
      epsilon, expected_delta):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, [-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.86749642, 0.67881797, 0.38292492, 0.12693674, 0.020923636]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.2, ADD, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
       [0.27768558, 0.21052945, 0.14204776, 0.076584985, 0.02337934,
        0.00020196, 0.0]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, [-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.86749642, 0.67881797, 0.38292492, 0.12693674, 0.020923636]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.2, REM, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0],
       [0.25918178, 0.1814346, 0.11631708, 0.076584985, 0.051816131,
        0.035738492, 0.012494629, 0.00229682]),
      )
  def test_gaussian_get_delta_for_epsilon_vectorized(
      self, standard_deviation, sensitivity, sampling_prob, adjacency_type,
      epsilon_values, expected_delta_values):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    for expected_delta, delta in zip(expected_delta_values,
                                     pl.get_delta_for_epsilon(epsilon_values)):
      self.assertAlmostEqual(expected_delta, delta)


class DiscreteLaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, [-10, 0, 1, 10],
       [3.3190008e-05, 7.3105858e-01, 9.0106198e-01, 9.9998779e-01]),
      (2.0, 1, 1.0, ADD, [-10, 0, 1, 10],
       [1.8154581e-09, 8.8079708e-01, 9.8386764e-01, 1.0000000e+00]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, [-10, 0, 1, 10],
       [3.3190008e-05, 7.3105858e-01, 9.0106198e-01, 9.9998779e-01]),
      (2.0, 1, 0.2, ADD, [-10, 0, 1, 10],
       [1.8154581e-09, 8.8079708e-01, 9.8386764e-01, 1.0000000e+00]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, [-10, 0, 1, 10],
       [9.0219796e-05, 9.0106198e-01, 9.6360274e-01, 9.9999551e-01]),
      (2.0, 1, 1.0, REM, [-10, 0, 1, 10],
       [1.3414522e-08, 9.8386764e-01, 9.9781672e-01, 1.0000000e+00]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, [-10, 0, 1, 10],
       [7.8813838e-05, 8.6706130e-01, 9.5109459e-01, 9.9999396e-01]),
      (2.0, 1, 0.2, REM, [-10, 0, 1, 10],
       [4.1352708e-09, 9.0141119e-01, 9.8665746e-01, 1.0000000e+00]),
  )
  def test_mu_upper_cdf(
      self, parameter, sensitivity, sampling_prob, adjacency_type,
      x, expected_mu_upper_cdf):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertSequenceAlmostEqual(pl.mu_upper_cdf(x), expected_mu_upper_cdf)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, [-30, 0, 1, 30],
       [-3.13132617e+01, -1.31326169, -3.13261688e-01, -6.83897383e-14]),
      (2.0, 1, 1.0, ADD, [-30, 0, 1, 30],
       [-62.12692801, -2.12692801, -0.12692801, 0.]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, [-30, 0, 1, 30],
       [-3.10178672e+01, -1.01786716, -2.67801985e-01, -5.97855099e-14]),
      (2.0, 1, 0.2, ADD, [-30, 0, 1, 30],
       [-6.03167975e+01, -3.16797514e-01, -3.74386343e-02, 0.0]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, [-30, 0, 1, 30],
       [-3.03132617e+01, -3.13261688e-01, -1.04181233e-01, -2.52020627e-14]),
      (2.0, 1, 1.0, REM, [-30, 0, 1, 30],
       [-6.01269280e+01, -1.26928011e-01, -1.62639044e-02, 0.0]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, [-30, 0, 1, 30],
       [-3.03132617e+01, -3.13261688e-01, -1.04181233e-01, -2.52020627e-14]),
      (2.0, 1, 0.2, REM, [-30, 0, 1, 30],
       [-6.01269280e+01, -1.26928011e-01, -1.62639044e-02, 0.0]),
  )
  def test_mu_lower_log_cdf(
      self, parameter, sensitivity, sampling_prob, adjacency_type,
      x, expected_mu_lower_log_cdf):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertSequenceAlmostEqual(pl.mu_lower_log_cdf(x),
                                   expected_mu_lower_log_cdf)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 0, 1.0),
      (1.0, 1, 1.0, ADD, 1, -1.0),
      (0.3, 2, 1.0, ADD, 0, 0.6),
      (0.3, 2, 1.0, ADD, 1, 0.0),
      (0.3, 2, 1.0, ADD, 2, -0.6),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 1, -0.86483972516319),
      (1.0, 1, 0.8, ADD, -1, 0.7046054708796525),
      (0.3, 2, 0.5, ADD, 2, -0.3443407699259402),
      (0.3, 3, 0.5, ADD, 2, -0.1612080639085818),
      (0.3, 2, 0.4, ADD, 1, 0),
      (0.3, 2, 0.3, ADD, 0, 0.1454380063386891),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, -1, 1.0),
      (1.0, 1, 1.0, REM, 0, -1.0),
      (0.3, 2, 1.0, REM, -2, 0.6),
      (0.3, 2, 1.0, REM, -1, 0),
      (0.3, 2, 1.0, REM, 0, -0.6),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, -1, 0.86483972516319),
      (1.0, 1, 0.8, REM, 1, -0.7046054708796525),
      (0.3, 2, 0.5, REM, -2, 0.3443407699259402),
      (0.3, 3, 0.5, REM, -2, 0.1612080639085818),
      (0.3, 2, 0.4, REM, -1, 0),
      (0.3, 2, 0.3, REM, 0, -0.1454380063386891))
  def test_discrete_laplace_privacy_loss(self, parameter, sensitivity,
                                         sampling_prob, adjacency_type, x,
                                         expected_privacy_loss):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters((1.0, 1, 0.4), (2.0, 7, -1.1))
  def test_discrete_laplace_privacy_loss_value_errors(
      self, parameter, sensitivity, x):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    with self.assertRaises(ValueError):
      pl.privacy_loss(x)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 1.1, -math.inf),
      (1.0, 1, 1.0, ADD, 0.9, 0.0),
      (1.0, 1, 1.0, ADD, -1.0, math.inf),
      (0.3, 2, 1.0, ADD, 0.7, -math.inf),
      (0.3, 2, 1.0, ADD, 0.2, 0),
      (0.3, 2, 1.0, ADD, 0.0, 1.0),
      (0.3, 2, 1.0, ADD, -0.6, math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 0.9, -math.inf),
      (1.0, 1, 0.8, ADD, 0.7, 0),
      (1.0, 1, 0.8, ADD, -0.9, math.inf),
      (0.3, 2, 0.5, ADD, 0.26, -math.inf),
      (0.3, 2, 0.4, ADD, 0.0, 1.0),
      (0.3, 2, 0.3, ADD, -0.23, math.inf),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, 1.1, -math.inf),
      (1.0, 1, 1.0, REM, 0.9, -1.0),
      (1.0, 1, 1.0, REM, -1.0, math.inf),
      (0.3, 2, 1.0, REM, 0.7, -math.inf),
      (0.3, 2, 1.0, REM, 0.2, -2.0),
      (0.3, 2, 1.0, REM, 0.0, -1.0),
      (0.3, 2, 1.0, REM, -0.6, math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, 0.9, -math.inf),
      (1.0, 1, 0.8, REM, 0.86483972516319, -1.0),
      (1.0, 1, 0.8, REM, -0.8, math.inf),
      (0.3, 2, 0.5, REM, 0.35, -math.inf),
      (0.3, 2, 0.4, REM, 0.0, -1.0),
      (0.3, 2, 0.3, REM, -0.15, math.inf))
  def test_discrete_laplace_inverse_privacy_loss(self, parameter, sensitivity,
                                                 sampling_prob, adjacency_type,
                                                 privacy_loss, expected_x):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 1, 0, {
          1: 0.73105858,
          -1: 0.26894142
      }),
      (0.3, 2, 1.0, ADD, 1, 1, {
          0.6: 0.57444252,
          -0.6: 0.31526074
      }),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 1, 0, {
          0.7046054708796525: 0.73105858,
          -0.86483972516319: 0.26894142
      }),
      (0.3, 2, 0.6, ADD, 1, 1, {
          0.3156879596155301: 0.5744425168116589,
          -0.4009692034808894: 0.3152607374933769
      }),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, 0, -1, {
          1: 0.73105858,
          -1: 0.26894142
      }),
      (0.3, 2, 1.0, REM, -1, -1, {
          0.6: 0.57444252,
          -0.6: 0.31526074
      }),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, 0, -1, {
          0.86483972516319: 0.638635147178003,
          -0.7046054708796525: 0.361364852821997
      }),
      (0.3, 2, 0.6, REM, -1, -1, {
          0.4009692034808894: 0.4707698050843462,
          -0.3156879596155301: 0.4189334492206898
      }))
  def test_discrete_laplace_privacy_loss_tail(
      self, parameter, sensitivity, sampling_prob, adjacency_type,
      expected_lower_x_truncation, expected_upper_x_truncation,
      expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.assert_dictionary_almost_equal(
        self, expected_tail_probability_mass_function,
        tail_pld.tail_probability_mass_function)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 0, 1),
      (0.3, 2, 1.0, ADD, 0, 2),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 0, 1),
      (0.3, 2, 0.6, ADD, 0, 2),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, -1, 0),
      (0.3, 2, 1.0, REM, -2, 0),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, -1, 0),
      (0.3, 2, 0.6, REM, -2, 0))
  def test_discrete_laplace_connect_dots_bounds(self, parameter, sensitivity,
                                                sampling_prob, adjacency_type,
                                                expected_lower_x,
                                                expected_upper_x):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    _assert_connect_bounds_equal(self, pl.connect_dots_bounds(),
                                 None, None,
                                 expected_lower_x, expected_upper_x)

  @parameterized.parameters((-3.0, 1), (0.0, 1), (2.0, 0.5),
                            (2.0, -1), (1.0, 0),
                            (2.0, 1, 0.0, ADD), (1.0, 1, 1.2, REM),
                            (2.0, 1, -0.1, ADD))
  def test_discrete_laplace_value_errors(self, parameter, sensitivity,
                                         sampling_prob=1.0, adjacency_type=ADD):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
          parameter,
          sensitivity=sensitivity,
          sampling_prob=sampling_prob,
          adjacency_type=adjacency_type)

  @parameterized.parameters((-1, 0.8, 1.0, 0.1), (0.5, 1.0, 1.0, 0.1),
                            (0, 1.0, 1.0, 0.2), (1, 1.0, 1.0, -0.1),
                            (1, 0.8, 1.0, 1.1), (1, 0.0, 1.0, 0.1),
                            (3, 1.1, 1.0, 0.1), (1, -0.2, 1.0, 0.1))
  def test_discrete_laplace_from_privacy_parameters_value_errors(
      self, sensitivity, sampling_prob, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteLaplacePrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity, sampling_prob=sampling_prob)

  @parameterized.parameters((1, 1.0, ADD, 1.0, 0.0, 1.0),
                            (1, 1.0, REM, 1.0, 0.1, 1.0),
                            (2, 1.0, ADD, 1.0, 0.01, 0.5),
                            (1, 1.0, REM, 3.0, 0.01, 3.0),
                            (1, 0.8, ADD, 1.0, 0.0, 1.1467204062),
                            (1, 0.7, REM, 1.0, 0.1, 1.2397322437),
                            (2, 0.3, ADD, 1.0, 0.01, 0.9531096869),
                            (1, 0.2, REM, 3.0, 0.01, 4.5687933452))
  def test_discrete_laplace_from_privacy_parameters(
      self, sensitivity, sampling_prob, adjacency_type,
      epsilon, delta, expected_parameter):
    pl = (privacy_loss_mechanism.DiscreteLaplacePrivacyLoss
          .from_privacy_guarantee(
              common.DifferentialPrivacyParameters(
                  epsilon, delta),
              sensitivity,
              sampling_prob=sampling_prob,
              adjacency_type=adjacency_type))
    self.assertAlmostEqual(expected_parameter, pl.parameter)
    self.assertEqual(adjacency_type, pl.adjacency_type)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 1.0, 0.0), (0.333333, 3, 1.0, ADD, 1.0, 0.0),
      (0.5, 4, 1.0, ADD, 2.0, 0.0), (0.5, 4, 1.0, ADD, 0.5, 0.54202002),
      (0.5, 4, 1.0, ADD, 1.0, 0.39346934),
      (0.5, 4, 1.0, ADD, -0.5, 0.72222110),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 1.0, 0.0),
      (0.333333, 3, 0.8, ADD, 1.0, 0.0),
      (0.5, 4, 0.7, ADD, 0.5, 0.2293628348747755),
      (0.5, 4, 0.6, ADD, 0.6, 0.07668344250639381),
      (0.5, 4, 0.3, ADD, 0.5, 0.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, 1.0, 0.0), (0.333333, 3, 1.0, REM, 1.0, 0.0),
      (0.5, 4, 1.0, REM, 2.0, 0.0), (0.5, 4, 1.0, REM, 0.5, 0.54202002),
      (0.5, 4, 1.0, REM, 1.0, 0.39346934),
      (0.5, 4, 1.0, REM, -0.5, 0.72222110),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, 1.0, 0.0),
      (0.333333, 3, 0.8, REM, 1.0, 0.0),
      (0.5, 4, 0.7, REM, 0.5, 0.3523838505224567),
      (0.5, 4, 0.6, REM, 1.0, 0.178181891763215),
      (0.5, 4, 0.3, REM, 0.5, 0.1068168460276349),
      (1.0, 1, 0.2, REM, -0.25, 0.2211992169285951))
  def test_discrete_laplace_get_delta_for_epsilon(self, parameter, sensitivity,
                                                  sampling_prob, adjacency_type,
                                                  epsilon, expected_delta):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (0.5, 4, 1.0, ADD, [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
       [0.86466472, 0.77686984, 0.7222211, 0.63212056, 0.54202002,
        0.39346934, 0.0]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (0.5, 4, 0.2, ADD, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
       [0.31709255, 0.25960017, 0.19633877, 0.12642411, 0.058632421, 0.0, 0.0]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (0.5, 4, 1.0, REM, [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
       [0.86466472, 0.77686984, 0.7222211, 0.63212056, 0.54202002,
        0.39346934, 0.0]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (0.5, 4, 0.2, REM, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
       [0.25918178, 0.18126925, 0.14821539, 0.12642411, 0.11181698,
        0.095673604, 0.07817137]),
      )
  def test_discrete_laplace_get_delta_for_epsilon_vectorized(
      self, parameter, sensitivity, sampling_prob, adjacency_type,
      epsilon_values, expected_delta_values):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    for expected_delta, delta in zip(expected_delta_values,
                                     pl.get_delta_for_epsilon(epsilon_values)):
      self.assertAlmostEqual(expected_delta, delta)


class DiscreteGaussianPrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, [-3, -2, -1, 0, 1],
       [0.0, 0.0, 0.27406862, 0.72593138, 1.0]),
      (3.0, 1, 1.0, ADD, [-3, -2, -1, 0, 1],
       [0.0, 0.0, 0.32710442, 0.67289558, 1.0]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, [-3, -2, -1, 0, 1],
       [0.0, 0.0, 0.274068619, 0.725931381, 1.0]),
      (2.0, 1, 0.2, ADD, [-3, -2, -1, 0, 1],
       [0.0, 0.0, 0.319167768, 0.680832232, 1.0]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, [-3, -2, -1, 0, 1],
       [0.0, 0.27406862, 0.72593138, 1.0, 1.0]),
      (3.0, 1, 1.0, REM, [-3, -2, -1, 0, 1],
       [0.0, 0.32710442, 0.67289558, 1.0, 1.0]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, [-3, -2, -1, 0, 1],
       [0.0, 0.219254895, 0.635558829, 0.945186276, 1.0]),
      (2.0, 1, 0.2, REM, [-3, -2, -1, 0, 1],
       [0.0, 0.063833554, 0.391500661, 0.744665785, 1.0]),
  )
  def test_mu_upper_cdf(
      self, sigma, sensitivity, sampling_prob, adjacency_type,
      x, expected_mu_upper_cdf):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        truncation_bound=1,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertSequenceAlmostEqual(pl.mu_upper_cdf(x), expected_mu_upper_cdf)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, [-1, 0, 1, 2],
       [-math.inf, -1.29437677, -0.32029979, 0.]),
      (3.0, 1, 1.0, ADD, [-1, 0, 1, 2],
       [-math.inf, -1.11747583, -0.39616512, 0.]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, [-1, 0, 1, 2],
       [-2.90381468, -1.00939014, -0.24750655, 0.]),
      (2.0, 1, 0.2, ADD, [-1, 0, 1, 2],
       [-1.36518195e+00, -4.96759453e-01, -6.59619911e-02, 2.22044605e-16]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, [-1, 0, 1, 2],
       [-1.29437677, -0.32029979, 0.0, 0.0]),
      (3.0, 1, 1.0, REM, [-1, 0, 1, 2],
       [-1.11747583, -0.39616512, 0., 0.]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, [-1, 0, 1, 2],
       [-1.29437677, -0.32029979, 0., 0.]),
      (2.0, 1, 0.2, REM, [-1, 0, 1, 2],
       [-1.14203839, -0.38443936, 0., 0.]),
  )
  def test_mu_lower_log_cdf(
      self, sigma, sensitivity, sampling_prob, adjacency_type,
      x, expected_mu_lower_log_cdf):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        truncation_bound=1,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertSequenceAlmostEqual(pl.mu_lower_log_cdf(x),
                                   expected_mu_lower_log_cdf)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 5, -4.5),
      (1, 1, 1, ADD, -3, 3.5),
      (1, 2, 1, ADD, 3, -4.0),
      (4.0, 4, 1.0, ADD, 20, -4.5),
      (5, 5, 1, ADD, -15, 3.5),
      (7.0, 14, 1.0, ADD, 21, -4.0),
      (1.0, 1, 1.0, ADD, -12, math.inf),
      (1.0, 1, 1.0, ADD, 13, -math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, -4, 1.565960898891332),
      (1.0, 1, 0.7, ADD, 4, -3.156183763141021),
      (1.0, 2, 0.4, ADD, -1, 0.4986891437585786),
      (4.0, 4, 0.3, ADD, -16, 0.3519252431310541),
      (5.0, 5, 0.4, ADD, 20, -2.628009438900115),
      (7.0, 14, 0.1, ADD, -7, 0.1033275126220077),
      (1.0, 1, 0.3, ADD, 13, -math.inf),
      (1.0, 1, 0.3, ADD, -12, 0.3566749439387324),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, 4, -4.5), (1, 1, 1, REM, -4, 3.5),
      (1.0, 2, 1.0, REM, 1, -4.0), (4, 4, 1, REM, 16, -4.5),
      (5.0, 5, 1.0, REM, -20, 3.5), (7, 14, 1, REM, 7, -4.0),
      (1.0, 1, 1.0, REM, -13, math.inf),
      (1.0, 1, 1.0, REM, 12, -math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, 4, -1.565960898891332),
      (1.0, 1, 0.7, REM, -4, 3.156183763141021),
      (1.0, 2, 0.4, REM, 1, -0.4986891437585786),
      (4.0, 4, 0.3, REM, 16, -0.3519252431310541),
      (5.0, 5, 0.4, REM, -20, 2.628009438900115),
      (7.0, 14, 0.1, REM, 7, -0.1033275126220077),
      (1.0, 1, 0.3, REM, -13, math.inf),
      (1.0, 1, 0.3, REM, 12, -0.3566749439387324))
  def test_discrete_gaussian_privacy_loss(self, sigma, sensitivity,
                                          sampling_prob, adjacency_type,
                                          x, expected_privacy_loss):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters((1.0, 1, 1.0, ADD, 0.4), (2.0, 7, 1.0, REM, -1.1),
                            (1.0, 1, 0.6, ADD, -13), (2.0, 1, 0.5, ADD, 26),
                            (1.0, 1, 0.6, REM, -14), (2.0, 1, 0.5, REM, 25))
  def test_discrete_gaussian_privacy_loss_value_errors(self, sigma, sensitivity,
                                                       sampling_prob,
                                                       adjacency_type, x):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    with self.assertRaises(ValueError):
      pl.privacy_loss(x)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, -4.5, 5),
      (1.0, 1, 1.0, ADD, 3.5, -3),
      (1.0, 2, 1.0, ADD, -4.0, 3),
      (4.0, 4, 1.0, ADD, -4.51, 20),
      (5.0, 5, 1.0, ADD, 3.49, -15),
      (7.0, 14, 1.0, ADD, -4.0, 21),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 1.565961, -5),
      (1.0, 1, 0.7, ADD, -3.156182, 3),
      (1.0, 2, 0.4, ADD, 0.4986892, -2),
      (4.0, 4, 0.3, ADD, 0.3519254, -17),
      (5.0, 5, 0.4, ADD, -2.6280094, 19),
      (7.0, 14, 0.1, ADD, 0.1033276, -8),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, -4.5, 4),
      (1.0, 1, 1.0, REM, 3.5, -4),
      (1.0, 2, 1.0, REM, -4.0, 1),
      (4.0, 4, 1.0, REM, -4.51, 16),
      (5.0, 5, 1.0, REM, 3.49, -20),
      (7.0, 14, 1.0, REM, -4.0, 7),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, -1.565961, 4),
      (1.0, 1, 0.7, REM, 3.156182, -4),
      (1.0, 2, 0.4, REM, -0.4986892, 1),
      (4.0, 4, 0.3, REM, -0.3519254, 16),
      (5.0, 5, 0.4, REM, 2.6280094, -20),
      (7.0, 14, 0.1, REM, -0.1033276, 7))
  def test_discrete_gaussian_inverse_privacy_loss(self, sigma, sensitivity,
                                                  sampling_prob, adjacency_type,
                                                  privacy_loss, expected_x):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 2, 1.0, ADD, -1, 2, {
          math.inf: 0.05448868
      }),
      (1.0, 2, 2, 1.0, ADD, 0, 2, {
          math.inf: 0.29869003
      }),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 2, 0.8, ADD, -2, 2, {
          math.inf: 0.0
      }),
      (1.0, 2, 2, 0.7, ADD, -2, 2, {
          math.inf: 0.0
      }),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 1.0, REM, -2, 1, {
          math.inf: 0.05448868
      }),
      (1.0, 2, 2, 1.0, REM, -2, 0, {
          math.inf: 0.29869003
      }),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 0.8, REM, -2, 2, {
          math.inf: 0.043590944
      }),
      (1.0, 2, 2, 0.7, REM, -2, 2, {
          math.inf: 0.209083021
      }))
  def test_discrete_gaussian_privacy_loss_tail(
      self, sigma, sensitivity, truncation_bound, sampling_prob, adjacency_type,
      expected_lower_x_truncation, expected_upper_x_truncation,
      expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        truncation_bound=truncation_bound,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.assert_dictionary_almost_equal(
        self, expected_tail_probability_mass_function,
        tail_pld.tail_probability_mass_function)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 2, 1.0, ADD, -1, 2),
      (1.0, 2, 2, 1.0, ADD, 0, 2),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 2, 0.8, ADD, -2, 2),
      (1.0, 2, 2, 0.7, ADD, -2, 2),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 1.0, REM, -2, 1),
      (1.0, 2, 2, 1.0, REM, -2, 0),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 0.8, REM, -2, 2),
      (1.0, 2, 2, 0.7, REM, -2, 2))
  def test_discrete_gaussian_connect_dots_bounds(
      self, sigma, sensitivity, truncation_bound, sampling_prob, adjacency_type,
      expected_lower_x, expected_upper_x):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        truncation_bound=truncation_bound,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    _assert_connect_bounds_equal(self, pl.connect_dots_bounds(),
                                 None, None,
                                 expected_lower_x, expected_upper_x)

  @parameterized.parameters((-3.0, 1), (0.0, 1), (2.0, 0.5), (1.0, 0),
                            (2.0, -1), (2.0, 4, 1, ADD, 1),
                            (2.0, 1, 0), (1.0, 1, 1.2), (2.0, 1, -0.1))
  def test_discrete_gaussian_value_errors(self, sigma, sensitivity,
                                          sampling_prob=1.0, adjacency_type=ADD,
                                          truncation_bound=None):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
          sigma,
          sensitivity=sensitivity,
          truncation_bound=truncation_bound,
          sampling_prob=sampling_prob,
          adjacency_type=adjacency_type)

  @parameterized.parameters((1.0, 1, 1, {
      -1.5: 0,
      -1: 0.27406862,
      0: 0.7259314,
      1: 1,
      1.5: 1
  }), (3.0, 2, 2, {
      -2.1: 0,
      -2: 0.17820326,
      -1: 0.38872553,
      0: 0.61127447,
      1: 0.82179674,
      2: 1,
      2.7: 1
  }))
  def test_discrete_gaussian_noise_cdf(self, sigma, sensitivity,
                                       truncation_bound, x_to_cdf_value):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity, truncation_bound=truncation_bound)
    for x, cdf_value in x_to_cdf_value.items():
      self.assertAlmostEqual(cdf_value, pl.noise_cdf(x))

  @parameterized.parameters((1.0, 1, 1, 0.7403629), (3.0, 2, 2, 1.3589226))
  def test_discrete_gaussian_std(self, sigma, sensitivity, truncation_bound,
                                 expected_std):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity, truncation_bound=truncation_bound)
    self.assertAlmostEqual(expected_std, pl.standard_deviation())

  @parameterized.parameters((-1, 1.0, 1.0, 0.1), (0.5, 1.0, 1.0, 0.1),
                            (0, 0.7, 1.0, 0.2), (1, 1.0, 1.0, 0.0),
                            (1, 1.0, 1.0, 1.1), (1, 1.0, 1.0, -0.1),
                            (1, 0.0, 1.0, 0.1), (1, 1.1, 1.0, 0.1),
                            (1, -0.1, 1.0, 0.1))
  def test_discrete_gaussian_from_privacy_parameters_value_errors(
      self, sensitivity, sampling_prob, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteGaussianPrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity,
          sampling_prob=sampling_prob)

  @parameterized.parameters(
      (1, 1.0, ADD, 1.0, 0.12693674, 1.041),
      (2, 1.0, REM, 1.0, 0.12693674, 1.972),
      (3, 1.0, ADD, 1.0, 0.78760074, 0.993),
      (6, 1.0, REM, 1.0, 0.78760074, 2.014),
      (1, 1.0, ADD, 2.0, 0.02092364, 1.038),
      (5, 1.0, REM, 2.0, 0.02092364, 5.008),
      (1, 1.0, ADD, 16.0, 1e-5, 0.306),
      (2, 1.0, REM, 16.0, 1e-5, 0.703),
      (1, 0.8, REM, 1.0, 0.07850075632001355, 1.041),
      (2, 0.7, ADD, 1.0, 0.06665777574091321, 1.972),
      (3, 0.4, REM, 1.0, 0.27122238416249084, 0.993),
      (6, 0.5, ADD, 1.0, 0.3604879495041193, 2.014),
      (1, 0.3, REM, 2.0, 0.0002834863230938751, 1.038),
      (5, 0.1, ADD, 2.0, 2.340272571167144e-06, 5.008),
      (2, 0.9, REM, 16.0, 4.518347272315105e-06, 0.703))
  def test_discrete_gaussian_from_privacy_parameters(self, sensitivity,
                                                     sampling_prob,
                                                     adjacency_type, epsilon,
                                                     delta, expected_sigma):
    pl = (
        privacy_loss_mechanism.DiscreteGaussianPrivacyLoss
        .from_privacy_guarantee(
            common.DifferentialPrivacyParameters(epsilon, delta),
            sensitivity,
            sampling_prob=sampling_prob,
            adjacency_type=adjacency_type))
    self.assertAlmostEqual(expected_sigma, pl._sigma, 3)
    self.assertEqual(adjacency_type, pl.adjacency_type)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 2, 1.0, ADD, 1.0, 0.150574425),
      (10.0, 3, 5, 1.0, ADD, 1.0, 0.263672394),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 2, 0.8, ADD, 1.0, 0.024865564),
      (5.0, 3, 5, 0.4, ADD, 0.1, 0.085584980),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 1.0, REM, 1.0, 0.150574425),
      (10.0, 3, 5, 1.0, REM, 1.0, 0.263672394),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 0.8, REM, 1.0, 0.101734157),
      (5.0, 3, 5, 0.4, REM, 0.1, 0.104486937))
  def test_discrete_gaussian_get_delta_for_epsilon(
      self, sigma, sensitivity, truncation_bound, sampling_prob, adjacency_type,
      epsilon, expected_delta):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        truncation_bound=truncation_bound,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 2, 1.0, ADD, [-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.872038958, 0.687513794, 0.402619947, 0.150574425, 0.054488685]),
      (10.0, 3, 5, 1.0, ADD, [-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.900348895, 0.729120212, 0.285480872, 0.263672394, 0.263672394]),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 2, 0.8, ADD, [-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.870564110, 0.669546464, 0.322095958, 0.024865564, 0.000000000]),
      (5.0, 3, 5, 0.4, ADD, [-0.2, -0.1, 0.0, 0.1, 0.2],
       [0.258926845, 0.189706273, 0.129522020, 0.085584980, 0.063350727]),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 1.0, REM, [-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.872038958, 0.687513794, 0.402619947, 0.150574425, 0.054488685]),
      (10.0, 3, 5, 1.0, REM, [-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.900348895, 0.729120212, 0.285480872, 0.263672394, 0.263672394]),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 0.8, REM, [-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.864664717, 0.641268089, 0.322095958, 0.101734157, 0.043590948]),
      (5.0, 3, 5, 0.4, REM, [-0.2, -0.1, 0.0, 0.1, 0.2],
       [0.233136435, 0.172603074, 0.129522020, 0.104486937, 0.094851204]))
  def test_discrete_gaussian_get_delta_for_epsilon_vectorized(
      self, sigma, sensitivity, truncation_bound, sampling_prob, adjacency_type,
      epsilon_values, expected_delta_values):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        truncation_bound=truncation_bound,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    for expected_delta, delta in zip(expected_delta_values,
                                     pl.get_delta_for_epsilon(epsilon_values)):
      self.assertAlmostEqual(expected_delta, delta)


def _add_and_remove_test_cases(test_cases):
  """Takes test cases and creates a copy of each for each adjacency type."""
  def add_test_case(test_case):
    add_test_case = test_case.copy()
    add_test_case['testcase_name'] += '_add'
    add_test_case['adjacency_type'] = ADD
    return add_test_case
  add_test_cases = [add_test_case(case) for case in test_cases]

  def remove_test_case(test_case):
    remove_test_case = test_case.copy()
    remove_test_case['testcase_name'] += '_remove'
    remove_test_case['adjacency_type'] = REM
    return remove_test_case
  remove_test_cases = [remove_test_case(case) for case in test_cases]

  return add_test_cases + remove_test_cases


def _gaussian_test_cases(num_tests):
  """Generates lists of inputs corresponding to Gaussians."""
  random.seed(0)
  def test(i):
    return {
        'testcase_name': f'gaussian_{i}',
        'standard_deviation': random.uniform(1, 10),
        'sensitivities': [random.uniform(1, 10)],
        'sampling_probs': [1.0],
    }
  return [test(i) for i in range(num_tests)]


def _subsampled_gaussian_test_cases(num_tests):
  """Generates lists of inputs corresponding to subsampled Gaussians."""
  random.seed(0)
  probs = [random.uniform(0, 1) for _ in range(num_tests)]

  def test(i):
    return {
        'testcase_name': f'subsampled_gaussian_{i}',
        'standard_deviation': random.uniform(1, 10),
        'sensitivities': [0, random.uniform(1, 10)],
        'sampling_probs': [probs[i], 1 - probs[i]],
    }

  return [test(i) for i in range(num_tests)]


def _mixture_gaussian_with_zero_test_cases(num_tests):
  """Generates lists of inputs corresponding to mixture Gaussians."""
  random.seed(0)
  probs = [random.uniform(0, 0.5) for _ in range(2 * num_tests)]
  prob_lists = [
      [1 - probs[2 * i] - probs[2 * i + 1], probs[2 * i], probs[2 * i + 1]]
      for i in range(num_tests)
  ]

  def test(i):
    return {
        'testcase_name': f'mixture_gaussian_with_zero_{i}',
        'standard_deviation': random.uniform(1, 10),
        'sensitivities': [0, random.uniform(1, 10), random.uniform(1, 10)],
        'sampling_probs': prob_lists[i],
    }

  return [test(i) for i in range(num_tests)]


def _mixture_gaussian_without_zero_test_cases(num_tests):
  """Generates lists of inputs corresponding to non-zero mixture Gaussians."""
  random.seed(0)
  probs = [random.uniform(0, 1.0) for _ in range(num_tests)]

  def test(i):
    return {
        'testcase_name': f'mixture_gaussian_without_zero_{i}',
        'standard_deviation': random.uniform(1, 10),
        'sensitivities': [random.uniform(1, 10), random.uniform(1, 10)],
        'sampling_probs': [probs[i], 1 - probs[i]],
    }

  return [test(i) for i in range(num_tests)]


class MixtureGaussianPrivacyLossTest(parameterized.TestCase):
  """Tests for privacy_loss_mechanism.MixtureGaussianPrivacyLoss class."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'negative_stdev',
          'standard_deviation': -1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
      },
      {
          'testcase_name': 'negative_sensitivity',
          'standard_deviation': 1.0,
          'sensitivities': [-1.0, 1.0],
          'sampling_probs': [0.5, 0.5],
      },
      {
          'testcase_name': 'negative_probability',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.75, 0.75, -0.5],
      },
      {
          'testcase_name': 'probability_greater_than_one',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [1.5, 0.5, 0.5],
      },
      {
          'testcase_name': 'probabilities_dont_add_up_to_one',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.2, 0.2],
      },
      {
          'testcase_name': 'list_lengths_differ_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [0.5, 0.5],
      },
      {
          'testcase_name': 'list_lengths_differ_2',
          'standard_deviation': 1.0,
          'sensitivities': [1.0, 2.0, 3.0],
          'sampling_probs': [0.5, 0.5],
      },
  )
  def test_init_raises_error(
      self,
      standard_deviation,
      sensitivities,
      sampling_probs,
  ):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
          standard_deviation=standard_deviation,
          sensitivities=sensitivities,
          sampling_probs=sampling_probs,
      )

  @parameterized.named_parameters(
      {
          'testcase_name': 'gaussian_add_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'adjacency_type': ADD,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-6.38249341e01, -1.84102165e00,
                                        -6.93147181e-01, -1.12858841e-19]
      },
      {
          'testcase_name': 'gaussian_add_2',
          'standard_deviation': 2.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'adjacency_type': ADD,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-1.77793764e01, -1.17591176e00,
                                        -6.93147181e-01, -3.39767890e-06]
      },
      {
          'testcase_name': 'sampled_gaussian_add_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': ADD,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-54.84062277, -1.48313922,
                                        -0.56516047, 0.0]
      },
      {
          'testcase_name': 'sampled_gaussian_add_2',
          'standard_deviation': 2.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.8, 0.2],
          'adjacency_type': ADD,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-1.52717161e01, -7.72823689e-01,
                                        -4.25917894e-01, -9.08856295e-07]
      },
      {
          'testcase_name': 'mixture_gaussian_add',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.4, 0.4],
          'adjacency_type': ADD,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-5.48406729e01, -1.75699779,
                                        -0.839952452, -2.22044605e-13]
      },
      {
          'testcase_name': 'gaussian_rem_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'adjacency_type': REM,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-5.32312852e01, -6.93147181e-01,
                                        -0.172753779, -7.61985302e-24]
      },
      {
          'testcase_name': 'gaussian_rem_2',
          'standard_deviation': 2.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'adjacency_type': REM,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-1.50649984e01, -6.93147181e-01,
                                        -3.68946415e-01, -2.86651613e-07]
      },
      {
          'testcase_name': 'sampled_gaussian_rem_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': REM,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-5.32312852e01, -6.93147181e-01,
                                        -1.72753779e-01, -7.61985302e-24]
      },
      {
          'testcase_name': 'sampled_gaussian_rem_2',
          'standard_deviation': 2.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.8, 0.2],
          'adjacency_type': REM,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-1.50649984e01, -6.93147181e-01,
                                        -3.68946415e-01, -2.86651613e-07]
      },
      {
          'testcase_name': 'mixture_gaussian_rem',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.4, 0.4],
          'adjacency_type': REM,
          'x': [-10, 0, 1, 10],
          'expected_mu_lower_log_cdf': [-5.32312852e01, -6.93147181e-01,
                                        -1.72753779e-01, -7.61985302e-24]
      },
  )
  def test_mu_lower_log_cdf(
      self,
      standard_deviation,
      sensitivities,
      sampling_probs,
      adjacency_type,
      x,
      expected_mu_lower_log_cdf,
  ):
    pl = privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
        standard_deviation=standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        adjacency_type=adjacency_type,
    )
    self.assertSequenceAlmostEqual(
        expected_mu_lower_log_cdf, pl.mu_lower_log_cdf(x)
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'gaussian_add_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'adjacency_type': ADD,
          'x': 5.0,
          'expected_privacy_loss': -4.5,
      },
      {
          'testcase_name': 'gaussian_add_2',
          'standard_deviation': 1.0,
          'sensitivities': [2.0],
          'sampling_probs': [1.0],
          'adjacency_type': ADD,
          'x': 3.0,
          'expected_privacy_loss': -4.0,
      },
      {
          'testcase_name': 'gaussian_add_3',
          'standard_deviation': 7.0,
          'sensitivities': [14.0],
          'sampling_probs': [1.0],
          'adjacency_type': ADD,
          'x': 21.0,
          'expected_privacy_loss': -4.0,
      },
      {
          'testcase_name': 'sampled_gaussian_add_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': ADD,
          'x': 0.5,
          'expected_privacy_loss': 0.0,
      },
      {
          'testcase_name': 'sampled_gaussian_add_2',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 2.0],
          'sampling_probs': [0.3, 0.7],
          'adjacency_type': ADD,
          'x': 0,
          'expected_privacy_loss': 0.929541389699331,
      },
      {
          'testcase_name': 'sampled_gaussian_add_3',
          'standard_deviation': 7.0,
          'sensitivities': [0.0, 14.0],
          'sampling_probs': [0.1, 0.9],
          'adjacency_type': ADD,
          'x': -7,
          'expected_privacy_loss': 2.150000710600199,
      },
      {
          'testcase_name': 'mixture_gaussian_add_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'adjacency_type': ADD,
          'x': 0.5,
          'expected_privacy_loss': 0.1351602748368097,
      },
      {
          'testcase_name': 'mixture_gaussian_add_2',
          'standard_deviation': 1.0,
          'sensitivities': [1.0, 2.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': ADD,
          'x': 0.5,
          'expected_privacy_loss': 0.7046054708796523,
      },
      {
          'testcase_name': 'mixture_gaussian_add_3',
          'standard_deviation': 7.0,
          'sensitivities': [0.0, 7.0, 14.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'adjacency_type': ADD,
          'x': 0.5,
          'expected_privacy_loss': 0.4746752545839654,
      },
      {
          'testcase_name': 'mixture_gaussian_add_4',
          'standard_deviation': 7.0,
          'sensitivities': [0.0, 7.0, 14.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'adjacency_type': ADD,
          'x': -0.5,
          'expected_privacy_loss': 0.5757291778782041,
      },
      {
          'testcase_name': 'gaussian_rem_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'adjacency_type': REM,
          'x': 4.0,
          'expected_privacy_loss': -4.5,
      },
      {
          'testcase_name': 'gaussian_rem_2',
          'standard_deviation': 1.0,
          'sensitivities': [2.0],
          'sampling_probs': [1.0],
          'adjacency_type': REM,
          'x': 1.0,
          'expected_privacy_loss': -4.0,
      },
      {
          'testcase_name': 'gaussian_rem_3',
          'standard_deviation': 7.0,
          'sensitivities': [14.0],
          'sampling_probs': [1.0],
          'adjacency_type': REM,
          'x': 7.0,
          'expected_privacy_loss': -4.0,
      },
      {
          'testcase_name': 'sampled_gaussian_rem_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': REM,
          'x': -0.5,
          'expected_privacy_loss': 0.0,
      },
      {
          'testcase_name': 'sampled_gaussian_rem_2',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 2.0],
          'sampling_probs': [0.3, 0.7],
          'adjacency_type': REM,
          'x': 0.0,
          'expected_privacy_loss': -0.929541389699331,
      },
      {
          'testcase_name': 'sampled_gaussian_rem_3',
          'standard_deviation': 7.0,
          'sensitivities': [0.0, 14.0],
          'sampling_probs': [0.1, 0.9],
          'adjacency_type': REM,
          'x': 7.0,
          'expected_privacy_loss': -2.150000710600199,
      },
      {
          'testcase_name': 'mixture_gaussian_rem_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'adjacency_type': REM,
          'x': 0.5,
          'expected_privacy_loss': -0.8423781325734492,
      },
      {
          'testcase_name': 'mixture_gaussian_rem_2',
          'standard_deviation': 1.0,
          'sensitivities': [1.0, 2.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': REM,
          'x': 0.5,
          'expected_privacy_loss': -2.176785009442309,
      },
      {
          'testcase_name': 'mixture_gaussian_rem_3',
          'standard_deviation': 7.0,
          'sensitivities': [0.0, 7.0, 14.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'adjacency_type': REM,
          'x': 0.5,
          'expected_privacy_loss': -0.5757291778782041,
      },
      {
          'testcase_name': 'mixture_gaussian_rem_4',
          'standard_deviation': 7.0,
          'sensitivities': [0.0, 7.0, 14.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'adjacency_type': REM,
          'x': -0.5,
          'expected_privacy_loss': -0.4746752545839654,
      }
  )
  def test_privacy_loss(
      self,
      standard_deviation,
      sensitivities,
      sampling_probs,
      adjacency_type,
      x,
      expected_privacy_loss,
  ):
    pl = privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
        standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        adjacency_type=adjacency_type,
    )
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.named_parameters(
      _add_and_remove_test_cases(
          _gaussian_test_cases(5)
          + _subsampled_gaussian_test_cases(5)
          + _mixture_gaussian_with_zero_test_cases(5)
          + _mixture_gaussian_without_zero_test_cases(5)
      )
  )
  def test_privacy_loss_tail(
      self, standard_deviation, sensitivities, sampling_probs, adjacency_type
  ):
    log_cutoff = -10.0
    pl = privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
        standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        adjacency_type=adjacency_type,
        log_mass_truncation_bound=log_cutoff,
    )
    tail_mass = math.exp(log_cutoff) / 2
    privacy_loss_tail = pl.privacy_loss_tail()
    lower = privacy_loss_tail.lower_x_truncation
    upper = privacy_loss_tail.upper_x_truncation
    lower_cdf = pl.mu_upper_cdf(lower)
    upper_cdf = pl.mu_upper_cdf(upper)
    # Any solution that reports a privacy loss that doesn't exceed the cutoff is
    # correct, so we use LessEqual instead of AlmostEqual to be robust to
    # changes that might e.g. sacrifice accuracy for efficiency (e.g., changing
    # the default discretization).
    self.assertLessEqual(lower_cdf, tail_mass)
    # Computing 1 - upper_cdf is numerically imprecise.
    self.assertLessEqual(1 - upper_cdf, tail_mass * (1 + 1e-7))
    self.assertDictEqual(
        privacy_loss_tail.tail_probability_mass_function,
        {
            math.inf: pl.mu_upper_cdf(lower),
            pl.privacy_loss(upper): 1 - pl.mu_upper_cdf(upper),
        },
    )

  @parameterized.named_parameters(
      _add_and_remove_test_cases(
          _gaussian_test_cases(5)
          + _subsampled_gaussian_test_cases(5)
          + _mixture_gaussian_with_zero_test_cases(5)
          + _mixture_gaussian_without_zero_test_cases(5)
      )
  )
  def test_connect_dots_bounds(
      self, standard_deviation, sensitivities, sampling_probs, adjacency_type
  ):
    log_cutoff = -10.0
    pl = privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
        standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        adjacency_type=adjacency_type,
        log_mass_truncation_bound=log_cutoff,
    )
    connect_dots_bounds = pl.connect_dots_bounds()
    privacy_loss_tail = pl.privacy_loss_tail()
    pl_lower = connect_dots_bounds.epsilon_lower
    pl_upper = connect_dots_bounds.epsilon_upper
    x_lower = privacy_loss_tail.lower_x_truncation
    x_upper = privacy_loss_tail.upper_x_truncation
    self.assertAlmostEqual(pl.privacy_loss(x_lower), pl_upper)
    self.assertAlmostEqual(pl.privacy_loss(x_upper), pl_lower)
    self.assertIsNone(connect_dots_bounds.lower_x)
    self.assertIsNone(connect_dots_bounds.upper_x)

  @parameterized.named_parameters(
      _add_and_remove_test_cases(
          _gaussian_test_cases(5)
          + _subsampled_gaussian_test_cases(5)
          + _mixture_gaussian_with_zero_test_cases(5)
          + _mixture_gaussian_without_zero_test_cases(5)
      )
  )
  def test_inverse_privacy_losses(
      self, standard_deviation, sensitivities, sampling_probs, adjacency_type
  ):
    pl = privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
        standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        adjacency_type=adjacency_type,
    )
    privacy_loss_tail = pl.privacy_loss_tail()
    pl_lower = pl.privacy_loss(privacy_loss_tail.upper_x_truncation)
    pl_upper = pl.privacy_loss(privacy_loss_tail.lower_x_truncation)
    pl_inputs = np.array(
        [pl_lower + alpha * (pl_upper - pl_lower) / 10 for alpha in range(1, 9)]
    )
    outputs = pl.inverse_privacy_losses(pl_inputs)
    for pl_input, output in zip(pl_inputs, outputs):
      # The method we're testing is only approximately implemented, so
      # we use a lower-precision test.
      self.assertAlmostEqual(pl_input, pl.privacy_loss(output), places=5)

  # The gaussian and sampled_gaussian test cases are from
  # the tests for GaussianPrivacyLoss. The mixture_gaussian deltas are
  # computed via numerical search: since the privacy loss is monotonic, the
  # delta for a given epsilon can be written as
  # max_t [Pr[y < t] - e^eps * Pr[x < t]], where y is the "upper" variable
  # and x is the "lower" variable. This is an increasing-then-decreasing
  # function of t for MoG mechanisms, so we can optimize it via binary search.
  @parameterized.named_parameters(
      {
          'testcase_name': 'gaussian_add_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'adjacency_type': ADD,
          'epsilon': 1.0,
          'expected_delta': 0.12693674,
      },
      {
          'testcase_name': 'gaussian_add_2',
          'standard_deviation': 1.0,
          'sensitivities': [3.0],
          'sampling_probs': [1.0],
          'adjacency_type': ADD,
          'epsilon': 1.0,
          'expected_delta': 0.78760074,
      },
      {
          'testcase_name': 'sampled_gaussian_add_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': ADD,
          'epsilon': 1.0,
          'expected_delta': 0.0231362104090899,
      },
      {
          'testcase_name': 'sampled_gaussian_add_2',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 3.0],
          'sampling_probs': [0.3, 0.7],
          'adjacency_type': ADD,
          'epsilon': 1.0,
          'expected_delta': 0.1195051215523554,
      },
      {
          'testcase_name': 'mixture_gaussian_add_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'adjacency_type': ADD,
          'epsilon': 1.0,
          'expected_delta': 0.036691263832032806,
      },
      {
          'testcase_name': 'mixture_gaussian_add_2',
          'standard_deviation': 1.0,
          'sensitivities': [1.0, 2.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': ADD,
          'epsilon': 1.0,
          'expected_delta': 0.3894964356580768,
      },
      {
          'testcase_name': 'gaussian_remove_1',
          'standard_deviation': 1.0,
          'sensitivities': [1.0],
          'sampling_probs': [1.0],
          'adjacency_type': REM,
          'epsilon': 1.0,
          'expected_delta': 0.12693674,
      },
      {
          'testcase_name': 'gaussian_remove_2',
          'standard_deviation': 1.0,
          'sensitivities': [3.0],
          'sampling_probs': [1.0],
          'adjacency_type': REM,
          'epsilon': 1.0,
          'expected_delta': 0.78760074,
      },
      {
          'testcase_name': 'sampled_gaussian_remove_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': REM,
          'epsilon': 1.0,
          'expected_delta': 0.0816951786585355,
      },
      {
          'testcase_name': 'sampled_gaussian_remove_2',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 3.0],
          'sampling_probs': [0.3, 0.7],
          'adjacency_type': REM,
          'epsilon': 1.0,
          'expected_delta': 0.5356298793262404,
      },
      {
          'testcase_name': 'mixture_gaussian_remove_1',
          'standard_deviation': 1.0,
          'sensitivities': [0.0, 1.0, 2.0],
          'sampling_probs': [0.2, 0.6, 0.2],
          'adjacency_type': REM,
          'epsilon': 1.0,
          'expected_delta': 0.15768284088654105,
      },
      {
          'testcase_name': 'mixture_gaussian_remove_2',
          'standard_deviation': 1.0,
          'sensitivities': [1.0, 2.0],
          'sampling_probs': [0.2, 0.8],
          'adjacency_type': REM,
          'epsilon': 1.0,
          'expected_delta': 0.433276675545065,
      },
  )
  def test_get_delta_for_epsilon(
      self,
      standard_deviation,
      sensitivities,
      sampling_probs,
      adjacency_type,
      epsilon,
      expected_delta,
  ):
    pl = privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
        standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        adjacency_type=adjacency_type,
    )
    self.assertAlmostEqual(pl.get_delta_for_epsilon(epsilon), expected_delta)

  @parameterized.named_parameters(
      _add_and_remove_test_cases(
          _gaussian_test_cases(5)
          + _subsampled_gaussian_test_cases(5)
          + _mixture_gaussian_with_zero_test_cases(5)
          + _mixture_gaussian_without_zero_test_cases(5)
      )
  )
  def test_zero_sampling_probs_ignored(
      self, standard_deviation, sensitivities, sampling_probs, adjacency_type
  ):
    """Tests that zeros in sampling_probs are ignored."""
    padded_sensitivities = sensitivities + [100.0]
    padded_sampling_probs = sampling_probs + [0.0]
    expected_pl = privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
        standard_deviation,
        sensitivities=sensitivities,
        sampling_probs=sampling_probs,
        adjacency_type=adjacency_type,
    )
    padded_pl = privacy_loss_mechanism.MixtureGaussianPrivacyLoss(
        standard_deviation,
        sensitivities=padded_sensitivities,
        sampling_probs=padded_sampling_probs,
        adjacency_type=adjacency_type,
    )
    self.assertAlmostEqual(expected_pl.get_delta_for_epsilon(1),
                           padded_pl.get_delta_for_epsilon(1))


if __name__ == '__main__':
  unittest.main()
