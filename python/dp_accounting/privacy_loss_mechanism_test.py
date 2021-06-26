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

"""Tests for privacy_loss_mechanism."""

import math
import unittest
from absl.testing import parameterized
from scipy import stats

from dp_accounting import common
from dp_accounting import privacy_loss_mechanism
from dp_accounting import test_util


class LaplacePrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, -0.1, 1), (1, 1, 2, -1), (1, 1, 0.3, 0.4),
                            (4, 4, -0.4, 1), (5, 5, 7, -1), (7, 7, 2.1, 0.4))
  def test_laplace_privacy_loss(self, parameter, sensitivity, x,
                                expected_privacy_loss):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters(
      (1, 1, 1, 0), (1, 1, -1, math.inf), (1, 1, 0.4, 0.3), (4, 4, 1, 0),
      (5, 5, -1, math.inf), (7, 7, 0.4, 2.1), (1, 1, 2, -math.inf),
      (3, 1, 3.1, -math.inf), (4, 4, 1.1, -math.inf))
  def test_laplace_inverse_privacy_loss(self, parameter, sensitivity,
                                        privacy_loss, expected_x):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

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
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.dictionary_almost_equal(self,
                                      expected_tail_probability_mass_function,
                                      tail_pld.tail_probability_mass_function)

  @parameterized.parameters((-3.0, 1), (0, 1), (1, 0), (2.0, -1.0))
  def test_laplace_value_errors(self, parameter, sensitivity):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.LaplacePrivacyLoss(
          parameter, sensitivity=sensitivity)

  @parameterized.parameters((1, 1, 0, 1), (1, 1, 0.1, 1), (2, 1, 0.01, 2),
                            (1, 3, 0.01, 0.33333333))
  def test_laplace_from_privacy_parameters(self, sensitivity, epsilon, delta,
                                           expected_parameter):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss.from_privacy_guarantee(
        common.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity)
    self.assertAlmostEqual(expected_parameter, pl.parameter)

  @parameterized.parameters((1, 1, 1, 0), (3, 3, 1, 0), (2, 4, 2, 0),
                            (2, 4, 0.5, 0.52763345), (1, 1, 0, 0.39346934),
                            (2, 2, 0, 0.39346934), (1, 1, -2, 0.86466472))
  def test_laplace_get_delta_for_epsilon(
      self, parameter, sensitivity, epsilon, expected_delta):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))


class GaussianPrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, 5, -4.5), (1, 1, -3, 3.5), (1, 2, 3, -4),
                            (4, 4, 20, -4.5), (5, 5, -15, 3.5), (7, 14, 21, -4))
  def test_gaussian_privacy_loss(self, standard_deviation, sensitivity, x,
                                 expected_privacy_loss):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters((1, 1, -4.5, 5), (1, 1, 3.5, -3), (1, 2, -4, 3),
                            (4, 4, -4.5, 20), (5, 5, 3.5, -15), (7, 14, -4, 21))
  def test_gaussian_inverse_privacy_loss(self, standard_deviation, sensitivity,
                                         privacy_loss, expected_x):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

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
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        pessimistic_estimate=pessimistic_estimate,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-1))
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.dictionary_almost_equal(self,
                                      expected_tail_probability_mass_function,
                                      tail_pld.tail_probability_mass_function)

  @parameterized.parameters((0, 1), (-10, 2), (4, 0), (2, -1), (1, 1, 1))
  def test_gaussian_value_errors(self, standard_deviation, sensitivity,
                                 log_mass_truncation_bound=-50):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.GaussianPrivacyLoss(
          standard_deviation,
          sensitivity=sensitivity,
          log_mass_truncation_bound=log_mass_truncation_bound)

  @parameterized.parameters((1, 1, 0), (1, 1, 1.1), (1, 1, -0.1))
  def test_gaussian_from_privacy_parameters_value_errors(
      self, sensitivity, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity)

  @parameterized.parameters((1, 1, 0.12693674, 1), (2, 1, 0.12693674, 2),
                            (3, 1, 0.78760074, 1), (6, 1, 0.78760074, 2),
                            (1, 2, 0.02092364, 1), (5, 2, 0.02092364, 5),
                            (1, 16, 1e-5, 0.344), (2, 16, 1e-5, 0.688))
  def test_gaussian_from_privacy_parameters(self, sensitivity, epsilon, delta,
                                            expected_standard_deviation):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
        common.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity)
    self.assertAlmostEqual(expected_standard_deviation, pl.standard_deviation,
                           3)

  @parameterized.parameters((1, 1, 1, 0.12693674), (2, 2, 1, 0.12693674),
                            (1, 3, 1, 0.78760074), (2, 6, 1, 0.78760074),
                            (1, 1, 2, 0.02092364), (5, 5, 2, 0.02092364))
  def test_gaussian_get_delta_for_epsilon(
      self, standard_deviation, sensitivity, epsilon, expected_delta):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))


class DiscreteLaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, 0, 1), (1, 1, 1, -1), (0.3, 2, 0, 0.6),
                            (0.3, 2, 1, 0), (0.3, 2, 2, -0.6))
  def test_discrete_laplace_privacy_loss(self, parameter, sensitivity, x,
                                         expected_privacy_loss):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters((1, 1, 0.4), (2, 7, -1.1))
  def test_discrete_laplace_privacy_loss_value_errors(
      self, parameter, sensitivity, x):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    with self.assertRaises(ValueError):
      pl.privacy_loss(x)

  @parameterized.parameters((1, 1, 1.1, -math.inf), (1, 1, 0.9, 0),
                            (1, 1, -1, math.inf), (0.3, 2, 0.7, -math.inf),
                            (0.3, 2, 0.2, 0), (0.3, 2, 0, 1),
                            (0.3, 2, -0.6, math.inf))
  def test_discrete_laplace_inverse_privacy_loss(self, parameter, sensitivity,
                                                 privacy_loss, expected_x):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

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
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.dictionary_almost_equal(self,
                                      expected_tail_probability_mass_function,
                                      tail_pld.tail_probability_mass_function)

  @parameterized.parameters((-3, 1), (0, 1), (2, 0.5), (2.0, -1), (1.0, 0))
  def test_discrete_laplace_value_errors(self, parameter, sensitivity):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
          parameter,
          sensitivity=sensitivity)

  @parameterized.parameters((-1, 1, 0.1), (0.5, 1, 0.1), (0, 1, 0.2),
                            (1, 1, -0.1), (1, 1, 1.1))
  def test_discrete_laplace_from_privacy_parameters_value_errors(
      self, sensitivity, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteLaplacePrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity)

  @parameterized.parameters((1, 1, 0, 1), (1, 1, 0.1, 1), (2, 1, 0.01, 0.5),
                            (1, 3, 0.01, 3))
  def test_discrete_laplace_from_privacy_parameters(self, sensitivity, epsilon,
                                                    delta, expected_parameter):
    pl = (privacy_loss_mechanism.DiscreteLaplacePrivacyLoss
          .from_privacy_guarantee(
              common.DifferentialPrivacyParameters(
                  epsilon, delta),
              sensitivity))
    self.assertAlmostEqual(expected_parameter, pl.parameter)

  @parameterized.parameters((1, 1, 1, 0), (0.333333, 3, 1, 0), (0.5, 4, 2, 0),
                            (0.5, 4, 0.5, 0.54202002), (0.5, 4, 1, 0.39346934),
                            (0.5, 4, -0.5, 0.72222110))
  def test_discrete_laplace_get_delta_for_epsilon(
      self, parameter, sensitivity, epsilon, expected_delta):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))


class DiscreteGaussianPrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters((1, 1, 5, -4.5), (1, 1, -3, 3.5), (1, 2, 3, -4),
                            (4, 4, 20, -4.5), (5, 5, -15, 3.5), (7, 14, 21, -4),
                            (1, 1, -12, math.inf))
  def test_discrete_gaussian_privacy_loss(self, sigma, sensitivity,
                                          x, expected_privacy_loss):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters((1, 1, 0.4), (2, 7, -1.1))
  def test_discrete_gaussian_privacy_loss_value_errors(
      self, sigma, sensitivity, x):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity)
    with self.assertRaises(ValueError):
      pl.privacy_loss(x)

  @parameterized.parameters((1, 1, -4.5, 5), (1, 1, 3.5, -3), (1, 2, -4, 3),
                            (4, 4, -4.51, 20), (5, 5, 3.49, -15),
                            (7, 14, -4, 21))
  def test_discrete_gaussian_inverse_privacy_loss(self, sigma, sensitivity,
                                                  privacy_loss, expected_x):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters((1, 1, 2, -1, 2, {
      math.inf: 0.05448868
  }), (1, 2, 2, 0, 2, {
      math.inf: 0.29869003
  }))
  def test_discrete_gaussian_privacy_loss_tail(
      self, sigma, sensitivity, truncation_bound, expected_lower_x_truncation,
      expected_upper_x_truncation, expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity, truncation_bound=truncation_bound)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.dictionary_almost_equal(self,
                                      expected_tail_probability_mass_function,
                                      tail_pld.tail_probability_mass_function)

  @parameterized.parameters((-3, 1), (0, 1), (2, 0.5), (1.0, 0), (2.0, -1),
                            (2.0, 4, 1))
  def test_discrete_gaussian_value_errors(self,
                                          sigma,
                                          sensitivity,
                                          truncation_bound=None):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
          sigma, sensitivity=sensitivity, truncation_bound=truncation_bound)

  @parameterized.parameters((1, 1, 1, {
      -1.5: 0,
      -1: 0.27406862,
      0: 0.7259314,
      1: 1,
      1.5: 1
  }), (3, 2, 2, {
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

  @parameterized.parameters((1, 1, 1, 0.7403629), (3, 2, 2, 1.3589226))
  def test_discrete_gaussian_std(self, sigma, sensitivity, truncation_bound,
                                 expected_std):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity, truncation_bound=truncation_bound)
    self.assertAlmostEqual(expected_std, pl.standard_deviation())

  @parameterized.parameters((-1, 1, 0.1), (0.5, 1, 0.1), (0, 1, 0.2), (1, 1, 0),
                            (1, 1, 1.1), (1, 1, -0.1))
  def test_discrete_gaussian_from_privacy_parameters_value_errors(
      self, sensitivity, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteGaussianPrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity)

  @parameterized.parameters(
      (1, 1, 0.12693674, 1.041), (2, 1, 0.12693674, 1.972),
      (3, 1, 0.78760074, 0.993), (6, 1, 0.78760074, 2.014),
      (1, 2, 0.02092364, 1.038), (5, 2, 0.02092364, 5.008),
      (1, 16, 1e-5, 0.306), (2, 16, 1e-5, 0.703))
  def test_discrete_gaussian_from_privacy_parameters(self, sensitivity, epsilon,
                                                     delta, expected_sigma):
    pl = (
        privacy_loss_mechanism.DiscreteGaussianPrivacyLoss
        .from_privacy_guarantee(
            common.DifferentialPrivacyParameters(epsilon, delta), sensitivity))
    self.assertAlmostEqual(expected_sigma, pl._sigma, 3)


if __name__ == '__main__':
  unittest.main()
