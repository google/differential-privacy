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
"""Tests for privacy_calculator."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from clustering import central_privacy_utils
from clustering import clustering_params
from clustering import privacy_calculator
from dp_accounting import dp_event
from dp_accounting import mechanism_calibration
from dp_accounting.pld import accountant
from dp_accounting.pld import common


class PrivacyCalculatorTest(parameterized.TestCase):

  def test_privacy_calculator_from_budget_split(self):
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=10, delta=1e-2)
    budget_split = clustering_params.PrivacyBudgetSplit(
        frac_sum=0.7, frac_group_count=0.3)
    radius = 7.2
    depth = 3
    pcalc = privacy_calculator.PrivacyCalculator.from_budget_split(
        privacy_param, budget_split, radius, depth)
    self.assertEqual(
        pcalc.average_privacy_param,
        central_privacy_utils.AveragePrivacyParam.from_budget_split(
            privacy_param, budget_split, radius))
    self.assertEqual(
        pcalc.count_privacy_param,
        central_privacy_utils.CountPrivacyParam.from_budget_split(
            privacy_param, budget_split, depth))

  def test_make_clustering_event(self):
    gaussian_std_dev = 5.4
    sensitivity = 2
    laplace_param = 5
    max_depth = 25
    clustering_event = privacy_calculator.make_clustering_event(
        gaussian_std_dev, laplace_param, sensitivity, max_depth)
    gaussian_event = dp_event.GaussianDpEvent(2.7)
    laplace_event = dp_event.SelfComposedDpEvent(
        dp_event.LaplaceDpEvent(0.2), 26)
    self.assertEqual(clustering_event,
                     dp_event.ComposedDpEvent([gaussian_event, laplace_event]))

  def test_make_clustering_event_zero_std_dev(self):
    gaussian_std_dev = 0
    sensitivity = 2
    laplace_param = 5
    max_depth = 25
    clustering_event = privacy_calculator.make_clustering_event(
        gaussian_std_dev, laplace_param, sensitivity, max_depth)
    laplace_event = dp_event.SelfComposedDpEvent(
        dp_event.LaplaceDpEvent(0.2), 26)
    self.assertEqual(
        clustering_event,
        dp_event.ComposedDpEvent([dp_event.NonPrivateDpEvent(), laplace_event]))

  def test_make_clustering_event_inf_laplace_event(self):
    gaussian_std_dev = 5.4
    sensitivity = 2
    laplace_param = np.inf
    max_depth = 25
    clustering_event = privacy_calculator.make_clustering_event(
        gaussian_std_dev, laplace_param, sensitivity, max_depth)
    gaussian_event = dp_event.GaussianDpEvent(2.7)
    self.assertEqual(
        clustering_event,
        dp_event.ComposedDpEvent([gaussian_event,
                                  dp_event.NonPrivateDpEvent()]))

  @mock.patch.object(
      privacy_calculator,
      "make_clustering_event",
      return_value=dp_event.ComposedDpEvent([
          dp_event.GaussianDpEvent(2.4),
          dp_event.SelfComposedDpEvent(dp_event.LaplaceDpEvent(0.8), 26)
      ]),
      autospec=True)
  def test_make_clustering_event_from_param(self, mock_make_clustering_event):
    multipliers = clustering_params.PrivacyCalculatorMultiplier(1.2, 0.2)
    sensitivity = 2
    max_depth = 25
    alpha = 4.0
    clustering_event = privacy_calculator.make_clustering_event_from_param(
        multipliers, sensitivity, max_depth, alpha)
    mock_args = mock_make_clustering_event.call_args[1]
    self.assertAlmostEqual(mock_args["sum_std_dev"], 9.6)
    self.assertAlmostEqual(mock_args["count_laplace_param"], 1.25)
    self.assertAlmostEqual(mock_args["sensitivity"], 2)
    self.assertAlmostEqual(mock_args["max_depth"], 25)
    self.assertEqual(
        clustering_event,
        dp_event.ComposedDpEvent([
            dp_event.GaussianDpEvent(2.4),
            dp_event.SelfComposedDpEvent(dp_event.LaplaceDpEvent(0.8), 26)
        ]))

  @mock.patch.object(
      accountant,
      "get_smallest_gaussian_noise",
      return_value=8.4,
      autospec=True)
  def test_get_alpha_interval(self, mock_smallest_gaussian_noise):
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=2.0, delta=1e-6)
    radius = 3.2
    multipliers = clustering_params.PrivacyCalculatorMultiplier(3.5, 2.1)
    interval = privacy_calculator.get_alpha_interval(privacy_param, radius,
                                                     multipliers)

    # Check arguments.
    mock_args = mock_smallest_gaussian_noise.call_args[1]
    self.assertAlmostEqual(mock_args["privacy_parameters"],
                           common.DifferentialPrivacyParameters(2.0, 1e-6))
    self.assertAlmostEqual(mock_args["num_queries"], 1)
    self.assertAlmostEqual(mock_args["sensitivity"], radius)

    self.assertEqual(interval,
                     mechanism_calibration.LowerEndpointAndGuess(0.75, 1.5))

  def test_get_alpha_interval_error(self):
    radius = 3.2
    multipliers = clustering_params.PrivacyCalculatorMultiplier(3.5, 2.1)
    # Infinite epsilon.
    with self.assertRaises(ValueError):
      _ = privacy_calculator.get_alpha_interval(
          clustering_params.DifferentialPrivacyParam(
              epsilon=np.inf, delta=1e-6), radius, multipliers)

    # Delta = 1.
    with self.assertRaises(ValueError):
      _ = privacy_calculator.get_alpha_interval(
          clustering_params.DifferentialPrivacyParam(epsilon=1.0, delta=1),
          radius, multipliers)

  @parameterized.named_parameters(("basic", 10, 1e-2),
                                  ("inf_eps", np.inf, 1e-2),
                                  ("one_delta", 10, 1))
  def test_validate_accounting(self, epsilon, delta):
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=epsilon, delta=delta)
    budget_split = clustering_params.PrivacyBudgetSplit(
        frac_sum=0.7, frac_group_count=0.3)
    radius = 7.2
    depth = 3
    pcalc = privacy_calculator.PrivacyCalculator.from_budget_split(
        privacy_param, budget_split, radius, depth)
    pcalc.validate_accounting(privacy_param, depth)

  @parameterized.named_parameters(("basic", 0.4, 4.0),
                                  ("no_sum_noise", 0, 0.01),
                                  ("no_count_noise", 0.5, np.inf))
  def test_validate_accounting_error(self, sum_std_dev, count_laplace_param):
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=10, delta=1e-2)
    depth = 3
    pcalc = privacy_calculator.PrivacyCalculator(
        central_privacy_utils.AveragePrivacyParam(sum_std_dev, 10),
        central_privacy_utils.CountPrivacyParam(count_laplace_param))
    with self.assertRaisesRegex(
        ValueError,
        expected_regex="Accounted privacy params greater than allowed: "
        r".* > \(10, 0\.01\)"):
      pcalc.validate_accounting(privacy_param, depth)

  @parameterized.named_parameters(("basic", 10, 1e-2),
                                  ("inf_eps", np.inf, 1e-2),
                                  ("one_delta", 10, 1))
  def test_from_mechanism_calibration(self, epsilon, delta):
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=epsilon, delta=delta)
    radius = 3.2
    max_depth = 12
    multipliers = clustering_params.PrivacyCalculatorMultiplier(3.5, 2.1)
    pcalc = privacy_calculator.PrivacyCalculator.from_mechanism_calibration(
        privacy_param, radius, max_depth, multipliers)
    # Result should be within the privacy budget.
    pcalc.validate_accounting(privacy_param, max_depth)
    self.assertEqual(pcalc.average_privacy_param.sensitivity, radius)

  @parameterized.named_parameters(("basic", 10, 1e-2),
                                  ("inf_eps", np.inf, 1e-2),
                                  ("one_delta", 10, 1))
  def test_from_mechanism_calibration_scaled_multipliers(self, epsilon, delta):
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=epsilon, delta=delta)
    radius = 3.2
    max_depth = 12
    multipliers = clustering_params.PrivacyCalculatorMultiplier(3.5, 2.1)
    pcalc1 = privacy_calculator.PrivacyCalculator.from_mechanism_calibration(
        privacy_param, radius, max_depth, multipliers)

    # When we scale the multipliers by a constant, the result should be the
    # same.
    two_x_multipliers = clustering_params.PrivacyCalculatorMultiplier(7.0, 4.2)
    pcalc2 = privacy_calculator.PrivacyCalculator.from_mechanism_calibration(
        privacy_param, radius, max_depth, two_x_multipliers)
    self.assertAlmostEqual(
        pcalc1.average_privacy_param.gaussian_standard_deviation,
        pcalc2.average_privacy_param.gaussian_standard_deviation,
        delta=1e-4)
    self.assertAlmostEqual(
        pcalc1.count_privacy_param.laplace_param,
        pcalc2.count_privacy_param.laplace_param,
        delta=1e-4)


if __name__ == "__main__":
  absltest.main()
