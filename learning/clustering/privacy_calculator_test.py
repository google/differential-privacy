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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from clustering import central_privacy_utils
from clustering import clustering_params
from clustering import privacy_calculator
from dp_accounting import dp_event


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

  @parameterized.named_parameters(
      ("basic", 10, 1e-2), ("inf_eps", np.inf, 1e-2), ("one_delta", 10, 1))
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


if __name__ == "__main__":
  absltest.main()
