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
from clustering import central_privacy_utils
from clustering import clustering_params
from clustering import privacy_calculator


class PrivacyCalculatorTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
