# Copyright 2021 Google LLC.
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
"""Test utilities for clustering."""

from clustering import central_privacy_utils
from clustering import clustering_params
from clustering import coreset_params
from clustering import privacy_calculator


def get_test_coreset_param(epsilon: float = 1.0,
                           delta: float = 1e-2,
                           gaussian_std_dev_multiplier: float = 2.3,
                           laplace_param_multiplier: float = 32.1,
                           min_num_points_in_branching_node: int = 4,
                           min_num_points_in_node: int = 2,
                           max_depth: int = 4,
                           radius: float = 1) -> coreset_params.CoresetParam:
  # pylint: disable=g-doc-args
  """Returns coreset_param with defaults for params not needed for testing.

  Usage: Explicitly pass in parameters that are relied on in the test.
  """
  privacy_param = clustering_params.DifferentialPrivacyParam(
      epsilon=epsilon, delta=delta)
  multipliers = clustering_params.PrivacyCalculatorMultiplier(
      gaussian_std_dev_multiplier=gaussian_std_dev_multiplier,
      laplace_param_multiplier=laplace_param_multiplier,
  )
  tree_param = clustering_params.TreeParam(
      min_num_points_in_branching_node=min_num_points_in_branching_node,
      min_num_points_in_node=min_num_points_in_node,
      max_depth=max_depth)
  pcalc = privacy_calculator.PrivacyCalculator(
      privacy_param, radius, max_depth, multipliers
  )
  coreset_param = coreset_params.CoresetParam(
      pcalc=pcalc,
      tree_param=tree_param,
      short_description='TestCoresetParam',
      radius=radius)
  return coreset_param


class TestPrivacyCalculator(privacy_calculator.PrivacyCalculator):
  def __init__(self, gaussian_std_dev, sensitivity, laplace_param):
    self.average_privacy_param = central_privacy_utils.AveragePrivacyParam(
        gaussian_std_dev, sensitivity
    )
    self.count_privacy_param = central_privacy_utils.CountPrivacyParam(
        laplace_param
    )


def get_test_privacy_calculator(
    gaussian_std_dev: float = 32.6,
    sensitivity: float = 6.4,
    laplace_param: float = 0.8) -> privacy_calculator.PrivacyCalculator:
  # pylint: disable=g-doc-args
  """Returns privacy calculator with defaults for params not needed for testing.

  Usage: Explicitly pass in parameters that are relied on in the test.
  """
  return TestPrivacyCalculator(gaussian_std_dev, sensitivity, laplace_param)
