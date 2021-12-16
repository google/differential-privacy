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
"""Default values for the clustering algorithm parameters."""

import typing
import numpy as np

from clustering import central_privacy_utils
from clustering import clustering_params
from dp_accounting import accountant
from dp_accounting import common

PrivateCount = int


def default_tree_param(
    k: int, data: clustering_params.Data,
    privacy_param: clustering_params.DifferentialPrivacyParam,
    privacy_budget_split: clustering_params.PrivacyBudgetSplit
) -> typing.Tuple[clustering_params.TreeParam, PrivateCount]:
  """Heuristic tree param based on the data and number of clusters.

  Args:
    k: Number of clusters to divide the data into.
    data: Data to find centers for.
    privacy_param: privacy parameters for the algorithm.
    privacy_budget_split: budget split between different computations.

  Returns:
    (default TreeParam, private count). The private count is provided so that
    it doesn't need to be re-computed.
  """
  # Note that max_depth is used for the private count calculation so it cannot
  # depend on the count.
  # Chosen experimentally over multiple datasets.
  max_depth = 20

  # Calculate the standard deviation for the sum noise using a sensitivity of 1.
  if privacy_param.epsilon == np.inf:
    sum_sigma = 0
  else:
    sum_sigma = accountant.get_smallest_gaussian_noise(
        common.DifferentialPrivacyParameters(
            privacy_param.epsilon * privacy_budget_split.frac_sum,
            privacy_param.delta),
        num_queries=1,
        sensitivity=1.0)

  private_count = central_privacy_utils.get_private_count(
      data.num_points,
      central_privacy_utils.CountPrivacyParam.compute_group_count_privacy_param(
          privacy_param, privacy_budget_split, max_depth))

  # We can consider the noise as distributed amongst the points that are being
  # summed. The noise has l2-norm roughly sqrt(dimension) * sum_sigma * radius,
  # so if we distribute among 10 * sqrt(dimension) * sum_sigma, each point
  # has noise roughly 0.1 * radius.
  num_points_in_node_for_low_noise = int(10 * np.sqrt(data.dim) * sum_sigma)

  # We want to at least have the ability to consider a node per cluster, even
  # if the noise might be higher than we'd like.
  min_num_points_in_node = min(num_points_in_node_for_low_noise,
                               private_count // (2 * k))

  # min_num_points_in_node must always be at least 1. Note it's possible that
  # the private_count is negative, so we should ensure this max is done last.
  min_num_points_in_node = max(1, min_num_points_in_node)
  min_num_points_in_branching_node = 3 * min_num_points_in_node

  return (clustering_params.TreeParam(
      min_num_points_in_branching_node=min_num_points_in_branching_node,
      min_num_points_in_node=min_num_points_in_node,
      max_depth=max_depth), private_count)
