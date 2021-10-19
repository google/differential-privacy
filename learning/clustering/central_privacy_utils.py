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
"""Utilities for adding noise to satisfy central privacy."""

import dataclasses

import numpy as np
from scipy import stats

from clustering import clustering_params
from dp_accounting import accountant
from dp_accounting import common


def get_private_average(nonprivate_points: np.ndarray, private_count: int,
                        clustering_param: clustering_params.ClusteringParam,
                        dim: int) -> np.ndarray:
  """Returns a differentially private average of the given data points.

  Args:
    nonprivate_points: data points to be averaged, may be empty.
    private_count: differentially private count of the number of data points.
      This is provided to save privacy budget since, in our applications, it is
      often already computed elsewhere. Required to be >= 1.
    clustering_param: parameters of the clustering algorithm.
    dim: dimension of the data points.

  Returns:
    A differentially private average of the given data points.
  """
  if private_count < 1:
    raise ValueError(
        f"get_private_average() called with private_count={private_count}")

  sum_points = np.sum(nonprivate_points, axis=0)
  epsilon_sum = (
      clustering_param.privacy_budget_split.frac_sum *
      clustering_param.privacy_param.epsilon)

  if epsilon_sum == np.inf:
    return sum_points / private_count

  gaussian_standard_deviation = accountant.get_smallest_gaussian_noise(
      common.DifferentialPrivacyParameters(
          epsilon_sum, clustering_param.privacy_param.delta),
      num_queries=1,
      sensitivity=clustering_param.radius)
  sum_points += np.random.normal(scale=gaussian_standard_deviation, size=dim)
  return sum_points / private_count


@dataclasses.dataclass
class PrivateCountParam():
  """Parameters for calling get_private_count().

  Attributes:
    privacy_param: privacy parameters for the algorithm
    privacy_budget_split: budget split between different computations
    max_tree_depth: maximum depth for the tree
  """
  privacy_param: clustering_params.DifferentialPrivacyParam
  privacy_budget_split: clustering_params.PrivacyBudgetSplit
  max_tree_depth: int

  @classmethod
  def from_clustering_param(
      cls, clustering_param: clustering_params.ClusteringParam):
    return cls(clustering_param.privacy_param,
               clustering_param.privacy_budget_split,
               clustering_param.tree_param.max_depth)


def get_private_count(nonprivate_count: int,
                      private_count_param: PrivateCountParam) -> int:
  """Computes differentially private count.

  Assume that the privacy budget for group count (specified in
  clustering_params) is divided equally across the levels of the tree.

  Args:
    nonprivate_count: the (unnoised) count of the number of data points in a
      group.
    private_count_param: parameters for calculating the private count.

  Returns:
    The differentially private count where a Discrete Laplace noise with
    appropriate parameter is added to the non-private count.
  """
  # Split epsilon between each level of the tree starting with level 0. Depth
  # is based on the number of edges in the path, so add one to the depth to get
  # the number of levels.
  epsilon_per_level = (
      private_count_param.privacy_budget_split.frac_group_count *
      private_count_param.privacy_param.epsilon /
      (private_count_param.max_tree_depth + 1))
  if epsilon_per_level == np.inf:
    return nonprivate_count
  return nonprivate_count + stats.dlaplace.rvs(epsilon_per_level)
