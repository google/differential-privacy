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
from typing import Type

import numpy as np
from scipy import stats

from clustering import clustering_params
from dp_accounting.pld import accountant
from dp_accounting.pld import common


@dataclasses.dataclass
class AveragePrivacyParam():
  """Privacy parameters for calling get_private_average()."""
  gaussian_standard_deviation: float
  sensitivity: float

  def __post_init__(self):
    # Standard deviation can be 0 to indicate no noise.
    if self.gaussian_standard_deviation < 0:
      raise ValueError(
          f'Gaussian standard deviation was {self.gaussian_standard_deviation}, '
          'but it must be nonnegative.')

    if self.sensitivity <= 0:
      raise ValueError(
          f'Sensitivity for averaging was {self.sensitivity}, but it must be '
          'positive.')

  @classmethod
  def from_budget_split(
      cls: Type['AveragePrivacyParam'],
      privacy_param: clustering_params.DifferentialPrivacyParam,
      privacy_budget_split: clustering_params.PrivacyBudgetSplit,
      radius: float) -> 'AveragePrivacyParam':
    """Calculates standard deviation by splitting the privacy budget."""
    split_epsilon = (privacy_budget_split.frac_sum * privacy_param.epsilon)
    if split_epsilon == np.inf:
      gaussian_standard_deviation = 0
    else:
      gaussian_standard_deviation = accountant.get_smallest_gaussian_noise(
          common.DifferentialPrivacyParameters(split_epsilon,
                                               privacy_param.delta),
          num_queries=1,
          sensitivity=radius)
    return cls(gaussian_standard_deviation, radius)


def get_private_average(nonprivate_points: np.ndarray, private_count: int,
                        average_privacy_param: AveragePrivacyParam,
                        dim: int) -> np.ndarray:
  """Returns a differentially private average of the given data points.

  Args:
    nonprivate_points: data points to be averaged, may be empty.
    private_count: differentially private count of the number of data points.
      This is provided to save privacy budget since, in our applications, it is
      often already computed elsewhere. Required to be >= 1.
    average_privacy_param: privacy parameters for the private average.
    dim: dimension of the data points.

  Returns:
    A differentially private average of the given data points.
  """
  if private_count < 1:
    raise ValueError(
        f'get_private_average() called with private_count={private_count}')

  sum_points = np.sum(nonprivate_points, axis=0)

  # Add noise.
  sum_points += np.random.normal(
      scale=average_privacy_param.gaussian_standard_deviation, size=dim)
  return sum_points / private_count


@dataclasses.dataclass
class CountPrivacyParam():
  """Privacy parameters for calling get_private_count()."""
  laplace_param: float

  def __post_init__(self):
    # No noise means laplace_param == inf, not 0. We invert the laplace param
    # for accounting.
    if self.laplace_param <= 0:
      raise ValueError(f'Laplace param was {self.laplace_param}, '
                       'but it must be positive.')

  @classmethod
  def from_budget_split(
      cls: Type['CountPrivacyParam'],
      clustering_privacy_param: clustering_params.DifferentialPrivacyParam,
      budget_split: clustering_params.PrivacyBudgetSplit,
      depth: int) -> 'CountPrivacyParam':
    """Computes laplace param by splitting the budget."""
    # Split epsilon between each level of the tree starting with level 0. Depth
    # is based on the number of edges in the path, so add one to the depth to
    # get the number of levels.
    split_epsilon = (budget_split.frac_group_count *
                     clustering_privacy_param.epsilon) / (
                         depth + 1.0)
    return cls(laplace_param=split_epsilon)


def get_private_count(nonprivate_count: int,
                      count_privacy_param: CountPrivacyParam) -> int:
  """Computes differentially private count.

  Args:
    nonprivate_count: the (unnoised) count of the number of data points in a
      group.
    count_privacy_param: privacy parameters for calculating the private count.

  Returns:
    The differentially private count where a Discrete Laplace noise with
    appropriate parameter is added to the non-private count.
  """
  if count_privacy_param.laplace_param == np.inf:
    return nonprivate_count
  return nonprivate_count + stats.dlaplace.rvs(
      count_privacy_param.laplace_param)
