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
"""Calculates average and count privacy params."""

import dataclasses
from typing import Type

from absl import logging
import numpy as np

from clustering import central_privacy_utils
from clustering import clustering_params
from dp_accounting import dp_event
from dp_accounting import dp_event_builder
from dp_accounting.pld import pld_privacy_accountant


def make_clustering_event(sum_std_dev: float, count_laplace_param: float,
                          sensitivity: float,
                          max_depth: int) -> dp_event.DpEvent:
  """Returns a DpEvent for clustering."""
  builder = dp_event_builder.DpEventBuilder()

  if sum_std_dev == 0:
    builder.compose(dp_event.NonPrivateDpEvent())
  else:
    builder.compose(dp_event.GaussianDpEvent(sum_std_dev / sensitivity))

  # Depth is based on the number of edges in the path, so add one to the depth
  # to get the number of levels.
  if count_laplace_param == np.inf:
    builder.compose(dp_event.NonPrivateDpEvent())
  else:
    builder.compose(
        dp_event.LaplaceDpEvent(1 / count_laplace_param), max_depth + 1)
  return builder.build()


@dataclasses.dataclass
class PrivacyCalculator():
  """Calculates and returns privacy parameters."""
  average_privacy_param: central_privacy_utils.AveragePrivacyParam
  count_privacy_param: central_privacy_utils.CountPrivacyParam

  @classmethod
  def from_budget_split(
      cls: Type['PrivacyCalculator'],
      privacy_param: clustering_params.DifferentialPrivacyParam,
      privacy_budget_split: clustering_params.PrivacyBudgetSplit, radius: float,
      max_depth: int) -> 'PrivacyCalculator':
    """Calculates privacy parameters by splitting the privacy budget."""
    if privacy_param.privacy_model != clustering_params.PrivacyModel.CENTRAL:
      raise NotImplementedError(
          f'Currently unsupported privacy model: {privacy_param.privacy_model}')

    average_privacy_param = central_privacy_utils.AveragePrivacyParam.from_budget_split(
        privacy_param, privacy_budget_split, radius)
    count_privacy_param = central_privacy_utils.CountPrivacyParam.from_budget_split(
        privacy_param, privacy_budget_split, max_depth)
    return cls(average_privacy_param, count_privacy_param)

  def validate_accounting(
      self, privacy_param: clustering_params.DifferentialPrivacyParam,
      max_depth: int):
    """Errors if the params exceed the privacy budget."""
    if privacy_param.epsilon == np.inf or privacy_param.delta >= 1:
      return

    clustering_event = make_clustering_event(
        self.average_privacy_param.gaussian_standard_deviation,
        self.count_privacy_param.laplace_param,
        self.average_privacy_param.sensitivity, max_depth)

    acct = pld_privacy_accountant.PLDAccountant()
    acct.compose(clustering_event)
    calculated_epsilon = acct.get_epsilon(privacy_param.delta)
    calculated_delta = acct.get_delta(privacy_param.epsilon)

    logging.info('Accounted epsilon: %s', calculated_epsilon)
    logging.info('Accounted delta: %s', calculated_delta)

    if (calculated_epsilon > privacy_param.epsilon or
        calculated_delta > privacy_param.delta):
      raise ValueError('Accounted privacy params greater than allowed: '
                       f'({calculated_epsilon}, {calculated_delta}) > '
                       f'({privacy_param.epsilon}, {privacy_param.delta})')
