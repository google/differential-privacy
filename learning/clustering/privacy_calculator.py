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

from clustering import central_privacy_utils
from clustering import clustering_params


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
