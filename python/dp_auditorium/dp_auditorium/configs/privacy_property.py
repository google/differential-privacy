# Copyright 2024 Google LLC.
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
"""Dataclass for privacy properties."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class PureDp:
  """Parameter for epsilon-DP."""

  epsilon: float


@dataclasses.dataclass
class RenyiDp:
  """Parameters for alpha, epsilon)-Renyi DP.

  Attributes:
    alpha: Renyi-DP parameter.
    epsilon: Renyi-DP parameter.
  """

  alpha: float
  epsilon: float


@dataclasses.dataclass
class ApproximateDp:
  """Parameters for (epsilon, delta)-DP."""

  epsilon: float
  delta: float


@dataclasses.dataclass
class PrivacyProperty:
  """Generic container containing a single property.

  Attributes:
    pure_dp: epsilon-DP parameter.
    renyi_dp: Renyi-DP parameters.
    approximate_dp: (epsilon, delta)-DP parameter.
  """

  pure_dp: Optional[PureDp] = None
  renyi_dp: Optional[RenyiDp] = None
  approximate_dp: Optional[ApproximateDp] = None

  def __post_init__(self):
    num_set_fields = 0
    if self.pure_dp is not None:
      num_set_fields += 1
    if self.renyi_dp is not None:
      num_set_fields += 1
    if self.approximate_dp is not None:
      num_set_fields += 1
    if num_set_fields != 1:
      raise ValueError(
          f'PrivacyProperty must have exactly one of pure_dp, renyi_dp,'
          f' approximate_dp'
      )
