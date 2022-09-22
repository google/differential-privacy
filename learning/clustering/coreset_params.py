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
"""Parameters for coreset generation."""

import dataclasses
import typing

from clustering import clustering_params
from clustering import privacy_calculator


@dataclasses.dataclass
class CoresetParam():
  """Parameters that determine the clustering algorithm.

  Attributes:
    pcalc: Privacy calculator.
    tree_param: Parameters for constructing LSH tree.
    short_description: Text description for clustering parameters.
    radius: Bound on the distance of each point from the origin in datapoints.
  """
  pcalc: privacy_calculator.PrivacyCalculator
  tree_param: clustering_params.TreeParam
  short_description: typing.Optional[str]
  radius: float
