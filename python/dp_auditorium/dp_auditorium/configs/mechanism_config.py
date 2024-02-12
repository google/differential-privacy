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
"""Dataclasses for Mechanism configus."""

import dataclasses
import enum
from typing import Optional


# NOTE: These configurations are intendend to initialize both private and
# non-private mechanisms and are here for the purposes of demonstrating
# the ability of the testers to detect issues in the implementations of
# a mechanism. In no way should these configurations be used for deploying
# these mechanisms into a production system.


@dataclasses.dataclass
class MeanMechanismConfig:
  """Configuration to initialize the MeanMechanism.

  Attributes: epsilon, delta: Privacy parameters.
    use_noised_counts_for_calculating_mean: Should be true for private
    mechanism.
    use_noised_counts_for_calculating_noise_scale: Should be true for private
    mechanism.
    max_value: Data values will be clipped to this maximum value.
    min_value: Data values will be clipped to this minimum value.
  """

  epsilon: float
  delta: float
  use_noised_counts_for_calculating_mean: bool
  use_noised_counts_for_calculating_noise_scale: bool
  max_value: float
  min_value: float


@dataclasses.dataclass
class NoisyMaxMechanismConfig:
  """Configuration to initialize the NoisyMaxMechanism.

  Attributes:
    epsilon: Privacy parameter.
    num_elements: Number of elements in the domain.
  """

  epsilon: float
  num_elements: int


@dataclasses.dataclass
class GradientDescentMechanismConfig:
  """Configuration to initialize the GradientDescentMechanism.

  Attributes:
    alpha: Privacy parameter determining the order of the Renyi divergence.
    epsilon: Privacy paramter for Renyi or approximate DP. When testing for
      approximate DP, delta will be determined using a range of alphas. For
      details see `get_delta_and_optimal_order` in the `dp_accounting` library.
    noise_scale_reduce_factor: Non negative scalar determining the fraction of
      noise added with respect to the correct noise scale to achieve the (alpha,
      epsilon) - Renyi DP or (epsilon, delta) - approximate DP guarantee. The
      guarantee is satisfied for values `noise_reduce_factor >= 1.0`.
    l2_norm_clip: Value to clip the l2 norm of gradients.
  """

  alpha: float
  epsilon: float
  noise_scale_reduce_factor: float
  l2_norm_clip: float


class QueryType(enum.Enum):
  """Query types for sparse vector technique mechanisms."""

  QUERY_TYPE_UNSPECIFIED = 0
  QUERY_TYPE_SUM = 1
  QUERY_TYPE_REVEAL_RECORDS = 2


class DataValues(enum.Enum):
  """Data values returned by sparse vector technique mechanisms.

  To maximize the likelihood that a property tester will detect a privacy
  violation, these values should not collide with values returned by any query.
  """

  SVT_NO_RESPONSE = 0
  SVT_ABOVE_THRESHOLD = 1
  SVT_BELOW_THRESHOLD = 2


@dataclasses.dataclass
class SVTMechanismConfig:
  """Configuration to initialize the sparse vector technique mechanisms.

  This configuration can be used for any of the mechanisms (SVT1Mechanism
  through SVT6Mechanism).

  Attributes:
    epsilon: Privacy parameter.
    max_value: Data values will be clipped to this maximum value.
    min_value: Data values will be clipped to this minimum value.
    query_type: Type of each query that will be computed on the data by the
      mechanism.
    num_queries: Number of queries that will be computed on the data by the
      mechanism.
    threshold: Threshold that each query value is compared to by the mechanism.
    max_answered_queries: Maximum number of queries where the mechanism responds
      that the query value is above the threshold.
  """

  epsilon: float
  max_value: float
  min_value: float
  query_type: QueryType
  num_queries: int
  threshold: float
  max_answered_queries: int
