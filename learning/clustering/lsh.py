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
"""Hashes data by projecting against stored random vectors and taking the sign."""

import dataclasses
import typing

import numpy as np


HashChar = str
HashCharToPoints = typing.Dict[HashChar, np.ndarray]


@dataclasses.dataclass
class SimHash():
  """Implements Locality Sensitive Hashing (LSH) using the Random Projection method.

  The LSH of length n for vectors in R^d is obtained by sampling n vectors
  {v_1, ..., v_n} from the standard normal distribution over R^d; a point
  x in R^d hashes to n bits given by (sign(<x, v_1>), ... , sign(<x, v_n>)).
  Reference:
  https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection

  Attributes:
    dim: Dimension of the data that will be hashed.
    max_hash_len: Maximum length of hash requested, also the maximum number of
      hash vectors that will be needed.
    projection_vectors: Random vectors sampled for LSH; 2D np.ndarray of shape
      (max_hash_len, dim), where i'th index corresponds to the i'th projection
      vector.
  """
  dim: int
  max_hash_len: int
  projection_vectors: typing.Optional[np.ndarray] = None

  def __post_init__(self):
    if self.projection_vectors is None:
      self.projection_vectors = np.random.normal(
          size=(self.max_hash_len, self.dim))

  def group_by_next_hash(self,
                         datapoints: np.ndarray,
                         hash_prefix: str = "") -> HashCharToPoints:
    """Groups points in datapoints by the next hash character after hash_prefix.

    Args:
      datapoints: Datapoints to group, required to have the same hash_prefix.
      hash_prefix: Prefix for the hash of all the datapoints.

    Returns:
      HashCharToPoints mapping the next character in the hash value to
        the datapoints with that next character.

    Raises:
      ValueError: if hash_prefix is not strictly smaller than max_hash_len.
    """
    prefix_length = len(hash_prefix)
    if prefix_length >= self.max_hash_len:
      raise ValueError(f"Hash prefix {hash_prefix} has length greater than or "
                       f"equal to max hash length ({self.max_hash_len})")
    projected_values = np.matmul(datapoints,
                                 self.projection_vectors[prefix_length])
    return {
        "0": datapoints[projected_values >= 0],
        "1": datapoints[projected_values < 0]
    }
