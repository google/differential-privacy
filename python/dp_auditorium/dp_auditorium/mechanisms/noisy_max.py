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
"""Noisy max mechanism that can be tested using DP-Auditorium."""

import collections

import numpy as np

from dp_auditorium.configs import mechanism_config


class NoisyMaxMechanism:
  """Implementation of noisy max mechanism.

  Assumes records in dataset indicate preference for one element in
  {0, ..., num_elements-1}. The mechanism returns the element with maximum noisy
  count using Gumbel noise. This is equivalent to an exponential mechanism
  where the utility function are the elements' counts. This implementation
  satisfies epsilon - differential privacy.

  Attributes:
    epsilon: Privacy budget parameter.
    num_elements: Number of elements records contribute to.
    rng: Random number generator.
  """

  def __init__(
      self,
      config: mechanism_config.NoisyMaxMechanismConfig,
      rng: np.random.BitGenerator,
  ):
    self.epsilon = config.epsilon
    self.num_elements = config.num_elements
    self.rng = rng

  def __call__(self, data: np.ndarray, num_samples: int) -> np.ndarray:
    counts = collections.Counter(data)
    counts_array = [counts[i] for i in range(self.num_elements)]

    noise = self.rng.gumbel(
        loc=0, scale=2 / self.epsilon, size=(num_samples, self.num_elements)
    )
    noisy_counts = counts_array + noise
    return np.argmax(noisy_counts, axis=1)
