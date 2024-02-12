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
"""Example mean mechanisms that can be tested using DP-Auditorium."""

from typing import Tuple, Union

import numpy as np

from dp_auditorium.configs import mechanism_config

_ZERO_TOLERANCE = 1e-12


class MeanMechanism:
  """An implementation of the Laplace and Gaussian mean mechanisms.

  Implements a mechanism to get a "private" version of the mean of a data set.
  The configuration passed however can potentially generate non-private versions
  of this mechanism.
  """

  def __init__(
      self,
      config: mechanism_config.MeanMechanismConfig,
      rng: np.random.BitGenerator,
  ):
    self._use_noised_counts_for_calculating_mean = (
        config.use_noised_counts_for_calculating_mean
    )
    self._use_noised_counts_for_calculating_noise_scale = (
        config.use_noised_counts_for_calculating_noise_scale
    )
    self._epsilon = config.epsilon
    self._delta = config.delta
    self._epsilon_budget_scale = 1.0
    self._rng = rng
    self._max_value = config.max_value
    self._min_value = config.min_value
    if (
        self._use_noised_counts_for_calculating_mean
        or self._use_noised_counts_for_calculating_noise_scale
    ):
      self._epsilon_budget_scale = 0.5
    if self._delta == 0.0:
      # Laplace mechanism.
      self._noise_factor = 1.0
      self._noise_function = self._rng.laplace
    else:
      # Gaussian mechanism.
      self._noise_factor = np.sqrt(2.0 * np.log(1.25 / self._delta))
      self._noise_function = self._rng.normal

  def _get_counts(
      self, data: np.ndarray, num_samples: int
  ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """Returns counts to be used to calculate the mean and noise scale.

    Args:
      data: Dataset
      num_samples: num samples to be output by the mechanism.

    Returns:
      A tuple with the first element corresponding to the counts to be used for
      mean calculation and the second to be used for scaling noise.
      If no noise is added, an entry in the tuple is an int, if noise is added
      the entry in the tuple is an array of length num_samples.
    """
    noise_scale = 1.0 / self._epsilon_budget_scale / self._epsilon
    noisy_counts = len(data) + self._rng.laplace(
        0, noise_scale, (num_samples, 1)
    )
    if self._use_noised_counts_for_calculating_mean:
      if self._use_noised_counts_for_calculating_noise_scale:
        return noisy_counts, noisy_counts
      else:
        return noisy_counts, len(data)
    else:
      if self._use_noised_counts_for_calculating_noise_scale:
        return len(data), noisy_counts
      else:
        return len(data), len(data)

  def __call__(self, data: np.ndarray, num_samples: int) -> np.ndarray:
    data = np.clip(data, self._min_value, self._max_value)
    counts_for_mean, counts_for_noise_scale = self._get_counts(
        data, num_samples
    )
    value_sum = np.sum(data)
    noise_scale = (
        self._noise_factor
        * np.abs(self._max_value - self._min_value)
        / self._epsilon_budget_scale
        / self._epsilon
        / counts_for_noise_scale
    )
    noise_scale = np.maximum(_ZERO_TOLERANCE, noise_scale)
    return value_sum / counts_for_mean + self._noise_function(
        0.0, noise_scale, (num_samples, 1)
    )
