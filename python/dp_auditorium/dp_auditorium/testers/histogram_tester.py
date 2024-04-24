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
"""Approximate DP tester from Gilbert and McMillan (2018).

The tester in this file corresponds to algorithm 2 in
https://arxiv.org/pdf/1806.06427.pdf. Parameter names follow naming in the
original algorithm.
"""

import collections
import functools

from etils.array_types import IntArray
import numpy as np
from typing_extensions import override

from dp_auditorium import interfaces
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config
from dp_auditorium.testers import property_tester_utils


def _estimate_discrete_distribution(
    samples: IntArray,
    universe_size: int,
) -> np.ndarray:
  """Returns estimated probability mass function over universe using samples.

  This function estimates a probability mass function over a finite universe
  using samples from an unknown underlying distribution. It assumes the sample
  values are in the range [0, universe_size-1].

  Args:
    samples: array of samples from the underlying distribution.
    universe_size: size of the universe.

  Raises:
    ValueError: If the `samples` array is empty or if `samples` contains
    elements outside [0,`universe_size`-1].
  """
  if samples.size == 0:
    raise ValueError("Cannot estimate probability mass from empty array.")
  # Discrete mechanisms return values in {0,...,universe_size-1}. Since
  # we enforce type int for the `samples` array checking for the maximum and the
  # minimum enforces that the histogram is created in the right domain.
  if np.amax(samples) >= universe_size or np.amin(samples) < 0:
    raise ValueError(
        "The samples from the mechanism have a different range than"
        f" {{0,..., {universe_size-1}}}."
    )
  counts = collections.Counter(samples)
  return np.array([counts[i] for i in range(universe_size)]) / samples.shape[0]


def _estimate_continuous_distribution(
    samples: np.ndarray, universe_size: int, min_value: float, max_value: float
) -> np.ndarray:
  """Estimates a discretized probability mass function from samples.

  Assuming a continuous random variable taking values over an interval
  `[min_value, max_value]`, this function estimates the continuous density
  function by discretizing the interval `[min_value, max_value]` and computing
  a normalized histogram from samples of the underlying distribution.

  Args:
    samples: array of samples from the underlying distribution distribution.
    universe_size: size of the discretized output space.
    min_value: minimum value of the random value.
    max_value: maximum value of the random value.

  Returns:
    Array where each entry determines the estimated probability of the
    variable taking values on the corresponding bin.

  Raises:
    ValueError: If the `samples` array is empty.
  """
  clipped_samples = np.clip(samples, min_value, max_value)
  if samples.size == 0:
    raise ValueError("Cannot estimate probability mass from empty array.")
  counts, _ = np.histogram(
      clipped_samples,
      bins=universe_size,
      range=(min_value, max_value),
  )
  return counts / samples.shape[0]


class HistogramTester(interfaces.PropertyTester):
  """Privacy tester introduced in https://arxiv.org/pdf/1806.06427.pdf.

  Implements the approximate differential privacy (DP) tester corresponding to
  Algorithm 2 in https://arxiv.org/pdf/1806.06427.pdf. The algorithm is given
  samples from an (epsilon, delta)-approximate DP mechanism on neighboring
  datasets and uses them to estimate the parameter `delta`. The algorithm is
  only defined for finite output mechanisms, so the implementation below places
  continuous-valued samples into discrete buckets. The error bound for the
  estimate of `delta` is a slightly improved version of the bound from the
  original paper, as it does not assume that the number of samples is Poisson
  distributed, and it holds for any given probability, rather than only with
  probability 2/3.
  """

  def __init__(
      self, config: property_tester_config.HistogramPropertyTesterConfig
  ):
    """Initializes an instance of a Histogram property tester.

    Args:
      config: Configuration for Histogram tester.
    """
    property_tester_utils.validate_approximate_dp_property(
        config.approximate_dp
    )

    # Set function to estimate a probability mass function.
    if config.test_discrete_mechanism:
      self._estimate_distribution = functools.partial(
          _estimate_discrete_distribution,
          universe_size=config.histogram_size,
      )
    else:
      self._estimate_distribution = functools.partial(
          _estimate_continuous_distribution,
          universe_size=config.histogram_size,
          min_value=config.min_value,
          max_value=config.max_value,
      )
    self._epsilon = config.approximate_dp.epsilon
    self._delta = config.approximate_dp.delta
    self._use_original_tester = config.use_original_tester
    self._histogram_size = config.histogram_size
    self._approximate_dp = config.approximate_dp

  @property
  def privacy_property(self) -> privacy_property.PrivacyProperty:
    """The privacy guarantee that the tester is being used to test for."""
    return privacy_property.PrivacyProperty(approximate_dp=self._approximate_dp)

  def _get_error_tolerance(
      self,
      num_samples: float,
      probabilities1: np.ndarray,
      probabilities2: np.ndarray,
      failure_probability: float
  ) -> float:
    """Gets error tolerance for Histogram property tester."""
    if self._use_original_tester:
      term_1 = (
          2.0
          * (1.0 + np.exp(self._epsilon))
          * np.sqrt(self._histogram_size / num_samples)
      )
    else:
      term_1a = (
          2.0 / np.sqrt(num_samples)
          * sum(np.sqrt(probabilities1))
      )
      term_1b = (
          2.0 * np.exp(self._epsilon) / np.sqrt(num_samples)
          * sum(np.sqrt(probabilities2))
      )
      term_1 = term_1a + term_1b
    term_2 = (
        6.0
        * (1.0 + np.exp(self._epsilon))
        * np.sqrt(np.log(4.0 / failure_probability) / (2.0 * num_samples))
    )
    return term_1 + term_2

  @override
  def estimate_lower_bound(
      self,
      samples1: np.ndarray,
      samples2: np.ndarray,
      failure_probability: float,
  ) -> float:
    """Estimates delta in approximate DP guarantee."""
    num_samples = min(samples1.shape[0], samples2.shape[0])
    probabilities1 = self._estimate_distribution(samples1)
    probabilities2 = self._estimate_distribution(samples2)
    per_outcome_delta = probabilities1 - np.exp(self._epsilon) * probabilities2
    estimated_delta = np.sum(per_outcome_delta[per_outcome_delta > 0])
    error_tolerance = self._get_error_tolerance(
        num_samples, probabilities1, probabilities2, failure_probability
    )
    return estimated_delta - error_tolerance

  @override
  def reject_property(self, lower_bound: float) -> bool:
    return lower_bound >= self._delta
