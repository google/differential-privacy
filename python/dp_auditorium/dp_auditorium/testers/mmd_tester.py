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
"""Approximate DP tester based on maximum mean discrepancy (MMD).

Implementation of the DP tester from Section 17 of
https://braintex.goog/project/63ecefd65ee2590082194a90.
"""

import numpy as np

from dp_auditorium import interfaces
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config
from dp_auditorium.testers import property_tester_utils


class MMDPropertyTester(interfaces.PropertyTester):
  """Approximate DP tester based on maximum mean discrepancy (MMD).

  Implements an approximate differential privacy (DP) tester. Given samples from
  a pair of distributions P and Q, which are induced by a mechanism on two
  neighboring datasets, the tester estimates the MMD, which is defined

  max_f E_{x ~ P}[f(x)] - E_{x ~ Q}[f(x)]

  where the maximization is over a reproducing kernel Hilbert space. If the
  mechanism is (epsilon, delta)-approximate DP then the square root of the MMD
  cannot exceed (exp(epsilon) - 1) + (1 + exp(epsilon)) * delta.
  """

  def __init__(
      self, config: property_tester_config.MMDPropertyTesterConfig
  ) -> None:
    """Initialize MMD tester.

    Args:
      config: Configuration for MMD tester.

    Raises:
      ValueError: Invalid kernel function.
    """
    property_tester_utils.validate_approximate_dp_property(
        approximate_dp=config.approximate_dp
    )
    self._approximate_dp = config.approximate_dp
    self._bandwidth = config.bandwidth
    if config.kernel == property_tester_config.Kernel.KERNEL_RBF:
      self._kernel = lambda x, y: (
          np.exp(-self._bandwidth * np.linalg.norm(x - y) ** 2)
      )
    elif config.kernel == property_tester_config.Kernel.KERNEL_LAPLACIAN:
      self._kernel = lambda x, y: (
          np.exp(-self._bandwidth * np.linalg.norm(np.atleast_1d(x - y), 1))
      )
    else:
      raise ValueError(f'Invalid kernel function: {config.kernel.name}')

  @property
  def privacy_property(self) -> privacy_property.PrivacyProperty:
    """The privacy guarantee that the tester is being used to test for."""
    return privacy_property.PrivacyProperty(approximate_dp=self._approximate_dp)

  def estimate_lower_bound(
      self,
      samples1: np.ndarray,
      samples2: np.ndarray,
      failure_probability: float,
  ) -> float:
    """Estimates a lower bound on delta.

    Args:
      samples1: Array of samples from the first distribution.
      samples2: Array of samples from the second distribution.
      failure_probability: Probability that the returned lower bound does not
        hold.

    Returns:
      Estimated lower bound on delta.
    """
    epsilon = self._approximate_dp.epsilon

    # Split each set of samples into two halves.
    samples1a, samples1b = property_tester_utils.split_train_test_samples(
        samples1
    )
    samples2a, samples2b = property_tester_utils.split_train_test_samples(
        samples2
    )

    # Effective number of samples is the size of the smallest set of samples.
    num_samples = min(
        samples1a.shape[0],
        samples1b.shape[0],
        samples2a.shape[0],
        samples2b.shape[0],
    )

    # Define pairwise kernel function.
    h = lambda xa, xb, ya, yb: (
        self._kernel(xa, xb) - 2 * self._kernel(xa, ya) + self._kernel(ya, yb)
    )

    # Calculate sample mean and sample variance of pairwise kernel function.
    total = 0
    total_squares = 0
    for xa, xb, ya, yb in zip(samples1a, samples1b, samples2a, samples2b):
      total += h(xa, xb, ya, yb)
      total_squares += h(xa, xb, ya, yb) ** 2
    sample_mean = total / num_samples
    sample_variance = max(
        0.0,  # Truncate at 0 in case of underflow.
        total_squares / num_samples - (sample_mean) ** 2,
    )

    # Compute two error tolerances. First error tolerance is worst-case,
    # second error tolerance is data-dependent.
    error_tol = np.zeros(2)
    error_tol[0] = np.sqrt(
        8.0 * np.log(1.0 / failure_probability) / num_samples
    )
    error_tol[1] = np.sqrt(
        2.0 * sample_variance * np.log(2.0 / failure_probability) / num_samples
    ) + (28.0 * np.log(2.0 / failure_probability) / (3.0 * (num_samples - 1)))

    # Define some convenience variables.
    exp_eps_minus_one = np.exp(epsilon) - 1
    exp_eps_minus_expinv_eps = np.exp(epsilon) - np.exp(-epsilon)
    one_plus_expinv_eps = 1.0 + np.exp(-epsilon)

    # Calculate two lower bounds on delta, based on the two error tolerances.
    delta_lower_bound = np.zeros(2)
    for i in (0, 1):
      delta_lower_bound[i] = (
          np.sqrt(
              max(
                  0.0,  # Truncate at 0 in case of underflow.
                  exp_eps_minus_expinv_eps**2
                  + one_plus_expinv_eps
                  * (sample_mean - error_tol[i] - exp_eps_minus_one**2),
              )
          )
          - exp_eps_minus_expinv_eps
      ) / one_plus_expinv_eps

    # Return maximum of the two lower bounds.
    return max(delta_lower_bound[0], delta_lower_bound[1])

  def reject_property(self, lower_bound: float) -> bool:
    """Returns false if and only if lower_bound exceeds delta."""
    return lower_bound > self._approximate_dp.delta
