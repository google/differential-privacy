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
"""Testers type aliases used throughout the dp-auditorium library."""

from typing import Any, Optional, Protocol, TypeVar

import numpy as np

from dp_auditorium.configs import privacy_property

_T = TypeVar('_T')

NeighboringDatasetsType = tuple[_T, _T]


class Mechanism(Protocol):

  def __call__(self, dataset: Any, num_samples: int) -> np.ndarray:
    """Computes `num_samples` times a randomized query on `dataset`.

    Main interface for mechanisms to be tested. The implementation should
    compute a batch of `num_samples` independent samples of the mechanism on the
    dataset.

    Args:
      dataset: Object containing records to be used by the mechanism to compute
        a query.
      num_samples: Number of independent samples from the mechanism computed on
        the dataset.

    Returns:
      Array of `num_samples` from the mechanism. The first dimension should
      correspond to the `num_samples` batch dimension.
    """


class DatasetGenerator(Protocol):

  def __call__(self, divergence: Optional[float]) -> NeighboringDatasetsType:
    """Generates a pair of neighboring datasets.

    Function that generates neighboring datasets to test for privacy violations.
    It should return two objects of the same type that are compatible with the
    input of the tested mechanism. The two datasets should be neighbors
    according to the desired tested property. For example if testing for pure
    differential privacy

    Args:
      divergence: Optional float informing the dataset generator to generate a
        new pair of datasets. For example, it can provide feedback from
        previously generated dataset pairs.

    Returns:
      A pair of neighboring datasets of the same type.
    """


class PropertyTester(Protocol):
  """Tester that reject if a the divergence between distributions is bounded.

  PropertyTesters check if there is evidence to reject that a given divergence
  between two distributions P and Q is bounded. This class implements two
  methods: (1) `estimate_lower_bound` that receives samples from P and Q and
  computes a lower bound for the divergence with a certain probability;
  (2) `reject_property` receives a lower bound on the divergence and
  outputs a boolean indicating if the lower bound value exceeds the expected
  divergence. If `reject_property_holds` returns `True`, then the property is
  rejected with a certain probability determined by specific instances of the
  PropertyTester. No formal guarantees are provided if the result is `False`.
  """

  @property
  def privacy_property(self) -> privacy_property.PrivacyProperty:
    """The privacy guarantee that the tester is being used to test for."""

  def estimate_lower_bound(
      self,
      samples1: np.ndarray,
      samples2: np.ndarray,
      failure_probability: float,
  ) -> float:
    """Estimates a divergence lower bound.

    Bound for the divergence between P and Q.

    Args:
      samples1: Array of samples from the first distribution.
      samples2: Array of samples from the second distribution.
      failure_probability: Probability that the returned lower bound does not
        hold.

    Returns:
      A float for the estimated lower bound on the divergence value.
    """

  def reject_property(self, lower_bound: float) -> bool:
    """Tests if `lower_bound` exceeds the expected divergence value.

    Args:
      lower_bound: An estimator for a lower bound on a divergence value.

    Returns:
      True if the lower bounds exceeds the expected divergence in which case
      the property does not hold with specific probability determined by
      specific instances. Returns false otherwise, in which case, no formal
      guarantees are provided about the property between the two distributions.
    """
