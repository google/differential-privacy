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
"""Tests for DP-Auditorium interfaces."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from dp_auditorium.configs import privacy_property
from dp_auditorium.testers import divergence_tester


@dataclasses.dataclass
class StubDivergencePropertyTesterConfig:
  """Configuration for stub divergence property tester."""

  estimated_divergence: float
  test_threshold: float


class StubDivergencePropertyTester(divergence_tester.DivergencePropertyTester):

  def __init__(
      self,
      config: StubDivergencePropertyTesterConfig,
      base_model: tf.keras.Model,
  ):
    self._config_test_threshold = config.test_threshold
    self._estimated_divergence = config.estimated_divergence
    self._base_model = base_model

  @property
  def _test_threshold(self) -> float:
    return self._config_test_threshold

  @property
  def privacy_property(self) -> privacy_property.PrivacyProperty:
    return privacy_property.PureDp(epsilon=0.1)

  def _get_optimized_divergence_estimation_model(
      self,
      samples_first_distribution: np.ndarray,
      samples_second_distribution: np.ndarray,
  ) -> tf.keras.Model:
    return self._base_model

  def _compute_divergence_on_samples(
      self,
      model: tf.keras.Model,
      samples_first_distribution: np.ndarray,
      samples_second_distribution: np.ndarray,
      failure_probability: float,
  ) -> float:
    del (
        model,
        samples_first_distribution,
        samples_second_distribution,
        failure_probability,
    )
    return self._estimated_divergence


class DivergencePropertyTesterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.base_model = tf.keras.Sequential([
        tf.keras.Input(shape=(3,)),
        tf.keras.layers.Dense(1, activation="relu"),
    ])

  @parameterized.parameters(0.3, 1.1, 2.5)
  def test_divergence_property_estimates_lower_bound_returns_expected_divergence(
      self, divergence
  ):
    samples1 = np.ones((100, 1))
    samples2 = np.ones((100, 1))

    config = StubDivergencePropertyTesterConfig(
        estimated_divergence=divergence, test_threshold=0.1
    )
    divergence_property_tester = StubDivergencePropertyTester(
        config, self.base_model
    )

    estimated_divergence = divergence_property_tester.estimate_lower_bound(
        samples1, samples2, failure_probability=0.1
    )
    self.assertAlmostEqual(estimated_divergence, divergence)

  @parameterized.product(lower_bound=[0.1, 0.5], threshold=[0.0, 0.5, 1.0])
  def test_divergence_property_tester_rejects_property(
      self, lower_bound, threshold
  ):
    # In this test the estimated divergence parameter will be unused and we can
    # set a dummy value.
    config = StubDivergencePropertyTesterConfig(
        estimated_divergence=0.314, test_threshold=threshold
    )
    divergence_property_tester = StubDivergencePropertyTester(
        config, self.base_model
    )
    result = divergence_property_tester.reject_property(lower_bound)
    self.assertEqual(lower_bound > threshold, result)


if __name__ == "__main__":
  absltest.main()
