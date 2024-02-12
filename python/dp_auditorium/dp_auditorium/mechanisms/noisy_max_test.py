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
"""Tests for noisy max mechanism."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from dp_auditorium.configs import mechanism_config
from dp_auditorium.mechanisms import noisy_max

_SEED = 12345
_RNG = np.random.default_rng(seed=_SEED)


def setUpModule():
  np.random.seed(_SEED)
  tf.random.set_seed(_SEED)


class NoisyMaxTest(tf.test.TestCase, parameterized.TestCase):

  def test_noisy_max_low_noise(self):
    # Initialize a low noise mechanism on 5 elements.
    config = mechanism_config.NoisyMaxMechanismConfig(
        epsilon=1000, num_elements=5
    )
    mechanism = noisy_max.NoisyMaxMechanism(config, _RNG)

    # All records select first element.
    data = np.zeros(1000)

    # Draw samples from mechanism.
    num_samples = 1000
    samples = mechanism(data, num_samples)

    # Verify that mechanism almost always outputs correct answer.
    _, counts = np.unique(samples, return_counts=True)
    actual_fraction_first_element = counts[0] / np.sum(counts)
    self.assertAlmostEqual(1.0, actual_fraction_first_element)

  def test_noisy_max_high_noise(self):
    # Initialize a high noise mechanism on 2 elements.
    config = mechanism_config.NoisyMaxMechanismConfig(epsilon=1, num_elements=2)
    mechanism = noisy_max.NoisyMaxMechanism(config, _RNG)

    # 10 records select first element, 5 records select second element.
    data = np.concatenate((np.zeros(10), np.ones(5)))

    # Draw samples from mechanism.
    num_samples = 10000
    samples = mechanism(data, num_samples)

    # Noisy max mechanism is equivalent to the exponential mechanism.
    _, counts = np.unique(samples, return_counts=True)
    expected_fraction_first_element = np.exp(5) / (np.exp(5) + np.exp(2.5))
    expected_fraction_second_element = np.exp(2.5) / (np.exp(5) + np.exp(2.5))
    actual_fraction_first_element = counts[0] / (counts[0] + counts[1])
    actual_fraction_second_element = counts[1] / (counts[0] + counts[1])

    # Verify output distribution of mechanism.
    self.assertAlmostEqual(
        expected_fraction_first_element, actual_fraction_first_element, places=2
    )
    self.assertAlmostEqual(
        expected_fraction_second_element,
        actual_fraction_second_element,
        places=2,
    )


if __name__ == "__main__":
  absltest.main()
