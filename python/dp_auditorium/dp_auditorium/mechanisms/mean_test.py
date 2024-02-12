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
"""Tests for mean mechanisms."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from dp_auditorium.mechanisms import mean
from dp_auditorium.mechanisms import mechanisms_utils

_SEED = 12345
_RNG = np.random.default_rng(seed=_SEED)


def setUpModule():
  np.random.seed(_SEED)
  tf.random.set_seed(_SEED)


class MeanTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(epsilon=(5.0, 10.0), delta=(0, 0.1))
  def test_mean_mechanisms_private(self, epsilon, delta):
    # Average of mechanism output over many trials should be close to data mean.
    data = np.linspace(0, 1)
    config = mechanisms_utils.default_mean_mechanism_config_generator(
        "private_mean",
        epsilon=epsilon,
        delta=delta)
    mechanism = mean.MeanMechanism(config, _RNG)
    samples = mechanism(data, 10_000)
    self.assertAlmostEqual(np.mean(samples), np.mean(data), places=3)

  @parameterized.parameters((0.1, 0.0), (0.001, 0.01))
  def test_mean_mechanisms_nonprivate(self, epsilon, delta):
    data = np.linspace(0, 1)
    # Draw samples from private and non-private mechanisms.
    samples = {}
    for mechanism_name in ("private_mean",
                           "non_private_mean_v1",
                           "non_private_mean_v2"):
      config = mechanisms_utils.default_mean_mechanism_config_generator(
          mechanism_name,
          epsilon=epsilon,
          delta=delta)
      mechanism = mean.MeanMechanism(config, _RNG)
      samples[mechanism_name] = mechanism(data, 1000)
    # Private samples should be much less concentrated than non-private samples.
    private_stddev = np.std(samples["private_mean"])
    non_private_v1_stddev = np.std(samples["non_private_mean_v1"])
    non_private_v2_stddev = np.std(samples["non_private_mean_v2"])
    self.assertGreater(private_stddev, 2 * non_private_v1_stddev)
    self.assertGreater(private_stddev, 2 * non_private_v2_stddev)


if __name__ == "__main__":
  absltest.main()
