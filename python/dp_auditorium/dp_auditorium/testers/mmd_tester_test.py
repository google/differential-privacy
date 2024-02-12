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
"""Tests for maximum mean discrepancy (MMD) approximate DP tester."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config
from dp_auditorium.testers import mmd_tester

_SEED = 12345
_RNG = np.random.default_rng(seed=_SEED)


def different_samples(n):
  """Returns n samples each from two different distributions."""
  return _RNG.normal(0, 1, n), _RNG.normal(5, 1, n)


def identical_samples(n):
  """Returns n samples each from two identical distributions."""
  return _RNG.normal(0, 1, n), _RNG.normal(0, 1, n)


def mmd_config(kernel):
  """Returns an MMD tester config with the specified kernel."""
  return property_tester_config.MMDPropertyTesterConfig(
      approximate_dp=privacy_property.ApproximateDp(epsilon=0.1, delta=0.01),
      bandwidth=0.1,
      kernel=kernel,
  )


class MMDTesterTest(tf.test.TestCase, parameterized.TestCase):

  def test_unspecified_kernel(self):
    with self.assertRaisesRegex(ValueError, 'Invalid kernel function'):
      config = mmd_config(property_tester_config.Kernel.KERNEL_UNSPECIFIED)
      mmd_tester.MMDPropertyTester(config)

  def test_privacy_property(self):
    config = mmd_config(property_tester_config.Kernel.KERNEL_RBF)
    tester = mmd_tester.MMDPropertyTester(config)
    self.assertEqual(
        config.approximate_dp, tester.privacy_property.approximate_dp
    )

  @parameterized.parameters(
      property_tester_config.Kernel.KERNEL_RBF,
      property_tester_config.Kernel.KERNEL_LAPLACIAN,
  )
  def test_estimate_lower_bound_identical(self, kernel):
    """Test estimate_lower_bound() on identical distributions."""
    config = mmd_config(kernel)
    tester = mmd_tester.MMDPropertyTester(config)
    samples1, samples2 = identical_samples(n=1000)
    lower_bound = tester.estimate_lower_bound(
        samples1, samples2, failure_probability=0.1
    )
    self.assertLess(lower_bound, 0)

  @parameterized.parameters(
      property_tester_config.Kernel.KERNEL_RBF,
      property_tester_config.Kernel.KERNEL_LAPLACIAN,
  )
  def test_estimate_lower_bound_different(self, kernel):
    """Test estimate_lower_bound() on different distributions."""
    config = mmd_config(kernel)
    tester = mmd_tester.MMDPropertyTester(config)
    samples1, samples2 = different_samples(n=1000)
    lower_bound = tester.estimate_lower_bound(
        samples1, samples2, failure_probability=0.1
    )
    self.assertGreater(lower_bound, config.approximate_dp.delta)

  def test_reject_property(self):
    """Test reject_property()."""
    config = mmd_config(property_tester_config.Kernel.KERNEL_RBF)
    tester = mmd_tester.MMDPropertyTester(config)
    self.assertEqual(config.approximate_dp.delta, 0.01)
    self.assertTrue(tester.reject_property(0.1))


if __name__ == '__main__':
  absltest.main()
