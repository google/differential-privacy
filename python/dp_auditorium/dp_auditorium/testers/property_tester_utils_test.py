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
"""Tests for property_tester_utils module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config
from dp_auditorium.testers import property_tester_utils


class PropertyTesterUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.training_config = property_tester_config.TrainingConfig(
        training_epochs=1,
        batch_size=4,
        optimizer_learning_rate=0.01,
        model_output_coordinate_bound=1.5,
    )

  @parameterized.parameters(-0.1, 0.0)
  def test_validate_training_config_wrong_lr(self, learning_rate):
    self.training_config.optimizer_learning_rate = learning_rate
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_training_config(self.training_config)

  @parameterized.parameters(-10, 0)
  def test_validate_training_config_wrong_training_epochs(
      self, training_epochs
  ):
    self.training_config.training_epochs = training_epochs
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_training_config(self.training_config)

  @parameterized.parameters(-5, 0)
  def test_validate_training_config_wrong_batch_size(self, batch_size):
    self.training_config.batch_size = batch_size
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_training_config(self.training_config)

  @parameterized.parameters(-0.1, 0.0)
  def test_validate_training_config_wrong_bound(self, bound):
    self.training_config.model_output_coordinate_bound = bound
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_training_config(self.training_config)

  def test_split_train_test_samples_raises_exception(self):
    with self.assertRaises(ValueError):
      property_tester_utils.split_train_test_samples(samples=np.ones(1))

  @parameterized.parameters((np.ones(100),), (np.ones(101),))
  def test_split_train_test_samples(self, samples):
    expected_train = samples[:50]
    expected_test = samples[50:]
    result = property_tester_utils.split_train_test_samples(samples=samples)
    self.assertAllClose(result[0], expected_train)
    self.assertAllClose(result[1], expected_test)

  @parameterized.parameters(-0.1, 0.0, 1.0, None)
  def test_validate_renyi_dp_property_wrong_alpha(self, alpha):
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_renyi_dp_property(
          privacy_property.RenyiDp(epsilon=0.1, alpha=alpha)
      )

  @parameterized.parameters([-0.1, 0.0, None])
  def test_validate_renyi_dp_property_wrong_epsilon(self, epsilon):
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_renyi_dp_property(
          privacy_property.RenyiDp(epsilon=epsilon, alpha=2.0)
      )

  @parameterized.parameters([-0.1, None])
  def test_validate_approximate_dp_property_wrong_delta(self, delta):
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_approximate_dp_property(
          privacy_property.ApproximateDp(epsilon=0.1, delta=delta)
      )

  @parameterized.parameters([-0.1, 0.0, None])
  def test_validate_approximate_dp_property_wrong_epsilon(self, epsilon):
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_approximate_dp_property(
          privacy_property.ApproximateDp(epsilon=epsilon, delta=0.1)
      )

  @parameterized.parameters([-0.1, 0.0, None])
  def test_validate_pure_dp_property_wrong_epsilon(self, epsilon):
    with self.assertRaises(ValueError):
      _ = property_tester_utils.validate_pure_dp_property(
          privacy_property.PureDp(epsilon=epsilon)
      )


if __name__ == '__main__':
  absltest.main()
