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
"""Tests for classification_dataset_generator."""
from absl.testing import absltest
import tensorflow as tf
from vizier.service import clients

from dp_auditorium.configs import dataset_generator_config
from dp_auditorium.generators import classification_dataset_generator


clients.environment_variables.servicer_use_sql_ram()


class ClassificationDatasetGeneratorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.sample_dim = 5
    self.num_samples = 4
    self.min_value = 13.7
    self.max_value = 15.9
    self.num_classes = 3

    self.generator_config = (
        dataset_generator_config.ClassificationDatasetGeneratorConfig(
            sample_dim=self.sample_dim,
            num_samples=self.num_samples,
            num_classes=self.num_classes,
            min_value=self.min_value,
            max_value=self.max_value,
            study_name='stub_study',
            study_owner='stub_owner',
            metric_name='stub_metric',
            search_algorithm='RANDOM_SEARCH',
        )
    )
    self.generator = (
        classification_dataset_generator.ClassificationDatasetGenerator(
            config=self.generator_config,
        )
    )

  def test_get_neighboring_datasets_from_vizier_params_produces_correct_pair(
      self,
  ):
    """Tests datasets have correct shapes and are adjacent."""

    data1, data2 = self.generator(None)
    # Check output shape
    with self.subTest('data1-images-have-correct-shape'):
      self.assertEqual(data1[0].shape, (self.num_samples, self.sample_dim))
    with self.subTest('data1-labels-have-correct-shape'):
      self.assertEqual(data1[1].shape, (self.num_samples,))
    with self.subTest('data2-images-have-correct-shape'):
      self.assertEqual(data2[0].shape, (self.num_samples - 1, self.sample_dim))
    with self.subTest('data2-labels-have-correct-shape'):
      self.assertEqual(data2[1].shape, (self.num_samples - 1,))

    # Check output values range.
    with self.subTest('data1-labels-in-range'):
      self.assertAllInRange(data1[1], 0, self.num_classes)
    with self.subTest('data2-labels-in-range'):
      self.assertAllInRange(data2[1], 0, self.num_classes)
    with self.subTest('data1-features-in-range'):
      self.assertAllInRange(data1[0], self.min_value, self.max_value)
    with self.subTest('data2-features-in-range'):
      self.assertAllInRange(data2[0], self.min_value, self.max_value)


if __name__ == '__main__':
  absltest.main()
