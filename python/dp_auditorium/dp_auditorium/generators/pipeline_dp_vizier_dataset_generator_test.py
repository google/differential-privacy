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
"""Tests for pipeline_dp_dataset_generator."""
import time

from absl.testing import absltest
import tensorflow as tf
from vizier.service import clients

from dp_auditorium.configs import dataset_generator_config
from dp_auditorium.generators import pipeline_dp_vizier_dataset_generator


clients.environment_variables.servicer_use_sql_ram()


class PipelineDpDatasetGeneratorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.num_params = 5
    self.min_value = 0.1
    self.max_value = 1.5
    self.max_num_partitions = 3
    self.max_num_privacy_ids = 2
    self.vizier_config = dataset_generator_config.VizierDatasetGeneratorConfig(
        study_name='stub_study' + str(time.time()),
        study_owner='owner',
        num_vizier_parameters=self.num_params,
        data_type=dataset_generator_config.DataType.DATA_TYPE_FLOAT,
        min_value=self.min_value,
        max_value=self.max_value,
        search_algorithm='GRID_SEARCH',
        metric_name='divergence_estimator',
    )
    self.pipeline_dp_config = (
        pipeline_dp_vizier_dataset_generator.PipelineDpDatasetGeneratorConfig(
            max_num_partitions=self.max_num_partitions,
            max_num_privacy_ids=self.max_num_privacy_ids,
        )
    )
    self.generator = (
        pipeline_dp_vizier_dataset_generator.PipelineDpDatasetGenerator(
            config=self.vizier_config,
            pipeline_dp_generator_config=self.pipeline_dp_config,
        )
    )

  def test_get_neighboring_datasets_from_vizier_params(self):
    """Tests datasets have correct shapes and are adjacent."""
    num_entries_data1 = (self.max_num_privacy_ids - 1) * self.max_num_partitions
    num_entries_data2 = self.max_num_privacy_ids * self.max_num_partitions

    data1, data2 = self.generator(None)
    with self.subTest('data1-has-correct-shape'):
      self.assertLen(data1, num_entries_data1)

    with self.subTest('data2-has-correct-shape'):
      self.assertLen(data2, num_entries_data2)

    # data1 should contain ids 0 through `self.max_num_privacy_ids - 2`
    # and data2 should contain ids 0 through `self.max_num_privacy_ids - 1`
    with self.subTest('data1-has-correct-ids'):
      self.assertAllInRange(
          [x[0] for x in data1], 0, self.max_num_privacy_ids - 2
      )
    with self.subTest('data2-has-correct-ids'):
      self.assertAllInRange(
          [x[0] for x in data2], 0, self.max_num_privacy_ids - 1
      )


if __name__ == '__main__':
  absltest.main()
