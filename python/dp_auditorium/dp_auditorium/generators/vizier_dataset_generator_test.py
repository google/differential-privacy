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
"""Tests for vizier_dataset_generator."""
import time
from absl.testing import absltest
import numpy as np
import tensorflow as tf
from vizier.service import clients

from dp_auditorium import interfaces
from dp_auditorium.configs import dataset_generator_config
from dp_auditorium.generators import vizier_dataset_generator


clients.environment_variables.servicer_use_sql_ram()

_STUB_DATA = np.ones(2)


class StubVizierGenerator(vizier_dataset_generator.VizierDatasetGenerator):
  """Concrete vizier generator class."""

  def get_neighboring_datasets_from_vizier_params(
      self, vizier_params: np.ndarray
  ) -> interfaces.NeighboringDatasetsType:
    return _STUB_DATA, _STUB_DATA


class VizierDatasetGeneratorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.num_params = 2
    self.min_value = 0.0
    self.max_value = 1.0
    self.config = dataset_generator_config.VizierDatasetGeneratorConfig(
        study_name='stub_study'+str(time.time()),
        study_owner='owner',
        num_vizier_parameters=self.num_params,
        data_type=dataset_generator_config.DataType.DATA_TYPE_FLOAT,
        min_value=self.min_value,
        max_value=self.max_value,
        search_algorithm='GRID_SEARCH',
        metric_name='divergence_estimator',
    )
    self.vizier_generator = StubVizierGenerator(self.config)

  def test_vizier_generator_initializes_attributes(self):
    with self.subTest('metric-name-is-correct'):
      self.assertEqual(
          'divergence_estimator', self.vizier_generator._metric_name
      )
    with self.subTest('trial-loaded-is-false'):
      self.assertFalse(
          self.vizier_generator._trial_loaded,
          'Generator initialized trial_loaded as True.',
      )
    with self.subTest('num-params-is-correct'):
      self.assertEqual(
          self.num_params, self.vizier_generator._num_vizier_params
      )

  def test_load_trial(self):
    self.vizier_generator._load_trial()
    with self.subTest('updates-trial-loaded-true'):
      self.assertTrue(
          self.vizier_generator._trial_loaded,
          'Loading a trial did not update `trial_loaded`',
      )
    with self.subTest('trial-is-not-none'):
      self.assertIsNotNone(self.vizier_generator._last_trial)

  def test_complete_trial(self):
    trial_result = 6.0
    self.vizier_generator._load_trial()
    self.vizier_generator._complete_trial(last_trial_result=trial_result)
    self.assertFalse(self.vizier_generator._trial_loaded)

  def test_extract_params_from_trial_raises_exception_with_no_trial(self):
    with self.assertRaisesRegex(
        ValueError,
        'Trying to extract parameters from trial but no trial is loaded.',
    ):
      _ = self.vizier_generator._extract_params_from_trial()

  def test_extract_params_from_trial_returns_data_with_loaded_trial(self):
    self.vizier_generator._load_trial()
    data = self.vizier_generator._extract_params_from_trial()
    self.assertShapeEqual(np.zeros(self.num_params), data)

  def test_call_method_returns_neighboring_datasets(self):
    dataset1, dataset2 = self.vizier_generator(None)
    with self.subTest('returns-neighboring-datasets-first-call'):
      self.assertAllEqual(_STUB_DATA, dataset1)
      self.assertAllEqual(_STUB_DATA, dataset2)

    with self.subTest('returns-neighboring-datasets-second-call'):
      dataset1, dataset2 = self.vizier_generator(last_trial_result=5.0)
      self.assertAllEqual(_STUB_DATA, dataset1)
      self.assertAllEqual(_STUB_DATA, dataset2)

  def test_vizier_generator_with_int_data(self):
    min_value = 1
    max_value = 10
    self.config.min_value = min_value
    self.config.max_value = max_value
    self.config.data_type = dataset_generator_config.DataType.DATA_TYPE_INT32
    self.config.study_name = 'int_study'

    int_type_generator = StubVizierGenerator(self.config)
    int_type_generator._load_trial()
    data = int_type_generator._extract_params_from_trial()
    self.assertAllGreaterEqual(data, min_value)
    self.assertAllLessEqual(data, max_value)


if __name__ == '__main__':
  absltest.main()
