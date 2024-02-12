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
"""Tests different concrete classes of sparse vector technique."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from dp_auditorium.configs import mechanism_config
from dp_auditorium.mechanisms import sparse_vector_technique

_EPSILON = 1000
_MAX_ANSWERED_QUERIES = 2
_MIN_VALUE = 0.1
_MAX_VALUE = 1.1
_THRESHOLD = 0.5
_NUM_QUERIES = 3
_SENSITIVITY = _MAX_VALUE - _MIN_VALUE
_SEED = 0
_RNG = np.random.default_rng(seed=_SEED)
_CONFIG = mechanism_config.SVTMechanismConfig(
    epsilon=_EPSILON,
    max_value=_MAX_VALUE,
    min_value=_MIN_VALUE,
    query_type=mechanism_config.QueryType.QUERY_TYPE_SUM,
    num_queries=_NUM_QUERIES,
    threshold=_THRESHOLD,
    max_answered_queries=_MAX_ANSWERED_QUERIES,
)


class SparseVectorTechniqueTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      sparse_vector_technique.SVT1,
      sparse_vector_technique.SVT2,
      sparse_vector_technique.SVT4,
      sparse_vector_technique.SVT5,
      sparse_vector_technique.SVT6,
  )
  def test_get_binary_outputs_mechanism(self, svt_mechanism_class):
    """Checks that svt_mechanism's outputs before aborting are correct."""
    svt_mechanism = svt_mechanism_class(_CONFIG, _RNG)

    dummy_noisy_queries = np.array([[30.1, -100, 5.0], [30.1, -100, 5.0]])
    dummy_threshold = np.zeros((dummy_noisy_queries.shape[0], 1))
    result_threshold = svt_mechanism.get_query_output(
        dummy_noisy_queries, dummy_threshold
    )

    above = mechanism_config.DataValues.SVT_ABOVE_THRESHOLD
    below = mechanism_config.DataValues.SVT_BELOW_THRESHOLD
    expected_query_output = (
        np.array([[above, below, above], [above, below, above]]),
        np.array([[True, False, True], [True, False, True]]),
    )
    self.assertAllEqual(expected_query_output, result_threshold)

  @parameterized.parameters(
      sparse_vector_technique.SVT1,
      sparse_vector_technique.SVT2,
      sparse_vector_technique.SVT4,
  )
  def test_binary_final_outputs_with_max_count_above_threshold(
      self, svt_mechanism_class
  ):
    """Checks outputs for svt_mechanism with binary outputs and max_counts."""
    svt_mechanism = svt_mechanism_class(_CONFIG, _RNG)
    # We test on data where the `sum` query will be above the 0.5 threshold used
    # to initialize the mechanism.
    data_above_threshold = [5]
    num_samples = 2
    final_result_above_threshold = svt_mechanism(
        data_above_threshold, num_samples=num_samples
    )

    above = mechanism_config.DataValues.SVT_ABOVE_THRESHOLD
    no_response = mechanism_config.DataValues.SVT_NO_RESPONSE
    expected_output_above_threshold = np.array(
        [[above, above, no_response], [above, above, no_response]]
    )
    self.assertAllEqual(
        expected_output_above_threshold, final_result_above_threshold
    )

  @parameterized.parameters(
      sparse_vector_technique.SVT1,
      sparse_vector_technique.SVT2,
      sparse_vector_technique.SVT4,
      sparse_vector_technique.SVT5,
      sparse_vector_technique.SVT6,
  )
  def test_binary_final_outputs_below_threshold(self, svt_mechanism_class):
    """Checks outputs for svt_mechanism with binary outputs and max_counts."""
    svt_mechanism = svt_mechanism_class(_CONFIG, _RNG)
    # We test on data where the `sum` query will be below the 0.5 threshold used
    # to initialize the mechanism.
    data_below_threshold = [-1]
    num_samples = 2
    final_result_below_threshold = svt_mechanism(
        data_below_threshold, num_samples=num_samples
    )

    below = mechanism_config.DataValues.SVT_BELOW_THRESHOLD
    expected_output_below_threshold = np.array(
        [[below, below, below], [below, below, below]]
    )
    self.assertAllEqual(
        expected_output_below_threshold, final_result_below_threshold
    )

  @parameterized.parameters(
      (
          sparse_vector_technique.SVT1,
          4 * _MAX_ANSWERED_QUERIES * _SENSITIVITY / _EPSILON,
      ),
      (
          sparse_vector_technique.SVT2,
          4 * _MAX_ANSWERED_QUERIES * _SENSITIVITY / _EPSILON,
      ),
      (
          sparse_vector_technique.SVT3,
          2 * _MAX_ANSWERED_QUERIES * _SENSITIVITY / _EPSILON,
      ),
      (sparse_vector_technique.SVT4, 4 * _SENSITIVITY / (3 * _EPSILON)),
      (sparse_vector_technique.SVT5, 0),
      (sparse_vector_technique.SVT6, 2 * _SENSITIVITY / _EPSILON),
  )
  def test_svt_has_correct_noise_scale(
      self,
      svt_mechanism_class,
      expected_query_scale,
  ):
    svt_mechanism = svt_mechanism_class(_CONFIG, _RNG)
    stub_queries = 0.0
    num_samples = 10000

    # noisy_queries will have shape (num_samples, num_queries), for a total of
    # `num_samples * num_queries` observations of the noise.
    noisy_queries = svt_mechanism.get_noisy_queries(
        queries_values=stub_queries,
        num_samples=num_samples,
    )

    error = (
        3 * expected_query_scale / np.sqrt(num_samples * _CONFIG.num_queries)
    )
    self.assertAllClose(stub_queries, np.mean(noisy_queries), atol=error)

  @parameterized.parameters(
      (sparse_vector_technique.SVT1, 2 * _SENSITIVITY / _EPSILON),
      (
          sparse_vector_technique.SVT2,
          2 * _MAX_ANSWERED_QUERIES * _SENSITIVITY / _EPSILON,
      ),
      (sparse_vector_technique.SVT3, 2 * _SENSITIVITY / _EPSILON),
      (sparse_vector_technique.SVT4, 4 * _SENSITIVITY / _EPSILON),
      (sparse_vector_technique.SVT5, 2 / _EPSILON),
      (sparse_vector_technique.SVT6, 2 * _SENSITIVITY / _EPSILON),
  )
  def test_svt_has_correct_threshold_noise_scale(
      self, svt_mechanism_class, expected_threshold_scale
  ):
    svt_mechanism = svt_mechanism_class(_CONFIG, _RNG)
    num_samples = 10000
    thresholds = svt_mechanism.get_noisy_thresholds(num_samples=num_samples)
    self.assertAllClose(
        _THRESHOLD,
        np.mean(thresholds),
        atol=3 * expected_threshold_scale / np.sqrt(num_samples),
    )


class QueriesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (np.array([-3, 5]), 0.4), (np.array([0.3, 0.8, -0.4]), 0.7)
  )
  def test_get_sum_query(self, data, expected_result):
    num_queries = 3
    min_value = -1.1
    max_value = 1.5
    expected_sensitivity = max_value - min_value
    sum_query = sparse_vector_technique._get_sum_query(
        min_value, max_value, num_queries
    )
    result = sum_query.query_calculator(data)
    with self.subTest("sensitivity-is-correct"):
      self.assertEqual(sum_query.sensitivity, expected_sensitivity)
    with self.subTest("result-is-correct"):
      self.assertAllClose(result, np.repeat(expected_result, num_queries))

  @parameterized.parameters((np.array([-3, 5]),), (np.array([0.3, 0.8, -0.4]),))
  def test_get_reveal_records_query(self, data):
    num_queries = 3
    min_value = -1.1
    max_value = 1.5
    expected_sensitivity = max_value - min_value
    clipped_data = np.clip(data, min_value, max_value)
    expected_result = np.array(
        [clipped_data[-1], clipped_data[-1], clipped_data[-2]]
    )

    sum_query = sparse_vector_technique._get_reveal_records_query(
        min_value, max_value, num_queries
    )
    result = sum_query.query_calculator(data)
    with self.subTest("sensitivity-is-correct"):
      self.assertEqual(sum_query.sensitivity, expected_sensitivity)
    with self.subTest("result-is-correct"):
      self.assertAllClose(result, expected_result)

  def test_get_reveal_records_query_on_neighbor_data(self):
    data1 = np.array([0.0, 1.0])
    data2 = np.array([0.0, 1.0, 0.0])

    min_value = 0.0
    max_value = 1.0

    expected_result_data1 = np.array([1.0, 1.0, 1.0, 0.0])
    expected_result_data2 = np.array([0.0, 0.0, 0.0, 1.0])

    sum_query = sparse_vector_technique._get_reveal_records_query(
        min_value, max_value, 4
    )
    result1 = sum_query.query_calculator(data1)
    result2 = sum_query.query_calculator(data2)

    with self.subTest("result-is-correct-in-data1"):
      self.assertAllClose(expected_result_data1, result1)
    with self.subTest("result-is-correct-in-data2"):
      self.assertAllClose(expected_result_data2, result2)


if __name__ == "__main__":
  absltest.main()
