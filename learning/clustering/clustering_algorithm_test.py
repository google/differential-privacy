# Copyright 2021 Google LLC.
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
"""Tests for clustering_algorithm."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from clustering import clustering_algorithm
from clustering import clustering_params


class ClusteringTest(parameterized.TestCase):

  def test_clustering_result_value_errors_unequal_dim(self):
    centers = np.array([[0, 0], [100, 100]])
    datapoints = np.array([[1, 0, 1], [101, 101, 99], [4, 0, 4]])
    labels = np.array([0, 1, 1], dtype=int)
    data = clustering_params.Data(datapoints=datapoints, radius=200)
    with self.assertRaises(ValueError):
      clustering_algorithm.ClusteringResult(data, centers, labels, loss=1.0)

  def test_clustering_result_value_errors_unequal_points(self):
    centers = np.array([[0, 0, 0], [1, 1, 1]])
    datapoints = np.array([[1, 0, 1], [101, 101, 99], [4, 0, 4]])
    labels = np.array([0, 1], dtype=int)
    data = clustering_params.Data(datapoints=datapoints, radius=200)
    with self.assertRaises(ValueError):
      clustering_algorithm.ClusteringResult(data, centers, labels, loss=1.0)

  def test_clustering_result_value_errors_labels_out_of_bounds(self):
    centers = np.array([[0, 0, 0], [1, 1, 1]])
    datapoints = np.array([[1, 0, 1], [101, 101, 99], [4, 0, 4]])
    data = clustering_params.Data(datapoints=datapoints, radius=200)
    for labels in [
        np.array([-1, 0, 1], dtype=int),
        np.array([0, 1, 2], dtype=int),
        np.array([0, 1, 1.1])
    ]:
      with self.assertRaises(ValueError):
        clustering_algorithm.ClusteringResult(data, centers, labels, loss=1.0)

  def test_clustering_result_value_errors_loss_label_only_one_init(self):
    centers = np.zeros((2, 3))
    datapoints = np.zeros((4, 3))
    data = clustering_params.Data(datapoints=datapoints, radius=2)
    cluster_labels = np.array([0, 0, 1, 1], dtype=int)
    loss = 1.0
    with self.assertRaises(ValueError):
      clustering_algorithm.ClusteringResult(data, centers, cluster_labels)
    with self.assertRaises(ValueError):
      clustering_algorithm.ClusteringResult(data, centers, loss=loss)

  def test_get_clustering_result(self):
    centers = np.array([[0, 0, 0], [100, 100, 100]])
    datapoints = np.array([[1, 0, 1], [101, 101, 99], [4, 0, 4]])
    data = clustering_params.Data(datapoints=datapoints, radius=200)

    clustering_result = clustering_algorithm.ClusteringResult(data, centers)

    self.assertLen(data.datapoints, 3)
    for i, datapoint in enumerate(clustering_result.data.datapoints):
      self.assertSequenceAlmostEqual(datapoints[i], datapoint)
    self.assertLen(centers, 2)
    for i, center in enumerate(clustering_result.centers):
      self.assertSequenceAlmostEqual(centers[i], center)

    self.assertListEqual(list(clustering_result.labels), [0, 1, 0])
    self.assertAlmostEqual(clustering_result.loss, 37)

  def test_clipped_data_used_for_clustering_and_not_result_calculation(
      self):
    # Clipped datapoints (radius=1): [[0.3, 0.2], [0.6, 0.8], [0.6, 0.8]]
    datapoints = np.array([[0.3, 0.2], [3, 4], [6, 8]])
    # Very small radius means the datapoint will be clipped for the center
    # calculation.
    data = clustering_params.Data(datapoints=datapoints, radius=1)
    # No noise
    privacy_param = clustering_params.DifferentialPrivacyParam(np.inf)
    # No branching, the coreset will just be the average of the points
    tree_param = clustering_params.TreeParam(1, 1, 0)
    clustering_result = clustering_algorithm.private_lsh_clustering(
        3,
        data,
        privacy_param,
        tree_param=tree_param,
        multipliers=clustering_params.PrivacyCalculatorMultiplier())

    # Center should be calculated using the clipped data.
    expected_center = np.array([0.5, 0.6])
    self.assertLen(clustering_result.centers, 1)
    self.assertSequenceAlmostEqual(clustering_result.centers[0],
                                   expected_center)

    self.assertListEqual(list(clustering_result.labels), [0, 0, 0])

    # Loss calculation should still be relative to the original points.
    self.assertAlmostEqual(clustering_result.loss, 103.02)


class ClusteringMetricsTest(absltest.TestCase):

  def test_value_error_no_true_labels(self):
    datapoints, radius = np.zeros(shape=(6, 4)), 1.0
    data = clustering_params.Data(datapoints, radius)
    centers = np.zeros(shape=(3, 4))
    cluster_labels = np.array([0, 0, 1, 1, 2, 2])
    clustering_result = clustering_algorithm.ClusteringResult(
        data, centers, cluster_labels, loss=1.0)
    with self.assertRaises(ValueError):
      clustering_result.cross_label_histogram()
    with self.assertRaises(ValueError):
      clustering_result.get_clustering_metrics()

  def test_get_clustering_metrics(self):
    datapoints, radius = np.zeros(shape=(6, 4)), 1.0
    labels = np.array([0, 0, 0, 1, 1, 1])
    data = clustering_params.Data(datapoints, radius, labels)
    centers = np.zeros(shape=(3, 4))
    cluster_labels = np.array([0, 0, 1, 1, 2, 2])
    clustering_result = clustering_algorithm.ClusteringResult(
        data, centers, cluster_labels, loss=1.0)
    clustering_metrics = clustering_result.get_clustering_metrics()

    expected_cross_label_histogram = np.array([[2, 0], [1, 1], [0, 2]],
                                              dtype=int)
    self.assertTrue((clustering_metrics.cross_label_histogram ==
                     expected_cross_label_histogram).all())
    self.assertEqual(clustering_metrics.num_points, 6)
    self.assertEqual(clustering_metrics.dominant_label_correct_count, 5)
    self.assertAlmostEqual(clustering_metrics.dominant_label_accuracy, 5 / 6)
    self.assertEqual(clustering_metrics.true_pairs, 6)
    self.assertEqual(clustering_metrics.true_nonmatch_count, 4)
    self.assertAlmostEqual(clustering_metrics.true_nonmatch_frac, 4 / 6)
    self.assertEqual(clustering_metrics.false_pairs, 9)
    self.assertEqual(clustering_metrics.false_match_count, 1)
    self.assertAlmostEqual(clustering_metrics.false_match_frac, 1 / 9)


class ClusteringEdgeCaseTest(parameterized.TestCase):
  baseline_k: int
  baseline_privacy_param: clustering_params.DifferentialPrivacyParam

  def setUp(self):
    super().setUp()
    self.baseline_k = 2
    self.baseline_privacy_param = clustering_params.DifferentialPrivacyParam()

  def test_small_dataset(self):
    datapoints = np.array([[0.3, 0.2]])
    data = clustering_params.Data(datapoints=datapoints, radius=1)
    self.assertIsNotNone(
        clustering_algorithm.private_lsh_clustering(
            self.baseline_k,
            data,
            self.baseline_privacy_param,
            multipliers=clustering_params.PrivacyCalculatorMultiplier(),
        )
    )

  def test_privacy_budget_split_does_not_error(self):
    datapoints = np.array([[0.3, 0.2]])
    data = clustering_params.Data(datapoints=datapoints, radius=1)
    self.assertIsNotNone(
        clustering_algorithm.private_lsh_clustering(
            self.baseline_k,
            data,
            self.baseline_privacy_param,
            privacy_budget_split=clustering_params.PrivacyBudgetSplit(),
        )
    )


if __name__ == '__main__':
  absltest.main()
