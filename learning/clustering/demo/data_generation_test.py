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
"""Tests for data generation."""

from absl.testing import absltest

import numpy as np

from clustering import clustering_params
from clustering.demo import data_generation


class DataGenerationTest(absltest.TestCase):

  def test_sample_uniform_sphere(self):
    num_points, dim, radius = 1000, 10, 3.0
    datapoints = data_generation.sample_uniform_sphere(num_points, dim, radius)
    self.assertEqual(datapoints.shape, (num_points, dim))
    max_norm = np.max(np.linalg.norm(datapoints, axis=1))
    self.assertLessEqual(max_norm, radius)

  def test_data_generation_shape_and_radius(self):
    num_points, dim, radius = 1000, 10, 3.0
    num_clusters, cluster_ratio = 60, 20.0
    data: clustering_params.Data = data_generation.generate_synthetic_dataset(
        num_points, dim, num_clusters, cluster_ratio, radius)
    self.assertEqual(data.datapoints.shape, (num_points, dim))
    # Test that all points are contained within the radius ball.
    max_norm = np.max(np.linalg.norm(data.datapoints, axis=1))
    self.assertLessEqual(max_norm, radius)
    # Also test that not all points are at the boundary of the radius ball.
    min_norm = np.min(np.linalg.norm(data.datapoints, axis=1))
    self.assertLess(min_norm, radius*(1 - 1.0/cluster_ratio))

  def test_size_of_clusters(self):
    num_points, dim, radius = 1000, 10, 3.0
    num_clusters, cluster_ratio = 60, 20.0
    data: clustering_params.Data = data_generation.generate_synthetic_dataset(
        num_points, dim, num_clusters, cluster_ratio, radius)
    for k in range(num_clusters):
      self.assertGreaterEqual(sum(data.labels == k), num_points // num_clusters)

  def test_average_distances(self):
    num_points, dim, radius = 1003, 10, 3.0
    num_clusters, cluster_ratio = 10, 20.0
    data: clustering_params.Data = data_generation.generate_synthetic_dataset(
        num_points, dim, num_clusters, cluster_ratio, radius)
    sum_intercluster_dist, num_intercluster_pairs = 0.0, 0
    sum_intracluster_dist, num_intracluster_pairs = 0.0, 0
    for i in range(num_points):
      for j in range(i+1, num_points):
        if data.labels[i] == data.labels[j]:
          sum_intercluster_dist += np.linalg.norm(data.datapoints[i] -
                                                  data.datapoints[j])
          num_intercluster_pairs += 1
        else:
          sum_intracluster_dist += np.linalg.norm(data.datapoints[i] -
                                                  data.datapoints[j])
          num_intracluster_pairs += 1
    avg_intercluster_dist = sum_intercluster_dist / num_intercluster_pairs
    avg_intracluster_dist = sum_intracluster_dist / num_intracluster_pairs
    self.assertGreater(avg_intracluster_dist / avg_intercluster_dist,
                       cluster_ratio)


if __name__ == '__main__':
  absltest.main()
