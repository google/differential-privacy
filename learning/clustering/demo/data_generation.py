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
"""Generates data for clustering experimentation."""

import numpy as np

from clustering import clustering_params


def sample_uniform_sphere(num_points: int,
                          dim: int,
                          radius: float = 1.0) -> clustering_params.Points:
  """Returns points sampled uniformly in a L2-ball of specified radius.

  Samples a point from the standard normal distribution, scales it to be of norm
  equal to the radius and finally scales it further by a factor u^{1/dim} for a
  uniformly random u in [0,1], to yield a point that is uniform within the
  L2-ball of specified radius.
  Reference:
  https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_within_the_n-ball

  Args:
    num_points: number of points to be sampled.
    dim: dimension of points to be sampled.
    radius: radius of the ball which contain all points.
  """
  points = np.random.normal(0.0, 1.0, size=(num_points, dim))
  new_radiuses = radius * (np.random.uniform(0, 1, num_points)**(1.0 / dim))
  scale = new_radiuses / np.linalg.norm(points, axis=1)
  result = (points.T * scale).T
  assert max(np.linalg.norm(result, axis=1)) <= radius, (
      f"Sampled points outside the sphere with radius {radius}, "
      f"got {max(np.linalg.norm(result, axis=1))}")
  return result


def generate_synthetic_dataset(
    num_points: int = 1000000,
    dim: int = 100,
    num_clusters: int = 64,
    cluster_ratio: float = 100.0,
    radius: float = 1.0) -> clustering_params.Data:
  """Generates a synthetic dataset.

  First samples cluster centers within a smaller radius of
  radius*(1-1/cluster_ratio), so that points added around them stay within
  radius. Next, num_points/num_clusters many points are sampled from the
  Gaussian distribution centered at each cluster (if num_points/num_clusters is
  not an integer, then excess points are in the last cluster). Finally, points
  are clipped to norm=radius.

  Args:
    num_points: The number of data points.
    dim: The dimension of data points.
    num_clusters: The number of clusters to divide the points evenly into;
      extras go in the last cluster.
    cluster_ratio: The ratio of the intercluster distance to intracluster
      distance.
    radius: The radius for all the data to be confined in. At the end, this
      radius is enforced by scaling any points that are outside the radius.

  Returns:
    Data containing sampled datapoints, radius, and labels.
  """
  center_radius = radius * (1 - 1 / float(cluster_ratio))
  rand_centers: np.ndarray = sample_uniform_sphere(
      num_clusters, dim, center_radius)  # shape=(num_clusters, dim)
  datapoints: np.ndarray = np.random.normal(
      0,
      np.sqrt(radius) / (float(cluster_ratio) * np.sqrt(dim)),
      size=(num_points, dim))

  num_points_per_cluster: np.ndarray = np.ones(num_clusters, dtype=int) * (
      num_points // num_clusters)
  num_points_per_cluster[-1] += num_points % num_clusters

  labels = np.concatenate([
      np.ones(k, dtype=int) * i for (i, k) in enumerate(num_points_per_cluster)
  ])
  shift_mat: np.ndarray = np.vstack([
      np.outer(np.ones(k), v)
      for (k, v) in zip(num_points_per_cluster, rand_centers)
  ])
  datapoints += shift_mat

  # Enforce the radius by scaling any points that are outside that range.
  data = clustering_params.Data(datapoints, radius, labels)
  return clustering_params.Data(data.clip_by_radius(), data.radius, data.labels)
