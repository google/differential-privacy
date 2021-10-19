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
"""Basic Example for Using Private Clustering."""

from typing import Sequence

from absl import app
from absl import flags
import numpy as np

from clustering import clustering_algorithm
from clustering import clustering_params
from clustering.demo import data_generation

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_points', 100000,
                     'Number of points in synthetic dataset.')
flags.DEFINE_integer('dim', 100, 'Dimension of points in synthetic dataset.')
flags.DEFINE_integer('num_clusters', 64,
                     'Number of clusters in synthetic dataset.')
flags.DEFINE_float(
    'cluster_ratio', 8.0,
    'Parameter controlling the ratio of distances between points in different '
    'vs. same cluster.')
flags.DEFINE_float(
    'radius', 1.0,
    'Radius of ball in which all points in synthetic dataset lie.')

flags.DEFINE_float(
    'fixed_epsilon', 1.0,
    'Value of epsilon to use when experimenting with varying k.')
flags.DEFINE_list('k_to_try', '2, 4, 8, 16, 32, 64',
                  'List of k values to use when experimenting with varying k.')
flags.DEFINE_integer('fixed_k', 64,
                     'Value of k when experimenting with varying epsilon.')
flags.DEFINE_list(
    'epsilon_to_try', '0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, inf',
    'List of epsilon values to use when experimenting with varying epsilon.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data: clustering_params.Data = data_generation.generate_synthetic_dataset(
      FLAGS.num_points, FLAGS.dim, FLAGS.num_clusters, FLAGS.cluster_ratio,
      FLAGS.radius)
  print('==== Synthentic Dataset Information ====\n'
        f'Number of datapoints: {FLAGS.num_points}\n'
        f'Dimensions: {FLAGS.dim}\n'
        f'Number of clusters: {FLAGS.num_clusters}\n'
        f'Radius: {FLAGS.radius}\n'
        f'Cluster centers drawn from: Uniform over ball of '
        f'radius={FLAGS.radius * (1 - 1 / float(FLAGS.cluster_ratio)):.4}\n'
        f'Each cluster drawn from: '
        f'N(cluster_center, '
        f'{FLAGS.radius / FLAGS.cluster_ratio / np.sqrt(FLAGS.dim):.4} * I) '
        f'clipped to ball of radius {FLAGS.radius}')

  eval_head = ('|  k | epsilon | clustering loss |    dominant label accuracy '
               '   | false match fraction | true non-match fraction |')

  def run_clustering(k: int, eps: float) -> None:
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=eps, delta=1e-6)
    clustering_result: clustering_algorithm.ClusteringResult = (
        clustering_algorithm.private_lsh_clustering(k, data, privacy_param))
    clustering_metrics: clustering_algorithm.ClusteringMetrics = (
        clustering_result.get_clustering_metrics())
    correct_pred = clustering_metrics.dominant_label_correct_count
    accuracy = clustering_metrics.dominant_label_accuracy
    false_match_frac = clustering_metrics.false_match_frac
    true_nonmatch_frac = clustering_metrics.true_nonmatch_frac
    print(f'| {k:>2} | {eps:>7} '
          f'| {clustering_result.loss:>15.8} '
          f'| {accuracy:>6.2} ({correct_pred:>6} out of {FLAGS.num_points:>6}) '
          f'| {false_match_frac:>20.4} '
          f'| {true_nonmatch_frac:>23.4} |')

  print(f'\n# Evaluation with epsilon = {FLAGS.fixed_epsilon} and '
        f'varying k in {list(map(int, FLAGS.k_to_try))}')
  print(eval_head)
  for k in list(map(int, FLAGS.k_to_try)):
    run_clustering(k, FLAGS.fixed_epsilon)

  print(f'\n# Evaluation with k = {FLAGS.fixed_k} and '
        f'varying epsilon in {list(map(float, FLAGS.epsilon_to_try))}')
  print(eval_head)
  for epsilon in list(map(float, FLAGS.epsilon_to_try)):
    run_clustering(FLAGS.fixed_k, epsilon)

  print('Note: all computations apart from cluster centers, such as loss, '
        'label accuracy, etc. above are not differentially private.')


if __name__ == '__main__':
  app.run(main)
