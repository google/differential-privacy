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

_NUM_POINTS = flags.DEFINE_integer('num_points', 100000,
                                   'Number of points in synthetic dataset.')
_DIM = flags.DEFINE_integer('dim', 100,
                            'Dimension of points in synthetic dataset.')
_NUM_CLUSTERS = flags.DEFINE_integer(
    'num_clusters', 64, 'Number of clusters in synthetic dataset.')
_CLUSTER_RATIO = flags.DEFINE_float(
    'cluster_ratio', 8.0,
    'Parameter controlling the ratio of distances between points in different '
    'vs. same cluster.')
_RADIUS = flags.DEFINE_float(
    'radius', 1.0,
    'Radius of ball in which all points in synthetic dataset lie.')

_FIXED_EPS = flags.DEFINE_float(
    'fixed_epsilon', 1.0,
    'Value of epsilon to use when experimenting with varying k.')
_K_TO_TRY = flags.DEFINE_list(
    'k_to_try', '2, 4, 8, 16, 32, 64',
    'List of k values to use when experimenting with varying k.')
_FIXED_K = flags.DEFINE_integer(
    'fixed_k', 64, 'Value of k when experimenting with varying epsilon.')
_EPS_TO_TRY = flags.DEFINE_list(
    'epsilon_to_try', '0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, inf',
    'List of epsilon values to use when experimenting with varying epsilon.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data: clustering_params.Data = data_generation.generate_synthetic_dataset(
      _NUM_POINTS.value, _DIM.value, _NUM_CLUSTERS.value, _CLUSTER_RATIO.value,
      _RADIUS.value)
  print('==== Synthetic Dataset Information ====\n'
        f'Number of datapoints: {_NUM_POINTS.value}\n'
        f'Dimensions: {_DIM.value}\n'
        f'Number of clusters: {_NUM_CLUSTERS.value}\n'
        f'Radius: {_RADIUS.value}\n'
        f'Cluster centers drawn from: Uniform over ball of '
        f'radius={_RADIUS.value * (1 - 1 / float(_CLUSTER_RATIO.value)):.4}\n'
        f'Each cluster drawn from: '
        f'N(cluster_center, '
        f'{_RADIUS.value / _CLUSTER_RATIO.value / np.sqrt(_DIM.value):.4} * I) '
        f'clipped to ball of radius {_RADIUS.value}')

  eval_head = ('|  k | epsilon | clustering loss |    dominant label accuracy '
               '   | false match fraction | true non-match fraction |')

  def run_clustering(k: int, eps: float) -> None:
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=eps, delta=1e-6)
    clustering_result: clustering_algorithm.ClusteringResult = (
        clustering_algorithm.private_lsh_clustering(
            k,
            data,
            privacy_param))
    clustering_metrics: clustering_algorithm.ClusteringMetrics = (
        clustering_result.get_clustering_metrics())
    correct_pred = clustering_metrics.dominant_label_correct_count
    accuracy = clustering_metrics.dominant_label_accuracy
    false_match_frac = clustering_metrics.false_match_frac
    true_nonmatch_frac = clustering_metrics.true_nonmatch_frac
    print(
        f'| {k:>2} | {eps:>7} '
        f'| {clustering_result.loss:>15.8} '
        f'| {accuracy:>6.2} ({correct_pred:>6} out of {_NUM_POINTS.value:>6}) '
        f'| {false_match_frac:>20.4} '
        f'| {true_nonmatch_frac:>23.4} |')

  print(f'\n# Evaluation with epsilon = {_FIXED_EPS.value} and '
        f'varying k in {list(map(int, _K_TO_TRY.value))}')
  print(eval_head)
  for k in list(map(int, _K_TO_TRY.value)):
    run_clustering(k, _FIXED_EPS.value)

  print(f'\n# Evaluation with k = {_FIXED_K.value} and '
        f'varying epsilon in {list(map(float, _EPS_TO_TRY.value))}')
  print(eval_head)
  for epsilon in list(map(float, _EPS_TO_TRY.value)):
    run_clustering(_FIXED_K.value, epsilon)

  print('Note: all computations apart from cluster centers, such as loss, '
        'label accuracy, etc. above are not differentially private.')


if __name__ == '__main__':
  app.run(main)
