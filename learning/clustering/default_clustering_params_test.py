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
"""Tests for default_clustering_params."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from clustering import central_privacy_utils
from clustering import clustering_params
from clustering import default_clustering_params
from dp_accounting import accountant
from dp_accounting import common


class ClusteringParamTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('basic', 100000, 90000, 10, 1.0, 567, 189, 20),
      ('inf_eps', 100000, 90000, 10, np.inf, 3, 1, 20),
      ('min_num_points_min', 1000, 900, 100, 1.0, 12, 4, 20),
      ('negative_private_count', 100000, -100, 10, 85, 3, 1, 20),
  )
  @mock.patch.object(central_privacy_utils, 'get_private_count', autospec=True)
  @mock.patch.object(
      accountant, 'get_smallest_gaussian_noise', return_value=6, autospec=True)
  def test_default_tree_param(self, points, returned_private_count, k, epsilon,
                              expected_min_num_points_in_branching_node,
                              expected_min_num_points_in_node,
                              expected_max_depth, mock_gaussian_noise,
                              mock_private_count):
    dim = 10
    mock_private_count.return_value = returned_private_count
    data = clustering_params.Data(np.ones(shape=(points, dim)), radius=1.0)
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=epsilon, delta=1e-2)
    budget_split = clustering_params.PrivacyBudgetSplit(
        frac_sum=0.8, frac_group_count=0.2)

    (tree_param, private_count) = default_clustering_params.default_tree_param(
        k, data, privacy_param, budget_split)
    self.assertEqual(tree_param.max_depth, expected_max_depth)
    if epsilon == np.inf:
      mock_gaussian_noise.assert_not_called()
    else:
      mock_gaussian_noise.assert_called_once_with(
          common.DifferentialPrivacyParameters(0.8 * epsilon, 1e-2), 1, 1.0)
    mock_private_count.assert_called_once_with(
        nonprivate_count=points,
        count_privacy_param=central_privacy_utils.CountPrivacyParam(
            epsilon=0.2 * epsilon / (tree_param.max_depth + 1), delta=1e-2))
    self.assertEqual(private_count, returned_private_count)
    self.assertEqual(tree_param.min_num_points_in_node,
                     expected_min_num_points_in_node)
    self.assertEqual(tree_param.min_num_points_in_branching_node,
                     expected_min_num_points_in_branching_node)


if __name__ == '__main__':
  absltest.main()
