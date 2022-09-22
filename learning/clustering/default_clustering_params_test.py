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
from clustering import test_utils


class ClusteringParamTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('basic', 100000, 90000, 10, 25.8, 567, 189, 20),
      ('zero_std_dev', 100000, 90000, 10, 0, 3, 1, 20),
      ('min_num_points_min', 1000, 900, 100, 25.8, 12, 4, 20),
      ('negative_private_count', 100000, -100, 10, 25.8, 3, 1, 20),
  )
  @mock.patch.object(central_privacy_utils, 'get_private_count', autospec=True)
  def test_default_tree_param(self, points, returned_private_count, k,
                              gaussian_std_dev,
                              expected_min_num_points_in_branching_node,
                              expected_min_num_points_in_node,
                              expected_max_depth, mock_private_count):
    dim = 10
    radius = 4.3
    mock_private_count.return_value = returned_private_count
    data = clustering_params.Data(np.ones(shape=(points, dim)), radius=radius)
    pcalc = test_utils.get_test_privacy_calculator(
        gaussian_std_dev=gaussian_std_dev, sensitivity=radius)
    (tree_param, private_count) = default_clustering_params.default_tree_param(
        k, data, pcalc, expected_max_depth)
    self.assertEqual(tree_param.max_depth, expected_max_depth)
    mock_private_count.assert_called_once_with(
        nonprivate_count=points, count_privacy_param=pcalc.count_privacy_param)
    self.assertEqual(private_count, returned_private_count)
    self.assertEqual(tree_param.min_num_points_in_node,
                     expected_min_num_points_in_node)
    self.assertEqual(tree_param.min_num_points_in_branching_node,
                     expected_min_num_points_in_branching_node)


if __name__ == '__main__':
  absltest.main()
