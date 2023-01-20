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
"""Tests for clustering_params."""

from absl.testing import absltest

import numpy as np

from clustering import clustering_params


class ClusteringParamTest(absltest.TestCase):

  def test_privacy_param_defaults(self):
    privacy_param = clustering_params.DifferentialPrivacyParam()
    self.assertEqual(privacy_param.epsilon, 1.0)
    self.assertEqual(privacy_param.delta, 1e-6)
    self.assertEqual(privacy_param.privacy_model,
                     clustering_params.PrivacyModel.CENTRAL)

  def test_privacy_budget_split_defaults(self):
    privacy_budget_split = clustering_params.PrivacyBudgetSplit()
    self.assertEqual(privacy_budget_split.frac_sum, 0.8)
    self.assertEqual(privacy_budget_split.frac_group_count, 0.2)

  def test_privacy_budget_split_invalid(self):
    with self.assertRaises(
        ValueError,
        msg="The provided privacy budget split (1.6) was greater than 1.0."):
      clustering_params.PrivacyBudgetSplit(frac_sum=0.7, frac_group_count=0.8)

  def test_tree_param(self):
    tree_param = clustering_params.TreeParam(
        min_num_points_in_branching_node=4,
        min_num_points_in_node=2,
        max_depth=5)
    self.assertEqual(tree_param.min_num_points_in_branching_node, 4)
    self.assertEqual(tree_param.min_num_points_in_node, 2)
    self.assertEqual(tree_param.max_depth, 5)

  def test_error_tree_param(self):
    with self.assertRaises(ValueError):
      clustering_params.TreeParam(
          min_num_points_in_branching_node=4,
          min_num_points_in_node=0,
          max_depth=5)
    with self.assertRaises(ValueError):
      clustering_params.TreeParam(
          min_num_points_in_branching_node=4,
          min_num_points_in_node=-2,
          max_depth=5)
    with self.assertRaises(ValueError):
      clustering_params.TreeParam(
          min_num_points_in_branching_node=4,
          min_num_points_in_node=20,
          max_depth=5)

  def test_data(self):
    (points, dim) = (10, 3)
    data = clustering_params.Data(np.ones(shape=(points, dim)), radius=1.0)
    self.assertEqual(data.num_points, points)
    self.assertEqual(data.dim, dim)
    self.assertEqual(data.radius, 1.0)

  def test_data_label_unequal_length(self):
    points, dim = 10, 3
    datapoints = np.zeros(shape=(points, dim))
    labels = np.ones(points - 1, dtype=int)
    with self.assertRaises(ValueError):
      clustering_params.Data(datapoints, radius=1.0, labels=labels)

  def test_clip_by_radius(self):
    datapoints = np.array([[3., 2., 4.], [1., 2., 3.]])
    data = clustering_params.Data(datapoints, radius=10.0)
    points_to_clip = np.array([[0., 0., 0., 0.], [1., 2., 3., 4.],
                               [5., 6., 7., 8.], [9., 10., 11., 12.],
                               [13., 14., 15., 16.]])
    clipped_datapoints = data.clip_by_radius(points_to_clip)
    self.assertLen(clipped_datapoints, 5)
    self.assertSequenceAlmostEqual(clipped_datapoints[0], [0., 0., 0., 0.])
    self.assertSequenceAlmostEqual(clipped_datapoints[1], [1., 2., 3., 4.])
    self.assertSequenceAlmostEqual(
        clipped_datapoints[2], [3.79049022, 4.54858826, 5.30668631, 6.06478435])
    self.assertSequenceAlmostEqual(
        clipped_datapoints[3], [4.26162351, 4.73513724, 5.20865096, 5.68216469])
    self.assertSequenceAlmostEqual(
        clipped_datapoints[4], [4.46949207, 4.81329915, 5.15710623, 5.50091331])

  def test_clip_by_radius_default_to_self(self):
    datapoints = np.array([[0., 0., 0., 0.], [1., 2., 3., 4.], [5., 6., 7., 8.],
                           [9., 10., 11., 12.], [13., 14., 15., 16.]])
    data = clustering_params.Data(datapoints, radius=10.0)
    clipped_datapoints = data.clip_by_radius()
    self.assertLen(clipped_datapoints, 5)
    self.assertSequenceAlmostEqual(clipped_datapoints[0], [0., 0., 0., 0.])
    self.assertSequenceAlmostEqual(clipped_datapoints[1], [1., 2., 3., 4.])
    self.assertSequenceAlmostEqual(
        clipped_datapoints[2], [3.79049022, 4.54858826, 5.30668631, 6.06478435])
    self.assertSequenceAlmostEqual(
        clipped_datapoints[3], [4.26162351, 4.73513724, 5.20865096, 5.68216469])
    self.assertSequenceAlmostEqual(
        clipped_datapoints[4], [4.46949207, 4.81329915, 5.15710623, 5.50091331])

  def test_privacy_calculator_multiplier(self):
    multiplier = clustering_params.PrivacyCalculatorMultiplier(
        gaussian_std_dev_multiplier=4.2, laplace_param_multiplier=5.1)
    alpha = 3.0
    sensitivity = 1.4
    std_dev = multiplier.get_gaussian_std_dev(alpha, sensitivity)
    self.assertEqual(std_dev, 17.64)
    self.assertEqual(multiplier.get_alpha(std_dev, sensitivity), alpha)
    self.assertEqual(multiplier.get_laplace_param(alpha), 1.0 / 15.3)


if __name__ == "__main__":
  absltest.main()
