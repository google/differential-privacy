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
"""Test utilities for clustering."""

from clustering import clustering_params


def get_test_clustering_param(epsilon=1.0,
                              delta=1e-2,
                              frac_sum=0.2,
                              frac_group_count=0.8,
                              min_num_points_in_branching_node=4,
                              min_num_points_in_node=2,
                              max_depth=4,
                              radius=1):
  # pylint: disable=g-doc-args
  """Returns clustering_param with defaults for params not needed for testing.

  Usage: Explicitly pass in parameters that are relied on in the test.
  """
  privacy_param = clustering_params.DifferentialPrivacyParam(
      epsilon=epsilon, delta=delta)
  privacy_budget_split = clustering_params.PrivacyBudgetSplit(
      frac_sum=frac_sum,
      frac_group_count=frac_group_count)
  tree_param = clustering_params.TreeParam(
      min_num_points_in_branching_node=min_num_points_in_branching_node,
      min_num_points_in_node=min_num_points_in_node,
      max_depth=max_depth)
  clustering_param = clustering_params.ClusteringParam(
      privacy_param=privacy_param,
      privacy_budget_split=privacy_budget_split,
      tree_param=tree_param,
      short_description='TestClusteringParam',
      radius=radius)
  return clustering_param
