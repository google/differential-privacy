# Copyright 2022 Google LLC.
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
"""Tests for coreset_params."""

from absl.testing import absltest

from clustering import clustering_params
from clustering import coreset_params
from clustering import test_utils


class ClusteringParamTest(absltest.TestCase):

  def test_coreset_param(self):
    pcalc = test_utils.get_test_privacy_calculator()
    tree_param = clustering_params.TreeParam(
        min_num_points_in_branching_node=4,
        min_num_points_in_node=2,
        max_depth=5)
    coreset_param = coreset_params.CoresetParam(
        pcalc=pcalc,
        tree_param=tree_param,
        short_description="TestCoresetParam",
        radius=20)
    self.assertEqual(coreset_param.pcalc, pcalc)
    self.assertEqual(coreset_param.tree_param, tree_param)
    self.assertEqual(coreset_param.short_description, "TestCoresetParam")
    self.assertEqual(coreset_param.radius, 20)


if __name__ == "__main__":
  absltest.main()
