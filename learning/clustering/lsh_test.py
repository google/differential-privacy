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
"""Tests for lsh."""

from absl.testing import absltest
import numpy as np

from clustering import lsh


class LshTest(absltest.TestCase):

  def test_projection_vectors_shape(self):
    dim, max_hash_len = 10, 6
    sh = lsh.SimHash(dim, max_hash_len)
    self.assertEqual(sh.projection_vectors.shape, (max_hash_len, dim))

  def test_value_errors(self):
    dim, max_hash_len = 10, 6
    num_points = 50
    sh = lsh.SimHash(dim, max_hash_len)
    datapoints = np.random.normal(size=(num_points, dim))
    with self.assertRaises(ValueError):
      sh.group_by_next_hash(datapoints, hash_prefix="010010")
    with self.assertRaises(ValueError):
      sh.group_by_next_hash(datapoints, hash_prefix="0101011")

  def test_group_by_next_hash_shape(self):
    dim, max_hash_len = 10, 6
    num_points = 50
    sh = lsh.SimHash(dim, max_hash_len)
    datapoints = np.random.normal(size=(num_points, dim))
    children = sh.group_by_next_hash(datapoints)
    self.assertEqual(children["0"].shape[0] + children["1"].shape[0],
                     num_points)

  def test_group_by_next_hash(self):
    dim, max_hash_len = 5, 2
    hash_prefix = "0"
    projection_vectors = np.array([[0, 1, 1, -1, 0], [1, 0, -1, 0, 0]])
    sh = lsh.SimHash(dim, max_hash_len, projection_vectors)
    datapoints = np.array([[1.5, 0, 1, 0.5, 0],
                           [1, 1, 0, 0.7, 0.1],
                           [0.8, 0.1, 1, 0.2, 0.4],
                           [0.1, 0.5, 0.3, 0.7, 0.8],
                           [-0.5, 0.1, -0.3, -0.4, 0.2]])
    children = sh.group_by_next_hash(datapoints, hash_prefix)
    self.assertTrue((children["0"] == datapoints[[0, 1]]).all())
    self.assertTrue((children["1"] == datapoints[[2, 3, 4]]).all())


if __name__ == "__main__":
  absltest.main()
