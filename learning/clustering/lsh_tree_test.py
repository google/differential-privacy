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
"""Tests for lsh_tree."""

import typing
from unittest import mock

from absl.testing import absltest
import numpy as np
from scipy import stats

from clustering import clustering_params
from clustering import lsh
from clustering import lsh_tree
from clustering import test_utils


def get_test_origin_points(nonprivate_count=5, dim=10):
  """Points with defaults for parameters not needed for the test."""
  return np.zeros((nonprivate_count, dim))


def get_test_sim_hash(dim=10, max_hash_len=1):
  """SimHash with defaults for parameters not needed for the test."""
  return lsh.SimHash(dim=dim, max_hash_len=max_hash_len)


class TestLshTreeNode(lsh_tree.LshTreeNode):
  """Test node for testing without the real children hashing logic.

  This test implementation always returns a fraction of the points with the next
  hash_prefix containing 0, and the rest with 1, setting the private count equal
  to the real count + 1. Equality is checked by just checking the hash_prefix
  and private_count.
  """
  # Fraction of the nonprivate_points to add a 0 to the hash_prefix.
  frac_zero: float

  def __init__(self, *args, frac_zero: float = 0.5, **kwargs):
    """Initializes test node.

    Args:
      *args: extra arguments for the parent class
      frac_zero: fraction of the nonprivate_points to add a 0 to the hash_prefix
      **kwargs: extra arguments for the parent class
    """
    super(TestLshTreeNode, self).__init__(*args, **kwargs)
    self.frac_zero = frac_zero

  def get_private_count(self) -> int:
    """Returns a fake private count."""
    self.private_count = len(self.nonprivate_points) + 1
    return self.private_count

  def children(self) -> typing.List[lsh_tree.LshTreeNode]:
    """Returns fake children for this node."""
    cutoff = int(len(self.nonprivate_points) * self.frac_zero)
    return [
        TestLshTreeNode(
            self.hash_prefix + '0',
            self.nonprivate_points[:cutoff],
            self.coreset_param,
            self.sim_hash,
            private_count=cutoff + 1,
            frac_zero=self.frac_zero),
        TestLshTreeNode(
            self.hash_prefix + '1',
            self.nonprivate_points[cutoff:],
            self.coreset_param,
            self.sim_hash,
            private_count=len(self.nonprivate_points) - cutoff + 1,
            frac_zero=self.frac_zero)
    ]

  def __eq__(self, other):
    """Returns whether hash_prefix and private_count are the same for tests."""
    if not isinstance(other, TestLshTreeNode):
      return False
    return self.hash_prefix == other.hash_prefix and (self.private_count
                                                      == other.private_count)


class LshTreeTest(absltest.TestCase):

  @mock.patch.object(stats.dlaplace, 'rvs', return_value=-5, autospec=True)
  def test_get_private_count_basic(self, mock_dlaplace_fn):
    nonprivate_count = 30
    nonprivate_points = get_test_origin_points(
        nonprivate_count=nonprivate_count)
    coreset_param = test_utils.get_test_coreset_param(
        epsilon=5, max_depth=9)
    sim_hash = get_test_sim_hash()
    lsh_tree_node = lsh_tree.LshTreeNode(
        hash_prefix='',
        nonprivate_points=nonprivate_points,
        coreset_param=coreset_param,
        sim_hash=sim_hash)
    self.assertEqual(lsh_tree_node.get_private_count(), 25)
    mock_dlaplace_fn.assert_called_once_with(
        coreset_param.pcalc.count_privacy_param.laplace_param
    )

  def test_get_private_count_cache(self):
    nonprivate_count = 30
    nonprivate_points = get_test_origin_points(
        nonprivate_count=nonprivate_count)
    coreset_param = test_utils.get_test_coreset_param(epsilon=0.01)
    sim_hash = get_test_sim_hash()
    lsh_tree_node = lsh_tree.LshTreeNode(
        hash_prefix='',
        nonprivate_points=nonprivate_points,
        coreset_param=coreset_param,
        sim_hash=sim_hash)

    first_private_count = lsh_tree_node.get_private_count()
    self.assertEqual(first_private_count, lsh_tree_node.get_private_count())

  def test_get_children(self):
    hash_prefix, dim, max_hash_len = '0', 5, 2
    datapoints = np.array([[1.5, 0, 1, 0.5, 0], [1, 1, 0, 0.7, 0.1],
                           [0.8, 0.1, 1, 0.2, 0.4], [0.1, 0.5, 0.3, 0.7, 0.8],
                           [-0.5, 0.1, -0.3, -0.4, 0.2]])
    # Returns children regardless of whether the node should branch. The
    # filtering in the algorithm is done after.
    coreset_param = test_utils.get_test_coreset_param(max_depth=max_hash_len)
    projection_vectors = np.array([[0, 1, 1, -1, 0], [1, 0, -1, 0, 0]])
    sh = lsh.SimHash(dim, max_hash_len, projection_vectors)
    node = lsh_tree.LshTreeNode(hash_prefix, datapoints, coreset_param, sh)
    children = node.children()

    self.assertSameElements([child.hash_prefix for child in children],
                            ['00', '01'])
    for child in children:
      self.assertEqual(child.coreset_param, coreset_param)
      self.assertEqual(child.sim_hash, sh)
      if child.hash_prefix == '00':
        self.assertTrue((child.nonprivate_points == datapoints[[0, 1]]).all())
      if child.hash_prefix == '01':
        self.assertTrue((child.nonprivate_points == datapoints[[2, 3,
                                                                4]]).all())

  def test_get_children_one_empty(self):
    hash_prefix, dim, max_hash_len = '0', 5, 2
    datapoints = np.array([[1.5, 0, 1, 0.5, 0], [1, 1, 0, 0.7, 0.1]])
    # Returns children regardless of whether the node should branch. The
    # filtering in the algorithm is done after.
    coreset_param = test_utils.get_test_coreset_param(max_depth=max_hash_len)
    projection_vectors = np.array([[0, 1, 1, -1, 0], [1, 0, -1, 0, 0]])
    sh = lsh.SimHash(dim, max_hash_len, projection_vectors)
    node = lsh_tree.LshTreeNode(hash_prefix, datapoints, coreset_param, sh)
    children = node.children()

    self.assertSameElements([child.hash_prefix for child in children],
                            ['00', '01'])
    for child in children:
      self.assertEqual(child.coreset_param, coreset_param)
      self.assertEqual(child.sim_hash, sh)
      if child.hash_prefix == '00':
        self.assertTrue((child.nonprivate_points == datapoints[[0, 1]]).all())
      if child.hash_prefix == '01':
        self.assertEmpty(child.nonprivate_points)

  def test_get_children_error(self):
    hash_prefix, dim, max_hash_len = '00', 5, 2
    datapoints = np.array([[1.5, 0, 1, 0.5, 0], [1, 1, 0, 0.7, 0.1]])
    # Returns children regardless of whether the node should branch. The
    # filtering in the algorithm is done after.
    coreset_param = test_utils.get_test_coreset_param(max_depth=max_hash_len)
    projection_vectors = np.array([[0, 1, 1, -1, 0], [1, 0, -1, 0, 0]])
    sh = lsh.SimHash(dim, max_hash_len, projection_vectors)
    node = lsh_tree.LshTreeNode(hash_prefix, datapoints, coreset_param, sh)

    with self.assertRaises(ValueError):
      node.children()

  def test_filter_branching_nodes_too_few_points(self):
    sim_hash = get_test_sim_hash()
    # private_count, not the nonprivate_count, should be used for the check.
    level: lsh_tree.LshTreeLevel = [
        lsh_tree.LshTreeNode(
            '0',
            get_test_origin_points(nonprivate_count=15),
            test_utils.get_test_coreset_param(
                min_num_points_in_branching_node=10),
            sim_hash,
            private_count=1),
    ]
    self.assertEmpty(lsh_tree.LshTree.filter_branching_nodes(level))

  def test_filter_branching_nodes_enough_points(self):
    sim_hash = get_test_sim_hash()
    level: lsh_tree.LshTreeLevel = [
        lsh_tree.LshTreeNode(
            '0',
            get_test_origin_points(nonprivate_count=15),
            test_utils.get_test_coreset_param(
                min_num_points_in_branching_node=10),
            sim_hash,
            private_count=20),
    ]
    self.assertSequenceEqual(
        lsh_tree.LshTree.filter_branching_nodes(level), level)

  def test_get_next_level_empty_list(self):
    self.assertEmpty(lsh_tree.LshTree.get_next_level([]))

  def test_get_next_level(self):
    sim_hash = get_test_sim_hash()
    coreset_param = test_utils.get_test_coreset_param(
        min_num_points_in_branching_node=10, min_num_points_in_node=5)
    level: lsh_tree.LshTreeLevel = [
        TestLshTreeNode(
            '0',
            get_test_origin_points(nonprivate_count=16),
            coreset_param,
            sim_hash,
            private_count=20),
    ]
    expected_next_level = [
        TestLshTreeNode(
            '00',
            get_test_origin_points(nonprivate_count=8),
            coreset_param,
            sim_hash,
            private_count=9),
        TestLshTreeNode(
            '01',
            get_test_origin_points(nonprivate_count=8),
            coreset_param,
            sim_hash,
            private_count=9),
    ]
    branching_nodes = lsh_tree.LshTree.filter_branching_nodes(level)
    self.assertSequenceEqual(
        lsh_tree.LshTree.get_next_level(branching_nodes), expected_next_level)

  def test_get_next_level_filters_children_node(self):
    sim_hash = get_test_sim_hash()
    coreset_param = test_utils.get_test_coreset_param(
        min_num_points_in_branching_node=10, min_num_points_in_node=9)
    level: lsh_tree.LshTreeLevel = [
        # The children test nodes have a private count of 6, which is less than
        # min_num_points_in_node.
        TestLshTreeNode(
            '0',
            get_test_origin_points(nonprivate_count=10),
            coreset_param,
            sim_hash,
            private_count=11),
        # The children test nodes have a private count of 3 and 9, only the node
        # with 9 should be in the result.
        TestLshTreeNode(
            '1',
            get_test_origin_points(nonprivate_count=10),
            coreset_param,
            sim_hash,
            private_count=11,
            frac_zero=0.2),
    ]
    expected_next_level = [
        TestLshTreeNode(
            '11',
            get_test_origin_points(nonprivate_count=8),
            coreset_param,
            sim_hash,
            private_count=9),
    ]
    branching_nodes = lsh_tree.LshTree.filter_branching_nodes(level)
    self.assertSequenceEqual(
        lsh_tree.LshTree.get_next_level(branching_nodes), expected_next_level)

  def test_root_node(self):
    nonprivate_points = [[1, 2, 1], [0.4, 0.2, 0.8], [3, 0, 3]]
    data = clustering_params.Data(nonprivate_points, radius=4.3)
    coreset_param = test_utils.get_test_coreset_param(radius=4.3, max_depth=20)
    root = lsh_tree.root_node(data, coreset_param)
    self.assertEqual(root.hash_prefix, '')
    self.assertSequenceEqual(root.nonprivate_points, nonprivate_points)
    self.assertEqual(root.coreset_param, coreset_param)
    self.assertEqual(root.sim_hash.dim, 3)
    self.assertEqual(root.sim_hash.max_hash_len, 20)
    self.assertIsNotNone(root.private_count)

  def test_root_node_provide_private_count(self):
    nonprivate_points = [[1, 2, 1], [0.4, 0.2, 0.8], [3, 0, 3]]
    data = clustering_params.Data(nonprivate_points, radius=4.3)
    coreset_param = test_utils.get_test_coreset_param(radius=4.3, max_depth=20)
    root = lsh_tree.root_node(data, coreset_param, private_count=10)
    self.assertEqual(root.hash_prefix, '')
    self.assertSequenceEqual(root.nonprivate_points, nonprivate_points)
    self.assertEqual(root.coreset_param, coreset_param)
    self.assertEqual(root.sim_hash.dim, 3)
    self.assertEqual(root.sim_hash.max_hash_len, 20)
    self.assertEqual(root.private_count, 10)

  def test_lsh_tree_empty_root_errors(self):
    test_root = lsh_tree.LshTreeNode(
        '0',
        get_test_origin_points(nonprivate_count=15),
        test_utils.get_test_coreset_param(),
        get_test_sim_hash(),
        private_count=0)
    with self.assertRaises(ValueError):
      lsh_tree.LshTree(test_root)

  def test_lsh_tree_negative_count_root_errors(self):
    test_root = lsh_tree.LshTreeNode(
        '0',
        get_test_origin_points(nonprivate_count=15),
        test_utils.get_test_coreset_param(),
        get_test_sim_hash(),
        private_count=-10)
    with self.assertRaises(ValueError):
      lsh_tree.LshTree(test_root)

  def test_lsh_tree(self):
    # Test tree:
    # Nodes are nonprivate count + 1.
    # Branches to the left are 0, to the right are 1.
    # Nodes in parentheses are filtered out.
    #           64+1
    #          /    \
    #      8+1      56+1
    #     /  \      /   \
    # (1+1)   7+1  7+1   49+1
    #                   /   \
    #                (6+1)  43+1
    nonprivate_count = 64
    sh = get_test_sim_hash()
    cp = test_utils.get_test_coreset_param(
        min_num_points_in_node=8,
        min_num_points_in_branching_node=9,
        max_depth=3)
    test_root = TestLshTreeNode(
        '', get_test_origin_points(nonprivate_count), cp, sh, frac_zero=0.125)
    expected_tree = {
        0: [TestLshTreeNode('', get_test_origin_points(64), cp, sh)],
        1: [
            TestLshTreeNode('0', get_test_origin_points(8), cp, sh),
            TestLshTreeNode('1', get_test_origin_points(56), cp, sh)
        ],
        2: [
            TestLshTreeNode('01', get_test_origin_points(7), cp, sh),
            TestLshTreeNode('10', get_test_origin_points(7), cp, sh),
            TestLshTreeNode('11', get_test_origin_points(49), cp, sh)
        ],
        3: [TestLshTreeNode('111', get_test_origin_points(43), cp, sh)],
    }
    tree = lsh_tree.LshTree(test_root)
    self.assertEqual(tree.tree, expected_tree)

  def test_lsh_tree_branching_node_becomes_leaf(self):
    # Test tree:
    # Nodes are nonprivate count + 1.
    # Branches to the left are 0, to the right are 1.
    # Nodes in parentheses are filtered out.
    #             64+1
    #          /       \
    #      32+1          32+1
    #     /   \         /   \
    # (16+1) (16+1)  (16+1) (16+1)
    nonprivate_count = 64
    sh = get_test_sim_hash()
    cp = test_utils.get_test_coreset_param(
        min_num_points_in_node=20,
        min_num_points_in_branching_node=30,
        max_depth=5)
    test_root = TestLshTreeNode(
        '', get_test_origin_points(nonprivate_count), cp, sh, frac_zero=0.5)
    expected_tree = {
        0: [TestLshTreeNode('', get_test_origin_points(64), cp, sh)],
        1: [
            TestLshTreeNode('0', get_test_origin_points(32), cp, sh),
            TestLshTreeNode('1', get_test_origin_points(32), cp, sh),
        ]
    }
    tree = lsh_tree.LshTree(test_root)
    self.assertEqual(tree.tree, expected_tree)

  def test_lsh_tree_leaves(self):
    # Test tree:
    # Nodes are nonprivate count + 1.
    # Branches to the left are 0, to the right are 1.
    # Nodes in parentheses are filtered out.
    #           64+1
    #          /    \
    #      8+1      56+1
    #     /  \      /   \
    # (1+1)   7+1  7+1   49+1
    #                   /   \
    #                (6+1)  43+1
    nonprivate_count = 64
    sh = get_test_sim_hash()
    cp = test_utils.get_test_coreset_param(
        min_num_points_in_node=8,
        min_num_points_in_branching_node=9,
        max_depth=3)
    test_root = TestLshTreeNode(
        '', get_test_origin_points(nonprivate_count), cp, sh, frac_zero=0.125)
    expected_leaves = [
        TestLshTreeNode('01', get_test_origin_points(7), cp, sh),
        TestLshTreeNode('10', get_test_origin_points(7), cp, sh),
        TestLshTreeNode('111', get_test_origin_points(43), cp, sh)
    ]
    tree = lsh_tree.LshTree(test_root)
    self.assertEqual(tree.leaves, expected_leaves)

  def test_lsh_tree_leaves_branching_node_becomes_leaf(self):
    # Test tree:
    # Nodes are nonprivate count + 1.
    # Branches to the left are 0, to the right are 1.
    # Nodes in parentheses are filtered out.
    #             64+1
    #          /       \
    #      32+1          32+1
    #     /   \         /   \
    # (16+1) (16+1)  (16+1) (16+1)
    nonprivate_count = 64
    sh = get_test_sim_hash()
    cp = test_utils.get_test_coreset_param(
        min_num_points_in_node=20,
        min_num_points_in_branching_node=30,
        max_depth=5)
    test_root = TestLshTreeNode(
        '', get_test_origin_points(nonprivate_count), cp, sh, frac_zero=0.5)
    expected_leaves = [
        TestLshTreeNode('0', get_test_origin_points(32), cp, sh),
        TestLshTreeNode('1', get_test_origin_points(32), cp, sh),
    ]
    tree = lsh_tree.LshTree(test_root)
    self.assertEqual(tree.leaves, expected_leaves)


if __name__ == '__main__':
  absltest.main()
