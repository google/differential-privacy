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
"""LSH tree for hierarchically grouping nearby points."""

import dataclasses
import typing

from absl import logging
import numpy as np

from clustering import central_privacy_utils
from clustering import clustering_params
from clustering import coreset_params
from clustering import lsh

HashPrefix = str


@dataclasses.dataclass
class LshTreeNode():
  """Node in a LSH tree corresponding to a single hash prefix.

  Attributes:
    hash_prefix: Hash prefix represented by this node.
    nonprivate_points: Points that hash to hash_prefix.
    coreset_param: Clustering params used for constructing this node.
    sim_hash: LSH used for generating the hashes.
    private_count: Private count of the points in nonprivate_points.
    private_average: Private average of the points in nonprivate_points if
      get_private_average has been called in the past, otherwise None.
  """
  hash_prefix: HashPrefix
  nonprivate_points: np.ndarray
  coreset_param: coreset_params.CoresetParam
  sim_hash: lsh.SimHash
  private_count: typing.Optional[int] = None
  private_average: typing.Optional[np.ndarray] = dataclasses.field(
      init=False, default=None)

  def __post_init__(self):
    if self.private_count is None:
      self.get_private_count()

  def get_private_average(self) -> np.ndarray:
    """Returns and saves private average of the points in the node.

    Requires that self.private_count >= 1.
    """
    # Reuse old results if they've been computed in the past.
    if self.private_average is not None:
      return self.private_average

    self.private_average = central_privacy_utils.get_private_average(
        self.nonprivate_points, self.private_count,
        self.coreset_param.pcalc.average_privacy_param, self.sim_hash.dim)
    return self.private_average

  def get_private_count(self) -> int:
    """Returns and saves private count of the points in the node."""
    if self.private_count is not None:
      return self.private_count

    self.private_count = central_privacy_utils.get_private_count(
        len(self.nonprivate_points),
        self.coreset_param.pcalc.count_privacy_param)
    return self.private_count

  def children(self) -> typing.List["LshTreeNode"]:
    """Returns the children for this node.

    There is a child for every hash_prefix equal to self.hash_prefix with one
    more hash character. Note that children are returned regardless of
    self.coreset_param.tree_param.
    """
    next_hash_char_to_points = self.sim_hash.group_by_next_hash(
        self.nonprivate_points, hash_prefix=self.hash_prefix)
    return [
        LshTreeNode(self.hash_prefix + next_hash_char,
                    nonprivate_points_with_hash_char, self.coreset_param,
                    self.sim_hash) for next_hash_char,
        nonprivate_points_with_hash_char in next_hash_char_to_points.items()
    ]

  def __repr__(self) -> str:
    """Represents nodes in the form of private_count(hash_prefix)."""
    return str(self.private_count) + "(" + self.hash_prefix + ")"


# List of leaves in the LSH tree.
LshTreeLeaves = typing.List[LshTreeNode]
# Index of a particular level of the tree
LevelIndex = int
# Nodes on one levels of the tree
LshTreeLevel = typing.List[LshTreeNode]
# Subset of a level including just the nodes that should be branched.
NodesToBranch = LshTreeLevel


def root_node(data: clustering_params.Data,
              coreset_param: coreset_params.CoresetParam,
              private_count: typing.Optional[int] = None):
  """Returns root node for an LSH prefix tree.

  Args:
    data: Data to use for generating the tree.
    coreset_param: Clustering parameters to use for generating the tree.
    private_count: Private count for the number of datapoints. If None, the
      private count will be computed.
  """
  sim_hash = lsh.SimHash(data.dim, coreset_param.tree_param.max_depth)
  return LshTreeNode(
      "", data.datapoints, coreset_param, sim_hash, private_count=private_count)


class LshTree():
  """Tree in which the data is split into groups based on prefixes from the LSH values.

  Attributes:
    tree: Maps level indices to each level.
    leaves: Leaf nodes of the tree.
  """
  tree: typing.Dict[LevelIndex, LshTreeLevel]
  leaves: LshTreeLeaves

  def __init__(self, root: LshTreeNode):
    """Initializes an LshTree with the given root.

    Args:
      root: Root to use for the LshTree. Required to have private count >= 1.
    """
    if root.private_count < 1:
      raise ValueError("Private count of the root must be at least 1.")
    coreset_param = root.coreset_param
    logging.debug("Starting tree construction with max_levels %s",
                  coreset_param.tree_param.max_depth)
    level_idx: LevelIndex = 0
    self.tree: typing.Dict[LevelIndex, LshTreeLevel] = dict()
    self.tree[level_idx] = [root]

    while level_idx < coreset_param.tree_param.max_depth:
      # Branch all the nodes that should be branched
      branching_nodes: NodesToBranch = LshTree.filter_branching_nodes(
          self.tree[level_idx])
      next_level = LshTree.get_next_level(branching_nodes)
      if next_level:
        level_idx += 1
        self.tree[level_idx] = next_level
      else:
        break
    logging.debug("Tree generated (level -> nodes): %s", self.tree)

    logging.debug("Starting to collect the leaves of the tree.")
    self.leaves = []
    for level_idx in self.tree:
      self.leaves.extend(list(filter(self.is_leaf, self.tree[level_idx])))
    logging.debug("Found %s leaves: %s", len(self.leaves), self.leaves)

  def is_leaf(self, node: LshTreeNode) -> bool:
    """Returns whether the node is a leaf.

    Args:
      node: LshTreeNode in this tree to check whether it has any children.
    """
    level_below = len(node.hash_prefix) + 1

    # If node is in the last level, it is a leaf.
    if level_below > max(self.tree.keys()):
      return True

    for maybe_child in self.tree.get(level_below):
      # Each level adds one character to the hash prefix.
      if node.hash_prefix == maybe_child.hash_prefix[:-1]:
        return False

    # No children were found, so this node is a leaf.
    return True

  @staticmethod
  def filter_branching_nodes(tree_level: LshTreeLevel) -> NodesToBranch:
    """Returns the nodes in tree_level that have enough points to branch.

    Args:
      tree_level: A level of the tree.
    """

    def enough_points_to_branch(node: LshTreeNode):
      tree_param = node.coreset_param.tree_param
      return node.private_count >= tree_param.min_num_points_in_branching_node

    return list(filter(enough_points_to_branch, tree_level))

  @staticmethod
  def get_next_level(nodes_to_branch: NodesToBranch) -> LshTreeLevel:
    """Returns the next level of the tree based on nodes_to_branch.

    Args:
      nodes_to_branch: Nodes to branch for getting the next level in the tree.
    """
    flatten_children = []
    for node in nodes_to_branch:
      flatten_children.extend(node.children())

    def enough_points(node: LshTreeNode):
      tree_param = node.coreset_param.tree_param
      return node.private_count >= tree_param.min_num_points_in_node

    return list(filter(enough_points, flatten_children))
