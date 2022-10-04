//
// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_COUNT_TREE_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_COUNT_TREE_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {
namespace internal {

// Maintains a tree of specified height where each node has branching_factor
// children. Each node contains a count, which can be incremented and read out.
// Nodes are identified by index. Nodes are indexed in sequence, starting with
// the root at index 0, and the rightmost leaf node at the maximum index. All of
// a node's children will have sequential indices. Contains numerous methods to
// traverse the tree. Also has the ability to serialize the tree to a proto, and
// to merge the counts from a serialized tree with identical parameters into the
// current tree.
//
// This is used as the underlying data structure for implementing quantile
// trees.
class CountTree {
 public:
  // height is the number of levels in the tree, not including the root.
  // branching_factor is the number of children each node will have.
  CountTree(int height, int branching_factor);

  // Methods for finding a particular node in the tree. Note that the rightmost
  // leaf will be at index LeftMostLeaf() + NumberOfLeaves().
  int GetLeftMostLeaf() const;
  int GetNthLeaf(int n) const;
  int GetBranchingFactor() const;
  int GetHeight() const;
  int GetRoot() const;

  int GetNumberOfNodes() const;
  int GetNumberOfLeaves() const;

  // Methods for navigating the tree from a given node.
  int Parent(int nodeIndex) const;
  int LeftMostChild(int nodeIndex) const;
  int RightMostChild(int nodeIndex) const;
  int LeftMostInSubtree(int nodeIndex) const;
  int RightMostInSubtree(int nodeIndex) const;

  bool IsLeaf(int nodeIndex) const;

  // Modify the count of a specified node.
  void IncrementNode(int nodeIndex);
  void IncrementNodeBy(int nodeIndex, int64_t increment);

  // Sets the counts of all nodes to 0.
  void ClearNodes();

  // Returns the count of a specified node.
  int64_t GetNodeCount(int nodeIndex) const;

  // Serializes the CountTree to a proto representation.
  BoundedQuantilesSummary Serialize();

  // Deserializes the proto representation and combines it with the current
  // CountTree. This will add the counts of each node together.
  // Returns an error if the summary is malformed, or if the parameters (height
  // and branching factor) of the serialized tree do not match.
  absl::Status Merge(const BoundedQuantilesSummary& summary);

  // Returns an estimate of the current memory footprint of the CountTree,
  // in bytes.
  int64_t MemoryUsed();

 private:
  const int height_;
  const int branching_factor_;
  // Quantities are all calculated from height and branching factor. Cached
  // to avoid re-calculation.
  const int number_of_nodes_;
  // The number of nodes in the tree that are in the lowest/largest level and
  // have no children.
  const int number_of_leaves_;
  // The index of the leaf node with the smallest index.
  const int left_most_leaf_;
  // The index of the root.
  static const int root_node_ = 0;
  // We store the tree as an unordered map. This gives fast lookups, and means
  // that we don't need space for empty nodes.
  absl::flat_hash_map<int, int64_t> tree_;
};

}  // namespace internal
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_COUNT_TREE_H_
