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

#include "algorithms/internal/count-tree.h"

#include <cmath>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {
namespace internal {

CountTree::CountTree(int height, int branching_factor)
    : height_(height),
      branching_factor_(branching_factor),
      number_of_nodes_((std::pow(branching_factor_, height_ + 1) - 1) /
                       (branching_factor_ - 1)),
      number_of_leaves_(std::pow(branching_factor_, height_)),
      left_most_leaf_(number_of_nodes_ - number_of_leaves_) {}

int CountTree::GetLeftMostLeaf() const { return left_most_leaf_; }
int CountTree::GetNthLeaf(int n) const { return GetLeftMostLeaf() + n; }
int CountTree::GetNumberOfNodes() const { return number_of_nodes_; }
int CountTree::GetNumberOfLeaves() const { return number_of_leaves_; }
int CountTree::GetBranchingFactor() const { return branching_factor_; }
int CountTree::GetHeight() const { return height_; }
int CountTree::GetRoot() const { return root_node_; }

int CountTree::Parent(int nodeIndex) const {
  return (nodeIndex - 1) / branching_factor_;
}
int CountTree::LeftMostChild(int nodeIndex) const {
  return nodeIndex * branching_factor_ + 1;
}
int CountTree::RightMostChild(int nodeIndex) const {
  return (nodeIndex + 1) * branching_factor_;
}

bool CountTree::IsLeaf(int nodeIndex) const {
  return nodeIndex >= GetLeftMostLeaf() && nodeIndex < GetNumberOfNodes();
}

int CountTree::LeftMostInSubtree(int nodeIndex) const {
  while (!IsLeaf(nodeIndex)) {
    nodeIndex = LeftMostChild(nodeIndex);
  }
  return nodeIndex;
}
int CountTree::RightMostInSubtree(int nodeIndex) const {
  while (!IsLeaf(nodeIndex)) {
    nodeIndex = RightMostChild(nodeIndex);
  }
  return nodeIndex;
}

void CountTree::IncrementNode(int nodeIndex) { ++tree_[nodeIndex]; }
void CountTree::IncrementNodeBy(int nodeIndex, int64_t increment) {
  tree_[nodeIndex] += increment;
}

void CountTree::ClearNodes() { tree_.clear(); }

int64_t CountTree::GetNodeCount(int nodeIndex) const {
  auto node = tree_.find(nodeIndex);
  if (node == tree_.end()) {
    return 0;
  }
  return node->second;
}

BoundedQuantilesSummary CountTree::Serialize() {
  BoundedQuantilesSummary to_return;
  to_return.mutable_quantile_tree()->insert(tree_.begin(), tree_.end());
  to_return.set_tree_height(height_);
  to_return.set_branching_factor(branching_factor_);
  return to_return;
}

absl::Status CountTree::Merge(const BoundedQuantilesSummary& summary) {
  if (!summary.has_tree_height() || !summary.has_branching_factor()) {
    return absl::InternalError(
        "Summary missing height and/or branching factor.");
  }
  if (summary.tree_height() != height_) {
    return absl::InternalError(
        absl::StrCat("Height mismatch. Tree had: ", height_,
                     " but summary had: ", summary.tree_height()));
  }
  if (summary.branching_factor() != branching_factor_) {
    return absl::InternalError(
        absl::StrCat("Branching factor mismatch. Tree had: ", branching_factor_,
                     " but summary had: ", summary.branching_factor()));
  }
  for (std::pair<int32_t, int64_t> node : summary.quantile_tree()) {
    tree_[node.first] += node.second;
  }
  return absl::OkStatus();
}

int64_t CountTree::MemoryUsed() {
  // https://abseil.io/docs/cpp/guides/container#memory-usage
  return sizeof(CountTree) +
         (sizeof(std::pair<int, int64_t>) + 1) * tree_.bucket_count();
}

}  // namespace internal
}  // namespace differential_privacy
