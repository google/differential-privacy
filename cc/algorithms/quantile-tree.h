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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_QUANTILE_TREE_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_QUANTILE_TREE_H_

#include <unordered_map>

#include "absl/status/status.h"
#include "base/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/bounded-algorithm.h"
#include "algorithms/internal/count-tree.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {

// A small tolerance on the quantile we're searching for. We'll be aiming to
// return a value that's within this tolerance of the chosen quantile. This is
// a post-processing parameter with no privacy implications.
constexpr double kNumericalTolerance = 1.0e-6;
// Default tree parameters. Will result in splitting the input space into 16^4
// = 65536 equal buckets. Using a larger height or branching factor will
// split the input space more finely, resulting in greater precision but also
// increasing space used. Increasing the height will increase the amount of
// noise that is added. These parameters were selected based on experiments.
constexpr int kDefaultTreeHeight = 4;
constexpr int kDefaultBranchingFactor = 16;
// Fraction a node needs to contribute to the total count of itself and its
// siblings to be considered during the search for a particular quantile. The
// idea of alpha is to filter out noisy empty nodes. This is a post processing
// parameter with no privacy implications.
constexpr double kAlpha = 0.005;

// Calculates differentially private quantiles using a tree-based data
// structure. See a full writeup of the algorithm at:
// https://github.com/google/differential-privacy/blob/main/common_docs/Differentially_Private_Quantile_Trees.pdf
//
// This algorithm can be used to calculate an arbitrarily large number of
// quantiles with no loss in accuracy or additional expenditure of privacy
// budget.
//
// This is not an Algorithm, and does not behave in the same way as other
// algorithms. For a quantile implementation that follows the Algorithm
// interface, see multi-quantile.h.
template <typename T>
class QuantileTree {
 public:
  class Builder;
  class Privatized;

  void AddEntry(const T& input) { AddMultipleEntries(input, 1); }

  // Removes all input from the QuantileTree. After calling this method, the
  // QuantileTree will be equivalent to one that is newly initialized with no
  // input added.
  void Reset() { tree_.ClearNodes(); }

  struct DPParams {
    double epsilon;
    double delta;
    int max_contributions_per_partition;
    int max_partitions_contributed_to;
    std::unique_ptr<NumericalMechanismBuilder> mechanism_builder;
  };

  // Returns a private version of the quantile tree, which can be used to get
  // differentially private quantiles. Each call to this method expends the
  // epsilon and delta specified in the params.
  base::StatusOr<Privatized> MakePrivate(const DPParams& params) {
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> mech,
        params.mechanism_builder->SetEpsilon(params.epsilon)
            .SetDelta(params.delta)
            .SetL0Sensitivity(params.max_partitions_contributed_to *
                              tree_.GetHeight())
            .SetLInfSensitivity(params.max_contributions_per_partition)
            .Build());
    return Privatized(upper_, lower_, std::move(mech), tree_);
  }

  BoundedQuantilesSummary Serialize() {
    BoundedQuantilesSummary to_return = tree_.Serialize();
    to_return.set_lower(lower_);
    to_return.set_upper(upper_);
    return to_return;
  }

  absl::Status Merge(const BoundedQuantilesSummary& summary) {
    if (static_cast<double>(lower_) != summary.lower() ||
        static_cast<double>(upper_) != summary.upper()) {
      return absl::InternalError(absl::StrCat(
          "Bounds mismatch. Tree: [", lower_, ", ", upper_, "] ",
          ", summary: [", summary.lower(), ", ", summary.upper(), "]"));
    }
    return tree_.Merge(summary);
  }

  int64_t MemoryUsed() {
    return sizeof(QuantileTree) - sizeof(internal::CountTree) +
           tree_.MemoryUsed();
  }

  int GetHeight() { return tree_.GetHeight(); }
  int GetBranchingFactor() { return tree_.GetBranchingFactor(); }

 private:
  QuantileTree(T lower, T upper, int tree_height, int branching_factor)
      : lower_(lower), upper_(upper), tree_(tree_height, branching_factor) {}

  int getLeafIndex(T input) {
    double leaf_fraction =
        static_cast<double>(input - lower_) / (upper_ - lower_);
    return tree_.GetNthLeaf(leaf_fraction * (tree_.GetNumberOfLeaves() - 1));
  }

  void AddMultipleEntries(const T& input, const int64_t times) {
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    if (std::isnan(static_cast<double>(input))) {
      return;
    }
    if (times <= 0) {
      return;
    }
    int currentNode = getLeafIndex(Clamp(lower_, upper_, input));
    while (currentNode > tree_.GetRoot()) {
      tree_.IncrementNodeBy(currentNode, times);
      currentNode = tree_.Parent(currentNode);
    }
  }

  T lower_;
  T upper_;
  internal::CountTree tree_;

  friend class QuantileTreeTestPeer;
};

// A private version of a quantile tree. Used for calculating differentially
// private quantiles. It will contain raw data internally, but only
// differentially private results can be accessed.
template <typename T>
class QuantileTree<T>::Privatized {
 public:
  base::StatusOr<double> GetQuantile(double quantile) {
    if (quantile < 0 || quantile > 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Requested quantile must be in [0, 1] but was ", quantile));
    }

    quantile = ClampQuantile(quantile);

    int current_node = raw_tree_.GetRoot();
    while (!raw_tree_.IsLeaf(current_node)) {
      int left_most_child = raw_tree_.LeftMostChild(current_node);
      int right_most_child = raw_tree_.RightMostChild(current_node);

      double total_count = 0.0;
      for (int i = left_most_child; i <= right_most_child; ++i) {
        total_count += GetNoisedCount(i);
      }

      // All child nodes appear to be empty. No need to continue down the tree.
      if (total_count <= 0) break;

      // Remove nodes that make up less than an alpha fraction of the total -
      // these are likely empty.
      double corrected_total_count = 0.0;
      for (int i = left_most_child; i <= right_most_child; ++i) {
        corrected_total_count +=
            GetNoisedCount(i) >= total_count * kAlpha ? GetNoisedCount(i) : 0.0;
      }

      // All child nodes have a negligible noisy count. We can't tell whether
      // they have any elements in them, and if so how many, so we can stop
      // and pick the middle of this range.
      if (corrected_total_count <= 0) break;

      double partial_count = 0.0;
      for (int i = left_most_child; i <= right_most_child; ++i) {
        double count = GetNoisedCount(i);
        // Ignore nodes we think are empty.
        partial_count += count >= total_count * kAlpha ? count : 0.0;
        if (partial_count / corrected_total_count >=
            quantile - kNumericalTolerance) {
          quantile =
              (quantile - (partial_count - count) / corrected_total_count) /
              (count / corrected_total_count);
          quantile = std::min(std::max(quantile, 0.0), 1.0);
          current_node = i;
          break;
        }
      }
    }

    double to_return = (1 - quantile) * GetSubtreeLowerBound(current_node) +
                       quantile * GetSubtreeUpperBound(current_node);
    return to_return;
  }

 private:
  friend class QuantileTree<T>;

  Privatized(T upper, T lower, std::unique_ptr<NumericalMechanism> mechanism,
             internal::CountTree raw_tree)
      : raw_tree_(raw_tree),
        upper_(upper),
        lower_(lower),
        mechanism_(std::move(mechanism)) {}

  double GetNoisedCount(int index) {
    if (noised_tree_.find(index) == noised_tree_.end()) {
      noised_tree_[index] = mechanism_->AddNoise(raw_tree_.GetNodeCount(index));
    }
    return noised_tree_[index];
  }

  double GetSubtreeLowerBound(int index) {
    int leaf_index =
        raw_tree_.LeftMostInSubtree(index) - raw_tree_.GetLeftMostLeaf();
    double quantile =
        static_cast<double>(leaf_index) / raw_tree_.GetNumberOfLeaves();
    return quantile * upper_ + (1 - quantile) * lower_;
  }

  double GetSubtreeUpperBound(int index) {
    int leaf_index =
        raw_tree_.RightMostInSubtree(index) - raw_tree_.GetLeftMostLeaf() + 1;
    double quantile =
        static_cast<double>(leaf_index) / raw_tree_.GetNumberOfLeaves();
    return quantile * upper_ + (1 - quantile) * lower_;
  }

  // Clamps a quantile to a value between 0.005 and 0.995. This mitigates the
  // inaccuracy of the quantile tree mechanism when finding a quantile close to
  // 0 or 1.
  static double ClampQuantile(double quantile) {
    return std::min(std::max(0.005, quantile), 0.995);
  }

  const T upper_;
  const T lower_;
  std::unique_ptr<NumericalMechanism> mechanism_;
  const internal::CountTree raw_tree_;
  std::unordered_map<int64_t, int64_t> noised_tree_;
};

template <typename T>
class QuantileTree<T>::Builder {
 public:
  Builder& SetTreeHeight(int tree_height) {
    tree_height_ = tree_height;
    return *static_cast<Builder*>(this);
  }

  Builder& SetBranchingFactor(int branching_factor) {
    branching_factor_ = branching_factor;
    return *static_cast<Builder*>(this);
  }

  Builder& SetLower(T lower) {
    lower_ = lower;
    return *static_cast<Builder*>(this);
  }

  Builder& SetUpper(T upper) {
    upper_ = upper;
    return *static_cast<Builder*>(this);
  }

  base::StatusOr<std::unique_ptr<QuantileTree<T>>> Build() {
    if (!tree_height_.has_value()) {
      tree_height_ = kDefaultTreeHeight;
    }
    if (!branching_factor_.has_value()) {
      branching_factor_ = kDefaultBranchingFactor;
    }
    if (!lower_.has_value() || !upper_.has_value()) {
      return absl::InvalidArgumentError(
          "Lower and upper bounds must both be set.");
    }

    if (tree_height_.value() < 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Tree height must be at least 1, but was ", tree_height_.value()));
    }
    if (branching_factor_.value() < 2) {
      return absl::InvalidArgumentError(
          absl::StrCat("Branching factor must be at least 2, but was ",
                       branching_factor_.value()));
    }
    if (lower_.value() >= upper_.value()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Lower bound must be less than upper bound, but lower: ",
                       lower_.value(), " >= upper: ", upper_.value()));
    }

    return std::unique_ptr<QuantileTree>(
        new QuantileTree(lower_.value(), upper_.value(), tree_height_.value(),
                         branching_factor_.value()));
  }

 private:
  absl::optional<int> tree_height_;
  absl::optional<int> branching_factor_;
  absl::optional<T> lower_;
  absl::optional<T> upper_;
};
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_QUANTILE_TREE_H_
