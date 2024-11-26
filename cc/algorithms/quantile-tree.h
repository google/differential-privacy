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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "algorithms/internal/count-tree.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/confidence-interval.pb.h"
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
  absl::StatusOr<Privatized> MakePrivate(const DPParams& params) {
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
  absl::StatusOr<double> GetQuantile(double quantile) {
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

  absl::StatusOr<ConfidenceInterval> ComputeNoiseConfidenceInterval(
      double quantile, double confidence_interval_level) {
    if (quantile < 0 || quantile > 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Quantile must be in [0, 1], but was ", quantile));
    }

    quantile = ClampQuantile(quantile);

    ASSIGN_OR_RETURN(double lower_bound,
                     ComputeNoiseConfidenceIntervalBound(
                         quantile, confidence_interval_level,
                         ConfidenceIntervalBoundType::LOWER));
    ASSIGN_OR_RETURN(double upper_bound,
                     ComputeNoiseConfidenceIntervalBound(
                         quantile, confidence_interval_level,
                         ConfidenceIntervalBoundType::UPPER));

    ConfidenceInterval confidence_interval;
    confidence_interval.set_lower_bound(lower_bound);
    confidence_interval.set_upper_bound(upper_bound);
    confidence_interval.set_confidence_level(confidence_interval_level);

    return confidence_interval;
  }

 private:
  friend class QuantileTree<T>;

  Privatized(T upper, T lower, std::unique_ptr<NumericalMechanism> mechanism,
             const internal::CountTree& raw_tree)
      : raw_tree_(raw_tree),
        upper_(upper),
        lower_(lower),
        mechanism_(std::move(mechanism)) {}

  // These are used by the confidence interval computation algorithm.
  enum class ConfidenceIntervalBoundType { LOWER, UPPER };
  struct IndexAndQuantile {
    int index;
    double quantile;
    IndexAndQuantile(int initial_index, double initial_quantile)
        : index(initial_index), quantile(initial_quantile) {}
  };

  // Returns a pair of the index of the child node visited next in the quantile
  // search together with the rank for the next iteration of the search. If all
  // child nodes are considered empty, null is returned.
  IndexAndQuantile GetNextIndexAndQuantile(
      double quantile, int leftmost_child_index, int rightmost_child_index,
      const std::unordered_map<int, double>& node_counts) {
    double total_count = 0.0;
    for (int i = leftmost_child_index; i <= rightmost_child_index; i++) {
      total_count += std::max(0.0, node_counts.at(i));
    }

    double corrected_total_count = 0.0;
    for (int i = leftmost_child_index; i <= rightmost_child_index; i++) {
      // Treat child nodes contributing less than a gamma fraction to the total
      // count as empty subtrees.
      double node_count_i = node_counts.at(i);
      corrected_total_count +=
          node_count_i >= total_count * kAlpha ? node_count_i : 0.0;
    }
    if (corrected_total_count == 0.0) {
      // Either all counts are 0.0 or no child node contributes more than an
      // alpha fraction to the total count (the latter can only happen when
      // alpha > 1 / branching factor, which is not the case for the default
      // branching factor). This means that all child nodes are considered
      // empty.
      return IndexAndQuantile(-1, -1);
    }

    // Determine the child node whose subtree contains the bound.
    double partial_count = 0.0;
    for (int i = leftmost_child_index; true; i++) {
      double count = node_counts.at(i);
      // Skip child nodes contributing less than gamma to the total count.
      if (count < total_count * kAlpha) {
        continue;
      }

      partial_count += count;
      // Check if the bound is in the current child's subtree.
      if (partial_count / corrected_total_count <
          quantile - kNumericalTolerance) {
        continue;
      }

      double next_quantile =
          (quantile - (partial_count - count) / corrected_total_count) /
          (count / corrected_total_count);
      // Clamping rank to a value between 0.0 and 1.0. Note that rank can
      // become greater than 1 because of the numerical tolerance. Values
      // less than 0.0 should not occur. The respective clamping is set in
      // place to be on the safe side.
      next_quantile = std::min(std::max(0.0, next_quantile), 1.0);
      return IndexAndQuantile(i, next_quantile);
    }
  }

  // The following computation of a lower or upper interval bound is based on
  // the same search algorithm used to compute the respective quantile. The
  // difference is that instead of using the noised node counts to determine the
  // direction of the search, the algorithm uses the confidence intervals bounds
  // of the node counts.
  absl::StatusOr<double> ComputeNoiseConfidenceIntervalBound(
      double quantile, double confidence_interval_level,
      const ConfidenceIntervalBoundType& bound_type) {
    // Let b be the branching factor and h the height of the tree. The search
    // for a quantile queries at most b * h node counts. Assigning a confidence
    // interval with error probability
    //    alpha_per_count = 1-confidence_interval_level^(1 / (b * h))
    // to each of these counts guarantees that the true counts are contained
    // within these confidence intervals with error probability
    //    1 - (1 - alpha_per_count)^(b * h)
    //  = 1 - (1 - (1 - (confidence_interval_level)^(1 / (b * h))))^(b * h)
    //  = 1 - confidence_interval_level,
    // which matches the specified error probability.
    double alpha_per_count =
        1 - std::pow(
                confidence_interval_level,
                1.0 / (raw_tree_.GetBranchingFactor() * raw_tree_.GetHeight()));

    // Confidence interval of a node count of 0 with error probability
    // confidence_level_per_count. All other node count confidence intervals are
    // computed by shifting this interval, which is faster than calling
    // ComputeConfidenceInterval() for each node count individually.
    // privacy_budget is set to the maximum of 1.0 because computing the noise
    // confidence intervals for quantiles does not consume any privacy budget.
    ASSIGN_OR_RETURN(ConfidenceInterval zero_confidence_interval,
                     mechanism_->NoiseConfidenceInterval(1 - alpha_per_count));

    // Value of the bound that is being computed. The value is set to the
    // tightest bound possible and loosened successively as needed.
    double bound =
        bound_type == ConfidenceIntervalBoundType::LOWER ? upper_ : lower_;

    int index = raw_tree_.GetRoot();
    // Search for the index of the leaf node containing the desired bound,
    // starting at the root.
    while (index < raw_tree_.GetLeftMostLeaf()) {
      int leftmost_child_index = raw_tree_.LeftMostChild(index);
      int rightmost_child_index = raw_tree_.RightMostChild(index);

      // Index of the node visited next in the search. The value is set to the
      // tightest index possible and loosened successively as needed.
      int next_index = bound_type == ConfidenceIntervalBoundType::LOWER
                           ? std::numeric_limits<int>::max()
                           : std::numeric_limits<int>::min();
      // Quantile used in the next iteration of the search. The value will be
      // set with the first update of next_index.
      double next_quantile = -1.0;

      std::unordered_map<int, ConfidenceInterval> child_confidence_intervals;
      for (int i = leftmost_child_index; i <= rightmost_child_index; i++) {
        ConfidenceInterval ci;
        ci.set_lower_bound(noised_tree_[i] +
                           zero_confidence_interval.lower_bound());
        ci.set_upper_bound(noised_tree_[i] +
                           zero_confidence_interval.upper_bound());
        child_confidence_intervals[i] = ci;
      }

      // Let [l_i, u_i] denote the confidence interval of child node i. To find
      // a lower bound b for the quantiles that can be reached via a particular
      // configuration of counts c_i such that l_i ≤ c_i ≤ u_i, the counts to
      // left of b should be as large as possible while the counts to the right
      // of b should be as small as possible. Thus, we set
      //    c_i = u_i if i <= j and c_i = l_i if i > j
      // for some index j. Similarly, an upper bound can be obtained by setting
      //    c_i = l_i if i <= j and c_i = u_i if i > j.
      //
      // Because we don't know the index j in advance, we go through all
      // possible indices j and pick whichever yields the smallest lower bound
      // or largest upper bound.
      for (int j = leftmost_child_index - 1; j <= rightmost_child_index; j++) {
        std::unordered_map<int, double> count_bounds;
        for (int i = leftmost_child_index; i <= j; i++) {
          count_bounds[i] = bound_type == ConfidenceIntervalBoundType::LOWER
                                ? child_confidence_intervals[i].upper_bound()
                                : child_confidence_intervals[i].lower_bound();
        }
        for (int i = j + 1; i <= rightmost_child_index; i++) {
          count_bounds[i] = bound_type == ConfidenceIntervalBoundType::LOWER
                                ? child_confidence_intervals[i].lower_bound()
                                : child_confidence_intervals[i].upper_bound();
        }

        IndexAndQuantile next_index_and_quantile =
            GetNextIndexAndQuantile(quantile, leftmost_child_index,
                                    rightmost_child_index, count_bounds);

        if (next_index_and_quantile.index == -1) {
          // All child nodes are considred empty. Update the bound with a linear
          // interpolation of the smallest and largest value associated with the
          // current node if the result yields a looser bound.
          if (bound_type == ConfidenceIntervalBoundType::LOWER) {
            bound =
                std::min(bound, (1 - quantile) * GetSubtreeLowerBound(index) +
                                    quantile * GetSubtreeUpperBound(index));
          } else {
            bound =
                std::max(bound, (1 - quantile) * GetSubtreeLowerBound(index) +
                                    quantile * GetSubtreeUpperBound(index));
          }
        } else {
          // Update nextIndex and nextRank if this results in a looser bound.
          if ((bound_type == ConfidenceIntervalBoundType::LOWER &&
               next_index_and_quantile.index <= next_index) ||
              (bound_type == ConfidenceIntervalBoundType::UPPER &&
               next_index_and_quantile.index >= next_index)) {
            if (next_index_and_quantile.index != next_index) {
              next_index = next_index_and_quantile.index;
              next_quantile = next_index_and_quantile.quantile;
            } else if ((bound_type == ConfidenceIntervalBoundType::LOWER &&
                        next_index_and_quantile.quantile < next_quantile) ||
                       (bound_type == ConfidenceIntervalBoundType::UPPER &&
                        next_index_and_quantile.quantile > next_quantile)) {
              next_quantile = next_index_and_quantile.quantile;
            }
          }
        }
      }

      // Check if the current node was considered empty for all values of j
      // (this is the case when nextRank is still its initial invalid value of
      // -1). If so, the search can be stopped and the bound returned.
      // Otherwise, continue the search in the next node with the respective
      // new rank.
      if (next_quantile == -1) {
        return bound;
      }
      index = next_index;
      quantile = next_quantile;
    }
    // The search has reached a leaf node. In this case, we either return a
    // linear interpolation between the smallest and the largest value
    // associated with the leaf node, or the bound computed so far should it be
    // looser.
    double linear_interpolation = (1 - quantile) * GetSubtreeLowerBound(index) +
                                  quantile * GetSubtreeUpperBound(index);
    return bound_type == ConfidenceIntervalBoundType::LOWER
               ? std::min(bound, linear_interpolation)
               : std::max(bound, linear_interpolation);
  }

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
  Builder() = default;

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

  absl::StatusOr<std::unique_ptr<QuantileTree<T>>> Build() {
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

    // Ensure (upper - lower) does not overflow.
    if constexpr (std::is_floating_point_v<T>) {
      if (!std::isfinite(upper_.value() - lower_.value())) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Upper and lower bounds failed floating point overflow check: "
            "upper - lower must be finite, but is ",
            upper_.value() - lower_.value()));
      }
    } else {
      if (lower_.value() < 0 &&
          upper_.value() > std::numeric_limits<T>::max() + lower_.value()) {
        return absl::InvalidArgumentError(
            "Upper and lower bounds failed signed integer overflow check for "
            "upper - lower");
      }
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
  std::optional<int> tree_height_;
  std::optional<int> branching_factor_;
  std::optional<T> lower_;
  std::optional<T> upper_;
};
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_QUANTILE_TREE_H_
