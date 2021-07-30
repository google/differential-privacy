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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_QUANTILES_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_QUANTILES_H_

#include "absl/status/status.h"
#include "base/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/bounded-algorithm.h"
#include "algorithms/quantile-tree.h"

namespace differential_privacy {

// Calculates multiple differentially private quantiles. Currently implemented
// using the quantile tree mechanism, see quantile-tree.h for more about the
// mechanism. The set of quantiles to be calculated are specified when building
// the algorithm.
//
// Each element of the output represents the value of one of the requested
// quantiles, in the same order as they were requested when building the
// algorithm.
//
// When constructing a MultiQuantile, upper and lower bounds on the input
// must be explicitly specified. MultiQuantile does not support ApproxBounds.
template <typename T>
class Quantiles : public Algorithm<T> {
  static_assert(std::is_arithmetic<T>::value,
                "BoundedSum can only be used for arithmetic types");

 public:
  class Builder;

  void AddEntry(const T& t) override { return tree_->AddEntry(t); }

  Summary Serialize() const override {
    Summary to_return;
    to_return.mutable_data()->PackFrom(tree_->Serialize());
    return to_return;
  }

  absl::Status Merge(const Summary& summary) {
    if (!summary.has_data()) {
      return absl::InternalError(
          "Cannot merge summary with no bounded quantiles data");
    }

    BoundedQuantilesSummary quantiles_summary;
    if (!summary.data().UnpackTo(&quantiles_summary)) {
      return absl::InternalError(
          "Bounded quantiles summary could not be unpacked.");
    }
    return tree_->Merge(quantiles_summary);
  }

  int64_t MemoryUsed() override {
    return tree_->MemoryUsed() + sizeof(Quantiles<T>) +
           sizeof(NumericalMechanismBuilder) +
           sizeof(double) * quantiles_.capacity();
  }

 protected:
  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    typename QuantileTree<T>::DPParams dp_params;
    dp_params.epsilon = Algorithm<T>::GetEpsilon() * privacy_budget;
    dp_params.delta = Algorithm<T>::GetDelta() * privacy_budget;
    dp_params.max_contributions_per_partition =
        max_contributions_per_partition_;
    dp_params.max_partitions_contributed_to = max_partitions_contributed_to_;
    dp_params.mechanism_builder = mechanism_builder_->Clone();
    base::StatusOr<typename QuantileTree<T>::Privatized> result =
        tree_->MakePrivate(dp_params);
    if (!result.ok()) {
      return result.status();
    }
    typename QuantileTree<T>::Privatized privatized_tree =
        std::move(result.value());

    Output output;
    for (double quantile : quantiles_) {
      double result;
      ASSIGN_OR_RETURN(result, privatized_tree.GetQuantile(quantile));
      AddToOutput<double>(&output, result);
    }
    return output;
  }

  void ResetState() override { tree_->Reset(); }

 private:
  Quantiles(std::unique_ptr<QuantileTree<T>> tree,
                std::vector<double> quantiles, double epsilon, double delta,
                int max_contributions_per_partition,
                int max_partitions_contributed_to,
                std::unique_ptr<NumericalMechanismBuilder> mechanism_builder)
      : Algorithm<T>(epsilon, delta),
        tree_(std::move(tree)),
        quantiles_(quantiles),
        max_contributions_per_partition_(max_contributions_per_partition),
        max_partitions_contributed_to_(max_partitions_contributed_to),
        mechanism_builder_(std::move(mechanism_builder)) {}

  std::unique_ptr<QuantileTree<T>> tree_;
  int max_contributions_per_partition_;
  int max_partitions_contributed_to_;
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_;
  std::vector<double> quantiles_;
};

template <typename T>
class Quantiles<T>::Builder
    : public AlgorithmBuilder<T, Quantiles<T>, Quantiles<T>::Builder> {
  using AlgorithmBuilder =
      differential_privacy::AlgorithmBuilder<T, Quantiles<T>,
                                             Quantiles<T>::Builder>;

 public:
  Quantiles<T>::Builder& SetLower(T lower) {
    lower_ = lower;
    return *this;
  }
  Quantiles<T>::Builder& SetUpper(T upper) {
    upper_ = upper;
    return *this;
  }

  // The list of quantiles to be produced. It is required; the algorithm will
  // fail to build without a list of quantiles. If this method is called
  // more than once, it will overwrite any previous list of quantiles rather
  // than appending to it.
  Quantiles<T>::Builder& SetQuantiles(const std::vector<double>& quantiles) {
    quantiles_ = quantiles;
    return *this;
  }

 protected:
  base::StatusOr<std::unique_ptr<Quantiles<T>>> BuildAlgorithm() override {
    typename QuantileTree<T>::Builder tree_builder;

    if (lower_.has_value()) {
      tree_builder.SetLower(lower_.value());
    }
    if (upper_.has_value()) {
      tree_builder.SetUpper(upper_.value());
    }
    std::unique_ptr<QuantileTree<T>> tree;
    ASSIGN_OR_RETURN(tree, tree_builder.Build());

    if (quantiles_.empty()) {
      return absl::InvalidArgumentError(
          "You must specify at least one quantile to calculate.");
    }
    for (double quantile : quantiles_) {
      if (quantile < 0 || quantile > 1) {
        return absl::InvalidArgumentError(
            "All quantiles to calculate must be in [0, 1].");
      }
    }

    // Try building a numerical mechanism so we can return an error now if any
    // parameters are invalid. Otherwise, the error wouldn't be returned until
    // we call MakePrivate in GenerateResult.
    std::unique_ptr<NumericalMechanismBuilder> mech_builder_clone =
        AlgorithmBuilder::GetMechanismBuilderClone();
    if (AlgorithmBuilder::GetEpsilon().has_value()) {
      mech_builder_clone->SetEpsilon(AlgorithmBuilder::GetEpsilon().value());
    }
    if (AlgorithmBuilder::GetDelta().has_value()) {
      mech_builder_clone->SetDelta(AlgorithmBuilder::GetDelta().value());
    }
    mech_builder_clone
        ->SetLInfSensitivity(
            AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1))
        .SetL0Sensitivity(
            AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1) *
            tree->GetHeight());
    RETURN_IF_ERROR(mech_builder_clone->Build().status());

    return std::unique_ptr<Quantiles>(new Quantiles(
        std::move(tree), quantiles_, AlgorithmBuilder::GetEpsilon().value(),
        AlgorithmBuilder::GetDelta().value_or(0),
        AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1),
        AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1),
        AlgorithmBuilder::GetMechanismBuilderClone()));
  }

 private:
  absl::optional<T> lower_;
  absl::optional<T> upper_;
  std::vector<double> quantiles_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_QUANTILES_H_
