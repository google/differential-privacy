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

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "algorithms/algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/quantile-tree.h"
#include "algorithms/util.h"
#include "base/status_macros.h"

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
// When constructing a Quantiles object, upper and lower bounds on the input
// must be explicitly specified. Quantiles do not support ApproxBounds.
template <typename T>
class Quantiles : public Algorithm<T> {
  static_assert(std::is_arithmetic<T>::value,
                "Quantiles can only be used for arithmetic types");

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

  std::vector<double> GetQuantiles() const { return quantiles_; }

 protected:
  absl::StatusOr<Output> GenerateResult(
      double confidence_interval_level) override {
    typename QuantileTree<T>::DPParams dp_params;
    dp_params.epsilon = Algorithm<T>::GetEpsilon();
    dp_params.delta = Algorithm<T>::GetDelta();
    dp_params.max_contributions_per_partition =
        max_contributions_per_partition_;
    dp_params.max_partitions_contributed_to = max_partitions_contributed_to_;
    dp_params.mechanism_builder = mechanism_builder_->Clone();
    absl::StatusOr<typename QuantileTree<T>::Privatized> result =
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
      // Add noise confidence interval.
      absl::StatusOr<ConfidenceInterval> interval =
          privatized_tree.ComputeNoiseConfidenceInterval(
              quantile, confidence_interval_level);

      if (interval.ok()) {
        AddToOutput(&output, result, interval.value());
      } else {
        AddToOutput<double>(&output, result);
      }
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
class Quantiles<T>::Builder {
 public:
  Quantiles<T>::Builder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  Quantiles<T>::Builder& SetDelta(double delta) {
    delta_ = delta;
    return *this;
  }

  Quantiles<T>::Builder& SetMaxPartitionsContributed(
      int max_partitions_contributed) {
    max_partitions_contributed_ = max_partitions_contributed;
    return *this;
  }

  Quantiles<T>::Builder& SetMaxContributionsPerPartition(
      int max_contributions_per_partition) {
    max_contributions_per_partition_ = max_contributions_per_partition;
    return *this;
  }

  Quantiles<T>::Builder& SetLower(T lower) {
    lower_ = lower;
    return *this;
  }

  Quantiles<T>::Builder& SetUpper(T upper) {
    upper_ = upper;
    return *this;
  }

  Quantiles<T>::Builder& SetLaplaceMechanism(
      std::unique_ptr<NumericalMechanismBuilder> builder) {
    mechanism_builder_ = std::move(builder);
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

  absl::StatusOr<std::unique_ptr<Quantiles<T>>> Build() {
    if (!epsilon_.has_value()) {
      epsilon_ = DefaultEpsilon();
      LOG(WARNING) << "Default epsilon of " << epsilon_.value()
                   << " is being used. Consider setting your own epsilon based "
                      "on privacy considerations.";
    }
    RETURN_IF_ERROR(ValidateEpsilon(epsilon_));
    RETURN_IF_ERROR(ValidateDelta(delta_));
    RETURN_IF_ERROR(ValidateBounds(lower_, upper_));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed_));
    RETURN_IF_ERROR(
        ValidateMaxContributionsPerPartition(max_contributions_per_partition_));
    RETURN_IF_ERROR(ValidateQuantiles(quantiles_));

    // Try building a numerical mechanism so we can return an error now if any
    // parameters are invalid. Otherwise, the error wouldn't be returned until
    // we call MakePrivate in GenerateResult.
    RETURN_IF_ERROR(mechanism_builder_->Clone()
                        ->SetEpsilon(epsilon_.value())
                        .SetDelta(delta_)
                        .SetL0Sensitivity(max_partitions_contributed_)
                        .SetLInfSensitivity(max_contributions_per_partition_)
                        .Build()
                        .status());

    // All validation passed; construct quantiles algorithm below.

    typename QuantileTree<T>::Builder tree_builder;
    if (lower_.has_value()) {
      tree_builder.SetLower(lower_.value());
    }
    if (upper_.has_value()) {
      tree_builder.SetUpper(upper_.value());
    }
    ASSIGN_OR_RETURN(std::unique_ptr<QuantileTree<T>> tree,
                     tree_builder.Build());

    return absl::WrapUnique(new Quantiles<T>(
        std::move(tree), quantiles_, epsilon_.value(), delta_,
        max_contributions_per_partition_, max_partitions_contributed_,
        mechanism_builder_->Clone()));
  }

 private:
  std::optional<double> epsilon_;
  double delta_ = 0;
  std::optional<T> upper_;
  std::optional<T> lower_;
  int max_partitions_contributed_ = 1;
  int max_contributions_per_partition_ = 1;
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_ =
      std::make_unique<LaplaceMechanism::Builder>();
  std::vector<double> quantiles_;

  static absl::Status ValidateQuantiles(std::vector<double>& quantiles) {
    if (quantiles.empty()) {
      return absl::InvalidArgumentError(
          "You must specify at least one quantile to calculate.");
    }
    for (double quantile : quantiles) {
      if (quantile < 0 || quantile > 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "All quantiles to calculate must be in [0, 1], but one was: ",
            quantile));
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_QUANTILES_H_
