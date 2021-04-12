//
// Copyright 2019 Google LLC
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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_SUM_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_SUM_H_

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cstdint>
#include "google/protobuf/any.pb.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "base/statusor.h"
#include "absl/strings/str_cat.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounded-algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/util.h"
#include "proto/confidence-interval.pb.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {

template <typename T>
class BoundedSum : public Algorithm<T> {
  static_assert(std::is_arithmetic<T>::value,
                "BoundedSum can only be used for arithmetic types");

 public:
  // Builder class that should be used to construct BoundedSum algorithms.
  class Builder;

  BoundedSum(double epsilon, double delta) : Algorithm<T>(epsilon, delta) {}

  virtual ~BoundedSum() = default;

  // Returns the lower bound when it has been set.
  virtual std::optional<T> lower() const = 0;

  // Returns the upper bound when it has been set.
  virtual std::optional<T> upper() const = 0;

 protected:
  // Check that bounds are appropriate.
  static absl::Status CheckLowerBound(T lower) {
    if (lower < -1 * std::numeric_limits<T>::max()) {
      return absl::InvalidArgumentError(
          "Lower bound cannot be higher in magnitude than the max "
          "numeric limit. If manually bounding, please increase it by "
          "at least 1.");
    }
    return absl::OkStatus();
  }

  // Build a numerical mechanism that will return adequate noise for the raw
  // sum to make the result DP.
  static base::StatusOr<std::unique_ptr<NumericalMechanism>> BuildMechanism(
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      const double epsilon, const double delta, const double l0_sensitivity,
      const double max_contributions_per_partition, const T lower,
      const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetDelta(delta)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(max_contributions_per_partition *
                            std::max(std::abs(lower), std::abs(upper)))
        .Build();
  }
};

// Bounded sum implementation that uses fixed bounds.
template <typename T>
class BoundedSumWithFixedBounds : public BoundedSum<T> {
 public:
  BoundedSumWithFixedBounds(const double epsilon, const double delta,
                            const T lower, const T upper,
                            std::unique_ptr<NumericalMechanism> mechanism)
      : BoundedSum<T>(epsilon, delta),
        lower_(lower),
        upper_(upper),
        mechanism_(std::move(mechanism)) {}

  void AddEntry(const T& t) override {
    if (std::isnan(static_cast<double>(t))) {
      return;
    }
    partial_sum_ += Clamp<T>(lower_, upper_, t);
  }

  Summary Serialize() const override {
    BoundedSumSummary sum_summary;
    // TODO: Use the partial_sum field of the proto.
    SetValue(sum_summary.add_pos_sum(), partial_sum_);

    Summary result;
    result.mutable_data()->PackFrom(sum_summary);
    return result;
  }

  absl::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return absl::InternalError("Cannot merge summary with no data.");
    }

    // Unpack sum summary
    BoundedSumSummary sum_summary;
    if (!summary.data().UnpackTo(&sum_summary)) {
      return absl::InternalError("Bounded sum summary unable to be unpacked.");
    }

    // Get required partial sum
    // TODO: Use the partial_sum field of the proto.
    if (sum_summary.pos_sum_size() != 1) {
      return absl::InternalError(absl::StrCat(
          "Bounded sum summary must have exactly one pos_sum but got ",
          sum_summary.pos_sum_size()));
    }
    partial_sum_ += GetValue<T>(sum_summary.pos_sum(0));

    return absl::Status();
  }

  int64_t MemoryUsed() override {
    return sizeof(BoundedSumWithFixedBounds) + mechanism_->MemoryUsed();
  }

  std::optional<T> upper() const override { return upper_; }

  std::optional<T> lower() const override { return lower_; }

  base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget = 1) override {
    return mechanism_->NoiseConfidenceInterval(confidence_level,
                                               privacy_budget);
  }

 protected:
  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    RETURN_IF_ERROR(ValidateIsPositive(privacy_budget, "Privacy budget",
                                       absl::StatusCode::kFailedPrecondition));

    Output output;

    // Add noise to the sum.
    double noisy_sum = mechanism_->AddNoise(partial_sum_, privacy_budget);
    if (std::is_integral<T>::value) {
      SafeOpResult<T> cast_result =
          SafeCastFromDouble<T>(std::round(noisy_sum));
      AddToOutput<T>(&output, cast_result.value);
    } else {
      AddToOutput<T>(&output, noisy_sum);
    }

    // Add noise confidence interval.
    base::StatusOr<ConfidenceInterval> interval =
        NoiseConfidenceInterval(noise_interval_level, privacy_budget);
    if (interval.ok()) {
      output.mutable_error_report()->set_allocated_noise_confidence_interval(
          new ConfidenceInterval(*interval));
    }

    return output;
  }

  void ResetState() override { partial_sum_ = 0; }

 private:
  // Bounds
  const T lower_;
  const T upper_;

  // (Partially) aggregated sum
  T partial_sum_ = 0;

  // Mechanism to add noise.
  std::unique_ptr<NumericalMechanism> mechanism_;
};

// Bounded sum implementation using privately inferred bounds as a single-pass
// algorithm using ApproxBounds.
template <typename T>
class BoundedSumWithApproxBounds : public BoundedSum<T> {
 public:
  BoundedSumWithApproxBounds(
      const double epsilon, const double delta, const double l0_sensitivity,
      const double max_contributions_per_partition,
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      std::unique_ptr<ApproxBounds<T>> approx_bounds)
      : BoundedSum<T>(epsilon, delta),
        mechanism_builder_(std::move(mechanism_builder)),
        l0_sensitivity_(l0_sensitivity),
        max_contributions_per_partition_(max_contributions_per_partition),
        approx_bounds_(std::move(approx_bounds)) {
    // We use partial values for each bin of the ApproxBounds logarithmic
    // histogram.
    pos_sum_.resize(approx_bounds_->NumPositiveBins(), 0);
    neg_sum_.resize(approx_bounds_->NumPositiveBins(), 0);
  }

  void AddEntry(const T& t) override {
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    if (std::isnan(static_cast<double>(t))) {
      return;
    }

    approx_bounds_->AddEntry(t);

    // Find partial sums.
    if (t >= 0) {
      approx_bounds_->template AddToPartialSums<T>(&pos_sum_, t);
    } else {
      approx_bounds_->template AddToPartialSums<T>(&neg_sum_, t);
    }
  }

  // Noise confidence interval is not known before finalizing the algorithm as
  // we are using approx bounds.
  base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget = 1) override {
    return absl::InvalidArgumentError(
        "NoiseConfidenceInterval changes per result generation for "
        "automatically-determined sensitivity.");
  }

  std::optional<T> lower() const override { return std::nullopt; }
  std::optional<T> upper() const override { return std::nullopt; }

  Summary Serialize() const override {
    // Create BoundedSumSummary.
    BoundedSumSummary bs_summary;
    for (T x : pos_sum_) {
      SetValue(bs_summary.add_pos_sum(), x);
    }
    for (T x : neg_sum_) {
      SetValue(bs_summary.add_neg_sum(), x);
    }
    Summary approx_bounds_summary = approx_bounds_->Serialize();
    approx_bounds_summary.data().UnpackTo(bs_summary.mutable_bounds_summary());

    // Create Summary.
    Summary summary;
    summary.mutable_data()->PackFrom(bs_summary);
    return summary;
  }

  absl::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return absl::InternalError(
          "Cannot merge summary with no bounded sum data.");
    }

    // Add bounded sum partial values.
    BoundedSumSummary bs_summary;
    if (!summary.data().UnpackTo(&bs_summary)) {
      return absl::InternalError("Bounded sum summary unable to be unpacked.");
    }
    if (pos_sum_.size() != bs_summary.pos_sum_size() ||
        neg_sum_.size() != bs_summary.neg_sum_size()) {
      return absl::InternalError(
          "Merged BoundedSum must have the same amount of partial sum "
          "values as this BoundedSum.");
    }
    for (int i = 0; i < pos_sum_.size(); ++i) {
      pos_sum_[i] += GetValue<T>(bs_summary.pos_sum(i));
    }
    for (int i = 0; i < neg_sum_.size(); ++i) {
      neg_sum_[i] += GetValue<T>(bs_summary.neg_sum(i));
    }

    // Merge approx bounds summary.
    Summary approx_bounds_summary;
    approx_bounds_summary.mutable_data()->PackFrom(bs_summary.bounds_summary());
    RETURN_IF_ERROR(approx_bounds_->Merge(approx_bounds_summary));

    return absl::OkStatus();
  }

  // Returns the total epsilon used by this single-pass algorithm that uses
  // approx bounds internally.
  double GetEpsilon() const override {
    return approx_bounds_->GetEpsilon() + Algorithm<T>::GetEpsilon();
  }

  // Returns the epsilon used to calculate approximate bounds.
  double GetBoundingEpsilon() const { return approx_bounds_->GetEpsilon(); }

  // Returns the epsilon used to calculate the noisy sum.  The overall algorithm
  // also uses epsilon for privately inferred bounds using approx bounds.
  double GetAggregationEpsilon() const { return Algorithm<T>::GetEpsilon(); }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BoundedSum<T>);
    memory += sizeof(T) * (pos_sum_.capacity() + neg_sum_.capacity());
    memory += approx_bounds_->MemoryUsed();
    memory += sizeof(*mechanism_builder_);
    return memory;
  }

 protected:
  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    RETURN_IF_ERROR(ValidateIsPositive(privacy_budget, "Privacy budget",
                                       absl::StatusCode::kFailedPrecondition));
    Output output;

    // Use a fraction of the privacy budget to find the approximate bounds.
    const double bounds_budget = privacy_budget / 2;
    const double remaining_budget = privacy_budget - bounds_budget;

    // Get results of approximate bounds.
    ASSIGN_OR_RETURN(Output bounds, approx_bounds_->PartialResult(
                                        bounds_budget, noise_interval_level));
    const T approx_bounds_lower = GetValue<T>(bounds.elements(0).value());
    const T approx_bounds_upper = GetValue<T>(bounds.elements(1).value());
    RETURN_IF_ERROR(BoundedSum<T>::CheckLowerBound(approx_bounds_lower));

    // Since sensitivity is determined only by the larger-magnitude bound,
    // set the smaller-magnitude bound to be the negative of the larger. This
    // minimizes clamping and so maximizes accuracy.
    const T lower = std::min(approx_bounds_lower, -1 * approx_bounds_upper);
    const T upper = std::max(approx_bounds_upper, -1 * approx_bounds_lower);

    // Populate the bounding report with ApproxBounds information.
    output.mutable_error_report()->set_allocated_bounding_report(
        new BoundingReport(approx_bounds_->GetBoundingReport(lower, upper)));

    // Construct NumericalMechanism.
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> mechanism,
        BoundedSum<T>::BuildMechanism(
            mechanism_builder_->Clone(), Algorithm<T>::GetEpsilon(),
            Algorithm<T>::GetDelta(), l0_sensitivity_,
            max_contributions_per_partition_, lower, upper));

    // To find the sum, pass the identity function as the transform. We pass
    // count = 0 because the count should never be used.
    ASSIGN_OR_RETURN(
        T sum, approx_bounds_->template ComputeFromPartials<T>(
                   pos_sum_, neg_sum_, [](T x) { return x; }, lower, upper, 0));

    // Add noise to sum. Use the remaining privacy budget.
    T noisy_sum = mechanism->AddNoise(sum, remaining_budget);
    AddToOutput<T>(&output, noisy_sum);

    // Add noise confidence interval to the error report.
    base::StatusOr<ConfidenceInterval> interval =
        mechanism->NoiseConfidenceInterval(noise_interval_level,
                                           remaining_budget);
    if (interval.ok()) {
      output.mutable_error_report()->set_allocated_noise_confidence_interval(
          new ConfidenceInterval(*interval));
    }

    return output;
  }

  void ResetState() override {
    std::fill(pos_sum_.begin(), pos_sum_.end(), 0);
    std::fill(neg_sum_.begin(), neg_sum_.end(), 0);
    approx_bounds_->Reset();
  }

 private:
  // Vectors of partial values stored for automatic clamping.
  std::vector<T> pos_sum_, neg_sum_;

  // Used to construct the numerical mechanism once bounds are obtained.
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_;
  const double l0_sensitivity_;
  const int max_contributions_per_partition_;

  // Algorithm to privately infer bounds.
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;
};

template <typename T>
class BoundedSum<T>::Builder
    : public BoundedAlgorithmBuilder<T, BoundedSum<T>, BoundedSum<T>::Builder> {
 private:
  using AlgorithmBuilder =
      differential_privacy::AlgorithmBuilder<T, BoundedSum<T>,
                                             BoundedSum<T>::Builder>;
  using BoundedBuilder =
      BoundedAlgorithmBuilder<T, BoundedSum<T>, BoundedSum<T>::Builder>;
  base::StatusOr<std::unique_ptr<BoundedSum<T>>> BuildBoundedAlgorithm()
      override {
    // We have to check epsilon now, otherwise the split during ApproxBounds
    // construction might make the error message confusing.
    RETURN_IF_ERROR(
        ValidateIsFiniteAndPositive(AlgorithmBuilder::GetEpsilon(), "Epsilon"));

    // Ensure that either bounds are manually set or ApproxBounds is made.
    RETURN_IF_ERROR(BoundedBuilder::BoundsSetup());

    if (BoundedBuilder::BoundsAreSet()) {
      // Construct mechanism directly so we can fail on build if sensitivity is
      // inappropriate.
      RETURN_IF_ERROR(CheckLowerBound(BoundedBuilder::GetLower().value()));

      const double epsilon = BoundedBuilder::GetEpsilon().value();
      const double delta = BoundedBuilder::GetDelta().value_or(0);
      const int max_partitions_contributed =
          AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1);
      const int max_contributions_per_partition =
          AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1);
      const T lower = BoundedBuilder::GetLower().value();
      const T upper = BoundedBuilder::GetUpper().value();

      ASSIGN_OR_RETURN(
          std::unique_ptr<NumericalMechanism> mechanism,
          BuildMechanism(AlgorithmBuilder::GetMechanismBuilderClone(), epsilon,
                         delta, max_partitions_contributed,
                         max_contributions_per_partition, lower, upper));

      // Construct BoundedSum with fixed bounds.
      return std::unique_ptr<BoundedSum<T>>(new BoundedSumWithFixedBounds<T>(
          epsilon, delta, lower, upper, std::move(mechanism)));
    }

    // Construct BoundedSum with approx bounds
    return std::unique_ptr<BoundedSum<T>>(new BoundedSumWithApproxBounds<T>(
        BoundedBuilder::GetRemainingEpsilon().value(),
        BoundedBuilder::GetDelta().value_or(0),
        AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1),
        AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1),
        AlgorithmBuilder::GetMechanismBuilderClone(),
        BoundedBuilder::MoveApproxBoundsPointer()));
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_SUM_H_
