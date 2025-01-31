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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds-as-bounds-provider.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounds-provider.h"
#include "algorithms/internal/clamped-calculation-without-bounds.h"
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
  static_assert(std::is_arithmetic<T>(),
                "BoundedSum can only be used for arithmetic types");
  static_assert(std::is_signed<T>(),
                "BoundedSum can only be used for signed types");

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
  static absl::StatusOr<std::unique_ptr<NumericalMechanism>> BuildMechanism(
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      const double epsilon, const double delta, const double l0_sensitivity,
      const double max_contributions_per_partition, const T lower,
      const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetDelta(delta)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(max_contributions_per_partition *
                            std::max(std::abs(static_cast<double>(lower)),
                                     std::abs(static_cast<double>(upper))))
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

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) override {
    return mechanism_->NoiseConfidenceInterval(confidence_level);
  }

 protected:
  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    Output output;

    // Add noise to the sum.
    double noisy_sum = mechanism_->AddNoise(partial_sum_);
    // Add noise confidence interval.
    absl::StatusOr<ConfidenceInterval> interval =
        NoiseConfidenceInterval(noise_interval_level);

    if (std::is_integral<T>::value) {
      SafeOpResult<T> cast_result =
          SafeCastFromDouble<T>(std::round(noisy_sum));
      if (interval.ok()) {
        output = MakeOutput<T>(cast_result.value, interval.value());
      } else {
        output = MakeOutput<T>(cast_result.value);
      }
    } else {
      if (interval.ok()) {
        output = MakeOutput<T>(noisy_sum, interval.value());
      } else {
        output = MakeOutput<T>(noisy_sum);
      }
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
      std::unique_ptr<BoundsProvider<T>> bounds_provider)
      : BoundedSum<T>(epsilon, delta),
        mechanism_builder_(std::move(mechanism_builder)),
        l0_sensitivity_(l0_sensitivity),
        max_contributions_per_partition_(max_contributions_per_partition),
        bounds_provider_(std::move(bounds_provider)),
        clamped_calculation_(
            bounds_provider_->CreateClampedCalculationWithoutBounds()) {
    // We use partial values for each bin of the ApproxBounds logarithmic
    // histogram.
    pos_sum_.resize(clamped_calculation_->GetNumBins(), 0);
    neg_sum_.resize(clamped_calculation_->GetNumBins(), 0);
  }

  void AddEntry(const T& t) override {
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    if (std::isnan(static_cast<double>(t))) {
      return;
    }

    bounds_provider_->AddEntry(t);

    // Find partial sums.
    if (t >= 0) {
      clamped_calculation_->template AddToPartialSums<T>(&pos_sum_, t);
    } else {
      clamped_calculation_->template AddToPartialSums<T>(&neg_sum_, t);
    }
  }

  // Noise confidence interval is not known before finalizing the algorithm as
  // we are using approx bounds.
  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) override {
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
    *bs_summary.mutable_bounds() = bounds_provider_->Serialize();

    // TODO: Remove this old serialization code that we keep for
    // limited backwards compatibility.
    if (bs_summary.bounds().has_approx_bounds_summary()) {
      *bs_summary.mutable_bounds_summary() =
          bs_summary.bounds().approx_bounds_summary();
    }

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

    // Merge bounds summary.
    if (bs_summary.has_bounds()) {
      RETURN_IF_ERROR(bounds_provider_->Merge(bs_summary.bounds()));
    } else if (bs_summary.has_bounds_summary()) {
      // TODO: Remove this old serialization code that we keep for
      // limited backwards compatibility.
      BoundsSummary bounds_summary;
      *bounds_summary.mutable_approx_bounds_summary() =
          bs_summary.bounds_summary();
      RETURN_IF_ERROR(bounds_provider_->Merge(bounds_summary));
    }

    return absl::OkStatus();
  }

  // Returns the epsilon used to calculate bounds.
  double GetBoundingEpsilon() const { return bounds_provider_->GetEpsilon(); }

  // Returns the epsilon used to calculate the noisy sum.  The overall algorithm
  // also uses epsilon for privately inferred bounds using approx bounds.
  double GetAggregationEpsilon() const {
    return Algorithm<T>::GetEpsilon() - bounds_provider_->GetEpsilon();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BoundedSum<T>);
    memory += sizeof(T) * (pos_sum_.capacity() + neg_sum_.capacity());
    memory += bounds_provider_->MemoryUsed();
    memory += clamped_calculation_->MemoryUsed();
    memory += sizeof(*mechanism_builder_);
    return memory;
  }

  // Use the following methods only for testing.
  int GetMaxContributionsPerPartitionForTesting() {
    return max_contributions_per_partition_;
  }

  double GetL0SensitivityForTesting() { return l0_sensitivity_; }

  BoundsProvider<T>* GetBoundsProviderForTesting() {
    return bounds_provider_.get();
  }

 protected:
  // Since sensitivity is determined only by the larger-magnitude bound,
  // set the smaller-magnitude bound to be the negative of the larger. This
  // minimizes clamping and so maximizes accuracy.
  static BoundsResult<T> RelaxBoundsSameSumSensitivity(
      const BoundsResult<T>& bounds_result) {
    BoundsResult<T> result = bounds_result;
    // We need to be careful with the numerical limits since -max == lowest + 1
    // for integers.
    if (bounds_result.lower_bound == std::numeric_limits<T>::lowest()) {
      result.upper_bound = std::numeric_limits<T>::max();
      return result;
    }

    result.lower_bound =
        std::min(bounds_result.lower_bound, -1 * bounds_result.upper_bound);
    result.upper_bound =
        std::max(bounds_result.upper_bound, -1 * bounds_result.lower_bound);
    return result;
  }

  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    ASSIGN_OR_RETURN(BoundsResult<T> bounds_result,
                     bounds_provider_->FinalizeAndCalculateBounds());

    if (bounds_result.lower_bound == 0 && bounds_result.upper_bound == 0) {
      // In case both bounds are 0, we need to enlarge the bounds range as we
      // would otherwise have 0 sensitivity.  This is a quick fix that works
      // with BoundsProvider returning powers of two.
      //
      // TODO: Find a better solution for this quick fix.
      bounds_result.upper_bound = 1;
      bounds_result.lower_bound = -1;
    }

    bounds_result = RelaxBoundsSameSumSensitivity(bounds_result);

    // Construct NumericalMechanism.
    ASSIGN_OR_RETURN(std::unique_ptr<NumericalMechanism> mechanism,
                     BoundedSum<T>::BuildMechanism(
                         mechanism_builder_->Clone(), GetAggregationEpsilon(),
                         Algorithm<T>::GetDelta(), l0_sensitivity_,
                         max_contributions_per_partition_,
                         bounds_result.lower_bound, bounds_result.upper_bound));

    // To find the sum, pass the identity function as the transform. We pass
    // count = 0 because the count should never be used.
    ASSIGN_OR_RETURN(
        T sum, clamped_calculation_->template ComputeFromPartials<T>(
                   pos_sum_, neg_sum_, [](T x) { return x; },
                   bounds_result.lower_bound, bounds_result.upper_bound, 0));

    // Add noise and confidence interval to the sum output. Use the remaining
    // privacy budget.
    T noisy_sum = mechanism->AddNoise(sum);
    absl::StatusOr<ConfidenceInterval> interval =
        mechanism->NoiseConfidenceInterval(noise_interval_level);

    Output output;

    if (interval.ok()) {
      output = MakeOutput<T>(noisy_sum, interval.value());
    } else {
      output = MakeOutput<T>(noisy_sum);
    }

    // Populate the bounding report with ApproxBounds information.
    output.mutable_error_report()->set_allocated_bounding_report(
        new BoundingReport(bounds_provider_->GetBoundingReport(bounds_result)));

    return output;
  }

  void ResetState() override {
    std::fill(pos_sum_.begin(), pos_sum_.end(), 0);
    std::fill(neg_sum_.begin(), neg_sum_.end(), 0);
    bounds_provider_->Reset();
  }

 private:
  // Vectors of partial values stored for automatic clamping.
  std::vector<T> pos_sum_, neg_sum_;

  // Used to construct the numerical mechanism once bounds are obtained.
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_;
  const double l0_sensitivity_;
  const int max_contributions_per_partition_;

  // Algorithm to privately infer bounds.
  std::unique_ptr<BoundsProvider<T>> bounds_provider_;
  std::unique_ptr<internal::ClampedCalculationWithoutBounds<T>>
      clamped_calculation_;
};

template <typename T>
class BoundedSum<T>::Builder {
 public:
  BoundedSum<T>::Builder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  BoundedSum<T>::Builder& SetDelta(double delta) {
    delta_ = delta;
    return *this;
  }

  BoundedSum<T>::Builder& SetMaxPartitionsContributed(
      int max_partitions_contributed) {
    max_partitions_contributed_ = max_partitions_contributed;
    return *this;
  }

  BoundedSum<T>::Builder& SetMaxContributionsPerPartition(
      int max_contributions_per_partition) {
    max_contributions_per_partition_ = max_contributions_per_partition;
    return *this;
  }

  BoundedSum<T>::Builder& SetUpper(T upper) {
    upper_ = upper;
    return *this;
  }

  BoundedSum<T>::Builder& SetLower(T lower) {
    lower_ = lower;
    return *this;
  }

  BoundedSum<T>::Builder& SetApproxBounds(
      std::unique_ptr<ApproxBounds<T>> approx_bounds) {
    approx_bounds_ = std::move(approx_bounds);
    return *this;
  }

  BoundedSum<T>::Builder& SetBoundsProvider(
      std::unique_ptr<BoundsProvider<T>> bounds_provider) {
    bounds_provider_ = std::move(bounds_provider);
    return *this;
  }

  BoundedSum<T>::Builder& SetLaplaceMechanism(
      std::unique_ptr<NumericalMechanismBuilder> builder) {
    mechanism_builder_ = std::move(builder);
    return *this;
  }

  absl::StatusOr<std::unique_ptr<BoundedSum<T>>> Build() {
    if (!epsilon_.has_value()) {
      epsilon_ = DefaultEpsilon();
      LOG(WARNING) << "Default epsilon of " << epsilon_.value()
                   << " is being used. Consider setting your own epsilon based "
                      "on privacy considerations.";
    }
    RETURN_IF_ERROR(ValidateEpsilon(epsilon_));
    RETURN_IF_ERROR(ValidateDelta(delta_));
    RETURN_IF_ERROR(ValidateBounds(lower_, upper_));
    if (lower_.has_value()) {
      RETURN_IF_ERROR(CheckLowerBound(lower_.value()));
    }
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed_));
    RETURN_IF_ERROR(
        ValidateMaxContributionsPerPartition(max_contributions_per_partition_));
    if (upper_.has_value() && lower_.has_value()) {
      return BuildSumWithFixedBounds();
    }
    return BuildSumWithApproxBounds();
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
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;
  std::unique_ptr<BoundsProvider<T>> bounds_provider_;

  absl::StatusOr<std::unique_ptr<BoundedSum<T>>> BuildSumWithFixedBounds() {
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> mechanism,
        BuildMechanism(mechanism_builder_->Clone(), epsilon_.value(), delta_,
                       max_partitions_contributed_,
                       max_contributions_per_partition_, lower_.value(),
                       upper_.value()));

    return absl::StatusOr<std::unique_ptr<BoundedSum<T>>>(
        std::make_unique<BoundedSumWithFixedBounds<T>>(
            epsilon_.value(), delta_, lower_.value(), upper_.value(),
            std::move(mechanism)));
  }

  absl::StatusOr<std::unique_ptr<BoundedSum<T>>> BuildSumWithApproxBounds() {
    if (approx_bounds_ != nullptr && bounds_provider_ != nullptr) {
      return absl::InvalidArgumentError(
          "ApproxBounds and a generic BoundsProvider have been set, but only "
          "one is allowed. Please use only one, either SetApproxBounds or "
          "SetBoundsProvider.");
    } else if (approx_bounds_ == nullptr && bounds_provider_ == nullptr) {
      ASSIGN_OR_RETURN(
          std::unique_ptr<ApproxBounds<T>> approx_bounds,
          typename ApproxBounds<T>::Builder()
              .SetEpsilon(epsilon_.value() / 2)
              .SetLaplaceMechanism(mechanism_builder_->Clone())
              .SetMaxContributionsPerPartition(max_contributions_per_partition_)
              .SetMaxPartitionsContributed(max_partitions_contributed_)
              .Build());
      bounds_provider_ = std::make_unique<ApproxBoundsAsBoundsProvider<T>>(
          std::move(approx_bounds));
    } else if (approx_bounds_ != nullptr && bounds_provider_ == nullptr) {
      bounds_provider_ = std::make_unique<ApproxBoundsAsBoundsProvider<T>>(
          std::move(approx_bounds_));
    }
    if (epsilon_.value() <= bounds_provider_->GetEpsilon()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Approx Bounds consumes more epsilon budget than available. Total "
          "Epsilon: ",
          epsilon_.value(),
          " Approx Bounds Epsilon: ", bounds_provider_->GetEpsilon()));
    }

    return absl::StatusOr<std::unique_ptr<BoundedSum<T>>>(
        std::make_unique<BoundedSumWithApproxBounds<T>>(
            epsilon_.value(), delta_, max_partitions_contributed_,
            max_contributions_per_partition_, mechanism_builder_->Clone(),
            std::move(bounds_provider_)));
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_SUM_H_
