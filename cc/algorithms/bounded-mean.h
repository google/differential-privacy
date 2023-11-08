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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_MEAN_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_MEAN_H_

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/internal/bounded-mean-ci.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/util.h"
#include "proto/confidence-interval.pb.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {

constexpr int kNumStepsOptMeanConfidenceInterval = 1000;

// Incrementally provides a differentially private average.
// All input vales are normalized to be their difference from the middle of the
// input range. That allows us to calculate the sum of all input values with
// half the sensitivity it would otherwise take for better accuracy (as compared
// to doing noisy sum / noisy count). This algorithm is taken from section 2.5.5
// of the following book (algorithm 2.4):
// https://books.google.com/books?id=WFttDQAAQBAJ&pg=PA24#v=onepage&q&f=false
template <typename T>
class BoundedMean : public Algorithm<T> {
  static_assert(std::is_arithmetic<T>::value,
                "BoundedMean can only be used for arithmetic types");

 public:
  // Builder for BoundedMean algorithm.
  class Builder;

  BoundedMean(double epsilon, double delta) : Algorithm<T>(epsilon, delta) {}

  virtual ~BoundedMean() = default;

  // For integral type, check for no overflow in the subtraction.
  template <typename T2 = T,
            std::enable_if_t<std::is_integral<T2>::value>* = nullptr>
  static absl::Status CheckBounds(T lower, T upper) {
    SafeOpResult<T> subtract_result = SafeSubtract(upper, lower);
    if (subtract_result.overflow) {
      return absl::InvalidArgumentError(
          "Upper - lower caused integer overflow.");
    }
    return absl::OkStatus();
  }

  // No checks for floating point type.
  template <typename T2 = T,
            std::enable_if_t<std::is_floating_point<T2>::value>* = nullptr>
  static absl::Status CheckBounds(T lower, T upper) {
    return absl::OkStatus();
  }

  // Numerical mechanism to add noise to the normalized sum.  Not to be confused
  // with the sum mechanism we are using in BoundedSum that is not normalized.
  static absl::StatusOr<std::unique_ptr<NumericalMechanism>>
  BuildMechanismForNormalizedSum(
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      const double epsilon, const double delta, const double l0_sensitivity,
      const double max_contributions_per_partition, const T lower,
      const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetDelta(delta)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(max_contributions_per_partition *
                            (std::abs(upper - lower) / 2.0))
        .Build();
  }

 protected:
  virtual void AddMultipleEntries(const T& input, int64_t num_of_entries) = 0;

  // Friend class for testing only.
  friend class BoundedMeanTestPeer;
};

template <typename T>
class BoundedMeanWithFixedBounds : public BoundedMean<T> {
 public:
  BoundedMeanWithFixedBounds(
      const double epsilon, const double delta, const T lower, const T upper,
      std::unique_ptr<NumericalMechanism> sum_mechanism,
      std::unique_ptr<NumericalMechanism> count_mechanism)
      : BoundedMean<T>(epsilon, delta),
        lower_(lower),
        upper_(upper),
        sum_mechanism_(std::move(sum_mechanism)),
        count_mechanism_(std::move(count_mechanism)) {}

  void AddEntry(const T& input) override { AddMultipleEntries(input, 1); }

  Summary Serialize() const override {
    BoundedMeanSummary mean_summary;
    mean_summary.set_count(partial_count_);
    SetValue(mean_summary.add_pos_sum(), partial_sum_);

    Summary summary;
    summary.mutable_data()->PackFrom(mean_summary);
    return summary;
  }

  absl::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return absl::InternalError(
          "Cannot merge summary with no bounded mean data.");
    }

    // Add counts and bounded sums.
    BoundedMeanSummary mean_summary;
    if (!summary.data().UnpackTo(&mean_summary)) {
      return absl::InternalError("Bounded mean summary unable to be unpacked.");
    }
    if (mean_summary.pos_sum_size() != 1) {
      return absl::InternalError(absl::StrCat(
          "Bounded mean summary must have exactly one pos_sum but got ",
          mean_summary.pos_sum_size()));
    }

    partial_sum_ += GetValue<T>(mean_summary.pos_sum(0));
    partial_count_ += mean_summary.count();

    return absl::OkStatus();
  }

  int64_t MemoryUsed() override {
    return sizeof(BoundedMeanWithFixedBounds) + sum_mechanism_->MemoryUsed() +
           count_mechanism_->MemoryUsed();
  }

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) override {
    return NoiseConfidenceInterval(confidence_level, 0, 0);
  }

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double noised_sum, double noised_count) {
    internal::BoundedMeanConfidenceIntervalParams params;
    params.confidence_level = confidence_level;
    params.lower_bound = lower_;
    params.upper_bound = upper_;
    params.noised_sum = noised_sum;
    params.noised_count = noised_count;
    params.count_mechanism = count_mechanism_.get();
    params.sum_mechanism = sum_mechanism_.get();
    return internal::BoundedMeanConfidenceInterval(params);
  }

  // Internal representation of an bounded mean result that also contains the
  // noised sum and count.
  struct BoundedMeanResult {
    double noised_mean;
    double noised_count;
    double noised_sum;
  };
  BoundedMeanResult GenerateBoundedMeanResult() {
    BoundedMeanResult result;
    result.noised_count = std::max(
        1.0, static_cast<double>(count_mechanism_->AddNoise(partial_count_)));
    result.noised_sum = sum_mechanism_->AddNoise(partial_sum_);
    if constexpr (!std::is_floating_point<T>::value) {
      // Normalize the sum for integers. In floating point case, the sum is
      // normalized, since each entry is normalized on addition.
      result.noised_sum -= partial_count_ * GetMidPoint();
    }
    result.noised_mean =
        (result.noised_sum / result.noised_count) + GetMidPoint();
    return result;
  }

 protected:
  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    const BoundedMeanResult result = GenerateBoundedMeanResult();
    const absl::StatusOr<ConfidenceInterval> ci = NoiseConfidenceInterval(
        noise_interval_level, result.noised_sum, result.noised_count);
    const double clamped_result =
        Clamp<double>(lower_, upper_, result.noised_mean);

    Output output;
    if (ci.ok()) {
      AddToOutput(&output, clamped_result, ci.value());
    } else {
      AddToOutput(&output, clamped_result);
    }
    return output;
  }

  void ResetState() override {
    partial_sum_ = 0;
    partial_count_ = 0;
  }

  void AddMultipleEntries(const T& input, int64_t num_of_entries) override {
    absl::Status status =
        ValidateIsPositive(num_of_entries, "Number of entries");
    if (std::isnan(static_cast<double>(input)) || !status.ok()) {
      return;
    }
    T processed_input = Clamp<T>(lower_, upper_, input);
    if constexpr (std::is_floating_point<T>::value) {
      // Normalize floating point input for for increasing numerical stability.
      processed_input -= GetMidPoint();
    }
    partial_sum_ += processed_input * num_of_entries;
    partial_count_ += num_of_entries;
  }

 private:
  double GetMidPoint() const { return lower_ + ((upper_ - lower_) / 2.0); }

  const T lower_;
  const T upper_;

  // Mechanisms to add noise to sum and count.
  std::unique_ptr<NumericalMechanism> sum_mechanism_;
  std::unique_ptr<NumericalMechanism> count_mechanism_;

  T partial_sum_ = 0;
  int64_t partial_count_ = 0;
};

template <typename T>
class BoundedMeanWithApproxBounds : public BoundedMean<T> {
 public:
  BoundedMeanWithApproxBounds(
      const double epsilon, const double delta, const double epsilon_for_sum,
      const double delta_for_sum, const double l0_sensitivity,
      const double max_contributions_per_partition,
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      std::unique_ptr<NumericalMechanism> count_mechanism,
      std::unique_ptr<ApproxBounds<T>> approx_bounds)
      : BoundedMean<T>(epsilon, delta),
        epsilon_for_sum_(epsilon_for_sum),
        delta_for_sum_(delta_for_sum),
        count_mechanism_(std::move(count_mechanism)),
        mechanism_builder_(std::move(mechanism_builder)),
        l0_sensitivity_(l0_sensitivity),
        max_contributions_per_partition_(max_contributions_per_partition),
        approx_bounds_(std::move(approx_bounds)) {
    // For automatically determining bounds, we need partial sums for each bin
    // of the ApproxBounds logarithmic histogram.
    pos_sum_.resize(approx_bounds_->NumPositiveBins(), 0);
    neg_sum_.resize(approx_bounds_->NumPositiveBins(), 0);
  }

  void AddEntry(const T& input) override { AddMultipleEntries(input, 1); }

  Summary Serialize() const override {
    // Create BoundedMeanSummary.
    BoundedMeanSummary bm_summary;
    bm_summary.set_count(partial_count_);
    for (T x : pos_sum_) {
      SetValue(bm_summary.add_pos_sum(), x);
    }
    for (T x : neg_sum_) {
      SetValue(bm_summary.add_neg_sum(), x);
    }

    // Add approx bounds summary
    Summary approx_bounds_summary = approx_bounds_->Serialize();
    approx_bounds_summary.data().UnpackTo(bm_summary.mutable_bounds_summary());

    // Create Summary.
    Summary summary;
    summary.mutable_data()->PackFrom(bm_summary);
    return summary;
  }

  absl::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return absl::InternalError(
          "Cannot merge summary with no bounded mean data.");
    }

    // Check bounded mean summary.
    BoundedMeanSummary bm_summary;
    if (!summary.data().UnpackTo(&bm_summary)) {
      return absl::InternalError("Bounded mean summary unable to be unpacked.");
    }
    if (pos_sum_.size() != bm_summary.pos_sum_size() ||
        neg_sum_.size() != bm_summary.neg_sum_size()) {
      return absl::InternalError(
          "Merged BoundedMeans must have equal number of partial sums.");
    }

    // Check and merge approx bounds summary.  This is the first operation that
    // modifies the internal state.
    Summary approx_bounds_summary;
    approx_bounds_summary.mutable_data()->PackFrom(bm_summary.bounds_summary());
    RETURN_IF_ERROR(approx_bounds_->Merge(approx_bounds_summary));

    // Merge partial counts and sum buckets.
    partial_count_ += bm_summary.count();
    for (int i = 0; i < pos_sum_.size(); ++i) {
      pos_sum_[i] += GetValue<T>(bm_summary.pos_sum(i));
    }
    for (int i = 0; i < neg_sum_.size(); ++i) {
      neg_sum_[i] += GetValue<T>(bm_summary.neg_sum(i));
    }

    return absl::OkStatus();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BoundedMean<T>);
    memory += sizeof(T) * (pos_sum_.capacity() + neg_sum_.capacity());
    memory += approx_bounds_->MemoryUsed();
    memory += sizeof(*mechanism_builder_);
    return memory;
  }

  // Returns the epsilon used to calculate approximate bounds.
  double GetBoundingEpsilon() const { return approx_bounds_->GetEpsilon(); }

  // Returns the epsilon used to calculate the noisy mean.
  double GetAggregationEpsilon() const {
    return Algorithm<T>::GetEpsilon() - GetBoundingEpsilon();
  }

  // Returns a pointer to the ApproxBounds object.  Does not transfer
  // ownsership.  Only use for testing.
  ApproxBounds<T>* GetApproxBoundsForTesting() { return approx_bounds_.get(); }

 protected:
  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    // Use a fraction of the privacy budget to find the approximate bounds.
    ASSIGN_OR_RETURN(Output bounds,
                     approx_bounds_->PartialResult(noise_interval_level));
    const T lower = GetValue<T>(bounds.elements(0).value());
    const T upper = GetValue<T>(bounds.elements(1).value());
    RETURN_IF_ERROR(BoundedMean<T>::CheckBounds(lower, upper));

    // To find the sum, pass the identity function as the transform.
    ASSIGN_OR_RETURN(const T sum,
                     approx_bounds_->template ComputeFromPartials<T>(
                         pos_sum_, neg_sum_, [](T x) { return x; }, lower,
                         upper, partial_count_));

    // Populate the bounding report with ApproxBounds information.
    Output output;
    *(output.mutable_error_report()->mutable_bounding_report()) =
        approx_bounds_->GetBoundingReport(lower, upper);

    // Construct the sum mechanism mechanism with the obtained bounds.
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> sum_mechanism,
        BoundedMean<T>::BuildMechanismForNormalizedSum(
            mechanism_builder_->Clone(), epsilon_for_sum_, delta_for_sum_,
            l0_sensitivity_, max_contributions_per_partition_, lower, upper));

    // We use the midpoint to normalize the sum.
    const double midpoint = lower + (upper - lower) / 2;

    const double noised_count = std::max(
        1.0, static_cast<double>(count_mechanism_->AddNoise(partial_count_)));
    const double normalized_sum =
        sum_mechanism->AddNoise(sum - partial_count_ * midpoint);
    const double mean = normalized_sum / noised_count + midpoint;

    // Calculate the confidence interval for the given noise and on the approx
    // bounds result.  This only takes the noise that is added into account and
    // *not* the probability for choosing the bounds.
    internal::BoundedMeanConfidenceIntervalParams ci_params;
    ci_params.lower_bound = lower;
    ci_params.upper_bound = upper;
    ci_params.confidence_level = noise_interval_level;
    ci_params.noised_sum = normalized_sum;
    ci_params.noised_count = noised_count;
    ci_params.sum_mechanism = sum_mechanism.get();
    ci_params.count_mechanism = count_mechanism_.get();
    const ConfidenceInterval ci =
        internal::BoundedMeanConfidenceInterval(ci_params);

    // Add mean to output and return the result.
    AddToOutput<double>(&output, Clamp<double>(lower, upper, mean), ci);
    return output;
  }

  void ResetState() override {
    std::fill(pos_sum_.begin(), pos_sum_.end(), 0);
    std::fill(neg_sum_.begin(), neg_sum_.end(), 0);
    partial_count_ = 0;
    approx_bounds_->Reset();
  }

  void AddMultipleEntries(const T& input, int64_t num_of_entries) override {
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    absl::Status status =
        ValidateIsPositive(num_of_entries, "Number of entries");
    if (std::isnan(static_cast<double>(input)) || !status.ok()) {
      return;
    }

    approx_bounds_->AddMultipleEntries(input, num_of_entries);

    // Find partial sums.
    if (input >= 0) {
      approx_bounds_->template AddMultipleEntriesToPartialSums<T>(
          &pos_sum_, input, num_of_entries);
    } else {
      approx_bounds_->template AddMultipleEntriesToPartialSums<T>(
          &neg_sum_, input, num_of_entries);
    }
    partial_count_ += num_of_entries;
  }

 private:
  // Vectors of partial values stored for automatic clamping.
  std::vector<T> pos_sum_, neg_sum_;

  // Raw count of the number of entries added.
  int64_t partial_count_ = 0;

  // The count mechanism does not depend on bounds and will be constructed in
  // the builder.  The sum mechanism depends on the bounds and will be
  // constructed when bounds are known (during finalization of the result).
  std::unique_ptr<NumericalMechanism> count_mechanism_;

  // Used to construct the sum mechanism once bounds are obtained for
  // auto-bounding.
  const double epsilon_for_sum_;
  const double delta_for_sum_;
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_;
  const double l0_sensitivity_;
  const double max_contributions_per_partition_;

  // Approx bounds instance to automatically determining bounds.
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;
};

template <typename T>
class BoundedMean<T>::Builder {
 public:
  BoundedMean<T>::Builder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  BoundedMean<T>::Builder& SetDelta(double delta) {
    delta_ = delta;
    return *this;
  }

  BoundedMean<T>::Builder& SetMaxPartitionsContributed(
      int max_partitions_contributed) {
    max_partitions_contributed_ = max_partitions_contributed;
    return *this;
  }

  BoundedMean<T>::Builder& SetMaxContributionsPerPartition(
      int max_contributions_per_partition) {
    max_contributions_per_partition_ = max_contributions_per_partition;
    return *this;
  }

  BoundedMean<T>::Builder& SetUpper(T upper) {
    upper_ = upper;
    return *this;
  }

  BoundedMean<T>::Builder& SetLower(T lower) {
    lower_ = lower;
    return *this;
  }

  BoundedMean<T>::Builder& SetApproxBounds(
      std::unique_ptr<ApproxBounds<T>> approx_bounds) {
    approx_bounds_ = std::move(approx_bounds);
    return *this;
  }

  BoundedMean<T>::Builder& SetLaplaceMechanism(
      std::unique_ptr<NumericalMechanismBuilder> builder) {
    mechanism_builder_ = std::move(builder);
    return *this;
  }

  absl::StatusOr<std::unique_ptr<BoundedMean<T>>> Build() {
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
    if (upper_.has_value() && lower_.has_value()) {
      return BuildMeanWithFixedBounds();
    }
    return BuildMeanWithApproxBounds();
  }

 private:
  absl::optional<double> epsilon_;
  double delta_ = 0;
  absl::optional<T> upper_;
  absl::optional<T> lower_;
  int max_partitions_contributed_ = 1;
  int max_contributions_per_partition_ = 1;
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_ =
      absl::make_unique<LaplaceMechanism::Builder>();
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;

  absl::StatusOr<std::unique_ptr<BoundedMean<T>>> BuildMeanWithFixedBounds() {
    RETURN_IF_ERROR(
        BoundedMean<T>::CheckBounds(lower_.value(), upper_.value()));
    ASSIGN_OR_RETURN(std::unique_ptr<NumericalMechanism> count_mechanism,
                     mechanism_builder_->Clone()
                         ->SetEpsilon(epsilon_.value() / 2)
                         .SetDelta(delta_ / 2)
                         .SetL0Sensitivity(max_partitions_contributed_)
                         .SetLInfSensitivity(max_contributions_per_partition_)
                         .Build());
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> sum_mechanism,
        BuildMechanismForNormalizedSum(
            mechanism_builder_->Clone(), epsilon_.value() / 2, delta_ / 2,
            /*l0_sensitivity=*/max_partitions_contributed_,
            max_contributions_per_partition_, lower_.value(), upper_.value()));

    return absl::StatusOr<std::unique_ptr<BoundedMean<T>>>(
        absl::make_unique<BoundedMeanWithFixedBounds<T>>(
            epsilon_.value(), delta_, lower_.value(), upper_.value(),
            std::move(sum_mechanism), std::move(count_mechanism)));
  }

  absl::StatusOr<std::unique_ptr<BoundedMean<T>>> BuildMeanWithApproxBounds() {
    if (!approx_bounds_) {
      ASSIGN_OR_RETURN(
          approx_bounds_,
          typename ApproxBounds<T>::Builder()
              .SetEpsilon(epsilon_.value() / 2)
              .SetLaplaceMechanism(mechanism_builder_->Clone())
              .SetMaxContributionsPerPartition(max_contributions_per_partition_)
              .SetMaxPartitionsContributed(max_partitions_contributed_)
              .Build());
    }

    if (epsilon_.value() <= approx_bounds_->GetEpsilon()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Approx Bounds consumes more epsilon budget than available. Total "
          "Epsilon: ",
          epsilon_.value(),
          " Approx Bounds Epsilon: ", approx_bounds_->GetEpsilon()));
    }

    // Budget calculation.
    const double epsilon_for_count =
        (epsilon_.value() - approx_bounds_->GetEpsilon()) / 2;
    const double epsilon_for_sum =
        epsilon_.value() - approx_bounds_->GetEpsilon() - epsilon_for_count;

    const double delta_for_count = delta_ / 2;
    const double delta_for_sum = delta_ - delta_for_count;

    ASSIGN_OR_RETURN(std::unique_ptr<NumericalMechanism> count_mechanism,
                     mechanism_builder_->Clone()
                         ->SetEpsilon(epsilon_for_count)
                         .SetDelta(delta_for_count)
                         .SetL0Sensitivity(max_partitions_contributed_)
                         .SetLInfSensitivity(max_contributions_per_partition_)
                         .Build());

    return absl::StatusOr<std::unique_ptr<BoundedMean<T>>>(
        absl::make_unique<BoundedMeanWithApproxBounds<T>>(
            epsilon_.value(), delta_, epsilon_for_sum, delta_for_sum,
            max_partitions_contributed_, max_contributions_per_partition_,
            mechanism_builder_->Clone(), std::move(count_mechanism),
            std::move(approx_bounds_)));
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_MEAN_H_
