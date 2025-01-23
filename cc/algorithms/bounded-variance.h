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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_VARIANCE_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_VARIANCE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
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
#include "proto/data.pb.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {

// Incrementally provides a differentially private variance for values in the
// range [lower..upper]. Values outside of this range will be clamped so they
// lie in the range. The output will also be clamped between 0 and (upper -
// lower)^2 / 4. Since the result is guaranteed to be positive, this algorithm
// can be used to compute a differentially private standard deviation.
//
// The algorithm uses O(1) memory and runs in O(n) time where n is the size of
// the dataset, making it a fast and efficient. The amount of noise added grows
// quadratically in (upper - lower) and decreases linearly in n, so it might not
// produce good results unless n >> (upper - lower)^2.
//
// The algorithm is a variation of the algorithm for differentially private mean
// from "Differential Privacy: From Theory to Practice", section 2.5.5:
// https://books.google.com/books?id=WFttDQAAQBAJ&pg=PA24#v=onepage&q&f=false
template <typename T>
class BoundedVariance : public Algorithm<T> {
  static_assert(std::is_arithmetic<T>::value,
                "BoundedVariance can only be used for arithmetic types");

 public:
  // Builder for BoundedVariance algorithm.
  class Builder;

  BoundedVariance(const double epsilon) : Algorithm<T>(epsilon) {}

  virtual ~BoundedVariance() = default;

  // Returns the epsilon used to calculate approximate bounds. If approximate
  // bounds are not used, returns 0.
  virtual double GetBoundingEpsilon() const = 0;

  // Returns the epsilon used to calculate the noisy mean. If bounds are
  // specified explicitly, this will be the total epsilon used by the algorithm.
  virtual double GetAggregationEpsilon() const = 0;

 protected:
  virtual void AddMultipleEntries(const T& input, int64_t num_of_entries) = 0;

  // For integral type, check for overflow in subtraction squared.
  template <typename T2 = T,
            std::enable_if_t<std::is_integral<T2>::value>* = nullptr>
  static absl::Status CheckBounds(T lower, T upper) {
    if (lower > upper) {
      return absl::InvalidArgumentError("Lower cannot be greater than upper.");
    }
    SafeOpResult<T> subtract_result = SafeSubtract(upper, lower);
    SafeOpResult<T> safe_square_result = SafeSquare(subtract_result.value);
    if (subtract_result.overflow || safe_square_result.overflow) {
      return absl::InvalidArgumentError(
          "Sensitivity calculation caused integer overflow.");
    }
    if (upper > std::sqrt(std::numeric_limits<T>::max()) ||
        lower < -1 * std::sqrt(std::numeric_limits<T>::max())) {
      return absl::InvalidArgumentError("Squaring the bounds caused overflow.");
    }
    return absl::OkStatus();
  }

  template <typename T2 = T,
            std::enable_if_t<std::is_floating_point<T2>::value>* = nullptr>
  static absl::Status CheckBounds(T lower, T upper) {
    if (lower > upper) {
      return absl::InvalidArgumentError("Lower cannot be greater than upper.");
    }
    return absl::OkStatus();
  }

  // Returns the width of the range of f(x) = x^2 where the domain of f is
  // [lower, upper].
  static double RangeOfSquares(T lower, T upper) {
    if (0 > lower && 0 < upper) {
      return std::max(lower * lower, upper * upper);
    }
    return std::abs(upper * upper - lower * lower);
  }

  static absl::StatusOr<std::unique_ptr<NumericalMechanism>> BuildSumMechanism(
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      const double epsilon, const double l0_sensitivity,
      const double max_contributions_per_partition, const T lower,
      const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(max_contributions_per_partition *
                            static_cast<double>(upper - lower) / 2.0)
        .Build();
  }

  static absl::StatusOr<std::unique_ptr<NumericalMechanism>>
  BuildSumOfSquaresMechanism(
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      const double epsilon, const double l0_sensitivity,
      const double max_contributions_per_partition, const T lower,
      const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(max_contributions_per_partition *
                            (RangeOfSquares(lower, upper) / 2))
        .Build();
  }

  static double IntervalLengthSquared(T lower, T upper) {
    return std::pow(static_cast<double>(upper - lower), 2);
  }

  // Returns the midpoint of the range of f(x) = x^2 where the domains of f is
  // [lower, upper].
  static double MidpointOfSquares(T lower, T upper) {
    DCHECK_GE(upper, lower);
    if (0 > lower && 0 < upper) {
      return std::max(lower * lower, upper * upper) / 2;
    }
    return lower * lower + (upper * upper - lower * lower) / 2;
  }

  // Friend class for testing only
  friend class BoundedVarianceTestPeer;
};

// Bounded variance implementation that uses fixed bounds.
template <typename T>
class BoundedVarianceWithFixedBounds : public BoundedVariance<T> {
 public:
  BoundedVarianceWithFixedBounds(
      const double epsilon, const T lower, const T upper,
      std::unique_ptr<NumericalMechanism> count_mechanism,
      std::unique_ptr<NumericalMechanism> sum_mechanism,
      std::unique_ptr<NumericalMechanism> sum_of_squares_mechanism)
      : BoundedVariance<T>(epsilon),
        lower_(lower),
        upper_(upper),
        count_mechanism_(std::move(count_mechanism)),
        sum_mechanism_(std::move(sum_mechanism)),
        sum_of_squares_mechanism_(std::move(sum_of_squares_mechanism)) {}

  void AddEntry(const T& t) override { AddMultipleEntries(t, 1); }

  Summary Serialize() const override {
    BoundedVarianceSummary variance_summary;
    variance_summary.set_count(partial_count_);
    SetValue(variance_summary.add_pos_sum(), partial_sum_);
    variance_summary.add_pos_sum_of_squares(partial_sum_of_squares_);

    // Pack variance summary into summary.
    Summary summary;
    summary.mutable_data()->PackFrom(variance_summary);
    return summary;
  }

  absl::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return absl::InternalError(
          "Cannot merge summary with no bounded variance data.");
    }

    // Unpack bounded variance summary.
    BoundedVarianceSummary variance_summary;
    if (!summary.data().UnpackTo(&variance_summary)) {
      return absl::InternalError(
          "Bounded variance summary unable to be unpacked.");
    }

    // Check for expected sizes of repeated fields.
    if (variance_summary.pos_sum_size() != 1) {
      return absl::InternalError(
          absl::StrCat("Expected positive sums of size exactly 1 but got ",
                       variance_summary.pos_sum_size()));
    }
    if (variance_summary.pos_sum_of_squares_size() != 1) {
      return absl::InternalError(absl::StrCat(
          "Expected positive sum of squares of size exactly 1 but got ",
          variance_summary.pos_sum_of_squares_size()));
    }

    // Verification successful.  Merge fields.
    partial_count_ += variance_summary.count();
    partial_sum_ += GetValue<T>(variance_summary.pos_sum(0));
    partial_sum_of_squares_ += variance_summary.pos_sum_of_squares(0);

    return absl::OkStatus();
  }

  int64_t MemoryUsed() override {
    return sizeof(BoundedVarianceWithFixedBounds) +
           count_mechanism_->MemoryUsed() + sum_mechanism_->MemoryUsed() +
           sum_of_squares_mechanism_->MemoryUsed();
  }

  double GetBoundingEpsilon() const override { return 0; }

  double GetAggregationEpsilon() const override {
    return Algorithm<T>::GetEpsilon();
  }

 protected:
  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    const double sum_midpoint = lower_ + ((upper_ - lower_) / 2);
    const double sum_of_squares_midpoint =
        BoundedVariance<T>::MidpointOfSquares(lower_, upper_);

    const double noised_count = count_mechanism_->AddNoise(partial_count_);
    const double noised_normalized_sum = sum_mechanism_->AddNoise(
        partial_sum_ - (partial_count_ * sum_midpoint));
    const double noised_normalized_sum_of_squares =
        sum_of_squares_mechanism_->AddNoise(
            partial_sum_of_squares_ -
            (partial_count_ * sum_of_squares_midpoint));

    double mean;
    double mean_of_squares;
    if (noised_count <= 1) {
      mean = sum_midpoint;
      mean_of_squares = sum_of_squares_midpoint;
    } else {
      mean = (noised_normalized_sum / noised_count) + sum_midpoint;
      mean_of_squares = (noised_normalized_sum_of_squares / noised_count) +
                        sum_of_squares_midpoint;
    }

    const double noised_variance = mean_of_squares - (mean * mean);

    Output output;
    AddToOutput<double>(
        &output,
        Clamp<double>(
            0, BoundedVariance<T>::IntervalLengthSquared(lower_, upper_) / 4,
            noised_variance));
    return output;
  }

  void ResetState() override {
    partial_count_ = 0;
    partial_sum_ = 0;
    partial_sum_of_squares_ = 0;
  }

  void AddMultipleEntries(const T& input, int64_t num_of_entries) override {
    absl::Status status =
        ValidateIsPositive(num_of_entries, "Number of entries");
    if (std::isnan(static_cast<double>(input)) || !status.ok()) {
      return;
    }
    partial_count_ += num_of_entries;
    const T clamped_input = Clamp<T>(lower_, upper_, input);
    partial_sum_ += clamped_input * num_of_entries;
    partial_sum_of_squares_ += std::pow(clamped_input, 2) * num_of_entries;
  }

 private:
  const T lower_;
  const T upper_;

  std::unique_ptr<NumericalMechanism> count_mechanism_;
  std::unique_ptr<NumericalMechanism> sum_mechanism_;
  std::unique_ptr<NumericalMechanism> sum_of_squares_mechanism_;

  int64_t partial_count_ = 0;
  T partial_sum_ = 0;
  double partial_sum_of_squares_ = 0;
};

template <typename T>
class BoundedVarianceWithApproxBounds : public BoundedVariance<T> {
 public:
  BoundedVarianceWithApproxBounds(
      const double epsilon, const double epsilon_for_sum,
      const double epsilon_for_squares, const double l0_sensitivity,
      const int max_contributions_per_partition,
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      std::unique_ptr<NumericalMechanism> count_mechanism,
      std::unique_ptr<BoundsProvider<T>> bounds_provider)
      : BoundedVariance<T>(epsilon),
        epsilon_for_sum_(epsilon_for_sum),
        epsilon_for_squares_(epsilon_for_squares),
        mechanism_builder_(std::move(mechanism_builder)),
        l0_sensitivity_(l0_sensitivity),
        max_contributions_per_partition_(max_contributions_per_partition),
        count_mechanism_(std::move(count_mechanism)),
        bounds_provider_(std::move(bounds_provider)),
        clamped_calculation_(
            bounds_provider_->CreateClampedCalculationWithoutBounds()) {
    // To determining bounds, we need partial values for each bin of the
    // ApproxBounds logarithmic histogram.
    pos_sum_.resize(clamped_calculation_->GetNumBins(), 0);
    neg_sum_.resize(clamped_calculation_->GetNumBins(), 0);
    pos_sum_of_squares_.resize(clamped_calculation_->GetNumBins(), 0);
    neg_sum_of_squares_.resize(clamped_calculation_->GetNumBins(), 0);
  }

  void AddEntry(const T& t) override { AddMultipleEntries(t, 1); }

  Summary Serialize() const override {
    // Create BoundedVarianceSummary.
    BoundedVarianceSummary bv_summary;
    bv_summary.set_count(partial_count_);
    for (T x : pos_sum_) {
      SetValue(bv_summary.add_pos_sum(), x);
    }
    for (T x : neg_sum_) {
      SetValue(bv_summary.add_neg_sum(), x);
    }
    for (double x : pos_sum_of_squares_) {
      bv_summary.add_pos_sum_of_squares(x);
    }
    for (double x : neg_sum_of_squares_) {
      bv_summary.add_neg_sum_of_squares(x);
    }

    // Serialize bounds data.
    *bv_summary.mutable_bounds() = bounds_provider_->Serialize();

    // TODO: Remove this old serialization code that we keep for
    // limited backwards compatibility.
    if (bv_summary.bounds().has_approx_bounds_summary()) {
      *bv_summary.mutable_bounds_summary() =
          bv_summary.bounds().approx_bounds_summary();
    }

    // Create Summary.
    Summary summary;
    summary.mutable_data()->PackFrom(bv_summary);
    return summary;
  }

  absl::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return absl::InternalError(
          "Cannot merge summary with no bounded variance data.");
    }

    // Unpack bounded variance summary.
    BoundedVarianceSummary bv_summary;
    if (!summary.data().UnpackTo(&bv_summary)) {
      return absl::InternalError(
          "Bounded variance summary unable to be unpacked.");
    }
    if (!bv_summary.has_bounds() && !bv_summary.has_bounds_summary()) {
      return absl::InternalError(
          "Merged BoundedVariance must have the same bounding strategy.");
    }
    if (pos_sum_.size() != bv_summary.pos_sum_size() ||
        neg_sum_.size() != bv_summary.neg_sum_size() ||
        pos_sum_of_squares_.size() != bv_summary.pos_sum_of_squares_size() ||
        neg_sum_of_squares_.size() != bv_summary.neg_sum_of_squares_size()) {
      return absl::InternalError(
          "Merged BoundedVariance must have the same amount of partial "
          "sum or sum of squares values as this BoundedVariance.");
    }

    // Merge bounds
    if (bv_summary.has_bounds()) {
      RETURN_IF_ERROR(bounds_provider_->Merge(bv_summary.bounds()));
    } else if (bv_summary.has_bounds_summary()) {
      // TODO: Remove this old serialization code that we keep for
      // limited backwards compatibility.
      BoundsSummary bounds_summary;
      *bounds_summary.mutable_approx_bounds_summary() =
          bv_summary.bounds_summary();
      RETURN_IF_ERROR(bounds_provider_->Merge(bounds_summary));
    }

    // Add count and partial values to current ones.
    partial_count_ += bv_summary.count();
    for (int i = 0; i < pos_sum_.size(); ++i) {
      pos_sum_[i] += GetValue<T>(bv_summary.pos_sum(i));
      pos_sum_of_squares_[i] += bv_summary.pos_sum_of_squares(i);
    }
    for (int i = 0; i < neg_sum_.size(); ++i) {
      neg_sum_[i] += GetValue<T>(bv_summary.neg_sum(i));
      neg_sum_of_squares_[i] += bv_summary.neg_sum_of_squares(i);
    }

    return absl::OkStatus();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BoundedVarianceWithApproxBounds<T>);
    memory += sizeof(T) * (pos_sum_.capacity() + neg_sum_.capacity());
    memory += sizeof(double) *
              (pos_sum_of_squares_.capacity() + neg_sum_of_squares_.capacity());
    memory += sizeof(*mechanism_builder_);
    memory += bounds_provider_->MemoryUsed();
    memory += clamped_calculation_->MemoryUsed();
    return memory;
  }

  // Returns the epsilon used to calculate bounds.
  double GetBoundingEpsilon() const override {
    return bounds_provider_->GetEpsilon();
  }

  // Returns the epsilon used to calculate the noisy mean.
  double GetAggregationEpsilon() const override {
    return Algorithm<T>::GetEpsilon() - GetBoundingEpsilon();
  }

  // Returns a pointer to the BoundsProvider object.  Does not transfer
  // ownership.  Only use for testing.
  BoundsProvider<T>* GetBoundsProviderForTesting() {
    return bounds_provider_.get();
  }

 private:
  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    Output output;

    ASSIGN_OR_RETURN(BoundsResult<T> bounds_result,
                     bounds_provider_->FinalizeAndCalculateBounds());

    const T lower = bounds_result.lower_bound;
    const T upper = bounds_result.upper_bound;
    RETURN_IF_ERROR(BoundedVariance<T>::CheckBounds(lower, upper));

    // To find the sum, pass the identity function as the transform.
    ASSIGN_OR_RETURN(const double sum,
                     clamped_calculation_->template ComputeFromPartials<T>(
                         pos_sum_, neg_sum_, [](T x) { return x; }, lower,
                         upper, partial_count_));

    // To find sum of squares, pass the square function.
    ASSIGN_OR_RETURN(
        const double sum_of_squares,
        clamped_calculation_->template ComputeFromPartials<double>(
            pos_sum_of_squares_, neg_sum_of_squares_, [](T x) { return x * x; },
            lower, upper, partial_count_));

    // Populate the bounding report with ApproxBounds information.
    *(output.mutable_error_report()->mutable_bounding_report()) =
        bounds_provider_->GetBoundingReport(bounds_result);

    const double noised_count = count_mechanism_->AddNoise(partial_count_);

    // Calculate noised normalized sum
    const T sum_midpoint = lower + ((upper - lower) / 2);
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> sum_mechanism,
        BoundedVariance<T>::BuildSumMechanism(
            mechanism_builder_->Clone(), epsilon_for_sum_, l0_sensitivity_,
            max_contributions_per_partition_, lower, upper));
    const double noised_normalized_sum =
        sum_mechanism->AddNoise(sum - (partial_count_ * sum_midpoint));

    // Calculate noised normalized sum of squares.
    const double sum_of_squares_midpoint =
        BoundedVariance<T>::MidpointOfSquares(lower, upper);
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> sum_of_squares_mechanism,
        BoundedVariance<T>::BuildSumOfSquaresMechanism(
            mechanism_builder_->Clone(), epsilon_for_squares_, l0_sensitivity_,
            max_contributions_per_partition_, lower, upper));
    const double noised_normalized_sum_of_squares =
        sum_of_squares_mechanism->AddNoise(
            sum_of_squares - (partial_count_ * sum_of_squares_midpoint));

    // Calculate the result from the noised values.  From this point everything
    // should be post-processing.
    double mean = sum_midpoint;
    double mean_of_square = sum_of_squares_midpoint;
    if (noised_count > 1) {
      mean = noised_normalized_sum / noised_count + sum_midpoint;
      mean_of_square = noised_normalized_sum_of_squares / noised_count +
                       sum_of_squares_midpoint;
    }

    const double noised_variance = mean_of_square - std::pow(mean, 2);

    AddToOutput<double>(
        &output,
        Clamp<double>(
            0.0, BoundedVariance<T>::IntervalLengthSquared(lower, upper) / 4,
            noised_variance));
    return output;
  }

  void ResetState() override {
    std::fill(pos_sum_.begin(), pos_sum_.end(), 0);
    std::fill(pos_sum_of_squares_.begin(), pos_sum_of_squares_.end(), 0);
    std::fill(neg_sum_.begin(), neg_sum_.end(), 0);
    std::fill(neg_sum_of_squares_.begin(), neg_sum_of_squares_.end(), 0);
    partial_count_ = 0;
    bounds_provider_->Reset();
  }

  void AddMultipleEntries(const T& input, int64_t num_of_entries) override {
    // Drop value if it is NaN.
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    absl::Status status =
        ValidateIsPositive(num_of_entries, "Number of entries");
    if (std::isnan(static_cast<double>(input)) || !status.ok()) {
      return;
    }

    // Count is unaffected by clamping.
    partial_count_ += num_of_entries;

    // Store partial results and feed input into ApproxBounds algorithm.
    for (int i = 0; i < num_of_entries; ++i) {
      bounds_provider_->AddEntry(input);
    }

    // Add to partial sums and sum of squares.
    auto difference_of_squares = [](T val1, T val2) {
      // Lessen the chance of becoming inf/-inf by calculating it like this.
      return (static_cast<double>(val1) + val2) *
             (static_cast<double>(val1) - val2);
    };

    if (input >= 0) {
      for (int i = 0; i < num_of_entries; ++i) {
        clamped_calculation_->template AddToPartialSums<T>(&pos_sum_, input);
        clamped_calculation_->template AddToPartials<double>(
            &pos_sum_of_squares_, input, difference_of_squares);
      }
    } else {
      for (int i = 0; i < num_of_entries; ++i) {
        clamped_calculation_->template AddToPartialSums<T>(&neg_sum_, input);
        clamped_calculation_->template AddToPartials<double>(
            &neg_sum_of_squares_, input, difference_of_squares);
      }
    }
  }

  // Vectors of partial values stored for automatic clamping.
  std::vector<T> pos_sum_, neg_sum_;
  std::vector<double> pos_sum_of_squares_, neg_sum_of_squares_;
  int64_t partial_count_ = 0;

  // Used to construct mechanism once bounds are obtained.
  const double epsilon_for_sum_;
  const double epsilon_for_squares_;
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_;
  const double l0_sensitivity_;
  const int max_contributions_per_partition_;

  std::unique_ptr<NumericalMechanism> count_mechanism_;

  std::unique_ptr<BoundsProvider<T>> bounds_provider_;

  std::unique_ptr<internal::ClampedCalculationWithoutBounds<T>>
      clamped_calculation_;
};

template <typename T>
class BoundedVariance<T>::Builder {
 public:
  BoundedVariance<T>::Builder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  BoundedVariance<T>::Builder& SetDelta(double delta) {
    delta_ = delta;
    return *this;
  }

  BoundedVariance<T>::Builder& SetMaxPartitionsContributed(
      int max_partitions_contributed) {
    max_partitions_contributed_ = max_partitions_contributed;
    return *this;
  }

  BoundedVariance<T>::Builder& SetMaxContributionsPerPartition(
      int max_contributions_per_partition) {
    max_contributions_per_partition_ = max_contributions_per_partition;
    return *this;
  }

  BoundedVariance<T>::Builder& SetUpper(T upper) {
    upper_ = upper;
    return *this;
  }

  BoundedVariance<T>::Builder& SetLower(T lower) {
    lower_ = lower;
    return *this;
  }

  BoundedVariance<T>::Builder& SetApproxBounds(
      std::unique_ptr<ApproxBounds<T>> approx_bounds) {
    bounds_provider_ = std::make_unique<ApproxBoundsAsBoundsProvider<T>>(
        std::move(approx_bounds));
    return *this;
  }

  BoundedVariance<T>::Builder& SetBoundsProvider(
      std::unique_ptr<BoundsProvider<T>> bounds_provider) {
    bounds_provider_ = std::move(bounds_provider);
    return *this;
  }

  BoundedVariance<T>::Builder& SetLaplaceMechanism(
      std::unique_ptr<NumericalMechanismBuilder> builder) {
    mechanism_builder_ = std::move(builder);
    return *this;
  }

  absl::StatusOr<std::unique_ptr<BoundedVariance<T>>> Build() {
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
      return BuildVarianceWithFixedBounds();
    }
    return BuildVarianceWithApproxBounds();
  }

 private:
  std::optional<double> epsilon_;
  double delta_ = 0;
  std::optional<T> upper_;
  std::optional<T> lower_;
  int max_partitions_contributed_ = 1;
  int max_contributions_per_partition_ = 1;
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_ =
      absl::make_unique<LaplaceMechanism::Builder>();
  std::unique_ptr<BoundsProvider<T>> bounds_provider_;

  absl::StatusOr<std::unique_ptr<BoundedVariance<T>>>
  BuildVarianceWithFixedBounds() {
    RETURN_IF_ERROR(CheckBounds(lower_.value(), upper_.value()));

    ASSIGN_OR_RETURN(std::unique_ptr<NumericalMechanism> count_mechanism,
                     mechanism_builder_->Clone()
                         ->SetEpsilon(epsilon_.value() / 3)
                         .SetL0Sensitivity(max_partitions_contributed_)
                         .SetLInfSensitivity(max_contributions_per_partition_)
                         .Build());
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> sum_mechanism,
        BoundedVariance<T>::BuildSumMechanism(
            mechanism_builder_->Clone(), epsilon_.value() / 3,
            max_partitions_contributed_, max_contributions_per_partition_,
            lower_.value(), upper_.value()));
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> sos_mechanism,
        BoundedVariance<T>::BuildSumOfSquaresMechanism(
            mechanism_builder_->Clone(), epsilon_.value() / 3,
            max_partitions_contributed_, max_contributions_per_partition_,
            lower_.value(), upper_.value()));

    return absl::StatusOr<std::unique_ptr<BoundedVariance<T>>>(
        absl::make_unique<BoundedVarianceWithFixedBounds<T>>(
            epsilon_.value(), lower_.value(), upper_.value(),
            std::move(count_mechanism), std::move(sum_mechanism),
            std::move(sos_mechanism)));
  }

  absl::StatusOr<std::unique_ptr<BoundedVariance<T>>>
  BuildVarianceWithApproxBounds() {
    if (bounds_provider_ == nullptr) {
      ASSIGN_OR_RETURN(
          std::unique_ptr<ApproxBounds<T>> approx_bounds,
          typename ApproxBounds<T>::Builder()
              .SetEpsilon(epsilon_.value() / 2.0)
              .SetLaplaceMechanism(mechanism_builder_->Clone())
              .SetMaxContributionsPerPartition(max_contributions_per_partition_)
              .SetMaxPartitionsContributed(max_partitions_contributed_)
              .Build());
      bounds_provider_ = std::make_unique<ApproxBoundsAsBoundsProvider<T>>(
          std::move(approx_bounds));
    }

    if (epsilon_.value() <= bounds_provider_->GetEpsilon()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The Bounds Provider consumes more epsilon budget than available. "
          "Total Epsilon: ",
          epsilon_.value(),
          " Bounds Provider Epsilon: ", bounds_provider_->GetEpsilon()));
    }

    // Budget calculation.
    const double remaining_epsilon =
        epsilon_.value() - bounds_provider_->GetEpsilon();

    const double epsilon_for_count = remaining_epsilon / 3;
    const double epsilon_for_sum = remaining_epsilon / 3;
    const double epsilon_for_squares =
        remaining_epsilon - epsilon_for_count - epsilon_for_sum;

    ASSIGN_OR_RETURN(std::unique_ptr<NumericalMechanism> count_mechanism,
                     mechanism_builder_->Clone()
                         ->SetEpsilon(epsilon_for_count)
                         .SetL0Sensitivity(max_partitions_contributed_)
                         .SetLInfSensitivity(max_contributions_per_partition_)
                         .Build());

    return absl::StatusOr<std::unique_ptr<BoundedVariance<T>>>(
        absl::make_unique<BoundedVarianceWithApproxBounds<T>>(
            epsilon_.value(), epsilon_for_sum, epsilon_for_squares,
            max_partitions_contributed_, max_contributions_per_partition_,
            mechanism_builder_->Clone(), std::move(count_mechanism),
            std::move(bounds_provider_)));
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_VARIANCE_H_
