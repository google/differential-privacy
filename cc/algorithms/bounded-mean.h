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

#include "google/protobuf/any.pb.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "base/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounded-algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/summary.pb.h"
#include "base/canonical_errors.h"
#include "base/status_macros.h"

namespace differential_privacy {

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

  BoundedMean(const double epsilon) : Algorithm<T>(epsilon) {}

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
  static base::StatusOr<std::unique_ptr<NumericalMechanism>>
  BuildMechanismForNormalizedSum(
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      const double epsilon, const double l0_sensitivity,
      const double max_contributions_per_partition, const T lower,
      const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(max_contributions_per_partition *
                            (std::abs(upper - lower) / 2))
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
      const double epsilon, const T lower, const T upper,
      std::unique_ptr<NumericalMechanism> sum_mechanism,
      std::unique_ptr<NumericalMechanism> count_mechanism)
      : BoundedMean<T>(epsilon),
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

 protected:
  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    RETURN_IF_ERROR(ValidateIsPositive(privacy_budget, "Privacy budget",
                                       absl::StatusCode::kFailedPrecondition));
    const double midpoint = lower_ + ((upper_ - lower_) / 2);

    // Split privacy budget for sum and count mechanisms.
    const double sum_budget = privacy_budget / 2;
    const double count_budget = privacy_budget - sum_budget;

    const double noised_count =
        std::max(1.0, static_cast<double>(count_mechanism_->AddNoise(
                          partial_count_, count_budget)));
    const double noised_normalized_sum = sum_mechanism_->AddNoise(
        partial_sum_ - (partial_count_ * midpoint), sum_budget);
    const double mean = (noised_normalized_sum / noised_count) + midpoint;

    Output output;
    AddToOutput<double>(&output, Clamp<double>(lower_, upper_, mean));
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
    partial_sum_ += Clamp<T>(lower_, upper_, input) * num_of_entries;
    partial_count_ += num_of_entries;
  }

 private:
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
      const double epsilon, const double l0_sensitivity,
      const double max_contributions_per_partition,
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder,
      std::unique_ptr<NumericalMechanism> count_mechanism,
      std::unique_ptr<ApproxBounds<T>> approx_bounds)
      : BoundedMean<T>(epsilon),
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

  double GetEpsilon() const override {
    return approx_bounds_->GetEpsilon() + Algorithm<T>::GetEpsilon();
  }

  // Returns the epsilon used to calculate approximate bounds.
  double GetBoundingEpsilon() const { return approx_bounds_->GetEpsilon(); }

  // Returns the epsilon used to calculate the noisy mean.
  double GetAggregationEpsilon() const { return Algorithm<T>::GetEpsilon(); }

 protected:
  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    RETURN_IF_ERROR(ValidateIsPositive(privacy_budget, "Privacy budget",
                                       absl::StatusCode::kFailedPrecondition));

    // Split privacy budget.  We use 1/2 for approx bounds, 1/4 for count and
    // 1/4 for sum.
    const double bounds_budget = privacy_budget / 2;
    const double sum_budget = (privacy_budget - bounds_budget) / 2;
    const double count_budget = privacy_budget - bounds_budget - sum_budget;

    Output output;

    // Use a fraction of the privacy budget to find the approximate bounds.
    ASSIGN_OR_RETURN(Output bounds, approx_bounds_->PartialResult(
                                        bounds_budget, noise_interval_level));
    const T lower = GetValue<T>(bounds.elements(0).value());
    const T upper = GetValue<T>(bounds.elements(1).value());
    RETURN_IF_ERROR(BoundedMean<T>::CheckBounds(lower, upper));

    // To find the sum, pass the identity function as the transform.
    ASSIGN_OR_RETURN(const T sum,
                     approx_bounds_->template ComputeFromPartials<T>(
                         pos_sum_, neg_sum_, [](T x) { return x; }, lower,
                         upper, partial_count_));

    // Populate the bounding report with ApproxBounds information.
    *(output.mutable_error_report()->mutable_bounding_report()) =
        approx_bounds_->GetBoundingReport(lower, upper);

    // Construct the sum mechanism mechanism with the obtained bounds.
    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> sum_mechanism,
        BoundedMean<T>::BuildMechanismForNormalizedSum(
            mechanism_builder_->Clone(), Algorithm<T>::GetEpsilon(),
            l0_sensitivity_, max_contributions_per_partition_, lower, upper));

    // We use the midpoint to normalize the sum.
    const double midpoint = lower + (upper - lower) / 2;

    const double noised_count =
        std::max(1.0, static_cast<double>(count_mechanism_->AddNoise(
                          partial_count_, count_budget)));
    const double normalized_sum =
        sum_mechanism->AddNoise(sum - partial_count_ * midpoint, sum_budget);
    const double mean = normalized_sum / noised_count + midpoint;

    // Add mean to output and return the result.
    AddToOutput<double>(&output, Clamp<double>(lower, upper, mean));
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
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_;
  const double l0_sensitivity_;
  const double max_contributions_per_partition_;

  // Approx bounds instance to automatically determining bounds.
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;
};

template <typename T>
class BoundedMean<T>::Builder
    : public BoundedAlgorithmBuilder<T, BoundedMean<T>, Builder> {
  using AlgorithmBuilder =
      differential_privacy::AlgorithmBuilder<T, BoundedMean<T>, Builder>;
  using BoundedBuilder = BoundedAlgorithmBuilder<T, BoundedMean<T>, Builder>;

 private:
  base::StatusOr<std::unique_ptr<BoundedMean<T>>> BuildBoundedAlgorithm()
      override {
    // We have to check epsilon now, otherwise the split during ApproxBounds
    // construction might make the error message confusing.
    RETURN_IF_ERROR(
        ValidateIsFiniteAndPositive(AlgorithmBuilder::GetEpsilon(), "Epsilon"));
    // Ensure that either bounds are manually set or ApproxBounds is made.
    RETURN_IF_ERROR(BoundedBuilder::BoundsSetup());

    // The count noising doesn't depend on the bounds, so we can always
    // construct the mechanism we use for it here.
    std::unique_ptr<NumericalMechanism> count_mechanism;
    ASSIGN_OR_RETURN(
        count_mechanism,
        AlgorithmBuilder::GetMechanismBuilderClone()
            ->SetEpsilon(BoundedBuilder::GetRemainingEpsilon().value())
            .SetL0Sensitivity(
                AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1))
            .SetLInfSensitivity(
                AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1))
            .Build());

    if (BoundedBuilder::BoundsAreSet()) {
      RETURN_IF_ERROR(CheckBounds(BoundedBuilder::GetLower().value(),
                                  BoundedBuilder::GetUpper().value()));
      ASSIGN_OR_RETURN(
          std::unique_ptr<NumericalMechanism> sum_mechanism,
          BoundedMean<T>::BuildMechanismForNormalizedSum(
              AlgorithmBuilder::GetMechanismBuilderClone(),
              BoundedBuilder::GetRemainingEpsilon().value(),
              AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1),
              AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1),
              BoundedBuilder::GetLower().value(),
              BoundedBuilder::GetUpper().value()));

      // Construct BoundedSum with fixed bounds.
      return std::unique_ptr<BoundedMean<T>>(new BoundedMeanWithFixedBounds<T>(
          BoundedBuilder::GetEpsilon().value(),
          BoundedBuilder::GetLower().value(),
          BoundedBuilder::GetUpper().value(), std::move(sum_mechanism),
          std::move(count_mechanism)));
    }

    // Construct BoundedMean.
    auto mech_builder = AlgorithmBuilder::GetMechanismBuilderClone();
    return std::unique_ptr<BoundedMean<T>>(new BoundedMeanWithApproxBounds<T>(
        BoundedBuilder::GetRemainingEpsilon().value(),
        AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1),
        AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1),
        std::move(mech_builder), std::move(count_mechanism),
        std::move(BoundedBuilder::MoveApproxBoundsPointer())));
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_MEAN_H_
