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

#include <type_traits>

#include "google/protobuf/any.pb.h"
#include "absl/random/distributions.h"
#include "base/status.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounded-algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {

// Incrementally provides a differentially private average.
// All input vales are normalized to be their difference from the middle of the
// input range. That allows us to calculate the sum of all input values with
// half the sensitivity it would otherwise take for better accuracy (as compared
// to doing noisy sum / noisy count). This algorithm is taken from section 2.5.5
// of the following book (algorithm 2.4):
// https://books.google.com/books?id=WFttDQAAQBAJ&pg=PA24#v=onepage&q&f=false
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
class BoundedMean : public Algorithm<T> {
 public:
  // Builder for BoundedMean algorithm.
  class Builder : public BoundedAlgorithmBuilder<T, BoundedMean<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, BoundedMean<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, BoundedMean<T>, Builder>;

   public:
    // For integral type, check for no overflow in the subtraction.
    template <typename T2 = T,
              std::enable_if_t<std::is_integral<T2>::value>* = nullptr>
    static base::Status CheckBounds(T lower, T upper) {
      T subtract_result;
      if (!SafeSubtract(upper, lower, &subtract_result)) {
        return base::InvalidArgumentError(
            "Upper - lower caused integer overflow.");
      }
      return base::OkStatus();
    }

    // No checks for floating point type.
    template <typename T2 = T,
              std::enable_if_t<std::is_floating_point<T2>::value>* = nullptr>
    static base::Status CheckBounds(T lower, T upper) {
      return base::OkStatus();
    }

   private:
    base::StatusOr<std::unique_ptr<BoundedMean<T>>> BuildAlgorithm() override {
      // Ensure that either bounds are manually set or ApproxBounds is made.
      RETURN_IF_ERROR(BoundedBuilder::BoundsSetup());

      // If manual bounding, check bounds and construct mechanism so we can fail
      // on build if sensitivity is inappropriate.
      std::unique_ptr<NumericalMechanism> sum_mechanism = nullptr;
      if (BoundedBuilder::BoundsAreSet()) {
        RETURN_IF_ERROR(CheckBounds(BoundedBuilder::lower_.value(),
                                    BoundedBuilder::upper_.value()));
        ASSIGN_OR_RETURN(
            sum_mechanism,
            BuildSumMechanism(AlgorithmBuilder::mechanism_builder_->Clone(),
                              AlgorithmBuilder::epsilon_.value(),
                              AlgorithmBuilder::l0_sensitivity_.value_or(1),
                              AlgorithmBuilder::linf_sensitivity_.value_or(1),
                              BoundedBuilder::upper_.value(),
                              BoundedBuilder::lower_.value()));
      }

      // The count noising doesn't depend on the bounds, so we can always
      // construct the mechanism we use for it here.
      std::unique_ptr<NumericalMechanism> count_mechanism;
      ASSIGN_OR_RETURN(
          count_mechanism,
          AlgorithmBuilder::mechanism_builder_
              ->SetEpsilon(AlgorithmBuilder::epsilon_.value())
              .SetL0Sensitivity(AlgorithmBuilder::l0_sensitivity_.value_or(1))
              .SetLInfSensitivity(
                  AlgorithmBuilder::linf_sensitivity_.value_or(1))
              .Build());

      // Construct BoundedMean.
      auto mech_builder = AlgorithmBuilder::mechanism_builder_->Clone();
      return absl::WrapUnique(
          new BoundedMean(AlgorithmBuilder::epsilon_.value(),
                          BoundedBuilder::lower_.value_or(0),
                          BoundedBuilder::upper_.value_or(0),
                          AlgorithmBuilder::l0_sensitivity_.value_or(1),
                          AlgorithmBuilder::linf_sensitivity_.value_or(1),
                          std::move(mech_builder), std::move(sum_mechanism),
                          std::move(count_mechanism),
                          std::move(BoundedBuilder::approx_bounds_)));
    }
  };

  void AddEntry(const T& t) override {
    if (std::isnan(t)) {
      return;
    }
    ++raw_count_;

    if (!approx_bounds_) {
      pos_sum_[0] += Clamp<T>(lower_, upper_, t);
    } else {
      approx_bounds_->AddEntry(t);

      // Find partial sums.
      if (t >= 0) {
        approx_bounds_->template AddToPartialSums<T>(&pos_sum_, t);
      } else {
        approx_bounds_->template AddToPartialSums<T>(&neg_sum_, t);
      }
    }
  }

  Summary Serialize() override {
    // Create BoundedMeanSummary.
    BoundedMeanSummary bm_summary;
    bm_summary.set_count(raw_count_);
    for (T x : pos_sum_) {
      SetValue(bm_summary.add_pos_sum(), x);
    }
    for (T x : neg_sum_) {
      SetValue(bm_summary.add_neg_sum(), x);
    }
    if (approx_bounds_) {
      Summary approx_bounds_summary = approx_bounds_->Serialize();
      approx_bounds_summary.data().UnpackTo(
          bm_summary.mutable_bounds_summary());
    }

    // Create Summary.
    Summary summary;
    summary.mutable_data()->PackFrom(bm_summary);
    return summary;
  }

  base::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return base::InvalidArgumentError(
          "Cannot merge summary with no bounded mean data.");
    }

    // Add counts and bounded sums.
    BoundedMeanSummary bm_summary;
    if (!summary.data().UnpackTo(&bm_summary)) {
      return base::InvalidArgumentError(
          "Bounded mean summary unable to be unpacked.");
    }
    raw_count_ += bm_summary.count();
    if (pos_sum_.size() != bm_summary.pos_sum_size() ||
        neg_sum_.size() != bm_summary.neg_sum_size()) {
      return base::InvalidArgumentError(
          "Merged BoundedMeans must have equal number of partial sums.");
    }
    for (int i = 0; i < pos_sum_.size(); ++i) {
      pos_sum_[i] += GetValue<T>(bm_summary.pos_sum(i));
    }
    for (int i = 0; i < neg_sum_.size(); ++i) {
      neg_sum_[i] += GetValue<T>(bm_summary.neg_sum(i));
    }
    if (approx_bounds_) {
      Summary approx_bounds_summary;
      approx_bounds_summary.mutable_data()->PackFrom(
          bm_summary.bounds_summary());
      RETURN_IF_ERROR(approx_bounds_->Merge(approx_bounds_summary));
    }

    return base::OkStatus();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BoundedMean<T>) +
                   sizeof(T) * (pos_sum_.capacity() + neg_sum_.capacity());
    if (approx_bounds_) {
      memory += approx_bounds_->MemoryUsed();
    }
    if (sum_mechanism_) {
      memory += sum_mechanism_->MemoryUsed();
    }
    if (mechanism_builder_) {
      memory += sizeof(*mechanism_builder_);
    }
    return memory;
  }

 private:
  BoundedMean(const double epsilon, T lower, T upper,
              const double l0_sensitivity, const double linf_sensitivity,
              std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
              std::unique_ptr<NumericalMechanism> sum_mechanism,
              std::unique_ptr<NumericalMechanism> count_mechanism,
              std::unique_ptr<ApproxBounds<T>> approx_bounds = nullptr)
      : Algorithm<T>(epsilon),
        raw_count_(0),
        lower_(lower),
        upper_(upper),
        midpoint_(lower + (upper - lower) / 2),
        l0_sensitivity_(l0_sensitivity),
        linf_sensitivity_(linf_sensitivity),
        mechanism_builder_(std::move(mechanism_builder)),
        sum_mechanism_(std::move(sum_mechanism)),
        count_mechanism_(std::move(count_mechanism)),
        approx_bounds_(std::move(approx_bounds)) {
    // If automatically determining bounds, we need partial sums for each bin
    // of the ApproxBounds logarithmic histogram. Otherwise, we only need to
    // store one already-clamped sum.
    if (approx_bounds_) {
      pos_sum_.resize(approx_bounds_->NumPositiveBins(), 0);
      neg_sum_.resize(approx_bounds_->NumPositiveBins(), 0);
    } else {
      pos_sum_.push_back(0);
    }
  }

  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    DCHECK_GT(privacy_budget, 0.0)
        << "Privacy budget should be greater than zero.";
    if (privacy_budget == 0.0) return Output();
    double sum = 0;
    double remaining_budget = privacy_budget;
    Output output;

    // Find bounds and sum.
    if (approx_bounds_) {
      // Use a fraction of the privacy budget to find the approximate bounds.
      double bounds_budget = privacy_budget / 2;
      remaining_budget -= bounds_budget;
      ASSIGN_OR_RETURN(Output bounds, approx_bounds_->PartialResult(
                                          bounds_budget, noise_interval_level));
      lower_ = GetValue<T>(bounds.elements(0).value());
      upper_ = GetValue<T>(bounds.elements(1).value());
      RETURN_IF_ERROR(Builder::CheckBounds(lower_, upper_));
      midpoint_ = lower_ + (upper_ - lower_) / 2;

      // To find the sum, pass the identity function as the transform.
      sum = approx_bounds_->template ComputeFromPartials<T>(
          pos_sum_, neg_sum_, [](T x) { return x; }, lower_, upper_,
          raw_count_);

      // Populate the bounding report with ApproxBounds information.
      *(output.mutable_error_report()->mutable_bounding_report()) =
          approx_bounds_->GetBoundingReport(lower_, upper_);

      // Clear the mechanism. The sensitivity might have changed.
      sum_mechanism_.reset();
    } else {
      // Manual bounds were set and clamping was done upon adding entries.
      sum = pos_sum_[0];
    }

    // Construct mechanism if needed.
    if (!sum_mechanism_) {
      ASSIGN_OR_RETURN(
          sum_mechanism_,
          BuildSumMechanism(mechanism_builder_->Clone(),
                            Algorithm<T>::GetEpsilon(), l0_sensitivity_,
                            linf_sensitivity_, lower_, upper_));
    }

    double count_budget = remaining_budget / 2;
    remaining_budget -= count_budget;
    double noised_count =
        std::max(1.0, count_mechanism_->AddNoise(raw_count_, count_budget));
    double normalized_sum = sum_mechanism_->AddNoise(
        sum - raw_count_ * midpoint_, remaining_budget);
    double average = normalized_sum / noised_count + midpoint_;
    AddToOutput<double>(&output, Clamp<double>(lower_, upper_, average));
    return output;
  }

  void ResetState() override {
    std::fill(pos_sum_.begin(), pos_sum_.end(), 0);
    std::fill(neg_sum_.begin(), neg_sum_.end(), 0);
    raw_count_ = 0;
    if (approx_bounds_) {
      approx_bounds_->Reset();
      sum_mechanism_ = nullptr;
    }
  }

  static base::StatusOr<std::unique_ptr<NumericalMechanism>> BuildSumMechanism(
      std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
      const double epsilon, const double l0_sensitivity,
      const double linf_sensitivity, const T lower, const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(linf_sensitivity * (std::abs(upper - lower) / 2))
        .Build();
  }

  // Vectors of partial values stored for automatic clamping.
  std::vector<T> pos_sum_, neg_sum_;

  size_t raw_count_;
  T lower_, upper_;
  double midpoint_;

  // Used to construct mechanism once bounds are obtained for auto-bounding.
  std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder_;
  const double l0_sensitivity_;
  const double linf_sensitivity_;

  // The count and the sum will have different sensitivites, so we need
  // different mechanisms to noise them.
  std::unique_ptr<NumericalMechanism> sum_mechanism_;
  std::unique_ptr<NumericalMechanism> count_mechanism_;

  // If this is not nullptr, we are automatically determining bounds. Otherwise,
  // lower and upper contain the manually set bounds.
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_MEAN_H_
