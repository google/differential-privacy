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

#include <type_traits>

#include "google/protobuf/any.pb.h"
#include "absl/memory/memory.h"
#include "base/status.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounded-algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/util.h"

namespace differential_privacy {

// Incrementally provides a differentially private variance for values in the
// range [lower..upper]. Values outside of this range will be clamped so they
// lie in the range. The output will also be clamped between 0 and (upper -
// lower)^2. Since the result is guaranteed to be positive, this algorithm can
// be used to compute a differentially private standard deviation.
//
// The algorithm uses O(1) memory and runs in O(n) time where n is the size of
// the dataset, making it a fast and efficient. The amount of noise added grows
// quadratically in (upper - lower) and decreases linearly in n, so it might not
// produce good results unless n >> (upper - lower)^2.
//
// The algorithm is a variation of the algorithm for differentially private mean
// from "Differential Privacy: From Theory to Practice", section 2.5.5:
// https://books.google.com/books?id=WFttDQAAQBAJ&pg=PA24#v=onepage&q&f=false
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
class BoundedVariance : public Algorithm<T> {
 public:
  // Builder for BoundedVariance algorithm.
  class Builder
      : public BoundedAlgorithmBuilder<T, BoundedVariance<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, BoundedVariance<T>, Builder>;
    using BoundedBuilder =
        BoundedAlgorithmBuilder<T, BoundedVariance<T>, Builder>;

   public:
    // For integral type, check for no overflow in subtraction squared.
    template <typename T2 = T,
              std::enable_if_t<std::is_integral<T2>::value>* = nullptr>
    static base::Status CheckBounds(T lower, T upper) {
      if (lower > upper) {
        return base::InvalidArgumentError(
            "Lower cannot be greater than upper.");
      }
      T subtract_result, square_result;
      if (!SafeSubtract(upper, lower, &subtract_result) ||
          !SafeSquare(subtract_result, &square_result)) {
        return base::InvalidArgumentError(
            "Sensitivity calculation caused integer overflow.");
      }
      if (upper > sqrt(std::numeric_limits<T>::max()) ||
          lower < -1 * sqrt(std::numeric_limits<T>::max())) {
        return base::InvalidArgumentError(
            "Squaring the bounds caused overflow.");
      }
      return base::OkStatus();
    }

    template <typename T2 = T,
              std::enable_if_t<std::is_floating_point<T2>::value>* = nullptr>
    static base::Status CheckBounds(T lower, T upper) {
      if (lower > upper) {
        return base::InvalidArgumentError(
            "Lower cannot be greater than upper.");
      }
      return base::OkStatus();
    }

   private:
    base::StatusOr<std::unique_ptr<BoundedVariance<T>>> BuildAlgorithm()
        override {
      // Ensure that either bounds are manually set or ApproxBounds is made.
      RETURN_IF_ERROR(BoundedBuilder::BoundsSetup());

      // If manual bounding, check bounds and construct mechanism so we can fail
      // on build if sensitivity is inappropriate.
      std::unique_ptr<NumericalMechanism> sum_mechanism = nullptr;
      std::unique_ptr<NumericalMechanism> sos_mechanism = nullptr;
      if (BoundedBuilder::BoundsAreSet()) {
        RETURN_IF_ERROR(CheckBounds(BoundedBuilder::lower_.value(),
                                    BoundedBuilder::upper_.value()));
        ASSIGN_OR_RETURN(
            sum_mechanism,
            BuildSumMechanism(AlgorithmBuilder::mechanism_builder_->Clone(),
                              AlgorithmBuilder::epsilon_.value(),
                              AlgorithmBuilder::l0_sensitivity_.value_or(1),
                              AlgorithmBuilder::linf_sensitivity_.value_or(1),
                              BoundedBuilder::lower_.value(),
                              BoundedBuilder::upper_.value()));
        ASSIGN_OR_RETURN(sos_mechanism,
                         BuildSumOfSquaresMechanism(
                             AlgorithmBuilder::mechanism_builder_->Clone(),
                             AlgorithmBuilder::epsilon_.value(),
                             AlgorithmBuilder::l0_sensitivity_.value_or(1),
                             AlgorithmBuilder::linf_sensitivity_.value_or(1),
                             BoundedBuilder::lower_.value(),
                             BoundedBuilder::upper_.value()));
      }

      std::unique_ptr<NumericalMechanism> count_mechanism;
      ASSIGN_OR_RETURN(
          count_mechanism,
          AlgorithmBuilder::mechanism_builder_
              ->SetEpsilon(AlgorithmBuilder::epsilon_.value())
              .SetL0Sensitivity(AlgorithmBuilder::l0_sensitivity_.value_or(1))
              .SetLInfSensitivity(
                  AlgorithmBuilder::linf_sensitivity_.value_or(1))
              .Build());

      // Construct bounded variance.
      auto mech_builder = AlgorithmBuilder::mechanism_builder_->Clone();
      return absl::WrapUnique(new BoundedVariance(
          AlgorithmBuilder::epsilon_.value(),
          BoundedBuilder::lower_.value_or(0),
          BoundedBuilder::upper_.value_or(0),
          AlgorithmBuilder::l0_sensitivity_.value_or(1),
          AlgorithmBuilder::linf_sensitivity_.value_or(1),
          std::move(mech_builder), std::move(sum_mechanism),
          std::move(sos_mechanism), std::move(count_mechanism),
          std::move(BoundedBuilder::approx_bounds_)));
    }
  };

  void AddEntry(const T& t) override {
    // Drop value if it is NaN.
    if (std::isnan(t)) {
      return;
    }

    // Count is unaffected by clamping.
    ++raw_count_;

    // If bounds exist, clamp and record. Otherwise, store partial results and
    // feed input into ApproxBounds algorithm.
    if (!approx_bounds_) {
      double clamped = Clamp<double>(lower_, upper_, t);
      pos_sum_[0] += clamped;
      pos_sum_of_squares_[0] += clamped * clamped;
    } else {
      approx_bounds_->AddEntry(t);

      // Add to partial sums and sum of squares.
      auto difference_of_squares = [](T val1, T val2) {
        // Lessen the chance of becoming inf/-inf by calculating it like this.
        return (static_cast<double>(val1) + val2) *
               (static_cast<double>(val1) - val2);
      };
      if (t >= 0) {
        approx_bounds_->template AddToPartialSums<T>(&pos_sum_, t);
        approx_bounds_->template AddToPartials<double>(&pos_sum_of_squares_, t,
                                                       difference_of_squares);
      } else {
        approx_bounds_->template AddToPartialSums<T>(&neg_sum_, t);
        approx_bounds_->template AddToPartials<double>(&neg_sum_of_squares_, t,
                                                       difference_of_squares);
      }
    }
  }

  Summary Serialize() override {
    // Create BoundedVarianceSummary.
    BoundedVarianceSummary bv_summary;
    bv_summary.set_count(raw_count_);
    for (T x : pos_sum_) {
      SetValue(bv_summary.add_pos_sum(), x);
    }
    for (T x : neg_sum_) {
      SetValue(bv_summary.add_neg_sum(), x);
    }
    for (T x : pos_sum_of_squares_) {
      bv_summary.add_pos_sum_of_squares(x);
    }
    for (T x : neg_sum_of_squares_) {
      bv_summary.add_neg_sum_of_squares(x);
    }
    if (approx_bounds_) {
      Summary approx_bounds_summary = approx_bounds_->Serialize();
      approx_bounds_summary.data().UnpackTo(
          bv_summary.mutable_bounds_summary());
    }

    // Create Summary.
    Summary summary;
    summary.mutable_data()->PackFrom(bv_summary);
    return summary;
  }

  base::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return base::InvalidArgumentError(
          "Cannot merge summary with no bounded variance data.");
    }

    // Unpack bounded variance summary.
    BoundedVarianceSummary bv_summary;
    if (!summary.data().UnpackTo(&bv_summary)) {
      return base::InvalidArgumentError(
          "Bounded variance summary unable to be unpacked.");
    }
    if ((approx_bounds_ != nullptr) != bv_summary.has_bounds_summary()) {
      return base::InvalidArgumentError(
          "Merged BoundedVariance must have the same bounding strategy.");
    }
    if (pos_sum_.size() != bv_summary.pos_sum_size() ||
        neg_sum_.size() != bv_summary.neg_sum_size() ||
        pos_sum_of_squares_.size() != bv_summary.pos_sum_of_squares_size() ||
        neg_sum_of_squares_.size() != bv_summary.neg_sum_of_squares_size()) {
      return base::InvalidArgumentError(
          "Merged BoundedVariance must have the same amount of partial "
          "sum or sum of squares values as this BoundedVariance.");
    }

    // Add count and partial values to current ones.
    raw_count_ += bv_summary.count();
    for (int i = 0; i < pos_sum_.size(); ++i) {
      pos_sum_[i] += GetValue<T>(bv_summary.pos_sum(i));
      pos_sum_of_squares_[i] += bv_summary.pos_sum_of_squares(i);
    }
    for (int i = 0; i < neg_sum_.size(); ++i) {
      neg_sum_[i] += GetValue<T>(bv_summary.neg_sum(i));
      neg_sum_of_squares_[i] += bv_summary.neg_sum_of_squares(i);
    }

    // Merge approx bounds if auto-clamping.
    if (approx_bounds_) {
      Summary approx_bounds_summary;
      approx_bounds_summary.mutable_data()->PackFrom(
          bv_summary.bounds_summary());
      RETURN_IF_ERROR(approx_bounds_->Merge(approx_bounds_summary));
    }

    return base::OkStatus();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BoundedVariance<T>) +
                   sizeof(T) * (pos_sum_.capacity() + neg_sum_.capacity()) +
                   sizeof(double) * (pos_sum_of_squares_.capacity() +
                                     neg_sum_of_squares_.capacity());
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
  BoundedVariance(const double epsilon, const T lower, const T upper,
                  const double l0_sensitivity, const double linf_sensitivity,
                  std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
                  std::unique_ptr<NumericalMechanism> sum_mechanism,
                  std::unique_ptr<NumericalMechanism> sos_mechanism,
                  std::unique_ptr<NumericalMechanism> count_mechanism,
                  std::unique_ptr<ApproxBounds<T>> approx_bounds = nullptr)
      : Algorithm<T>(epsilon),
        raw_count_(0),
        lower_(lower),
        upper_(upper),
        l0_sensitivity_(l0_sensitivity),
        linf_sensitivity_(linf_sensitivity),
        mechanism_builder_(std::move(mechanism_builder)),
        sum_mechanism_(std::move(sum_mechanism)),
        sos_mechanism_(std::move(sos_mechanism)),
        count_mechanism_(std::move(count_mechanism)),
        approx_bounds_(std::move(approx_bounds)) {
    // If automatically determining bounds, we need partial values for each bin
    // of the ApproxBounds logarithmic histogram. Otherwise, we only need to
    // store one already-clamped value.
    if (approx_bounds_) {
      pos_sum_.resize(approx_bounds_->NumPositiveBins(), 0);
      neg_sum_.resize(approx_bounds_->NumPositiveBins(), 0);
      pos_sum_of_squares_.resize(approx_bounds_->NumPositiveBins(), 0);
      neg_sum_of_squares_.resize(approx_bounds_->NumPositiveBins(), 0);
    } else {
      pos_sum_.push_back(0);
      pos_sum_of_squares_.push_back(0);
    }
  }

  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    DCHECK_GT(privacy_budget, 0.0)
        << "Privacy budget should be greater than zero.";
    if (privacy_budget == 0.0) return Output();
    double remaining_budget = privacy_budget;
    Output output;

    // We need these values to find the final variance.
    double sum = 0;
    double sos = 0;  // Sum of squares.

    if (approx_bounds_) {
      // Get bounds with a fraction of the privacy budget.
      double bounds_budget = privacy_budget / 2;
      remaining_budget -= bounds_budget;
      ASSIGN_OR_RETURN(Output bounds, approx_bounds_->PartialResult(
                                          bounds_budget, noise_interval_level));
      lower_ = GetValue<T>(bounds.elements(0).value());
      upper_ = GetValue<T>(bounds.elements(1).value());
      RETURN_IF_ERROR(Builder::CheckBounds(lower_, upper_));

      // To find the sum, pass the identity function as the transform.
      sum = approx_bounds_->template ComputeFromPartials<T>(
          pos_sum_, neg_sum_, [](T x) { return x; }, lower_, upper_,
          raw_count_);

      // To find sum of squares, pass the square function.
      sos = approx_bounds_->template ComputeFromPartials<double>(
          pos_sum_of_squares_, neg_sum_of_squares_, [](T x) { return x * x; },
          lower_, upper_, raw_count_);

      // Populate the bounding report with ApproxBounds information.
      *(output.mutable_error_report()->mutable_bounding_report()) =
          approx_bounds_->GetBoundingReport(lower_, upper_);

      // Clear the mechanism. The sensitivity might have changed.
      sum_mechanism_.reset();
      sos_mechanism_.reset();
    } else {
      // In this case, lower and upper were manually set. The clamped partial
      // values are stored and do not need processing.
      sum = pos_sum_[0];
      sos = pos_sum_of_squares_[0];
    }

    // From this point lower_ and upper_ are guaranteed to be set, either from
    // ApproxBounds results or manually at construction. Construct mechanism
    // with the correct noise if needed.
    if (!sum_mechanism_) {
      ASSIGN_OR_RETURN(
          sum_mechanism_,
          BuildSumMechanism(mechanism_builder_->Clone(),
                            Algorithm<T>::GetEpsilon(), l0_sensitivity_,
                            linf_sensitivity_, lower_, upper_));
    }
    if (!sos_mechanism_) {
      ASSIGN_OR_RETURN(
          sos_mechanism_,
          BuildSumOfSquaresMechanism(
              mechanism_builder_->Clone(), Algorithm<T>::GetEpsilon(),
              l0_sensitivity_, linf_sensitivity_, lower_, upper_));
    }

    T sum_midpoint = lower_ + (upper_ - lower_) / 2;
    T sos_midpoint = MidpointOfSquares(lower_, upper_);

    double count_budget = remaining_budget / 4;
    remaining_budget -= count_budget;
    double noised_sum_count =
        count_mechanism_->AddNoise(raw_count_, count_budget);
    remaining_budget -= count_budget;
    double noised_sos_count =
        count_mechanism_->AddNoise(raw_count_, count_budget);

    // Exact output is sum_of_squares/count - sum*sum/(count*count).
    double sum_budget = remaining_budget / 2;
    remaining_budget -= sum_budget;
    double normalized_sum = sum_mechanism_->AddNoise(
        sum - static_cast<double>(raw_count_) * sum_midpoint, sum_budget);
    double normalized_sos = sos_mechanism_->AddNoise(
        sos - static_cast<double>(raw_count_) * sos_midpoint, remaining_budget);

    double mean;
    if (noised_sum_count <= 1) {
      mean = sum_midpoint;
    } else {
      mean = normalized_sum / noised_sum_count + sum_midpoint;
    }

    double mean_of_square;
    if (noised_sos_count <= 1) {
      mean_of_square = sos_midpoint;
    } else {
      mean_of_square = normalized_sos / noised_sos_count + sos_midpoint;
    }

    double noised_variance = mean_of_square - pow(mean, 2);
    AddToOutput<double>(
        &output, Clamp<double>(0.0, IntervalLengthSquared(lower_, upper_) / 4,
                               noised_variance));
    return output;
  }

  void ResetState() override {
    std::fill(pos_sum_.begin(), pos_sum_.end(), 0);
    std::fill(pos_sum_of_squares_.begin(), pos_sum_of_squares_.end(), 0);
    std::fill(neg_sum_.begin(), neg_sum_.end(), 0);
    std::fill(neg_sum_of_squares_.begin(), neg_sum_of_squares_.end(), 0);
    raw_count_ = 0;

    if (approx_bounds_) {
      approx_bounds_->Reset();
      sum_mechanism_ = nullptr;
      sos_mechanism_ = nullptr;
    }
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

  // Returns the width of the range of f(x) = x^2 where the domain of f is
  // [lower, upper].
  static double RangeOfSquares(T lower, T upper) {
    if (0 > lower && 0 < upper) {
      return std::max(lower * lower, upper * upper);
    }
    return std::abs(upper * upper - lower * lower);
  }

  static base::StatusOr<std::unique_ptr<NumericalMechanism>> BuildSumMechanism(
      std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
      const double epsilon, const double l0_sensitivity,
      const double linf_sensitivity, const T lower, const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(linf_sensitivity *
                            static_cast<double>((upper - lower) / 2))
        .Build();
  }

  static base::StatusOr<std::unique_ptr<NumericalMechanism>>
  BuildSumOfSquaresMechanism(
      std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
      const double epsilon, const double l0_sensitivity,
      const double linf_sensitivity, const T lower, const T upper) {
    return mechanism_builder->SetEpsilon(epsilon)
        .SetL0Sensitivity(l0_sensitivity)
        .SetLInfSensitivity(linf_sensitivity *
                            (RangeOfSquares(lower, upper) / 2))
        .Build();
  }

  // Vectors of partial values stored for automatic clamping.
  std::vector<T> pos_sum_, neg_sum_;
  std::vector<double> pos_sum_of_squares_, neg_sum_of_squares_;
  size_t raw_count_;
  T lower_, upper_;

  // Used to construct mechanism once bounds are obtained.
  std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder_;
  const double l0_sensitivity_;
  const double linf_sensitivity_;

  std::unique_ptr<NumericalMechanism> sum_mechanism_;
  std::unique_ptr<NumericalMechanism> sos_mechanism_;
  std::unique_ptr<NumericalMechanism> count_mechanism_;

  // If this is not nullptr, we are automatically determining bounds. Otherwise,
  // lower and upper contain the manually set bounds.
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_VARIANCE_H_
