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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_STANDARD_DEVIATION_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_STANDARD_DEVIATION_H_

#include <type_traits>

#include "absl/memory/memory.h"
#include "base/status.h"
#include "algorithms/algorithm.h"
#include "algorithms/bounded-algorithm.h"
#include "algorithms/bounded-variance.h"
#include "algorithms/numerical-mechanisms.h"

namespace differential_privacy {

// Incrementally provides a differentially private standard deviation for values
// in the range [lower..upper]. Values outside of this range will be clamped so
// they lie in the range. The output will also be clamped between 0 and (upper -
// lower).
//
// The implementation simply computes the bounded variance and takes the square
// root, which is differentially private by the post-processing theorem. It
// relies on the fact that the bounded variance algorithm guarantees that the
// output is non-negative.
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
class BoundedStandardDeviation : public Algorithm<T> {
 public:
  // Builder for BoundedStandardDeviation algorithm.
  class Builder : public BoundedAlgorithmBuilder<T, BoundedStandardDeviation<T>,
                                                 Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, BoundedStandardDeviation<T>,
                                               Builder>;
    using BoundedBuilder =
        BoundedAlgorithmBuilder<T, BoundedStandardDeviation<T>, Builder>;

   private:
    base::StatusOr<std::unique_ptr<BoundedStandardDeviation<T>>>
    BuildAlgorithm() override {
      // Set bounding info if appropriate.
      if (BoundedBuilder::lower_.has_value()) {
        variance_builder_.SetLower(BoundedBuilder::lower_.value());
      }
      if (BoundedBuilder::upper_.has_value()) {
        variance_builder_.SetUpper(BoundedBuilder::upper_.value());
      }
      if (BoundedBuilder::approx_bounds_) {
        variance_builder_.SetApproxBounds(
            std::move(BoundedBuilder::approx_bounds_));
      }

      // Construct bounded variance.
      std::unique_ptr<BoundedVariance<T>> variance;
      auto mech_builder = AlgorithmBuilder::mechanism_builder_->Clone();
      ASSIGN_OR_RETURN(
          variance,
          variance_builder_.SetEpsilon(AlgorithmBuilder::epsilon_.value())
              .SetLaplaceMechanism(std::move(mech_builder))
              .Build());

      return absl::WrapUnique(new BoundedStandardDeviation(
          AlgorithmBuilder::epsilon_.value(), std::move(variance)));
    }

    typename BoundedVariance<T>::Builder variance_builder_;
  };

  void AddEntry(const T& t) override { variance_->AddEntry(t); }

  // Returns a BoundedVarianceSummary.
  Summary Serialize() override { return variance_->Serialize(); }

  // Merges from BoundedVarianceSummary.
  base::Status Merge(const Summary& summary) override {
    return variance_->Merge(summary);
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BoundedStandardDeviation<T>);
    if (variance_) {
      memory += variance_->MemoryUsed();
    }
    return memory;
  }

 private:
  BoundedStandardDeviation(const double epsilon,
                           std::unique_ptr<BoundedVariance<T>> variance)
      : Algorithm<T>(epsilon), variance_(std::move(variance)) {}

  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    ASSIGN_OR_RETURN(
        Output variance_output,
        variance_->PartialResult(privacy_budget, noise_interval_level));
    double stdev = std::sqrt(GetValue<double>(variance_output));
    SetValue<double>(variance_output.mutable_elements(0)->mutable_value(),
                     stdev);
    return variance_output;
  }

  void ResetState() override { variance_->Reset(); }
  std::unique_ptr<BoundedVariance<T>> variance_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_STANDARD_DEVIATION_H_
