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

#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounded-variance.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

// Bounded Stddev is deprecated in favor of using variance and taking the sqrt
// of the result.

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
template <typename T>
class ABSL_DEPRECATED(
    "Use BoundedVariance instead and take the sqrt of the result")
    BoundedStandardDeviation : public Algorithm<T> {
  static_assert(
      std::is_arithmetic<T>::value,
      "BoundedStandardDeviation can only be used for arithmetic types");

 public:
  // Builder for BoundedStandardDeviation algorithm.
  class ABSL_DEPRECATED(
      "Use BoundedVariance instead and take the sqrt of the result") Builder;

  void AddEntry(const T& t) override { variance_->AddEntry(t); }

  // Returns a BoundedVarianceSummary.
  Summary Serialize() const override { return variance_->Serialize(); }

  // Merges from BoundedVarianceSummary.
  absl::Status Merge(const Summary& summary) override {
    return variance_->Merge(summary);
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BoundedStandardDeviation<T>);
    if (variance_) {
      memory += variance_->MemoryUsed();
    }
    return memory;
  }

  double GetEpsilon() const override { return variance_->GetEpsilon(); }

  // Returns the epsilon used to calculate approximate bounds. If approximate
  // bounds are not used, returns 0.
  double GetBoundingEpsilon() const { return variance_->GetBoundingEpsilon(); }

  // Returns the epsilon used to calculate the noisy mean. If bounds are
  // specified explicitly, this will be the total epsilon used by the algorithm.
  double GetAggregationEpsilon() const {
    return variance_->GetAggregationEpsilon();
  }

 private:
  BoundedStandardDeviation(const double epsilon,
                           std::unique_ptr<BoundedVariance<T>> variance)
      : Algorithm<T>(epsilon), variance_(std::move(variance)) {}

  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    ASSIGN_OR_RETURN(Output variance_output,
                     variance_->PartialResult(noise_interval_level));
    double stdev = std::sqrt(GetValue<double>(variance_output));
    SetValue<double>(variance_output.mutable_elements(0)->mutable_value(),
                     stdev);
    return variance_output;
  }

  void ResetState() override { variance_->Reset(); }
  std::unique_ptr<BoundedVariance<T>> variance_;
};

template <typename T>
class BoundedStandardDeviation<T>::Builder {
 public:
  BoundedStandardDeviation<T>::Builder& SetEpsilon(double epsilon) {
    variance_builder_.SetEpsilon(epsilon);
    return *this;
  }

  BoundedStandardDeviation<T>::Builder& SetDelta(double delta) {
    variance_builder_.SetDelta(delta);
    return *this;
  }

  BoundedStandardDeviation<T>::Builder& SetMaxPartitionsContributed(
      int max_partitions_contributed) {
    variance_builder_.SetMaxPartitionsContributed(max_partitions_contributed);
    return *this;
  }

  BoundedStandardDeviation<T>::Builder& SetMaxContributionsPerPartition(
      int max_contributions_per_partition) {
    variance_builder_.SetMaxContributionsPerPartition(
        max_contributions_per_partition);
    return *this;
  }

  BoundedStandardDeviation<T>::Builder& SetUpper(T upper) {
    variance_builder_.SetUpper(upper);
    return *this;
  }

  BoundedStandardDeviation<T>::Builder& SetLower(T lower) {
    variance_builder_.SetLower(lower);
    return *this;
  }

  BoundedStandardDeviation<T>::Builder& SetApproxBounds(
      std::unique_ptr<ApproxBounds<T>> approx_bounds) {
    variance_builder_.SetApproxBounds(std::move(approx_bounds));
    return *this;
  }

  BoundedStandardDeviation<T>::Builder& SetLaplaceMechanism(
      std::unique_ptr<NumericalMechanismBuilder> builder) {
    variance_builder_.SetLaplaceMechanism(std::move(builder));
    return *this;
  }

  absl::StatusOr<std::unique_ptr<BoundedStandardDeviation<T>>> Build() {
    ASSIGN_OR_RETURN(std::unique_ptr<BoundedVariance<T>> variance,
                     variance_builder_.Build());
    const double epsilon = variance->GetEpsilon();
    return absl::WrapUnique(
        new BoundedStandardDeviation(epsilon, std::move(variance)));
  }

 private:
  typename BoundedVariance<T>::Builder variance_builder_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_STANDARD_DEVIATION_H_
