//
// Copyright 2024 Google LLC
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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_APPROX_BOUNDS_AS_BOUNDS_PROVIDER_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_APPROX_BOUNDS_AS_BOUNDS_PROVIDER_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounds-provider.h"
#include "algorithms/internal/clamped-calculation-without-bounds.h"
#include "base/status_macros.h"

namespace differential_privacy {

// Adapter to use ApproxBounds as a BoundsProvider.  Wraps an ApproxBounds
// object.
template <typename T>
class ApproxBoundsAsBoundsProvider final : public BoundsProvider<T> {
 public:
  explicit ApproxBoundsAsBoundsProvider(
      std::unique_ptr<ApproxBounds<T>> approx_bounds)
      : approx_bounds_(std::move(approx_bounds)) {}

  void AddEntry(const T& entry) override { approx_bounds_->AddEntry(entry); }

  virtual absl::StatusOr<BoundsResult<T>> FinalizeAndCalculateBounds()
      override {
    ASSIGN_OR_RETURN(Output output, approx_bounds_->PartialResult());
    BoundsResult<T> result;
    result.lower_bound = GetValue<T>(output.elements(0).value());
    result.upper_bound = GetValue<T>(output.elements(1).value());
    return result;
  }

  BoundingReport GetBoundingReport(
      const BoundsResult<T>& bounds_result) override {
    return approx_bounds_->GetBoundingReport(bounds_result.lower_bound,
                                             bounds_result.upper_bound);
  }

  void Reset() override { approx_bounds_->Reset(); }

  int64_t MemoryUsed() const override {
    return sizeof(*this) + approx_bounds_->MemoryUsed();
  }

  double GetEpsilon() const override { return approx_bounds_->GetEpsilon(); }

  double GetDelta() const override { return approx_bounds_->GetDelta(); }

  BoundsSummary Serialize() const override {
    Summary summary = approx_bounds_->Serialize();
    BoundsSummary bounds_summary;
    summary.data().UnpackTo(bounds_summary.mutable_approx_bounds_summary());
    return bounds_summary;
  }

  absl::Status Merge(const BoundsSummary& bounds_summary) {
    if (!bounds_summary.has_approx_bounds_summary()) {
      return absl::InternalError("approx_bounds_summary must be set");
    }
    Summary summary;
    const bool pack_successful = summary.mutable_data()->PackFrom(
        bounds_summary.approx_bounds_summary());
    if (!pack_successful) {
      return absl::InternalError("Cannot pack bounds summary");
    }
    return approx_bounds_->Merge(summary);
  }

  std::unique_ptr<internal::ClampedCalculationWithoutBounds<T>>
  CreateClampedCalculationWithoutBounds() const override {
    return approx_bounds_->CreateClampedCalculationWithoutBounds();
  }

  // Returns a pointer to the wrapped ApproxBounds object.  Does not transfer
  // ownership.  Only use for testing.
  ApproxBounds<T>* GetApproxBoundsForTesting() { return approx_bounds_.get(); }

 private:
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_APPROX_BOUNDS_AS_BOUNDS_PROVIDER_H_
