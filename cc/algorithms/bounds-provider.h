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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_BOUNDS_PROVIDER_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_BOUNDS_PROVIDER_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/internal/clamped-calculation-without-bounds.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"

namespace differential_privacy {

template <typename T>
struct BoundsResult {
  T lower_bound;
  T upper_bound;
};

// BoundsProvider is an interface for algorithms that provide automatic bounds
// to be used in bounding algorithms.
template <typename T>
class BoundsProvider {
 public:
  virtual void AddEntry(const T& entry) = 0;

  // This method should only be called once and Reset() should be called
  // afterwards.
  virtual absl::StatusOr<BoundsResult<T>> FinalizeAndCalculateBounds() = 0;

  virtual BoundingReport GetBoundingReport(
      const BoundsResult<T>& bounds_result) = 0;

  // Reset state for another calculation.
  virtual void Reset() = 0;

  // Provides an estimate for the memory used for this object.
  virtual int64_t MemoryUsed() const = 0;

  // Getter for epsilon and delta for privacy budget calculation.
  virtual double GetEpsilon() const = 0;
  virtual double GetDelta() const = 0;

  // Serialization on merging with serialized data.
  virtual BoundsSummary Serialize() const = 0;
  virtual absl::Status Merge(const BoundsSummary& bounds_summary) = 0;

  // Create a new clamped calculation that can be used with this bounds
  // provider.
  virtual std::unique_ptr<internal::ClampedCalculationWithoutBounds<T>>
  CreateClampedCalculationWithoutBounds() const = 0;

  virtual ~BoundsProvider() = default;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_BOUNDS_PROVIDER_H_
