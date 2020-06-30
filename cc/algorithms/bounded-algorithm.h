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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_ALGORITHM_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_ALGORITHM_H_

#include <memory>

#include "base/status.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "base/status.h"

namespace differential_privacy {

// BoundedAlgorithmBuilder is used to build algorithms which need lower and
// upper input bounds to determine sensitivity and/or clamp inputs. It provides
// three ways to provide bounds for an algorithm built by this type of builder:
//
//   1. Manually set bounds using the SetLower and SetUpper functions.
//   2. Automatically determine bounds with manually set options. To do this,
//      pass in a constructed ApproxBound into the SetApproxBounds function.
//   3. Automatically determine bounds with default options. If no manual bounds
//      or ApproxBounds algorithm are passed in, then a default ApproxBounds
//      may be constructed. Child builders can call BoundSettingSetup() upon
//      Build to do this.
//
// Currently, all bounded algorithms use the Laplace mechanism.
template <typename T, class Algorithm, class Builder>
class BoundedAlgorithmBuilder : public AlgorithmBuilder<T, Algorithm, Builder> {
  using AlgorithmBuilder =
      differential_privacy::AlgorithmBuilder<T, Algorithm, Builder>;

 public:
  Builder& SetLower(T lower) {
    lower_ = lower;
    return *static_cast<Builder*>(this);
  }

  Builder& SetUpper(T upper) {
    upper_ = upper;
    return *static_cast<Builder*>(this);
  }

  // ClearBounds resets the builder. Erases bounds and bounding objects that
  // were previously set.
  Builder& ClearBounds() {
    lower_.reset();
    upper_.reset();
    approx_bounds_ = nullptr;
    return *static_cast<Builder*>(this);
  }

  // Setting ApproxBounds removes manually set bounds. If automatic bounds are
  // desired, this field is optional.
  Builder& SetApproxBounds(std::unique_ptr<ApproxBounds<T>> approx_bounds) {
    ClearBounds();
    approx_bounds_ = std::move(approx_bounds);
    return *static_cast<Builder*>(this);
  }

 protected:
  // Returns whether bounds have been set for this builder.
  inline bool BoundsAreSet() {
    return lower_.has_value() && upper_.has_value();
  }

  base::Status BoundsSetup() {
    // If either bound is not set and we do not have an ApproxBounds,
    // construct the default one.
    if (!BoundsAreSet() && !approx_bounds_) {
      auto mech_builder = AlgorithmBuilder::mechanism_builder_->Clone();
      ASSIGN_OR_RETURN(approx_bounds_,
                       typename ApproxBounds<T>::Builder()
                           .SetEpsilon(AlgorithmBuilder::epsilon_.value())
                           .SetLaplaceMechanism(std::move(mech_builder))
                           .Build());
    }
    return base::OkStatus();
  }

  // Bounds are optional and do not need to be set.  If they are not set,
  // automatic bounds will be determined.
  absl::optional<T> lower_;
  absl::optional<T> upper_;

  // Used to automatically determine approximate mimimum and maximum to become
  // lower and upper bounds, respectively.
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_ALGORITHM_H_
