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
#include <optional>

#include "absl/status/status.h"
#include "base/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "base/status_macros.h"

namespace differential_privacy {

// If the user does not manually specify an ApproxBounds object, we will use
// this fraction of the total epsilon to calculate them.
const double kDefaultBoundsBudgetFraction = 0.5;

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
  // This method needs to be overwritten by childs to build bounded algorithms.
  virtual base::StatusOr<std::unique_ptr<Algorithm>>
  BuildBoundedAlgorithm() = 0;

  // Returns whether bounds have been set for this builder.
  inline bool BoundsAreSet() {
    return lower_.has_value() && upper_.has_value();
  }

  absl::Status BoundsSetup() {
    // If either bound is not set and we do not have an ApproxBounds,
    // construct the default one.
    if (!BoundsAreSet() && !approx_bounds_) {
      double bounds_epsilon =
          AlgorithmBuilder::GetEpsilon().value() * kDefaultBoundsBudgetFraction;
      remaining_epsilon_ =
          AlgorithmBuilder::GetEpsilon().value() - bounds_epsilon;
      auto mech_builder = AlgorithmBuilder::GetMechanismBuilderClone();
      const int max_partitions_contributed =
          AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1);
      const int max_contributions_per_partition =
          AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1);
      ASSIGN_OR_RETURN(approx_bounds_,
                       typename ApproxBounds<T>::Builder()
                           .SetEpsilon(bounds_epsilon)
                           .SetLaplaceMechanism(std::move(mech_builder))
                           .SetMaxPartitionsContributed(
                               max_partitions_contributed)
                           .SetMaxContributionsPerPartition(
                               max_contributions_per_partition)
                           .Build());
    }
    return absl::OkStatus();
  }

  // Returns the epsilon allotted for calculating the aggregation. If bounds
  // are set manually, or an ApproximateBounds object has been manually
  // specified, this will be the full epsilon. If an ApproxBounds was created
  // automatically this will be the full epsilon - epsilon spent on that
  // ApproxBounds. If called before BoundsSetup this will always return the full
  // epsilon.
  absl::optional<double> GetRemainingEpsilon() {
    if (remaining_epsilon_.has_value()) {
      return remaining_epsilon_;
    }
    return AlgorithmBuilder::GetEpsilon();
  }

  std::unique_ptr<ApproxBounds<T>> MoveApproxBoundsPointer() {
    return std::move(approx_bounds_);
  }

  absl::optional<T> GetLower() const { return lower_; }
  absl::optional<T> GetUpper() const { return upper_; }
  ApproxBounds<T>* GetApproxBounds() const { return approx_bounds_.get(); }

 private:
  // Bounds are optional and do not need to be set.  If they are not set,
  // automatic bounds will be determined.
  absl::optional<T> lower_;
  absl::optional<T> upper_;

  // Epsilon left over after creating an ApproxBounds.
  absl::optional<double> remaining_epsilon_;

  // Used to automatically determine approximate mimimum and maximum to become
  // lower and upper bounds, respectively.
  std::unique_ptr<ApproxBounds<T>> approx_bounds_;

  // Common initialization and checks for building bounded algorithms.
  base::StatusOr<std::unique_ptr<Algorithm>> BuildAlgorithm() final {
    if (lower_.has_value() != upper_.has_value()) {
      return absl::InvalidArgumentError(
          "Lower and upper bounds must either both be set or both be unset.");
    }

    if (BoundsAreSet()) {
      RETURN_IF_ERROR(ValidateIsFinite(lower_.value(), "Lower bound"));
      RETURN_IF_ERROR(ValidateIsFinite(upper_.value(), "Upper bound"));

      if (lower_.value() > upper_.value()) {
        return absl::InvalidArgumentError(
            "Lower bound cannot be greater than upper bound.");
      }
    }

    return BuildBoundedAlgorithm();
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BOUNDED_ALGORITHM_H_
