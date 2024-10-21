// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DIFFERENTIAL_PRIVACY_ACCOUNTING_COMMON_COMMON_H_
#define DIFFERENTIAL_PRIVACY_ACCOUNTING_COMMON_COMMON_H_

#include <functional>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace differential_privacy {
namespace accounting {
// Top-level aliases

// Representation of the differential privacy parameters of a mechanism.
struct EpsilonDelta {
  double epsilon;
  double delta;

  absl::Status Validate() const {
    if (epsilon < 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("epsilon should be positive: %lf.", epsilon));
    }
    if (delta < 0 || delta > 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("delta should be between 0 and 1: %lf.", delta));
    }
    return absl::OkStatus();
  }
};

// Parameters used for binary search.
struct BinarySearchParameters {
  // An upper bound on the binary search range.
  double lower_bound;
  // A lower bound on the binary search range.
  double upper_bound;
  // An initial guess to start the search with. Must be positive. When this
  // guess is close to the true value, it can help make the binary search
  // faster.
  std::optional<double> initial_guess;
  // An acceptable error on the returned value.
  double tolerance = 1e-7;
  // Whether the search is over integers.
  bool discrete = false;
};

// Inverses a monotone function. Specifically, computes x such that f(x) is no
// more than value, when such x exists. It is guaranteed that the returned x is
// within search_parameters.tolerance of the smallest (for monotonically
// decreasing f) or the largest (for monotonically increasing f) such x. When no
// such x exists within the given range, returns NotFoundError.
absl::StatusOr<double> InverseMonotoneFunction(
    absl::FunctionRef<absl::StatusOr<double>(double x)> func, double value,
    BinarySearchParameters search_parameters, bool increasing = false);

// Each probability mass function (PMF) is hash map with outcome as key and
// and probability of this outcome (mass) as value.
// Each mass should be > 0 and the sum of total masses is equal to (roughly) 1.
//
// Most often T will be integer type (i.e. representing a probability
// distribution over integers) but it can be anything.
template <typename T>
using ProbabilityMassFunctionOf = absl::flat_hash_map<T, double>;

using ProbabilityMassFunction = absl::flat_hash_map<int, double>;
using CumulativeDensityFunction = absl::flat_hash_map<int, double>;

enum class EstimateType : int { kOptimistic, kPessimistic };

enum class NoiseType : int { kDiscrete, kContinuous };
}  // namespace accounting
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ACCOUNTING_COMMON_COMMON_H_
