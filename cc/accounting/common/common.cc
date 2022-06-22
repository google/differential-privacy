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

#include "accounting/common/common.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "base/status_macros.h"

namespace differential_privacy {
namespace accounting {
absl::StatusOr<double> InverseMonotoneFunction(
    const absl::FunctionRef<absl::StatusOr<double>(double x)> func,
    const double value, const BinarySearchParameters search_parameters,
    const bool increasing) {
  double lower_x = search_parameters.lower_bound;
  double upper_x = search_parameters.upper_bound;

  double min_value = -std::numeric_limits<double>::infinity();
  if (increasing && lower_x != -std::numeric_limits<double>::infinity()) {
    ASSIGN_OR_RETURN(min_value, func(lower_x));
  } else if (!increasing &&
             upper_x != std::numeric_limits<double>::infinity()) {
    ASSIGN_OR_RETURN(min_value, func(upper_x));
  }
  if (min_value > value) {
    return absl::NotFoundError(absl::StrFormat(
        "Cannot find x in range (%lf, %lf) whose value is at most %lf.",
        lower_x, upper_x, value));
  }

  std::function<bool(const double)> solution_above_current_x;
  if (increasing) {
    solution_above_current_x = [value](const double func_value) {
      return func_value <= value;
    };
  } else {
    solution_above_current_x = [value](const double func_value) {
      return func_value > value;
    };
  }

  if (search_parameters.initial_guess.has_value()) {
    // If |initial_guess| is specified, start from |initial_guess| and keep
    // doubling until the value of x becomes too large. This is done only for
    // efficiency: When |initial_guess| is close to a true solution and
    // |upper_x| is much larger this can help reduce the number of evaluations
    // to the function.
    double initial_guess_x = search_parameters.initial_guess.value();
    while (initial_guess_x < upper_x) {
      double func_value;
      ASSIGN_OR_RETURN(func_value, func(initial_guess_x));
      if (!solution_above_current_x(func_value)) break;
      lower_x = initial_guess_x;
      initial_guess_x *= 2;
    }
    upper_x = std::min(upper_x, initial_guess_x);
  }

  double tolerance = search_parameters.tolerance;
  if (search_parameters.discrete) tolerance = 1;

  while (upper_x - lower_x > tolerance) {
    double mid_x = (upper_x + lower_x) / 2;
    if (search_parameters.discrete) mid_x = std::floor(mid_x);

    double func_value;
    ASSIGN_OR_RETURN(func_value, func(mid_x));
    if (solution_above_current_x(func_value)) {
      lower_x = mid_x;
    } else {
      upper_x = mid_x;
    }
  }

  return increasing ? lower_x : upper_x;
}
}  // namespace accounting
}  // namespace differential_privacy
