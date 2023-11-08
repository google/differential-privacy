//
// Copyright 2022 Google LLC
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

#include "algorithms/internal/bounded-mean-ci.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/confidence-interval.pb.h"

namespace differential_privacy {
namespace internal {
namespace {

constexpr int kNumStepsOptMeanConfidenceInterval = 1000;

// Given a valid split of the confidence levels, this function returns a
// confidence interval for the mean aggregation.
NumericalMechanism::NoiseConfidenceIntervalResult
NoiseConfidenceIntervalForFixedNumAndDenom(
    const BoundedMeanConfidenceIntervalParams &params, double num_conf_level,
    double denom_conf_level) {
  const NumericalMechanism::NoiseConfidenceIntervalResult sum_ci =
      params.sum_mechanism->UncheckedNoiseConfidenceInterval(num_conf_level,
                                                             params.noised_sum);
  NumericalMechanism::NoiseConfidenceIntervalResult count_ci =
      params.count_mechanism->UncheckedNoiseConfidenceInterval(
          denom_conf_level, params.noised_count);

  // Lower and upper CI for count must be at least 1.
  count_ci.lower = std::max<double>(1, count_ci.lower);
  count_ci.upper = std::max<double>(1, count_ci.upper);

  double mean_lower;
  if (sum_ci.lower >= 0) {
    mean_lower = sum_ci.lower / count_ci.upper;
  } else {
    mean_lower = sum_ci.lower / count_ci.lower;
  }

  double mean_upper;
  if (sum_ci.upper >= 0) {
    mean_upper = sum_ci.upper / count_ci.lower;
  } else {
    mean_upper = sum_ci.upper / count_ci.upper;
  }

  NumericalMechanism::NoiseConfidenceIntervalResult result;
  const double midpoint =
      params.lower_bound + ((params.upper_bound - params.lower_bound) / 2.0);
  result.lower = Clamp<double>(params.lower_bound, params.upper_bound,
                               midpoint + mean_lower);
  result.upper = Clamp<double>(params.lower_bound, params.upper_bound,
                               midpoint + mean_upper);
  return result;
}

}  // namespace

// Details about the current implementation: This function is currently
// implemented using the noise confidence intervals of the numerator, i.e., the
// sum aggregation, and the denominator, i.e., the count aggregation, that are
// used to calculate the anonymized mean.
//
// The confidence interval [low, up] of bounded mean is derived from
// confidence intervals [lowNum, upNum] and [lowDen, upDen] of the mean's
// numerator and denominator, such that
//   Pr(low < raw < up)
//     >= Pr(lowNum < rawNum < upNum & lowDen < rawDen < upDen).
//
// See the NoiseConfidenceIntervalForFixedNumAndDenom method for details of
// how to set [low, up] based on lowNum, upNum, lowDen and upDen.
//
// Because the confidence intervals of the numerator and denominator are
// independent, we can lower bound the confidence level of the mean in terms
// of the confidence level of the numerator and the denominator like this:
//   Pr(low < raw < up)
//     >= Pr(lowNum < rawNum < upNum & lowDen < rawDen < upDen)
//      = Pr(lowNum < rawNum < upNum) * Pr(lowDen < rawDen < upDen)
//     >= confidenceLevelNum * confidenceLevelDen
//
// This means that we can choose confidenceLevelNum and alphaDen arbitrarily
// as long as
//   confidenceLevelNum * confidenceLevelDen = confidenceLevel.
//
// This implementation uses a brute force search for confidenceLevelNum that
// minimizes the size of the confidence interval of bounded mean.
//
// In case one of the input parameters is not finite, this function will return
// a default-constructed ConfidenceInterval with no values set.
ConfidenceInterval BoundedMeanConfidenceInterval(
    const BoundedMeanConfidenceIntervalParams &params) {
  if (!std::isfinite(params.confidence_level) ||
      !std::isfinite(params.noised_sum) || !std::isfinite(params.lower_bound) ||
      !std::isfinite(params.upper_bound) ||
      !std::isfinite(params.noised_count)) {
    return ConfidenceInterval();
  }
  NumericalMechanism::NoiseConfidenceIntervalResult tightest_ci;
  double tightest_ci_size = std::numeric_limits<double>::max();
  for (int i = 1; i < kNumStepsOptMeanConfidenceInterval; ++i) {
    // Setting the confidence level of the numerator and denominator such that
    // overall_conf_level = num_conf_level * denom_conf_level
    // and all confidence levels are between 0 and 1.
    //
    // num_conf_level has to be between confidence_level and 1.  We are
    // dividing this interval into kNumStepsOptMeanConfidenceInterval steps
    // for the brute force search.
    const double num_conf_level =
        params.confidence_level +
        (i / (double)kNumStepsOptMeanConfidenceInterval) *
            (1.0 - params.confidence_level);
    const double denom_conf_level = params.confidence_level / num_conf_level;
    const NumericalMechanism::NoiseConfidenceIntervalResult ci =
        NoiseConfidenceIntervalForFixedNumAndDenom(params, num_conf_level,
                                                   denom_conf_level);
    const double ci_size = ci.upper - ci.lower;
    if (ci_size < tightest_ci_size) {
      tightest_ci = ci;
      tightest_ci_size = ci_size;
    }
  }

  ConfidenceInterval ci;
  ci.set_lower_bound(tightest_ci.lower);
  ci.set_upper_bound(tightest_ci.upper);
  ci.set_confidence_level(params.confidence_level);
  return ci;
}

}  // namespace internal
}  // namespace differential_privacy
