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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_BOUNDED_MEAN_CI_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_BOUNDED_MEAN_CI_H_

#include "algorithms/numerical-mechanisms.h"
#include "proto/confidence-interval.pb.h"

namespace differential_privacy {
namespace internal {

struct BoundedMeanConfidenceIntervalParams {
  // Requested confidence level of the confidence interval.  Confidence levels
  // are between 0 and 1.
  double confidence_level;

  // Anonymized sum and count results.
  double noised_sum;
  double noised_count;

  // Lower and upper bound for the bounded aggregation.
  double lower_bound;
  double upper_bound;

  // Mechanisms for noising the sum and the count.  Does not assume ownership.
  NumericalMechanism* sum_mechanism;
  NumericalMechanism* count_mechanism;
};

// Given a confidence_level between 0 and 1, this function will return a
// confidence interval of the mean function.  In case the confidence_level is
// not between 0 and 1, the behavior is undefined.
ConfidenceInterval BoundedMeanConfidenceInterval(
    const BoundedMeanConfidenceIntervalParams& params);

}  // namespace internal
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_BOUNDED_MEAN_CI_H_
