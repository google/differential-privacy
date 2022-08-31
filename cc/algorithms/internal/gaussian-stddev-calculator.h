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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_GAUSSIAN_STDDEV_CALCULATOR_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_GAUSSIAN_STDDEV_CALCULATOR_H_

namespace differential_privacy {
namespace internal {

// Calculates the standard deviation that is required for the Gaussian
// distribution to have required DP guarantees.
double CalculateGaussianStddev(double epsilon, double delta,
                               double l2_sensitivity);

// Returns delta when using a given standard deviation.  Exposed for testing.
double CalculateDeltaForGaussianStddev(double epsilon, double l2_sensitivity,
                                       double stddev);

struct BoundsForGaussianStddev {
  double upper;
  double lower;
};

// Returns an upper bound for the standard deviation.  Exposed for testing.
BoundsForGaussianStddev CalculateBoundsForGaussianStddev(double epsilon,
                                                         double delta,
                                                         double l2_sensitivity);

}  // namespace internal
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_GAUSSIAN_STDDEV_CALCULATOR_H_
