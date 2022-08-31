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

#include "algorithms/internal/gaussian-stddev-calculator.h"

#include <cmath>
#include <limits>

namespace differential_privacy {
namespace internal {
namespace {

// The relative accuracy at which to stop the binary search to find the tightest
// sigma such that Gaussian noise satisfies (epsilon, delta)-differential
// privacy given the sensitivities.
constexpr double kGaussianSigmaAccuracy = 1e-3;

// Cdf for the Gaussian distribution with stddev = 1.
double StandardGaussianCDF(double x) {
  return std::erfc(-x / (std::sqrt(2.0))) / 2.0;
}

}  // namespace

// Calculates the standard deviation by first using an exponential search (via
// CalculateBounds) and then doing a binary search until bounds are tight
// enough.  The returned standard deviation might be slightly higher than the
// required standard deviation to be on the safe side.
double CalculateGaussianStddev(double epsilon, double delta,
                               double l2_sensitivity) {
  BoundsForGaussianStddev bounds =
      CalculateBoundsForGaussianStddev(epsilon, delta, l2_sensitivity);
  while (bounds.upper - bounds.lower > kGaussianSigmaAccuracy * bounds.lower) {
    const double middle = (bounds.lower + bounds.upper) / 2.0;
    if (CalculateDeltaForGaussianStddev(epsilon, l2_sensitivity,
                                        /* stddev= */ middle) > delta) {
      bounds.lower = middle;
    } else {
      bounds.upper = middle;
    }
  }
  return bounds.upper;
}

// This implementation uses Theorem 8 of https://arxiv.org/pdf/1805.06530v2.pdf
// to calculate the delta for a given standard deviation.
double CalculateDeltaForGaussianStddev(double epsilon, double l2_sensitivity,
                                       double stddev) {
  const double a = l2_sensitivity / (2 * stddev);
  const double b = epsilon * stddev / l2_sensitivity;
  const double c = std::exp(epsilon);
  if (std::isnan(b)) {
    // If either l2_sensitivity goes to 0 or e^epsilon goes to infinity,
    // delta goes to 0.
    return 0;
  }
  return StandardGaussianCDF(a - b) - c * StandardGaussianCDF(-a - b);
}

// Calculates upper and lower bounds for the standard deviation by using an
// exponential search.
BoundsForGaussianStddev CalculateBoundsForGaussianStddev(
    double epsilon, double delta, double l2_sensitivity) {
  BoundsForGaussianStddev bounds;
  bounds.lower = std::numeric_limits<double>::min();
  bounds.upper = l2_sensitivity;
  while (CalculateDeltaForGaussianStddev(epsilon, l2_sensitivity,
                                         /* stddev= */ bounds.upper) > delta) {
    bounds.lower = bounds.upper;
    bounds.upper = bounds.upper * 2.0;
  }
  return bounds;
}

}  // namespace internal
}  // namespace differential_privacy
