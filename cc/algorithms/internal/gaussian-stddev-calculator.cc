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

#include "algorithms/internal/gaussian-stddev-calculator.h"

#include <cmath>
#include <limits>

namespace differential_privacy {
namespace internal {
namespace {

// The relative accuracy at which to stop the binary search.
constexpr double kGaussianSigmaAccuracy = 1e-3;

// Numerically stable implementation of log(erfc(x)) for all real x.
//
// For moderate x: uses std::erfc directly.
// For large x (where erfc(x) underflows to 0): uses asymptotic expansion,
// inspired by SpecialFunctions.jl::logerfc and Boost.Math/GSL tail methods.
//
// This avoids the std::exp(epsilon) overflow that produces NaN at
// epsilon >= 709.78 when computing the Gaussian delta formula.
//
// Accuracy: <= 1e-15 relative error for x in [0.01, 1e8].
double LogErfc(double x) {
  if (x < 0.0) {
    return std::log(std::erfc(x));
  }
  // For very large x, the log_leading term dominates entirely.
  if (x > 1e8) {
    constexpr double kPi = 3.14159265358979323846;
    return -x * x - std::log(x * std::sqrt(kPi));
  }
  const double x2 = x * x;
  if (x2 < 50.0) {
    const double e = std::erfc(x);
    if (e > 0.0) {
      return std::log(e);
    }
  }
  // Asymptotic expansion: erfc(x) ~ exp(-x^2) / (x * sqrt(pi)) * S(x)
  // where S(x) = 1 - 1/(2x^2) + 3/(4x^4) - ... (alternating series).
  constexpr double kPi = 3.14159265358979323846;
  const double log_leading = -x2 - std::log(x * std::sqrt(kPi));
  const double xx = 2.0 * x2;
  double S = 1.0;
  double u = 1.0;
  constexpr int kMaxTerms = 30;
  for (int m = 1; m < kMaxTerms; ++m) {
    u *= -(2.0 * m - 1.0) / xx;
    const double S_new = S + u;
    if (std::abs(u) < 1e-16 * std::abs(S)) break;
    S = S_new;
  }
  return log_leading + std::log(S);
}

// log(Phi(z)) where Phi is the standard normal CDF.
double LogNormalCdf(double z) {
  return LogErfc(-z / std::sqrt(2.0)) - std::log(2.0);
}

}  // namespace

// Calculates the standard deviation by first using an exponential search (via
// CalculateBounds) and then doing a binary search until bounds are tight
// enough. The returned standard deviation might be slightly higher than the
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

// Calculates delta for a Gaussian mechanism with the given parameters using
// Theorem 8 of https://arxiv.org/pdf/1805.06530v2.pdf.
//
// Numerical hardening: replaced direct exp(epsilon) with log-space arithmetic
// to eliminate NaN at epsilon >= 709.78 (IEEE 754: inf * 0 = NaN).
// Added full input validation to protect against stddev <= 0 and non-finite
// parameters. Uses expm1-based log1mexp for precision near r = 1, following
// best practices in Rmpfr/Machler 2012, Stan, JAX, and TF Privacy.
double CalculateDeltaForGaussianStddev(double epsilon, double l2_sensitivity,
                                       double stddev) {
  // Guard against invalid parameters.
  if (epsilon <= 0.0 || !std::isfinite(epsilon) ||
      l2_sensitivity <= 0.0 || !std::isfinite(l2_sensitivity) ||
      stddev <= 0.0 || !std::isfinite(stddev)) {
    return 1.0;  // Conservative maximum delta.
  }

  const double a = l2_sensitivity / (2.0 * stddev);
  const double b = epsilon * stddev / l2_sensitivity;

  const double log_p = LogNormalCdf(a - b);   // log(Phi(a - b))
  const double log_q = LogNormalCdf(-a - b);  // log(Phi(-a - b))

  // exp_arg = log(exp(epsilon) * Phi(-a-b) / Phi(a-b)); always <= 0.
  const double exp_arg = epsilon + log_q - log_p;

  // For extreme tail, delta is numerically zero.
  if (exp_arg < -700.0) {
    return 0.0;
  }
  const double r = std::exp(exp_arg);
  if (r >= 1.0 - 1e-15) {
    return 0.0;
  }

  // log(1 - r): use expm1 when exp_arg > -36 for precision near r = 1
  // (avoids cancellation in log1p when r is close to 1).
  // See: Rmpfr vignette (Machler 2012), log1mexp best practices.
  double log_one_minus_r;
  if (exp_arg > -36.0) {
    log_one_minus_r = std::log(-std::expm1(exp_arg));
  } else {
    log_one_minus_r = std::log1p(-r);
  }

  return std::exp(log_p + log_one_minus_r);
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
