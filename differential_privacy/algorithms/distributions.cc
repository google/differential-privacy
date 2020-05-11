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
#include "differential_privacy/algorithms/distributions.h"

#include <cmath>

#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "differential_privacy/algorithms/rand.h"
#include "differential_privacy/algorithms/util.h"

namespace differential_privacy {
namespace internal {

LaplaceDistribution::LaplaceDistribution(double b) : b_(b) {
  CHECK_GE(b, 0.0);
}

LaplaceDistribution::LaplaceDistribution(double epsilon,
                                                     double sensitivity)
    : LaplaceDistribution(sensitivity / epsilon) {}

double LaplaceDistribution::GetUniformDouble() { return UniformDouble(); }

// Generates samples from the Laplace Distribution according to the
// Ratio of Uniforms method outlined in Section 4.7
// Devroye, Luc. "Non-Uniform Random Variate Generation" (1987): 195. Cleaner
// and more accurate than the typical Inverse CDF method under fixed precision
// arithmetic.
double LaplaceDistribution::Sample(double scale) {
  DCHECK_GT(scale, 0);
  double u1 = GetUniformDouble();
  double u2 = GetUniformDouble();

  const double value = std::log(u1 / u2) * (scale * b_);
  if (std::isnan(value)) {
    return 0.0;
  }
  return value;
}

double LaplaceDistribution::Sample() { return Sample(1.0); }

double LaplaceDistribution::GetDiversity() { return b_; }

double LaplaceDistribution::cdf(double b, double x) {
  if (x > 0) {
    return 1 - .5 * exp(-x / b);
  }
  return .5 * exp(x / b);
}

int64_t LaplaceDistribution::MemoryUsed() {
  return sizeof(LaplaceDistribution);
}

GaussianDistribution::GaussianDistribution(double stddev) : stddev_(stddev) {
  DCHECK_GE(stddev, 0.0);
}

double GaussianDistribution::Sample(double scale) {
  DCHECK_GT(scale, 0);
  const double value =
      absl::Gaussian<double>(SecureURBG::GetSingleton(), 0, scale * stddev_);
  if (std::isnan(value)) {
    return 0.0;
  }
  return value;
}

double GaussianDistribution::Sample() { return Sample(1.0); }

double GaussianDistribution::Stddev() { return stddev_; }

}  // namespace internal
}  // namespace differential_privacy
