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
#include "algorithms/distributions.h"

#include <cmath>

#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "algorithms/rand.h"
#include "algorithms/util.h"
#include "base/canonical_errors.h"

namespace differential_privacy {
namespace internal {

namespace {

static constexpr double kPi = 3.14159265358979323846;

// The square root of the maximum number n of Bernoulli trials from which a
// binomial sample is drawn. Larger values result in more fine grained noise,
// but increase the chance of sampling inaccuracies due to overflows. The
// probability of such an event will be roughly 2^-45 or less, if the square
// root is set to 2^57.
static constexpr double kBinomialBound = (double)(1LL << 57);

// Approximates the probability of a random sample m + n / 2 drawn from a
// binomial distribution of n Bernoulli trials that have a success probability
// of 1 / 2 each. The approximation is taken from Lemma 7 of the noise
// generation documentation available in
// https://github.com/google/differential-privacy/blob/master/common_docs/Secure_Noise_Generation.pdf
double ApproximateBinomialProbability(double sqrt_n, int64_t m) {
  if (abs(m) > sqrt_n * sqrt(log(sqrt_n) / 2)) return 0;

  return sqrt(2 / kPi) / sqrt_n * exp(-2.0 * m * m / (sqrt_n * sqrt_n)) *
         (1 - 0.4 * (2 * pow(log(sqrt_n), 1.5)) / sqrt_n);
}

}  // namespace

LegacyLaplaceDistribution::LegacyLaplaceDistribution(double b) : b_(b) {
  CHECK_GE(b, 0.0);
}

LegacyLaplaceDistribution::LegacyLaplaceDistribution(double epsilon,
                                                     double sensitivity)
    : LegacyLaplaceDistribution(sensitivity / epsilon) {}

double LegacyLaplaceDistribution::GetUniformDouble() { return UniformDouble(); }

// Generates samples from the Laplace Distribution according to the
// Ratio of Uniforms method outlined in Section 4.7
// Devroye, Luc. "Non-Uniform Random Variate Generation" (1987): 195. Cleaner
// and more accurate than the typical Inverse CDF method under fixed precision
// arithmetic.
double LegacyLaplaceDistribution::Sample(double scale) {
  DCHECK_GT(scale, 0);
  double u1 = GetUniformDouble();
  double u2 = GetUniformDouble();

  const double value = std::log(u1 / u2) * (scale * b_);
  if (std::isnan(value)) {
    return 0.0;
  }
  return value;
}

double LegacyLaplaceDistribution::Sample() { return Sample(1.0); }

double LegacyLaplaceDistribution::GetDiversity() { return b_; }

double LegacyLaplaceDistribution::cdf(double b, double x) {
  if (x > 0) {
    return 1 - .5 * exp(-x / b);
  }
  return .5 * exp(x / b);
}

int64_t LegacyLaplaceDistribution::MemoryUsed() {
  return sizeof(LegacyLaplaceDistribution);
}

GaussianDistribution::GaussianDistribution(double stddev)
    : stddev_(stddev),
      granularity_(GetNextPowerOfTwo(2 * stddev / kBinomialBound)) {
  DCHECK_GE(stddev, 0.0);
}

double GaussianDistribution::Sample(double scale) {
  DCHECK_GT(scale, 0);
  // TODO: make graceful behaviour when sigma is too big.
  double sigma = scale * stddev_;

  // The square root of n is chosen in a way that ensures that the respective
  // binomial distribution approximates a Gaussian distribution close enough.
  // The sqrt(n) is taken instead of n, to ensure that all results of arithmetic
  // operations fit in 64 bit integer range.
  double sqrt_n = 2.0 * sigma / granularity_;
  return SampleBinomial(sqrt_n) * granularity_;
}

double GaussianDistribution::Sample() { return Sample(1.0); }

double GaussianDistribution::Stddev() { return stddev_; }

double GaussianDistribution::GetGranularity() { return granularity_; }

GeometricDistribution::GeometricDistribution(double lambda) : lambda_(lambda) {
  DCHECK_GE(lambda, 0);
}

double GaussianDistribution::SampleGeometric() {
  int geom_sample = 0;
  while (absl::Bernoulli(SecureURBG::GetSingleton(), 0.5)) ++geom_sample;
  return geom_sample;
}

// Returns a random sample m where {@code m + n / 2} is drawn from a binomial
// distribution of n Bernoulli trials that have a success probability of 1 / 2
// each. The sampling technique is based on Bringmann et al.'s rejection
// sampling approach proposed in "Internal DLA: Efficient Simulation of a
// Physical Growth Model", available
// https://people.mpi-inf.mpg.de/~kbringma/paper/2014ICALP.pdf. The square root
// of n must be at least 10^6. This is to ensure an accurate approximation of a
// Gaussian distribution.
double GaussianDistribution::SampleBinomial(double sqrt_n) {
  long long step_size = static_cast<long long>(round(sqrt(2.0) * sqrt_n + 1));

  SecureURBG& random = SecureURBG::GetSingleton();
  while (true) {
    int geom_sample = SampleGeometric();
    int two_sided_geom =
        absl::Bernoulli(random, 0.5) ? geom_sample : (-geom_sample - 1);
    long long uniform_sample = absl::Uniform(random, 0u, step_size);
    long long result = step_size * two_sided_geom + uniform_sample;

    double result_prob = ApproximateBinomialProbability(sqrt_n, result);
    double reject_prob = UniformDouble();

    if (result_prob > 0 && reject_prob > 0 &&
        reject_prob < result_prob * step_size * pow(2.0, geom_sample - 2)) {
      return result;
    }
  }
}

double GeometricDistribution::GetUniformDouble() { return UniformDouble(); }

int64_t GeometricDistribution::Sample() { return Sample(1.0); }

int64_t GeometricDistribution::Sample(double scale) {
  if (lambda_ == std::numeric_limits<double>::infinity()) {
    return 0;
  }
  double lambda = lambda_ / scale;

  if (GetUniformDouble() >
      -1.0 * expm1(-1.0 * lambda * std::numeric_limits<int64_t>::max())) {
    return std::numeric_limits<int64_t>::max();
  }

  // Performs a binary search for the sample over the range of possible output
  // values. At each step we split the remaining range in two and pick the left
  // or right side proportional to the probability that the output falls within
  // that range, ending when we have only a single possible sample remaining.
  int64_t lo = 0;
  int64_t hi = std::numeric_limits<int64_t>::max();
  while (hi - lo > 1) {
    int64_t mid =
        lo -
        static_cast<int64_t>(std::floor(
            (std::log(0.5) + std::log1p(exp(lambda * (lo - hi)))) / lambda));
    mid = std::min(std::max(mid, lo + 1), hi - 1);

    double q = std::expm1(lambda * (lo - mid)) / expm1(lambda * (lo - hi));
    if (GetUniformDouble() <= q) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return hi - 1;
}

double GeometricDistribution::Lambda() { return lambda_; }

// This is 2^K, with K hardcoded as 40. To generate laplace noise, we sample an
// integer from a geometric distribution, randomly flip the sign, then multiply
// it by a small power of two. That small power of 2 is the smallest power of 2
// larger than ((sensitivity / epsilon) / 2^K). As a consequence, large values
// of K will result in more fine grained noise, but increase the chance of an
// overflow during noise sampling. The probability of such an event will be well
// below 2^-1000, if the granularity parameter is set to a value of 2^40 or less
// and the epsilon passed to addNoise is at least 2^-50.
const double GRANULARITY_PARAM = static_cast<double>(int64_t{1} << 40);

base::StatusOr<double> CalculateGranularity(double epsilon,
                                            double sensitivity) {
  double gran = GetNextPowerOfTwo((sensitivity / epsilon) / GRANULARITY_PARAM);
  double lambda = gran * epsilon / (sensitivity + gran);
  if (lambda < 1.0 / (int64_t{1} << 59)) {
    return base::InvalidArgumentError(
        "The provided parameters may cause an overflow. Probably epsilon is "
        "too small.");
  }
  return gran;
}

LaplaceDistribution::LaplaceDistribution(double epsilon, double sensitivity) {
  epsilon_ = epsilon;
  sensitivity_ = sensitivity;
  granularity_ = CalculateGranularity(epsilon_, sensitivity_).ValueOrDie();

  double lambda;
  if (sensitivity_ == 0) {
    lambda = std::numeric_limits<double>::infinity();
  } else {
    lambda = granularity_ * epsilon_ / (sensitivity_ + granularity_);
  }
  geometric_distro_ = absl::make_unique<GeometricDistribution>(lambda);
}

double LaplaceDistribution::GetUniformDouble() { return UniformDouble(); }

bool LaplaceDistribution::GetBoolean() {
  return absl::Bernoulli(SecureURBG::GetSingleton(), 0.5);
}
double LaplaceDistribution::Sample() { return Sample(1.0); }

double LaplaceDistribution::Sample(double scale) {
  int64_t sample;
  bool sign;
  do {
    sample = geometric_distro_->Sample(scale);
    sign = GetBoolean();
    // Keep a sample of 0 only if the sign is positive. Otherwise, the
    // probability of 0 would be twice as high as it should be.
  } while (sample == 0 && !sign);
  sample = sign ? sample : -sample;
  return sample * granularity_;
}

double LaplaceDistribution::GetGranularity() { return granularity_; }

double LaplaceDistribution::GetDiversity() { return sensitivity_ / epsilon_; }

int64_t LaplaceDistribution::MemoryUsed() {
  int64_t memory = sizeof(LaplaceDistribution);
  if (geometric_distro_ != nullptr) {
    memory += sizeof(*geometric_distro_);
  }
  return memory;
}

}  // namespace internal
}  // namespace differential_privacy
