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
#include <cstdlib>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "algorithms/rand.h"
#include "algorithms/util.h"
#include "third_party/cephes/inverse_gaussian_cdf.h"
#include "base/status_macros.h"

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
// https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf
double ApproximateBinomialProbability(double sqrt_n, int64_t m) {
  const double n = sqrt_n * sqrt_n;
  if (std::abs(m) > sqrt_n * std::sqrt(std::log(n)) / 2) {
    return 0;
  }
  return std::sqrt(2 / kPi) / sqrt_n * std::exp(-2.0 * m * m / n) *
         (1 - (0.4 * std::pow(std::log(n), 1.5) / sqrt_n));
}

}  // namespace

GaussianDistribution::Builder& GaussianDistribution::Builder::SetStddev(
    double stddev) {
  stddev_ = stddev;
  return *this;
}

absl::StatusOr<std::unique_ptr<GaussianDistribution>>
GaussianDistribution::Builder::Build() {
  RETURN_IF_ERROR(
      ValidateIsFiniteAndNonNegative(stddev_, "Standard deviation"));
  return absl::WrapUnique<GaussianDistribution>(
      new GaussianDistribution(stddev_));
}

GaussianDistribution::GaussianDistribution(double stddev) : stddev_(stddev) {
  DCHECK_GE(stddev, 0.0);
}

double GaussianDistribution::Sample(double scale) {
  DCHECK_GT(scale, 0);
  // TODO: make graceful behaviour when sigma is too big.
  double sigma = scale * stddev_;
  // Use at least the lowest positive floating point number as granularity when
  // sigma is very small.
  double granularity =
      std::max(GetGranularity(scale), std::numeric_limits<double>::min());

  // The square root of n is chosen in a way that ensures that the respective
  // binomial distribution approximates a Gaussian distribution close enough.
  // The sqrt(n) is taken instead of n, to ensure that all results of arithmetic
  // operations fit in 64 bit integer range.
  double sqrt_n = (sigma / granularity) * 2.0;
  return SampleBinomial(sqrt_n) * granularity;
}

double GaussianDistribution::Sample() { return Sample(1.0); }

double GaussianDistribution::Stddev() const { return stddev_; }

double GaussianDistribution::GetGranularity(double scale) const {
  double sigma = scale * stddev_;
  return GetNextPowerOfTwo((sigma / kBinomialBound) * 2.0);
}

double GaussianDistribution::cdf(double stddev, double x) {
  DCHECK_GT(stddev, 0);
  return (std::erfc(-x / (stddev * std::sqrt(2)))) / 2;
}

double GaussianDistribution::Quantile(double stddev, double x) {
  DCHECK_GT(stddev, 0);
  return stddev * third_party::cephes::InverseCdfStandardGaussian(x);
}

GeometricDistribution::Builder& GeometricDistribution::Builder::SetLambda(
    double lambda) {
  lambda_ = lambda;
  return *this;
}

absl::StatusOr<std::unique_ptr<GeometricDistribution>>
GeometricDistribution::Builder::Build() {
  RETURN_IF_ERROR(ValidateIsFiniteAndNonNegative(lambda_, "Lambda"));

  return absl::WrapUnique<GeometricDistribution>(
      new GeometricDistribution(lambda_));
}

GeometricDistribution::GeometricDistribution(double lambda) : lambda_(lambda) {
  DCHECK_GE(lambda, 0);
}

double GaussianDistribution::SampleGeometric() {
  int geom_sample = 0;
  while (absl::Bernoulli(SecureURBG::GetInstance(), 0.5)) ++geom_sample;
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
  long long step_size =
      static_cast<long long>(std::round(std::sqrt(2.0) * sqrt_n + 1));

  SecureURBG& random = SecureURBG::GetInstance();
  while (true) {
    int geom_sample = SampleGeometric();
    int two_sided_geom =
        absl::Bernoulli(random, 0.5) ? geom_sample : (-geom_sample - 1);
    int64_t uniform_sample = absl::Uniform(random, 0u, step_size);
    int64_t result = step_size * two_sided_geom + uniform_sample;

    double result_prob = ApproximateBinomialProbability(sqrt_n, result);
    double reject_prob = UniformDouble();

    if (result_prob > 0 && reject_prob > 0 &&
        reject_prob <
            result_prob * step_size * std::pow(2.0, geom_sample - 2)) {
      return result;
    }
  }
}

double GeometricDistribution::GetUniformDouble() { return UniformDouble(); }

int64_t GeometricDistribution::Sample() { return Sample(1.0); }

const std::vector<double>& GeometricDistribution::GetProbs(double lambda) {
  if (probs_.empty() || lambda != cached_lambda_) {
    probs_.clear();
    probs_.reserve(63);
    for (int i = 0; i < 63; ++i) {
      double c = lambda * (1LL << i);
      if (c > 100) {
        // If c is large, exp(c) is huge and 1 / (exp(c) + 1) is close to 0, no
        // need to compute them.
        break;
      } else {
        probs_.push_back(1.0 / (std::exp(c) + 1));
      }
    }
    cached_lambda_ = lambda;
  }
  return probs_;
}

int64_t GeometricDistribution::Sample(double scale) {
  if (lambda_ == std::numeric_limits<double>::infinity()) {
    return 0;
  }
  double lambda = lambda_ / scale;
  // Samples an integer in [0, 2^63 - 1] from a Geometric distribution (q =
  // e^-lambda). lambda is order of 2^-40, see the comment to
  // kLaplaceGranularityParam.
  //
  // For exceedingly small lambda values (e.g., 10^-12), standard inverse
  // transform sampling (k = floor(ln(U) / -lambda)) suffers from catastrophic
  // floating-point precision loss. This algorithm avoids calculating absolute
  // cumulative probabilities entirely by conditionally generating the result
  // bit-by-bit.
  //
  // MATHEMATICAL FOUNDATION:
  // We perform a binary search over the integer space. Due to the memoryless
  // property of the Geometric distribution, the probability of the sample
  // falling into the right half of any interval depends ONLY on the size of the
  // half-interval (d), not on its starting position (L).
  //
  // Specifically, if an interval has 2d elements, the probability of branching
  // right is:
  //     P(Right) = 1 / (1 + exp(d * lambda))
  //
  // BITWISE MAPPING:
  // Constructing a 63-bit integer from MSB (bit 62) down to LSB (bit 0) is
  // perfectly isomorphic to this binary search:
  //
  // 1. Loop Index (i): Represents the current bit. The half-interval size is d
  // = 2^i.
  // 2. Precomputation: Because 'd' only takes 63 deterministic values (2^62
  // down to 2^0), we precompute the exact branching probability for each bit
  // and store it in `probs`. probs[i] = 1 / (1 + exp(2^i * lambda)).
  // 3. Branching: We draw a uniform random float. If U < probs[i], the sample
  // belongs in the right half of the current sub-interval.
  // 4. Accumulation: Setting the i-th bit to 1 (result |= 1LL << i) adds 2^i to
  // the result. This is mathematically equivalent to moving the left boundary
  // of our search space (L = L + d). Leaving it as 0 is equivalent to narrowing
  // the right boundary.

  const std::vector<double>& probs = GetProbs(lambda);

  int64_t result = 0;
  for (int i = probs.size() - 1; i >= 0; --i) {
    if (GetUniformDouble() < probs[i]) {
      result |= (1LL << i);
    }
  }
  return result;
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
constexpr double kLaplaceGranularityParam =
    static_cast<double>(int64_t{1} << 40);

absl::Status LaplaceDistribution::ValidateEpsilon(double epsilon) {
  RETURN_IF_ERROR(ValidateIsFiniteAndPositive(epsilon, "Epsilon"));
  if (epsilon < kMinEpsilon) {
    return absl::InvalidArgumentError(
        absl::StrCat("Epsilon must be at least 2^-50, but is ", epsilon, "."));
  }
  return absl::OkStatus();
}

// Calculates 'r' from the secure noise paper (see
// ../../common_docs/Secure_Noise_Generation.pdf)
double CalculateGranularityForLaplace(double diversity) {
  return GetNextPowerOfTwo(diversity / kLaplaceGranularityParam);
}

absl::StatusOr<double> LaplaceDistribution::CalculateGranularity(
    double epsilon, double sensitivity) {
  RETURN_IF_ERROR(ValidateEpsilon(epsilon));
  RETURN_IF_ERROR(ValidateIsFiniteAndPositive(sensitivity, "Sensitivity"));
  return CalculateGranularityForLaplace(sensitivity / epsilon);
}

LaplaceDistribution::Builder& LaplaceDistribution::Builder::SetEpsilon(
    double epsilon) {
  epsilon_ = epsilon;
  return *this;
}

LaplaceDistribution::Builder& LaplaceDistribution::Builder::SetSensitivity(
    double sensitivity) {
  sensitivity_ = sensitivity;
  return *this;
}

absl::StatusOr<std::unique_ptr<LaplaceDistribution>>
LaplaceDistribution::Builder::Build() {
  RETURN_IF_ERROR(ValidateEpsilon(epsilon_));
  RETURN_IF_ERROR(ValidateIsFiniteAndPositive(sensitivity_, "Sensitivity"));
  const double diversity = sensitivity_ / epsilon_;
  const double granularity = CalculateGranularityForLaplace(diversity);
  ASSIGN_OR_RETURN(
      std::unique_ptr<GeometricDistribution> geometric_distro,
      GeometricDistribution::Builder()
          .SetLambda(granularity * epsilon_ / (sensitivity_ + granularity))
          .Build());
  return absl::WrapUnique<LaplaceDistribution>(new LaplaceDistribution(
      epsilon_, sensitivity_, granularity, std::move(geometric_distro)));
}

LaplaceDistribution::LaplaceDistribution(
    double epsilon, double sensitivity, double granularity,
    std::unique_ptr<GeometricDistribution> geometric_distro)
    : epsilon_(epsilon),
      sensitivity_(sensitivity),
      granularity_(granularity),
      geometric_distro_(std::move(geometric_distro)) {}

LaplaceDistribution::LaplaceDistribution(double epsilon, double sensitivity)
    : epsilon_(epsilon), sensitivity_(sensitivity) {
  absl::StatusOr<double> granularity =
      CalculateGranularity(epsilon_, sensitivity_);
  CHECK(granularity.ok()) << granularity.status();
  granularity_ = granularity.value();

  double lambda;
  if (sensitivity_ == 0) {
    // Builder validation should prevent this case from ever happening, but if
    // not, prevent a possible divide-by-zero.
    lambda = std::numeric_limits<double>::infinity();
  } else {
    lambda = granularity_ * epsilon_ / (sensitivity_ + granularity_);
  }
  GeometricDistribution::Builder builder;
  geometric_distro_ = builder.SetLambda(lambda).Build().value();
}

double LaplaceDistribution::GetUniformDouble() { return UniformDouble(); }

bool LaplaceDistribution::GetBoolean() {
  return absl::Bernoulli(SecureURBG::GetInstance(), 0.5);
}

double LaplaceDistribution::Sample() {
  int64_t sample;
  bool sign;
  do {
    sample = geometric_distro_->Sample();
    sign = GetBoolean();
    // Keep a sample of 0 only if the sign is positive. Otherwise, the
    // probability of 0 would be twice as high as it should be.
  } while (sample == 0 && !sign);
  const int64_t signed_sample = sign ? sample : -sample;
  return signed_sample * granularity_;
}

double LaplaceDistribution::GetGranularity() { return granularity_; }

double LaplaceDistribution::GetVariance() const {
  return 2.0 * std::pow(GetDiversity(), 2);
}

double LaplaceDistribution::GetDiversity() const {
  return sensitivity_ / epsilon_;
}

double LaplaceDistribution::GetMinEpsilon() { return kMinEpsilon; }

double LaplaceDistribution::cdf(double b, double x) {
  if (x > 0) {
    return 1 - .5 * std::exp(-x / b);
  }
  return .5 * std::exp(x / b);
}

double LaplaceDistribution::Quantile(double b, double p) {
  if (p > 0.5) {
    return -b * std::log(2 - 2 * p);
  }
  return b * std::log(2 * p);
}

int64_t LaplaceDistribution::MemoryUsed() {
  int64_t memory = sizeof(LaplaceDistribution);
  if (geometric_distro_ != nullptr) {
    memory += sizeof(*geometric_distro_);
  }
  return memory;
}

}  // namespace internal
}  // namespace differential_privacy
