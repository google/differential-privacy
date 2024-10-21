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
#include "algorithms/numerical-mechanisms.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "algorithms/distributions.h"
#include "algorithms/internal/gaussian-stddev-calculator.h"
#include "algorithms/rand.h"
#include "algorithms/util.h"
#include "proto/confidence-interval.pb.h"
#include "proto/numerical-mechanism.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {
namespace {

// The maximum allowable probability that the noise will overflow.
const double kMaxOverflowProbability = std::pow(2.0, -64);

// The relative accuracy at which to stop the binary search to find the tightest
// sigma such that Gaussian noise satisfies (epsilon, delta)-differential
// privacy given the sensitivities.
constexpr double kGaussianSigmaAccuracy = 1e-3;

// Validates and returns the calculated L1 sensitivity based on the L0 and LInf
// sensitivities.
absl::StatusOr<double> CalculateL1SensitivityFromL0AndLInf(
    double l0_sensitivity, double linf_sensitivity) {
  RETURN_IF_ERROR(
      ValidateIsFiniteAndPositive(l0_sensitivity, "L0 sensitivity"));
  RETURN_IF_ERROR(
      ValidateIsFiniteAndPositive(linf_sensitivity, "LInf sensitivity"));
  const double l1_sensitivity = l0_sensitivity * linf_sensitivity;
  if (!std::isfinite(l1_sensitivity)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The result of the L1 sensitivity calculation is not finite: %g. "
        "Please check your contribution and sensitivity settings.",
        l1_sensitivity));
  }
  if (l1_sensitivity == 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The result of the L1 sensitivity calculation is 0, likely because "
        "either L0 sensitivity (%g) and/or LInf sensitivity (%g) are too "
        "small. Please check your contribution and sensitivity settings.",
        l0_sensitivity, linf_sensitivity));
  }
  return l1_sensitivity;
}

// Validates and returns either (1) the L1 sensitivity directly, or (2) the L1
// sensitivity calculated from L0 and LInf, depending on which of the parameters
// are available.
absl::StatusOr<double> CalculateL1Sensitivity(
    std::optional<double> l0_sensitivity, std::optional<double> l1_sensitivity,
    std::optional<double> linf_sensitivity) {
  if (l1_sensitivity.has_value()) {
    RETURN_IF_ERROR(
        ValidateIsFiniteAndPositive(l1_sensitivity, "L1 sensitivity"));
    return l1_sensitivity.value();
  }
  if (l0_sensitivity.has_value() && linf_sensitivity.has_value()) {
    return CalculateL1SensitivityFromL0AndLInf(l0_sensitivity.value(),
                                               linf_sensitivity.value());
  }
  std::string error_string =
      "LaplaceMechanism requires either L1 or (L0 and LInf) sensitivities to "
      "be set";
  if (l0_sensitivity.has_value()) {
    error_string.append(", but only L0 was set.");
  } else if (linf_sensitivity.has_value()) {
    error_string.append(", but only LInf was set.");
  } else {
    error_string.append(", but none were set.");
  }
  return absl::InvalidArgumentError(error_string);
}

}  // namespace

absl::StatusOr<std::unique_ptr<NumericalMechanism>>
LaplaceMechanism::Builder::Build() {
  RETURN_IF_ERROR(ValidateIsFiniteAndPositive(GetEpsilon(), "Epsilon"));
  double epsilon = GetEpsilon().value();
  ASSIGN_OR_RETURN(
      double l1, CalculateL1Sensitivity(GetL0Sensitivity(), GetL1Sensitivity(),
                                        GetLInfSensitivity()));
  // Check that generated noise is not likely to overflow.
  double diversity = l1 / epsilon;
  double overflow_probability =
      (1 - internal::LaplaceDistribution::cdf(
               diversity, std::numeric_limits<double>::max())) +
      internal::LaplaceDistribution::cdf(diversity,
                                         std::numeric_limits<double>::lowest());
  if (overflow_probability >= kMaxOverflowProbability) {
    return absl::InvalidArgumentError("Sensitivity is too high.");
  }
  absl::StatusOr<double> gran_or_status =
      internal::LaplaceDistribution::CalculateGranularity(epsilon, l1);
  if (!gran_or_status.ok()) return gran_or_status.status();

  return absl::StatusOr<std::unique_ptr<NumericalMechanism>>(
      absl::make_unique<LaplaceMechanism>(epsilon, l1));
}

LaplaceMechanism::LaplaceMechanism(double epsilon, double sensitivity)
    : NumericalMechanism(epsilon),
      sensitivity_(sensitivity),
      diversity_(sensitivity / epsilon) {
  absl::StatusOr<std::unique_ptr<internal::LaplaceDistribution>>
      status_or_distro = internal::LaplaceDistribution::Builder()
                             .SetEpsilon(GetEpsilon())
                             .SetSensitivity(sensitivity)
                             .Build();
  DCHECK(status_or_distro.status().ok()) << status_or_distro.status().message();
  distro_ = std::move(status_or_distro.value());
}

absl::StatusOr<std::unique_ptr<NumericalMechanism>>
LaplaceMechanism::Deserialize(const serialization::LaplaceMechanism& proto) {
  Builder builder;
  if (proto.has_epsilon()) {
    builder.SetEpsilon(proto.epsilon());
  }
  if (proto.has_l1_sensitivity()) {
    builder.SetL1Sensitivity(proto.l1_sensitivity());
  }
  return builder.Build();
}

NumericalMechanism::NoiseConfidenceIntervalResult
LaplaceMechanism::UncheckedNoiseConfidenceInterval(double confidence_level,
                                                   double noised_result) const {
  const double bound = diversity_ * std::log(1.0 - confidence_level);
  NoiseConfidenceIntervalResult ci;
  // bound is negative as log(x) with 0 < x < 1 is negative.
  ci.lower = noised_result + bound;
  ci.upper = noised_result - bound;
  return ci;
}

absl::StatusOr<ConfidenceInterval> LaplaceMechanism::NoiseConfidenceInterval(
    double confidence_level, double noised_result) {
  RETURN_IF_ERROR(CheckConfidenceLevel(confidence_level));
  NoiseConfidenceIntervalResult ci =
      UncheckedNoiseConfidenceInterval(confidence_level, noised_result);
  ConfidenceInterval result;
  result.set_lower_bound(ci.lower);
  result.set_upper_bound(ci.upper);
  result.set_confidence_level(confidence_level);
  return result;
}

serialization::LaplaceMechanism LaplaceMechanism::Serialize() const {
  serialization::LaplaceMechanism output;
  output.set_epsilon(NumericalMechanism::GetEpsilon());
  output.set_l1_sensitivity(sensitivity_);
  return output;
}

int64_t LaplaceMechanism::MemoryUsed() {
  int64_t memory = sizeof(LaplaceMechanism);
  if (distro_) {
    memory += distro_->MemoryUsed();
  }
  return memory;
}

double LaplaceMechanism::AddDoubleNoise(double result) {
  double sample = distro_->Sample();
  return RoundToNearestMultiple(result, distro_->GetGranularity()) + sample;
}

int64_t LaplaceMechanism::AddInt64Noise(int64_t result) {
  double sample = distro_->Sample();
  SafeOpResult<int64_t> noise_cast_result =
      SafeCastFromDouble<int64_t>(std::round(sample));

  // Granularity should be a power of 2, and thus can be cast without losing
  // any meaningful fraction. If granularity is <1 (i.e., 2^x, where x<0),
  // then flooring the granularity we use here to 1 should be fine for this
  // function. If granularity is greater than an int64_t can represent, then
  // it's so high that the return value likely won't be terribly meaningful,
  // so just cap the granularity at the largest number int64_t can represent.
  int64_t granularity;
  SafeOpResult<int64_t> granularity_cast_result =
      SafeCastFromDouble<int64_t>(std::max(distro_->GetGranularity(), 1.0));
  if (granularity_cast_result.overflow) {
    granularity = std::numeric_limits<int64_t>::max();
  } else {
    granularity = granularity_cast_result.value;
  }

  return RoundToNearestInt64Multiple(result, granularity) +
         noise_cast_result.value;
}

absl::StatusOr<std::unique_ptr<NumericalMechanism>>
GaussianMechanism::Builder::Build() {
  internal::GaussianDistribution::Builder builder;
  ASSIGN_OR_RETURN(std::unique_ptr<internal::GaussianDistribution> distro,
                   builder.SetStddev(1).Build());

  if (stddev_.has_value()) {
    if (GetEpsilon().has_value() || GetDelta().has_value() ||
        GetL0Sensitivity().has_value() || GetLInfSensitivity().has_value() ||
        l2_sensitivity_.has_value()) {
      return absl::InvalidArgumentError(
          "If standard deviation is set directly it must be the only "
          "parameter.");
    }
    if (!std::isfinite(stddev_.value()) || stddev_.value() < 0) {
      return absl::InvalidArgumentError(
          "Standard deviation must be finite and positive.");
    }
    std::unique_ptr<NumericalMechanism> result =
        absl::make_unique<GaussianMechanism>(stddev_.value(),
                                             std::move(distro));
    return result;
  }  // Else construct from DP parameters.

  std::optional<double> epsilon = GetEpsilon();
  RETURN_IF_ERROR(ValidateIsFiniteAndPositive(epsilon, "Epsilon"));
  RETURN_IF_ERROR(DeltaIsSetAndValid());
  ASSIGN_OR_RETURN(double l2, CalculateL2Sensitivity());

  return absl::StatusOr<std::unique_ptr<NumericalMechanism>>(
      absl::make_unique<GaussianMechanism>(epsilon.value(), GetDelta().value(),
                                           l2, std::move(distro)));
}

absl::StatusOr<double> GaussianMechanism::Builder::CalculateL2Sensitivity() {
  if (l2_sensitivity_.has_value()) {
    absl::Status status =
        ValidateIsFiniteAndPositive(l2_sensitivity_, "L2 sensitivity");
    if (status.ok()) {
      return l2_sensitivity_.value();
    } else {
      return status;
    }
  } else if (GetL0Sensitivity().has_value() &&
             GetLInfSensitivity().has_value()) {
    // Try to calculate L2 sensitivity from L0 and LInf sensitivities
    RETURN_IF_ERROR(
        ValidateIsFiniteAndPositive(GetL0Sensitivity(), "L0 sensitivity"));
    double l0 = GetL0Sensitivity().value();
    RETURN_IF_ERROR(
        ValidateIsFiniteAndPositive(GetLInfSensitivity(), "LInf sensitivity"));
    double linf = GetLInfSensitivity().value();
    double l2 = std::sqrt(l0) * linf;
    if (!std::isfinite(l2) || l2 <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The calculated L2 sensitivity must be positive and finite but "
          "is ",
          l2,
          ". Contribution or sensitivity settings might be too high or too "
          "low."));
    }
    return l2;
  }
  return absl::InvalidArgumentError(
      "Gaussian Mechanism requires either L2 sensitivity or both L0 "
      "and LInf sensitivity to be set.");
}

GaussianMechanism::GaussianMechanism(double epsilon, double delta,
                                     double l2_sensitivity)
    : NumericalMechanism(epsilon),
      delta_(delta),
      l2_sensitivity_(l2_sensitivity),
      stddev_(CalculateStddev(epsilon, delta, l2_sensitivity)) {
  absl::StatusOr<std::unique_ptr<internal::GaussianDistribution>>
      status_or_distro =
          internal::GaussianDistribution::Builder().SetStddev(1).Build();
  DCHECK(status_or_distro.status().ok());
  standard_gaussian_ = std::move(status_or_distro.value());
}

absl::StatusOr<std::unique_ptr<NumericalMechanism>>
GaussianMechanism::Deserialize(const serialization::GaussianMechanism& proto) {
  Builder builder;
  if (proto.has_epsilon()) {
    builder.SetEpsilon(proto.epsilon());
  }
  if (proto.has_delta()) {
    builder.SetDelta(proto.delta());
  }
  if (proto.has_l2_sensitivity()) {
    builder.SetL2Sensitivity(proto.l2_sensitivity());
  }
  return builder.Build();
}

bool GaussianMechanism::NoisedValueAboveThreshold(double result,
                                                  double threshold) {
  double stddev = CalculateStddev();
  return UniformDouble() >
         internal::GaussianDistribution::cdf(stddev, threshold - result);
}

double GaussianMechanism::ProbabilityOfNoisedValueAboveThreshold(
    double result, double threshold) {
  double stddev = CalculateStddev();
  return 1 - internal::GaussianDistribution::cdf(stddev, threshold - result);
}

serialization::GaussianMechanism GaussianMechanism::Serialize() const {
  serialization::GaussianMechanism result;
  result.set_epsilon(NumericalMechanism::GetEpsilon());
  result.set_delta(delta_);
  result.set_l2_sensitivity(l2_sensitivity_);
  return result;
}

int64_t GaussianMechanism::MemoryUsed() {
  int64_t memory = sizeof(GaussianMechanism);
  if (standard_gaussian_) {
    memory += sizeof(internal::GaussianDistribution);
  }
  return memory;
}

NumericalMechanism::NoiseConfidenceIntervalResult
GaussianMechanism::UncheckedNoiseConfidenceInterval(
    double confidence_level, double noised_result) const {
  const double stddev = CalculateStddev(GetEpsilon(), delta_, l2_sensitivity_);
  // calculated using the symmetric properties of the Gaussian distribution
  // and the cumulative distribution function for the distribution
  double bound =
      InverseErrorFunction(-1 * confidence_level) * stddev * std::sqrt(2.0);
  NoiseConfidenceIntervalResult ci;
  // bound is negative.
  ci.lower = noised_result + bound;
  ci.upper = noised_result - bound;
  return ci;
}

absl::StatusOr<ConfidenceInterval> GaussianMechanism::NoiseConfidenceInterval(
    double confidence_level, double noised_result) {
  RETURN_IF_ERROR(CheckConfidenceLevel(confidence_level));
  NoiseConfidenceIntervalResult ci =
      UncheckedNoiseConfidenceInterval(confidence_level, noised_result);
  ConfidenceInterval result;
  result.set_lower_bound(ci.lower);
  result.set_upper_bound(ci.upper);
  result.set_confidence_level(confidence_level);
  return result;
}

double GaussianMechanism::CalculateStddev(double epsilon, double delta,
                                          double l2_sensitivity) {
  return internal::CalculateGaussianStddev(epsilon, delta, l2_sensitivity);
}

double GaussianMechanism::CalculateStddev() const { return stddev_; }

double GaussianMechanism::AddDoubleNoise(double result) {
  double stddev = CalculateStddev();
  double sample = standard_gaussian_->Sample(stddev);

  return RoundToNearestMultiple(result,
                                standard_gaussian_->GetGranularity(stddev)) +
         sample;
}

int64_t GaussianMechanism::AddInt64Noise(int64_t result) {
  double stddev = CalculateStddev();
  double sample = standard_gaussian_->Sample(stddev);

  SafeOpResult<int64_t> noise_cast_result =
      SafeCastFromDouble<int64_t>(std::round(sample));

  // Granularity should be a power of 2, and thus can be cast without losing
  // any meaningful fraction. If granularity is <1 (i.e., 2^x, where x<0),
  // then flooring the granularity we use here to 1 should be fine for this
  // function. If granularity is greater than an int64_t can represent, then
  // it's so high that the return value likely won't be terribly meaningful,
  // so just cap the granularity at the largest number int64_t can represent.
  int64_t granularity;
  SafeOpResult<int64_t> granularity_cast_result = SafeCastFromDouble<int64_t>(
      std::max(standard_gaussian_->GetGranularity(stddev), 1.0));
  if (granularity_cast_result.overflow) {
    granularity = std::numeric_limits<int64_t>::max();
  } else {
    granularity = granularity_cast_result.value;
  }

  return RoundToNearestInt64Multiple(result, granularity) +
         noise_cast_result.value;
}

}  // namespace differential_privacy
