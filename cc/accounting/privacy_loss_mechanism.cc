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
#include "accounting/privacy_loss_mechanism.h"

#include <cmath>
#include <optional>

#include "base/status_macros.h"

namespace differential_privacy {
namespace accounting {

absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> LaplacePrivacyLoss::Create(
    double parameter, double sensitivity) {
  if (parameter <= 0) {
    return absl::InvalidArgumentError("parameter should be positive.");
  }
  if (sensitivity <= 0) {
    return absl::InvalidArgumentError("sensitivity should be positive.");
  }
  return absl::WrapUnique(new LaplacePrivacyLoss(parameter, sensitivity));
}

absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> LaplacePrivacyLoss::Create(
    const EpsilonDelta& epsilon_delta) {
  RETURN_IF_ERROR(epsilon_delta.Validate());
  constexpr double sensitivity = 1;
  const double parameter = sensitivity / epsilon_delta.epsilon;
  return Create(parameter, sensitivity);
}

double LaplacePrivacyLoss::InversePrivacyLoss(double privacy_loss) const {
  if (privacy_loss > sensitivity_ / parameter_) {
    return -std::numeric_limits<double>::infinity();
  }
  if (privacy_loss <= -sensitivity_ / parameter_) {
    return std::numeric_limits<double>::infinity();
  }
  return 0.5 * (sensitivity_ - privacy_loss * parameter_);
}

double LaplacePrivacyLoss::NoiseCdf(double x) const {
  return boost::math::cdf(distribution_, x);
}

double LaplacePrivacyLoss::PrivacyLoss(double x) const {
  return (std::abs(x - sensitivity_) - std::abs(x)) / parameter_;
}

PrivacyLossTail LaplacePrivacyLoss::PrivacyLossDistributionTail() const {
  ProbabilityMassFunctionOf<double> pmf;

  //  When x <= 0, the privacy loss is simply sensitivity / parameter; this
  //  happens with probability 0.5.
  pmf[sensitivity_ / parameter_] = 0.5;

  // When x >= sensitivity, the privacy loss is simply
  // - sensitivity / parameter; this happens with probability
  //  1 - CDF(sensitivity) = CDF(-sensitivity).
  pmf[-sensitivity_ / parameter_] =
      boost::math::cdf(distribution_, -sensitivity_);

  return PrivacyLossTail{0, sensitivity_, pmf};
}

absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>>
GaussianPrivacyLoss::Create(double standard_deviation, double sensitivity,
                            EstimateType estimate_type,
                            double log_mass_truncation_bound) {
  if (standard_deviation <= 0) {
    return absl::InvalidArgumentError("standard_deviation should be positive.");
  }
  if (sensitivity <= 0) {
    return absl::InvalidArgumentError("sensitivity should be positive.");
  }
  if (log_mass_truncation_bound > 0) {
    return absl::InvalidArgumentError(
        "log_mass_truncation_bound cannot be "
        "positive.");
  }
  return absl::WrapUnique(new GaussianPrivacyLoss(standard_deviation,
                                                  sensitivity, estimate_type,
                                                  log_mass_truncation_bound));
}

absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>>
GaussianPrivacyLoss::Create(const EpsilonDelta& epsilon_delta,
                            EstimateType estimate_type,
                            double log_mass_truncation_bound) {
  RETURN_IF_ERROR(epsilon_delta.Validate());
  if (epsilon_delta.delta == 0) {
    return absl::InvalidArgumentError(
        "delta should be positive for the "
        "Gaussian mechanism.");
  }
  constexpr double sensitivity = 1;
  // Use binary search to find the smallest possible standard deviation of the
  // Gaussian noise for which the protocol is (epsilon, delta)-differentially
  // private.

  // The initial standard deviation is set to
  // sqrt(2 * ln(1.5/delta) * sensitivity / epsilon. It is known that, when
  // epsilon is no more than one, the Gaussian mechanism with this standard
  // deviation is (epsilon, delta)-DP. See e.g. Appendix A in Dwork and Roth
  // book, "The Algorithmic Foundations of Differential Privacy".
  double initial_standard_deviation =
      std::sqrt(2 * std::log(1.5 / epsilon_delta.delta)) * sensitivity /
      epsilon_delta.epsilon;

  BinarySearchParameters search_parameters = {
      /*lower_bound=*/0,
      /*upper_bound=*/std::numeric_limits<double>::infinity(),
      /*.initial_guess=*/initial_standard_deviation};

  auto compute_delta = [estimate_type,
                        epsilon_delta](double standard_deviation) {
    auto privacy_loss = GaussianPrivacyLoss::Create(
        standard_deviation, sensitivity, estimate_type,
        /*log_mass_truncation_bound=*/0);
    return privacy_loss.value()->GetDeltaForEpsilon(epsilon_delta.epsilon);
  };

  auto inverse_result = InverseMonotoneFunction(
      compute_delta, epsilon_delta.delta, search_parameters);

  return GaussianPrivacyLoss::Create(inverse_result.value(), sensitivity,
                                     estimate_type, log_mass_truncation_bound);
}

absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>>
GaussianPrivacyLoss::Compose(int num_times) {
  // The composition with itself num_times is the same as the
  // GaussianPrivacyLoss with sensitivity scaled up by a factor of square root
  // of num_times.
  return GaussianPrivacyLoss::Create(
      standard_deviation_, sensitivity_ * std::sqrt(num_times), estimate_type_,
      log_mass_truncation_bound_);
}

double GaussianPrivacyLoss::InversePrivacyLoss(double privacy_loss) const {
  return 0.5 * sensitivity_ -
         privacy_loss * (std::pow(standard_deviation_, 2) / sensitivity_);
}

double GaussianPrivacyLoss::NoiseCdf(double x) const {
  return boost::math::cdf(distribution_, x);
}

double GaussianPrivacyLoss::PrivacyLoss(double x) const {
  return 0.5 * sensitivity_ * (sensitivity_ - 2 * x) /
         std::pow(standard_deviation_, 2);
}

PrivacyLossTail GaussianPrivacyLoss::PrivacyLossDistributionTail() const {
  // We set lower_x_truncation so that CDF(lower_x_truncation) =
  // 0.5 * exp(log_mass_truncation_bound), and then set upper_x_truncation
  // to be -lower_x_truncation.
  const double p = 0.5 * std::exp(log_mass_truncation_bound_);
  const double lower_x_truncation = boost::math::quantile(distribution_, p);
  const double upper_x_truncation = -lower_x_truncation;
  ProbabilityMassFunctionOf<double> pmf;
  if (estimate_type_ == EstimateType::kPessimistic) {
    pmf[std::numeric_limits<double>::infinity()] = p;
    pmf[PrivacyLoss(upper_x_truncation)] = p;
  } else {
    pmf[PrivacyLoss(lower_x_truncation)] = p;
  }
  return PrivacyLossTail{lower_x_truncation, upper_x_truncation, pmf};
}

absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>>
DiscreteLaplacePrivacyLoss::Create(double parameter, int sensitivity) {
  if (parameter <= 0) {
    return absl::InvalidArgumentError("parameter should be positive.");
  }
  if (sensitivity <= 0) {
    return absl::InvalidArgumentError("sensitivity should be positive.");
  }
  return absl::WrapUnique(
      new DiscreteLaplacePrivacyLoss(parameter, sensitivity));
}

absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>>
DiscreteLaplacePrivacyLoss::Create(const EpsilonDelta& epsilon_delta,
                                   const int sensitivity) {
  RETURN_IF_ERROR(epsilon_delta.Validate());
  if (sensitivity <= 0) {
    return absl::InvalidArgumentError(
        "sensitivity for discrete Laplace mechanism should be a positive "
        "integer.");
  }
  const double parameter = epsilon_delta.epsilon / sensitivity;
  return Create(parameter, sensitivity);
}

double DiscreteLaplacePrivacyLoss::InversePrivacyLoss(
    double privacy_loss) const {
  if (privacy_loss > sensitivity_ * parameter_) {
    return -std::numeric_limits<double>::infinity();
  }
  if (privacy_loss <= -sensitivity_ * parameter_) {
    return std::numeric_limits<double>::infinity();
  }
  return std::floor(0.5 * (sensitivity_ - privacy_loss / parameter_));
}

double DiscreteLaplacePrivacyLoss::NoiseCdf(double x) const {
  double floor_x = std::floor(x);
  if (floor_x < 0) {
    return std::exp(parameter_ * (floor_x + 1)) / (std::exp(parameter_) + 1);
  } else {
    return 1 - std::exp(-parameter_ * floor_x) / (std::exp(parameter_) + 1);
  }
}

double DiscreteLaplacePrivacyLoss::PrivacyLoss(double x) const {
  return (std::abs(x - sensitivity_) - std::abs(x)) * parameter_;
}

PrivacyLossTail DiscreteLaplacePrivacyLoss::PrivacyLossDistributionTail()
    const {
  ProbabilityMassFunctionOf<double> pmf;

  //  When x <= 0, the privacy loss is simply sensitivity * parameter; this
  // happens with probability CDF(0).
  pmf[sensitivity_ * parameter_] = NoiseCdf(0);

  // When x >= sensitivity, the privacy loss is simply
  // - sensitivity * parameter; this happens with probability
  //  1 - CDF(sensitivity - 1) = CDF(-sensitivity).
  pmf[-sensitivity_ * parameter_] = NoiseCdf(-sensitivity_);

  return PrivacyLossTail{1, sensitivity_ - 1, pmf};
}
absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>>
DiscreteGaussianPrivacyLoss::Create(double sigma, int sensitivity,
                                    std::optional<int> truncation_bound) {
  if (sigma <= 0) {
    return absl::InvalidArgumentError("sigma should be positive.");
  }
  if (sensitivity <= 0) {
    return absl::InvalidArgumentError("sensitivity should be positive.");
  }
  // Tail bound from Canonne et al. ensures that the mass that gets
  // truncated is at most 1e-30. (See Proposition 1 in the supplementary
  // material.)
  int truncation_bound_value =
      truncation_bound.value_or(std::ceil(11.6 * sigma));
  if (truncation_bound_value * 2 < sensitivity) {
    return absl::InvalidArgumentError(
        "Truncation bound should be at least half of sensitivity");
  }
  ProbabilityMassFunction noise_pmf;
  CumulativeDensityFunction noise_cdf;

  for (int x = -truncation_bound_value; x <= truncation_bound_value; ++x) {
    noise_pmf[x] = std::exp(-0.5 * std::pow(x, 2) / std::pow(sigma, 2));
    noise_cdf[x] = noise_cdf[x - 1] + noise_pmf[x];
  }
  for (int x = -truncation_bound_value; x <= truncation_bound_value; ++x) {
    noise_pmf[x] /= noise_cdf[truncation_bound_value];
    noise_cdf[x] /= noise_cdf[truncation_bound_value];
  }
  return absl::WrapUnique(new DiscreteGaussianPrivacyLoss(
      sigma, sensitivity, truncation_bound_value, noise_pmf, noise_cdf));
}

absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>>
DiscreteGaussianPrivacyLoss::Create(const EpsilonDelta& epsilon_delta,
                                    int sensitivity) {
  // Use binary search to find the smallest possible sigma of the Discrete
  // Gaussian noise for which the protocol is (epsilon, delta)-differentially
  // private.

  // The initial standard deviation is set to
  // sqrt(2 * ln(1.5/delta)) * sensitivity / epsilon. It is known that, when
  // epsilon is no more than one, the continuous Gaussian mechanism with this
  // sigma is (epsilon, delta)-DP. See e.g. Appendix A in Dwork and Roth
  // book, "The Algorithmic Foundations of Differential Privacy".
  RETURN_IF_ERROR(epsilon_delta.Validate());
  if (epsilon_delta.delta == 0) {
    return absl::InvalidArgumentError(
        "delta should be positive for the "
        "discrete Gaussian mechanism.");
  }
  if (sensitivity <= 0) {
    return absl::InvalidArgumentError("sensitivity should be positive.");
  }
  double initial_sigma = std::sqrt(2 * std::log(1.5 / epsilon_delta.delta)) *
                         sensitivity / epsilon_delta.epsilon;

  BinarySearchParameters search_parameters = {
      .lower_bound = 0,
      .upper_bound = std::numeric_limits<double>::infinity(),
      .initial_guess = initial_sigma};

  auto compute_delta = [sensitivity,
                        epsilon_delta](double sigma) -> absl::StatusOr<double> {
    ASSIGN_OR_RETURN(std::unique_ptr<DiscreteGaussianPrivacyLoss> privacy_loss,
                     DiscreteGaussianPrivacyLoss::Create(sigma, sensitivity));
    return privacy_loss->GetDeltaForEpsilon(epsilon_delta.epsilon);
  };

  ASSIGN_OR_RETURN(double sigma,
                   InverseMonotoneFunction(compute_delta, epsilon_delta.delta,
                                           search_parameters));

  return DiscreteGaussianPrivacyLoss::Create(sigma, sensitivity);
}

double DiscreteGaussianPrivacyLoss::InversePrivacyLoss(
    double privacy_loss) const {
  return std::floor(0.5 * sensitivity_ -
                    privacy_loss * (std::pow(sigma_, 2) / sensitivity_));
}

double DiscreteGaussianPrivacyLoss::NoiseCdf(double x) const {
  if (x >= truncation_bound_) return 1;
  if (x < -truncation_bound_) return 0;
  return noise_cdf_.at(std::floor(x));
}

double DiscreteGaussianPrivacyLoss::PrivacyLoss(double x) const {
  if (x >= sensitivity_ - truncation_bound_)
    return 0.5 * sensitivity_ * (sensitivity_ - 2 * x) / std::pow(sigma_, 2);
  return std::numeric_limits<double>::infinity();
}

PrivacyLossTail DiscreteGaussianPrivacyLoss::PrivacyLossDistributionTail()
    const {
  // When x < -truncation_bound + sensitivity, the privacy loss is infinity.
  // Due to truncation, x > truncation_bound never occurs.
  return PrivacyLossTail{static_cast<double>(sensitivity_ - truncation_bound_),
                         static_cast<double>(truncation_bound_),
                         {{std::numeric_limits<double>::infinity(),
                           NoiseCdf(sensitivity_ - truncation_bound_ - 1)}}};
}

double DiscreteGaussianPrivacyLoss::StandardDeviation() const {
  double variance = 0;
  for (auto [x, p] : noise_pmf_) variance += (p * std::pow(x, 2));
  return std::sqrt(variance);
}

}  // namespace accounting
}  // namespace differential_privacy
