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

#ifndef DIFFERENTIAL_PRIVACY_ACCOUNTING_PRIVACY_LOSS_MECHANISM_H_
#define DIFFERENTIAL_PRIVACY_ACCOUNTING_PRIVACY_LOSS_MECHANISM_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "boost/math/distributions/laplace.hpp"
#include "boost/math/distributions/normal.hpp"
#include "accounting/common/common.h"
// Implementing privacy loss for additive noise mechanisms.
// Please refer to the supplementary material below for more details:
// ../../common_docs/Privacy_Loss_Distributions.pdf

namespace differential_privacy {
namespace accounting {

// Representation of the tail of privacy loss distribution.
struct PrivacyLossTail {
  // The minimum value of x that should be considered after the tail
  // is discarded.
  double lower_x_truncation = 0;
  // The maximum value of x that should be considered after the tail
  // is discarded.
  double upper_x_truncation = 0;
  // The probability mass of the privacy loss distribution that has to be added
  // due to the discarded tail. Each key is a privacy loss value and the
  // corresponding value is the probability mass that the value occurs.
  ProbabilityMassFunctionOf<double> probability_mass_function = {};
};

// Privacy loss of additive noise mechanisms.
//
// An additive noise mechanism for computing a scalar-valued function f is a
// mechanism that outputs the sum of the true value of the function and a noise
// drawn from a certain distribution mu.
//
// We assume that the noise mu is such that the algorithm is more private as
// the sensitivity of f decreases. (Recall that the sensitivity of f is the
// maximum absolute change in f when an input to a single user changes.)
// Under this assumption, the PLD of the mechanism is exactly generated as
// follows: pick x from mu and let the privacy loss be
// ln(P(x) / P(x - sensitivity)). Note that when mu is discrete,
// P(x) and P(x - sensitivity) are the probability masses of mu at x and
// x - sensitivity respectively. When mu is continuous, P(x) and
// P(x - sensitivity) are the probability densities of mu at x and
// x - sensitivity respectively.
//
// This class also assumes the privacy loss is non-increasing as x increases.
class AdditiveNoisePrivacyLoss {
 public:
  virtual ~AdditiveNoisePrivacyLoss() {}

  // Whether the noise is discrete.
  virtual NoiseType Discrete() const = 0;

  // Computes the tail of the privacy loss distribution.
  virtual PrivacyLossTail PrivacyLossDistributionTail() const = 0;

  // Computes the privacy loss at a given point.
  virtual double PrivacyLoss(double x) const = 0;

  // Returns the largest x such that the privacy loss at x is at least
  // privacy_loss.
  // We assume that privacy loss is non-increasing as x increases
  // This is true for Laplace, Gaussian and other widely used mechanisms such
  // as DiscreteLaplace.
  virtual double InversePrivacyLoss(double privacy_loss) const = 0;

  // Cumulative density function of the noise distribution.
  // Returns cumulative density function of noise at x, i.e. the probability
  // that mu is less than or equal to x.
  virtual double NoiseCdf(double x) const = 0;

  // Computes the epsilon-hockey stick divergence of the mechanism.
  // That is for a given epsilon returns delta such that the mechanism is
  // (epsilon, delta)-DP.
  virtual double GetDeltaForEpsilon(double epsilon) const {
    const double x_cutoff = InversePrivacyLoss(epsilon);
    return NoiseCdf(x_cutoff) -
           std::exp(epsilon) * NoiseCdf(x_cutoff - sensitivity_);
  }

  double Sensitivity() const { return sensitivity_; }

 protected:
  explicit AdditiveNoisePrivacyLoss(double sensitivity = 1)
      : sensitivity_(sensitivity) {}

  const double sensitivity_;
};

// Privacy loss of the Laplace mechanism.
//
// The Laplace mechanism for computing a scalar-valued function f simply outputs
// the sum of the true value of the function and a noise drawn from the Laplace
// distribution. Recall that the Laplace distribution with parameter b has
// probability density function 0.5/b * exp(-|x|/b) at x for any real number x.
//
// The privacy loss distribution of the Laplace mechanism is equivalent to the
// privacy loss distribution between the Laplace distribution and the same
// distribution but shifted by the sensitivity of f. Specifically, the privacy
// loss distribution of the Laplace mechanism is generated as follows: first
// pick x according to the Laplace noise. Then, let the privacy loss be
// ln(PDF(x) / PDF(x - sensitivity)) which is equal to
// (|x - sensitivity| - |x|) / parameter.
class LaplacePrivacyLoss : public AdditiveNoisePrivacyLoss {
 public:
  // Creates LaplacePrivacyLoss from these parameters:
  // parameter: the parameter of the Laplace distribution.
  // sensitivity: the sensitivity of function f. (i.e. the maximum absolute
  //   change in f when an input to a single user changes.)
  static absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> Create(
      double parameter, double sensitivity);

  // Creates LaplacePrivacyLoss from epsilon and delta.
  static absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> Create(
      const EpsilonDelta& epsilon_delta);

  NoiseType Discrete() const override { return NoiseType::kContinuous; }

  double InversePrivacyLoss(double privacy_loss) const override;

  double NoiseCdf(double x) const override;

  double PrivacyLoss(double x) const override;

  // Computes the privacy loss at the tail of Laplace distribution.
  PrivacyLossTail PrivacyLossDistributionTail() const override;

  double Parameter() const { return parameter_; }

 private:
  LaplacePrivacyLoss(double parameter, double sensitivity)
      : AdditiveNoisePrivacyLoss(sensitivity), parameter_(parameter) {
    distribution_ = boost::math::laplace_distribution<double>(0, parameter);
  }

  boost::math::laplace_distribution<double> distribution_;
  const double parameter_;
};

// Privacy loss of the Gaussian mechanism.
//
// The Gaussian mechanism for computing a scalar-valued function f simply
// outputs the sum of the true value of the function and a noise drawn from the
// Gaussian distribution. Recall that the (centered) Gaussian distribution with
// standard deviation sigma has probability density function
// 1/(sigma * sqrt(2 * pi)) * exp(-0.5 x^2/sigma^2) at x for any real number x.

// The privacy loss distribution of the Gaussian mechanism is
// equivalent to the privacy loss distribution between the Gaussian distribution
// and the same distribution but shifted by the sensitivity of f. Specifically,
// the privacy loss distribution of the Gaussian mechanism is generated as
// follows: first pick x according to the Gaussian noise. Then, let the privacy
// loss be ln(PDF(x) / PDF(x - sensitivity)) which is equal to
// 0.5 * sensitivity * (sensitivity - 2 * x) / sigma^2.
class GaussianPrivacyLoss : public AdditiveNoisePrivacyLoss {
 public:
  // Creates GaussianPrivacyLoss from these parameters:
  // standard_deviation: standard deviation of Gaussian noise
  // sensitivity: the sensitivity of function f. (i.e. the maximum absolute
  //   change in f when an input to a single user changes.)
  // estimate_type: kPessimistic denoting that the rounding is done in
  //     such a way that the resulting epsilon-hockey stick divergence
  //     computation gives an upper estimate to the real value.
  // log_mass_truncation_bound: the ln of the probability mass that might be
  //   discarded from the noise distribution. The larger this number,
  //   the more error it may introduce in divergence calculations.
  static absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> Create(
      double standard_deviation, double sensitivity,
      EstimateType estimate_type = EstimateType::kPessimistic,
      double log_mass_truncation_bound = -50);

  NoiseType Discrete() const override { return NoiseType::kContinuous; }

  static absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> Create(
      const EpsilonDelta& epsilon_delta,
      EstimateType estimate_type = EstimateType::kPessimistic,
      double log_mass_truncation_bound = -50);

  // Composes with itself num_times.
  // The composition with itself num_times is the same as the
  // GaussianPrivacyLoss with sensitivity scaled up by a factor of square root
  // of num_times.
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> Compose(int num_times);

  double InversePrivacyLoss(double privacy_loss) const override;

  double NoiseCdf(double x) const override;

  double PrivacyLoss(double x) const override;

  PrivacyLossTail PrivacyLossDistributionTail() const override;

  double StandardDeviation() const { return standard_deviation_; }

 private:
  GaussianPrivacyLoss(double standard_deviation, double sensitivity,
                      EstimateType estimate_type,
                      double log_mass_truncation_bound)
      : AdditiveNoisePrivacyLoss(sensitivity),
        standard_deviation_(standard_deviation),
        estimate_type_(estimate_type),
        log_mass_truncation_bound_(log_mass_truncation_bound) {
    distribution_ =
        boost::math::normal_distribution<double>(0, standard_deviation);
  }
  const double standard_deviation_;
  const EstimateType estimate_type_;
  const double log_mass_truncation_bound_;
  boost::math::normal_distribution<double> distribution_;
};

// Privacy loss of the discrete Laplace mechanism.
//
// The discrete Laplace mechanism for computing an integer-valued function f
// simply outputs the sum of the true value of the function and a noise drawn
// from the discrete Laplace distribution. Recall that the discrete Laplace
// distribution with parameter a > 0 has probability mass function
// Z * exp(-a * |x|) at x for any integer x, where Z = (e^a - 1) / (e^a + 1).
//
// The privacy loss distribution of the discrete Laplace mechanism is equivalent
// to that between the discrete Laplace distribution and the same distribution
// but shifted by the sensitivity. More specifically, the privacy loss
// distribution of the discrete Laplace mechanism is generated as follows: first
// pick x according to the discrete Laplace noise. Then, let the privacy loss be
// ln(PMF(x) / PMF(x - sensitivity)) which is equal to
// parameter * (|x - sensitivity| - |x|).
class DiscreteLaplacePrivacyLoss : public AdditiveNoisePrivacyLoss {
 public:
  // Creates DiscreteLaplacePrivacyLoss from these parameters:
  // parameter: the parameter of the discrete Laplace distribution.
  // sensitivity: the sensitivity of function f. (i.e. the maximum absolute
  //   change in f when an input to a single user changes.)
  static absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> Create(
      double parameter, int sensitivity);

  // Creates DiscreteLaplacePrivacyLoss from epsilon, delta and sensitivity.
  static absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> Create(
      const EpsilonDelta& epsilon_delta, const int sensitivity);

  NoiseType Discrete() const override { return NoiseType::kDiscrete; }

  double InversePrivacyLoss(double privacy_loss) const override;

  double NoiseCdf(double x) const override;

  double PrivacyLoss(double x) const override;

  // Computes the privacy loss at the tail of discrete Laplace distribution.
  PrivacyLossTail PrivacyLossDistributionTail() const override;

  double Parameter() const { return parameter_; }

 private:
  DiscreteLaplacePrivacyLoss(double parameter, int sensitivity)
      : AdditiveNoisePrivacyLoss(sensitivity), parameter_(parameter) {}

  const double parameter_;
};

// Privacy loss of the discrete Gaussian mechanism.
//
// The discrete Gaussian mechanism for computing a scalar-valued function f
//  simply outputs the sum of the true value of the function and a noise drawn
//  from the discrete Gaussian distribution. Recall that the (centered) discrete
//  Gaussian distribution with parameter sigma has probability mass function
//  proportional to exp(-0.5 x^2/sigma^2) at x for any integer x. Since its
//  normalization factor and cumulative density function do not have a closed
//  form, we will instead consider the truncated version where the noise x is
//  restricted to only be in [-truncated_bound, truncated_bound].

// The privacy loss distribution of the discrete Gaussian mechanism is
//  equivalent to the privacy loss distribution between the discrete Gaussian
//  distribution and the same distribution but shifted by the sensitivity of f.
//  Specifically, the privacy loss distribution of the discrete Gaussian
//  mechanism is generated as follows: first pick x according to the discrete
//  Gaussian noise. Then, let the privacy loss be
//  ln(PMF(x) / PMF(x - sensitivity)) which is equal to
//  0.5 * sensitivity * (sensitivity - 2 * x) / sigma^2. Note that since we
//  consider the truncated version of the noise, we set the privacy loss to
//  infinity when x < -truncation_bound + sensitivity.
//
//  Reference:
//  Canonne, Kamath, Steinke. "The Discrete Gaussian for Differential Privacy".
//  In NeurIPS 2020.
class DiscreteGaussianPrivacyLoss : public AdditiveNoisePrivacyLoss {
 public:
  // Creates DiscreteGaussianPrivacyLoss from these parameters:
  // sigma: the parameter of the discrete Gaussian distribution. Note that
  //   unlike the (continuous) Gaussian distribution this is not equal to the
  //   standard deviation of the noise.
  // sensitivity: the sensitivity of function f. (i.e. the maximum absolute
  //   change in f when an input to a single user changes.)
  // truncation_bound: bound for truncating the noise, i.e. the noise will only
  //   have a support in [-truncation_bound, truncation_bound]. When not set,
  //   truncation_bound will be chosen in such a way that the mass of the noise
  //   outside of this range is at most 1e-30.
  static absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> Create(
      double sigma, int sensitivity,
      std::optional<int> truncation_bound = std::nullopt);

  NoiseType Discrete() const override { return NoiseType::kDiscrete; }

  static absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> Create(
      const EpsilonDelta& epsilon_delta, int sensitivity);

  double InversePrivacyLoss(double privacy_loss) const override;

  double NoiseCdf(double x) const override;

  double PrivacyLoss(double x) const override;

  PrivacyLossTail PrivacyLossDistributionTail() const override;

  double Sigma() const { return sigma_; }

  double StandardDeviation() const;

  int TruncationBound() const { return truncation_bound_; }

 private:
  DiscreteGaussianPrivacyLoss(double sigma, int sensitivity,
                              int truncation_bound,
                              ProbabilityMassFunction noise_pmf,
                              CumulativeDensityFunction noise_cdf)
      : AdditiveNoisePrivacyLoss(sensitivity),
        sigma_(sigma),
        truncation_bound_(truncation_bound),
        noise_pmf_(noise_pmf),
        noise_cdf_(noise_cdf) {}
  const double sigma_;
  const int truncation_bound_;
  const ProbabilityMassFunction noise_pmf_;
  const CumulativeDensityFunction noise_cdf_;
};
}  // namespace accounting
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ACCOUNTING_PRIVACY_LOSS_MECHANISM_H_
