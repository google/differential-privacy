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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <cstdint>
#include "base/logging.h"
#include "absl/base/attributes.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "algorithms/distributions.h"
#include "algorithms/rand.h"
#include "algorithms/util.h"
#include "proto/confidence-interval.pb.h"
#include "proto/numerical-mechanism.pb.h"
#include "base/status_macros.h"

// A set of classes for adding differentially private noise to numerical data.
// Rather than sampling directly from the distribution, these classes allow
// passing in the differential privacy parameters (epsilon, delta, sensitivity).
// Then, given a numerical value they automatically calculate the parameters of
// the distribution based on the differential privacy parameters and return the
// value plus noise. They also handle any other necessary steps to achieve
// differentially private results (e.g. snapping).
namespace differential_privacy {

// The maximum allowable probability that the noise will overflow.
static const double kMaxOverflowProbability = std::pow(2.0, -64);

// The relative accuracy at which to stop the binary search to find the tightest
// sigma such that Gaussian noise satisfies (epsilon, delta)-differential
// privacy given the sensitivities.
static const double kGaussianSigmaAccuracy = 1e-3;

// Provides a common abstraction for NumericalMechanism.  Numerical mechanisms
// can add noise to data.
class NumericalMechanism {
 public:
  NumericalMechanism(double epsilon) : epsilon_(epsilon) {}

  virtual ~NumericalMechanism() = default;

  template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  T AddNoise(T result) {
    return AddInt64Noise(result);
  }

  template <typename T,
            std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  T AddNoise(T result) {
    return AddDoubleNoise(result);
  }

  // Quickly determines if result with added noise is greater than threshold.
  // This method allows for quicker thresholding decisions by using a uniform
  // random number instead of the slower (i.e., more complex to compute) noise
  // value from the distribution. Using a faster randomness generation method
  // for thresholding decisions is still privacy-safe, because the thresholding
  // decision is binary, so there is no risk of violating DP from the least
  // significant bits of the returned result (see On Significance of the Least
  // Significant Bits For Differential Privacy by Ilya Mironov:
  // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.366.5957&rep=rep1&type=pdf).
  virtual bool NoisedValueAboveThreshold(double result, double threshold) = 0;

  virtual double ProbabilityOfNoisedValueAboveThreshold(double result,
                                                        double threshold) = 0;

  virtual int64_t MemoryUsed() = 0;

  virtual absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double noised_result) = 0;

  virtual absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) = 0;

  struct NoiseConfidenceIntervalResult {
    double upper;
    double lower;
  };

  // An optimized but unchecked method for NoiseConfidenceInterval.  End-users
  // should use NoiseConfidenceInterval instead.
  //
  // If confidence_level is in the open interval (0,1), this method returns the
  // noise confidence interval for the noised result.  If confidence_level is
  // outside of (0,1), the behavior is unspecified.
  virtual NoiseConfidenceIntervalResult UncheckedNoiseConfidenceInterval(
      double confidence_level, double noised_result) const = 0;

  double GetEpsilon() const { return epsilon_; }

  // Returns the variance of the noise that will be added by the underlying
  // distribution.
  virtual double GetVariance() const { return 0; }

  // Returns the value of the cumulative density function, i.e. the probability
  // that the noise added is no greater than x.
  virtual double Cdf(double x) const = 0;

  // Returns the value of the quantile function (inverse cumulative density),
  // i.e. the value x such that with probability p the noise added is no greater
  // than x.
  virtual double Quantile(double p) const = 0;

 protected:
  virtual double AddDoubleNoise(double result) = 0;

  virtual int64_t AddInt64Noise(int64_t result) = 0;

  static absl::Status CheckConfidenceLevel(const double confidence_level) {
    return ValidateIsInExclusiveInterval(confidence_level, 0, 1,
                                         "Confidence level");
  }

 private:
  const double epsilon_;
};

// Provides a common abstraction for Builders for NumericalMechanism.
class NumericalMechanismBuilder {
 public:
  virtual ~NumericalMechanismBuilder() = default;

  NumericalMechanismBuilder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  NumericalMechanismBuilder& SetDelta(double delta) {
    delta_ = delta;
    return *this;
  }

  NumericalMechanismBuilder& SetL0Sensitivity(double l0_sensitivity) {
    l0_sensitivity_ = l0_sensitivity;
    return *this;
  }

  NumericalMechanismBuilder& SetLInfSensitivity(double linf_sensitivity) {
    linf_sensitivity_ = linf_sensitivity;
    return *this;
  }

  virtual absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() = 0;

  virtual std::unique_ptr<NumericalMechanismBuilder> Clone() const = 0;

 protected:
  // Checks if delta is set and valid to be used in the Gaussian mechanism.
  absl::Status DeltaIsSetAndValid() const {
    RETURN_IF_ERROR(ValidateIsFinite(delta_, "Delta"));
    RETURN_IF_ERROR(ValidateIsInExclusiveInterval(delta_, 0, 1, "Delta"));
    return absl::OkStatus();
  }

  absl::optional<double> GetEpsilon() const { return epsilon_; }
  absl::optional<double> GetDelta() const { return delta_; }
  absl::optional<double> GetL0Sensitivity() const { return l0_sensitivity_; }
  absl::optional<double> GetLInfSensitivity() const {
    return linf_sensitivity_;
  }

 private:
  absl::optional<double> epsilon_;
  absl::optional<double> delta_;
  absl::optional<double> l0_sensitivity_;
  absl::optional<double> linf_sensitivity_;
};

// Provides differential privacy by adding Laplace noise. This class also
// contains supporting functions related to the noise added.
class LaplaceMechanism : public NumericalMechanism {
 public:
  // Builder for LaplaceMechanism.
  class Builder : public NumericalMechanismBuilder {
   public:
    ABSL_DEPRECATED(
        "Set LInf and L0 sensitivity instead or use SetL1Sensitivity in tests.")
    Builder& SetSensitivity(double l1_sensitivity) {
      return SetL1Sensitivity(l1_sensitivity);
    }

    Builder& SetL1Sensitivity(double sensitivity_l1) {
      l1_sensitivity_ = sensitivity_l1;
      return *this;
    }

    absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      RETURN_IF_ERROR(ValidateIsFiniteAndPositive(GetEpsilon(), "Epsilon"));
      double epsilon = GetEpsilon().value();
      ASSIGN_OR_RETURN(double L1, CalculateL1Sensitivity());
      // Check that generated noise is not likely to overflow.
      double diversity = L1 / epsilon;
      double overflow_probability =
          (1 - internal::LaplaceDistribution::cdf(
                   diversity, std::numeric_limits<double>::max())) +
          internal::LaplaceDistribution::cdf(
              diversity, std::numeric_limits<double>::lowest());
      if (overflow_probability >= kMaxOverflowProbability) {
        return absl::InvalidArgumentError("Sensitivity is too high.");
      }
      absl::StatusOr<double> gran_or_status =
          internal::LaplaceDistribution::CalculateGranularity(epsilon, L1);
      if (!gran_or_status.ok()) return gran_or_status.status();

      return absl::StatusOr<std::unique_ptr<NumericalMechanism>>(
          absl::make_unique<LaplaceMechanism>(epsilon, L1));
    }

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }

   protected:
    absl::optional<double> GetL1Sensitivity() const { return l1_sensitivity_; }

   private:
    absl::optional<double> l1_sensitivity_;

    // Returns the L1 sensitivity when it has been set or returns an upper bound
    // on the L1 sensitivity calculated from L0 and Linf sensitivities.
    absl::StatusOr<double> CalculateL1Sensitivity() {
      if (l1_sensitivity_.has_value()) {
        absl::Status status =
            ValidateIsFiniteAndPositive(l1_sensitivity_, "L1 sensitivity");
        if (status.ok()) {
          return l1_sensitivity_.value();
        } else {
          return status;
        }
      }
      if (GetL0Sensitivity().has_value() && GetLInfSensitivity().has_value()) {
        RETURN_IF_ERROR(
            ValidateIsFiniteAndPositive(GetL0Sensitivity(), "L0 sensitivity"));
        double l0 = GetL0Sensitivity().value();
        RETURN_IF_ERROR(ValidateIsFiniteAndPositive(GetLInfSensitivity(),
                                                    "LInf sensitivity"));
        double linf = GetLInfSensitivity().value();
        double l1 = l0 * linf;
        if (!std::isfinite(l1)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The result of the L1 sensitivity calculation is not finite: ",
              l1,
              ". Please check your contribution and sensitivity settings."));
        }
        if (l1 == 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The result of the L1 sensitivity calculation is 0, likely "
              "because either L0 sensitivity (",
              l0, ") and/or LInf sensitivity (", linf,
              ") are too small. Please check your contribution and sensitivity "
              "settings."));
        }
        return l1;
      }
      // Sensitivity of 1 has been the default previously.  This will only
      // be set for LaplaceMechanism for backwards compatibility.
      return 1;
    }
  };

  ABSL_DEPRECATED(
      "Use LaplaceMechanism::Builder instead of LaplaceMechanism constructor.")
  explicit LaplaceMechanism(double epsilon, double sensitivity = 1.0)
      : NumericalMechanism(epsilon),
        sensitivity_(sensitivity),
        diversity_(sensitivity / epsilon) {
    absl::StatusOr<std::unique_ptr<internal::LaplaceDistribution>>
        status_or_distro = internal::LaplaceDistribution::Builder()
                               .SetEpsilon(GetEpsilon())
                               .SetSensitivity(sensitivity)
                               .Build();
    DCHECK(status_or_distro.status().ok())
        << status_or_distro.status().message();
    distro_ = std::move(status_or_distro.value());
  }

  // Deserialize the LaplaceMechanism from a proto.
  static absl::StatusOr<std::unique_ptr<NumericalMechanism>> Deserialize(
      const serialization::LaplaceMechanism& proto) {
    Builder builder;
    if (proto.has_epsilon()) {
      builder.SetEpsilon(proto.epsilon());
    }
    if (proto.has_l1_sensitivity()) {
      builder.SetL1Sensitivity(proto.l1_sensitivity());
    }
    return builder.Build();
  }

  LaplaceMechanism(double epsilon, double sensitivity,
                   std::unique_ptr<internal::LaplaceDistribution> distro)
      : LaplaceMechanism(epsilon, sensitivity) {
    distro_ = std::move(distro);
  }

  virtual ~LaplaceMechanism() = default;

  using NumericalMechanism::AddNoise;

  // Quickly determines if result is greater than threshold.
  bool NoisedValueAboveThreshold(double result, double threshold) override {
    return GetUniformDouble() >
           internal::LaplaceDistribution::cdf(diversity_, threshold - result);
  }

  double ProbabilityOfNoisedValueAboveThreshold(double result,
                                                double threshold) override {
    return 1 -
           internal::LaplaceDistribution::cdf(diversity_, threshold - result);
  }

  virtual double GetUniformDouble() { return distro_->GetUniformDouble(); }

  // Returns the confidence interval of the specified confidence level of the
  // noise that AddNoise() would add.
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].
  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) override {
    return NoiseConfidenceInterval(confidence_level, 0);
  }

  NoiseConfidenceIntervalResult UncheckedNoiseConfidenceInterval(
      double confidence_level, double noised_result) const override {
    const double bound = diversity_ * std::log(1.0 - confidence_level);
    NoiseConfidenceIntervalResult ci;
    // bound is negative as log(x) with 0 < x < 1 is negative.
    ci.lower = noised_result + bound;
    ci.upper = noised_result - bound;
    return ci;
  }

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double noised_result) override {
    RETURN_IF_ERROR(CheckConfidenceLevel(confidence_level));
    NoiseConfidenceIntervalResult ci =
        UncheckedNoiseConfidenceInterval(confidence_level, noised_result);
    ConfidenceInterval result;
    result.set_lower_bound(ci.lower);
    result.set_upper_bound(ci.upper);
    result.set_confidence_level(confidence_level);
    return result;
  }

  serialization::LaplaceMechanism Serialize() const {
    serialization::LaplaceMechanism output;
    output.set_epsilon(NumericalMechanism::GetEpsilon());
    output.set_l1_sensitivity(sensitivity_);
    return output;
  }

  // Returns the memory usage of the mechanism.
  virtual int64_t MemoryUsed() {
    int64_t memory = sizeof(LaplaceMechanism);
    if (distro_) {
      memory += distro_->MemoryUsed();
    }
    return memory;
  }

  double GetSensitivity() const { return sensitivity_; }

  // Returns the calculated diversity of the underlying laplace distribution.
  double GetDiversity() const { return diversity_; }
  static double GetMinEpsilon() {
    return internal::LaplaceDistribution::GetMinEpsilon();
  }

  double GetVariance() const override { return distro_->GetVariance(); }

  double Cdf(double x) const override {
    return internal::LaplaceDistribution::cdf(diversity_, x);
  }

  double Quantile(double p) const override {
    return internal::LaplaceDistribution::Quantile(diversity_, p);
  }

 protected:
  // Adds differentially private noise to a provided value.
  double AddDoubleNoise(double result) override {
    double sample = distro_->Sample();
    return RoundToNearestMultiple(result, distro_->GetGranularity()) + sample;
  }

  // Adds noise to an integral value (that could overflow). Calling AddNoise on
  // 0.0 avoids casting result to a double value, which could cause a SIGILL
  // error and is not secure privacy-wise, as it can have unforeseen effects on
  // the sensitivity of x. Rounding and adding the noise to result is a
  // privacy-safe operation (for noise of moderate magnitude, i.e. < 2^53).
  int64_t AddInt64Noise(int64_t result) override {
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

 private:
  const double sensitivity_;
  const double diversity_;
  std::unique_ptr<internal::LaplaceDistribution> distro_;
};

class GaussianMechanism : public NumericalMechanism {
 public:
  class Builder : public NumericalMechanismBuilder {
   public:
    Builder& SetL2Sensitivity(double l2_sensitivity) {
      l2_sensitivity_ = l2_sensitivity;
      return *this;
    }

    absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      internal::GaussianDistribution::Builder builder;
      ASSIGN_OR_RETURN(std::unique_ptr<internal::GaussianDistribution> distro,
                       builder.SetStddev(1).Build());

      absl::optional<double> epsilon = GetEpsilon();
      RETURN_IF_ERROR(ValidateIsFiniteAndPositive(epsilon, "Epsilon"));
      RETURN_IF_ERROR(DeltaIsSetAndValid());
      ASSIGN_OR_RETURN(double l2, CalculateL2Sensitivity());

      return absl::StatusOr<std::unique_ptr<NumericalMechanism>>(
          absl::make_unique<GaussianMechanism>(
              epsilon.value(), GetDelta().value(), l2, std::move(distro)));
    }

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }

   protected:
    absl::optional<double> l2_sensitivity_;

   private:
    // Returns the l2 sensitivity when it has been set or returns an upper bound
    // on the l2 sensitivity calculated from l0 and linf sensitivities.
    absl::StatusOr<double> CalculateL2Sensitivity() {
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
        RETURN_IF_ERROR(ValidateIsFiniteAndPositive(GetLInfSensitivity(),
                                                    "LInf sensitivity"));
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
  };

  ABSL_DEPRECATED(
      "Use GaussianMechanism::Builder instead of GaussianMechanism "
      "constructor.")
  explicit GaussianMechanism(double epsilon, double delta,
                             double l2_sensitivity)
      : NumericalMechanism(epsilon),
        delta_(delta),
        l2_sensitivity_(l2_sensitivity) {
    absl::StatusOr<std::unique_ptr<internal::GaussianDistribution>>
        status_or_distro =
            internal::GaussianDistribution::Builder().SetStddev(1).Build();
    DCHECK(status_or_distro.status().ok());
    standard_gaussian_ = std::move(status_or_distro.value());
  }

  GaussianMechanism(
      double epsilon, double delta, double l2_sensitivity,
      std::unique_ptr<internal::GaussianDistribution> standard_gaussian)
      : GaussianMechanism(epsilon, delta, l2_sensitivity) {
    standard_gaussian_ = std::move(standard_gaussian);
  }

  // Deserialize the GaussianMechanism from a proto.
  static absl::StatusOr<std::unique_ptr<NumericalMechanism>> Deserialize(
      const serialization::GaussianMechanism& proto) {
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

  virtual ~GaussianMechanism() = default;

  using NumericalMechanism::AddNoise;

  // Quickly determines if result is greater than threshold.
  bool NoisedValueAboveThreshold(double result, double threshold) override {
    double stddev = CalculateStddev();
    return UniformDouble() >
           internal::GaussianDistribution::cdf(stddev, threshold - result);
  }

  double ProbabilityOfNoisedValueAboveThreshold(double result,
                                                double threshold) override {
    double stddev = CalculateStddev();
    return 1 - internal::GaussianDistribution::cdf(stddev, threshold - result);
  }

  serialization::GaussianMechanism Serialize() const {
    serialization::GaussianMechanism result;
    result.set_epsilon(NumericalMechanism::GetEpsilon());
    result.set_delta(delta_);
    result.set_l2_sensitivity(l2_sensitivity_);
    return result;
  }

  virtual int64_t MemoryUsed() {
    int64_t memory = sizeof(GaussianMechanism);
    if (standard_gaussian_) {
      memory += sizeof(internal::GaussianDistribution);
    }
    return memory;
  }

  // Returns the confidence interval of the specified confidence level of the
  // noise that AddNoise() would add.
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].
  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) override {
    return NoiseConfidenceInterval(confidence_level, 0);
  }

  NoiseConfidenceIntervalResult UncheckedNoiseConfidenceInterval(
      double confidence_level, double noised_result) const override {
    const double stddev =
        CalculateStddev(GetEpsilon(), delta_, l2_sensitivity_);
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

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double noised_result) override {
    RETURN_IF_ERROR(CheckConfidenceLevel(confidence_level));
    NoiseConfidenceIntervalResult ci =
        UncheckedNoiseConfidenceInterval(confidence_level, noised_result);
    ConfidenceInterval result;
    result.set_lower_bound(ci.lower);
    result.set_upper_bound(ci.upper);
    result.set_confidence_level(confidence_level);
    return result;
  }

  // Returns the standard deviation of the Gaussian noise necessary to obtain
  // (epsilon, delta)-differential privacy for the given L_2 sensitivity. The
  // result will deviate from the tightest possible value sigma_tight by at most
  // kGaussianSigmaAccuracy * sigma_tight. To be on the safe side, the lowest
  // result from this method is the minimum positive floating point number.
  //
  // This implementation uses a binary search. Its runtime is roughly
  // log(kGaussianSigmaAccuracy)
  // + log(max{sigma_tight / l2_sensitivity, l2_sensitivity / sigma_tight}).
  static double CalculateStddev(double epsilon, double delta,
                                double l2_sensitivity) {
    // l2_sensitivity is used as a starting guess for the upper bound, since
    // the required noise grows linearly with sensitivity.
    double upper_bound = l2_sensitivity;
    double lower_bound = std::numeric_limits<double>::min();

    // Increase lower_bound and upper_bound until upper_bound is actually an
    // upper bound of sigma_tight, using exponential search.
    while (CalculateDelta(upper_bound, epsilon, l2_sensitivity) > delta) {
      lower_bound = upper_bound;
      upper_bound = upper_bound * 2;
    }

    // Binary search [lower_bound, upper_bound] to find a good enough
    // approximation of sigma_tight.
    while (upper_bound - lower_bound > kGaussianSigmaAccuracy * lower_bound) {
      double middle = lower_bound * 0.5 + upper_bound * 0.5;
      if (CalculateDelta(middle, epsilon, l2_sensitivity) > delta) {
        lower_bound = middle;
      } else {
        upper_bound = middle;
      }
    }

    // Return the over-approximation to err on the safe side.
    return upper_bound;
  }

  double CalculateStddev() const {

    return CalculateStddev(GetEpsilon(), delta_, l2_sensitivity_);
  }

  double GetDelta() const { return delta_; }

  double GetL2Sensitivity() const { return l2_sensitivity_; }

  double GetVariance() const override {
    return std::pow(CalculateStddev(GetEpsilon(), GetDelta(), l2_sensitivity_),
                    2);
  }

  double Cdf(double x) const override {
    return internal::GaussianDistribution::cdf(CalculateStddev(), x);
  }

  double Quantile(double p) const override {
    return internal::GaussianDistribution::Quantile(CalculateStddev(), p);
  }

  // Returns the smallest delta such that the Gaussian mechanism with standard
  // deviation sigma obtains (epsilon, delta)-differential
  // privacy with respect to the provided L_2 sensitivity. The calculation is
  // based on Theorem 8 of Balle and Wang's "Improving the Gaussian Mechanism
  // for Differential Privacy: Analytical Calibration and Optimal Denoising",
  // available <a href="https://arxiv.org/abs/1805.06530v2">here</a>.
  static double CalculateDelta(double sigma, double epsilon,
                               double l2_sensitivity) {
    // Denoting by CDF the CDF function of the standard Gaussian distribution
    // (mean 0, variance 1), and s the L2 sensitivity, the tight choice of delta
    // is:
    //    CDF(s/(2*sigma) - epsilon*sigma/s) - exp(epsilon)*CDF(-s/(2*sigma) -
    //    epsilon*sigma/s)
    // To simplify the reasoning floating-point underflow/overflows, we rewrite
    // this formula into:
    //    CDF(a - b) - c * CDF(-a - b)
    // where a = s / (2 * sigma), b = epsilon * sigma / s, and c = exp(epsilon).
    double a = l2_sensitivity / (2 * sigma);
    double b = epsilon * sigma / l2_sensitivity;
    double c = std::exp(epsilon);

    if (std::isinf(b)) {
      // If either l2_sensitivity goes to 0 or e^epsilon goes to infinity,
      // delta goes to 0.
      return 0;
    }
    return StandardNormalDistributionCDF(a - b) -
           c * StandardNormalDistributionCDF(-a - b);
  }

 protected:
  // Adds differentially private noise to a provided value.
  double AddDoubleNoise(double result) override {
    double stddev = CalculateStddev(GetEpsilon(), delta_, l2_sensitivity_);
    double sample = standard_gaussian_->Sample(stddev);

    return RoundToNearestMultiple(result,
                                  standard_gaussian_->GetGranularity(stddev)) +
           sample;
  }

  int64_t AddInt64Noise(int64_t result) override {
    double stddev = CalculateStddev(GetEpsilon(), delta_, l2_sensitivity_);
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

 private:
  const double delta_;
  const double l2_sensitivity_;
  std::unique_ptr<internal::GaussianDistribution> standard_gaussian_;

  static double StandardNormalDistributionCDF(double x) {
    return internal::GaussianDistribution::cdf(1, x);
  }
};

// Mechanism builder that returns the mechanism with minimum variance for given
// parameters.  Chooses between Gaussian and Laplace mechanism.
class MinVarianceMechanismBuilder : public NumericalMechanismBuilder {
 public:
  // Returns the numerical mechanism with the lower variance.  If only one
  // mechanism can be build, this method returns that mechanism.  If no
  // mechanism can be build, this method returns an error.
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
    LaplaceMechanism::Builder laplace_builder;
    absl::StatusOr<std::unique_ptr<NumericalMechanism>> laplace =
        GetMechanismFromBuilder(laplace_builder);

    GaussianMechanism::Builder gaussian_builder;
    absl::StatusOr<std::unique_ptr<NumericalMechanism>> gaussian =
        GetMechanismFromBuilder(gaussian_builder);

    if (laplace.ok() && gaussian.ok()) {
      if (laplace.value()->GetVariance() < gaussian.value()->GetVariance()) {
        return laplace;
      } else {
        return gaussian;
      }
    }

    if (laplace.ok()) {
      return laplace;
    }

    if (gaussian.ok()) {
      return gaussian;
    }

    // Both builders returned errors, so we are also returning an error.
    if (laplace.status() == gaussian.status()) {
      return laplace.status();
    } else {
      return absl::Status(
          laplace.status().code(),
          absl::StrCat("Laplace and Gaussian returned different errors during "
                       "build. Laplace: ",
                       laplace.status().message(),
                       "; Gaussian: ", gaussian.status().message()));
    }
  }

  std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
    return absl::make_unique<MinVarianceMechanismBuilder>(*this);
  }

 private:
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> GetMechanismFromBuilder(
      NumericalMechanismBuilder& builder) {
    if (GetEpsilon()) {
      builder.SetEpsilon(GetEpsilon().value());
    }
    if (GetDelta().has_value()) {
      builder.SetDelta(GetDelta().value());
    }
    if (GetL0Sensitivity().has_value()) {
      builder.SetL0Sensitivity(GetL0Sensitivity().value());
    }
    if (GetLInfSensitivity().has_value()) {
      builder.SetLInfSensitivity(GetLInfSensitivity().value());
    }
    return builder.Build();
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_H_
