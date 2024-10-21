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

#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

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

// Provides a common abstraction for NumericalMechanism.  Numerical mechanisms
// can add noise to data.
class NumericalMechanism {
 public:
  NumericalMechanism(double epsilon) : epsilon_(epsilon) {}

  virtual ~NumericalMechanism() = default;

  template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  int64_t AddNoise(T result) {
    return AddInt64Noise(result);
  }

  template <typename T,
            std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  double AddNoise(T result) {
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

  std::optional<double> GetEpsilon() const { return epsilon_; }
  std::optional<double> GetDelta() const { return delta_; }
  std::optional<double> GetL0Sensitivity() const { return l0_sensitivity_; }
  std::optional<double> GetLInfSensitivity() const { return linf_sensitivity_; }

 private:
  std::optional<double> epsilon_;
  std::optional<double> delta_;
  std::optional<double> l0_sensitivity_;
  std::optional<double> linf_sensitivity_;
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

    absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override;

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }

   protected:
    std::optional<double> GetL1Sensitivity() const { return l1_sensitivity_; }

   private:
    std::optional<double> l1_sensitivity_;
  };

  ABSL_DEPRECATED(
      "Use LaplaceMechanism::Builder instead of LaplaceMechanism constructor.")
  explicit LaplaceMechanism(double epsilon)
      : LaplaceMechanism(epsilon, /* sensitivity= */ 1.0) {}

  ABSL_DEPRECATED(
      "Use LaplaceMechanism::Builder instead of LaplaceMechanism constructor.")
  LaplaceMechanism(double epsilon, double sensitivity);

  LaplaceMechanism(double epsilon, double sensitivity,
                   std::unique_ptr<internal::LaplaceDistribution> distro)
      : LaplaceMechanism(epsilon, sensitivity) {
    distro_ = std::move(distro);
  }

  virtual ~LaplaceMechanism() = default;

  // Deserialize the LaplaceMechanism from a proto.
  static absl::StatusOr<std::unique_ptr<NumericalMechanism>> Deserialize(
      const serialization::LaplaceMechanism& proto);

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
      double confidence_level, double noised_result) const override;

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double noised_result) override;

  serialization::LaplaceMechanism Serialize() const;

  // Returns the memory usage of the mechanism.
  int64_t MemoryUsed() override;

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
  double AddDoubleNoise(double result) override;

  // Adds noise to an integral value (that could overflow). Calling AddNoise on
  // 0.0 avoids casting result to a double value, which could cause a SIGILL
  // error and is not secure privacy-wise, as it can have unforeseen effects on
  // the sensitivity of x. Rounding and adding the noise to result is a
  // privacy-safe operation (for noise of moderate magnitude, i.e. < 2^53).
  int64_t AddInt64Noise(int64_t result) override;

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

    // Allows users to set the noise standard deviation directly, skipping
    // the calculations based on differential privacy parameters. Only use this
    // if you know what you're doing - the code will apply the exact amount of
    // noise you specify, which might not be enough to achieve your desired
    // privacy guarantee.
    //
    // Either this, or the differential privacy parameters (epsilon and delta)
    // should be specified. In case both are specified, this method will return
    // an error.
    Builder& SetStandardDeviation(double stddev) {
      stddev_ = stddev;
      return *this;
    }

    absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override;

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }

   protected:
    std::optional<double> l2_sensitivity_;
    std::optional<double> stddev_;

   private:
    // Returns the l2 sensitivity when it has been set or returns an upper bound
    // on the l2 sensitivity calculated from l0 and linf sensitivities.
    absl::StatusOr<double> CalculateL2Sensitivity();
  };

  ABSL_DEPRECATED(
      "Use GaussianMechanism::Builder instead of GaussianMechanism "
      "constructor.")
  GaussianMechanism(double epsilon, double delta, double l2_sensitivity);

  ABSL_DEPRECATED(
      "Use GaussianMechanism::Builder instead of GaussianMechanism "
      "constructor.")
  GaussianMechanism(
      double epsilon, double delta, double l2_sensitivity,
      std::unique_ptr<internal::GaussianDistribution> standard_gaussian)
      : GaussianMechanism(epsilon, delta, l2_sensitivity) {
    standard_gaussian_ = std::move(standard_gaussian);
  }

  ABSL_DEPRECATED(
      "Use GaussianMechanism::Builder instead of GaussianMechanism "
      "constructor.")
  GaussianMechanism(
      double stddev,
      std::unique_ptr<internal::GaussianDistribution> standard_gaussian)
      : NumericalMechanism(0), delta_(0), l2_sensitivity_(0), stddev_(stddev) {
    standard_gaussian_ = std::move(standard_gaussian);
  }

  // Deserialize the GaussianMechanism from a proto.
  static absl::StatusOr<std::unique_ptr<NumericalMechanism>> Deserialize(
      const serialization::GaussianMechanism& proto);

  virtual ~GaussianMechanism() = default;

  using NumericalMechanism::AddNoise;

  // Quickly determines if result is greater than threshold.
  bool NoisedValueAboveThreshold(double result, double threshold) override;

  double ProbabilityOfNoisedValueAboveThreshold(double result,
                                                double threshold) override;

  serialization::GaussianMechanism Serialize() const;

  int64_t MemoryUsed() override;

  // Returns the confidence interval of the specified confidence level of the
  // noise that AddNoise() would add.
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].
  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) override {
    return NoiseConfidenceInterval(confidence_level, 0);
  }

  NoiseConfidenceIntervalResult UncheckedNoiseConfidenceInterval(
      double confidence_level, double noised_result) const override;

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double noised_result) override;

  // Returns the standard deviation of the Gaussian noise necessary to obtain
  // (epsilon, delta)-differential privacy for the given L_2 sensitivity. The
  // result will deviate from the tightest possible value sigma_tight by at most
  // kGaussianSigmaAccuracy * sigma_tight. To be on the safe side, the lowest
  // result from this method is the minimum positive floating point number.
  //
  // This implementation uses a binary search. Its runtime is roughly
  // log(kGaussianSigmaAccuracy)
  // + log(max{sigma_tight / l2_sensitivity, l2_sensitivity / sigma_tight}).
  //
  // The calculation is based on <a
  // href="https://arxiv.org/abs/1805.06530v2">Balle and Wang's "Improving the
  // Gaussian Mechanism for Differential Privacy: Analytical Calibration and
  // Optimal Denoising"</a>. The paper states that the lower bound on sigma from
  // the original analysis of the Gaussian mechanism (sigma ≥ sqrt(2 *
  // l2_sensitivity^2 * log(1.25/𝛿) / 𝜖^2)) is far from tight and binary search
  // can give us a better lower bound.
  static double CalculateStddev(double epsilon, double delta,
                                double l2_sensitivity);

  double CalculateStddev() const;

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

 protected:
  // Adds differentially private noise to a provided value.
  double AddDoubleNoise(double result) override;

  int64_t AddInt64Noise(int64_t result) override;

 private:
  const double delta_;
  const double l2_sensitivity_;
  std::unique_ptr<internal::GaussianDistribution> standard_gaussian_;
  const double stddev_;
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
