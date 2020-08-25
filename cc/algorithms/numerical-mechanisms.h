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

#include <math.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <cstdint>
#include "base/logging.h"
#include "absl/base/attributes.h"
#include "absl/memory/memory.h"
#include "base/status.h"
#include "base/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "algorithms/distributions.h"
#include "algorithms/util.h"
#include "proto/confidence-interval.pb.h"
#include "base/canonical_errors.h"
#include "base/status_macros.h"

// A set of classes for adding differentially private noise to numerical data.
// Rather than sampling directly from the distribution, these classes allow
// passing in the differential privacy parameters (epsilon, delta, sensitivity).
// Then, given a numerical value they automatically calculate the parameters of
// the distribution based on the differential privacy parameters and return the
// value plus noise. They also handle any other necessary steps to achieve
// differentially private results (e.g. snapping).
namespace differential_privacy {

// Clamping factor.
// Using a factor of 2^39 means that the clamp+round-to-power-of-2 approach
// adds at most a factor of 2^-10 extra (i.e. around 0.1%) to the privacy
// budget (Theorem 1, Mironov 2012).
static const double kClampFactor = std::pow(2.0, 39);

// The maximum allowable probability that the noise will overflow.
static const double kMaxOverflowProbability = std::pow(2.0, -64);

// The relative accuracy at which to stop the binary search to find the tightest
// sigma such that Gaussian noise satisfies (epsilon, delta)-differential
// privacy given the sensitivities.
static const double kGaussianSigmaAccuracy = 1e-3;

template <typename T>
T UpperBound() {
  if (std::numeric_limits<T>::max() > kClampFactor) {
    return static_cast<T>(kClampFactor);
  }
  return std::numeric_limits<T>::max();
}

template <typename T>
T LowerBound() {
  if (std::numeric_limits<T>::lowest() < -kClampFactor) {
    return static_cast<T>(-kClampFactor);
  }
  return std::numeric_limits<T>::lowest();
}

// Provides a common abstraction for NumericalMechanism.  Numerical mechanisms
// can add noise to data and track the remaining privacy budget.
class NumericalMechanism {
 public:
  NumericalMechanism(double epsilon) : epsilon_(epsilon) {}

  virtual ~NumericalMechanism() = default;

  virtual double AddNoise(double result, double privacy_budget) = 0;

  double AddNoise(double result) { return AddNoise(result, 1.0); }

  virtual int64_t MemoryUsed() = 0;

  virtual base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget, double noised_result) {
    return base::UnimplementedError(
        "NoiseConfidenceInterval() unsupported for this numerical mechanism.");
  }

  virtual base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget) {
    return base::UnimplementedError(
        "NoiseConfidenceInterval() unsupported for this numerical mechanism.");
  }

  double GetEpsilon() { return epsilon_; }

 protected:
  base::Status CheckConfidenceLevel(double confidence_level) {
    if (std::isnan(confidence_level) ||
        !(0 < confidence_level && confidence_level < 1)) {
      return base::InvalidArgumentError(absl::StrCat(
          "Confidence level has to be in the open interval (0,1), but is ",
          confidence_level));
    }
    return base::OkStatus();
  }

  base::Status CheckPrivacyBudget(double privacy_budget) {
    if (std::isnan(privacy_budget) ||
        !(0 < privacy_budget && privacy_budget <= 1)) {
      return base::InvalidArgumentError(absl::StrCat(
          "privacy_budget has to be in the interval (0, 1], but is ",
          privacy_budget));
    }
    return base::OkStatus();
  }

  // Checks and clamps the budget so that it is in the interval (0,1].  Should
  // only be used in methods where we no longer can return an base::Status, as
  // it tries some recovery from invalid states.
  double CheckAndClampBudget(double privacy_budget) {
    base::Status status = CheckPrivacyBudget(privacy_budget);
    LOG_IF(ERROR, !status.ok()) << status.message();
    if (std::isnan(privacy_budget)) {
      // Recover from this invalid state by returning the minimal possible
      // privacy budget.
      return std::numeric_limits<double>::min();
    }
    return Clamp<double>(std::numeric_limits<double>::min(), 1, privacy_budget);
  }

 private:
  double epsilon_;
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

  virtual base::StatusOr<std::unique_ptr<NumericalMechanism>> Build() = 0;

  virtual std::unique_ptr<NumericalMechanismBuilder> Clone() const = 0;

 protected:
  // Checks if delta is set and valid to be used in the Gaussian mechanism.
  base::Status DeltaIsSetAndValid() const {
    ASSIGN_OR_RETURN(double delta, GetValueIfSetAndFinite(delta_, "Delta"));
    if (delta <= 0 || 1 <= delta) {
      return base::InvalidArgumentError(absl::StrCat(
          "Delta has to be in the interval (0, 1) but is ", delta));
    }
    return base::OkStatus();
  }

  absl::optional<double> GetEpsilon() const { return epsilon_; }
  absl::optional<double> GetDelta() const { return delta_; }
  absl::optional<double> GetL0Sensitivity() const { return l0_sensitivity_; }
  absl::optional<double> GetLInfSensitivity() const {
    return linf_sensitivity_;
  }

  // Returns the value of optional `opt` if it is set and finite.  Will return
  // an InvalidArgumentError otherwise that includes `name` in the error
  // message.
  static base::StatusOr<double> GetValueIfSetAndFinite(
      absl::optional<double> opt, absl::string_view name) {
    if (!opt.has_value()) {
      return base::InvalidArgumentError(absl::StrCat(name, " has to be set."));
    }
    if (!std::isfinite(opt.value())) {
      return base::InvalidArgumentError(
          absl::StrCat(name, " has to be finite but is ", opt.value()));
    }
    return opt.value();
  }

  // Returns the value of optional `opt` if it is set, finite, and positive.
  // Will return an InvalidArgumentError otherwise that includes `name` in the
  // error message.
  static base::StatusOr<double> GetValueIfSetAndPositive(
      absl::optional<double> opt, absl::string_view name) {
    ASSIGN_OR_RETURN(double d, GetValueIfSetAndFinite(opt, name));
    if (d <= 0) {
      return base::InvalidArgumentError(
          absl::StrCat(name, " has to be positive but is ", d));
    }
    return d;
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
        "Set LInf and L0 sensitivity instead or use SetL1Sensitivity in tests")
    Builder& SetSensitivity(double l1_sensitivity) {
      return SetL1Sensitivity(l1_sensitivity);
    }

    Builder& SetL1Sensitivity(double sensitivity_l1) {
      l1_sensitivity_ = sensitivity_l1;
      return *this;
    }

    base::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      ASSIGN_OR_RETURN(double epsilon,
                       GetValueIfSetAndPositive(GetEpsilon(), "Epsilon"));
      ASSIGN_OR_RETURN(double L1, CalculateL1Sensitivity());
      // Check that generated noise is not likely to overflow.
      double diversity = L1 / epsilon;
      double overflow_probability =
          (1 - internal::LaplaceDistribution::cdf(
                   diversity, std::numeric_limits<double>::max())) +
          internal::LaplaceDistribution::cdf(
              diversity, std::numeric_limits<double>::lowest());
      if (overflow_probability >= kMaxOverflowProbability) {
        return base::InvalidArgumentError("Sensitivity is too high.");
      }
      base::StatusOr<double> gran_or_status =
          internal::CalculateGranularity(epsilon, L1);
      if (!gran_or_status.ok()) return gran_or_status.status();

      std::unique_ptr<NumericalMechanism> result =
          absl::make_unique<LaplaceMechanism>(epsilon, L1);
      return result;
    }

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }

   protected:
    absl::optional<double> GetL1Sensitivity() const { return l1_sensitivity_; }

   private:
    absl::optional<double> l1_sensitivity_;

    // Returns the l1 sensitivity when it has been set or returns an upper bound
    // on the l1 sensitivity calculated from l0 and linf sensitivities.
    base::StatusOr<double> CalculateL1Sensitivity() {
      if (l1_sensitivity_.has_value()) {
        return GetValueIfSetAndPositive(l1_sensitivity_, "L1 sensitivity");
      }
      if (GetL0Sensitivity().has_value() && GetLInfSensitivity().has_value()) {
        ASSIGN_OR_RETURN(double l0, GetValueIfSetAndPositive(GetL0Sensitivity(),
                                                             "L0 sensitivity"));
        ASSIGN_OR_RETURN(
            double linf,
            GetValueIfSetAndPositive(GetLInfSensitivity(), "LInf sensitivity"));
        double l1 = l0 * linf;
        if (!std::isfinite(l1)) {
          return base::InvalidArgumentError(absl::StrCat(
              "The result of the L1 sensitivity calculation is not finite: ",
              l1,
              ". Please check your contribution and sensitivity settings."));
        }
        return l1;
      }
      // Sensitivity of 1 has been the default previously.  This will only
      // be set for LaplaceMechanism for backwards compatibility.
      return 1;
    }
  };

  explicit LaplaceMechanism(double epsilon, double sensitivity = 1.0)
      : NumericalMechanism(epsilon),
        sensitivity_(sensitivity),
        diversity_(sensitivity / epsilon),
        distro_(absl::make_unique<internal::LaplaceDistribution>(
            GetEpsilon(), sensitivity_)) {}

  LaplaceMechanism(double epsilon, double sensitivity,
                   std::unique_ptr<internal::LaplaceDistribution> distro)
      : LaplaceMechanism(epsilon, sensitivity) {
    distro_ = std::move(distro);
  }

  virtual ~LaplaceMechanism() = default;

  using NumericalMechanism::AddNoise;

  // Adds differentially private noise to a provided value. The privacy_budget
  // is multiplied with epsilon for this particular result. Privacy budget
  // should be in (0, 1], and is a way to divide an epsilon between multiple
  // values. For instance, if a user wanted to add noise to two different values
  // with a given epsilon then they could add noise to each value with a privacy
  // budget of 0.5 (or 0.4 and 0.6, etc).
  double AddNoise(double result, double privacy_budget) override {
    privacy_budget = CheckAndClampBudget(privacy_budget);
    double sample = distro_->Sample(1.0 / privacy_budget);
    return RoundToNearestMultiple(result, distro_->GetGranularity()) + sample;
  }

  virtual double GetUniformDouble() { return distro_->GetUniformDouble(); }

  // Returns the confidence interval of the specified confidence level of the
  // noise that AddNoise() would add with the specified privacy budget.
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].
  base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget) override {
    return NoiseConfidenceInterval(confidence_level, privacy_budget, 0);
  }

  base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget,
      double noised_result) override {
    RETURN_IF_ERROR(CheckConfidenceLevel(confidence_level));
    RETURN_IF_ERROR(CheckPrivacyBudget(privacy_budget));

    double bound = diversity_ * log(1 - confidence_level) / privacy_budget;

    ConfidenceInterval confidence;
    confidence.set_lower_bound(noised_result + bound);
    confidence.set_upper_bound(noised_result - bound);
    confidence.set_confidence_level(confidence_level);
    return confidence;
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

 private:
  double sensitivity_;
  double diversity_;
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

    base::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      ASSIGN_OR_RETURN(double epsilon,
                       GetValueIfSetAndPositive(GetEpsilon(), "Epsilon"));
      RETURN_IF_ERROR(DeltaIsSetAndValid());
      ASSIGN_OR_RETURN(double l2, CalculateL2Sensitivity());
      std::unique_ptr<NumericalMechanism> result =
          absl::make_unique<GaussianMechanism>(epsilon, GetDelta().value(), l2);
      return result;
    }

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }

   protected:
    absl::optional<double> l2_sensitivity_;

   private:
    // Returns the l2 sensitivity when it has been set or returns an upper bound
    // on the l2 sensitivity calculated from l0 and linf sensitivities.
    base::StatusOr<double> CalculateL2Sensitivity() {
      if (l2_sensitivity_.has_value()) {
        return GetValueIfSetAndPositive(l2_sensitivity_, "L2 sensitivity");
      } else if (GetL0Sensitivity().has_value() &&
                 GetLInfSensitivity().has_value()) {
        // Try to calculate L2 sensitivity from L0 and LInf sensitivities
        ASSIGN_OR_RETURN(double l0, GetValueIfSetAndPositive(GetL0Sensitivity(),
                                                             "L0 sensitivity"));
        ASSIGN_OR_RETURN(
            double linf,
            GetValueIfSetAndPositive(GetLInfSensitivity(), "LInf sensitivity"));
        double l2 = std::sqrt(l0) * linf;
        if (!std::isfinite(l2) || l2 <= 0) {
          return base::InvalidArgumentError(absl::StrCat(
              "The calculated L2 sensitivity has to be positive and finite but "
              "is ",
              l2,
              ". Contribution or sensitivity settings might be too high or too "
              "low."));
        }
        return l2;
      }
      return base::InvalidArgumentError(
          "Gaussian Mechanism requires either L2 sensitivity or both, L0 "
          "and LInf sensitivity to be set.");
    }
  };

  explicit GaussianMechanism(double epsilon, double delta,
                             double l2_sensitivity)
      : NumericalMechanism(epsilon),
        delta_(delta),
        l2_sensitivity_(l2_sensitivity),
        distro_(absl::make_unique<internal::GaussianDistribution>(1)) {}

  virtual ~GaussianMechanism() = default;

  using NumericalMechanism::AddNoise;

  // Adds differentially private noise to a provided value. The privacy_budget
  // is multiplied with epsilon and delta for this particular result. Privacy
  // budget should be in (0, 1], and is a way to divide an epsilon between
  // multiple values. For instance, if a user wanted to add noise to two
  // different values with a given epsilon and delta then they could add noise
  // to each value with a privacy budget of 0.5 (or 0.4 and 0.6, etc).
  double AddNoise(double result, double privacy_budget) override {
    privacy_budget = CheckAndClampBudget(privacy_budget);

    double local_epsilon = privacy_budget * GetEpsilon();
    double local_delta = privacy_budget * delta_;
    double stddev = CalculateStddev(local_epsilon, local_delta);
    double sample = distro_->Sample(stddev);

    return RoundToNearestMultiple(result, distro_->GetGranularity(stddev)) +
           sample;
  }

  virtual int64_t MemoryUsed() {
    int64_t memory = sizeof(GaussianMechanism);
    if (distro_) {
      memory += sizeof(internal::GaussianDistribution);
    }
    return memory;
  }

  // Returns the confidence interval of the specified confidence level of the
  // noise that AddNoise() would add with the specified privacy budget.
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].

  base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget) override {
    return NoiseConfidenceInterval(confidence_level, privacy_budget, 0);
  }

  base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget,
      double noised_result) override {
    RETURN_IF_ERROR(CheckConfidenceLevel(confidence_level));
    RETURN_IF_ERROR(CheckPrivacyBudget(privacy_budget));

    double local_epsilon = privacy_budget * GetEpsilon();
    double local_delta = privacy_budget * delta_;
    double stddev = CalculateStddev(local_epsilon, local_delta);

    ConfidenceInterval confidence;
    // calculated using the symmetric properties of the Gaussian distribution
    // and the cumulative distribution function for the distribution
    float bound =
        InverseErrorFunction(-1 * confidence_level) * stddev * std::sqrt(2);
    confidence.set_lower_bound(noised_result + bound);
    confidence.set_upper_bound(noised_result - bound);
    confidence.set_confidence_level(confidence_level);

    return confidence;
  }

  // Returns the standard deviation of the Gaussian noise necessary to obtain
  // (epsilon, delta)-differential privacy for the given L_2 sensitivity. The
  // result will deviate from the tightest possible value sigma_tight by at most
  // kGaussianSigmaAccuracy * sigma_tight.
  //
  // This implementation uses a binary search. Its runtime is roughly
  // log(kGaussianSigmaAccuracy)
  // + log(max{sigma_tight / l2_sensitivity_, l2_sensitivity_ / sigma_tight}).
  double CalculateStddev(double epsilon, double delta) {
    // l2_sensitivity_ is used as a starting guess for the upper bound, since
    // the required noise grows linearly with sensitivity.
    double upper_bound = l2_sensitivity_;
    double lower_bound = 0;

    // Increase lower_bound and upper_bound until upper_bound is actually an
    // upper bound of sigma_tight, using exponential search.
    while (CalculateDelta(upper_bound, epsilon) > delta) {
      lower_bound = upper_bound;
      upper_bound = upper_bound * 2;
    }

    // Binary search [lower_bound, upper_bound] to find a good enough
    // approximation of sigma_tight.
    while (upper_bound - lower_bound > kGaussianSigmaAccuracy * lower_bound) {
      double middle = lower_bound * 0.5 + upper_bound * 0.5;
      if (CalculateDelta(middle, epsilon) > delta) {
        lower_bound = middle;
      } else {
        upper_bound = middle;
      }
    }

    // Return the over-approximation to err on the safe side.
    return upper_bound;
  }

  double GetDelta() const { return delta_; }

  double GetL2Sensitivity() const { return l2_sensitivity_; }

 private:
  double delta_;
  double l2_sensitivity_;
  std::unique_ptr<internal::GaussianDistribution> distro_;

  double StandardNormalDistributionCDF(double x) {
    return (1 + std::erf(x / sqrt(2))) / 2;
  }

  // Returns the smallest delta such that the Gaussian mechanism with standard
  // deviation sigma obtains (epsilon, delta)-differential
  // privacy with respect to the provided L_2 sensitivity. The calculation is
  // based on Theorem 8 of Balle and Wang's "Improving the Gaussian Mechanism
  // for Differential Privacy: Analytical Calibration and Optimal Denoising",
  // available <a href="https://arxiv.org/abs/1805.06530v2">here</a>.

  double CalculateDelta(double sigma, double epsilon) {
    // Denoting by CDF the CDF function of the standard Gaussian distribution
    // (mean 0, variance 1), and s the L2 sensitivity, the tight choice of delta
    // is:
    //    CDF(s/(2*sigma) - epsilon*sigma/s) - exp(epsilon)*CDF(-s/(2*sigma) -
    //    epsilon*sigma/s)
    // To simplify the reasoning floating-point underflow/overflows, we rewrite
    // this formula into:
    //    CDF(a - b) - c * CDF(-a - b)
    // where a = s / (2 * sigma), b = epsilon * sigma / s, and c = exp(epsilon).
    double a = l2_sensitivity_ / (2 * sigma);
    double b = epsilon * sigma / l2_sensitivity_;
    double c = exp(epsilon);

    if (std::isinf(b)) {
      // If either l2_sensitivity_ goes to 0 or e^epsilon goes to infinity,
      // delta goes to 0.
      return 0;
    }
    return StandardNormalDistributionCDF(a - b) -
           c * StandardNormalDistributionCDF(-a - b);
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_H_
