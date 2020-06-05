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

#include <limits>
#include <memory>
#include <string>

#include "algorithms/distributions.h"
#include "algorithms/util.h"
#include "proto/confidence-interval.pb.h"
#include "base/canonical_errors.h"

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
      double confidence_level, double privacy_budget) {
    return base::UnimplementedError(
        "NoiseConfidenceInterval() unsupported for this numerical mechanism.");
  }

  double GetEpsilon() { return epsilon_; }

 protected:
  double epsilon_;

  base::Status CheckConfidenceLevel(double confidence_level) {
    if (!(0 < confidence_level && confidence_level < 1)) {
      return base::InvalidArgumentError(absl::StrCat(
          "Confidence level has to be in the open interval (0,1), but is ",
          confidence_level));
    }
    return base::OkStatus();
  }

  base::Status CheckPrivacyBudget(double privacy_budget) {
    if (!(0 < privacy_budget && privacy_budget <= 1)) {
      return base::InvalidArgumentError(absl::StrCat(
          "privacy_budget has to be in the interval (0, 1], but is ",
          privacy_budget));
    }
    return base::OkStatus();
  }

  // Checks and clamps the budget so that it is in the interval (0,1].
  double CheckAndClampBudget(double privacy_budget) {
    base::Status status = CheckPrivacyBudget(privacy_budget);
    LOG_IF(ERROR, !status.ok()) << status.message();
    return Clamp<double>(std::numeric_limits<double>::min(), 1, privacy_budget);
  }
};

// Provides a common abstraction for Builders for NumericalMechanism.
template <class Mechanism, class Builder = typename Mechanism::Builder>
class NumericalMechanismBuilder {
 public:
  virtual ~NumericalMechanismBuilder() = default;

  Builder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *static_cast<Builder*>(this);
  }

  Builder& SetDelta(double delta) {
    delta_ = delta;
    return *static_cast<Builder*>(this);
  }

  Builder& SetL0Sensitivity(double l0_sensitivity) {
    l0_sensitivity_ = l0_sensitivity;
    return *static_cast<Builder*>(this);
  }

  Builder& SetLInfSensitivity(double linf_sensitivity) {
    linf_sensitivity_ = linf_sensitivity;
    return *static_cast<Builder*>(this);
  }

  virtual base::StatusOr<std::unique_ptr<Mechanism>> Build() = 0;

  virtual std::unique_ptr<Builder> Clone() const {
    Builder clone;
    if (epsilon_.has_value()) {
      clone.SetEpsilon(epsilon_.value());
    }
    if (delta_.has_value()) {
      clone.SetDelta(delta_.value());
    }
    if (l0_sensitivity_.has_value()) {
      clone.SetL0Sensitivity(l0_sensitivity_.value());
    }
    if (linf_sensitivity_.has_value()) {
      clone.SetLInfSensitivity(linf_sensitivity_.value());
    }
    return absl::make_unique<Builder>(clone);
  }

 protected:
  absl::optional<double> epsilon_;
  absl::optional<double> delta_;
  absl::optional<double> l0_sensitivity_;
  absl::optional<double> linf_sensitivity_;

  // Checks if epsilon is set and valid to be used for building any of the
  // mechanisms.
  base::Status EpsilonIsSetAndValid() {
    if (!epsilon_.has_value()) {
      return base::InvalidArgumentError("Epsilon has to be set.");
    }
    if (!std::isfinite(epsilon_.value())) {
      return base::InvalidArgumentError(
          absl::StrCat("Epsilon has to be finite but is ", epsilon_.value()));
    }
    if (epsilon_.value() <= 0) {
      return base::InvalidArgumentError(
          absl::StrCat("Epsilon has to be positive but is ", epsilon_.value()));
    }
    return base::OkStatus();
  }

  // Checks if delta is set and valid to be used in the Gaussian mechanism.
  base::Status DeltaIsSetAndValid() {
    if (!delta_.has_value()) {
      return base::InvalidArgumentError("Delta has to be set.");
    }
    if (!std::isfinite(delta_.value())) {
      return base::InvalidArgumentError(
          absl::StrCat("Delta has to be finite but is ", delta_.value()));
    }
    if (delta_.value() < 0 || 1 <= delta_.value()) {
      return base::InvalidArgumentError(absl::StrCat(
          "Delta has to be in the interval [0,1) but is ", delta_.value()));
    }
    return base::OkStatus();
  }
};

// Provides differential privacy by adding Laplace noise. This class also
// contains supporting functions related to the noise added.
class LaplaceMechanism : public NumericalMechanism {
 public:
  // Builder for LaplaceMechanism.
  class Builder : public NumericalMechanismBuilder<LaplaceMechanism> {
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

    base::StatusOr<std::unique_ptr<LaplaceMechanism>> Build() override {
      base::Status epsilon_status = EpsilonIsSetAndValid();
      if (!epsilon_status.ok()) {
        return epsilon_status;
      }
      // Check if L1 sensitivity is provided or make an estimate.
      if (!l1_sensitivity_.has_value()) {
        if (l0_sensitivity_.has_value() && linf_sensitivity_.has_value()) {
          l1_sensitivity_ = l0_sensitivity_.value() * linf_sensitivity_.value();
        } else {
          // Sensitivity of 1 has been the default previously.  This will only
          // be set for LaplaceMechanism for backwards compatabability.
          l1_sensitivity_ = 1;
        }
      }
      // Check that generated noise is not likely to overflow.
      double diversity = l1_sensitivity_.value() / epsilon_.value();
      double overflow_probability =
          (1 - internal::LaplaceDistribution::cdf(
                   diversity, std::numeric_limits<double>::max())) +
          internal::LaplaceDistribution::cdf(
              diversity, std::numeric_limits<double>::lowest());
      if (overflow_probability >= kMaxOverflowProbability) {
        return base::InvalidArgumentError("Sensitivity is too high.");
      }

      return absl::make_unique<LaplaceMechanism>(epsilon_.value(),
                                                l1_sensitivity_.value());
    }

    std::unique_ptr<Builder> Clone() const override {
      std::unique_ptr<Builder> clone =
          NumericalMechanismBuilder<LaplaceMechanism>::Clone();
      if (l1_sensitivity_.has_value()) {
        clone->l1_sensitivity_ = l1_sensitivity_.value();
      }
      return clone;
    }

   protected:
    absl::optional<double> l1_sensitivity_;
  };

  explicit LaplaceMechanism(double epsilon, double sensitivity = 1.0)
      : NumericalMechanism(epsilon),
        sensitivity_(sensitivity),
        diversity_(sensitivity / epsilon),
        distro_(absl::make_unique<internal::LaplaceDistribution>(
            epsilon_, sensitivity_)) {}

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
    // Implements the snapping mechanism defined by
    // Mironov (2012, "On Significance of the Least Significant Bits For
    // Differential Privacy").
    double noise = distro_->Sample(1.0 / privacy_budget);
    double noised_result =
        Clamp<double>(LowerBound<double>(), UpperBound<double>(), result) +
        noise;
    double nearest_power = GetNextPowerOfTwo(diversity_ / privacy_budget);
    double rounded_result =
        RoundToNearestMultiple(noised_result, nearest_power);
    return Clamp<double>(LowerBound<double>(), UpperBound<double>(),
                         rounded_result);
  }

  virtual double GetUniformDouble() { return distro_->GetUniformDouble(); }

  // Returns the confidence interval of the specified confidence level of the
  // noise that AddNoise() would add with the specified privacy budget.
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].
  virtual base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget) override {
    base::Status status = CheckConfidenceLevel(confidence_level);
    status.Update(CheckPrivacyBudget(privacy_budget));
    if (!status.ok()) {
      return status;
    }

    double bound = diversity_ * log(1 - confidence_level) / privacy_budget;

    ConfidenceInterval confidence;
    confidence.set_lower_bound(bound);
    confidence.set_upper_bound(-bound);
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

  double GetSensitivity() { return sensitivity_; }

  // Returns the calculated diversity of the underlying laplace distribution.
  double GetDiversity() { return diversity_; }

 private:
  double sensitivity_;
  double diversity_;
  std::unique_ptr<internal::LaplaceDistribution> distro_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_H_
