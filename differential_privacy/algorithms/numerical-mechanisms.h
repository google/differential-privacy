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

#include "differential_privacy/algorithms/confidence-interval.pb.h"
#include "differential_privacy/algorithms/distributions.h"
#include "differential_privacy/algorithms/util.h"
#include "differential_privacy/base/canonical_errors.h"

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

template <typename T>
T ClampDouble(T lower, T upper, double value) {
  if (value > upper) {
    return upper;
  }
  if (value < lower) {
    return lower;
  }
  return value;
}

// Provides a common abstraction for NumericalMechanism.  Numerical mechanisms
// can add noise to data and track the remaining privacy budget.
class NumericalMechanism {
 public:
  virtual ~NumericalMechanism() = default;

  virtual double AddNoise(double result, double privacy_budget) = 0;

  double AddNoise(double result) { return AddNoise(result, 1.0); }

  virtual int64_t MemoryUsed() = 0;
};

// Provides a common abstraction for Builders for NumericalMechanism.
template <class Builder, class Mechanism>
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
};

// Provides differential privacy by adding Laplace noise. This class also
// contains supporting functions related to the noise added.
class LaplaceMechanism : public NumericalMechanism {
 public:
  // Builder for LaplaceMechanism.
  class Builder : public NumericalMechanismBuilder<Builder, LaplaceMechanism> {
   public:
    // This method is deprecated and SetL1Sensitivity() should be used instead.
    Builder& SetSensitivity(double l1_sensitivity) {
      return SetL1Sensitivity(l1_sensitivity);
    }

    Builder& SetL1Sensitivity(double sensitivity_l1) {
      l1_sensitivity_ = sensitivity_l1;
      return *this;
    }

    base::StatusOr<std::unique_ptr<LaplaceMechanism>> Build() override {
      if (!epsilon_.has_value()) {
        // Epsilon has been previously set to 1 as default.  Need to check if
        // anything breaks.
        return base::InvalidArgumentError(
            "Laplace Mechanism requires epsilon to be set.");
      }
      if (std::isnan(epsilon_.value()) ||
          std::abs(epsilon_.value()) ==
              std::numeric_limits<double>::infinity()) {
        return base::InvalidArgumentError("Epsilon must be a real number.");
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
          NumericalMechanismBuilder<Builder, LaplaceMechanism>::Clone();
      if (l1_sensitivity_.has_value()) {
        clone->l1_sensitivity_ = l1_sensitivity_.value();
      }
      return clone;
    }

   protected:
    absl::optional<double> l1_sensitivity_;
  };

  explicit LaplaceMechanism(double epsilon, double sensitivity = 1.0)
      : epsilon_(epsilon),
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
    if (privacy_budget <= 0) {
      privacy_budget = std::numeric_limits<double>::min();
    }
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
    return ClampDouble<double>(LowerBound<double>(), UpperBound<double>(),
                               rounded_result);
  }

  virtual double GetUniformDouble() { return distro_->GetUniformDouble(); }

  // Returns the confidence interval of the specified confidence level of the
  // noise that AddNoise() would add with the specified privacy budget, as shown
  // in: (broken link)
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].
  virtual ConfidenceInterval NoiseConfidenceInterval(double confidence_level,
                                                     double privacy_budget) {
    ConfidenceInterval confidence;
    if (epsilon_ <= 0 || privacy_budget <= 0) {
      confidence.set_lower_bound(std::numeric_limits<double>::lowest());
      confidence.set_upper_bound(std::numeric_limits<double>::max());
    }

    double bound = diversity_ * log(1 - confidence_level) / privacy_budget;
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

  double GetEpsilon() { return epsilon_; }
  double GetSensitivity() { return sensitivity_; }

  // Returns the calculated diversity of the underlying laplace distribution.
  double GetDiversity() { return diversity_; }

 private:
  double epsilon_;
  double sensitivity_;
  double diversity_;
  std::unique_ptr<internal::LaplaceDistribution> distro_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_H_
