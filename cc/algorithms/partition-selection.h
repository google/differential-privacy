//
// Copyright 2020 Google LLC
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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_PARTITION_SELECTION_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_PARTITION_SELECTION_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/distributions.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/rand.h"
#include "algorithms/util.h"
#include "base/status_macros.h"

namespace differential_privacy {

// Provides a common abstraction for PartitionSelectionStrategy. Calling
// ShouldKeep will return true if a partition with the given number of users
// should be kept based on the values the partition selection strategy was
// instantiated with (while ShouldKeep will return false if the partition should
// have been dropped).
class PartitionSelectionStrategy {
 public:
  virtual ~PartitionSelectionStrategy() = default;

  double GetEpsilon() const { return epsilon_; }

  double GetDelta() const { return delta_; }

  int64_t GetMaxPartitionsContributed() const {
    return max_partitions_contributed_;
  }

  // This is set with the results from `CalculateAdjustedDelta`.
  double GetAdjustedDelta() const { return adjusted_delta_; }

  int GetPreThreshold() const { return pre_threshold_; }

  // ShouldKeep returns true when a partition with a given number of users
  // should be kept and false otherwise.
  virtual bool ShouldKeep(double num_users) = 0;

  virtual double ProbabilityOfKeep(double num_users) const = 0;

 protected:
  [[deprecated(
      "Deprecated in favour of the one that also supports setting pre_threshold"
      "")]] PartitionSelectionStrategy(double epsilon, double delta,
                                       int64_t max_partitions_contributed,
                                       double adjusted_delta)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions_contributed,
                                   adjusted_delta, 1) {}
  PartitionSelectionStrategy(double epsilon, double delta,
                             int64_t max_partitions_contributed,
                             double adjusted_delta, int pre_threshold)
      : epsilon_(epsilon),
        delta_(delta),
        max_partitions_contributed_(max_partitions_contributed),
        adjusted_delta_(adjusted_delta),
        pre_threshold_(pre_threshold) {}

  // We must derive an adjusted delta, to be used as the probability of keeping
  // a single partition with one user, from delta, the probability we keep any
  // of the partitions contributed to by a single user.  Since the probability
  // we drop a partition with a single user is 1 - adjusted_delta_, and raising
  // this expression to the power of the max number of partitions one user can
  // contribute to will get us delta, we can solve to get the following formula.
  static absl::StatusOr<double> CalculateAdjustedDelta(
      double delta, int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(ValidateDelta(delta));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed));

    // Numerically stable equivalent of
    // 1- pow(1 - delta, 1 / max_partitions_contributed).
    if (delta == 1) {  // Avoid NaN from log1p(-1) -> log(0)
      return 1;
    }
    return -std::expm1(log1p(-delta) / max_partitions_contributed);
  }

  // Inverse of CalculateAdjustedDelta()
  static absl::StatusOr<double> CalculateUnadjustedDelta(
      double adjusted_delta, int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(ValidateDelta(adjusted_delta));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed));

    // Numerically stable equivalent of
    // 1 - pow(1 - adjusted_delta, max_partitions_contributed).
    if (adjusted_delta == 1) {  // Avoid NaN from log1p(-1) -> log(0)
      return 1;
    }
    return -std::expm1(max_partitions_contributed * log1p(-adjusted_delta));
  }

 private:
  double epsilon_;
  double delta_;
  int max_partitions_contributed_;
  double adjusted_delta_;
  const int pre_threshold_;
};

// Provides a common abstraction for PartitionSelectionStrategy builders. Each
// partition selection strategy builder inherits from this builder.
class PartitionSelectionStrategyBuilder {
 public:
  virtual ~PartitionSelectionStrategyBuilder() = default;

  PartitionSelectionStrategyBuilder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  PartitionSelectionStrategyBuilder& SetDelta(double delta) {
    delta_ = delta;
    return *this;
  }

  PartitionSelectionStrategyBuilder& SetMaxPartitionsContributed(
      int64_t max_partitions_contributed) {
    max_partitions_contributed_ = max_partitions_contributed;
    return *this;
  }

  PartitionSelectionStrategyBuilder& SetPreThreshold(int pre_threshold) {
    pre_threshold_ = pre_threshold;
    return *this;
  }

  virtual absl::StatusOr<std::unique_ptr<PartitionSelectionStrategy>>
  Build() = 0;

 protected:
  std::optional<double> GetEpsilon() { return epsilon_; }

  std::optional<double> GetDelta() { return delta_; }

  std::optional<int> GetPreThreshold() { return pre_threshold_; }

  std::optional<int64_t> GetMaxPartitionsContributed() {
    return max_partitions_contributed_;
  }

 private:
  std::optional<double> epsilon_;
  std::optional<double> delta_;
  std::optional<int> pre_threshold_;
  std::optional<int64_t> max_partitions_contributed_;
};

// NearTruncatedGeometricPartitionSelection implements magic partition selection
// - instead of calculating a specific threshold to determine whether or not a
// partition should be kept, magic partition selection uses a formula derived
// from the original probablistic definition of differential privacy to generate
// the probability with which a partition should be kept. The math is shown in
// https://arxiv.org/pdf/2006.03684.pdf. We use the term "near truncated
// geometric" because the selection algorithm's output distribution is close to
// that of a thresholding approach based on truncated geometric noise (Theorem 5
// in the aforementioned paper).
class NearTruncatedGeometricPartitionSelection
    : public PartitionSelectionStrategy {
 public:
  // Builder for NearTruncatedGeometricPartitionSelection
  class Builder : public PartitionSelectionStrategyBuilder {
   public:
    absl::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      RETURN_IF_ERROR(ValidateEpsilon(GetEpsilon()));
      RETURN_IF_ERROR(ValidateDelta(GetDelta()));
      RETURN_IF_ERROR(
          ValidateMaxPartitionsContributed(GetMaxPartitionsContributed()));
      RETURN_IF_ERROR(ValidatePreThresholdOptional(GetPreThreshold()));

      ASSIGN_OR_RETURN(
          double adjusted_delta,
          CalculateAdjustedDelta(GetDelta().value(),
                                 GetMaxPartitionsContributed().value()));

      std::unique_ptr<PartitionSelectionStrategy> magic_selection =
          absl::WrapUnique(new NearTruncatedGeometricPartitionSelection(
              GetEpsilon().value(), GetDelta().value(),
              GetMaxPartitionsContributed().value(), adjusted_delta,
              GetPreThreshold().value_or(1)));
      return magic_selection;
    }
  };

  virtual ~NearTruncatedGeometricPartitionSelection() = default;

  double GetAdjustedEpsilon() const { return adjusted_epsilon_; }

  double GetFirstCrossover() const { return crossover_1_; }

  double GetSecondCrossover() const { return crossover_2_; }

  bool ShouldKeep(double num_users) override {
    // generate a random number between 0 and 1
    double rand_num = UniformDouble();
    // only keep partition if random number < expected probability of keep
    return (rand_num <= ProbabilityOfKeep(num_users));
  }

  // ProbabilityOfKeep returns the probability with which a partition with
  // num_users
  // users should be kept, Thm. 1 of https://arxiv.org/pdf/2006.03684.pdf
  double ProbabilityOfKeep(double num_users) const override {
    const double adjusted_delta = GetAdjustedDelta();
    if (num_users < GetPreThreshold()) {
      return 0;
    } else if (num_users <= crossover_1_) {
      return (
          (std::expm1((num_users - GetPreThreshold() + 1) * adjusted_epsilon_) /
           std::expm1(adjusted_epsilon_)) *
          adjusted_delta);
    } else if (num_users > crossover_1_ && num_users <= crossover_2_) {
      const double m = num_users - crossover_1_;
      const double p_crossover = ProbabilityOfKeep(crossover_1_);
      return p_crossover - (1 - p_crossover +
                            (adjusted_delta / std::expm1(adjusted_epsilon_))) *
                               std::expm1(-m * adjusted_epsilon_);
    } else {
      return 1;
    }
  }

 protected:
  NearTruncatedGeometricPartitionSelection(double epsilon, double delta,
                                           int max_partitions,
                                           double adjusted_delta)
      : NearTruncatedGeometricPartitionSelection(epsilon, delta, max_partitions,
                                                 adjusted_delta, 1) {}

  NearTruncatedGeometricPartitionSelection(double epsilon, double delta,
                                           int max_partitions,
                                           double adjusted_delta,
                                           int pre_threshold)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions,
                                   adjusted_delta, pre_threshold),
        adjusted_epsilon_(epsilon / static_cast<double>(max_partitions)) {
    SetCrossovers(adjusted_epsilon_, adjusted_delta);
  }

 private:
  inline void SetCrossovers(double adjusted_epsilon, double adjusted_delta) {
    crossover_1_ =
        1 +
        floor(log1p(tanh(adjusted_epsilon_ / 2) * (1 / adjusted_delta - 1)) /
              adjusted_epsilon_) +
        GetPreThreshold() - 1;
    crossover_2_ =
        crossover_1_ +
        floor((1.0 / adjusted_epsilon_) *
              log1p((std::expm1(adjusted_epsilon_) / adjusted_delta) *
                    (1 - ProbabilityOfKeep(crossover_1_))));
  }

  double adjusted_epsilon_;
  double crossover_1_;
  double crossover_2_;
};

// PreaggPartitionSelection is the deprecated name for
// NearTruncatedGeometricPartitionSelection.
using PreaggPartitionSelection = NearTruncatedGeometricPartitionSelection;

// LaplacePartitionSelection calculates a threshold based on the CDF of the
// Laplace distribution, delta, epsilon, and the max number of partitions a
// single user can contribute to. If the number of users in a partition
// + Laplace noise is greater than this threshold, the partition should be kept.
class LaplacePartitionSelection : public PartitionSelectionStrategy {
 public:
  // Builder for LaplacePartitionSelection
  class Builder : public PartitionSelectionStrategyBuilder {
   public:
    Builder& SetLaplaceMechanism(
        std::unique_ptr<LaplaceMechanism::Builder> laplace_builder) {
      laplace_builder_ = std::move(laplace_builder);
      return *this;
    }

    absl::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      RETURN_IF_ERROR(ValidateEpsilon(GetEpsilon()));
      RETURN_IF_ERROR(ValidateDelta(GetDelta()));
      RETURN_IF_ERROR(
          ValidateMaxPartitionsContributed(GetMaxPartitionsContributed()));
      RETURN_IF_ERROR(ValidatePreThresholdOptional(GetPreThreshold()));

      if (laplace_builder_ == nullptr) {
        laplace_builder_ = absl::make_unique<LaplaceMechanism::Builder>();
      }

      double epsilon = GetEpsilon().value();
      double delta = GetDelta().value();
      int64_t max_partitions_contributed =
          GetMaxPartitionsContributed().value();

      ASSIGN_OR_RETURN(
          double adjusted_delta,
          CalculateAdjustedDelta(delta, max_partitions_contributed));

      ASSIGN_OR_RETURN(
          double threshold,
          CalculateThreshold(epsilon, delta, max_partitions_contributed));
      threshold += GetPreThreshold().value_or(1) - 1;

      std::unique_ptr<NumericalMechanism> mechanism_;
      ASSIGN_OR_RETURN(mechanism_,
                       laplace_builder_->SetEpsilon(epsilon)
                           .SetL0Sensitivity(max_partitions_contributed)
                           .SetLInfSensitivity(1)
                           .Build());
      std::unique_ptr<PartitionSelectionStrategy> laplace =
          absl::WrapUnique(new LaplacePartitionSelection(
              epsilon, delta, max_partitions_contributed, adjusted_delta,
              GetPreThreshold().value_or(1), threshold, std::move(mechanism_)));

      return laplace;
    }

   private:
    std::unique_ptr<LaplaceMechanism::Builder> laplace_builder_;
  };

  virtual ~LaplacePartitionSelection() = default;

  bool ShouldKeep(double num_users) override {
    if (num_users < GetPreThreshold()) {
      return false;
    }
    return mechanism_->NoisedValueAboveThreshold(num_users, threshold_);
  }

  std::optional<double> NoiseValueIfShouldKeep(double num_users) {
    if (num_users < GetPreThreshold()) {
      return std::optional<double>();
    }
    double noised_value = mechanism_->AddNoise(num_users);
    return noised_value > threshold_ ? noised_value : std::optional<double>();
  }

  double ProbabilityOfKeep(double num_users) const override {
    if (num_users < GetPreThreshold()) {
      return 0;
    }
    return mechanism_->ProbabilityOfNoisedValueAboveThreshold(num_users,
                                                              threshold_);
  }

  static absl::StatusOr<double> CalculateDelta(
      double epsilon, double threshold, int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(ValidateEpsilon(epsilon));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed));

    if (threshold < 1) {
      return CalculateUnadjustedDelta(
          1 - (std::exp(
                   (threshold - 1) /
                   CalculateDiversity(epsilon, max_partitions_contributed)) /
               2),
          max_partitions_contributed);
    } else {
      return CalculateUnadjustedDelta(
          (std::exp((1 - threshold) /
                    CalculateDiversity(epsilon, max_partitions_contributed))) /
              2,
          max_partitions_contributed);
    }
  }

  static absl::StatusOr<double> CalculateThreshold(
      double epsilon, double delta, int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(ValidateEpsilon(epsilon));
    RETURN_IF_ERROR(ValidateDelta(delta));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed));

    ASSIGN_OR_RETURN(double adjusted_delta,
                     CalculateAdjustedDelta(delta, max_partitions_contributed));

    if (delta > 0.5) {
      return 1 + CalculateDiversity(epsilon, max_partitions_contributed) *
                     std::log(2 * (1 - adjusted_delta));
    } else {
      return 1 - CalculateDiversity(epsilon, max_partitions_contributed) *
                     (std::log(2 * adjusted_delta));
    }
  }

  int64_t GetL1Sensitivity() const { return l1_sensitivity_; }

  double GetDiversity() const { return diversity_; }

  double GetThreshold() const { return threshold_; }

 protected:
  [[deprecated(
      "Deprecated in favour of the one that also supports setting pre_threshold"
      "")]] LaplacePartitionSelection(double epsilon, double delta,
                                      int64_t max_partitions_contributed,
                                      double adjusted_delta, double threshold,
                                      std::unique_ptr<NumericalMechanism>
                                          laplace)
      : LaplacePartitionSelection(epsilon, delta, max_partitions_contributed,
                                  adjusted_delta, 1, threshold,
                                  std::move(laplace)) {}
  LaplacePartitionSelection(double epsilon, double delta,
                            int64_t max_partitions_contributed,
                            double adjusted_delta, int pre_threshold,
                            double threshold,
                            std::unique_ptr<NumericalMechanism> laplace)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions_contributed,
                                   adjusted_delta, pre_threshold),
        l1_sensitivity_(max_partitions_contributed),
        diversity_(CalculateDiversity(epsilon, l1_sensitivity_)),
        threshold_(threshold),
        mechanism_(std::move(laplace)) {}

  static double CalculateDiversity(double epsilon, int64_t l1_sensitivity) {
    return l1_sensitivity / epsilon;
  }

 private:
  int64_t l1_sensitivity_;
  double diversity_;
  // threshold_ includes the pre_threshold value as well.
  double threshold_;
  std::unique_ptr<NumericalMechanism> mechanism_;
};

// GaussianPartitionSelection calculates a threshold based on the CDF of the
// Gaussian distribution, delta, epsilon, and the max number of partitions a
// single user can contribute to. If the number of users in a partition
// + Gaussian noise is greater than this threshold, the partition should be
// kept.
class GaussianPartitionSelection : public PartitionSelectionStrategy {
 public:
  // Builder for GaussianPartitionSelection
  class Builder : public PartitionSelectionStrategyBuilder {
   public:
    Builder& SetGaussianMechanism(
        std::unique_ptr<GaussianMechanism::Builder> gaussian_builder) {
      gaussian_builder_ = std::move(gaussian_builder);
      return *this;
    }

    absl::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      RETURN_IF_ERROR(ValidateEpsilon(GetEpsilon()));
      RETURN_IF_ERROR(ValidateDelta(GetDelta()));
      RETURN_IF_ERROR(
          ValidateMaxPartitionsContributed(GetMaxPartitionsContributed()));
      RETURN_IF_ERROR(ValidatePreThresholdOptional(GetPreThreshold()));
      if (gaussian_builder_ == nullptr) {
        gaussian_builder_ = absl::make_unique<GaussianMechanism::Builder>();
      }

      double epsilon = GetEpsilon().value();
      double delta = GetDelta().value();
      int64_t max_partitions_contributed =
          GetMaxPartitionsContributed().value();
      double threshold_delta = delta / 2.;
      double noise_delta = delta - threshold_delta;

      std::unique_ptr<NumericalMechanism> mechanism_;
      ASSIGN_OR_RETURN(mechanism_,
                       gaussian_builder_->SetEpsilon(epsilon)
                           .SetL0Sensitivity(max_partitions_contributed)
                           .SetLInfSensitivity(1)
                           .SetDelta(noise_delta)
                           .Build());

      ASSIGN_OR_RETURN(double threshold,
                       CalculateThreshold(epsilon, noise_delta, threshold_delta,
                                          max_partitions_contributed));
      threshold += GetPreThreshold().value_or(1) - 1;

      ASSIGN_OR_RETURN(
          double adjusted_threshold_delta,
          CalculateAdjustedDelta(threshold_delta, max_partitions_contributed));

      std::unique_ptr<PartitionSelectionStrategy> gaussian =
          absl::WrapUnique(new GaussianPartitionSelection(
              epsilon, delta, threshold_delta, noise_delta,
              max_partitions_contributed, adjusted_threshold_delta,
              GetPreThreshold().value_or(1), threshold, std::move(mechanism_)));

      return gaussian;
    }

   private:
    std::unique_ptr<GaussianMechanism::Builder> gaussian_builder_;
  };

  virtual ~GaussianPartitionSelection() = default;

  double GetThresholdDelta() const { return threshold_delta_; }

  double GetNoiseDelta() const { return noise_delta_; }

  bool ShouldKeep(double num_users) override {
    if (num_users < GetPreThreshold()) {
      return false;
    }
    return mechanism_->NoisedValueAboveThreshold(num_users, threshold_);
  }

  std::optional<double> NoiseValueIfShouldKeep(double num_users) {
    if (num_users < GetPreThreshold()) {
      return std::optional<double>();
    }
    double noised_value = mechanism_->AddNoise(num_users);
    return noised_value > threshold_ ? noised_value : std::optional<double>();
  }

  double ProbabilityOfKeep(double num_users) const override {
    if (num_users < GetPreThreshold()) {
      return 0;
    }
    return mechanism_->ProbabilityOfNoisedValueAboveThreshold(num_users,
                                                              threshold_);
  }

  // CalculateThresholdDelta returns the threshold_delta for a threshold k. This
  // is the inverse of CalculateThreshold.
  static absl::StatusOr<double> CalculateThresholdDelta(
      double epsilon, double noise_delta, double threshold,
      int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(ValidateEpsilon(epsilon));
    RETURN_IF_ERROR(ValidateDelta(noise_delta));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed));

    double sigma = GaussianMechanism::CalculateStddev(
        epsilon, noise_delta, std::sqrt(max_partitions_contributed));

    const double max_contribution = 1;
    const double adjusted_threshold_delta =
        1 - internal::GaussianDistribution::cdf(sigma,
                                                threshold - max_contribution);

    return CalculateUnadjustedDelta(adjusted_threshold_delta,
                                    max_partitions_contributed);
  }

  // CalculateThreshold returns the smallest threshold k to use in a
  // differentially private histogram with added Gaussian noise of the given
  // standard deviation.
  //
  // See
  // https://github.com/google/differential-privacy/blob/main/common_docs/Delta_For_Thresholding.pdf
  // for details on the math underlying this.
  static absl::StatusOr<double> CalculateThresholdFromStddev(
      double stddev, double threshold_delta,
      int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(ValidateIsFiniteAndPositive(stddev, "Stddev"));
    RETURN_IF_ERROR(ValidateDelta(threshold_delta));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed));
    ASSIGN_OR_RETURN(
        double adjusted_threshold_delta,
        CalculateAdjustedDelta(threshold_delta, max_partitions_contributed));

    const double max_contribution = 1;

    // Note: Quantile(1-delta) = -Quantile(delta). We chose the second option
    // here, because for small delta, 1 - delta is approximately 1. This would
    // thus lead to a numerically unstable algorithm.
    return max_contribution - internal::GaussianDistribution::Quantile(
                                  stddev, adjusted_threshold_delta);
  }

  // Returns the smallest threshold k to use in a
  // differentially private histogram with added Gaussian noise.
  //
  // See
  // https://github.com/google/differential-privacy/blob/main/common_docs/Delta_For_Thresholding.pdf
  // for details on the math underlying this.
  static absl::StatusOr<double> CalculateThreshold(
      double epsilon, double noise_delta, double threshold_delta,
      int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(ValidateEpsilon(epsilon));
    RETURN_IF_ERROR(ValidateDelta(noise_delta));
    RETURN_IF_ERROR(ValidateDelta(threshold_delta));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed));

    const double stddev = GaussianMechanism::CalculateStddev(
        epsilon, noise_delta, std::sqrt(max_partitions_contributed));

    return CalculateThresholdFromStddev(stddev, threshold_delta,
                                        max_partitions_contributed);
  }

  double GetThreshold() const { return threshold_; }

 protected:
  [[deprecated(
      "Deprecated in favour of the one that also supports setting pre_threshold"
      "")]] GaussianPartitionSelection(double epsilon, double delta,
                                       double threshold_delta,
                                       double noise_delta,
                                       int64_t max_partitions_contributed,
                                       double adjusted_delta, double threshold,
                                       std::unique_ptr<NumericalMechanism>
                                           gaussian)
      : GaussianPartitionSelection(epsilon, delta, threshold_delta, noise_delta,
                                   max_partitions_contributed, adjusted_delta,
                                   1, threshold, std::move(gaussian)) {}
  GaussianPartitionSelection(double epsilon, double delta,
                             double threshold_delta, double noise_delta,
                             int64_t max_partitions_contributed,
                             double adjusted_delta, int pre_threshold,
                             double threshold,
                             std::unique_ptr<NumericalMechanism> gaussian)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions_contributed,
                                   adjusted_delta, pre_threshold),
        threshold_delta_(threshold_delta),
        noise_delta_(noise_delta),
        threshold_(threshold),
        mechanism_(std::move(gaussian)) {}

 private:
  double threshold_delta_;
  double noise_delta_;
  // threshold_ includes the pre_threshold value as well.
  double threshold_;
  std::unique_ptr<NumericalMechanism> mechanism_;
};

// Prethresholds the user count before delegating the partition selection logic
// to the wrapped partition selection strategy.
class [[deprecated(
    "This class is deprecated in favour of the pre_threshold attribute of "
    "other "
    "strategies classes.")]] PartitionSelectionStrategyWithPreThresholding
    : public PartitionSelectionStrategy {
 public:
  enum class PartitionSelectionStrategyType {
    kNearTruncatedGeometric,
    kLaplace,
    kGaussian
  };
  // Builder for PartitionSelectionStrategyWithPreThresholding
  class Builder {
   public:
    Builder& SetPreThreshold(int pre_threshold) {
      pre_threshold_ = pre_threshold;
      return *this;
    }

    Builder& SetEpsilon(double epsilon) {
      epsilon_ = epsilon;
      return *this;
    }

    Builder& SetDelta(double delta) {
      delta_ = delta;
      return *this;
    }

    Builder& SetMaxPartitionsContributed(int max_partitions_contributed) {
      max_partitions_contributed_ = max_partitions_contributed;
      return *this;
    }

    Builder& SetPartitionSelectionStrategy(
        PartitionSelectionStrategyType strategy_type) {
      switch (strategy_type) {
        case PartitionSelectionStrategyType::kNearTruncatedGeometric:
          strategy_builder_ = std::make_unique<
              NearTruncatedGeometricPartitionSelection::Builder>();
          break;
        case PartitionSelectionStrategyType::kLaplace:
          strategy_builder_ =
              std::make_unique<LaplacePartitionSelection::Builder>();
          break;
        case PartitionSelectionStrategyType::kGaussian:
          strategy_builder_ =
              std::make_unique<GaussianPartitionSelection::Builder>();
          break;
          // No default. Builder guarantees an error will be thrown if unset.
      }
      return *this;
    }

    // These partition selection strategy setters are used in mocking tests.
    Builder& SetPartitionSelectionStrategy(
        std::unique_ptr<NearTruncatedGeometricPartitionSelection::Builder>
            strategy_builder) {
      strategy_builder_ = std::move(strategy_builder);
      return *this;
    }

    Builder& SetPartitionSelectionStrategy(
        std::unique_ptr<LaplacePartitionSelection::Builder> strategy_builder) {
      strategy_builder_ = std::move(strategy_builder);
      return *this;
    }

    Builder& SetPartitionSelectionStrategy(
        std::unique_ptr<GaussianPartitionSelection::Builder> strategy_builder) {
      strategy_builder_ = std::move(strategy_builder);
      return *this;
    }

    absl::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build() {
      RETURN_IF_ERROR(ValidateEpsilon(epsilon_));
      RETURN_IF_ERROR(ValidateDelta(delta_));
      RETURN_IF_ERROR(
          ValidateMaxPartitionsContributed(max_partitions_contributed_));
      RETURN_IF_ERROR(ValidatePreThreshold(pre_threshold_));
      RETURN_IF_ERROR(ValidatePartitionStrategy());
      ASSIGN_OR_RETURN(
          std::unique_ptr<PartitionSelectionStrategy> wrapped_strategy,
          strategy_builder_->SetEpsilon(epsilon_.value())
              .SetDelta(delta_.value())
              .SetMaxPartitionsContributed(max_partitions_contributed_.value())
              .Build());
      return std::unique_ptr<PartitionSelectionStrategyWithPreThresholding>(
          new PartitionSelectionStrategyWithPreThresholding(
              pre_threshold_.value(), std::move(wrapped_strategy)));
    }

   private:
    std::optional<int> pre_threshold_;
    std::unique_ptr<PartitionSelectionStrategyBuilder> strategy_builder_;
    std::optional<double> epsilon_;
    std::optional<double> delta_;
    std::optional<int> max_partitions_contributed_;

    absl::Status ValidatePartitionStrategy() {
      if (strategy_builder_ == nullptr) {
        return absl::InvalidArgumentError(
            "Partition Selection Strategy must be set.");
      }
      return absl::OkStatus();
    }
  };

  double GetPreThreshold() const { return pre_threshold_; }

  bool ShouldKeep(double num_users) override {
    double thresholded_num_users = num_users - (pre_threshold_ - 1);
    if (thresholded_num_users <= 0) return false;
    return wrapped_strategy_->ShouldKeep(thresholded_num_users);
  }

  double ProbabilityOfKeep(double num_users) const override {
    double thresholded_num_users = num_users - (pre_threshold_ - 1);
    if (thresholded_num_users <= 0) return 0;
    return wrapped_strategy_->ProbabilityOfKeep(thresholded_num_users);
  }

 protected:
  PartitionSelectionStrategyWithPreThresholding(
      int pre_threshold, std::unique_ptr<PartitionSelectionStrategy> strategy)
      : PartitionSelectionStrategy(strategy->GetEpsilon(), strategy->GetDelta(),
                                   strategy->GetMaxPartitionsContributed(),
                                   strategy->GetAdjustedDelta()),
        pre_threshold_(pre_threshold),
        wrapped_strategy_(std::move(strategy)) {}

 private:
  const int pre_threshold_;
  std::unique_ptr<PartitionSelectionStrategy> wrapped_strategy_;
};
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_PARTITION_SELECTION_H_
