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
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "base/statusor.h"
#include "algorithms/distributions.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/rand.h"
#include "algorithms/util.h"
#include "base/canonical_errors.h"
#include "base/status_macros.h"

namespace differential_privacy {

// Provides a common abstraction for PartitionSelectionStrategy. Each partition
// selection strategy class has a builder with which it can be instantiated, and
// calling ShouldKeep will return true if a partition with the given number of
// users should be kept based on the values the partition selection strategy
// was instantiated with (while ShouldKeep will return false if the partition
// should have been dropped).
class PartitionSelectionStrategy {
 public:
  // Builder base class
  class Builder {
   public:
    virtual ~Builder() = default;

    Builder& SetEpsilon(double epsilon) {
      epsilon_ = epsilon;
      return *this;
    }

    Builder& SetDelta(double delta) {
      delta_ = delta;
      return *this;
    }

    Builder& SetMaxPartitionsContributed(int64_t max_partitions_contributed) {
      max_partitions_contributed_ = max_partitions_contributed;
      return *this;
    }

    virtual base::StatusOr<std::unique_ptr<PartitionSelectionStrategy>>
    Build() = 0;

   protected:
    // Convenience methods to check if the Builder variables are set & valid
    absl::Status EpsilonIsSetAndValid() {
      return PartitionSelectionStrategy::EpsilonIsSetAndValid(epsilon_);
    }
    absl::Status DeltaIsSetAndValid() {
      return PartitionSelectionStrategy::DeltaIsSetAndValid(delta_);
    }
    absl::Status MaxPartitionsContributedIsSetAndValid() {
      return PartitionSelectionStrategy::MaxPartitionsContributedIsSetAndValid(
          max_partitions_contributed_);
    }

    absl::optional<double> GetEpsilon() { return epsilon_; }

    absl::optional<double> GetDelta() { return delta_; }

    absl::optional<int64_t> GetMaxPartitionsContributed() {
      return max_partitions_contributed_;
    }

    absl::optional<double> delta_;

   private:
    absl::optional<double> epsilon_;
    absl::optional<int64_t> max_partitions_contributed_;
  };

  virtual ~PartitionSelectionStrategy() = default;

  double GetEpsilon() const { return epsilon_; }

  double GetDelta() const { return delta_; }

  int64_t GetMaxPartitionsContributed() const {
    return max_partitions_contributed_;
  }

  // ShouldKeep returns true when a partition with a given number of users
  // should be kept and false otherwise.
  virtual bool ShouldKeep(int num_users) = 0;

 protected:
  PartitionSelectionStrategy(double epsilon, double delta,
                             int64_t max_partitions_contributed,
                             double adjusted_delta)
      : epsilon_(epsilon),
        delta_(delta),
        max_partitions_contributed_(max_partitions_contributed),
        adjusted_delta_(adjusted_delta) {}

  // Checks if epsilon is set and valid.
  static absl::Status EpsilonIsSetAndValid(absl::optional<double> epsilon) {
    RETURN_IF_ERROR(ValidateIsFiniteAndPositive(epsilon, "Epsilon"));
    return absl::OkStatus();
  }

  // Checks if delta is set and valid.
  static absl::Status DeltaIsSetAndValid(absl::optional<double> delta) {
    RETURN_IF_ERROR(ValidateIsInInclusiveInterval(delta, 0, 1, "Delta"));
    return absl::OkStatus();
  }

  // Checks if the max number of partitions contributed to is set and valid.
  static absl::Status MaxPartitionsContributedIsSetAndValid(
      absl::optional<int64_t> max_partitions_contributed) {
    RETURN_IF_ERROR(ValidateIsPositive(
        max_partitions_contributed,
        "Max number of partitions a user can contribute to"));
    return absl::OkStatus();
  }

  double GetAdjustedDelta() const { return adjusted_delta_; }

  // We must derive an adjusted delta, to be used as the probability of keeping
  // a single partition with one user, from delta, the probability we keep any
  // of the partitions contributed to by a single user.  Since the probability
  // we drop a partition with a single user is 1 - adjusted_delta_, and raising
  // this expression to the power of the max number of partitions one user can
  // contribute to will get us delta, we can solve to get the following formula.
  static base::StatusOr<double> CalculateAdjustedDelta(
      double delta, int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(PartitionSelectionStrategy::DeltaIsSetAndValid(delta));
    RETURN_IF_ERROR(
        PartitionSelectionStrategy::MaxPartitionsContributedIsSetAndValid(
            max_partitions_contributed));

    // Numerically stable equivalent of
    // 1- pow(1 - delta, 1 / max_partitions_contributed).
    if (delta == 1) {  // Avoid NaN from log1p(-1) -> log(0)
      return 1;
    }
    return -expm1(log1p(-delta) / max_partitions_contributed);
  }

  // Inverse of CalculateAdjustedDelta()
  static base::StatusOr<double> CalculateUnadjustedDelta(
      double adjusted_delta, int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(
        PartitionSelectionStrategy::DeltaIsSetAndValid(adjusted_delta));
    RETURN_IF_ERROR(
        PartitionSelectionStrategy::MaxPartitionsContributedIsSetAndValid(
            max_partitions_contributed));

    // Numerically stable equivalent of
    // 1 - pow(1 - adjusted_delta, max_partitions_contributed).
    if (adjusted_delta == 1) {  // Avoid NaN from log1p(-1) -> log(0)
      return 1;
    }
    return -expm1(max_partitions_contributed * log1p(-adjusted_delta));
  }

 private:
  double epsilon_;
  double delta_;
  int max_partitions_contributed_;
  double adjusted_delta_;
};

// PreAggPartitionSelection implements magic partition selection - instead of
// calculating a specific threshold to determine whether or not a partition
// should be kept, magic partition selection uses a formula derived from the
// original probablistic definition of differential privacy to generate the
// probability with which a partition should be kept. The math is shown in
// https://arxiv.org/pdf/2006.03684.pdf.
class PreaggPartitionSelection : public PartitionSelectionStrategy {
 public:
  // Builder for PreaggPartitionSelection
  class Builder : public PartitionSelectionStrategy::Builder {
   public:
    base::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      RETURN_IF_ERROR(EpsilonIsSetAndValid());
      RETURN_IF_ERROR(DeltaIsSetAndValid());
      RETURN_IF_ERROR(MaxPartitionsContributedIsSetAndValid());

      ASSIGN_OR_RETURN(
          double adjusted_delta,
          CalculateAdjustedDelta(GetDelta().value(),
                                 GetMaxPartitionsContributed().value()));

      std::unique_ptr<PartitionSelectionStrategy> magic_selection =
          absl::WrapUnique(new PreaggPartitionSelection(
              GetEpsilon().value(), GetDelta().value(),
              GetMaxPartitionsContributed().value(), adjusted_delta));
      return magic_selection;
    }
  };

  virtual ~PreaggPartitionSelection() = default;

  double GetAdjustedEpsilon() const { return adjusted_epsilon_; }

  double GetFirstCrossover() const { return crossover_1_; }

  double GetSecondCrossover() const { return crossover_2_; }

  bool ShouldKeep(int num_users) override {
    // generate a random number between 0 and 1
    double rand_num = UniformDouble();
    // only keep partition if random number < expected probability of keep
    return (rand_num <= ProbabilityOfKeep(num_users));
  }

 protected:
  PreaggPartitionSelection(double epsilon, double delta, int max_partitions,
                           double adjusted_delta)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions,
                                   adjusted_delta),
        adjusted_epsilon_(epsilon / static_cast<double>(max_partitions)) {
    crossover_1_ =
        1 +
        floor(log1p(tanh(adjusted_epsilon_ / 2) * (1 / adjusted_delta - 1)) /
              adjusted_epsilon_);
    crossover_2_ =
        crossover_1_ + floor((1.0 / adjusted_epsilon_) *
                             log1p((expm1(adjusted_epsilon_) / adjusted_delta) *
                                   (1 - ProbabilityOfKeep(crossover_1_))));
  }

 private:
  double adjusted_epsilon_;
  double crossover_1_;
  double crossover_2_;

  // ProbabilityOfKeep returns the probability with which a partition with n
  // users should be kept, Thm. 1 of https://arxiv.org/pdf/2006.03684.pdf
  double ProbabilityOfKeep(double n) const {
    const double adjusted_delta = GetAdjustedDelta();
    if (n == 0) {
      return 0;
    } else if (n <= crossover_1_) {
      return ((expm1(n * adjusted_epsilon_) / expm1(adjusted_epsilon_)) *
              adjusted_delta);
    } else if (n > crossover_1_ && n <= crossover_2_) {
      const double m = n - crossover_1_;
      const double p_crossover = ProbabilityOfKeep(crossover_1_);
      return p_crossover -
             (1 - p_crossover + (adjusted_delta / expm1(adjusted_epsilon_))) *
                 expm1(-m * adjusted_epsilon_);
    } else {
      return 1;
    }
  }
};

// LaplacePartitionSelection calculates a threshold based on the CDF of the
// Laplace distribution, delta, epsilon, and the max number of partitions a
// single user can contribute to. If the number of users in a partition
// + Laplace noise is greater than this threshold, the partition should be kept.
class LaplacePartitionSelection : public PartitionSelectionStrategy {
 public:
  // Builder for LaplacePartitionSelection
  class Builder : public PartitionSelectionStrategy::Builder {
   public:
    Builder& SetLaplaceMechanism(
        std::unique_ptr<LaplaceMechanism::Builder> laplace_builder) {
      laplace_builder_ = std::move(laplace_builder);
      return *this;
    }

    base::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      RETURN_IF_ERROR(EpsilonIsSetAndValid());
      RETURN_IF_ERROR(DeltaIsSetAndValid());
      RETURN_IF_ERROR(MaxPartitionsContributedIsSetAndValid());
      if (laplace_builder_ == nullptr) {
        laplace_builder_ = absl::make_unique<LaplaceMechanism::Builder>();
      }

      double epsilon = GetEpsilon().value();
      double delta = GetDelta().value();
      int64_t max_partitions_contributed = GetMaxPartitionsContributed().value();

      ASSIGN_OR_RETURN(
          double adjusted_delta,
          CalculateAdjustedDelta(delta, max_partitions_contributed));

      ASSIGN_OR_RETURN(
          double threshold,
          CalculateThreshold(epsilon, delta, max_partitions_contributed));

      std::unique_ptr<NumericalMechanism> mechanism_;
      ASSIGN_OR_RETURN(mechanism_,
                       laplace_builder_->SetEpsilon(epsilon)
                           .SetL0Sensitivity(max_partitions_contributed)
                           .SetLInfSensitivity(1)
                           .Build());
      std::unique_ptr<PartitionSelectionStrategy> laplace =
          absl::WrapUnique(new LaplacePartitionSelection(
              epsilon, delta, max_partitions_contributed, adjusted_delta,
              threshold, std::move(mechanism_)));

      return laplace;
    }

   private:
    std::unique_ptr<LaplaceMechanism::Builder> laplace_builder_;
  };

  virtual ~LaplacePartitionSelection() = default;

  bool ShouldKeep(int num_users) override {
    return mechanism_->NoisedValueAboveThreshold(num_users, threshold_);
  }

  static base::StatusOr<double> CalculateDelta(
      double epsilon, double threshold, int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(PartitionSelectionStrategy::EpsilonIsSetAndValid(epsilon));
    RETURN_IF_ERROR(
        PartitionSelectionStrategy::MaxPartitionsContributedIsSetAndValid(
            max_partitions_contributed));

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

  static base::StatusOr<double> CalculateThreshold(
      double epsilon, double delta, int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(PartitionSelectionStrategy::EpsilonIsSetAndValid(epsilon));
    RETURN_IF_ERROR(PartitionSelectionStrategy::DeltaIsSetAndValid(delta));
    RETURN_IF_ERROR(
        PartitionSelectionStrategy::MaxPartitionsContributedIsSetAndValid(
            max_partitions_contributed));

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
  LaplacePartitionSelection(double epsilon, double delta,
                            int64_t max_partitions_contributed,
                            double adjusted_delta, double threshold,
                            std::unique_ptr<NumericalMechanism> laplace)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions_contributed,
                                   adjusted_delta),
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
  class Builder : public PartitionSelectionStrategy::Builder {
   public:
    Builder& SetGaussianMechanism(
        std::unique_ptr<GaussianMechanism::Builder> gaussian_builder) {
      gaussian_builder_ = std::move(gaussian_builder);
      return *this;
    }

    base::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      RETURN_IF_ERROR(EpsilonIsSetAndValid());
      RETURN_IF_ERROR(DeltaIsSetAndValid());
      RETURN_IF_ERROR(MaxPartitionsContributedIsSetAndValid());
      if (gaussian_builder_ == nullptr) {
        gaussian_builder_ = absl::make_unique<GaussianMechanism::Builder>();
      }

      double epsilon = GetEpsilon().value();
      double delta = GetDelta().value();
      int64_t max_partitions_contributed = GetMaxPartitionsContributed().value();
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

      ASSIGN_OR_RETURN(
          double adjusted_threshold_delta,
          CalculateAdjustedDelta(threshold_delta, max_partitions_contributed));

      std::unique_ptr<PartitionSelectionStrategy> gaussian =
          absl::WrapUnique(new GaussianPartitionSelection(
              epsilon, delta, threshold_delta, noise_delta,
              max_partitions_contributed, adjusted_threshold_delta, threshold,
              std::move(mechanism_)));

      return gaussian;
    }

   private:
    std::unique_ptr<GaussianMechanism::Builder> gaussian_builder_;
  };

  virtual ~GaussianPartitionSelection() = default;

  double GetThresholdDelta() const { return threshold_delta_; }

  double GetNoiseDelta() const { return noise_delta_; }

  bool ShouldKeep(int num_users) override {
    return mechanism_->NoisedValueAboveThreshold(num_users, threshold_);
  }

  // CalculateThresholdDelta returns the threshold_delta for a threshold k. This
  // is the inverse of CalculateThreshold.
  static base::StatusOr<double> CalculateThresholdDelta(
      double epsilon, double noise_delta, double threshold,
      int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(PartitionSelectionStrategy::EpsilonIsSetAndValid(epsilon));
    RETURN_IF_ERROR(DeltaIsSetAndValid(noise_delta));
    RETURN_IF_ERROR(
        PartitionSelectionStrategy::MaxPartitionsContributedIsSetAndValid(
            max_partitions_contributed));

    double sigma = GaussianMechanism::CalculateStddev(
        epsilon, noise_delta, max_partitions_contributed);

    const double max_contribution = 1;
    const double adjusted_threshold_delta =
        1 - internal::GaussianDistribution::cdf(sigma,
                                                threshold - max_contribution);

    return CalculateUnadjustedDelta(adjusted_threshold_delta,
                                    max_partitions_contributed);
  }

  // CalculateThreshold returns the smallest threshold k to use in a
  // differentially private histogram with added Gaussian noise.
  //
  // See
  // https://github.com/google/differential-privacy/blob/main/common_docs/Delta_For_Thresholding.pdf
  // for details on the math underlying this.
  static base::StatusOr<double> CalculateThreshold(
      double epsilon, double noise_delta, double threshold_delta,
      int64_t max_partitions_contributed) {
    RETURN_IF_ERROR(PartitionSelectionStrategy::EpsilonIsSetAndValid(epsilon));
    RETURN_IF_ERROR(DeltaIsSetAndValid(noise_delta));
    RETURN_IF_ERROR(DeltaIsSetAndValid(threshold_delta));
    RETURN_IF_ERROR(
        PartitionSelectionStrategy::MaxPartitionsContributedIsSetAndValid(
            max_partitions_contributed));

    double sigma = GaussianMechanism::CalculateStddev(
        epsilon, noise_delta, max_partitions_contributed);

    ASSIGN_OR_RETURN(
        double adjusted_threshold_delta,
        CalculateAdjustedDelta(threshold_delta, max_partitions_contributed));

    const double max_contribution = 1;

    return max_contribution + internal::GaussianDistribution::Quantile(
                                  sigma, 1 - adjusted_threshold_delta);
  }

  double GetThreshold() const { return threshold_; }

 protected:
  GaussianPartitionSelection(double epsilon, double delta,
                             double threshold_delta, double noise_delta,
                             int64_t max_partitions_contributed,
                             double adjusted_delta, double threshold,
                             std::unique_ptr<NumericalMechanism> gaussian)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions_contributed,
                                   adjusted_delta),
        threshold_delta_(threshold_delta),
        noise_delta_(noise_delta),
        threshold_(threshold),
        mechanism_(std::move(gaussian)) {}

  // CalculateDelta computes the smallest δ such that the Gaussian mechanism
  // with fixed standard deviation σ is (ε,δ)-differentially private. The
  // calculation is based on Theorem 8 of Balle and Wang's "Improving the
  // Gaussian Mechanism for Differential Privacy: Analytical Calibration and
  // Optimal Denoising" (https://arxiv.org/abs/1805.06530v2).
  static double CalculateDelta(double sigma, double epsilon,
                               int64_t max_partitions_contributed) {
    const double l2_sensitivity = std::sqrt(max_partitions_contributed);

    return GaussianMechanism::CalculateDelta(sigma, epsilon, l2_sensitivity);
  }

 private:
  double threshold_delta_;
  double noise_delta_;
  double threshold_;
  std::unique_ptr<NumericalMechanism> mechanism_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_PARTITION_SELECTION_H_
