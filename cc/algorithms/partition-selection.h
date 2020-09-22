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

#include <math.h>

#include <iostream>
#include <string>
#include <utility>

#include "base/statusor.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/rand.h"
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

    Builder& SetMaxPartitionsContributed(int max_partitions_contributed) {
      max_partitions_contributed_ = max_partitions_contributed;
      return *this;
    }

    virtual base::StatusOr<std::unique_ptr<PartitionSelectionStrategy>>
    Build() = 0;

   protected:
    // Checks if the max number of partitions contributed  to is set and valid.
    base::Status MaxPartitionsContributedIsSetAndValid() {
      if (!max_partitions_contributed_.has_value()) {
        return base::InvalidArgumentError(
            "Max number of partitions a user can contribute to has to be set.");
      }
      if (max_partitions_contributed_.value() <= 0) {
        return base::InvalidArgumentError(absl::StrCat(
            "Max number of partitions a user can contribute to has to be"
            " positive but is ",
            max_partitions_contributed_.value()));
      }
      return base::OkStatus();
    }

    // Checks if delta is set and valid.
    base::Status DeltaIsSetAndValid() {
      if (!delta_.has_value()) {
        return base::InvalidArgumentError("Delta has to be set.");
      }
      if (!std::isfinite(delta_.value())) {
        return base::InvalidArgumentError(
            absl::StrCat("Delta has to be finite but is ", delta_.value()));
      }
      if (delta_.value() <= 0 || delta_.value() >= 1) {
        return base::InvalidArgumentError(absl::StrCat(
            "Delta has to be in the interval (0,1) but is ", delta_.value()));
      }
      return base::OkStatus();
    }

    absl::optional<double> GetEpsilon() { return epsilon_; }

    absl::optional<double> GetDelta() { return delta_; }

    absl::optional<int> GetMaxPartitionsContributed() {
      return max_partitions_contributed_;
    }

   private:
    absl::optional<double> epsilon_;
    absl::optional<double> delta_;
    absl::optional<int> max_partitions_contributed_;
  };

  virtual ~PartitionSelectionStrategy() = default;

  double GetEpsilon() const { return epsilon_; }

  double GetDelta() const { return delta_; }

  double GetMaxPartitionsContributed() const {
    return max_partitions_contributed_;
  }

  // ShouldKeep returns true when a partition with a given number of users
  // should be kept and false otherwise.
  virtual bool ShouldKeep(int num_users) = 0;

 protected:
  PartitionSelectionStrategy(double epsilon, double delta,
                             int max_partitions_contributed)
      : epsilon_(epsilon),
        delta_(delta),
        max_partitions_contributed_(max_partitions_contributed),
        adjusted_delta_(AdjustDelta(delta)) {}

 private:
  double epsilon_;
  double delta_;
  int max_partitions_contributed_;
  double adjusted_delta_;

 protected:
  double GetAdjustedDelta() const { return adjusted_delta_; }

  // We must derive adjusted_delta_, the probability of keeping a single
  // partition with one user, from delta, the probability we keep any of the
  // partitions contributed to by a single user.  Since the probability we drop
  // a partition with a single user is 1 - adjusted_delta_, and raising this
  // expression to the power of the max number of partitions one user can
  // contribute to will get us delta, we can solve to get the following formula.
  double AdjustDelta(double delta) const {
    // Numerically stable equivalent of
    // 1- pow(1 - delta, 1 / max_partitions_contributed_).
    return -expm1(log1p(-delta) / max_partitions_contributed_);
  }
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
      std::unique_ptr<PartitionSelectionStrategy> magic_selection =
          absl::WrapUnique(new PreaggPartitionSelection(
              GetEpsilon().value(), GetDelta().value(),
              GetMaxPartitionsContributed().value()));
      return magic_selection;
    }

   private:
    base::Status EpsilonIsSetAndValid() {
      if (!GetEpsilon().has_value()) {
        return base::InvalidArgumentError("Epsilon has to be set.");
      }
      if (!std::isfinite(GetEpsilon().value())) {
        return base::InvalidArgumentError(absl::StrCat(
            "Epsilon has to be finite but is ", GetEpsilon().value()));
      }
      if (GetEpsilon().value() <= 0) {
        return base::InvalidArgumentError(absl::StrCat(
            "Epsilon has to be positive but is ", GetEpsilon().value()));
      }
      return base::OkStatus();
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
  PreaggPartitionSelection(double epsilon, double delta, int max_partitions)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions),
        adjusted_epsilon_(epsilon / static_cast<double>(max_partitions)) {
    const double adjusted_delta = GetAdjustedDelta();
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
      RETURN_IF_ERROR(MaxPartitionsContributedIsSetAndValid());
      RETURN_IF_ERROR(DeltaIsSetAndValid());
      if (!GetEpsilon().has_value()) {
        return base::InvalidArgumentError("Epsilon has to be set.");
      }
      if (laplace_builder_ == nullptr) {
        laplace_builder_ = absl::make_unique<LaplaceMechanism::Builder>();
      }
      std::unique_ptr<NumericalMechanism> mechanism_;
      ASSIGN_OR_RETURN(
          mechanism_,
          laplace_builder_->SetEpsilon(GetEpsilon().value())
              .SetL0Sensitivity(GetMaxPartitionsContributed().value())
              .SetLInfSensitivity(1)
              .Build());
      std::unique_ptr<PartitionSelectionStrategy> laplace =
          absl::WrapUnique(new LaplacePartitionSelection(
              GetEpsilon().value(), GetDelta().value(),
              GetMaxPartitionsContributed().value(), std::move(mechanism_)));
      return laplace;
    }

   private:
    std::unique_ptr<LaplaceMechanism::Builder> laplace_builder_;
  };

  virtual ~LaplacePartitionSelection() = default;

  bool ShouldKeep(int num_users) override {
    const double noised_result =
        mechanism_->AddNoise(static_cast<double>(num_users), 1.0);
    return (noised_result > threshold_);
  }

  double GetL1Sensitivity() const { return l1_sensitivity_; }

  double GetDiversity() const { return diversity_; }

  double GetThreshold() const { return threshold_; }

 protected:
  LaplacePartitionSelection(double epsilon, double delta,
                            int max_partitions_contributed,
                            std::unique_ptr<NumericalMechanism> laplace)
      : PartitionSelectionStrategy(epsilon, delta, max_partitions_contributed),
        l1_sensitivity_(max_partitions_contributed),
        diversity_(l1_sensitivity_ / epsilon),
        mechanism_(std::move(laplace)) {
    threshold_ = 1 - diversity_ * (log(2 * GetAdjustedDelta()));
  }

 private:
  double l1_sensitivity_;
  double diversity_;
  double threshold_;
  std::unique_ptr<NumericalMechanism> mechanism_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_PARTITION_SELECTION_H_
