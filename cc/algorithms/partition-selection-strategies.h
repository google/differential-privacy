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

//TODO once this is done do a thorough comb through according to the style guide

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_PARTITION_SELECTION_STRATEGIES_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_PARTITION_SELECTION_STRATEGIES_H_

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "base/statusor.h"
#include "base/canonical_errors.h"
#include "algorithms/numerical-mechanisms.h"

namespace differential_privacy{

// Provides a common abstraction for PartitionSelectionStrategy.  Partition
// selection strategies decide whether or not a specific partition should be
// dropped or kept.
class PartitionSelectionStrategy {
 public:
  //Builder base class
  class Builder {
    public:
      virtual ~Builder() = default;

      Builder& SetEpsilon(double epsilon) {
        epsilon_ = epsilon;
        return *static_cast<Builder*>(this);
      }

      Builder& SetDelta(double delta) {
        delta_ = delta;
        return *static_cast<Builder*>(this);
      }

      Builder& SetMaxPartitionsContributed(int max_partitions_contributed) {
        max_partitions_contributed_ = max_partitions_contributed;
        return *static_cast<Builder*>(this);
      }

      virtual base::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build() = 0;

    protected:
      absl::optional<double> epsilon_;
      absl::optional<double> delta_;
      absl::optional<int> max_partitions_contributed_;

      //Checks if the max number of partitions contributed  to is set and valid.
      base::Status MaxPartitionsContributedIsSetAndValid() {
        if (!max_partitions_contributed_.has_value()) {
          return base::InvalidArgumentError("Max number of partitions a user can contribute to has to be set.");
        }
        if (max_partitions_contributed_ <= 0) {
          return base::InvalidArgumentError(absl::StrCat(
              "The maximum number of partitions a user can contribute to has to be positive but is ", delta_.value()));
        }
        return base::OkStatus();
      }
  };

  PartitionSelectionStrategy(double epsilon, double delta, int max_partitions_contributed)
  	: epsilon_(epsilon), delta_(delta),
      adjusted_delta_(1.0 - pow(1 - delta,(double) max_partitions_contributed)),
      max_partitions_contributed_(max_partitions_contributed) {}

  virtual ~PartitionSelectionStrategy() = default;

  double GetEpsilon() { return epsilon_; }

  double GetDelta() { return delta_; }

  double GetAdjustedDelta() { return adjusted_delta_; }

  double GetMaxPartitionsContributed() { return max_partitions_contributed_; }

  virtual bool ShouldKeep(int num_users) = 0;

 protected:
  double epsilon_;
  double delta_;
  double adjusted_delta_;
  int max_partitions_contributed_;
};


class PreaggPartitionSelection : public PartitionSelectionStrategy {
 public:
  //Builder for PreaggPartitionSelection
  class Builder : public PartitionSelectionStrategy::Builder {
    public:
      base::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build() override {
        base::Status epsilon_status = EpsilonIsSetAndValid();
        base::Status delta_status = DeltaIsSetAndValid();
        base::Status max_partitions_contributed_status = MaxPartitionsContributedIsSetAndValid();

        if (!epsilon_status.ok()) {
          return epsilon_status;
        }
        else if (!delta_status.ok()) {
          return delta_status;
        }
        else if (!max_partitions_contributed_status.ok()) {
          return max_partitions_contributed_status;
        }
        else {
          std::unique_ptr<PartitionSelectionStrategy> magic = absl::make_unique<PreaggPartitionSelection>(epsilon_.value(), delta_.value(), max_partitions_contributed_.value());
          return magic;
        }
      }
    protected:
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

      // Checks if delta is set and valid.
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

  PreaggPartitionSelection(double epsilon, double delta, int max_partitions)
  	: PartitionSelectionStrategy(epsilon, delta, max_partitions) {
    srand(time(NULL));
    if(epsilon_ != 0) {
      adjusted_epsilon_ = epsilon / (double) max_partitions;
    }
    else {
      adjusted_epsilon_ = 0;
    }
    crossover_1_ = 1 + floor((1.0/adjusted_epsilon_)
                       *  log((exp(adjusted_epsilon_) + 2.0 * adjusted_delta_ - 1.0)
                          / ((exp(adjusted_epsilon_) + 1) * adjusted_delta_)));
    crossover_2_ = crossover_1_ + floor((1.0/adjusted_epsilon_)
                        *  log(1 + ((exp(adjusted_epsilon_) - 1) / adjusted_delta_)
                          * (1 - ProbabilityOfKeep(crossover_1_))));
  }

  virtual ~PreaggPartitionSelection() = default;

  double GetAdjustedEpsilon() { return adjusted_epsilon_; }

  double GetFirstCrossover() { return crossover_1_; }

  double GetSecondCrossover() { return crossover_2_; }

  bool ShouldKeep(int num_users) override {
    //generate a random number between 0 and 1
    double rand_num = ((double) rand() / RAND_MAX);
    //only keep partition if random number is less than expected probability of keep
    return (rand_num <= ProbabilityOfKeep(num_users));
  }

 protected:
  double adjusted_epsilon_;
  double crossover_1_;
  double crossover_2_;

  double ProbabilityOfKeep(double n) {
    if(n == 0) {
      return 0;
    }
    else if (n <= crossover_1_) {
      return (((exp(n * adjusted_epsilon_) - 1) / (exp(adjusted_epsilon_) - 1))
              * adjusted_delta_);
    }
    else if (n > crossover_1_ && n <= crossover_2_) {
      double m = n - crossover_1_;
      return ((1 - exp(-1 * m * adjusted_epsilon_))
               * (1 + adjusted_delta_ / (exp(adjusted_epsilon_) - 1))
               + exp(-1 * m * adjusted_epsilon_) * ProbabilityOfKeep(crossover_1_));
    }
    else {
      return 1;
    }
  }

};


class LaplacePartitionSelection : public PartitionSelectionStrategy {
 public:
  //Builder for LaplacePartitionSelection
  class Builder : public PartitionSelectionStrategy::Builder {
    public:
      Builder& SetLaplaceMechanism(LaplaceMechanism::Builder laplace_builder) {
        laplace_builder_ = laplace_builder;
        return *static_cast<Builder*>(this);
      }

      base::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build() override {
        base::Status max_partitions_contributed_status = MaxPartitionsContributedIsSetAndValid();
        if (!max_partitions_contributed_status.ok()) {
          return max_partitions_contributed_status;
        }
        else if (laplace_builder_.has_value()) {
          std::unique_ptr<LaplaceMechanism> mechanism_ = laplace_builder_.value().Build();
          std::unique_ptr<PartitionSelectionStrategy> laplace = absl::make_unique<LaplacePartitionSelection>(
            epsilon_.value(), delta_.value(), max_partitions_contributed_.value(), mechanism_);
          return laplace;
        }
        else {
          return base::InvalidArgumentError("you need a builder fix this message later");
        }
      }

    protected:
      absl::optional<LaplaceMechanism::Builder> laplace_builder_;
  };

  LaplacePartitionSelection(double epsilon, double delta, int max_partitions_contributed, LaplaceMechanism laplace)
    : PartitionSelectionStrategy(epsilon, delta, max_partitions_contributed),
      l1_sensitivity_(max_partitions_contributed),
      diversity_(l1_sensitivity_ / epsilon), mechanism_(laplace) {
        threshold_ = 1 - diversity_ * (log(2 * adjusted_delta_));
      }

  virtual ~LaplacePartitionSelection() = default;

  bool ShouldKeep(int num_users) override {
    double noised_result = mechanism_.AddNoise((double) num_users, 1.0);
    return (noised_result > threshold_);
  }

  double GetL1Sensitivity() { return l1_sensitivity_; }

  double GetDiversity() { return diversity_; }

  double GetThreshold() { return threshold_; }

  protected:
    double l1_sensitivity_;
    double diversity_;
    double threshold_;
    LaplaceMechanism mechanism_;

};

//We're leaving this be for now
/*
class GaussianPartitionSelection : public PartitionSelectionStrategy {
 public:
  GaussianPartitionSelection(double epsilon, double delta, int max_partitions)
    : PartitionSelectionStrategy(epsilon, delta, max_partitions) {
      //TODO
  }

  virtual ~GaussianPartitionSelection() = default;

  bool shouldKeep(int num_users) override {
    //TODO
  }

  double getL2Sensitivity() { return l2_sensitivity_; }

  double getStandardDev() { return standard_dev_; }

  double getThreshold() { return threshold_; }

 protected:
  double l2_sensitivity_;
  double standard_dev_;
  double threshold_;

};*/

} // namespace differential_privacy

#endif //DIFFERENTIAL_PRIVACY_ALGORITHMS_PARTITION_SELECTION_STRATEGIES_H_