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

namespace differential_privacy{

// Provides a common abstraction for PartitionSelectionStrategy.  Partition
// selection strategies decide whether or not a specific partition should be
// dropped or kept.
class PartitionSelectionStrategy {
 public:
  PartitionSelectionStrategy(double epsilon, double delta, int num_users, int max_partitions)
  	: epsilon_(epsilon), delta_(delta), adjusted_delta_(1.0 - pow(1 - delta, (double) max_partitions)),
    num_users_(num_users), max_partitions_(max_partitions) {}

  virtual ~PartitionSelectionStrategy() = default;

  double GetEpsilon() { return epsilon_; }

  double GetDelta() { return delta_; }

  double GetAdjustedDelta() { return adjusted_delta_; }

  int GetNumUsers() { return num_users_; }


  virtual bool shouldKeep() = 0;

 protected:
  double epsilon_;
  double delta_;
  double adjusted_delta_;
  int num_users_;
  int max_partitions_;
};



class MagicPartitionSelection : public PartitionSelectionStrategy {
 public:
  MagicPartitionSelection(double epsilon, double delta, int num_users, int max_partitions)
  	: PartitionSelectionStrategy(epsilon, delta, num_users, max_partitions) {
    if(epsilon_ != 0) {
      adjusted_epsilon_ = epsilon / (double)max_partitions;
    }
    else {
      adjusted_epsilon_ = 0;
    }
    crossover_1_ = 1 + floor((1.0/adjusted_epsilon_)
                       *  log((exp(adjusted_epsilon_) + 2.0 * delta_ - 1.0)
                          / ((exp(adjusted_epsilon_) + 1) * delta_)));
    crossover_2_ = crossover_1_ + floor((1.0/adjusted_epsilon_)
                        *  log(1 + ((exp(adjusted_epsilon_) - 1) / delta_)
                          * (1 - probabilityOfKeep(crossover_1_))));
  }

  virtual ~MagicPartitionSelection() = default;

  bool shouldKeep() override {
    //generate a random number between 0 and 1
    srand(time(NULL));
    double rand_num = ((double) rand() / RAND_MAX);
    //only keep partition if random number is less than expected probability of keep
    return (rand_num <= probabilityOfKeep(num_users_));
  }

 protected:
  double adjusted_epsilon_;
  double crossover_1_;
  double crossover_2_;

  double probabilityOfKeep(double n) {
    if(n == 0 || delta_ == 0) {
      return 0;
    }
    else if (adjusted_epsilon_ == 0) {
      return fmin(1.0, n * delta_);
    }
    else if (n <= crossover_1_) {
      return (((exp(n * adjusted_epsilon_) - 1) / (exp(adjusted_epsilon_) - 1))
              * delta_);
    }
    else if (n > crossover_1_ && n <= crossover_2_) {
      double m = n - crossover_1_;
      return ((1 - exp(-1 * m * adjusted_epsilon_))
               * (1 + delta_ / (exp(adjusted_epsilon_) - 1))
               + exp(-1 * m * adjusted_epsilon_) * probabilityOfKeep(crossover_1_));
    }
    else {
      return 1;
    }
  }

};

/*
//TODO
class LaplacePartitionSelection : public PartitionSelectionStrategy {
 public:
  LaplacePartitionSelection(double epsilon, double delta, int num_users, int max_partitions)
    : PartitionSelectionStrategy(epsilon, delta, num_users, max_partitions) {
    //diversity here if that's the right name
  }

  virtual ~LaplacePartitionSelection() = default;

  bool shouldKeep() override {
    return base::UnimplementedError(
        "shouldKeep() unsupported for this numerical mechanism.");
  }

 protected:
  //more stuff

};

//TODO
class GaussianPartitionSelection : public PartitionSelectionStrategy {
 public:
  GaussianPartitionSelection(double epsilon, double delta, int num_users, int max_partitions)
    : PartitionSelectionStrategy(epsilon, delta, num_users, max_partitions) {
    //diversity here if that's the right name
  }

  virtual ~GaussianPartitionSelection() = default;

  bool shouldKeep() override {
    return base::UnimplementedError(
        "sshouldKeep() unsupported for this numerical mechanism.");
  }

 protected:
  //more stuff

};*/

} // namespace differential_privacy

#endif //DIFFERENTIAL_PRIVACY_ALGORITHMS_PARTITION_SELECTION_STRATEGIES_H_