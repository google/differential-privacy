
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

#ifndef ALGORITHMS_WITH_INSUFFICIENT_NOISE_H_
#define ALGORITHMS_WITH_INSUFFICIENT_NOISE_H_

#include "algorithms/count.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/bounded-mean.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iterator>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/random/distributions.h"

#include "algorithms/algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/util.h"
#include "base/statusor.h"
#include "proto/data.pb.h"
#include "testing/sequence.h"
#include "testing/stochastic_tester.h"

namespace differential_privacy {

namespace testing {

// Runs the Stochastic Tester on a continuous range of three algorithms (Count, 
// BoundedSum, and BoundedMean) in which the privacy protection claimed exceeds 
// the amount of noise applied. Sends results of each run to an output file.

template <typename T>
class CountWithInsufficientNoise : public differential_privacy::Count<T> {
 public:
  CountWithInsufficientNoise(double epsilon, double ratio)
    : Count<T>(epsilon,
    LaplaceMechanism::Builder()
      .SetEpsilon(epsilon)
      .SetL0Sensitivity(1)
      .SetLInfSensitivity(1)
      .Build()
      .ValueOrDie()),
      ratio_(ratio) {} 
  double GetEpsilon() const override { return Algorithm<T>::GetEpsilon()
    * ratio_; }
  private:
    double ratio_;
};

template <typename T, typename std::enable_if<std::is_integral<T>::value || 
	std::is_floating_point<T>::value>::type* = nullptr>
class SumWithInsufficientNoise : public differential_privacy::BoundedSum<T> {
 public:
  SumWithInsufficientNoise(double epsilon, T lower, T upper,
    std::unique_ptr<LaplaceMechanism::Builder> builder, double ratio)
    : BoundedSum<T>(epsilon, lower, upper, 1, 1, std::move(builder),
    nullptr, nullptr), ratio_(ratio) {} // set sensitivity values to 1
// Overrides epsilon such that amount of noise applied is only a fraction of
// what privacy protection claimed
  double GetEpsilon() const override { return Algorithm<T>::GetEpsilon() 
   * ratio_; } 
 private: 
  double ratio_;
};

template <typename T, typename std::enable_if<std::is_integral<T>::value || 
std::is_floating_point<T>::value>::type* = nullptr>
class MeanWithInsufficientNoise : public differential_privacy::BoundedMean<T> {
 public:
  MeanWithInsufficientNoise(double epsilon, T lower, T upper,
    std::unique_ptr<LaplaceMechanism::Builder> builder, double ratio)
    : BoundedMean<T>(epsilon, lower, upper, 1, 1, std::move(builder),
    LaplaceMechanism::Builder()
      .SetEpsilon(epsilon)
      .SetL0Sensitivity(1)
      .SetLInfSensitivity(1)
      .Build()
      .ValueOrDie(),
    LaplaceMechanism::Builder()
      .SetEpsilon(epsilon)
      .Build()
      .ValueOrDie(), nullptr), ratio_(ratio) {}
    double GetEpsilon() const override { return Algorithm<T>::GetEpsilon()
     * ratio_; }
 private:
  double ratio_;
};

bool RunStochasticTester(double ratio, double num_datasets,
  double num_samples_per_histogram);

void GetTestResults(std::ofstream& datafile, double num_datasets, 
  double num_samples_per_histogram, double ratio_min, double ratio_max);

} // namespace testing  
} // namespace differential_privacy

#endif  // ALGORITHMS_WITH_INSUFFICIENT_NOISE_H_