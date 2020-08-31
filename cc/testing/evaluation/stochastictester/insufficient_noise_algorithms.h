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
  
// Runs the Stochastic Tester on three algorithm types, each of which has been
// deliberately constructed to violate differential privacy. Measures the 
// Stochastic Tester's ability to detect the differential privacy violations
// over a continuous range of ratio values.

template <typename T>
class CountWithInsufficientNoise : public differential_privacy::Count<T> {
 public:
  CountWithInsufficientNoise(double epsilon,
    std::unique_ptr<LaplaceMechanism::Builder> count_builder, double ratio)
    : Count<T>(epsilon,
    count_builder->SetEpsilon(epsilon)
    .Build()
    .ValueOrDie()),
    ratio_(ratio) {}
// Overrides epsilon such that amount of noise applied is only a fraction of
// what privacy protection claimed 
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
    const double l0_sensitivity, const double max_contributions_per_partition,
    std::unique_ptr<LaplaceMechanism::Builder> sum_builder, double ratio)
    : BoundedSum<T>(epsilon, lower, upper, l0_sensitivity, 
    max_contributions_per_partition, std::move(sum_builder),
    LaplaceMechanism::Builder()
    .SetEpsilon(epsilon)
    .SetL0Sensitivity(l0_sensitivity)
    .SetLInfSensitivity(max_contributions_per_partition * std::max(std::abs(lower),std::abs(upper)))
    .Build()
    .ValueOrDie(),nullptr),
    ratio_(ratio) {}
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
    const double l0_sensitivity, const double max_contributions_per_partition, 
    std::unique_ptr<LaplaceMechanism::Builder> mean_builder, double ratio)
    : BoundedMean<T>(epsilon, lower, upper, l0_sensitivity,
    max_contributions_per_partition, std::move(mean_builder),
    LaplaceMechanism::Builder()
    .SetEpsilon(epsilon)
    .SetL0Sensitivity(l0_sensitivity)
    .SetLInfSensitivity(max_contributions_per_partition * (std::abs(upper - lower) / 2))
    .Build()
    .ValueOrDie(),
    LaplaceMechanism::Builder()
    .SetEpsilon(epsilon)
    .Build()
    .ValueOrDie(),nullptr),
    ratio_(ratio) {}
// Overrides epsilon such that amount of noise applied is only a fraction of
// what privacy protection claimed
  double GetEpsilon() const override { return Algorithm<T>::GetEpsilon()
    * ratio_; }
 private:
  double ratio_;
};

bool RunStochasticTesterOnCount(double ratio, int num_datasets,
  int num_samples_per_histogram);

bool RunStochasticTesterOnSum(double ratio, int num_datasets,
  int num_samples_per_histogram);

bool RunStochasticTesterOnMean(double ratio, int num_datasets,
  int num_samples_per_histogram);

struct SummaryResults {
      int num_tests;
      int num_tests_passed;
      double maximum_ratio;
      std::chrono::duration<double> total_time;
};

base::StatusOr<SummaryResults> GetTestResultsForCount(int num_datasets,
  int num_samples_per_histogram, int ratio_min, int ratio_max,
  std::ofstream& datafile);

base::StatusOr<SummaryResults> GetTestResultsForSum(int num_datasets,
  int num_samples_per_histogram, int ratio_min, int ratio_max,
  std::ofstream& datafile);

base::StatusOr<SummaryResults> GetTestResultsForMean(int num_datasets,
  int num_samples_per_histogram, int ratio_min, int ratio_max,
  std::ofstream& datafile);

} // namespace testing  
} // namespace differential_privacy

#endif  // ALGORITHMS_WITH_INSUFFICIENT_NOISE_H_