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

#include "insufficient_noise_algorithms.h"

namespace differential_privacy {

namespace testing {

// Runs the Stochastic Tester on three algorithm types, each of which has been
// deliberately constructed to violate differential privacy. Measures the 
// Stochastic Tester's ability to detect the differential privacy violations
// over algorithms with increasingly more subtle violations.

const double epsilon_value = std::log(3);

bool RunStochasticTesterOnCount(double ratio, int num_datasets,
  int num_samples_per_histogram) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
    DefaultDatasetSize(), true, DefaultDataScale(), DefaultDataOffset());
  auto algorithm = absl::make_unique<CountWithInsufficientNoise<double>>(
    epsilon_value, absl::make_unique<LaplaceMechanism::Builder>(),
    ratio);
  StochasticTester<double, int64_t> tester(std::move(algorithm),
    std::move(sequence), num_datasets, num_samples_per_histogram);
// Desired outcome is false, since algorithm has been engineered to violate DP.
  bool algo_is_dp = tester.Run();
  return algo_is_dp; 
}

bool RunStochasticTesterOnSum(double ratio, int num_datasets,
  int num_samples_per_histogram) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
    DefaultDatasetSize(), true, DefaultDataScale(), DefaultDataOffset());
  auto algorithm = absl::make_unique<SumWithInsufficientNoise<double>>(
    epsilon_value, sequence->RangeMin(), sequence->RangeMax(),1,1,
    absl::make_unique<LaplaceMechanism::Builder>(), ratio);
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
    num_datasets, num_samples_per_histogram);
// Desired outcome is false, since algorithm has been engineered to violate DP.
  bool algo_is_dp = tester.Run();
  return algo_is_dp;
}

bool RunStochasticTesterOnMean(double ratio, int num_datasets,
  int num_samples_per_histogram) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
    DefaultDatasetSize(), true, DefaultDataScale(), DefaultDataOffset());
  auto algorithm = absl::make_unique<MeanWithInsufficientNoise<double>>(
    epsilon_value,sequence->RangeMin(), sequence->RangeMax(),1,1,
    absl::make_unique<LaplaceMechanism::Builder>(), ratio);
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
    num_datasets, num_samples_per_histogram);
// Desired outcome is false, since algorithm has been engineered to violate DP.
  bool algo_is_dp = tester.Run();
  return algo_is_dp;   
}

// Applies the Stochastic Tester to the constructed Count algorithm over a 
// continuous range of ratio values and sends the test results to an output file.
base::StatusOr<SummaryResults> GetTestResultsForCount(int num_datasets,
  int num_samples_per_histogram, double ratio_min, double ratio_max,
  double increment, std::ofstream& datafile) {

  SummaryResults sr;
  double num_tests = 0;
  double num_tests_passed = 0;
  double maximum_ratio_passed = 0;
  auto start = std::chrono::high_resolution_clock::now();

  int num_iterations = ceil((ratio_max - ratio_min) / increment);
  for (double i=0; i <= num_iterations; ++i) {
    auto start_test_run = std::chrono::high_resolution_clock::now();
    double ratio = i * increment + ratio_min;
    std::cout << "Now calculating count algorithm with ratio: " << 
      ratio << std::endl;
    bool outcome = RunStochasticTesterOnCount(ratio, num_datasets, 
      num_samples_per_histogram); 
    num_tests++;
    if (outcome == 0) { 
      num_tests_passed++;
      maximum_ratio_passed = ratio;
    }

    auto finish_test_run = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> test_run_length =
      std::chrono::duration_cast<std::chrono::duration<double>>(
      finish_test_run - start_test_run);
    datafile << "insufficient_noise," << "count" << ",0," << outcome << "," <<
      ratio << "," << num_datasets << "," << num_samples_per_histogram << "," <<
      test_run_length.count() << "\n";
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_elapsed =
    std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  sr.num_tests = num_tests;
  sr.num_tests_passed = num_tests_passed;
  sr.maximum_ratio = maximum_ratio_passed;
  sr.total_time = time_elapsed;
  return sr;
}

// Applies the Stochastic Tester to the constructed Bounded Sum algorithm over a
// continuous range of ratio values and sends the test results to an output file.
base::StatusOr<SummaryResults> GetTestResultsForSum(int num_datasets,
  int num_samples_per_histogram, double ratio_min, double ratio_max,
  double increment, std::ofstream& datafile) {

  SummaryResults sr;
  double num_tests = 0;
  double num_tests_passed = 0;
  double maximum_ratio_passed = 0;
  auto start = std::chrono::high_resolution_clock::now();

  int num_iterations = ceil((ratio_max - ratio_min) / increment);
  for (double i=0; i <= num_iterations; ++i) {
    auto start_test_run = std::chrono::high_resolution_clock::now();
    double ratio = i * increment + ratio_min;
    std::cout << "Now calculating bounded_sum algorithm with ratio: " << 
      ratio << std::endl;
    bool outcome = RunStochasticTesterOnSum(ratio, num_datasets, 
      num_samples_per_histogram);  
    num_tests++;
    if (outcome == 0) { 
      num_tests_passed++;
      maximum_ratio_passed = ratio;
    }

    auto finish_test_run = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> test_run_length =
      std::chrono::duration_cast<std::chrono::duration<double>>(
      finish_test_run - start_test_run);
    datafile << "insufficient_noise," << "bounded_sum" << ",0," << outcome << "," <<
      ratio << "," << num_datasets << "," << num_samples_per_histogram << "," <<
      test_run_length.count() << "\n";
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_elapsed =
    std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  sr.num_tests = num_tests;
  sr.num_tests_passed = num_tests_passed;
  sr.maximum_ratio = maximum_ratio_passed;
  sr.total_time = time_elapsed;
  return sr;
}

base::StatusOr<SummaryResults> GetTestResultsForMean(int num_datasets,
  int num_samples_per_histogram, double ratio_min, double ratio_max,
  double increment, std::ofstream& datafile) {

  SummaryResults sr;
  double num_tests = 0;
  double num_tests_passed = 0;
  double maximum_ratio_passed = 0;
  auto start = std::chrono::high_resolution_clock::now();

  int num_iterations = ceil((ratio_max - ratio_min) / increment);
  for (double i=0; i <= num_iterations; ++i) {
    auto start_test_run = std::chrono::high_resolution_clock::now();
    double ratio = i * increment + ratio_min;
    std::cout << "Now calculating bounded_mean algorithm with ratio: " << 
      ratio << std::endl;
    bool outcome = RunStochasticTesterOnMean(ratio, num_datasets, 
      num_samples_per_histogram);  
    num_tests++;
    if (outcome == 0) { 
      num_tests_passed++;
      maximum_ratio_passed = ratio;
    }

    auto finish_test_run = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> test_run_length =
      std::chrono::duration_cast<std::chrono::duration<double>>(
      finish_test_run - start_test_run);
    datafile << "insufficient_noise," << "bounded_mean" << ",0," << outcome << "," <<
      ratio << "," << num_datasets << "," << num_samples_per_histogram << "," <<
      test_run_length.count() << "\n";
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_elapsed =
    std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  sr.num_tests = num_tests;
  sr.num_tests_passed = num_tests_passed;
  sr.maximum_ratio = maximum_ratio_passed;
  sr.total_time = time_elapsed;
  return sr;
} 

} // namespace testing
} // namespace differential_privacy