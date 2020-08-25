 
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

#include "algorithms_with_insufficient_noise.h"

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
#include "algorithms/bounded-sum.h"
#include "algorithms/bounded-mean.h"
#include "algorithms/count.h"
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

const double epsilon_value = std::log(3);

bool RunStochasticTesterOnCount(double ratio, int num_datasets,
  int num_samples_per_histogram) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
    DefaultDatasetSize(), true, DefaultDataScale(), DefaultDataOffset());
  auto algorithm = absl::make_unique<CountWithInsufficientNoise<double>>(
    epsilon_value, ratio);
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
    absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>(), ratio);
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
    absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>(), ratio);
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
    num_datasets, num_samples_per_histogram);
// Desired outcome is false, since algorithm has been engineered to violate DP.
  bool algo_is_dp = tester.Run();
  return algo_is_dp;   
}

 struct SummaryResults {
      int num_tests;
      int num_tests_passed;
      double maximum_ratio;
      std::chrono::duration<double> total_time;
};

// Runs the Stochastic Tester on the specified algorithm for a continuous range
// of ratios and sends the test results to an output file.
base::StatusOr<SummaryResults> GetTestResultsForCount(int num_datasets,
  int num_samples_per_histogram, int ratio_min, int ratio_max, std::ofstream& datafile) {

  SummaryResults sr;
  double num_tests = 0;
  double num_tests_passed = 0;
  double maximum_ratio_passed = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i=ratio_min; i<=ratio_max; i++) {
    auto start_test_run = std::chrono::high_resolution_clock::now();
    double ratio = static_cast<double>(i)/100.0;
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

base::StatusOr<SummaryResults> GetTestResultsForSum(int num_datasets,
  int num_samples_per_histogram, int ratio_min, int ratio_max, std::ofstream& datafile) {

  SummaryResults sr;
  double num_tests = 0;
  double num_tests_passed = 0;
  double maximum_ratio_passed = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i=ratio_min; i<=ratio_max; i++) {
    auto start_test_run = std::chrono::high_resolution_clock::now();
    double ratio = static_cast<double>(i)/100.0;
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
  int num_samples_per_histogram, int ratio_min, int ratio_max, std::ofstream& datafile) {

  SummaryResults sr;
  double num_tests = 0;
  double num_tests_passed = 0;
  double maximum_ratio_passed = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i=ratio_min; i<=ratio_max; i++) {
    auto start_test_run = std::chrono::high_resolution_clock::now();
    double ratio = static_cast<double>(i)/100.0;
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

// TODO: Make sure ratio_min and ratio_max cannot exceed 100

int main(int argc, char *argv[]) {

  std::ofstream countfile;
  std::ofstream sumfile;
  std::ofstream meanfile;

  differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> count_summary;
  differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> sum_summary;
  differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> mean_summary;

  int const count_num_datasets = 10;
  int const sum_num_datasets = 17;
  int const mean_num_datasets = 22;

  double const num_samples_per_histogram = 100;
  double const ratio_min = 90.0;
  double const ratio_max = 91.0;
  std::string header = "test_name,algorithm,expected,actual,ratio,num_datasets,num_samples,time(sec)";

  if (argc >= 2) {
//    datafile.open(argv[1]);
    std::cout << "This is a test!" << std::endl;
  }

  else {
    countfile.open("testing/evaluation/stochastic_tester_results_counttest.txt");
    countfile << header << "\n";
    count_summary = differential_privacy::testing::GetTestResultsForCount(
      count_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,countfile);
    countfile.close();

    sumfile.open("testing/evaluation/stochastic_tester_results_sumtest.txt");
    sumfile << header << "\n";
    sum_summary = differential_privacy::testing::GetTestResultsForSum(
      sum_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,sumfile);
    sumfile.close();

    meanfile.open("testing/evaluation/stochastic_tester_results_meantest.txt");
    meanfile << header << "\n";
    mean_summary = differential_privacy::testing::GetTestResultsForMean(
      mean_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,meanfile);
    meanfile.close();
  }
  return 0;
}
