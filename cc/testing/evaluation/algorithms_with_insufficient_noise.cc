 
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

// Runs the Stochastic Tester on a continuous range of three algorithms (Count, 
// BoundedSum, and BoundedMean) in which the privacy protection claimed exceeds 
// the amount of noise applied. Sends results of each run to an output file.

bool RunStochasticTester(std::string algorithm, double ratio, double num_datasets,
  double num_samples_per_histogram) {

  auto sequence = absl::make_unique<HaltonSequence<double>>(
    DefaultDatasetSize(), true, DefaultDataScale(), DefaultDataOffset());

  if (algorithm == "count") {

    auto sequence = absl::make_unique<HaltonSequence<int64_t>>(
      DefaultDatasetSize(), true, DefaultDataScale(), DefaultDataOffset());

    auto algorithm = absl::make_unique<CountWithInsufficientNoise<int64_t>>(
      std::log(3), ratio);
    
    StochasticTester<int64_t> tester( std::move(algorithm), std::move(sequence),
      num_datasets, num_samples_per_histogram);

    bool algo_is_dp = tester.Run(); // false = 0, true = 1
    std::cout << algo_is_dp << std::endl; 
    return algo_is_dp; // correct outcome is false, e.g., DP premise rejected
  }

  else if (algorithm == "boundedsum") {

    auto algorithm = absl::make_unique<SumWithInsufficientNoise<double>>(
      std::log(3), sequence->RangeMin(), sequence->RangeMax(),
      absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>(), ratio);

    StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
      num_datasets, num_samples_per_histogram);

    bool algo_is_dp = tester.Run(); // false = 0, true = 1
    std::cout << algo_is_dp << std::endl; 
    return algo_is_dp; // correct outcome is false, e.g., DP premise rejected

  }

  else if (algorithm == "boundedmean") {

    auto algorithm = absl::make_unique<MeanWithInsufficientNoise<double>>(
      std::log(3),sequence->RangeMin(), sequence->RangeMax(),
      absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>(), ratio);

    StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
      num_datasets, num_samples_per_histogram);

    bool algo_is_dp = tester.Run(); // false = 0, true = 1
    std::cout << algo_is_dp << std::endl; 
    return algo_is_dp; // correct outcome is false, e.g., DP premise rejected
  }

  else {
    std::cout << "The algorithm specified was not recognized." << 
      "Try count, boundedsum, or boundedmean instead." << std::endl;
  }
}


void GetTestResults(std::ofstream& datafile, std::string algorithm,
  double num_datasets, double num_samples_per_histogram, double ratio_min,
  double ratio_max) {

  double num_tests = 0;
  double num_tests_passed = 0;
  double maximum_ratio_passed = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (double i=ratio_min; i<=ratio_max; i++) {
    auto start_test_run = std::chrono::high_resolution_clock::now();
    double ratio = i/100.0;
    std::cout << "Now calculating " << algorithm << " algorithm with ratio: " << 
      ratio << std::endl;
    bool outcome = RunStochasticTester(algorithm, ratio, num_datasets,
      num_samples_per_histogram);
    num_tests++;

    if (outcome == 0) { // correct outcome is false, e.g., DP predicate rejected 
      num_tests_passed++;
      maximum_ratio_passed = ratio;
    }

    auto finish_test_run = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> test_run_length =
      std::chrono::duration_cast<std::chrono::duration<double>>(
      finish_test_run - start_test_run);
    datafile << "insufficient_noise," << algorithm << ",0," << outcome << "," <<
      ratio << "," << num_datasets << "," << num_samples_per_histogram << "," <<
      test_run_length.count() << "\n";
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_elapsed =
    std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  std::cout << num_tests_passed << " out of " << num_tests << " passed. " <<
    "The largest ratio value that passed was " << maximum_ratio_passed << "." <<
    std::endl;
    }
  } // namespace testing
} // namespace differential_privacy

int main(int argc, char *argv[]) {

  double num_datasets = 15;
  double num_samples_per_histogram = 10;
  double ratio_min = 80.0;
  double ratio_max = 81.0;
  std::ofstream datafile;
  std::vector<std::string> algorithms={"count","boundedsum","boundedmean"};

    if (argc >= 2) {
      datafile.open(argv[1]);
    }

    else {
      datafile.open("stochastic_tester_results.txt");
    }

  datafile << "test_name,algorithm,expected,actual,ratio,num_datasets," <<
    "num_samples,time(sec)" << "\n";

  for (std::string& algorithm : algorithms) {
    // run the test on each algorithm 10 times
    for (int i=1; i<=3; i++) {
      differential_privacy::testing::GetTestResults(datafile,algorithm,
        num_datasets,num_samples_per_histogram,ratio_min,ratio_max);
      std::cout << "Number of iterations completed: " << i << "/10" << std::endl;
    }
  }

datafile.close();
return 0;
}
