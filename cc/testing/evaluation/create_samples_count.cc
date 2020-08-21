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

#include "create_samples_count.h"
#include "algorithms/count.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <vector>
#include <string>

#include "absl/random/distributions.h"
#include "absl/memory/memory.h"

#include "algorithms/algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/util.h"
#include "base/statusor.h"
#include "proto/data.pb.h"
#include "testing/sequence.h"

namespace differential_privacy {

namespace testing {

/** Creates samples of data pairs using the Count algorithm. Each data pair
and its associated input parameters replicates a scenario constructed in 
the 
* @see [textproto](https://github.com/google/differential-privacy/blob/main/proto/testing/count_dp_test_cases.textproto)
for CountDpTest.java.
*/

const std::string folder_name = "CountSamples";

// Construct Count algorithm.
int64_t CountAlgorithm(const std::vector<int>& values, double epsilon,
  int max_partitions) {
  std::unique_ptr<Count<int64_t>> count = Count<int64_t>::Builder()
    .SetEpsilon(epsilon)
    .SetMaxPartitionsContributed(max_partitions) 
    .Build()
    .ValueOrDie();
  base::StatusOr<Output> result = count->Result(values.begin(), values.end());
  Output obj = result.ValueOrDie();
  return GetValue<int64_t>(obj);
}

// Creates a folder to contain all samples with a particular ratio value
// (e.g., R95). Every folder contains 10 subfolders for each distinct data pair.
// Every subfolder contains seven iterations of each data pair.

void CreateSingleScenario(int scenario, double true_value, int number_of_samples,
  int increment, int max_partitions, double epsilon, double ratio) { 
  double neighbor_value = true_value + increment;
  double implemented_epsilon = epsilon / ratio;

// Create pairs of vectors with arbitrary value of 100.
  std::vector<int> sampleA(true_value,100);
  std::vector<int> sampleB(neighbor_value,100);
  std::string filepath = folder_name+"/R"
    +std::to_string(static_cast<int>(ratio*100))+"/Scenario"+std::to_string(scenario);
  mkdir(filepath.c_str(), 0777);
// For each sample, run CountAlgorithm 1M times. Run each sample seven times. 
// Generates 14 files (seven pairs of files) with each run.
  for (int i=0; i<7; i++) {
    std::ofstream samplefileA;
    std::ofstream samplefileB;
    samplefileA.open(filepath+"/TestCase"+std::to_string(i)+"A.txt");
    samplefileB.open(filepath+"/TestCase"+std::to_string(i)+"B.txt");
    for (int i=0; i<number_of_samples; i++) {
      int64_t outputA = CountAlgorithm(sampleA, implemented_epsilon, max_partitions);
      samplefileA << outputA << "\n";
      int64_t outputB = CountAlgorithm(sampleB, implemented_epsilon,max_partitions);
        samplefileB << outputB << "\n";
    }
  samplefileA.close();
  samplefileB.close();
  }
}

// Run each data pair to mirror parameters specified in the [textproto](https://github.com/google/differential-privacy/blob/main/proto/testing/count_dp_test_cases.textproto)
// for CountDpTest.java.

void GenerateAllScenarios(double ratio) {
  const int num_of_samples = 1000000;
  double small_epsilon = 0.01;
  double default_epsilon = std::log(3);
  double large_epsilon = 2*std::log(3);

// Laplace noise, empty count, default parameters
  differential_privacy::testing::CreateSingleScenario(1,0,num_of_samples,1,1,
    default_epsilon,ratio);

// Laplace noise, empty count, two partitions contributed
  differential_privacy::testing::CreateSingleScenario(2,0,num_of_samples,2,2,
    default_epsilon,ratio);

// Laplace noise, empty count, many partitions contributed
  differential_privacy::testing::CreateSingleScenario(3,0,num_of_samples,250,250,
    default_epsilon,ratio);

// Laplace noise, empty count, small epsilon
  differential_privacy::testing::CreateSingleScenario(4,0,num_of_samples,1,1,
    small_epsilon,ratio);

// Laplace noise, empty count, large epsilon
  differential_privacy::testing::CreateSingleScenario(5,0,num_of_samples,1,1,
    large_epsilon,ratio);

// Laplace noise, small count, default parameters
  differential_privacy::testing::CreateSingleScenario(6,28,num_of_samples,1,1,
    default_epsilon,ratio);

// Laplace noise, small count, two partitions contributed
  differential_privacy::testing::CreateSingleScenario(7,28,num_of_samples,2,2,
    default_epsilon,ratio);

// Laplace noise, small count, many partitions contributed
  differential_privacy::testing::CreateSingleScenario(8,28,num_of_samples,250,250,
    default_epsilon,ratio);

// Laplace noise, small count, small epsilon
  differential_privacy::testing::CreateSingleScenario(9,28,num_of_samples,1,1,
    small_epsilon,ratio);

// Laplace noise, small count, large epsilon
  differential_privacy::testing::CreateSingleScenario(10,28,num_of_samples,1,1,
    large_epsilon,ratio);

  }
} // testing
} // differential_privacy

int main(int argc, char** argv) {
// Create folder to hold the samples.
  mkdir(differential_privacy::testing::folder_name.c_str(), 0777);
  for (int i = 80; i <= 99; i++) {
    std::cout << i << std::endl;
    std::string filepath = differential_privacy::testing::folder_name
      +"/R"+std::to_string(i);
    mkdir(filepath.c_str(), 0777);
    const double r = i / 100.0;
    differential_privacy::testing::GenerateAllScenarios(r);
  }
  return 0;
}