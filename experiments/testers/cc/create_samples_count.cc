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

namespace differential_privacy {

namespace testing {

// Creates pairs of samples of differentially private counts. 
// Each sample-pair replicates a unique scenario constructed in the proto for
// CountDpTest.java, available here:
// https://github.com/google/differential-privacy/blob/main/proto/testing/count_dp_test_cases.textproto.

const std::string count_samples_folder = "../statisticaltester/countsamples";

// Construct Count algorithm.
int64_t DPCount(const std::vector<int>& values, double epsilon,
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
// (e.g., R95). Every folder contains 10 subfolders for each unique sample-pair.
// Every subfolder contains seven runs of each sample-pair (14 files in total).
void CreateSingleScenarioCount(int scenario, double true_value, int number_of_samples,
  int increment, int max_partitions, double epsilon, double ratio) { 
  double neighbor_value = true_value + increment;
  double implemented_epsilon = epsilon / ratio;
// Create pairs of vectors with arbitrary value of 100.
  std::vector<int> sampleA(true_value,100);
  std::vector<int> sampleB(neighbor_value,100);
  std::string filepath = count_samples_folder+"/R"
    +std::to_string(static_cast<int>(ratio*100))+"/Scenario"+std::to_string(scenario);
  mkdir(filepath.c_str(), 0777);
  for (int i=0; i<7; i++) {
    std::ofstream samplefileA;
    std::ofstream samplefileB;
    samplefileA.open(filepath+"/TestCase"+std::to_string(i)+"A.txt");
    samplefileB.open(filepath+"/TestCase"+std::to_string(i)+"B.txt");
    for (int i=0; i<number_of_samples; i++) {
      int64_t outputA = DPCount(sampleA, implemented_epsilon, max_partitions);
      samplefileA << outputA << "\n";
      int64_t outputB = DPCount(sampleB, implemented_epsilon,max_partitions);
        samplefileB << outputB << "\n";
    }
  samplefileA.close();
  samplefileB.close();
  }
}

// Runs each sample-pair with parameters that replicate those specified in:
// https://github.com/google/differential-privacy/blob/main/proto/testing/count_dp_test_cases.textproto
void GenerateAllScenariosCount(double ratio) {
  const int num_of_samples = 100;
  double small_epsilon = 0.01;
  double default_epsilon = std::log(3);
  double large_epsilon = 2*std::log(3);

// Laplace noise, empty count, default parameters
  CreateSingleScenarioCount(1,0,num_of_samples,1,1,default_epsilon,ratio);

// Laplace noise, empty count, two partitions contributed
  CreateSingleScenarioCount(2,0,num_of_samples,2,2,default_epsilon,ratio);

// Laplace noise, empty count, many partitions contributed
  CreateSingleScenarioCount(3,0,num_of_samples,250,250,default_epsilon,ratio);

// Laplace noise, empty count, small epsilon
  CreateSingleScenarioCount(4,0,num_of_samples,1,1,small_epsilon,ratio);

// Laplace noise, empty count, large epsilon
  CreateSingleScenarioCount(5,0,num_of_samples,1,1,large_epsilon,ratio);

// Laplace noise, small count, default parameters
  CreateSingleScenarioCount(6,28,num_of_samples,1,1,default_epsilon,ratio);

// Laplace noise, small count, two partitions contributed
  CreateSingleScenarioCount(7,28,num_of_samples,2,2,default_epsilon,ratio);

// Laplace noise, small count, many partitions contributed
  CreateSingleScenarioCount(8,28,num_of_samples,250,250,default_epsilon,ratio);

// Laplace noise, small count, small epsilon
  CreateSingleScenarioCount(9,28,num_of_samples,1,1,small_epsilon,ratio);

// Laplace noise, small count, large epsilon
  CreateSingleScenarioCount(10,28,num_of_samples,1,1,large_epsilon,ratio);
}
} // testing
} // differential_privacy
