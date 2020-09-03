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

#include "create_samples_sum.h"

// Creates pairs of samples of differentially private sums. 
// Each sample-pair replicates a unique scenario constructed in the proto for
// BoundedSumDpTest.java, available here:
// https://github.com/google/differential-privacy/blob/main/proto/testing/bounded_sum_dp_test_cases.textproto.

namespace differential_privacy {

namespace testing {

const std::string sum_samples_folder = "../statisticaltester/boundedsumsamples";

double DiscretizeSum(double true_value, double granularity) {
  if (granularity > 0) {
    double abs_value = abs(true_value);
    double scaled_sample = true_value/granularity;
    if (abs_value >= 1L << 54) {
      double discretized_value = scaled_sample * granularity;
      return discretized_value;
  	} else {
   	double discretized_value = std::round(scaled_sample) * granularity;
    return discretized_value;
    }
  }
  else {
    std::cout << "Granularity must be positive. Try again, please." << std::endl;
  }
}

// Construct the BoundedSum algorithm.
double BoundedSumAlgorithm(std::vector<double> values, double granularity,
  double epsilon, int max_partitions, int lower, int upper) {
    std::unique_ptr<BoundedSum<double>> boundedsum = BoundedSum<double>::Builder()
    .SetEpsilon(epsilon)
    .SetMaxPartitionsContributed(max_partitions)
    .SetLower(lower)
    .SetUpper(upper) 
    .Build()
    .ValueOrDie();

  base::StatusOr<Output> result = boundedsum->Result(values.begin(),
    values.end());
  Output obj = result.ValueOrDie();
  return GetValue<double>(obj);
}

// Creates a folder to contain all samples with a particular ratio value
// (e.g., R95). Every folder contains 17 subfolders for each unique sample-pair.
// Every subfolder contains seven runs of each sample-pair (14 files in total).
void CreateSingleScenarioSum(int scenario, std::vector<double> values,
  double granularity, double epsilon, int max_partitions,
	int lower, int upper, int number_of_samples, int increment, double ratio) { 
  std::vector<double> neighbor_values = values;
  neighbor_values.push_back(increment);
  double implemented_epsilon = epsilon / ratio;
  std::string filepath = sum_samples_folder+"/R"
    +std::to_string(static_cast<int>(ratio*100))+"/Scenario"
    +std::to_string(scenario);
  mkdir(filepath.c_str(), 0777);
  for (int i=0; i<7; i++) {
    std::ofstream samplefileA;
    std::ofstream samplefileB;
    samplefileA.open(filepath+"/TestCase"+std::to_string(i)+"A.txt");
    samplefileB.open(filepath+"/TestCase"+std::to_string(i)+"B.txt");
    for (int i=0; i<number_of_samples; i++) {
      int64_t outputA = BoundedSumAlgorithm(values, granularity,
        implemented_epsilon, max_partitions, lower, upper);
      double discretized_outputA = DiscretizeSum(outputA, granularity);
      samplefileA << discretized_outputA << "\n";
      int64_t outputB = BoundedSumAlgorithm(neighbor_values, granularity,
        implemented_epsilon, max_partitions, lower, upper);
      double discretized_outputB = DiscretizeSum(outputB, granularity);
      samplefileB << discretized_outputB << "\n";
    }
  samplefileA.close();
  samplefileB.close();
  }
}

void GenerateAllScenariosSum(double ratio) {
  const int num_of_samples = 100;
  double small_epsilon = 0.1;
  double default_epsilon = std::log(3);
  double large_epsilon = 2*std::log(3);

// Laplace noise, empty sum, default parameters
  std::vector<double>zero_vector{0};
  CreateSingleScenarioSum(1,zero_vector,0.015625,default_epsilon,1,0,1,num_of_samples,
    1000,ratio);

// Laplace noise, empty sum, many partitions contributed
  CreateSingleScenarioSum(2,zero_vector,0.125,default_epsilon,25,0,1,num_of_samples,
    25000,ratio);

// Laplace noise, empty sum, large bounds
  CreateSingleScenarioSum(3,zero_vector,0.25,default_epsilon,1,-50,49,num_of_samples,
    -50000,ratio);

// Laplace noise, empty sum, small epsilon
  CreateSingleScenarioSum(4,zero_vector,0.0625,default_epsilon,1,0,1,num_of_samples,
    1000,ratio);

// Laplace noise, empty sum, large epsilon
  CreateSingleScenarioSum(5,zero_vector,0.03125,large_epsilon,1,0,1,num_of_samples,
    1000,ratio);

// Laplace noise, small positive sum, default parameters
  std::vector<double>vec1{0.64872,0.12707,0.00128,0.14684,0.86507};
  CreateSingleScenarioSum(6,vec1,0.015625,default_epsilon,1,0,1,num_of_samples,1000,
    ratio);

// Laplace noise, small positive sum, many partitions contributed
  CreateSingleScenarioSum(7,vec1,0.125,default_epsilon,25,0,1,num_of_samples,25000,
    ratio);

// Laplace noise, small positive sum, small epsilon
  CreateSingleScenarioSum(8,vec1,0.0625,small_epsilon,1,0,1,num_of_samples,1000,ratio);

// Laplace noise, small positive sum, large epsilon
  CreateSingleScenarioSum(9,vec1,0.0625,large_epsilon,1,0,1,num_of_samples,1000,ratio);

// Laplace noise, large positive sum, default parameters
  std::vector<double>vec2{32.43606,35.35006,40.73424,32.53939,7.081785};
  CreateSingleScenarioSum(10,vec2,0.25,default_epsilon,1,0,50,num_of_samples,50000,
    ratio);

// Laplace noise, large positive sum, many partitions contributed
  CreateSingleScenarioSum(11,vec2,1,default_epsilon,25,0,50,num_of_samples,50000*25,
    ratio);

// Laplace noise, large positive sum, small epsilon
  CreateSingleScenarioSum(12,vec2,1,small_epsilon,1,0,50,num_of_samples,50000,ratio);

// Laplace noise, large positive sum, large epsilon
  CreateSingleScenarioSum(13,vec2,0.5,large_epsilon,1,0,50,num_of_samples,50000,ratio);

// Laplace noise, large mixed sum, default parameters
  std::vector<double>vec3{-32.43606,35.35006,-40.73424,-32.53939,7.081785};
  CreateSingleScenarioSum(14,vec3,0.25,default_epsilon,1,-50,49.9,num_of_samples,
    -50000,ratio);

// Laplace noise, large mixed sum, many partitions contributed
  CreateSingleScenarioSum(15,vec3,1,default_epsilon,1,-50,49.9,num_of_samples,
    -50000*25,ratio);

// Laplace noise, large mixed sum, small epsilon
  CreateSingleScenarioSum(16,vec3,1,small_epsilon,1,-50,49.9,num_of_samples,-50000,
    ratio);

// Laplace noise, large mixed sum, large epsilon
  CreateSingleScenarioSum(17,vec3,0.5,large_epsilon,1,-50,49.9,num_of_samples,-50000,
    ratio);
}
} // testing
} // differential_privacy

