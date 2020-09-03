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

#include "create_samples_mean.h"

// Creates pairs of samples of differentially private means. 
// Each sample-pair replicates a unique scenario constructed in the proto for
// BoundedMeanDpTest.java, available here:
// https://github.com/google/differential-privacy/blob/main/proto/testing/bounded_mean_dp_test_cases.textproto.

namespace differential_privacy {

namespace testing {

const std::string mean_samples_folder = "../statisticaltester/boundedmeansamples";

double DiscretizeMean(double true_value, double granularity) {
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

// Construct the BoundedMean algorithm.
double DPMean(std::vector<double> values, double granularity,
  double epsilon, int max_partitions, int max_contributions, int lower, int upper) {
    std::unique_ptr<BoundedMean<double>> boundedmean =
    BoundedMean<double>::Builder()
    .SetEpsilon(epsilon)
    .SetMaxPartitionsContributed(max_partitions)
    .SetMaxContributionsPerPartition(max_contributions)
    .SetLower(lower)
    .SetUpper(upper) 
    .Build()
    .ValueOrDie();

  base::StatusOr<Output> result = boundedmean->Result(values.begin(),
    values.end());
  Output obj = result.ValueOrDie();
  return GetValue<double>(obj);
}

// Construct the BoundedMean algorithm for large values.
// Additional parameters enable large values to be added
// to the algorithm one by one.
double DPLargeMean(double initial_value, double extra_values_length, 
  double extra_value, double granularity, double epsilon, int max_partitions,
  int max_contributions, int lower, int upper) {
    std::unique_ptr<BoundedMean<double>> boundedmean =
    BoundedMean<double>::Builder()
    .SetEpsilon(epsilon)
    .SetMaxPartitionsContributed(max_partitions)
    .SetMaxContributionsPerPartition(max_contributions)
    .SetLower(lower)
    .SetUpper(upper) 
    .Build()
    .ValueOrDie();

// Add entry with initial value
  boundedmean->AddEntry(initial_value,granularity);
// Add entry with subsequent values
  for (int i=0; i<extra_values_length; i++) {
    boundedmean->AddEntry(extra_value);
  }
  return GetValue<double>(boundedmean->PartialResult().ValueOrDie());
}

// Creates a folder to contain all samples with a particular ratio value
// (e.g., R95). Every folder contains 22 subfolders for each unique sample-pair.
// Every subfolder contains seven runs of each sample-pair (14 files in total).
void CreateSingleScenarioMean(int scenario, std::vector<double>valuesA,
  std::vector<double>valuesB, double granularity, double epsilon,
  int max_partitions, int max_contributions, int lower, int upper,
  int number_of_samples, double ratio, double initial_value,
  double extra_values_length, double extra_value) { 

  double implemented_epsilon = epsilon / ratio;
  std::string filepath = mean_samples_folder+"/R"
    +std::to_string(static_cast<int>(ratio*100))+"/Scenario"
    +std::to_string(scenario);
  mkdir(filepath.c_str(), 0777);

  for (int i=0; i<7; i++) {
    std::ofstream samplefileA;
    std::ofstream samplefileB;
    samplefileA.open(filepath+"/TestCase"+std::to_string(i)+"A.txt");
    samplefileB.open(filepath+"/TestCase"+std::to_string(i)+"B.txt");

    if (extra_values_length == 0) {
      for (int i=0; i<number_of_samples; i++) {
        double outputA = DPMean(valuesA, granularity,
          implemented_epsilon, max_partitions, max_contributions, lower, upper);
        double discretized_outputA = DiscretizeMean(outputA, granularity);
        samplefileA << discretized_outputA << "\n";
        double outputB = DPMean(valuesB, granularity,
          implemented_epsilon, max_partitions, max_contributions, lower, upper);
        double discretized_outputB = DiscretizeMean(outputB, granularity);
        samplefileB << discretized_outputB << "\n";
      }
      samplefileA.close();
      samplefileB.close();
    }
    else {
      for (int i=0; i<number_of_samples; i++) {
      double outputA = DPMean(valuesA, granularity,
        implemented_epsilon, max_partitions, max_contributions, lower, upper);
      double discretized_outputA = DiscretizeMean(outputA, granularity);
        samplefileA << discretized_outputA << "\n";
      double outputB = DPLargeMean(initial_value, extra_values_length,
        extra_value, granularity, implemented_epsilon, max_partitions,
        max_contributions, lower, upper);
      double discretized_outputB = DiscretizeMean(outputB, granularity);
        samplefileB << discretized_outputB << "\n";
      }
      samplefileA.close();
      samplefileB.close();
    }
  }   
}
// Runs each sample-pair with parameters that replicate those specified in:
// https://github.com/google/differential-privacy/blob/main/proto/testing/bounded_mean_dp_test_cases.textproto.
void GenerateAllScenariosMean(double ratio) {

  const int num_of_samples = 100;
  double small_epsilon = 0.1;
  double default_epsilon = std::log(3);
  double large_epsilon = 2*std::log(3);

// Laplace noise, empty mean, default parameters
  std::vector<double>zero_vector{0};
  std::vector<double>valuesB1{1000};
  CreateSingleScenarioMean(1,zero_vector,valuesB1,0.0078125,default_epsilon,1,1,0,1,num_of_samples,ratio);

// Laplace noise, empty mean, many partitions contributed
  std::vector<double>valuesB2(1000,25);
  CreateSingleScenarioMean(2,zero_vector,valuesB2,0.25,default_epsilon,25,1,0,1,num_of_samples,ratio);

// Laplace noise, empty mean, many contributions per partition
  std::vector<double>valuesB3(1000,10);
  CreateSingleScenarioMean(3,zero_vector,valuesB3,0.25,default_epsilon,1,10,0,1,num_of_samples,ratio);

// Laplace noise, empty mean, large bounds
  std::vector<double>valuesB4{-50000};
  CreateSingleScenarioMean(4,zero_vector,valuesB4,0.25,default_epsilon,1,1,-50,50,num_of_samples,ratio);

// Laplace noise, empty mean, small epsilon
  std::vector<double>valuesB5{1000};
  CreateSingleScenarioMean(5,zero_vector,valuesB5,0.0625,small_epsilon,1,1,0,1,num_of_samples,ratio);

// Laplace noise, empty mean, large epsilon
  std::vector<double>valuesB6{1000};
  CreateSingleScenarioMean(6,zero_vector,valuesB6,0.03125,large_epsilon,1,1,0,1,num_of_samples,ratio);

// Laplace noise, small positive mean, default parameters
  std::vector<double>valuesA7{1};
  std::vector<double>valuesB7{1,-1000};
  CreateSingleScenarioMean(7,valuesA7,valuesB7,0.015625,default_epsilon,1,1,0,1,num_of_samples,ratio);

// Laplace noise, small positive mean, many partitions contributed
  CreateSingleScenarioMean(8,zero_vector,zero_vector,0.25,default_epsilon,25,1,0,1,num_of_samples,ratio,
      1,25,-1000);

// Laplace noise, small positive mean, many contributions per partition
  CreateSingleScenarioMean(9,zero_vector,zero_vector,0.25,default_epsilon,1,10,0,1,num_of_samples,ratio,
    1,10,-1000);

// Laplace noise, small positive mean, small epsilon
  std::vector<double>valuesA10{1};
  std::vector<double>valuesB10{1,-1000};
  CreateSingleScenarioMean(10,valuesA10,valuesB10,0.0625,small_epsilon,1,1,0,1,num_of_samples,ratio);

// Laplace noise, small positive mean, large epsilon
  std::vector<double>valuesA11{1};
  std::vector<double>valuesB11{1,-1000};
  CreateSingleScenarioMean(11,valuesA11,valuesB11,0.0625,large_epsilon,1,1,0,1,num_of_samples,ratio);

// Laplace noise, small positive mean, multiple entries
  std::vector<double>valuesA12{0.64872,0.12707,0.00128,0.14684,0.86507};
  std::vector<double>valuesB12{0.64872,0.12707,0.00128,0.14684,0.86507,1000};
  CreateSingleScenarioMean(12,valuesA12,valuesB12,0.015625,default_epsilon,1,1,0,1,num_of_samples,ratio);

// Laplace noise, large positive mean, default parameters
  std::vector<double>valuesA13{50};
  std::vector<double>valuesB13{50,-1000};
  CreateSingleScenarioMean(13,valuesA13,valuesB13,0.25,default_epsilon,1,1,0,50,num_of_samples,ratio);

// Laplace noise, large positive mean, many partitions contributed 
  CreateSingleScenarioMean(14,zero_vector,zero_vector,2,default_epsilon,25,1,0,50,
    num_of_samples,ratio,50,25,-1000);

// Laplace noise, large positive mean, many contributions per partition
  CreateSingleScenarioMean(15,zero_vector,zero_vector,1,default_epsilon,1,10,0,50,
    num_of_samples,ratio,50,10,-1000);

// Laplace noise, large positive mean, small epsilon
  std::vector<double>valuesA16{50};
  std::vector<double>valuesB16{50,-1000};
  CreateSingleScenarioMean(16,valuesA16,valuesB16,2,small_epsilon,1,1,0,50,num_of_samples,ratio);

// Laplace noise, large positive mean, large epsilon
  std::vector<double>valuesA17{50};
  std::vector<double>valuesB17{50,-1000};
  CreateSingleScenarioMean(17,valuesA17,valuesB17,0.5,large_epsilon,1,1,0,50,num_of_samples,ratio);

// Laplace noise, large positive mean, multiple entries
  std::vector<double>valuesA18{32,43606,35.35006,40.73424,32.53939,7.081785};
  std::vector<double>valuesB18{32,43606,35.35006,40.73424,32.53939,7.081785,-1000};
  CreateSingleScenarioMean(18,valuesA18,valuesB18,0.25,default_epsilon,1,1,0,50,num_of_samples,ratio);

// Laplace noise, large mixed mean, default parameters
  std::vector<double>valuesA19{-50};
  std::vector<double>valuesB19{-50,50000};
  CreateSingleScenarioMean(19,valuesA19,valuesB19,0.5,default_epsilon,1,1,-50,50,num_of_samples,ratio);

// Laplace noise, large mixed mean, many partitions contributed
  CreateSingleScenarioMean(20,zero_vector,zero_vector,2,default_epsilon,25,1,-50,50,
    num_of_samples,ratio,-50,25,50000);

// Laplace noise, large mixed mean, many contributions per partition
  CreateSingleScenarioMean(21,zero_vector,zero_vector,2,default_epsilon,1,10,-50,50,
    num_of_samples,ratio,-50,10,50000);

// Laplace noise, large mixed mean, multiple entries
  std::vector<double>valuesA22{-32.43606,35.35006,-40.73424,-32.53939,7.081785};
  std::vector<double>valuesB22{-32.43606,35.35006,-40.73424,-32.53939,7.081785,50000};
  CreateSingleScenarioMean(22,valuesA22,valuesB22,0.5,default_epsilon,1,1,-50,50,num_of_samples,ratio);
}
} // testing
} // differential_privacy
