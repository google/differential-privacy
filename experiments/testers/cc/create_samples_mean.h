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

#ifndef CREATE_SAMPLES_MEAN_H
#define CREATE_SAMPLES_MEAN_H

#include "algorithms/bounded-mean.h"

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
#include <sys/stat.h>

#include "absl/random/distributions.h"
#include "absl/memory/memory.h"

#include "algorithms/algorithm.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/count.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "base/statusor.h"
#include "proto/data.pb.h"
#include "testing/sequence.h"

namespace differential_privacy {

namespace testing {

// Creates pairs of samples of differentially private means. 
// Each sample-pair replicates a unique scenario constructed in the proto for
// BoundedMeanDpTest.java, available here:
// https://github.com/google/differential-privacy/blob/main/proto/testing/bounded_mean_dp_test_cases.textproto.

extern const std::string mean_samples_folder;

double DiscretizeMean(double true_value, double granularity);

// Construct the BoundedMean algorithm.
double DPMean(std::vector<double> values, double granularity,
  double epsilon, int max_partitions, int max_contributions, int lower, int upper);

// Construct the BoundedMean algorithm for large values.
double DPLargeMean(double initial_value, double extra_values_length, 
  double extra_value, double granularity, double epsilon, int max_partitions,
  int max_contributions, int lower, int upper);

// Creates a folder to contain all samples with a particular ratio value
// (e.g., R95). Every folder contains 22 subfolders for each unique sample-pair.
// Every subfolder contains seven runs of each sample-pair (14 files in total).
void CreateSingleScenarioMean(int scenario, std::vector<double>valuesA,
  std::vector<double>valuesB, double granularity, double epsilon,
  int max_partitions, int max_contributions, int lower, int upper,
  int number_of_samples, double ratio, double initial_value = 0,
  double extra_values_length = 0, double extra_value = 0);

// Runs each sample-pair with parameters that replicate those specified in:
// https://github.com/google/differential-privacy/blob/main/proto/testing/bounded_mean_dp_test_cases.textproto.
void GenerateAllScenariosMean(double ratio);

} // testing
} // differential_privacy

#endif // CREATE_SAMPLES_MEAN_H