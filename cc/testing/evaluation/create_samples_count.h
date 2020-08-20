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

#ifndef CREATE_SAMPLES_COUNT_H_
#define CREATE_SAMPLES_COUNT_H_

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
#include "algorithms/bounded-mean.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/util.h"
#include "base/statusor.h"
#include "proto/data.pb.h"
#include "testing/sequence.h"

namespace differential_privacy {

namespace testing {

// Creates samples of data pairs using the Count algorithm. Each data pair
// and its associated input parameters replicates a scenario constructed in 
// the [textproto](https://github.com/google/differential-privacy/blob/main/proto/testing/count_dp_test_cases.textproto)
// for CountDpTest.java.

extern const std::string folder_name;

// Construct Count algorithm.
int64_t CountAlgorithm(const std::vector<int>& values, double epsilon,
  int max_partitions);

// Creates a folder to contain all samples with a particular ratio value
// (e.g., R95). Every folder contains 10 subfolders for each distinct data pair.
// Every subfolder contains seven iterations of each data pair.
void CreateSingleScenario(int scenario, double true_value, int number_of_samples,
  int increment, int max_partitions, double epsilon, double ratio);

// Run each data pair to mirror parameters specified in the [textproto](https://github.com/google/differential-privacy/blob/main/proto/testing/count_dp_test_cases.textproto)
// for CountDpTest.java.
void GenerateAllScenarios(double ratio); 

} // testing
} // differential_privacy

#endif  // CREATE_SAMPLES_COUNT_H_