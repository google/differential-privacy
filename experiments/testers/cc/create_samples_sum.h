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

#ifndef CREATE_SAMPLES_SUM_H
#define CREATE_SAMPLES_SUM_H

#include "algorithms/bounded-sum.h"

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
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "base/statusor.h"
#include "proto/data.pb.h"
#include "testing/sequence.h"

namespace differential_privacy {

namespace testing {

extern const std::string sum_samples_folder;

double DiscretizeSum(double true_value, double granularity);

// Construct the BoundedSum algorithm.
double BoundedSumAlgorithm(std::vector<double> values, double granularity,
  double epsilon, int max_partitions, int lower, int upper);

// Creates a folder to contain all samples with a particular ratio value
// (e.g., R95). Every folder contains 17 subfolders for each unique sample-pair.
// Every subfolder contains seven runs of each sample-pair (14 files in total).
void CreateSingleScenarioSum(int scenario, std::vector<double> values,
  double granularity, double epsilon, int max_partitions,
	int lower, int upper, int number_of_samples, int increment, double ratio);

void GenerateAllScenariosSum(double ratio);

} // testing
} // differential_privacy

#endif // CREATE_SAMPLES_SUM_H