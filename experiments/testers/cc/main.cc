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
#include "create_samples_count.h"
#include "create_samples_sum.h"
#include "create_samples_mean.h"

#include <chrono>
#include <ctime>
#include <filesystem>
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
#include <sys/stat.h>

#include "algorithms/algorithm.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/bounded-mean.h"
#include "algorithms/count.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "base/statusor.h"
#include "proto/data.pb.h"
#include "testing/sequence.h"
#include "testing/stochastic_tester.h"

int main(int argc, char *argv[]) {

// Runs Stochastic Tester over series of algorithms with insufficient noise.
  std::ofstream countfile;
  std::ofstream sumfile;
  std::ofstream meanfile;

  differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> count_summary;
  differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> sum_summary;
  differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> mean_summary;

// Dataset values have been hard-coded in order to maintain consistency with 
// the Statistical Tester. They should not be changed under any circumstances.
  int const count_num_datasets = 10;
  int const sum_num_datasets = 17;
  int const mean_num_datasets = 22;

// Default values which can be changed by the user.
  double num_samples_per_histogram = 100;
  double ratio_min = 0.80;
  double ratio_max = 0.85;
  double increment = 0.01;

  std::string header = "test_name,algorithm,expected,actual,ratio,num_datasets,num_samples,time(sec)";
  std::string filepath = "../results/";

  time_t now = time(0); 
  char* dt = ctime(&now);

// Specify ratio_min, ratio_max, number of samples, and name of files on the command line. 
  if (argc==7) {

    double ratio_min = strtod(argv[1],NULL);
    double ratio_max = strtod(argv[2],NULL);
    double num_samples_per_histogram = strtod(argv[3],NULL);

    countfile.open(filepath+argv[4]);
    sumfile.open(filepath+argv[5]);
    meanfile.open(filepath+argv[6]);
  }

    else {
      std::cout << "Invalid parameter(s) specified." << std::endl; 
    }
// Use default parameter values.
    countfile.open(filepath+"stochastic_tester_results_count.txt");
    countfile << dt << "\n";
    countfile << header << "\n";
    count_summary = differential_privacy::testing::GetTestResultsForCount(
    count_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,0.01,countfile);
    countfile.close();

    sumfile.open(filepath+"stochastic_tester_results_sum.txt");
    sumfile << dt << "\n";
    sumfile << header << "\n";
    sum_summary = differential_privacy::testing::GetTestResultsForSum(
      sum_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,0.01,sumfile);
    sumfile.close();

    meanfile.open(filepath+"stochastic_tester_results_mean.txt");
    meanfile << dt << "\n";
    meanfile << header << "\n";
    mean_summary = differential_privacy::testing::GetTestResultsForMean(
      mean_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,0.01,meanfile);
    meanfile.close();

// Generates samples of Count algorithm with insufficient noise.
  mkdir(differential_privacy::testing::count_samples_folder.c_str(), 0777);
  int num_iterations = ceil((ratio_max - ratio_min) / increment);
  for (double i=0; i <= num_iterations; ++i) {
    double ratio = i * increment + ratio_min;
    int ratio_name = (int)lround(ratio*100);
    std::string count_path = differential_privacy::testing::count_samples_folder
      +"/R"+std::to_string(ratio_name);
    mkdir(count_path.c_str(), 0777);
    differential_privacy::testing::GenerateAllScenariosCount(ratio);
  }

// // Generates samples of BoundedSum algorithm with insufficient noise.
  mkdir(differential_privacy::testing::sum_samples_folder.c_str(), 0777);
  for (double i=0; i <= num_iterations; ++i) {
    double ratio = i * increment + ratio_min;
    int ratio_name = (int)lround(ratio*100);
    std::string sum_path = differential_privacy::testing::sum_samples_folder
      +"/R"+std::to_string(ratio_name);
    mkdir(sum_path.c_str(), 0777);
    differential_privacy::testing::GenerateAllScenariosSum(ratio);
  }

// // Generates samples of BoundedMean algorithm with insufficient noise.
  mkdir(differential_privacy::testing::mean_samples_folder.c_str(), 0777);
  for (double i=0; i <= num_iterations; ++i) {
    double ratio = i * increment + ratio_min;
    int ratio_name = (int)lround(ratio*100);
    std::string mean_path = differential_privacy::testing::mean_samples_folder
      +"/R"+std::to_string(ratio_name);
    mkdir(mean_path.c_str(), 0777);
    differential_privacy::testing::GenerateAllScenariosMean(ratio);
  }
  
  return 0;
}

    // if ((ratio_min > 0) && (ratio_max > 0) && (ratio_min < 100) && (ratio_max < 100)
    //   && (num_samples_per_histogram > 0)) {
    //   countfile << dt << "\n";
    //   countfile << header << "\n";
    //   count_summary = differential_privacy::testing::GetTestResultsForCount(
    //   count_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,0.1,countfile);
    //   countfile.close();

    //   sumfile << dt << "\n";
    //   sumfile << header << "\n";
    //   sum_summary = differential_privacy::testing::GetTestResultsForSum(
    //     sum_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,0.1,sumfile);
    //   sumfile.close();

    //   meanfile << dt << "\n";
    //   meanfile << header << "\n";
    //   mean_summary = differential_privacy::testing::GetTestResultsForMean(
    //     mean_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,0.1,meanfile);
    //   meanfile.close();
    // }