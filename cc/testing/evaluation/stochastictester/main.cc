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
//   std::ofstream countfile;
//   std::ofstream sumfile;
//   std::ofstream meanfile;

//   differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> count_summary;
//   differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> sum_summary;
//   differential_privacy::base::StatusOr<differential_privacy::testing::SummaryResults> mean_summary;

// // Dataset values have been hard-coded in order to maintain consistency with 
// // the Statistical Tester. They should not be changed under any circumstances.
//   int const count_num_datasets = 10;
//   int const sum_num_datasets = 17;
//   int const mean_num_datasets = 22;

// Default values which can be changed by the user.
  double num_samples_per_histogram = 100;
  double ratio_min = 80.0;
  double ratio_max = 85.0;

//   std::string header = "test_name,algorithm,expected,actual,ratio,num_datasets,num_samples,time(sec)";
//   std::string filepath = "testing/evaluation/results/";

//   time_t now = time(0); 
//   char* dt = ctime(&now);

// // Specify name of files, ratio_min, ratio_max, number of samples on the command line. 
//   if (argc == 6) {
//     countfile.open(filepath+argv[1]);
//     sumfile.open(filepath+argv[2]);
//     meanfile.open(filepath+argv[3]);

//     double ratio_min = strtod(argv[4],NULL);
//     double ratio_max = strtod(argv[5],NULL);
//     double num_samples_per_histogram = strtod(argv[6],NULL);

//     if ((ratio_min > 0) && (ratio_max > 0) && (ratio_min < 1) && (ratio_max < 1)
//       && (num_samples_per_histogram > 0)) {
//       countfile << dt << "\n";
//       countfile << header << "\n";
//       count_summary = differential_privacy::testing::GetTestResultsForCount(
//       count_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,countfile);
//       countfile.close();

//       sumfile << dt << "\n";
//       sumfile << header << "\n";
//       sum_summary = differential_privacy::testing::GetTestResultsForSum(
//         sum_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,sumfile);
//       sumfile.close();

//       meanfile << dt << "\n";
//       meanfile << header << "\n";
//       mean_summary = differential_privacy::testing::GetTestResultsForMean(
//         mean_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,meanfile);
//       meanfile.close();
//     }
//     else {
//       std::cout << "Invalid parameter(s) specified." << std::endl; 
//     }
//   }

//   else {
// // Use default parameter values.
//     countfile.open(filepath+"stochastic_tester_results_count.txt");
//     countfile << dt << "\n";
//     countfile << header << "\n";
//     count_summary = differential_privacy::testing::GetTestResultsForCount(
//     count_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,countfile);
//     countfile.close();

//     sumfile.open(filepath+"stochastic_tester_results_sum.txt");
//     sumfile << dt << "\n";
//     sumfile << header << "\n";
//     sum_summary = differential_privacy::testing::GetTestResultsForSum(
//       sum_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,sumfile);
//     sumfile.close();

//     meanfile.open(filepath+"stochastic_tester_results_mean.txt");
//     meanfile << dt << "\n";
//     meanfile << header << "\n";
//     mean_summary = differential_privacy::testing::GetTestResultsForMean(
//       mean_num_datasets,num_samples_per_histogram,ratio_min,ratio_max,meanfile);
//     meanfile.close();
//   }

// Generates samples of Count algorithm with insufficient noise.
//   mkdir(differential_privacy::testing::count_samples_folder.c_str(), 0777);
//   for (int i = ratio_min; i <= ratio_max; i++) {
//     std::string count_path = differential_privacy::testing::count_samples_folder
//       +"/R"+std::to_string(i);
//     mkdir(count_path.c_str(), 0777);
//     const double r = i / 100.0;
//     differential_privacy::testing::GenerateAllScenariosCount(r);
//   }

// // Generates samples of BoundedSum algorithm with insufficient noise.
//   mkdir(differential_privacy::testing::sum_samples_folder.c_str(), 0777);
//   for (int i = ratio_min; i <= ratio_max; i++) {
//     std::string sum_path = differential_privacy::testing::sum_samples_folder
//       +"/R"+std::to_string(i);
//     mkdir(sum_path.c_str(), 0777);
//     const double r = i / 100.0;
//     differential_privacy::testing::GenerateAllScenariosSum(r);
//   }

// Generates samples of BoundedMean algorithm with insufficient noise.
  mkdir(differential_privacy::testing::mean_samples_folder.c_str(), 0777);
  for (int i = ratio_min; i <= ratio_max; i++) {
    std::string mean_path = differential_privacy::testing::mean_samples_folder
      +"/R"+std::to_string(i);
    mkdir(mean_path.c_str(), 0777);
    const double r = i / 100.0;
    differential_privacy::testing::GenerateAllScenariosMean(r);
  }
  
  return 0;
}