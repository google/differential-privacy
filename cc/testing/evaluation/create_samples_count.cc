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

// #include <bits/stdc++.h>
// #include <sys/stat.h>
// #include <sys/types.h>

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
  std::string filepath = "/usr/local/google/home/krosman/myproject/CountSamples/R"
    +std::to_string(static_cast<int>(ratio*100))+"/Scenario"
    +std::to_string(scenario);
  mkdir(filepath.c_str(), 0777);
// For each sample, run CountAlgorithm 1M times. Run each sample seven times. 
// Generate 14 files (seven pairs of files) with each run.
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

  // // Laplace noise, large count, default parameters
  // differential_privacy::testing::CreateSingleScenario(11,1000000,num_of_samples,1,1,default_epsilon,ratio);

  // // Laplace noise, large count, two partitions contributed
  // differential_privacy::testing::CreateSingleScenario(12,1000000,num_of_samples,2,2,large_epsilon,ratio);

  // // Laplace noise, large count, many partitions contributed
  // differential_privacy::testing::CreateSingleScenario(13,1000000,num_of_samples,250,250,default_epsilon,ratio);

  // // Laplace noise, small count, large epsilon
  // differential_privacy::testing::CreateSingleScenario(14,1000000,num_of_samples,1,1,small_epsilon,ratio);

  // // Laplace noise, small count, large epsilon
  // differential_privacy::testing::CreateSingleScenario(15,1000000,num_of_samples,1,1,large_epsilon,ratio);

    }
  }
}

int main(int argc, char** argv) {
  for (int i = 81; i <= 99; i++) {
    std::string filepath = "/usr/local/google/home/krosman/myproject/CountSamples/R"+std::to_string(i);
    mkdir(filepath.c_str(), 0777);
    const double r = i / 100.0;
    differential_privacy::testing::GenerateAllScenarios(r);
  }
  return 0;
}

//   std::vector<double> a = {1,2,3,50};

//   for (int i=1; i<=20; i++){

//   double result = differential_privacy::testing::boundedmean(a, 0.15625, std::default_epsilon, 1, 0, 1);
//   std::cout << result << std::endl;
// }

// double discretize(double true_value, double granularity) {
//   if (granularity > 0) {
//     double abs_value = abs(true_value);
//     double scaled_sample = true_value/granularity;
//     if (abs_value >= 9007199254740991L) {
//       double discretized_value = scaled_sample * granularity;
//       return discretized_value;
//   } else {
//     double discretized_value = std::round(scaled_sample) * granularity;
//     return discretized_value;
//     }
//   }
//   else {
//     std::cout << "Granularity must be positive. Try again, please." << std::endl;
//   }
// }

// // Construct BoundedSum algorithm.
// int64_t boundedsum(double true_value, double granularity, double epsilon, int max_partitions,
//   int lower, int upper) {
//     std::unique_ptr<BoundedSum<int64_t>> boundedsum =
//         BoundedSum<int64_t>::Builder().SetEpsilon(epsilon)
//                                 .SetMaxPartitionsContributed(max_partitions)
//                                 .SetLower(lower)
//                                 .SetUpper(upper) 
//                                 .Build()
//                                 .ValueOrDie();

//     boundedsum->AddEntry(discretize(true_value, granularity));
//     return GetValue<int64_t>(boundedsum->PartialResult().ValueOrDie());
//     }

// void foo(int a) {
//     std::cout << a << "\n";

// }

// // Construct BoundedMean algorithm.
// double boundedmean(const std::vector<double>& values, double granularity, double epsilon, int max_partitions,
// int lower, int upper) {
//   std::unique_ptr<BoundedMean<double>> boundedmean = 
//     BoundedMean<double>::Builder()
//           .SetEpsilon(epsilon)
//           .SetMaxPartitionsContributed(max_partitions)
//           .SetLower(lower)
//           .SetUpper(upper)
//           .Build()
//           .ValueOrDie();

//   std::vector<double> vect;
//   for (double i : values) {
// //    std::cout << i << std::endl;
//     double discretized_value = discretize(i, granularity);
//  //   std::cout << discretized_value << std::endl;
//     vect.push_back(discretized_value);
//   }

//   // for (double i: vect) {
//   //   std::cout << i << std::endl;
//   // }
//     // Compute the count and get the result.
//     base::StatusOr<Output> result = boundedmean->Result(vect.begin(), vect.end());

//     // Convert result to an Output object
//     Output obj = result.ValueOrDie();

//     // GetValue can be used to extract the value from an Output protobuf. For
//     // count, this is always an int64_t value.
//     // std::cout << GetValue<int64_t>(obj) << std::endl;

//     return GetValue<double>(obj);
//   }
// }
// }

  // for (int i=1; i<=20; i++) {

  //   int outcome = differential_privacy::testing::boundedsum(1.78898, small_epsilon5625, std::default_epsilon, 1, 0, 1);
  //   std::cout << outcome << std::endl;

  // differential_privacy::testing::discretize(1.7);

// }

//  differential_privacy::testing::CreateCases(1,55,10000,1,1,std::default_epsilon,0.9);

  // auto static start = std::chrono::high_resolution_clock::now();

  // std::cout << "This is a test!" << std::endl;

  // for (int r=85; r<100; r++) {

  //     double ratio = r/100.0; 

  //     std::cout << r << std::endl;
      
  //     differential_privacy::testing::CreateFolders(ratio);
  //   }

  // auto static finish = std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> elapsed = finish - start;
  // std::cout << "Elapsed time: " << elapsed.count() << "s\n";

// void CreateFolders(double ratio) {

//   int num_of_samples = 1000000;

//   // Laplace noise, empty count, default parameters
//   differential_privacy::testing::CreateSamples(1,0,num_of_samples,1,1,default_epsilon,ratio);

//   // Laplace noise, empty count, two partitions contributed
//   differential_privacy::testing::CreateSamples(2,0,num_of_samples,2,2,default_epsilon,ratio);

//   // Laplace noise, empty count, many partitions contributed
//   differential_privacy::testing::CreateSamples(3,0,num_of_samples,250,250,default_epsilon,ratio);

//   // Laplace noise, empty count, small epsilon
//   differential_privacy::testing::CreateSamples(4,0,num_of_samples,1,1,small_epsilon,ratio);

//   // Laplace noise, empty count, large epsilon
//   differential_privacy::testing::CreateSamples(5,0,num_of_samples,1,1,large_epsilon,ratio);

//   // Laplace noise, small count, default parameters
//   differential_privacy::testing::CreateSamples(6,28,num_of_samples,1,1,default_epsilon,ratio);

//   // Laplace noise, small count, two partitions contributed
//   differential_privacy::testing::CreateSamples(7,28,num_of_samples,2,2,default_epsilon,ratio);

//   // Laplace noise, small count, many partitions contributed
//   differential_privacy::testing::CreateSamples(8,28,num_of_samples,250,250,default_epsilon,ratio);

//   // Laplace noise, small count, small epsilon
//   differential_privacy::testing::CreateSamples(9,28,num_of_samples,1,1,small_epsilon,ratio);

//   // Laplace noise, small count, large epsilon
//   differential_privacy::testing::CreateSamples(10,28,num_of_samples,1,1,large_epsilon,ratio);

//     }
//   }
// }

    // // Compute the count and get the result.
    // base::StatusOr<Output> result = boundedsum->Result(values.begin(), values.end());

    // // Convert result to an Output object
    // Output obj = result.ValueOrDie();

    // // GetValue can be used to extract the value from an Output protobuf. For
    // // count, this is always an int64_t value.
    // return GetValue<int64_t>(obj);

    // boundedsum->AddEntry(10);
    // boundedsum->AddEntry(10);
    // boundedsum->AddEntry(10);

    // double fraction = true_value/100.0;

    // for (int i=0; i<100; i++) {
    //   boundedsum->AddEntry(fraction);
    // }

  // advertised: 1.10
  // ratio = 1.10/implemented
  // 0.50 = 1.10/implemented
  // 0.50 * implemented = 1.10
  // implemented = 1.10/ratio

  // // Laplace noise, empty count, default parameters
  // differential_privacy::testing::CreateCases(1,0,num_of_samples,1,1,default_epsilon,1);

  // // Laplace noise, empty count, two partitions contributed
  // differential_privacy::testing::CreateCases(2,0,num_of_samples,2,2,default_epsilon,1);

  // // Laplace noise, empty count, many partitions contributed
  // differential_privacy::testing::CreateCases(3,0,num_of_samples,250,250,default_epsilon,1);

  // // Laplace noise, empty count, small epsilon
  // differential_privacy::testing::CreateCases(4,0,num_of_samples,1,1,small_epsilon,1);

  // // Laplace noise, empty count, large epsilon
  // differential_privacy::testing::CreateCases(5,0,num_of_samples,1,1,large_epsilon,1);

  // // Laplace noise, small count, default parameters
  // differential_privacy::testing::CreateCases(6,28,num_of_samples,1,1,default_epsilon,1);

  // // Laplace noise, small count, two partitions contributed
  // differential_privacy::testing::CreateCases(7,28,num_of_samples,2,2,default_epsilon,1);

  // // Laplace noise, small count, many partitions contributed
  // differential_privacy::testing::CreateCases(8,28,num_of_samples,250,250,default_epsilon,1);

  // // Laplace noise, small count, small epsilon
  // differential_privacy::testing::CreateCases(9,28,num_of_samples,1,1,small_epsilon,1);

  // // Laplace noise, small count, large epsilon
  // differential_privacy::testing::CreateCases(10,28,num_of_samples,1,1,large_epsilon,1);

      // if (empty == false) {

      //   std::ofstream samplefileA;
      //   std::ofstream samplefileB;

      //   samplefileA.open(filepath+"/TestCase"+num+"A.txt");
      //   samplefileB.open(filepath+"/TestCase"+num+"B.txt");

      //   std::vector<double> sampleA(true_value,100);
      //   std::vector<double> sampleB(neighbor_value,100);

      //   samplefileA << std::fixed << std::showpoint;
      //   samplefileA << std::setprecision(1);

      //   samplefileB << std::fixed << std::showpoint;
      //   samplefileB << std::setprecision(1);

      // for (int i=0; i<number_of_samples; i++) {

      //     std::cout << i << std::endl;

      //     double outputA = count(sampleA, epsilon, max_partitions);

      //     samplefileA << outputA << "\n";

      //     double outputB = count(sampleB, epsilon, max_partitions);

      //     samplefileB << outputB << "\n";

      // }

      // samplefileA.close();
      // samplefileB.close();

  // long long int x = 2147483647;
  // long long int y = 2146736422;
  // long long int z = 1383713414;
  // long long int q = 2146374624;
  // long long int p = 1989764249;
  // long long int r = x+y+z+q+p;


  // long int x = 2147483647;
  // long int y = 2146736422;
  // long int z = 1383713414;
  // long int q = 2146374624;
  // long int p = 1989764249;
  // long int r = x+y+z+q+p;

  // // // Laplace noise, large count, default parameters
  // CreateCases(11,r,num_of_samples,1,1,default_epsilon,false);

  // // // Laplace noise, large count, two partitions contributed
  // CreateCases(12,r,num_of_samples,2,2,default_epsilon,false);

  // // // Laplace noise, large count, many partitions contributed
  // CreateCases(13,r,num_of_samples,250,250,default_epsilon,false);

  // // // Laplace noinse, large count, small epsilon
  // CreateCases(14,r,num_of_samples,1,1,small_epsilon,false);

  // // // Laplace noise, large count, large epsilon
  // CreateCases(15,r,num_of_samples,1,1,large_epsilon,false);


//  long long x = 2147483647+2146736422+1383713414+2146374624+1989764249;

  // // Laplace noise, large count, default parameters
  // differential_privacy::testing::CreateCases(11,x,num_of_samples,1,1,default_epsilon,false);

  // // // Laplace noise, large count, two partitions contributed
  // differential_privacy::testing::CreateCases(12,r,num_of_samples,2,2,default_epsilon,false);

  // // // Laplace noise, large count, many partitions contributed
  // differential_privacy::testing::CreateCases(13,r,num_of_samples,250,250,default_epsilon,false);

  // // // Laplace noinse, large count, small epsilon
  // differential_privacy::testing::CreateCases(14,r,num_of_samples,1,1,small_epsilon,false);

  // // // Laplace noise, large count, large epsilon
  // differential_privacy::testing::CreateCases(15,r,num_of_samples,1,1,large_epsilon,false);