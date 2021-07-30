//
// Copyright 2021 Google LLC
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
//

#ifndef DIFFERENTIAL_PRIVACY_CPP_TESTING_STATISTICAL_TESTS_UTILS_H_
#define DIFFERENTIAL_PRIVACY_CPP_TESTING_STATISTICAL_TESTS_UTILS_H_

#include <fstream>
#include <functional>
#include <iostream>

#include "google/protobuf/text_format.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "algorithms/rand.h"

namespace differential_privacy::testing {

// Sample a value from Laplace distribution, implemented with absl random
// generators (i.e. non-secure).
double SampleReferenceLaplacian(double mean, double variance,
                                SecureURBG* random);

// Generates number_of_samples samples from both sample_generator_a and
// sample_generator_b generators and decides whether these 2 sets of random
// samples were likely drawn from similar discrete distributions. See
// (broken link) for more information.
bool GenerateClosenessVote(std::function<double()> sample_generator_a,
                           std::function<double()> sample_generator_b,
                           int number_of_samples, double l2_tolerance,
                           double granularity);

// Decides whether two sets of random samples were likely drawn from a pair of
// discrete distributions that approximately satisfy (ε,δ) differential privacy.
//
// The two distributions are considered to be (ε,δ) differentially private if
// the likelihood of any event with respect to the first distribution is at most
// δ plus e^ε times the likelihood of the same event in the second distribution
// and vice versa. Moreover, the distributions are considered approximately
// (ε,δ) differentially private if there exists a δ' such that the distributions
// are (ε,δ') differentially private and |δ' - δ| is less than half of a given
// tolerance α. Otherwise if no δ' exists such that |δ' - δ| is less than α, the
// distributions are not considered approximately (ε, δ) differentially private.
// Assuming that α > (m / n)^0.5 * (1 + e^(2 * ε)), the error probability is at
// most (1 + e^(2 * ε)) / (n * (α - (m / n)^0.5 * (1 + e^(2 * ε)))^2), where m
// is the size of the support of the distributions and n is the expected value
// of a Poisson distribution from which the number of samples is drawn. See
// (broken link) for more information.
bool VerifyApproximateDp(const std::vector<double>& samples_a,
                         const std::vector<double>& samples_b, double epsilon,
                         double delta, double delta_tolerance);

// Generates number_of_samples samples from both sample_generator_a and
// sample_generator_b generators and decides whether two sets of random samples
// were likely drawn from a pair of discrete distributions that approximately
// satisfy (ε,δ) differential privacy. See (broken link) for more
// information. The test will fail if this pair of samples do not satisfy
// (ε,δ + delta_tolerance) differential privacy.
bool GenerateApproximateDpVote(std::function<double()> sample_generator_a,
                               std::function<double()> sample_generator_b,
                               int number_of_samples, double epsilon,
                               double delta, double delta_tolerance,
                               double granularity);

// Generates number_of_votes of votes from vote_generator to determine a
// majority. Stops early as soon as the majority is clear. Returns the majority.
bool RunBallot(std::function<bool()> vote_generator, int number_of_votes);

template <typename T>
std::optional<T> ReadProto(std::istream* proto_file) {
  T tests;
  std::string serialized_protobuf;

  std::string line;
  while (getline(*proto_file, line)) {
    absl::StrAppend(&serialized_protobuf, line, "\n");
  }
  *proto_file >> serialized_protobuf;
  if (!google::protobuf::TextFormat::ParseFromString(serialized_protobuf, &tests)) {
    return std::optional<T>();
  }
  return tests;
}

template <typename T>
std::optional<T> ReadProto(const std::string& path) {
  std::ifstream proto_file(path);
  if (!proto_file.is_open()) {
    return std::optional<T>();
  }
  return ReadProto<T>(&proto_file);
}

// Partitions the interval between lower and upper into num_buckets subintervals
// of equal size and return the index (from 0 to num_buckets - 1) of the
// subinterval that contains the specified sample.
// Samples outside the bounds will be assigned to the lowest or highest bin as
// appropriate. Samples that are exactly equal to bin boundaries will be
// assigned to the higher bin.
int Bucketize(double sample, double lower, double upper, int num_buckets);

}  // namespace differential_privacy::testing

#endif  // DIFFERENTIAL_PRIVACY_CPP_TESTING_STATISTICAL_TESTS_UTILS_H_
