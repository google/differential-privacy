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

#include "testing/statistical_tests_utils.h"

#include <cmath>
#include <cstdlib>

#include "absl/random/distributions.h"
#include "absl/status/statusor.h"
#include "algorithms/rand.h"
#include "algorithms/util.h"
#include "base/status_macros.h"

namespace differential_privacy::testing {

double SampleReferenceLaplacian(double mean, double variance,
                                SecureURBG* random) {
  double b = std::sqrt(variance / 2);
  double exp = absl::Exponential<double>(*random, 1 / b);
  double flip = absl::Bernoulli(*random, 0.5);
  return mean + (flip ? -exp : exp);
}

namespace {

// Decides whether two sets of random samples were likely drawn from similar
// discrete distributions up to tolerance l2_tolerance.
//
// The distributions are considered similar if the l2 distance between them is
// less than half the specified l2 tolerance t. Otherwise, if the distance is
// greater than t, they are considered dissimilar. The error probability is at
// most 4014 / (n * t^2), where n is the number of samples contained in one of
// the sets. See (broken link) for more information.
bool VerifyCloseness(const std::vector<double>& samples_a,
                     const std::vector<double>& samples_b,
                     double l2_tolerance) {
  DCHECK(samples_a.size() == samples_b.size())
      << "The sample sets must be of equal size.";
  DCHECK(!samples_a.empty()) << "The sample sets must not be empty";
  DCHECK(l2_tolerance > 0) << "The l2 tolerance must be positive";
  DCHECK(l2_tolerance < 1) << "The l2 tolerance should be less than 1";

  absl::flat_hash_map<double, int64_t> histogram_a = BuildHistogram(samples_a);
  absl::flat_hash_map<double, int64_t> histogram_b = BuildHistogram(samples_b);

  int64_t self_collision_count_a = 0;
  int64_t self_collision_count_b = 0;
  int64_t cross_collision_count = 0;

  for (const auto& key_count : histogram_a) {
    int64_t count = key_count.second;
    self_collision_count_a += (count * (count - 1)) / 2;
  }

  for (const auto& key_count : histogram_b) {
    int64_t count = key_count.second;
    self_collision_count_b += (count * (count - 1)) / 2;
  }

  for (const auto& key_count : histogram_a) {
    auto it = histogram_b.find(key_count.first);
    if (it == histogram_b.end()) continue;
    int64_t count_a = key_count.second;
    int64_t count_b = it->second;

    cross_collision_count += count_a * count_b;
  }

  double test_value =
      self_collision_count_a + self_collision_count_b -
      ((samples_a.size() - 1.0) / samples_a.size()) * cross_collision_count;
  double threshold = (l2_tolerance * (samples_a.size() - 1)) *
                     (l2_tolerance * samples_a.size()) / 4.0;
  return test_value < threshold;
}

}  // namespace

bool RunBallot(std::function<bool()> vote_generator, int number_of_votes) {
  return RunBallot([vote_generator = std::move(vote_generator)]()
                       -> absl::StatusOr<bool> { return vote_generator(); },
                   number_of_votes)
      .value_or(false);
}

absl::StatusOr<bool> RunBallot(
    std::function<absl::StatusOr<bool>()> vote_generator, int number_of_votes) {
  DCHECK(number_of_votes > 0) << "The number of votes must be positive";
  int accept_votes = 0;
  int reject_votes = 0;
  while (std::max(accept_votes, reject_votes) <= number_of_votes / 2) {
    ASSIGN_OR_RETURN(const bool result, vote_generator());
    (result ? accept_votes : reject_votes)++;
  }
  return accept_votes > reject_votes;
}

bool GenerateClosenessVote(std::function<double()> sample_generator_a,
                           std::function<double()> sample_generator_b,
                           int number_of_samples, double l2_tolerance,
                           double granularity) {
  std::vector<double> samples_a(number_of_samples);
  std::vector<double> samples_b(number_of_samples);
  for (int i = 0; i < number_of_samples; i++) {
    samples_a[i] =
        RoundToNearestDoubleMultiple(sample_generator_a(), granularity);
    samples_b[i] =
        RoundToNearestDoubleMultiple(sample_generator_b(), granularity);
  }
  return VerifyCloseness(samples_a, samples_b, l2_tolerance);
}

bool GenerateApproximateDpVote(std::function<double()> sample_generator_a,
                               std::function<double()> sample_generator_b,
                               int number_of_samples, double epsilon,
                               double delta, double delta_tolerance,
                               double granularity) {
  std::vector<double> samples_a(number_of_samples);
  std::vector<double> samples_b(number_of_samples);
  for (int i = 0; i < number_of_samples; ++i) {
    samples_a[i] = RoundToNearestMultiple(sample_generator_a(), granularity);
    samples_b[i] = RoundToNearestMultiple(sample_generator_b(), granularity);
  }
  return VerifyApproximateDp(samples_a, samples_b, epsilon, delta,
                             delta_tolerance);
}

int Bucketize(double sample, double lower, double upper, int num_buckets) {
  return std::max(
      0, std::min(num_buckets - 1,
                  static_cast<int>(floor(((sample - lower) / (upper - lower)) *
                                         num_buckets))));
}

}  // namespace differential_privacy::testing
