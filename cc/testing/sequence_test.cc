//
// Copyright 2019 Google LLC
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

#include "testing/sequence.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "algorithms/util.h"

namespace differential_privacy {
namespace testing {
namespace {

const int64_t kDimensions = 10;
constexpr int64_t kNumSamples = 5000;
constexpr double kUnitUniformMean = 0.5;
constexpr double kUnitUniformVariance = 1.0 / 12;

std::vector<std::vector<double>> GenerateSamplesFromSequence(
    Sequence<double>* sequence, int64_t num_samples) {
  std::vector<std::vector<double>> samples(num_samples);
  std::generate(samples.begin(), samples.end(),
                [&sequence] { return sequence->GetSample(); });
  return samples;
}

TEST(StoredSequenceTest, CheckStoredSequenceReturnsExpectedOutput) {
  std::vector<std::vector<double>> stored_sequence(
      {{1.0}, {1.0, 2.0}, {1.0, 2.0, 3.0}});
  StoredSequence<double> sequence(stored_sequence);
  // Using 2 * the test vector's size + 1 for the number of samples allows us to
  // exercise the repeating behavior while also ending out of period.
  std::vector<std::vector<double>> samples(
      GenerateSamplesFromSequence(&sequence, 2 * stored_sequence.size() + 1));
  std::vector<std::vector<double>> expected_samples({{1.0},
                                                     {1.0, 2.0},
                                                     {1.0, 2.0, 3.0},
                                                     {1.0},
                                                     {1.0, 2.0},
                                                     {1.0, 2.0, 3.0},
                                                     {1.0}});
  EXPECT_EQ(samples, expected_samples);
}

TEST(StoredSequenceTest, NextNDimensions) {
  StoredSequence<double> sequence({{1.0}, {1.0, 2.0}, {1.0, 2.0, 3.0}});
  const std::vector<int64_t> expected = {1, 2, 3, 1};
  const std::vector<int64_t> dimensions =
      sequence.NextNDimensions(expected.size());
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(dimensions[i], expected[i]);
  }
}

void CheckUniformStatistics(const std::vector<std::vector<double>>& samples) {
  for (int i = 0; i < kDimensions; ++i) {
    // Generate vector of values on the k^th dimension.
    std::vector<double> column(samples.size());
    std::transform(samples.begin(), samples.end(), column.begin(),
                   [i](const std::vector<double>& v) { return v[i]; });

    EXPECT_NEAR(Mean(column), kUnitUniformMean, 0.01);
    EXPECT_NEAR(Variance(column), kUnitUniformVariance, 0.01);
    EXPECT_GE(*std::min_element(column.begin(), column.end()), 0.0);
    EXPECT_LE(*std::max_element(column.begin(), column.end()), 1.0);
  }
}

TEST(HaltonSequenceTest, CheckHaltonSequenceStatisticsForUnitValues) {
  HaltonSequence<double> sequence(kDimensions);
  std::vector<std::vector<double>> samples(
      GenerateSamplesFromSequence(&sequence, kNumSamples));
  CheckUniformStatistics(samples);
}

// Check whether the samples have a correlation within 0.05 of 0 between each
// pair of dimensions.
void CheckLowCorrelation(const std::vector<std::vector<double>>& samples) {
  for (int i = 0; i < kDimensions; ++i) {
    std::vector<double> column_first(samples.size());
    std::transform(samples.begin(), samples.end(), column_first.begin(),
                   [i](const std::vector<double>& v) { return v[i]; });
    for (int j = i + 1; j < kDimensions; ++j) {
      std::vector<double> column_second(samples.size());
      std::transform(samples.begin(), samples.end(), column_second.begin(),
                     [j](const std::vector<double>& v) { return v[j]; });
      EXPECT_NEAR(0.0, Correlation(column_first, column_second), 0.05);
    }
  }
}

TEST(HaltonSequenceTest, CheckHaltonSequenceForLowCorrelation) {
  HaltonSequence<double> sequence(kDimensions);
  std::vector<std::vector<double>> samples(
      GenerateSamplesFromSequence(&sequence, kNumSamples));
  CheckLowCorrelation(samples);
}

TEST(HaltonSequenceTest, NextNDimensions) {
  HaltonSequence<double> sequence(kDimensions);
  const std::vector<int64_t> dimensions = sequence.NextNDimensions(4);
  for (int i = 0; i < dimensions.size(); ++i) {
    EXPECT_EQ(dimensions[i], kDimensions);
  }
}

TEST(HaltonTest, Base2) {
  Halton h(2);
  std::vector<double> seq = {0,         1.0 / 2.0, 1.0 / 4.0, 3.0 / 4.0,
                             1.0 / 8.0, 5.0 / 8.0, 3.0 / 8.0, 7.0 / 8.0};
  for (int i = 1; i < seq.size(); ++i) {
    EXPECT_EQ(h.Get(i), seq[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace differential_privacy
