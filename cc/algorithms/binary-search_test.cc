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

#include "algorithms/binary-search.h"

#include <climits>
#include <memory>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "algorithms/algorithm.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/util.h"

namespace differential_privacy {
namespace {

static constexpr size_t kDataSize = 10000;
static constexpr size_t kStatsSize = 500;

using ::differential_privacy::test_utils::ZeroNoiseMechanism;

template <typename T>
class TestPercentileSearch : public BinarySearch<T> {
 public:
  TestPercentileSearch(
      double quantile, double epsilon, T lower, T upper,
      int64_t max_partitions_contributed,
      int64_t max_contributions_per_partition,
      std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder)
      : BinarySearch<T>(epsilon, lower, upper, max_partitions_contributed,
                        max_contributions_per_partition, quantile,
                        std::move(mechanism_builder),
                        absl::make_unique<base::Percentile<T>>()
        ) {}
};

TEST(BinarySearchTest, MedianTest) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search.PartialResult().value()), 100);
}

TEST(BinarySearchTest, PercentileTest) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      .6, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search.PartialResult().value()), 120);
}

TEST(BinarySearchTest, RepeatedResultTest) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search1(
      .5, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  TestPercentileSearch<int64_t> search2(
      .5, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (int64_t i = 0; i < kDataSize; ++i) {
    search1.AddEntry(std::round(200 * i / kDataSize));
    search2.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search1.PartialResult().value()),
            GetValue<int64_t>(search2.PartialResult().value()));
}

TEST(BinarySearchTest, MinTest) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      0, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_NEAR(GetValue<int64_t>(search.PartialResult().value()), 0, 10);
}

TEST(BinarySearchTest, MaxTest) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      1, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_NEAR(GetValue<int64_t>(search.PartialResult().value()), 200, 10);
}

TEST(BinarySearchTest, SerializeMergeTest) {
  // Serialize into a summary.
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (int64_t i = 0; i < 100; ++i) {
    search.AddEntry(100);
    search.AddEntry(200);
  }
  Summary summary = search.Serialize();

  BinarySearchSummary bs_summary;
  EXPECT_TRUE(summary.has_data());
  EXPECT_TRUE(summary.data().UnpackTo(&bs_summary));

  // Merge the summary back.
  TestPercentileSearch<int64_t> search_2(
      .5, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (int64_t i = 0; i < 100; ++i) {
    search_2.AddEntry(300);
  }

  EXPECT_OK(search_2.Merge(summary));
  EXPECT_EQ(GetValue<int64_t>(search_2.PartialResult().value()), 200);
}

TEST(BinarySearchTest, DropNanEntries) {
  double epsilon = 1;
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<double> search(
      .5, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
    search.AddEntry(NAN);
  }
  EXPECT_NEAR(GetValue<double>(search.PartialResult().value()), 100, .001);
}

// Binary search for the minimum with extreme bounds is extremely inaccurate.
TEST(BinarySearchTest, ExtremeBoundsMedianSearch) {
  double epsilon = std::log(3);
  int64_t lower = std::numeric_limits<int64_t>::lowest();
  int64_t upper = std::numeric_limits<int64_t>::max();
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search.PartialResult().value()), 100);
}

TEST(BinarySearchTest, ErrorConfidenceInterval) {
  double epsilon = std::log(3);
  double lower = 0, upper = 1000;
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (int64_t i = 0; i < kDataSize; ++i) {
    search.AddEntry(100);
  }
  Output output = search.PartialResult().value();
  ConfidenceInterval interval = GetNoiseConfidenceInterval(output);
  EXPECT_EQ(interval.confidence_level(), kDefaultConfidenceLevel);
  EXPECT_NEAR(interval.upper_bound(), 0, std::pow(10, -6));
  EXPECT_NEAR(interval.lower_bound(), 0, std::pow(10, -6));
}

TEST(BinarySearchTest, MemoryUsed) {
  TestPercentileSearch<double> search(
      .5, std::log(3), 1, 2, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  EXPECT_GT(search.MemoryUsed(), 0);
}

TEST(BinarySearchTest, LowerEqualsUpper) {
  TestPercentileSearch<int64_t> search(
      .5, std::log(3), 1, 1, 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  Output output = search.PartialResult().value();
  ConfidenceInterval interval = GetNoiseConfidenceInterval(output);
  EXPECT_EQ(GetValue<int64_t>(output), 1);
  EXPECT_EQ(interval.lower_bound(), 1);
  EXPECT_EQ(interval.upper_bound(), 1);
}

}  // namespace
}  // namespace differential_privacy
