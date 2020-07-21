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

namespace differential_privacy {
namespace {

static constexpr size_t kDataSize = 10000;
static constexpr size_t kStatsSize = 500;

using ::differential_privacy::test_utils::ZeroNoiseMechanism;

template <typename T>
class TestPercentileSearch : public BinarySearch<T> {
 public:
  TestPercentileSearch(double percentile, double epsilon, T lower, T upper,
                       std::unique_ptr<LaplaceMechanism::Builder> builder)
      : BinarySearch<T>(
            epsilon, lower, upper, percentile,
            absl::WrapUnique<LaplaceMechanism>(dynamic_cast<LaplaceMechanism*>(
                builder->Build().ValueOrDie().release())),
            absl::make_unique<base::Percentile<T>>()
        ) {}
};

TEST(BinarySearchTest, MedianTest) {
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search.PartialResult(1.0).ValueOrDie()), 100);
}

TEST(BinarySearchTest, PercentileTest) {
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      .6, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search.PartialResult(1.0).ValueOrDie()), 120);
}

TEST(BinarySearchTest, RepeatedResultTest) {
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (int64_t i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search.PartialResult(0.5).ValueOrDie()),
            GetValue<int64_t>(search.PartialResult(0.5).ValueOrDie()));
}

TEST(BinarySearchTest, MinTest) {
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      0, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_NEAR(GetValue<int64_t>(search.PartialResult(1.0).ValueOrDie()), 0, 10);
}

TEST(BinarySearchTest, MaxTest) {
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      1, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_NEAR(GetValue<int64_t>(search.PartialResult(1.0).ValueOrDie()), 200, 10);
}

TEST(BinarySearchTest, SerializeMergeTest) {
  // Serialize into a summary.
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper,
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
      .5, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (int64_t i = 0; i < 100; ++i) {
    search_2.AddEntry(300);
  }

  EXPECT_OK(search_2.Merge(summary));
  EXPECT_EQ(GetValue<int64_t>(search_2.PartialResult(1.0).ValueOrDie()), 200);
}

TEST(BinarySearchTest, DropNanEntries) {
  double epsilon = 1;
  int64_t lower = 0, upper = 400;
  TestPercentileSearch<double> search(
      .5, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
    search.AddEntry(NAN);
  }
  EXPECT_NEAR(GetValue<double>(search.PartialResult(1.0).ValueOrDie()), 100,
              .001);
}

// Binary search for the minimum with extreme bounds is extremely inaccurate.
TEST(BinarySearchTest, ExtremeBoundsMedianSearch) {
  double epsilon = DefaultEpsilon();
  int64_t lower = std::numeric_limits<int64_t>::lowest();
  int64_t upper = std::numeric_limits<int64_t>::max();
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (double i = 0; i < kDataSize; ++i) {
    search.AddEntry(std::round(200 * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search.PartialResult(1.0).ValueOrDie()), 100);
}

TEST(BinarySearchTest, ErrorConfidenceInterval) {
  double epsilon = DefaultEpsilon();
  double lower = 0, upper = 1000;
  TestPercentileSearch<int64_t> search(
      .5, epsilon, lower, upper,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  for (int64_t i = 0; i < kDataSize; ++i) {
    search.AddEntry(100);
  }
  Output output = search.PartialResult().ValueOrDie();
  ConfidenceInterval interval =
      output.error_report().noise_confidence_interval();
  EXPECT_EQ(interval.confidence_level(), kDefaultConfidenceLevel);
  EXPECT_NEAR(interval.upper_bound(), 0, std::pow(10, -6));
  EXPECT_NEAR(interval.lower_bound(), 0, std::pow(10, -6));
}

TEST(BinarySearchTest, MemoryUsed) {
  TestPercentileSearch<double> search(
      .5, DefaultEpsilon(), 1, 2,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  EXPECT_GT(search.MemoryUsed(), 0);
}

TEST(BinarySearchTest, LowerEqualsUpper) {
  TestPercentileSearch<int64_t> search(
      .5, DefaultEpsilon(), 1, 1,
      absl::make_unique<test_utils::ZeroNoiseMechanism::Builder>());
  Output output = search.PartialResult(1).ValueOrDie();
  EXPECT_EQ(GetValue<int64_t>(output), 1);
  EXPECT_EQ(output.error_report().noise_confidence_interval().lower_bound(), 1);
  EXPECT_EQ(output.error_report().noise_confidence_interval().upper_bound(), 1);
}

}  // namespace
}  // namespace differential_privacy
