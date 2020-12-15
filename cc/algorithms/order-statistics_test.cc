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

#include "algorithms/order-statistics.h"

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/util.h"

namespace differential_privacy {
namespace continuous {
namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::testing::HasSubstr;
using ::differential_privacy::base::testing::StatusIs;

static constexpr size_t kDataSize = 10000;
static constexpr double kNumSamples = 10000;

TEST(OrderStatisticsTest, Max) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 2048;
  base::StatusOr<std::unique_ptr<Max<int64_t>>> search =
      Max<int64_t>::Builder()
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(search);
  for (int64_t i = 0; i < kDataSize; ++i) {
    (*search)->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  EXPECT_NEAR(GetValue<int64_t>((*search)->PartialResult(1.0).ValueOrDie()), 200,
              10);
}

TEST(OrderStatisticsTest, Min) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 2048;
  base::StatusOr<std::unique_ptr<Min<int64_t>>> search =
      Min<int64_t>::Builder()
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  for (int64_t i = 0; i < kDataSize; ++i) {
    (*search)->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  base::StatusOr<Output> result = (*search)->PartialResult(1.0);
  ASSERT_OK(result);
  EXPECT_NEAR(GetValue<int64_t>(*result), 0, 10);
}

TEST(OrderStatisticsTest, Median) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 2048;
  base::StatusOr<std::unique_ptr<Median<int64_t>>> search =
      Median<int64_t>::Builder()
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(search);
  for (int64_t i = 0; i < kDataSize; ++i) {
    (*search)->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  base::StatusOr<Output> result = (*search)->PartialResult(1.0);
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<int64_t>(*result), 100);
}

TEST(OrderStatisticsTest, MedianLinfIncreasesVariance) {
  // Median is 0
  const std::vector<double> input = {1, 0, 0, -1};

  std::function<double(int)> sample_variance_for_max_contributions =
      [&input](int max_contributions) {
        double sum = 0;
        for (int i = 0; i < kNumSamples; ++i) {
          base::StatusOr<std::unique_ptr<Median<double>>> bounded_sum =
              Median<double>::Builder()
                  .SetMaxContributionsPerPartition(max_contributions)
                  .SetEpsilon(1)
                  .SetLower(-1)
                  .SetUpper(1)
                  .Build();
          CHECK_EQ(bounded_sum.status(), absl::OkStatus());
          base::StatusOr<Output> out =
              (*bounded_sum)->Result(input.begin(), input.end());
          CHECK_EQ(out.status(), absl::OkStatus());
          sum += std::pow(GetValue<double>(*out), 2);
        }
        return sum / (kNumSamples - 1);
      };

  // We expect the sample variance with max contribution 3 to be
  // bigger than with max contribution 1.
  EXPECT_GT(sample_variance_for_max_contributions(3),
            1.1 * sample_variance_for_max_contributions(1));
}

TEST(OrderStatisticsTest, Percentile) {
  double epsilon = std::log(3);
  int64_t lower = 0, upper = 2048;
  base::StatusOr<std::unique_ptr<Percentile<int64_t>>> search =
      Percentile<int64_t>::Builder()
          .SetPercentile(.45)
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(search);
  for (int64_t i = 0; i < kDataSize; ++i) {
    (*search)->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  base::StatusOr<Output> result = (*search)->PartialResult(1.0);
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<int64_t>(*result), 90);
}

TEST(OrderStatisticsTest, PercentileGetter) {
  double epsilon = std::log(3), expectedPercentile = 0.9;
  int64_t lower = 0, upper = 2048;
  base::StatusOr<std::unique_ptr<Percentile<int64_t>>> percentile =
      Percentile<int64_t>::Builder()
          .SetPercentile(expectedPercentile)
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .Build();
  ASSERT_OK(percentile);
  EXPECT_EQ((*percentile)->GetPercentile(), expectedPercentile);
}

TEST(OrderStatisticsTest, InvalidParameters) {
  Percentile<int64_t>::Builder builder;
  EXPECT_OK(builder.SetPercentile(.9).SetLower(1).SetUpper(2).Build());
  EXPECT_THAT(
      builder.SetLower(3).Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Lower bound cannot be greater than upper bound")));
  EXPECT_THAT(
      builder.SetLower(1).SetPercentile(-1).Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Percentile must be in the inclusive interval [0,1]")));
  EXPECT_THAT(
      builder.SetPercentile(2).Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Percentile must be in the inclusive interval [0,1]")));
}

TEST(OrderStatisticsTest, Median_DefaultBounds) {
  double epsilon = std::log(3);
  base::StatusOr<std::unique_ptr<Median<int64_t>>> search =
      Median<int64_t>::Builder()
          .SetEpsilon(epsilon)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(search);
  for (int64_t i = 0; i < kDataSize; ++i) {
    (*search)->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  base::StatusOr<Output> result = (*search)->PartialResult(1.0);
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<int64_t>(*result), 100);
}

}  // namespace
}  // namespace continuous
}  // namespace differential_privacy
