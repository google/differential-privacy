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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/util.h"

namespace differential_privacy {
namespace continuous {
namespace {

using test_utils::ZeroNoiseMechanism;

static constexpr size_t kDataSize = 10000;
static constexpr size_t kStatsSize = 500;
static constexpr double kNumSamples = 10000;

TEST(OrderStatisticsTest, Max) {
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 2048;
  std::unique_ptr<Max<int64_t>> search =
      typename Max<int64_t>::Builder()
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  for (int64_t i = 0; i < kDataSize; ++i) {
    search->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  EXPECT_NEAR(GetValue<int64_t>(search->PartialResult(1.0).ValueOrDie()), 200,
              10);
}

TEST(OrderStatisticsTest, Min) {
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 2048;
  std::unique_ptr<Min<int64_t>> search =
      typename Min<int64_t>::Builder()
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  for (int64_t i = 0; i < kDataSize; ++i) {
    search->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  EXPECT_NEAR(GetValue<int64_t>(search->PartialResult(1.0).ValueOrDie()), 0, 10);
}

TEST(OrderStatisticsTest, Median) {
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 2048;
  std::unique_ptr<Median<int64_t>> search =
      typename Median<int64_t>::Builder()
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  for (int64_t i = 0; i < kDataSize; ++i) {
    search->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search->PartialResult(1.0).ValueOrDie()), 100);
}

TEST(OrderStatisticsTest, MedianLinfIncreasesVariance) {
  // Median is 0
  const std::vector<double> input = {1, 0, 0, -1};

  std::function<double(int)> sample_variance_for_max_contributions =
      [&input](int max_contributions) {
        double sum = 0;
        for (int i = 0; i < kNumSamples; ++i) {
          auto bounded_sum =
              typename Median<double>::Builder()
                  .SetMaxContributionsPerPartition(max_contributions)
                  .SetEpsilon(1)
                  .SetLower(-1)
                  .SetUpper(1)
                  .Build();
          CHECK_EQ(bounded_sum.status(), base::OkStatus());
          auto out = (*bounded_sum)->Result(input.begin(), input.end());
          CHECK_EQ(out.status(), base::OkStatus());
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
  double epsilon = DefaultEpsilon();
  int64_t lower = 0, upper = 2048;
  std::unique_ptr<Percentile<int64_t>> search =
      typename Percentile<int64_t>::Builder()
          .SetPercentile(.45)
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  for (int64_t i = 0; i < kDataSize; ++i) {
    search->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search->PartialResult(1.0).ValueOrDie()), 90);
}

TEST(OrderStatisticsTest, PercentileGetter) {
  double epsilon = DefaultEpsilon(), expectedPercentile = 0.9;
  int64_t lower = 0, upper = 2048;
  std::unique_ptr<Percentile<int64_t>> percentile =
      typename Percentile<int64_t>::Builder()
          .SetPercentile(expectedPercentile)
          .SetEpsilon(epsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .Build()
          .ValueOrDie();
  EXPECT_EQ(percentile->percentile(), expectedPercentile);
}

TEST(OrderStatisticsTest, InvalidParameters) {
  Percentile<int64_t>::Builder builder;
  EXPECT_TRUE(builder.SetPercentile(.9)
                  .SetLower(1)
                  .SetUpper(2)
                  .Build()
                  .ok());
  EXPECT_FALSE(builder.SetLower(3).Build().ok());
  EXPECT_FALSE(builder.SetLower(1).SetPercentile(-1).Build().ok());
  EXPECT_FALSE(builder.SetPercentile(2).Build().ok());
}

TEST(OrderStatisticsTest, Median_DefaultBounds) {
  double epsilon = DefaultEpsilon();
  std::unique_ptr<Median<int64_t>> search =
      typename Median<int64_t>::Builder()
          .SetEpsilon(epsilon)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  for (int64_t i = 0; i < kDataSize; ++i) {
    search->AddEntry(std::round(static_cast<double>(200) * i / kDataSize));
  }
  EXPECT_EQ(GetValue<int64_t>(search->PartialResult(1.0).ValueOrDie()), 100);
}

}  // namespace
}  // namespace continuous
}  // namespace differential_privacy
