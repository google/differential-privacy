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
//

#include "algorithms/partition-selection.h"

#include <cmath>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/partition-selection-testing.h"

namespace differential_privacy {
namespace {

using ::differential_privacy::test_utils::MockGaussianStrategy;
using ::differential_privacy::test_utils::MockLaplaceStrategy;
using ::differential_privacy::test_utils::MockNearTruncatedStrategy;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::HasSubstr;
using ::testing::TestWithParam;
using ::differential_privacy::base::testing::IsOkAndHolds;
using ::differential_privacy::base::testing::StatusIs;

constexpr int kNumSamples = 10000000;
constexpr int kSmallNumSamples = 1000000;
constexpr int kTinyNumSamples = 10000;

constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::min();
constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kNegInf = -std::numeric_limits<double>::infinity();
constexpr double kPosInf = std::numeric_limits<double>::infinity();
constexpr double kDoubleMin = std::numeric_limits<double>::lowest();
constexpr double kDoubleMinPos = std::numeric_limits<double>::min();
constexpr double kDoubleMax = std::numeric_limits<double>::max();

constexpr double kCalcDeltaTestDefaultTolerance = 0.001;
constexpr double kCalcThresholdTestDefaultTolerance = 0.05;

// NearTruncatedGeometricregationPartitionSelection Tests

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionUnsetEpsilon) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(test_builder.SetDelta(0.1).SetMaxPartitionsContributed(2).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be set")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionNanEpsilon) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(test_builder.SetEpsilon(NAN)
                  .SetDelta(0.3)
                  .SetMaxPartitionsContributed(4)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be a valid numeric value")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionNotFiniteEpsilon) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(test_builder.SetEpsilon(std::numeric_limits<double>::infinity())
                  .SetDelta(0.3)
                  .SetMaxPartitionsContributed(4)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be finite")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionNegativeEpsilon) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(test_builder.SetEpsilon(-5.0)
                  .SetDelta(0.6)
                  .SetMaxPartitionsContributed(7)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be finite and positive")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionUnsetDelta) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetEpsilon(8.0).SetMaxPartitionsContributed(9).Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be set.")));
}

TEST(PartitionSelectionTest, NearTruncatedGeometricPartitionSelectionNanDelta) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(test_builder.SetEpsilon(1.2)
                  .SetDelta(NAN)
                  .SetMaxPartitionsContributed(3)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Delta must be a valid numeric value")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionNotFiniteDelta) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(test_builder.SetEpsilon(1.2)
                  .SetDelta(std::numeric_limits<double>::infinity())
                  .SetMaxPartitionsContributed(3)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionInvalidDelta) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(test_builder.SetEpsilon(4.5)
                  .SetDelta(6.0)
                  .SetMaxPartitionsContributed(7)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionUnsetMaxPartitionsContributed) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetEpsilon(0.8).SetDelta(0.9).Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be set")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionNegativeMaxPartitionsContributed) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetEpsilon(0.1)
          .SetDelta(0.2)
          .SetMaxPartitionsContributed(-3)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be positive, but is -3.")));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionZeroMaxPartitionsContributed) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetEpsilon(0.1)
          .SetDelta(0.2)
          .SetMaxPartitionsContributed(0)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be positive, but is 0.")));
}

// We expect the probability of keeping a partition with one user
// will be approximately delta
TEST(PartitionSelectionTest, NearTruncatedGeometricPartitionSelectionOneUser) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ShouldKeep(1)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(1), strategy->GetDelta(), 1e-12);
  EXPECT_THAT(static_cast<double>(num_kept) / kSmallNumSamples,
              DoubleNear(strategy->GetDelta(), 0.001));
}

// We expect the probability of keeping a partition with no users will be zero
TEST(PartitionSelectionTest, NearTruncatedGeometricPartitionSelectionNoUsers) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  EXPECT_EQ(strategy->ProbabilityOfKeep(0), 0.0);
  for (int i = 0; i < 1000; i++) {
    EXPECT_FALSE(strategy->ShouldKeep(0));
  }
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionFirstCrossover) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  NearTruncatedGeometricPartitionSelection* magic =
      dynamic_cast<NearTruncatedGeometricPartitionSelection*>(strategy.get());
  EXPECT_THAT(magic->GetFirstCrossover(), DoubleEq(6));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionSecondCrossover) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  NearTruncatedGeometricPartitionSelection* magic =
      dynamic_cast<NearTruncatedGeometricPartitionSelection*>(strategy.get());
  EXPECT_THAT(magic->GetSecondCrossover(), DoubleEq(11));
}

// Values calculated with formula
TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionNumUsersEqFirstCrossover) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  EXPECT_NEAR(strategy->ProbabilityOfKeep(6), 0.58840484458, 1e-10);
  int num_kept = 0;
  for (int i = 0; i < kNumSamples; i++) {
    if (strategy->ShouldKeep(6)) num_kept++;
  }
  EXPECT_THAT(static_cast<double>(num_kept) / kNumSamples,
              DoubleNear(0.58840484458, 0.001));
}

// Values calculated with formula
TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionNumUsersBtwnCrossovers) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kNumSamples; i++) {
    if (strategy->ShouldKeep(8)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(8), 0.86807080625, 1e-10);
  EXPECT_THAT(static_cast<double>(num_kept) / kNumSamples,
              DoubleNear(0.86807080625, 0.001));
}

// Values calculated with formula - 15 should be so large that this partition is
// always kept.
TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionNumUsersGreaterThanCrossovers) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  EXPECT_EQ(strategy->ProbabilityOfKeep(15), 1);
  for (int i = 0; i < 1000; i++) {
    EXPECT_TRUE(strategy->ShouldKeep(15));
  }
}

// For tiny epsilon probability of keeping is basically n * delta.
TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionTinyEpsilon) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(1e-20).SetDelta(0.02).SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kNumSamples; i++) {
    if (strategy->ShouldKeep(6)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(6), 0.12, 1e-10);
  EXPECT_THAT(static_cast<double>(num_kept) / kNumSamples,
              DoubleNear(0.12, 0.001));
}

TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionTinyEpsilonLargeDelta) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(1e-20).SetDelta(0.15).SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kNumSamples; i++) {
    if (strategy->ShouldKeep(3)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(3), 0.45, 1e-10);
  EXPECT_THAT(static_cast<double>(num_kept) / kNumSamples,
              DoubleNear(0.45, 0.001));
}

// For tiny epsilon probability of keeping is basically n * delta.
TEST(PartitionSelectionTest,
     NearTruncatedGeometricPartitionSelectionTinyEpsilonBtwnCrossovers) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(1e-20).SetDelta(0.02).SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kNumSamples; i++) {
    if (strategy->ShouldKeep(40)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(40), 0.8, 1e-10);
  EXPECT_THAT(static_cast<double>(num_kept) / kNumSamples,
              DoubleNear(0.8, 0.001));
}
// LaplacePartitionSelection Tests
// Due to the inheritance, SetLaplaceMechanism must be
// called before SetDelta, SetEpsilon, etc.

TEST(PartitionSelectionTest,
     LaplacePartitionSelectionUnsetMaxPartitionsContributed) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetDelta(0.1)
          .SetEpsilon(2)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be set")));
}

TEST(PartitionSelectionTest,
     LaplacePartitionSelectionNegativeMaxPartitionsContributed) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetDelta(0.1)
          .SetEpsilon(2)
          .SetMaxPartitionsContributed(-3)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be positive, but is -3.")));
}

TEST(PartitionSelectionTest,
     LaplacePartitionSelectionZeroMaxPartitionsContributed) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetDelta(0.1)
          .SetEpsilon(2)
          .SetMaxPartitionsContributed(0)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be positive, but is 0.")));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionUnsetEpsilon) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetDelta(0.1)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Epsilon must be set.")));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionUnsetDelta) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be set")));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionNanDelta) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetDelta(NAN)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be a valid numeric value")));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionNotFiniteDelta) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetDelta(std::numeric_limits<double>::infinity())
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionInvalidPositiveDelta) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetDelta(5.2)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionInvalidNegativeDelta) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetDelta(-0.1)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval")));
}

// We expect the probability of keeping a partition with one user
// will be approximately delta
TEST(PartitionSelectionTest, LaplacePartitionSelectionOneUser) {
  LaplacePartitionSelection::Builder test_builder;
  test_builder
      .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
      .SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ShouldKeep(1)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(1), strategy->GetDelta(), 1e-12);
  EXPECT_THAT(static_cast<double>(num_kept) / kSmallNumSamples,
              DoubleNear(strategy->GetDelta(), 0.0006));
}

// When the number of users is at the threshold, we expect drop/keep is 50/50.
// These numbers should make the threshold approximately 5.
TEST(PartitionSelectionTest, LaplacePartitionSelectionAtThreshold) {
  LaplacePartitionSelection::Builder test_builder;
  test_builder
      .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
      .SetEpsilon(0.5)
      .SetDelta(0.06766764161)
      .SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ShouldKeep(5)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(5), 0.5, 1e-10);
  EXPECT_THAT(static_cast<double>(num_kept) / kSmallNumSamples,
              DoubleNear(0.5, 0.0025));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionThreshold) {
  LaplacePartitionSelection::Builder test_builder;
  test_builder
      .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
      .SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  LaplacePartitionSelection* laplace =
      dynamic_cast<LaplacePartitionSelection*>(strategy.get());
  EXPECT_THAT(laplace->GetThreshold(), DoubleNear(7.43775164974, 0.001));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionUnsetBuilderThreshold) {
  LaplacePartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  LaplacePartitionSelection* laplace =
      dynamic_cast<LaplacePartitionSelection*>(strategy.get());
  ASSERT_THAT(laplace, testing::NotNull());
  EXPECT_THAT(laplace->GetThreshold(), DoubleNear(7.43775164974, 0.001));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionNoiseValueIfShouldKeep) {
  LaplacePartitionSelection::Builder test_builder;
  test_builder
      .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
      .SetEpsilon(0.5)
      .SetDelta(0.06766764161)
      .SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  auto* laplace_ps = dynamic_cast<LaplacePartitionSelection*>(strategy.get());
  for (int i = 0; i < kTinyNumSamples; ++i) {
    auto noised_value = laplace_ps->NoiseValueIfShouldKeep(5);
    if (noised_value.has_value()) {
      ++num_kept;
      EXPECT_GE(noised_value.value(), laplace_ps->GetThreshold());
    }
  }

  EXPECT_THAT(static_cast<double>(num_kept) / kTinyNumSamples,
              DoubleNear(0.5, 0.02));
}

// CalculateDelta and CalculateThreshold structs and tests

TEST(PartitionSelectionTest,
     LaplacePartitionSelectionCalculateDeltaThresholdSymmetryAround1) {
  double epsilon = std::log(3);
  int64_t max_partitions_contributed = 1;
  double delta_1_minus_i, delta_1_plus_i;
  for (double i = 0.1; i < 5; i += 0.1) {
    delta_1_minus_i = LaplacePartitionSelection::CalculateDelta(
                          epsilon, 1 - i, max_partitions_contributed)
                          .value();
    delta_1_plus_i = LaplacePartitionSelection::CalculateDelta(
                         epsilon, 1 + i, max_partitions_contributed)
                         .value();
    EXPECT_THAT(delta_1_minus_i, DoubleNear(1 - delta_1_plus_i, 0.0001));
  }
}

struct CalculateDeltaTest {
  CalculateDeltaTest(double epsilon_in, double threshold_in,
                     int64_t max_partitions_contributed_in,
                     std::optional<double> expected_delta_in,
                     double tolerance_in = kCalcDeltaTestDefaultTolerance)
      : epsilon(epsilon_in),
        threshold(threshold_in),
        max_partitions_contributed(max_partitions_contributed_in),
        expected_delta(expected_delta_in),
        tolerance(tolerance_in) {}

  const double epsilon;
  const double threshold;
  const int64_t max_partitions_contributed;
  // Missing implies an error is expected to be returned.
  const std::optional<double> expected_delta;
  const double tolerance;
};

static CalculateDeltaTest DeltaTest(
    double epsilon, double threshold, int64_t max_partitions_contributed,
    double expected_delta,
    double tolerance = kCalcThresholdTestDefaultTolerance) {
  return CalculateDeltaTest(epsilon, threshold, max_partitions_contributed,
                            expected_delta, tolerance);
}

static CalculateDeltaTest DeltaErrorTest(
    double epsilon, double threshold, int64_t max_partitions_contributed,
    double tolerance = kCalcThresholdTestDefaultTolerance) {
  return CalculateDeltaTest(epsilon, threshold, max_partitions_contributed,
                            std::optional<double>(), tolerance);
}

TEST(PartitionSelectionTest, CalculateDeltaTests) {
  std::vector<CalculateDeltaTest> delta_test_cases = {
    // In all tests, "max_pc" is shorthand for "max_partitions_contributed".
    //
    // Fix epsilon = ln(3) & max_pc = 1, and vary threshold.
    //
    //                                      expected       test
    //         epsilon   threshold  max_pc  delta          tolerance
    //        --------  ----------  ------  -------------  ---------
    DeltaTest(std::log(3.0),          1,      1, 0.5),
    DeltaTest(std::log(3.0),          2,      1, 0.16666667),
    DeltaTest(std::log(3.0),          3,      1, 0.05555555556, 1e-05),
    DeltaTest(std::log(3.0),          4,      1, 0.01851851852, 1e-05),
    DeltaTest(std::log(3.0),          5,      1, 0.00617283960, 1e-06),
    DeltaTest(std::log(3.0),         10,      1, 2.5402631e-05, 1e-08),
    DeltaTest(std::log(3.0),         20,      1, 4.3019580e-10, 1e-13),
    DeltaTest(std::log(3.0),         50,      1, 2.0894334e-24, 1e-27),
    DeltaTest(std::log(3.0),         75,      1, 2.4660232e-36, 1e-39),
    DeltaTest(std::log(3.0),         87,      1, 4.6402600e-42, 6e-46),
    DeltaTest(std::log(3.0),         93,      1, 6.3652400e-45, 1e-48),
    DeltaTest(std::log(3.0),         94,      1, 2.1217500e-45, 1e-48),
    DeltaTest(std::log(3.0),         95,      1, 7.0724900e-46, 1e-49),
    DeltaTest(std::log(3.0),         96,      1, 2.3575000e-46, 1e-49),
    DeltaTest(std::log(3.0),        100,      1, 2.9104900e-48, 1e-51),
    DeltaTest(std::log(3.0),       1000,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),      10000,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),     100000,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),    1000000,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0), kDoubleMax,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),    kPosInf,      1, 0.0,           1e-100),

    // Fix threshold = 50 & max_pc = 1, and vary epsilon.
    //
    //                                          expected       test
    //              epsilon  threshold  max_pc  delta          tolerance
    //        -------------  ---------  ------  -------------  ---------
    DeltaTest(kDoubleMinPos,        50,      1, 0.5),
    DeltaTest(       1e-308,        50,      1, 0.5),
    DeltaTest(       1e-100,        50,      1, 0.5),
    DeltaTest(        1e-50,        50,      1, 0.5),
    DeltaTest(        1e-20,        50,      1, 0.5),
    DeltaTest(        1e-10,        50,      1, 0.5),
    DeltaTest(         1e-5,        50,      1, 0.49975505),
    DeltaTest(         1e-2,        50,      1, 0.30631319),
    DeltaTest(         1e-1,        50,      1, 0.0037232914,  1e-06),
    DeltaTest(          0.5,        50,      1, 1.1448674e-11, 1e-14),
    DeltaTest(          1.0,        50,      1, 2.6214428e-22, 1e-25),
    DeltaTest(std::log(3.0),        50,      1, 2.0894334e-24, 1e-27),
    DeltaTest(          1.5,        50,      1, 6.0024092e-33, 1e-36),
    DeltaTest(          2.0,        50,      1, 1.3732725e-43, 1.2e-46),
    DeltaTest(          5.0,        50,      1, 0.0,           1e-100),
    DeltaTest(          1e1,        50,      1, 0.0,           1e-100),
    DeltaTest(          1e2,        50,      1, 0.0,           1e-100),
    DeltaTest(          1e5,        50,      1, 0.0,           1e-100),
    DeltaTest(         1e10,        50,      1, 0.0,           1e-100),
    DeltaTest(         1e20,        50,      1, 0.0,           1e-100),
    DeltaTest(         1e50,        50,      1, 0.0,           1e-100),
    DeltaTest(        1e100,        50,      1, 0.0,           1e-100),
    DeltaTest(        1e308,        50,      1, 0.0,           1e-100),
    DeltaTest(   kDoubleMax,        50,      1, 0.0,           1e-100),

    // Fix epsilon & threshold and vary max_pc.
    //                                        expected       test
    //         epsilon  threshold     max_pc  delta          tolerance
    //        --------  ---------  ---------  -------------  ---------
    DeltaTest(std::log(3.0),        50,         1, 2.0894334e-24, 1e-27),
    DeltaTest(std::log(3.0),        50,         2, 2.0442300e-12, 1e-15),
    DeltaTest(std::log(3.0),        50,         3, 2.4160800e-08, 1e-11),
    DeltaTest(std::log(3.0),        50,         4, 2.8595300e-06, 1e-09),
    DeltaTest(std::log(3.0),        50,         5, 5.2740300e-05, 1e-08),
    DeltaTest(std::log(3.0),        50,        10, 0.0227296,     1e-05),
    DeltaTest(std::log(3.0),        50,       100, 1.0),
    DeltaTest(std::log(3.0),        50,      1000, 1.0),
    DeltaTest(std::log(3.0),        50,     10000, 1.0),
    DeltaTest(std::log(3.0),        50,    100000, 1.0),
    DeltaTest(std::log(3.0),        50,   1000000, 1.0),
    DeltaTest(std::log(3.0),        50, kInt64Max, 1.0),

    // Error cases.
    //
    // Epsilon must be finite and greater than 0.
    //
    //                epsilon  threshold  max_pc
    //             ----------  ---------  ------
    DeltaErrorTest(kDoubleMin,        50, 1),
    DeltaErrorTest(      -1.0,        50, 1),
    DeltaErrorTest(       0.0,        50, 1),
    DeltaErrorTest(   kPosInf,        50, 1),
    DeltaErrorTest(   kNegInf,        50, 1),
    DeltaErrorTest(      kNaN,        50, 1),

    // Threshold must be finite.
    //
    //              epsilon  threshold  max_pc
    //             --------  ---------  ------
    DeltaErrorTest(std::log(3.0),      kNaN, 1),

    // Max_partitions_contributed (max_pc) must be greater than 0.
    //
    //              epsilon  threshold  max_pc
    //             --------  ---------  ---------
    DeltaErrorTest(std::log(3.0),        50, kInt64Min),
    DeltaErrorTest(std::log(3.0),        50, -1),
    DeltaErrorTest(std::log(3.0),        50, 0),
  };

  for (const CalculateDeltaTest& test_case : delta_test_cases) {
    const double epsilon = test_case.epsilon;
    const double threshold = test_case.threshold;
    const int64_t max_partitions_contributed =
        test_case.max_partitions_contributed;

    const std::string test_case_string = absl::StrCat(
        "epsilon: ", epsilon, ", threshold: ", threshold,
        ", max_partitions_contributed: ", max_partitions_contributed);

    absl::StatusOr<double> status_or_delta =
        LaplacePartitionSelection::CalculateDelta(epsilon, threshold,
                                                  max_partitions_contributed);

    if (test_case.expected_delta.has_value()) {
      // We expect the test to succeed, and a delta to be computed.
      ASSERT_TRUE(status_or_delta.ok()) << status_or_delta.status() << "\n"
                                        << test_case_string;
      const double delta = status_or_delta.value();
      EXPECT_THAT(delta, DoubleNear(test_case.expected_delta.value(),
                                    test_case.tolerance))
          << test_case_string;
    } else {
      EXPECT_FALSE(status_or_delta.ok())
          << test_case_string
          << "\nunexpected successfully computed delta value: "
          << status_or_delta.value();
    }
  }
}

struct CalculateThresholdTest {
  CalculateThresholdTest(
      double epsilon_in, double delta_in, int64_t max_partitions_contributed_in,
      std::optional<double> expected_threshold_in,
      double tolerance_in = kCalcThresholdTestDefaultTolerance)
      : epsilon(epsilon_in),
        delta(delta_in),
        max_partitions_contributed(max_partitions_contributed_in),
        expected_threshold(expected_threshold_in),
        tolerance(tolerance_in) {}

  const double epsilon;
  const double delta;
  const int64_t max_partitions_contributed;
  // Missing implies an error is expected to be returned.
  const std::optional<double> expected_threshold;
  const double tolerance;
};

static CalculateThresholdTest ThresholdTest(
    double epsilon, double delta, int64_t max_partitions_contributed,
    double expected_threshold,
    double tolerance = kCalcThresholdTestDefaultTolerance) {
  return CalculateThresholdTest(epsilon, delta, max_partitions_contributed,
                                expected_threshold, tolerance);
}

static CalculateThresholdTest ThresholdErrorTest(
    double epsilon, double delta, int64_t max_partitions_contributed,
    double tolerance = kCalcThresholdTestDefaultTolerance) {
  return CalculateThresholdTest(epsilon, delta, max_partitions_contributed,
                                std::optional<double>(), tolerance);
}

TEST(PartitionSelectionTest, CalculateThresholdTests) {
  std::vector<CalculateThresholdTest> threshold_test_cases = {
    // In all tests, "max_pc" is shorthand for "max_partitions_contributed".
    //
    // Fix epsilon = ln(3) & max_pc = 1, and vary delta.
    //
    //                                              expected  test
    //             epsilon          delta  max_pc  threshold  tolerance
    //            --------  -------------  ------  ---------  ---------
    ThresholdTest(std::log(3.0),           0.0,      1,   kPosInf),
    ThresholdTest(std::log(3.0), kDoubleMinPos,      1, 645.17900),
    ThresholdTest(std::log(3.0),        1e-308,      1, 645.90700),
    ThresholdTest(std::log(3.0),        1e-256,      1, 536.92000),
    ThresholdTest(std::log(3.0),        1e-128,      1, 268.64500),
    ThresholdTest(std::log(3.0),         1e-64,      1, 134.50700),
    ThresholdTest(std::log(3.0),         1e-32,      1,  67.43800),
    ThresholdTest(std::log(3.0), 2.0894334e-24,      1,  50.00000),
    ThresholdTest(std::log(3.0),         1e-16,      1,  33.90350),
    ThresholdTest(std::log(3.0),          1e-8,      1,  17.13630),
    ThresholdTest(std::log(3.0),          1e-4,      1,   8.75268),
    ThresholdTest(std::log(3.0),          1e-2,      1,   4.56088),
    ThresholdTest(std::log(3.0),           0.1,      1,   2.46497),
    ThresholdTest(std::log(3.0),           0.3,      1,   1.46497),
    ThresholdTest(std::log(3.0),           0.5,      1,   1.00000),
    ThresholdTest(std::log(3.0),           0.7,      1,   0.53503),
    ThresholdTest(std::log(3.0),           0.9,      1,  -0.46497),
    ThresholdTest(std::log(3.0),           1.0,      1,   kNegInf,  1e-46),

    // Fix epsilon = 10^9 & max_pc = 1, and vary delta
    //
    //                                               expected   test
    //               epsilon          delta  max_pc  threshold  tolerance
    //            ----------  -------------  ------  ---------  ---------
    ThresholdTest(1000000000,           0.0,      1, kPosInf),
    ThresholdTest(1000000000, kDoubleMinPos,      1, 1),
    ThresholdTest(1000000000,        1e-308,      1, 1),
    ThresholdTest(1000000000,        1e-200,      1, 1),
    ThresholdTest(1000000000,        1e-100,      1, 1),
    ThresholdTest(1000000000,         1e-50,      1, 1),
    ThresholdTest(1000000000,          1e-8,      1, 1),
    ThresholdTest(1000000000,           0.1,      1, 1),
    ThresholdTest(1000000000,           0.5,      1, 1),
    ThresholdTest(1000000000,           0.8,      1, 1),
    ThresholdTest(1000000000,           1.0,      1, kNegInf,  1e-46),

    // Fix delta = 2.0894334e-24 & max_pc = 1, and vary epsilon.
    //
    //                                                    expected  test
    //                  epsilon          delta  max_pc   threshold  tolerance
    //            -------------  -------------  ------  ----------  -----------
    ThresholdTest(kDoubleMinPos, 2.0894334e-24,      1,    kPosInf),
    ThresholdTest(       1e-308, 2.0894334e-24,      1,    kPosInf),
    ThresholdTest(       1e-100, 2.0894334e-24,      1, 5.3832e101, 2.12256e94),
    ThresholdTest(        1e-50, 2.0894334e-24,      1, 5.3832e051, 2.12256e44),
    ThresholdTest(        1e-20, 2.0894334e-24,      1, 5.3832e021, 2.12256e14),
    ThresholdTest(        1e-10, 2.0894334e-24,      1, 5.3832e011, 21227),
    ThresholdTest(         1e-5, 2.0894334e-24,      1, 5.3832e006, 1.3),
    ThresholdTest(         1e-2, 2.0894334e-24,      1,  5384.2000, 0.15),
    ThresholdTest(         1e-1, 2.0894334e-24,      1,   539.3200),
    ThresholdTest(          0.5, 2.0894334e-24,      1,   108.6640),
    ThresholdTest(          1.0, 2.0894334e-24,      1,    54.8320),
    ThresholdTest(std::log(3.0), 2.0894334e-24,      1,    50.0000),
    ThresholdTest(          1.5, 2.0894334e-24,      1,    36.8880),
    ThresholdTest(          2.0, 2.0894334e-24,      1,    27.9160),
    ThresholdTest(          5.0, 2.0894334e-24,      1,    11.7664),
    ThresholdTest(          1e1, 2.0894334e-24,      1,     6.3832),
    ThresholdTest(          1e2, 2.0894334e-24,      1,     1.53832),
    ThresholdTest(          1e5, 2.0894334e-24,      1,     1.00054),
    ThresholdTest(         1e10, 2.0894334e-24,      1,     1.0000),
    ThresholdTest(         1e20, 2.0894334e-24,      1,     1.0000),
    ThresholdTest(         1e50, 2.0894334e-24,      1,     1.0000),
    ThresholdTest(        1e100, 2.0894334e-24,      1,     1.0000),
    ThresholdTest(        1e308, 2.0894334e-24,      1,     1.0000),
    ThresholdTest(   kDoubleMax, 2.0894334e-24,      1,     1.0000),

    // Fix epsilon & delta and vary max_pc.
    //                                                  expected  test
    //             epsilon          delta     max_pc   threshold  tolerance
    //            --------  -------------  ---------  ----------  ---------
    ThresholdTest(std::log(3.0), 2.0894334e-24,         1,     50.000),
    ThresholdTest(std::log(3.0), 2.0894334e-24,         2,    100.262),
    ThresholdTest(std::log(3.0), 2.0894334e-24,         3,    151.000),
    ThresholdTest(std::log(3.0), 2.0894334e-24,         4,    202.047),
    ThresholdTest(std::log(3.0), 2.0894334e-24,         5,    253.325),
    ThresholdTest(std::log(3.0), 2.0894334e-24,        10,    511.959),
    ThresholdTest(std::log(3.0), 2.0894334e-24,       100,   5320.180),
    ThresholdTest(std::log(3.0), 2.0894334e-24,      1000,  55288.700),
    ThresholdTest(std::log(3.0), 2.0894334e-24,     10000, 573837.000, 0.131),
    ThresholdTest(std::log(3.0), 2.0894334e-24,    100000,  5.94795e6, 2.640),
    ThresholdTest(std::log(3.0), 2.0894334e-24,   1000000,  6.15754e7, 20.63),
    ThresholdTest(std::log(3.0), 2.0894334e-24, kInt64Max, 8.18561e20, 1.19938e14),

    // Test that we can legitimately compute a negative threshold.
    ThresholdTest(0.001, 0.99999999999, 1, -24634.3),

    // Error cases.
    //
    // Epsilon must be finite and greater than 0.
    //
    //                    epsilon          delta  max_pc
    //                 ----------  -------------  ------
    ThresholdErrorTest(kDoubleMin, 2.0894334e-24,      1),
    ThresholdErrorTest(      -1.0, 2.0894334e-24,      1),
    ThresholdErrorTest(       0.0, 2.0894334e-24,      1),
    ThresholdErrorTest(   kPosInf, 2.0894334e-24,      1),
    ThresholdErrorTest(   kNegInf, 2.0894334e-24,      1),
    ThresholdErrorTest(      kNaN, 2.0894334e-24,      1),

    // Max_pc must be greater than 0.
    //
    //                   epsilon         delta  max_pc
    //                 --------- -------------  ------
    ThresholdErrorTest(std::log(3.0), 2.0894334e-24, kInt64Min),
    ThresholdErrorTest(std::log(3.0), 2.0894334e-24, -1),
    ThresholdErrorTest(std::log(3.0), 2.0894334e-24, 0),

    // Delta must be in range [0, 1].
    //
    //                  epsilon              delta  max_pc
    //                 --------  -----------------  ------
    ThresholdErrorTest(std::log(3.0),           kNegInf,      1),
    ThresholdErrorTest(std::log(3.0),           kNegInf,      1),
    ThresholdErrorTest(std::log(3.0),                -1,      1),
    ThresholdErrorTest(std::log(3.0),                -1,      1),
    ThresholdErrorTest(std::log(3.0), 0 - kDoubleMinPos,      1),
    ThresholdErrorTest(std::log(3.0), 0 - kDoubleMinPos,      1),
    ThresholdErrorTest(std::log(3.0),   1.0000000000001,      1),
    ThresholdErrorTest(std::log(3.0),   1.0000000000001,      1),
    ThresholdErrorTest(std::log(3.0),                 2,      1),
    ThresholdErrorTest(std::log(3.0),                 2,      1),
    ThresholdErrorTest(std::log(3.0),           kPosInf,      1),
    ThresholdErrorTest(std::log(3.0),           kPosInf,      1),
  };

  for (const CalculateThresholdTest& test_case : threshold_test_cases) {
    const double epsilon = test_case.epsilon;
    const double delta = test_case.delta;
    const int64_t max_partitions_contributed =
        test_case.max_partitions_contributed;

    const std::string test_case_string = absl::StrCat(
        "epsilon: ", epsilon, ", delta: ", delta,
        ", max_partitions_contributed: ", max_partitions_contributed);

    absl::StatusOr<double> status_or_threshold =
        LaplacePartitionSelection::CalculateThreshold(
            epsilon, delta, max_partitions_contributed);

    if (test_case.expected_threshold.has_value()) {
      // We expect the test to succeed, and a threshold to be computed.
      ASSERT_TRUE(status_or_threshold.ok())
          << status_or_threshold.status() << "\n"
          << test_case_string;
      const double threshold = status_or_threshold.value();
      EXPECT_THAT(threshold, DoubleNear(test_case.expected_threshold.value(),
                                        test_case.tolerance))
          << test_case_string;
    } else {
      EXPECT_FALSE(status_or_threshold.ok())
          << test_case_string
          << "\nunexpected successfully computed threshold value: "
          << status_or_threshold.value();
    }
  }
}

TEST(PartitionSelectionTest, RoundTripThresholdTests) {
  // Vary threshold from 1 to 100, and max_partitions_contributed from 1 to 5.
  // Calculate delta for each each tuple (epsilon, threshold,
  // max_partitions_contributed), and then re-compute threshold to ensure it is
  // the same. Note that once the threshold gets high enough so that the
  // computed delta is 0, round-tripping back to the original threshold longer
  // works so we stop early in that case.
  const double epsilon = std::log(3);
  for (int64_t max_partitions_contributed = 1; max_partitions_contributed < 5;
       ++max_partitions_contributed) {
    bool computed_delta_of_zero = false;
    for (int64_t threshold = -20; threshold < 1000; threshold += 10) {
      std::string test_case_string = absl::StrCat(
          "threshold: ", threshold,
          ", max_partitions_contributed: ", max_partitions_contributed);
      absl::StatusOr<double> status_or_delta =
          LaplacePartitionSelection::CalculateDelta(epsilon, threshold,
                                                    max_partitions_contributed);
      ASSERT_TRUE(status_or_delta.ok()) << status_or_delta.status() << "\n"
                                        << test_case_string;
      const double delta = status_or_delta.value();
      absl::StrAppend(&test_case_string, ", computed delta: ", delta);

      // If the computed delta is 0, then computing threshold from this will
      // result in the maximum value. Therefore round tripping will not work.
      // Note that all remaining input threshold values are higher that this one
      // within this loop, which implies that all remaining computed deltas will
      // be 0 as well.
      if (computed_delta_of_zero) {
        EXPECT_EQ(delta, 0);
        continue;
      }
      if (delta == 0) {
        computed_delta_of_zero = true;
        continue;
      }

      absl::StatusOr<double> status_or_threshold =
          LaplacePartitionSelection::CalculateThreshold(
              epsilon, delta, max_partitions_contributed);
      ASSERT_TRUE(status_or_threshold.ok())
          << status_or_threshold.status() << "\n"
          << test_case_string;

      // We normally expect that the original threshold and the round tripped
      // threshold are the same. However, because of the loss of precision
      // due to floating point calculations, there may be some variance.
      const double computed_threshold = status_or_threshold.value();

      EXPECT_THAT(computed_threshold, DoubleNear(threshold, 0.001))
          << test_case_string;
    }
  }
}

TEST(PartitionSelectionTest, RoundTripDeltaTests) {
  // Ideally, vary max_partitions_contributed from 1 to 5, and delta from 0 to 1
  // (exclusively). Calculate threshold for each triple (epsilon, delta,
  // max_partitions_contributed), and then re-compute the delta.
  //
  // We can't check the recomputed delta for equality against the original
  // delta because there are many deltas that map to the same threshold.
  // We also cannot check that the recomputed delta is closer to the
  // original delta than to computed deltas for threshold +/-1, and we
  // cannot check that the original delta is closer to the recomputed delta
  // than to the computed deltas for threshold +/1. This is because the
  // curve is not linear, and, for example, the original delta can be closer
  // to the computed delta for threshold-1 than to the recomputed delta.
  //
  // So we just test here that the original delta is between the recomputed
  // delta and either the computed delta for threshold+1 or threshold-1.
  const double epsilon = std::log(3);
  for (int64_t max_partitions_contributed = 1; max_partitions_contributed < 5;
       ++max_partitions_contributed) {
    for (double delta = 1e-308; delta < 1.0; delta *= 10) {
      std::string test_case_string = absl::StrCat(
          "delta: ", delta,
          ", max_partitions_contributed: ", max_partitions_contributed);

      absl::StatusOr<double> status_or_threshold =
          LaplacePartitionSelection::CalculateThreshold(
              epsilon, delta, max_partitions_contributed);
      ASSERT_TRUE(status_or_threshold.ok())
          << status_or_threshold.status() << "\n"
          << test_case_string;
      const double threshold = status_or_threshold.value();

      absl::StrAppend(&test_case_string, ", computed threshold: ", threshold);

      const absl::StatusOr<double> computed_delta =
          LaplacePartitionSelection::CalculateDelta(epsilon, threshold,
                                                    max_partitions_contributed);

      const absl::StatusOr<double> computed_delta_plus_one =
          LaplacePartitionSelection::CalculateDelta(epsilon, threshold + 1,
                                                    max_partitions_contributed);

      const absl::StatusOr<double> computed_delta_minus_one =
          LaplacePartitionSelection::CalculateDelta(epsilon, threshold - 1,
                                                    max_partitions_contributed);

      // Expect that the original delta is closer to the computed delta than
      // to the computed delta for threshold+1 or threshold-1.
      ASSERT_TRUE(computed_delta.ok());
      ASSERT_TRUE(computed_delta_plus_one.ok());
      ASSERT_TRUE(computed_delta_minus_one.ok());

      absl::StrAppend(&test_case_string,
                      ", computed delta: ", computed_delta.value());
      absl::StrAppend(&test_case_string, ", computed delta (k plus_one): ",
                      computed_delta_plus_one.value());
      absl::StrAppend(&test_case_string, ", computed delta (k minus one): ",
                      computed_delta_minus_one.value());

      // The original delta and recomputed delta for threshold is greater
      // than the computed delta for threshold+1.
      // The original delta and recomputed delta for threshold is less than
      // the computed delta for threshold-1.
      EXPECT_GT(delta, computed_delta_plus_one.value());
      EXPECT_LT(delta, computed_delta_minus_one.value());

      EXPECT_GT(computed_delta.value(), computed_delta_plus_one.value());
      EXPECT_LT(computed_delta.value(), computed_delta_minus_one.value());

      EXPECT_TRUE((delta >= computed_delta.value() &&
                   delta <= computed_delta_minus_one.value()) ||
                  (delta <= computed_delta.value() &&
                   delta >= computed_delta_plus_one.value()));
    }
  }
}

// GaussianPartitionSelection Tests
// Due to the inheritance, SetGaussianMechanism must be
// called before SetDelta, SetEpsilon, etc.

TEST(PartitionSelectionTest,
     GaussianPartitionSelectionUnsetMaxPartitionsContributed) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetDelta(0.1)
          .SetEpsilon(2)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be set")));
}

TEST(PartitionSelectionTest,
     GaussianPartitionSelectionNegativeMaxPartitionsContributed) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetDelta(0.1)
          .SetEpsilon(2)
          .SetMaxPartitionsContributed(-3)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be positive")));
}

TEST(PartitionSelectionTest,
     GaussianPartitionSelectionZeroMaxPartitionsContributed) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetDelta(0.1)
          .SetEpsilon(2)
          .SetMaxPartitionsContributed(0)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be positive, but is 0.")));
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionUnsetEpsilon) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetDelta(0.1)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Epsilon must be set.")));
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionUnsetDelta) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be set.")));
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionNanDelta) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetDelta(NAN)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be a valid numeric value")));
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionNotFiniteDelta) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetDelta(std::numeric_limits<double>::infinity())
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionInvalidPositiveDelta) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetDelta(5.2)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionInvalidNegativeDelta) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder
          .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
          .SetEpsilon(0.1)
          .SetDelta(-0.1)
          .SetMaxPartitionsContributed(2)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionTest, CalculateThresholdFromStddevNegativeStd) {
  EXPECT_THAT(GaussianPartitionSelection::CalculateThresholdFromStddev(
                  /* stddev= */ -0.1, /* threshold_delta =*/1e-5,
                  /* max_partitions_contributed= */ 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Stddev must be finite and positive")));
}

TEST(PartitionSelectionTest, CalculateThresholdFromStddevNegativeStdDelta) {
  EXPECT_THAT(
      GaussianPartitionSelection::CalculateThresholdFromStddev(
          /* stddev= */ 0.5, /* threshold_delta =*/-1e-5,
          /* max_partitions_contributed= */ 1),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval [0,1]")));
}

TEST(PartitionSelectionTest, CalculateThresholdFromStddevZeroMaxPc) {
  EXPECT_THAT(GaussianPartitionSelection::CalculateThresholdFromStddev(
                  /* stddev= */ 0.5, /* threshold_delta =*/1e-5,
                  /* max_partitions_contributed= */ 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("L0 sensitivity) must be positive")));
}

struct ThresholdTestFromStddev {
  double stddev;
  double threshold_delta;
  int64_t max_partitions_contributed;

  double expected_threshold;
};

class CalculateThresholdFromStddevTest
    : public testing::TestWithParam<ThresholdTestFromStddev> {};

TEST_P(CalculateThresholdFromStddevTest, Test) {
  auto& [stddev, threshold_delta, max_partitions_contributed, expected] =
      GetParam();

  EXPECT_THAT(GaussianPartitionSelection::CalculateThresholdFromStddev(
                  stddev, threshold_delta, max_partitions_contributed),
              IsOkAndHolds(DoubleNear(expected, 1e-12)))
      << absl::StrFormat(
             "stddev=%v threshold_delta=%v max_partitions_contributed=%v",
             stddev, threshold_delta, max_partitions_contributed);
}

INSTANTIATE_TEST_SUITE_P(
    CalculateThresholdFromStddevTests, CalculateThresholdFromStddevTest,
    testing::ValuesIn<ThresholdTestFromStddev>({
        // "max_pc" is shorthand for "max_partitions_contributed".
        //
        //                    stddev delta max_pc expected_threshold
        ThresholdTestFromStddev{0.01, 1e-15, 1, 1.07941345326171},
        ThresholdTestFromStddev{0.01, 1e-10, 1, 1.0636134090240406},
        ThresholdTestFromStddev{0.01, 1e-05, 1, 1.0426489079392283},
        ThresholdTestFromStddev{0.1, 1e-15, 1, 1.7941345326170999},
        ThresholdTestFromStddev{0.1, 1e-10, 1, 1.6361340902404056},
        ThresholdTestFromStddev{0.1, 1e-05, 1, 1.4264890793922824},
        ThresholdTestFromStddev{1, 1e-15, 1, 8.941345326170998},
        ThresholdTestFromStddev{1, 1e-10, 1, 7.361340902404056},
        ThresholdTestFromStddev{1, 1e-05, 1, 5.264890793922825},
        ThresholdTestFromStddev{10, 1e-15, 1, 80.41345326170998},
        ThresholdTestFromStddev{10, 1e-10, 1, 64.61340902404055},
        ThresholdTestFromStddev{10, 1e-05, 1, 43.648907939228245},
        ThresholdTestFromStddev{100, 1e-15, 1, 795.1345326170998},
        ThresholdTestFromStddev{100, 1e-10, 1, 637.1340902404056},
        ThresholdTestFromStddev{100, 1e-05, 1, 427.48907939228246},
        ThresholdTestFromStddev{10, 1e-15, 1, 80.41345326170998},
        ThresholdTestFromStddev{10, 1e-15, 10, 83.22082216130434},
        ThresholdTestFromStddev{10, 1e-15, 100, 85.93793224109598},
        ThresholdTestFromStddev{10, 1e-10, 1, 64.61340902404055},
        ThresholdTestFromStddev{10, 1e-10, 10, 68.06023155488566},
        ThresholdTestFromStddev{10, 1e-10, 100, 71.3448382529423},
        ThresholdTestFromStddev{10, 1e-05, 1, 43.648907939228245},
        ThresholdTestFromStddev{10, 1e-05, 10, 48.53423399421888},
        ThresholdTestFromStddev{10, 1e-05, 100, 52.99336662034604},
    }));

TEST(PartitionSelectionTest, CalculateGaussianThresholdTests) {
  std::vector<CalculateThresholdTest> threshold_test_cases = {
    // In all tests, "max_pc" is shorthand for "max_partitions_contributed".
    //
    // Fix epsilon = ln(3) & max_pc = 1, and vary delta.
    //
    //                                              expected  test
    //             epsilon          delta  max_pc  threshold  tolerance
    //            --------  -------------  ------  ---------  ---------
    ThresholdTest(std::log(3.0),           0.0,      1,   kPosInf),
    ThresholdTest(std::log(3.0), kDoubleMinPos,      1,   1278.4594),
    ThresholdTest(std::log(3.0),        1e-308,      1,   1279.1838),
    ThresholdTest(std::log(3.0),        1e-256,      1,   1061.4645),
    ThresholdTest(std::log(3.0),        1e-128,      1,   526.3769),
    ThresholdTest(std::log(3.0),         1e-64,      1,   259.1447),
    ThresholdTest(std::log(3.0),         1e-32,      1,   126.3767),
    ThresholdTest(std::log(3.0),         1e-16,      1,   60.5608),
    ThresholdTest(std::log(3.0),          1e-8,      1,   28.3774),
    ThresholdTest(std::log(3.0),          1e-4,      1,   13.0061),
    ThresholdTest(std::log(3.0),          1e-2,      1,   6.02085),
    ThresholdTest(std::log(3.0),           0.1,      1,   3.06731),
    ThresholdTest(std::log(3.0),           0.3,      1,   1.93016),
    ThresholdTest(std::log(3.0),           0.5,      1,   1.49072),
    ThresholdTest(std::log(3.0),           0.7,      1,   1.23706),
    ThresholdTest(std::log(3.0),           0.9,      1,   1.06663),
    ThresholdTest(std::log(3.0),           1.0,      1,   1),
    ThresholdTest(std::log(3.0),           2.0,      1,   kNegInf),

    // Fix epsilon = 10^9 & max_pc = 1, and vary delta
    //
    //                                               expected   test
    //               epsilon          delta  max_pc  threshold  tolerance
    //            ----------  -------------  ------  ---------  ---------
    ThresholdTest(1000000000,           0.0,      1, kPosInf),
    ThresholdTest(1000000000, kDoubleMinPos,      1, 1),
    ThresholdTest(1000000000,        1e-308,      1, 1),
    ThresholdTest(1000000000,        1e-200,      1, 1),
    ThresholdTest(1000000000,        1e-100,      1, 1),
    ThresholdTest(1000000000,         1e-50,      1, 1),
    ThresholdTest(1000000000,          1e-8,      1, 1),
    ThresholdTest(1000000000,           0.1,      1, 1),
    ThresholdTest(1000000000,           0.5,      1, 1),
    ThresholdTest(1000000000,           0.8,      1, 1),
    ThresholdTest(1000000000,           1.0,      1, 1),
    ThresholdTest(1000000000,           2.0,      1, kNegInf),

    // Fix delta = 2.0894334e-24 & max_pc = 1, and vary epsilon.
    //
    //                                                    expected  test
    //                  epsilon          delta  max_pc   threshold  tolerance
    //            -------------  -------------  ------  ----------  -----------
    ThresholdTest(kDoubleMinPos, 2.0894334e-14,     1,    2.90787e+14, 1e+09),
    ThresholdTest(      1e-308, 2.0894334e-14,      1,    2.90787e+14, 1e+09),
    ThresholdTest(      1e-100, 2.0894334e-14,      1,    2.90787e+14, 1e+09),
    ThresholdTest(       1e-50, 2.0894334e-14,      1,    2.90787e+14, 1e+09),
    ThresholdTest(       1e-20, 2.0894334e-14,      1,    2.91837e+14, 1e+09),
    ThresholdTest(       1e-10, 2.0894334e-14,      1,    2.33308e+11, 1e+06),
    ThresholdTest(         1e-5, 2.0894334e-14,      1,    4.13343e+06, 10),
    ThresholdTest(         1e-2, 2.0894334e-14,      1,    4958.78),
    ThresholdTest(         1e-1, 2.0894334e-14,      1,    521.815),
    ThresholdTest(          0.5, 2.0894334e-14,      1,    108.925),
    ThresholdTest(          1.0, 2.0894334e-14,      1,    55.9185),
    ThresholdTest(std::log(3.0), 2.0894334e-14,      1,    51.1404),
    ThresholdTest(          1.5, 2.0894334e-14,      1,    38.0603),
    ThresholdTest(          2.0, 2.0894334e-14,      1,    29.0565),
    ThresholdTest(          5.0, 2.0894334e-14,      1,    12.6840),
    ThresholdTest(          1e1, 2.0894334e-14,      1,    7.1257),
    ThresholdTest(          1e2, 2.0894334e-14,      1,    1.87689),
    ThresholdTest(          1e5, 2.0894334e-14,      1,    1.00054),
    ThresholdTest(         1e10, 2.0894334e-14,      1,    1.0000),
    ThresholdTest(         1e20, 2.0894334e-14,      1,    1.0000),
    ThresholdTest(         1e50, 2.0894334e-14,      1,    1.0000),
    ThresholdTest(        1e100, 2.0894334e-14,      1,    1.0000),
    ThresholdTest(        1e308, 2.0894334e-14,      1,    1.0000),
    ThresholdTest(   kDoubleMax, 2.0894334e-14,      1,    1.0000),

    // Fix epsilon & delta and vary max_pc.
    //                                                        expected
    //             epsilon        delta            max_pc     threshold
    //            --------        -------------   ---------   ----------
    ThresholdTest(std::log(3.0), 2.0894334e-14,         1,    51.1404),
    ThresholdTest(std::log(3.0), 2.0894334e-14,         2,    72.7319),
    ThresholdTest(std::log(3.0), 2.0894334e-14,         3,    89.4375),
    ThresholdTest(std::log(3.0), 2.0894334e-14,         4,    103.5948),
    ThresholdTest(std::log(3.0), 2.0894334e-14,         5,    116.1157),
    ThresholdTest(std::log(3.0), 2.0894334e-14,        10,    165.5916),
    ThresholdTest(std::log(3.0), 2.0894334e-14,       100,   539.9088),
    ThresholdTest(std::log(3.0), 2.0894334e-14,      1000,  1761.5664),
    ThresholdTest(std::log(3.0), 2.0894334e-14,     10000,  5741.3149),
    ThresholdTest(std::log(3.0), 2.0894334e-14,    100000,  18684.6603),
    ThresholdTest(std::log(3.0), 2.0894334e-14,   1000000,  60718.2913),
    ThresholdTest(std::log(3.0), 2.0894334e-14, kInt64Max,  239762230815.2955),

    // Error cases.
    //
    // Epsilon must be finite and greater than 0.
    //
    //                    epsilon          delta  max_pc
    //                 ----------  -------------  ------
    ThresholdErrorTest(kDoubleMin, 2.0894334e-24,      1),
    ThresholdErrorTest(      -1.0, 2.0894334e-24,      1),
    ThresholdErrorTest(       0.0, 2.0894334e-24,      1),
    ThresholdErrorTest(   kPosInf, 2.0894334e-24,      1),
    ThresholdErrorTest(   kNegInf, 2.0894334e-24,      1),
    ThresholdErrorTest(      kNaN, 2.0894334e-24,      1),

    // Max_pc must be greater than 0.
    //
    //                   epsilon         delta  max_pc
    //                 --------- -------------  ------
    ThresholdErrorTest(std::log(3.0), 2.0894334e-24, kInt64Min),
    ThresholdErrorTest(std::log(3.0), 2.0894334e-24, -1),
    ThresholdErrorTest(std::log(3.0), 2.0894334e-24, 0),

    // Delta must be in range [0, 2].
    //
    //                  epsilon              delta  max_pc
    //                 --------  -----------------  ------
    ThresholdErrorTest(std::log(3.0),           kNegInf,      1),
    ThresholdErrorTest(std::log(3.0),           kNegInf,      1),
    ThresholdErrorTest(std::log(3.0),                -1,      1),
    ThresholdErrorTest(std::log(3.0),                -1,      1),
    ThresholdErrorTest(std::log(3.0), 0 - kDoubleMinPos,      1),
    ThresholdErrorTest(std::log(3.0), 0 - kDoubleMinPos,      1),
    ThresholdErrorTest(std::log(3.0),   2.0000000000001,      1),
    ThresholdErrorTest(std::log(3.0),   2.0000000000001,      1),
    ThresholdErrorTest(std::log(3.0),                 3,      1),
    ThresholdErrorTest(std::log(3.0),                 3,      1),
    ThresholdErrorTest(std::log(3.0),           kPosInf,      1),
    ThresholdErrorTest(std::log(3.0),           kPosInf,      1),
  };

  for (const CalculateThresholdTest& test_case : threshold_test_cases) {
    const double epsilon = test_case.epsilon;
    const double delta = test_case.delta;
    const int64_t max_partitions_contributed =
        test_case.max_partitions_contributed;

    const std::string test_case_string = absl::StrCat(
        "epsilon: ", epsilon, ", delta: ", delta,
        ", max_partitions_contributed: ", max_partitions_contributed);

    const double noise_delta = delta / 2;
    const double threshold_delta = delta - noise_delta;
    absl::StatusOr<double> status_or_threshold =
        GaussianPartitionSelection::CalculateThreshold(
            epsilon, noise_delta, threshold_delta, max_partitions_contributed);

    if (test_case.expected_threshold.has_value()) {
      // We expect the test to succeed, and a threshold to be computed.
      ASSERT_TRUE(status_or_threshold.ok())
          << status_or_threshold.status() << "\n"
          << test_case_string;
      const double threshold = status_or_threshold.value();
      EXPECT_THAT(threshold, DoubleNear(test_case.expected_threshold.value(),
                                        test_case.tolerance))
          << test_case_string;
    } else {
      EXPECT_FALSE(status_or_threshold.ok())
          << test_case_string
          << "\nunexpected successfully computed threshold value: "
          << status_or_threshold.value();
    }
  }
}

TEST(PartitionSelectionTest, CalculateGaussianDeltaTests) {
  std::vector<CalculateDeltaTest> delta_test_cases = {
    // In all tests, "max_pc" is shorthand for "max_partitions_contributed".
    //
    // Fix epsilon = ln(3) & max_pc = 1, and vary threshold.
    //
    //                                      expected       test
    //         epsilon   threshold  max_pc  delta          tolerance
    //        --------  ----------  ------  -------------  ---------
    DeltaTest(std::log(3.0),          1,      1, 0.5),
    DeltaTest(std::log(3.0),          2,      1, 0.0216028),
    DeltaTest(std::log(3.0),          3,      1, 2.63368e-05,   1e-05),
    DeltaTest(std::log(3.0),          4,      1, 6.5914e-10,    1e-05),
    DeltaTest(std::log(3.0),          5,      1, 3.33067e-16,   1e-06),
    DeltaTest(std::log(3.0),         10,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),         20,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),         50,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),         75,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),         87,      1, 0.0,           6e-100),
    DeltaTest(std::log(3.0),         93,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),         94,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),         95,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),         96,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),        100,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),       1000,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),      10000,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),     100000,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),    1000000,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0), kDoubleMax,      1, 0.0,           1e-100),
    DeltaTest(std::log(3.0),    kPosInf,      1, 0.0,           1e-100),

    // Fix threshold = 2 & max_pc = 1, and vary epsilon.
    //
    //                                          expected       test
    //              epsilon  threshold  max_pc  delta          tolerance
    //        -------------  ---------  ------  -------------  ---------
    DeltaTest(kDoubleMinPos,        2,      1, 0.0887881),
    DeltaTest(       1e-308,        2,      1, 0.0887881),
    DeltaTest(       1e-100,        2,      1, 0.0887881),
    DeltaTest(        1e-50,        2,      1, 0.0887881),
    DeltaTest(        1e-20,        2,      1, 0.0887881),
    DeltaTest(        1e-10,        2,      1, 0.0887881),
    DeltaTest(         1e-5,        2,      1, 0.0887881),
    DeltaTest(         1e-2,        2,      1, 0.0875032),
    DeltaTest(         1e-1,        2,      1, 0.0771943),
    DeltaTest(          0.5,        2,      1, 0.0454027),
    DeltaTest(          1.0,        2,      1, 0.0243547),
    DeltaTest(std::log(3.0),        2,      1, 0.0216028),
    DeltaTest(          1.5,        2,      1, 0.0134125),
    DeltaTest(          2.0,        2,      1, 0.00750155),
    DeltaTest(          5.0,        2,      1, 0.000273579),
    DeltaTest(          1e1,        2,      1, 1.38959e-06),
    DeltaTest(          1e2,        2,      1, 0.0,           1e-100),
    DeltaTest(          1e5,        2,      1, 0.0,           1e-100),
    DeltaTest(         1e10,        2,      1, 0.0,           1e-100),
    DeltaTest(         1e20,        2,      1, 0.0,           1e-100),
    DeltaTest(         1e50,        2,      1, 0.0,           1e-100),
    DeltaTest(        1e100,        2,      1, 0.0,           1e-100),
    DeltaTest(        1e308,        2,      1, 0.0,           1e-100),
    DeltaTest(   kDoubleMax,        2,      1, 0.0,           1e-100),

    // Fix epsilon & threshold and vary max_pc.
    //                                        expected       test
    //         epsilon  threshold     max_pc  delta          tolerance
    //        --------  ---------  ---------  -------------  ---------
    DeltaTest(std::log(3.0),        2,         1, 0.0216028),
    DeltaTest(std::log(3.0),        2,         2, 0.14700),
    DeltaTest(std::log(3.0),        2,         3, 0.322138),
    DeltaTest(std::log(3.0),        2,         4, 0.492678),
    DeltaTest(std::log(3.0),        2,         5, 0.635904),
    DeltaTest(std::log(3.0),        2,        10, 1.0),
    DeltaTest(std::log(3.0),        2,       100, 1.0),
    DeltaTest(std::log(3.0),        2,      1000, 1.0),
    DeltaTest(std::log(3.0),        2,     10000, 1.0),
    DeltaTest(std::log(3.0),        2,    100000, 1.0),
    DeltaTest(std::log(3.0),        2,   1000000, 1.0),
    DeltaTest(std::log(3.0),        2, kInt64Max, 1.0),

    // Error cases.
    //
    // Epsilon must be finite and greater than 0.
    //
    //                epsilon  threshold  max_pc
    //             ----------  ---------  ------
    DeltaErrorTest(kDoubleMin,        2, 1),
    DeltaErrorTest(      -1.0,        2, 1),
    DeltaErrorTest(       0.0,        2, 1),
    DeltaErrorTest(   kPosInf,        2, 1),
    DeltaErrorTest(   kNegInf,        2, 1),
    DeltaErrorTest(      kNaN,        2, 1),

    // Threshold must be finite.
    //
    //              epsilon  threshold  max_pc
    //             --------  ---------  ------
    DeltaErrorTest(std::log(3.0),      kNaN, 1),

    // Max_partitions_contributed (max_pc) must be greater than 0.
    //
    //              epsilon  threshold  max_pc
    //             --------  ---------  ---------
    DeltaErrorTest(std::log(3.0),        2, kInt64Min),
    DeltaErrorTest(std::log(3.0),        2, -1),
    DeltaErrorTest(std::log(3.0),        2, 0),
  };

  for (const CalculateDeltaTest& test_case : delta_test_cases) {
    const double epsilon = test_case.epsilon;
    const double threshold = test_case.threshold;
    const double noise_delta = 0.5;
    const int64_t max_partitions_contributed =
        test_case.max_partitions_contributed;

    const std::string test_case_string = absl::StrCat(
        "epsilon: ", epsilon, ", threshold: ", threshold,
        ", max_partitions_contributed: ", max_partitions_contributed);

    absl::StatusOr<double> status_or_delta =
        GaussianPartitionSelection::CalculateThresholdDelta(
            epsilon, noise_delta, threshold, max_partitions_contributed);

    if (test_case.expected_delta.has_value()) {
      // We expect the test to succeed, and a delta to be computed.
      ASSERT_TRUE(status_or_delta.ok()) << status_or_delta.status() << "\n"
                                        << test_case_string;
      const double delta = status_or_delta.value();
      EXPECT_THAT(delta, DoubleNear(test_case.expected_delta.value(),
                                    test_case.tolerance))
          << test_case_string;
    } else {
      EXPECT_FALSE(status_or_delta.ok())
          << test_case_string
          << "\nunexpected successfully computed delta value: "
          << status_or_delta.value();
    }
  }
}

TEST(PartitionSelectionTest, GaussianRoundTripThresholdTests) {
  // Vary threshold from 1 to 1000, and max_partitions_contributed from 1 to 5.
  // Calculate delta for each tuple (epsilon, threshold,
  // max_partitions_contributed), and then re-compute threshold to ensure it is
  // the same. Note that once the threshold gets high enough so that the
  // computed delta is 0, round-tripping back to the original threshold longer
  // works so we stop early in that case.
  const double epsilon = std::log(3);
  const double noise_delta = 0.5;
  for (int64_t max_partitions_contributed = 1; max_partitions_contributed < 5;
       ++max_partitions_contributed) {
    bool computed_delta_of_zero = false;
    for (int64_t threshold = 0; threshold < 1000; threshold += 10) {
      std::string test_case_string = absl::StrCat(
          "threshold: ", threshold,
          ", max_partitions_contributed: ", max_partitions_contributed);
      absl::StatusOr<double> status_or_delta =
          GaussianPartitionSelection::CalculateThresholdDelta(
              epsilon, noise_delta, threshold, max_partitions_contributed);
      ASSERT_TRUE(status_or_delta.ok()) << status_or_delta.status() << "\n"
                                        << test_case_string;
      const double delta = status_or_delta.value();
      absl::StrAppend(&test_case_string, ", computed delta: ", delta);

      // If the computed delta is 0, then computing threshold from this will
      // result in the maximum value. Therefore round tripping will not work.
      // Note that all remaining input threshold values are higher that this one
      // within this loop, which implies that all remaining computed deltas will
      // be 0 as well.
      if (computed_delta_of_zero) {
        EXPECT_EQ(delta, 0);
        continue;
      }
      if (delta == 0) {
        computed_delta_of_zero = true;
        continue;
      }

      absl::StatusOr<double> status_or_threshold =
          GaussianPartitionSelection::CalculateThreshold(
              epsilon, noise_delta, delta, max_partitions_contributed);
      ASSERT_TRUE(status_or_threshold.ok())
          << status_or_threshold.status() << "\n"
          << test_case_string;

      // We normally expect that the original threshold and the round tripped
      // threshold are the same. However, because of the loss of precision
      // due to floating point calculations, there may be some variance.
      const double computed_threshold = status_or_threshold.value();

      EXPECT_THAT(computed_threshold, DoubleNear(threshold, 0.001))
          << test_case_string;
    }
  }
}

TEST(PartitionSelectionTest, GaussianRoundTripDeltaTests) {
  // Ideally, vary max_partitions_contributed from 1 to 5, and threshold delta
  // from 0 to 0.5 (exclusively). Calculate threshold for each triple (epsilon,
  // delta, max_partitions_contributed), and then re-compute the delta.
  //
  // We can't check the recomputed delta for equality against the original
  // delta because there are many deltas that map to the same threshold.
  // We also cannot check that the recomputed delta is closer to the
  // original delta than to computed deltas for threshold +/-1, and we
  // cannot check that the original delta is closer to the recomputed delta
  // than to the computed deltas for threshold +/1. This is because the
  // curve is not linear, and, for example, the original delta can be closer
  // to the computed delta for threshold-1 than to the recomputed delta.
  //
  // So we just test here that the original delta is between the recomputed
  // delta and either the computed delta for threshold+1 or threshold-1.
  const double epsilon = std::log(3);
  const double noise_delta = 0.5;
  for (int64_t max_partitions_contributed = 1; max_partitions_contributed < 5;
       ++max_partitions_contributed) {
    for (double delta = 0.00001; delta < 0.5; delta *= 10) {
      std::string test_case_string = absl::StrCat(
          "delta: ", delta,
          ", max_partitions_contributed: ", max_partitions_contributed);

      absl::StatusOr<double> status_or_threshold =
          GaussianPartitionSelection::CalculateThreshold(
              epsilon, noise_delta, delta, max_partitions_contributed);
      ASSERT_TRUE(status_or_threshold.ok())
          << status_or_threshold.status() << "\n"
          << test_case_string;
      const double threshold = status_or_threshold.value();

      absl::StrAppend(&test_case_string, ", computed threshold: ", threshold);

      const absl::StatusOr<double> computed_delta =
          GaussianPartitionSelection::CalculateThresholdDelta(
              epsilon, noise_delta, threshold, max_partitions_contributed);

      const absl::StatusOr<double> computed_delta_plus_one =
          GaussianPartitionSelection::CalculateThresholdDelta(
              epsilon, noise_delta, threshold + 1, max_partitions_contributed);

      const absl::StatusOr<double> computed_delta_minus_one =
          GaussianPartitionSelection::CalculateThresholdDelta(
              epsilon, noise_delta, threshold - 1, max_partitions_contributed);

      // Expect that the original delta is closer to the computed delta than
      // to the computed delta for threshold+1 or threshold-1.
      ASSERT_TRUE(computed_delta.ok());
      ASSERT_TRUE(computed_delta_plus_one.ok());
      ASSERT_TRUE(computed_delta_minus_one.ok());

      absl::StrAppend(&test_case_string,
                      ", computed delta: ", computed_delta.value());
      absl::StrAppend(&test_case_string, ", computed delta (k plus_one): ",
                      computed_delta_plus_one.value());
      absl::StrAppend(&test_case_string, ", computed delta (k minus one): ",
                      computed_delta_minus_one.value());

      // The original delta and recomputed delta for threshold is greater
      // than the computed delta for threshold+1.
      // The original delta and recomputed delta for threshold is less than
      // the computed delta for threshold-1.
      EXPECT_GT(delta, computed_delta_plus_one.value());
      EXPECT_LT(delta, computed_delta_minus_one.value());

      EXPECT_GT(computed_delta.value(), computed_delta_plus_one.value());
      EXPECT_LT(computed_delta.value(), computed_delta_minus_one.value());

      EXPECT_TRUE((delta >= computed_delta.value() &&
                   delta <= computed_delta_minus_one.value()) ||
                  (delta <= computed_delta.value() &&
                   delta >= computed_delta_plus_one.value()));
    }
  }
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionNoiseValueIfShouldKeep) {
  GaussianPartitionSelection::Builder test_builder;
  test_builder
      .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
      .SetEpsilon(0.5)
      .SetDelta(0.01)
      .SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  auto* gaussian_ps = dynamic_cast<GaussianPartitionSelection*>(strategy.get());
  for (int i = 0; i < kTinyNumSamples; ++i) {
    auto noised_value = gaussian_ps->NoiseValueIfShouldKeep(5);
    if (noised_value.has_value()) {
      ++num_kept;
      EXPECT_GE(noised_value.value(), gaussian_ps->GetThreshold());
    }
  }

  EXPECT_THAT(static_cast<double>(num_kept) / kTinyNumSamples,
              DoubleNear(0.07, 0.02));
}

TEST(PartitionSelectionTest,
     LaplacePartitionSelectionNoiseValueIfShouldKeepNoUsers) {
  LaplacePartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  auto* laplace_ps = dynamic_cast<LaplacePartitionSelection*>(strategy.get());
  ASSERT_THAT(laplace_ps, testing::NotNull());
  std::vector<std::optional<double>> noised_values;
  for (int i = 0; i < kTinyNumSamples; i++) {
    if (laplace_ps->NoiseValueIfShouldKeep(0).has_value()) {
      noised_values.push_back(laplace_ps->NoiseValueIfShouldKeep(0));
    }
  }
  EXPECT_THAT(noised_values, testing::Each(testing::Eq(std::nullopt)));
}

TEST(PartitionSelectionTest, LaplacePartitionSelectionShouldKeepNoUsers) {
  LaplacePartitionSelection::Builder test_builder;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  auto* laplace_ps = dynamic_cast<LaplacePartitionSelection*>(strategy.get());
  ASSERT_THAT(laplace_ps, testing::NotNull());
  int num_kept = 0;
  for (int i = 0; i < kTinyNumSamples; i++) {
    if (laplace_ps->ShouldKeep(0)) {
      ++num_kept;
    }
  }
  EXPECT_EQ(laplace_ps->ProbabilityOfKeep(0), 0);
  EXPECT_EQ(num_kept, 0);
}

TEST(PartitionSelectionTest,
     GaussianPartitionSelectionNoiseValueIfShouldKeepNoUsers) {
  GaussianPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetEpsilon(0.5)
      .SetDelta(0.01)
      .SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  auto* gaussian_ps = dynamic_cast<GaussianPartitionSelection*>(strategy.get());
  ASSERT_THAT(gaussian_ps, testing::NotNull());
  std::vector<std::optional<double>> noised_values;
  for (int i = 0; i < kTinyNumSamples; i++) {
    noised_values.push_back(gaussian_ps->NoiseValueIfShouldKeep(0));
  }
  EXPECT_THAT(noised_values, testing::Each(testing::Eq(std::nullopt)));
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionShouldKeepNoUsers) {
  GaussianPartitionSelection::Builder test_builder;
  test_builder
      .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
      .SetEpsilon(0.5)
      .SetDelta(0.01)
      .SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  auto* gaussian_ps = dynamic_cast<GaussianPartitionSelection*>(strategy.get());
  ASSERT_THAT(gaussian_ps, testing::NotNull());
  std::vector<bool> should_keep;
  for (int i = 0; i < kTinyNumSamples; i++) {
    should_keep.push_back(gaussian_ps->ShouldKeep(0));
  }
  EXPECT_EQ(gaussian_ps->ProbabilityOfKeep(0), 0);
  // With small probability, ShouldKeep evaluates to true.
  EXPECT_THAT(should_keep, testing::Each(testing::Eq(false)));
}

TEST(PartitionSelectionTest, GaussianPartitionSelectionShouldKeep) {
  GaussianPartitionSelection::Builder test_builder;
  test_builder
      .SetGaussianMechanism(absl::make_unique<GaussianMechanism::Builder>())
      .SetEpsilon(0.5)
      .SetDelta(0.01)
      .SetMaxPartitionsContributed(1);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  auto* gaussian_ps = dynamic_cast<GaussianPartitionSelection*>(strategy.get());
  ASSERT_THAT(gaussian_ps, testing::NotNull());
  int num_kept = 0;
  for (int i = 0; i < kTinyNumSamples; ++i) {
    if (gaussian_ps->ShouldKeep(5)) {
      ++num_kept;
    }
  }
  EXPECT_THAT(gaussian_ps->ProbabilityOfKeep(5), DoubleNear(0.07, 0.02));
  EXPECT_THAT(static_cast<double>(num_kept) / kTinyNumSamples,
              DoubleNear(0.07, 0.02));
}

constexpr auto kNearTruncatedGeometric =
    PartitionSelectionStrategyWithPreThresholding::
        PartitionSelectionStrategyType::kNearTruncatedGeometric;
constexpr auto kLaplace = PartitionSelectionStrategyWithPreThresholding::
    PartitionSelectionStrategyType::kLaplace;
constexpr auto kGaussian = PartitionSelectionStrategyWithPreThresholding::
    PartitionSelectionStrategyType::kGaussian;

TEST(PartitionSelectionWithPreThresholdingTest,
     NearTruncatedGeometricPartitionSelectionWithPreThresholdOK) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  ASSERT_OK(test_builder.SetEpsilon(0.5)
                .SetDelta(0.02)
                .SetMaxPartitionsContributed(1)
                .SetPartitionSelectionStrategy(kNearTruncatedGeometric)
                .SetPreThreshold(10)
                .Build());
}

TEST(PartitionSelectionWithPreThresholdingTest,
     LaplacePartitionSelectionWithPreThresholdOK) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  ASSERT_OK(test_builder.SetEpsilon(0.5)
                .SetDelta(0.02)
                .SetMaxPartitionsContributed(1)
                .SetPartitionSelectionStrategy(kLaplace)
                .SetPreThreshold(10)
                .Build());
}

TEST(PartitionSelectionWithPreThresholdingTest,
     GaussianPartitionSelectionWithPreThresholdOK) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  ASSERT_OK(test_builder.SetEpsilon(0.5)
                .SetDelta(0.02)
                .SetMaxPartitionsContributed(1)
                .SetPartitionSelectionStrategy(kGaussian)
                .SetPreThreshold(10)
                .Build());
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionwithPreThresholdEmptyStrategy) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(test_builder.SetPreThreshold(10)
                  .SetEpsilon(0.5)
                  .SetDelta(0.02)
                  .SetMaxPartitionsContributed(1)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Partition Selection Strategy must be set")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdEmptyEpsilon) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetDelta(0.02)
          .SetMaxPartitionsContributed(1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Epsilon must be set")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdEmptyDelta) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetEpsilon(0.05)
          .SetMaxPartitionsContributed(1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be set")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdEmptyMaxPartitionsContributed) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetEpsilon(0.05)
          .SetDelta(0.02)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Maximum number of partitions that can be contributed "
                         "to (i.e., L0 sensitivity) must be set")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdNaNEpsilon) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetEpsilon(NAN)
          .SetDelta(0.02)
          .SetMaxPartitionsContributed(1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Epsilon must be a valid numeric value")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdNegativeDelta) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetEpsilon(0.05)
          .SetDelta(-1)
          .SetMaxPartitionsContributed(1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdLargeDelta) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetEpsilon(0.05)
          .SetDelta(2)
          .SetMaxPartitionsContributed(1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be in the inclusive interval")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdNaNDelta) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetEpsilon(0.05)
          .SetDelta(NAN)
          .SetMaxPartitionsContributed(1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Delta must be a valid numeric value")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdNegativeMaxPartitionsContributed) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetEpsilon(0.05)
          .SetDelta(0.5)
          .SetMaxPartitionsContributed(-1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Maximum number of partitions that can be contributed "
                         "to (i.e., L0 sensitivity) must be positive")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionWithPreThresholdZeroMaxPartitionsContributed) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(10)
          .SetEpsilon(0.05)
          .SetDelta(0.5)
          .SetMaxPartitionsContributed(0)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Maximum number of partitions that can be contributed "
                         "to (i.e., L0 sensitivity) must be positive")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionNegativePreThreshold) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(-1)
          .SetEpsilon(0.05)
          .SetDelta(0.5)
          .SetMaxPartitionsContributed(1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Pre Threshold must be greater than 0, but is -1")));
}

TEST(PartitionSelectionWithPreThresholdingTest,
     PartitionSelectionZeroPreThreshold) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetPartitionSelectionStrategy(kNearTruncatedGeometric)
          .SetPreThreshold(0)
          .SetEpsilon(0.05)
          .SetDelta(0.5)
          .SetMaxPartitionsContributed(1)
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Pre Threshold must be greater than 0, but is 0")));
}

// Mocking tests to check the pre-thresholding logic is delegated correctly.
TEST(PartitionSelectionWithPreThresholdingTest,
     NearTruncatedGeometricPartitionSelectionWithPreThresholdWrapperOK) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  std::unique_ptr<MockNearTruncatedStrategy::Builder> mock_builder =
      absl::make_unique<MockNearTruncatedStrategy::Builder>();
  MockNearTruncatedStrategy* mock_builder_ptr = mock_builder->mock();
  // Check that wrapped ShouldKeep is called with (pre-threshold - 1) removed
  EXPECT_CALL(*mock_builder_ptr, ShouldKeep(14)).Times(1);
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(std::move(mock_builder))
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  strategy->ShouldKeep(23);
}

TEST(PartitionSelectionWithPreThresholdingTest,
     LaplacePartitionSelectionWithPreThresholdWrapperOK) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  std::unique_ptr<MockLaplaceStrategy::Builder> mock_builder =
      absl::make_unique<MockLaplaceStrategy::Builder>();
  MockLaplaceStrategy* mock_builder_ptr = mock_builder->mock();
  // Check that wrapped ShouldKeep is called with (pre-threshold - 1) removed
  EXPECT_CALL(*mock_builder_ptr, ShouldKeep(14)).Times(1);
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(std::move(mock_builder))
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  strategy->ShouldKeep(23);
}

TEST(PartitionSelectionWithPreThresholdingTest,
     GaussianPartitionSelectionWithPreThresholdWrapperOK) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  std::unique_ptr<MockGaussianStrategy::Builder> mock_builder =
      absl::make_unique<MockGaussianStrategy::Builder>();
  MockGaussianStrategy* mock_builder_ptr = mock_builder->mock();
  // Check that wrapped ShouldKeep is called with (pre-threshold - 1) removed
  EXPECT_CALL(*mock_builder_ptr, ShouldKeep(14)).Times(1);
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(std::move(mock_builder))
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  strategy->ShouldKeep(23);
}

// Tests that pre-thresholding ShouldKeep deterministically fails with low user
// count.
TEST(PartitionSelectionWithPreThresholdingTest,
     NearTruncatedGeometricPartitionSelectionPreThresholdFail) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kNearTruncatedGeometric)
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ShouldKeep(9)) num_kept++;
  }
  EXPECT_THAT(num_kept, 0);
}

TEST(PartitionSelectionWithPreThresholdingTest,
     LaplacePartitionSelectionPreThresholdFail) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kLaplace)
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ShouldKeep(9)) num_kept++;
  }
  EXPECT_THAT(num_kept, 0);
}

TEST(PartitionSelectionWithPreThresholdingTest,
     GaussianPartitionSelectionPreThresholdFail) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kGaussian)
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ShouldKeep(9)) num_kept++;
  }
  EXPECT_THAT(num_kept, 0);
}

TEST(
    PartitionSelectionWithPreThresholdingTest,
    NearTruncatedGeometricPartitionSelectionPreThresholdZeroProbabilityOfKeep) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kNearTruncatedGeometric)
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int prob_of_keep = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ProbabilityOfKeep(9)) prob_of_keep++;
  }
  EXPECT_THAT(prob_of_keep, 0);
}

TEST(PartitionSelectionWithPreThresholdingTest,
     LaplacePartitionSelectionPreThresholdZeroProbabilityOfKeep) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kLaplace)
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int prob_of_keep = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ProbabilityOfKeep(9)) prob_of_keep++;
  }
  EXPECT_THAT(prob_of_keep, 0);
}

TEST(PartitionSelectionWithPreThresholdingTest,
     GaussianPartitionSelectionPreThresholdZeroProbabilityOfKeep) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kGaussian)
      .SetPreThreshold(10);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int prob_of_keep = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ProbabilityOfKeep(9)) prob_of_keep++;
  }
  EXPECT_THAT(prob_of_keep, 0);
}

// This test is similar to
// `NearTruncatedGeometricPartitionSelectionNumUsersBtwnCrossovers`.
TEST(
    PartitionSelectionWithPreThresholdingTest,
    NearTruncatedGeometricWithPreThresholdingPartitionSelectionNumUsersBtwnCrossovers) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kNearTruncatedGeometric)
      .SetPreThreshold(5);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kNumSamples; i++) {
    if (strategy->ShouldKeep(12)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(12), 0.86807080625, 1e-10);
  EXPECT_THAT(static_cast<double>(num_kept) / kNumSamples,
              DoubleNear(0.86807080625, 0.001));
}

// This test is similar to `LaplacePartitionSelectionAtThreshold`.
// These numbers should make the threshold approximately 5.
// The number of users passed to the laplace partition selection with
// pre-threshold subtraction is near the threshold, so we expect drop/keep is
// 50/50.
TEST(PartitionSelectionWithPreThresholdingTest,
     LaplacePartitionSelectionWithPreThresholdingAtThreshold) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.06766764161)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kLaplace)
      .SetPreThreshold(5);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kSmallNumSamples; i++) {
    if (strategy->ShouldKeep(9)) num_kept++;
  }
  EXPECT_NEAR(strategy->ProbabilityOfKeep(9), 0.5, 1e-10);
  EXPECT_THAT(static_cast<double>(num_kept) / kSmallNumSamples,
              DoubleNear(0.5, 0.0025));
}

// This test is similar to `GaussianPartitionSelectionShouldKeep`.
TEST(PartitionSelectionWithPreThresholdingTest,
     GaussianWithThresholdingPartitionSelectionShouldKeep) {
  PartitionSelectionStrategyWithPreThresholding::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.01)
      .SetMaxPartitionsContributed(1)
      .SetPartitionSelectionStrategy(kGaussian)
      .SetPreThreshold(5);
  auto built_strategy = test_builder.Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  int num_kept = 0;
  for (int i = 0; i < kTinyNumSamples; ++i) {
    if (strategy->ShouldKeep(9)) num_kept++;
  }
  EXPECT_THAT(strategy->ProbabilityOfKeep(9), DoubleNear(0.07, 0.02));
  EXPECT_THAT(static_cast<double>(num_kept) / kTinyNumSamples,
              DoubleNear(0.07, 0.02));
}

// PartitionSelectionPreThresholdParameterBuildTest checks that different
// partition selection strategies are built with different pre_threshold values.
// They also check the value returned by the getter.

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestNearTruncatedGeometricBuildPreThresholdUnset) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5).SetDelta(0.02).SetMaxPartitionsContributed(1);
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), 1);
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestNearTruncatedGeometricBuildPreThreshold1) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPreThreshold(1);
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), 1);
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestNearTruncatedGeometricBuildPreThresholdMax) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPreThreshold(std::numeric_limits<int>::max());
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), std::numeric_limits<int>::max());
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestGaussianBuildPreThresholdUnset) {
  GaussianPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5).SetDelta(0.02).SetMaxPartitionsContributed(1);
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), 1);
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestGaussianBuildPreThreshold1) {
  GaussianPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPreThreshold(1);
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), 1);
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestGaussianBuildPreThresholdMax) {
  GaussianPartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPreThreshold(std::numeric_limits<int>::max());
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), std::numeric_limits<int>::max());
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplaceBuildPreThresholdUnset) {
  LaplacePartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5).SetDelta(0.02).SetMaxPartitionsContributed(1);
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), 1);
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplaceBuildPreThreshold1) {
  LaplacePartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPreThreshold(1);
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), 1);
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplaceBuildPreThresholdMax) {
  LaplacePartitionSelection::Builder test_builder;
  test_builder.SetEpsilon(0.5)
      .SetDelta(0.02)
      .SetMaxPartitionsContributed(1)
      .SetPreThreshold(std::numeric_limits<int>::max());
  auto built = test_builder.Build();
  ASSERT_OK(built);
  EXPECT_EQ(built.value()->GetPreThreshold(), std::numeric_limits<int>::max());
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestNearTruncatedGeometricBuildPreThreshold0) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetEpsilon(0.5)
          .SetDelta(0.02)
          .SetMaxPartitionsContributed(1)
          .SetPreThreshold(0)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "pre_threshold should be either unset or positive, but is 0")));
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestGaussianBuildPreThreshold0) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetEpsilon(0.5)
          .SetDelta(0.02)
          .SetMaxPartitionsContributed(1)
          .SetPreThreshold(0)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "pre_threshold should be either unset or positive, but is 0")));
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestGaussianBuildPreThresholdNegative) {
  GaussianPartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetEpsilon(0.5)
          .SetDelta(0.02)
          .SetMaxPartitionsContributed(1)
          .SetPreThreshold(-42)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "pre_threshold should be either unset or positive, but is -42")));
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplaceBuildPreThreshold0) {
  LaplacePartitionSelection::Builder test_builder;
  EXPECT_THAT(
      test_builder.SetEpsilon(0.5)
          .SetDelta(0.02)
          .SetMaxPartitionsContributed(1)
          .SetPreThreshold(0)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "pre_threshold should be either unset or positive, but is 0")));
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplaceBuildPreThresholdGetThresholdNoChange) {
  LaplacePartitionSelection::Builder test_builder;
  const double kExpThresholdWithoutPre = 7.43775164974;
  const int kPreThreshold = 1;
  auto strategy = test_builder.SetEpsilon(0.5)
                      .SetDelta(0.02)
                      .SetMaxPartitionsContributed(1)
                      .SetPreThreshold(kPreThreshold)
                      .Build();
  ASSERT_OK(strategy);
  LaplacePartitionSelection* laplace =
      dynamic_cast<LaplacePartitionSelection*>(strategy.value().get());
  ASSERT_THAT(laplace, testing::NotNull());
  EXPECT_THAT(laplace->GetThreshold(),
              DoubleNear(kExpThresholdWithoutPre + kPreThreshold - 1, 1e-6));
}

TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplaceBuildPreThresholdGetThresholdChange) {
  LaplacePartitionSelection::Builder test_builder;
  const double kExpThresholdWithoutPre = 7.43775164974;
  const int kPreThreshold = 10;
  auto strategy = test_builder.SetEpsilon(0.5)
                      .SetDelta(0.02)
                      .SetMaxPartitionsContributed(1)
                      .SetPreThreshold(kPreThreshold)
                      .Build();
  ASSERT_OK(strategy);
  LaplacePartitionSelection* laplace =
      dynamic_cast<LaplacePartitionSelection*>(strategy.value().get());
  ASSERT_THAT(laplace, testing::NotNull());
  EXPECT_THAT(laplace->GetThreshold(),
              DoubleNear(kExpThresholdWithoutPre + kPreThreshold - 1, 1e-6));
}

// Check that pre thresholding shifts the argument of ProbabilityOfKeep.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestNearTruncatedGeometricEqFirstCrossoverPreThresholdProbabilityOfKeep) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 6;
  const double kExpProbabilityOfKeep = 0.58840484458;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  std::vector<double> probs;
  for (int num_users = 0; num_users < kPreThreshold; ++num_users) {
    probs.push_back(strategy->ProbabilityOfKeep(num_users));
  }
  EXPECT_THAT(probs, testing::Each(DoubleEq(0)));
  double value = strategy->ProbabilityOfKeep(kNumUsers + kPreThreshold - 1);
  EXPECT_NEAR(value, kExpProbabilityOfKeep, 1e-6);
}

// Check that pre thresholding shifts the argument of ProbabilityOfKeep.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestNearTruncatedGeometricBtwnCrossoversPreThresholdProbabilityOfKeep) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 8;
  const double kExpProbabilityOfKeep = 0.86807080625;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  std::vector<double> probs;
  for (int num_users = 0; num_users < kPreThreshold; ++num_users) {
    probs.push_back(strategy->ProbabilityOfKeep(num_users));
  }
  EXPECT_THAT(probs, testing::Each(DoubleEq(0)));
  double value = strategy->ProbabilityOfKeep(kNumUsers + kPreThreshold - 1);
  EXPECT_NEAR(value, kExpProbabilityOfKeep, 1e-6);
}

// Check that pre thresholding shifts the argument of ProbabilityOfKeep.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestNearTruncatedGeometricExceedsCrossoversPreThresholdProbabilityOfKeep) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 15;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  std::vector<double> probs;
  for (int num_users = 0; num_users < kPreThreshold; ++num_users) {
    probs.push_back(strategy->ProbabilityOfKeep(num_users));
  }
  EXPECT_THAT(probs, testing::Each(DoubleEq(0)));
  double value = strategy->ProbabilityOfKeep(kNumUsers + kPreThreshold - 1);
  EXPECT_EQ(value, 1);
}

// Check that pre thresholding shifts the argument of ProbabilityOfKeep.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplacePreThresholdProbabilityOfKeep) {
  LaplacePartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 9;
  const double kExpProbabilityOfKeep = 0.771054506;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  std::vector<double> probs;
  for (int num_users = 0; num_users < kPreThreshold; ++num_users) {
    probs.push_back(strategy->ProbabilityOfKeep(num_users));
  }
  EXPECT_THAT(probs, testing::Each(DoubleEq(0)));
  double value = strategy->ProbabilityOfKeep(kNumUsers + kPreThreshold - 1);
  EXPECT_NEAR(value, kExpProbabilityOfKeep, 1e-6);
}

// Check that pre thresholding shifts the argument of ProbabilityOfKeep.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestGaussianPreThresholdProbabilityOfKeep) {
  GaussianPartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 9;
  const double kExpProbabilityOfKeep = 0.584958494;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  std::vector<double> probs;
  for (int num_users = 0; num_users < kPreThreshold; ++num_users) {
    probs.push_back(strategy->ProbabilityOfKeep(num_users));
  }
  EXPECT_THAT(probs, testing::Each(DoubleEq(0)));
  double value = strategy->ProbabilityOfKeep(kNumUsers + kPreThreshold - 1);
  EXPECT_NEAR(value, kExpProbabilityOfKeep, 1e-6);
}

// Check that pre thresholding shifts the argument of ShouldKeep:
// NearTruncatedGeometric.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestNearTruncatedGeometricPreThresholdShouldKeep) {
  NearTruncatedGeometricPartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 6;
  const double kExpProbabilityOfKeep = 0.58840484458;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  double kept = 0;
  for (int i = 0; i < kTinyNumSamples; i++) {
    if (strategy->ShouldKeep(kNumUsers + kPreThreshold - 1)) kept++;
  }
  EXPECT_NEAR(kept / kTinyNumSamples, kExpProbabilityOfKeep, 0.1);
}

// Check that pre thresholding shifts the argument of ShouldKeep.
// Laplace.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplacePreThresholdShouldKeep) {
  LaplacePartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 9;
  const double kExpProbabilityOfKeep = 0.771054506;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  double kept = 0;
  for (int i = 0; i < kTinyNumSamples; i++) {
    if (strategy->ShouldKeep(kNumUsers + kPreThreshold - 1)) kept++;
  }
  EXPECT_NEAR(kept / kTinyNumSamples, kExpProbabilityOfKeep, 0.1);
}

// Check that pre thresholding shifts the argument of ShouldKeep.
// Gaussian.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestGaussianPreThresholdShouldKeep) {
  GaussianPartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 9;
  const double kExpProbabilityOfKeep = 0.584958494;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  double kept = 0;
  for (int i = 0; i < kTinyNumSamples; i++) {
    if (strategy->ShouldKeep(kNumUsers + kPreThreshold - 1)) kept++;
  }
  EXPECT_NEAR(kept / kTinyNumSamples, kExpProbabilityOfKeep, 0.1);
}

// Check that pre thresholding shifts the argument of NoiseValueIfShouldKeep.
// Laplace.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestLaplacePreThresholdNoiseValueIfShouldKeep) {
  LaplacePartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 9;
  const double kExpNoiseWithoutPrethreshold = 7.7598163430355003;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  double noise_sum = 0;
  LaplacePartitionSelection* laplace =
      dynamic_cast<LaplacePartitionSelection*>(strategy.get());
  ASSERT_THAT(laplace, testing::NotNull());
  for (int i = 0; i < kTinyNumSamples; i++) {
    noise_sum += laplace->NoiseValueIfShouldKeep(kNumUsers + kPreThreshold - 1)
                     .value_or(kPreThreshold - 1);
  }
  EXPECT_NEAR(noise_sum / kTinyNumSamples,
              kExpNoiseWithoutPrethreshold + kPreThreshold - 1, 0.2);
}

// Check that pre thresholding shifts the argument of NoiseValueIfShouldKeep.
// Gaussian.
TEST(PartitionSelectionPreThresholdParameterBuildTest,
     TestGaussianPreThresholdNoiseValueIfShouldKeep) {
  GaussianPartitionSelection::Builder test_builder;
  const int kPreThreshold = 10, kNumUsers = 9;
  const double kExpNoiseWithoutPrethreshold = 6.4945305204431927;
  auto built_strategy = test_builder.SetEpsilon(0.5)
                            .SetDelta(0.02)
                            .SetMaxPartitionsContributed(1)
                            .SetPreThreshold(kPreThreshold)
                            .Build();
  ASSERT_OK(built_strategy);
  auto strategy = std::move(built_strategy.value());
  double noise_sum = 0;
  GaussianPartitionSelection* gaussian =
      dynamic_cast<GaussianPartitionSelection*>(strategy.get());
  ASSERT_THAT(gaussian, testing::NotNull());
  for (int i = 0; i < kTinyNumSamples; i++) {
    noise_sum += gaussian->NoiseValueIfShouldKeep(kNumUsers + kPreThreshold - 1)
                     .value_or(kPreThreshold - 1);
  }
  EXPECT_NEAR(noise_sum / kTinyNumSamples,
              kExpNoiseWithoutPrethreshold + kPreThreshold - 1, 0.2);
}

}  // namespace
}  // namespace differential_privacy
