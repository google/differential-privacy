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

#include "algorithms/bounded-sum.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/approx-bounds-as-bounds-provider.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "proto/data.pb.h"

namespace differential_privacy {
namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::testing::DoubleEq;
using ::testing::Eq;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::differential_privacy::base::testing::IsOk;
using ::differential_privacy::base::testing::IsOkAndHolds;
using ::differential_privacy::base::testing::StatusIs;

constexpr double kNumSamples = 10000;
constexpr double kDefaultEpsilon = 1.1;

template <typename T>
class BoundedSumTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Typed test to iterate all test cases through all supported versions of
// BoundedSumTest, currently (int64_t, double).
typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(BoundedSumTest, NumericTypes);

TYPED_TEST(BoundedSumTest, BasicIO) {
  std::vector<TypeParam> a = {1, 2, 3, 4};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon)
          .SetLower(0)
          .SetUpper(10)
          .Build();
  ASSERT_OK(bs);
  auto output = (*bs)->Result(a.begin(), a.end());
  ASSERT_OK(output);
  EXPECT_THAT(GetValue<TypeParam>(*output), Eq(static_cast<TypeParam>(10)));
}

TYPED_TEST(BoundedSumTest, BasicIOWithLaplaceReturnsOk) {
  std::vector<TypeParam> a = {1, 2, 3, 4};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<LaplaceMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(10)
          .Build();
  ASSERT_OK(bs);
  auto output = (*bs)->Result(a.begin(), a.end());
  EXPECT_OK(output);
}

TYPED_TEST(BoundedSumTest, BasicIOWithGaussianReturnsOk) {
  std::vector<TypeParam> a = {1, 2, 3, 4};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<GaussianMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetDelta(0.0001)
          .SetLower(0)
          .SetUpper(10)
          .Build();
  ASSERT_OK(bs);
  auto output = (*bs)->Result(a.begin(), a.end());
  EXPECT_OK(output);
}

TYPED_TEST(BoundedSumTest, GaussianWithoutDeltaFails) {
  std::vector<TypeParam> a = {1, 2, 3, 4};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<GaussianMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(10)
          .Build();
  ASSERT_THAT(bs,
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Delta")));
}

TYPED_TEST(BoundedSumTest, BasicIOWithoutIterator) {
  std::vector<TypeParam> a = {0, 0, 10, 10};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(2)
          .Build();
  ASSERT_OK(bs);
  for (const auto& input : a) {
    (*bs)->AddEntry(input);
  }
  auto output = (*bs)->PartialResult();
  ASSERT_OK(output);
  EXPECT_THAT(GetValue<TypeParam>(*output), Eq(static_cast<TypeParam>(6)));
}

TEST(BoundedSumTest, AddEntryInvalidInputIgnored) {
  std::vector<float> a = {0, 0, std::numeric_limits<float>::quiet_NaN(), 10,
                          10};
  auto bs =
      typename BoundedSum<float>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(2)
          .Build();
  ASSERT_OK(bs);
  for (const auto& input : a) {
    (*bs)->AddEntry(input);
  }
  auto output = (*bs)->PartialResult();
  ASSERT_OK(output);
  EXPECT_FLOAT_EQ(GetValue<float>(*output), 6);
}

TYPED_TEST(BoundedSumTest, InsufficientPrivacyBudgetTest) {
  std::vector<TypeParam> a = {0, 0, 10, 10};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(10)
          .Build();
  ASSERT_OK(bs);
  (*bs)->AddEntries(a.begin(), a.end());
  ASSERT_OK((*bs)->PartialResult());
  EXPECT_THAT((*bs)->PartialResult(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("can only produce results once")));
}

TYPED_TEST(BoundedSumTest, RepeatedResultsTest) {
  std::vector<TypeParam> a = {1, 2, 3, 4};
  typename BoundedSum<TypeParam>::Builder builder;
  builder.SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
      .SetEpsilon(1.0)
      .SetLower(0)
      .SetUpper(10);
  absl::StatusOr<std::unique_ptr<BoundedSum<TypeParam>>> bs1 = builder.Build();
  ASSERT_OK(bs1);
  absl::StatusOr<std::unique_ptr<BoundedSum<TypeParam>>> bs2 = builder.Build();
  ASSERT_OK(bs2);
  (*bs1)->AddEntries(a.begin(), a.end());
  (*bs2)->AddEntries(a.begin(), a.end());
  auto output1 = (*bs1)->PartialResult();
  ASSERT_OK(output1);
  auto output2 = (*bs2)->PartialResult();
  ASSERT_OK(output2);
  EXPECT_EQ(GetValue<TypeParam>(*output1), GetValue<TypeParam>(*output2));
}

TYPED_TEST(BoundedSumTest, BuildWithMoreApproxBoundsEpsilonThanEpsilonFails) {
  auto bounds = typename ApproxBounds<TypeParam>::Builder()
                    .SetEpsilon(kDefaultEpsilon)
                    .Build();
  ASSERT_OK(bounds);
  auto bs = typename BoundedSum<TypeParam>::Builder()
                .SetEpsilon(kDefaultEpsilon)
                .SetApproxBounds(std::move(bounds.value()))
                .Build();
  EXPECT_THAT(
      bs.status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Approx Bounds consumes more epsilon budget than available")));
}

TYPED_TEST(BoundedSumTest, ClampTest) {
  std::vector<TypeParam> a = {0, 0, 10, 10};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(2)
          .Build();
  ASSERT_OK(bs);
  auto output = (*bs)->Result(a.begin(), a.end());
  ASSERT_OK(output);
  EXPECT_THAT(GetValue<TypeParam>(*output), Eq(static_cast<TypeParam>(6)));
}

TEST(BoundedSumTest, ConfidenceIntervalWithLaplaceTest) {
  double epsilon = 0.5;
  double upperBound = 2;
  double lowerBound = 1;
  double level = .95;
  auto bs =
      BoundedSum<int>::Builder()
          .SetEpsilon(epsilon)
          .SetLaplaceMechanism(std::make_unique<LaplaceMechanism::Builder>())
          .SetLower(lowerBound)
          .SetUpper(upperBound)
          .Build();
  ASSERT_OK(bs);
  ConfidenceInterval wantConfidenceInterval;
  double interval_bound = upperBound * std::log(1 - level) / epsilon;
  wantConfidenceInterval.set_lower_bound(interval_bound);
  wantConfidenceInterval.set_upper_bound(-interval_bound);
  wantConfidenceInterval.set_confidence_level(level);
  EXPECT_THAT((*bs)->NoiseConfidenceInterval(level),
              IsOkAndHolds(EqualsProto(wantConfidenceInterval)));
  auto result = (*bs)->PartialResult();
  ASSERT_OK(result);
  EXPECT_THAT(GetNoiseConfidenceInterval(*result),
              EqualsProto(wantConfidenceInterval));
}

TYPED_TEST(BoundedSumTest, BoundGettersForFixedBounds) {
  int expectedLower = 1, expectedUpper = 2;
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(expectedLower)
          .SetUpper(expectedUpper)
          .Build();
  ASSERT_OK(bs);
  EXPECT_THAT((*bs)->lower().value(), Eq(expectedLower));
  EXPECT_THAT((*bs)->upper().value(), Eq(expectedUpper));
}

TYPED_TEST(BoundedSumTest, BoundGettersForApproxBounds) {
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .Build();
  ASSERT_OK(bs);
  EXPECT_FALSE((*bs)->lower().has_value());
  EXPECT_FALSE((*bs)->upper().has_value());
}

TYPED_TEST(BoundedSumTest, SensitivityTooHigh) {
  // Increase the lower bound by one so that taking the magnitude wont overflow.
  EXPECT_EQ(BoundedSum<double>::Builder()
                .SetEpsilon(1.0)
                .SetLower(std::numeric_limits<double>::lowest() + 1)
                .SetUpper(std::numeric_limits<double>::max())
                .Build()
                .status()
                .message(),
            "Sensitivity is too high.");
}

TYPED_TEST(BoundedSumTest, SensitivityTooHighApproxBounds) {
  auto bounds =
      ApproxBounds<double>::Builder()
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetThresholdForTest(1)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs = BoundedSum<double>::Builder()
                .SetEpsilon(kDefaultEpsilon)
                .SetApproxBounds(std::move(*bounds))
                .Build();
  ASSERT_OK(bs);
  // Divide lower numeric limit by 4 so that the auto-determined lower bound
  // will not be the lower numeric limit and absolute value wont overflow.
  (*bs)->AddEntry(std::numeric_limits<double>::lowest() / 4);
  (*bs)->AddEntry(std::numeric_limits<double>::max());
  EXPECT_EQ((*bs)->PartialResult().status().message(),
            "Sensitivity is too high.");
}

TYPED_TEST(BoundedSumTest, LowerBoundMagnitudeOverflows) {
  EXPECT_THAT(typename BoundedSum<TypeParam>::Builder()
                  .SetEpsilon(1.0)
                  .SetLower(std::numeric_limits<TypeParam>::lowest())
                  .SetUpper(std::numeric_limits<TypeParam>::lowest() + 1)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TYPED_TEST(BoundedSumTest, MaxContributionsVarianceTest) {
  // Use following inputs with sum 0.
  const std::vector<TypeParam> input = {1, -1, 1, -1, 0, 0, 0};

  std::function<TypeParam(int)> sample_variance_for_max_contributions =
      [&input](int max_contributions) {
        double sum = 0;
        for (int i = 0; i < kNumSamples; ++i) {
          auto bounded_sum =
              typename BoundedSum<TypeParam>::Builder()
                  .SetMaxContributionsPerPartition(max_contributions)
                  .SetEpsilon(1)
                  .SetLower(-1)
                  .SetUpper(1)
                  .Build();
          CHECK_EQ(bounded_sum.status(), absl::OkStatus());
          auto out = (*bounded_sum)->Result(input.begin(), input.end());
          CHECK_EQ(out.status(), absl::OkStatus());
          sum += std::pow(GetValue<TypeParam>(*out), 2);
        }
        return sum / (kNumSamples - 1);
      };

  // We expect the sample variance with max contribution 2 to be (significantly)
  // bigger than with max contribution 1.
  EXPECT_GT(sample_variance_for_max_contributions(2),
            1.1 * sample_variance_for_max_contributions(1));
}

TYPED_TEST(BoundedSumTest, SetZeroNoiseMechanismBuilder) {
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(10)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>(
              ZeroNoiseMechanism::Builder()))
          .Build();
  ASSERT_OK(bs);
  (*bs)->AddEntry(1);
  auto output = (*bs)->PartialResult();
  ASSERT_OK(output);
  EXPECT_EQ(GetValue<TypeParam>(*output), 1);
}

TYPED_TEST(BoundedSumTest, SerializeMergeTest) {
  typename BoundedSum<TypeParam>::Builder builder;

  // Get summary of first BoundedSum between entries.
  auto bs1 =
      builder.SetLower(0)
          .SetUpper(3)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs1);
  (*bs1)->AddEntry(-2);
  Summary summary = (*bs1)->Serialize();
  (*bs1)->AddEntry(6);

  // Merge summary into second BoundedVariance.
  auto bs2 = builder.Build();
  ASSERT_OK(bs2);
  (*bs2)->AddEntry(6);
  EXPECT_OK((*bs2)->Merge(summary));

  // Check equality.
  auto output1 = (*bs1)->PartialResult();
  ASSERT_OK(output1);
  auto output2 = (*bs2)->PartialResult();
  ASSERT_OK(output2);
  EXPECT_EQ(GetValue<TypeParam>(*output1), GetValue<TypeParam>(*output2));
}

TYPED_TEST(BoundedSumTest, SerializeMergePartialSumsTest) {
  typename ApproxBounds<TypeParam>::Builder bounds_builder;
  typename BoundedSum<TypeParam>::Builder builder;

  // BoundedSums have automatic bounding, so entries will be split and stored as
  // partial sums.
  auto bounds1 =
      bounds_builder.SetThresholdForTest(0.5)
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetNumBins(50)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds1);
  auto bs1 =
      builder.SetEpsilon(kDefaultEpsilon)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(*bounds1))
          .Build();
  ASSERT_OK(bs1);
  (*bs1)->AddEntry(-10);
  (*bs1)->AddEntry(4);
  Summary summary = (*bs1)->Serialize();
  (*bs1)->AddEntry(6);

  // Merge summary into second BoundedSum.
  auto bounds2 = bounds_builder.Build();
  ASSERT_OK(bounds2);
  auto bs2 = builder.SetApproxBounds(std::move(*bounds2)).Build();
  ASSERT_OK(bs2);
  (*bs2)->AddEntry(6);
  EXPECT_OK((*bs2)->Merge(summary));

  // Check equality. Bounds are set to [-16, 16].
  auto output1 = (*bs1)->PartialResult();
  ASSERT_OK(output1);
  auto output2 = (*bs2)->PartialResult();
  ASSERT_OK(output2);
  EXPECT_EQ(GetValue<TypeParam>(*output1), GetValue<TypeParam>(*output2));
}

// This test will be removed when removing backwards compatibility for the
// `bounds_summary` field.
TYPED_TEST(BoundedSumTest, SerializeMergeApproxBoundsBackwardsCompatability) {
  absl::StatusOr<std::unique_ptr<BoundedSum<TypeParam>>> bounds1 =
      typename BoundedSum<TypeParam>::Builder().SetEpsilon(1e10).Build();
  ASSERT_OK(bounds1.status());

  for (int i = 0; i < 100; ++i) {
    bounds1.value()->AddEntry(10);
  }

  absl::StatusOr<std::unique_ptr<BoundedSum<TypeParam>>> bounds2 =
      typename BoundedSum<TypeParam>::Builder().SetEpsilon(1.0).Build();
  ASSERT_OK(bounds2.status());

  // Remove the newly introduced field as this field is ignored by versions
  // before the proto change.
  Summary bounds1_summary = bounds1.value()->Serialize();
  BoundedSumSummary bs_summary;
  bounds1_summary.data().UnpackTo(&bs_summary);
  bs_summary.clear_bounds();
  bounds1_summary.mutable_data()->PackFrom(bs_summary);

  ASSERT_OK(bounds2.value()->Merge(bounds1_summary));
  EXPECT_THAT(bounds2.value()->PartialResult(), IsOk());
}

TEST(BoundedSumTest, OverflowFromAddNoiseTypeCast) {
  // Overflowing should result in the sum + noise eventually wrapping around and
  // become negative.
  int i;
  for (i = 0; i < 100; ++i) {
    auto bs =
        typename BoundedSum<int64_t>::Builder()
            .SetLaplaceMechanism(std::make_unique<LaplaceMechanism::Builder>())
            .SetEpsilon(10.0)  // Large epsilon avoids noise > INT64_MAX
            .SetLower(1)
            .SetUpper(std::numeric_limits<int64_t>::max())
            .Build();
    ASSERT_OK(bs);
    (*bs)->AddEntry(std::numeric_limits<int64_t>::max());
    auto result = (*bs)->PartialResult();
    ASSERT_OK(result);
    if (GetValue<int64_t>(result.value()) < 0) {
      // An overflow has happened, so return to end the test as a success.
      return;
    }
  }
  FAIL() << "No overflow occurred after " << i << " iterations.";
}

TEST(BoundedSumTest, OverflowAddEntryManualBounds) {
  typename BoundedSum<int64_t>::Builder builder;

  absl::StatusOr<std::unique_ptr<BoundedSum<int64_t>>> bs =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(std::numeric_limits<int64_t>::max())
          .Build();
  ASSERT_OK(bs);
  (*bs)->AddEntry(std::numeric_limits<int64_t>::max());
  (*bs)->AddEntry(1);
  (*bs)->AddEntry(1);
  (*bs)->AddEntry(std::numeric_limits<int64_t>::max());

  absl::StatusOr<Output> result = (*bs)->PartialResult();
  ASSERT_OK(result);
  // Overflowing should result in the running sum wrapping around to zero.
  EXPECT_EQ(GetValue<int64_t>(result.value()), 0);
}

TEST(BoundedSumTest, UnderflowAddEntryManualBounds) {
  typename BoundedSum<int64_t>::Builder builder;

  absl::StatusOr<std::unique_ptr<BoundedSum<int64_t>>> bs =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(std::numeric_limits<int64_t>::lowest() + 1)
          .SetUpper(0)
          .Build()
          .value();
  (*bs)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);
  (*bs)->AddEntry(-1);
  (*bs)->AddEntry(-1);
  (*bs)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);

  absl::StatusOr<Output> result = (*bs)->PartialResult();
  ASSERT_OK(result);
  // Underflowing should result in the running sum wrapping around to zero.
  EXPECT_EQ(GetValue<int64_t>(result.value()), 0);
}

TEST(BoundedSumTest, OverflowMergeManualBoundsTest) {
  typename BoundedSum<int64_t>::Builder builder;

  absl::StatusOr<std::unique_ptr<BoundedSum<int64_t>>> bs =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(std::numeric_limits<int64_t>::max())
          .Build();
  ASSERT_OK(bs);
  (*bs)->AddEntry(std::numeric_limits<int64_t>::max());
  Summary summary = (*bs)->Serialize();

  absl::StatusOr<std::unique_ptr<BoundedSum<int64_t>>> bs2 = builder.Build();
  ASSERT_OK(bs2);
  (*bs2)->AddEntry(1);
  (*bs2)->AddEntry(1);
  (*bs2)->AddEntry(std::numeric_limits<int64_t>::max());

  ASSERT_OK((*bs2)->Merge(summary));

  absl::StatusOr<Output> result = (*bs2)->PartialResult();
  EXPECT_OK(result);
  // Overflowing should result in the running sum wrapping around to zero.
  EXPECT_EQ(GetValue<int64_t>(result.value()), 0);

  // Test post-overflow serialize & merge
  summary = (*bs2)->Serialize();
  bs2 = builder.Build();
  EXPECT_OK((*bs2)->Merge(summary));
  result = (*bs2)->PartialResult();
  EXPECT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0);
}

TEST(BoundedSumTest, UnderflowMergeManualBoundsTest) {
  typename BoundedSum<int64_t>::Builder builder;

  absl::StatusOr<std::unique_ptr<BoundedSum<int64_t>>> bs =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(std::numeric_limits<int64_t>::lowest() + 1)
          .SetUpper(0)
          .Build()
          .value();
  (*bs)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);
  Summary summary = (*bs)->Serialize();

  absl::StatusOr<std::unique_ptr<BoundedSum<int64_t>>> bs2 = builder.Build();
  ASSERT_OK(bs2);
  (*bs2)->AddEntry(-1);
  (*bs2)->AddEntry(-1);
  (*bs2)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);

  EXPECT_OK((*bs2)->Merge(summary));

  absl::StatusOr<Output> result = (*bs2)->PartialResult();
  EXPECT_OK(result);
  // Underflowing should result in the running sum wrapping around to zero.
  EXPECT_EQ(GetValue<int64_t>(result.value()), 0);

  // Test post-overflow serialize & merge
  summary = (*bs2)->Serialize();
  bs2 = builder.Build();
  EXPECT_OK((*bs2)->Merge(summary));
  result = (*bs2)->PartialResult();
  EXPECT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0);
}

TEST(BoundedSumTest, DropNanEntriesManualBounds) {
  std::vector<double> a = {NAN, 1};
  auto bs =
      BoundedSum<double>::Builder()
          .SetLower(0)
          .SetUpper(10)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);
  auto output = (*bs)->Result(a.begin(), a.end());
  ASSERT_OK(output);
  EXPECT_DOUBLE_EQ(GetValue<double>(*output), 1.0);
}

TEST(BoundedSumTest, DropNanEntriesApproxBounds) {
  std::vector<double> a = {NAN, 1};
  auto bounds =
      ApproxBounds<double>::Builder()
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(0.5)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      BoundedSum<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(*bounds))
          .Build();
  ASSERT_OK(bs);

  // Bounds are set to [-1, 1].
  auto output = (*bs)->Result(a.begin(), a.end());
  ASSERT_OK(output);
  EXPECT_DOUBLE_EQ(GetValue<double>(*output), 1.0);
}

TYPED_TEST(BoundedSumTest, PropagateApproxBoundsError) {
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);

  // Automatic bounds are needed but there is no input, so the count-threshold
  // should exceed any bin count.
  EXPECT_FALSE((*bs)->PartialResult().ok());
}

// Test when 0 is in [lower, upper] with ApproxBounds.
TYPED_TEST(BoundedSumTest, AutomaticBoundsContainZero) {
  std::vector<TypeParam> a = {0,
                              0,
                              8,
                              8,
                              std::numeric_limits<TypeParam>::lowest(),
                              std::numeric_limits<TypeParam>::max()};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1.5)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);
  (*bs)->AddEntries(a.begin(), a.end());
  auto output = (*bs)->PartialResult();
  ASSERT_OK(output);

  BoundingReport expected_report;
  SetValue<TypeParam>(expected_report.mutable_lower_bound(), -8);
  SetValue<TypeParam>(expected_report.mutable_upper_bound(), 8);
  expected_report.set_num_inputs(a.size());
  expected_report.set_num_outside(2);

  EXPECT_EQ(GetValue<TypeParam>(output->elements(0).value()), 16);
  EXPECT_THAT(output->error_report().bounding_report(),
              EqualsProto(expected_report));
}

TYPED_TEST(BoundedSumTest, AutomaticBoundsNegative) {
  std::vector<TypeParam> a = {9, -2, -2, -4, -6, -6};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);
  (*bs)->AddEntries(a.begin(), a.end());
  auto output = (*bs)->PartialResult();
  ASSERT_OK(output);

  BoundingReport expected_report;
  SetValue<TypeParam>(expected_report.mutable_lower_bound(), -8);
  SetValue<TypeParam>(expected_report.mutable_upper_bound(), 8);
  expected_report.set_num_inputs(a.size());
  expected_report.set_num_outside(1);

  // 9 gets clamped to 8.
  EXPECT_EQ(GetValue<TypeParam>(output->elements(0).value()), -12);
  EXPECT_THAT(output->error_report().bounding_report(),
              EqualsProto(expected_report));
}

TYPED_TEST(BoundedSumTest, AutomaticBoundsPositive) {
  std::vector<TypeParam> a = {-9, 2, 2, 4, 6, 6};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1.5)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);
  (*bs)->AddEntries(a.begin(), a.end());
  auto output = (*bs)->PartialResult();
  ASSERT_OK(output);

  BoundingReport expected_report;
  SetValue<TypeParam>(expected_report.mutable_lower_bound(), -8);
  SetValue<TypeParam>(expected_report.mutable_upper_bound(), 8);
  expected_report.set_num_inputs(a.size());
  expected_report.set_num_outside(1);

  // -9 gets clamped to -8.
  EXPECT_EQ(GetValue<TypeParam>(output->elements(0).value()), 12);
  EXPECT_THAT(output->error_report().bounding_report(),
              EqualsProto(expected_report));
}

// Test not providing ApproxBounds and instead using the default.
TYPED_TEST(BoundedSumTest, AutomaticBoundsDefault) {
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);

  // Threshold is below 100.
  std::vector<TypeParam> big(1001, 10);
  std::vector<TypeParam> small(1000, -10);
  (*bs)->AddEntries(big.begin(), big.end());
  (*bs)->AddEntries(small.begin(), small.end());
  auto output = (*bs)->PartialResult();
  ASSERT_OK(output);

  BoundingReport expected_report;
  SetValue<TypeParam>(expected_report.mutable_lower_bound(), -16);
  SetValue<TypeParam>(expected_report.mutable_upper_bound(), 16);
  expected_report.set_num_inputs(big.size() + small.size());
  expected_report.set_num_outside(0);

  EXPECT_NEAR(GetValue<TypeParam>(output->elements(0).value()), 10.0,
              std::pow(10, -10));
  EXPECT_THAT(output->error_report().bounding_report(),
              EqualsProto(expected_report));
}

TYPED_TEST(BoundedSumTest, Reset) {
  // Construct bounded sum with approximate bounding.
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetThresholdForTest(0.5)
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);

  // Reset between adding vectors.
  std::vector<TypeParam> a = {-10, 1000};
  std::vector<TypeParam> b = {-100, 100, 1};
  (*bs)->AddEntries(a.begin(), a.end());
  (*bs)->Reset();
  (*bs)->AddEntries(b.begin(), b.end());

  // Check result is only affected by vector b.
  auto result = (*bs)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<TypeParam>(result->elements(0).value()), 1);
}

TYPED_TEST(BoundedSumTest, Memory) {
  auto bounds_small = typename ApproxBounds<TypeParam>::Builder()
                          .SetEpsilon(kDefaultEpsilon / 2)
                          .SetNumBins(1)
                          .Build();
  ASSERT_OK(bounds_small);
  auto bounds_big = typename ApproxBounds<TypeParam>::Builder()
                        .SetEpsilon(kDefaultEpsilon / 2)
                        .SetNumBins(1)
                        .Build();
  ASSERT_OK(bounds_big);

  auto bs_small = typename BoundedSum<TypeParam>::Builder()
                      .SetEpsilon(kDefaultEpsilon)
                      .SetApproxBounds(std::move(*bounds_small))
                      .Build();
  ASSERT_OK(bs_small);
  auto bs_big = typename BoundedSum<TypeParam>::Builder()
                    .SetEpsilon(kDefaultEpsilon)
                    .SetApproxBounds(std::move(*bounds_big))
                    .Build();
  ASSERT_OK(bs_big);

  EXPECT_GE((*bs_big)->MemoryUsed(), (*bs_small)->MemoryUsed());
}

TYPED_TEST(BoundedSumTest, SplitsEpsilonWithAutomaticBounds) {
  double epsilon = 1.0;

  auto bs =
      typename BoundedSum<TypeParam>::Builder().SetEpsilon(epsilon).Build();
  ASSERT_OK(bs);

  auto* sum_with_approx_bounds =
      static_cast<BoundedSumWithApproxBounds<TypeParam>*>((*bs).get());

  EXPECT_NEAR(sum_with_approx_bounds->GetEpsilon(), epsilon, 1e-10);
  EXPECT_NEAR(sum_with_approx_bounds->GetEpsilon(),
              sum_with_approx_bounds->GetBoundingEpsilon() +
                  sum_with_approx_bounds->GetAggregationEpsilon(),
              1e-10);
  EXPECT_GT(sum_with_approx_bounds->GetBoundingEpsilon(), 0);
  EXPECT_LT(sum_with_approx_bounds->GetBoundingEpsilon(), epsilon);
  EXPECT_GT(sum_with_approx_bounds->GetAggregationEpsilon(), 0);
  EXPECT_LT(sum_with_approx_bounds->GetAggregationEpsilon(), epsilon);
}

TEST(BoundedSumTest, ApproxBoundsSumHasExpectedMaxPartitionsContributed) {
  const int max_contributions_per_partition = 1234;
  absl::StatusOr<std::unique_ptr<BoundedSum<double>>> bs =
      BoundedSum<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetMaxContributionsPerPartition(max_contributions_per_partition)
          .Build();
  ASSERT_OK(bs);

  auto* bs_with_approx_bounds =
      static_cast<BoundedSumWithApproxBounds<double>*>(bs.value().get());
  ASSERT_THAT(bs_with_approx_bounds, NotNull());

  EXPECT_THAT(
      bs_with_approx_bounds->GetMaxContributionsPerPartitionForTesting(),
      Eq(max_contributions_per_partition));
}

TEST(BoundedSumTest, ApproxBoundsSumHasExpectedL0Sensitivity) {
  const int max_partitions_contributed = 1234;
  absl::StatusOr<std::unique_ptr<BoundedSum<double>>> bs =
      BoundedSum<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetMaxPartitionsContributed(max_partitions_contributed)
          .Build();
  ASSERT_OK(bs);

  auto* bs_with_approx_bounds =
      static_cast<BoundedSumWithApproxBounds<double>*>(bs.value().get());
  ASSERT_THAT(bs_with_approx_bounds, NotNull());

  EXPECT_THAT(bs_with_approx_bounds->GetL0SensitivityForTesting(),
              DoubleEq(max_partitions_contributed));
}

TEST(BoundedSumTest, ApproxBoundsMechanismHasExpectedVariance) {
  const int max_partitions_contributed = 2;
  const int max_contributions_per_partition = 3;
  const double expected_variance =
      LaplaceMechanism::Builder()
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetL0Sensitivity(max_partitions_contributed)
          .SetLInfSensitivity(max_contributions_per_partition)
          .Build()
          .value()
          ->GetVariance();

  absl::StatusOr<std::unique_ptr<BoundedSum<double>>> bs =
      BoundedSum<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetMaxPartitionsContributed(max_partitions_contributed)
          .SetMaxContributionsPerPartition(max_contributions_per_partition)
          .Build();
  ASSERT_OK(bs);

  auto* bs_with_approx_bounds =
      static_cast<BoundedSumWithApproxBounds<double>*>(bs.value().get());
  ASSERT_THAT(bs_with_approx_bounds, NotNull());
  auto* approx_bounds_as_bounds_provider =
      static_cast<ApproxBoundsAsBoundsProvider<double>*>(
          bs_with_approx_bounds->GetBoundsProviderForTesting());
  ASSERT_THAT(approx_bounds_as_bounds_provider, NotNull());

  EXPECT_THAT(approx_bounds_as_bounds_provider->GetApproxBoundsForTesting()
                  ->GetMechanismForTesting()
                  ->GetVariance(),
              DoubleEq(expected_variance));
}

TEST(BoundedSumTest, ApproxBoundsOnInt64LowestUsesFullRangeBounds) {
  absl::StatusOr<std::unique_ptr<BoundedSum<int64_t>>> bs =
      BoundedSum<int64_t>::Builder().SetEpsilon(kDefaultEpsilon).Build();
  ASSERT_OK(bs);

  for (int i = 0; i < 1000; ++i) {
    bs.value()->AddEntry(std::numeric_limits<int64_t>::lowest());
  }

  absl::StatusOr<Output> result = bs.value()->PartialResult();
  ASSERT_OK(result);

  EXPECT_EQ(
      GetValue<int64_t>(result->error_report().bounding_report().lower_bound()),
      std::numeric_limits<int64_t>::lowest());
  EXPECT_EQ(
      GetValue<int64_t>(result->error_report().bounding_report().upper_bound()),
      std::numeric_limits<int64_t>::max());
}

TEST(BoundedSumTest, BuilderReturnsErrorWhenApproxBoundsAndBoundsProviderSet) {
  auto approx_bounds1 = ApproxBounds<int64_t>::Builder().SetEpsilon(1).Build();
  ASSERT_OK(approx_bounds1.status());
  auto approx_bounds2 = ApproxBounds<int64_t>::Builder().SetEpsilon(1).Build();
  ASSERT_OK(approx_bounds2.status());

  EXPECT_THAT(BoundedSum<int64_t>::Builder()
                  .SetEpsilon(3)
                  .SetApproxBounds(std::move(approx_bounds1).value())
                  .SetBoundsProvider(
                      std::make_unique<ApproxBoundsAsBoundsProvider<int64_t>>(
                          std::move(approx_bounds2).value()))
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("only one is allowed")));
}

TEST(BoundedSumTest, BuilderUsesBoundsProvider) {
  auto approx_bounds =
      ApproxBounds<int64_t>::Builder().SetEpsilon(1.23).Build();
  ASSERT_OK(approx_bounds.status());
  auto bounded_sum =
      BoundedSum<int64_t>::Builder()
          .SetEpsilon(2)
          .SetBoundsProvider(
              std::make_unique<ApproxBoundsAsBoundsProvider<int64_t>>(
                  std::move(approx_bounds).value()))
          .Build();
  ASSERT_OK(bounded_sum.status());
  auto* bounded_sum_with_approx_bounds =
      dynamic_cast<BoundedSumWithApproxBounds<int64_t>*>(bounded_sum->get());
  ASSERT_THAT(bounded_sum_with_approx_bounds, NotNull());

  EXPECT_THAT(bounded_sum_with_approx_bounds->GetBoundsProviderForTesting()
                  ->GetEpsilon(),
              Eq(1.23));
}

}  //  namespace
}  // namespace differential_privacy
