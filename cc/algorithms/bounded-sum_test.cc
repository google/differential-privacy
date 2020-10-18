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

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"

namespace differential_privacy {
namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::testing::Eq;
using ::differential_privacy::base::testing::EqualsProto;
using ::differential_privacy::base::testing::IsOkAndHolds;

constexpr double kNumSamples = 10000;

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
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(10)
          .Build();
  ASSERT_OK(bs);
  auto output = (*bs)->Result(a.begin(), a.end());
  ASSERT_OK(output);
  EXPECT_THAT(GetValue<TypeParam>(*output), Eq(static_cast<TypeParam>(10)));
}

TYPED_TEST(BoundedSumTest, BasicIOWithoutIterator) {
  std::vector<TypeParam> a = {0, 0, 10, 10};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
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

TYPED_TEST(BoundedSumTest, RepeatedResultTest) {
  std::vector<TypeParam> a = {0, 0, 10, 10};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(2)
          .Build();
  ASSERT_OK(bs);
  (*bs)->AddEntries(a.begin(), a.end());
  auto output1 = (*bs)->PartialResult(0.5);
  ASSERT_OK(output1);
  auto output2 = (*bs)->PartialResult(0.5);
  ASSERT_OK(output2);
  EXPECT_EQ(GetValue<TypeParam>(*output1), GetValue<TypeParam>(*output2));
}

TYPED_TEST(BoundedSumTest, ClampTest) {
  std::vector<TypeParam> a = {0, 0, 10, 10};
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(2)
          .Build();
  ASSERT_OK(bs);
  auto output = (*bs)->Result(a.begin(), a.end());
  ASSERT_OK(output);
  EXPECT_THAT(GetValue<TypeParam>(*output), Eq(static_cast<TypeParam>(6)));
}

TEST(BoundedSumTest, ConfidenceIntervalTest) {
  double epsilon = 0.5;
  double upperBound = 2;
  double lowerBound = 1;
  double level = .95;
  double budget = .4;
  auto bs = BoundedSum<int>::Builder()
                .SetEpsilon(epsilon)
                .SetLower(lowerBound)
                .SetUpper(upperBound)
                .Build();
  ASSERT_OK(bs);
  ConfidenceInterval wantConfidenceInterval;
  double interval_bound = upperBound * log(1 - level) / epsilon / budget;
  wantConfidenceInterval.set_lower_bound(interval_bound);
  wantConfidenceInterval.set_upper_bound(-interval_bound);
  wantConfidenceInterval.set_confidence_level(level);
  EXPECT_THAT((*bs)->NoiseConfidenceInterval(level, budget),
              IsOkAndHolds(EqualsProto(wantConfidenceInterval)));
  auto result = (*bs)->PartialResult(budget);
  ASSERT_OK(result);
  EXPECT_THAT((*result).error_report().noise_confidence_interval(),
              EqualsProto(wantConfidenceInterval));
}

TYPED_TEST(BoundedSumTest, BoundGetters) {
  int expectedLower = 1, expectedUpper = 2;
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(expectedLower)
          .SetUpper(expectedUpper)
          .Build();
  ASSERT_OK(bs);
  EXPECT_THAT((*bs)->lower(), Eq(expectedLower));
  EXPECT_THAT((*bs)->upper(), Eq(expectedUpper));
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
          .SetEpsilon(1)
          .SetThreshold(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs = BoundedSum<double>::Builder()
                .SetEpsilon(1)
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
  EXPECT_FALSE(typename BoundedSum<TypeParam>::Builder()
                   .SetEpsilon(1.0)
                   .SetLower(std::numeric_limits<TypeParam>::lowest())
                   .SetUpper(std::numeric_limits<TypeParam>::lowest() + 1)
                   .Build()
                   .ok());
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
          CHECK_EQ(bounded_sum.status(), base::OkStatus());
          auto out = (*bounded_sum)->Result(input.begin(), input.end());
          CHECK_EQ(out.status(), base::OkStatus());
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
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>(
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
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
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
      bounds_builder.SetThreshold(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds1);
  auto bs1 =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(*bounds1))
          .Build();
  ASSERT_OK(bs1);
  (*bs1)->AddEntry(-10);
  (*bs1)->AddEntry(4);
  Summary summary = (*bs1)->Serialize();
  (*bs1)->AddEntry(6);

  // Merge summary into second BoundedVariance.
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

TEST(BoundedSumTest, DropNanEntriesManualBounds) {
  std::vector<double> a = {NAN, 1};
  auto bs =
      BoundedSum<double>::Builder()
          .SetLower(0)
          .SetUpper(10)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);
  auto output = (*bs)->Result(a.begin(), a.end());
  ASSERT_OK(output);
  EXPECT_EQ(GetValue<double>(*output), 1.0);
}

TEST(BoundedSumTest, DropNanEntriesApproxBounds) {
  std::vector<double> a = {NAN, 1};
  auto bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      BoundedSum<double>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(*bounds))
          .Build();
  ASSERT_OK(bs);

  // Bounds are set to [-1, 1].
  auto output = (*bs)->Result(a.begin(), a.end());
  ASSERT_OK(output);
  EXPECT_EQ(GetValue<double>(*output), 1.0);
}

TYPED_TEST(BoundedSumTest, PropagateApproxBoundsError) {
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bs);

  // Automatic bounds are needed but there is no input, so the count-threshhold
  // should exceed any bin count.
  EXPECT_FALSE((*bs)->PartialResult().ok());
}

// Test when 0 is in [lower, upper].
TYPED_TEST(BoundedSumTest, AutomaticBoundsContainZero) {
  std::vector<TypeParam> a = {0,
                              0,
                              8,
                              8,
                              std::numeric_limits<TypeParam>::lowest(),
                              std::numeric_limits<TypeParam>::max()};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
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
          .SetEpsilon(1)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
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
          .SetEpsilon(1)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
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
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
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
          .SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetThreshold(1)
          .Build();
  ASSERT_OK(bounds);
  auto bs =
      typename BoundedSum<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
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
  auto bounds_small =
      typename ApproxBounds<TypeParam>::Builder().SetNumBins(1).Build();
  ASSERT_OK(bounds_small);
  auto bounds_big =
      typename ApproxBounds<TypeParam>::Builder().SetNumBins(1).Build();
  ASSERT_OK(bounds_big);

  auto bs_small = typename BoundedSum<TypeParam>::Builder()
                      .SetApproxBounds(std::move(*bounds_small))
                      .Build();
  ASSERT_OK(bs_small);
  auto bs_big = typename BoundedSum<TypeParam>::Builder()
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

  EXPECT_NEAR((*bs)->GetEpsilon(), epsilon, 1e-10);
  EXPECT_NEAR((*bs)->GetEpsilon(),
              (*bs)->GetBoundingEpsilon() + (*bs)->GetAggregationEpsilon(),
              1e-10);
  EXPECT_GT((*bs)->GetBoundingEpsilon(), 0);
  EXPECT_LT((*bs)->GetBoundingEpsilon(), epsilon);
  EXPECT_GT((*bs)->GetAggregationEpsilon(), 0);
  EXPECT_LT((*bs)->GetAggregationEpsilon(), epsilon);
}

}  //  namespace
}  // namespace differential_privacy
