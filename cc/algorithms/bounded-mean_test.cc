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

#include "algorithms/bounded-mean.h"

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/numerical-mechanisms-testing.h"

namespace differential_privacy {
namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::differential_privacy::base::testing::EqualsProto;

constexpr double kSmallEpsilon = 0.00000001;
constexpr double kNumSamples = 10000;

template <typename T>
class BoundedMeanTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(BoundedMeanTest, NumericTypes);

TYPED_TEST(BoundedMeanTest, BasicTest) {
  std::vector<TypeParam> a = {2, 4, 6, 8};
  std::unique_ptr<BoundedMean<TypeParam>> mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .Build()
          .ValueOrDie();
  Output result = mean->Result(a.begin(), a.end()).ValueOrDie();
  EXPECT_GE(GetValue<double>(result), 1);
  EXPECT_LE(GetValue<double>(result), 9);
}

TYPED_TEST(BoundedMeanTest, RepeatedResultTest) {
  std::vector<TypeParam> a = {2, 4, 6, 8};

  std::unique_ptr<BoundedMean<TypeParam>> mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .Build()
          .ValueOrDie();
  mean->AddEntries(a.begin(), a.end());

  EXPECT_EQ(GetValue<double>(mean->PartialResult(0.5).ValueOrDie()),
            GetValue<double>(mean->PartialResult(0.5).ValueOrDie()));
}

TYPED_TEST(BoundedMeanTest, BasicTestWithoutIterator) {
  std::vector<TypeParam> a = {2, 4, 6, 8};
  std::unique_ptr<BoundedMean<TypeParam>> mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .Build()
          .ValueOrDie();
  for (const auto& input : a) {
    mean->AddEntry(input);
  }
  Output result = mean->PartialResult().ValueOrDie();
  EXPECT_GE(GetValue<double>(result), 1);
  EXPECT_LE(GetValue<double>(result), 9);
}

// This test verifies that BoundedMean never returns a value outside of the
// bounds, even if BoundedSum/Count would be outside the bounds.
TYPED_TEST(BoundedMeanTest, LowClampTest) {
  std::vector<TypeParam> a = {0, 0, 0, 0};

  for (int i = 0; i < kNumSamples; ++i) {
    std::unique_ptr<BoundedMean<TypeParam>> mean =
        typename BoundedMean<TypeParam>::Builder()
            .SetEpsilon(kSmallEpsilon)
            .SetLower(0)
            .SetUpper(10)
            .Build()
            .ValueOrDie();
    Output result = mean->Result(a.begin(), a.end()).ValueOrDie();
    EXPECT_GE(GetValue<double>(result), 0);
  }
}

TYPED_TEST(BoundedMeanTest, HighClampTest) {
  std::vector<TypeParam> a = {10, 10, 10, 10};

  for (int i = 0; i < kNumSamples; ++i) {
    std::unique_ptr<BoundedMean<TypeParam>> mean =
        typename BoundedMean<TypeParam>::Builder()
            .SetEpsilon(kSmallEpsilon)
            .SetLower(0)
            .SetUpper(10)
            .Build()
            .ValueOrDie();
    Output result = mean->Result(a.begin(), a.end()).ValueOrDie();
    EXPECT_LE(GetValue<double>(result), 10);
  }
}

TYPED_TEST(BoundedMeanTest, LargeEpsilonTest) {
  std::vector<TypeParam> a = {6, 3, 5, 1, 7, 2, 3, 3, 4, 6, 5, 1};

  // Compute the expected mean
  double expected = 0;
  for (auto value : a) {
    expected += value;
  }
  expected /= a.size();

  std::unique_ptr<BoundedMean<TypeParam>> mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(std::pow(10, 20))
          .SetLower(1)
          .SetUpper(7)
          .Build()
          .ValueOrDie();
  Output actual = mean->Result(a.begin(), a.end()).ValueOrDie();

  EXPECT_DOUBLE_EQ(GetValue<double>(actual), expected);
}

TYPED_TEST(BoundedMeanTest, PropagateApproxBoundsError) {
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();

  // Automatic bounds are needed but there is no input, so the count-threshhold
  // should exceed any bin count.
  EXPECT_FALSE(bm->PartialResult().ok());
}

TYPED_TEST(BoundedMeanTest, MaxContributionsVarianceTest) {
  // Use following inputs with mean 0.
  const std::vector<TypeParam> input = {1, 1, -1, -1};

  std::function<double(int)> sample_variance_for_max_contributions =
      [&input](int max_contributions) {
        double sum = 0;
        for (int i = 0; i < kNumSamples; ++i) {
          auto mean = typename BoundedMean<TypeParam>::Builder()
                          .SetMaxContributionsPerPartition(max_contributions)
                          .SetEpsilon(1)
                          .SetLower(-1)
                          .SetUpper(1)
                          .Build();
          CHECK_EQ(mean.status(), base::OkStatus());
          auto out = (*mean)->Result(input.begin(), input.end());
          CHECK_EQ(out.status(), base::OkStatus());
          sum += std::pow(GetValue<double>(*out), 2);
        }
        return sum / (kNumSamples - 1);
      };

  // We expect the sample variance with max contribution 2 to be (significantly)
  // bigger than with max contribution 1.
  EXPECT_GT(sample_variance_for_max_contributions(2),
            1.1 * sample_variance_for_max_contributions(1));
}

TYPED_TEST(BoundedMeanTest, SerializeMergeTest) {
  typename BoundedMean<TypeParam>::Builder builder;

  std::unique_ptr<BoundedMean<TypeParam>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(3)
          .Build()
          .ValueOrDie();
  bm->AddEntry(1);
  Summary summary = bm->Serialize();
  bm->AddEntry(3);

  std::unique_ptr<BoundedMean<TypeParam>> bm2 = builder.Build().ValueOrDie();
  EXPECT_OK(bm2->Merge(summary));
  bm2->AddEntry(3);

  EXPECT_EQ(GetValue<double>(bm->PartialResult().ValueOrDie()),
            GetValue<double>(bm2->PartialResult().ValueOrDie()));
}

TYPED_TEST(BoundedMeanTest, SerializeMergePartialSumsTest) {
  typename ApproxBounds<TypeParam>::Builder bounds_builder;
  typename BoundedMean<TypeParam>::Builder builder;

  // Automatic bounding, so entries will be split and stored as partial sums.
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      bounds_builder.SetThreshold(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(bounds))
          .Build()
          .ValueOrDie();
  bm->AddEntry(-10);
  bm->AddEntry(4);
  Summary summary = bm->Serialize();
  bm->AddEntry(6);

  // Merge summary into second BoundedVariance.
  std::unique_ptr<ApproxBounds<TypeParam>> bounds2 =
      bounds_builder.Build().ValueOrDie();
  std::unique_ptr<BoundedMean<TypeParam>> bm2 =
      builder.SetApproxBounds(std::move(bounds2)).Build().ValueOrDie();
  bm2->AddEntry(6);
  EXPECT_OK(bm2->Merge(summary));

  // Check equality.  Bounds are set to [-16, 8].
  EXPECT_EQ(GetValue<double>(bm->PartialResult().ValueOrDie()),
            GetValue<double>(bm2->PartialResult().ValueOrDie()));
}

TYPED_TEST(BoundedMeanTest, AutomaticBoundsNegative) {
  std::vector<TypeParam> a = {9, -2, -2, -1, -6, -6};
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bm->AddEntries(a.begin(), a.end());

  // 9 gets clamped to -1.
  Output expected_output;
  AddToOutput<double>(&expected_output, -3);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), -8);
  SetValue<TypeParam>(report->mutable_upper_bound(), -1);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  EXPECT_THAT(bm->PartialResult().ValueOrDie(), EqualsProto(expected_output));
}

TYPED_TEST(BoundedMeanTest, AutomaticBoundsPositive) {
  std::vector<TypeParam> a = {-9, 2, 2, 1, 6, 6};
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bm->AddEntries(a.begin(), a.end());

  // -9 gets clamped to 1.
  Output expected_output;
  AddToOutput<double>(&expected_output, 3);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), 1);
  SetValue<TypeParam>(report->mutable_upper_bound(), 8);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  EXPECT_THAT(bm->PartialResult().ValueOrDie(), EqualsProto(expected_output));
}

TEST(BoundedMeanTest, DropNanEntries) {
  std::vector<double> a = {2, 4, 6, NAN, 8};
  std::unique_ptr<BoundedMean<double>> mean =
      typename BoundedMean<double>::Builder()
          .SetEpsilon(1)
          .SetLower(1)
          .SetUpper(9)
          .Build()
          .ValueOrDie();
  Output result = mean->Result(a.begin(), a.end()).ValueOrDie();
  EXPECT_GE(GetValue<double>(result), 1);
  EXPECT_LE(GetValue<double>(result), 9);
}

TEST(BoundedMeanTest, SensitivityOverflow) {
  // Check for error when upper - lower causes integer overflow.
  EXPECT_EQ(typename BoundedMean<int>::Builder()
                .SetEpsilon(1.0)
                .SetLower(INT_MIN)
                .SetUpper(INT_MAX)
                .Build()
                .status()
                .message(),
            "Upper - lower caused integer overflow.");
}

TEST(BoundedMeanTest, SensitivityOverflowApproxBounds) {
  std::unique_ptr<ApproxBounds<int>> bounds =
      typename ApproxBounds<int>::Builder()
          .SetEpsilon(1)
          .SetThreshold(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedMean<int>> bm = typename BoundedMean<int>::Builder()
                                             .SetEpsilon(1)
                                             .SetApproxBounds(std::move(bounds))
                                             .Build()
                                             .ValueOrDie();

  // Adding these two entries make the bounds [-1, max]. Sensitivity is
  // calculated |max - (-1)|, which overflowss.
  bm->AddEntry(-1);
  bm->AddEntry(INT_MAX);

  EXPECT_EQ(bm->PartialResult().status().message(),
            "Upper - lower caused integer overflow.");
}

// Test when 0 is in [lower, upper].
TYPED_TEST(BoundedMeanTest, AutomaticBoundsContainZero) {
  std::vector<TypeParam> a = {4,
                              4,
                              -1,
                              -1,
                              std::numeric_limits<TypeParam>::lowest(),
                              std::numeric_limits<TypeParam>::max()};
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bm->AddEntries(a.begin(), a.end());

  Output expected_output;
  AddToOutput<double>(&expected_output, 1.5);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), -1);
  SetValue<TypeParam>(report->mutable_upper_bound(), 4);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  EXPECT_THAT(bm->PartialResult().ValueOrDie(), EqualsProto(expected_output));
}

// Test not providing ApproxBounds and instead using the default.
TYPED_TEST(BoundedMeanTest, AutomaticBoundsDefault) {
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::vector<TypeParam> big(100, 10);
  std::vector<TypeParam> small(100, -10);
  bm->AddEntries(big.begin(), big.end());
  bm->AddEntries(small.begin(), small.end());

  BoundingReport bounding_report;
  SetValue<TypeParam>(bounding_report.mutable_lower_bound(), -16);
  SetValue<TypeParam>(bounding_report.mutable_upper_bound(), 16);
  bounding_report.set_num_inputs(big.size() + small.size());
  bounding_report.set_num_outside(0);
  Output::ErrorReport expected_report;
  *(expected_report.mutable_bounding_report()) = bounding_report;

  Output result = bm->PartialResult().ValueOrDie();
  EXPECT_THAT(result.error_report(), EqualsProto(expected_report));
  EXPECT_NEAR(GetValue<double>(result.elements(0).value()), 0.0,
              std::pow(10, -10));
}

// Test when a bound is 0.
TYPED_TEST(BoundedMeanTest, AutomaticBoundsZero) {
  std::vector<TypeParam> a = {0, 0, 4, 4, -2, 2, 7};
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bm->AddEntries(a.begin(), a.end());

  // -2 gets clamped to 0. 7 gets clamped to 4.
  Output expected_output;
  AddToOutput<double>(&expected_output, 2);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), 0);
  SetValue<TypeParam>(report->mutable_upper_bound(), 4);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  EXPECT_THAT(bm->PartialResult().ValueOrDie(), EqualsProto(expected_output));
}

TYPED_TEST(BoundedMeanTest, Reset) {
  // Construct bounded sum with approximate bounding.
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetThreshold(1)
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();

  // Reset between adding vectors.
  std::vector<TypeParam> a = {-10, 1000};
  std::vector<TypeParam> b = {-100, 100, 3};
  bm->AddEntries(a.begin(), a.end());
  bm->Reset();
  bm->AddEntries(b.begin(), b.end());

  // Check result is only affected by vector b.
  auto result = bm->PartialResult();
  EXPECT_OK(result);
  EXPECT_EQ(GetValue<double>(result.ValueOrDie().elements(0).value()), 1);
}

TYPED_TEST(BoundedMeanTest, MemoryUsed) {
  std::unique_ptr<BoundedMean<TypeParam>> bm =
      typename BoundedMean<TypeParam>::Builder().Build().ValueOrDie();
  EXPECT_GT(bm->MemoryUsed(), 0);
}

}  //  namespace
}  // namespace differential_privacy
