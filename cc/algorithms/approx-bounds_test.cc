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

#include "algorithms/approx-bounds.h"

#include <limits>

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "algorithms/numerical-mechanisms-testing.h"

namespace differential_privacy {
namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::HasSubstr;
using ::differential_privacy::base::testing::StatusIs;

template <typename T>
class ApproxBoundsTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(ApproxBoundsTest, NumericTypes);

TEST(ApproxBoundsTest, BasicTest) {
  std::vector<int64_t> a = {0, -5, -5, INT_MIN, -7, 7, 7, 3, -6, 6, 5, 1};

  // Make ApproxBounds.
  base::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      ApproxBounds<int64_t>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetThreshold(3)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  base::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().int_value(), -8);
  EXPECT_EQ(result->elements(1).value().int_value(), 8);
}

TYPED_TEST(ApproxBoundsTest, EmptyHistogramTest) {
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetSuccessProbability(.95)  // k threshold = 3.04886
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  EXPECT_THAT(
      (*bounds)->PartialResult(),
      StatusIs(
          base::StatusCode::kFailedPrecondition,
          HasSubstr(
              "run over a larger dataset or decrease success_probability")));
}

TEST(ApproxBoundsTest, SmallScale) {
  std::vector<double> a = {0, -.5, -.5, .1, -.7, .7, .8, .3, -.5, .6, .5, .1};
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(.1)
          .SetSuccessProbability(.95)  // k threshold = 3.04886
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  base::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().float_value(), -.8);
  EXPECT_EQ(result->elements(1).value().float_value(), .8);
}

TEST(ApproxBoundsTest, InputBeyondBins) {
  std::vector<double> a = {-1, -1, -1, -1, 3, 9, 9, 9, 28, 12, 34};
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetSuccessProbability(.95)  // k threshold = 3.04886
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  (*bounds)->AddEntries(a.begin(), a.end());
  base::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().float_value(), -1);
  EXPECT_EQ(result->elements(1).value().float_value(), 8);
}

TEST(ApproxBoundsTest, NegativeMax) {
  std::vector<double> a = {-3, -3, -3, -3, -8, -8, -8, -8};
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(4)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  (*bounds)->AddEntries(a.begin(), a.end());
  base::StatusOr<Output> result = (*bounds)->PartialResult();
  EXPECT_EQ(result->elements(0).value().float_value(), -8);
  EXPECT_EQ(result->elements(1).value().float_value(), -2);
}

TYPED_TEST(ApproxBoundsTest, InvalidParameters) {
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(0)
                  .SetScale(1)
                  .SetBase(2)
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(base::StatusCode::kInvalidArgument,
                       HasSubstr("Must have one or more bins")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(0)
                  .SetBase(2)
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(base::StatusCode::kInvalidArgument,
                       HasSubstr("Scale must be positive")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(1)
                  .SetBase(0.5)
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(base::StatusCode::kInvalidArgument,
                       HasSubstr("Base must be greater than 1")));
  EXPECT_THAT(
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(2)
          .SetScale(1)
          .SetBase(2)
          .SetSuccessProbability(1)
          .Build(),
      StatusIs(base::StatusCode::kInvalidArgument,
               HasSubstr("Success percentage must be between 0 and 1")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(1)
                  .SetBase(2)
                  .SetThreshold(-1)
                  .Build(),
              StatusIs(base::StatusCode::kInvalidArgument,
                       HasSubstr("k threshold must be nonnegative")));
}

TEST(ApproxBoundsTest, DefaultIntTest) {
  std::vector<int> a = {INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN,
                        INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};
  base::StatusOr<std::unique_ptr<ApproxBounds<int>>> bounds =
      ApproxBounds<int>::Builder()
          .SetThreshold(4)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  base::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().int_value(), INT_MIN);
  EXPECT_EQ(result->elements(1).value().int_value(), INT_MAX);
}

TEST(ApproxBoundsTest, DefaultDoubleTest) {
  std::vector<double> big(30, std::numeric_limits<double>::max());
  std::vector<double> small(30, std::numeric_limits<double>::lowest());
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  (*bounds)->AddEntries(big.begin(), big.end());
  (*bounds)->AddEntries(small.begin(), small.end());
  base::StatusOr<Output> result = (*bounds)->PartialResult();
  EXPECT_EQ(result->elements(0).value().float_value(),
            std::numeric_limits<double>::lowest());
  EXPECT_EQ(result->elements(1).value().float_value(),
            std::numeric_limits<double>::max());
}

TYPED_TEST(ApproxBoundsTest, SerializeAndMergeTest) {
  std::vector<TypeParam> a = {-1, -11, 6};
  std::vector<TypeParam> b = {3, 5, 15, 56};
  typename ApproxBounds<TypeParam>::Builder builder;

  // Serialize bounds with only data from a.
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds1 =
      builder.SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds1);
  (*bounds1)->AddEntries(a.begin(), a.end());
  Summary summary = (*bounds1)->Serialize();
  (*bounds1)->AddEntries(b.begin(), b.end());

  // Create bounds2 with part of its data from merge.
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds2 =
      builder.Build();
  ASSERT_OK(bounds2);
  EXPECT_OK((*bounds2)->Merge(summary));
  (*bounds2)->AddEntries(b.begin(), b.end());

  // Check that results are the same.
  base::StatusOr<Output> result1 = (*bounds1)->PartialResult();
  ASSERT_OK(result1);
  base::StatusOr<Output> result2 = (*bounds2)->PartialResult();
  ASSERT_OK(result2);
  EXPECT_EQ(result1->elements(0).value().float_value(),
            result2->elements(0).value().float_value());
  EXPECT_EQ(result1->elements(1).value().float_value(),
            result2->elements(1).value().float_value());
}

TEST(ApproxBoundsTest, DropNanEntries) {
  std::vector<double> a = {1, 1, 1, NAN};
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(2)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  for (const auto& element : a) {
    (*bounds)->AddEntry(element);
  }
  base::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().float_value(), 0);
  EXPECT_EQ(result->elements(1).value().float_value(), 1);
}

TEST(ApproxBounds, HandleInfinityEntries) {
  std::vector<double> a = {1, 1, 1, INFINITY, INFINITY};
  const double bins = 13;
  const double base = 2;
  const double scale = 7;
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(bins)
          .SetBase(base)
          .SetScale(scale)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  base::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().float_value(), 0);
  const double max_result = scale * std::pow(base, bins - 1);
  EXPECT_EQ(result->elements(1).value().float_value(), max_result);
}

TEST(ApproxBoundsTest, NumPositiveBins) {
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(2)
          .SetBase(2)
          .SetScale(1)
          .SetSuccessProbability(.95)
          .Build();
  ASSERT_OK(bounds);
  EXPECT_EQ((*bounds)->NumPositiveBins(), 2);
}

TEST(ApproxBoundsTest, MostSignificantBit) {
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetSuccessProbability(.95)
          .Build();
  ASSERT_OK(bounds);
  EXPECT_EQ((*bounds)->MostSignificantBit(1), 0);
  EXPECT_EQ((*bounds)->MostSignificantBit(0), 0);
  EXPECT_EQ((*bounds)->MostSignificantBit(64), 3);
  EXPECT_EQ((*bounds)->MostSignificantBit(-8), 3);
}

TEST(ApproxBoundsTest, ThresholdByPrivacyBudget) {
  ApproxBounds<int>::Builder builder;
  std::vector<int> a = {1, 1};

  // Threshold = 1.9 / privacy_budget = 3.8, so no bounds are found.
  base::StatusOr<std::unique_ptr<ApproxBounds<int>>> bounds1 =
      builder.SetSuccessProbability(.01)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds1);
  (*bounds1)->AddEntries(a.begin(), a.end());
  EXPECT_THAT((*bounds1)->PartialResult(.5),
              StatusIs(base::StatusCode::kFailedPrecondition,
                       HasSubstr("decrease success_probability")));

  // Threshold = 1.9 / privacy_budget = 1.9, so bounds are found.
  base::StatusOr<std::unique_ptr<ApproxBounds<int>>> bounds2 = builder.Build();
  ASSERT_OK(bounds2);
  (*bounds2)->AddEntries(a.begin(), a.end());
  base::StatusOr<Output> result2 = (*bounds2)->PartialResult();
  ASSERT_OK(result2);
  EXPECT_EQ(GetValue<int>(result2->elements(1).value()), 1);
}

TYPED_TEST(ApproxBoundsTest, AddToPartials) {
  int n_bins = 4;
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(n_bins)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };

  // Test positive number.
  std::vector<TypeParam> sums(n_bins, 0);
  std::vector<TypeParam> expected = {1, 1, 2, 2};
  (*bounds)->template AddToPartials<TypeParam>(&sums, 6, difference);
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }

  // Test negative number.
  std::fill(sums.begin(), sums.end(), 0);
  expected = {-1, -1, -1, 0};
  (*bounds)->template AddToPartials<TypeParam>(&sums, -3, difference);
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }

  // Test 0.
  std::fill(sums.begin(), sums.end(), 0);
  std::fill(expected.begin(), expected.end(), 0);
  (*bounds)->template AddToPartials<TypeParam>(&sums, 0, difference);
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }
}

TYPED_TEST(ApproxBoundsTest, AddToPartialSums) {
  int n_bins = 4;
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(n_bins)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);

  // Test positive number.
  std::vector<TypeParam> sums(n_bins, 0);
  std::vector<TypeParam> expected = {1, 1, 2, 2};
  (*bounds)->template AddToPartialSums<TypeParam>(&sums, 6);
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }

  // Test negative number.
  std::fill(sums.begin(), sums.end(), 0);
  expected = {-1, -1, -1, 0};
  (*bounds)->template AddToPartialSums<TypeParam>(&sums, -3);
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }

  // Test 0.
  std::fill(sums.begin(), sums.end(), 0);
  std::fill(expected.begin(), expected.end(), 0);
  (*bounds)->template AddToPartialSums<TypeParam>(&sums, 0);
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }
}

TYPED_TEST(ApproxBoundsTest, ComputeSumFromPartials) {
  int n_bins = 4;
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(n_bins)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);
  std::vector<TypeParam> pos_sum(n_bins, 0);
  std::vector<TypeParam> neg_sum(n_bins, 0);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };
  (*bounds)->template AddToPartials<TypeParam>(&pos_sum, 6, difference);
  (*bounds)->template AddToPartials<TypeParam>(&neg_sum, -3, difference);

  EXPECT_EQ((*bounds)->template ComputeFromPartials<TypeParam>(
                pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, 4, 2),
            1);
  EXPECT_EQ((*bounds)->template ComputeFromPartials<TypeParam>(
                pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, -1, 2),
            -4);
  EXPECT_EQ((*bounds)->template ComputeFromPartials<TypeParam>(
                pos_sum, neg_sum, [](TypeParam x) { return x; }, 1, 4, 2),
            5);
}

TEST(ApproxBoundsText, ComputeSumFromPartialsAcrossOne) {
  base::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder().Build();
  ASSERT_OK(bounds);
  std::vector<double> pos_sum((*bounds)->NumPositiveBins(), 0);
  std::vector<double> neg_sum((*bounds)->NumPositiveBins(), 0);
  auto difference = [](double val1, double val2) { return val1 - val2; };
  (*bounds)->template AddToPartials<double>(&pos_sum, 6, difference);
  (*bounds)->template AddToPartials<double>(&neg_sum, -3, difference);

  EXPECT_EQ((*bounds)->template ComputeFromPartials<double>(
                pos_sum, neg_sum, [](double x) { return x; }, -4, -0.5, 2),
            -3.5);
  EXPECT_EQ((*bounds)->template ComputeFromPartials<double>(
                pos_sum, neg_sum, [](double x) { return x; }, 0.5, 4, 2),
            4.5);
}

TYPED_TEST(ApproxBoundsTest, GetBoundingReport_NoInputs) {
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);
  BoundingReport expected;
  SetValue<TypeParam>(expected.mutable_lower_bound(), -8);
  SetValue<TypeParam>(expected.mutable_upper_bound(), 8);
  EXPECT_THAT((*bounds)->GetBoundingReport(-8, 8), EqualsProto(expected));
}

TYPED_TEST(ApproxBoundsTest, GetBoundingReport) {
  std::vector<TypeParam> a = {0, -5, -1, -100, -7, 7, 8, 3, -6, 6, 5, 1};
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(3)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  EXPECT_OK((*bounds)->PartialResult());
  EXPECT_EQ((*bounds)->GetBoundingReport(0, 8).num_inputs(), a.size());
  EXPECT_EQ((*bounds)->GetBoundingReport(-8, 8).num_outside(), 1);   // [-8, 8]
  EXPECT_EQ((*bounds)->GetBoundingReport(1, 8).num_outside(), 7);    // (1, 8]
  EXPECT_EQ((*bounds)->GetBoundingReport(-8, -2).num_outside(), 9);  // [-8, -2)
  EXPECT_EQ((*bounds)->GetBoundingReport(0, 1).num_outside(), 10);   // [0, 1]
  EXPECT_EQ((*bounds)->GetBoundingReport(-1, 0).num_outside(), 11);  // [-1, 0)
}

TYPED_TEST(ApproxBoundsTest, Memory) {
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds_small =
      typename ApproxBounds<TypeParam>::Builder().SetNumBins(1).Build();
  ASSERT_OK(bounds_small);
  base::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds_big =
      typename ApproxBounds<TypeParam>::Builder().SetNumBins(2).Build();
  ASSERT_OK(bounds_big);

  // Extra memory comes from extra element in pos_bins_ and neg_bins_.
  EXPECT_GE((*bounds_big)->MemoryUsed(), (*bounds_small)->MemoryUsed());
}

}  //  namespace
}  // namespace differential_privacy
