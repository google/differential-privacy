//
// Copyright 2025 Google LLC
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

#include "algorithms/approx-bounds-provider.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "algorithms/bounds-provider.h"
#include "algorithms/internal/clamped-calculation-without-bounds.h"
#include "algorithms/numerical-mechanisms-testing.h"

namespace differential_privacy {

namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Lt;

template <typename T>
class ApproxBoundsProviderTypedTest : public ::testing::Test {};

using TestingTypes = ::testing::Types<int64_t, double>;
TYPED_TEST_SUITE(ApproxBoundsProviderTypedTest, TestingTypes);

TYPED_TEST(ApproxBoundsProviderTypedTest, ReturnsErrorOnEmptyEpsilon) {
  typename ApproxBoundsProvider<TypeParam>::Options options{};
  EXPECT_THAT(
      ApproxBoundsProvider<TypeParam>::Create(options),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Epsilon must be finite and positive, but is 0")));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           ReturnsErrorOnEmptyMaxPartitionsContributed) {
  typename ApproxBoundsProvider<TypeParam>::Options options{1};
  EXPECT_THAT(
      ApproxBoundsProvider<TypeParam>::Create(options),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Maximum number of partitions that can be contributed to "
                    "(i.e., L0 sensitivity) must be positive, but is 0")));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           ReturnsErrorOnEmptyMaxContributionsPerPartition) {
  typename ApproxBoundsProvider<TypeParam>::Options options{1, 1};
  EXPECT_THAT(ApproxBoundsProvider<TypeParam>::Create(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Maximum number of contributions per "
                                 "partition must be positive, but is 0")));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsReturnsErrorOnEmptyInput) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1.0,  // epsilon
      1,    // max_partitions_contributed
      1     // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  EXPECT_THAT((*approx_bound_provider)->FinalizeAndCalculateBounds(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("run over a larger dataset")));
}

TYPED_TEST(ApproxBoundsProviderTypedTest, BasicTestIntegerDivByZero) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1.0,  // epsilon
      1,    // max_partitions_contributed
      1,    // max_contributions_per_partition
      2,    // scale
      1,    // base
      2,    // num_bins
      0.9,  // success_probability
      std::make_unique<ZeroNoiseMechanism::Builder>(),  // mechanism_builder
  };
  EXPECT_THAT(ApproxBoundsProvider<TypeParam>::Create(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Base must be greater than 1")));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsReturnsResultOnLargeInput) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1e20,  // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  for (int i = -10000; i < 10000; ++i) {
    (*approx_bound_provider)->AddEntry(i);
  }

  EXPECT_THAT(
      (*approx_bound_provider)->FinalizeAndCalculateBounds(),
      IsOkAndHolds(AllOf(
          Field("lower_bound", &BoundsResult<TypeParam>::lower_bound, Lt(0)),
          Field("upper_bound", &BoundsResult<TypeParam>::upper_bound, Gt(0)))));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsReturnsCorrectLowerAndUpperBounds) {
  std::vector<TypeParam> a = {0, 0, 0, 0, 1, 3, 7, 8, 8, 8};
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1.0,  // epsilon
      1,    // max_partitions_contributed
      1,    // max_contributions_per_partition
      1,    // scale
      2,    // base
      10,   // num_bins
      0.9,  // success_probability
      std::make_unique<ZeroNoiseMechanism::Builder>(),  // mechanism_builder
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  for (const auto& input : a) {
    (*approx_bound_provider)->AddEntry(input);
  }

  const absl::StatusOr<BoundsResult<TypeParam>> result =
      (*approx_bound_provider)->FinalizeAndCalculateBounds();
  ASSERT_OK(result);
  EXPECT_EQ(result->lower_bound, 0);
  EXPECT_EQ(result->upper_bound, 1);
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsInputBeyondBins) {
  std::vector<TypeParam> a = {-1, -1, -1, -1, 3, 9, 9, 9, 28, 12, 34};
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1e20,  // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
      1,     // scale
      2,     // base
      4,     // num_bins
      0.95,  // success_probability, threshold = 3.04886
      std::make_unique<ZeroNoiseMechanism::Builder>(),  // mechanism_builder
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  for (const auto& input : a) {
    (*approx_bound_provider)->AddEntry(input);
  }
  absl::StatusOr<BoundsResult<TypeParam>> result =
      (*approx_bound_provider)->FinalizeAndCalculateBounds();
  ASSERT_OK(result);
  EXPECT_FLOAT_EQ(result->lower_bound, -1);
  EXPECT_FLOAT_EQ(result->upper_bound, 8);
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsHandleOverflowPosBins) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,     // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
      1,     // scale
      2,     // base
      2,     // num_bins
      0.95,  // success_probability
      std::make_unique<ZeroNoiseMechanism::Builder>(),  // mechanism_builder
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  // Add std::numeric_limits<int64_t>::max() + 3 entries to the same bin to try to
  // cause an overflow.
  (*approx_bound_provider)->AddEntry(std::numeric_limits<int64_t>::max());
  (*approx_bound_provider)->AddEntry(1);
  (*approx_bound_provider)->AddEntry(1);
  (*approx_bound_provider)->AddEntry(1);
  absl::StatusOr<BoundsResult<TypeParam>> result =
      (*approx_bound_provider)->FinalizeAndCalculateBounds();
  // An overflow should cause a negative bin count and all bins to be below
  // the threshold, resulting in an error when there are no bins to return.
  // Thus, if there is no error, there was no overflow.
  EXPECT_THAT(result.status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsHandleOverflowNegBins) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,     // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
      1,     // scale
      2,     // base
      2,     // num_bins
      0.95,  // success_probability
      std::make_unique<ZeroNoiseMechanism::Builder>(),  // mechanism_builder
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  // Add std::numeric_limits<int64_t>::max() + 3 entries to the same bin to try to
  // cause an overflow.
  (*approx_bound_provider)->AddEntry(std::numeric_limits<int64_t>::max());
  (*approx_bound_provider)->AddEntry(-1);
  (*approx_bound_provider)->AddEntry(-1);
  (*approx_bound_provider)->AddEntry(-1);
  absl::StatusOr<BoundsResult<TypeParam>> result =
      (*approx_bound_provider)->FinalizeAndCalculateBounds();
  // An overflow should cause a negative bin count and all bins to be below
  // the threshold, resulting in an error when there are no bins to return.
  // Thus, if there is no error, there was no overflow.
  EXPECT_THAT(result.status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));
}

TEST(ApproxBoundsProviderTest,
     FinalizeAndCalculateBoundsHandleInfinityEntries) {
  std::vector<double> a = {1, 1, 1, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity()};
  typename ApproxBoundsProvider<double>::Options options{
      1e20,  // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
      7,     // scale
      2,     // base
      13,    // num_bins
      0.5,   // success_probability
      std::make_unique<ZeroNoiseMechanism::Builder>(),  // mechanism_builder
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<double>>>
      approx_bound_provider = ApproxBoundsProvider<double>::Create(options);
  for (const auto& input : a) {
    (*approx_bound_provider)->AddEntry(input);
  }
  absl::StatusOr<BoundsResult<double>> result =
      (*approx_bound_provider)->FinalizeAndCalculateBounds();
  ASSERT_OK(result);
  EXPECT_FLOAT_EQ(result->lower_bound, 0);
  EXPECT_FLOAT_EQ(result->upper_bound,
                  28672);  // scale * std::pow(base, bins - 1)
}

TEST(ApproxBoundsProviderTest, FinalizeAndCalculateBoundsNumPositiveBins) {
  std::vector<double> a = {1, 1, 1, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity()};
  typename ApproxBoundsProvider<double>::Options options{
      1e20,  // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
      1,     // scale
      2,     // base
      2,     // num_bins
      0.5,   // success_probability
      std::make_unique<ZeroNoiseMechanism::Builder>(),  // mechanism_builder
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<double>>>
      approx_bound_provider = ApproxBoundsProvider<double>::Create(options);
  ASSERT_OK(approx_bound_provider.status());
  for (const auto& input : a) {
    (*approx_bound_provider)->AddEntry(input);
  }
  absl::StatusOr<BoundsResult<double>> result =
      (*approx_bound_provider)->FinalizeAndCalculateBounds();
  ASSERT_OK(result);
  EXPECT_FLOAT_EQ(result->lower_bound, 0);
  EXPECT_FLOAT_EQ(result->upper_bound, 2);
}

TEST(ApproxBoundsProviderTest, FinalizeAndCalculateBoundsSmallScale) {
  std::vector<double> a = {0,   -0.5, -0.5, 0.1, -0.7, 0.7,
                           0.8, 0.3,  -0.5, 0.6, 0.5,  0.1};
  typename ApproxBoundsProvider<double>::Options options{
      1,    // epsilon
      1,    // max_partitions_contributed
      1,    // max_contributions_per_partition
      0.1,  // scale
      2,    // base
      4,    // num_bins
      0.9,  // success_probability
      std::make_unique<ZeroNoiseMechanism::Builder>(),  // mechanism_builder
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<double>>>
      approx_bound_provider = ApproxBoundsProvider<double>::Create(options);
  ASSERT_OK(approx_bound_provider.status());
  for (const auto& input : a) {
    (*approx_bound_provider)->AddEntry(input);
  }
  absl::StatusOr<BoundsResult<double>> result =
      (*approx_bound_provider)->FinalizeAndCalculateBounds();
  ASSERT_OK(result);
  EXPECT_FLOAT_EQ(result->lower_bound, -0.8);
  EXPECT_FLOAT_EQ(result->upper_bound, 0.8);
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsAfterResetReturnsError) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,  // epsilon
      1,  // max_partitions_contributed
      1,  // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  for (int i = -10000; i < 10000; ++i) {
    (*approx_bound_provider)->AddEntry(i);
  }
  (*approx_bound_provider)->Reset();

  EXPECT_THAT((*approx_bound_provider)->FinalizeAndCalculateBounds(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("run over a larger dataset")));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           GetBoundingReportReturnsReportWithSameBounds) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1e20,  // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  for (int i = -10000; i < 10000; ++i) {
    (*approx_bound_provider)->AddEntry(i);
  }
  const absl::StatusOr<BoundsResult<TypeParam>> result =
      (*approx_bound_provider)->FinalizeAndCalculateBounds();
  ASSERT_OK(result.status());

  const BoundingReport report =
      (*approx_bound_provider)->GetBoundingReport(result.value());

  EXPECT_THAT(GetValue<TypeParam>(report.lower_bound()),
              Eq(result->lower_bound));
  EXPECT_THAT(GetValue<TypeParam>(report.upper_bound()),
              Eq(result->upper_bound));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           MemoryUsedReturnsValueLargerThanObjectSize) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,  // epsilon
      1,  // max_partitions_contributed
      1,  // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  EXPECT_THAT((*approx_bound_provider)->MemoryUsed(),
              Gt(sizeof(ApproxBoundsProvider<TypeParam>)));
}

TYPED_TEST(ApproxBoundsProviderTypedTest, GetEpsilonReturnsExpectedValue) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1.234,  // epsilon
      1,      // max_partitions_contributed
      1,      // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);

  EXPECT_THAT((*approx_bound_provider)->GetEpsilon(), Eq(1.234));
}

TYPED_TEST(ApproxBoundsProviderTypedTest, GetDeltaReturnsZero) {
  // Delta is currently ignored for approx bounds.
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,  // epsilon
      1,  // max_partitions_contributed
      1,  // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  EXPECT_THAT((*approx_bound_provider)->GetDelta(), Eq(0));
}

TYPED_TEST(ApproxBoundsProviderTypedTest, MergeWithEmptyBoundsSummaryFails) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,  // epsilon
      1,  // max_partitions_contributed
      1,  // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  EXPECT_THAT(
      (*approx_bound_provider)->Merge(BoundsSummary()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Cannot merge summary with no histogram data")));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           SerializeFromPopulatedAndMergeEmptyBoundsProviderReturnsBounds) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1e20,  // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      populated_approx_bounds_provider =
          ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(populated_approx_bounds_provider.status());

  for (int i = -5; i < 5; ++i) {
    (*populated_approx_bounds_provider)->AddEntry(i);
  }
  typename ApproxBoundsProvider<TypeParam>::Options empty_approx_bounds_options{
      1e20,  // epsilon
      1,     // max_partitions_contributed
      1,     // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      empty_approx_bounds_provider =
          ApproxBoundsProvider<TypeParam>::Create(empty_approx_bounds_options);
  ASSERT_OK(empty_approx_bounds_provider.status());

  ASSERT_OK((*empty_approx_bounds_provider)
                ->Merge((*populated_approx_bounds_provider)->Serialize()));

  EXPECT_THAT(
      (*empty_approx_bounds_provider)->FinalizeAndCalculateBounds(),
      IsOkAndHolds(AllOf(
          Field("lower_bound", &BoundsResult<TypeParam>::lower_bound, Lt(0)),
          Field("upper_bound", &BoundsResult<TypeParam>::upper_bound, Gt(0)))));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           CreateClampedCalculationWithoutBoundsHasSameProperties) {
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,   // epsilon
      1,   // max_partitions_contributed
      1,   // max_contributions_per_partition
      2,   // scale
      3,   // base
      10,  // num_bins
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());
  std::unique_ptr<internal::ClampedCalculationWithoutBounds<TypeParam>>
      clamped_calculation =
          (*approx_bound_provider)->CreateClampedCalculationWithoutBounds();

  EXPECT_THAT(clamped_calculation->GetNumBins(), Eq(10));
  EXPECT_THAT(clamped_calculation->GetBaseForTesting(), Eq(3));
  EXPECT_THAT(clamped_calculation->GetScaleForTesting(), Eq(2));
}

TYPED_TEST(ApproxBoundsProviderTypedTest,
           ComputeFromPartialsCountValidityTest) {
  int n_bins = 4;
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,       // epsilon
      1,       // max_partitions_contributed
      1,       // max_contributions_per_partition
      1,       // scale
      2,       // base
      n_bins,  // num_bins
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());

  std::vector<TypeParam> pos_sum(n_bins, 0);
  std::vector<TypeParam> neg_sum(n_bins, 0);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };
  (*approx_bound_provider)
      ->template AddToPartials<TypeParam>(&pos_sum, 6, difference);
  (*approx_bound_provider)
      ->template AddToPartials<TypeParam>(&neg_sum, -3, difference);

  std::vector<int64_t> invalid_entries{-1,
                                       std::numeric_limits<int64_t>::lowest()};
  absl::StatusOr<TypeParam> result;
  for (int64_t n_entries : invalid_entries) {
    result = (*approx_bound_provider)
                 ->template ComputeFromPartials<TypeParam>(
                     pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, 4,
                     n_entries);
    EXPECT_THAT(result.status(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Count must be non-negative")));
  }

  result = (*approx_bound_provider)
               ->template ComputeFromPartials<TypeParam>(
                   pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, 4, 0);
  EXPECT_OK(result.status());
}

TYPED_TEST(ApproxBoundsProviderTypedTest, ComputeSumFromPartials) {
  int n_bins = 4;
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,       // epsilon
      1,       // max_partitions_contributed
      1,       // max_contributions_per_partition
      1,       // scale
      2,       // base
      n_bins,  // num_bins
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);
  ASSERT_OK(approx_bound_provider.status());
  std::vector<TypeParam> pos_sum(n_bins, 0);
  std::vector<TypeParam> neg_sum(n_bins, 0);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };
  (*approx_bound_provider)
      ->template AddToPartials<TypeParam>(&pos_sum, 6, difference);
  (*approx_bound_provider)
      ->template AddToPartials<TypeParam>(&neg_sum, -3, difference);

  absl::StatusOr<TypeParam> result =
      (*approx_bound_provider)
          ->template ComputeFromPartials<TypeParam>(
              pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, 4, 2);
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), 1);

  result = (*approx_bound_provider)
               ->template ComputeFromPartials<TypeParam>(
                   pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, -1, 2);
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), -4);

  result = (*approx_bound_provider)
               ->template ComputeFromPartials<TypeParam>(
                   pos_sum, neg_sum, [](TypeParam x) { return x; }, 1, 4, 2);
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), 5);
}

TYPED_TEST(ApproxBoundsProviderTypedTest, OverflowComputeFromPartials) {
  int64_t int64lowest = std::numeric_limits<int64_t>::lowest();
  int64_t int64max = std::numeric_limits<int64_t>::max();
  std::function<int64_t(int64_t)> value_transform = [](int64_t x) { return x; };
  typename ApproxBoundsProvider<TypeParam>::Options options{
      1,  // epsilon
      1,  // max_partitions_contributed
      1,  // max_contributions_per_partition
      1,  // scale
      2,  // base
      4,  // num_bins
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<TypeParam>>>
      approx_bound_provider = ApproxBoundsProvider<TypeParam>::Create(options);

  std::vector<int64_t> neg_sum = {0, -1, -2, int64lowest};
  std::vector<int64_t> pos_sum = {0, 0, 0, 0};
  absl::StatusOr<int64_t> result =
      (*approx_bound_provider)
          ->template ComputeFromPartials<int64_t>(
              pos_sum, neg_sum, value_transform, int64lowest, int64max, 2);
  ASSERT_OK(result);
  // The negative sums should overflow to positive
  EXPECT_GT(result.value(), 0);

  result = (*approx_bound_provider)
               ->template ComputeFromPartials<int64_t>(
                   pos_sum, neg_sum, value_transform, int64lowest, -1, 2);
  ASSERT_OK(result);
  // The negative sums should overflow to positive
  EXPECT_GT(result.value(), 0);

  neg_sum = {0, 0, 0, 0};
  pos_sum = {0, 1, 2, int64max};
  result = (*approx_bound_provider)
               ->template ComputeFromPartials<int64_t>(
                   pos_sum, neg_sum, value_transform, int64lowest, int64max, 2);
  ASSERT_OK(result);
  // The positive sums should overflow to negative
  EXPECT_LT(result.value(), 0);

  result = (*approx_bound_provider)
               ->template ComputeFromPartials<int64_t>(
                   pos_sum, neg_sum, value_transform, 1, int64max, 2);
  ASSERT_OK(result);
  // The positive sums should overflow to negative
  EXPECT_LT(result.value(), 0);
}

TEST(ApproxBoundsProviderTest, ComputeSumFromPartialsAcrossOne) {
  typename ApproxBoundsProvider<double>::Options options{
      1,  // epsilon
      1,  // max_partitions_contributed
      1,  // max_contributions_per_partition
  };
  absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<double>>>
      approx_bound_provider = ApproxBoundsProvider<double>::Create(options);
  ASSERT_OK(approx_bound_provider);
  std::vector<double> pos_sum((*approx_bound_provider)->NumPositiveBins(), 0);
  std::vector<double> neg_sum((*approx_bound_provider)->NumPositiveBins(), 0);
  auto difference = [](double val1, double val2) { return val1 - val2; };
  (*approx_bound_provider)
      ->template AddToPartials<double>(&pos_sum, 6, difference);
  (*approx_bound_provider)
      ->template AddToPartials<double>(&neg_sum, -3, difference);

  absl::StatusOr<double> result =
      (*approx_bound_provider)
          ->template ComputeFromPartials<double>(
              pos_sum, neg_sum, [](double x) { return x; }, -4, -0.5, 2);
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(result.value(), -3.5);

  result = (*approx_bound_provider)
               ->template ComputeFromPartials<double>(
                   pos_sum, neg_sum, [](double x) { return x; }, 0.5, 4, 2);
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(result.value(), 4.5);
}

}  // namespace
}  // namespace differential_privacy
