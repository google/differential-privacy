//
// Copyright 2024 Google LLC
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

#include "algorithms/approx-bounds-as-bounds-provider.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounds-provider.h"
#include "algorithms/internal/clamped-calculation-without-bounds.h"

namespace differential_privacy {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Lt;

template <typename T>
class ApproxBoundsAsBoundsProviderTypedTest : public ::testing::Test {};

using TestingTypes = ::testing::Types<int64_t, double>;
TYPED_TEST_SUITE(ApproxBoundsAsBoundsProviderTypedTest, TestingTypes);

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsReturnsErrorOnEmptyInput) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1).Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());

  EXPECT_THAT(bounds_provider.FinalizeAndCalculateBounds(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("run over a larger dataset")));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsReturnsResultOnLargeInput) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1e20).Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());

  for (int i = -10000; i < 10000; ++i) {
    bounds_provider.AddEntry(i);
  }

  EXPECT_THAT(
      bounds_provider.FinalizeAndCalculateBounds(),
      IsOkAndHolds(AllOf(
          Field("lower_bound", &BoundsResult<TypeParam>::lower_bound, Lt(0)),
          Field("upper_bound", &BoundsResult<TypeParam>::upper_bound, Gt(0)))));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           GetBoundingReportReturnsReportWithSameBounds) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1e20).Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());

  for (int i = -10000; i < 10000; ++i) {
    bounds_provider.AddEntry(i);
  }
  const absl::StatusOr<BoundsResult<TypeParam>> result =
      bounds_provider.FinalizeAndCalculateBounds();
  ASSERT_OK(result.status());

  const BoundingReport report =
      bounds_provider.GetBoundingReport(result.value());

  EXPECT_THAT(GetValue<TypeParam>(report.lower_bound()),
              Eq(result->lower_bound));
  EXPECT_THAT(GetValue<TypeParam>(report.upper_bound()),
              Eq(result->upper_bound));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           FinalizeAndCalculateBoundsAfterResetReturnsError) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1).Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());

  for (int i = -10000; i < 10000; ++i) {
    bounds_provider.AddEntry(i);
  }
  bounds_provider.Reset();

  EXPECT_THAT(bounds_provider.FinalizeAndCalculateBounds(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("run over a larger dataset")));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           MemoryUsedReturnsValueLargerThanObjectSize) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1).Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());

  EXPECT_THAT(bounds_provider.MemoryUsed(),
              Gt(sizeof(ApproxBoundsAsBoundsProvider<TypeParam>)));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           GetEpsilonReturnsExpectedValue) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1.234).Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());

  EXPECT_THAT(bounds_provider.GetEpsilon(), Eq(1.234));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest, GetDeltaReturnsZero) {
  // Delta is currently ignored for approx bounds.
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetDelta(0)
          .Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());

  EXPECT_THAT(bounds_provider.GetDelta(), Eq(0));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           MergeWithEmptyBoundsSummaryFails) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1e20).Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());

  EXPECT_THAT(bounds_provider.Merge(BoundsSummary()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("approx_bounds_summary must be set")));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           SerialzieFromPopulatedAndMergeEmptyBoundsProviderReturnsBounds) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>>
      populated_approx_bounds =
          typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1e20).Build();
  ASSERT_OK(populated_approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> populated_bounds_provider(
      std::move(populated_approx_bounds).value());
  for (int i = -10000; i < 10000; ++i) {
    populated_bounds_provider.AddEntry(i);
  }
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> empty_approx_bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1e20).Build();
  ASSERT_OK(empty_approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> empty_bounds_provider(
      std::move(empty_approx_bounds).value());

  ASSERT_OK(empty_bounds_provider.Merge(populated_bounds_provider.Serialize()));

  EXPECT_THAT(
      empty_bounds_provider.FinalizeAndCalculateBounds(),
      IsOkAndHolds(AllOf(
          Field("lower_bound", &BoundsResult<TypeParam>::lower_bound, Lt(0)),
          Field("upper_bound", &BoundsResult<TypeParam>::upper_bound, Gt(0)))));
}

TYPED_TEST(ApproxBoundsAsBoundsProviderTypedTest,
           CreateClampedCalculationWithoutBoundsHasSameProperties) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> approx_bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetNumBins(10)
          .SetBase(3)
          .SetScale(2)
          .Build();
  ASSERT_OK(approx_bounds.status());
  ApproxBoundsAsBoundsProvider<TypeParam> bounds_provider(
      std::move(approx_bounds).value());
  std::unique_ptr<internal::ClampedCalculationWithoutBounds<TypeParam>>
      clamped_calculation =
          bounds_provider.CreateClampedCalculationWithoutBounds();

  EXPECT_THAT(clamped_calculation->GetNumBins(), Eq(10));
  EXPECT_THAT(clamped_calculation->GetBaseForTesting(), Eq(3));
  EXPECT_THAT(clamped_calculation->GetScaleForTesting(), Eq(2));
}

}  // namespace
}  // namespace differential_privacy
