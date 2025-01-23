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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "algorithms/internal/clamped-calculation-without-bounds.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/data.pb.h"

namespace differential_privacy {

// Provides limited-scope static methods for interacting with a ApproxBounds
// object for testing purposes.
class ApproxBoundsTestPeer {
 public:
  template <typename T>
  static void AddMultipleEntries(const T& t, int64_t num_of_entries,
                                 ApproxBounds<T>* ab) {
    ab->AddMultipleEntries(t, num_of_entries);
  }

  template <typename T, typename T2>
  static void AddMultipleEntriesToPartials(std::vector<T2>* partials, T value,
                                           int64_t num_of_entries,
                                           std::function<T2(T, T)> make_partial,
                                           ApproxBounds<T>* ab) {
    ab->AddMultipleEntriesToPartials(partials, value, num_of_entries,
                                     make_partial);
  }

  template <typename T, typename T2>
  static void AddMultipleEntriesToPartialSums(std::vector<T2>* sums, T value,
                                              int64_t num_of_entries,
                                              ApproxBounds<T>* ab) {
    ab->AddMultipleEntriesToPartialSums(sums, value, num_of_entries);
  }
};

namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::testing::Eq;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::differential_privacy::base::testing::StatusIs;

template <typename T>
class ApproxBoundsTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(ApproxBoundsTest, NumericTypes);

TEST(ApproxBoundsTest, BasicTest) {
  std::vector<int64_t> a = {0, -5, -5, INT_MIN, -7, 7, 7, 3, -6, 6, 5, 1};

  // Make ApproxBounds.
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      ApproxBounds<int64_t>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetThresholdForTest(3)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().int_value(), -8);
  EXPECT_EQ(result->elements(1).value().int_value(), 8);
}

TEST(ApproxBoundsTest, BasicMultipleEntriesTest) {
  std::vector<int64_t> a = {1, 2, 3, 5, 8, 13};

  // Make ApproxBounds.
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      typename ApproxBounds<int64_t>::Builder()
          .SetNumBins(10)
          .SetScale(1)
          .SetBase(2)
          .SetThresholdForTest(2.5)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  for (const auto& input : a) {
    ApproxBoundsTestPeer::AddMultipleEntries(input, input,
                                             bounds.value().get());
  }
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().int_value(), 2);
  EXPECT_EQ(result->elements(1).value().int_value(), 16);
}

TEST(ApproxBoundsTest, AddMultipleEntriesInvalidInputTest) {
  std::vector<float> a = {1.0, 2.0, 3.0, 5.0, 8.0, 13.0};

  // Make ApproxBounds.
  absl::StatusOr<std::unique_ptr<ApproxBounds<float>>> bounds =
      typename ApproxBounds<float>::Builder()
          .SetNumBins(10)
          .SetScale(1)
          .SetBase(2)
          .SetThresholdForTest(2.5)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);

  // Add some normal entries whose result we know what to expect.
  for (const auto& input : a) {
    ApproxBoundsTestPeer::AddMultipleEntries(input, input,
                                             bounds.value().get());
  }

  // Try adding an invalid entry, which we expect should be ignored.
  ApproxBoundsTestPeer::AddMultipleEntries(
      std::numeric_limits<float>::quiet_NaN(), 1, bounds.value().get());

  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);

  EXPECT_FLOAT_EQ(result->elements(0).value().float_value(), 2.0);
  EXPECT_FLOAT_EQ(result->elements(1).value().float_value(), 16.0);
}

TEST(ApproxBoundsTest, AddMultipleEntriesInvalidNumberOfEntriesTest) {
  std::vector<int64_t> a = {1, 2, 3, 5, 8, 13};

  // Make ApproxBounds.
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      typename ApproxBounds<int64_t>::Builder()
          .SetNumBins(10)
          .SetScale(1)
          .SetBase(2)
          .SetThresholdForTest(2.5)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);

  // Add some normal entries whose result we know what to expect.
  for (const auto& input : a) {
    ApproxBoundsTestPeer::AddMultipleEntries(input, input,
                                             bounds.value().get());
  }

  // Expect adding an invalid number of entries to be ignored.
  std::vector<int64_t> invalid_entries{0, -1,
                                       std::numeric_limits<int64_t>::lowest()};
  for (int64_t n_entries : invalid_entries) {
    ApproxBoundsTestPeer::AddMultipleEntries<int64_t>(1, n_entries,
                                                      bounds.value().get());
  }

  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);

  EXPECT_FLOAT_EQ(result->elements(0).value().int_value(), 2);
  EXPECT_FLOAT_EQ(result->elements(1).value().int_value(), 16);
}

TYPED_TEST(ApproxBoundsTest, EmptyHistogramTest) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetSuccessProbability(.95)  // k threshold = 3.04886
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  absl::StatusOr<Output> result = bounds->get()->PartialResult();
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kFailedPrecondition,
                               HasSubstr("Bin count threshold was too large")));
  EXPECT_THAT(
      result.status().GetPayload(
          "type.googleapis.com/differential_privacy.ApproxBoundsNotEnoughData"),
      Optional(Eq(absl::Cord())));
}

TEST(ApproxBoundsTest, RetriesBoundingTest) {
  int64_t num_bins = 4;
  double threshold = 1.001;
  // We choose a large epsilon to ensure that the success probability
  // corresponding to the threshold is larger than kMinSuccessProbability
  double epsilon = 18;
  auto mechanism = ZeroNoiseMechanism::Builder().SetEpsilon(epsilon).Build();
  // if we set an explicit threshold, then we won't retry bounding, so instead
  // we set a success probability that corresponds to the desired threshold.
  double success_probability =
      std::pow((*mechanism)->Cdf(threshold), 2 * num_bins);
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      typename ApproxBounds<double>::Builder()
          .SetNumBins(num_bins)
          .SetBase(2)
          .SetScale(1)
          .SetEpsilon(epsilon)
          .SetSuccessProbability(success_probability)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntry(3);

  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);

  EXPECT_FLOAT_EQ(result->elements(0).value().float_value(), 2);
  EXPECT_FLOAT_EQ(result->elements(1).value().float_value(), 4);
}

TYPED_TEST(ApproxBoundsTest, ExplicitThresholdNotRelaxed) {
  int64_t num_bins = 4;
  double threshold = 1.0001;
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(num_bins)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(threshold)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntry(3);

  absl::StatusOr<Output> result = (*bounds)->PartialResult();

  EXPECT_THAT(result, StatusIs(absl::StatusCode::kFailedPrecondition,
                               HasSubstr("Bin count threshold was too large")));
  EXPECT_THAT(
      result.status().GetPayload(
          "type.googleapis.com/differential_privacy.ApproxBoundsNotEnoughData"),
      Optional(Eq(absl::Cord())));
}

TEST(ApproxBoundsTest, InsufficientPrivacyBudgetTest) {
  std::vector<int64_t> a = {0, -5, -5, INT_MIN, -7, 7, 7, 3, -6, 6, 5, 1};

  // Make ApproxBounds.
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      ApproxBounds<int64_t>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetThresholdForTest(3)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  absl::StatusOr<Output> output = bounds->get()->PartialResult();
  EXPECT_THAT(output, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("can only produce results once")));
  EXPECT_THAT(
      output.status().GetPayload(
          "type.googleapis.com/differential_privacy.ApproxBoundsNotEnoughData"),
      Eq(std::nullopt));
}

TEST(ApproxBoundsTest, SmallScale) {
  std::vector<double> a = {0, -.5, -.5, .1, -.7, .7, .8, .3, -.5, .6, .5, .1};
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(.1)
          .SetSuccessProbability(.95)  // k threshold = 3.04886
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_FLOAT_EQ(result->elements(0).value().float_value(), -.8);
  EXPECT_FLOAT_EQ(result->elements(1).value().float_value(), .8);
}

TEST(ApproxBoundsTest, InputBeyondBins) {
  std::vector<double> a = {-1, -1, -1, -1, 3, 9, 9, 9, 28, 12, 34};
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetSuccessProbability(.95)  // k threshold = 3.04886
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  (*bounds)->AddEntries(a.begin(), a.end());
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_FLOAT_EQ(result->elements(0).value().float_value(), -1);
  EXPECT_FLOAT_EQ(result->elements(1).value().float_value(), 8);
}

TEST(ApproxBoundsTest, NegativeMax) {
  std::vector<double> a = {-3, -3, -3, -3, -8, -8, -8, -8};
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(4)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  (*bounds)->AddEntries(a.begin(), a.end());
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  EXPECT_FLOAT_EQ(result->elements(0).value().float_value(), -8);
  EXPECT_FLOAT_EQ(result->elements(1).value().float_value(), -2);
}

TYPED_TEST(ApproxBoundsTest, InvalidParameters) {
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(0)
                  .SetScale(1)
                  .SetBase(2)
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Number of bins must be positive")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(0)
                  .SetBase(2)
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Scale must be finite and positive")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(std::numeric_limits<double>::infinity())
                  .SetBase(2)
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Scale must be finite and positive")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(std::numeric_limits<double>::quiet_NaN())
                  .SetBase(2)
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Scale must be a valid numeric value")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(1)
                  .SetBase(0.5)
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Base must be greater than or equal to 1")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(1)
                  .SetBase(std::numeric_limits<double>::infinity())
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Base must be finite")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(1)
                  .SetBase(std::numeric_limits<double>::quiet_NaN())
                  .SetSuccessProbability(.95)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Base must be a valid numeric value")));
  EXPECT_THAT(
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(2)
          .SetScale(1)
          .SetBase(2)
          .SetSuccessProbability(1)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Success probability must be in the exclusive interval (0,1)")));
  EXPECT_THAT(
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(2)
          .SetScale(1)
          .SetBase(2)
          .SetSuccessProbability(0)
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Success probability must be in the exclusive interval (0,1)")));
  EXPECT_THAT(
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(2)
          .SetScale(1)
          .SetBase(2)
          .SetSuccessProbability(std::numeric_limits<double>::infinity())
          .Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Success probability must be in the exclusive interval (0,1)")));
  EXPECT_THAT(
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(2)
          .SetScale(1)
          .SetBase(2)
          .SetSuccessProbability(std::numeric_limits<double>::quiet_NaN())
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Success probability must be a valid numeric value")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(1)
                  .SetBase(2)
                  .SetThresholdForTest(-1)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("k threshold must be non-negative")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(1)
                  .SetBase(2)
                  .SetThresholdForTest(std::numeric_limits<double>::infinity())
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("k threshold must be finite")));
  EXPECT_THAT(typename ApproxBounds<TypeParam>::Builder()
                  .SetNumBins(2)
                  .SetScale(1)
                  .SetBase(2)
                  .SetThresholdForTest(std::numeric_limits<double>::quiet_NaN())
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("k threshold must be a valid numeric value")));
}

TYPED_TEST(ApproxBoundsTest, ComputeFromPartialsCountValidityTest) {
  int n_bins = 4;
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
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

  std::vector<int64_t> invalid_entries{-1,
                                       std::numeric_limits<int64_t>::lowest()};
  absl::StatusOr<TypeParam> result;
  for (int64_t n_entries : invalid_entries) {
    result = (*bounds)->template ComputeFromPartials<TypeParam>(
        pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, 4, n_entries);
    EXPECT_THAT(result.status(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Count must be non-negative")));
  }

  result = (*bounds)->template ComputeFromPartials<TypeParam>(
      pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, 4, 0);
  EXPECT_OK(result.status());
}

TEST(ApproxBoundsTest, DefaultIntTest) {
  std::vector<int> a = {INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN,
                        INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};
  absl::StatusOr<std::unique_ptr<ApproxBounds<int>>> bounds =
      ApproxBounds<int>::Builder()
          .SetThresholdForTest(4)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(result->elements(0).value().int_value(), INT_MIN);
  EXPECT_EQ(result->elements(1).value().int_value(), INT_MAX);
}

TEST(ApproxBoundsTest, DefaultDoubleTest) {
  std::vector<double> big(30, std::numeric_limits<double>::max());
  std::vector<double> small(30, std::numeric_limits<double>::lowest());
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  (*bounds)->AddEntries(big.begin(), big.end());
  (*bounds)->AddEntries(small.begin(), small.end());
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  EXPECT_DOUBLE_EQ(result->elements(0).value().float_value(),
                   std::numeric_limits<double>::lowest());
  EXPECT_DOUBLE_EQ(result->elements(1).value().float_value(),
                   std::numeric_limits<double>::max());
}

TYPED_TEST(ApproxBoundsTest, SerializeAndMergeTest) {
  std::vector<TypeParam> a = {-1, -11, 6};
  std::vector<TypeParam> b = {3, 5, 15, 56};
  typename ApproxBounds<TypeParam>::Builder builder;

  // Serialize bounds with only data from a.
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds1 =
      builder.SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds1);
  (*bounds1)->AddEntries(a.begin(), a.end());
  Summary summary = (*bounds1)->Serialize();
  (*bounds1)->AddEntries(b.begin(), b.end());

  // Create bounds2 with part of its data from merge.
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds2 =
      builder.Build();
  ASSERT_OK(bounds2);
  EXPECT_OK((*bounds2)->Merge(summary));
  (*bounds2)->AddEntries(b.begin(), b.end());

  // Check that results are the same.
  absl::StatusOr<Output> result1 = (*bounds1)->PartialResult();
  ASSERT_OK(result1);
  absl::StatusOr<Output> result2 = (*bounds2)->PartialResult();
  ASSERT_OK(result2);
  EXPECT_FLOAT_EQ(result1->elements(0).value().float_value(),
                  result2->elements(0).value().float_value());
  EXPECT_FLOAT_EQ(result1->elements(1).value().float_value(),
                  result2->elements(1).value().float_value());
}

TYPED_TEST(ApproxBoundsTest, SerializeAndMergeOverflowPosBinsTest) {
  typename ApproxBounds<int64_t>::Builder builder;

  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      builder.SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  ApproxBoundsTestPeer::AddMultipleEntries<int64_t>(
      1, std::numeric_limits<int64_t>::max(), (*bounds).get());
  Summary summary = (*bounds)->Serialize();
  (*bounds)->AddEntry(1);
  (*bounds)->AddEntry(1);
  (*bounds)->AddEntry(1);

  // Create bounds2 with part of its data from merge.
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds2 =
      builder.Build();
  ASSERT_OK(bounds2);
  EXPECT_OK((*bounds2)->Merge(summary));
  (*bounds2)->AddEntry(1);
  (*bounds2)->AddEntry(1);
  (*bounds2)->AddEntry(1);

  // The bin counts should have overflowed and be smaller than the threshold.
  EXPECT_THAT((*bounds2)->PartialResult().status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));

  // Ensure a pre-merge overflow is passed on during a serialize & merge
  summary = (*bounds2)->Serialize();
  bounds2 = builder.Build();
  EXPECT_OK((*bounds2)->Merge(summary));
  // The bin counts should have overflowed and be smaller than the threshold.
  EXPECT_THAT((*bounds2)->PartialResult().status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));
}

TYPED_TEST(ApproxBoundsTest, SerializeAndMergeOverflowNegBinsTest) {
  typename ApproxBounds<int64_t>::Builder builder;

  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      builder.SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  ApproxBoundsTestPeer::AddMultipleEntries<int64_t>(
      -1, std::numeric_limits<int64_t>::max(), (*bounds).get());
  Summary summary = (*bounds)->Serialize();
  (*bounds)->AddEntry(-1);
  (*bounds)->AddEntry(-1);
  (*bounds)->AddEntry(-1);

  // Create bounds2 with part of its data from merge.
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds2 =
      builder.Build();
  ASSERT_OK(bounds2);
  EXPECT_OK((*bounds2)->Merge(summary));
  (*bounds2)->AddEntry(-1);
  (*bounds2)->AddEntry(-1);
  (*bounds2)->AddEntry(-1);

  // The bin counts should have overflowed and be smaller than the threshold.
  EXPECT_THAT((*bounds2)->PartialResult().status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));

  // Ensure a pre-merge overflow is passed on during a serialize & merge
  summary = (*bounds2)->Serialize();
  bounds2 = builder.Build();
  EXPECT_OK((*bounds2)->Merge(summary));
  // The bin counts should have overflowed and be smaller than the threshold.
  EXPECT_THAT((*bounds2)->PartialResult().status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));
}

TEST(ApproxBoundsTest, DropNanEntries) {
  std::vector<double> a = {1, 1, 1, NAN};
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(2)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  for (const auto& element : a) {
    (*bounds)->AddEntry(element);
  }
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_FLOAT_EQ(result->elements(0).value().float_value(), 0);
  EXPECT_FLOAT_EQ(result->elements(1).value().float_value(), 1);
}

TEST(ApproxBoundsTest, HandleOverflowPosBins) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      ApproxBounds<int64_t>::Builder()
          .SetNumBins(2)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  // Add std::numeric_limits<int64_t>::max() + 3 entries to the same bin to try to
  // cause an overflow.
  ApproxBoundsTestPeer::AddMultipleEntries<int64_t>(
      1, std::numeric_limits<int64_t>::max(), (*bounds).get());
  (*bounds)->AddEntry(1);
  (*bounds)->AddEntry(1);
  (*bounds)->AddEntry(1);
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  // An overflow should cause a negative bin count and all bins to be below
  // the threshold, resulting in an error when there are no bins to return.
  // Thus, if there is no error, there was no overflow.
  EXPECT_THAT(result.status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));
}

TEST(ApproxBoundsTest, HandleOverflowNegBins) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      ApproxBounds<int64_t>::Builder()
          .SetNumBins(2)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  // Add std::numeric_limits<int64_t>::max() + 3 entries to the same bin to try to
  // cause an overflow.
  ApproxBoundsTestPeer::AddMultipleEntries<int64_t>(
      -1, std::numeric_limits<int64_t>::max(), (*bounds).get());
  (*bounds)->AddEntry(-1);
  (*bounds)->AddEntry(-1);
  (*bounds)->AddEntry(-1);
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  // An overflow should cause a negative bin count and all bins to be below
  // the threshold, resulting in an error when there are no bins to return.
  // Thus, if there is no error, there was no overflow.
  EXPECT_THAT(result.status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));
}

TEST(ApproxBoundsTest, HandleInfinityEntries) {
  std::vector<double> a = {1, 1, 1, INFINITY, INFINITY};
  const double bins = 13;
  const double base = 2;
  const double scale = 7;
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetNumBins(bins)
          .SetBase(base)
          .SetScale(scale)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  (*bounds)->AddEntries(a.begin(), a.end());
  absl::StatusOr<Output> result = (*bounds)->PartialResult();
  ASSERT_OK(result);
  EXPECT_FLOAT_EQ(result->elements(0).value().float_value(), 0);
  const double max_result = scale * std::pow(base, bins - 1);
  EXPECT_FLOAT_EQ(result->elements(1).value().float_value(), max_result);
}

TEST(ApproxBoundsTest, NumPositiveBins) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
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
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
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

TYPED_TEST(ApproxBoundsTest, AddToPartials) {
  int n_bins = 4;
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
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

TYPED_TEST(ApproxBoundsTest, AddMultipleEntriesToPartials) {
  int n_bins = 4;
  int n_entries = 5;
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(n_bins)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };

  // Test positive number.
  std::vector<TypeParam> sums(n_bins, 0);
  std::vector<TypeParam> expected = {5, 5, 10, 10};
  ApproxBoundsTestPeer::AddMultipleEntriesToPartials<TypeParam, TypeParam>(
      &sums, 6, n_entries, difference, bounds.value().get());

  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }

  // Test negative number.
  std::fill(sums.begin(), sums.end(), 0);
  expected = {-5, -5, -5, 0};
  ApproxBoundsTestPeer::AddMultipleEntriesToPartials<TypeParam, TypeParam>(
      &sums, -3, n_entries, difference, bounds.value().get());
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }

  // Test 0.
  std::fill(sums.begin(), sums.end(), 0);
  std::fill(expected.begin(), expected.end(), 0);
  ApproxBoundsTestPeer::AddMultipleEntriesToPartials<TypeParam, TypeParam>(
      &sums, 0, n_entries, difference, bounds.value().get());
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }
}

TEST(ApproxBoundsTest, AddMultipleEntriesToPartialsInvalidValueTest) {
  int n_bins = 4;
  int n_entries = 5;
  absl::StatusOr<std::unique_ptr<ApproxBounds<float>>> bounds =
      typename ApproxBounds<float>::Builder()
          .SetNumBins(n_bins)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);
  auto difference = [](float val1, float val2) { return val1 - val2; };

  std::vector<float> sums(n_bins, 0);
  std::vector<float> expected(n_bins, 0);
  ApproxBoundsTestPeer::AddMultipleEntriesToPartials<float, float>(
      &sums, std::numeric_limits<float>::quiet_NaN(), n_entries, difference,
      bounds.value().get());

  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }
}

TYPED_TEST(ApproxBoundsTest,
           AddMultipleEntriesToPartialsInvalidNumberOfEntriesTest) {
  int n_bins = 4;
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(n_bins)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };

  std::vector<TypeParam> sums(n_bins, 0);
  std::vector<TypeParam> expected(n_bins, 0);

  std::vector<int64_t> invalid_entries{0, -1,
                                       std::numeric_limits<int64_t>::lowest()};

  for (int64_t n_entries : invalid_entries) {
    ApproxBoundsTestPeer::AddMultipleEntriesToPartials<TypeParam, TypeParam>(
        &sums, 1, n_entries, difference, bounds.value().get());
  }

  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }
}

TYPED_TEST(ApproxBoundsTest, AddToPartialSums) {
  int n_bins = 4;
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
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

TYPED_TEST(ApproxBoundsTest, AddMultipleEntriesToPartialSums) {
  int n_bins = 4;
  int n_entries = 5;
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(n_bins)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);

  // Test positive number.
  std::vector<TypeParam> sums(n_bins, 0);
  std::vector<TypeParam> expected = {5, 5, 10, 10};
  ApproxBoundsTestPeer::AddMultipleEntriesToPartialSums<TypeParam, TypeParam>(
      &sums, 6, n_entries, bounds.value().get());
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }

  // Test negative number.
  std::fill(sums.begin(), sums.end(), 0);
  expected = {-5, -5, -5, 0};
  ApproxBoundsTestPeer::AddMultipleEntriesToPartialSums<TypeParam, TypeParam>(
      &sums, -3, n_entries, bounds.value().get());
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }

  // Test 0.
  std::fill(sums.begin(), sums.end(), 0);
  std::fill(expected.begin(), expected.end(), 0);
  ApproxBoundsTestPeer::AddMultipleEntriesToPartialSums<TypeParam, TypeParam>(
      &sums, 0, n_entries, bounds.value().get());
  for (int i = 0; i < n_bins; ++i) {
    EXPECT_EQ(sums[i], expected[i]);
  }
}

TEST(ApproxBoundsTest, OverflowAddMultipleEntriesToPartialSums) {
  int n_bins = 4;
  int64_t n_entries = std::numeric_limits<int64_t>::max();
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      typename ApproxBounds<int64_t>::Builder()
          .SetNumBins(n_bins)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);

  std::vector<int64_t> sums(n_bins, 0);
  ApproxBoundsTestPeer::AddMultipleEntriesToPartialSums<int64_t, int64_t>(
      &sums, 6, n_entries, bounds.value().get());

  // If there is an overflow, at least one of the bins will be negative
  EXPECT_THAT(sums, testing::Contains(testing::Lt(0)));
}

TYPED_TEST(ApproxBoundsTest, ComputeSumFromPartials) {
  int n_bins = 4;
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
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

  absl::StatusOr<TypeParam> result =
      (*bounds)->template ComputeFromPartials<TypeParam>(
          pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, 4, 2);
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), 1);

  result = (*bounds)->template ComputeFromPartials<TypeParam>(
      pos_sum, neg_sum, [](TypeParam x) { return x; }, -4, -1, 2);
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), -4);

  result = (*bounds)->template ComputeFromPartials<TypeParam>(
      pos_sum, neg_sum, [](TypeParam x) { return x; }, 1, 4, 2);
  ASSERT_OK(result);
  EXPECT_EQ(result.value(), 5);
}

TEST(ApproxBoundsTest, OverflowNoiseFromTypeCast) {
  // Overflowing should result in the sum + noise eventually wrapping around and
  // become negative.
  absl::Status partial_result_status;
  int i = 0;
  do {
    absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
        ApproxBounds<int64_t>::Builder()
            .SetNumBins(2)
            .SetBase(2)
            .SetScale(1)
            .SetThresholdForTest(2)
            .SetEpsilon(1)
            .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
            .Build();
    ASSERT_OK(bounds);
    ApproxBoundsTestPeer::AddMultipleEntries<int64_t>(
        1, std::numeric_limits<int64_t>::max(), (*bounds).get());
    absl::StatusOr<Output> result = (*bounds)->PartialResult();
    partial_result_status.Update(result.status());
    ++i;
  } while (i < 100 && partial_result_status.ok());
  // An overflow should have been caused by a negative bin count and all bins
  // to be below the threshold, resulting in an error when there are no bins to
  // return.
  EXPECT_THAT(partial_result_status,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large to find "
                                 "approximate bounds.")));
}

TYPED_TEST(ApproxBoundsTest, OverflowComputeFromPartials) {
  int64_t int64lowest = std::numeric_limits<int64_t>::lowest();
  int64_t int64max = std::numeric_limits<int64_t>::max();
  std::function<int64_t(int64_t)> value_transform = [](int64_t x) { return x; };
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      typename ApproxBounds<int64_t>::Builder()
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .Build();
  ASSERT_OK(bounds);

  std::vector<int64_t> neg_sum = {0, -1, -2, int64lowest};
  std::vector<int64_t> pos_sum = {0, 0, 0, 0};
  absl::StatusOr<int64_t> result =
      (*bounds)->template ComputeFromPartials<int64_t>(
          pos_sum, neg_sum, value_transform, int64lowest, int64max, 2);
  ASSERT_OK(result);
  // The negative sums should overflow to positive
  EXPECT_GT(result.value(), 0);

  result = (*bounds)->template ComputeFromPartials<int64_t>(
      pos_sum, neg_sum, value_transform, int64lowest, -1, 2);
  ASSERT_OK(result);
  // The negative sums should overflow to positive
  EXPECT_GT(result.value(), 0);

  neg_sum = {0, 0, 0, 0};
  pos_sum = {0, 1, 2, int64max};
  result = (*bounds)->template ComputeFromPartials<int64_t>(
      pos_sum, neg_sum, value_transform, int64lowest, int64max, 2);
  ASSERT_OK(result);
  // The positive sums should overflow to negative
  EXPECT_LT(result.value(), 0);

  result = (*bounds)->template ComputeFromPartials<int64_t>(
      pos_sum, neg_sum, value_transform, 1, int64max, 2);
  ASSERT_OK(result);
  // The positive sums should overflow to negative
  EXPECT_LT(result.value(), 0);
}

TEST(ApproxBoundsText, ComputeSumFromPartialsAcrossOne) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder().Build();
  ASSERT_OK(bounds);
  std::vector<double> pos_sum((*bounds)->NumPositiveBins(), 0);
  std::vector<double> neg_sum((*bounds)->NumPositiveBins(), 0);
  auto difference = [](double val1, double val2) { return val1 - val2; };
  (*bounds)->template AddToPartials<double>(&pos_sum, 6, difference);
  (*bounds)->template AddToPartials<double>(&neg_sum, -3, difference);

  absl::StatusOr<double> result =
      (*bounds)->template ComputeFromPartials<double>(
          pos_sum, neg_sum, [](double x) { return x; }, -4, -0.5, 2);
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(result.value(), -3.5);

  result = (*bounds)->template ComputeFromPartials<double>(
      pos_sum, neg_sum, [](double x) { return x; }, 0.5, 4, 2);
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(result.value(), 4.5);
}

TYPED_TEST(ApproxBoundsTest, GetBoundingReport_NoInputs) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
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
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(3)
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
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds_small =
      typename ApproxBounds<TypeParam>::Builder().SetNumBins(1).Build();
  ASSERT_OK(bounds_small);
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds_big =
      typename ApproxBounds<TypeParam>::Builder().SetNumBins(2).Build();
  ASSERT_OK(bounds_big);

  // Extra memory comes from extra element in pos_bins_ and neg_bins_.
  EXPECT_GE((*bounds_big)->MemoryUsed(), (*bounds_small)->MemoryUsed());
}

TEST(ApproxBoundsTest, DefaultNumBinsForInt64Is64) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<int64_t>>> bounds =
      ApproxBounds<int64_t>::Builder().SetEpsilon(1.1).SetScale(1.0).Build();
  ASSERT_OK(bounds);
  EXPECT_EQ(bounds.value()->GetNumPosBinsForTesting(), 64);
}

TYPED_TEST(ApproxBoundsTest,
           CreateClampedCalculationWithoutBoundsHasSameProperties) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1.1)
          .SetScale(2.0)
          .SetNumBins(5)
          .SetBase(3)
          .Build();
  ASSERT_OK(bounds);

  std::unique_ptr<internal::ClampedCalculationWithoutBounds<TypeParam>>
      clamped_calculation =
          bounds.value()->CreateClampedCalculationWithoutBounds();

  EXPECT_THAT(clamped_calculation->GetNumBins(), Eq(5));
  EXPECT_THAT(clamped_calculation->GetScaleForTesting(), Eq(2.0));
  EXPECT_THAT(clamped_calculation->GetBaseForTesting(), Eq(3));
}

}  //  namespace
}  // namespace differential_privacy
