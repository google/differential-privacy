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

#include <limits.h>

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cstdint>
#include "base/logging.h"
#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "base/statusor.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"

namespace differential_privacy {

// Provides limited-scope static methods for interacting with a BoundedMean
// object for testing purposes.
class BoundedMeanTestPeer {
 public:
  template <typename T>
  static void AddMultipleEntries(const T& t, int64_t num_of_entries,
                                 BoundedMean<T>* bm) {
    bm->AddMultipleEntries(t, num_of_entries);
  }
};

namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::testing::_;
using ::testing::DoubleEq;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::HasSubstr;
using ::differential_privacy::base::testing::StatusIs;

constexpr double kSmallEpsilon = 0.00000001;
constexpr double kNumSamples = 10000;

template <typename T>
class BoundedMeanTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(BoundedMeanTest, NumericTypes);

TYPED_TEST(BoundedMeanTest, BasicTest) {
  std::vector<TypeParam> a = {2, 4, 6, 8};
  auto mean = typename BoundedMean<TypeParam>::Builder()
                  .SetEpsilon(1.0)
                  .SetLower(1)
                  .SetUpper(9)
                  .Build();
  ASSERT_OK(mean);
  auto result = (*mean)->Result(a.begin(), a.end());
  ASSERT_OK(result);
  EXPECT_GE(GetValue<double>(*result), 1);
  EXPECT_LE(GetValue<double>(*result), 9);
}

TYPED_TEST(BoundedMeanTest, BasicTestWithExplicitLaplace) {
  std::vector<TypeParam> a = {2, 4, 6, 8};
  auto mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .SetLaplaceMechanism(std::make_unique<LaplaceMechanism::Builder>())
          .Build();
  ASSERT_OK(mean);
  auto result = (*mean)->Result(a.begin(), a.end());
  ASSERT_OK(result);
  EXPECT_GE(GetValue<double>(*result), 1);
  EXPECT_LE(GetValue<double>(*result), 9);
}

TYPED_TEST(BoundedMeanTest, BasicTestWithExplicitGaussian) {
  std::vector<TypeParam> a = {2, 4, 6, 8};
  auto mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1.0)
          .SetDelta(1e-7)
          .SetLower(1)
          .SetUpper(9)
          .SetLaplaceMechanism(std::make_unique<GaussianMechanism::Builder>())
          .Build();
  ASSERT_OK(mean);
  auto result = (*mean)->Result(a.begin(), a.end());
  ASSERT_OK(result);
  EXPECT_GE(GetValue<double>(*result), 1);
  EXPECT_LE(GetValue<double>(*result), 9);
}

TYPED_TEST(BoundedMeanTest, RepeatedResultTest) {
  std::vector<TypeParam> a = {2, 4, 6, 8};

  auto mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .Build();
  ASSERT_OK(mean);
  (*mean)->AddEntries(a.begin(), a.end());

  auto result1 = (*mean)->PartialResult(0.5);
  ASSERT_OK(result1);
  auto result2 = (*mean)->PartialResult(0.5);
  ASSERT_OK(result2);

  EXPECT_DOUBLE_EQ(GetValue<double>(*result1), GetValue<double>(*result2));
}

TYPED_TEST(BoundedMeanTest, BasicTestWithoutIterator) {
  std::vector<TypeParam> a = {2, 4, 6, 8};
  auto mean = typename BoundedMean<TypeParam>::Builder()
                  .SetEpsilon(1.0)
                  .SetLower(1)
                  .SetUpper(9)
                  .Build();
  ASSERT_OK(mean);
  for (const auto& input : a) {
    (*mean)->AddEntry(input);
  }
  auto result = (*mean)->PartialResult();
  ASSERT_OK(result);
  EXPECT_GE(GetValue<double>(*result), 1);
  EXPECT_LE(GetValue<double>(*result), 9);
}

TYPED_TEST(BoundedMeanTest, BasicMultipleEntriesTest) {
  std::vector<TypeParam> a = {1, 2, 3, 4, 5};
  auto mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(1)
          .SetUpper(5)
          .Build();
  ASSERT_OK(mean);
  for (const auto& input : a) {
    BoundedMeanTestPeer::AddMultipleEntries<TypeParam>(input, input,
                                                       mean.value().get());
  }
  auto result = (*mean)->PartialResult();
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result), 11.0 / 3.0);
}

TEST(BoundedMeanTest, AddMultipleEntriesInvalidInputTest) {
  auto mean =
      typename BoundedMean<float>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-5)
          .SetUpper(5)
          .Build();
  ASSERT_OK(mean);
  (*mean)->AddEntry(4);
  BoundedMeanTestPeer::AddMultipleEntries<float>(
      std::numeric_limits<float>::quiet_NaN(), 1, mean.value().get());
  auto result = (*mean)->PartialResult();
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result), 4);
}

TYPED_TEST(BoundedMeanTest, AddMultipleEntriesInvalidNumberOfEntriesTest) {
  auto mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-5)
          .SetUpper(5)
          .Build();
  ASSERT_OK(mean);

  (*mean)->AddEntry(4);

  std::vector<int64_t> invalid_entries{0, -1,
                                     std::numeric_limits<int64_t>::lowest()};
  for (int64_t n_entries : invalid_entries) {
    BoundedMeanTestPeer::AddMultipleEntries<TypeParam>(1, n_entries,
                                                       mean.value().get());
  }
  auto result = (*mean)->PartialResult();
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result), 4);
}

TEST(BoundedMeanTest, InvalidParametersTest) {
  EXPECT_THAT(BoundedMean<double>::Builder()
                  .SetEpsilon(std::numeric_limits<double>::infinity())
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be finite")));

  EXPECT_THAT(BoundedMean<double>::Builder()
                  .SetEpsilon(std::numeric_limits<double>::quiet_NaN())
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be a valid numeric value")));

  EXPECT_THAT(BoundedMean<double>::Builder().SetEpsilon(-0.5).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be finite and positive")));

  EXPECT_THAT(BoundedMean<double>::Builder().SetEpsilon(0).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be finite and positive")));
}

TYPED_TEST(BoundedMeanTest, InsufficientPrivacyBudgetTest) {
  std::vector<TypeParam> a = {2, 4, 6, 8};

  auto mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .Build();
  ASSERT_OK(mean);
  (*mean)->AddEntries(a.begin(), a.end());

  ASSERT_OK((*mean)->PartialResult());
  EXPECT_THAT((*mean)->PartialResult(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Privacy budget must be positive")));
}

// This test verifies that BoundedMean never returns a value outside of the
// bounds, even if BoundedSum/Count would be outside the bounds.
TYPED_TEST(BoundedMeanTest, LowClampTest) {
  std::vector<TypeParam> a = {0, 0, 0, 0};

  for (int i = 0; i < kNumSamples; ++i) {
    auto mean = typename BoundedMean<TypeParam>::Builder()
                    .SetEpsilon(kSmallEpsilon)
                    .SetLower(0)
                    .SetUpper(10)
                    .Build();
    ASSERT_OK(mean);
    auto result = (*mean)->Result(a.begin(), a.end());
    ASSERT_OK(result);
    EXPECT_GE(GetValue<double>(*result), 0);
  }
}

TYPED_TEST(BoundedMeanTest, HighClampTest) {
  std::vector<TypeParam> a = {10, 10, 10, 10};

  for (int i = 0; i < kNumSamples; ++i) {
    auto mean = typename BoundedMean<TypeParam>::Builder()
                    .SetEpsilon(kSmallEpsilon)
                    .SetLower(0)
                    .SetUpper(10)
                    .Build();
    ASSERT_OK(mean);
    auto result = (*mean)->Result(a.begin(), a.end());
    EXPECT_LE(GetValue<double>(*result), 10);
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

  auto mean = typename BoundedMean<TypeParam>::Builder()
                  .SetEpsilon(std::pow(10, 20))
                  .SetLower(1)
                  .SetUpper(7)
                  .Build();
  ASSERT_OK(mean);
  auto actual = (*mean)->Result(a.begin(), a.end());
  ASSERT_OK(actual);

  EXPECT_DOUBLE_EQ(GetValue<double>(*actual), expected);
}

TYPED_TEST(BoundedMeanTest, PropagateApproxBoundsError) {
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);

  // Automatic bounds are needed but there is no input, so the count-threshhold
  // should exceed any bin count.
  EXPECT_THAT((*bm)->PartialResult(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Bin count threshold was too large")));
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
          CHECK_EQ(mean.status(), absl::OkStatus());
          auto out = (*mean)->Result(input.begin(), input.end());
          CHECK_EQ(out.status(), absl::OkStatus());
          sum += std::pow(GetValue<double>(*out), 2);
        }
        return sum / (kNumSamples - 1);
      };

  // We expect the sample variance with max contribution 2 to be (significantly)
  // bigger than with max contribution 1.
  EXPECT_GT(sample_variance_for_max_contributions(2),
            1.1 * sample_variance_for_max_contributions(1));
}

TEST(BoundedMeanTest, OverflowRawCountTest) {
  typename BoundedMean<double>::Builder builder;

  base::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-10)
          .SetUpper(10)
          .Build();
  ASSERT_OK(bm);
  BoundedMeanTestPeer::AddMultipleEntries<double>(
      0, std::numeric_limits<int64_t>::max(), (*bm).get());
  BoundedMeanTestPeer::AddMultipleEntries<double>(
      0, std::numeric_limits<int64_t>::max(), (*bm).get());
  (*bm)->AddEntry(1);
  (*bm)->AddEntry(1);
  (*bm)->AddEntry(1);
  (*bm)->AddEntry(1);

  base::StatusOr<Output> result = (*bm)->PartialResult();
  ASSERT_OK(result);
  // If the int64_t partial_count_ overflows, it should wrap around to 2,
  // resulting in a mean of (1+1+1+1) / 2 = 2, instead of the correct mean of
  // nearly 0.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 2.0);
}

TEST(BoundedMeanTest, OverflowCountFromAddNoiseTypeCast) {
  const double kBound = std::numeric_limits<int64_t>::max() / 2;
  int i;
  for (i = 0; i < 100; ++i) {
    typename BoundedMean<double>::Builder builder;

    base::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
        builder
            .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
            .SetLower(-kBound)
            .SetUpper(kBound)
            .Build();
    ASSERT_OK(bm);
    BoundedMeanTestPeer::AddMultipleEntries<double>(
        1, std::numeric_limits<int64_t>::max(), (*bm).get());

    base::StatusOr<Output> result = (*bm)->PartialResult();
    ASSERT_OK(result);
    // The noise applied to the count should eventually cause an overflow,
    // resulting in a noised_count = 1, and thus a mean of around
    // (1 * INT64_MAX) / 1 = INT64_MAX, clamped to the upper limit.
    if (GetValue<double>(result.value()) >= kBound) {
      // An overflow has happened, so return to end the test as a success.
      return;
    }
  }
  FAIL() << "No overflow occurred after " << i << " iterations.";
}

TEST(BoundedMeanTest, OverflowAddMultipleEntriesManualBoundsTest) {
  typename BoundedMean<int64_t>::Builder builder;

  base::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-std::numeric_limits<int64_t>::max() / 2)
          .SetUpper(std::numeric_limits<int64_t>::max() / 2)
          .Build();
  ASSERT_OK(bm);
  BoundedMeanTestPeer::AddMultipleEntries<int64_t>(
      2, std::numeric_limits<int64_t>::max(), (*bm).get());

  base::StatusOr<Output> result = (*bm)->PartialResult();
  ASSERT_OK(result);
  // Adding 2 * int64_max should overflow and wrap around to -2, resulting in a
  // mean of -2 / int64_max.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()),
                   -2.0 / std::numeric_limits<int64_t>::max());
}

TEST(BoundedMeanTest, OverflowAddEntryManualBoundsTest) {
  typename BoundedMean<int64_t>::Builder builder;

  base::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(std::numeric_limits<int64_t>::max())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::max());
  (*bm)->AddEntry(1);
  (*bm)->AddEntry(1);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::max());

  base::StatusOr<Output> result = (*bm)->PartialResult();
  EXPECT_OK(result);
  // Overflowing should result in the running sum wrapping around to zero.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0);
}

TEST(BoundedMeanTest, UnderflowAddEntryManualBoundsTest) {
  typename BoundedMean<int64_t>::Builder builder;

  base::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(std::numeric_limits<int64_t>::lowest() + 1)
          .SetUpper(0)
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);
  (*bm)->AddEntry(-1);
  (*bm)->AddEntry(-1);
  (*bm)->AddEntry(-1);
  (*bm)->AddEntry(-1);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);

  base::StatusOr<Output> result = (*bm)->PartialResult();
  EXPECT_OK(result);
  // Underflowing should result in the running sum wrapping around to zero.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0);
}

TEST(BoundedMeanTest, OverflowRawCountMergeManualBoundsTest) {
  typename BoundedMean<double>::Builder builder;

  base::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-10)
          .SetUpper(10)
          .Build();
  ASSERT_OK(bm);
  BoundedMeanTestPeer::AddMultipleEntries<double>(
      0, std::numeric_limits<int64_t>::max(), (*bm).get());
  Summary summary = (*bm)->Serialize();

  base::StatusOr<std::unique_ptr<BoundedMean<double>>> bm2 = builder.Build();
  ASSERT_OK(bm2);
  BoundedMeanTestPeer::AddMultipleEntries<double>(
      0, std::numeric_limits<int64_t>::max(), (*bm2).get());
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(1);

  ASSERT_OK((*bm2)->Merge(summary));

  base::StatusOr<Output> result = (*bm2)->PartialResult();
  ASSERT_OK(result);
  // If the int64_t partial_count_ overflows, it should wrap around to 2,
  // resulting in a mean of (1+1+1+1) / 2 = 2, instead of the correct mean of
  // nearly 0.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 2);

  // Test post-overflow serialize & merge
  summary = (*bm2)->Serialize();
  bm2 = builder.Build();
  EXPECT_OK((*bm2)->Merge(summary));
  result = (*bm2)->PartialResult();
  EXPECT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 2);
}

TEST(BoundedMeanTest, OverflowMergeManualBoundsTest) {
  typename BoundedMean<int64_t>::Builder builder;

  base::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(std::numeric_limits<int64_t>::max())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::max());
  Summary summary = (*bm)->Serialize();

  base::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm2 = builder.Build();
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(std::numeric_limits<int64_t>::max());

  ASSERT_OK((*bm2)->Merge(summary));

  base::StatusOr<Output> result = (*bm2)->PartialResult();
  EXPECT_OK(result);
  // Overflowing should result in the running sum wrapping around to zero.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0.0);

  // Test post-overflow serialize & merge
  summary = (*bm2)->Serialize();
  bm2 = builder.Build();
  EXPECT_OK((*bm2)->Merge(summary));
  result = (*bm2)->PartialResult();
  EXPECT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0.0);
}

TEST(BoundedMeanTest, UnderflowMergeManualBoundsTest) {
  typename BoundedMean<int64_t>::Builder builder;

  base::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(std::numeric_limits<int64_t>::lowest() + 1)
          .SetUpper(0)
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);
  (*bm)->AddEntry(-1);
  Summary summary = (*bm)->Serialize();

  base::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm2 = builder.Build();
  (*bm2)->AddEntry(-1);
  (*bm2)->AddEntry(-1);
  (*bm2)->AddEntry(-1);
  (*bm2)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);

  ASSERT_OK((*bm2)->Merge(summary));

  base::StatusOr<Output> result = (*bm2)->PartialResult();
  EXPECT_OK(result);
  // Underflowing should result in the running sum wrapping around to zero.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0.0);

  // Test post-overflow serialize & merge
  summary = (*bm2)->Serialize();
  bm2 = builder.Build();
  EXPECT_OK((*bm2)->Merge(summary));
  result = (*bm2)->PartialResult();
  EXPECT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0.0);
}

TYPED_TEST(BoundedMeanTest, SerializeMergeTest) {
  typename BoundedMean<TypeParam>::Builder builder;

  auto bm1 =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(3)
          .Build();
  ASSERT_OK(bm1);
  (*bm1)->AddEntry(1);
  Summary summary = (*bm1)->Serialize();
  (*bm1)->AddEntry(3);

  auto bm2 = builder.Build();
  ASSERT_OK(bm2);
  EXPECT_OK((*bm2)->Merge(summary));
  (*bm2)->AddEntry(3);

  auto result1 = (*bm1)->PartialResult();
  ASSERT_OK(result1);
  auto result2 = (*bm2)->PartialResult();
  ASSERT_OK(result2);

  EXPECT_DOUBLE_EQ(GetValue<double>(*result1), GetValue<double>(*result2));
}

TYPED_TEST(BoundedMeanTest, SerializeMergePartialSumsTest) {
  typename ApproxBounds<TypeParam>::Builder bounds_builder;
  typename BoundedMean<TypeParam>::Builder builder;

  // Automatic bounding, so entries will be split and stored as partial sums.
  auto bounds =
      bounds_builder.SetThresholdForTest(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm1 =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(*bounds))
          .Build();
  ASSERT_OK(bm1);
  (*bm1)->AddEntry(-10);
  (*bm1)->AddEntry(4);
  Summary summary = (*bm1)->Serialize();
  (*bm1)->AddEntry(6);

  // Merge summary into second BoundedVariance.
  auto bounds2 = bounds_builder.Build();
  ASSERT_OK(bounds2);
  auto bm2 = builder.SetApproxBounds(std::move(*bounds2)).Build();
  ASSERT_OK(bm2);
  (*bm2)->AddEntry(6);
  EXPECT_OK((*bm2)->Merge(summary));

  // Check equality.  Bounds are set to [-16, 8].
  auto result1 = (*bm1)->PartialResult();
  ASSERT_OK(result1);
  auto result2 = (*bm2)->PartialResult();
  ASSERT_OK(result2);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result1), GetValue<double>(*result2));
}

TYPED_TEST(BoundedMeanTest, AutomaticBoundsNegative) {
  std::vector<TypeParam> a = {9, -2, -2, -1, -6, -6};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntries(a.begin(), a.end());

  // 9 gets clamped to -1.
  Output expected_output;
  AddToOutput<double>(&expected_output, -3);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), -8);
  SetValue<TypeParam>(report->mutable_upper_bound(), -1);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  auto actual_output = (*bm)->PartialResult();
  ASSERT_OK(actual_output);

  EXPECT_THAT(*actual_output, EqualsProto(expected_output));
}

TYPED_TEST(BoundedMeanTest, AutomaticBoundsPositive) {
  std::vector<TypeParam> a = {-9, 2, 2, 1, 6, 6};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntries(a.begin(), a.end());

  // -9 gets clamped to 1.
  Output expected_output;
  AddToOutput<double>(&expected_output, 3);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), 1);
  SetValue<TypeParam>(report->mutable_upper_bound(), 8);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  auto actual_output = (*bm)->PartialResult();
  ASSERT_OK(actual_output);

  EXPECT_THAT(*actual_output, EqualsProto(expected_output));
}

TEST(BoundedMeanTest, DropNanEntries) {
  std::vector<double> a = {2, 4, 6, NAN, 8};
  auto mean = BoundedMean<double>::Builder()
                  .SetEpsilon(1)
                  .SetLower(1)
                  .SetUpper(9)
                  .Build();
  ASSERT_OK(mean);
  auto result = (*mean)->Result(a.begin(), a.end());
  EXPECT_GE(GetValue<double>(*result), 1);
  EXPECT_LE(GetValue<double>(*result), 9);
}

TEST(BoundedMeanTest, SensitivityOverflow) {
  // Check for error when upper - lower causes integer overflow.
  EXPECT_EQ(BoundedMean<int>::Builder()
                .SetEpsilon(1.0)
                .SetLower(INT_MIN)
                .SetUpper(INT_MAX)
                .Build()
                .status()
                .message(),
            "Upper - lower caused integer overflow.");
}

TEST(BoundedMeanTest, SensitivityOverflowApproxBounds) {
  auto bounds =
      ApproxBounds<int>::Builder()
          .SetEpsilon(1)
          .SetThresholdForTest(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm = BoundedMean<int>::Builder()
                .SetEpsilon(1)
                .SetApproxBounds(std::move(*bounds))
                .Build();
  ASSERT_OK(bm);

  // Adding these two entries make the bounds [-1, max]. Sensitivity is
  // calculated |max - (-1)|, which overflowss.
  (*bm)->AddEntry(-1);
  (*bm)->AddEntry(INT_MAX);

  EXPECT_THAT((*bm)->PartialResult(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Upper - lower caused integer overflow.")));
}

// Test when 0 is in [lower, upper].
TYPED_TEST(BoundedMeanTest, AutomaticBoundsContainZero) {
  std::vector<TypeParam> a = {4,
                              4,
                              -1,
                              -1,
                              std::numeric_limits<TypeParam>::lowest(),
                              std::numeric_limits<TypeParam>::max()};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntries(a.begin(), a.end());

  Output expected_output;
  AddToOutput<double>(&expected_output, 1.5);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), -1);
  SetValue<TypeParam>(report->mutable_upper_bound(), 4);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  auto actual_output = (*bm)->PartialResult();
  ASSERT_OK(actual_output);

  EXPECT_THAT(*actual_output, EqualsProto(expected_output));
}

// Test not providing ApproxBounds and instead using the default.
TYPED_TEST(BoundedMeanTest, AutomaticBoundsDefault) {
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);
  std::vector<TypeParam> big(1000, 10);
  std::vector<TypeParam> small(1000, -10);
  (*bm)->AddEntries(big.begin(), big.end());
  (*bm)->AddEntries(small.begin(), small.end());

  BoundingReport bounding_report;
  SetValue<TypeParam>(bounding_report.mutable_lower_bound(), -16);
  SetValue<TypeParam>(bounding_report.mutable_upper_bound(), 16);
  bounding_report.set_num_inputs(big.size() + small.size());
  bounding_report.set_num_outside(0);
  Output::ErrorReport expected_report;
  *(expected_report.mutable_bounding_report()) = bounding_report;

  auto result = (*bm)->PartialResult();
  ASSERT_OK(result);
  EXPECT_THAT(result->error_report(), EqualsProto(expected_report));
  EXPECT_NEAR(GetValue<double>(result->elements(0).value()), 0.0,
              std::pow(10, -10));
}

// Test when a bound is 0.
TYPED_TEST(BoundedMeanTest, AutomaticBoundsZero) {
  std::vector<TypeParam> a = {0, 0, 4, 4, -2, 2, 7};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntries(a.begin(), a.end());

  // -2 gets clamped to 0. 7 gets clamped to 4.
  Output expected_output;
  AddToOutput<double>(&expected_output, 2);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), 0);
  SetValue<TypeParam>(report->mutable_upper_bound(), 4);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  auto actual_output = (*bm)->PartialResult();
  ASSERT_OK(actual_output);

  EXPECT_THAT(*actual_output, EqualsProto(expected_output));
}

TYPED_TEST(BoundedMeanTest, Reset) {
  // Construct bounded sum with approximate bounding.
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetThresholdForTest(1)
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);

  // Reset between adding vectors.
  std::vector<TypeParam> a = {-10, 1000};
  std::vector<TypeParam> b = {-100, 100, 3};
  (*bm)->AddEntries(a.begin(), a.end());
  (*bm)->Reset();
  (*bm)->AddEntries(b.begin(), b.end());

  // Check result is only affected by vector b.
  auto result = (*bm)->PartialResult();
  EXPECT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(result->elements(0).value()), 1);
}

TYPED_TEST(BoundedMeanTest, MemoryUsed) {
  auto bm = typename BoundedMean<TypeParam>::Builder().Build();
  EXPECT_GT((*bm)->MemoryUsed(), 0);
}

TYPED_TEST(BoundedMeanTest, SplitsEpsilonWithAutomaticBounds) {
  double epsilon = 1.0;
  base::StatusOr<std::unique_ptr<BoundedMean<TypeParam>>> bm =
      typename BoundedMean<TypeParam>::Builder().SetEpsilon(epsilon).Build();
  ASSERT_OK(bm);
  auto* bmi = dynamic_cast<BoundedMeanWithApproxBounds<TypeParam>*>(bm->get());
  EXPECT_NEAR(bmi->GetEpsilon(), epsilon, 1e-10);
  EXPECT_NEAR(bmi->GetEpsilon(),
              bmi->GetBoundingEpsilon() + bmi->GetAggregationEpsilon(), 1e-10);
  EXPECT_GT(bmi->GetBoundingEpsilon(), 0);
  EXPECT_LT(bmi->GetBoundingEpsilon(), epsilon);
  EXPECT_GT(bmi->GetAggregationEpsilon(), 0);
  EXPECT_LT(bmi->GetAggregationEpsilon(), epsilon);
}

TEST(BoundedMeanWithApproxBoundsTest, ConsumesAllBudgetOfNumericalMechanisms) {
  std::unique_ptr<test_utils::MockLaplaceMechanism> mock_count_mechanism =
      std::make_unique<test_utils::MockLaplaceMechanism>();
  std::unique_ptr<test_utils::MockLaplaceMechanism> mock_sum_mechanism =
      std::make_unique<test_utils::MockLaplaceMechanism>();

  test_utils::MockLaplaceMechanism* mock_count_ptr = mock_count_mechanism.get();
  test_utils::MockLaplaceMechanism* mock_sum_ptr = mock_sum_mechanism.get();

  // For a double bounded mean, we add int noise to the count and double noise
  // to the sum.
  EXPECT_CALL(*mock_count_ptr, AddInt64Noise(_, DoubleEq(1))).Times(1);
  EXPECT_CALL(*mock_sum_ptr, AddDoubleNoise(_, DoubleEq(1))).Times(1);

  BoundedMeanWithFixedBounds<double> bm(
      /*epsilon=*/1.0,
      /*delta=*/0,
      /*lower=*/-1,
      /*upper=*/1, std::move(mock_sum_mechanism),
      std::move(mock_count_mechanism));

  for (int i = 0; i < 10; ++i) {
    bm.AddEntry(1.0);
  }

  EXPECT_OK(bm.PartialResult());
}

}  //  namespace
}  // namespace differential_privacy
