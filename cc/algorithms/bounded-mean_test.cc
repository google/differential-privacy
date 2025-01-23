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
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
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
#include "algorithms/bounds-provider.h"
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
using ::testing::DoubleNear;
using ::testing::Eq;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Le;
using ::testing::Lt;
using ::testing::NotNull;
using ::testing::Property;
using ::differential_privacy::base::testing::IsOk;
using ::differential_privacy::base::testing::StatusIs;

constexpr double kSmallEpsilon = 0.00000001;
constexpr double kNumSamples = 10000;
constexpr double kDefaultEpsilon = 1.1;

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

  auto mean1 =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .Build();
  ASSERT_OK(mean1);

  auto mean2 =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .Build();
  ASSERT_OK(mean2);

  (*mean2)->AddEntries(a.begin(), a.end());

  auto result1 = (*mean1)->PartialResult();
  ASSERT_OK(result1);
  auto result2 = (*mean2)->PartialResult();
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
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
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
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
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
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
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

TEST(BoundedMeanTest, NormalizedSumHasExpectedSensitivity) {
  // Checking for rounding issues in the sensitivity calculation when the type
  // of the bounds is int.
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> m =
      BoundedMean<int>::BuildMechanismForNormalizedSum(
          std::make_unique<LaplaceMechanism::Builder>(), kDefaultEpsilon,
          /*delta=*/0,
          /*l0_sensitivity=*/1,
          /*max_contribution_per_partition=*/1, /*lower=*/0, /*upper=*/3);
  ASSERT_OK(m);

  LaplaceMechanism* lm = dynamic_cast<LaplaceMechanism*>(m->get());
  ASSERT_THAT(lm, NotNull());

  EXPECT_THAT(lm->GetSensitivity(), DoubleEq(3.0 / 2.0));
}

TYPED_TEST(BoundedMeanTest, InsufficientPrivacyBudgetTest) {
  std::vector<TypeParam> a = {2, 4, 6, 8};

  auto mean =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(9)
          .Build();
  ASSERT_OK(mean);
  (*mean)->AddEntries(a.begin(), a.end());

  ASSERT_OK((*mean)->PartialResult());
  EXPECT_THAT((*mean)->PartialResult(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("can only produce results once")));
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
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);

  // Automatic bounds are needed but there is no input, so the count-threshold
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

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
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

  absl::StatusOr<Output> result = (*bm)->PartialResult();
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

    absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
        builder
            .SetLaplaceMechanism(std::make_unique<LaplaceMechanism::Builder>())
            .SetLower(-kBound)
            .SetUpper(kBound)
            .Build();
    ASSERT_OK(bm);
    BoundedMeanTestPeer::AddMultipleEntries<double>(
        1, std::numeric_limits<int64_t>::max(), (*bm).get());

    absl::StatusOr<Output> result = (*bm)->PartialResult();
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

  absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-std::numeric_limits<int64_t>::max() / 2)
          .SetUpper(std::numeric_limits<int64_t>::max() / 2)
          .Build();
  ASSERT_OK(bm);
  BoundedMeanTestPeer::AddMultipleEntries<int64_t>(
      2, std::numeric_limits<int64_t>::max(), (*bm).get());

  absl::StatusOr<Output> result = (*bm)->PartialResult();
  ASSERT_OK(result);
  // Adding 2 * int64_max should overflow and wrap around to -2, resulting in a
  // mean of -2 / int64_max.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()),
                   -2.0 / std::numeric_limits<int64_t>::max());
}

TEST(BoundedMeanTest, OverflowAddEntryManualBoundsTest) {
  typename BoundedMean<int64_t>::Builder builder;

  absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(std::numeric_limits<int64_t>::max())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::max());
  (*bm)->AddEntry(1);
  (*bm)->AddEntry(1);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::max());

  absl::StatusOr<Output> result = (*bm)->PartialResult();
  EXPECT_OK(result);
  // Overflowing should result in the running sum wrapping around to zero.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0);
}

TEST(BoundedMeanTest, UnderflowAddEntryManualBoundsTest) {
  typename BoundedMean<int64_t>::Builder builder;

  absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
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

  absl::StatusOr<Output> result = (*bm)->PartialResult();
  EXPECT_OK(result);
  // Underflowing should result in the running sum wrapping around to zero.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 0);
}

TEST(BoundedMeanTest, OverflowRawCountMergeManualBoundsTest) {
  typename BoundedMean<double>::Builder builder;

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-10)
          .SetUpper(10)
          .Build();
  ASSERT_OK(bm);
  BoundedMeanTestPeer::AddMultipleEntries<double>(
      0, std::numeric_limits<int64_t>::max(), (*bm).get());
  Summary summary = (*bm)->Serialize();

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm2 = builder.Build();
  ASSERT_OK(bm2);
  BoundedMeanTestPeer::AddMultipleEntries<double>(
      0, std::numeric_limits<int64_t>::max(), (*bm2).get());
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(1);

  ASSERT_OK((*bm2)->Merge(summary));

  absl::StatusOr<Output> result = (*bm2)->PartialResult();
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

  absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(std::numeric_limits<int64_t>::max())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::max());
  Summary summary = (*bm)->Serialize();

  absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm2 = builder.Build();
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(1);
  (*bm2)->AddEntry(std::numeric_limits<int64_t>::max());

  ASSERT_OK((*bm2)->Merge(summary));

  absl::StatusOr<Output> result = (*bm2)->PartialResult();
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

  absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm =
      builder
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(std::numeric_limits<int64_t>::lowest() + 1)
          .SetUpper(0)
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);
  (*bm)->AddEntry(-1);
  Summary summary = (*bm)->Serialize();

  absl::StatusOr<std::unique_ptr<BoundedMean<int64_t>>> bm2 = builder.Build();
  (*bm2)->AddEntry(-1);
  (*bm2)->AddEntry(-1);
  (*bm2)->AddEntry(-1);
  (*bm2)->AddEntry(std::numeric_limits<int64_t>::lowest() + 1);

  ASSERT_OK((*bm2)->Merge(summary));

  absl::StatusOr<Output> result = (*bm2)->PartialResult();
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
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
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
  // Automatic bounding, so entries will be split and stored as partial sums.
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetThresholdForTest(0.5)
          .SetNumBins(10)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds);
  auto bm1 =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(bounds).value())
          .SetEpsilon(kDefaultEpsilon)
          .Build();
  ASSERT_OK(bm1);
  (*bm1)->AddEntry(-10);
  (*bm1)->AddEntry(4);
  Summary summary = (*bm1)->Serialize();
  (*bm1)->AddEntry(6);

  // Merge summary into second BoundedVariance.
  auto bounds2 =
      typename ApproxBounds<TypeParam>::Builder()
          .SetThresholdForTest(0.5)
          .SetNumBins(10)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds2);
  auto bm2 =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(bounds2).value())
          .SetEpsilon(kDefaultEpsilon)
          .Build();
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

TYPED_TEST(BoundedMeanTest, SerializeMergePartialSumsWithBoundsProvider) {
  // Automatic bounding, so entries will be split and stored as partial sums.
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetThresholdForTest(0.5)
          .SetNumBins(10)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds);
  std::unique_ptr<BoundsProvider<TypeParam>> bounds_provider =
      std::make_unique<ApproxBoundsAsBoundsProvider<TypeParam>>(
          std::move(bounds).value());
  auto bm1 =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetBoundsProvider(std::move(bounds_provider))
          .SetEpsilon(kDefaultEpsilon)
          .Build();
  ASSERT_OK(bm1);
  (*bm1)->AddEntry(-10);
  (*bm1)->AddEntry(4);
  Summary summary = (*bm1)->Serialize();
  (*bm1)->AddEntry(6);

  // Merge summary into second BoundedVariance.
  auto bounds2 =
      typename ApproxBounds<TypeParam>::Builder()
          .SetThresholdForTest(0.5)
          .SetNumBins(10)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds2);
  std::unique_ptr<BoundsProvider<TypeParam>> bounds_provider2 =
      std::make_unique<ApproxBoundsAsBoundsProvider<TypeParam>>(
          std::move(bounds2).value());
  auto bm2 =
      typename BoundedMean<TypeParam>::Builder()
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetBoundsProvider(std::move(bounds_provider2))
          .SetEpsilon(kDefaultEpsilon)
          .Build();
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

// This test will be removed when removing backwards compatibility for the
// `bounds_summary` field.
TYPED_TEST(BoundedMeanTest, SerializeMergeApproxBoundsBackwardsCompatability) {
  absl::StatusOr<std::unique_ptr<BoundedMean<TypeParam>>> bounds1 =
      typename BoundedMean<TypeParam>::Builder().SetEpsilon(1e10).Build();
  ASSERT_OK(bounds1.status());

  for (int i = 0; i < 100; ++i) {
    bounds1.value()->AddEntry(10);
  }

  absl::StatusOr<std::unique_ptr<BoundedMean<TypeParam>>> bounds2 =
      typename BoundedMean<TypeParam>::Builder().SetEpsilon(1.0).Build();
  ASSERT_OK(bounds2.status());

  // Remove the newly introduced field as this field is ignored by versions
  // before the proto change.
  Summary bounds1_summary = bounds1.value()->Serialize();
  BoundedMeanSummary bm_summary;
  bounds1_summary.data().UnpackTo(&bm_summary);
  bm_summary.clear_bounds();
  bounds1_summary.mutable_data()->PackFrom(bm_summary);

  ASSERT_OK(bounds2.value()->Merge(bounds1_summary));
  EXPECT_THAT(bounds2.value()->PartialResult(), IsOk());
}

TYPED_TEST(BoundedMeanTest, AutomaticBoundsNegative) {
  std::vector<TypeParam> a = {9, -2, -2, -1, -6, -6};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1.5)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntries(a.begin(), a.end());

  absl::StatusOr<Output> output = (*bm)->PartialResult();
  ASSERT_OK(output);

  // 9 gets clamped to -1.
  EXPECT_THAT(output->elements_size(), Eq(1));
  EXPECT_THAT(GetValue<double>(output->elements(0).value()), DoubleEq(-3.0));
  EXPECT_THAT(GetValue<TypeParam>(
                  output->error_report().bounding_report().lower_bound()),
              Eq(-8));
  EXPECT_THAT(GetValue<TypeParam>(
                  output->error_report().bounding_report().upper_bound()),
              Eq(-1));
  EXPECT_THAT(output->error_report().bounding_report().num_inputs(),
              DoubleEq(a.size()));
  EXPECT_THAT(output->error_report().bounding_report().num_outside(),
              DoubleEq(2.0));
}

TYPED_TEST(BoundedMeanTest, AutomaticBoundsPositive) {
  std::vector<TypeParam> a = {-9, 2, 2, 1, 6, 6};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1.5)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon)
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntries(a.begin(), a.end());

  absl::StatusOr<Output> output = (*bm)->PartialResult();
  ASSERT_OK(output);

  // -9 gets clamped to 1.
  EXPECT_THAT(output->elements_size(), Eq(1));
  EXPECT_THAT(GetValue<double>(output->elements(0).value()), DoubleEq(3.0));
  EXPECT_THAT(GetValue<TypeParam>(
                  output->error_report().bounding_report().lower_bound()),
              Eq(1));
  EXPECT_THAT(GetValue<TypeParam>(
                  output->error_report().bounding_report().upper_bound()),
              Eq(8));
  EXPECT_THAT(output->error_report().bounding_report().num_inputs(),
              DoubleEq(a.size()));
  EXPECT_THAT(output->error_report().bounding_report().num_outside(),
              DoubleEq(2.0));
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
          .SetEpsilon(0.5)
          .SetThresholdForTest(0.5)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm = BoundedMean<int>::Builder()
                .SetEpsilon(1)
                .SetApproxBounds(std::move(bounds).value())
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
          .SetEpsilon(0.5)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(2)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntries(a.begin(), a.end());

  absl::StatusOr<Output> output = (*bm)->PartialResult();
  ASSERT_OK(output);

  EXPECT_THAT(output->elements_size(), Eq(1));
  EXPECT_THAT(GetValue<double>(output->elements(0).value()), DoubleEq(1.5));
  EXPECT_THAT(GetValue<TypeParam>(
                  output->error_report().bounding_report().lower_bound()),
              Eq(-1));
  EXPECT_THAT(GetValue<TypeParam>(
                  output->error_report().bounding_report().upper_bound()),
              Eq(4));
  EXPECT_THAT(output->error_report().bounding_report().num_inputs(),
              DoubleEq(a.size()));
  EXPECT_THAT(output->error_report().bounding_report().num_outside(),
              DoubleEq(2.0));
}

// Test not providing ApproxBounds and instead using the default.
TYPED_TEST(BoundedMeanTest, AutomaticBoundsDefault) {
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
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

TEST(BoundedMeanTest, BuilderWithApproxBoundsMoreBudgetThanTotalBudgetFails) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder().SetEpsilon(1.1).Build();
  ASSERT_OK(bounds);
  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      BoundedMean<double>::Builder()
          .SetEpsilon(1.09)
          .SetApproxBounds(std::move(bounds).value())
          .Build();
  ASSERT_THAT(
      bm.status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("consumes more epsilon budget than available")));
}

// Test when a bound is 0.
TYPED_TEST(BoundedMeanTest, AutomaticBoundsZero) {
  std::vector<TypeParam> a = {0, 0, 4, 4, -2, 2, 7};
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(0.5)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1.5)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bm);
  (*bm)->AddEntries(a.begin(), a.end());

  absl::StatusOr<Output> output = (*bm)->PartialResult();
  ASSERT_OK(output);

  // -2 gets clamped to 0. 7 gets clamped to 4.
  EXPECT_THAT(output->elements_size(), Eq(1));
  EXPECT_THAT(GetValue<double>(output->elements(0).value()), DoubleEq(2.0));
  EXPECT_THAT(GetValue<TypeParam>(
                  output->error_report().bounding_report().lower_bound()),
              Eq(0));
  EXPECT_THAT(GetValue<TypeParam>(
                  output->error_report().bounding_report().upper_bound()),
              Eq(4));
  EXPECT_THAT(output->error_report().bounding_report().num_inputs(),
              DoubleEq(a.size()));
  EXPECT_THAT(output->error_report().bounding_report().num_outside(),
              DoubleEq(2.0));
}

TYPED_TEST(BoundedMeanTest, Reset) {
  // Construct bounded sum with approximate bounding.
  auto bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetThresholdForTest(0.5)
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds);
  auto bm =
      typename BoundedMean<TypeParam>::Builder()
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon)
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
  absl::StatusOr<std::unique_ptr<BoundedMean<TypeParam>>> bm =
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
  EXPECT_CALL(*mock_count_ptr, AddInt64Noise(_)).Times(1);
  EXPECT_CALL(*mock_sum_ptr, AddDoubleNoise(_)).Times(1);

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

TEST(BoundedMeanWithFixedBoundsTest, ApproxBoundsMechanismHasExpectedVariance) {
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

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      BoundedMean<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetMaxPartitionsContributed(max_partitions_contributed)
          .SetMaxContributionsPerPartition(max_contributions_per_partition)
          .Build();
  ASSERT_OK(bm);

  auto* bm_with_approx_bounds =
      static_cast<BoundedMeanWithApproxBounds<double>*>(bm.value().get());
  ASSERT_THAT(bm_with_approx_bounds, NotNull());
  auto* approx_bounds_as_bounds_provider =
      static_cast<ApproxBoundsAsBoundsProvider<double>*>(
          bm_with_approx_bounds->GetBoundsProviderForTesting());
  ASSERT_THAT(approx_bounds_as_bounds_provider, NotNull());

  EXPECT_THAT(approx_bounds_as_bounds_provider->GetApproxBoundsForTesting()
                  ->GetMechanismForTesting()
                  ->GetVariance(),
              DoubleEq(expected_variance));
}

TEST(BoundedMeanTest, ConfidenceIntervalWithNoisedResultOkWithPosMidpoint) {
  const double lower = -1;
  const double upper = 1;
  const double confidence_level = 0.95;
  // Use 1000 as noised contribution count and 500 as noised total sum -> CI
  // midpoint should be close to 0.5.
  const double noised_count = 1000;
  const double noised_sum = 500;

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      typename BoundedMean<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .Build();
  ASSERT_OK(bm);

  BoundedMeanWithFixedBounds<double>* fixed_bm =
      dynamic_cast<BoundedMeanWithFixedBounds<double>*>(bm->get());
  ASSERT_THAT(fixed_bm, NotNull());

  absl::StatusOr<ConfidenceInterval> ci = fixed_bm->NoiseConfidenceInterval(
      confidence_level, noised_sum, noised_count);
  ASSERT_OK(ci);
  EXPECT_THAT(ci->lower_bound(), Lt(ci->upper_bound()));
  EXPECT_THAT(ci->confidence_level(), DoubleEq(confidence_level));

  const double ci_midpoint = (ci->upper_bound() + ci->lower_bound()) / 2.0;
  ASSERT_THAT(ci_midpoint, DoubleNear(0.5, 0.01));
}

TEST(BoundedMeanTest, ConfidenceIntervalWithNoisedResultOkWithNegMidpoint) {
  const double lower = -1;
  const double upper = 1;
  const double confidence_level = 0.95;
  // Use 1000 as noised contribution count and -500 as noised total sum -> CI
  // midpoint should be close to -0.5.
  const double noised_count = 1000;
  const double noised_sum = -500;

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      typename BoundedMean<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .Build();
  ASSERT_OK(bm);

  BoundedMeanWithFixedBounds<double>* fixed_bm =
      dynamic_cast<BoundedMeanWithFixedBounds<double>*>(bm->get());
  ASSERT_THAT(fixed_bm, NotNull());

  absl::StatusOr<ConfidenceInterval> ci = fixed_bm->NoiseConfidenceInterval(
      confidence_level, noised_sum, noised_count);
  ASSERT_OK(ci);
  EXPECT_THAT(ci->lower_bound(), Lt(ci->upper_bound()));
  EXPECT_THAT(ci->confidence_level(), DoubleEq(confidence_level));

  const double ci_midpoint = (ci->upper_bound() + ci->lower_bound()) / 2.0;
  ASSERT_THAT(ci_midpoint, DoubleNear(-0.5, 0.01));
}

TEST(BoundedMeanTest, ConfidenceIntervalWithLowerLevelGetsTighter) {
  const double lower = -1;
  const double upper = 1;
  const double confidence_level_higher = 0.95;
  const double confidence_level_lower = 0.8;
  const double noised_count = 1000;
  const double noised_sum = -500;

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      typename BoundedMean<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .Build();
  ASSERT_OK(bm);

  BoundedMeanWithFixedBounds<double>* fixed_bm =
      dynamic_cast<BoundedMeanWithFixedBounds<double>*>(bm->get());
  ASSERT_THAT(fixed_bm, NotNull());

  absl::StatusOr<ConfidenceInterval> ci_higher =
      fixed_bm->NoiseConfidenceInterval(confidence_level_higher, noised_sum,
                                        noised_count);
  ASSERT_OK(ci_higher);
  absl::StatusOr<ConfidenceInterval> ci_lower =
      fixed_bm->NoiseConfidenceInterval(confidence_level_lower, noised_sum,
                                        noised_count);
  ASSERT_OK(ci_lower);

  // The CI returned with higher confidence level has to be included in the CI
  // for the lower confidence level.
  EXPECT_THAT(ci_higher->lower_bound(), Lt(ci_lower->lower_bound()));
  EXPECT_THAT(ci_higher->upper_bound(), Gt(ci_lower->upper_bound()));
}

TEST(BoundedMeanTest, ConfidenceIntervalWithMoreDataGetsTighter) {
  const double lower = -1;
  const double upper = 1;
  const double confidence_level = 0.95;

  const double noised_count_many_users = 1000;
  const double noised_sum_many_users = -500;

  const double noised_count_fewer_users = 100;
  const double noised_sum_fewer_users = -50;

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      typename BoundedMean<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .Build();
  ASSERT_OK(bm);

  BoundedMeanWithFixedBounds<double>* fixed_bm =
      dynamic_cast<BoundedMeanWithFixedBounds<double>*>(bm->get());
  ASSERT_THAT(fixed_bm, NotNull());

  absl::StatusOr<ConfidenceInterval> ci_many_users =
      fixed_bm->NoiseConfidenceInterval(confidence_level, noised_sum_many_users,
                                        noised_count_many_users);
  ASSERT_OK(ci_many_users);
  absl::StatusOr<ConfidenceInterval> ci_fewer_users =
      fixed_bm->NoiseConfidenceInterval(
          confidence_level, noised_sum_fewer_users, noised_count_fewer_users);
  ASSERT_OK(ci_fewer_users);

  // The CI returned with fewer users has to be included in the CI for many
  // users.
  EXPECT_THAT(ci_fewer_users->lower_bound(), Lt(ci_many_users->lower_bound()));
  EXPECT_THAT(ci_fewer_users->upper_bound(), Gt(ci_many_users->upper_bound()));
}

TEST(BoundedMeanTest, EmptyFixedBoundsMeanOutputHasConfidenceInterval) {
  const double confidence_level = 0.987654;

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      BoundedMean<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLower(0.0)
          .SetUpper(1.0)
          .Build();
  ASSERT_OK(bm);

  absl::StatusOr<Output> output = bm->get()->PartialResult(confidence_level);
  ASSERT_OK(output);

  EXPECT_TRUE(output->elements(0).has_noise_confidence_interval());
  EXPECT_THAT(
      output->elements(0).noise_confidence_interval().confidence_level(),
      DoubleEq(confidence_level));
}

TEST(BoundedMeanTest, FixedBoundsMeanOutputHasConfidenceIntervalWithinBounds) {
  const double confidence_level = 0.987654;
  const double lower = -0.01;
  const double upper = 1.01;
  const int num_contributions = 1000;
  const double input = (lower + upper) / 2.0;

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      BoundedMean<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLower(lower)
          .SetUpper(upper)
          .Build();
  ASSERT_OK(bm);

  for (int i = 0; i < num_contributions; ++i) {
    bm->get()->AddEntry(input);
  }
  absl::StatusOr<Output> output = bm->get()->PartialResult(confidence_level);
  ASSERT_OK(output);

  EXPECT_THAT(
      output->elements(0).noise_confidence_interval().confidence_level(),
      DoubleEq(confidence_level));
  EXPECT_THAT(output->elements(0).noise_confidence_interval().lower_bound(),
              Gt(lower));
  EXPECT_THAT(output->elements(0).noise_confidence_interval().upper_bound(),
              Lt(upper));
}

TEST(BoundedMeanTest, ApproxBoundsMeanOutputHasConfidenceInterval) {
  const double confidence_level = 0.95432;
  const int num_contributions = 1000;
  const double input = 3.2;

  absl::StatusOr<std::unique_ptr<BoundedMean<double>>> bm =
      BoundedMean<double>::Builder().SetEpsilon(kDefaultEpsilon).Build();
  ASSERT_OK(bm);

  for (int i = 0; i < num_contributions; ++i) {
    bm->get()->AddEntry(input);
  }
  absl::StatusOr<Output> output = bm->get()->PartialResult(confidence_level);
  ASSERT_OK(output);

  const double lower_bound =
      GetValue<double>(output->error_report().bounding_report().lower_bound());
  const double upper_bound =
      GetValue<double>(output->error_report().bounding_report().upper_bound());
  EXPECT_THAT(
      output->elements(0).noise_confidence_interval(),
      AllOf(Property("confidence_level", &ConfidenceInterval::confidence_level,
                     DoubleEq(confidence_level)),
            Property("lower_bound", &ConfidenceInterval::lower_bound,
                     Ge(lower_bound)),
            Property("upper_bound", &ConfidenceInterval::upper_bound,
                     Le(upper_bound))));
}

}  //  namespace
}  // namespace differential_privacy
