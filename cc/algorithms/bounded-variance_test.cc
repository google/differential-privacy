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

#include "algorithms/bounded-variance.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/approx-bounds-as-bounds-provider.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"

namespace differential_privacy {

// Provides limited-scope static methods for interacting with a BoundedVariance
// object for testing purposes.
class BoundedVarianceTestPeer {
 public:
  template <typename T,
            std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  static void AddMultipleEntries(const T& t, int64_t num_of_entries,
                                 BoundedVariance<T>* bv) {
    bv->AddMultipleEntries(t, num_of_entries);
  }
};

namespace {

using test_utils::ZeroNoiseMechanism;
using ::testing::_;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::differential_privacy::base::testing::IsOk;
using ::differential_privacy::base::testing::StatusIs;

constexpr double kSmallEpsilon = 0.00000001;
constexpr int64_t kNumSamples = 10000;
constexpr double kDefaultEpsilon = 1.1;
// Max upper bound (and negative lower bound) BoundedVariance will accept
// Used in overflow-related tests
const int64_t kSqrtInt64Max = std::sqrt(std::numeric_limits<int64_t>::max());

template <typename T>
class BoundedVarianceTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 private:
  double Variance(std::vector<T> values) {
    if (values.empty()) {
      return 0;
    }
    int num_of_values = values.size();
    double mean = accumulate(values.begin(), values.end(), 0.0) / num_of_values;
    double variance = 0;
    for (int i = 0; i < num_of_values; ++i) {
      variance += (values[i] - mean) * (values[i] - mean);
    }
    variance /= num_of_values;
    return variance;
  }
};

// Typed test to iterate all test cases through all supported versions of
// BoundedVarianceTest, currently (int64_t, double).
typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(BoundedVarianceTest, NumericTypes);

TYPED_TEST(BoundedVarianceTest, BasicTest) {
  std::vector<TypeParam> a = {1, 2, 3, 4, 5};
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(6)
          .Build();
  ASSERT_OK(bv);
  absl::StatusOr<Output> result = (*bv)->Result(a.begin(), a.end());
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result), 2.0);
}

TYPED_TEST(BoundedVarianceTest, BasicMultipleEntriesTest) {
  std::vector<TypeParam> a = {1, 2, 3, 4, 5};
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(6)
          .Build();
  ASSERT_OK(bv);
  for (const auto& input : a) {
    BoundedVarianceTestPeer::AddMultipleEntries<TypeParam>(input, input,
                                                           (*bv).get());
  }
  absl::StatusOr<Output> result = (*bv)->PartialResult();
  ASSERT_OK(result);
  EXPECT_NEAR(GetValue<double>(*result), 14.0 / 9.0, 0.0000001);
}

TEST(BoundedVarianceTest, AddMultipleEntriesInvalidInputTest) {
  absl::StatusOr<std::unique_ptr<BoundedVariance<float>>> bv =
      typename BoundedVariance<float>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(6)
          .Build();
  ASSERT_OK(bv);

  // Add a few basic entries so we can expect a predictable variance.
  std::vector<float> a = {1, 2, 3, 4, 5};
  for (const auto& input : a) {
    BoundedVarianceTestPeer::AddMultipleEntries<float>(input, input,
                                                       (*bv).get());
  }

  BoundedVarianceTestPeer::AddMultipleEntries<float>(
      std::numeric_limits<float>::quiet_NaN(), 1, (*bv).get());
  absl::StatusOr<Output> result = (*bv)->PartialResult();
  ASSERT_OK(result);
  EXPECT_NEAR(GetValue<double>(*result), 14.0 / 9.0, 0.0000001);
}

TYPED_TEST(BoundedVarianceTest, AddMultipleEntriesInvalidNumberOfEntriesTest) {
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(6)
          .Build();
  ASSERT_OK(bv);

  // Add a few basic entries so we can expect a predictable variance.
  std::vector<TypeParam> a = {1, 2, 3, 4, 5};
  for (const auto& input : a) {
    BoundedVarianceTestPeer::AddMultipleEntries<TypeParam>(input, input,
                                                           (*bv).get());
  }

  std::vector<int64_t> invalid_entries{0, -1,
                                       std::numeric_limits<int64_t>::lowest()};
  for (int64_t n_entries : invalid_entries) {
    BoundedVarianceTestPeer::AddMultipleEntries<TypeParam>(1, n_entries,
                                                           (*bv).get());
  }

  absl::StatusOr<Output> result = (*bv)->PartialResult();
  ASSERT_OK(result);
  EXPECT_NEAR(GetValue<double>(*result), 14.0 / 9.0, 0.0000001);
}

TYPED_TEST(BoundedVarianceTest, RepeatedResultTest) {
  std::vector<TypeParam> a = {1, 2, 3, 4, 5};
  typename BoundedVariance<TypeParam>::Builder builder;
  builder.SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
      .SetEpsilon(1.0)
      .SetLower(0)
      .SetUpper(6);

  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv1 =
      builder.Build();
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv2 =
      builder.Build();
  ASSERT_OK(bv1);
  ASSERT_OK(bv2);
  (*bv1)->AddEntries(a.begin(), a.end());
  (*bv2)->AddEntries(a.begin(), a.end());
  absl::StatusOr<Output> result1 = (*bv1)->PartialResult(0.5);
  ASSERT_OK(result1);
  absl::StatusOr<Output> result2 = (*bv2)->PartialResult(0.5);
  ASSERT_OK(result2);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result1), GetValue<double>(*result2));
}

TYPED_TEST(BoundedVarianceTest, InsufficientPrivacyBudgetTest) {
  std::vector<TypeParam> a = {1, 2, 3, 4, 5};
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(6)
          .Build();
  ASSERT_OK(bv);
  (*bv)->AddEntries(a.begin(), a.end());
  ASSERT_OK((*bv)->PartialResult());
  EXPECT_THAT((*bv)->PartialResult(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("can only produce results once")));
}

TYPED_TEST(BoundedVarianceTest, ClampInputTest) {
  std::vector<TypeParam> a = {0, 0, 1, 1};
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(3)
          .Build();
  ASSERT_OK(bv);
  absl::StatusOr<Output> result = (*bv)->Result(a.begin(), a.end());
  ASSERT_OK(result);
  // The input will be clamped to {1, 1, 1, 1} returning variance 0.
  EXPECT_DOUBLE_EQ(GetValue<double>(*result), 0.0);
}

TYPED_TEST(BoundedVarianceTest, ClampOutputLowerTest) {
  std::vector<TypeParam> a = {1, 1, 1, 1};

  for (int i = 0; i < kNumSamples; ++i) {
    absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
        typename BoundedVariance<TypeParam>::Builder()
            .SetEpsilon(kSmallEpsilon)
            .SetLower(1)
            .SetUpper(2)
            .Build();
    ASSERT_OK(bv);
    absl::StatusOr<Output> result = (*bv)->Result(a.begin(), a.end());
    EXPECT_GE(GetValue<double>(*result), 0.0);
  }
}

TYPED_TEST(BoundedVarianceTest, ClampOutputUpperTest) {
  std::vector<TypeParam> a = {0, 10};

  for (int i = 0; i < kNumSamples; ++i) {
    absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
        typename BoundedVariance<TypeParam>::Builder()
            .SetEpsilon(kSmallEpsilon)
            .SetLower(0)
            .SetUpper(10)
            .Build();
    ASSERT_OK(bv);
    absl::StatusOr<Output> result = (*bv)->Result(a.begin(), a.end());
    ASSERT_OK(result);
    EXPECT_LE(GetValue<double>(*result), 25.0);
  }
}

TYPED_TEST(BoundedVarianceTest, EmptyInputsBoundsTest) {
  std::vector<TypeParam> a;
  double lower = 0;
  double upper = 10;
  // See header comment in BoundedVariance that states "The output will also be
  // clamped between 0 and (upper - lower)^2."
  double result_lower = 0;
  double result_upper = std::pow(upper - lower, 2);
  int num_of_trials = 100;  // Chosen abitrarily
  for (int i = 0; i < num_of_trials; ++i) {
    absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
        typename BoundedVariance<TypeParam>::Builder()
            .SetLaplaceMechanism(
                absl::make_unique<ZeroNoiseMechanism::Builder>())
            .SetEpsilon(1.0)
            .SetLower(lower)
            .SetUpper(upper)
            .Build();
    ASSERT_OK(bv);
    absl::StatusOr<Output> result = (*bv)->Result(a.begin(), a.end());
    ASSERT_OK(result);
    EXPECT_GE(GetValue<double>(*result), result_lower);
    EXPECT_LE(GetValue<double>(*result), result_upper);
  }
}

TYPED_TEST(BoundedVarianceTest, GaussianInputTest) {
  // Test samples points from a Gaussian and checks that the differentially
  // private variance is the variance used to generate the distribution.
  absl::StatusOr<std::unique_ptr<BoundedVariance<double>>> bv =
      BoundedVariance<double>::Builder()
          .SetLower(-20)
          .SetUpper(20)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv);
  // Generate the large sample of points from a seeded Laplace distribution.
  constexpr int samples = 100000;
  std::mt19937 gen;
  double stdev = 2;
  for (int i = 0; i < samples; i++) {
    (*bv)->AddEntry(std::round(absl::Gaussian<double>(gen, 0, stdev)));
  }
  absl::StatusOr<Output> result = (*bv)->PartialResult();
  ASSERT_OK(result);
  EXPECT_NEAR(GetValue<double>(*result), stdev * stdev, 0.1);
}

TYPED_TEST(BoundedVarianceTest, MaxContributionsVarianceTest) {
  const std::vector<TypeParam> input = {-1, -1, 1, 1};

  // Calculate variance of input.
  const double real_mean =
      std::accumulate(input.begin(), input.end(), 0.0) / input.size();
  double sum = 0;
  for (const auto& i : input) {
    sum += std::pow(i - real_mean, 2);
  }
  const double real_variance = sum / (input.size() - 1);

  std::function<double(int)> sample_variance_for_max_contribution =
      [&input, real_variance](int max_contribution) {
        double sum = 0;
        for (int i = 0; i < kNumSamples; ++i) {
          auto variance = typename BoundedVariance<TypeParam>::Builder()
                              .SetMaxContributionsPerPartition(max_contribution)
                              .SetEpsilon(1)
                              .SetLower(-1)
                              .SetUpper(1)
                              .Build();
          CHECK_EQ(variance.status(), absl::OkStatus());
          auto out = (*variance)->Result(input.begin(), input.end());
          CHECK_EQ(out.status(), absl::OkStatus());
          sum += std::pow(GetValue<double>(*out) - real_variance, 2);
        }
        return sum / (kNumSamples - 1);
      };

  // We expect the sample variance with max contribution 2 to be (significantly)
  // bigger than with max contribution 1.
  EXPECT_GT(sample_variance_for_max_contribution(2),
            1.1 * sample_variance_for_max_contribution(1));
}

TYPED_TEST(BoundedVarianceTest, MergeDifferentBoundingStrategy) {
  // Manual bounding.
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv1 =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLower(0)
          .SetUpper(3)
          .Build();
  ASSERT_OK(bv1);
  Summary summary = (*bv1)->Serialize();

  // Auto bounding.
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv2 =
      typename BoundedVariance<TypeParam>::Builder().Build();
  ASSERT_OK(bv2);

  // Error due to different strategies.
  ASSERT_THAT(
      (*bv2)->Merge(summary),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "Merged BoundedVariance must have the same bounding strategy.")));
}

TYPED_TEST(BoundedVarianceTest, SerializeMergeTest) {
  // Get summary of first BoundedVariance between entries.
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv1 =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(.5)
          .SetLower(0)
          .SetUpper(3)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv1);
  (*bv1)->AddEntry(2);
  Summary summary = (*bv1)->Serialize();
  (*bv1)->AddEntry(6);

  // Merge summary into second BoundedVariance.
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv2 =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(.5)
          .SetLower(0)
          .SetUpper(3)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv2);
  (*bv2)->AddEntry(6);
  EXPECT_OK((*bv2)->Merge(summary));

  // Check equality.
  absl::StatusOr<Output> result1 = (*bv1)->PartialResult();
  ASSERT_OK(result1);
  absl::StatusOr<Output> result2 = (*bv2)->PartialResult();
  ASSERT_OK(result2);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result1), GetValue<double>(*result2));
}

TYPED_TEST(BoundedVarianceTest,
           SerializeMergePartialValuesWithApproxBoundsTest) {
  typename ApproxBounds<TypeParam>::Builder bounds_builder;
  typename BoundedVariance<TypeParam>::Builder builder;

  // Automatic bounding, so entries will be split and stored as partials.
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds1 =
      typename ApproxBounds<TypeParam>::Builder()
          .SetThresholdForTest(0.5)
          .SetNumBins(50)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds1);
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv1 =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon)
          .SetApproxBounds(std::move(bounds1).value())
          .Build();
  ASSERT_OK(bv1);
  (*bv1)->AddEntry(-10);
  (*bv1)->AddEntry(4);
  Summary summary = (*bv1)->Serialize();
  (*bv1)->AddEntry(6);

  // Merge summary into second BoundedVariance.
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds2 =
      typename ApproxBounds<TypeParam>::Builder()
          .SetThresholdForTest(0.5)
          .SetNumBins(50)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds2);
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv2 =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon)
          .SetApproxBounds(std::move(bounds2).value())
          .Build();
  ASSERT_OK(bv2);
  (*bv2)->AddEntry(6);
  EXPECT_OK((*bv2)->Merge(summary));

  // Check equality. Bounds are set to [-16, 8].
  absl::StatusOr<Output> result1 = (*bv1)->PartialResult();
  ASSERT_OK(result1);
  absl::StatusOr<Output> result2 = (*bv2)->PartialResult();
  ASSERT_OK(result2);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result1), GetValue<double>(*result2));
}

TYPED_TEST(BoundedVarianceTest,
           SerializeMergePartialValuesWithBoundsProviderTest) {
  typename ApproxBounds<TypeParam>::Builder bounds_builder;
  typename BoundedVariance<TypeParam>::Builder builder;

  // Automatic bounding, so entries will be split and stored as partials.
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds1 =
      typename ApproxBounds<TypeParam>::Builder()
          .SetThresholdForTest(0.5)
          .SetNumBins(50)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds1);
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv1 =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon)
          .SetBoundsProvider(
              std::make_unique<ApproxBoundsAsBoundsProvider<TypeParam>>(
                  std::move(bounds1).value()))
          .Build();
  ASSERT_OK(bv1);
  (*bv1)->AddEntry(-10);
  (*bv1)->AddEntry(4);
  Summary summary = (*bv1)->Serialize();
  (*bv1)->AddEntry(6);

  // Merge summary into second BoundedVariance.
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds2 =
      typename ApproxBounds<TypeParam>::Builder()
          .SetThresholdForTest(0.5)
          .SetNumBins(50)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .Build();
  ASSERT_OK(bounds2);
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv2 =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon)
          .SetBoundsProvider(
              std::make_unique<ApproxBoundsAsBoundsProvider<TypeParam>>(
                  std::move(bounds2).value()))
          .Build();
  ASSERT_OK(bv2);
  (*bv2)->AddEntry(6);
  EXPECT_OK((*bv2)->Merge(summary));

  // Check equality. Bounds are set to [-16, 8].
  absl::StatusOr<Output> result1 = (*bv1)->PartialResult();
  ASSERT_OK(result1);
  absl::StatusOr<Output> result2 = (*bv2)->PartialResult();
  ASSERT_OK(result2);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result1), GetValue<double>(*result2));
}

// This test will be removed when removing backwards compatibility for the
// `bounds_summary` field.
TYPED_TEST(BoundedVarianceTest,
           SerializeMergeApproxBoundsBackwardsCompatability) {
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bounds1 =
      typename BoundedVariance<TypeParam>::Builder().SetEpsilon(1e10).Build();
  ASSERT_OK(bounds1.status());

  for (int i = 0; i < 100; ++i) {
    bounds1.value()->AddEntry(10);
  }

  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bounds2 =
      typename BoundedVariance<TypeParam>::Builder().SetEpsilon(1.0).Build();
  ASSERT_OK(bounds2.status());

  // Remove the newly introduced field as this field is ignored by versions
  // before the proto change.
  Summary bounds1_summary = bounds1.value()->Serialize();
  BoundedVarianceSummary bv_summary;
  bounds1_summary.data().UnpackTo(&bv_summary);
  bv_summary.clear_bounds();
  bounds1_summary.mutable_data()->PackFrom(bv_summary);

  ASSERT_OK(bounds2.value()->Merge(bounds1_summary));
  EXPECT_THAT(bounds2.value()->PartialResult(), IsOk());
}

TEST(BoundedVarianceTest, OverflowRawCountTest) {
  typename BoundedVariance<double>::Builder builder;

  std::unique_ptr<BoundedVariance<double>> bv =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-1)
          .SetUpper(1)
          .Build()
          .value();
  BoundedVarianceTestPeer::AddMultipleEntries<double>(
      -0.5, std::numeric_limits<int64_t>::max(), bv.get());
  BoundedVarianceTestPeer::AddMultipleEntries<double>(-0.5, 1, bv.get());
  BoundedVarianceTestPeer::AddMultipleEntries<double>(
      0.5, std::numeric_limits<int64_t>::max(), bv.get());
  BoundedVarianceTestPeer::AddMultipleEntries<double>(0.5, 1, bv.get());
  BoundedVarianceTestPeer::AddMultipleEntries<double>(1, 5, bv.get());

  auto result = bv->PartialResult();
  EXPECT_OK(result.status());
  // A partial_count_ overflow should result in a variance of 1.0.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 1.0);
}

TEST(BoundedVarianceTest, OverflowAddEntryManualBounds) {
  typename BoundedVariance<int32_t>::Builder builder;

  std::unique_ptr<BoundedVariance<int32_t>> bv =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-1)
          .SetUpper(1)
          .Build()
          .value();

  BoundedVarianceTestPeer::AddMultipleEntries<int32_t>(
      1, std::numeric_limits<int32_t>::max(), bv.get());
  BoundedVarianceTestPeer::AddMultipleEntries<int32_t>(
      1, std::numeric_limits<int32_t>::max(), bv.get());

  auto result = bv->PartialResult();
  ASSERT_OK(result.status());
  // Overflow should result in a variance of 1.0.
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 1.0);
}

TEST(BoundedVarianceTest, UnderflowAddEntryManualBounds) {
  typename BoundedVariance<int32_t>::Builder builder;

  std::unique_ptr<BoundedVariance<int32_t>> bv =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-1)
          .SetUpper(1)
          .Build()
          .value();
  BoundedVarianceTestPeer::AddMultipleEntries<int32_t>(
      -1, std::numeric_limits<int32_t>::max(), bv.get());
  BoundedVarianceTestPeer::AddMultipleEntries<int32_t>(
      -1, std::numeric_limits<int32_t>::max(), bv.get());

  auto result = bv->PartialResult();
  EXPECT_OK(result.status());
  // Overflow should result in a variance of 1.0
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 1.0);
}

TEST(BoundedVarianceTest, OverflowRawCountMergeManualBoundsTest) {
  typename BoundedVariance<double>::Builder builder;

  std::unique_ptr<BoundedVariance<double>> bv =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(0)
          .SetUpper(10)
          .Build()
          .value();
  BoundedVarianceTestPeer::AddMultipleEntries<double>(
      10, std::numeric_limits<int64_t>::max(), bv.get());

  Summary summary = bv->Serialize();

  std::unique_ptr<BoundedVariance<double>> bv2 = builder.Build().value();
  BoundedVarianceTestPeer::AddMultipleEntries<double>(
      10, std::numeric_limits<int64_t>::max(), bv.get());

  EXPECT_OK(bv2->Merge(summary));

  bv2->AddEntry(10);
  bv2->AddEntry(10);

  auto result = bv2->PartialResult();
  EXPECT_OK(result.status());
  // An overflow should cause the count of entries wrap around to 1, which
  // should result in the variance so large that it becomes clamped to
  // IntervalLengthSquared(lower, upper) / 4, instead of based upon the actual
  // data entries (which would be 0 if there was no count overflow, since all
  // entries are the same and do not vary).
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 25.0);
}

TEST(BoundedVarianceTest, OverflowMergeManualBoundsTest) {
  typename BoundedVariance<int32_t>::Builder builder;

  std::unique_ptr<BoundedVariance<int32_t>> bv =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-1)
          .SetUpper(1)
          .Build()
          .value();
  BoundedVarianceTestPeer::AddMultipleEntries<int32_t>(
      1, std::numeric_limits<int32_t>::max(), bv.get());
  Summary summary = bv->Serialize();

  std::unique_ptr<BoundedVariance<int32_t>> bv2 = builder.Build().value();
  BoundedVarianceTestPeer::AddMultipleEntries<int32_t>(
      1, std::numeric_limits<int32_t>::max(), bv2.get());

  ASSERT_OK(bv2->Merge(summary));

  auto result = bv2->PartialResult();
  ASSERT_OK(result.status());
  // Overflow should result in a variance of 1.0
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 1.0);
}

TEST(BoundedVarianceTest, UnderflowMergeManualBoundsTest) {
  typename BoundedVariance<int32_t>::Builder builder;

  std::unique_ptr<BoundedVariance<int32_t>> bv =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetLower(-1)
          .SetUpper(1)
          .Build()
          .value();
  BoundedVarianceTestPeer::AddMultipleEntries<int32_t>(
      -1, std::numeric_limits<int32_t>::max(), bv.get());
  Summary summary = bv->Serialize();

  std::unique_ptr<BoundedVariance<int32_t>> bv2 = builder.Build().value();
  BoundedVarianceTestPeer::AddMultipleEntries<int32_t>(
      -1, std::numeric_limits<int32_t>::max(), bv2.get());

  EXPECT_OK(bv2->Merge(summary));

  auto result = bv2->PartialResult();
  EXPECT_OK(result.status());
  // Underflow should result in a variance of 1.0
  EXPECT_DOUBLE_EQ(GetValue<double>(result.value()), 1.0);
}

TEST(BoundedVarianceTest, SensitivityOverflow) {
  absl::StatusOr<std::unique_ptr<BoundedVariance<int64_t>>> failed_bv =
      BoundedVariance<int64_t>::Builder()
          .SetEpsilon(1.0)
          .SetLower(std::numeric_limits<int64_t>::lowest())
          .SetUpper(std::numeric_limits<int64_t>::max())
          .Build();
  EXPECT_THAT(
      failed_bv,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Sensitivity calculation caused integer overflow.")));
}

TYPED_TEST(BoundedVarianceTest, SensitivityTooHigh) {
  // Make bounds so taking the interval squared won't overflow.
  absl::StatusOr<std::unique_ptr<BoundedVariance<double>>> failed_bv =
      BoundedVariance<double>::Builder()
          .SetLower(0)
          .SetUpper(std::pow(std::numeric_limits<double>::max() / 2, .5))
          .Build();
  EXPECT_THAT(failed_bv, StatusIs(absl::StatusCode::kInvalidArgument,
                                  HasSubstr("Sensitivity is too high.")));
}

TEST(BoundedVarianceTest, DropNanEntries) {
  std::vector<double> a = {1, 2, 3, 4, NAN, NAN, 5};
  absl::StatusOr<std::unique_ptr<BoundedVariance<double>>> bv =
      BoundedVariance<double>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1)
          .SetLower(0)
          .SetUpper(6)
          .Build();
  ASSERT_OK(bv);
  absl::StatusOr<Output> result = (*bv)->Result(a.begin(), a.end());
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(*result), 2.0);
}

TYPED_TEST(BoundedVarianceTest, PropagateApproxBoundsError) {
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder().SetEpsilon(1).Build();
  ASSERT_OK(bv);

  // Automatic bounds are needed but there is no input, so the count-threshhold
  // should exceed any bin count.
  EXPECT_THAT((*bv)->PartialResult(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("run over a larger dataset")));
}

TYPED_TEST(BoundedVarianceTest, AutomaticBoundsContainZero) {
  std::vector<TypeParam> a = {0, 8, -8,
                              std::numeric_limits<TypeParam>::lowest(),
                              std::numeric_limits<TypeParam>::max()};
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(0.5)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(*bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv);
  (*bv)->AddEntries(a.begin(), a.end());

  Output expected_output;
  AddToOutput<double>(&expected_output, 51.2);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), -8);
  SetValue<TypeParam>(report->mutable_upper_bound(), 8);
  report->set_num_inputs(a.size());
  report->set_num_outside(0);

  absl::StatusOr<Output> actual_output = (*bv)->PartialResult();
  ASSERT_OK(actual_output);
  EXPECT_THAT(*actual_output, EqualsProto(expected_output));
}

TEST(BoundedVarianceTest, AutomaticBoundsNegative) {
  std::vector<double> a = {5, -2, -2, -4, -6, -6};
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetEpsilon(0.5)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1.5)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  absl::StatusOr<std::unique_ptr<BoundedVariance<double>>> bv =
      BoundedVariance<double>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv);
  (*bv)->AddEntries(a.begin(), a.end());

  absl::StatusOr<Output> result = (*bv)->PartialResult();
  ASSERT_OK(result);

  // 5 gets clamped to -1
  double expected_sum = -21;
  double expected_sos = 97;
  double expected_variance =
      (expected_sos - expected_sum * expected_sum / a.size()) / a.size();

  EXPECT_THAT(GetValue<double>(*result),
              DoubleNear(expected_variance, expected_variance / 10000));

  BoundingReport expected_report;
  SetValue<double>(expected_report.mutable_lower_bound(), -8);
  SetValue<double>(expected_report.mutable_upper_bound(), -1);
  expected_report.set_num_inputs(a.size());
  expected_report.set_num_outside(1);

  EXPECT_THAT(result->error_report().bounding_report(),
              EqualsProto(expected_report));
}

TEST(BoundedVarianceTest, AutomaticBoundsPositive) {
  std::vector<double> a = {-5, 2, 2, 4, 6, 6};
  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetEpsilon(0.5)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1.5)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  absl::StatusOr<std::unique_ptr<BoundedVariance<double>>> bv =
      BoundedVariance<double>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv);
  (*bv)->AddEntries(a.begin(), a.end());

  absl::StatusOr<Output> result = (*bv)->PartialResult();
  ASSERT_OK(result);

  // -5 gets clamped to 1
  double expected_sum = 21;
  double expected_sos = 97;
  double expected_variance =
      (expected_sos - expected_sum * expected_sum / a.size()) / a.size();

  EXPECT_THAT(GetValue<double>(*result),
              DoubleNear(expected_variance, expected_variance / 10000));

  BoundingReport expected_report;
  SetValue<double>(expected_report.mutable_lower_bound(), 1);
  SetValue<double>(expected_report.mutable_upper_bound(), 8);
  expected_report.set_num_inputs(a.size());
  expected_report.set_num_outside(1);

  EXPECT_THAT(result->error_report().bounding_report(),
              EqualsProto(expected_report));
}

// Test not providing ApproxBounds and instead using the default.
TYPED_TEST(BoundedVarianceTest, AutomaticBoundsDefault) {
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv);
  std::vector<TypeParam> big(570, 10);
  std::vector<TypeParam> small(570, -10);
  (*bv)->AddEntries(big.begin(), big.end());
  (*bv)->AddEntries(small.begin(), small.end());

  Output expected_output;
  AddToOutput<double>(&expected_output, 100);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), -16);
  SetValue<TypeParam>(report->mutable_upper_bound(), 16);
  report->set_num_inputs(big.size() + small.size());
  report->set_num_outside(0);

  absl::StatusOr<Output> actual_output = (*bv)->PartialResult();
  ASSERT_OK(actual_output);
  EXPECT_THAT(*actual_output, EqualsProto(expected_output));
}

// Test when a bound is 0.
TYPED_TEST(BoundedVarianceTest, AutomaticBoundsZero) {
  std::vector<TypeParam> a = {0, 0, 4, 4, -2, 7};
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(0.5)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThresholdForTest(1.5)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bounds);
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv);
  (*bv)->AddEntries(a.begin(), a.end());

  // Bounds are [0, 4]. -2 gets clamped to 0. 7 gets clamped to 4.
  absl::StatusOr<Output> result = (*bv)->PartialResult();
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(result->elements(0).value()), 4);
}

TYPED_TEST(BoundedVarianceTest, Reset) {
  // Construct approximate variance.
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(kDefaultEpsilon / 2)
          .SetThresholdForTest(0.5)
          .Build();
  ASSERT_OK(bounds);
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds).value())
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(bv);

  // Reset between adding vectors.
  std::vector<TypeParam> a = {-10, -10, -10, 1000, 1000, 1000};
  std::vector<TypeParam> b = {-100, -100, -100, 100, 100, 100};
  (*bv)->AddEntries(a.begin(), a.end());
  (*bv)->Reset();
  (*bv)->AddEntries(b.begin(), b.end());

  // Check result is only affected by vector b. Bounds are [-100, 100].
  absl::StatusOr<Output> result = (*bv)->PartialResult();
  ASSERT_OK(result);
  EXPECT_DOUBLE_EQ(GetValue<double>(result->elements(0).value()), 10000);
}

TYPED_TEST(BoundedVarianceTest, MemoryUsed) {
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder().Build();
  ASSERT_OK(bv);
  EXPECT_GT((*bv)->MemoryUsed(), 0);
}

TYPED_TEST(BoundedVarianceTest, SplitsEpsilonWithAutomaticBounds) {
  double epsilon = 1.0;

  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(epsilon)
          .Build();
  ASSERT_OK(bv);
  auto* bvi =
      dynamic_cast<BoundedVarianceWithApproxBounds<TypeParam>*>(bv->get());

  EXPECT_NEAR((*bv)->GetEpsilon(), epsilon, 1e-10);
  EXPECT_NEAR((*bv)->GetEpsilon(),
              bvi->GetBoundingEpsilon() + bvi->GetAggregationEpsilon(), 1e-10);
  EXPECT_GT(bvi->GetBoundingEpsilon(), 0);
  EXPECT_LT(bvi->GetBoundingEpsilon(), epsilon);
  EXPECT_GT(bvi->GetAggregationEpsilon(), 0);
  EXPECT_LT(bvi->GetAggregationEpsilon(), epsilon);
}

TYPED_TEST(BoundedVarianceTest,
           BuilderWithApproxBoundsMoreBudgetThanTotalBudgetFails) {
  absl::StatusOr<std::unique_ptr<ApproxBounds<TypeParam>>> bounds =
      typename ApproxBounds<TypeParam>::Builder().SetEpsilon(1.1).Build();
  ASSERT_OK(bounds);
  absl::StatusOr<std::unique_ptr<BoundedVariance<TypeParam>>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1.09)
          .SetApproxBounds(std::move(bounds).value())
          .Build();
  ASSERT_THAT(bv.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Bounds Provider consumes more epsilon")));
}

TEST(BoundedVarianceWithFixedBoundsTest,
     ConsumesAllBudgetOfNumericalMechanisms) {
  std::unique_ptr<test_utils::MockLaplaceMechanism> mock_count_mechanism =
      std::make_unique<test_utils::MockLaplaceMechanism>();
  std::unique_ptr<test_utils::MockLaplaceMechanism> mock_sum_mechanism =
      std::make_unique<test_utils::MockLaplaceMechanism>();
  std::unique_ptr<test_utils::MockLaplaceMechanism>
      mock_sum_of_squares_mechanism =
          std::make_unique<test_utils::MockLaplaceMechanism>();

  test_utils::MockLaplaceMechanism* mock_count_ptr = mock_count_mechanism.get();
  test_utils::MockLaplaceMechanism* mock_sum_ptr = mock_sum_mechanism.get();
  test_utils::MockLaplaceMechanism* mock_sum_of_squares_ptr =
      mock_sum_of_squares_mechanism.get();

  // For a double bounded variance, we add int noise to the count and double
  // noise to the sum and the sum of squares.
  EXPECT_CALL(*mock_count_ptr, AddInt64Noise(_)).Times(1);
  EXPECT_CALL(*mock_sum_ptr, AddDoubleNoise(_)).Times(1);
  EXPECT_CALL(*mock_sum_of_squares_ptr, AddDoubleNoise(_)).Times(1);

  BoundedVarianceWithFixedBounds<double> bv(
      /*epsilon=*/1.0,
      /*lower=*/-1,
      /*upper=*/1, std::move(mock_count_mechanism),
      std::move(mock_sum_mechanism), std::move(mock_sum_of_squares_mechanism));

  for (int i = 0; i < 10; ++i) {
    bv.AddEntry(1.0);
  }

  EXPECT_OK(bv.PartialResult());
}

TEST(BoundedVarianceTest, ApproxBoundsMechanismHasExpectedVariance) {
  const int max_partitions_contributed = 2;
  const int max_contributions_per_partition = 3;
  const double expected_variance =
      LaplaceMechanism::Builder()
          .SetEpsilon(kDefaultEpsilon / 2.0)
          .SetL0Sensitivity(max_partitions_contributed)
          .SetLInfSensitivity(max_contributions_per_partition)
          .Build()
          .value()
          ->GetVariance();

  absl::StatusOr<std::unique_ptr<BoundedVariance<double>>> bv =
      BoundedVariance<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetMaxPartitionsContributed(max_partitions_contributed)
          .SetMaxContributionsPerPartition(max_contributions_per_partition)
          .Build();
  ASSERT_OK(bv);

  // Cast all the way down to get an ApproxBoundsAsBoundsProvider pointer.
  auto* bv_with_approx_bounds =
      static_cast<BoundedVarianceWithApproxBounds<double>*>(bv.value().get());
  ASSERT_THAT(bv_with_approx_bounds, NotNull());
  auto* approx_bounds_as_bounds_provider =
      static_cast<ApproxBoundsAsBoundsProvider<double>*>(
          bv_with_approx_bounds->GetBoundsProviderForTesting());
  ASSERT_THAT(approx_bounds_as_bounds_provider, NotNull());

  EXPECT_THAT(approx_bounds_as_bounds_provider->GetApproxBoundsForTesting()
                  ->GetMechanismForTesting()
                  ->GetVariance(),
              DoubleEq(expected_variance));
}

}  //  namespace
}  // namespace differential_privacy
