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

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/numerical-mechanisms-testing.h"

namespace differential_privacy {
namespace {

using test_utils::ZeroNoiseMechanism;
using ::testing::DoubleNear;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::Return;

constexpr double kSmallEpsilon = 0.00000001;
constexpr int64_t kNumSamples = 10000;

template <typename T>
class BoundedVarianceTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Typed test to iterate all test cases through all supported versions of
// BoundedVarianceTest, currently (int64_t, double).
typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(BoundedVarianceTest, NumericTypes);

TYPED_TEST(BoundedVarianceTest, BasicTest) {
  std::vector<TypeParam> a = {1, 2, 3, 4, 5};
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(6)
          .Build()
          .ValueOrDie();
  EXPECT_EQ(GetValue<double>(bv->Result(a.begin(), a.end()).ValueOrDie()), 2.0);
}

TYPED_TEST(BoundedVarianceTest, RepeatedResultTest) {
  std::vector<TypeParam> a = {1, 2, 3, 4, 5};
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(0)
          .SetUpper(6)
          .Build()
          .ValueOrDie();

  bv->AddEntries(a.begin(), a.end());

  EXPECT_EQ(GetValue<double>(bv->PartialResult(0.5).ValueOrDie()),
            GetValue<double>(bv->PartialResult(0.5).ValueOrDie()));
}

TYPED_TEST(BoundedVarianceTest, ClampInputTest) {
  std::vector<TypeParam> a = {0, 0, 10, 10};
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1.0)
          .SetLower(1)
          .SetUpper(3)
          .Build()
          .ValueOrDie();
  EXPECT_EQ(GetValue<double>(bv->Result(a.begin(), a.end()).ValueOrDie()), 1.0);
}

TYPED_TEST(BoundedVarianceTest, ClampOutputLowerTest) {
  std::vector<TypeParam> a = {1, 1, 1, 1};

  for (int i = 0; i < kNumSamples; ++i) {
    std::unique_ptr<BoundedVariance<TypeParam>> bv =
        typename BoundedVariance<TypeParam>::Builder()
            .SetEpsilon(kSmallEpsilon)
            .SetLower(1)
            .SetUpper(2)
            .Build()
            .ValueOrDie();
    EXPECT_GE(GetValue<double>(bv->Result(a.begin(), a.end()).ValueOrDie()),
              0.0);
  }
}

TYPED_TEST(BoundedVarianceTest, ClampOutputUpperTest) {
  std::vector<TypeParam> a = {0, 10};

  for (int i = 0; i < kNumSamples; ++i) {
    std::unique_ptr<BoundedVariance<TypeParam>> bv =
        typename BoundedVariance<TypeParam>::Builder()
            .SetEpsilon(kSmallEpsilon)
            .SetLower(0)
            .SetUpper(10)
            .Build()
            .ValueOrDie();
    EXPECT_LE(GetValue<double>(bv->Result(a.begin(), a.end()).ValueOrDie()),
              25.0);
  }
}

TYPED_TEST(BoundedVarianceTest, EmptyInputsBoundsTest) {
  std::vector<TypeParam> a;
  double lower = 0;
  double upper = 10;
  // See header comment in BoundedVariance that states "The output will also be
  // clamped between 0 and (upper - lower)^2."
  double result_lower = 0;
  double result_upper = pow(upper - lower, 2);
  int num_of_trials = 100;  // Chosen abitrarily
  double result;
  for (int i = 0; i < num_of_trials; ++i) {
    std::unique_ptr<BoundedVariance<TypeParam>> bv =
        typename BoundedVariance<TypeParam>::Builder()
            .SetLaplaceMechanism(
                absl::make_unique<ZeroNoiseMechanism::Builder>())
            .SetEpsilon(1.0)
            .SetLower(lower)
            .SetUpper(upper)
            .Build()
            .ValueOrDie();
    result = GetValue<double>(bv->Result(a.begin(), a.end()).ValueOrDie());
    EXPECT_GE(result, result_lower);
    EXPECT_LE(result, result_upper);
  }
}

TYPED_TEST(BoundedVarianceTest, GaussianInputTest) {
  // Test samples points from a Gaussian and checks that the differentially
  // private variance is the variance used to generate the distribution.
  std::unique_ptr<BoundedVariance<double>> bv =
      BoundedVariance<double>::Builder()
          .SetLower(-20)
          .SetUpper(20)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();

  // Generate the large sample of points from a seeded Laplace distribution.
  constexpr int samples = 100000;
  std::mt19937 gen;
  double stdev = 2;
  for (int i = 0; i < samples; i++) {
    bv->AddEntry(std::round(absl::Gaussian<double>(gen, 0, stdev)));
  }
  EXPECT_NEAR(GetValue<double>(bv->PartialResult().ValueOrDie()), stdev * stdev,
              0.1);
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
          CHECK_EQ(variance.status(), base::OkStatus());
          auto out = (*variance)->Result(input.begin(), input.end());
          CHECK_EQ(out.status(), base::OkStatus());
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
  typename BoundedVariance<TypeParam>::Builder bv_builder;

  // Manual bounding.
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      bv_builder.SetLower(0).SetUpper(3).Build().ValueOrDie();
  Summary summary = bv->Serialize();

  // Auto bounding.
  std::unique_ptr<BoundedVariance<TypeParam>> bv2 =
      bv_builder.ClearBounds().Build().ValueOrDie();

  // Error due to different strategies.
  EXPECT_EQ(bv2->Merge(summary).message(),
            "Merged BoundedVariance must have the same bounding strategy.");
}

TYPED_TEST(BoundedVarianceTest, SerializeMergeTest) {
  typename BoundedVariance<TypeParam>::Builder bv_builder;

  // Get summary of first BoundedVariance between entries.
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      bv_builder.SetEpsilon(.5)
          .SetLower(0)
          .SetUpper(3)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bv->AddEntry(2);
  Summary summary = bv->Serialize();
  bv->AddEntry(6);

  // Merge summary into second BoundedVariance.
  std::unique_ptr<BoundedVariance<TypeParam>> bv2 =
      bv_builder.Build().ValueOrDie();
  bv2->AddEntry(6);
  EXPECT_OK(bv2->Merge(summary));

  // Check equality.
  EXPECT_EQ(GetValue<double>(bv->PartialResult().ValueOrDie()),
            GetValue<double>(bv2->PartialResult().ValueOrDie()));
}

TYPED_TEST(BoundedVarianceTest, SerializeMergePartialValuesTest) {
  typename ApproxBounds<TypeParam>::Builder bounds_builder;
  typename BoundedVariance<TypeParam>::Builder builder;

  // Automatic bounding, so entries will be split and stored as partials.
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      bounds_builder.SetThreshold(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      builder
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetApproxBounds(std::move(bounds))
          .Build()
          .ValueOrDie();
  bv->AddEntry(-10);
  bv->AddEntry(4);
  Summary summary = bv->Serialize();
  bv->AddEntry(6);

  // Merge summary into second BoundedVariance.
  std::unique_ptr<ApproxBounds<TypeParam>> bounds2 =
      bounds_builder.Build().ValueOrDie();
  std::unique_ptr<BoundedVariance<TypeParam>> bv2 =
      builder.SetApproxBounds(std::move(bounds2)).Build().ValueOrDie();
  bv2->AddEntry(6);
  EXPECT_OK(bv2->Merge(summary));

  // Check equality. Bounds are set to [-16, 8].
  EXPECT_EQ(GetValue<double>(bv->PartialResult().ValueOrDie()),
            GetValue<double>(bv2->PartialResult().ValueOrDie()));
}

TEST(BoundedVarianceTest, SensitivityOverflow) {
  auto statusor = typename BoundedVariance<int64_t>::Builder()
                      .SetEpsilon(1.0)
                      .SetLower(std::numeric_limits<int64_t>::lowest())
                      .SetUpper(std::numeric_limits<int64_t>::max())
                      .Build();
  EXPECT_EQ(statusor.status().message(),
            "Sensitivity calculation caused integer overflow.");
}

TYPED_TEST(BoundedVarianceTest, SensitivityTooHigh) {
  // Make bounds so taking the interval squared won't overflow.
  EXPECT_EQ(typename BoundedVariance<double>::Builder()
                .SetLower(0)
                .SetUpper(std::pow(std::numeric_limits<double>::max() / 2, .5))
                .Build()
                .status()
                .message(),
            "Sensitivity is too high.");
}

TEST(BoundedVarianceTest, DropNanEntries) {
  std::vector<double> a = {1, 2, 3, 4, NAN, NAN, 5};
  std::unique_ptr<BoundedVariance<double>> bv =
      BoundedVariance<double>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1)
          .SetLower(0)
          .SetUpper(6)
          .Build()
          .ValueOrDie();
  EXPECT_EQ(GetValue<double>(bv->Result(a.begin(), a.end()).ValueOrDie()), 2.0);
}

TYPED_TEST(BoundedVarianceTest, PropagateApproxBoundsError) {
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .Build()
          .ValueOrDie();

  // Automatic bounds are needed but there is no input, so the count-threshhold
  // should exceed any bin count.
  EXPECT_FALSE(bv->PartialResult().ok());
}

TYPED_TEST(BoundedVarianceTest, AutomaticBoundsContainZero) {
  std::vector<TypeParam> a = {0, 8, -8,
                              std::numeric_limits<TypeParam>::lowest(),
                              std::numeric_limits<TypeParam>::max()};
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetNumBins(4)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bv->AddEntries(a.begin(), a.end());

  Output expected_output;
  AddToOutput<double>(&expected_output, 51.2);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), -8);
  SetValue<TypeParam>(report->mutable_upper_bound(), 8);
  report->set_num_inputs(a.size());
  report->set_num_outside(0);

  EXPECT_THAT(bv->PartialResult().ValueOrDie(), EqualsProto(expected_output));
}

TEST(BoundedVarianceTest, AutomaticBoundsNegative) {
  std::vector<double> a = {5, -2, -2, -4, -6, -6};
  std::unique_ptr<ApproxBounds<double>> bounds =
      typename ApproxBounds<double>::Builder()
          .SetEpsilon(1)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedVariance<double>> bv =
      typename BoundedVariance<double>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bv->AddEntries(a.begin(), a.end());

  Output result = bv->PartialResult().ValueOrDie();

  double expected_sum = -21;
  double expected_sos = 97;
  double expected_variance =
      (expected_sos - expected_sum * expected_sum / a.size()) / a.size();

  EXPECT_THAT(GetValue<double>(result),
              DoubleNear(expected_variance, expected_variance / 10000));

  BoundingReport expected_report;
  SetValue<double>(expected_report.mutable_lower_bound(), -8);
  SetValue<double>(expected_report.mutable_upper_bound(), -1);
  expected_report.set_num_inputs(a.size());
  expected_report.set_num_outside(1);

  EXPECT_THAT(result.error_report().bounding_report(),
              EqualsProto(expected_report));
}

TEST(BoundedVarianceTest, AutomaticBoundsPositive) {
  std::vector<double> a = {-5, 2, 2, 4, 6, 6};
  std::unique_ptr<ApproxBounds<double>> bounds =
      typename ApproxBounds<double>::Builder()
          .SetEpsilon(1)
          .SetNumBins(5)
          .SetBase(2)
          .SetScale(1)
          .SetThreshold(2)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedVariance<double>> bv =
      typename BoundedVariance<double>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bv->AddEntries(a.begin(), a.end());

  Output result = bv->PartialResult().ValueOrDie();

  double expected_sum = 21;
  double expected_sos = 97;
  double expected_variance =
      (expected_sos - expected_sum * expected_sum / a.size()) / a.size();

  EXPECT_THAT(GetValue<double>(result),
              DoubleNear(expected_variance, expected_variance / 10000));

  BoundingReport expected_report;
  SetValue<double>(expected_report.mutable_lower_bound(), 1);
  SetValue<double>(expected_report.mutable_upper_bound(), 8);
  expected_report.set_num_inputs(a.size());
  expected_report.set_num_outside(1);

  EXPECT_THAT(result.error_report().bounding_report(),
              EqualsProto(expected_report));
}

// Test not providing ApproxBounds and instead using the default.
TYPED_TEST(BoundedVarianceTest, AutomaticBoundsDefault) {
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::vector<TypeParam> big(57, 10);
  std::vector<TypeParam> small(57, -10);
  bv->AddEntries(big.begin(), big.end());
  bv->AddEntries(small.begin(), small.end());

  Output expected_output;
  AddToOutput<double>(&expected_output, 100);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), -16);
  SetValue<TypeParam>(report->mutable_upper_bound(), 16);
  report->set_num_inputs(big.size() + small.size());
  report->set_num_outside(0);

  EXPECT_THAT(bv->PartialResult().ValueOrDie(), EqualsProto(expected_output));
}

// Test when a bound is 0.
TYPED_TEST(BoundedVarianceTest, AutomaticBoundsZero) {
  std::vector<TypeParam> a = {0, 0, 4, 4, -2, 7};
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
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bv->AddEntries(a.begin(), a.end());

  // Bounds are [0, 4]. -2 gets clamped to 0. 7 gets clamped to 4.
  EXPECT_EQ(
      GetValue<double>(bv->PartialResult().ValueOrDie().elements(0).value()),
      4);
}

TYPED_TEST(BoundedVarianceTest, Reset) {
  // Construct approximate variance.
  std::unique_ptr<ApproxBounds<TypeParam>> bounds =
      typename ApproxBounds<TypeParam>::Builder()
          .SetNumBins(3)
          .SetBase(10)
          .SetScale(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetThreshold(3)
          .Build()
          .ValueOrDie();
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();

  // Reset between adding vectors.
  std::vector<TypeParam> a = {-10, -10, -10, 1000, 1000, 1000};
  std::vector<TypeParam> b = {-100, -100, -100, 100, 100, 100};
  bv->AddEntries(a.begin(), a.end());
  bv->Reset();
  bv->AddEntries(b.begin(), b.end());

  // Check result is only affected by vector b. Bounds are [-100, 100].
  auto result = bv->PartialResult();
  EXPECT_OK(result);
  EXPECT_EQ(GetValue<double>(result.ValueOrDie().elements(0).value()), 10000);
}

TYPED_TEST(BoundedVarianceTest, MemoryUsed) {
  std::unique_ptr<BoundedVariance<TypeParam>> bv =
      typename BoundedVariance<TypeParam>::Builder().Build().ValueOrDie();
  EXPECT_GT(bv->MemoryUsed(), 0);
}

}  //  namespace
}  // namespace differential_privacy
