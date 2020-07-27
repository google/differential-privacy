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

#include "differential_privacy/algorithms/bounded-standard-deviation.h"

#include "differential_privacy/base/testing/proto_matchers.h"
#include "differential_privacy/base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/random/distributions.h"
#include "differential_privacy/algorithms/approx-bounds.h"
#include "differential_privacy/algorithms/distributions.h"
#include "differential_privacy/algorithms/numerical-mechanisms-testing.h"

namespace differential_privacy {
namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::differential_privacy::base::testing::EqualsProto;

template <typename T>
class BoundedStandardDeviationTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Typed test to iterate all test cases through all supported versions of
// BoundedVarianceTest, currently (int64_t, double).
typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(BoundedStandardDeviationTest, NumericTypes);

TYPED_TEST(BoundedStandardDeviationTest, BasicTest) {
  std::vector<TypeParam> a = {1, 5, 7, 9, 13};
  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
      typename BoundedStandardDeviation<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1)
          .SetLower(0)
          .SetUpper(15)
          .Build()
          .ValueOrDie();
  EXPECT_EQ(GetValue<double>(bsd->Result(a.begin(), a.end()).ValueOrDie()),
            4.0);
}

TYPED_TEST(BoundedStandardDeviationTest, RepeatedResultTest) {
  std::vector<TypeParam> a = {1, 5, 7, 9, 13};
  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
      typename BoundedStandardDeviation<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1)
          .SetLower(0)
          .SetUpper(15)
          .Build()
          .ValueOrDie();
  bsd->AddEntries(a.begin(), a.end());

  EXPECT_EQ(GetValue<double>(bsd->PartialResult(0.5).ValueOrDie()),
            GetValue<double>(bsd->PartialResult(0.5).ValueOrDie()));
}

TYPED_TEST(BoundedStandardDeviationTest, ClampInputTest) {
  std::vector<TypeParam> a = {0, 0, 10, 10};
  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
      typename BoundedStandardDeviation<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .SetEpsilon(1)
          .SetLower(1)
          .SetUpper(3)
          .Build()
          .ValueOrDie();
  EXPECT_EQ(GetValue<double>(bsd->Result(a.begin(), a.end()).ValueOrDie()),
            1.0);
}

TYPED_TEST(BoundedStandardDeviationTest, ClampOutputMinStdDevTest) {
  // To keep these tests from depending too much on the implementation details
  // of bounded variance, we check to make sure the output is clamped within the
  // range for various dataset sizes.
  constexpr int max_dataset_size = 10;
  int64_t num_samples = 1000;
  constexpr int lower = -5;
  constexpr int upper = 5;

  for (int dataset_size = 0; dataset_size <= max_dataset_size; ++dataset_size) {
    for (int i = 0; i < num_samples; ++i) {
      std::vector<double> a(dataset_size);
      // Dataset with minimum possible stddev.
      std::fill(a.begin(), a.end(), 5);

      std::unique_ptr<BoundedStandardDeviation<double>> bsd =
          BoundedStandardDeviation<double>::Builder()
              .SetEpsilon(1)
              .SetLower(lower)
              .SetUpper(upper)
              .Build()
              .ValueOrDie();
      double result =
          GetValue<double>(bsd->Result(a.begin(), a.end()).ValueOrDie());
      EXPECT_GE(result, 0.0);
    }
  }
}

TYPED_TEST(BoundedStandardDeviationTest, ClampOutputMaxStdDevTest) {
  // To keep these tests from depending too much on the implementation details
  // of bounded variance, we check to make sure the output is clamped within the
  // range for various dataset sizes.
  constexpr int max_dataset_size = 10;
  int64_t num_samples = 1000;
  constexpr int lower = -5;
  constexpr int upper = 5;
  for (int dataset_size = 2; dataset_size <= max_dataset_size;
       dataset_size += 2) {
    for (int i = 0; i < num_samples; ++i) {
      std::vector<TypeParam> a(dataset_size);
      // Dataset with maximum possible stddev.
      std::fill(a.begin(), a.begin() + dataset_size / 2, -5);
      std::fill(a.begin() + dataset_size / 2 + 1, a.end(), 5);

      std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
          typename BoundedStandardDeviation<TypeParam>::Builder()
              .SetEpsilon(1)
              .SetLower(lower)
              .SetUpper(upper)
              .Build()
              .ValueOrDie();
      TypeParam result =
          GetValue<TypeParam>(bsd->Result(a.begin(), a.end()).ValueOrDie());
      EXPECT_LE(result, upper - lower);
    }
  }
}

TYPED_TEST(BoundedStandardDeviationTest, RandGaussianTest) {
  // Test samples points from a Gaussian and checks that the differentially
  // private standard deviation is "close enough" to the standard deviation used
  // to generate the distribution.
  constexpr int num_trials = 10;
  constexpr double epsilon = 1.0;
  constexpr int samples = 50000;
  constexpr int range = 30;
  constexpr double mean = 10.0;
  constexpr double stddev = 3.0;
  // Upper bound on the error such that dp stddev is within actual stddev 99.99%
  // of the time. Larger than the theoretical error because square roots don't
  // play nice with additions, but a good enough estimation for the test.
  const double error =
      range * (std::sqrt(-log(0.0001) / epsilon / samples + 1) + 1);

  for (int i = 0; i < num_trials; i++) {
    std::mt19937 rand_gen;
    std::unique_ptr<BoundedStandardDeviation<double>> bsd =
        BoundedStandardDeviation<double>::Builder()
            .SetEpsilon(epsilon)
            .SetLower(mean - range / 2)
            .SetUpper(mean + range / 2)
            .Build()
            .ValueOrDie();
    for (int i = 0; i < samples; i++) {
      bsd->AddEntry(mean + absl::Gaussian<double>(rand_gen));
    }
    EXPECT_NEAR(GetValue<double>(bsd->PartialResult().ValueOrDie()), stddev,
                error);
  }
}

TYPED_TEST(BoundedStandardDeviationTest, SerializeAndMergeTest) {
  typename BoundedStandardDeviation<TypeParam>::Builder builder;

  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
      builder.SetEpsilon(.5)
          .SetLower(0)
          .SetUpper(3)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bsd->AddEntry(1);
  Summary summary = bsd->Serialize();
  bsd->AddEntry(2);

  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd2 =
      builder.Build().ValueOrDie();
  EXPECT_OK(bsd2->Merge(summary));
  bsd2->AddEntry(2);

  EXPECT_EQ(GetValue<double>(bsd->PartialResult().ValueOrDie()),
            GetValue<double>(bsd2->PartialResult().ValueOrDie()));
}

TYPED_TEST(BoundedStandardDeviationTest, TwoAlgorithmsOneBuilder) {
  std::vector<TypeParam> a = {-2, 2};
  typename BoundedStandardDeviation<TypeParam>::Builder builder;
  builder.SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>());

  // First algorithm doesn't clamp anything in a.
  auto alg1 = builder.SetLower(-5).SetUpper(5).Build().ValueOrDie();
  EXPECT_EQ(GetValue<double>(alg1->Result(a.begin(), a.end()).ValueOrDie()), 2);

  // Second algorithm clamps to [-1, 1].
  auto alg2 = builder.SetLower(-1).SetUpper(1).Build().ValueOrDie();
  EXPECT_EQ(GetValue<double>(alg2->Result(a.begin(), a.end()).ValueOrDie()), 1);
}

TYPED_TEST(BoundedStandardDeviationTest, AutomaticBounds) {
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
  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
      typename BoundedStandardDeviation<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetApproxBounds(std::move(bounds))
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  bsd->AddEntries(a.begin(), a.end());

  Output expected_output;
  AddToOutput<double>(&expected_output, 2);
  BoundingReport* report =
      expected_output.mutable_error_report()->mutable_bounding_report();
  SetValue<TypeParam>(report->mutable_lower_bound(), 0);
  SetValue<TypeParam>(report->mutable_upper_bound(), 4);
  report->set_num_inputs(a.size());
  report->set_num_outside(2);

  EXPECT_THAT(bsd->PartialResult().ValueOrDie(), EqualsProto(expected_output));
}

// Test not providing ApproxBounds and instead using the default.
TYPED_TEST(BoundedStandardDeviationTest, AutomaticBoundsDefault) {
  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
      typename BoundedStandardDeviation<TypeParam>::Builder()
          .SetEpsilon(1)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  std::vector<TypeParam> big(100, 10);
  std::vector<TypeParam> small(100, -10);
  bsd->AddEntries(big.begin(), big.end());
  bsd->AddEntries(small.begin(), small.end());

  EXPECT_NEAR(
      GetValue<double>(bsd->PartialResult().ValueOrDie().elements(0).value()),
      10, std::pow(10, -10));
}

TYPED_TEST(BoundedStandardDeviationTest, PropagateApproxBoundsError) {
  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
      typename BoundedStandardDeviation<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();

  // Automatic bounds are needed but there is no input, so the count-threshhold
  // should exceed any bin count.
  EXPECT_FALSE(bsd->PartialResult().ok());
}

TYPED_TEST(BoundedStandardDeviationTest, MemoryUsed) {
  std::unique_ptr<BoundedStandardDeviation<TypeParam>> bsd =
      typename BoundedStandardDeviation<TypeParam>::Builder()
          .Build()
          .ValueOrDie();
  EXPECT_GT(bsd->MemoryUsed(), 0);
}

}  // namespace
}  // namespace differential_privacy
