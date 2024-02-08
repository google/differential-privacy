//
// Copyright 2021 Google LLC
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

#include "algorithms/quantiles.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms-testing.h"

namespace differential_privacy {
namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::HasSubstr;
using ::differential_privacy::base::testing::StatusIs;

const int kDefaultDatasetSize = 1001;
const int kNumRanksToTest = 10;

template <typename T>
class QuantilesTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(QuantilesTest, NumericTypes);

TEST(QuantilesTest, InvalidParametersTest) {
  EXPECT_THAT(Quantiles<double>::Builder()
                  .SetLower(0)
                  .SetUpper(1)
                  .SetQuantiles({})
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must specify at least one quantile")));
  EXPECT_THAT(Quantiles<double>::Builder()
                  .SetLower(0)
                  .SetUpper(1)
                  .SetQuantiles({-1})
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("quantiles to calculate must be in [0, 1]")));
  EXPECT_THAT(Quantiles<double>::Builder()
                  .SetLower(0)
                  .SetUpper(1)
                  .SetQuantiles({1.5})
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("quantiles to calculate must be in [0, 1]")));
  EXPECT_THAT(
      Quantiles<double>::Builder()
          .SetLower(2)
          .SetUpper(1)
          .SetQuantiles({0.5})
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Lower bound cannot be greater than upper bound")));
  EXPECT_THAT(
      Quantiles<double>::Builder()
          .SetLower(1)
          .SetUpper(1)
          .SetQuantiles({0.5})
          .Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Lower bound cannot be equal to upper bound")));
  EXPECT_THAT(Quantiles<double>::Builder().SetQuantiles({0.5}).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lower and upper bounds must both be set")));
}

// Input must be sorted.
template <typename T>
double TrueQuantileFromSorted(std::vector<T> inputs, double quantile) {
  int rank = std::round(quantile * (inputs.size() - 1));
  return inputs[rank];
}

TYPED_TEST(QuantilesTest, ApproximatesTrueQuantile) {
  std::vector<double> quantiles;
  for (int i = 0; i < kNumRanksToTest; ++i) {
    quantiles.push_back(static_cast<double>(i) / kNumRanksToTest);
  }

  std::unique_ptr<Quantiles<TypeParam>> test_quantiles =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();

  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(absl::BitGen(), -25, 25));
  }

  for (TypeParam input : inputs) {
    test_quantiles->AddEntry(input);
  }

  double tolerance = 0.01;  // > upper - lower / branchingFactor ^ treeHeight
  std::sort(inputs.begin(), inputs.end());

  Output results = test_quantiles->PartialResult().value();

  for (int i = 0; i < quantiles.size(); ++i) {
    double quantile = quantiles[i];

    EXPECT_NEAR(results.elements(i).value().float_value(),
                TrueQuantileFromSorted(inputs, quantile), tolerance);
  }
}

TYPED_TEST(QuantilesTest, InputOrderInvariant) {
  std::vector<double> quantiles;
  for (int i = 0; i < kNumRanksToTest; ++i) {
    quantiles.push_back(static_cast<double>(i) / kNumRanksToTest);
  }

  std::unique_ptr<Quantiles<TypeParam>> test_quantiles1 =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();
  std::unique_ptr<Quantiles<TypeParam>> test_quantiles2 =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();

  absl::BitGen gen;
  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(gen, -25, 25));
  }

  for (TypeParam input : inputs) {
    test_quantiles1->AddEntry(input);
  }
  std::shuffle(inputs.begin(), inputs.end(), gen);
  for (TypeParam input : inputs) {
    test_quantiles2->AddEntry(input);
  }

  Output results1 = test_quantiles1->PartialResult().value();
  Output results2 = test_quantiles2->PartialResult().value();

  EXPECT_THAT(results1, ::differential_privacy::base::testing::EqualsProto(results2));
}

// This should hold even with noise enabled.
TYPED_TEST(QuantilesTest, ResultsIncreaseMonotonically) {
  std::vector<double> quantiles;
  for (int i = 0; i < kNumRanksToTest; ++i) {
    quantiles.push_back(static_cast<double>(i) / kNumRanksToTest);
  }

  std::unique_ptr<Quantiles<TypeParam>> test_quantiles =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();

  absl::BitGen gen;
  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(gen, -25, 25));
  }

  for (TypeParam input : inputs) {
    test_quantiles->AddEntry(input);
  }

  Output result = test_quantiles->PartialResult().value();

  double last_result = std::numeric_limits<double>::lowest();
  for (int i = 0; i < result.elements_size(); ++i) {
    EXPECT_GE(result.elements(i).value().float_value(), last_result);
    last_result = result.elements(i).value().float_value();
  }
}

TYPED_TEST(QuantilesTest, SerializeMergeTest) {
  std::vector<double> quantiles;
  for (int i = 0; i < kNumRanksToTest; ++i) {
    quantiles.push_back(static_cast<double>(i) / kNumRanksToTest);
  }

  std::unique_ptr<Quantiles<TypeParam>> test_quantiles1 =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();
  std::unique_ptr<Quantiles<TypeParam>> test_quantiles2 =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();

  absl::BitGen gen;
  std::vector<TypeParam> first_inputs;
  std::vector<TypeParam> second_inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    TypeParam input = absl::Uniform(gen, -25, 25);
    if (i < kDefaultDatasetSize / 2) {
      first_inputs.push_back(input);
    } else {
      second_inputs.push_back(input);
    }
  }

  for (TypeParam input : first_inputs) {
    test_quantiles1->AddEntry(input);
  }

  EXPECT_OK(test_quantiles2->Merge(test_quantiles1->Serialize()));

  for (TypeParam input : second_inputs) {
    test_quantiles1->AddEntry(input);
    test_quantiles2->AddEntry(input);
  }

  Output results1 = test_quantiles1->PartialResult().value();

  Output results2 = test_quantiles2->PartialResult().value();

  EXPECT_THAT(results1, ::differential_privacy::base::testing::EqualsProto(results2));
}

TEST(QuantilesTest, MergeFailsWithBadBounds) {
  std::unique_ptr<Quantiles<double>> test_quantiles =
      typename Quantiles<double>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles({0.5})
          .Build()
          .value();
  std::unique_ptr<Quantiles<double>> wrong_lower =
      typename Quantiles<double>::Builder()
          .SetUpper(50)
          .SetLower(-49)
          .SetQuantiles({0.5})
          .Build()
          .value();
  std::unique_ptr<Quantiles<double>> wrong_upper =
      typename Quantiles<double>::Builder()
          .SetUpper(49)
          .SetLower(-50)
          .SetQuantiles({0.5})
          .Build()
          .value();

  EXPECT_THAT(wrong_lower->Merge(test_quantiles->Serialize()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Bounds")));
  EXPECT_THAT(wrong_upper->Merge(test_quantiles->Serialize()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Bounds")));
}

TYPED_TEST(QuantilesTest, MergeInDifferentOrderReturnsSameProto) {
  std::vector<double> quantiles = {0.3, 0.5, 0.9};
  absl::StatusOr<std::unique_ptr<Quantiles<TypeParam>>> q1 =
      typename Quantiles<TypeParam>::Builder()
          .SetEpsilon(1.1)
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .Build();
  ASSERT_OK(q1);
  for (int i = -10; i < 0; ++i) {
    q1.value()->AddEntry(i);
  }
  Summary summary1 = q1.value()->Serialize();

  absl::StatusOr<std::unique_ptr<Quantiles<TypeParam>>> q2 =
      typename Quantiles<TypeParam>::Builder()
          .SetEpsilon(1.1)
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .Build();
  ASSERT_OK(q2);
  for (int i = 0; i < 10; ++i) {
    q2.value()->AddEntry(i);
  }
  Summary summary2 = q2.value()->Serialize();

  absl::StatusOr<std::unique_ptr<Quantiles<TypeParam>>> merger1 =
      typename Quantiles<TypeParam>::Builder()
          .SetEpsilon(1.1)
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .Build();
  ASSERT_OK(merger1);
  ASSERT_OK(merger1.value()->Merge(summary1));
  ASSERT_OK(merger1.value()->Merge(summary2));

  absl::StatusOr<std::unique_ptr<Quantiles<TypeParam>>> merger2 =
      typename Quantiles<TypeParam>::Builder()
          .SetEpsilon(1.1)
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .Build();
  ASSERT_OK(merger2);
  // Different order here as above for merger1.
  ASSERT_OK(merger2.value()->Merge(summary2));
  ASSERT_OK(merger2.value()->Merge(summary1));

  EXPECT_THAT(merger1.value()->Serialize(),
              EqualsProto(merger2.value()->Serialize()));
}

TEST(QuantilesTest, IgnoresNaN) {
  std::unique_ptr<Quantiles<double>> test_quantiles1 =
      typename Quantiles<double>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles({0.5})
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();

  std::unique_ptr<Quantiles<double>> test_quantiles2 =
      typename Quantiles<double>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles({0.5})
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();

  test_quantiles1->AddEntry(5.0);
  test_quantiles2->AddEntry(5.0);

  for (int i = 0; i < 100; ++i) {
    test_quantiles2->AddEntry(std::nan(""));
  }

  Output results1 = test_quantiles1->PartialResult().value();
  Output results2 = test_quantiles2->PartialResult().value();

  EXPECT_THAT(results1, ::differential_privacy::base::testing::EqualsProto(results2));
}

TYPED_TEST(QuantilesTest, MemoryUsed) {
  std::unique_ptr<Quantiles<TypeParam>> empty =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles({0.5})
          .Build()
          .value();
  std::unique_ptr<Quantiles<TypeParam>> once =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles({0.5})
          .Build()
          .value();
  std::unique_ptr<Quantiles<TypeParam>> twice =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles({0.5})
          .Build()
          .value();

  once->AddEntry(-49);
  twice->AddEntry(49);
  twice->AddEntry(49);
  EXPECT_GT(once->MemoryUsed(), empty->MemoryUsed());
  EXPECT_EQ(once->MemoryUsed(), twice->MemoryUsed());
}

TYPED_TEST(QuantilesTest, Reset) {
  std::vector<double> quantiles;
  for (int i = 0; i < kNumRanksToTest; ++i) {
    quantiles.push_back(static_cast<double>(i) / kNumRanksToTest);
  }

  std::unique_ptr<Quantiles<TypeParam>> test_quantiles1 =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();
  std::unique_ptr<Quantiles<TypeParam>> test_quantiles2 =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .SetLaplaceMechanism(std::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .value();

  std::vector<TypeParam> first_inputs;
  std::vector<TypeParam> second_inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    TypeParam input = absl::Uniform(absl::BitGen(), -25, 25);
    if (i < kDefaultDatasetSize / 2) {
      first_inputs.push_back(input);
    } else {
      second_inputs.push_back(input);
    }
  }

  for (TypeParam input : first_inputs) {
    test_quantiles1->AddEntry(input);
  }

  test_quantiles1->Reset();

  for (TypeParam input : second_inputs) {
    test_quantiles1->AddEntry(input);
    test_quantiles2->AddEntry(input);
  }

  Output results1 = test_quantiles1->PartialResult().value();

  Output results2 = test_quantiles2->PartialResult().value();

  EXPECT_THAT(results1, ::differential_privacy::base::testing::EqualsProto(results2));
}

TYPED_TEST(QuantilesTest, GetQuantiles) {
  std::vector<double> quantiles = {0.3, 0.5, 0.9};
  std::unique_ptr<Quantiles<TypeParam>> test_quantiles =
      typename Quantiles<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetQuantiles(quantiles)
          .Build()
          .value();

  EXPECT_THAT(test_quantiles->GetQuantiles(),
              testing::UnorderedElementsAre(0.3, 0.5, 0.9));
}

}  // namespace
}  // namespace differential_privacy
