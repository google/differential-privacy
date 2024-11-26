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

#include "algorithms/quantile-tree.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/confidence-interval.pb.h"

namespace differential_privacy {

// Provides limited-scope static methods for interacting with a QuantileTree
// object for testing purposes.
class QuantileTreeTestPeer {
 public:
  template <typename T>
  static void AddMultipleEntries(const T& t, int64_t num_of_entries,
                                 QuantileTree<T>* qt) {
    qt->AddMultipleEntries(t, num_of_entries);
  }
};

namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::testing::HasSubstr;
using ::differential_privacy::base::testing::StatusIs;

const double kTestDefaultEpsilon = 0.5;
const double kDefaultDelta = 1e-5;
const int kDefaultMaxContributionsPerPartition = 5;
const int kDefaultMaxPartitionsContributed = 12;
const int kDefaultDatasetSize = 1001;
const int kNumRanksToTest = 10;
const std::vector<double> kQuantilesToTest{0.0,  0.005, 0.01, 0.05,  0.25, 0.5,
                                           0.75, 0.95,  0.99, 0.995, 1.0};
const std::vector<double> kConfidenceLevelsToTest{
    0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999};

template <typename T>
class QuantileTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  void StatisticallyAssertConfidenceLevel(
      const std::vector<T> entries,
      typename QuantileTree<T>::Builder tree_builder,
      std::unique_ptr<NumericalMechanismBuilder> mech_builder) {
    // Prepare a hit counter that counts the number of times a confidence
    // interval contains the raw value keyed by quantile and confidence level.
    std::unordered_map<double, std::unordered_map<double, int>> hit_counter;

    std::unique_ptr<QuantileTree<T>> quantile_tree =
        tree_builder.Build().value();
    for (const T& entry : entries) {
      quantile_tree->AddEntry(entry);
    }

    typename QuantileTree<T>::DPParams dp_params;
    dp_params.epsilon = kTestDefaultEpsilon;
    dp_params.delta = kDefaultDelta;
    dp_params.max_contributions_per_partition =
        kDefaultMaxContributionsPerPartition;
    dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
    dp_params.mechanism_builder =
        std::make_unique<ZeroNoiseMechanism::Builder>();

    typename QuantileTree<T>::Privatized zero_noise_results =
        quantile_tree->MakePrivate(dp_params).value();

    // Approximate the raw quantile values by querying a zero noise instance of
    // the quantiles mechanism.
    std::unordered_map<double, double> zero_noise_quantiles;
    for (double quantile : kQuantilesToTest) {
      zero_noise_quantiles[quantile] =
          zero_noise_results.GetQuantile(quantile).value();
    }

    dp_params.mechanism_builder = std::move(mech_builder);

    // Sample the hit frequencies.
    for (int i = 0; i < kNumberOfSamples_; i++) {
      typename QuantileTree<T>::Privatized noised_results =
          quantile_tree->MakePrivate(dp_params).value();

      // Check whether the confidence intervals contain the respective raw value
      // for all quantiles and confidence levels.
      for (double quantile : kQuantilesToTest) {
        for (double level : kConfidenceLevelsToTest) {
          absl::StatusOr<ConfidenceInterval> noised_ci_or_status =
              noised_results.ComputeNoiseConfidenceInterval(quantile, level);
          EXPECT_OK(noised_ci_or_status);
          ConfidenceInterval noised_ci = noised_ci_or_status.value();
          if (noised_ci.lower_bound() <= zero_noise_quantiles[quantile] &&
              zero_noise_quantiles[quantile] <= noised_ci.upper_bound()) {
            ++hit_counter[quantile][level];
          }
        }
      }
    }

    // Expect that the hit frequency was sufficiently large for all quantiles
    // and confidence levels.
    for (double quantile : kQuantilesToTest) {
      for (double level : kConfidenceLevelsToTest) {
        EXPECT_GE(hit_counter[quantile][level], kAcceptableThresholds_[level]);
      }
    }
  }

  const int kNumberOfSamples_ = 2500;
  // Minimum number of times the raw value needs to be contained in the
  // confidence interval for a given alpha so that the statistical test accepts.
  // The failure probability is less than 10^-6.
  std::unordered_map<double, int> kAcceptableThresholds_ = {
      {0.9999, 2494}, {0.999, 2486}, {0.99, 2447}, {0.95, 2319},
      {0.9, 2175},    {0.75, 1769},  {0.5, 1130},  {0.25, 523},
      {0.1, 181},     {0.05, 76},    {0.01, 4}};
};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(QuantileTreeTest, NumericTypes);

TEST(QuantileTreeTest, InvalidParametersTest) {
  EXPECT_THAT(QuantileTree<double>::Builder()
                  .SetTreeHeight(0)
                  .SetLower(0)
                  .SetUpper(1)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Tree height must be at least 1")));
  EXPECT_THAT(QuantileTree<double>::Builder()
                  .SetBranchingFactor(1)
                  .SetLower(0)
                  .SetUpper(1)
                  .Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Branching factor must be at least 2")));
  EXPECT_THAT(QuantileTree<double>::Builder().SetLower(2).SetUpper(1).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lower bound must be less than upper")));
  EXPECT_THAT(QuantileTree<double>::Builder().Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lower and upper bounds")));
}

// Input must be sorted.
template <typename T>
double TrueQuantileFromSorted(std::vector<T> inputs, double quantile) {
  int rank = std::round(quantile * (inputs.size() - 1));
  return inputs[rank];
}

TYPED_TEST(QuantileTreeTest, ApproximatesTrueQuantile) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(absl::BitGen(), -25, 25));
  }

  for (TypeParam input : inputs) {
    test_quantiles->AddEntry(input);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  double tolerance = 0.01;  // > upper - lower / branchingFactor ^ treeHeight
  std::sort(inputs.begin(), inputs.end());

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_NEAR(results.GetQuantile(quantile).value(),
                TrueQuantileFromSorted(inputs, quantile), tolerance);
  }
}

TYPED_TEST(QuantileTreeTest, EmptyLinearlyDistributed) {
  double lower = -50;
  double upper = 50;
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(upper)
          .SetLower(lower)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  double tolerance = 0.01;  // < (upper - lower) / branchingFactor ^ treeHeight

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;
    // Avoid extreme quantiles, as they're special-cased.
    if (quantile < 0.1 || quantile > 0.9) continue;
    double expected = quantile * (upper - lower) + lower;
    EXPECT_NEAR(results.GetQuantile(quantile).value(), expected, tolerance);
  }
}

TYPED_TEST(QuantileTreeTest, LowerBoundClamps) {
  double lower = -50;
  double upper = 50;
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(upper)
          .SetLower(lower)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    test_quantiles->AddEntry(-100);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_GE(results.GetQuantile(quantile).value(), lower);
  }
}

TYPED_TEST(QuantileTreeTest, UpperBoundClamps) {
  double lower = -50;
  double upper = 50;
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(upper)
          .SetLower(lower)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    test_quantiles->AddEntry(100);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_LE(results.GetQuantile(quantile).value(), upper);
  }
}

TYPED_TEST(QuantileTreeTest, ApproximatesTrueQuantileNearUpperBound) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    test_quantiles->AddEntry(50);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();
  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  double tolerance = 0.01;  // < (upper - lower) / branchingFactor ^ treeHeight
  EXPECT_NEAR(results.GetQuantile(0.5).value(), 50, tolerance);
}

TYPED_TEST(QuantileTreeTest, ApproximatesTrueQuantileNearLowerBound) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    test_quantiles->AddEntry(-50);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();
  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  double tolerance = 0.01;  // < (upper - lower) / branchingFactor ^ treeHeight
  EXPECT_NEAR(results.GetQuantile(0.5).value(), -50, tolerance);
}

TYPED_TEST(QuantileTreeTest, InputOrderInvariant) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles1 =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles2 =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
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

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results1 =
      test_quantiles1->MakePrivate(dp_params).value();

  typename QuantileTree<TypeParam>::Privatized results2 =
      test_quantiles2->MakePrivate(dp_params).value();

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_EQ(results1.GetQuantile(quantile).value(),
              results2.GetQuantile(quantile).value());
  }
}

// This should hold even with noise enabled.
TYPED_TEST(QuantileTreeTest, RepeatedResultsIdentical) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
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

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_EQ(results.GetQuantile(quantile).value(),
              results.GetQuantile(quantile).value());
  }
}

// This should hold even with noise enabled.
TYPED_TEST(QuantileTreeTest, ResultsIncreaseMonotonically) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
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

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  double last_result = std::numeric_limits<double>::lowest();
  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_GE(results.GetQuantile(quantile).value(), last_result);
    last_result = results.GetQuantile(quantile).value();
  }
}

TYPED_TEST(QuantileTreeTest, InvalidRanksReturnErrors) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  EXPECT_THAT(results.GetQuantile(-0.5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("quantile must be in [0, 1]")));
  EXPECT_THAT(results.GetQuantile(1.5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("quantile must be in [0, 1]")));
}

TYPED_TEST(QuantileTreeTest, SerializeMergeTest) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles1 =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles2 =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
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

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results1 =
      test_quantiles1->MakePrivate(dp_params).value();

  typename QuantileTree<TypeParam>::Privatized results2 =
      test_quantiles2->MakePrivate(dp_params).value();

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_EQ(results1.GetQuantile(quantile).value(),
              results2.GetQuantile(quantile).value());
  }
}

TEST(QuantileTreeTest, MergeFailsWithBadBounds) {
  std::unique_ptr<QuantileTree<double>> test_quantiles =
      typename QuantileTree<double>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();
  std::unique_ptr<QuantileTree<double>> wrong_lower =
      typename QuantileTree<double>::Builder()
          .SetUpper(50)
          .SetLower(-49)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();
  std::unique_ptr<QuantileTree<double>> wrong_upper =
      typename QuantileTree<double>::Builder()
          .SetUpper(49)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  EXPECT_THAT(wrong_lower->Merge(test_quantiles->Serialize()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Bounds")));
  EXPECT_THAT(wrong_upper->Merge(test_quantiles->Serialize()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Bounds")));
}

TYPED_TEST(QuantileTreeTest, Reset) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles1 =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles2 =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
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

  test_quantiles1->Reset();

  for (TypeParam input : second_inputs) {
    test_quantiles1->AddEntry(input);
    test_quantiles2->AddEntry(input);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results1 =
      test_quantiles1->MakePrivate(dp_params).value();

  typename QuantileTree<TypeParam>::Privatized results2 =
      test_quantiles2->MakePrivate(dp_params).value();

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_EQ(results1.GetQuantile(quantile).value(),
              results2.GetQuantile(quantile).value());
  }
}

TEST(QuantileTreeTest, IgnoresNaN) {
  std::unique_ptr<QuantileTree<double>> test_quantiles =
      typename QuantileTree<double>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  typename QuantileTree<double>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  test_quantiles->AddEntry(5.0);
  typename QuantileTree<double>::Privatized results1 =
      test_quantiles->MakePrivate(dp_params).value();

  for (int i = 0; i < 100; ++i) {
    test_quantiles->AddEntry(std::nan(""));
  }

  typename QuantileTree<double>::Privatized results2 =
      test_quantiles->MakePrivate(dp_params).value();

  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;

    EXPECT_EQ(results1.GetQuantile(quantile).value(),
              results2.GetQuantile(quantile).value());
  }
}

TEST(QuantileTreeTest, TreeOverflowsWithInputs) {
  std::unique_ptr<QuantileTree<int64_t>> test_quantiles =
      typename QuantileTree<int64_t>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  QuantileTreeTestPeer::AddMultipleEntries<int64_t>(
      25, std::numeric_limits<int64_t>::max(), test_quantiles.get());
  test_quantiles->AddEntry(25);

  typename QuantileTree<int64_t>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<int64_t>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  // With no noise and no overflow, we should always get 25. With overflow,
  // those nodes should register as empty, meaning the whole tree will be empty,
  // meaning the median should be 0 (middle of the range).
  EXPECT_EQ(results.GetQuantile(0.5).value(), 0);
}

TEST(QuantileTreeTest, TreeOverflowsWithNoise) {
  std::unique_ptr<QuantileTree<int64_t>> test_quantiles =
      typename QuantileTree<int64_t>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  QuantileTreeTestPeer::AddMultipleEntries<int64_t>(
      25, std::numeric_limits<int64_t>::max(), test_quantiles.get());

  typename QuantileTree<int64_t>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

  // All the entries are in one leaf node. If the counts don't overflow, they
  // should always be much larger than all the noisy zeroes, and so we should
  // ~always get a median in that bucket. If the count can overflow, it should
  // do so ~50% of the time. If the count overflows, the total count should
  // be negative, and we should conclude the tree is empty. Therefore, we will
  // pick the middle of the range (0).
  //
  // That means that if overflows can happen, we should get a 0 ~50% of the
  // time. Given 10^3 tries, an event with p=.5 should ~always occur.
  for (int i = 0; i < 1e3; ++i) {
    typename QuantileTree<int64_t>::Privatized results =
        test_quantiles->MakePrivate(dp_params).value();

    if (results.GetQuantile(0.5).value() == 0) {
      // An overflow occurred, so we can return from the test with a success.
      return;
    }
  }
  FAIL() << "No overflow occurred after 1e3 iterations.";
}

TYPED_TEST(QuantileTreeTest, PrivatizedConstantWithExtraInput) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  for (int i = 0; i < 100; ++i) {
    test_quantiles->AddEntry(-25);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  double tolerance = 0.01;  // > upper - lower / branchingFactor ^ treeHeight
  EXPECT_NEAR(results.GetQuantile(0.5).value(), -25, tolerance);

  for (int i = 0; i < 1000; ++i) {
    test_quantiles->AddEntry(25);
  }

  EXPECT_NEAR(results.GetQuantile(0.5).value(), -25, tolerance);
}

TYPED_TEST(QuantileTreeTest, ZeroNoiseConfidenceIntervalsMatchQuantile) {
  std::unique_ptr<QuantileTree<TypeParam>> test_quantiles =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(absl::BitGen(), -25, 25));
  }

  for (const TypeParam& input : inputs) {
    test_quantiles->AddEntry(input);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  double tolerance = 0.01;  // > upper - lower / branchingFactor ^ treeHeight
  std::sort(inputs.begin(), inputs.end());

  double confidence_level = 0.95;
  for (int i = 0; i < kNumRanksToTest; ++i) {
    double quantile = static_cast<double>(i) / kNumRanksToTest;
    double quantile_result = results.GetQuantile(quantile).value();
    absl::StatusOr<ConfidenceInterval> ci_or_status =
        results.ComputeNoiseConfidenceInterval(quantile, confidence_level);

    EXPECT_OK(ci_or_status);
    EXPECT_NEAR(quantile_result, ci_or_status.value().lower_bound(), tolerance);
    EXPECT_NEAR(quantile_result, ci_or_status.value().upper_bound(), tolerance);
    EXPECT_NEAR(confidence_level, ci_or_status.value().confidence_level(),
                tolerance);
  }
}

TYPED_TEST(QuantileTreeTest,
           ConfidenceIntervalsLowerBoundLessThanOrEqualsUpperBoundWithNoInput) {
  for (int i = 0; i < 1000; i++) {
    std::unique_ptr<QuantileTree<TypeParam>> quantile_tree =
        typename QuantileTree<TypeParam>::Builder()
            .SetUpper(50)
            .SetLower(-50)
            .SetTreeHeight(4)
            .SetBranchingFactor(10)
            .Build()
            .value();

    typename QuantileTree<TypeParam>::DPParams dp_params;
    dp_params.epsilon = kTestDefaultEpsilon;
    dp_params.delta = kDefaultDelta;
    dp_params.max_contributions_per_partition =
        kDefaultMaxContributionsPerPartition;
    dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
    dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

    typename QuantileTree<TypeParam>::Privatized results =
        quantile_tree->MakePrivate(dp_params).value();

    // Use a small confidence level to increase the chance of a violation.
    double confidence_level = 0.01;
    for (double quantile : kQuantilesToTest) {
      absl::StatusOr<ConfidenceInterval> ci_or_status =
          results.ComputeNoiseConfidenceInterval(quantile, confidence_level);
      EXPECT_OK(ci_or_status);
      EXPECT_LE(ci_or_status.value().lower_bound(),
                ci_or_status.value().upper_bound());
    }
  }
}

TYPED_TEST(QuantileTreeTest,
           ConfidenceIntervalsLowerBoundLessThanOrEqualsUpperBoundWithInput) {
  for (int i = 0; i < 1000; i++) {
    std::unique_ptr<QuantileTree<TypeParam>> quantile_tree =
        typename QuantileTree<TypeParam>::Builder()
            .SetUpper(50)
            .SetLower(-50)
            .SetTreeHeight(4)
            .SetBranchingFactor(10)
            .Build()
            .value();

    std::vector<TypeParam> inputs;
    for (int i = 0; i < kDefaultDatasetSize; ++i) {
      inputs.push_back(absl::Uniform(absl::BitGen(), -25, 25));
    }

    for (TypeParam input : inputs) {
      quantile_tree->AddEntry(input);
    }

    typename QuantileTree<TypeParam>::DPParams dp_params;
    dp_params.epsilon = kTestDefaultEpsilon;
    dp_params.delta = kDefaultDelta;
    dp_params.max_contributions_per_partition =
        kDefaultMaxContributionsPerPartition;
    dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
    dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

    typename QuantileTree<TypeParam>::Privatized results =
        quantile_tree->MakePrivate(dp_params).value();

    // Use a small confidence level to increase the chance of a violation.
    double confidence_level = 0.01;
    for (double quantile : kQuantilesToTest) {
      absl::StatusOr<ConfidenceInterval> ci_or_status =
          results.ComputeNoiseConfidenceInterval(quantile, confidence_level);
      EXPECT_OK(ci_or_status);
      EXPECT_LE(ci_or_status.value().lower_bound(),
                ci_or_status.value().upper_bound());
    }
  }
}

TYPED_TEST(QuantileTreeTest, ConfidenceIntervalWithinBounds) {
  const TypeParam lower_bound = 1;
  const TypeParam upper_bound = 2;

  std::unique_ptr<QuantileTree<TypeParam>> quantile_tree =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(upper_bound)
          .SetLower(lower_bound)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  for (int i = 0; i < 10; ++i) {
    quantile_tree->AddEntry(lower_bound);
    quantile_tree->AddEntry(upper_bound);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      quantile_tree->MakePrivate(dp_params).value();

  // To increase the chance of a violation, we use a high confidence level and
  // test the confidence intervals of the min and max quantiles, which match the
  // bounds of the input range.
  absl::StatusOr<ConfidenceInterval> ci_or_status =
      results.ComputeNoiseConfidenceInterval(0.0, 0.99);
  EXPECT_OK(ci_or_status);
  EXPECT_GE(ci_or_status.value().lower_bound(), lower_bound);

  ci_or_status = results.ComputeNoiseConfidenceInterval(1.0, 0.99);
  EXPECT_OK(ci_or_status);
  EXPECT_LE(ci_or_status.value().upper_bound(), upper_bound);
}

TYPED_TEST(QuantileTreeTest,
           ConfidenceIntervalWithGaussianNoiseReturnsSameResults) {
  std::unique_ptr<QuantileTree<TypeParam>> quantile_tree =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(absl::BitGen(), -25, 25));
  }

  for (TypeParam input : inputs) {
    quantile_tree->AddEntry(input);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<GaussianMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      quantile_tree->MakePrivate(dp_params).value();

  double confidence_level = 0.95;
  for (double quantile : kQuantilesToTest) {
    absl::StatusOr<ConfidenceInterval> ci_or_status_1 =
        results.ComputeNoiseConfidenceInterval(quantile, confidence_level);
    absl::StatusOr<ConfidenceInterval> ci_or_status_2 =
        results.ComputeNoiseConfidenceInterval(quantile, confidence_level);
    EXPECT_OK(ci_or_status_1);
    EXPECT_OK(ci_or_status_2);
    EXPECT_EQ(ci_or_status_1.value().lower_bound(),
              ci_or_status_2.value().lower_bound());
    EXPECT_EQ(ci_or_status_1.value().upper_bound(),
              ci_or_status_2.value().upper_bound());
  }
}

TYPED_TEST(QuantileTreeTest,
           ConfidenceIntervalWithLaplaceNoiseReturnsSameResults) {
  std::unique_ptr<QuantileTree<TypeParam>> quantile_tree =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(absl::BitGen(), -25, 25));
  }

  for (TypeParam input : inputs) {
    quantile_tree->AddEntry(input);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      quantile_tree->MakePrivate(dp_params).value();

  double confidence_level = 0.95;
  for (double quantile : kQuantilesToTest) {
    absl::StatusOr<ConfidenceInterval> ci_or_status_1 =
        results.ComputeNoiseConfidenceInterval(quantile, confidence_level);
    absl::StatusOr<ConfidenceInterval> ci_or_status_2 =
        results.ComputeNoiseConfidenceInterval(quantile, confidence_level);
    EXPECT_OK(ci_or_status_1);
    EXPECT_OK(ci_or_status_2);
    EXPECT_EQ(ci_or_status_1.value().lower_bound(),
              ci_or_status_2.value().lower_bound());
    EXPECT_EQ(ci_or_status_1.value().upper_bound(),
              ci_or_status_2.value().upper_bound());
  }
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithGaussianNoiseReturnsLowerLevelIntervalsWithinHigherLevelIntervals) {
  std::unique_ptr<QuantileTree<TypeParam>> quantile_tree =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(absl::BitGen(), -25, 25));
  }

  for (TypeParam input : inputs) {
    quantile_tree->AddEntry(input);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<GaussianMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      quantile_tree->MakePrivate(dp_params).value();

  for (double quantile : kQuantilesToTest) {
    absl::StatusOr<ConfidenceInterval> ci_or_status_1 =
        results.ComputeNoiseConfidenceInterval(quantile, 0.9);
    absl::StatusOr<ConfidenceInterval> ci_or_status_2 =
        results.ComputeNoiseConfidenceInterval(quantile, 0.99);
    EXPECT_OK(ci_or_status_1);
    EXPECT_OK(ci_or_status_2);
    EXPECT_LE(ci_or_status_1.value().lower_bound(),
              ci_or_status_2.value().lower_bound());
    EXPECT_GE(ci_or_status_1.value().upper_bound(),
              ci_or_status_2.value().upper_bound());
  }
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithLaplaceNoiseReturnsLowerLevelIntervalsWithinHigherLevelIntervals) {
  std::unique_ptr<QuantileTree<TypeParam>> quantile_tree =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  std::vector<TypeParam> inputs;
  for (int i = 0; i < kDefaultDatasetSize; ++i) {
    inputs.push_back(absl::Uniform(absl::BitGen(), -25, 25));
  }

  for (TypeParam input : inputs) {
    quantile_tree->AddEntry(input);
  }

  typename QuantileTree<TypeParam>::DPParams dp_params;
  dp_params.epsilon = kTestDefaultEpsilon;
  dp_params.delta = kDefaultDelta;
  dp_params.max_contributions_per_partition =
      kDefaultMaxContributionsPerPartition;
  dp_params.max_partitions_contributed_to = kDefaultMaxPartitionsContributed;
  dp_params.mechanism_builder = std::make_unique<LaplaceMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      quantile_tree->MakePrivate(dp_params).value();

  // Test that all higher levels' confidence intervals are within lower levels'
  // confidence intervals.
  for (double quantile : kQuantilesToTest) {
    for (int i = 0; i < kConfidenceLevelsToTest.size(); ++i) {
      for (int j = i + 1; j < kConfidenceLevelsToTest.size(); ++j) {
        absl::StatusOr<ConfidenceInterval> ci_or_status_1 =
            results.ComputeNoiseConfidenceInterval(quantile,
                                                   kConfidenceLevelsToTest[i]);
        absl::StatusOr<ConfidenceInterval> ci_or_status_2 =
            results.ComputeNoiseConfidenceInterval(quantile,
                                                   kConfidenceLevelsToTest[j]);
        EXPECT_OK(ci_or_status_1);
        EXPECT_OK(ci_or_status_2);
        EXPECT_LE(ci_or_status_1.value().lower_bound(),
                  ci_or_status_2.value().lower_bound());
        EXPECT_GE(ci_or_status_1.value().upper_bound(),
                  ci_or_status_2.value().upper_bound());
      }
    }
  }
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithGaussianNoiseSatisfiesConfidenceLevelWithOneEntry) {
  typename QuantileTree<TypeParam>::Builder builder =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10);

  std::vector<TypeParam> entries = {0};

  this->StatisticallyAssertConfidenceLevel(
      entries, builder, std::make_unique<GaussianMechanism::Builder>());
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithLaplaceNoiseSatisfiesConfidenceLevelWithOneEntry) {
  typename QuantileTree<TypeParam>::Builder builder =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10);

  std::vector<TypeParam> entries = {0};

  this->StatisticallyAssertConfidenceLevel(
      entries, builder, std::make_unique<LaplaceMechanism::Builder>());
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithGaussianNoiseSatisfiesConfidenceLevelWithUniformEntries) {
  typename QuantileTree<TypeParam>::Builder builder =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(250)
          .SetLower(-250)
          .SetTreeHeight(4)
          .SetBranchingFactor(10);

  std::vector<TypeParam> entries;
  for (int i = -250; i <= 250; ++i) {
    entries.push_back(i);
  }

  this->StatisticallyAssertConfidenceLevel(
      entries, builder, std::make_unique<GaussianMechanism::Builder>());
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithLaplaceNoiseSatisfiesConfidenceLevelWithUniformEntries) {
  typename QuantileTree<TypeParam>::Builder builder =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(250)
          .SetLower(-250)
          .SetTreeHeight(4)
          .SetBranchingFactor(10);

  std::vector<TypeParam> entries;
  for (int i = -250; i <= 250; ++i) {
    entries.push_back(i);
  }

  this->StatisticallyAssertConfidenceLevel(
      entries, builder, std::make_unique<LaplaceMechanism::Builder>());
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithGaussianNoiseSatisfiesConfidenceLevelWithConstantEntries) {
  typename QuantileTree<TypeParam>::Builder builder =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10);

  std::vector<TypeParam> entries;
  for (int i = 0; i <= 20; ++i) {
    entries.push_back(3);
  }

  this->StatisticallyAssertConfidenceLevel(
      entries, builder, std::make_unique<GaussianMechanism::Builder>());
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithLaplaceNoiseSatisfiesConfidenceLevelWithConstantEntries) {
  typename QuantileTree<TypeParam>::Builder builder =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10);

  std::vector<TypeParam> entries;
  for (int i = 0; i <= 20; ++i) {
    entries.push_back(3);
  }

  this->StatisticallyAssertConfidenceLevel(
      entries, builder, std::make_unique<LaplaceMechanism::Builder>());
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithGaussianNoiseSatisfiesConfidenceLevelWithBernoulliEntries) {
  typename QuantileTree<TypeParam>::Builder builder =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10);

  std::vector<TypeParam> entries;
  for (int i = 0; i <= 100; ++i) {
    entries.push_back(1);
    entries.push_back(-1);
  }

  this->StatisticallyAssertConfidenceLevel(
      entries, builder, std::make_unique<GaussianMechanism::Builder>());
}

TYPED_TEST(
    QuantileTreeTest,
    ConfidenceIntervalWithLaplaceNoiseSatisfiesConfidenceLevelWithBernoulliEntries) {
  typename QuantileTree<TypeParam>::Builder builder =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10);

  std::vector<TypeParam> entries;
  for (int i = 0; i <= 100; ++i) {
    entries.push_back(1);
    entries.push_back(-1);
  }

  this->StatisticallyAssertConfidenceLevel(
      entries, builder, std::make_unique<LaplaceMechanism::Builder>());
}

TYPED_TEST(QuantileTreeTest, MemoryUsed) {
  std::unique_ptr<QuantileTree<TypeParam>> empty =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();
  std::unique_ptr<QuantileTree<TypeParam>> once =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();
  std::unique_ptr<QuantileTree<TypeParam>> twice =
      typename QuantileTree<TypeParam>::Builder()
          .SetUpper(50)
          .SetLower(-50)
          .SetTreeHeight(4)
          .SetBranchingFactor(10)
          .Build()
          .value();

  once->AddEntry(-49);
  twice->AddEntry(49);
  twice->AddEntry(49);
  EXPECT_GT(once->MemoryUsed(), empty->MemoryUsed());
  EXPECT_EQ(once->MemoryUsed(), twice->MemoryUsed());
}

TEST(QuantileTreeTest, BuildWithLargeDoubleBoundsFails) {
  absl::StatusOr<std::unique_ptr<QuantileTree<double>>> quantile_tree =
      QuantileTree<double>::Builder()
          .SetUpper(1.7e308)
          .SetLower(-1.7e308)
          .Build();
  EXPECT_THAT(quantile_tree,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("failed floating point overflow check")));
}

TEST(QuantileTreeTest, BuildWithLargeInt64BoundsFails) {
  absl::StatusOr<std::unique_ptr<QuantileTree<int64_t>>> quantile_tree =
      QuantileTree<int64_t>::Builder()
          .SetUpper(std::numeric_limits<int64_t>::max())
          .SetLower(std::numeric_limits<int64_t>::lowest())
          .Build();
  EXPECT_THAT(quantile_tree,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("failed signed integer overflow check")));
}

}  // namespace
}  // namespace differential_privacy
