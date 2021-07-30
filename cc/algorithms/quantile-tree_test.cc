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

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "algorithms/numerical-mechanisms-testing.h"

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

template <typename T>
class QuantileTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();
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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();
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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<LaplaceMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<LaplaceMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<LaplaceMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<LaplaceMechanism::Builder>();

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
  dp_params.mechanism_builder = absl::make_unique<ZeroNoiseMechanism::Builder>();

  typename QuantileTree<TypeParam>::Privatized results =
      test_quantiles->MakePrivate(dp_params).value();

  double tolerance = 0.01;  // > upper - lower / branchingFactor ^ treeHeight
  EXPECT_NEAR(results.GetQuantile(0.5).value(), -25, tolerance);

  for (int i = 0; i < 1000; ++i) {
    test_quantiles->AddEntry(25);
  }

  EXPECT_NEAR(results.GetQuantile(0.5).value(), -25, tolerance);
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

}  // namespace
}  // namespace differential_privacy
