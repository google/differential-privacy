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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounded-mean.h"
#include "algorithms/bounded-standard-deviation.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/bounded-variance.h"
#include "algorithms/count.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/order-statistics.h"
#include "algorithms/util.h"
#include "testing/sequence.h"
#include "testing/stochastic_tester.h"

namespace differential_privacy {
namespace {

using ::differential_privacy::continuous::Max;
using ::differential_privacy::continuous::Median;
using ::differential_privacy::continuous::Min;
using ::differential_privacy::continuous::Percentile;
using ::differential_privacy::test_utils::SeededLaplaceMechanism;
using ::differential_privacy::testing::HaltonSequence;
using ::differential_privacy::testing::StochasticTester;
using ::differential_privacy::testing::StoredSequence;

constexpr int kNumDatasetsToTest = 500;
constexpr int kSmallNumDatasetsToTest = 100;
constexpr int kVerySmallNumDatasetsToTest = 5;
constexpr int kNumSamplesPerHistogram = 20000;

const double kDefaultOverallEpsilon = std::log(3);
const double kDefaultBoundsEpsilon = std::log(3) / 2;

template <typename T>
class StochasticDifferentialPrivacyTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<BoundedSum<double>, BoundedMean<double>,
                         BoundedVariance<double>,
                         BoundedStandardDeviation<double>>
    BoundedDpAlgorithms;
TYPED_TEST_SUITE(StochasticDifferentialPrivacyTest, BoundedDpAlgorithms);

TYPED_TEST(StochasticDifferentialPrivacyTest, AllBoundedDpAlgorithms) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      testing::DefaultDatasetSize(), true /* sorted_only */,
      testing::DefaultDataScale(), testing::DefaultDataOffset());
  auto algorithm = typename TypeParam::Builder()
                       .SetLaplaceMechanism(
                           absl::make_unique<SeededLaplaceMechanism::Builder>())
                       .SetEpsilon(std::log(3))
                       .SetLower(sequence->RangeMin())
                       .SetUpper(sequence->RangeMax())
                       .Build()
                       .value();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  kSmallNumDatasetsToTest,
                                  kNumSamplesPerHistogram);
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticDifferentialPrivacyTest, Max) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      testing::DefaultDatasetSize(), true /* sorted_only */,
      testing::DefaultDataScale(), testing::DefaultDataOffset());

  double lower = sequence->RangeMin();
  double upper = sequence->RangeMax();
  auto algorithm = Max<double>::Builder()
                       .SetLaplaceMechanism(
                           absl::make_unique<SeededLaplaceMechanism::Builder>())
                       .SetEpsilon(std::log(3))
                       .SetLower(lower)
                       .SetUpper(upper)
                       .Build()
                       .value();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  kVerySmallNumDatasetsToTest,
                                  kNumSamplesPerHistogram);
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticDifferentialPrivacyTest, Min) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      testing::DefaultDatasetSize(), true /* sorted_only */,
      testing::DefaultDataScale(), testing::DefaultDataOffset());

  double lower = sequence->RangeMin();
  double upper = sequence->RangeMax();
  auto algorithm = Min<double>::Builder()
                       .SetLaplaceMechanism(
                           absl::make_unique<SeededLaplaceMechanism::Builder>())
                       .SetEpsilon(std::log(3))
                       .SetLower(lower)
                       .SetUpper(upper)
                       .Build()
                       .value();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  kVerySmallNumDatasetsToTest,
                                  kNumSamplesPerHistogram);
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticDifferentialPrivacyTest, Median) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      testing::DefaultDatasetSize(), true /* sorted_only */,
      testing::DefaultDataScale(), testing::DefaultDataOffset());

  double lower = sequence->RangeMin();
  double upper = sequence->RangeMax();
  auto algorithm = Median<double>::Builder()
                       .SetLaplaceMechanism(
                           absl::make_unique<SeededLaplaceMechanism::Builder>())
                       .SetEpsilon(std::log(3))
                       .SetLower(lower)
                       .SetUpper(upper)
                       .Build()
                       .value();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  kVerySmallNumDatasetsToTest,
                                  kNumSamplesPerHistogram);
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticDifferentialPrivacyTest, Percentile) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      testing::DefaultDatasetSize(), true /* sorted_only */,
      testing::DefaultDataScale(), testing::DefaultDataOffset());

  double percentile = 0.9;
  double lower = sequence->RangeMin();
  double upper = sequence->RangeMax();
  auto algorithm = Percentile<double>::Builder()
                       .SetLaplaceMechanism(
                           absl::make_unique<SeededLaplaceMechanism::Builder>())
                       .SetPercentile(percentile)
                       .SetEpsilon(std::log(3))
                       .SetLower(lower)
                       .SetUpper(upper)
                       .Build()
                       .value();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  kVerySmallNumDatasetsToTest,
                                  kNumSamplesPerHistogram);
  EXPECT_TRUE(tester.Run());
}

// For Count, by creating a single dataset of a particular size, executing the
// DP tester for this dataset on a single path in the search space is
// equivalent to testing all datasets of that size, assuming that the
// algorithm does not depend on the actual values of in the dataset.
TEST(StochasticDifferentialPrivacyTest, CountNonBranchingSearch) {
  constexpr int kCountNonBranchingSearchDatasetSize = 2500;
  std::vector<int64_t> dataset(kCountNonBranchingSearchDatasetSize, 0);
  std::vector<std::vector<int64_t>> datasets;
  datasets.emplace_back(dataset);
  auto sequence = absl::make_unique<StoredSequence<int64_t>>(datasets);
  std::unique_ptr<Count<int64_t>> algorithm =
      Count<int64_t>::Builder()
          .SetLaplaceMechanism(
              absl::make_unique<SeededLaplaceMechanism::Builder>())
          .SetEpsilon(std::log(3))
          .Build()
          .value();
  StochasticTester<int64_t> tester(std::move(algorithm), std::move(sequence),
                                   /*num_datasets=*/1, kNumSamplesPerHistogram,
                                   /*disable_search_branching=*/true);
  EXPECT_TRUE(tester.Run());
}

// The stochaster tester tests the first output element: approximate minimum.
TEST(StochasticDifferentialPrivacyTest, ApproxBoundsMinimum) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      testing::DefaultDatasetSize(), true, testing::DefaultDataScale(),
      testing::DefaultDataOffset());
  auto algorithm = ApproxBounds<double>::Builder()
                       .SetLaplaceMechanism(
                           absl::make_unique<SeededLaplaceMechanism::Builder>())
                       .SetEpsilon(1)
                       .SetBase(2)
                       .SetNumBins(1)
                       .SetScale(.2)
                       .SetSuccessProbability(.9)
                       .Build()
                       .value();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  kNumDatasetsToTest, kNumSamplesPerHistogram);
  EXPECT_TRUE(tester.Run());
}

TYPED_TEST(StochasticDifferentialPrivacyTest, AllApproxBoundedDpAlgorithms) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      testing::DefaultDatasetSize(), true, testing::DefaultDataScale(),
      testing::DefaultDataOffset());

  absl::StatusOr<std::unique_ptr<ApproxBounds<double>>> bounds =
      ApproxBounds<double>::Builder()
          .SetLaplaceMechanism(
              absl::make_unique<SeededLaplaceMechanism::Builder>())
          .SetEpsilon(kDefaultBoundsEpsilon)
          .SetScale(.2)
          .SetBase(2)
          .SetNumBins(1)
          .SetSuccessProbability(.90)
          .Build();
  ASSERT_TRUE(bounds.ok()) << bounds.status().message();

  // The sum mechanism is remade for every run of the algorithm. Thus, we need
  // to ensure an outside generator is passed into the mechanism builder.
  std::seed_seq seed({1, 1, 1, 1, 1});
  std::mt19937 rand_gen(seed);
  auto mech_builder = SeededLaplaceMechanism::Builder().rand_gen(&rand_gen);
  absl::StatusOr<std::unique_ptr<TypeParam>> algorithm =
      typename TypeParam::Builder()
          .SetLaplaceMechanism(
              absl::make_unique<SeededLaplaceMechanism::Builder>(mech_builder))
          .SetEpsilon(kDefaultOverallEpsilon)
          .SetApproxBounds(std::move(bounds).value())
          .Build();
  ASSERT_TRUE(algorithm.ok()) << algorithm.status().message();

  StochasticTester<double> tester(std::move(algorithm).value(),
                                  std::move(sequence), kNumDatasetsToTest,
                                  kNumSamplesPerHistogram);
  EXPECT_TRUE(tester.Run());
}

}  // namespace
}  // namespace differential_privacy
