//
// Copyright 2024 Google LLC
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

#include "algorithms/internal/clamped-calculation-without-bounds.h"

#include <cstdint>
#include <limits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"

namespace differential_privacy::internal {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Gt;

template <typename T>
class ClampedCalcualtionWithoutBoundsTypedTest : public ::testing::Test {};

using TestTypes = ::testing::Types<int64_t, double>;
TYPED_TEST_SUITE(ClampedCalcualtionWithoutBoundsTypedTest, TestTypes);

TYPED_TEST(ClampedCalcualtionWithoutBoundsTypedTest,
           AddToPartialsWithPositiveInputModifiesVector) {
  typename ClampedCalculationWithoutBounds<TypeParam>::Options options;
  options.num_bins = 6;
  options.scale = 1;
  options.base = 2;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<TypeParam>::Create(options);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };
  std::vector<TypeParam> sums(6, 0);

  clamped_calculation->template AddToPartials<TypeParam>(&sums, 7, difference);

  EXPECT_THAT(sums, ElementsAre(1, 1, 2, 3, 0, 0));
}

TYPED_TEST(ClampedCalcualtionWithoutBoundsTypedTest,
           AddToPartialsWithBase3AndPositiveInputModifiesVector) {
  typename ClampedCalculationWithoutBounds<TypeParam>::Options options;
  options.num_bins = 6;
  options.scale = 1;
  options.base = 3;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<TypeParam>::Create(options);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };
  std::vector<TypeParam> sums(6, 0);

  clamped_calculation->template AddToPartials<TypeParam>(&sums, 12, difference);

  EXPECT_THAT(sums, ElementsAre(1, 2, 6, 3, 0, 0));
}

TYPED_TEST(ClampedCalcualtionWithoutBoundsTypedTest,
           AddToPartialsWithScale3Base7AndPositiveInputModifiesVector) {
  typename ClampedCalculationWithoutBounds<TypeParam>::Options options;
  options.num_bins = 6;
  options.scale = 3;
  options.base = 7;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<TypeParam>::Create(options);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };
  std::vector<TypeParam> sums(6, 0);

  clamped_calculation->template AddToPartials<TypeParam>(&sums, 162,
                                                         difference);

  EXPECT_THAT(sums, ElementsAre(3, 18, 126, 15, 0, 0));
}

TYPED_TEST(ClampedCalcualtionWithoutBoundsTypedTest,
           AddToPartialsWithNegativeInputModifiesVector) {
  typename ClampedCalculationWithoutBounds<TypeParam>::Options options;
  options.num_bins = 6;
  options.scale = 1;
  options.base = 2;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<TypeParam>::Create(options);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };
  std::vector<TypeParam> sums(6, 0);

  clamped_calculation->template AddToPartials<TypeParam>(&sums, -9, difference);

  EXPECT_THAT(sums, ElementsAre(-1, -1, -2, -4, -1, 0));
}

TYPED_TEST(ClampedCalcualtionWithoutBoundsTypedTest,
           AddToPartialsWithZeroInputDoesNotModifyVector) {
  typename ClampedCalculationWithoutBounds<TypeParam>::Options options;
  options.num_bins = 6;
  options.scale = 1;
  options.base = 2;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<TypeParam>::Create(options);
  auto difference = [](TypeParam val1, TypeParam val2) { return val1 - val2; };
  std::vector<TypeParam> sums(6, 0);

  clamped_calculation->template AddToPartials<TypeParam>(&sums, 0, difference);

  EXPECT_THAT(sums, ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(ClampedCalcualtionWithoutBoundsTest, AddToPartialsWithNanInputIgnored) {
  ClampedCalculationWithoutBounds<double>::Options options;
  options.num_bins = 6;
  options.scale = 1;
  options.base = 2;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<double>::Create(options);

  auto difference = [](double val1, double val2) { return val1 - val2; };
  std::vector<double> sums(6, 0);

  clamped_calculation->AddToPartials<double>(
      &sums, std::numeric_limits<double>::quiet_NaN(), difference);

  EXPECT_THAT(sums, ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(ClampedCalcualtionWithoutBoundsTest,
     AddToPartialsWithInfiniteInputModifiesVector) {
  ClampedCalculationWithoutBounds<double>::Options options;
  options.num_bins = 6;
  options.scale = 1;
  options.base = 2;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<double>::Create(options);

  auto difference = [](double val1, double val2) { return val1 - val2; };
  std::vector<double> sums(6, 0);

  clamped_calculation->AddToPartials<double>(
      &sums, std::numeric_limits<double>::infinity(), difference);

  EXPECT_THAT(sums, ElementsAre(1, 1, 2, 4, 8, 16));
}

TYPED_TEST(ClampedCalcualtionWithoutBoundsTypedTest,
           ComputeSumFromPartialsCalculatesClampedSum) {
  typename ClampedCalculationWithoutBounds<TypeParam>::Options options;
  options.num_bins = 6;
  options.scale = 1;
  options.base = 2;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<TypeParam>::Create(options);

  std::vector<TypeParam> pos_sum(6, 0), neg_sum(6, 0);
  clamped_calculation->AddToPartialSums(&pos_sum, 6);
  clamped_calculation->AddToPartialSums(&neg_sum, -3);

  EXPECT_THAT(clamped_calculation->template ComputeFromPartials<TypeParam>(
                  pos_sum, neg_sum, [](TypeParam x) { return x; }, /*lower=*/-4,
                  /*upper=*/4, 2),
              IsOkAndHolds(Eq(1)));

  EXPECT_THAT(clamped_calculation->template ComputeFromPartials<TypeParam>(
                  pos_sum, neg_sum, [](TypeParam x) { return x; }, /*lower=*/-4,
                  /*upper=*/-1, 2),
              IsOkAndHolds(Eq(-4)));

  EXPECT_THAT(clamped_calculation->template ComputeFromPartials<TypeParam>(
                  pos_sum, neg_sum, [](TypeParam x) { return x; }, /*lower=*/2,
                  /*upper=*/4, 2),
              IsOkAndHolds(Eq(6)));
}

TYPED_TEST(ClampedCalcualtionWithoutBoundsTypedTest,
           MemoryUsedReturnsPositiveValue) {
  typename ClampedCalculationWithoutBounds<TypeParam>::Options options;
  options.num_bins = 6;
  options.scale = 1;
  options.base = 2;
  auto clamped_calculation =
      ClampedCalculationWithoutBounds<TypeParam>::Create(options);
  EXPECT_THAT(clamped_calculation->MemoryUsed(), Gt(0));
}

}  // namespace
}  // namespace differential_privacy::internal
