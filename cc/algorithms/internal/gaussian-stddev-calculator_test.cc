//
// Copyright 2022 Google LLC
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

#include "algorithms/internal/gaussian-stddev-calculator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace differential_privacy {
namespace internal {
namespace {

using ::testing::DoubleEq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Le;

// The standard deviation as calcualted from Theorem 1 of
// https://arxiv.org/pdf/1805.06530v2.pdf.  Only holds for epsilon < 1.
double Theorem1Stddev(double epsilon, double delta, double l2_sensitivity) {
  return l2_sensitivity * std::sqrt(2.0 * std::log(1.25 / delta)) / epsilon;
}

TEST(GaussianStddevCalculatorTest,
     CalculateGaussianStddevReturnsPositiveValue) {
  const double epsilon = std::log(3);
  const double delta = 1e-7;
  const double l2_sensitivity = 1.2;
  EXPECT_THAT(CalculateGaussianStddev(epsilon, delta, l2_sensitivity), Gt(0.0));
}

TEST(GaussianStddevCalculatorTest,
     CalculateGaussianStddevReturnsNonZeroForHighDelta) {
  const double epsilon = std::log(3);
  const double delta = 1.0;
  const double l2_sensitivity = 1.2;
  EXPECT_THAT(CalculateGaussianStddev(epsilon, delta, l2_sensitivity), Gt(0.0));
}

TEST(GaussianStddevCalculator,
     CalculateGaussianStddevReturnsLowerStddevThanTheorem1) {
  const double epsilon = std::log(3);
  const double delta = 1e-7;
  const double l2_sensitivity = 1.2;
  EXPECT_THAT(CalculateGaussianStddev(epsilon, delta, l2_sensitivity),
              Le(Theorem1Stddev(epsilon, delta, l2_sensitivity)));
}

TEST(GaussianStddevCalculatorTest,
     CalculateDeltaForGaussianStddevReturnsPositiveValue) {
  const double epsilon = std::log(3);
  const double l2_sensitivity = 1.0;
  const double stddev = 1.1;
  EXPECT_THAT(CalculateDeltaForGaussianStddev(epsilon, l2_sensitivity, stddev),
              Gt(0.0));
}

TEST(GaussianStddevCalculatorTest,
     CalculateDeltaForStddevWithLowSensitivityAndHighEpsReturnsZero) {
  const double epsilon = 100;
  const double l2_sensitivity = 2.1;
  const double stddev = 1;
  EXPECT_THAT(CalculateDeltaForGaussianStddev(epsilon, l2_sensitivity, stddev),
              DoubleEq(0.0));
}

TEST(GaussianStddevCalculatorTest,
     CalculateDeltaForGaussianStddevTighterThanTheorem1) {
  const double epsilon = std::log(3);
  const double delta = 1e-7;
  const double l2_sensitivity = 1.1;
  EXPECT_THAT(CalculateDeltaForGaussianStddev(
                  epsilon, l2_sensitivity,
                  /* stddev= */ Theorem1Stddev(epsilon, delta, l2_sensitivity)),
              Le(delta));
}

TEST(GaussianStddevCalculatorTest,
     CalculateBoundsForGaussianStddevReturnsPositiveBounds) {
  const double epsilon = std::log(3);
  const double delta = 1e-7;
  const double l2_sensitivity = 1.2;
  const BoundsForGaussianStddev bounds =
      CalculateBoundsForGaussianStddev(epsilon, delta, l2_sensitivity);
  EXPECT_THAT(bounds.lower, Gt(0.0));
  EXPECT_THAT(bounds.upper, Gt(0.0));
}

TEST(GaussianStddevCalculatorTest,
     CalculateBoundsForGaussianStddevReturnsLowerThatIsLowerThanUpper) {
  const double epsilon = std::log(3);
  const double delta = 1e-7;
  const double l2_sensitivity = 1.2;
  const BoundsForGaussianStddev bounds =
      CalculateBoundsForGaussianStddev(epsilon, delta, l2_sensitivity);
  EXPECT_THAT(bounds.lower, Le(bounds.upper));
}

TEST(GaussianStddevCalculatorTest,
     CalculateBoundsForGaussianStddevReturnsValidBounds) {
  const double epsilon = std::log(3);
  const double delta = 1e-7;
  const double l2_sensitivity = 1.3;
  const BoundsForGaussianStddev bounds =
      CalculateBoundsForGaussianStddev(epsilon, delta, l2_sensitivity);
  EXPECT_THAT(CalculateDeltaForGaussianStddev(epsilon, l2_sensitivity,
                                              /* stddev= */ bounds.upper),
              Le(delta));
  EXPECT_THAT(CalculateDeltaForGaussianStddev(epsilon, l2_sensitivity,
                                              /* stddev= */ bounds.lower),
              Ge(delta));
}

}  // namespace
}  // namespace internal
}  // namespace differential_privacy
