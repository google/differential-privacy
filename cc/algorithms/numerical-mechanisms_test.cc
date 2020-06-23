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

#include "algorithms/numerical-mechanisms.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "algorithms/distributions.h"

namespace differential_privacy {
namespace {

using testing::_;
using testing::DoubleEq;
using testing::DoubleNear;
using testing::Eq;
using testing::Ge;
using testing::MatchesRegex;
using testing::Return;

class MockLaplaceDistribution : public internal::LaplaceDistribution {
 public:
  MockLaplaceDistribution() : internal::LaplaceDistribution(1.0, 1.0) {}
  MOCK_METHOD1(Sample, double(double));
};

template <typename T>
class NumericalMechanismsTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(NumericalMechanismsTest, NumericTypes);

TYPED_TEST(NumericalMechanismsTest, LaplaceBuilder) {
  LaplaceMechanism::Builder test_builder;
  std::unique_ptr<LaplaceMechanism> test_mechanism =
      test_builder.SetEpsilon(1).SetSensitivity(3).Build().ValueOrDie();

  EXPECT_DOUBLE_EQ(test_mechanism->GetEpsilon(), 1);
  EXPECT_DOUBLE_EQ(test_mechanism->GetSensitivity(), 3);
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonNotSet) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Epsilon has to be set.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonZero) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(1).SetEpsilon(0).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Epsilon has to be positive.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonNegative) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(1).SetEpsilon(-1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Epsilon has to be positive.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonNan) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(1).SetEpsilon(NAN).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Epsilon has to be finite.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonInfinity) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL1Sensitivity(1).SetEpsilon(INFINITY).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Epsilon has to be finite.*"));
}

TYPED_TEST(NumericalMechanismsTest, LaplaceBuilderSensitivityTooHigh) {
  LaplaceMechanism::Builder test_builder;
  base::StatusOr<std::unique_ptr<LaplaceMechanism>> test_mechanism =
      test_builder.SetEpsilon(1)
          .SetSensitivity(std::numeric_limits<double>::max())
          .Build();
  EXPECT_FALSE(test_mechanism.ok());
}

TEST(NumericalMechanismsTest, LaplaceAddsNoise) {
  auto distro = absl::make_unique<MockLaplaceDistribution>();
  ON_CALL(*distro, Sample(_)).WillByDefault(Return(10.0));
  LaplaceMechanism mechanism(1.0, 1.0, std::move(distro));

  EXPECT_THAT(mechanism.AddNoise(0.0), DoubleNear(10.0, 5.0));
}

TEST(NumericalMechanismsTest, LaplaceAddsNoNoiseWhenSensitivityIsZero) {
  LaplaceMechanism mechanism(1.0, 0.0);

  EXPECT_THAT(mechanism.AddNoise(12.3), DoubleEq(12.3));
}

TEST(NumericalMechanismsTest, LaplaceDiversityCorrect) {
  LaplaceMechanism mechanism(1.0, 1.0);
  EXPECT_EQ(mechanism.GetDiversity(), 1.0);

  LaplaceMechanism mechanism2(2.0, 1.0);
  EXPECT_EQ(mechanism2.GetDiversity(), 0.5);

  LaplaceMechanism mechanism3(2.0, 3.0);
  EXPECT_EQ(mechanism3.GetDiversity(), 1.5);
}

TEST(NumericalMechanismsTest, LaplaceBudgetCorrect) {
  auto distro = absl::make_unique<MockLaplaceDistribution>();
  EXPECT_CALL(*distro, Sample(1.0)).Times(1);
  EXPECT_CALL(*distro, Sample(2.0)).Times(1);
  EXPECT_CALL(*distro, Sample(4.0)).Times(1);
  LaplaceMechanism mechanism(1.0, 1.0, std::move(distro));

  mechanism.AddNoise(0.0, 1.0);
  mechanism.AddNoise(0.0, 0.5);
  mechanism.AddNoise(0.0, 0.25);
}

TEST(NumericalMechanismsTest, LaplaceSnaps) {
  auto distro = absl::make_unique<MockLaplaceDistribution>();
  EXPECT_CALL(*distro, Sample(_))
      .WillOnce(Return(10.0))
      .WillOnce(Return(10.001));
  LaplaceMechanism mechanism(1.0, 1.0, std::move(distro));

  EXPECT_THAT(mechanism.AddNoise(0.0),
              DoubleNear(mechanism.AddNoise(0.0), 0.0001));
}

TEST(NumericalMechanismsTest, LaplaceWorksForIntegers) {
  auto distro = absl::make_unique<MockLaplaceDistribution>();
  ON_CALL(*distro, Sample(_)).WillByDefault(Return(10.0));
  LaplaceMechanism mechanism(1.0, 1.0, std::move(distro));

  EXPECT_EQ(static_cast<int64_t>(mechanism.AddNoise(0)), 10);
}

TEST(NumericalMechanismsTest, LaplaceConfidenceInterval) {
  double epsilon = 0.5;
  double sensitivity = 1.0;
  double level = .95;
  double budget = .5;
  LaplaceMechanism mechanism(epsilon, sensitivity);
  base::StatusOr<ConfidenceInterval> confidence_interval =
      mechanism.NoiseConfidenceInterval(level, budget);
  EXPECT_TRUE(confidence_interval.ok());
  EXPECT_EQ(confidence_interval.ValueOrDie().lower_bound(),
            log(1 - level) / epsilon / budget);
  EXPECT_EQ(confidence_interval.ValueOrDie().upper_bound(),
            -log(1 - level) / epsilon / budget);
  EXPECT_EQ(confidence_interval.ValueOrDie().confidence_level(), level);

  double result = 19.3;
  base::StatusOr<ConfidenceInterval> confidence_interval_with_result =
      mechanism.NoiseConfidenceInterval(level, budget, result);
  EXPECT_TRUE(confidence_interval.ok());
  EXPECT_EQ(confidence_interval_with_result.ValueOrDie().lower_bound(),
            result + (log(1 - level) / epsilon / budget));
  EXPECT_EQ(confidence_interval_with_result.ValueOrDie().upper_bound(),
            result - (log(1 - level) / epsilon / budget));
  EXPECT_EQ(confidence_interval_with_result.ValueOrDie().confidence_level(), level);

}

TYPED_TEST(NumericalMechanismsTest, LaplaceBuilderClone) {
  LaplaceMechanism::Builder test_builder;
  std::unique_ptr<LaplaceMechanism::Builder> clone =
      test_builder.SetEpsilon(1).SetSensitivity(3).Clone();
  std::unique_ptr<LaplaceMechanism> test_mechanism =
      clone->Build().ValueOrDie();

  EXPECT_DOUBLE_EQ(test_mechanism->GetEpsilon(), 1);
  EXPECT_DOUBLE_EQ(test_mechanism->GetSensitivity(), 3);
}

TEST(NumericalMechanismsTest, LaplaceEstimatesL1WithL0AndLInf) {
  LaplaceMechanism::Builder builder;
  std::unique_ptr<LaplaceMechanism> mechanism = builder.SetEpsilon(1)
                                                    .SetL0Sensitivity(5)
                                                    .SetLInfSensitivity(3)
                                                    .Build()
                                                    .ValueOrDie();
  EXPECT_THAT(mechanism->GetSensitivity(), Ge(3));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaNotSet) {
  GaussianMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL2Sensitivity(1).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Delta has to be set.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaNan) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL2Sensitivity(1).SetEpsilon(1).SetDelta(NAN).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Delta has to be finite.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaNegative) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL2Sensitivity(1).SetEpsilon(1).SetDelta(-1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Delta has to be in the interval.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaOne) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL2Sensitivity(1).SetEpsilon(1).SetDelta(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(base::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(std::string(failed_build.status().message()));
  EXPECT_THAT(message, MatchesRegex("^Delta has to be in the interval.*"));
}

TEST(NumericalMechanismsTest, GaussianMechanismAddsNoise) {
  GaussianMechanism mechanism(1.0, 0.5, 1.0);

  EXPECT_TRUE(mechanism.AddNoise(1.0) != 1.0);
  EXPECT_TRUE(mechanism.AddNoise(1.1) != 1.1);

  // Test values that should be clamped.
  EXPECT_FALSE(std::isnan(mechanism.AddNoise(1.1, 2.0)));
  EXPECT_FALSE(std::isnan(mechanism.AddNoise(1.1, -2.0)));
}

TEST(NumericalMechanismsTest, GaussNoiseConfidenceInterval) {
  // Tests the NoiseConfidenceInterval method for Gaussian noise.
  // Standard deviations are pre-calculated using CalculateStdDev
  // in the Gaussian mechanism class. True bounds are also pre-calculated
  // using a confidence interval calcualtor.

  int NUM_TESTS = 3;
  double epsilons[] = {1.2, 1.0, 10.0};
  double deltas[] = {0.3, 0.5, 0.5};
  double sensitivities[] = {1.0, 1.0, 1.0};
  double levels[] = {.9, .95, .95};
  double budgets[] = {.5, .5, .75};
  double results[] = {0,1.3,2.7};
  // calculated using standard deviations of
  // 3.4855, 3.60742, 0.367936 respectively
  double true_bounds[] = {-5.733, -7.07, -0.7211};

  for (int i = 0; i < NUM_TESTS; i++) {
    double epsilon = epsilons[i];
    double delta = deltas[i];
    double sensitivity = sensitivities[i];
    double level = levels[i];
    double budget = budgets[i];
    double result = results[i];
    double true_lower_bound = result + true_bounds[i];
    double true_upper_bound = result - true_bounds[i];

    GaussianMechanism mechanism(epsilon, delta, sensitivity);
    base::StatusOr<ConfidenceInterval> confidence_interval =
      mechanism.NoiseConfidenceInterval(level, budget, result);

    EXPECT_TRUE(confidence_interval.ok());
    EXPECT_NEAR(confidence_interval.ValueOrDie().lower_bound(),
            true_lower_bound, 0.001);
    EXPECT_NEAR(confidence_interval.ValueOrDie().upper_bound(),
            true_upper_bound, 0.001);
    EXPECT_EQ(confidence_interval.ValueOrDie().confidence_level(), level);
  }
}

TEST(NumericalMechanismsTest, GaussianBuilderClone) {
  GaussianMechanism::Builder test_builder;
  auto clone =
      test_builder.SetEpsilon(1.1).SetDelta(0.5).SetL2Sensitivity(1.2).Clone();
  auto mechanism = clone->Build().ValueOrDie();

  EXPECT_DOUBLE_EQ(mechanism->GetEpsilon(), 1.1);
  EXPECT_DOUBLE_EQ(mechanism->GetDelta(), 0.5);
  EXPECT_DOUBLE_EQ(mechanism->GetL2Sensitivity(), 1.2);
}
}  // namespace
}  // namespace differential_privacy
