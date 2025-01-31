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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "algorithms/distributions.h"

namespace differential_privacy {
namespace {

using ::testing::_;
using ::testing::Contains;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::Lt;
using ::testing::MatchesRegex;
using ::testing::Not;
using ::testing::Return;
using ::differential_privacy::base::testing::StatusIs;

// Number of samples to use when taking samples is inexpensive.
constexpr int kNumSamples = 1e6;
// Number of samples to use when taking samples is expensive, such as using a
// NumericalMechanism to add noise.
constexpr int kSmallNumSamples = 1e4;

class MockLaplaceDistribution : public internal::LaplaceDistribution {
 public:
  MockLaplaceDistribution() : internal::LaplaceDistribution(1.0, 1.0) {}
  MOCK_METHOD(double, Sample, (), (override));
};

template <typename T>
class NumericalMechanismsTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(NumericalMechanismsTest, NumericTypes);

TYPED_TEST(NumericalMechanismsTest, LaplaceBuilder) {
  LaplaceMechanism::Builder test_builder;
  auto test_mechanism = test_builder.SetL1Sensitivity(3).SetEpsilon(1).Build();
  ASSERT_OK(test_mechanism);

  EXPECT_DOUBLE_EQ((*test_mechanism)->GetEpsilon(), 1);
  EXPECT_DOUBLE_EQ(
      dynamic_cast<LaplaceMechanism *>(test_mechanism->get())->GetSensitivity(),
      3);
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonNotSet) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^Epsilon must be set.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonZero) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(1).SetEpsilon(0).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^Epsilon must be finite and positive.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonNegative) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(1).SetEpsilon(-1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^Epsilon must be finite and positive.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonNan) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(1).SetEpsilon(NAN).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^Epsilon must be a valid numeric value.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsEpsilonInfinity) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL1Sensitivity(1).SetEpsilon(INFINITY).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^Epsilon must be finite.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsNoSensitivities) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              HasSubstr("LaplaceMechanism requires either L1 or (L0 and LInf) "
                        "sensitivities to be set, but none were set"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsOnlyL0Set) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(1).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              HasSubstr("LaplaceMechanism requires either L1 or (L0 and LInf) "
                        "sensitivities to be set, but only L0 was set"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsOnlyLInfSet) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetLInfSensitivity(1).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              HasSubstr("LaplaceMechanism requires either L1 or (L0 and LInf) "
                        "sensitivities to be set, but only LInf was set"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsCalculatedL1NotFinite) {
  LaplaceMechanism::Builder test_builder;
  const double high_sensitivity = std::numeric_limits<double>::max() - 1.0;
  auto failed_build = test_builder.SetL0Sensitivity(high_sensitivity)
                          .SetLInfSensitivity(high_sensitivity)
                          .SetEpsilon(1)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(
      message,
      HasSubstr(
          "The result of the L1 sensitivity calculation is not finite: inf. "
          "Please check your contribution and sensitivity settings"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsL1SensitivityZero) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(0).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^L1 sensitivity must be finite and positive.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsL1SensitivityNegative) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(-1).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^L1 sensitivity must be finite and positive.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsL1SensitivityNan) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL1Sensitivity(NAN).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^L1 sensitivity must be a valid numeric value..*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsL1SensitivityInfinity) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL1Sensitivity(INFINITY).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^L1 sensitivity must be finite.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsL0SensitivityZero) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(0)
                          .SetLInfSensitivity(1)
                          .SetEpsilon(1)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^L0 sensitivity must be finite and positive.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsL0SensitivityNegative) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(-1)
                          .SetLInfSensitivity(1)
                          .SetEpsilon(1)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^L0 sensitivity must be finite and positive.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsL0SensitivityNan) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(NAN)
                          .SetLInfSensitivity(1)
                          .SetEpsilon(1)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^L0 sensitivity must be a valid numeric value.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsLInfSensitivityZero) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(1)
                          .SetLInfSensitivity(0)
                          .SetEpsilon(1)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(
      message,
      MatchesRegex("^LInf sensitivity must be finite and positive, but is.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsLInfSensitivityNegative) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(1)
                          .SetLInfSensitivity(-1)
                          .SetEpsilon(1)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(
      message,
      MatchesRegex("^LInf sensitivity must be finite and positive, but is.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsLInfSensitivityNan) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(1)
                          .SetLInfSensitivity(NAN)
                          .SetEpsilon(1)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(
      message,
      MatchesRegex("^LInf sensitivity must be a valid numeric value.*"));
}

TEST(NumericalMechanismsTest, LaplaceBuilderFailsL0SensitivityInfinity) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(INFINITY)
                          .SetLInfSensitivity(1)
                          .SetEpsilon(1)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^L0 sensitivity must be finite.*"));
}

// When L1 is not directly provided, it is calculated as L0 * LInf. This tests
// ensures that floating-point limitations resulting in
// L1 = L0 * LInf = TINY_DOUBLE * TINY_DOUBLE = 0 are caught and invalidated.
TEST(NumericalMechanismsTest, LaplaceBuilderFailsSmallL0LargeLInfSensitivity) {
  LaplaceMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL0Sensitivity(std::numeric_limits<double>::min())
          .SetLInfSensitivity(std::numeric_limits<double>::min())
          .SetEpsilon(1)
          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(
      message,
      MatchesRegex("^The result of the L1 sensitivity calculation is 0.*"));
}

TYPED_TEST(NumericalMechanismsTest, LaplaceBuilderSensitivityTooHigh) {
  LaplaceMechanism::Builder test_builder;
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> test_mechanism =
      test_builder.SetL1Sensitivity(std::numeric_limits<double>::max())
          .SetEpsilon(1)
          .Build();
  EXPECT_FALSE(test_mechanism.ok());
}

TEST(NumericalMechanismsTest, LaplaceAddsNoise) {
  auto distro = absl::make_unique<MockLaplaceDistribution>();
  ON_CALL(*distro, Sample()).WillByDefault(Return(10.0));
  LaplaceMechanism mechanism(1.0, 1.0, std::move(distro));

  EXPECT_THAT(mechanism.AddNoise(0.0), DoubleNear(10.0, 5.0));
}

TEST(NumericalMechanismsTest, LaplaceNoisedValueAboveThreshold) {
  LaplaceMechanism::Builder builder;
  std::unique_ptr<NumericalMechanism> mechanism =
      builder.SetL1Sensitivity(1).SetEpsilon(1).Build().value();

  struct TestScenario {
    double input;
    double threshold;
    double expected_probability;
  };

  // To reduce flakiness from randomness, perform multiple trials and declare
  // the test successful if a sufficient expected number of trials provide the
  // expected result.
  std::vector<TestScenario> test_scenarios = {
      {-0.5, -0.5, 0.5000}, {0.0, -0.5, 0.6967}, {0.5, -0.5, 0.8160},
      {-0.5,  0.0, 0.3035}, {0.0,  0.0, 0.5000}, {0.5,  0.0, 0.6967},
      {-0.5,  0.5, 0.1840}, {0.0,  0.5, 0.3035}, {0.5,  0.5, 0.5000},
  };

  double num_above_thresold;
  for (TestScenario ts : test_scenarios) {
    num_above_thresold = 0;
    for (int i = 0; i < kNumSamples; ++i) {
      if (mechanism->NoisedValueAboveThreshold(ts.input, ts.threshold))
        ++num_above_thresold;
    }
    EXPECT_NEAR(num_above_thresold / kNumSamples, ts.expected_probability,
                0.0025);
  }
}

TEST(NumericalMechanismsTest, LaplaceDiversityCorrect) {
  LaplaceMechanism mechanism(1.0, 1.0);
  EXPECT_EQ(mechanism.GetDiversity(), 1.0);

  LaplaceMechanism mechanism2(2.0, 1.0);
  EXPECT_EQ(mechanism2.GetDiversity(), 0.5);

  LaplaceMechanism mechanism3(2.0, 3.0);
  EXPECT_EQ(mechanism3.GetDiversity(), 1.5);
}

TEST(NumericalMechanismsTest, LaplaceVarianceCorrect) {
  const double epsilon = 2.123;
  const double l0 = 3.456;
  const double linf = 5.789;

  // Calculate diversity and variance directly from above parameters.
  const double diversity = (l0 * linf) / epsilon;
  const double variance = 2.0 * std::pow(diversity, 2);

  absl::StatusOr<std::unique_ptr<NumericalMechanism>> mechanism =
      LaplaceMechanism::Builder()
          .SetEpsilon(epsilon)
          .SetL0Sensitivity(l0)
          .SetLInfSensitivity(linf)
          .Build();

  ASSERT_OK(mechanism.status());
  EXPECT_NEAR(mechanism->get()->GetVariance(), variance, 1e-6);
}

TEST(NumericalMechanismsTest, LaplaceWorksForIntegers) {
  auto distro = absl::make_unique<MockLaplaceDistribution>();
  ON_CALL(*distro, Sample()).WillByDefault(Return(10.0));
  LaplaceMechanism mechanism(1.0, 1.0, std::move(distro));

  EXPECT_EQ(static_cast<int64_t>(mechanism.AddNoise(0)), 10);
}

TEST(NumericalMechanismsTest, LaplaceRoundsToGranularity_Double) {
  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^-10. Granularity ~= sensitivity / epsilon / 2^40 ~= 1 / 2^-30 / 2^40
  // = 2^30 / 2^40.
  std::unique_ptr<NumericalMechanism> small_granularity_mech =
      LaplaceMechanism::Builder()
          .SetEpsilon(4.7e-10)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1)
          .Build()
          .value();

  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^0. Granularity ~= sensitivity / epsilon / 2^40 ~= 1 / 2^-40 / 2^40
  // = 2^40 / 2^40.
  std::unique_ptr<NumericalMechanism> med_granularity_mech =
      LaplaceMechanism::Builder()
          .SetEpsilon(9.1e-13)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1)
          .Build()
          .value();

  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^10. Granularity ~= sensitivity / epsilon / 2^40 ~= 1 / 2^-50 / 2^40
  // = 2^50 / 2^40.
  std::unique_ptr<NumericalMechanism> large_granularity_mech =
      LaplaceMechanism::Builder()
          .SetEpsilon(8.9e-16)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1)
          .Build()
          .value();

  for (int i = 0; i < kSmallNumSamples; ++i) {
    // The rounding process should be independent of the value of x. Setting x
    // to a value between -1*10^6 and 10^6 at random should covere a broad range
    // of congruence classes.
    double input = UniformDouble() * 2e6 - 1e6;

    EXPECT_EQ(std::fmod(small_granularity_mech->AddNoise(input), 1.0 / 1024.0),
              0);
    EXPECT_EQ(std::fmod(med_granularity_mech->AddNoise(input), 1), 0);
    EXPECT_EQ(std::fmod(large_granularity_mech->AddNoise(input), 1024), 0);
  }
}

TEST(NumericalMechanismsTest, LaplaceRoundsToGranularity_Int) {
  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^1. Granularity ~= sensitivity / epsilon / 2^40 ~= 1 / 2^-41 / 2^40
  // = 2^41 / 2^40 = 2.
  std::unique_ptr<NumericalMechanism> med_granularity_mech =
      LaplaceMechanism::Builder()
          .SetEpsilon(4.6e-13)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1)
          .Build()
          .value();

  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^10. Granularity ~= sensitivity / epsilon / 2^40 ~= 1 / 2^-50 / 2^40
  // = 2^50 / 2^40 = 2^10 = 1024.
  std::unique_ptr<NumericalMechanism> large_granularity_mech =
      LaplaceMechanism::Builder()
          .SetEpsilon(8.9e-16)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1)
          .Build()
          .value();

  for (int i = 0; i < kSmallNumSamples; ++i) {
    // The rounding process should be independent of the value of x. Setting x
    // to a value between -1*10^6 and 10^6 at random should covere a broad range
    // of congruence classes.
    int64_t input = UniformDouble() * 2e6 - 1e6;
    EXPECT_EQ(std::fmod(med_granularity_mech->AddNoise(input), 2), 0);
    EXPECT_EQ(std::fmod(large_granularity_mech->AddNoise(input), 1024), 0);
  }
}

TEST(NumericalMechanismsTest, LaplaceConfidenceInterval) {
  double epsilon = 0.5;
  double sensitivity = 1.0;
  double level = .95;
  LaplaceMechanism mechanism(epsilon, sensitivity);
  absl::StatusOr<ConfidenceInterval> confidence_interval =
      mechanism.NoiseConfidenceInterval(level);
  ASSERT_OK(confidence_interval);
  EXPECT_LT(confidence_interval->lower_bound(),
            confidence_interval->upper_bound());
  EXPECT_EQ(confidence_interval->lower_bound(), std::log(1 - level) / epsilon);
  EXPECT_EQ(confidence_interval->upper_bound(), -std::log(1 - level) / epsilon);
  EXPECT_EQ(confidence_interval->confidence_level(), level);

  double result = 19.3;
  absl::StatusOr<ConfidenceInterval> confidence_interval_with_result =
      mechanism.NoiseConfidenceInterval(level, result);
  ASSERT_OK(confidence_interval_with_result);
  EXPECT_EQ(confidence_interval_with_result->lower_bound(),
            result + (std::log(1 - level) / epsilon));
  EXPECT_EQ(confidence_interval_with_result->upper_bound(),
            result - (std::log(1 - level) / epsilon));
  EXPECT_EQ(confidence_interval_with_result->confidence_level(), level);
}

TEST(NumericalMechanismsTest,
     LaplaceConfidenceIntervalFailsForConfidenceLevelNan) {
  LaplaceMechanism mechanism(1.0, 1.0);
  auto failed_confidence_interval = mechanism.NoiseConfidenceInterval(NAN);
  EXPECT_THAT(
      failed_confidence_interval,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Confidence level must be a valid numeric value")));
}

TEST(NumericalMechanismsTest, LaplaceCdfPositive) {
  auto mechanism =
      LaplaceMechanism::Builder().SetL1Sensitivity(3).SetEpsilon(2).Build();
  ASSERT_OK(mechanism);

  EXPECT_DOUBLE_EQ((*mechanism)->Cdf(0.5), 0.641734344713105374787);
}

TEST(NumericalMechanismsTest, LaplaceCdfNegative) {
  auto mechanism =
      LaplaceMechanism::Builder().SetL1Sensitivity(3).SetEpsilon(2).Build();
  ASSERT_OK(mechanism);

  EXPECT_DOUBLE_EQ((*mechanism)->Cdf(-0.7), 0.313544542636528);
}

TEST(NumericalMechanismsTest, LaplaceQuantilePositive) {
  auto mechanism =
      LaplaceMechanism::Builder().SetL1Sensitivity(3).SetEpsilon(2).Build();
  ASSERT_OK(mechanism);

  EXPECT_DOUBLE_EQ((*mechanism)->Quantile(0.641734344713105374787), 0.5);
}

TEST(NumericalMechanismsTest, LaplaceQuantileNegative) {
  auto mechanism =
      LaplaceMechanism::Builder().SetL1Sensitivity(3).SetEpsilon(2).Build();
  ASSERT_OK(mechanism);

  EXPECT_DOUBLE_EQ((*mechanism)->Quantile(0.313544542636528), -0.7);
}

TYPED_TEST(NumericalMechanismsTest, LaplaceBuilderClone) {
  LaplaceMechanism::Builder test_builder;
  std::unique_ptr<NumericalMechanismBuilder> clone =
      test_builder.SetL1Sensitivity(3).SetEpsilon(1).Clone();
  auto test_mechanism = clone->Build();
  ASSERT_OK(test_mechanism);

  EXPECT_DOUBLE_EQ((*test_mechanism)->GetEpsilon(), 1);
  EXPECT_DOUBLE_EQ(
      dynamic_cast<LaplaceMechanism *>(test_mechanism->get())->GetSensitivity(),
      3);
}

class NoiseIntervalMultipleParametersTests
    : public ::testing::TestWithParam<struct conf_int_params> {};

struct conf_int_params {
  double epsilon;
  double delta;
  double sensitivity;
  double level;
  double result;
  double true_bound;
};

// True bounds calculated using standard deviations of
// 0.644043, 0.507324, 0.213379, respectively.
struct conf_int_params gauss_params1 = {/*epsilon =*/1.2,
                                        /*delta =*/0.3,
                                        /*sensitivity =*/1.0,
                                        /*level =*/.9,
                                        /*result =*/0,
                                        /*true_bound =*/-1.05936};

struct conf_int_params gauss_params2 = {/*epsilon =*/1.0,
                                        /*delta =*/0.5,
                                        /*sensitivity =*/1.0,
                                        /*level =*/.95,
                                        /*result =*/1.3,
                                        /*true_bound =*/-0.994337};

struct conf_int_params gauss_params3 = {/*epsilon =*/10.0,
                                        /*delta =*/0.5,
                                        /*sensitivity =*/1.0,
                                        /*level =*/.95,
                                        /*result =*/2.7,
                                        /*true_bound =*/-0.418215};

INSTANTIATE_TEST_SUITE_P(TestSuite, NoiseIntervalMultipleParametersTests,
                         testing::Values(gauss_params1, gauss_params2,
                                         gauss_params3));

TEST_P(NoiseIntervalMultipleParametersTests, GaussNoiseConfidenceInterval) {
  // Tests the NoiseConfidenceInterval method for Gaussian noise.
  // Standard deviations are pre-calculated using CalculateStdDev
  // in the Gaussian mechanism class. True bounds are also pre-calculated
  // using a confidence interval calcualtor.

  struct conf_int_params params = GetParam();
  double epsilon = params.epsilon;
  double delta = params.delta;
  double sensitivity = params.sensitivity;
  double conf_level = params.level;
  double result = params.result;
  double true_lower_bound = params.result + params.true_bound;
  double true_upper_bound = params.result - params.true_bound;

  GaussianMechanism mechanism(epsilon, delta, sensitivity);
  absl::StatusOr<ConfidenceInterval> confidence_interval =
      mechanism.NoiseConfidenceInterval(conf_level, result);

  ASSERT_OK(confidence_interval);
  EXPECT_NEAR(confidence_interval->lower_bound(), true_lower_bound, 0.001);
  EXPECT_NEAR(confidence_interval->upper_bound(), true_upper_bound, 0.001);
  EXPECT_EQ(confidence_interval->confidence_level(), conf_level);
}

TEST(NumericalMechanismsTest, LaplaceEstimatesL1WithL0AndLInf) {
  LaplaceMechanism::Builder builder;
  auto mechanism =
      builder.SetEpsilon(1).SetL0Sensitivity(5).SetLInfSensitivity(3).Build();
  ASSERT_OK(mechanism);
  EXPECT_THAT(
      dynamic_cast<LaplaceMechanism *>(mechanism->get())->GetSensitivity(),
      Ge(3));
}

TEST(NumericalMechanismsTest, AddNoise) {
  auto distro = absl::make_unique<MockLaplaceDistribution>();
  double granularity = distro->GetGranularity();
  ON_CALL(*distro, Sample()).WillByDefault(Return(10));
  LaplaceMechanism mechanism(1.0, 1.0, std::move(distro));

  double remainder =
      std::fmod(mechanism.AddNoise(0.1 * granularity), granularity);
  EXPECT_EQ(remainder, 0);
  EXPECT_THAT(mechanism.AddNoise(0.1 * granularity),
              DoubleNear(10.0, 0.000001));
}

TEST(NumericalMechanismsTest, LambdaTooSmall) {
  LaplaceMechanism::Builder test_builder;
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> test_mechanism_or =
      test_builder.SetL1Sensitivity(3)
          .SetEpsilon(1.0 / std::pow(10, 100))
          .Build();
  EXPECT_FALSE(test_mechanism_or.ok());
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaNotSet) {
  GaussianMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL2Sensitivity(1).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^Delta must be set.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaNan) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL2Sensitivity(1).SetEpsilon(1).SetDelta(NAN).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^Delta must be a valid numeric value.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaNegative) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL2Sensitivity(1).SetEpsilon(1).SetDelta(-1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^Delta must be in the exclusive interval.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaOne) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL2Sensitivity(1).SetEpsilon(1).SetDelta(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^Delta must be in the exclusive interval.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsDeltaZero) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL2Sensitivity(1).SetEpsilon(1).SetDelta(0).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^Delta must be in the exclusive interval.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsL0SensitivityNan) {
  GaussianMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(NAN)
                          .SetLInfSensitivity(1)
                          .SetEpsilon(1)
                          .SetDelta(0.2)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^L0 sensitivity must be a valid numeric value.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsLInfSensitivityInfinity) {
  GaussianMechanism::Builder test_builder;
  auto failed_build = test_builder.SetL0Sensitivity(1)
                          .SetLInfSensitivity(INFINITY)
                          .SetEpsilon(1)
                          .SetDelta(0.2)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^LInf sensitivity must be finite.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsL2SensitivityNan) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetL2Sensitivity(NAN).SetEpsilon(1).SetDelta(0.2).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(message,
              MatchesRegex("^L2 sensitivity must be a valid numeric value.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsCalculatedL2SensitivityZero) {
  GaussianMechanism::Builder test_builder;
  auto failed_build = test_builder.SetEpsilon(1)
                          .SetDelta(0.2)
                          // Use very low L0 and LInf sensitivities so that the
                          // calculation of l2 will result in 0.
                          .SetL0Sensitivity(4.94065645841247e-323)
                          .SetLInfSensitivity(5.24566986113514e-317)
                          .Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  // Convert message to std::string so that the matcher works in the open source
  // version.
  std::string message(failed_build.status().message());
  EXPECT_THAT(
      message,
      MatchesRegex(
          "^The calculated L2 sensitivity must be positive and finite.*"));
}

TEST(NumericalMechanismsTest, GaussianBuilderFailsWithStddevAndOtherParams) {
  GaussianMechanism::Builder test_builder;
  auto failed_build =
      test_builder.SetStandardDeviation(1).SetEpsilon(1).Build();
  EXPECT_THAT(failed_build.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  std::string message(failed_build.status().message());
  EXPECT_THAT(message, MatchesRegex("^If standard deviation is set directly it "
                                    "must be the only parameter.*"));

  auto failed_build2 =
      test_builder.SetStandardDeviation(1).SetDelta(0.1).Build();
  EXPECT_THAT(failed_build2.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  message = failed_build2.status().message();
  EXPECT_THAT(message, MatchesRegex("^If standard deviation is set directly it "
                                    "must be the only parameter.*"));

  auto failed_build3 =
      test_builder.SetStandardDeviation(1).SetL0Sensitivity(1).Build();
  EXPECT_THAT(failed_build3.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  message = failed_build3.status().message();
  EXPECT_THAT(message, MatchesRegex("^If standard deviation is set directly it "
                                    "must be the only parameter.*"));

  auto failed_build4 =
      test_builder.SetStandardDeviation(1).SetLInfSensitivity(1).Build();
  EXPECT_THAT(failed_build4.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  message = failed_build4.status().message();
  EXPECT_THAT(message, MatchesRegex("^If standard deviation is set directly it "
                                    "must be the only parameter.*"));

  auto failed_build5 =
      test_builder.SetStandardDeviation(1).SetL2Sensitivity(1).Build();
  EXPECT_THAT(failed_build5.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  message = failed_build5.status().message();
  EXPECT_THAT(message, MatchesRegex("^If standard deviation is set directly it "
                                    "must be the only parameter.*"));
}

TEST(NumericalMechanismsTest, GaussianMechanismAddsNoise) {
  GaussianMechanism mechanism(1.0, 0.5, 1.0);

  EXPECT_TRUE(mechanism.AddNoise(1.0) != 1.0);
  EXPECT_TRUE(mechanism.AddNoise(1.1) != 1.1);
}

TEST(NumericalMechanismsTest,
     GaussianMechanismAddsNoiseForHighEpsilonAndLowDelta) {
  auto test_mechanism = GaussianMechanism::Builder()
                            .SetL2Sensitivity(6.2324042213746395e-184)
                            .SetDelta(2.7161546250836291e-312)
                            .SetEpsilon(1.257239018692402e+232)
                            .Build();
  EXPECT_TRUE(test_mechanism.ok());

  const double raw_value = 2.7161546250836291e-312;
  double noised_value = (*test_mechanism)->AddNoise(raw_value);
  EXPECT_TRUE(std::isfinite(noised_value));
}

TEST(NumericalMechanismsTest, GaussianMechanismAddsNoiseForLowDelta) {
  auto test_mechanism = GaussianMechanism::Builder()
                            .SetL2Sensitivity(1.0)
                            .SetDelta(6.486452831e-47)
                            .SetEpsilon(1.0)
                            .Build();
  ASSERT_TRUE(test_mechanism.ok());
  EXPECT_NEAR(dynamic_cast<GaussianMechanism *>(test_mechanism->get())
                  ->CalculateStddev(1, 6.486452831e-47, 1),
              14.0, 0.0001);
}

TEST(NumericalMechanismsTest, GaussianRoundsToGranularity_Double) {
  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^-10. Granularity ~= 2 * sigma / 2^57. We pick a sigma that yields a
  // granularity of 2^-10, and then adjust sensitivity (sigma scales linearly
  // with LInf sensitivity) to adjust the granularity.
  std::unique_ptr<NumericalMechanism> small_granularity_mech =
      GaussianMechanism::Builder()
          .SetEpsilon(1.0e-15)
          .SetDelta(1.0e-14)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1)
          .Build()
          .value();

  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^0 = 1.
  std::unique_ptr<NumericalMechanism> med_granularity_mech =
      GaussianMechanism::Builder()
          .SetEpsilon(1.0e-15)
          .SetDelta(1.0e-14)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1024)
          .Build()
          .value();

  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^10 = 1024.
  std::unique_ptr<NumericalMechanism> large_granularity_mech =
      GaussianMechanism::Builder()
          .SetEpsilon(1.0e-15)
          .SetDelta(1.0e-14)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1048576.0)
          .Build()
          .value();

  for (int i = 0; i < kSmallNumSamples; ++i) {
    // The rounding process should be independent of the value of x. Setting x
    // to a value between -1*10^6 and 10^6 at random should cover a broad range
    // of congruence classes.
    double input = UniformDouble() * 2000000.0 - 1000000.0;

    EXPECT_EQ(std::fmod(small_granularity_mech->AddNoise(input), 1.0 / 1024.0),
              0);
    EXPECT_EQ(std::fmod(med_granularity_mech->AddNoise(input), 1), 0);
    EXPECT_EQ(std::fmod(large_granularity_mech->AddNoise(input), 1024), 0);
  }
}

TEST(NumericalMechanismsTest, GaussianRoundsToGranularity_Int) {
  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^1 = 2.
  std::unique_ptr<NumericalMechanism> med_granularity_mech =
      GaussianMechanism::Builder()
          .SetEpsilon(1.0e-15)
          .SetDelta(1.0e-14)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(2048)
          .Build()
          .value();

  // These choices of epsilon and sensitivities should result in a granularity
  // of 2^10 = 1024.
  std::unique_ptr<NumericalMechanism> large_granularity_mech =
      GaussianMechanism::Builder()
          .SetEpsilon(1.0e-15)
          .SetDelta(1.0e-14)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1048576)
          .Build()
          .value();

  for (int i = 0; i < kSmallNumSamples; ++i) {
    // The rounding process should be independent of the value of x. Setting x
    // to a value between -1*10^6 and 10^6 at random should covere a broad range
    // of congruence classes.
    int64_t input = UniformDouble() * 2e6 - 1e6;

    EXPECT_EQ(std::fmod(med_granularity_mech->AddNoise(input), 2), 0);
    EXPECT_EQ(std::fmod(large_granularity_mech->AddNoise(input), 1024), 0);
  }
}

TEST(NumericalMechanismsTest, GaussianMechanismCdf) {
  auto mechanism = GaussianMechanism::Builder()
                       .SetL2Sensitivity(1)
                       .SetEpsilon(1)
                       .SetDelta(0.5)
                       .Build()
                       .value();
  EXPECT_NEAR(mechanism->GetVariance(), 0.257378, 1e-6);

  EXPECT_THAT(mechanism->Cdf(0.5), DoubleNear(0.837826, 1e-6));
}

TEST(NumericalMechanismsTest, GaussianMechanismQuantile) {
  auto mechanism = GaussianMechanism::Builder()
                       .SetL2Sensitivity(1)
                       .SetEpsilon(1)
                       .SetDelta(0.5)
                       .Build()
                       .value();
  EXPECT_NEAR(mechanism->GetVariance(), 0.257378, 1e-6);

  EXPECT_THAT(mechanism->Quantile(0.837826), DoubleNear(0.5, 1e-6));
}

TEST(NumericalMechanismsTest, GaussianMechanismNoisedValueAboveThreshold) {
  GaussianMechanism::Builder builder;
  std::unique_ptr<NumericalMechanism> mechanism =
      builder.SetL2Sensitivity(1).SetEpsilon(1).SetDelta(0.5).Build().value();
  // If the computed variance changes, then we need to update the probabilities
  // in test_scenarios below.
  EXPECT_NEAR(mechanism->GetVariance(), 0.257378, 1e-6);

  struct TestScenario {
    double input;
    double threshold;
    double expected_probability;
  };

  // To reduce flakiness from randomness, perform multiple trials and declare
  // the test successful if a sufficient expected number of trials provide the
  // expected result.
  std::vector<TestScenario> test_scenarios = {
      {-0.5, -0.5, 0.5000}, {0.0, -0.5, 0.8378}, {0.5, -0.5, 0.9756},
      {-0.5,  0.0, 0.1622}, {0.0,  0.0, 0.5000}, {0.5,  0.0, 0.8378},
      {-0.5,  0.5, 0.0244}, {0.0,  0.5, 0.1622}, {0.5,  0.5, 0.5000},
  };

  double num_above_thresold;
  for (TestScenario ts : test_scenarios) {
    num_above_thresold = 0;
    for (int i = 0; i < kNumSamples; ++i) {
      if (mechanism->NoisedValueAboveThreshold(ts.input, ts.threshold))
        ++num_above_thresold;
    }
    EXPECT_NEAR(num_above_thresold / kNumSamples, ts.expected_probability,
                0.0025);
  }
}

TEST(NumericalMechanismsTest, GaussianBuilderClone) {
  GaussianMechanism::Builder test_builder;
  auto clone =
      test_builder.SetL2Sensitivity(1.2).SetEpsilon(1.1).SetDelta(0.5).Clone();
  auto mechanism = clone->Build();
  ASSERT_OK(mechanism);

  EXPECT_DOUBLE_EQ((*mechanism)->GetEpsilon(), 1.1);
  EXPECT_DOUBLE_EQ(
      dynamic_cast<GaussianMechanism *>(mechanism->get())->GetDelta(), 0.5);
  EXPECT_DOUBLE_EQ(
      dynamic_cast<GaussianMechanism *>(mechanism->get())->GetL2Sensitivity(),
      1.2);
}

TEST(NumericalMechanismsTest, Stddev) {
  auto mechanism = GaussianMechanism::Builder()
                       .SetL2Sensitivity(1.0)
                       .SetEpsilon(std::log(3))
                       .SetDelta(0.00001)
                       .Build();
  auto gaussian = dynamic_cast<GaussianMechanism *>(mechanism.value().get());
  EXPECT_DOUBLE_EQ(gaussian->CalculateStddev(), 3.42578125);
  // Call CalculateStddev with parameters differing from the attributes.
  EXPECT_DOUBLE_EQ(gaussian->CalculateStddev(std::log(4), 0.00002, 3.0),
                   7.986328125);
}

TEST(NumericalMechanismsTest, GaussianVarianceReturnsWallysResult) {
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> mechanism =
      GaussianMechanism::Builder()
          .SetEpsilon(1)
          .SetDelta(1e-6)
          .SetL0Sensitivity(2)
          .SetLInfSensitivity(3)
          .Build();
  ASSERT_OK(mechanism.status());

  // Ensure the returned variance roughly matches what (broken link) returns.
  // Currently Wally returns a stddev of 17.922 for these values.
  EXPECT_NEAR(mechanism->get()->GetVariance(), std::pow(17.922, 2), 0.5);
}

TEST(NumericalMechanismsTest, GaussianSetStddev) {
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> mechanism =
      GaussianMechanism::Builder().SetStandardDeviation(3.5).Build();
  ASSERT_OK(mechanism.status());

  std::vector<double> samples;
  for (int i = 0; i < kNumSamples; ++i) {
    samples.push_back((*mechanism)->AddNoise(0));
  }

  EXPECT_NEAR(StandardDev(samples), 3.5, 0.1);
}

TEST(NumericalMechanismsTest, LaplaceMechanismSerialization) {
  const double epsilon = std::log(3);
  const double l0 = 2;
  const double linf = 3;
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> test_mechanism =
      LaplaceMechanism::Builder()
          .SetEpsilon(epsilon)
          .SetL0Sensitivity(l0)
          .SetLInfSensitivity(linf)
          .Build();
  ASSERT_OK(test_mechanism);
  auto *laplace_mechanism =
      dynamic_cast<LaplaceMechanism *>(test_mechanism->get());

  serialization::LaplaceMechanism serialized_data =
      laplace_mechanism->Serialize();

  absl::StatusOr<std::unique_ptr<NumericalMechanism>> deserialized =
      LaplaceMechanism::Deserialize(serialized_data);
  ASSERT_OK(deserialized);

  EXPECT_EQ((*deserialized)->GetEpsilon(), epsilon);
  EXPECT_EQ(
      laplace_mechanism->GetSensitivity(),
      dynamic_cast<LaplaceMechanism *>(deserialized->get())->GetSensitivity());
}

TEST(NumericalMechanismsTest, GaussianMechanismSerialization) {
  const double epsilon = std::log(3);
  const double delta = 10e-8;
  const double l0 = 2;
  const double linf = 3;
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> test_mechanism =
      GaussianMechanism::Builder()
          .SetEpsilon(epsilon)
          .SetDelta(delta)
          .SetL0Sensitivity(l0)
          .SetLInfSensitivity(linf)
          .Build();
  ASSERT_OK(test_mechanism);
  auto *gaussian_mechanism =
      dynamic_cast<GaussianMechanism *>(test_mechanism->get());

  serialization::GaussianMechanism serialized_data =
      gaussian_mechanism->Serialize();

  absl::StatusOr<std::unique_ptr<NumericalMechanism>> deserialized =
      GaussianMechanism::Deserialize(serialized_data);
  ASSERT_OK(deserialized);

  EXPECT_EQ((*deserialized)->GetEpsilon(), epsilon);
  auto *deserialized_gaussian =
      dynamic_cast<GaussianMechanism *>(deserialized->get());
  EXPECT_EQ(deserialized_gaussian->GetDelta(), delta);
  EXPECT_EQ(deserialized_gaussian->GetL2Sensitivity(),
            gaussian_mechanism->GetL2Sensitivity());
}

TEST(NumericalMechanismTest,
     MinVarianceMechanismBuilderWithLowSensitivityReturnsLaplace) {
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> must_be_laplace =
      MinVarianceMechanismBuilder()
          .SetEpsilon(1)
          .SetDelta(1e-5)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(2)
          .Build();
  ASSERT_OK(must_be_laplace);

  // dynamic_cast will return null when it cannot cast to the specified type.
  LaplaceMechanism *laplace =
      dynamic_cast<LaplaceMechanism *>(must_be_laplace->get());
  EXPECT_THAT(laplace, Not(IsNull()));

  GaussianMechanism *will_be_null =
      dynamic_cast<GaussianMechanism *>(must_be_laplace->get());
  EXPECT_THAT(will_be_null, IsNull());
}

TEST(NumericalMechanismTest,
     MinVarianceMechanismBuilderWithHighSensitivityReturnsGaussian) {
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> must_be_gaussian =
      MinVarianceMechanismBuilder()
          .SetEpsilon(1)
          .SetDelta(1e-5)
          .SetL0Sensitivity(10)
          .SetLInfSensitivity(20)
          .Build();
  ASSERT_OK(must_be_gaussian);

  // dynamic_cast will return null when it cannot cast to the specified type.
  GaussianMechanism *gaussian =
      dynamic_cast<GaussianMechanism *>(must_be_gaussian->get());
  EXPECT_THAT(gaussian, Not(IsNull()));

  LaplaceMechanism *will_be_null =
      dynamic_cast<LaplaceMechanism *>(must_be_gaussian->get());
  EXPECT_THAT(will_be_null, IsNull());
}

TEST(NumericalMechanismTest,
     MinVarianceMechanismBuilderWithoutDeltaReturnsLaplace) {
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> must_be_laplace =
      MinVarianceMechanismBuilder()
          .SetEpsilon(1)
          .SetL0Sensitivity(10)
          .SetLInfSensitivity(20)
          .Build();
  ASSERT_OK(must_be_laplace);

  // dynamic_cast will return null when it cannot cast to the specified type.
  LaplaceMechanism *laplace =
      dynamic_cast<LaplaceMechanism *>(must_be_laplace->get());
  EXPECT_THAT(laplace, Not(IsNull()));

  GaussianMechanism *will_be_null =
      dynamic_cast<GaussianMechanism *>(must_be_laplace->get());
  EXPECT_THAT(will_be_null, IsNull());
}

TEST(NumericalMechanismTest, MinVarianceMechanismBuilderFailsWithoutEpsilon) {
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> fails =
      MinVarianceMechanismBuilder()
          .SetDelta(1e-5)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(2)
          .Build();

  EXPECT_THAT(fails.status().code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(fails.status().message(), HasSubstr("Epsilon must be set"));
}

TEST(NumericalMechanismTest, AddNoiseReturnsNegativeValuesForUnsignedInt) {
  // Flakiness of this test is approximately 1 / 2**50 ~= 8e-16
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> mechanism =
      LaplaceMechanism::Builder()
          .SetEpsilon(0.001)  // low epsilon for more variance
          .SetL0Sensitivity(20)
          .SetLInfSensitivity(100)
          .Build();
  ASSERT_OK(mechanism);

  std::vector<int64_t> results;
  for (int i = 0; i < 50; ++i) {
    results.push_back(mechanism.value()->AddNoise<uint8_t>(0));
  }

  EXPECT_THAT(results, Contains(Lt(0)));
}

}  // namespace
}  // namespace differential_privacy
