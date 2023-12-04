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

#include "algorithms/internal/bounded-mean-ci.h"

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/confidence-interval.pb.h"

namespace differential_privacy {
namespace internal {
namespace {

using ::testing::_;
using ::testing::AtLeast;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::Gt;
using ::testing::Lt;
using ::testing::StrictMock;

constexpr double kDefaultEpsilon = 1.1;

class MockLaplaceMechanism : public LaplaceMechanism {
 public:
  MockLaplaceMechanism() : LaplaceMechanism(kDefaultEpsilon / 2.0) {}
  MOCK_METHOD(NumericalMechanism::NoiseConfidenceIntervalResult,
              UncheckedNoiseConfidenceInterval,
              (double confidence_level, double noised_result), (const));
};

TEST(BoundedMeanCiTest, BoundedMeanConfidenceIntervalHasBasicProperties) {
  BoundedMeanConfidenceIntervalParams params;
  params.confidence_level = 0.97654;

  params.lower_bound = -1.0;
  params.upper_bound = 1.0;

  // 1000 contributions with 500 sum is within lower and upper bound.
  params.noised_count = 1000;
  params.noised_sum = 500;

  StrictMock<MockLaplaceMechanism> sum_mock;
  EXPECT_CALL(sum_mock,
              UncheckedNoiseConfidenceInterval(_, DoubleEq(params.noised_sum)))
      .Times(AtLeast(1))
      .WillRepeatedly([](double confidence_level, double noised_sum) {
        NumericalMechanism::NoiseConfidenceIntervalResult r;
        r.lower = -0.5;
        r.upper = 0.5;
        return r;
      });
  params.sum_mechanism = &sum_mock;

  StrictMock<MockLaplaceMechanism> count_mock;
  EXPECT_CALL(count_mock, UncheckedNoiseConfidenceInterval(
                              _, DoubleEq(params.noised_count)))
      .Times(AtLeast(1))
      .WillRepeatedly([](double confidence_level, double noised_sum) {
        NumericalMechanism::NoiseConfidenceIntervalResult r;
        r.lower = -0.5;
        r.upper = 0.5;
        return r;
      });
  params.count_mechanism = &count_mock;

  const ConfidenceInterval ci = BoundedMeanConfidenceInterval(params);
  EXPECT_THAT(ci.confidence_level(), DoubleEq(params.confidence_level));
  EXPECT_THAT(ci.lower_bound(), Gt(params.lower_bound));
  EXPECT_THAT(ci.upper_bound(), Lt(params.upper_bound));
  EXPECT_LT(ci.lower_bound(), ci.upper_bound());
}

// Integration tests with real NumericalMechanism objects follow below.

// Returns the typically used Laplace mechanism for noising the sum in the mean
// aggregation.  There are further integration and statistical tests for bounded
// mean to make sure propertis for the overall aggregation hold as well.
std::unique_ptr<NumericalMechanism> LaplaceSumForParams(
    const BoundedMeanConfidenceIntervalParams &params) {
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> m =
      LaplaceMechanism::Builder()
          .SetEpsilon(kDefaultEpsilon / 2.0)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(
              std::abs(params.lower_bound - params.upper_bound) / 2.0)
          .Build();
  CHECK(m.ok()) << "Could not build Laplace for sum: " << m.status();
  return std::move(m).value();
}

// Returns the typically used Laplace mechanism for noising the count in the
// mean aggregation.
std::unique_ptr<NumericalMechanism> LaplaceCountForParams(
    const BoundedMeanConfidenceIntervalParams &params) {
  absl::StatusOr<std::unique_ptr<NumericalMechanism>> m =
      LaplaceMechanism::Builder()
          .SetEpsilon(kDefaultEpsilon / 2.0)
          .SetL0Sensitivity(1)
          .SetLInfSensitivity(1)
          .Build();
  CHECK(m.ok()) << "Could not build Laplace for count: " << m.status();
  return std::move(m).value();
}

TEST(BoundedMeanCiTest,
     BoundedMeanConfidenceIntervalReturnsExpectedPosMidpoint) {
  BoundedMeanConfidenceIntervalParams params;
  params.lower_bound = -1;
  params.upper_bound = 1;
  params.confidence_level = 0.95;

  // Use 1000 as noised contribution count and 500 as noised total sum -> CI
  // midpoint should be close to 0.5.
  params.noised_count = 1000;
  params.noised_sum = 500;

  std::unique_ptr<NumericalMechanism> count_mechanism =
      LaplaceCountForParams(params);
  params.count_mechanism = count_mechanism.get();
  std::unique_ptr<NumericalMechanism> sum_mechanism =
      LaplaceSumForParams(params);
  params.sum_mechanism = sum_mechanism.get();

  const ConfidenceInterval ci = BoundedMeanConfidenceInterval(params);

  EXPECT_LT(ci.lower_bound(), ci.upper_bound());
  EXPECT_THAT(ci.confidence_level(), DoubleEq(params.confidence_level));

  const double ci_midpoint = (ci.upper_bound() + ci.lower_bound()) / 2.0;
  ASSERT_THAT(ci_midpoint, DoubleNear(0.5, 0.01));
}

TEST(BoundedMeanCiTest,
     BoundedMeanConfidenceIntervalReturnsExpectedNegMidpoint) {
  BoundedMeanConfidenceIntervalParams params;
  params.lower_bound = -1;
  params.upper_bound = 1;
  params.confidence_level = 0.95;

  // Use 1000 as noised contribution count and -500 as noised total sum -> CI
  // midpoint should be close to -0.5.
  params.noised_count = 1000;
  params.noised_sum = -500;

  std::unique_ptr<NumericalMechanism> count_mechanism =
      LaplaceCountForParams(params);
  params.count_mechanism = count_mechanism.get();
  std::unique_ptr<NumericalMechanism> sum_mechanism =
      LaplaceSumForParams(params);
  params.sum_mechanism = sum_mechanism.get();

  const ConfidenceInterval ci = BoundedMeanConfidenceInterval(params);

  EXPECT_LT(ci.lower_bound(), ci.upper_bound());
  EXPECT_THAT(ci.confidence_level(), DoubleEq(params.confidence_level));

  const double ci_midpoint = (ci.upper_bound() + ci.lower_bound()) / 2.0;
  ASSERT_THAT(ci_midpoint, DoubleNear(-0.5, 0.01));
}

TEST(BoundedMeanCiTest,
     BoundedMeanConfidenceIntervalWithLowerLevelGetsTighter) {
  BoundedMeanConfidenceIntervalParams params;
  params.lower_bound = -1;
  params.upper_bound = 1;
  params.noised_count = 1000;
  params.noised_sum = 250;

  std::unique_ptr<NumericalMechanism> count_mechanism =
      LaplaceCountForParams(params);
  params.count_mechanism = count_mechanism.get();

  std::unique_ptr<NumericalMechanism> sum_mechanism =
      LaplaceSumForParams(params);
  params.sum_mechanism = sum_mechanism.get();

  BoundedMeanConfidenceIntervalParams params_higher_conf_level = params;
  params_higher_conf_level.confidence_level = 0.95;

  BoundedMeanConfidenceIntervalParams params_lower_conf_level = params;
  params_lower_conf_level.confidence_level = 0.8;

  const ConfidenceInterval ci_higher =
      BoundedMeanConfidenceInterval(params_higher_conf_level);
  const ConfidenceInterval ci_lower =
      BoundedMeanConfidenceInterval(params_lower_conf_level);

  // The CI returned with higher confidence level has to be included in the CI
  // for the lower confidence level.
  EXPECT_LT(ci_higher.lower_bound(), ci_lower.lower_bound());
  EXPECT_GT(ci_higher.upper_bound(), ci_lower.upper_bound());
}

TEST(BoundedMeanCiTest,
     BoundedMeanConfidenceIntervalWithMorePrivacyUnitsGetsTighter) {
  BoundedMeanConfidenceIntervalParams params;
  params.confidence_level = 0.94321;
  params.lower_bound = -1;
  params.upper_bound = 1;

  std::unique_ptr<NumericalMechanism> count_mechanism =
      LaplaceCountForParams(params);
  params.count_mechanism = count_mechanism.get();

  std::unique_ptr<NumericalMechanism> sum_mechanism =
      LaplaceSumForParams(params);
  params.sum_mechanism = sum_mechanism.get();

  BoundedMeanConfidenceIntervalParams params_more_units = params;
  params_more_units.noised_count = 1000;
  params_more_units.noised_sum = -500;

  BoundedMeanConfidenceIntervalParams params_fewer_units = params;
  params_fewer_units.noised_count = 100;
  params_fewer_units.noised_sum = -50;

  const ConfidenceInterval ci_more_units =
      BoundedMeanConfidenceInterval(params_more_units);
  const ConfidenceInterval ci_fewer_units =
      BoundedMeanConfidenceInterval(params_fewer_units);

  // The CI returned with fewer privacy units has to be included in the CI with
  // more privacy units.
  EXPECT_LT(ci_fewer_units.lower_bound(), ci_more_units.lower_bound());
  EXPECT_GT(ci_fewer_units.upper_bound(), ci_more_units.upper_bound());
}

TEST(BoundedMeanCiTest, InfiniteNoisedParamsReturnsDefaultCi) {
  BoundedMeanConfidenceIntervalParams params;
  params.confidence_level = 0.9;
  params.lower_bound = -1.0;
  params.upper_bound = 1.0;
  params.noised_sum = std::numeric_limits<double>::infinity();
  params.noised_count = std::numeric_limits<double>::infinity();
  params.sum_mechanism = nullptr;
  params.count_mechanism = nullptr;

  EXPECT_THAT(BoundedMeanConfidenceInterval(params), EqualsProto(""));
}

}  // namespace
}  // namespace internal
}  // namespace differential_privacy
