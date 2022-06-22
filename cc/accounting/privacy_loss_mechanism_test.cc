// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "accounting/privacy_loss_mechanism.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "base/testing/status_matchers.h"

namespace differential_privacy {
namespace accounting {
namespace {
using ::testing::DoubleNear;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::Values;
using ::differential_privacy::base::testing::StatusIs;

constexpr double kMaxError = 1e-4;

TEST(LaplacePrivacyLoss, InvalidParameter) {
  absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> mechanism =
      LaplacePrivacyLoss::Create(/*parameter=*/-1, /*sensitivity=*/1);

  EXPECT_THAT(mechanism, StatusIs(absl::InvalidArgumentError("").code(),
                                  HasSubstr("parameter should be positive")));
}

TEST(LaplacePrivacyLoss, InvalidSensitivity) {
  absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> mechanism =
      LaplacePrivacyLoss::Create(/*parameter=*/1, /*sensitivity=*/-1);

  EXPECT_THAT(mechanism, StatusIs(absl::InvalidArgumentError("").code(),
                                  HasSubstr("sensitivity should be positive")));
}

TEST(LaplacePrivacyLoss, CreateFromEpsilonDelta) {
  EpsilonDelta epsilon_delta = {3, 0.01};
  absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> mechanism =
      LaplacePrivacyLoss::Create(epsilon_delta);
  ASSERT_OK(mechanism);
  EXPECT_NEAR(mechanism.value()->Parameter(), 0.333333, kMaxError);
}

TEST(LaplacePrivacyLoss, PrivacyLoss) {
  absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> mechanism =
      LaplacePrivacyLoss::Create(/*parameter=*/1, /*sensitivity=*/1);
  ASSERT_OK(mechanism);
  EXPECT_DOUBLE_EQ(mechanism.value()->PrivacyLoss(-0.1), 1);

  mechanism = LaplacePrivacyLoss::Create(7, 7);
  ASSERT_OK(mechanism);
  EXPECT_DOUBLE_EQ(mechanism.value()->PrivacyLoss(2.1), 0.4);
}

TEST(LaplacePrivacyLoss, InversePrivacyLoss) {
  absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> mechanism =
      LaplacePrivacyLoss::Create(/*parameter=*/1, /*sensitivity=*/1);
  ASSERT_OK(mechanism);
  EXPECT_DOUBLE_EQ(mechanism.value()->InversePrivacyLoss(1), 0);
}

TEST(LaplacePrivacyLoss, PrivacyLossTail) {
  absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> mechanism =
      LaplacePrivacyLoss::Create(/*parameter=*/1, /*sensitivity=*/2);
  ASSERT_OK(mechanism);
  PrivacyLossTail result = mechanism.value()->PrivacyLossDistributionTail();
  EXPECT_DOUBLE_EQ(result.lower_x_truncation, 0);
  EXPECT_DOUBLE_EQ(result.upper_x_truncation, 2);
  EXPECT_THAT(
      result.probability_mass_function,
      UnorderedElementsAre(
          Pair(DoubleNear(2, kMaxError), DoubleNear(0.5, kMaxError)),
          Pair(DoubleNear(-2, kMaxError), DoubleNear(0.06766764, kMaxError))));
}

TEST(LaplacePrivacyLoss, GetDeltaForEpsilon) {
  absl::StatusOr<std::unique_ptr<LaplacePrivacyLoss>> mechanism =
      LaplacePrivacyLoss::Create(/*parameter=*/2, /*sensitivity=*/4);
  ASSERT_OK(mechanism);
  EXPECT_THAT(mechanism.value()->GetDeltaForEpsilon(0.5),
              DoubleNear(0.52763345, kMaxError));
}

TEST(GaussianPrivacyLoss, InvalidStdDeviation) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(/*standard_deviation=*/-1, /*sensitivity=*/1);

  EXPECT_THAT(mechanism,
              StatusIs(absl::InvalidArgumentError("").code(),
                       HasSubstr("standard_deviation should be positive")));
}

TEST(GaussianPrivacyLoss, InvalidSensitivity) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(/*standard_deviation=*/1, /*sensitivity=*/-1);

  EXPECT_THAT(mechanism, StatusIs(absl::InvalidArgumentError("").code(),
                                  HasSubstr("sensitivity should be positive")));
}

TEST(GaussianPrivacyLoss, InvalidMassTruncation) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(
          /*standard_deviation=*/1,
          /*sensitivity=*/1,
          /*estimate_type=*/EstimateType::kPessimistic,
          /*log_mass_truncation_bound=*/1);

  EXPECT_THAT(mechanism, StatusIs(absl::InvalidArgumentError("").code(),
                                  HasSubstr("log_mass_truncation_bound cannot "
                                            "be positive")));
}

TEST(GaussianPrivacyLoss, PrivacyLoss) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(7, 14);
  ASSERT_OK(mechanism);
  EXPECT_DOUBLE_EQ(mechanism.value()->PrivacyLoss(21), -4);
}

TEST(GaussianPrivacyLoss, InversePrivacyLoss) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(7, 14);
  ASSERT_OK(mechanism);
  EXPECT_DOUBLE_EQ(mechanism.value()->InversePrivacyLoss(-4), 21);
}

TEST(GaussianPrivacyLoss, PrivacyLossTailPessimistic) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(
          /*standard_deviation=*/4,
          /*sensitivity=*/8,
          /*estimate_type=*/EstimateType::kPessimistic,
          /*log_mass_truncation_bound=*/-1.147874464449318);
  ASSERT_OK(mechanism);
  PrivacyLossTail result = mechanism.value()->PrivacyLossDistributionTail();
  EXPECT_DOUBLE_EQ(result.lower_x_truncation, -4);
  EXPECT_DOUBLE_EQ(result.upper_x_truncation, 4);
  EXPECT_THAT(result.probability_mass_function,
              UnorderedElementsAre(Pair(std::numeric_limits<double>::infinity(),
                                        DoubleNear(0.15865525, kMaxError)),
                                   Pair(DoubleNear(0, kMaxError),
                                        DoubleNear(0.15865525, kMaxError))));
}

TEST(GaussianPrivacyLoss, PrivacyLossTailOptimistic) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(
          /*standard_deviation=*/4,
          /*sensitivity=*/8,
          /*estimate_type=*/EstimateType::kOptimistic,
          /*log_mass_truncation_bound=*/-1.147874464449318);
  ASSERT_OK(mechanism);
  PrivacyLossTail result = mechanism.value()->PrivacyLossDistributionTail();
  EXPECT_DOUBLE_EQ(result.lower_x_truncation, -4);
  EXPECT_DOUBLE_EQ(result.upper_x_truncation, 4);
  EXPECT_THAT(result.probability_mass_function,
              UnorderedElementsAre(Pair(DoubleNear(4, kMaxError),
                                        DoubleNear(0.15865525, kMaxError))));
}

struct GaussianGetDeltaForEpsilonParam {
  double standard_deviation;
  double sensitivity;
  double epsilon;
  double expected_delta;
};

class GaussianGetDeltaForEpsilonTest
    : public testing::TestWithParam<GaussianGetDeltaForEpsilonParam> {};

INSTANTIATE_TEST_SUITE_P(
    GaussianSuite, GaussianGetDeltaForEpsilonTest,
    Values(GaussianGetDeltaForEpsilonParam{.standard_deviation = 2,
                                           .sensitivity = 6,
                                           .epsilon = 1,
                                           .expected_delta = 0.78760074},
           GaussianGetDeltaForEpsilonParam{.standard_deviation = 1,
                                           .sensitivity = 3,
                                           .epsilon = 1,
                                           .expected_delta = 0.78760074},
           GaussianGetDeltaForEpsilonParam{.standard_deviation = 1,
                                           .sensitivity = 1,
                                           .epsilon = 1,
                                           .expected_delta = 0.12693674},
           GaussianGetDeltaForEpsilonParam{.standard_deviation = 2,
                                           .sensitivity = 2,
                                           .epsilon = 1,
                                           .expected_delta = 0.12693674},
           GaussianGetDeltaForEpsilonParam{.standard_deviation = 5,
                                           .sensitivity = 5,
                                           .epsilon = 2,
                                           .expected_delta = 0.02092364},
           GaussianGetDeltaForEpsilonParam{.standard_deviation = 1,
                                           .sensitivity = 1,
                                           .epsilon = 2,
                                           .expected_delta = 0.02092364}));

TEST_P(GaussianGetDeltaForEpsilonTest, GetDeltaForEpsilon) {
  GaussianGetDeltaForEpsilonParam param = GetParam();
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(param.standard_deviation, param.sensitivity);

  ASSERT_OK(mechanism);
  EXPECT_THAT(mechanism.value()->GetDeltaForEpsilon(param.epsilon),
              DoubleNear(param.expected_delta, kMaxError));
}

TEST(GaussianPrivacyLoss, InvalidDelta) {
  EpsilonDelta epsilon_delta = {/*epsilon=*/1, /*delta=*/0};
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(epsilon_delta);
  EXPECT_THAT(mechanism,
              StatusIs(absl::InvalidArgumentError("").code(),
                       HasSubstr("delta should be positive for the Gaussian "
                                 "mechanism.")));
}

struct GaussianFromEpsilonDeltaParam {
  double epsilon;
  double delta;
  double expected_standard_deviation;
};

class GaussianFromEpsilonDeltaTest
    : public testing::TestWithParam<GaussianFromEpsilonDeltaParam> {};

INSTANTIATE_TEST_SUITE_P(
    GaussianSuite, GaussianFromEpsilonDeltaTest,
    Values(GaussianFromEpsilonDeltaParam{.epsilon = 1,
                                         .delta = 0.12693674,
                                         .expected_standard_deviation = 1},
           GaussianFromEpsilonDeltaParam{
               .epsilon = 16,
               .delta = 0.00001,
               .expected_standard_deviation = 0.34418}));

TEST_P(GaussianFromEpsilonDeltaTest, FromEpsilonDelta) {
  GaussianFromEpsilonDeltaParam param = GetParam();
  EpsilonDelta epsilon_delta = {param.epsilon, param.delta};
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> mechanism =
      GaussianPrivacyLoss::Create(epsilon_delta);
  ASSERT_OK(mechanism);
  ASSERT_NEAR(mechanism.value()->StandardDeviation(),
              param.expected_standard_deviation, kMaxError);
}

TEST(DiscreteLaplacePrivacyLoss, InvalidParameter) {
  absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> mechanism =
      DiscreteLaplacePrivacyLoss::Create(/*parameter=*/-1, /*sensitivity=*/1);

  EXPECT_THAT(mechanism, StatusIs(absl::InvalidArgumentError("").code(),
                                  HasSubstr("parameter should be positive")));
}

TEST(DiscreteLaplacePrivacyLoss, InvalidSensitivity) {
  absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> mechanism =
      DiscreteLaplacePrivacyLoss::Create(/*parameter=*/1, /*sensitivity=*/-1);

  EXPECT_THAT(mechanism, StatusIs(absl::InvalidArgumentError("").code(),
                                  HasSubstr("sensitivity should be positive")));
}

struct DiscreteLaplacePrivacyLossParam {
  double parameter;
  double sensitivity;
  double x;
  double expected_privacy_loss;
};

class DiscreteLaplacePrivacyLossTest
    : public testing::TestWithParam<DiscreteLaplacePrivacyLossParam> {};

INSTANTIATE_TEST_SUITE_P(
    DiscreteLaplacePrivacyLossSuite, DiscreteLaplacePrivacyLossTest,
    Values(DiscreteLaplacePrivacyLossParam{.parameter = 1,
                                           .sensitivity = 1,
                                           .x = 0,
                                           .expected_privacy_loss = 1},
           DiscreteLaplacePrivacyLossParam{.parameter = 1,
                                           .sensitivity = 1,
                                           .x = 1,
                                           .expected_privacy_loss = -1},
           DiscreteLaplacePrivacyLossParam{.parameter = 0.3,
                                           .sensitivity = 2,
                                           .x = 0,
                                           .expected_privacy_loss = 0.6},
           DiscreteLaplacePrivacyLossParam{.parameter = 0.3,
                                           .sensitivity = 2,
                                           .x = 1,
                                           .expected_privacy_loss = 0},
           DiscreteLaplacePrivacyLossParam{.parameter = 0.3,
                                           .sensitivity = 2,
                                           .x = 2,
                                           .expected_privacy_loss = -0.6}));

TEST_P(DiscreteLaplacePrivacyLossTest, PrivacyLoss) {
  DiscreteLaplacePrivacyLossParam param = GetParam();
  absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> mechanism =
      DiscreteLaplacePrivacyLoss::Create(param.parameter, param.sensitivity);

  ASSERT_OK(mechanism);
  EXPECT_THAT(mechanism.value()->PrivacyLoss(param.x),
              DoubleNear(param.expected_privacy_loss, kMaxError));
}

struct DiscreteLaplaceInversePrivacyLossParam {
  double parameter;
  double sensitivity;
  double privacy_loss;
  double expected_x;
};

class DiscreteLaplaceInversePrivacyLossTest
    : public testing::TestWithParam<DiscreteLaplaceInversePrivacyLossParam> {};

INSTANTIATE_TEST_SUITE_P(
    DiscreteLaplacePrivacyLossSuite, DiscreteLaplaceInversePrivacyLossTest,
    Values(
        DiscreteLaplaceInversePrivacyLossParam{
            .parameter = 1,
            .sensitivity = 1,
            .privacy_loss = 1.1,
            .expected_x = -std::numeric_limits<double>::infinity()},
        DiscreteLaplaceInversePrivacyLossParam{.parameter = 1,
                                               .sensitivity = 1,
                                               .privacy_loss = 0.9,
                                               .expected_x = 0},
        DiscreteLaplaceInversePrivacyLossParam{
            .parameter = 1,
            .sensitivity = 1,
            .privacy_loss = -1,
            .expected_x = std::numeric_limits<double>::infinity()},
        DiscreteLaplaceInversePrivacyLossParam{
            .parameter = 0.3,
            .sensitivity = 2,
            .privacy_loss = 0.7,
            .expected_x = -std::numeric_limits<double>::infinity()},
        DiscreteLaplaceInversePrivacyLossParam{.parameter = 0.3,
                                               .sensitivity = 2,
                                               .privacy_loss = 0.2,
                                               .expected_x = 0},
        DiscreteLaplaceInversePrivacyLossParam{.parameter = 0.3,
                                               .sensitivity = 2,
                                               .privacy_loss = 0,
                                               .expected_x = 1},
        DiscreteLaplaceInversePrivacyLossParam{
            .parameter = 0.3,
            .sensitivity = 2,
            .privacy_loss = -0.6,
            .expected_x = std::numeric_limits<double>::infinity()}));

TEST_P(DiscreteLaplaceInversePrivacyLossTest, InversePrivacyLoss) {
  DiscreteLaplaceInversePrivacyLossParam param = GetParam();
  absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> mechanism =
      DiscreteLaplacePrivacyLoss::Create(param.parameter, param.sensitivity);

  ASSERT_OK(mechanism);
  EXPECT_THAT(mechanism.value()->InversePrivacyLoss(param.privacy_loss),
              DoubleNear(param.expected_x, kMaxError));
}

TEST(DiscreteLaplacePrivacyLoss, PrivacyLossTail) {
  absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> mechanism =
      DiscreteLaplacePrivacyLoss::Create(
          /*parameter=*/0.3,
          /*sensitivity=*/2);
  ASSERT_OK(mechanism);
  PrivacyLossTail result = mechanism.value()->PrivacyLossDistributionTail();
  EXPECT_DOUBLE_EQ(result.lower_x_truncation, 1);
  EXPECT_DOUBLE_EQ(result.upper_x_truncation, 1);
  EXPECT_THAT(result.probability_mass_function,
              UnorderedElementsAre(Pair(DoubleNear(0.6, kMaxError),
                                        DoubleNear(0.57444252, kMaxError)),
                                   Pair(DoubleNear(-0.6, kMaxError),
                                        DoubleNear(0.31526074, kMaxError))));
}

TEST(DiscreteLaplacePrivacyLoss, CreateFromEpsilonDelta) {
  EpsilonDelta epsilon_delta = {3, 0.01};
  absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> mechanism =
      DiscreteLaplacePrivacyLoss::Create(epsilon_delta, /*sensitivity=*/2);
  ASSERT_OK(mechanism);
  EXPECT_NEAR(mechanism.value()->Parameter(), 1.5, kMaxError);
}

struct DiscreteLaplaceGetDeltaForEpsilonParam {
  double parameter;
  double sensitivity;
  double epsilon;
  double expected_delta;
};

class DiscreteLaplaceGetDeltaForEpsilonTest
    : public testing::TestWithParam<DiscreteLaplaceGetDeltaForEpsilonParam> {};

INSTANTIATE_TEST_SUITE_P(
    DiscreteLaplaceSuite, DiscreteLaplaceGetDeltaForEpsilonTest,
    Values(DiscreteLaplaceGetDeltaForEpsilonParam{.parameter = 1,
                                                  .sensitivity = 1,
                                                  .epsilon = 1,
                                                  .expected_delta = 0},
           DiscreteLaplaceGetDeltaForEpsilonParam{.parameter = 0.333333,
                                                  .sensitivity = 3,
                                                  .epsilon = 1,
                                                  .expected_delta = 0},
           DiscreteLaplaceGetDeltaForEpsilonParam{.parameter = 0.5,
                                                  .sensitivity = 4,
                                                  .epsilon = 2,
                                                  .expected_delta = 0},
           DiscreteLaplaceGetDeltaForEpsilonParam{.parameter = 0.5,
                                                  .sensitivity = 4,
                                                  .epsilon = 0.5,
                                                  .expected_delta = 0.54202002},
           DiscreteLaplaceGetDeltaForEpsilonParam{.parameter = 0.5,
                                                  .sensitivity = 4,
                                                  .epsilon = 1,
                                                  .expected_delta = 0.39346934},
           DiscreteLaplaceGetDeltaForEpsilonParam{
               .parameter = 0.5,
               .sensitivity = 4,
               .epsilon = -0.5,
               .expected_delta = 0.72222110}));

TEST_P(DiscreteLaplaceGetDeltaForEpsilonTest, GetDeltaForEpsilon) {
  DiscreteLaplaceGetDeltaForEpsilonParam param = GetParam();
  absl::StatusOr<std::unique_ptr<DiscreteLaplacePrivacyLoss>> mechanism =
      DiscreteLaplacePrivacyLoss::Create(param.parameter, param.sensitivity);

  ASSERT_OK(mechanism);
  EXPECT_THAT(mechanism.value()->GetDeltaForEpsilon(param.epsilon),
              DoubleNear(param.expected_delta, kMaxError));
}

TEST(DiscreteGaussianPrivacyLoss, InvalidSigma) {
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(/*sigma=*/-1, /*sensitivity=*/1);

  EXPECT_THAT(mechanism, StatusIs(absl::InvalidArgumentError("").code(),
                                  HasSubstr("sigma should be positive")));
}

TEST(DiscreteGaussianPrivacyLoss, InvalidSensitivity) {
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(/*sigma=*/1, /*sensitivity=*/-1);

  EXPECT_THAT(mechanism, StatusIs(absl::InvalidArgumentError("").code(),
                                  HasSubstr("sensitivity should be positive")));
}

struct DiscreteGaussianPrivacyLossParam {
  double sigma;
  int sensitivity;
  double x;
  double expected_privacy_loss;
};

class DiscreteGaussianPrivacyLossTest
    : public testing::TestWithParam<DiscreteGaussianPrivacyLossParam> {};

INSTANTIATE_TEST_SUITE_P(
    DiscreteGaussianPrivacyLossSuite, DiscreteGaussianPrivacyLossTest,
    Values(
        DiscreteGaussianPrivacyLossParam{.sigma = 1,
                                         .sensitivity = 1,
                                         .x = 5,
                                         .expected_privacy_loss = -4.5},
        DiscreteGaussianPrivacyLossParam{.sigma = 1,
                                         .sensitivity = 1,
                                         .x = -3,
                                         .expected_privacy_loss = 3.5},
        DiscreteGaussianPrivacyLossParam{
            .sigma = 1, .sensitivity = 2, .x = 3, .expected_privacy_loss = -4},
        DiscreteGaussianPrivacyLossParam{.sigma = 4,
                                         .sensitivity = 4,
                                         .x = 20,
                                         .expected_privacy_loss = -4.5},
        DiscreteGaussianPrivacyLossParam{.sigma = 5,
                                         .sensitivity = 5,
                                         .x = -15,
                                         .expected_privacy_loss = 3.5},
        DiscreteGaussianPrivacyLossParam{.sigma = 7,
                                         .sensitivity = 14,
                                         .x = 21,
                                         .expected_privacy_loss = -4},
        DiscreteGaussianPrivacyLossParam{
            .sigma = 1,
            .sensitivity = 1,
            .x = -12,
            .expected_privacy_loss = std::numeric_limits<double>::infinity()}));

TEST_P(DiscreteGaussianPrivacyLossTest, PrivacyLoss) {
  DiscreteGaussianPrivacyLossParam param = GetParam();
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(param.sigma, param.sensitivity);

  ASSERT_OK(mechanism);
  EXPECT_THAT(mechanism.value()->PrivacyLoss(param.x),
              DoubleNear(param.expected_privacy_loss, kMaxError));
}

struct DiscreteGaussianInversePrivacyLossParam {
  double sigma;
  int sensitivity;
  double privacy_loss;
  double expected_x;
};

class DiscreteGaussianInversePrivacyLossTest
    : public testing::TestWithParam<DiscreteGaussianInversePrivacyLossParam> {};

INSTANTIATE_TEST_SUITE_P(
    DiscreteGaussianInversePrivacyLossSuite,
    DiscreteGaussianInversePrivacyLossTest,
    Values(DiscreteGaussianInversePrivacyLossParam{.sigma = 1,
                                                   .sensitivity = 1,
                                                   .privacy_loss = -4.5,
                                                   .expected_x = 5},
           DiscreteGaussianInversePrivacyLossParam{.sigma = 1,
                                                   .sensitivity = 1,
                                                   .privacy_loss = 3.5,
                                                   .expected_x = -3},
           DiscreteGaussianInversePrivacyLossParam{.sigma = 1,
                                                   .sensitivity = 2,
                                                   .privacy_loss = -4,
                                                   .expected_x = 3},
           DiscreteGaussianInversePrivacyLossParam{.sigma = 4,
                                                   .sensitivity = 4,
                                                   .privacy_loss = -4.51,
                                                   .expected_x = 20},
           DiscreteGaussianInversePrivacyLossParam{.sigma = 5,
                                                   .sensitivity = 5,
                                                   .privacy_loss = 3.49,
                                                   .expected_x = -15},
           DiscreteGaussianInversePrivacyLossParam{.sigma = 7,
                                                   .sensitivity = 14,
                                                   .privacy_loss = -4,
                                                   .expected_x = 21}));

TEST_P(DiscreteGaussianInversePrivacyLossTest, InversePrivacyLoss) {
  DiscreteGaussianInversePrivacyLossParam param = GetParam();
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(param.sigma, param.sensitivity);

  ASSERT_OK(mechanism);
  EXPECT_THAT(mechanism.value()->InversePrivacyLoss(param.privacy_loss),
              DoubleNear(param.expected_x, kMaxError));
}

TEST(DiscreteGaussianPrivacyLossTest, Truncation) {
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(
          /*sigma=*/3,
          /*sensitivity=*/1);
  ASSERT_OK(mechanism);
  EXPECT_EQ(35, mechanism.value()->TruncationBound());
}

TEST(DiscreteGaussianPrivacyLossTest, PrivacyLossTail) {
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(
          /*sigma=*/1,
          /*sensitivity=*/2,
          /*truncation_bound=*/2);
  ASSERT_OK(mechanism);
  PrivacyLossTail result = mechanism.value()->PrivacyLossDistributionTail();
  EXPECT_DOUBLE_EQ(result.lower_x_truncation, 0);
  EXPECT_DOUBLE_EQ(result.upper_x_truncation, 2);
  EXPECT_THAT(
      result.probability_mass_function,
      UnorderedElementsAre(
          Pair(DoubleNear(std::numeric_limits<double>::infinity(), kMaxError),
               DoubleNear(0.29869003, kMaxError))));
}

TEST(DiscreteGaussianPrivacyLossTest, NoiseCdf) {
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(
          /*sigma=*/3,
          /*sensitivity=*/2,
          /*truncation_bound=*/2);
  ASSERT_OK(mechanism);

  absl::flat_hash_map<double, double> expected_noise_cdf = {
      {-2.1, 0},       {-2, 0.17820326}, {-1, 0.38872553}, {0, 0.61127447},
      {1, 0.82179674}, {2, 1},           {2.7, 1}};
  for (auto [x, expected_cdf_value] : expected_noise_cdf) {
    double cdf_value = mechanism.value()->NoiseCdf(x);
    EXPECT_THAT(cdf_value, DoubleNear(cdf_value, kMaxError));
  }
}

TEST(DiscreteGaussianPrivacyLossTest, StandardDeviation) {
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(
          /*sigma=*/3,
          /*sensitivity=*/2,
          /*truncation_bound=*/2);
  ASSERT_OK(mechanism);

  double std = mechanism.value()->StandardDeviation();
  EXPECT_THAT(std, DoubleNear(1.3589226, kMaxError));
}

TEST(DiscreteGaussianPrivacyLoss, InvalidDelta) {
  EpsilonDelta epsilon_delta = {/*epsilon=*/1, /*delta=*/0};
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism =
      DiscreteGaussianPrivacyLoss::Create(epsilon_delta,
                                          /*sensitivity=*/1);
  EXPECT_THAT(mechanism,
              StatusIs(absl::InvalidArgumentError("").code(),
                       HasSubstr("delta should be positive for the discrete "
                                 "Gaussian mechanism.")));
}

struct DiscreteGaussianFromEpsilonDeltaParam {
  int sensitivity;
  double epsilon;
  double delta;
  double expected_sigma;
};

class DiscreteGaussianFromEpsilonDeltaTest
    : public testing::TestWithParam<DiscreteGaussianFromEpsilonDeltaParam> {};

INSTANTIATE_TEST_SUITE_P(
    DiscreteGaussianFromEpsilonDeltaSuite, DiscreteGaussianFromEpsilonDeltaTest,
    Values(DiscreteGaussianFromEpsilonDeltaParam{.sensitivity = 1,
                                                 .epsilon = 1,
                                                 .delta = 0.12693674,
                                                 .expected_sigma = 1.0407},
           DiscreteGaussianFromEpsilonDeltaParam{.sensitivity = 1,
                                                 .epsilon = 16,
                                                 .delta = 1e-5,
                                                 .expected_sigma = 0.3062},
           DiscreteGaussianFromEpsilonDeltaParam{.sensitivity = 3,
                                                 .epsilon = 1,
                                                 .delta = 0.78760074,
                                                 .expected_sigma = 0.9928}));

TEST_P(DiscreteGaussianFromEpsilonDeltaTest, CreateFromEpsilonDelta) {
  DiscreteGaussianFromEpsilonDeltaParam param = GetParam();
  absl::StatusOr<std::unique_ptr<DiscreteGaussianPrivacyLoss>> mechanism;
  mechanism = DiscreteGaussianPrivacyLoss::Create(
      EpsilonDelta{param.epsilon, param.delta}, param.sensitivity);

  ASSERT_OK(mechanism);
  double sigma = (*mechanism)->Sigma();
  EXPECT_THAT(sigma, DoubleNear(param.expected_sigma, kMaxError));
}
}  // namespace
}  // namespace accounting
}  // namespace differential_privacy
