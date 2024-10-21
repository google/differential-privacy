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

#include "accounting/privacy_loss_distribution.h"

#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "accounting/common/common.h"
#include "accounting/common/test_util.h"
#include "proto/accounting/privacy-loss-distribution.pb.h"
#include "base/testing/status_matchers.h"

namespace differential_privacy {
namespace accounting {

// Test peer is required to set private instance variables in order to test
// behaviors independently.
class PrivacyLossDistributionTestPeer {
 public:
  static std::unique_ptr<PrivacyLossDistribution> Create(
      const ProbabilityMassFunction& probability_mass_function,
      double infinity_mass = 0, double discretization_interval = 1e-4,
      EstimateType estimate_type = EstimateType::kPessimistic) {
    return absl::WrapUnique(
        new PrivacyLossDistribution(discretization_interval, infinity_mass,
                                    probability_mass_function, estimate_type));
  }
};

namespace {
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::Matcher;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::Values;
using ::differential_privacy::base::testing::StatusIs;

constexpr double kMaxError = 1e-4f;

Matcher<ProbabilityMassFunction> PMFIsNear(
    const ProbabilityMassFunction& expected) {
  return differential_privacy::accounting::PMFIsNear(expected, kMaxError);
}

TEST(PrivacyLossDistributionTest, CreateBasic) {
  ProbabilityMassFunction pmf_lo = {{1, 0.5}, {2, 0.5}};
  ProbabilityMassFunction pmf_hi = {{1, 0.6}, {2, 0.4}};

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::Create(pmf_lo, pmf_hi);

  EXPECT_THAT(pld->Pmf(), UnorderedElementsAre(
                              FieldsAre(Eq(-2231), DoubleNear(0.4, kMaxError)),
                              FieldsAre(Eq(1824), DoubleNear(0.6, kMaxError))));

  EXPECT_EQ(pld->InfinityMass(), 0);
  EXPECT_EQ(pld->DiscretizationInterval(), 0.0001);
}

TEST(PrivacyLossDistributionTest, CreateInfinityMass) {
  ProbabilityMassFunction pmf_lo = {{1, 0.5}, {3, 0.5}};
  ProbabilityMassFunction pmf_hi = {{1, 0.6}, {2, 0.4}};

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::Create(pmf_lo, pmf_hi);

  EXPECT_THAT(pld->Pmf(), UnorderedElementsAre(Pair(1824, 0.6)));
  EXPECT_NEAR(pld->InfinityMass(), 0.4, kMaxError);
}

TEST(PrivacyLossDistributionTest, CreatePessimistic) {
  ProbabilityMassFunction pmf_lo = {{1, 0.5}, {2, 0.5}};
  ProbabilityMassFunction pmf_hi = {{1, 0.6}, {2, 0.4}};

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::Create(pmf_lo, pmf_hi,
                                      EstimateType::kPessimistic, 1e-4, -0.55);

  EXPECT_THAT(pld->Pmf(), UnorderedElementsAre(Pair(1824, 0.6)));
  EXPECT_NEAR(pld->InfinityMass(), 0.40, kMaxError);
}

TEST(PrivacyLossDistributionTest, CreateOptimistic) {
  ProbabilityMassFunction pmf_lo = {{1, 0.5}, {2, 0.5}};
  ProbabilityMassFunction pmf_hi = {{1, 0.6}, {2, 0.4}};

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::Create(pmf_lo, pmf_hi, EstimateType::kOptimistic,
                                      1e-4, -0.55);

  ProbabilityMassFunction expected_pmf = {{1823, 0.6}};

  EXPECT_THAT(pld->Pmf(), UnorderedElementsAre(
                              FieldsAre(Eq(1823), DoubleNear(0.6, kMaxError))));
  EXPECT_NEAR(pld->InfinityMass(), 0.0, kMaxError);
}

TEST(PrivacyLossDistributionTest, CreateMismatch) {
  ProbabilityMassFunction pmf_lo = {{1, 0.5}, {2, 0.5}};
  ProbabilityMassFunction pmf_hi = {{4, 0.6}};

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::Create(pmf_lo, pmf_hi,
                                      EstimateType::kPessimistic);

  EXPECT_EQ(pld->Pmf().size(), 0);
}

TEST(PrivacyLossDistributionTest, CreateIdentity) {
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateIdentity(1e-3);

  ProbabilityMassFunction expected_pmf = {{0, 1}};

  EXPECT_THAT(pld->Pmf(),
              UnorderedElementsAre(FieldsAre(Eq(0), DoubleNear(1, kMaxError))));
  EXPECT_NEAR(pld->InfinityMass(), 0.0, kMaxError);
  EXPECT_NEAR(pld->DiscretizationInterval(), 1e-3, kMaxError);
}

TEST(PrivacyLossDistributionTest, HockeyStickDivergence) {
  // 1824: is ceiling(value_discretization_interval * ln(1.2))
  // ceiling(1000 * 0.1823).
  // 1.2: so that hockey stick divergence at 0 is
  // (1 - e^{-ln(1.2)}) * 0.6 = 0.1
  ProbabilityMassFunction pmf = {{1824, 0.6}, {-2231, 0.4}};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(pmf);

  double value = pld->GetDeltaForEpsilon(0);

  EXPECT_NEAR(value, 0.1, kMaxError);
  EXPECT_GE(value, 1e-5);
}

TEST(PrivacyLossDistributionTest, Compose) {
  ProbabilityMassFunction pmf = {{0, 1}, {1, 4}, {2, 2}, {3, 5}};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(pmf,
                                              /*infinity_mass=*/0.3,
                                              /*discretization_interval=*/1e-4);

  ProbabilityMassFunction pmf_other = {{0, 3}, {1, 4}, {2, 1}};
  std::unique_ptr<PrivacyLossDistribution> pld_other =
      PrivacyLossDistributionTestPeer::Create(pmf_other,
                                              /*infinity_mass=*/0.2,
                                              /*discretization_interval=*/1e-4);

  EXPECT_OK(pld->Compose(*pld_other));
  EXPECT_THAT(pld->InfinityMass(), DoubleNear(0.44, kMaxError));
  EXPECT_FALSE(pld->Pmf().empty());
}

TEST(PrivacyLossDistributionTest, GetDeltaForEpsilonForComposedPLD) {
  ProbabilityMassFunction pmf = {{0, 0.1}, {1, 0.7}, {2, 0.1}};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(pmf,
                                              /*infinity_mass=*/0.1,
                                              /*discretization_interval=*/0.4);

  ProbabilityMassFunction pmf_other = {{1, 0.1}, {2, 0.6}, {3, 0.25}};
  std::unique_ptr<PrivacyLossDistribution> pld_other =
      PrivacyLossDistributionTestPeer::Create(pmf_other,
                                              /*infinity_mass=*/0.05,
                                              /*discretization_interval=*/0.4);

  absl::StatusOr<double> delta =
      pld->GetDeltaForEpsilonForComposedPLD(*pld_other, /*epsilon=*/1.1);
  ASSERT_OK(delta);
  EXPECT_THAT(*delta, DoubleNear(0.2956, kMaxError));
}

TEST(PrivacyLossDistributionTest, ComposeTruncation) {
  ProbabilityMassFunction pmf = {{0, 0.1}, {1, 0.7}, {2, 0.1}};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(pmf,
                                              /*infinity_mass=*/0.1,
                                              /*discretization_interval=*/1);

  ProbabilityMassFunction pmf_other = {{0, 0.1}, {1, 0.6}, {2, 0.2}};
  std::unique_ptr<PrivacyLossDistribution> pld_other =
      PrivacyLossDistributionTestPeer::Create(pmf_other,
                                              /*infinity_mass=*/0.1,
                                              /*discretization_interval=*/1);

  EXPECT_OK(pld->Compose(*pld_other, 0.021));
  EXPECT_THAT(pld->InfinityMass(), DoubleNear(0.211, kMaxError));
  EXPECT_THAT(pld->Pmf(), UnorderedElementsAre(
                              FieldsAre(Eq(1), DoubleNear(0.13, kMaxError)),
                              FieldsAre(Eq(2), DoubleNear(0.45, kMaxError)),
                              FieldsAre(Eq(3), DoubleNear(0.20, kMaxError)),
                              FieldsAre(Eq(4), DoubleNear(0.02, kMaxError))));
}

TEST(PrivacyLossDistributionTest, ComposeNumTimes) {
  ProbabilityMassFunction pmf = {{1, 0.6}, {2, 0.4}};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(pmf,
                                              /*infinity_mass=*/0.3,
                                              /*discretization_interval=*/1e-4);

  constexpr int num_times = 2;
  pld->Compose(num_times);

  EXPECT_THAT(pld->InfinityMass(), DoubleNear(0.51, kMaxError));
  EXPECT_THAT(pld->Pmf(), UnorderedElementsAre(
                              FieldsAre(Eq(2), DoubleNear(0.36, kMaxError)),
                              FieldsAre(Eq(3), DoubleNear(0.48, kMaxError)),
                              FieldsAre(Eq(4), DoubleNear(0.16, kMaxError))));
}

TEST(PrivacyLossDistributionTest, ComposeNumTimesForEpsilonZero) {
  EpsilonDelta epsilon_delta = {0, 0};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForPrivacyParameters(epsilon_delta);

  constexpr int num_times = 60;
  pld->Compose(num_times);

  EXPECT_THAT(pld->InfinityMass(), DoubleNear(0, kMaxError));
  EXPECT_THAT(pld->Pmf(),
              UnorderedElementsAre(FieldsAre(Eq(0), DoubleNear(1, kMaxError))));
}

TEST(PrivacyLossDistributionTest, ComposeNumTimesForEpsilonZeroDeltaNonZero) {
  EpsilonDelta epsilon_delta = {0, 1e-4};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForPrivacyParameters(epsilon_delta);

  constexpr int num_times = 70;
  pld->Compose(num_times);

  EXPECT_THAT(pld->InfinityMass(), DoubleNear(0.007, kMaxError));
  EXPECT_THAT(pld->Pmf(), UnorderedElementsAre(
                              FieldsAre(Eq(0), DoubleNear(0.993, kMaxError))));
}

TEST(PrivacyLossDistributionTest, ComposeNumTimesTruncation) {
  // Use Gaussian mechanism because it has closed form formula even afer
  // composition. For the setting of parameter below, the privacy loss after
  // composition should be the same as privacy loss with standard_deviation =
  // sensitivity.
  int standard_deviation = 20;
  int num_composition = standard_deviation * standard_deviation;
  absl::StatusOr<std::unique_ptr<AdditiveNoisePrivacyLoss>> noise_privacy_loss =
      GaussianPrivacyLoss::Create(standard_deviation, /*sensitivity=*/1);
  ASSERT_OK(noise_privacy_loss);

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForAdditiveNoise(
          *noise_privacy_loss.value(), EstimateType::kPessimistic,
          /*discretization_interval=*/1e-5);

  pld->Compose(num_composition, /*tail_mass_truncation=*/1e-7);
  EXPECT_NEAR(0.00153, pld->GetDeltaForEpsilon(3), kMaxError);
}

TEST(PrivacyLossDistributionTest,
     ComposeNumTimesTruncationAccountForTruncatedMass) {
  int num_composition = 2;
  double epsilon_initial = 1;

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(
          {{epsilon_initial, 0.7}, {-epsilon_initial, 0.3}});

  pld->Compose(num_composition, /*tail_mass_truncation=*/0.5);
  EXPECT_NEAR(0.5, pld->GetDeltaForEpsilon(num_composition * epsilon_initial),
              kMaxError);
}

TEST(PrivacyLossDistributionTest, ComposeNumTimesNoTruncationOptimistic) {
  int num_composition = 2;
  double epsilon_initial = 1;

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(
          {{epsilon_initial, 0.7}, {-epsilon_initial, 0.3}},
          /*infinity_mass=*/0,
          /*discretization_interval=*/epsilon_initial,
          /*estimate_type=*/EstimateType::kOptimistic);

  pld->Compose(num_composition, /*tail_mass_truncation=*/0.5);
  EXPECT_NEAR(0, pld->GetDeltaForEpsilon(num_composition * epsilon_initial),
              kMaxError);
}

TEST(PrivacyLossDistributionTest,
     ComposeErrorDifferentDiscretizationIntervals) {
  ProbabilityMassFunction pmf = {};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(pmf, 0.3, 1e-4);

  std::unique_ptr<PrivacyLossDistribution> pld_other =
      PrivacyLossDistributionTestPeer::Create(pmf, 0.3, 2e-4);

  std::string error_msg = "discretization interval";
  EXPECT_THAT(
      pld->ValidateComposition(*pld_other),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(error_msg)));
  EXPECT_THAT(
      pld->Compose(*pld_other),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(error_msg)));
  EXPECT_THAT(
      pld->GetDeltaForEpsilonForComposedPLD(*pld_other, /*epsilon=*/1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(error_msg)));
}

TEST(PrivacyLossDistributionTest, ComposeErrorDifferentEstimateTypes) {
  ProbabilityMassFunction pmf;
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(
          pmf, /*infinity_mass=*/0.3, /*discretization_interval=*/1e-4,
          /*estimate_type=*/EstimateType::kPessimistic);

  std::unique_ptr<PrivacyLossDistribution> pld_other =
      PrivacyLossDistributionTestPeer::Create(
          pmf, /*infinity_mass=*/0.3, /*discretization_interval=*/1e-4,
          /*estimate_type=*/EstimateType::kOptimistic);

  std::string error_msg = "estimate type";
  EXPECT_THAT(
      pld->ValidateComposition(*pld_other),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(error_msg)));
  EXPECT_THAT(
      pld->Compose(*pld_other),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(error_msg)));
  EXPECT_THAT(
      pld->GetDeltaForEpsilonForComposedPLD(*pld_other, /*epsilon=*/1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(error_msg)));
}

struct GetEpsilonFromDeltaParam {
  double discretization_interval;
  double infinity_mass;
  ProbabilityMassFunction pmf;
  double delta;
  double expected_epsilon;
};

class GetEpsilonFromDeltaTest
    : public testing::TestWithParam<GetEpsilonFromDeltaParam> {};

INSTANTIATE_TEST_SUITE_P(
    PrivacyLossDistributionSuite, GetEpsilonFromDeltaTest,
    Values(GetEpsilonFromDeltaParam{.discretization_interval = 0.5,
                                    .infinity_mass = 0.1,
                                    .pmf = {{4, 0.2}, {2, 0.7}},
                                    .delta = 0.5,
                                    .expected_epsilon = 0.56358432},
           GetEpsilonFromDeltaParam{.discretization_interval = 0.5,
                                    .infinity_mass = 0.1,
                                    .pmf = {{4, 0.2}, {2, 0.7}},
                                    .delta = 0.2,
                                    .expected_epsilon = 1.30685282},
           GetEpsilonFromDeltaParam{.discretization_interval = 1,
                                    .infinity_mass = 0.1,
                                    .pmf = {{1, 0.2}, {-1, 0.7}},
                                    .delta = 0.4,
                                    .expected_epsilon = 0},
           GetEpsilonFromDeltaParam{.discretization_interval = 1,
                                    .infinity_mass = 0,
                                    .pmf = {{-1, 0.1}},
                                    .delta = 0,
                                    .expected_epsilon = 0},
           // Test for the case delta = infinity_mass
           GetEpsilonFromDeltaParam{.discretization_interval = 1,
                                    .infinity_mass = 0.1,
                                    .pmf = {{2, 0.1}},
                                    .delta = 0.1,
                                    .expected_epsilon = 2},
           // Test resilience against overflow
           GetEpsilonFromDeltaParam{.discretization_interval = 1,
                                    .infinity_mass = 0,
                                    .pmf = {{5000, 1}},
                                    .delta = 0.1,
                                    .expected_epsilon = 5000},
           GetEpsilonFromDeltaParam{
               .discretization_interval = 1,
               .infinity_mass = 0.1,
               .pmf = {{5000, 0.2}, {4000, 0.1}, {3000, 0.7}},
               .delta = 0.4,
               .expected_epsilon = 4000}));

TEST_P(GetEpsilonFromDeltaTest, EpsilonFromDeltaBasic) {
  GetEpsilonFromDeltaParam param = GetParam();
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(param.pmf, param.infinity_mass,
                                              param.discretization_interval);
  EXPECT_NEAR(pld->GetEpsilonForDelta(param.delta), param.expected_epsilon,
              kMaxError);
}

TEST(GetEpsilonFromDeltaTest, EpsilonFromDeltaInfinity) {
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create({{1, 0.2}, {-1, 0.7}},
                                              /*infinity_mass=*/0.5,
                                              /*discretization_interval=*/1);
  EXPECT_EQ(pld->GetEpsilonForDelta(0.4),
            std::numeric_limits<double>::infinity());
}

struct CreateParam {
  double discretization_interval;
  ProbabilityMassFunction expected_pmf;
};

class DiscretizationTest : public testing::TestWithParam<CreateParam> {};

INSTANTIATE_TEST_SUITE_P(PrivacyLossDistributionSuite, DiscretizationTest,
                         Values(CreateParam{.discretization_interval = 0.5,
                                            .expected_pmf = {{3, 0.12447741},
                                                             {2, 0.19146246},
                                                             {1, 0.19146246},
                                                             {0, 0.30853754}}},
                                CreateParam{
                                    .discretization_interval = 0.3,
                                    .expected_pmf = {{5, 0.05790353},
                                                     {4, 0.10261461},
                                                     {3, 0.11559390},
                                                     {2, 0.11908755},
                                                     {1, 0.11220275},
                                                     {0, 0.09668214},
                                                     {-1, 0.21185540}}}));

TEST_P(DiscretizationTest, Gaussian) {
  CreateParam param = GetParam();
  absl::StatusOr<std::unique_ptr<AdditiveNoisePrivacyLoss>> noise_privacy_loss =
      GaussianPrivacyLoss::Create(
          /*standard_deviation=*/1,
          /*sensitivity=*/1,
          /*estimate_type=*/EstimateType::kPessimistic,
          /*log_mass_truncation_bound=*/-0.999345626001393);
  ASSERT_OK(noise_privacy_loss);

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForAdditiveNoise(
          *noise_privacy_loss.value(), EstimateType::kPessimistic,
          param.discretization_interval);

  EXPECT_THAT(pld->InfinityMass(), DoubleNear(0.184060, kMaxError));

  EXPECT_THAT(pld->Pmf(), PMFIsNear(param.expected_pmf));
}

TEST(PrivacyLossDistributionTest, DivergenceFromMechansim) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> noise_privacy_loss =
      GaussianPrivacyLoss::Create(
          /*standard_deviation=*/1,
          /*sensitivity=*/1);
  ASSERT_OK(noise_privacy_loss);

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForAdditiveNoise(
          *noise_privacy_loss.value(), EstimateType::kPessimistic,
          /*discretization_interval=*/1e-4);

  constexpr double epsilon = 1;
  // Delta for the noise privacy loss of the mechanism
  double delta_mechanism =
      noise_privacy_loss.value()->GetDeltaForEpsilon(epsilon);

  // Delta for the PMF discretized from that noise privacy loss
  double delta_pmf = pld->GetDeltaForEpsilon(epsilon);

  EXPECT_NEAR(delta_mechanism, delta_pmf, kMaxError);
}

TEST(PrivacyLossDistributionTest, GaussianOptimistic) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> noise_privacy_loss =
      GaussianPrivacyLoss::Create(
          /*standard_deviation=*/1,
          /*sensitivity=*/2,
          /*estimate_type=*/EstimateType::kOptimistic,
          /*log_mass_truncation_bound=*/-0.999345);
  ASSERT_OK(noise_privacy_loss);

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForAdditiveNoise(
          *noise_privacy_loss.value(), EstimateType::kOptimistic,
          /*discretization_interval=*/1);

  ProbabilityMassFunction expected_pmf = {
      {0, 0.124477}, {1, 0.191462}, {2, 0.191462}, {3, 0.308537}};

  EXPECT_THAT(
      pld->Pmf(),
      UnorderedElementsAre(FieldsAre(Eq(0), DoubleNear(0.124477, kMaxError)),
                           FieldsAre(Eq(1), DoubleNear(0.191462, kMaxError)),
                           FieldsAre(Eq(2), DoubleNear(0.191462, kMaxError)),
                           FieldsAre(Eq(3), DoubleNear(0.308537, kMaxError))));
  EXPECT_NEAR(pld->InfinityMass(), 0, kMaxError);
}

struct RandomizedResponseParam {
  double noise_parameter;
  double num_buckets;
  ProbabilityMassFunction expected_pmf;
  EstimateType estimate_type = EstimateType::kPessimistic;
};

class RandomizedResponse
    : public testing::TestWithParam<RandomizedResponseParam> {};

INSTANTIATE_TEST_SUITE_P(
    PrivacyLossDistribution, RandomizedResponse,
    Values(RandomizedResponseParam{.noise_parameter = 0.5,
                                   .num_buckets = 2,
                                   .expected_pmf = {{2, 0.75},
                                                    {-1, 0.25},
                                                    {0, 0}}},
           RandomizedResponseParam{
               .noise_parameter = 0.2,
               .num_buckets = 4,
               .expected_pmf = {{3, 0.85}, {-2, 0.05}, {0, 0.1}}}));

TEST_P(RandomizedResponse, Create) {
  RandomizedResponseParam param = GetParam();
  absl::StatusOr<std::unique_ptr<PrivacyLossDistribution>> pld =
      PrivacyLossDistribution::CreateForRandomizedResponse(
          param.noise_parameter, param.num_buckets, param.estimate_type,
          /*discretization_interval=*/1);

  ASSERT_OK(pld);

  EXPECT_THAT(pld.value()->Pmf(), PMFIsNear(param.expected_pmf));
}

struct CreateForPrivacyParametersParam {
  double epsilon;
  double delta;
  double discretization_interval;
  ProbabilityMassFunction expected_pmf;
  double expected_infinity_mass;
};

class CreateForPrivacyParameters
    : public testing::TestWithParam<CreateForPrivacyParametersParam> {};

INSTANTIATE_TEST_SUITE_P(
    PrivacyLossDistribution, CreateForPrivacyParameters,
    Values(CreateForPrivacyParametersParam{.epsilon = 1,
                                           .delta = 0,
                                           .discretization_interval = 1,
                                           .expected_pmf = {{1, 0.73105858},
                                                            {-1, 0.26894142}},
                                           .expected_infinity_mass = 0},
           CreateForPrivacyParametersParam{
               .epsilon = 1,
               .delta = 0,
               .discretization_interval = 0.3,
               .expected_pmf = {{4, 0.73105858}, {-3, 0.26894142}},
               .expected_infinity_mass = 0},
           CreateForPrivacyParametersParam{
               .epsilon = 0.5,
               .delta = 0.2,
               .discretization_interval = 0.5,
               .expected_pmf = {{1, 0.49796746}, {-1, 0.30203254}},
               .expected_infinity_mass = 0.2},
           CreateForPrivacyParametersParam{
               .epsilon = 0.5,
               .delta = 0.2,
               .discretization_interval = 0.07,
               .expected_pmf = {{8, 0.49796746}, {-7, 0.30203254}},
               .expected_infinity_mass = 0.2}));

TEST_P(CreateForPrivacyParameters, Create) {
  CreateForPrivacyParametersParam param = GetParam();
  EpsilonDelta epsilon_delta = {param.epsilon, param.delta};

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForPrivacyParameters(
          epsilon_delta, param.discretization_interval);

  EXPECT_THAT(pld->Pmf(), PMFIsNear(param.expected_pmf));
  EXPECT_NEAR(pld->InfinityMass(), param.expected_infinity_mass, kMaxError);
}

struct DiscreteLaplacePrivacyLossDistributionParam {
  std::string test_name;
  double parameter;
  double sensitivity;
  EstimateType estimate_type;
  double discretization_interval;
  ProbabilityMassFunction expected_pmf;
};

class DiscreteLaplacePrivacyLossDistribution
    : public testing::TestWithParam<
          DiscreteLaplacePrivacyLossDistributionParam> {};

INSTANTIATE_TEST_SUITE_P(
    DiscreteLaplacePrivacyLossDistributionParam,
    DiscreteLaplacePrivacyLossDistribution,
    Values(
        DiscreteLaplacePrivacyLossDistributionParam{
            .test_name = "basic",
            .parameter = 1.0,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1,
            .expected_pmf = {{1, 0.73105858}, {-1, 0.26894142}}},
        DiscreteLaplacePrivacyLossDistributionParam{
            .test_name = "varying_sensitivity",
            .parameter = 1.0,
            .sensitivity = 2,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1,
            .expected_pmf = {{2, 0.73105858},
                             {0, 0.17000340},
                             {-2, 0.09893802}}},
        DiscreteLaplacePrivacyLossDistributionParam{
            .test_name = "varying_sensitivity_and_parameter",
            .parameter = 0.8,
            .sensitivity = 2,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1,
            .expected_pmf = {{2, 0.68997448},
                             {0, 0.17072207},
                             {-1, 0.13930345}}},
        DiscreteLaplacePrivacyLossDistributionParam{
            .test_name = "varying_sensitivity_and_parameter_2",
            .parameter = 0.8,
            .sensitivity = 3,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1,
            .expected_pmf = {{3, 0.68997448},
                             {1, 0.17072207},
                             {0, 0.07671037},
                             {-2, 0.06259307}}},
        DiscreteLaplacePrivacyLossDistributionParam{
            .test_name = "varying_discretization_interval",
            .parameter = 1,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 0.03,
            .expected_pmf = {{34, 0.73105858}, {-33, 0.26894142}}},
        DiscreteLaplacePrivacyLossDistributionParam{
            .test_name = "optimistic",
            .parameter = 1,
            .sensitivity = 1,
            .estimate_type = EstimateType::kOptimistic,
            .discretization_interval = 0.03,
            .expected_pmf = {{33, 0.73105858}, {-34, 0.26894142}}}),
    [](const ::testing::TestParamInfo<
        DiscreteLaplacePrivacyLossDistribution::ParamType>& info) {
      return info.param.test_name;
    });

TEST_P(DiscreteLaplacePrivacyLossDistribution, Create) {
  DiscreteLaplacePrivacyLossDistributionParam param = GetParam();

  absl::StatusOr<std::unique_ptr<PrivacyLossDistribution>> pld =
      PrivacyLossDistribution::CreateForDiscreteLaplaceMechanism(
          /*parameter=*/param.parameter, /*sensitivity=*/param.sensitivity,
          /*estimate_type=*/param.estimate_type,
          /*discretization_interval=*/param.discretization_interval);

  ASSERT_OK(pld);

  EXPECT_DOUBLE_EQ((*pld)->InfinityMass(), 0);
  EXPECT_DOUBLE_EQ((*pld)->DiscretizationInterval(),
                   param.discretization_interval);
  EXPECT_EQ((*pld)->GetEstimateType(), param.estimate_type);
  EXPECT_THAT((*pld)->Pmf(), PMFIsNear(param.expected_pmf));
}

struct LaplacePrivacyLossDistributionParam {
  std::string test_name;
  double parameter;
  double sensitivity;
  EstimateType estimate_type;
  double discretization_interval;
  ProbabilityMassFunction expected_pmf;
};

class LaplacePrivacyLossDistribution
    : public testing::TestWithParam<LaplacePrivacyLossDistributionParam> {};

INSTANTIATE_TEST_SUITE_P(
    LaplacePrivacyLossDistributionParam, LaplacePrivacyLossDistribution,
    Values(
        LaplacePrivacyLossDistributionParam{
            .test_name = "basic",
            .parameter = 1,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1,
            .expected_pmf = {{1, 0.69673467},
                             {0, 0.11932561},
                             {-1, 0.18393972}}},
        LaplacePrivacyLossDistributionParam{
            .test_name = "varying_parameter_and_sensitivity",
            .parameter = 1,
            .sensitivity = 2,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1,
            .expected_pmf = {{2, 0.69673467},
                             {1, 0.11932561},
                             {0, 0.07237464},
                             {-1, 0.04389744},
                             {-2, 0.06766764}}},
        LaplacePrivacyLossDistributionParam{
            .test_name = "varying_discretization_interval",
            .parameter = 1,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 0.3,
            .expected_pmf = {{4, 0.52438529},
                             {3, 0.06624934},
                             {2, 0.05702133},
                             {1, 0.04907872},
                             {0, 0.04224244},
                             {-1, 0.03635841},
                             {-2, 0.03129397},
                             {-3, 0.19337051}}},
        LaplacePrivacyLossDistributionParam{
            .test_name = "optimistic_estimate",
            .parameter = 1,
            .sensitivity = 2,
            .estimate_type = EstimateType::kOptimistic,
            .discretization_interval = 1,
            .expected_pmf = {{2, 0.5},
                             {1, 0.19673467},
                             {0, 0.11932561},
                             {-1, 0.07237464},
                             {-2, 0.11156508}}}),
    [](const ::testing::TestParamInfo<
        LaplacePrivacyLossDistribution::ParamType>& info) {
      return info.param.test_name;
    });

TEST_P(LaplacePrivacyLossDistribution, CreatePLD) {
  LaplacePrivacyLossDistributionParam param = GetParam();

  absl::StatusOr<std::unique_ptr<PrivacyLossDistribution>> pld =
      PrivacyLossDistribution::CreateForLaplaceMechanism(
          /*parameter=*/param.parameter, /*sensitivity=*/param.sensitivity,
          /*estimate_type=*/param.estimate_type,
          /*discretization_interval=*/param.discretization_interval);

  ASSERT_OK(pld);

  EXPECT_DOUBLE_EQ((*pld)->InfinityMass(), 0);
  EXPECT_DOUBLE_EQ((*pld)->DiscretizationInterval(),
                   param.discretization_interval);
  EXPECT_EQ((*pld)->GetEstimateType(), param.estimate_type);
  EXPECT_THAT((*pld)->Pmf(), PMFIsNear(param.expected_pmf));
}

struct GaussianPrivacyLossDistributionParam {
  std::string test_name;
  double standard_deviation;
  double sensitivity;
  EstimateType estimate_type;
  double discretization_interval;
  ProbabilityMassFunction expected_pmf;
  double expected_infinity_mass;
};

class GaussianPrivacyLossDistribution
    : public testing::TestWithParam<GaussianPrivacyLossDistributionParam> {};

INSTANTIATE_TEST_SUITE_P(
    GaussianPrivacyLossDistributionParam, GaussianPrivacyLossDistribution,
    Values(
        GaussianPrivacyLossDistributionParam{
            .test_name = "basic",
            .standard_deviation = 1,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1,
            .expected_pmf = {{2, 0.12447741}, {1, 0.38292492}, {0, 0.30853754}},
            .expected_infinity_mass = 0.18406013,  // = CDF_normal(-0.9)
        },
        GaussianPrivacyLossDistributionParam{
            .test_name = "varying_standard_deviation_and_sensitivity",
            .standard_deviation = 3,
            .sensitivity = 6,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1,
            .expected_pmf = {{4, 0.12447741},
                             {3, 0.19146246},
                             {2, 0.19146246},
                             {1, 0.30853754}},
            .expected_infinity_mass = 0.18406013,  // = CDF_normal(-0.9)
        },
        GaussianPrivacyLossDistributionParam{
            .test_name = "varying_discretization_interval",
            .standard_deviation = 1,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 0.3,
            .expected_pmf = {{5, 0.05790353},
                             {4, 0.10261461},
                             {3, 0.11559390},
                             {2, 0.11908755},
                             {1, 0.11220275},
                             {0, 0.09668214},
                             {-1, 0.21185540}},
            .expected_infinity_mass = 0.18406013,  // = CDF_normal(-0.9)
        },
        GaussianPrivacyLossDistributionParam{
            .test_name = "optimistic_estimate",
            .standard_deviation = 1,
            .sensitivity = 1,
            .estimate_type = EstimateType::kOptimistic,
            .discretization_interval = 1,
            .expected_pmf = {{1, 0.30853754},
                             {0, 0.38292492},
                             {-1, 0.12447741}},
            .expected_infinity_mass = 0}),
    [](const ::testing::TestParamInfo<
        GaussianPrivacyLossDistribution::ParamType>& info) {
      return info.param.test_name;
    });

TEST_P(GaussianPrivacyLossDistribution, CreatePLD) {
  GaussianPrivacyLossDistributionParam param = GetParam();
  // mass_truncation_bound = ln(2) + log(CDF_normal(-0.9)).
  double mass_truncation_bound = -0.999345626001393;

  absl::StatusOr<std::unique_ptr<PrivacyLossDistribution>> pld =
      PrivacyLossDistribution::CreateForGaussianMechanism(
          /*standard_deviation=*/param.standard_deviation,
          /*sensitivity=*/param.sensitivity,
          /*estimate_type=*/param.estimate_type,
          /*discretization_interval=*/param.discretization_interval,
          /*mass_truncation_bound=*/mass_truncation_bound);

  ASSERT_OK(pld);

  EXPECT_NEAR((*pld)->InfinityMass(), param.expected_infinity_mass, kMaxError);
  EXPECT_DOUBLE_EQ((*pld)->DiscretizationInterval(),
                   param.discretization_interval);
  EXPECT_EQ((*pld)->GetEstimateType(), param.estimate_type);
  EXPECT_THAT((*pld)->Pmf(), PMFIsNear(param.expected_pmf));
}

struct DiscreteGaussianPrivacyLossDistributionParam {
  std::string test_name;
  double sigma;
  double sensitivity;
  EstimateType estimate_type;
  double discretization_interval;
  ProbabilityMassFunction expected_pmf;
  double expected_infinity_mass;
  std::optional<int> truncation_bound = std::nullopt;
};

class DiscreteGaussianPrivacyLossDistribution
    : public testing::TestWithParam<
          DiscreteGaussianPrivacyLossDistributionParam> {};

INSTANTIATE_TEST_SUITE_P(
    DiscreteGaussianPrivacyLossDistributionParam,
    DiscreteGaussianPrivacyLossDistribution,
    Values(
        DiscreteGaussianPrivacyLossDistributionParam{
            .test_name = "basic",
            .sigma = 1,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1e-2,
            .expected_pmf = {{50, 0.45186276}, {-50, 0.27406862}},
            .expected_infinity_mass = 0.27406862,
            .truncation_bound = 1,
        },
        DiscreteGaussianPrivacyLossDistributionParam{
            .test_name = "varying_sensitivity",
            .sigma = 1,
            .sensitivity = 2,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1e-2,
            .expected_pmf = {{0, 0.27406862}},
            .expected_infinity_mass = 0.72593138,
            .truncation_bound = 1,
        },
        DiscreteGaussianPrivacyLossDistributionParam{
            .test_name = "varying_sigma",
            .sigma = 3,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1e-2,
            .expected_pmf = {{6, 0.34579116}, {-5, 0.32710442}},
            .expected_infinity_mass = 0.32710442,
            .truncation_bound = 1,
        },
        DiscreteGaussianPrivacyLossDistributionParam{
            .test_name = "varying_discretization_interval",
            .sigma = 3,
            .sensitivity = 1,
            .estimate_type = EstimateType::kPessimistic,
            .discretization_interval = 1e-3,
            .expected_pmf = {{56, 0.34579116}, {-55, 0.32710442}},
            .expected_infinity_mass = 0.32710442,
            .truncation_bound = 1,
        },
        DiscreteGaussianPrivacyLossDistributionParam{
            .test_name = "optimistic",
            .sigma = 3,
            .sensitivity = 1,
            .estimate_type = EstimateType::kOptimistic,
            .discretization_interval = 1e-4,
            .expected_pmf = {{555, 0.34579116}, {-556, 0.32710442}},
            .expected_infinity_mass = 0.32710442,
            .truncation_bound = 1,
        },
        DiscreteGaussianPrivacyLossDistributionParam{
            .test_name = "default_truncation",
            .sigma = 3,
            .sensitivity = 1,
            .estimate_type = EstimateType::kOptimistic,
            .discretization_interval = 1,
            .expected_pmf =
                {{1, 0.00221}, {0, 0.56428}, {-1, 0.43278}, {-2, 0.00073}},
            .expected_infinity_mass = 0,
        }),
    [](const ::testing::TestParamInfo<
        DiscreteGaussianPrivacyLossDistribution::ParamType>& info) {
      return info.param.test_name;
    });

TEST_P(DiscreteGaussianPrivacyLossDistribution, CreatePLD) {
  DiscreteGaussianPrivacyLossDistributionParam param = GetParam();

  absl::StatusOr<std::unique_ptr<PrivacyLossDistribution>> pld =
      PrivacyLossDistribution::CreateForDiscreteGaussianMechanism(
          /*sigma=*/param.sigma,
          /*sensitivity=*/param.sensitivity,
          /*estimate_type=*/param.estimate_type,
          /*discretization_interval=*/param.discretization_interval,
          /*truncation_bound=*/param.truncation_bound);

  ASSERT_OK(pld);

  EXPECT_NEAR((*pld)->InfinityMass(), param.expected_infinity_mass, kMaxError);
  EXPECT_DOUBLE_EQ((*pld)->DiscretizationInterval(),
                   param.discretization_interval);
  EXPECT_EQ((*pld)->GetEstimateType(), param.estimate_type);
  EXPECT_THAT((*pld)->Pmf(), PMFIsNear(param.expected_pmf));
}

TEST(PrivacyLossDistributionTest, AccurateComposition) {
  absl::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> noise_privacy_loss =
      GaussianPrivacyLoss::Create(
          /*standard_deviation=*/4,
          /*sensitivity=*/1);
  ASSERT_OK(noise_privacy_loss);

  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForAdditiveNoise(
          *noise_privacy_loss.value(), EstimateType::kPessimistic,
          /*discretization_interval=*/1e-4);

  constexpr int num_times = 40;
  pld->Compose(num_times);

  constexpr double epsilon = 10;
  double delta = pld->GetDeltaForEpsilon(epsilon);

  EXPECT_NEAR(delta, 3.33762759e-9, 1e-10);
}

TEST(PrivacyLossDistributionTest, Serialization) {
  ProbabilityMassFunction pmf = {{1, 0.6}, {5, 0.3}};
  double infinity_mass = 0.1;
  double discretization_interval = 1e-3;
  EstimateType estimate_type = EstimateType::kPessimistic;
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(
          pmf, infinity_mass, discretization_interval, estimate_type);

  absl::StatusOr<serialization::PrivacyLossDistribution> serialized_result =
      pld->Serialize();
  ASSERT_OK(serialized_result);

  absl::StatusOr<std::unique_ptr<PrivacyLossDistribution>> deserialized_result =
      PrivacyLossDistribution::Deserialize(*serialized_result);
  ASSERT_OK(deserialized_result);

  EXPECT_THAT(
      (*deserialized_result)->Pmf(),
      UnorderedElementsAre(FieldsAre(Eq(1), DoubleNear(0.6, kMaxError)),
                           FieldsAre(Eq(5), DoubleNear(0.3, kMaxError))));
  EXPECT_NEAR((*deserialized_result)->InfinityMass(), infinity_mass, kMaxError);
  EXPECT_NEAR((*deserialized_result)->DiscretizationInterval(),
              discretization_interval, kMaxError);
}

TEST(PrivacyLossDistributionTest, SerializationOptimisticError) {
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(
          /*probability_mass_function=*/{}, /*infinity_mass=*/0,
          /*discretization_interval=*/1e-4,
          /*estimate_type=*/EstimateType::kOptimistic);

  EXPECT_THAT(pld->Serialize(), StatusIs(absl::StatusCode::kInvalidArgument,
                                         HasSubstr("optimistic")));
}

TEST(PrivacyLossDistributionTest, DeserializationNoPMFError) {
  serialization::PrivacyLossDistribution serialized_pld;
  EXPECT_THAT(PrivacyLossDistribution::Deserialize(serialized_pld),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("PMF")));
}
}  // namespace
}  // namespace accounting
}  // namespace differential_privacy
