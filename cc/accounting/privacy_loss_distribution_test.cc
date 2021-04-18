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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "base/statusor.h"
#include "accounting/common/common.h"
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
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;
using ::testing::Values;
using ::differential_privacy::base::testing::IsOk;
using ::differential_privacy::base::testing::StatusIs;

constexpr double kMaxError = 1e-4f;

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

  ProbabilityMassFunction expected_pmf = {{2, 0.36}, {3, 0.48}, {4, 0.16}};
  EXPECT_THAT(pld->InfinityMass(), DoubleNear(0.51, kMaxError));
  EXPECT_THAT(pld->Pmf(), UnorderedElementsAre(
                              FieldsAre(Eq(2), DoubleNear(0.36, kMaxError)),
                              FieldsAre(Eq(3), DoubleNear(0.48, kMaxError)),
                              FieldsAre(Eq(4), DoubleNear(0.16, kMaxError))));
}

TEST(PrivacyLossDistributionTest,
     ComposeErrorDifferentDiscretizationIntervals) {
  ProbabilityMassFunction pmf = {};
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistributionTestPeer::Create(pmf, 0.3, 1e-4);

  std::unique_ptr<PrivacyLossDistribution> pld_other =
      PrivacyLossDistributionTestPeer::Create(pmf, 0.3, 2e-4);

  EXPECT_THAT(pld->Compose(*pld_other),
              StatusIs(absl::InvalidArgumentError("").code(),
                       HasSubstr("Cannot compose, discretization intervals "
                                 "are different - 0.000200 vs 0.000100")));
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

  EXPECT_THAT(pld->Compose(*pld_other),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       Eq("Cannot compose, estimate types are different")));
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

TEST(PrivacyLossDistributionTest, DivergenceFromMechansim) {
  base::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> noise_privacy_loss =
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
  base::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> noise_privacy_loss =
      GaussianPrivacyLoss::Create(
          /*standard_deviation=*/1,
          /*sensitivity=*/2,
          /*estimate_type=*/EstimateType::kOptimistic,
          /*mass_truncation_bound=*/-0.999345);
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

TEST(PrivacyLossDistributionTest, AccurateComposition) {
  base::StatusOr<std::unique_ptr<GaussianPrivacyLoss>> noise_privacy_loss =
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
}  // namespace
}  // namespace accounting
}  // namespace differential_privacy
