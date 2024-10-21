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

#include "accounting/accountant.h"

#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "accounting/common/common.h"
#include "accounting/privacy_loss_distribution.h"
#include "base/testing/status_matchers.h"

namespace differential_privacy {
namespace accounting {

namespace {

constexpr double kMaxError = 1e-1;
using ::testing::Values;
using ::differential_privacy::base::testing::StatusIs;

struct TestCaseParam {
  EpsilonDelta epsilon_delta;
  int num_queries;
  double sensitivity = 1;
  NoiseFunction noise_function;
  double expected_parameter;
};

class AccountantTest : public testing::TestWithParam<TestCaseParam> {};

std::unique_ptr<AdditiveNoisePrivacyLoss> LaplaceNoiseFunction(
    double parameter, double sensitivity) {
  return LaplacePrivacyLoss::Create(parameter, sensitivity).value();
}

INSTANTIATE_TEST_SUITE_P(
    AccountantSuite, AccountantTest,
    Values(
        TestCaseParam{
            .epsilon_delta = EpsilonDelta{.epsilon = 3, .delta = 0},
            .num_queries = 5,
            .sensitivity = 2.1,
            .noise_function = LaplaceNoiseFunction,
            .expected_parameter = 3.5,
        },
        TestCaseParam{
            .epsilon_delta = EpsilonDelta{.epsilon = 1, .delta = 0.0001},
            .num_queries = 20,
            .sensitivity = 1,
            .noise_function = LaplaceNoiseFunction,
            .expected_parameter = 13.6},
        TestCaseParam{
            .epsilon_delta = EpsilonDelta{.epsilon = 1, .delta = 0.0001},
            .num_queries = 20,
            .sensitivity = 0.5,
            .noise_function = LaplaceNoiseFunction,
            .expected_parameter = 6.8}));

TEST_P(AccountantTest, Basic) {
  TestCaseParam param = GetParam();

  auto result = GetSmallestParameter(param.epsilon_delta, param.num_queries,
                                     param.sensitivity, param.noise_function,
                                     std::optional<double>());
  EXPECT_OK(result.status());
  EXPECT_NEAR(result.value(), param.expected_parameter, kMaxError);
}

struct AdvancedCompositionParam {
  double epsilon;
  double delta;
  double total_delta;
  int num_queries;
  double expected_total_epsilon;
};

class AdvancedCompositionTest
    : public testing::TestWithParam<AdvancedCompositionParam> {};

INSTANTIATE_TEST_SUITE_P(
    AdvancedCompositionSuite, AdvancedCompositionTest,
    Values(
        // Basic Composition
        AdvancedCompositionParam{.epsilon = 1,
                                 .delta = 0,
                                 .total_delta = 0,
                                 .num_queries = 30,
                                 .expected_total_epsilon = 30},
        // Advantage over basic #1
        AdvancedCompositionParam{.epsilon = 1,
                                 .delta = 0.001,
                                 .total_delta = 0.06,
                                 .num_queries = 30,
                                 .expected_total_epsilon = 22},
        // Advantage over basic #2
        AdvancedCompositionParam{.epsilon = 1,
                                 .delta = 0.001,
                                 .total_delta = 0.1,
                                 .num_queries = 30,
                                 .expected_total_epsilon = 20}));

TEST_P(AdvancedCompositionTest, Basic) {
  AdvancedCompositionParam param = GetParam();

  auto result = AdvancedComposition(
      EpsilonDelta{.epsilon = param.epsilon, .delta = param.delta},
      param.num_queries, param.total_delta);
  EXPECT_OK(result.status());
  EXPECT_NEAR(result.value(), param.expected_total_epsilon, kMaxError);
}

TEST(AdvancedCompositionNotFoundTest, TotalDeltaTooSmall) {
  int num_queries = 30;
  double total_delta = 0.26;
  auto result = AdvancedComposition(EpsilonDelta{.epsilon = 0.1, .delta = 0.2},
                                    num_queries, total_delta);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kNotFound));
}
}  // namespace
}  // namespace accounting
}  // namespace differential_privacy
