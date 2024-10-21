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

#include "accounting/common/common.h"

#include <cmath>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/testing/status_matchers.h"

namespace differential_privacy {
namespace accounting {
namespace {

using ::testing::Values;
using ::differential_privacy::base::testing::StatusIs;

constexpr double kMaxError = 1e-4;

struct EpsilonDeltaParam {
  double epsilon;
  double delta;
};

class EpsilonDeltaTest : public testing::TestWithParam<EpsilonDeltaParam> {};

INSTANTIATE_TEST_SUITE_P(EpsilonDeltaSuite, EpsilonDeltaTest,
                         Values(
                             // epsilon negative
                             EpsilonDeltaParam{.epsilon = -0.1, .delta = 0.1},
                             // delta negative
                             EpsilonDeltaParam{.epsilon = 1, .delta = -0.1},
                             // delta greater than one
                             EpsilonDeltaParam{.epsilon = 1, .delta = 1.1}));

TEST_P(EpsilonDeltaTest, ParametersOutOfRange) {
  const EpsilonDeltaParam& param = GetParam();
  EXPECT_THAT((EpsilonDelta{param.epsilon, param.delta}.Validate()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

struct InverseMonotoneFunctionParam {
  std::function<double(double x)> func;
  double value;
  double lower_x;
  double upper_x;
  std::optional<double> initial_guess;
  bool increasing;
  double expected_x;
  bool discrete = false;
};

class InverseMonotoneFunctionTest
    : public testing::TestWithParam<InverseMonotoneFunctionParam> {};

INSTANTIATE_TEST_SUITE_P(
    InverseMonotoneFunctionSuite, InverseMonotoneFunctionTest,
    Values(
        // without initial guess
        InverseMonotoneFunctionParam{.func = [](double x) { return -x; },
                                     .value = -4.5,
                                     .lower_x = 0,
                                     .upper_x = 10,
                                     .initial_guess = std::nullopt,
                                     .increasing = false,
                                     .expected_x = 4.5},
        // with initial guess
        InverseMonotoneFunctionParam{.func = [](double x) { return -x; },
                                     .value = -5,
                                     .lower_x = 0,
                                     .upper_x = 10,
                                     .initial_guess = 2,
                                     .increasing = false,
                                     .expected_x = 5},
        // infinite upper bound
        InverseMonotoneFunctionParam{
            .func = [](double x) { return -1 / (1 / x); },
            .value = -5,
            .lower_x = 0,
            .upper_x = std::numeric_limits<double>::infinity(),
            .initial_guess = 2,
            .increasing = false,
            .expected_x = 5},
        // increasing, without initial guess
        InverseMonotoneFunctionParam{
            .func = [](double x) { return std::pow(x, 2); },
            .value = 25,
            .lower_x = 0,
            .upper_x = 10,
            .initial_guess = std::nullopt,
            .increasing = true,
            .expected_x = 5},
        // increasing, with initial guess
        InverseMonotoneFunctionParam{
            .func = [](double x) { return std::pow(x, 2); },
            .value = 25,
            .lower_x = 0,
            .upper_x = 10,
            .initial_guess = 2,
            .increasing = true,
            .expected_x = 5},
        // discrete
        InverseMonotoneFunctionParam{.func = [](double x) { return -x; },
                                     .value = -4.5,
                                     .lower_x = 0,
                                     .upper_x = 10,
                                     .initial_guess = std::nullopt,
                                     .increasing = false,
                                     .expected_x = 5,
                                     .discrete = true}));

TEST_P(InverseMonotoneFunctionTest, InverseMonotoneFunctionBasic) {
  const InverseMonotoneFunctionParam& param = GetParam();
  BinarySearchParameters search_parameters = {
      .lower_bound = param.lower_x,
      .upper_bound = param.upper_x,
      .initial_guess = param.initial_guess,
      .tolerance = 1e-7,
      .discrete = param.discrete};

  absl::StatusOr<double> x = InverseMonotoneFunction(
      param.func, param.value, search_parameters, param.increasing);

  EXPECT_OK(x);
  EXPECT_NEAR(x.value(), param.expected_x, kMaxError);
}

TEST(InverseMonotoneFunctionTest, InverseMonotoneFunctionNotFoundTooLarge) {
  BinarySearchParameters search_parameters = {.lower_bound = -5,
                                              .upper_bound = 4};

  auto decreasing_func = [](double x) { return -x; };
  absl::StatusOr<double> x =
      InverseMonotoneFunction(decreasing_func, -5, search_parameters);
  EXPECT_THAT(x, StatusIs(absl::StatusCode::kNotFound));
}

TEST(InverseMonotoneFunctionTest, InverseMonotoneFunctionNotFoundTooSmall) {
  BinarySearchParameters search_parameters = {.lower_bound = -5,
                                              .upper_bound = 4};

  // inverse is too small for increasing function
  auto increasing_func = [](double x) { return x; };
  absl::StatusOr<double> x =
      InverseMonotoneFunction(increasing_func, -6, search_parameters,
                              /*increasing=*/true);
  EXPECT_THAT(x, StatusIs(absl::StatusCode::kNotFound));
}

TEST(InverseMonotoneFunctionTest, InverseMonotoneFunctionNotEvaluateInfinity) {
  BinarySearchParameters search_parameters = {
      .lower_bound = 3,
      .upper_bound = std::numeric_limits<double>::infinity(),
      .initial_guess = 6};

  auto decreasing_func = [](double x) {
    if (x != std::numeric_limits<double>::infinity())
      return -x;
    else
      return 1000.0;
  };
  absl::StatusOr<double> x =
      InverseMonotoneFunction(decreasing_func, -5, search_parameters);
  EXPECT_OK(x);
  EXPECT_NEAR(x.value(), 5, kMaxError);
}

TEST(InverseMonotoneFunctionTest,
     InverseMonotoneFunctionNotEvaluateMinusInfinity) {
  BinarySearchParameters search_parameters = {
      .lower_bound = -std::numeric_limits<double>::infinity(),
      .upper_bound = 7,
      .initial_guess = 2};

  auto increasing_func = [](double x) {
    if (x != -std::numeric_limits<double>::infinity())
      return x;
    else
      return 1000.0;
  };
  absl::StatusOr<double> x =
      InverseMonotoneFunction(increasing_func, 5, search_parameters,
                              /*increasing=*/true);
  EXPECT_OK(x);
  EXPECT_NEAR(x.value(), 5, kMaxError);
}

TEST(InverseMonotoneFunctionTest,
     InverseMonotoneFunctionPropagageStatusUpperBound) {
  BinarySearchParameters search_parameters = {
      .lower_bound = -1, .upper_bound = 7, .initial_guess = 2};

  auto increasing_func = [](double x) -> absl::StatusOr<double> {
    if (x != 7)
      return -x;
    else
      return absl::InvalidArgumentError("Error: upper bound");
  };
  absl::StatusOr<double> x =
      InverseMonotoneFunction(increasing_func, -5, search_parameters);
  EXPECT_THAT(x, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(InverseMonotoneFunctionTest,
     InverseMonotoneFunctionPropagageStatusLowerBound) {
  BinarySearchParameters search_parameters = {
      .lower_bound = -1, .upper_bound = 7, .initial_guess = 2};

  auto increasing_func = [](double x) -> absl::StatusOr<double> {
    if (x != -1)
      return x;
    else
      return absl::InvalidArgumentError("Error: lower bound");
  };
  absl::StatusOr<double> x = InverseMonotoneFunction(
      increasing_func, 5, search_parameters, /*increasing=*/true);
  EXPECT_THAT(x, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(InverseMonotoneFunctionTest, InverseMonotoneFunctionPropagageStatus) {
  BinarySearchParameters search_parameters = {
      .lower_bound = -1, .upper_bound = 7, .initial_guess = 2};

  auto increasing_func = [](double x) -> absl::StatusOr<double> {
    if (x != 4)
      return x;
    else
      return absl::InvalidArgumentError("Error");
  };
  absl::StatusOr<double> x = InverseMonotoneFunction(
      increasing_func, 5, search_parameters, /*increasing=*/true);
  EXPECT_THAT(x, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace accounting
}  // namespace differential_privacy
