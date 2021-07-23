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

#include "accounting/convolution.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "accounting/common/test_util.h"

namespace differential_privacy {
namespace accounting {
namespace {
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Key;
using ::testing::Le;
using ::testing::Matcher;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::Values;

constexpr double kMaxError = 1e-5;

Matcher<ProbabilityMassFunction> PMFIsNear(
    const ProbabilityMassFunction& expected) {
  return differential_privacy::accounting::PMFIsNear(expected, kMaxError);
}

TEST(Convolution, UnpackProbabilityMassFunction) {
  ProbabilityMassFunction pmf = {{5, 2.3}, {3, 3.14}, {1, 1.2}};

  UnpackedProbabilityMassFunction result = UnpackProbabilityMassFunction(pmf);

  EXPECT_EQ(result.min_key, 1);
  EXPECT_THAT(result.items, ElementsAre(1.2, 0, 3.14, 0, 2.3));
}

TEST(Convolution, UnpackEmptyPmf) {
  UnpackedProbabilityMassFunction result =
      UnpackProbabilityMassFunction(ProbabilityMassFunction());

  EXPECT_EQ(result.min_key, 0);
  EXPECT_TRUE(result.items.empty());
}

TEST(Convolution, CreateProbabilityMassFunction) {
  UnpackedProbabilityMassFunction input = {1, {1.2, 0, 3.14, 0, 2.3}};

  ProbabilityMassFunction result = CreateProbabilityMassFunction(input);

  EXPECT_THAT(result,
              UnorderedElementsAre(Pair(5, 2.3), Pair(3, 3.14), Pair(1, 1.2)));
}

TEST(Convolution, CreateProbabilityMassFunctionTruncationBothSide) {
  UnpackedProbabilityMassFunction input = {1, {0.2, 0.5, 0.3}};

  ProbabilityMassFunction result = CreateProbabilityMassFunction(input, 0.601);

  EXPECT_THAT(result, UnorderedElementsAre(Pair(2, 0.5)));
}

TEST(Convolution, CreateProbabilityMassFunctionTruncationLowerOnly) {
  UnpackedProbabilityMassFunction input = {1, {0.2, 0.5, 0.3}};

  ProbabilityMassFunction result = CreateProbabilityMassFunction(input, 0.4);

  EXPECT_THAT(result, UnorderedElementsAre(Pair(2, 0.5), Pair(3, 0.3)));
}

TEST(Convolution, CreateProbabilityMassFunctionTruncationUpperOnly) {
  UnpackedProbabilityMassFunction input = {1, {0.4, 0.5, 0.1}};

  ProbabilityMassFunction result = CreateProbabilityMassFunction(input, 0.2);

  EXPECT_THAT(result, UnorderedElementsAre(Pair(1, 0.4), Pair(2, 0.5)));
}

TEST(Convolution, CreateProbabilityMassFunctionTruncationAll) {
  UnpackedProbabilityMassFunction input = {1, {0.4, 0.5, 0.1}};

  ProbabilityMassFunction result = CreateProbabilityMassFunction(input, 3);

  EXPECT_THAT(result, IsEmpty());
}

TEST(Convolution, Convolve) {
  ProbabilityMassFunction pmf_x = {{1, 2}, {3, 4}};
  ProbabilityMassFunction pmf_y = {{2, 3}, {4, 6}};

  ProbabilityMassFunction result = Convolve(pmf_x, pmf_y);

  EXPECT_THAT(result,
              UnorderedElementsAre(Pair(3, DoubleNear(6.0, kMaxError)),
                                   Pair(5, DoubleNear(24.0, kMaxError)),
                                   Pair(7, DoubleNear(24.0, kMaxError))));
}

TEST(Convolution, ConvolveTruncation) {
  ProbabilityMassFunction pmf_x = {{1, 0.4}, {2, 0.6}};
  ProbabilityMassFunction pmf_y = {{1, 0.7}, {3, 0.3}};

  ProbabilityMassFunction result = Convolve(pmf_x, pmf_y, 0.57);

  EXPECT_THAT(result,
              UnorderedElementsAre(Pair(3, DoubleNear(0.42, kMaxError)),
                                   Pair(4, DoubleNear(0.12, kMaxError))));
}

TEST(Convolution, ConvolveOutputResize) {
  ProbabilityMassFunction pmf_x = {{1, 2}, {4001, 4}};
  ProbabilityMassFunction pmf_y = {{1, 3}, {3050, 6}};
  EXPECT_THAT(Convolve(pmf_x, pmf_y), Each(Key(Le(7051))));
}

TEST(Convolution, ConvolveMultiple) {
  ProbabilityMassFunction pmf = {{1, 2}, {3, 5}, {4, 6}};
  EXPECT_THAT(Convolve(pmf, 3),
              UnorderedElementsAre(Pair(3, DoubleNear(8.0, kMaxError)),
                                   Pair(5, DoubleNear(60.0, kMaxError)),
                                   Pair(6, DoubleNear(72.0, kMaxError)),
                                   Pair(7, DoubleNear(150.0, kMaxError)),
                                   Pair(8, DoubleNear(360.0, kMaxError)),
                                   Pair(9, DoubleNear(341.0, kMaxError)),
                                   Pair(10, DoubleNear(450.0, kMaxError)),
                                   Pair(11, DoubleNear(540.0, kMaxError)),
                                   Pair(12, DoubleNear(216.0, kMaxError))));
}

TEST(Convolution, ConvolveMultipleOutputResize) {
  ProbabilityMassFunction pmf = {{1, 2}, {1673, 5}};
  EXPECT_THAT(Convolve(pmf, /*num_times=*/2, /*tail_mass_truncation=*/0),
              Each(Key(Le(3346))));
}

struct ComputeConvolutionTruncationBoundsParam {
  std::vector<double> pmf;
  int num_times;
  double order;
  double tail_mass_truncation;
  int expected_lower_bound;
  int expected_upper_bound;
};

class ComputeConvolutionTruncationBoundsTest
    : public testing::TestWithParam<ComputeConvolutionTruncationBoundsParam> {};

INSTANTIATE_TEST_SUITE_P(
    ComputeConvolutionTruncationBoundsSuite,
    ComputeConvolutionTruncationBoundsTest,
    Values(
        // Test lower bound computation.
        ComputeConvolutionTruncationBoundsParam{.pmf = {0.1, 0.4, 0.5},
                                                .num_times = 3,
                                                .order = -1,
                                                .tail_mass_truncation = 0.5,
                                                .expected_lower_bound = 2,
                                                .expected_upper_bound = 6},
        // Test upper bound computation.
        ComputeConvolutionTruncationBoundsParam{.pmf = {0.2, 0.6, 0.2},
                                                .num_times = 3,
                                                .order = 1,
                                                .tail_mass_truncation = 0.7,
                                                .expected_lower_bound = 0,
                                                .expected_upper_bound = 5}));

TEST_P(ComputeConvolutionTruncationBoundsTest,
       ComputeConvolutionTruncationBoundsBasic) {
  ComputeConvolutionTruncationBoundsParam param = GetParam();
  UnpackedProbabilityMassFunction x;
  x.items = param.pmf;
  ConvolutionTruncationBounds result = ComputeConvolutionTruncationBounds(
      x, param.num_times, param.tail_mass_truncation, {{param.order}});
  EXPECT_EQ(param.expected_lower_bound, result.lower_bound);
  EXPECT_EQ(param.expected_upper_bound, result.upper_bound);
}

struct ConvolveMultipleTruncationParam {
  ProbabilityMassFunction input_pmf;
  int num_times;
  double tail_mass_truncation;
  ProbabilityMassFunction expected_output_pmf;
};

class ConvolveMultipleTruncationTest
    : public testing::TestWithParam<ConvolveMultipleTruncationParam> {};
INSTANTIATE_TEST_SUITE_P(
    ConvolveMultipleTruncationSuite, ConvolveMultipleTruncationTest,
    Values(
        // Test lower bound computation.
        ConvolveMultipleTruncationParam{
            .input_pmf = {{0, 0.1}, {1, 0.4}, {2, 0.5}},
            .num_times = 3,
            .tail_mass_truncation = 0.5,
            .expected_output_pmf = ProbabilityMassFunction(
                {{2, 0.063}, {3, 0.184}, {4, 0.315}, {5, 0.300}, {6, 0.126}})},
        // Test upper bound computation.
        ConvolveMultipleTruncationParam{
            .input_pmf = {{0, 0.2}, {1, 0.6}, {2, 0.2}},
            .num_times = 3,
            .tail_mass_truncation = 0.7,
            .expected_output_pmf = ProbabilityMassFunction(
                {{1, 0.072}, {2, 0.24}, {3, 0.36}, {4, 0.24}, {5, 0.072}})}));

TEST_P(ConvolveMultipleTruncationTest, ConvolveMultipleTruncationBasic) {
  ConvolveMultipleTruncationParam param = GetParam();
  ProbabilityMassFunction result =
      Convolve(param.input_pmf, param.num_times, param.tail_mass_truncation);
  EXPECT_THAT(result, PMFIsNear(param.expected_output_pmf));
}
}  // namespace
}  // namespace accounting
}  // namespace differential_privacy
