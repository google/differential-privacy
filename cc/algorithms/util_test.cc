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

#include "algorithms/util.h"

#include <limits>

#include "gtest/gtest.h"
#include "algorithms/distributions.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "base/status.h"

namespace differential_privacy {
namespace {

const char kSeedString[] = "ABCDEFGHIJKLMNOP";
constexpr int64_t kStatsSize = 50000;
constexpr double kTolerance = 1e-5;

TEST(XorStringsTest, XorsSameLength) {
  std::string first = "foo";
  std::string second = "bar";

  std::string result = XorStrings(first, second);

  EXPECT_EQ('f' ^ 'b', result[0]);
  EXPECT_EQ('o' ^ 'a', result[1]);
  EXPECT_EQ('o' ^ 'r', result[2]);
}

TEST(XorStringsTest, ShorterStringRepeated) {
  std::string first = "foobar";
  std::string second = "baz";

  std::string result = XorStrings(first, second);

  EXPECT_EQ('b' ^ 'b', result[3]);
  EXPECT_EQ('a' ^ 'a', result[4]);
  EXPECT_EQ('z' ^ 'r', result[5]);
}

TEST(XorStringsTest, EmptyStringReturnsUnchanged) {
  std::string first = "foo";
  std::string second = "";

  std::string result = XorStrings(first, second);

  EXPECT_EQ(result, "foo");
}

TEST(XorStringsTest, DoubleEmptyString) {
  std::string first = "";
  std::string second = "";

  std::string result = XorStrings(first, second);

  EXPECT_EQ(result, "");
}

TEST(EpsilonRiskValuesTest, DefaultEpsilon) {
  EXPECT_EQ(DefaultEpsilon(), std::log(3));
}

TEST(NextPowerTest, PositivesPowers) {
  EXPECT_NEAR(GetNextPowerOfTwo(3.0), 4.0, kTolerance);
  EXPECT_NEAR(GetNextPowerOfTwo(5.0), 8.0, kTolerance);
  EXPECT_NEAR(GetNextPowerOfTwo(7.9), 8.0, kTolerance);
}

TEST(NextPowerTest, ExactPositivePowers) {
  EXPECT_NEAR(GetNextPowerOfTwo(2.0), 2.0, kTolerance);
  EXPECT_NEAR(GetNextPowerOfTwo(8.0), 8.0, kTolerance);
}

TEST(NextPowerTest, One) {
  EXPECT_NEAR(GetNextPowerOfTwo(1.0), 1.0, kTolerance);
}

TEST(NextPowerTest, NegativePowers) {
  EXPECT_NEAR(GetNextPowerOfTwo(0.4), 0.5, kTolerance);
  EXPECT_NEAR(GetNextPowerOfTwo(0.2), 0.25, kTolerance);
}

TEST(NextPowerTest, ExactNegativePowers) {
  EXPECT_NEAR(GetNextPowerOfTwo(0.5), 0.5, kTolerance);
  EXPECT_NEAR(GetNextPowerOfTwo(0.125), 0.125, kTolerance);
}

TEST(InverseErrorTest, ProperResults) {
  // true values are pre-calculated
  EXPECT_NEAR(InverseErrorFunction(0.24), 0.216, 0.001);
  EXPECT_NEAR(InverseErrorFunction(0.9999), 2.751, 0.001);
  EXPECT_NEAR(InverseErrorFunction(0.0012), 0.001, 0.001);
  EXPECT_NEAR(InverseErrorFunction(0.5), 0.476, 0.001);
  EXPECT_NEAR(InverseErrorFunction(0.39), 0.360, 0.001);
  EXPECT_NEAR(InverseErrorFunction(0.0067), 0.0059, 0.001);

  double max = 1;
  double min = -1;
  for (int i = 0; i < 1000; i++) {
    double n = (max - min) * ((double)rand() / RAND_MAX) + min;
    EXPECT_NEAR(std::erf(InverseErrorFunction(n)), n, 0.001);
  }
}

TEST(InverseErrorTest, EdgeCases) {
  EXPECT_EQ(InverseErrorFunction(-1),
            -1 * std::numeric_limits<double>::infinity());
  EXPECT_EQ(InverseErrorFunction(1), std::numeric_limits<double>::infinity());
  EXPECT_EQ(InverseErrorFunction(0), 0);
}

// In RoundToNearestMultiple tests exact comparison of double is used, because
// for rounding to multiple of power of 2 RoundToNearestMultiple should provide
// exact value.
TEST(RoundTest, PositiveNoTies) {
  EXPECT_EQ(RoundToNearestMultiple(4.9, 2.0), 4.0);
  EXPECT_EQ(RoundToNearestMultiple(5.1, 2.0), 6.0);
}

TEST(RoundTest, NegativesNoTies) {
  EXPECT_EQ(RoundToNearestMultiple(-4.9, 2.0), -4.0);
  EXPECT_EQ(RoundToNearestMultiple(-5.1, 2.0), -6.0);
}

TEST(RoundTest, PositiveTies) {
  EXPECT_EQ(RoundToNearestMultiple(5.0, 2.0), 6.0);
}

TEST(RoundTest, NegativeTies) {
  EXPECT_EQ(RoundToNearestMultiple(-5.0, 2.0), -4.0);
}

TEST(RoundTest, NegativePowerOf2) {
  EXPECT_EQ(RoundToNearestMultiple(0.2078795763, 0.25), 0.25);
  EXPECT_EQ(RoundToNearestMultiple(0.1, 1.0 / (1 << 10)), 0.099609375);
  EXPECT_EQ(RoundToNearestMultiple(0.3, 1.0 / (1 << 30)),
            322122547.0 / (1 << 30));
}

TEST(QnormTest, InvalidProbability) {
  EXPECT_EQ(Qnorm(-0.1).status().code(), base::StatusCode::kInvalidArgument);
  EXPECT_EQ(Qnorm(0).status().code(), base::StatusCode::kInvalidArgument);
  EXPECT_EQ(Qnorm(1).status().code(), base::StatusCode::kInvalidArgument);
  EXPECT_EQ(Qnorm(2).status().code(), base::StatusCode::kInvalidArgument);
}
TEST(QnormTest, Accuracy) {
  double theoretical_accuracy = 4.5 * std::pow(10, -4);
  std::vector<double> p = {0.0000001, 0.00001, 0.001,   0.05,     0.15, 0.25,
                           0.35,      0.45,    0.55,    0.65,     0.75, 0.85,
                           0.95,      0.999,   0.99999, 0.9999999};
  std::vector<double> exact = {
      -5.199337582187471,   -4.264890793922602,   -3.090232306167813,
      -1.6448536269514729,  -1.0364333894937896,  -0.6744897501960817,
      -0.38532046640756773, -0.12566134685507402, 0.12566134685507402,
      0.38532046640756773,  0.6744897501960817,   1.0364333894937896,
      1.6448536269514729,   3.090232306167813,    4.264890793922602,
      5.199337582187471};
  for (int i = 0; i < p.size(); ++i) {
    EXPECT_LE(std::abs(exact[i] - Qnorm(p[i]).ValueOrDie()),
              theoretical_accuracy);
  }
}

TEST(ClampTest, DefaultTest) {
  EXPECT_EQ(Clamp(1, 3, 2), 2);
  EXPECT_EQ(Clamp(1.0, 3.0, 4.0), 3);
  EXPECT_EQ(Clamp(1.0, 3.0, -2.0), 1);
}

TEST(SafeOperationsTest, SafeAdd) {
  int64_t int_result;
  EXPECT_TRUE(SafeAdd<int64_t>(10, 20, &int_result));
  EXPECT_EQ(int_result, 30);
  EXPECT_TRUE(SafeAdd<int64_t>(std::numeric_limits<int64_t>::max(),
                             std::numeric_limits<int64_t>::lowest(),
                             &int_result));
  EXPECT_EQ(int_result, -1);
  EXPECT_FALSE(
      SafeAdd<int64_t>(std::numeric_limits<int64_t>::max(), 1, &int_result));
  EXPECT_FALSE(
      SafeAdd<int64_t>(std::numeric_limits<int64_t>::lowest(), -1, &int_result));
  EXPECT_TRUE(
      SafeAdd<int64_t>(std::numeric_limits<int64_t>::lowest(), 0, &int_result));
  EXPECT_EQ(int_result, std::numeric_limits<int64_t>::lowest());
}

TEST(SafeOperationsTest, SafeSubtract) {
  int64_t int_result;
  EXPECT_TRUE(SafeSubtract<int64_t>(10, 20, &int_result));
  EXPECT_EQ(int_result, -10);
  EXPECT_FALSE(SafeSubtract<int64_t>(1, std::numeric_limits<int64_t>::lowest(),
                                   &int_result));
  EXPECT_TRUE(SafeSubtract<int64_t>(-1, std::numeric_limits<int64_t>::lowest(),
                                  &int_result));
  EXPECT_EQ(int_result, std::numeric_limits<int64_t>::max());
  EXPECT_TRUE(SafeSubtract<int64_t>(std::numeric_limits<int64_t>::lowest(),
                                  std::numeric_limits<int64_t>::lowest(),
                                  &int_result));
  EXPECT_EQ(int_result, 0);

  uint64_t uint_result;
  EXPECT_TRUE(SafeSubtract<uint64_t>(1, std::numeric_limits<uint64_t>::lowest(),
                                   &uint_result));
  EXPECT_EQ(uint_result, 1);
}

TEST(SafeOperationsTest, SafeSquare) {
  int64_t int_result;
  EXPECT_TRUE(SafeSquare<int64_t>(-9, &int_result));
  EXPECT_EQ(int_result, 81);
  EXPECT_FALSE(
      SafeSquare<int64_t>(std::numeric_limits<int64_t>::max() - 1, &int_result));
  EXPECT_FALSE(
      SafeSquare<int64_t>(std::numeric_limits<int64_t>::lowest() + 1, &int_result));
  EXPECT_FALSE(
      SafeSquare<int64_t>(std::numeric_limits<int64_t>::lowest(), &int_result));

  uint64_t uint_result;
  EXPECT_TRUE(
      SafeSquare<uint64_t>(std::numeric_limits<uint64_t>::lowest(), &uint_result));
}

TEST(StatisticsTest, VectorStatistics) {
  std::vector<double> a = {1, 5, 7, 9, 13};
  EXPECT_EQ(Mean(a), 7);
  EXPECT_EQ(Variance(a), 16);
  EXPECT_EQ(StandardDev(a), 4);
  EXPECT_EQ(OrderStatistic(.60, a), 8);
  EXPECT_EQ(OrderStatistic(0, a), 1);
  EXPECT_EQ(OrderStatistic(1, a), 13);
}

TEST(VectorUtilTest, VectorFilter) {
  std::vector<double> v = {1, 2, 2, 3};
  std::vector<bool> selection = {false, true, true, false};
  std::vector<double> expected = {2, 2};
  EXPECT_THAT(VectorFilter(v, selection), testing::ContainerEq(expected));
}

TEST(VectorUtilTest, VectorToString) {
  std::vector<double> v = {1, 2, 2, 3};
  EXPECT_EQ(VectorToString(v), "[1, 2, 2, 3]");
}

}  // namespace
}  // namespace differential_privacy
