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
#include <optional>
#include <vector>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "algorithms/distributions.h"
#include "algorithms/numerical-mechanisms-testing.h"

namespace differential_privacy {
namespace {

using ::testing::HasSubstr;
using ::differential_privacy::base::testing::StatusIs;

const char kSeedString[] = "ABCDEFGHIJKLMNOP";
constexpr int64_t kStatsSize = 50000;
constexpr double kTolerance = 1e-5;

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
TEST(RoundDoubleTest, PositiveNoTies) {
  EXPECT_EQ(RoundToNearestMultiple(4.9, 2.0), 4.0);
  EXPECT_EQ(RoundToNearestMultiple(5.1, 2.0), 6.0);
}

TEST(RoundDoubleTest, NegativesNoTies) {
  EXPECT_EQ(RoundToNearestMultiple(-4.9, 2.0), -4.0);
  EXPECT_EQ(RoundToNearestMultiple(-5.1, 2.0), -6.0);
}

TEST(RoundDoubleTest, PositiveTies) {
  EXPECT_EQ(RoundToNearestMultiple(5.0, 2.0), 6.0);
}

TEST(RoundDoubleTest, NegativeTies) {
  EXPECT_EQ(RoundToNearestMultiple(-5.0, 2.0), -4.0);
}

TEST(RoundDoubleTest, NegativePowerOf2) {
  EXPECT_EQ(RoundToNearestMultiple(0.2078795763, 0.25), 0.25);
  EXPECT_EQ(RoundToNearestMultiple(0.1, 1.0 / (1 << 10)), 0.099609375);
  EXPECT_EQ(RoundToNearestMultiple(0.3, 1.0 / (1 << 30)),
            322122547.0 / (1 << 30));
}

TEST(RoundInt64Test, PositiveNoTies) {
  EXPECT_EQ(RoundToNearestMultiple(7, 3), 6);
  EXPECT_EQ(RoundToNearestMultiple(8, 3), 9);
  EXPECT_EQ(RoundToNearestMultiple(9, 4), 8);
  EXPECT_EQ(RoundToNearestMultiple(11, 4), 12);
  EXPECT_EQ(RoundToNearestMultiple(10, 5), 10);
  EXPECT_EQ(RoundToNearestMultiple(11, 5), 10);
  EXPECT_EQ(RoundToNearestMultiple(12, 5), 10);
  EXPECT_EQ(RoundToNearestMultiple(13, 5), 15);
  EXPECT_EQ(RoundToNearestMultiple(14, 5), 15);
  EXPECT_EQ(RoundToNearestMultiple(15, 5), 15);
  EXPECT_EQ(RoundToNearestMultiple(14, 7), 14);
  EXPECT_EQ(RoundToNearestMultiple(15, 7), 14);
  EXPECT_EQ(RoundToNearestMultiple(16, 7), 14);
  EXPECT_EQ(RoundToNearestMultiple(17, 7), 14);
  EXPECT_EQ(RoundToNearestMultiple(18, 7), 21);
  EXPECT_EQ(RoundToNearestMultiple(19, 7), 21);
  EXPECT_EQ(RoundToNearestMultiple(20, 7), 21);
  EXPECT_EQ(RoundToNearestMultiple(21, 7), 21);
}

TEST(RoundInt64Test, PositiveTies) {
  EXPECT_EQ(RoundToNearestMultiple(5, 2), 6);
  EXPECT_EQ(RoundToNearestMultiple(10, 4), 12);
  EXPECT_EQ(RoundToNearestMultiple(9, 6), 12);
  EXPECT_EQ(RoundToNearestMultiple(12, 8), 16);
  EXPECT_EQ(RoundToNearestMultiple(15, 10), 20);
}

TEST(RoundInt64Test, NegativeNoTies) {
  EXPECT_EQ(RoundToNearestMultiple(-7, 3), -6);
  EXPECT_EQ(RoundToNearestMultiple(-8, 3), -9);
  EXPECT_EQ(RoundToNearestMultiple(-9, 4), -8);
  EXPECT_EQ(RoundToNearestMultiple(-11, 4), -12);
}

TEST(RoundInt64Test, NegativeTies) {
  EXPECT_EQ(RoundToNearestMultiple(-5, 2), -4);
  EXPECT_EQ(RoundToNearestMultiple(-10, 4), -8);
}

TEST(RoundInt64Test, LargeValues) {
  const int64_t k2Pow29 = 1 << 29;
  const int64_t k2Pow30 = 1 << 30;

  EXPECT_EQ(RoundToNearestMultiple(k2Pow29, k2Pow30), k2Pow30);
  // Expect 0 since ties are broken towards +inf
  EXPECT_EQ(RoundToNearestMultiple(-k2Pow29, k2Pow30), 0);

  EXPECT_EQ(RoundToNearestMultiple(k2Pow30 - 5, k2Pow30), k2Pow30);
  EXPECT_EQ(RoundToNearestMultiple(-k2Pow30 + 5, k2Pow30), -k2Pow30);
}

TEST(QnormTest, InvalidProbability) {
  EXPECT_EQ(Qnorm(-0.1).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Qnorm(0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Qnorm(1).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Qnorm(2).status().code(), absl::StatusCode::kInvalidArgument);
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
    EXPECT_LE(std::abs(exact[i] - Qnorm(p[i]).value()), theoretical_accuracy);
  }
}

TEST(ClampTest, DefaultTest) {
  EXPECT_EQ(Clamp(1, 3, 2), 2);
  EXPECT_EQ(Clamp(1.0, 3.0, 4.0), 3);
  EXPECT_EQ(Clamp(1.0, 3.0, -2.0), 1);
}

TEST(SafeOperationsTest, SafeAddInt) {
  SafeOpResult<int64_t> safe_add_result;

  safe_add_result = SafeAdd<int64_t>(10, 20);
  EXPECT_FALSE(safe_add_result.overflow);
  EXPECT_EQ(safe_add_result.value, 30);

  safe_add_result = SafeAdd<int64_t>(std::numeric_limits<int64_t>::max(),
                                     std::numeric_limits<int64_t>::lowest());
  EXPECT_FALSE(safe_add_result.overflow);
  EXPECT_EQ(safe_add_result.value, -1);

  safe_add_result = SafeAdd<int64_t>(std::numeric_limits<int64_t>::max(), 1);
  EXPECT_TRUE(safe_add_result.overflow);
  EXPECT_EQ(safe_add_result.value, std::numeric_limits<int64_t>::max());

  safe_add_result =
      SafeAdd<int64_t>(std::numeric_limits<int64_t>::lowest(), -1);
  EXPECT_TRUE(safe_add_result.overflow);
  EXPECT_EQ(safe_add_result.value, std::numeric_limits<int64_t>::lowest());

  safe_add_result = SafeAdd<int64_t>(std::numeric_limits<int64_t>::lowest(), 0);
  EXPECT_FALSE(safe_add_result.overflow);
  EXPECT_EQ(safe_add_result.value, std::numeric_limits<int64_t>::lowest());
}

TEST(SafeOperationsTest, SafeAddDouble) {
  SafeOpResult<double> safe_add_result;

  safe_add_result = SafeAdd<double>(10, 20);
  EXPECT_FALSE(safe_add_result.overflow);
  EXPECT_EQ(safe_add_result.value, 30);

  safe_add_result = SafeAdd<double>(std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::lowest());
  EXPECT_FALSE(safe_add_result.overflow);
  EXPECT_DOUBLE_EQ(safe_add_result.value, 0);

  safe_add_result = SafeAdd<double>(std::numeric_limits<double>::max(), 1.0);
  EXPECT_FALSE(safe_add_result.overflow);
  EXPECT_DOUBLE_EQ(safe_add_result.value,
                   std::numeric_limits<double>::infinity());

  safe_add_result =
      SafeAdd<double>(std::numeric_limits<double>::lowest(), -1.0);
  EXPECT_FALSE(safe_add_result.overflow);
  EXPECT_DOUBLE_EQ(safe_add_result.value,
                   -std::numeric_limits<double>::infinity());

  safe_add_result = SafeAdd<double>(std::numeric_limits<double>::lowest(), 0.0);
  EXPECT_FALSE(safe_add_result.overflow);
  EXPECT_DOUBLE_EQ(safe_add_result.value,
                   std::numeric_limits<double>::lowest());
}

TEST(SafeOperationsTest, SafeSubtractInt) {
  SafeOpResult<int64_t> safe_subtract_int64_result;

  safe_subtract_int64_result = SafeSubtract<int64_t>(10, 20);
  EXPECT_FALSE(safe_subtract_int64_result.overflow);
  EXPECT_EQ(safe_subtract_int64_result.value, -10);

  safe_subtract_int64_result =
      SafeSubtract<int64_t>(0, std::numeric_limits<int64_t>::lowest());
  EXPECT_TRUE(safe_subtract_int64_result.overflow);
  EXPECT_EQ(safe_subtract_int64_result.value,
            std::numeric_limits<int64_t>::max());

  safe_subtract_int64_result =
      SafeSubtract<int64_t>(1, std::numeric_limits<int64_t>::lowest());
  EXPECT_TRUE(safe_subtract_int64_result.overflow);
  EXPECT_EQ(safe_subtract_int64_result.value,
            std::numeric_limits<int64_t>::max());

  safe_subtract_int64_result =
      SafeSubtract<int64_t>(-1, std::numeric_limits<int64_t>::lowest());
  EXPECT_FALSE(safe_subtract_int64_result.overflow);
  EXPECT_EQ(safe_subtract_int64_result.value,
            std::numeric_limits<int64_t>::max());

  safe_subtract_int64_result =
      SafeSubtract<int64_t>(std::numeric_limits<int64_t>::lowest(),
                            std::numeric_limits<int64_t>::lowest());
  EXPECT_FALSE(safe_subtract_int64_result.overflow);
  EXPECT_EQ(safe_subtract_int64_result.value, 0);

  SafeOpResult<uint64_t> safe_subtract_uint64_result =
      SafeSubtract<uint64_t>(1, std::numeric_limits<uint64_t>::lowest());
  EXPECT_FALSE(safe_subtract_uint64_result.overflow);
  EXPECT_EQ(safe_subtract_uint64_result.value, 1);
}

TEST(SafeOperationsTest, SafeSubtractDouble) {
  SafeOpResult<double> safe_subtract_result;

  safe_subtract_result = SafeSubtract<double>(10.0, 20.0);
  EXPECT_FALSE(safe_subtract_result.overflow);
  EXPECT_DOUBLE_EQ(safe_subtract_result.value, -10.0);

  safe_subtract_result =
      SafeSubtract<double>(1.0, std::numeric_limits<double>::lowest());
  EXPECT_FALSE(safe_subtract_result.overflow);
  EXPECT_DOUBLE_EQ(safe_subtract_result.value,
                   std::numeric_limits<double>::infinity());

  safe_subtract_result =
      SafeSubtract<double>(-1.0, std::numeric_limits<double>::lowest());
  EXPECT_FALSE(safe_subtract_result.overflow);
  EXPECT_DOUBLE_EQ(safe_subtract_result.value,
                   std::numeric_limits<double>::infinity());

  safe_subtract_result =
      SafeSubtract<double>(std::numeric_limits<double>::lowest(),
                           std::numeric_limits<double>::lowest());
  EXPECT_FALSE(safe_subtract_result.overflow);
  EXPECT_DOUBLE_EQ(safe_subtract_result.value, 0);
}

TEST(SafeOperationsTest, SafeSquare) {
  SafeOpResult<int64_t> safe_square_int64_result;

  safe_square_int64_result = SafeSquare<int64_t>(-9);
  EXPECT_FALSE(safe_square_int64_result.overflow);
  EXPECT_EQ(safe_square_int64_result.value, 81);

  safe_square_int64_result =
      SafeSquare<int64_t>(std::numeric_limits<int64_t>::max() - 1);
  EXPECT_TRUE(safe_square_int64_result.overflow);
  EXPECT_EQ(safe_square_int64_result.value, 0);

  safe_square_int64_result =
      SafeSquare<int64_t>(std::numeric_limits<int64_t>::lowest() + 1);
  EXPECT_TRUE(safe_square_int64_result.overflow);
  EXPECT_EQ(safe_square_int64_result.value, 0);

  safe_square_int64_result =
      SafeSquare<int64_t>(std::numeric_limits<int64_t>::lowest());
  EXPECT_TRUE(safe_square_int64_result.overflow);
  EXPECT_EQ(safe_square_int64_result.value, 0);

  SafeOpResult<uint64_t> safe_square_uint64_result =
      SafeSquare<uint64_t>(std::numeric_limits<uint64_t>::lowest());
  EXPECT_FALSE(safe_square_uint64_result.overflow);
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

TEST(StatisticTest, VectorStatistics) {
  std::vector<double> a;
  EXPECT_EQ(Mean(a), 0);
  EXPECT_EQ(Variance(a), 0);
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

TEST(SafeCastFromDoubleTest, Converts20ToIntegral) {
  SafeOpResult<int64_t> cast_result = SafeCastFromDouble<int64_t>(20.0);
  EXPECT_EQ(cast_result.value, 20);
  EXPECT_FALSE(cast_result.overflow);
}

TEST(SafeCastFromDoubleTest, ConvertsMaxValueToMaxIntegral) {
  SafeOpResult<int64_t> cast_result =
      SafeCastFromDouble<int64_t>(std::numeric_limits<int64_t>::max());
  EXPECT_EQ(cast_result.value, std::numeric_limits<int64_t>::max());
  EXPECT_FALSE(cast_result.overflow);
}

TEST(SafeCastFromDoubleTest, ConvertsMinDoubleValuesToZero) {
  SafeOpResult<int64_t> cast_result =
      SafeCastFromDouble<int64_t>(std::numeric_limits<double>::min());
  EXPECT_EQ(cast_result.value, 0);
  EXPECT_FALSE(cast_result.overflow);

  cast_result =
      SafeCastFromDouble<int64_t>(-std::numeric_limits<double>::min());
  EXPECT_EQ(cast_result.value, 0);
  EXPECT_FALSE(cast_result.overflow);
}

TEST(SafeCastFromDoubleTest, ConvertsLowestValueToLowestIntegral) {
  SafeOpResult<int64_t> cast_result =
      SafeCastFromDouble<int64_t>(std::numeric_limits<int64_t>::lowest());
  EXPECT_EQ(cast_result.value, std::numeric_limits<int64_t>::lowest());
  EXPECT_FALSE(cast_result.overflow);
}

TEST(SafeCastFromDoubleTest, ConvertsOverLimitValueToOverflowedIntegral) {
  std::vector<double> overflow_values = {
      1,
      2,
      3,
      4,
      5,
      10,
      1234,
      static_cast<double>(std::numeric_limits<int16_t>::max() - 2),
      static_cast<double>(std::numeric_limits<int16_t>::max() - 1),
      static_cast<double>(std::numeric_limits<int16_t>::max())};
  SafeOpResult<int16_t> cast_result;
  for (double overflow : overflow_values) {
    cast_result = SafeCastFromDouble<int16_t>(
        static_cast<double>(std::numeric_limits<int16_t>::max()) + overflow);
    EXPECT_EQ(cast_result.value,
              std::numeric_limits<int16_t>::lowest() - 1 + overflow);
    EXPECT_TRUE(cast_result.overflow);
  }
}

TEST(SafeCastFromDoubleTest, ConvertsUnderLimitValueToUnderflowedIntegral) {
  std::vector<double> overflow_values = {
      1,
      2,
      3,
      4,
      5,
      10,
      1234,
      static_cast<double>(std::numeric_limits<int16_t>::max() - 2),
      static_cast<double>(std::numeric_limits<int16_t>::max() - 1),
      static_cast<double>(std::numeric_limits<int16_t>::max())};
  SafeOpResult<int16_t> cast_result;
  for (double overflow : overflow_values) {
    cast_result = SafeCastFromDouble<int16_t>(
        static_cast<double>(std::numeric_limits<int16_t>::lowest()) - overflow);
    EXPECT_EQ(cast_result.value,
              std::numeric_limits<int16_t>::max() + 1 - overflow);
    EXPECT_TRUE(cast_result.overflow);
  }
}

TEST(SafeCastFromDoubleTest, ConvertsHighValueToOverflowedIntegral) {
  SafeOpResult<int64_t> cast_result = SafeCastFromDouble<int64_t>(1.0e200);
  EXPECT_LE(cast_result.value, std::numeric_limits<int64_t>::max());
  EXPECT_TRUE(cast_result.overflow);
}

TEST(SafeCastFromDoubleTest, ConvertsLowValueToUnderflowedIntegral) {
  SafeOpResult<int64_t> cast_result = SafeCastFromDouble<int64_t>(-1.0e200);
  EXPECT_GE(cast_result.value, std::numeric_limits<int64_t>::lowest());
  EXPECT_TRUE(cast_result.overflow);
}

TEST(SafeCastFromDoubleTest, ReturnsFalseOnNanForIntegrals) {
  SafeOpResult<int64_t> cast_result = SafeCastFromDouble<int64_t>(NAN);
  EXPECT_EQ(cast_result.value, std::numeric_limits<int64_t>::quiet_NaN());
  EXPECT_TRUE(cast_result.overflow);
}

// Combine all tests for float outputs.  Should be nothing unexpected here since
// this is just a cast from double to float.
TEST(SafeCastFromDoubleTest, ForFloat) {
  SafeOpResult<float> cast_result;

  // Normal case.
  cast_result = SafeCastFromDouble<float>(0.5);
  EXPECT_EQ(cast_result.value, 0.5);
  EXPECT_FALSE(cast_result.overflow);

  // NaN double should convert into NaN float.
  cast_result = SafeCastFromDouble<float>(NAN);
  EXPECT_TRUE(std::isnan(cast_result.value));
  EXPECT_FALSE(cast_result.overflow);

  // High double should convert into infinite float.
  cast_result = SafeCastFromDouble<float>(1.0e200);
  EXPECT_TRUE(std::isinf(cast_result.value));
  EXPECT_FALSE(cast_result.overflow);
}

TEST(ValidateTest, IsSet) {
  std::optional<double> opt;
  EXPECT_THAT(ValidateIsSet(opt, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be set.")));

  opt = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THAT(ValidateIsSet(opt, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be a valid numeric value")));

  std::vector<double> success_values = {
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::lowest(),
      -1,
      0,
      std::numeric_limits<double>::min(),
      1,
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity()};

  for (double value : success_values) {
    EXPECT_OK(ValidateIsSet(value, "Test value"));
  }
}

TEST(ValidateTest, IsPositive) {
  std::vector<double> success_values = {
      std::numeric_limits<double>::min(), 1, std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity()};
  std::vector<double> error_values = {-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::lowest(),
                                      -10, -1, 0};

  for (double value : success_values) {
    EXPECT_OK(ValidateIsPositive(value, "Test value"));
  }

  for (double value : error_values) {
    EXPECT_THAT(ValidateIsPositive(value, "Test value"),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Test value must be positive")));
  }
}

TEST(ValidateTest, IsNonNegative) {
  std::vector<double> success_values = {
      0, std::numeric_limits<double>::min(), 1,
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity()};
  std::vector<double> error_values = {-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::lowest(),
                                      -10, -1};

  for (double value : success_values) {
    EXPECT_OK(ValidateIsNonNegative(value, "Test value"));
  }

  for (double value : error_values) {
    EXPECT_THAT(ValidateIsNonNegative(value, "Test value"),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Test value must be non-negative")));
  }
}

TEST(ValidateTest, IsFinite) {
  std::vector<double> success_values = {std::numeric_limits<double>::lowest(),
                                        -1,
                                        0,
                                        std::numeric_limits<double>::min(),
                                        1,
                                        std::numeric_limits<double>::max()};

  std::vector<double> error_values = {-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()};

  for (double value : success_values) {
    EXPECT_OK(ValidateIsFinite(value, "Test value"));
  }

  for (double value : error_values) {
    EXPECT_THAT(ValidateIsFinite(value, "Test value"),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Test value must be finite")));
  }
}

TEST(ValidateTest, IsFiniteAndPositive) {
  std::vector<double> success_values = {std::numeric_limits<double>::min(), 1,
                                        std::numeric_limits<double>::max()};
  std::vector<double> error_values = {-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::lowest(),
                                      -10,
                                      -1,
                                      0,
                                      std::numeric_limits<double>::infinity()};

  for (double value : success_values) {
    EXPECT_OK(ValidateIsFiniteAndPositive(value, "Test value"));
  }

  for (double value : error_values) {
    EXPECT_THAT(ValidateIsFiniteAndPositive(value, "Test value"),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Test value must be finite and positive")));
  }
}

TEST(ValidateTest, IsFiniteAndNonNegative) {
  std::vector<double> success_values = {0, std::numeric_limits<double>::min(),
                                        1, std::numeric_limits<double>::max()};
  std::vector<double> error_values = {-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::lowest(),
                                      -10, -1,
                                      std::numeric_limits<double>::infinity()};

  for (double value : success_values) {
    EXPECT_OK(ValidateIsFiniteAndNonNegative(value, "Test value"));
  }

  for (double value : error_values) {
    EXPECT_THAT(
        ValidateIsFiniteAndNonNegative(value, "Test value"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Test value must be finite and non-negative")));
  }
}

TEST(ValidateTest, IsLesserThanOkStatus) {
  struct LesserThanParams {
    double value;
    double upper_bound;
  };

  std::vector<LesserThanParams> success_params = {
      {-std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::lowest()},
      {-1, 1},
      {0, std::numeric_limits<double>::min()},
      {std::numeric_limits<double>::max(),
       std::numeric_limits<double>::infinity()},
  };

  for (LesserThanParams params : success_params) {
    EXPECT_OK(
        ValidateIsLesserThan(params.value, params.upper_bound, "Test value"));
  }
}

TEST(ValidateTest, IsLesserThanError) {
  struct LesserThanParams {
    double value;
    double upper_bound;
  };

  std::vector<LesserThanParams> no_equal_error_params = {
      {-std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity()},
      {std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest()},
      {-1, -1},
      {std::numeric_limits<double>::min(), std::numeric_limits<double>::min()},
      {0, 0},
      {1, -1},
      {1, 1},
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity()}};

  for (LesserThanParams params : no_equal_error_params) {
    EXPECT_THAT(
        ValidateIsLesserThan(params.value, params.upper_bound, "Test value"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Test value must be lesser than")));
  }
}

TEST(ValidateTest, IsLesserThanUnsetError) {
  std::optional<double> test_unset;

  EXPECT_THAT(ValidateIsLesserThan(test_unset, 1, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be set")));
}

TEST(ValidateTest, IsLesserThanNaNError) {
  EXPECT_THAT(ValidateIsLesserThan(std::numeric_limits<double>::quiet_NaN(), 1,
                                   "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be a valid numeric value")));
}

TEST(ValidateTest, IsLesserThanOrEqualToOkStatus) {
  struct LesserThanParams {
    double value;
    double upper_bound;
  };

  std::vector<LesserThanParams> success_params = {
      {-std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity()},
      {std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest()},
      {-1, -1},
      {-1, 1},
      {0, 0},
      {std::numeric_limits<double>::min(), std::numeric_limits<double>::min()},
      {
          1,
          1,
      },
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity()}};

  for (LesserThanParams params : success_params) {
    EXPECT_OK(ValidateIsLesserThanOrEqualTo(params.value, params.upper_bound,
                                            "Test value"));
  }
}

TEST(ValidateTest, IsLesserThanOrEqualToError) {
  struct LesserThanParams {
    double value;
    double upper_bound;
  };

  std::vector<LesserThanParams> or_equal_error_params = {
      {std::numeric_limits<double>::lowest(),
       -std::numeric_limits<double>::infinity()},
      {std::numeric_limits<double>::min(), 0},
      {1, -1},
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::max()}};

  for (LesserThanParams params : or_equal_error_params) {
    EXPECT_THAT(
        ValidateIsLesserThanOrEqualTo(params.value, params.upper_bound,
                                      "Test value"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Test value must be lesser than or equal to")));
  }
}

TEST(ValidateTest, IsLesserThanOrEqualToUnsetError) {
  std::optional<double> test_unset;

  EXPECT_THAT(ValidateIsLesserThanOrEqualTo(test_unset, 1, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be set")));
}

TEST(ValidateTest, IsLesserThanOrEqualToNaNError) {
  EXPECT_THAT(ValidateIsLesserThanOrEqualTo(
                  std::numeric_limits<double>::quiet_NaN(), 1, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be a valid numeric value")));
}

TEST(ValidateTest, IsGreaterThanOkStatus) {
  struct GreaterThanParams {
    double value;
    double lower_bound;
  };

  std::vector<GreaterThanParams> success_params = {
      {std::numeric_limits<double>::lowest(),
       -std::numeric_limits<double>::infinity()},
      {std::numeric_limits<double>::min(), 0},
      {1, -1},
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::max()},
  };

  for (GreaterThanParams params : success_params) {
    EXPECT_OK(
        ValidateIsGreaterThan(params.value, params.lower_bound, "Test value"));
  }
}

TEST(ValidateTest, IsGreaterThanError) {
  struct GreaterThanParams {
    double value;
    double lower_bound;
  };

  std::vector<GreaterThanParams> no_equal_error_params = {
      {-std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity()},
      {std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest()},
      {-1, -1},
      {std::numeric_limits<double>::min(), std::numeric_limits<double>::min()},
      {0, 0},
      {-1, 1},
      {1, 1},
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity()}};

  for (GreaterThanParams params : no_equal_error_params) {
    EXPECT_THAT(
        ValidateIsGreaterThan(params.value, params.lower_bound, "Test value"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Test value must be greater than")));
  }
}

TEST(ValidateTest, IsGreaterThanUnsetError) {
  std::optional<double> test_unset;

  EXPECT_THAT(ValidateIsGreaterThan(test_unset, 1, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be set")));
}

TEST(ValidateTest, IsGreaterThanNaNError) {
  EXPECT_THAT(ValidateIsGreaterThan(std::numeric_limits<double>::quiet_NaN(), 1,
                                    "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be a valid numeric value")));
}

TEST(ValidateTest, IsGreaterThanOrEqualToOkStatus) {
  struct GreaterThanParams {
    double value;
    double lower_bound;
  };

  std::vector<GreaterThanParams> success_params = {
      {-std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity()},
      {std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest()},
      {-1, -1},
      {0, 0},
      {1, -1},
      {std::numeric_limits<double>::min(), std::numeric_limits<double>::min()},
      {1, 1},
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity()}};

  for (GreaterThanParams params : success_params) {
    EXPECT_OK(ValidateIsGreaterThanOrEqualTo(params.value, params.lower_bound,
                                             "Test value"));
  }
}

TEST(ValidateTest, IsGreaterThanOrEqualToError) {
  struct GreaterThanParams {
    double value;
    double lower_bound;
  };

  std::vector<GreaterThanParams> or_equal_error_params = {
      {-std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::lowest()},
      {0, std::numeric_limits<double>::min()},
      {-1, 1},
      {std::numeric_limits<double>::max(),
       std::numeric_limits<double>::infinity()}};

  for (GreaterThanParams params : or_equal_error_params) {
    EXPECT_THAT(
        ValidateIsGreaterThanOrEqualTo(params.value, params.lower_bound,
                                       "Test value"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Test value must be greater than or equal to")));
  }
}

TEST(ValidateTest, IsGreaterThanOrEqualToUnsetError) {
  std::optional<double> test_unset;

  EXPECT_THAT(ValidateIsGreaterThanOrEqualTo(test_unset, 1, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be set")));
}

TEST(ValidateTest, IsGreaterThanOrEqualToNaNError) {
  EXPECT_THAT(ValidateIsGreaterThanOrEqualTo(
                  std::numeric_limits<double>::quiet_NaN(), 1, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be a valid numeric value")));
}

TEST(ValidateTest, IsInIntervalOkStatus) {
  struct IntervalParams {
    double value;
    double lower_bound;
    double upper_bound;
    bool include_lower;
    bool include_upper;
  };

  std::vector<IntervalParams> success_params = {
      {std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest(), false, true},
      {std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest(), true, false},
      {std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest(), true, true},
      {0, -1, 1, false, false},
      {0, -1, 1, true, false},
      {0, -1, 1, false, true},
      {0, -1, 1, true, true},
      {0, 0, 0, false, true},
      {0, 0, 0, true, false},
      {0, 0, 0, true, true},
      {0.0, 0.0 - std::numeric_limits<double>::min(),
       0.0 + std::numeric_limits<double>::min(), false, false},
      {-1, -1, 1, true, false},
      {1, -1, 1, false, true},
      {1, 1, 1, false, true},
      {1, 1, 1, true, false},
      {1, 1, 1, true, true},
      {std::numeric_limits<double>::min(), std::numeric_limits<double>::min(),
       std::numeric_limits<double>::min(), false, true},
      {std::numeric_limits<double>::min(), std::numeric_limits<double>::min(),
       std::numeric_limits<double>::min(), true, false},
      {std::numeric_limits<double>::min(), std::numeric_limits<double>::min(),
       std::numeric_limits<double>::min(), true, true},
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
       std::numeric_limits<double>::max(), false, true},
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
       std::numeric_limits<double>::max(), true, false},
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
       std::numeric_limits<double>::max(), true, true},
  };

  for (IntervalParams params : success_params) {
    EXPECT_OK(ValidateIsInInterval(params.value, params.lower_bound,
                                   params.upper_bound, params.include_lower,
                                   params.include_upper, "Test value"));
  }
}

TEST(ValidateTest, IsOutsideExclusiveInterval) {
  struct IntervalParams {
    double value;
    double lower_bound;
    double upper_bound;
    bool include_lower;
    bool include_upper;
  };

  std::vector<IntervalParams> exclusive_error_params = {
      {std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest(),
       std::numeric_limits<double>::lowest(), false, false},
      {-1, 0, 1, false, false},
      {-1, -1, -1, false, false},
      {0, 0, 0, false, false},
      {1, 1, 1, false, false},
      {std::numeric_limits<double>::min(), std::numeric_limits<double>::min(),
       std::numeric_limits<double>::min(), false, false},
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
       std::numeric_limits<double>::max(), false, false},
  };

  for (IntervalParams params : exclusive_error_params) {
    EXPECT_THAT(
        ValidateIsInInterval(params.value, params.lower_bound,
                             params.upper_bound, params.include_lower,
                             params.include_upper, "Test value"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Test value must be in the exclusive interval (")));
  }
}

TEST(ValidateTest, IsOutsideInclusiveInterval) {
  struct IntervalParams {
    double value;
    double lower_bound;
    double upper_bound;
    bool include_lower;
    bool include_upper;
  };

  std::vector<IntervalParams> inclusive_error_params = {
      {-1, 0, 1, true, true},
      {0 - std::numeric_limits<double>::min(), 0,
       std::numeric_limits<double>::min(), true, true},
  };

  for (IntervalParams params : inclusive_error_params) {
    EXPECT_THAT(
        ValidateIsInInterval(params.value, params.lower_bound,
                             params.upper_bound, params.include_lower,
                             params.include_upper, "Test value"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Test value must be in the inclusive interval [")));
  }
}

TEST(ValidateTest, IsOutsideHalfClosedInterval) {
  struct IntervalParams {
    double value;
    double lower_bound;
    double upper_bound;
    bool include_lower;
    bool include_upper;
  };

  EXPECT_THAT(ValidateIsInInterval(-1, 0, 1, true, false, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be in the interval [0,1)")));

  EXPECT_THAT(ValidateIsInInterval(-1, 0, 1, false, true, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be in the interval (0,1]")));

  EXPECT_THAT(ValidateIsInInterval(-1, -1, 1, false, true, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be in the interval (-1,1]")));

  EXPECT_THAT(ValidateIsInInterval(1, -1, 1, true, false, "Test value"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Test value must be in the interval [-1,1)")));
}

// These tests document cases that result in known, incorrect behaviour
TEST(ValidateTest, IsInIntervalBadBehaviour) {
  struct IntervalParams {
    double value;
    double lower_bound;
    double upper_bound;
    bool include_lower;
    bool include_upper;
  };

  std::vector<IntervalParams> bad_exclusive_error_params = {
      // These test parameters should result in an OK_STATUS since the value is
      // within the bounds, but instead returns a kInvalidArgument status
      // because of double (im)precision.
      {-1.0, -1.0 - std::numeric_limits<double>::min(),
       -1.0 + std::numeric_limits<double>::min(), false, false},
      {1.0, 1.0 - std::numeric_limits<double>::min(),
       1.0 + std::numeric_limits<double>::min(), false, false},
  };

  for (IntervalParams params : bad_exclusive_error_params) {
    EXPECT_THAT(
        ValidateIsInInterval(params.value, params.lower_bound,
                             params.upper_bound, params.include_lower,
                             params.include_upper, "Test value"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Test value must be in the exclusive interval (")));
  }

  std::vector<IntervalParams> bad_success_params = {
      // These test parameters should result in an kInvalidArgument status since
      // the value falls outside of the bounds, but instead returns an OK_STATUS
      // because of double (im)precision.
      {-1.0 - std::numeric_limits<double>::min(), -1.0,
       -1.0 + std::numeric_limits<double>::min(), true, true},
      {1.0 - std::numeric_limits<double>::min(), 1.0,
       1.0 + std::numeric_limits<double>::min(), true, true},
  };

  for (IntervalParams params : bad_success_params) {
    EXPECT_OK(ValidateIsInInterval(params.value, params.lower_bound,
                                   params.upper_bound, params.include_lower,
                                   params.include_upper, "Test value"));
  }
}

TEST(ValidateTest, ValidateEpsilonFailsForNegativeAndNan) {
  EXPECT_THAT(ValidateEpsilon(-1), StatusIs(absl::StatusCode::kInvalidArgument,
                                            HasSubstr("positive")));
  EXPECT_THAT(ValidateEpsilon(0), StatusIs(absl::StatusCode::kInvalidArgument,
                                           HasSubstr("positive")));
  EXPECT_THAT(
      ValidateEpsilon(std::numeric_limits<double>::infinity()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("finite")));
  EXPECT_THAT(ValidateEpsilon(std::numeric_limits<double>::quiet_NaN()),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("valid")));
  EXPECT_THAT(ValidateEpsilon(std::numeric_limits<double>::signaling_NaN()),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("valid")));
}

TEST(ValidateTest, ValidateEpsilonReturnsOkForPositive) {
  EXPECT_THAT(ValidateEpsilon(0.1), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateEpsilon(1.1), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateEpsilon(std::log(3) / 2),
              StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateEpsilon(std::log(3)), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateEpsilon(100), StatusIs(absl::StatusCode::kOk));
}

TEST(ValidateTest, ValidateDeltaFailsForOutOfRangeValues) {
  EXPECT_THAT(ValidateDelta(-1), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("inclusive interval")));
  EXPECT_THAT(ValidateDelta(1.1), StatusIs(absl::StatusCode::kInvalidArgument,
                                           HasSubstr("inclusive interval")));
  EXPECT_THAT(
      ValidateDelta(std::numeric_limits<double>::quiet_NaN()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("valid numeric")));
}

TEST(ValidateTest, ValidateDeltaReturnsOkForInRange) {
  EXPECT_THAT(ValidateDelta(0), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateDelta(1e-50), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateDelta(1e-8), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateDelta(0.2), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateDelta(1), StatusIs(absl::StatusCode::kOk));
}

TEST(ValidateTest, ValidateMaxPartitionsContributedFailsForNonPositive) {
  EXPECT_THAT(
      ValidateMaxPartitionsContributed(-1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
  EXPECT_THAT(
      ValidateMaxPartitionsContributed(0),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(ValidateTest, ValidateMaxPartitionsContributedReturnsOkForPositive) {
  EXPECT_THAT(ValidateMaxPartitionsContributed(1),
              StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateMaxPartitionsContributed(10),
              StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateMaxPartitionsContributed(100),
              StatusIs(absl::StatusCode::kOk));
}

TEST(ValidateTest, ValidateMaxContributions) {
  EXPECT_THAT(
      ValidateMaxContributions(-1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
  EXPECT_THAT(
      ValidateMaxContributions(0),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
  EXPECT_THAT(ValidateMaxPartitionsContributed(10),
              StatusIs(absl::StatusCode::kOk));
}

TEST(ValidateTest, ValidateMaxContributionsPerPartitionFailsForNonPositive) {
  EXPECT_THAT(
      ValidateMaxContributionsPerPartition(-1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
  EXPECT_THAT(
      ValidateMaxContributionsPerPartition(0),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(ValidateTest, ValidateMaxContributionsPerPartitionReturnsOkForPositive) {
  EXPECT_THAT(ValidateMaxContributionsPerPartition(1),
              StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateMaxContributionsPerPartition(10),
              StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateMaxContributionsPerPartition(100),
              StatusIs(absl::StatusCode::kOk));
}

TEST(ValidateTest, ValidateBoundsChecksOrder) {
  EXPECT_THAT(ValidateBounds<double>(1, 2), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateBounds<double>(-2, 2), StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(ValidateBounds<double>(-2, -1), StatusIs(absl::StatusCode::kOk));

  EXPECT_THAT(ValidateBounds<double>(2, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("bound cannot be greater")));
  EXPECT_THAT(ValidateBounds<double>(2, -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("bound cannot be greater")));
}

TEST(ValidateTest, ValidateBoundsChecksBothSetOrUnset) {
  EXPECT_THAT(ValidateBounds<double>(1, std::nullopt),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("both")));
  EXPECT_THAT(ValidateBounds<double>(std::nullopt, 1),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("both")));
  EXPECT_THAT(ValidateBounds<double>(std::nullopt, std::nullopt),
              StatusIs(absl::StatusCode::kOk));
}

TEST(ValidateTest, ValidateBoundsFailsForNanBounds) {
  EXPECT_THAT(
      ValidateBounds<double>(1, std::numeric_limits<double>::infinity()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("finite")));
  EXPECT_THAT(
      ValidateBounds<double>(-std::numeric_limits<double>::infinity(), 1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("finite")));
}

TEST(ValidateTest, ValidateTreeHeightForInvalidNumeric) {
  EXPECT_THAT(ValidateTreeHeight(-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Height must be greater than or equal to 1")));
  EXPECT_THAT(ValidateTreeHeight(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Height must be greater than or equal to 1")));
}

TEST(ValidateTest, ValidateTreeHeightForEmpty) {
  EXPECT_THAT(ValidateTreeHeight(std::nullopt),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Tree Height must be set.")));
}

TEST(ValidateTest, ValidateTreeHeightForOK) {
  EXPECT_THAT(ValidateTreeHeight(1), StatusIs(absl::StatusCode::kOk));
}

TEST(ValidateTest, ValidateBranchingFactorForInvalidNumeric) {
  EXPECT_THAT(
      ValidateBranchingFactor(-1),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Branching Factor must be greater than or equal to 2")));
  EXPECT_THAT(
      ValidateBranchingFactor(0),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Branching Factor must be greater than or equal to 2")));
  EXPECT_THAT(
      ValidateBranchingFactor(1),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Branching Factor must be greater than or equal to 2")));
}

TEST(ValidateTest, ValidateBranchingFactorForEmpty) {
  EXPECT_THAT(ValidateBranchingFactor(std::nullopt),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Branching Factor must be set")));
}

TEST(ValidateTest, ValidateBranchingFactorForOK) {
  EXPECT_THAT(ValidateBranchingFactor(2), StatusIs(absl::StatusCode::kOk));
}

}  // namespace
}  // namespace differential_privacy
