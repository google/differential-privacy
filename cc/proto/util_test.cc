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

#include "proto/util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace differential_privacy {

namespace {

using ::testing::Eq;

const double kLowerBound = -2.5;
const double kUpperBound = 3.2;
const double kConfidenceLevel = 0.95;
const std::vector<double> kLowerBounds = {-1.2, -3.4, -5.6, -7.8, -9.0};
const std::vector<double> kUpperBounds = {1.2, 3.4, 5.6, 7.8, 9.0};
const std::vector<double> kConfidenceLevels = {0.1, 0.25, 0.50, 0.75, 0.99};

TEST(UtilTest, GetSetValueTypeInt) {
  ValueType v;
  SetValue(&v, 10);
  EXPECT_TRUE(v.has_int_value());
  EXPECT_THAT(GetValue<int64_t>(v), Eq(10));
}

TEST(UtilTest, GetSetValueTypeString) {
  ValueType v;
  SetValue(&v, "test");
  EXPECT_TRUE(v.has_string_value());
  EXPECT_THAT(GetValue<std::string>(v), Eq("test"));
}

TEST(UtilTest, GetSetValueTypeFloat) {
  ValueType v;
  SetValue(&v, 10.0);
  EXPECT_TRUE(v.has_float_value());
  EXPECT_THAT(GetValue<double>(v), Eq(10.0));
}

uint64_t BitCastDoubleToUint64(double d) {
  static_assert(sizeof(double) == sizeof(uint64_t));
  uint64_t result;
  std::memcpy(&result, &d, sizeof(uint64_t));
  return result;
}

TEST(UtilTest, SetValueNormalizesSignalingNanToQuietNan) {
  ValueType v;
  const double signaling_nan = std::numeric_limits<double>::signaling_NaN();

  SetValue(&v, signaling_nan);

  // NaN comparison is always false, we therefore compare the bit patterns.
  EXPECT_TRUE(BitCastDoubleToUint64(GetValue<double>(v)) ==
              BitCastDoubleToUint64(std::numeric_limits<double>::quiet_NaN()));
}

TEST(UtilTest, SetValueKeepsSignOfZeros) {
  ValueType v1, v2;
  const double positive_zero = 0.0;
  const double negative_zero = -0.0;

  SetValue(&v1, positive_zero);
  SetValue(&v2, negative_zero);

  EXPECT_FALSE(std::signbit(GetValue<double>(v1)));
  EXPECT_TRUE(std::signbit(GetValue<double>(v2)));
}

TEST(UtilTest, SetValueKeepsSignOfInf) {
  ValueType v1, v2;
  const double positive_inf = std::numeric_limits<double>::infinity();
  const double negative_inf = -std::numeric_limits<double>::infinity();

  SetValue(&v1, positive_inf);
  SetValue(&v2, negative_inf);

  EXPECT_FALSE(std::signbit(GetValue<double>(v1)));
  EXPECT_TRUE(std::signbit(GetValue<double>(v2)));
}

TEST(UtilTest, MakeOutputString) {
  std::string s = "hello";
  Output output = MakeOutput<std::string>(s);
  EXPECT_EQ(GetValue<std::string>(output), s);
}

TEST(UtilTest, MakeOutputInt) {
  int i = 1;
  Output output = MakeOutput<int>(i);
  EXPECT_EQ(GetValue<int>(output), i);
}

TEST(UtilTest, MakeOutputFloat) {
  double d = .1;
  Output output = MakeOutput<double>(d);
  EXPECT_EQ(GetValue<double>(output), d);
}

TEST(UtilTest, MakeOutputStringWithConfidenceInterval) {
  std::string s = "hello";
  ConfidenceInterval ci;
  ci.set_lower_bound(kLowerBound);
  ci.set_upper_bound(kUpperBound);
  ci.set_confidence_level(kConfidenceLevel);
  Output output = MakeOutput<std::string>(s, ci);
  ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output);

  EXPECT_EQ(GetValue<std::string>(output), s);
  EXPECT_EQ(output_ci.lower_bound(), ci.lower_bound());
  EXPECT_EQ(output_ci.upper_bound(), ci.upper_bound());
  EXPECT_EQ(output_ci.confidence_level(), ci.confidence_level());
}

TEST(UtilTest, MakeOutputIntWithConfidenceInterval) {
  int i = 1;
  ConfidenceInterval ci;
  ci.set_lower_bound(kLowerBound);
  ci.set_upper_bound(kUpperBound);
  ci.set_confidence_level(kConfidenceLevel);
  Output output = MakeOutput<int>(i, ci);
  ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output);

  EXPECT_EQ(GetValue<int>(output), i);
  EXPECT_EQ(output_ci.lower_bound(), ci.lower_bound());
  EXPECT_EQ(output_ci.upper_bound(), ci.upper_bound());
  EXPECT_EQ(output_ci.confidence_level(), ci.confidence_level());
}

TEST(UtilTest, MakeOutputFloatWithConfidenceInterval) {
  double d = .1;
  ConfidenceInterval ci;
  ci.set_lower_bound(kLowerBound);
  ci.set_upper_bound(kUpperBound);
  ci.set_confidence_level(kConfidenceLevel);
  Output output = MakeOutput<double>(d, ci);
  ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output);

  EXPECT_EQ(GetValue<double>(output), d);
  EXPECT_EQ(output_ci.lower_bound(), ci.lower_bound());
  EXPECT_EQ(output_ci.upper_bound(), ci.upper_bound());
  EXPECT_EQ(output_ci.confidence_level(), ci.confidence_level());
}

TEST(UtilTest, AddToOutputString) {
  Output output;
  AddToOutput<std::string>(&output, "1");
  EXPECT_EQ(GetValue<std::string>(output), "1");
}

TEST(UtilTest, AddToOutputInt) {
  Output output;
  AddToOutput<int>(&output, 1);
  EXPECT_EQ(GetValue<int>(output), 1);
}

TEST(UtilTest, AddToOutputFloat) {
  Output output;
  AddToOutput<double>(&output, .5);
  EXPECT_EQ(GetValue<double>(output), .5);
}

TEST(UtilTest, AddToOutputStringWithConfidenceInterval) {
  Output output;
  ConfidenceInterval ci;
  ci.set_lower_bound(kLowerBound);
  ci.set_upper_bound(kUpperBound);
  ci.set_confidence_level(kConfidenceLevel);
  AddToOutput<std::string>(&output, "1", ci);
  ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output);

  EXPECT_EQ(GetValue<std::string>(output), "1");
  EXPECT_EQ(output_ci.lower_bound(), ci.lower_bound());
  EXPECT_EQ(output_ci.upper_bound(), ci.upper_bound());
  EXPECT_EQ(output_ci.confidence_level(), ci.confidence_level());
}

TEST(UtilTest, AddToOutputIntWithConfidenceInterval) {
  Output output;
  ConfidenceInterval ci;
  ci.set_lower_bound(kLowerBound);
  ci.set_upper_bound(kUpperBound);
  ci.set_confidence_level(kConfidenceLevel);
  AddToOutput<int>(&output, 1, ci);
  ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output);

  EXPECT_EQ(GetValue<int>(output), 1);
  EXPECT_EQ(output_ci.lower_bound(), ci.lower_bound());
  EXPECT_EQ(output_ci.upper_bound(), ci.upper_bound());
  EXPECT_EQ(output_ci.confidence_level(), ci.confidence_level());
}

TEST(UtilTest, AddToOutputFloatWithConfidenceInterval) {
  Output output;
  ConfidenceInterval ci;
  ci.set_lower_bound(kLowerBound);
  ci.set_upper_bound(kUpperBound);
  ci.set_confidence_level(kConfidenceLevel);
  AddToOutput<double>(&output, .5, ci);
  ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output);

  EXPECT_EQ(GetValue<double>(output), .5);
  EXPECT_EQ(output_ci.lower_bound(), ci.lower_bound());
  EXPECT_EQ(output_ci.upper_bound(), ci.upper_bound());
  EXPECT_EQ(output_ci.confidence_level(), ci.confidence_level());
}

TEST(UtilTest, MultipleAddToOutputStringWithConfidenceInterval) {
  std::vector<std::string> data = {"1", "-2", "3", "-4", "5"};
  Output output;
  for (int i = 0; i < data.size(); ++i) {
    ConfidenceInterval ci;
    ci.set_lower_bound(kLowerBounds[i]);
    ci.set_upper_bound(kUpperBounds[i]);
    ci.set_confidence_level(kConfidenceLevels[i]);
    AddToOutput<std::string>(&output, data[i], ci);
  }

  for (int i = 0; i < data.size(); ++i) {
    ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output, i);
    EXPECT_EQ(GetValue<std::string>(output, i), data[i]);
    EXPECT_EQ(output_ci.lower_bound(), kLowerBounds[i]);
    EXPECT_EQ(output_ci.upper_bound(), kUpperBounds[i]);
    EXPECT_EQ(output_ci.confidence_level(), kConfidenceLevels[i]);
  }
}

TEST(UtilTest, MultipleAddToOutputIntWithConfidenceInterval) {
  std::vector<int> data = {1, -2, 3, -4, 5};
  Output output;
  for (int i = 0; i < data.size(); ++i) {
    ConfidenceInterval ci;
    ci.set_lower_bound(kLowerBounds[i]);
    ci.set_upper_bound(kUpperBounds[i]);
    ci.set_confidence_level(kConfidenceLevels[i]);
    AddToOutput<int>(&output, data[i], ci);
  }

  for (int i = 0; i < data.size(); ++i) {
    ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output, i);
    EXPECT_EQ(GetValue<int>(output, i), data[i]);
    EXPECT_EQ(output_ci.lower_bound(), kLowerBounds[i]);
    EXPECT_EQ(output_ci.upper_bound(), kUpperBounds[i]);
    EXPECT_EQ(output_ci.confidence_level(), kConfidenceLevels[i]);
  }
}

TEST(UtilTest, MultipleAddToOutputFloatWithConfidenceInterval) {
  std::vector<double> data = {1.0, -2.3, 4.5, -6.7, 8.9};
  Output output;
  for (int i = 0; i < data.size(); ++i) {
    ConfidenceInterval ci;
    ci.set_lower_bound(kLowerBounds[i]);
    ci.set_upper_bound(kUpperBounds[i]);
    ci.set_confidence_level(kConfidenceLevels[i]);
    AddToOutput<double>(&output, data[i], ci);
  }

  for (int i = 0; i < data.size(); ++i) {
    ConfidenceInterval output_ci = GetNoiseConfidenceInterval(output, i);
    EXPECT_EQ(GetValue<double>(output, i), data[i]);
    EXPECT_EQ(output_ci.lower_bound(), kLowerBounds[i]);
    EXPECT_EQ(output_ci.upper_bound(), kUpperBounds[i]);
    EXPECT_EQ(output_ci.confidence_level(), kConfidenceLevels[i]);
  }
}

TEST(UtilTest, MakeValueType) {
  ValueType v = MakeValueType(1.0);
  EXPECT_TRUE(v.has_float_value());
  EXPECT_EQ(v.float_value(), 1.0);
}

}  // namespace

}  // namespace differential_privacy
