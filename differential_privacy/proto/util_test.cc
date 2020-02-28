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

#include "differential_privacy/proto/util.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace differential_privacy {

namespace {

using ::testing::Eq;

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

TEST(UtilTest, MakeOutputString) {
  std::string s = "hello";
  Output output = MakeOutput<std::string>(s);
  EXPECT_EQ(output.elements(0).value().string_value(), s);
}

TEST(UtilTest, MakeOutputInt) {
  int i = 1;
  Output output = MakeOutput<int>(i);
  EXPECT_EQ(output.elements(0).value().int_value(), i);
}

TEST(UtilTest, MakeOutputFloat) {
  double d = .1;
  Output output = MakeOutput<double>(d);
  EXPECT_EQ(output.elements(0).value().float_value(), d);
}

TEST(UtilTest, AddToOutputString) {
  Output output;
  AddToOutput<std::string>(&output, "1");
  EXPECT_EQ(output.elements(0).value().string_value(), "1");
}

TEST(UtilTest, AddToOutputInt) {
  Output output;
  AddToOutput<int>(&output, 1);
  EXPECT_EQ(output.elements(0).value().int_value(), 1);
}

TEST(UtilTest, AddToOutputFloat) {
  Output output;
  AddToOutput<double>(&output, .5);
  EXPECT_EQ(output.elements(0).value().float_value(), .5);
}

TEST(UtilTest, MakeValueType) {
  ValueType v = MakeValueType(1.0);
  EXPECT_TRUE(v.has_float_value());
  EXPECT_EQ(v.float_value(), 1.0);
}

}  // namespace

}  // namespace differential_privacy
