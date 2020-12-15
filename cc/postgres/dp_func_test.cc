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

#include "postgres/dp_func.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

template <typename T>
class BoundedDpFuncTest : public ::testing::Test {};

typedef ::testing::Types<DpSum, DpMean, DpVariance, DpStandardDeviation>
    BoundedDpFuncs;
TYPED_TEST_SUITE(BoundedDpFuncTest, BoundedDpFuncs);

TEST(DpCount, BadEpsilon) {
  std::string err;
  auto dp_count = DpCount(&err, false, 0);
  EXPECT_EQ(err, "Epsilon must be finite and positive, but is 0.");
}

TEST(DpCount, AddEntryMissingAlgorithm) {
  std::string err;
  auto dp_count = DpCount(&err, false, 0);
  EXPECT_FALSE(dp_count.AddEntry(1));
}

TEST(DpCount, ResultMissingAlgorithm) {
  std::string err;
  auto dp_count = DpCount(&err, false, 0);
  EXPECT_EQ(dp_count.Result(&err), 0);
  EXPECT_EQ(err, "Underlying algorithm was never constructed.");
}

TEST(DpCount, BasicTest) {
  std::string err;
  auto func = DpCount(&err, true, 0);
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(func.AddEntry(1));
  static_cast<void>(func.Result(&err));
  EXPECT_TRUE(err.empty());
}

TYPED_TEST(BoundedDpFuncTest, BasicTest) {
  std::string err;
  auto func = TypeParam(&err, true, 0, false, 0, 5);
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(func.AddEntry(1));
  static_cast<void>(func.Result(&err));
  EXPECT_TRUE(err.empty());
}

TEST(DpNtile, BadPercentile) {
  std::string err;
  auto func = DpNtile(&err, -1, 0, 10);
  EXPECT_EQ(err,
            "Percentile must be in the inclusive interval [0,1], but is -1.");
}

TEST(DpNtile, BadBounds) {
  std::string err;
  auto func = DpNtile(&err, -1, 5, 1);
  EXPECT_EQ(err, "Lower bound cannot be greater than upper bound.");
}

TEST(DpNtile, BasicTest) {
  std::string err;
  auto func = DpNtile(&err, .5, 0, 10);
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(func.AddEntry(1));
  static_cast<void>(func.Result(&err));
  EXPECT_TRUE(err.empty());
}

}  // namespace
