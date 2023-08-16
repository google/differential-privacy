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

#include "testing/density_estimation.h"

#include <cstdint>
#include <limits>
#include <string>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace differential_privacy {
namespace testing {
namespace {

template <typename T>
class HistogramTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(HistogramTest, NumericTypes);

TYPED_TEST(HistogramTest, EmptyHistogram) {
  Histogram<TypeParam> hist(-2, .5, 8);
  for (int i = 0; i < hist.NumBins(); ++i) {
    EXPECT_EQ(0, hist.BinCountOrDie(i));
  }
  EXPECT_EQ(hist.Total(), 0);
  EXPECT_EQ(hist.MaxBinCount(), 0);
}

TYPED_TEST(HistogramTest, AddOutOfBounds) {
  Histogram<TypeParam> hist(-2, .5, 8);
  EXPECT_EQ(hist.Add(-3).code(), absl::StatusCode::kInvalidArgument);
}

TYPED_TEST(HistogramTest, BinCountOutOfBounds) {
  const int num_bins = 8;
  Histogram<TypeParam> hist(-2, .5, num_bins);
  EXPECT_EQ(hist.BinCount(-1).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(hist.BinCount(num_bins + 1).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TYPED_TEST(HistogramTest, BinBoundaries) {
  Histogram<TypeParam> hist(-2, .5, 8);
  EXPECT_EQ(hist.BinBoundary(0), -2);
  EXPECT_EQ(hist.BinBoundary(4), 0);
  EXPECT_EQ(hist.BinBoundary(8), std::numeric_limits<TypeParam>::max());
}

TYPED_TEST(HistogramTest, SmallInputSet) {
  Histogram<TypeParam> hist(-2, 2, 3);
  EXPECT_OK(hist.Add(-1));
  EXPECT_OK(hist.Add(0));
  EXPECT_OK(hist.Add(0));
  EXPECT_OK(hist.Add(100));
  EXPECT_EQ(hist.BinCountOrDie(0), 1);
  EXPECT_EQ(hist.BinCountOrDie(1), 2);
  EXPECT_EQ(hist.BinCountOrDie(2), 1);
  EXPECT_EQ(hist.Total(), 4);
  EXPECT_EQ(hist.MaxBinCount(), 2);
}

TEST(HistogramTest, ToStringEmpty) {
  Histogram<double> hist(0, 1.0 / 3.0, 2);
  const std::string expected = absl::StrCat(
      "\n", "-------------------------------------------------------------\n",
      "Empty histogram.\n",
      "-------------------------------------------------------------");
  EXPECT_EQ(hist.ToString(), expected);
}

TEST(HistogramTest, ToString) {
  Histogram<double> hist(0, 1.0 / 3.0, 2);
  ASSERT_OK(hist.Add(.25));
  ASSERT_OK(hist.Add(.4));
  ASSERT_OK(hist.Add(.4));
  const std::string expected = absl::StrCat(
      "\n", "-------------------------------------------------------------\n",
      "[                0,         0.333333) 1  33.333% ######\n",
      "[         0.333333,              inf) 2  66.667% ############\n",
      "-------------------------------------------------------------");
  EXPECT_EQ(hist.ToString(), expected);
}

}  // namespace
}  // namespace testing
}  // namespace differential_privacy
