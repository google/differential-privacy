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

#include "base/percentile.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "proto/summary.pb.h"

namespace differential_privacy {
namespace base {
namespace {

template <typename T>
class PercentileTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(PercentileTest, NumericTypes);

TYPED_TEST(PercentileTest, EmptyInputSet) {
  Percentile<TypeParam> percentile;
  EXPECT_EQ(percentile.num_values(), 0);
  EXPECT_EQ(std::make_pair(0.0, 1.0), percentile.GetRelativeRank(1));
}

TYPED_TEST(PercentileTest, SingletonInputSet) {
  Percentile<TypeParam> percentile;
  percentile.Add(1);
  EXPECT_EQ(percentile.num_values(), 1);
  EXPECT_EQ(std::make_pair(0.0, 0.0), percentile.GetRelativeRank(0));
  EXPECT_EQ(std::make_pair(0.0, 1.0), percentile.GetRelativeRank(1));
  EXPECT_EQ(std::make_pair(1.0, 1.0), percentile.GetRelativeRank(10));
}

TYPED_TEST(PercentileTest, SmallInputSet) {
  Percentile<TypeParam> percentile;
  percentile.Add(5);
  percentile.Add(3);
  percentile.Add(3);
  percentile.Add(5);
  percentile.Add(1);
  EXPECT_EQ(percentile.num_values(), 5);
  EXPECT_EQ(std::make_pair(0.0, 0.0), percentile.GetRelativeRank(-1));
  EXPECT_EQ(std::make_pair(0.0, 0.2), percentile.GetRelativeRank(1));
  EXPECT_EQ(std::make_pair(0.2, 0.2), percentile.GetRelativeRank(2));
  EXPECT_EQ(std::make_pair(0.2, 0.6), percentile.GetRelativeRank(3));
  EXPECT_EQ(std::make_pair(0.6, 0.6), percentile.GetRelativeRank(4));
  EXPECT_EQ(std::make_pair(0.6, 1.0), percentile.GetRelativeRank(5));
  EXPECT_EQ(std::make_pair(1.0, 1.0), percentile.GetRelativeRank(6));
}

TYPED_TEST(PercentileTest, LargeInputSet) {
  Percentile<TypeParam> percentile;
  int num_repeats = 3;
  TypeParam num_values = 10000;
  for (TypeParam i = num_values; i > 0; --i) {
    for (int j = 0; j < num_repeats; ++j) {
      percentile.Add(i);
    }
  }
  EXPECT_EQ(std::make_pair(52.0 / num_values, 53.0 / num_values),
            percentile.GetRelativeRank(53));
  EXPECT_EQ(std::make_pair(0.0, 1.0 / num_values),
            percentile.GetRelativeRank(1));
  EXPECT_EQ(std::make_pair((num_values - 1.0) / num_values, 1.0),
            percentile.GetRelativeRank(num_values));
}

TYPED_TEST(PercentileTest, Reset) {
  Percentile<TypeParam> percentile;
  percentile.Add(1);
  percentile.Reset();
  EXPECT_EQ(percentile.num_values(), 0);
}

TYPED_TEST(PercentileTest, Memory) {
  Percentile<TypeParam> percentile;
  int64_t small_memory = percentile.Memory();
  percentile.Add(1);
  int64_t large_memory = percentile.Memory();
  EXPECT_LT(small_memory, large_memory);
}

TYPED_TEST(PercentileTest, SerializeMerge) {
  Percentile<TypeParam> percentile;
  percentile.Add(4);
  BinarySearchSummary summary;
  percentile.SerializeToProto(summary.mutable_input());

  Percentile<TypeParam> percentile2;
  percentile2.Add(2);
  percentile2.MergeFromProto(summary.input());
  percentile2.Add(3);
  percentile2.Add(1);
  EXPECT_EQ(std::make_pair(.25, .5), percentile2.GetRelativeRank(2));
}

}  // namespace
}  // namespace base
}  // namespace differential_privacy
