//
// Copyright 2020 Google LLC
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

#include "algorithms/partition-selection-strategies.h"

#include "gmock/gmock.h"  //TODO do I need this?
#include "gtest/gtest.h"  //TODO do I need this?

namespace differential_privacy {
namespace {

//TODO literally just nabbed this from numerical-mechanisms_test.cc, fix later
using testing::_;
using testing::DoubleEq;
using testing::DoubleNear;
using testing::Eq;
using testing::Ge;
using testing::MatchesRegex;
using testing::Return;

class PartitionSelectionStrategiesTest : public ::testing::Test {};

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionNumUsersZero) {
	PreaggPartitionSelection pa1(1.0, 1.0, 0, 1);
	PreaggPartitionSelection pa2(1.2, 3.4, 0, 5);
	EXPECT_FALSE(pa1.shouldKeep());
	EXPECT_FALSE(pa2.shouldKeep());
}

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionDeltaZero) {
	PreaggPartitionSelection pa(0.6, 0.0, 78, 9);
	EXPECT_FALSE(pa.shouldKeep());
}

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionEpsilonZero) {
	//TODO
}

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionNonZeroCases) {
	//TODO
}

} //namespace
} //namespace differential_privacy