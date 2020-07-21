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
/*using testing::_;
using testing::DoubleEq;
using testing::DoubleNear;
using testing::Eq;
using testing::Ge;
using testing::MatchesRegex;
using testing::Return;

class PartitionSelectionStrategiesTest : public ::testing::Test {};*/

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionValidInstantiation) {
	PreaggPartitionSelection::Builder test_builder;
	test_builder.SetEpsilon(1.0).SetDelta(0.2).SetMaxPartitionsContributed(3); 
}

} //namespace
} //namespace differential_privacy