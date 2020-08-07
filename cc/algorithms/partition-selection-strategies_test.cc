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
#include "algorithms/numerical-mechanisms-testing.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace differential_privacy {
namespace {

using testing::_;
using testing::Eq;
using testing::DoubleNear;
using testing::DoubleEq;
using testing::MatchesRegex;

//PreaggregationPartitionSelection Tests

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionUnsetEpsilon) {
  	PreaggPartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetDelta(0.1)
  									.SetMaxPartitionsContributed(2).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Epsilon has to be set.*"));
}

TEST(PartitionSelectionStrategiesTest,
	 PreaggPartitionSelectionNotFiniteEpsilon) {
  	PreaggPartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetEpsilon(NAN).SetDelta(0.3)
  									.SetMaxPartitionsContributed(4).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Epsilon has to be finite.*"));
}

TEST(PartitionSelectionStrategiesTest,
	 PreaggPartitionSelectionNegativeEpsilon) {
	PreaggPartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetEpsilon(-5.0).SetDelta(0.6)
  									.SetMaxPartitionsContributed(7).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
   	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Epsilon has to be positive.*"));
}

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionUnsetDelta) {
	PreaggPartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetEpsilon(8.0)
  									.SetMaxPartitionsContributed(9).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
   	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Delta has to be set.*"));
}

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionNotFiniteDelta) {
	PreaggPartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetEpsilon(1.2).SetDelta(NAN)
  									.SetMaxPartitionsContributed(3).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
   	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Delta has to be finite.*"));
}

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionInvalidDelta) {
	PreaggPartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetEpsilon(4.5).SetDelta(6.0)
  									.SetMaxPartitionsContributed(7).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
   	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Delta has to be in the interval.*"));
}

TEST(PartitionSelectionStrategiesTest,
	 PreaggPartitionSelectionUnsetMaxPartitionsContributed) {
	PreaggPartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetEpsilon(0.8).SetDelta(0.9).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
   	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Max number of partitions a user can"
  									  " contribute to has to be set.*"));
}

TEST(PartitionSelectionStrategiesTest,
	 PreaggPartitionSelectionNegativeMaxPartitionsContributed) {
	PreaggPartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetEpsilon(0.1).SetDelta(0.2)
  									.SetMaxPartitionsContributed(-3).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
   	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Max number of partitions a user can"
  									  " contribute to has to be positive.*"));
}

//We expect the probability of keeping a partition with one user
//will be approximately delta
TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionOneUser) {
	PreaggPartitionSelection::Builder test_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetEpsilon(0.5).SetDelta(0.02)
					.SetMaxPartitionsContributed(1).Build().ValueOrDie();
  	double num_kept = 0.0;
  	for(int i = 0; i < 1000000; i++) {
  		if(build->ShouldKeep(1))
  			num_kept++;
  	}
  	EXPECT_THAT(num_kept/1000000, DoubleNear(build->GetDelta(), 0.001));
 }

//We expect the probability of keeping a partition
//with no users will always be zero
TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionNoUsers) {
	PreaggPartitionSelection::Builder test_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetEpsilon(0.5).SetDelta(0.02)
					.SetMaxPartitionsContributed(1).Build().ValueOrDie();
  	double num_kept = 0.0;
  	for(int i = 0; i < 100000; i++) {
  		if(build->ShouldKeep(0))
  			num_kept++;
  	}
  	EXPECT_THAT(num_kept/100000, DoubleEq(0));
 }

TEST(PartitionSelectionStrategiesTest, PreaggPartitionSelectionFirstCrossover) {
	 PreaggPartitionSelection::Builder test_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetEpsilon(0.5).SetDelta(0.02)
					.SetMaxPartitionsContributed(1).Build().ValueOrDie();
	PreaggPartitionSelection* magic =
		dynamic_cast<PreaggPartitionSelection*>(build.get());
  	EXPECT_THAT(magic->GetFirstCrossover(), DoubleEq(6));
 }


 TEST(PartitionSelectionStrategiesTest,
 	  PreaggPartitionSelectionSecondCrossover) {
	PreaggPartitionSelection::Builder test_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetEpsilon(0.5).SetDelta(0.02)
					.SetMaxPartitionsContributed(1).Build().ValueOrDie();
	PreaggPartitionSelection* magic =
		dynamic_cast<PreaggPartitionSelection*>(build.get());
  	EXPECT_THAT(magic->GetSecondCrossover(), DoubleEq(11));
 }

//Values calculated with formula
TEST(PartitionSelectionStrategiesTest,
	 PreaggPartitionSelectionNumUsersEqFirstCrossover) {
	PreaggPartitionSelection::Builder test_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetEpsilon(0.5).SetDelta(0.02)
					.SetMaxPartitionsContributed(1).Build().ValueOrDie();
  	double num_kept = 0.0;
  	for(int i = 0; i < 10000000; i++) {
  		if(build->ShouldKeep(6))
  			num_kept++;
  	}
  	EXPECT_THAT(num_kept/10000000, DoubleNear(0.58840484458, 0.001));
 }

//Values calculated with formula
TEST(PartitionSelectionStrategiesTest,
	 PreaggPartitionSelectionNumUsersBtwnCrossovers) {
	PreaggPartitionSelection::Builder test_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetEpsilon(0.5).SetDelta(0.02)
					.SetMaxPartitionsContributed(1).Build().ValueOrDie();
  	double num_kept = 0.0;
  	for(int i = 0; i < 10000000; i++) {
  		if(build->ShouldKeep(8))
  			num_kept++;
  	}
  	EXPECT_THAT(num_kept/10000000, DoubleNear(0.86807080625, 0.001));
 }

//Values calculated with formula
TEST(PartitionSelectionStrategiesTest,
	 PreaggPartitionSelectionNumUsersGreaterThanCrossovers) {
	PreaggPartitionSelection::Builder test_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetEpsilon(0.5).SetDelta(0.02)
					.SetMaxPartitionsContributed(1).Build().ValueOrDie();
  	double num_kept = 0.0;
  	for(int i = 0; i < 10000000; i++) {
  		if(build->ShouldKeep(15))
  			num_kept++;
  	}
  	EXPECT_THAT(num_kept/10000000, DoubleEq(1.0));
 }

//LaplacePartitionSelection Tests
//Due to the inheritance, SetLaplaceMechanism must be
//called before SetDelta, SetEpsilon, etc.

TEST(PartitionSelectionStrategiesTest,
	 LaplacePartitionSelectionUnsetMaxPartitionsContributed) {
  	LaplacePartitionSelection::Builder test_builder;
  	LaplaceMechanism::Builder laplace_builder;
  	auto failed_build = test_builder.SetLaplaceMechanism(&laplace_builder)
  									.SetDelta(0.1).SetEpsilon(2).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Max number of partitions a user can"
  									  " contribute to has to be set.*"));
}

TEST(PartitionSelectionStrategiesTest,
	 LaplacePartitionSelectionNegativeMaxPartitionsContributed) {
  	LaplacePartitionSelection::Builder test_builder;
  	LaplaceMechanism::Builder laplace_builder;
  	auto failed_build = test_builder.SetLaplaceMechanism(&laplace_builder)
  									.SetDelta(0.1).SetEpsilon(2)
  									.SetMaxPartitionsContributed(-3).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Max number of partitions a user can"
  									  " contribute to has to be positive.*"));
}

TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionUnsetEpsilon) {
  	LaplacePartitionSelection::Builder test_builder;
  	LaplaceMechanism::Builder laplace_builder;
  	auto failed_build = test_builder.SetLaplaceMechanism(&laplace_builder)
  									.SetDelta(0.1)
  									.SetMaxPartitionsContributed(2).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Epsilon has to be set.*"));
}

TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionUnsetDelta) {
  	LaplacePartitionSelection::Builder test_builder;
  	LaplaceMechanism::Builder laplace_builder;
  	auto failed_build = test_builder.SetLaplaceMechanism(&laplace_builder)
  									.SetEpsilon(0.1)
  									.SetMaxPartitionsContributed(2).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Delta has to be set.*"));
}

TEST(PartitionSelectionStrategiesTest,
	 LaplacePartitionSelectionNotFiniteDelta) {
  	LaplacePartitionSelection::Builder test_builder;
  	LaplaceMechanism::Builder laplace_builder;
  	auto failed_build = test_builder.SetLaplaceMechanism(&laplace_builder)
  									.SetEpsilon(0.1).SetDelta(NAN)
  									.SetMaxPartitionsContributed(2).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Delta has to be finite.*"));
}

TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionInvalidDelta) {
  	LaplacePartitionSelection::Builder test_builder;
  	LaplaceMechanism::Builder laplace_builder;
  	auto failed_build = test_builder.SetLaplaceMechanism(&laplace_builder)
  									.SetEpsilon(0.1).SetDelta(5.2)
  									.SetMaxPartitionsContributed(2).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^Delta has to be in the interval.*"));
}

TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionUnsetBuilder) {
  	LaplacePartitionSelection::Builder test_builder;
  	auto failed_build = test_builder.SetDelta(0.3).SetEpsilon(4)
  									.SetMaxPartitionsContributed(5).Build();
  	EXPECT_THAT(failed_build.status().code(),
                Eq(base::StatusCode::kInvalidArgument));
  	std::string message(std::string(failed_build.status().message()));
  	EXPECT_THAT(message, MatchesRegex("^The builder has to be set.*"));
}

//We expect the probability of keeping a partition with one user
// will be approximately delta
TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionOneUser) {
	LaplacePartitionSelection::Builder test_builder;
	LaplaceMechanism::Builder laplace_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetLaplaceMechanism(&laplace_builder).SetEpsilon(0.5)
					.SetDelta(0.02).SetMaxPartitionsContributed(1).Build()
					.ValueOrDie();
  	double num_kept = 0.0;
  	for(int i = 0; i < 1000000; i++) {
  		if(build->ShouldKeep(1))
  			num_kept++;
  	}
  	EXPECT_THAT(num_kept/1000000, DoubleNear(build->GetDelta(), 0.001));
 }

 //When the number of users is at the threshold, we expect drop/keep is 50/50.
 //These numbers should make the threshold approximately 5.
TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionAtThreshold) {
	LaplacePartitionSelection::Builder test_builder;
	LaplaceMechanism::Builder laplace_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetLaplaceMechanism(&laplace_builder).SetEpsilon(0.5)
					.SetDelta(0.06766764161).SetMaxPartitionsContributed(1)
					.Build().ValueOrDie();
  	double num_kept = 0.0;
  	for(int i = 0; i < 1000000; i++) {
  		if(build->ShouldKeep(5))
  			num_kept++;
  	}
  	EXPECT_THAT(num_kept/1000000, DoubleNear(0.5, 0.01));
 }

TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionThreshold) {
	LaplacePartitionSelection::Builder test_builder;
	LaplaceMechanism::Builder laplace_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetLaplaceMechanism(&laplace_builder).SetEpsilon(0.5)
					.SetDelta(0.02).SetMaxPartitionsContributed(1).Build()
					.ValueOrDie();
	LaplacePartitionSelection* laplace =
		dynamic_cast<LaplacePartitionSelection*>(build.get());
  	EXPECT_THAT(laplace->GetThreshold(), DoubleNear(7.43775164974, 0.001));
 }

TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionLow) {
	LaplacePartitionSelection::Builder test_builder;
	test_utils::ZeroNoiseMechanism::Builder laplace_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetLaplaceMechanism(&laplace_builder).SetEpsilon(0.5)
					.SetDelta(0.02).SetMaxPartitionsContributed(1)
					.Build().ValueOrDie();
  	EXPECT_THAT(build->ShouldKeep(7), Eq(false));
 }

 TEST(PartitionSelectionStrategiesTest, LaplacePartitionSelectionHigh) {
	LaplacePartitionSelection::Builder test_builder;
	test_utils::ZeroNoiseMechanism::Builder laplace_builder;
	std::unique_ptr<PartitionSelectionStrategy> build =
		test_builder.SetLaplaceMechanism(&laplace_builder).SetEpsilon(0.5)
					.SetDelta(0.02).SetMaxPartitionsContributed(1).Build()
					.ValueOrDie();
  	EXPECT_THAT(build->ShouldKeep(8), Eq(true));
 }

} //namespace
} //namespace differential_privacy