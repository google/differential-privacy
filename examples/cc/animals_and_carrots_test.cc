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

#include "animals_and_carrots.h"

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

namespace differential_privacy {
namespace example {
namespace {

constexpr char kDatafile[] =
    "animals_and_carrots.csv";

TEST(CarrotReporterTest, TrueStatistics) {
  CarrotReporter reporter(kDatafile, 1);
  EXPECT_EQ(reporter.Mean(),
            static_cast<double>(reporter.Sum()) / reporter.CountAbove(-1));
  EXPECT_EQ(reporter.Max(), 100);
}

TEST(CarrotReporterTest, TooLittleBudget) {
  CarrotReporter reporter(kDatafile, 1);
  EXPECT_EQ(reporter.PrivateCountAbove(2, 50).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(reporter.PrivateMax(2).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(reporter.PrivateMean(2).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(reporter.PrivateSum(2).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(reporter.RemainingEpsilon(), 1.0);
}

TEST(CarrotReporterTest, PrivacyBudget) {
  CarrotReporter reporter(kDatafile, 1);
  EXPECT_EQ(reporter.RemainingEpsilon(), 1.0);
  EXPECT_OK(reporter.PrivateMax(.2));
  EXPECT_EQ(reporter.RemainingEpsilon(), .8);
  EXPECT_OK(reporter.PrivateMax(.8));
  EXPECT_EQ(reporter.RemainingEpsilon(), 0.0);
}

}  // namespace
}  // namespace example
}  // namespace differential_privacy
