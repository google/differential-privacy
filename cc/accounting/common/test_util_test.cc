// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "accounting/common/test_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "accounting/common/common.h"

namespace differential_privacy {
namespace accounting {

namespace {
using ::testing::Not;

constexpr double kMaxError = 1e-6;

TEST(PMFIsNearTest, Match) {
  ProbabilityMassFunction pmf1 = {{1, 0.2}, {3, 0.3}, {2, 0.0000001}};
  ProbabilityMassFunction pmf2 = {{1, 0.2000001}, {3, 0.2999999}};
  EXPECT_THAT(pmf1, PMFIsNear(pmf2, kMaxError));
}

TEST(PMFIsNearTest, UnmatchZeroMass) {
  ProbabilityMassFunction pmf1 = {{1, 0.2}, {3, 0.3}, {2, 0.0001}};
  ProbabilityMassFunction pmf2 = {{1, 0.2000001}, {3, 0.2999999}};
  EXPECT_THAT(pmf1, Not(PMFIsNear(pmf2, kMaxError)));
}

TEST(PMFIsNearTest, UnmatchNonzeroMass) {
  ProbabilityMassFunction pmf1 = {{1, 0.2}, {3, 0.31}, {2, 0.0000001}};
  ProbabilityMassFunction pmf2 = {{1, 0.2000001}, {3, 0.2999999}};
  EXPECT_THAT(pmf1, Not(PMFIsNear(pmf2, kMaxError)));
}

TEST(PMFIsNearTest, UnmatchMissingOutcome) {
  ProbabilityMassFunction pmf1 = {{3, 0.31}, {2, 0.0000001}};
  ProbabilityMassFunction pmf2 = {{1, 0.2000001}, {3, 0.2999999}};
  EXPECT_THAT(pmf1, Not(PMFIsNear(pmf2, kMaxError)));
}

}  // namespace
}  // namespace accounting
}  // namespace differential_privacy
