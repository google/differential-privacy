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

#include "algorithms/numerical-mechanisms-testing.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace differential_privacy {
namespace test_utils {
namespace {

template <typename T>
class NumericalMechanismsTestingTest : public ::testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(NumericalMechanismsTestingTest, NumericTypes);

TYPED_TEST(NumericalMechanismsTestingTest, DifferentSeeds) {
  SeededLaplaceMechanism a(1.0, 1.0);
  SeededLaplaceMechanism b(1.0, 1.0);
  bool is_equal = true;
  for (int i = 0; i < 10; ++i) {
    is_equal &= (a.AddNoise(i) == b.AddNoise(i));
  }
  EXPECT_FALSE(is_equal);
}

TYPED_TEST(NumericalMechanismsTestingTest, PassedSeedConsistency) {
  std::seed_seq seed({1, 1, 1, 1, 1});
  std::seed_seq seed2({1, 1, 1, 1, 1});
  std::mt19937 gen(seed);
  std::mt19937 gen2(seed2);
  SeededLaplaceMechanism a(1, 10, &gen);
  SeededLaplaceMechanism b(1, 10, &gen2);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(a.AddNoise(i), b.AddNoise(i));
  }
}

TYPED_TEST(NumericalMechanismsTestingTest, DifferentSeedInconsistency) {
  std::seed_seq seed({1, 2, 3, 4});
  std::seed_seq seed2({1, 1, 1, 1, 1});
  std::mt19937 gen(seed);
  std::mt19937 gen2(seed2);
  SeededLaplaceMechanism a(1, 10, &gen);
  SeededLaplaceMechanism b(1, 10, &gen2);
  bool same_noise = true;
  // Have 1000 changes to get different noise in order to ensure this test will
  // fail by chance with an unreasonably small chance.
  for (int i = 0; i < 1000; ++i) {
    if (a.AddNoise(i) != b.AddNoise(i)) {
      same_noise = false;
      break;
    }
  }
  EXPECT_FALSE(same_noise);
}

}  // namespace
}  // namespace test_utils
}  // namespace differential_privacy
