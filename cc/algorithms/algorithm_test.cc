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

#include "algorithms/algorithm.h"

#include <forward_list>
#include <list>
#include <vector>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "base/statusor.h"

namespace differential_privacy {
namespace {

using ::testing::DoubleNear;

const double kTestPrecision = 1e-5;

template <typename T>
class TestAlgorithm : public Algorithm<T> {
 public:
  TestAlgorithm() : Algorithm<T>(1.0) {}
  void AddEntry(const T& t) override {}
  Summary Serialize() override { return Summary(); }
  base::Status Merge(const Summary& summary) override {
    return base::OkStatus();
  }
  int64_t MemoryUsed() override { return sizeof(TestAlgorithm<T>); }

 protected:
  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    Output output;
    output.mutable_error_report()
        ->mutable_noise_confidence_interval()
        ->set_confidence_level(noise_interval_level);
    return output;
  }
  void ResetState() override {}
};

TEST(IncrementalAlgorithmTest, StartsWithFullBudget) {
  TestAlgorithm<double> alg;
  EXPECT_THAT(alg.RemainingPrivacyBudget(), DoubleNear(1.0, kTestPrecision));
}

TEST(IncrementalAlgorithmTest, PartialResultConsumesBudget) {
  TestAlgorithm<double> alg;
  alg.PartialResult().ValueOrDie();
  EXPECT_THAT(alg.RemainingPrivacyBudget(), DoubleNear(0.0, kTestPrecision));
}

TEST(IncrementalAlgorithmTest, PartialResultConsumesPartialBudget) {
  TestAlgorithm<double> alg;
  alg.PartialResult(0.5).ValueOrDie();
  EXPECT_THAT(alg.RemainingPrivacyBudget(), DoubleNear(0.5, kTestPrecision));
}

TEST(IncrementalAlgorithmTest, PartialResultConsumesPartialBudgetMultiRound) {
  TestAlgorithm<double> alg;
  alg.PartialResult(0.5).ValueOrDie();
  alg.ConsumePrivacyBudget(0.5);
  EXPECT_THAT(alg.RemainingPrivacyBudget(), DoubleNear(0.0, kTestPrecision));
}

TEST(IncrementalAlgorithmTest, PartialResultPassesConfidenceLevel) {
  TestAlgorithm<double> alg;
  const double level = .9;
  const Output output = alg.PartialResult(1, level).ValueOrDie();
  EXPECT_EQ(
      output.error_report().noise_confidence_interval().confidence_level(),
      level);
}

TEST(IncrementalAlgorithmTest, Reset) {
  TestAlgorithm<double> alg;
  alg.PartialResult(0.5).ValueOrDie();
  alg.ConsumePrivacyBudget(0.5);
  alg.Reset();
  EXPECT_THAT(alg.RemainingPrivacyBudget(), DoubleNear(1.0, kTestPrecision));
}

TEST(IncrementalAlgorithmTest, MergedPartialResultConsumesBudget) {
  // Serialize and merge alg_1 into alg_2.
  TestAlgorithm<double> alg_1;
  TestAlgorithm<double> alg_2;
  Summary summary_1 = alg_1.Serialize();
  EXPECT_OK(alg_2.Merge(summary_1));
  alg_2.PartialResult().ValueOrDie();
  EXPECT_THAT(alg_2.RemainingPrivacyBudget(), DoubleNear(0.0, kTestPrecision));
}

TEST(IncrementalAlgorithmDeathTest, BudgetTooHigh) {
  TestAlgorithm<double> alg;
  alg.PartialResult(0.5).ValueOrDie();
  EXPECT_DEATH(alg.ConsumePrivacyBudget(0.6), "Requested budget.*");
}

}  // namespace
}  // namespace differential_privacy
