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

#include "differential_privacy/algorithms/bounded-algorithm.h"

#include "differential_privacy/base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "differential_privacy/algorithms/algorithm.h"
#include "differential_privacy/algorithms/approx-bounds.h"

namespace differential_privacy {
namespace {

template <typename T>
class BoundedAlgorithmTest : public testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(BoundedAlgorithmTest, NumericTypes);

template <typename T>
class BoundedAlgorithm : public Algorithm<T> {
 public:
  class Builder
      : public BoundedAlgorithmBuilder<T, BoundedAlgorithm<T>, Builder> {
    using BoundedBuilder =
        BoundedAlgorithmBuilder<T, BoundedAlgorithm<T>, Builder>;

   public:
    // Methods for testing.
    T Lower() { return BoundedBuilder::lower_; }
    T Upper() { return BoundedBuilder::upper_; }
    bool HasLower() { return BoundedBuilder::has_lower_; }
    bool HasUpper() { return BoundedBuilder::has_upper_; }
    ApproxBounds<T>* GetApproxBounds() {
      return BoundedBuilder::approx_bounds_.get();
    }

   private:
    base::StatusOr<std::unique_ptr<BoundedAlgorithm<T>>> BuildAlgorithm()
        override {
      RETURN_IF_ERROR(BoundedBuilder::BoundsSetup());
      return absl::WrapUnique(new BoundedAlgorithm());
    }
  };

  // Trivial implementations of virtual functions.
  void AddEntry(const T& t) override {}
  base::StatusOr<Output> GenerateResult(
      double /*privacy_budget*/, double /*noise_interval_level*/) override {
    return Output();
  }
  void ResetState() override {}
  Summary Serialize() override { return Summary(); }
  base::Status Merge(const Summary& summary) override {
    return base::OkStatus();
  }
  int64_t MemoryUsed() override { return sizeof(BoundedAlgorithm<T>); };

 private:
  BoundedAlgorithm() : Algorithm<T>(/*epsilon=*/1) {}
};

TYPED_TEST(BoundedAlgorithmTest, ManualBoundsTest) {
  typename BoundedAlgorithm<TypeParam>::Builder builder;
  builder.SetLower(1).SetUpper(2);
  EXPECT_OK(builder.Build());
  EXPECT_EQ(builder.Lower(), 1);
  EXPECT_EQ(builder.Upper(), 2);
  EXPECT_TRUE(builder.HasLower());
  EXPECT_TRUE(builder.HasUpper());
  EXPECT_FALSE(builder.GetApproxBounds());
}

TYPED_TEST(BoundedAlgorithmTest, ApproxBoundsClearsManualBounds) {
  typename BoundedAlgorithm<TypeParam>::Builder builder;
  builder.SetLower(1).SetUpper(2).SetApproxBounds(
      typename ApproxBounds<TypeParam>::Builder().Build().ValueOrDie());
  EXPECT_OK(builder.Build());
  EXPECT_FALSE(builder.HasLower());
  EXPECT_FALSE(builder.HasUpper());
  EXPECT_TRUE(builder.GetApproxBounds());
}

TYPED_TEST(BoundedAlgorithmTest, AutomaticApproxBounds) {
  typename BoundedAlgorithm<TypeParam>::Builder builder;
  EXPECT_OK(builder.Build());
  EXPECT_FALSE(builder.HasLower());
  EXPECT_FALSE(builder.HasUpper());
  EXPECT_TRUE(builder.GetApproxBounds());
}

}  // namespace
}  // namespace differential_privacy
