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

#include "algorithms/bounded-algorithm.h"

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/approx-bounds.h"

namespace differential_privacy {
namespace {

using ::testing::HasSubstr;
using ::differential_privacy::base::testing::StatusIs;

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
    T Lower() { return BoundedBuilder::GetLower().value(); }
    T Upper() { return BoundedBuilder::GetUpper().value(); }
    bool HasLower() { return BoundedBuilder::GetLower().has_value(); }
    bool HasUpper() { return BoundedBuilder::GetUpper().has_value(); }

   private:
    absl::StatusOr<std::unique_ptr<BoundedAlgorithm<T>>> BuildBoundedAlgorithm()
        override {
      return absl::WrapUnique(new BoundedAlgorithm());
    }
  };

  // Trivial implementations of virtual functions.
  void AddEntry(const T& t) override {}
  absl::StatusOr<Output> GenerateResult(
      double /*noise_interval_level*/) override {
    return Output();
  }
  void ResetState() override {}
  Summary Serialize() const override { return Summary(); }
  absl::Status Merge(const Summary& summary) override {
    return absl::OkStatus();
  }
  int64_t MemoryUsed() override { return sizeof(BoundedAlgorithm<T>); };

 private:
  BoundedAlgorithm() : Algorithm<T>(/*epsilon=*/1) {}
};

TYPED_TEST(BoundedAlgorithmTest, ManualBoundsTest) {
  typename BoundedAlgorithm<TypeParam>::Builder builder;
  builder.SetEpsilon(1.1).SetLower(1).SetUpper(2);
  EXPECT_OK(builder.Build());
  EXPECT_EQ(builder.Lower(), 1);
  EXPECT_EQ(builder.Upper(), 2);
  EXPECT_TRUE(builder.HasLower());
  EXPECT_TRUE(builder.HasUpper());
}

TEST(BoundedAlgorithmTest, InvalidParameters) {
  typename BoundedAlgorithm<double>::Builder builder;

  builder.SetEpsilon(1.1);
  builder.SetLower(-std::numeric_limits<double>::infinity());
  builder.SetUpper(0.5);
  EXPECT_THAT(builder.Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lower bound must be finite")));

  builder.SetLower(-0.5);
  builder.SetUpper(std::numeric_limits<double>::infinity());
  EXPECT_THAT(builder.Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Upper bound must be finite")));

  builder.SetLower(0.5);
  builder.SetUpper(-0.5);
  EXPECT_THAT(
      builder.Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Lower bound cannot be greater than upper bound.")));

  builder.ClearBounds();
  builder.SetLower(-0.5);
  EXPECT_THAT(builder.Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lower and upper bounds must either both be "
                                 "set or both be unset.")));

  builder.ClearBounds();
  builder.SetUpper(-0.5);
  EXPECT_THAT(builder.Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lower and upper bounds must either both be "
                                 "set or both be unset.")));

  builder.SetLower(std::numeric_limits<double>::quiet_NaN());
  builder.SetUpper(0.5);
  EXPECT_THAT(builder.Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lower bound must be a valid numeric value")));

  builder.SetLower(-0.5);
  builder.SetUpper(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THAT(builder.Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Upper bound must be a valid numeric value")));
}

}  // namespace
}  // namespace differential_privacy
