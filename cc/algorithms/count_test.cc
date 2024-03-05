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

#include "algorithms/count.h"

#include <stdint.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "proto/confidence-interval.pb.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"

namespace differential_privacy {

// Provides limited-scope static methods for interacting with a Count object for
// testing purposes.
class CountTestPeer {
 public:
  template <typename T>
  static void AddMultipleEntries(const T& v, int64_t num_of_entries,
                                 Count<T>* c) {
    c->AddMultipleEntries(v, num_of_entries);
  }
};

namespace {

using ::differential_privacy::test_utils::ZeroNoiseMechanism;
using ::differential_privacy::base::testing::EqualsProto;
using ::testing::HasSubstr;
using ::differential_privacy::base::testing::IsOkAndHolds;
using ::differential_privacy::base::testing::StatusIs;

template <typename T>
class CountTest : public testing::Test {};

constexpr double kDefaultEpsilon = 1.1;

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(CountTest, NumericTypes);

TYPED_TEST(CountTest, BasicTest) {
  std::vector<TypeParam> c = {1, 2, 3, 4, 2, 3};
  absl::StatusOr<std::unique_ptr<Count<TypeParam>>> count =
      typename Count<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(count);
  auto result = (*count)->Result(c.begin(), c.end());
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<int64_t>(*result), 6);
}

TYPED_TEST(CountTest, RepeatedResultTest) {
  std::vector<TypeParam> c = {1, 2, 3, 4, 2, 3};
  typename Count<TypeParam>::Builder builder;
  builder.SetEpsilon(kDefaultEpsilon)
      .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>());
  absl::StatusOr<std::unique_ptr<Count<TypeParam>>> count1 =

      builder.Build();
  ASSERT_OK(count1);

  absl::StatusOr<std::unique_ptr<Count<TypeParam>>> count2 =

      builder.Build();
  ASSERT_OK(count2);

  (*count1)->AddEntries(c.begin(), c.end());
  (*count2)->AddEntries(c.begin(), c.end());

  auto result1 = (*count1)->PartialResult(0.5);
  ASSERT_OK(result1);
  auto result2 = (*count2)->PartialResult(0.5);
  ASSERT_OK(result2);

  EXPECT_EQ(GetValue<int64_t>(*result1), GetValue<int64_t>(*result2));
}

TYPED_TEST(CountTest, AddMultipleEntriesInvalidNumberOfEntriesTest) {
  absl::StatusOr<std::unique_ptr<Count<TypeParam>>> count =
      typename Count<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(count);

  std::vector<int64_t> invalid_entries{-1, -10,
                                       std::numeric_limits<int64_t>::lowest()};
  for (int64_t n_entries : invalid_entries) {
    CountTestPeer::AddMultipleEntries<TypeParam>(1, n_entries,
                                                 count.value().get());
  }

  auto result = (*count)->PartialResult();
  ASSERT_OK(result);

  // Expect nothing to have been added to the count
  EXPECT_EQ(GetValue<int64_t>(*result), 0);
}

TYPED_TEST(CountTest, InsufficientPrivacyBudgetTest) {
  std::vector<TypeParam> c = {1, 2, 3, 4, 2, 3};
  absl::StatusOr<std::unique_ptr<Count<TypeParam>>> count =
      typename Count<TypeParam>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(count);

  (*count)->AddEntries(c.begin(), c.end());

  ASSERT_OK((*count)->PartialResult());
  EXPECT_THAT((*count)->PartialResult(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("can only produce results once")));
}

TEST(CountTest, ConfidenceIntervalTest) {
  double epsilon = 0.5;
  double level = .95;
  auto count = Count<double>::Builder().SetEpsilon(0.5).Build();
  ASSERT_OK(count);

  ConfidenceInterval wantConfidenceInterval;
  wantConfidenceInterval.set_lower_bound(std::log(1 - level) / epsilon);
  wantConfidenceInterval.set_upper_bound(-std::log(1 - level) / epsilon);
  wantConfidenceInterval.set_confidence_level(level);

  absl::StatusOr<ConfidenceInterval> confidenceInterval =
      (*count)->NoiseConfidenceInterval(level);
  EXPECT_THAT(confidenceInterval,
              IsOkAndHolds(EqualsProto(wantConfidenceInterval)));

  auto actual_result = (*count)->PartialResult();
  ASSERT_OK(actual_result);

  EXPECT_THAT(GetNoiseConfidenceInterval(*actual_result),
              EqualsProto(wantConfidenceInterval));
}

TEST(CountTest, BasicOverflowTest) {
  absl::StatusOr<std::unique_ptr<Count<int64_t>>> count =
      typename Count<int64_t>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();

  CountTestPeer::AddMultipleEntries<int64_t>(
      1, std::numeric_limits<int64_t>::max(), &**count);
  (*count)->AddEntry(1);

  auto result = (*count)->PartialResult();
  ASSERT_OK(result);

  EXPECT_EQ(GetValue<int64_t>(*result), std::numeric_limits<int64_t>::lowest());
}

TEST(CountTest, OverflowFromAddNoseTypeCastTest) {
  int i;
  for (i = 0; i < 100; ++i) {
    absl::StatusOr<std::unique_ptr<Count<int64_t>>> count =
        typename Count<int64_t>::Builder()
            .SetEpsilon(kDefaultEpsilon)
            .SetLaplaceMechanism(absl::make_unique<LaplaceMechanism::Builder>())
            .Build();

    CountTestPeer::AddMultipleEntries<int64_t>(
        1, std::numeric_limits<int64_t>::max(), &**count);

    auto result = (*count)->PartialResult();
    ASSERT_OK(result);
    // The added noise should eventually cause the count to overflow, resulting
    // in a negative count.
    if (GetValue<int64_t>(*result) < 0) {
      // An overflow has happened, so return to end the test as a success.
      return;
    }
  }
  FAIL() << "No overflow occurred after " << i << " iterations.";
}

TEST(CountTest, SerializeTest) {
  auto count = Count<double>::Builder().SetEpsilon(0.5).Build();
  ASSERT_OK(count);
  (*count)->AddEntry(1);
  (*count)->AddEntry(2);
  Summary summary = (*count)->Serialize();

  CountSummary count_summary;
  EXPECT_TRUE(summary.has_data());
  EXPECT_TRUE(summary.data().UnpackTo(&count_summary));
  EXPECT_EQ(count_summary.count(), 2);
}

TEST(CountTest, MergeTest) {
  // Create summary.
  CountSummary count_summary;
  count_summary.set_count(2);
  Summary summary;
  summary.mutable_data()->PackFrom(count_summary);

  // Merge.
  absl::StatusOr<std::unique_ptr<Count<double>>> count =
      Count<double>::Builder()
          .SetEpsilon(kDefaultEpsilon)
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(count);
  (*count)->AddEntry(0);

  EXPECT_OK((*count)->Merge(summary));

  auto result = (*count)->PartialResult();
  ASSERT_OK(result);

  EXPECT_EQ(GetValue<int64_t>(*result), 3);
}

TEST(CountTest, SerializeAndMergeOverflowTest) {
  Count<int64_t>::Builder builder;
  builder.SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>());
  absl::StatusOr<std::unique_ptr<Count<int64_t>>> count1 =
      builder.SetEpsilon(kDefaultEpsilon).Build();
  ASSERT_OK(count1);
  CountTestPeer::AddMultipleEntries<int64_t>(
      1, std::numeric_limits<int64_t>::max(), &**count1);
  Summary summary = (*count1)->Serialize();

  absl::StatusOr<std::unique_ptr<Count<int64_t>>> count2 = builder.Build();
  ASSERT_OK(count2);
  (*count2)->AddEntry(1);
  EXPECT_OK((*count2)->Merge(summary));

  absl::StatusOr<Output> result = (*count2)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<int64_t>(*result), std::numeric_limits<int64_t>::lowest());

  // Test post-overflow serialize & merge
  summary = (*count2)->Serialize();
  count2 = builder.Build();
  ASSERT_OK((*count2)->Merge(summary));
  result = (*count2)->PartialResult();
  ASSERT_OK(result.status());
  EXPECT_DOUBLE_EQ(GetValue<int64_t>(result.value()),
                   std::numeric_limits<int64_t>::lowest());
}

TEST(CountTest, MemoryUsed) {
  absl::StatusOr<std::unique_ptr<Count<double>>> count =
      Count<double>::Builder().SetEpsilon(kDefaultEpsilon).Build();
  ASSERT_OK(count);
  EXPECT_GT((*count)->MemoryUsed(), 0);
}

TEST(CountTest, DeltaNotSetGaussian) {
  absl::StatusOr<std::unique_ptr<Count<double>>> failed_count =
      Count<double>::Builder()
          .SetEpsilon(0.5)
          .SetLaplaceMechanism(
              absl::make_unique<
                  differential_privacy::GaussianMechanism::Builder>())
          .Build();
  EXPECT_THAT(failed_count,
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Delta")));
}

TEST(CountTest, BasicGaussian) {
  std::vector<int> c = {1, 2, 3, 4, 2, 3};
  absl::StatusOr<std::unique_ptr<Count<int>>> count =
      typename Count<int>::Builder()
          .SetEpsilon(1e100)
          .SetDelta(0.99)
          .SetLaplaceMechanism(
              absl::make_unique<
                  differential_privacy::GaussianMechanism::Builder>())
          .Build();
  ASSERT_OK(count);
  auto result = (*count)->Result(c.begin(), c.end());
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<int64_t>(*result), 6);
}

}  // namespace
}  // namespace differential_privacy
