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

#include <memory>

#include "google/protobuf/any.pb.h"
#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "base/statusor.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"

namespace differential_privacy {

// Provides limited-scope static methods for interacting with a Count object for
// testing purposes.
class CountTestPeer {
 public:
  template <typename T>
  static void AddMultipleEntries(const T& v, uint64_t num_of_entries,
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

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(CountTest, NumericTypes);

TYPED_TEST(CountTest, BasicTest) {
  std::vector<TypeParam> c = {1, 2, 3, 4, 2, 3};
  auto count =
      typename Count<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(count);
  auto result = (*count)->Result(c.begin(), c.end());
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<int64_t>(*result), 6);
}

TYPED_TEST(CountTest, RepeatedResultTest) {
  std::vector<TypeParam> c = {1, 2, 3, 4, 2, 3};
  auto count =
      typename Count<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(count);

  (*count)->AddEntries(c.begin(), c.end());

  auto result1 = (*count)->PartialResult(0.5);
  ASSERT_OK(result1);
  auto result2 = (*count)->PartialResult(0.5);
  ASSERT_OK(result2);

  EXPECT_EQ(GetValue<int64_t>(*result1), GetValue<int64_t>(*result2));
}

TYPED_TEST(CountTest, InsufficientPrivacyBudgetTest) {
  std::vector<TypeParam> c = {1, 2, 3, 4, 2, 3};
  auto count =
      typename Count<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();
  ASSERT_OK(count);

  (*count)->AddEntries(c.begin(), c.end());

  ASSERT_OK((*count)->PartialResult());
  EXPECT_THAT((*count)->PartialResult(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Privacy budget must be positive")));
}

TEST(CountTest, ConfidenceIntervalTest) {
  double epsilon = 0.5;
  double level = .95;
  auto count = Count<double>::Builder().SetEpsilon(0.5).Build();
  ASSERT_OK(count);

  ConfidenceInterval wantConfidenceInterval;
  wantConfidenceInterval.set_lower_bound(log(1 - level) / epsilon);
  wantConfidenceInterval.set_upper_bound(-log(1 - level) / epsilon);
  wantConfidenceInterval.set_confidence_level(level);

  base::StatusOr<ConfidenceInterval> confidenceInterval =
      (*count)->NoiseConfidenceInterval(level);
  EXPECT_THAT(confidenceInterval,
              IsOkAndHolds(EqualsProto(wantConfidenceInterval)));

  auto actual_result = (*count)->PartialResult();
  ASSERT_OK(actual_result);

  EXPECT_THAT(actual_result->error_report().noise_confidence_interval(),
              EqualsProto(wantConfidenceInterval));
}

TEST(CountTest, OverflowTest) {
  auto count =
      typename Count<uint64_t>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build();

  CountTestPeer::AddMultipleEntries<uint64_t>(
      1, std::numeric_limits<uint64_t>::max(), &**count);
  (*count)->AddEntry(1);

  auto result = (*count)->PartialResult();
  ASSERT_OK(result);

  EXPECT_EQ(GetValue<int64_t>(*result), 0);
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
  auto count =
      Count<double>::Builder()
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
  Count<uint64_t>::Builder builder;
  builder.SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>());
  base::StatusOr<std::unique_ptr<Count<uint64_t>>> count1 = builder.Build();
  ASSERT_OK(count1);
  CountTestPeer::AddMultipleEntries<uint64_t>(
      1, std::numeric_limits<uint64_t>::max(), &**count1);
  Summary summary = (*count1)->Serialize();

  base::StatusOr<std::unique_ptr<Count<uint64_t>>> count2 = builder.Build();
  ASSERT_OK(count2);
  (*count2)->AddEntry(1);
  EXPECT_OK((*count2)->Merge(summary));

  base::StatusOr<Output> result = (*count2)->PartialResult();
  ASSERT_OK(result);
  EXPECT_EQ(GetValue<int64_t>(*result), 0);

  // Test post-overflow serialize & merge
  summary = (*count2)->Serialize();
  count2 = builder.Build();
  ASSERT_OK((*count2)->Merge(summary));
  result = (*count2)->PartialResult();
  ASSERT_OK(result.status());
  EXPECT_DOUBLE_EQ(GetValue<int64_t>(result.value()), 0);
}

TEST(CountTest, MemoryUsed) {
  auto count = Count<double>::Builder().Build();
  ASSERT_OK(count);
  EXPECT_GT((*count)->MemoryUsed(), 0);
}

TEST(CountTest, DeltaNotSetGaussian) {
  auto failed_count =
      Count<double>::Builder()
          .SetEpsilon(0.5)
          .SetLaplaceMechanism(
              absl::make_unique<
                  differential_privacy::GaussianMechanism::Builder>())
          .Build();
  EXPECT_THAT(failed_count, StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Delta has to be set")));
}

TEST(CountTest, BasicGaussian) {Z
  std::vector<int> c = {1, 2, 3, 4, 2, 3};
  auto count = typename Count<int>::Builder()
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
