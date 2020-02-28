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

#include "differential_privacy/algorithms/count.h"

#include <memory>

#include "google/protobuf/any.pb.h"
#include "differential_privacy/base/testing/proto_matchers.h"
#include "differential_privacy/base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "differential_privacy/algorithms/numerical-mechanisms-testing.h"
#include "differential_privacy/proto/data.pb.h"
#include "differential_privacy/proto/summary.pb.h"

namespace differential_privacy {
namespace {

using test_utils::ZeroNoiseMechanism;
using testing::Contains;
using ::differential_privacy::base::testing::EqualsProto;
using ::differential_privacy::base::testing::IsOkAndHolds;

template <typename T>
class CountTest : public testing::Test {};

typedef ::testing::Types<int64_t, double> NumericTypes;
TYPED_TEST_SUITE(CountTest, NumericTypes);

TYPED_TEST(CountTest, BasicTest) {
  std::vector<TypeParam> c = {1, 2, 3, 4, 2, 3};
  std::unique_ptr<Count<TypeParam>> count =
      typename Count<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  EXPECT_EQ(GetValue<int64_t>(count->Result(c.begin(), c.end()).ValueOrDie()), 6);
}

TYPED_TEST(CountTest, RepeatedResultTest) {
  std::vector<TypeParam> c = {1, 2, 3, 4, 2, 3};
  std::unique_ptr<Count<TypeParam>> count =
      typename Count<TypeParam>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();

  count->AddEntries(c.begin(), c.end());

  EXPECT_EQ(GetValue<int64_t>(count->PartialResult(0.5).ValueOrDie()),
            GetValue<int64_t>(count->PartialResult(0.5).ValueOrDie()));
}

TEST(CountTest, ConfidenceIntervalTest) {
  double epsilon = 0.5;
  double level = .95;
  std::unique_ptr<Count<double>> count =
      Count<double>::Builder().SetEpsilon(0.5).Build().ValueOrDie();
  ConfidenceInterval wantConfidenceInterval;
  wantConfidenceInterval.set_lower_bound(log(1 - level) / epsilon);
  wantConfidenceInterval.set_upper_bound(-log(1 - level) / epsilon);
  wantConfidenceInterval.set_confidence_level(level);

  base::StatusOr<ConfidenceInterval> confidenceInterval =
      count->NoiseConfidenceInterval(level);
  EXPECT_THAT(confidenceInterval,
              IsOkAndHolds(EqualsProto(wantConfidenceInterval)));
  EXPECT_THAT(count->PartialResult()
                  .ValueOrDie()
                  .error_report()
                  .noise_confidence_interval(),
              EqualsProto(wantConfidenceInterval));
}

TEST(CountTest, SerializeTest) {
  std::unique_ptr<Count<double>> count =
      Count<double>::Builder().SetEpsilon(0.5).Build().ValueOrDie();
  count->AddEntry(1);
  count->AddEntry(2);
  Summary summary = count->Serialize();

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
  std::unique_ptr<Count<double>> count =
      Count<double>::Builder()
          .SetLaplaceMechanism(absl::make_unique<ZeroNoiseMechanism::Builder>())
          .Build()
          .ValueOrDie();
  count->AddEntry(0);

  EXPECT_OK(count->Merge(summary));
  EXPECT_EQ(GetValue<int64_t>(count->PartialResult().ValueOrDie()), 3);
}

TEST(CountTest, MemoryUsed) {
  std::unique_ptr<Count<double>> count =
      Count<double>::Builder().Build().ValueOrDie();
  EXPECT_GT(count->MemoryUsed(), 0);
}

}  // namespace
}  // namespace differential_privacy
