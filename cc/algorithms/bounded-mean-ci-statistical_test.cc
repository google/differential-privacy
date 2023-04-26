//
// Copyright 2022 Google LLC
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

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "algorithms/approx-bounds.h"
#include "algorithms/bounded-mean.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/confidence-interval.pb.h"
#include "proto/data.pb.h"

namespace differential_privacy {
namespace {

using ::testing::_;
using ::testing::Gt;
using ::testing::NiceMock;
using ::testing::Return;

TEST(BoundedMeanTest, ConfidenceIntervalBasicPropertyTestForFixedBounds) {
  // Runs the algorithm multiple times and checks that the true mean value is in
  // the confidence interval according to the confidence level.
  const double epsilon = std::log(3);
  const double lower = -1.0;
  const double upper = 1.0;
  const double confidence_level = 0.95;
  const int num_trials = 100000;
  const std::vector<double> inputs = {-1.0, 0.0, 1.0, -0.5, 0.5,
                                      0.0,  0.0, 0.8, 0.7,  -0.2};

  double inputs_sum = 0;
  for (double d : inputs) {
    inputs_sum += d;
  }
  const double raw_mean = inputs_sum / inputs.size();

  int hits = 0;
  for (int i = 0; i < num_trials; ++i) {
    absl::StatusOr<std::unique_ptr<BoundedMean<double>>> mean =
        BoundedMean<double>::Builder()
            .SetEpsilon(epsilon)
            .SetLower(lower)
            .SetUpper(upper)
            .Build();
    ASSERT_OK(mean);

    for (double d : inputs) {
      mean->get()->AddEntry(d);
    }
    absl::StatusOr<Output> output =
        mean->get()->PartialResult(confidence_level);
    ASSERT_OK(output);

    const ConfidenceInterval ci =
        output->elements(0).noise_confidence_interval();
    if (ci.lower_bound() <= raw_mean && raw_mean <= ci.upper_bound()) {
      ++hits;
    }
  }

  // This test fails with a probability of 1e-8.  If it fails more frequently it
  // should be considered as constantly failing.  To get to 94609, we calculate
  // the CDF of the binomial distribution for the 100,000 trials and the success
  // probability of 0.95 and find the greatest number for which the CDF is less
  // than 1e-8.
  ASSERT_THAT(hits, Gt(94609));
}

class MockApproxBounds : public ApproxBounds<double> {
 public:
  MockApproxBounds()
      : ApproxBounds(
            /*epsilon=*/0.1,
            /*num_bins=*/1024,
            /*scale=*/1.0,
            /*base=*/2.0,
            /*success_probability=*/1.0 - 1e-9,
            /*has_user_set_threshold=*/false,
            /*mechanism=*/
            LaplaceMechanism::Builder()
                .SetL1Sensitivity(1)
                .SetEpsilon(0.1)
                .Build()
                .value()) {}

  MOCK_METHOD(absl::StatusOr<Output>, GenerateResult,
              (double noise_interval_level), (override));
};

TEST(BoundedMeanTest, ConfidenceIntervalBasicPropertyTestForApproxBounds) {
  const double confidence_level = 0.95;
  const int num_trials = 100000;
  const std::vector<double> inputs = {-1.0, 0.0, 1.0, -0.5, 0.5,
                                      0.0,  0.0, 0.8, 0.7,  -0.2};

  double inputs_sum = 0;
  for (double d : inputs) {
    inputs_sum += d;
  }
  const double raw_mean = inputs_sum / inputs.size();

  int hits = 0;
  for (int i = 0; i < num_trials; ++i) {
    auto mock_approx_bounds = std::make_unique<NiceMock<MockApproxBounds>>();
    Output approx_bounds_result;
    approx_bounds_result.add_elements()->mutable_value()->set_float_value(-1.0);
    approx_bounds_result.add_elements()->mutable_value()->set_float_value(1.0);
    EXPECT_CALL(*mock_approx_bounds, GenerateResult(_))
        .WillRepeatedly(Return(approx_bounds_result));

    absl::StatusOr<std::unique_ptr<BoundedMean<double>>> mean =
        BoundedMean<double>::Builder()
            .SetEpsilon(1.1)
            .SetApproxBounds(std::move(mock_approx_bounds))
            .Build();
    ASSERT_OK(mean);

    for (double d : inputs) {
      mean->get()->AddEntry(d);
    }

    absl::StatusOr<Output> output =
        mean->get()->PartialResult(confidence_level);
    ASSERT_OK(output);

    const ConfidenceInterval ci =
        output->elements(0).noise_confidence_interval();

    if (ci.lower_bound() <= raw_mean && raw_mean <= ci.upper_bound()) {
      ++hits;
    }
  }

  // This test fails with a probability of 1e-8.  If it fails more frequently it
  // should be considered as constantly failing.  To get to 94609, we calculate
  // the CDF of the binomial distribution for the 100,000 trials and the success
  // probability of 0.95 and find the greatest number for which the CDF is less
  // than 1e-8.
  ASSERT_THAT(hits, Gt(94609));
}

}  // namespace
}  // namespace differential_privacy
