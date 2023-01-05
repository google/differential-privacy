//
// Copyright 2021 Google LLC
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

#include "testing/statistical_tests_utils.h"

#include "google/protobuf/text_format.h"
#include "base/testing/proto_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "algorithms/util.h"
#include "proto/testing/statistical_tests.pb.h"

namespace differential_privacy::testing {
namespace {

const double kLowL2Tolerance = 0.000000001;
const double kDefaultL2Tolerance = 0.001;
const double kHighL2Tolerance = 0.5;

const double kDefaultEpsilon = 1.0;
const double kDefaultDelta = 0.00001;
const double kLowDeltaTolerance = 0.0000000001;
const double kDefaultDeltaTolerance = 0.00001;
const double kHighDeltaTolerance = 0.5;

const double kNumSamples = 1000000;

// A callable object that will return the items in its input vector in order.
struct VectorGenerator {
  std::vector<double> samples_;
  VectorGenerator(std::vector<double> samples) : samples_(samples) {}
  double operator()() {
    double to_return = *samples_.begin();
    samples_.erase(samples_.begin());
    return to_return;
  }
};

TEST(ClosenessVoteTest, AcceptsIdenticalSamples) {
  std::vector<double> samples = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0,
                                 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};

  // We use granularity = 1.0 because the samples are already multiples of 1.0.
  EXPECT_TRUE(GenerateClosenessVote(VectorGenerator(samples),
                                    VectorGenerator(samples), samples.size(),
                                    kLowL2Tolerance, /*granularity=*/1.0));
  EXPECT_TRUE(GenerateClosenessVote(VectorGenerator(samples),
                                    VectorGenerator(samples), samples.size(),
                                    kHighL2Tolerance, /*granularity=*/1.0));
}

TEST(ClosenessVoteTest, RejectsDifferentSamples) {
  std::vector<double> samples_a = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0,
                                   4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  std::vector<double> samples_b = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                   2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};

  // We use granularity = 1.0 because the samples are already multiples of 1.0.
  EXPECT_FALSE(GenerateClosenessVote(
      VectorGenerator(samples_a), VectorGenerator(samples_b), samples_a.size(),
      kDefaultL2Tolerance, /*granularity=*/1.0));
  EXPECT_FALSE(GenerateClosenessVote(
      VectorGenerator(samples_a), VectorGenerator(samples_b), samples_a.size(),
      kLowL2Tolerance, /*granularity=*/1.0));
}

TEST(ClosenessVoteTest, AcceptsDifferentSamplesWithHighTolernace) {
  std::vector<double> samples_a = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0,
                                   4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  std::vector<double> samples_b = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                   2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};

  // We use granularity = 1.0 because the samples are already multiples of 1.0.
  EXPECT_TRUE(GenerateClosenessVote(
      VectorGenerator(samples_a), VectorGenerator(samples_b), samples_a.size(),
      kHighL2Tolerance, /*granularity=*/1.0));
}

TEST(ClosenessVoteTest, InvariantToSampleOrder) {
  std::vector<double> samples_a = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0,
                                   4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  std::vector<double> samples_a_unsorted = {5.0, 3.0, 2.0, 3.0, 1.0,
                                            2.0, 5.0, 3.0, 4.0, 5.0,
                                            5.0, 4.0, 5.0, 4.0, 4.0};
  std::vector<double> samples_b = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                   2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};
  std::vector<double> samples_b_unsorted = {4.0, 1.0, 5.0, 1.0, 4.0,
                                            2.0, 3.0, 3.0, 1.0, 2.0,
                                            2.0, 1.0, 3.0, 1.0, 2.0};

  EXPECT_TRUE(GenerateClosenessVote(
      VectorGenerator(samples_a), VectorGenerator(samples_a_unsorted),
      samples_a.size(), kDefaultL2Tolerance, /*granularity=*/1.0));
  EXPECT_TRUE(GenerateClosenessVote(
      VectorGenerator(samples_a_unsorted), VectorGenerator(samples_a),
      samples_a_unsorted.size(), kDefaultL2Tolerance, /*granularity=*/1.0));
  EXPECT_FALSE(GenerateClosenessVote(
      VectorGenerator(samples_a), VectorGenerator(samples_b_unsorted),
      samples_a.size(), kDefaultL2Tolerance, /*granularity=*/1.0));
  EXPECT_FALSE(GenerateClosenessVote(
      VectorGenerator(samples_a_unsorted), VectorGenerator(samples_b),
      samples_a_unsorted.size(), kDefaultL2Tolerance, /*granularity=*/1.0));
  EXPECT_FALSE(GenerateClosenessVote(
      VectorGenerator(samples_a_unsorted), VectorGenerator(samples_b_unsorted),
      samples_a_unsorted.size(), kDefaultL2Tolerance, /*granularity=*/1.0));
}

TEST(ApproximateDpVoteTest, AcceptsIdenticalSamples) {
  std::vector<double> samples = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0,
                                 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};

  // Identical sample sets should accept with epsilon and delta = 0, and almost
  // any delta tolerance. We use granularity = 1.0 because the samples are
  // already multiples of 1.0.
  EXPECT_TRUE(GenerateApproximateDpVote(
      VectorGenerator(samples), VectorGenerator(samples), samples.size(),
      /*epsilon=*/0.0, /*delta=*/0.0, kLowDeltaTolerance, /*granularity=*/1.0));
  EXPECT_TRUE(GenerateApproximateDpVote(
      VectorGenerator(samples), VectorGenerator(samples), samples.size(),
      /*epsilon=*/0.0, /*delta=*/0.0, kHighL2Tolerance, /*granularity=*/1.0));
}

TEST(ApproximateDpVoteTest, RejectsDifferentSamples) {
  std::vector<double> samples_a = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0,
                                   4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  std::vector<double> samples_b = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                   2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};

  // We use granularity = 1.0 because the samples are already multiples of 1.0.
  EXPECT_FALSE(GenerateApproximateDpVote(
      VectorGenerator(samples_a), VectorGenerator(samples_b), samples_a.size(),
      kDefaultEpsilon, kDefaultDelta, kDefaultDeltaTolerance,
      /*granularity=*/1.0));
  EXPECT_FALSE(GenerateApproximateDpVote(
      VectorGenerator(samples_a), VectorGenerator(samples_b), samples_a.size(),
      kDefaultEpsilon, kDefaultDelta, kLowDeltaTolerance, /*granularity=*/1.0));
}

TEST(ApproximateDpVoteTest, AcceptsDifferentSamplesWithHighTolerance) {
  std::vector<double> samples_a = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0,
                                   4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  std::vector<double> samples_b = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                   2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};

  // We use granularity = 1.0 because the samples are already multiples of 1.0.
  EXPECT_TRUE(GenerateApproximateDpVote(
      VectorGenerator(samples_a), VectorGenerator(samples_b), samples_a.size(),
      kDefaultEpsilon, kDefaultDelta, kHighDeltaTolerance,
      /*granularity=*/1.0));
}

TEST(ApproximateDpVoteTest, InvariantToSampleOrder) {
  std::vector<double> samples_a = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0,
                                   4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  std::vector<double> samples_a_unsorted = {5.0, 3.0, 2.0, 3.0, 1.0,
                                            2.0, 5.0, 3.0, 4.0, 5.0,
                                            5.0, 4.0, 5.0, 4.0, 4.0};
  std::vector<double> samples_b = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                   2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};
  std::vector<double> samples_b_unsorted = {4.0, 1.0, 5.0, 1.0, 4.0,
                                            2.0, 3.0, 3.0, 1.0, 2.0,
                                            2.0, 1.0, 3.0, 1.0, 2.0};

  // Identical sample sets should accept with epsilon and delta = 0, and almost
  // any delta tolerance. We use granularity = 1.0 because the samples are
  // already multiples of 1.0.
  EXPECT_TRUE(GenerateApproximateDpVote(
      VectorGenerator(samples_a), VectorGenerator(samples_a_unsorted),
      samples_a.size(),
      /*epsilon=*/0.0, /*delta=*/0.0, kDefaultL2Tolerance,
      /*granularity=*/1.0));
  EXPECT_TRUE(GenerateApproximateDpVote(
      VectorGenerator(samples_a_unsorted), VectorGenerator(samples_a),
      samples_a_unsorted.size(),
      /*epsilon=*/0.0, /*delta=*/0.0, kDefaultL2Tolerance,
      /*granularity=*/1.0));
  EXPECT_FALSE(GenerateApproximateDpVote(
      VectorGenerator(samples_a), VectorGenerator(samples_b_unsorted),
      samples_a.size(), kDefaultEpsilon, kDefaultDelta, kDefaultL2Tolerance,
      /*granularity=*/1.0));
  EXPECT_FALSE(GenerateApproximateDpVote(
      VectorGenerator(samples_a_unsorted), VectorGenerator(samples_b),
      samples_a_unsorted.size(), kDefaultEpsilon, kDefaultDelta,
      kDefaultL2Tolerance, /*granularity=*/1.0));
  EXPECT_FALSE(GenerateApproximateDpVote(
      VectorGenerator(samples_a_unsorted), VectorGenerator(samples_b_unsorted),
      samples_a_unsorted.size(), kDefaultEpsilon, kDefaultDelta,
      kDefaultL2Tolerance, /*granularity=*/1.0));
}

TEST(RunBallotTest, AcceptsMajorityTrue) {
  std::vector<bool> votes = {true, true, true, true, false, false, false};
  auto vote_it = votes.begin();
  std::function<bool()> vote_generator = [&vote_it]() { return *(vote_it++); };
  EXPECT_TRUE(RunBallot(vote_generator, votes.size()));
}

TEST(RunBallotTest, RejectsMajorityFalse) {
  std::vector<bool> votes = {true, true, true, false, false, false, false};
  auto vote_it = votes.begin();
  std::function<bool()> vote_generator = [&vote_it]() { return *(vote_it++); };
  EXPECT_FALSE(RunBallot(vote_generator, votes.size()));
}

TEST(ReferenceLaplaceTest, HasAccurateStatisticalProperties) {
  double mean = 0.0;
  double variance = 2.0;

  std::vector<double> samples;
  for (int i = 0; i < kNumSamples; ++i) {
    samples.push_back(
        SampleReferenceLaplacian(mean, variance, &SecureURBG::GetInstance()));
  }

  EXPECT_NEAR(Mean(samples), mean, 0.1);
  EXPECT_NEAR(Variance(samples), variance, 0.5);
}

TEST(ReadProtoTest, ReadProtoFromFile) {
  std::istringstream proto_file(
      R"(
  name: "Foo"
  dp_test_parameters {
    epsilon: 1.09861228866810969140  # = ln(3)
    delta: 0.0
    delta_tolerance: 0.01125
    granularity: 0.015625
  }
  noise_sampling_parameters {
    number_of_samples: 1000000
    l0_sensitivity: 1
    linf_sensitivity: 1.0
    epsilon: 1.09861228866810969140  # = ln(3)
    raw_input: 0.0
  }
)");

  ::testing::DistributionDpTestCase expected;
  google::protobuf::TextFormat::ParseFromString(
      R"pb(
        name: "Foo"
        dp_test_parameters {
          epsilon: 1.09861228866810969140  # = ln(3)
          delta: 0.0
          delta_tolerance: 0.01125
          granularity: 0.015625
        }
        noise_sampling_parameters {
          number_of_samples: 1000000
          l0_sensitivity: 1
          linf_sensitivity: 1.0
          epsilon: 1.09861228866810969140  # = ln(3)
          raw_input: 0.0
        }
      )pb",
      &expected);

  std::optional<::testing::DistributionDpTestCase> test_case =
      ReadProto<::testing::DistributionDpTestCase>(&proto_file);
  ASSERT_TRUE(test_case.has_value());
  EXPECT_THAT(test_case.value(), ::differential_privacy::base::testing::EqualsProto(expected));
}

TEST(BucketizeTest, BucketizesCorrectly) {
  EXPECT_EQ(Bucketize(0.5, 0, 10, 10), 0);
  EXPECT_EQ(Bucketize(5.5, 0, 10, 10), 5);
  EXPECT_EQ(Bucketize(9.6, 0, 10, 10), 9);

  EXPECT_EQ(Bucketize(-4.5, -5, 5, 10), 0);
  EXPECT_EQ(Bucketize(4.5, -5, 5, 10), 9);

  EXPECT_EQ(Bucketize(8, 0, 35, 5), 1);
  EXPECT_EQ(Bucketize(20, 0, 35, 5), 2);

  EXPECT_EQ(Bucketize(-5.5, -5, 5, 10), 0);
  EXPECT_EQ(Bucketize(-5, -5, 5, 10), 0);
  EXPECT_EQ(Bucketize(5, -5, 5, 10), 9);
  EXPECT_EQ(Bucketize(5.5, -5, 5, 10), 9);

  EXPECT_EQ(Bucketize(-1, -5, 5, 10), 4);
  EXPECT_EQ(Bucketize(0, -5, 5, 10), 5);
  EXPECT_EQ(Bucketize(1, -5, 5, 10), 6);
}

}  // namespace
}  // namespace differential_privacy::testing
