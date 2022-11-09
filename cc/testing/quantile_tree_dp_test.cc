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

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/quantile-tree.h"
#include "testing/statistical_tests_utils.h"
#include "proto/testing/statistical_tests.pb.h"

namespace differential_privacy {
namespace testing {
namespace {

using differential_privacy::QuantileTree;
using ::testing::BoundedQuantilesDpTestCase;
using ::testing::BoundedQuantilesDpTestCaseCollection;
using ::testing::BoundedQuantilesSamplingParameters;
using ::testing::DpTestParameters;
using ::testing::Message;

class QuantileTreeDpTest
    : public ::testing::TestWithParam<BoundedQuantilesDpTestCase> {};

constexpr char kTestCaseProtoPath[] =
    "external/com_google_differential_privacy/proto/testing/"
    "bounded_quantiles_dp_test_cases.textproto";

static bool GenerateVote(
    std::function<std::vector<double>()> sample_generator_a,
    std::function<std::vector<double>()> sample_generator_b, int num_samples,
    int num_ranks, double lower, double upper, double epsilon, double delta,
    double delta_tolerance, int num_buckets) {
  std::vector<std::vector<double>> samples_a(num_ranks);
  std::vector<std::vector<double>> samples_b(num_ranks);

  for (int i = 0; i < num_samples; ++i) {
    std::vector<double> sample_a = sample_generator_a();
    std::vector<double> sample_b = sample_generator_b();
    for (int j = 0; j < num_ranks; ++j) {
      samples_a[j].push_back(Bucketize(sample_a[j], lower, upper, num_buckets));
      samples_b[j].push_back(Bucketize(sample_b[j], lower, upper, num_buckets));
    }
  }

  // Only vote to accept if all quantiles pass the test.
  for (int j = 0; j < num_ranks; ++j) {
    if (!VerifyApproximateDp(samples_a[j], samples_b[j], epsilon, delta,
                             delta_tolerance)) {
      return false;
    }
  }
  return true;
}

// Execute a test case from bounded_quantiles_dp_test_cases.textproto.
// We set up quantile trees as the parameters specify, add the specified inputs,
// and then check that the output distribution does not violate the differential
// privacy definition.
TEST_P(QuantileTreeDpTest, RunTestCasesAndCountVotes) {
  BoundedQuantilesDpTestCase test_case = GetParam();
  SCOPED_TRACE(Message() << "Test case " << test_case.name());

  BoundedQuantilesSamplingParameters sampling_params =
      test_case.bounded_quantiles_sampling_parameters();
  DpTestParameters dp_test_params = test_case.dp_test_parameters();

  QuantileTree<double>::Builder quantile_builder;
  quantile_builder.SetTreeHeight(sampling_params.tree_height())
      .SetBranchingFactor(sampling_params.branching_factor())
      .SetLower(sampling_params.lower_bound())
      .SetUpper(sampling_params.upper_bound());

  std::unique_ptr<QuantileTree<double>> tree = quantile_builder.Build().value();
  for (double raw_entry : sampling_params.raw_entry()) {
    tree->AddEntry(raw_entry);
  }

  std::unique_ptr<QuantileTree<double>> neighbor_tree =
      quantile_builder.Build().value();
  for (double neighbor_entry : sampling_params.neighbour_raw_entry()) {
    neighbor_tree->AddEntry(neighbor_entry);
  }

  QuantileTree<double>::DPParams dp_params;
  dp_params.epsilon = sampling_params.epsilon();
  dp_params.max_contributions_per_partition =
      sampling_params.max_contributions_per_partition();
  dp_params.max_partitions_contributed_to =
      sampling_params.max_partitions_contributed();

  switch (sampling_params.noise_type()) {
    case ::testing::NoiseType::LAPLACE:
      dp_params.mechanism_builder =
          std::make_unique<LaplaceMechanism::Builder>();
      dp_params.delta = 0;
      break;
    case ::testing::NoiseType::GAUSSIAN:
      dp_params.mechanism_builder =
          std::make_unique<GaussianMechanism::Builder>();
      dp_params.delta = sampling_params.delta();
      break;
    default:
      FAIL() << "Unknown noise type";
  }

  std::function<std::vector<double>()> sample_generator = [&tree, &dp_params,
                                                           &sampling_params]() {
    QuantileTree<double>::Privatized privatized_tree =
        tree->MakePrivate(dp_params).value();
    std::vector<double> results;
    for (double rank : sampling_params.rank()) {
      results.push_back(privatized_tree.GetQuantile(rank).value());
    }
    return results;
  };

  std::function<std::vector<double>()> neighbor_generator =
      [&neighbor_tree, &dp_params, &sampling_params]() {
        QuantileTree<double>::Privatized privatized_tree =
            neighbor_tree->MakePrivate(dp_params).value();
        std::vector<double> results;
        for (double rank : sampling_params.rank()) {
          results.push_back(privatized_tree.GetQuantile(rank).value());
        }
        return results;
      };

  std::function<bool()> vote_generator = [&sample_generator,
                                          &neighbor_generator, &dp_test_params,
                                          &sampling_params]() {
    return GenerateVote(
        sample_generator, neighbor_generator,
        sampling_params.number_of_samples(), sampling_params.rank_size(),
        sampling_params.lower_bound(), sampling_params.upper_bound(),
        dp_test_params.epsilon(), dp_test_params.delta(),
        dp_test_params.delta_tolerance(), dp_test_params.num_of_buckets());
  };

  int number_of_votes =
      ReadProto<BoundedQuantilesDpTestCaseCollection>(kTestCaseProtoPath)
          ->voting_parameters()
          .number_of_votes();

  EXPECT_TRUE(RunBallot(vote_generator, number_of_votes));
}

INSTANTIATE_TEST_SUITE_P(
    ShardedStatTests, QuantileTreeDpTest,
    ValuesIn(ReadProto<BoundedQuantilesDpTestCaseCollection>(kTestCaseProtoPath)
                 ->bounded_quantiles_dp_test_case()));

}  // namespace
}  // namespace testing
}  // namespace differential_privacy
