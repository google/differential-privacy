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

#include "algorithms/rand.h"

#include <numeric>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace differential_privacy {
namespace {
const int sample_size = 1000000;
const double tolerance = 0.01;

// This test asserts the mean and variance of the samples.
class RandTest : public ::testing::Test,
                 public testing::WithParamInterface<bool> {
 public:

  template <typename T>
  static void MeanAndVar(T (*rng)(), double* mean, double* var) {
    std::vector<T> samples(sample_size);
    for (int i = 0; i < sample_size; i++) {
      samples[i] = rng();
    }
    *mean = std::accumulate(samples.begin(), samples.end(), 0.0) / sample_size;
    *var = std::accumulate(samples.begin(), samples.end(), 0.0,
                           [mean](double x, double y) {
                             return x + std::pow(y - *mean, 2);
                           }) /
           (sample_size - 1);
  }

  template <typename T>
  static void RunTest(T (*rng)(), double expected_mean, double expected_var) {
    double mean, var;
    MeanAndVar(rng, &mean, &var);
    // Assert relative difference between sampled mean/variance and actual
    // mean/variance is smaller than epsilon_.
    EXPECT_LT(std::fabs(mean - expected_mean), tolerance * expected_mean);
    EXPECT_LT(std::fabs(var - expected_var), tolerance * expected_var);
  }
};

INSTANTIATE_TEST_SUITE_P(EnableDefaultDenyDelay, RandTest, testing::Bool());

TEST_P(RandTest, UniformDouble) {
  RunTest(UniformDouble, /*expected_mean=*/0.5, /*expected_var=*/1.0 / 12.0);
}

TEST_P(RandTest, Geometric) {
  RunTest(Geometric, /*expected_mean=*/2, /*expected_var=*/2);
}

}  // namespace
}  // namespace differential_privacy
