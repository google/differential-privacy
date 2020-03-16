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

#include "differential_privacy/algorithms/distributions.h"

#include <unordered_map>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "differential_privacy/algorithms/numerical-mechanisms-testing.h"
#include "differential_privacy/algorithms/util.h"
#include "differential_privacy/base/status.h"

namespace differential_privacy {
namespace internal {
namespace {

constexpr int64_t kNumSamples = 10000000;
constexpr double kOneOverLog2 = 1.44269504089;

double Skew(const std::vector<double>& samples, double mu, double sigma) {
  double skew = std::accumulate(
      samples.begin(), samples.end(), 0.0, [&mu](double lhs, double rhs) {
        return lhs + (rhs - mu) * (rhs - mu) * (rhs - mu);
      });
  return skew / (samples.size() * sigma * sigma * sigma);
}

double Kurtosis(const std::vector<double>& samples, double mu, double var) {
  double kurt = std::accumulate(samples.begin(), samples.end(), 0.0,
                                [mu](double lhs, double rhs) {
                                  double m4 = (rhs - mu) * (rhs - mu);
                                  m4 *= m4;
                                  return lhs + m4;
                                });

  int64_t n = samples.size();
  kurt = (n + 1) * kurt / (n * var * var);
  kurt -= 3 * (n - 1);
  kurt *= static_cast<double>(n - 1) / (n - 2) / (n - 3);
  return kurt;
}

TEST(LaplaceDistributionTest, CheckStatisticsForUnitValues) {
  LaplaceDistribution dist(1.0);
  std::vector<double> samples(kNumSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist]() { return dist.Sample(); });
  double mean = Mean(samples);
  double var = Variance(samples);
  EXPECT_NEAR(0.0, mean, 0.01);
  EXPECT_NEAR(2.0, var, 0.1);
  EXPECT_NEAR(0.0, Skew(samples, mean, std::sqrt(var)), 0.1);
  EXPECT_NEAR(3.0, Kurtosis(samples, mean, var), 0.1);
}

TEST(LaplaceDistributionTest, CheckStatisticsForSpecificDistribution) {
  double b = kOneOverLog2;
  LaplaceDistribution dist(b);
  std::vector<double> samples(kNumSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist]() { return dist.Sample(); });
  double mean = Mean(samples);
  double var = Variance(samples);

  EXPECT_NEAR(0.0, mean, 0.01);
  EXPECT_NEAR(2.0 * b * b, var, 0.1);
  EXPECT_NEAR(0.0, Skew(samples, mean, std::sqrt(var)), 0.1);
  EXPECT_NEAR(3.0, Kurtosis(samples, mean, var), 0.1);
}

TEST(LaplaceDistributionTest, CheckStatisticsForSpecificScaledDistribution) {
  double b = kOneOverLog2;
  double scale = 3.0;
  LaplaceDistribution dist(b);
  std::vector<double> samples(kNumSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist, scale]() { return dist.Sample(scale); });
  EXPECT_NEAR(0.0, Mean(samples), 0.01 * scale);
  EXPECT_NEAR(2.0 * scale * scale * b * b, Variance(samples), 0.15 * scale);
}

TEST(LaplaceDistributionTest, DiversityGetter) {
  double b = kOneOverLog2;
  LaplaceDistribution dist(b);

  EXPECT_EQ(dist.GetDiversity(), b);
}

TEST(LaplaceDistributionTest, Cdf) {
  EXPECT_EQ(LaplaceDistribution::cdf(5, 0), .5);
  EXPECT_EQ(LaplaceDistribution::cdf(1, -1), .5 * exp(-1));
  EXPECT_EQ(LaplaceDistribution::cdf(1, 1), 1 - .5 * exp(-1));
}

TEST(GaussDistributionTest, CheckStatisticsForUnitValues) {
  GaussianDistribution dist(1.0);
  std::vector<double> samples(kNumSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist]() { return dist.Sample(); });
  EXPECT_NEAR(0.0, Mean(samples), 0.01);
  EXPECT_NEAR(1.0, Variance(samples), 0.1);
}

TEST(GaussDistributionTest, CheckStatisticsForSpecificDistribution) {
  double stddev = kOneOverLog2;
  GaussianDistribution dist(stddev);
  std::vector<double> samples(kNumSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist]() { return dist.Sample(); });
  EXPECT_NEAR(0.0, Mean(samples), 0.01);
  EXPECT_NEAR(stddev * stddev, Variance(samples), 0.1);
}

TEST(GaussDistributionTest, CheckStatisticsForSpecificScaledDistribution) {
  double stddev = kOneOverLog2;
  double scale = 3.0;
  GaussianDistribution dist(stddev);
  std::vector<double> samples(kNumSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist, scale]() { return dist.Sample(scale); });
  EXPECT_NEAR(0.0, Mean(samples), 0.01 * scale);
  EXPECT_NEAR(stddev * stddev * scale * scale, Variance(samples), 0.1 * scale);
}

TEST(GaussDistributionTest, StandardDeviationGetter) {
  double stddev = kOneOverLog2;
  GaussianDistribution dist(stddev);

  EXPECT_EQ(dist.Stddev(), stddev);
}

}  // namespace
}  // namespace internal
}  // namespace differential_privacy
