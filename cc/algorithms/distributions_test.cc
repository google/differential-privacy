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

#include "algorithms/distributions.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <unordered_map>

#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/util.h"

namespace differential_privacy {
namespace internal {
namespace {

using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Lt;
using ::differential_privacy::base::testing::StatusIs;

constexpr int64_t kNumSamples = 10000000;
constexpr int64_t kNumGeometricSamples = 1000000;
constexpr int64_t kGaussianSamples = 1000000;
constexpr double kOneOverLog2 = 1.44269504089;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

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

TEST(LaplaceDistributionTest, ParameterValidation) {
  LaplaceDistribution::Builder builder;

  EXPECT_THAT(builder.SetEpsilon(-1).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be finite and positive")));

  EXPECT_THAT(builder.SetEpsilon(0).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be finite and positive")));

  EXPECT_THAT(builder.SetEpsilon(kNan).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be a valid numeric value")));

  EXPECT_THAT(builder.SetEpsilon(kInfinity).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be finite and positive")));

  builder.SetEpsilon(1);
  EXPECT_THAT(builder.SetSensitivity(-1).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Sensitivity must be finite and positive")));

  EXPECT_THAT(builder.SetSensitivity(0).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Sensitivity must be finite and positive")));

  EXPECT_THAT(builder.SetSensitivity(kNan).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Sensitivity must be a valid numeric value")));

  EXPECT_THAT(builder.SetSensitivity(kInfinity).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Sensitivity must be finite and positive")));
}

TEST(LaplaceDistributionTest, CheckStatisticsForGeoUnitValues) {
  LaplaceDistribution::Builder builder;
  std::unique_ptr<LaplaceDistribution> dist =
      builder.SetEpsilon(1.0).SetSensitivity(1.0).Build().value();
  std::vector<double> samples(kNumGeometricSamples);
  std::generate(samples.begin(), samples.end(),
                [dist = std::move(dist)]() { return dist->Sample(); });
  double mean = Mean(samples);
  double var = Variance(samples);
  EXPECT_NEAR(0.0, mean, 0.01);
  EXPECT_NEAR(2.0, var, 0.1);
  EXPECT_NEAR(0.0, Skew(samples, mean, std::sqrt(var)), 0.1);
  EXPECT_NEAR(3.0, Kurtosis(samples, mean, var), 0.1);
}

TEST(LaplaceDistributionTest, CheckStatisticsForGeoSpecificDistribution) {
  double sensitivity = kOneOverLog2;
  LaplaceDistribution::Builder builder;
  std::unique_ptr<LaplaceDistribution> dist =
      builder.SetEpsilon(1.0).SetSensitivity(sensitivity).Build().value();
  std::vector<double> samples(kNumGeometricSamples);
  std::generate(samples.begin(), samples.end(),
                [dist = std::move(dist)]() { return dist->Sample(); });
  double mean = Mean(samples);
  double var = Variance(samples);

  EXPECT_NEAR(0.0, mean, 0.01);
  EXPECT_NEAR(2.0 * sensitivity * sensitivity, var, 0.1);
  EXPECT_NEAR(0.0, Skew(samples, mean, std::sqrt(var)), 0.1);
  EXPECT_NEAR(3.0, Kurtosis(samples, mean, var), 0.1);
}

TEST(LaplaceDistributionTest, Cdf) {
  EXPECT_EQ(LaplaceDistribution::cdf(5, 0), .5);
  EXPECT_EQ(LaplaceDistribution::cdf(1, -1), .5 * std::exp(-1));
  EXPECT_EQ(LaplaceDistribution::cdf(1, 1), 1 - .5 * std::exp(-1));
}

TEST(LaplaceDistributionTest, GetVarianceReturnsExpectedValue) {
  // Use epsilon = 1 and l1 sensitivity = 2 => 2 * 2^2 = 8
  absl::StatusOr<std::unique_ptr<LaplaceDistribution>> laplace_distribution =
      LaplaceDistribution::Builder().SetEpsilon(1).SetSensitivity(2).Build();
  ASSERT_OK(laplace_distribution);
  EXPECT_NEAR(laplace_distribution->get()->GetVariance(), 8, 1e-6);
}

TEST(LaplaceCalculateGranularityTest, InvalidParameters) {
  const double valid_param = kOneOverLog2;
  std::vector<double> invalid_params = {0, -0.1, -1, -10, kInfinity};

  for (double invalid_param : invalid_params) {
    EXPECT_THAT(
        LaplaceDistribution::CalculateGranularity(invalid_param, valid_param),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Epsilon must be finite and positive")));
    EXPECT_THAT(
        LaplaceDistribution::CalculateGranularity(valid_param, invalid_param),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Sensitivity must be finite and positive")));
  }

  // Test NaN
  EXPECT_THAT(LaplaceDistribution::CalculateGranularity(kNan, valid_param),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be a valid numeric value")));
  EXPECT_THAT(LaplaceDistribution::CalculateGranularity(valid_param, kNan),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Sensitivity must be a valid numeric value")));

  // Test epsilon < 2^-50. See (broken link)
  EXPECT_THAT(LaplaceDistribution::CalculateGranularity(
                  1.0 / (int64_t{1} << 51), valid_param),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Epsilon must be at least 2^-50")));
  EXPECT_THAT(LaplaceDistribution::CalculateGranularity(
                  1.0 / (int64_t{1} << 50), valid_param),
              StatusIs(absl::StatusCode::kOk));
}

TEST(LaplaceDistributionTest, GranularityIsBetweenZeroAndDiversity) {
  const double epsilon = 1.1;
  const double sensitivity = 2.0;

  absl::StatusOr<std::unique_ptr<LaplaceDistribution>> ld =
      LaplaceDistribution::Builder()
          .SetEpsilon(epsilon)
          .SetSensitivity(sensitivity)
          .Build();
  ASSERT_OK(ld);

  const double granularity = ld.value()->GetGranularity();
  EXPECT_THAT(granularity, Gt(0));

  const double diversity = sensitivity / epsilon;
  EXPECT_THAT(granularity, Lt(diversity));
}

TEST(GaussDistributionTest, ParameterValidation) {
  GaussianDistribution::Builder builder;

  EXPECT_THAT(
      builder.SetStddev(-1).Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Standard deviation must be finite and non-negative")));

  EXPECT_THAT(
      builder.SetStddev(kNan).Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Standard deviation must be a valid numeric value")));

  EXPECT_THAT(
      builder.SetStddev(kInfinity).Build(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Standard deviation must be finite and non-negative")));

  EXPECT_OK(builder.SetStddev(0).Build());
}

TEST(GaussDistributionTest, CheckStatisticsForUnitValues) {
  GaussianDistribution::Builder builder;
  std::unique_ptr<GaussianDistribution> dist =
      builder.SetStddev(1.0).Build().value();
  std::vector<double> samples(kGaussianSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist]() { return dist->Sample(); });
  EXPECT_NEAR(0.0, Mean(samples), 0.01);
  EXPECT_NEAR(1.0, Variance(samples), 0.1);
}

TEST(GaussDistributionTest, CheckStatisticsForSpecificDistribution) {
  double stddev = kOneOverLog2;
  GaussianDistribution::Builder builder;
  std::unique_ptr<GaussianDistribution> dist =
      builder.SetStddev(stddev).Build().value();
  std::vector<double> samples(kGaussianSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist]() { return dist->Sample(); });
  EXPECT_NEAR(0.0, Mean(samples), 0.01);
  EXPECT_NEAR(stddev * stddev, Variance(samples), 0.1);
}

TEST(GaussDistributionTest, CheckStatisticsForSpecificScaledDistribution) {
  double stddev = kOneOverLog2;
  double scale = 3.0;
  GaussianDistribution::Builder builder;
  std::unique_ptr<GaussianDistribution> dist =
      builder.SetStddev(stddev).Build().value();
  std::vector<double> samples(kGaussianSamples);
  std::generate(samples.begin(), samples.end(),
                [&dist, scale]() { return dist->Sample(scale); });
  EXPECT_NEAR(0.0, Mean(samples), 0.01 * scale);
  EXPECT_NEAR(stddev * stddev * scale * scale, Variance(samples), 0.1 * scale);
}

TEST(GaussDistributionTest, StandardDeviationGetter) {
  double stddev = kOneOverLog2;
  GaussianDistribution::Builder builder;
  std::unique_ptr<GaussianDistribution> dist =
      builder.SetStddev(stddev).Build().value();

  EXPECT_EQ(dist->Stddev(), stddev);
}

TEST(GaussDistributionTest, Cdf) {
  EXPECT_NEAR(GaussianDistribution::cdf(1, 0), 0.5, 0.000001);
  EXPECT_NEAR(GaussianDistribution::cdf(5, 0), 0.5, 0.000001);
  EXPECT_NEAR(GaussianDistribution::cdf(1, -1), 0.15865525, 0.000001);
  EXPECT_NEAR(GaussianDistribution::cdf(1, -5), 2.8665157e-7, 1e-13);
  EXPECT_NEAR(GaussianDistribution::cdf(1, -9), 1.1285884e-19, 1e-25);
  EXPECT_NEAR(GaussianDistribution::cdf(5, -1), 0.42074029, 0.000001);
  EXPECT_NEAR(GaussianDistribution::cdf(1, 1), 0.84134475, 0.000001);
  EXPECT_NEAR(GaussianDistribution::cdf(5, 1), 0.57925971, 0.000001);
  EXPECT_DEATH(GaussianDistribution::cdf(0, 1), "");
  EXPECT_DEATH(GaussianDistribution::cdf(-1, 1), "");
}

TEST(GaussDistributionTest, Quantile) {
  EXPECT_NEAR(GaussianDistribution::Quantile(1, 0.5), 0, 0.000001);
  EXPECT_NEAR(GaussianDistribution::Quantile(5, 0.5), 0, 0.000001);
  EXPECT_NEAR(GaussianDistribution::Quantile(1, 0.15865525), -1, 0.000001);
  EXPECT_NEAR(GaussianDistribution::Quantile(1, 2.8665157e-7), -5, 0.000001);
  // No test for 1.1285884e-19 as the precision is too low, and it evaluates to
  // +inf.
  EXPECT_NEAR(GaussianDistribution::Quantile(5, 0.42074029), -1, 0.000001);
  EXPECT_NEAR(GaussianDistribution::Quantile(1, 0.84134475), 1, 0.000001);
  EXPECT_NEAR(GaussianDistribution::Quantile(5, 0.57925971), 1, 0.000001);
  EXPECT_DEATH(GaussianDistribution::Quantile(0, 1), "");
  EXPECT_DEATH(GaussianDistribution::Quantile(-1, 1), "");
}

TEST(GeometricDistributionTest, ParameterValidation) {
  GeometricDistribution::Builder builder;

  EXPECT_THAT(builder.SetLambda(-1).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lambda must be finite and non-negative")));

  EXPECT_THAT(builder.SetLambda(kNan).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lambda must be a valid numeric value")));

  EXPECT_THAT(builder.SetLambda(kInfinity).Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Lambda must be finite and non-negative")));

  EXPECT_OK(builder.SetLambda(0).Build());
}

TEST(GeometricDistributionTest, SmallProbabilityStats) {
  GeometricDistribution::Builder builder;
  std::unique_ptr<GeometricDistribution> dist =
      builder.SetLambda(-1.0 * std::log(1.0 - 1e-6)).Build().value();

  std::vector<int64_t> samples(kNumGeometricSamples);
  std::generate(samples.begin(), samples.end(),
                [dist = std::move(dist)]() { return dist->Sample() + 1; });
  EXPECT_NEAR(1000000, Mean(samples), 10000);
  EXPECT_NEAR(999999.5, std::sqrt(Variance(samples)), 10000);
}

TEST(GeometricDistributionTest, LargeProbabilityStats) {
  GeometricDistribution::Builder builder;
  std::unique_ptr<GeometricDistribution> dist =
      builder.SetLambda(-1.0 * std::log(1.0 - 0.5)).Build().value();
  std::vector<int64_t> samples(kNumGeometricSamples);
  std::generate(samples.begin(), samples.end(),
                [dist = std::move(dist)]() { return dist->Sample() + 1; });
  EXPECT_NEAR(2, Mean(samples), 0.01);
  EXPECT_NEAR(std::sqrt(2), std::sqrt(Variance(samples)), 0.05);
}

TEST(GeometricDistributionTest, Ratios) {
  double p = 1e-2;
  GeometricDistribution::Builder builder;
  std::unique_ptr<GeometricDistribution> dist =
      builder.SetLambda(-1.0 * std::log(1.0 - p)).Build().value();
  std::vector<int64_t> counts(51, 0);
  for (int i = 0; i < kNumGeometricSamples; ++i) {
    int64_t sample = dist->Sample();
    if (sample < counts.size()) {
      ++counts[sample];
    }
  }
  std::vector<double> ratios;
  for (int i = 0; i < counts.size() - 1; ++i) {
    ratios.push_back(static_cast<double>(counts[i + 1]) /
                     static_cast<double>(counts[i]));
  }
  EXPECT_NEAR(p, Mean(ratios), p / 1e-2);
}

// For Binomial/Poisson RVs this is mult standard deviations since var= mean.
// Probability of failure with mult = 7 ~1e-23 if Gaussian approx holds, but it
// does not for low values of x, so we have to  add another fudge factor.
// Individual sample failure probability ~1e-13 and we test a few million
// samples overall so the overall flakiness of these tests is < 1e-6.
double AllowedError(int x, double mult = 7) {
  return (std::sqrt(x) + std::log1p(x)) * mult;
}

// Generates a std::string of the first N values from counts.
template <size_t N, typename T>
std::string FirstN(const std::vector<T>& counts) {
  auto spans = absl::MakeConstSpan(counts);
  if (spans.empty()) {
    return "<empty>";
  } else if (N < spans.size()) {
    spans = spans.first(N);
  }
  return absl::StrJoin(spans, ", ");
}

// Calculates the probability mass function of the Geometric
// distribution at an integer x:
//
// PMF(x) = (1-p)^x * p
//
double PMF(const double p, const int x) { return p * std::pow(1.0 - p, x); }

// Returns a list of the expected number of times a geometric distribution
// with parameter lambda will return each integer. Only returns integers
// for which the expected value is >= 2. Always returns the first 10 integers.
std::vector<double> ExpectedCounts(const double p, const double one_minus_p,
                                   const int num_samples) {
  std::vector<double> result;
  int i = 0;
  double expected_samples = PMF(p, i) * num_samples;
  while (i < 10 && expected_samples >= 2) {
    result.push_back(expected_samples);
    ++i;
    expected_samples = PMF(p, i) * num_samples;
  }

  return result;
}

class GeometricDistributionTest : public ::testing::TestWithParam<double> {};

TEST_P(GeometricDistributionTest, Distribution) {
  auto lambda = GetParam();

  GeometricDistribution::Builder builder;
  std::unique_ptr<GeometricDistribution> distribution =
      builder.SetLambda(lambda).Build().value();

  // Choose bucket sizes so that the expected count in the first bucket is
  // 150.
  const int64_t kBucketSize =
      std::ceil(-std::log1p(-150.0 / kNumGeometricSamples) / lambda);

  const int64_t kMaxValue = std::max(100.0, 100 / lambda);
  const size_t num_buckets = (kMaxValue / kBucketSize) + 1;
  std::vector<size_t> counts(num_buckets, 0);
  for (int i = 0; i < kNumGeometricSamples; ++i) {
    int64_t val = distribution->Sample() / kBucketSize;
    // Probability  of exceeding < exp(-100) * num_iterations;
    ASSERT_GE(val, 0);
    ASSERT_LE(val, kMaxValue);
    if (val < kMaxValue) {
      ++counts[val];
    }
  }
  double bucket_lambda = lambda * kBucketSize;
  auto expected_buckets =
      ExpectedCounts(-std::expm1(-bucket_lambda), std::exp(-bucket_lambda),
                     kNumGeometricSamples);

  for (size_t i = 0; i < expected_buckets.size() && i < counts.size(); i++) {
    ASSERT_NEAR(counts[i], expected_buckets[i],
                AllowedError(expected_buckets[i]))
        << i;
  }
  // For small lambdas, try to get large enough buckets to have some
  // statistically powerful results. Choose bucket sizes so that the expected
  // probability in the first bucket is 0.2 which is a reasonably large
  // fraction.
  const int kBigBucketSize = std::ceil(-std::log1p(-0.2) / bucket_lambda);
  // Otherwise the smaller buckets are themselves big enough;
  if (kBigBucketSize > 10) {
    bucket_lambda *= kBigBucketSize;
    expected_buckets =
        ExpectedCounts(-std::expm1(-bucket_lambda), std::exp(-bucket_lambda),
                       kNumGeometricSamples);
    std::vector<int> sampled(counts.size() / kBigBucketSize + 1, 0);
    for (int i = 0; i < counts.size(); ++i) {
      sampled[i / kBigBucketSize] += counts[i];
    }

    for (size_t i = 0; i < expected_buckets.size() && i < sampled.size(); i++) {
      EXPECT_NEAR(sampled[i], expected_buckets[i],
                  AllowedError(expected_buckets[i]))
          << i;
    }
  }
}

std::vector<double> GenParams() {
  return std::vector<double>({
      30,
      3,
      1e-2,
      1e-3,
      1e-4,
      1e-5,
      1e-7,
      1e-12,
      1e-15,
  });
}

std::string ParamName(const ::testing::TestParamInfo<double>& info) {
  const auto& p = info.param;
  std::string name = absl::StrCat("L_", p);
  return absl::StrReplaceAll(name, {{"-", "_"}, {".", "_"}});
}

INSTANTIATE_TEST_SUITE_P(All, GeometricDistributionTest,
                         ::testing::ValuesIn(GenParams()), ParamName);

TEST(GeometricDistribution, ImpossibleDoubles) {
  // Using std::geometric_distribution<int64_t> would fail this test, since it
  // can't generate large odd values.
  GeometricDistribution::Builder builder;
  std::unique_ptr<GeometricDistribution> distribution =
      builder.SetLambda(1e-15).Build().value();

  double count = 0;
  constexpr int kIter = 1000000;
  for (int i = 0; i < kIter; ++i) {
    int64_t val = distribution->Sample();
    ASSERT_GE(val, 0);

    if (val > (1LL << 53) && val % 2) {
      ++count;
    }
  }
  // 61 ~ (exp(-2**53 * 1e-15))/2 * 1e6.
  EXPECT_GT(count, 0);
}

}  // namespace
}  // namespace internal
}  // namespace differential_privacy
