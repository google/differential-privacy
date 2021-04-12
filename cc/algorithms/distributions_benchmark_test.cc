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

#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "algorithms/distributions.h"

namespace differential_privacy {
namespace internal {
namespace {

constexpr int64_t kNumSamples = 10000000;

// For Chi Squared Test, the following two variables are coupled, if
// kChiSquaredBins changes, update using Mathematica:
// kPChiSquared = N@InverseCDF[ChiSquareDistribution[kChiSquaredBins - 1],
//                             kChiSquaredConfidence]
constexpr int64_t kChiSquaredBins = 100;
constexpr double kChiSquaredConfidence = .999;
// This value reflects the likelihood that, at greater than 99.9% confidence,
// the generating function differs from the expected Distribution. (A lower
// value corresponds to a higher likelyhood that the sample came from the target
// distribution).
constexpr double kPChiSquared = 148.23;

// Partitions the Laplace Distribution into kChiSquaredBins equally likely
// regions and performs a Pearson's chi-squared test on the results.
// While not a perfect indicator of correctness, it will detect the general
// fitness (up to sampling resolution) of the LaplaceDistribution.
double LaplaceChiSquared(const std::vector<double>& v, double scale) {
  std::vector<int64_t> histogram(kChiSquaredBins, 0.0);
  auto BinFromX = [scale](double x) {
    int64_t n = kChiSquaredBins - 1;

    // For the Laplace Distribution, the sequence
    //   Log( (2 n + 1) / (2 (n - k)) ) * scale
    // partitions the PDF into equiprobable domains. This function just inverts
    // this and determines the appropriate bin for x.
    int64_t bin_mag =
        std::floor((0.5 * std::exp(-std::abs(x / scale)) *
                    (-1.0 - n + std::exp(std::abs(x / scale)) * n)) +
                   0.5);
    if (x < 0.0)
      return 2 * bin_mag + 1;
    else
      return 2 * bin_mag;
  };

  for (double x : v) {
    int64_t bin = BinFromX(x);
    histogram[bin]++;
  }

  double expected = v.size() / static_cast<double>(kChiSquaredBins);
  double chi_squared = 0.0;
  for (int64_t i : histogram) {
    chi_squared += (i - expected) * (i - expected) / expected;
  }
  return chi_squared;
}

void BM_laplace_chi_squared(benchmark::State& state) {
  LaplaceDistribution::Builder builder;
  std::unique_ptr<LaplaceDistribution> dist =
      builder.SetEpsilon(1.0).SetSensitivity(1.0).Build().value();
  std::vector<double> samples(kNumSamples);
  for (auto _ : state) {
    std::generate(samples.begin(), samples.end(),
                  [dist = std::move(dist)]() { return dist->Sample(); });
    double chi_sq = LaplaceChiSquared(samples, 1.0);
    state.SetLabel(
        absl::StrFormat("\nLaplace chi squared: %.2f\n"
                        "Threshold at %.3f confidence: %.2f\n",
                        chi_sq, kChiSquaredConfidence, kPChiSquared));
  }
}
BENCHMARK(BM_laplace_chi_squared);

}  // namespace
}  // namespace internal
}  // namespace differential_privacy
