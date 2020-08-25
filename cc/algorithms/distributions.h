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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_DISTRIBUTIONS_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_DISTRIBUTIONS_H_

#include <memory>

#include <cstdint>
#include "base/statusor.h"

namespace differential_privacy {
namespace internal {

// Allows samples to be drawn from a Gaussian distribution over a given stddev
// and mean 0 with optional per-sample scaling.
// The Gaussian noise is generated according to the binomial sampling mechanism
// described in
// https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf
// This approach is robust against unintentional privacy leaks due to artifacts
// of floating point arithmetic.
class GaussianDistribution {
 public:
  // Constructor for Gaussian with specified stddev.
  explicit GaussianDistribution(double stddev);

  virtual ~GaussianDistribution() {}

  virtual double Sample();

  // Samples the Gaussian with distribution Gauss(scale*stddev).
  virtual double Sample(double scale);

  // Returns the standard deviation of this distribution.
  double Stddev();

  // Returns the granularity that is also used when calculating Sample(). Be
  // careful when using GetGranularity() together with Sample() and make sure to
  // use the same parameter for scale in such cases.
  double GetGranularity(double scale) const;

 private:
  // Sample from geometric distribution with probability 0.5. It is much faster
  // then using GeometricDistribution which is suitable for any probability.
  double SampleGeometric();
  double SampleBinomial(double sqrt_n);

  double stddev_;
};

// Returns a sample drawn from the geometric distribution of probability
// p = 1 - e^-lambda, i.e. the number of bernoulli trial failures before the
// first success where the success probability is as defined above. lambda must
// be positive. If the result would be higher than the maximum int64_t, returns
// the maximum int64_t, which means that users should be careful around the edges
// of their distribution.
class GeometricDistribution {
 public:
  explicit GeometricDistribution(double lambda);

  virtual ~GeometricDistribution() {}

  virtual double GetUniformDouble();

  virtual int64_t Sample();

  virtual int64_t Sample(double scale);

  double Lambda();

 private:
  double lambda_;
};

// Calculates 'r' from the secure noise paper (see
// ../../common_docs/Secure_Noise_Generation.pdf)
base::StatusOr<double> CalculateGranularity(double epsilon, double sensitivity);

// DO NOT USE. Use LaplaceMechanism instead. LaplaceMechanism has an interface
// that directly accepts DP parameters, rather than requiring an error-prone
// conversion to laplace parameters.
//
// Allows sampling from a secure laplace distribution, which uses a geometric
// distribution to generate its noise in order to avoid the attack from
// Mironov's 2012 paper, "On Significance of the Least Significant Bits For
// Differential Privacy".
class LaplaceDistribution {
 public:
  explicit LaplaceDistribution(double epsilon, double sensitivity);

  virtual ~LaplaceDistribution() = default;

  virtual double GetUniformDouble();

  virtual double Sample();

  // Samples the Laplacian with distribution Lap(scale*b)
  virtual double Sample(double scale);

  virtual int64_t MemoryUsed();

  virtual bool GetBoolean();

  virtual double GetGranularity();

  // Returns the parameter defining this distribution, often labeled b.
  double GetDiversity();

  // Returns the cdf of the laplacian distribution with scale b at point x.
  static double cdf(double b, double x);

 private:
  double epsilon_;
  double sensitivity_;
  double granularity_;

 protected:
  std::unique_ptr<GeometricDistribution> geometric_distro_;
};

}  // namespace internal
}  // namespace differential_privacy
#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_DISTRIBUTIONS_H_
