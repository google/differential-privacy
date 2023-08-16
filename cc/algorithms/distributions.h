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

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

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
  // Builder for GaussianDistribution.
  class Builder {
   public:
    Builder& SetStddev(double stddev);

    absl::StatusOr<std::unique_ptr<GaussianDistribution>> Build();

   private:
    double stddev_;
  };

  virtual ~GaussianDistribution() {}

  virtual double Sample();

  // Samples the Gaussian with distribution Gauss(scale*stddev).
  virtual double Sample(double scale);

  // Returns the standard deviation of this distribution.
  double Stddev() const;

  // Returns the granularity that is also used when calculating Sample(). Be
  // careful when using GetGranularity() together with Sample() and make sure to
  // use the same parameter for scale in such cases.
  double GetGranularity(double scale) const;

  // Returns the cdf of the Gaussian distribution with standard deviation stddev
  // at point x.
  static double cdf(double stddev, double x);

  // Returns the quantile (inverse cdf) of the Gaussian distribution with
  // standard deviation stddev at point x.
  static double Quantile(double stddev, double x);

 private:
  // Constructor for Gaussian with specified stddev.
  explicit GaussianDistribution(double stddev);

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
// the maximum int64_t, which means that users should be careful around the
// edges of their distribution.
class GeometricDistribution {
 public:
  // Builder for GeometricDistribution.
  class Builder {
   public:
    Builder& SetLambda(double lambda);

    absl::StatusOr<std::unique_ptr<GeometricDistribution>> Build();

   private:
    double lambda_;
  };

  virtual ~GeometricDistribution() {}

  virtual double GetUniformDouble();

  virtual int64_t Sample();

  virtual int64_t Sample(double scale);

  double Lambda();

 protected:
  explicit GeometricDistribution(double lambda);

 private:
  double lambda_;
};

// DO NOT USE. Use LaplaceMechanism instead. LaplaceMechanism has an interface
// that directly accepts DP parameters, rather than requiring an error-prone
// conversion to Laplace parameters.
//
// Allows sampling from a secure Laplace distribution, which uses a geometric
// distribution to generate its noise in order to avoid the attack from
// Mironov's 2012 paper, "On Significance of the Least Significant Bits For
// Differential Privacy":
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.366.5957&rep=rep1&type=pdf
class LaplaceDistribution {
 public:
  // Builder for LaplaceDistribution.
  class Builder {
   public:
    Builder& SetEpsilon(double epsilon);

    Builder& SetSensitivity(double sensitivity);

    absl::StatusOr<std::unique_ptr<LaplaceDistribution>> Build();

   private:
    double epsilon_;
    double sensitivity_;
  };

  virtual ~LaplaceDistribution() = default;

  virtual double GetUniformDouble();

  virtual double Sample();

  virtual int64_t MemoryUsed();

  virtual bool GetBoolean();

  virtual double GetGranularity();

  virtual double GetVariance() const;

  // Returns the parameter defining this distribution, often labeled b.
  double GetDiversity() const;

  // Returns the smallest possible valid epsilon.
  static double GetMinEpsilon();

  // Returns the cdf of the Laplace distribution with scale b at point x.
  static double cdf(double b, double x);

  // Returns the quantile (inverse cdf) of the Laplace distribution with
  // scale b at the point p.
  static double Quantile(double b, double p);

  // Calculates 'r' from the secure noise paper (see
  // ../../common_docs/Secure_Noise_Generation.pdf)
  static absl::StatusOr<double> CalculateGranularity(double epsilon,
                                                     double sensitivity);

 protected:
  explicit LaplaceDistribution(
      double epsilon, double sensitivity, double granularity,
      std::unique_ptr<GeometricDistribution> geometric_distro);

  // Constructor that might fail during initialization of granularity or the
  // GeometricDistribution.
  explicit LaplaceDistribution(double epsilon, double sensitivity);

 private:
  static absl::Status ValidateEpsilon(double epsilon);

  double epsilon_;
  double sensitivity_;
  double granularity_;

  // Inclusive lower bound for epsilon when calculating granularity.
  static constexpr double kMinEpsilon = 1.0 / (int64_t{1} << 50);

 protected:
  std::unique_ptr<GeometricDistribution> geometric_distro_;
};

}  // namespace internal
}  // namespace differential_privacy
#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_DISTRIBUTIONS_H_
