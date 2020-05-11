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

#include "differential_privacy/base/statusor.h"

namespace differential_privacy {
namespace internal {
// DO NOT USE. Use LaplaceMechanism instead. LaplaceMechanism has several
// improvements (snapping, conversion from DP parameters to laplace parameter)
// that are error-prone to replicate.
//
// Allows samples to be drawn from a LaplaceDistribution over a given
// parameter with optional per-sample scaling.
// https://en.wikipedia.org/wiki/Laplace_distribution
// LaplaceDistribution is thread compatible ((broken link) but not
// necessarily thread safe ((broken link)
class LaplaceDistribution {
 public:
  // Constructor for Laplace parameter b.
  explicit LaplaceDistribution(double b);
  explicit LaplaceDistribution(double epsilon, double sensitivity);

  virtual ~LaplaceDistribution() {}

  virtual double GetUniformDouble();

  virtual double Sample();

  // Samples the Laplacian with distribution Lap(scale*b)
  virtual double Sample(double scale);

  // Returns the parameter defining this distribution, often labeled b.
  double GetDiversity();

  // Returns the cdf of the laplacian distribution with scale b at point x.
  static double cdf(double b, double x);

  virtual int64_t MemoryUsed();

 private:
  double b_;
};

// Allows samples to be drawn from a Gaussian distribution over a given stddev
// and mean 0 with optional per-sample scaling.
// https://en.wikipedia.org/wiki/Normal_distribution
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

 private:
  double stddev_;
};

}  // namespace internal
}  // namespace differential_privacy
#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_DISTRIBUTIONS_H_
