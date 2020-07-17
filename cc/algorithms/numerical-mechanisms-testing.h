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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_TESTING_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_TESTING_H_

#include <random>

#include "gmock/gmock.h"
#include "absl/random/random.h"
#include "algorithms/distributions.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/confidence-interval.pb.h"
#include "base/statusor.h"

namespace differential_privacy {
namespace test_utils {

// A numerical mechanism that adds no noise to its input and does not perform
// snapping. Returns whatever is passed to it unmodified. Use only for testing.
// Not differentially private.
class ZeroNoiseMechanism : public LaplaceMechanism {
 public:
  class Builder : public LaplaceMechanism::Builder {
   public:
    Builder() : LaplaceMechanism::Builder() {}

    base::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      return base::StatusOr<std::unique_ptr<LaplaceMechanism>>(
          absl::make_unique<ZeroNoiseMechanism>(epsilon_.value_or(1),
                                               l1_sensitivity_.value_or(1)));
    }

    std::unique_ptr<LaplaceMechanism::Builder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }
  };

  ZeroNoiseMechanism(double epsilon, double sensitivity)
      : LaplaceMechanism(epsilon, sensitivity) {}

  double AddNoise(double result, double privacy_budget) override {
    return result;
  }

  base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget) override {
    ConfidenceInterval confidence;
    confidence.set_lower_bound(0);
    confidence.set_upper_bound(0);
    confidence.set_confidence_level(confidence_level);
    return confidence;
  }

  int64_t MemoryUsed() override { return sizeof(ZeroNoiseMechanism); }
};

class SeededGeometricDistribution : public internal::GeometricDistribution {
 public:
  SeededGeometricDistribution(double lambda, std::mt19937* rand_gen)
      : internal::GeometricDistribution(lambda), rand_gen_(rand_gen) {}

  double GetUniformDouble() override {
    return absl::Uniform(*rand_gen_, 0, 1.0);
  }

 private:
  std::mt19937* rand_gen_;
};

// A numerical distribution that generates consistent noise from a pre-seeded
// RNG, intended to make statistical tests completely reliable. Does not perform
// snapping. Use only for testing. Not differentially private.
class SeededLaplaceDistribution : public internal::LaplaceDistribution {
 public:
  explicit SeededLaplaceDistribution(double epsilon, double sensitivity,
                                     std::mt19937* rand_gen = nullptr)
      : internal::LaplaceDistribution(epsilon, sensitivity) {
    if (rand_gen) {
      rand_gen_ = rand_gen;
    } else {
      n_calls_++;
      std::seed_seq seed({n_calls_});
      owned_rand_gen_ = std::mt19937(seed);
      rand_gen_ = &owned_rand_gen_;
    }
    geometric_distro_ = absl::make_unique<SeededGeometricDistribution>(
        geometric_distro_->Lambda(), rand_gen_);
  }

  double GetUniformDouble() override {
    return absl::Uniform(*rand_gen_, 0, 1.0);
  }

  bool GetBoolean() override { return absl::Bernoulli(*rand_gen_, 0.5); }

 protected:
  std::mt19937* rand_gen_;
  std::mt19937 owned_rand_gen_;

 private:
  // Used to ensure that different SeededLaplaceDistribution objects have
  // different seeds.
  static int n_calls_;
};

int SeededLaplaceDistribution::n_calls_ = 0;

// A numerical mechanism using a distribution that generates consistent noise
// from a pre-seeded RNG, intended to make statistical tests completely
// reliable. Does not perform snapping. Use only for testing. Not differentially
// private.
class SeededLaplaceMechanism : public LaplaceMechanism {
 public:
  // Builder for SeededLaplaceMechanism.
  class Builder : public LaplaceMechanism::Builder {
   public:
    Builder() : LaplaceMechanism::Builder() {}

    base::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      double sensitivity;
      if (l1_sensitivity_.has_value()) {
        sensitivity = *l1_sensitivity_;
      } else {
        sensitivity =
            l0_sensitivity_.value_or(1) * linf_sensitivity_.value_or(1);
      }
      return base::StatusOr<std::unique_ptr<LaplaceMechanism>>(
          absl::make_unique<SeededLaplaceMechanism>(epsilon_.value_or(1),
                                                   sensitivity, rand_gen_));
    }

    std::unique_ptr<LaplaceMechanism::Builder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }

    SeededLaplaceMechanism::Builder& rand_gen(std::mt19937* rand_gen) {
      rand_gen_ = rand_gen;
      return *this;
    }

   private:
    std::mt19937* rand_gen_ = nullptr;
  };

  explicit SeededLaplaceMechanism(double epsilon, double sensitivity = 1.0)
      : LaplaceMechanism(epsilon, sensitivity,
                         absl::make_unique<SeededLaplaceDistribution>(
                             epsilon, sensitivity)) {}

  explicit SeededLaplaceMechanism(double epsilon, double sensitivity,
                                  std::mt19937* rand_gen)
      : LaplaceMechanism(epsilon, sensitivity,
                         absl::make_unique<SeededLaplaceDistribution>(
                             epsilon, sensitivity, rand_gen)) {}
};

// A mock Laplace mechanism using gmock. Can be set to return any value.
class MockLaplaceMechanism : public LaplaceMechanism {
 public:
  // Builder for MockLaplaceMechanism.
  class Builder : public LaplaceMechanism::Builder {
   public:
    Builder() : mock_(absl::make_unique<MockLaplaceMechanism>()) {}

    // Can only be called once.
    base::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      return base::StatusOr<std::unique_ptr<LaplaceMechanism>>(
          std::unique_ptr<LaplaceMechanism>(mock_.release()));
    }

    MockLaplaceMechanism* mock() { return mock_.get(); }

    std::unique_ptr<LaplaceMechanism::Builder> Clone() const override {
      return absl::make_unique<MockLaplaceMechanism::Builder>();
    }

   private:
    std::unique_ptr<MockLaplaceMechanism> mock_;
  };

  MockLaplaceMechanism() : LaplaceMechanism(1, 1) {}
  MockLaplaceMechanism(double epsilon, double sensitivity)
      : LaplaceMechanism(epsilon, sensitivity) {}
  MOCK_METHOD2_T(AddNoise, double(double result, double privacy_budget));
  MOCK_METHOD2_T(NoiseConfidenceInterval,
                 base::StatusOr<ConfidenceInterval>(double confidence_level,
                                                    double privacy_budget));
  MOCK_METHOD0_T(MemoryUsed, int64_t());
};

}  // namespace test_utils
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_TESTING_H_
