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

#include <cstdint>
#include <memory>
#include <optional>
#include <random>

#include "gmock/gmock.h"
#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "algorithms/distributions.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/confidence-interval.pb.h"

namespace differential_privacy {
namespace test_utils {

// A full mock for the NumericalMechanism class using gmock.
class MockNoiseMechanism : public NumericalMechanism {
 public:
  MockNoiseMechanism() : NumericalMechanism(/*epsilon=*/1.0) {}

  MOCK_METHOD(double, AddDoubleNoise, (double result), (override));
  MOCK_METHOD(int64_t, AddInt64Noise, (int64_t result), (override));
  MOCK_METHOD(NumericalMechanism::NoiseConfidenceIntervalResult,
              UncheckedNoiseConfidenceInterval,
              (double confidence_level, double noised_result),
              (override, const));
  MOCK_METHOD(absl::StatusOr<ConfidenceInterval>, NoiseConfidenceInterval,
              (double confidence_level, double noised_result), (override));
  MOCK_METHOD(absl::StatusOr<ConfidenceInterval>, NoiseConfidenceInterval,
              (double confidence_level), (override));
  MOCK_METHOD(bool, NoisedValueAboveThreshold,
              (double result, double threshold), (override));
  MOCK_METHOD(double, ProbabilityOfNoisedValueAboveThreshold,
              (double result, double threshold), (override));
  MOCK_METHOD(int64_t, MemoryUsed, (), (override));
  MOCK_METHOD(double, Cdf, (double x), (override, const));
  MOCK_METHOD(double, Quantile, (double p), (override, const));
};

// A numerical mechanism that adds no noise to its input and does not perform
// snapping. Returns whatever is passed to it unmodified. Use only for testing.
// Not differentially private.
class ZeroNoiseMechanism : public LaplaceMechanism {
 public:
  class Builder : public LaplaceMechanism::Builder {
   public:
    Builder() : LaplaceMechanism::Builder() {}

    absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      return absl::StatusOr<std::unique_ptr<LaplaceMechanism>>(
          absl::make_unique<ZeroNoiseMechanism>(
              GetEpsilon().value_or(1), GetL1Sensitivity().value_or(1)));
    }

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
      return absl::make_unique<Builder>(*this);
    }
  };

  ZeroNoiseMechanism(double epsilon, double sensitivity)
      : LaplaceMechanism(epsilon, sensitivity) {}

  double AddDoubleNoise(double result) override { return result; }

  int64_t AddInt64Noise(int64_t result) override { return result; }

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) override {
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
      std::seed_seq seed({GetNumInstances()});
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
  // different seeds. This is *not* thread safe.
  static int GetNumInstances() {
    static int num_calls = 0;
    num_calls++;
    return num_calls;
  }
};

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

    absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      double sensitivity;
      if (GetL1Sensitivity().has_value()) {
        sensitivity = *GetL1Sensitivity();
      } else {
        sensitivity =
            GetL0Sensitivity().value_or(1) * GetLInfSensitivity().value_or(1);
      }
      return absl::StatusOr<std::unique_ptr<LaplaceMechanism>>(
          absl::make_unique<SeededLaplaceMechanism>(GetEpsilon().value_or(1),
                                                    sensitivity, rand_gen_));
    }

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
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
    Builder()
        : LaplaceMechanism::Builder(),
          mock_(absl::make_unique<MockLaplaceMechanism>()) {}

    // Can only be called once.
    absl::StatusOr<std::unique_ptr<NumericalMechanism>> Build() override {
      return absl::StatusOr<std::unique_ptr<LaplaceMechanism>>(
          std::unique_ptr<LaplaceMechanism>(mock_.release()));
    }

    MockLaplaceMechanism* mock() { return mock_.get(); }

    std::unique_ptr<NumericalMechanismBuilder> Clone() const override {
      return absl::make_unique<MockLaplaceMechanism::Builder>();
    }

   private:
    std::unique_ptr<MockLaplaceMechanism> mock_;
  };

  MockLaplaceMechanism() : LaplaceMechanism(1, 1) {
  }
  MockLaplaceMechanism(double epsilon, double sensitivity)
      : LaplaceMechanism(epsilon, sensitivity) {
  }
  MOCK_METHOD(double, AddDoubleNoise, (double result), (override));
  MOCK_METHOD(int64_t, AddInt64Noise, (int64_t result), (override));

  MOCK_METHOD(absl::StatusOr<ConfidenceInterval>, NoiseConfidenceInterval,
              (double confidence_level), (override));
  MOCK_METHOD(int64_t, MemoryUsed, (), (override));
};

}  // namespace test_utils
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_TESTING_H_
