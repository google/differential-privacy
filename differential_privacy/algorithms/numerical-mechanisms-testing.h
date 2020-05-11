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
#include "differential_privacy/algorithms/confidence-interval.pb.h"
#include "differential_privacy/algorithms/distributions.h"
#include "differential_privacy/algorithms/numerical-mechanisms.h"
#include "differential_privacy/base/statusor.h"

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

    base::StatusOr<std::unique_ptr<LaplaceMechanism>> Build() override {
      return base::StatusOr<std::unique_ptr<LaplaceMechanism>>(
          absl::make_unique<ZeroNoiseMechanism>(epsilon_.value_or(1),
                                               l1_sensitivity_.value_or(1)));
    }

    std::unique_ptr<LaplaceMechanism::Builder> Clone() const override {
      Builder clone;
      if (epsilon_.has_value()) {
        clone.SetEpsilon(epsilon_.value());
      }
      if (l1_sensitivity_.has_value()) {
        clone.SetL1Sensitivity(l1_sensitivity_.value());
      }
      return absl::make_unique<Builder>(clone);
    }
  };

  ZeroNoiseMechanism(double epsilon, double sensitivity)
      : LaplaceMechanism(epsilon, sensitivity) {}

  double AddNoise(double result, double privacy_budget) override {
    return result;
  }

  ConfidenceInterval NoiseConfidenceInterval(double confidence_level,
                                             double privacy_budget) override {
    ConfidenceInterval confidence;
    confidence.set_lower_bound(0);
    confidence.set_upper_bound(0);
    confidence.set_confidence_level(confidence_level);
    return confidence;
  }

  int64_t MemoryUsed() override { return sizeof(ZeroNoiseMechanism); }
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
      std::seed_seq seed({1, 2, 3});
      owned_rand_gen_ = std::mt19937(seed);
      rand_gen_ = &owned_rand_gen_;
    }
  }

  double GetUniformDouble() override {
    return absl::Uniform(*rand_gen_, 0, 1.0);
  }

 protected:
  std::mt19937* rand_gen_;
  std::mt19937 owned_rand_gen_;
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

    base::StatusOr<std::unique_ptr<LaplaceMechanism>> Build() override {
      return base::StatusOr<std::unique_ptr<LaplaceMechanism>>(
          absl::make_unique<SeededLaplaceMechanism>(
              epsilon_.value_or(1), l1_sensitivity_.value_or(1), rand_gen_));
    }

    std::unique_ptr<LaplaceMechanism::Builder> Clone() const override {
      SeededLaplaceMechanism::Builder clone;
      clone.rand_gen(rand_gen_);
      if (epsilon_.has_value()) {
        clone.SetEpsilon(epsilon_.value());
      }
      if (l1_sensitivity_.has_value()) {
        clone.SetL1Sensitivity(l1_sensitivity_.value());
      }
      return absl::make_unique<Builder>(clone);
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
    base::StatusOr<std::unique_ptr<LaplaceMechanism>> Build() override {
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
                 ConfidenceInterval(double confidence_level,
                                    double privacy_budget));
  MOCK_METHOD0_T(MemoryUsed, int64_t());
};

}  // namespace test_utils
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_NUMERICAL_MECHANISMS_TESTING_H_
