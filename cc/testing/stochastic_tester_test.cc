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

#include "testing/stochastic_tester.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "absl/status/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/count.h"
#include "algorithms/numerical-mechanisms-testing.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "testing/sequence.h"

namespace differential_privacy {
namespace testing {
namespace {

// Trivial non-DP sum that returns the exact value determinisitically.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value ||
                                  std::is_floating_point<T>::value>::type* =
              nullptr>
class NonDpSum : public Algorithm<T> {
 public:
  NonDpSum() : Algorithm<T>(1), result_(0) {}
  void AddEntry(const T& t) override { result_ += t; }

  absl::StatusOr<Output> GenerateResult(
      double /*noise_interval_level*/) override {
    return MakeOutput<T>(result_);
  }
  void ResetState() override { result_ = 0; }

  Summary Serialize() const override { return Summary(); }
  absl::Status Merge(const Summary& summary) override {
    return absl::OkStatus();
  }
  int64_t MemoryUsed() override { return sizeof(NonDpSum<T>); };

 private:
  T result_;
};

// Trivial non-DP count that returns the exact value determinisitically.
template <typename T>
class NonDpCount : public Algorithm<T> {
 public:
  NonDpCount() : Algorithm<T>(1), result_(0) {}
  void AddEntry(const T& t) override { ++result_; }

  absl::StatusOr<Output> GenerateResult(
      double /*noise_interval_level*/) override {
    return MakeOutput<int64_t>(result_);
  }
  void ResetState() override { result_ = 0; }

  Summary Serialize() const override { return Summary(); }
  absl::Status Merge(const Summary& summary) override {
    return absl::OkStatus();
  }
  int64_t MemoryUsed() override { return sizeof(NonDpCount<T>); };

 private:
  int64_t result_;
};

// A version of BoundedSum where Epsilon() is overridden to report half of the
// actual epsilon value, so we have an algorithm that claims stronger privacy
// guarantees than it actually provides.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value ||
                                  std::is_floating_point<T>::value>::type* =
              nullptr>
class BoundedSumWithInsufficientNoise : public BoundedSumWithFixedBounds<T> {
 public:
  BoundedSumWithInsufficientNoise(
      const double epsilon, const double delta, const T lower, const T upper,
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder)
      : BoundedSumWithFixedBounds<T>(
            epsilon, 0, lower, upper,
            BoundedSum<T>::BuildMechanism(std::move(mechanism_builder), epsilon,
                                          delta, 1, 1, lower, upper)
                .value()) {}
  double GetEpsilon() const override { return Algorithm<T>::GetEpsilon() / 2; }
};

// BoundedSum but it returns a error status with a fixed probability.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value ||
                                  std::is_floating_point<T>::value>::type* =
              nullptr>
class BoundedSumWithError : public BoundedSumWithFixedBounds<T> {
 public:
  BoundedSumWithError(
      const double epsilon, const double delta, const T lower, const T upper,
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder)
      : BoundedSumWithFixedBounds<T>(
            epsilon, delta, lower, upper,
            BoundedSum<T>::BuildMechanism(mechanism_builder->Clone(), epsilon,
                                          delta, 1, 1, lower, upper)
                .value()) {}

  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    if (UniformDouble() < 0.25) {
      return absl::InvalidArgumentError("BoundedSumWithError returns error.");
    }
    return BoundedSumWithFixedBounds<T>::GenerateResult(noise_interval_level);
  }
};

// Count but returns error without dp for some results.
template <typename T>
class CountNoDpError : public Count<T> {
 public:
  explicit CountNoDpError(double epsilon,
                          std::unique_ptr<NumericalMechanism> laplace_mechanism)
      : Count<T>(epsilon, 0, std::move(laplace_mechanism)) {}

  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    if (Count<T>::GetCount() == 0) {
      return absl::InvalidArgumentError("CountNoDpError returns error.");
    }
    return Count<T>::GenerateResult(noise_interval_level);
  }
};

// Trivial class that only returns error.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value ||
                                  std::is_floating_point<T>::value>::type* =
              nullptr>
class AlwaysError : public Algorithm<T> {
 public:
  AlwaysError() : Algorithm<T>(1), result_(0) {}
  void AddEntry(const T& t) override {}

  absl::StatusOr<Output> GenerateResult(
      double /*noise_interval_level*/) override {
    return absl::InvalidArgumentError("AlwaysError returns error.");
  }
  void ResetState() override {}

  Summary Serialize() const override { return Summary(); }
  absl::Status Merge(const Summary& summary) override {
    return absl::OkStatus();
  }
  int64_t MemoryUsed() override { return sizeof(AlwaysError<T>); };

 private:
  T result_;
};

TEST(StochasticTesterTest, SingleDatasetBoundedSumTest) {
  auto sequence = std::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), true /* sorted_only */, DefaultDataScale(),
      DefaultDataOffset());
  absl::StatusOr<std::unique_ptr<BoundedSum<double>>> algorithm =
      BoundedSum<double>::Builder()
          .SetLaplaceMechanism(
              std::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(std::log(3))
          .SetLower(sequence->RangeMin())
          .SetUpper(sequence->RangeMax())
          .Build();
  ASSERT_TRUE(algorithm.ok());
  StochasticTester<double> tester(std::move(*algorithm), std::move(sequence),
                                  /*num_datasets=*/1,
                                  DefaultNumSamplesPerHistogram());
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetNonDpSumTest) {
  auto sequence = std::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), true /* sorted_only */, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm = std::make_unique<NonDpSum<double>>();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  /*num_datasets=*/1,
                                  DefaultNumSamplesPerHistogram());
  EXPECT_FALSE(tester.Run());
}

TEST(StochasticTesterTest, EmptySamplesCountTest) {
  std::vector<std::vector<double>> datasets({{1.0}});
  auto sequence = std::make_unique<StoredSequence<double>>(datasets);
  absl::StatusOr<std::unique_ptr<Count<double>>> algorithm =
      Count<double>::Builder()
          .SetLaplaceMechanism(
              std::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(std::log(3))
          .Build();
  ASSERT_TRUE(algorithm.ok());
  StochasticTester<double, int64_t> tester(
      std::move(*algorithm), std::move(sequence),
      /*num_datasets=*/1, /*num_samples_per_histogram=*/0);
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetCountTest) {
  std::vector<std::vector<double>> datasets({{1.0, 2.0, 3.0}});
  auto sequence = std::make_unique<StoredSequence<double>>(datasets);
  absl::StatusOr<std::unique_ptr<Count<double>>> algorithm =
      Count<double>::Builder()
          .SetLaplaceMechanism(
              std::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(std::log(3))
          .Build();
  ASSERT_TRUE(algorithm.ok());
  StochasticTester<double, int64_t> tester(
      std::move(*algorithm), std::move(sequence),
      /*num_datasets=*/1, DefaultNumSamplesPerHistogram());
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetNonDpCountTest) {
  std::vector<std::vector<double>> datasets({{1.0, 2.0, 3.0}});
  auto sequence = std::make_unique<StoredSequence<double>>(datasets);
  auto algorithm = std::make_unique<NonDpCount<double>>();
  StochasticTester<double, int64_t> tester(
      std::move(algorithm), std::move(sequence),
      /*num_datasets=*/1, DefaultNumSamplesPerHistogram());
  EXPECT_FALSE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetCountNoBranchingTest) {
  std::vector<std::vector<double>> datasets({{1.0, 2.0, 3.0}});
  auto sequence = std::make_unique<StoredSequence<double>>(datasets);
  absl::StatusOr<std::unique_ptr<Count<double>>> algorithm =
      Count<double>::Builder()
          .SetLaplaceMechanism(
              std::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(std::log(3))
          .Build();
  ASSERT_TRUE(algorithm.ok());
  StochasticTester<double, int64_t> tester(
      std::move(*algorithm), std::move(sequence),
      /*num_datasets=*/1, DefaultNumSamplesPerHistogram(),
      /*disable_search_branching=*/true);
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetNonDpCountNoBranchingTest) {
  std::vector<std::vector<double>> datasets({{1.0, 2.0, 3.0}});
  auto sequence = std::make_unique<StoredSequence<double>>(datasets);
  auto algorithm = std::make_unique<NonDpCount<double>>();
  StochasticTester<double, int64_t> tester(
      std::move(algorithm), std::move(sequence),
      /*num_datasets=*/1, DefaultNumSamplesPerHistogram(),
      /*disable_search_branching=*/true);
  EXPECT_FALSE(tester.Run());
}

TEST(StochasticTesterTest, MultipleDatasetBoundedSumTest) {
  auto sequence = std::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  absl::StatusOr<std::unique_ptr<BoundedSum<double>>> algorithm =
      BoundedSum<double>::Builder()
          .SetLaplaceMechanism(
              std::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(std::log(3))
          .SetLower(sequence->RangeMin())
          .SetUpper(sequence->RangeMax())
          .Build();
  ASSERT_TRUE(algorithm.ok());
  StochasticTester<double, int64_t> tester(std::move(*algorithm),
                                           std::move(sequence));
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, MultipleDatasetBoundedSumWithInsufficientNoiseTest) {
  auto sequence = std::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto mechanism_builder =
      std::make_unique<test_utils::SeededLaplaceMechanism::Builder>();
  auto algorithm = std::make_unique<BoundedSumWithInsufficientNoise<double>>(
      std::log(3), 0, sequence->RangeMin(), sequence->RangeMax(),
      std::move(mechanism_builder));
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence));
  EXPECT_FALSE(tester.Run());
}

TEST(StochasticTesterTest, ReplaceErrorWithValue) {
  auto sequence = std::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto mechanism_builder =
      std::make_unique<test_utils::SeededLaplaceMechanism::Builder>();
  auto algorithm = std::make_unique<BoundedSumWithError<double>>(
      std::log(3), 0, sequence->RangeMin(), sequence->RangeMax(),
      std::move(mechanism_builder));
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence));
  EXPECT_TRUE(tester.Run());
}

// Test an algorithm that throws error deterministically.
TEST(StochasticTesterTest, ErrorStatusWithoutDP) {
  const double epsilon = std::log(3);
  auto sequence = std::make_unique<HaltonSequence<int64_t>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto mechanism = LaplaceMechanism::Builder()
                       .SetL1Sensitivity(1)
                       .SetEpsilon(epsilon)
                       .Build();
  ASSERT_TRUE(mechanism.ok());
  auto algorithm =
      std::make_unique<CountNoDpError<int64_t>>(epsilon, std::move(*mechanism));
  StochasticTester<int64_t> tester(std::move(algorithm), std::move(sequence));
  EXPECT_FALSE(tester.Run());
}

// Test a class that always determines error.
TEST(StochasticTesterTest, AllError) {
  auto sequence = std::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm = std::make_unique<AlwaysError<double>>();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence));
  EXPECT_TRUE(tester.Run());
}

}  // namespace
}  // namespace testing
}  // namespace differential_privacy
