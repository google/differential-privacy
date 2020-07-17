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
  NonDpSum() : Algorithm<T>(0), result_(0) {}
  void AddEntry(const T& t) override { result_ += t; }

  base::StatusOr<Output> GenerateResult(
      double /*privacy_budget*/, double /*noise_interval_level*/) override {
    return MakeOutput<T>(result_);
  }
  void ResetState() override { result_ = 0; }

  Summary Serialize() override { return Summary(); }
  base::Status Merge(const Summary& summary) override {
    return base::OkStatus();
  }
  int64_t MemoryUsed() override { return sizeof(NonDpSum<T>); };

 private:
  T result_;
};

// Trivial non-DP count that returns the exact value determinisitically.
template <typename T>
class NonDpCount : public Algorithm<T> {
 public:
  NonDpCount() : Algorithm<T>(0), result_(0) {}
  void AddEntry(const T& t) override { ++result_; }

  base::StatusOr<Output> GenerateResult(
      double /*privacy_budget*/, double /*noise_interval_level*/) override {
    return MakeOutput<int64_t>(result_);
  }
  void ResetState() override { result_ = 0; }

  Summary Serialize() override { return Summary(); }
  base::Status Merge(const Summary& summary) override {
    return base::OkStatus();
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
class BoundedSumWithInsufficientNoise : public BoundedSum<T> {
 public:
  BoundedSumWithInsufficientNoise(
      double epsilon, T lower, T upper,
      std::unique_ptr<LaplaceMechanism::Builder> builder)
      : BoundedSum<T>(epsilon, lower, upper, 1, 1, std::move(builder), nullptr,
                      nullptr) {}
  double GetEpsilon() const override { return Algorithm<T>::GetEpsilon() / 2; }
};

// BoundedSum but it returns a error status with a fixed probability.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value ||
                                  std::is_floating_point<T>::value>::type* =
              nullptr>
class BoundedSumWithError : public BoundedSum<T> {
 public:
  BoundedSumWithError(double epsilon, T lower, T upper,
                      std::unique_ptr<LaplaceMechanism::Builder> builder)
      : BoundedSum<T>(epsilon, lower, upper, 1, 1, builder->Clone(), nullptr,
                      nullptr),
        mechanism_(absl::WrapUnique(dynamic_cast<LaplaceMechanism*>(
            builder->Build().ValueOrDie().release()))) {}

  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    if (mechanism_->GetUniformDouble() < 0.25) {
      return base::InvalidArgumentError("BoundedSumWithError returns error.");
    }
    return BoundedSum<T>::GenerateResult(privacy_budget, noise_interval_level);
  }

 private:
  std::unique_ptr<LaplaceMechanism> mechanism_;
};

// Count but returns error without dp for some results.
template <typename T>
class CountNoDpError : public Count<T> {
 public:
  explicit CountNoDpError(double epsilon)
      : Count<T>(epsilon, LaplaceMechanism::Builder()
                              .SetEpsilon(epsilon)
                              .Build()
                              .ValueOrDie()) {}

  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    if (Count<T>::count_ == 0) {
      return base::InvalidArgumentError("CountNoDpError returns error.");
    }
    return Count<T>::GenerateResult(privacy_budget, noise_interval_level);
  }
};

// Trivial class that only returns error.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value ||
                                  std::is_floating_point<T>::value>::type* =
              nullptr>
class AlwaysError : public Algorithm<T> {
 public:
  AlwaysError() : Algorithm<T>(0), result_(0) {}
  void AddEntry(const T& t) override {}

  base::StatusOr<Output> GenerateResult(
      double /*privacy_budget*/, double /*noise_interval_level*/) override {
    return base::InvalidArgumentError("AlwaysError returns error.");
  }
  void ResetState() override {}

  Summary Serialize() override { return Summary(); }
  base::Status Merge(const Summary& summary) override {
    return base::OkStatus();
  }
  int64_t MemoryUsed() override { return sizeof(AlwaysError<T>); };

 private:
  T result_;
};

TEST(StochasticTesterTest, SingleDatasetBoundedSumTest) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), true /* sorted_only */, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm =
      BoundedSum<double>::Builder()
          .SetLaplaceMechanism(
              absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(DefaultEpsilon())
          .SetLower(sequence->RangeMin())
          .SetUpper(sequence->RangeMax())
          .Build()
          .ValueOrDie();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  /*num_datasets=*/1,
                                  DefaultNumSamplesPerHistogram());
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetNonDpSumTest) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), true /* sorted_only */, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm = absl::make_unique<NonDpSum<double>>();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence),
                                  /*num_datasets=*/1,
                                  DefaultNumSamplesPerHistogram());
  EXPECT_FALSE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetCountTest) {
  std::vector<std::vector<double>> datasets({{1.0, 2.0, 3.0}});
  auto sequence = absl::make_unique<StoredSequence<double>>(datasets);
  std::unique_ptr<Count<double>> algorithm =
      Count<double>::Builder()
          .SetLaplaceMechanism(
              absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(DefaultEpsilon())
          .Build()
          .ValueOrDie();
  StochasticTester<double, int64_t> tester(
      std::move(algorithm), std::move(sequence),
      /*num_datasets=*/1, DefaultNumSamplesPerHistogram());
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetNonDpCountTest) {
  std::vector<std::vector<double>> datasets({{1.0, 2.0, 3.0}});
  auto sequence = absl::make_unique<StoredSequence<double>>(datasets);
  auto algorithm = absl::make_unique<NonDpCount<double>>();
  StochasticTester<double, int64_t> tester(
      std::move(algorithm), std::move(sequence),
      /*num_datasets=*/1, DefaultNumSamplesPerHistogram());
  EXPECT_FALSE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetCountNoBranchingTest) {
  std::vector<std::vector<double>> datasets({{1.0, 2.0, 3.0}});
  auto sequence = absl::make_unique<StoredSequence<double>>(datasets);
  std::unique_ptr<Count<double>> algorithm =
      Count<double>::Builder()
          .SetLaplaceMechanism(
              absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(DefaultEpsilon())
          .Build()
          .ValueOrDie();
  StochasticTester<double, int64_t> tester(
      std::move(algorithm), std::move(sequence),
      /*num_datasets=*/1, DefaultNumSamplesPerHistogram(),
      /*disable_search_branching=*/true);
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, SingleDatasetNonDpCountNoBranchingTest) {
  std::vector<std::vector<double>> datasets({{1.0, 2.0, 3.0}});
  auto sequence = absl::make_unique<StoredSequence<double>>(datasets);
  auto algorithm = absl::make_unique<NonDpCount<double>>();
  StochasticTester<double, int64_t> tester(
      std::move(algorithm), std::move(sequence),
      /*num_datasets=*/1, DefaultNumSamplesPerHistogram(),
      /*disable_search_branching=*/true);
  EXPECT_FALSE(tester.Run());
}

TEST(StochasticTesterTest, MultipleDatasetBoundedSumTest) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm =
      BoundedSum<double>::Builder()
          .SetLaplaceMechanism(
              absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>())
          .SetEpsilon(DefaultEpsilon())
          .SetLower(sequence->RangeMin())
          .SetUpper(sequence->RangeMax())
          .Build()
          .ValueOrDie();
  StochasticTester<double, int64_t> tester(std::move(algorithm),
                                         std::move(sequence));
  EXPECT_TRUE(tester.Run());
}

TEST(StochasticTesterTest, MultipleDatasetBoundedSumWithInsufficientNoiseTest) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm = absl::make_unique<BoundedSumWithInsufficientNoise<double>>(
      DefaultEpsilon(), sequence->RangeMin(), sequence->RangeMax(),
      absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>());
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence));
  EXPECT_FALSE(tester.Run());
}

TEST(StochasticTesterTest, ReplaceErrorWithValue) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm = absl::make_unique<BoundedSumWithError<double>>(
      DefaultEpsilon(), sequence->RangeMin(), sequence->RangeMax(),
      absl::make_unique<test_utils::SeededLaplaceMechanism::Builder>());
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence));
  EXPECT_TRUE(tester.Run());
}

// Test an algorithm that throws error deterministically.
TEST(StochasticTesterTest, ErrorStatusWithoutDP) {
  auto sequence = absl::make_unique<HaltonSequence<int64_t>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm = absl::make_unique<CountNoDpError<int64_t>>(DefaultEpsilon());
  StochasticTester<int64_t> tester(std::move(algorithm), std::move(sequence));
  EXPECT_FALSE(tester.Run());
}

// Test a class that always determines error.
TEST(StochasticTesterTest, AllError) {
  auto sequence = absl::make_unique<HaltonSequence<double>>(
      DefaultDatasetSize(), /*sorted_only=*/true, DefaultDataScale(),
      DefaultDataOffset());
  auto algorithm = absl::make_unique<AlwaysError<double>>();
  StochasticTester<double> tester(std::move(algorithm), std::move(sequence));
  EXPECT_TRUE(tester.Run());
}

}  // namespace
}  // namespace testing
}  // namespace differential_privacy
