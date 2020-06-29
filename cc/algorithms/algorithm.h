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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_ALGORITHM_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_ALGORITHM_H_

#include <cstddef>
#include <iterator>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "base/status.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/util.h"
#include "proto/confidence-interval.pb.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"
#include "base/canonical_errors.h"
#include "base/status.h"
#include "base/statusor.h"

namespace differential_privacy {

constexpr double kDefaultDelta = 0.0;
constexpr double kDefaultConfidenceLevel = .95;

// Abstract superclass for differentially private algorithms.
//
// Includes a notion of privacy budget in addition to epsilon to allow for
// intermediate calls that still respect the total privacy budget.
//
// e.g. a->AddEntry(1.0); a->AddEntry(2.0); if(a->PartialResult(0.1) > 0.0) ...
//   would allow an intermediate inspection using 10% of the privacy budget and
//   allow 90% to be used at some later point.
//
// Generic call to Result consumes 100% of the privacy budget by default.
template <typename T>
class Algorithm {
 public:
  //
  // Epsilon is a standard parameter of differentially private
  // algorithms. See "The Algorithmic Foundations of Differential Privacy" p17.
  explicit Algorithm(double epsilon)
      : epsilon_(epsilon), privacy_budget_(kFullPrivacyBudget) {}

  virtual ~Algorithm() = default;

  Algorithm(const Algorithm&) = delete;
  Algorithm& operator=(const Algorithm&) = delete;

  // Adds one input to the algorithm.
  virtual void AddEntry(const T& t) = 0;

  // Adds multiple inputs to the algorithm.
  template <typename Iterator>
  void AddEntries(Iterator begin, Iterator end) {
    for (auto it = begin; it != end; ++it) {
      AddEntry(*it);
    }
  }

  // Runs the algorithm on the input using the epsilon parameter
  // provided in the constructor and returns output.
  template <typename Iterator>
  base::StatusOr<Output> Result(Iterator begin, Iterator end) {
    Reset();
    AddEntries(begin, end);
    return PartialResult();
  }

  // Gets the algorithm result, consuming the remaining privacy budget.
  base::StatusOr<Output> PartialResult() {
    return PartialResult(RemainingPrivacyBudget());
  }

  // Same as above, but consumes only the `privacy_budget` amount of budget.
  // Privacy budget, defined on [0,1], represents the fraction of the total
  // budget to consume.
  base::StatusOr<Output> PartialResult(double privacy_budget) {
    return GenerateResult(ConsumePrivacyBudget(privacy_budget),
                          kDefaultConfidenceLevel);
  }

  // Same as above, but provides the confidence level of the noise confidence
  // interval, which may be included in the algorithm output.
  base::StatusOr<Output> PartialResult(double privacy_budget,
                                       double noise_interval_level) {
    return GenerateResult(ConsumePrivacyBudget(privacy_budget),
                          noise_interval_level);
  }

  double RemainingPrivacyBudget() { return privacy_budget_; }

  // Strictly reduces privacy budget, so is safe to make public.
  double ConsumePrivacyBudget(double privacy_budget_fraction) {
    DCHECK_GE(privacy_budget_fraction, 0.0)
        << "Requested budget " << privacy_budget_fraction
        << " should be positive.";
    DCHECK_LE(privacy_budget_fraction, privacy_budget_)
        << "Requested budget " << privacy_budget_fraction
        << " exceeds remaining budget of " << privacy_budget_;
    privacy_budget_fraction = Clamp(0.0, 1.0, privacy_budget_fraction);
    double budget = privacy_budget_;
    privacy_budget_ = std::max(0.0, privacy_budget_ - privacy_budget_fraction);
    return budget - privacy_budget_;
  }

  // Resets the algorithm to a state in which it has received no input. After
  // Reset is called, the algorithm should only consider input added after the
  // last Reset call when providing output.
  void Reset() {
    privacy_budget_ = kFullPrivacyBudget;
    ResetState();
  }

  // Serializes summary data of current entries into Summary proto. This allows
  // results from distributed aggregation to be recorded and later merged.
  // Returns empty summary for algorithms for which serialize is unimplemented.
  virtual Summary Serialize() = 0;

  // Merges serialized summary data into this algorithm. The summary proto must
  // represent data from the same algorithm type with identical parameters. The
  // data field must contain the algorithm summary type of the corresponding
  // algorithm used. The summary proto cannot be empty.
  virtual base::Status Merge(const Summary& summary) = 0;

  // Returns the memory currently used by the algorithm in bytes.
  virtual int64_t MemoryUsed() = 0;

  // Returns the confidence_level confidence interval of noise added within the
  // algorithm with specified privacy budget, using epsilon and other relevant,
  // algorithm-specific parameters (e.g. bounds) provided by the constructor.
  // This metric may be used to gauge the error rate introduced by the noise.
  //
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].
  //
  // By default, NoiseConfidenceInterval() returns an error. Algorithms for
  // which a confidence interval can feasibly be calculated override this and
  // output the relevant value.
  // Conservatively, we do not release the error rate for algorithms whose
  // confidence intervals rely on input size.
  virtual base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget) {
    return base::UnimplementedError(
        "NoiseConfidenceInterval() unsupported for this algorithm");
  }

  virtual double GetEpsilon() const { return epsilon_; }

 protected:
  // Returns the result of the algorithm when run on all the input that has been
  // provided via AddEntr[y|ies] since the last call to Reset.
  // Apportioning of privacy budget is handled by calls from PartialResult
  // above.
  virtual base::StatusOr<Output> GenerateResult(
      double privacy_budget, double noise_interval_level) = 0;

  // Allows child classes to reset their state as part of a global reset.
  virtual void ResetState() = 0;

 private:
  static constexpr double kFullPrivacyBudget = 1.0;

  const double epsilon_;
  double privacy_budget_;
};

template <typename T, class Algorithm, class Builder>
class AlgorithmBuilder {
 public:
  virtual ~AlgorithmBuilder() = default;

  base::StatusOr<std::unique_ptr<Algorithm>> Build() {
    // Default epsilon is used whenever epsilon is not set. This value should
    // only be used for testing convenience. For any production use case, please
    // set your own epsilon based on privacy considerations.
    if (!epsilon_.has_value()) {
      epsilon_ = DefaultEpsilon();
      LOG(WARNING) << "Default epsilon of " << epsilon_.value()
                   << " is being used. Consider setting your own epsilon based "
                      "on privacy considerations.";
    }
    return BuildAlgorithm();
  }

  Builder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *static_cast<Builder*>(this);
  }

  Builder& SetDelta(double delta) {
    delta_ = delta;
    return *static_cast<Builder*>(this);
  }

  Builder& SetMaxPartitionsContributed(int max_partitions) {
    l0_sensitivity_ = max_partitions;
    return *static_cast<Builder*>(this);
  }

  // Note for BoundedAlgorithm, this does not specify the contribution that will
  // be clamped, but the number of contributions to any partition.
  Builder& SetMaxContributionsPerPartition(int max_contribution) {
    linf_sensitivity_ = max_contribution;
    return *static_cast<Builder*>(this);
  }

  Builder& SetLaplaceMechanism(
      std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder) {
    mechanism_builder_ = std::move(mechanism_builder);
    return *static_cast<Builder*>(this);
  }

 protected:
  virtual base::StatusOr<std::unique_ptr<Algorithm>> BuildAlgorithm() = 0;

  absl::optional<double> epsilon_;
  absl::optional<double> delta_;
  absl::optional<int> l0_sensitivity_;
  absl::optional<int> linf_sensitivity_;

  // The mechanism builder is used to interject custom mechanisms for testing.
  std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder_ =
      absl::make_unique<LaplaceMechanism::Builder>();

  base::StatusOr<std::unique_ptr<NumericalMechanism>>
  UpdateAndBuildMechanism() {
    auto clone = mechanism_builder_->Clone();
    if (epsilon_.has_value()) {
      clone->SetEpsilon(epsilon_.value());
    }
    if (delta_.has_value()) {
      clone->SetDelta(delta_.value());
    }
    // If not set, we are using 1 as default value for both, L0 and Linf, as
    // fallback for existing clients.
    return clone->SetL0Sensitivity(l0_sensitivity_.value_or(1))
        .SetLInfSensitivity(linf_sensitivity_.value_or(1))
        .Build();
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_ALGORITHM_H_
