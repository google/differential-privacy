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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/util.h"
#include "proto/confidence-interval.pb.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {

constexpr double kDefaultDelta = 0.0;
constexpr double kDefaultConfidenceLevel = .95;

// Abstract superclass for differentially private algorithms.
//
// Algorithm instances are typically *not* thread safe.  Entries must be added
// from a single thread only.  In case you want to use multiple threads, you can
// use per-thread instances of the Algorithm child class, serialize them, and
// then merge them together in a single thread.
template <typename T>
class Algorithm {
 public:
  //
  // Epsilon, delta are standard parameters of differentially private
  // algorithms. See "The Algorithmic Foundations of Differential Privacy" p17.
  explicit Algorithm(double epsilon, double delta)
      : epsilon_(epsilon), delta_(delta) {
    DCHECK_NE(epsilon, std::numeric_limits<double>::infinity());
    DCHECK_GT(epsilon, 0.0);
  }
  explicit Algorithm(double epsilon) : Algorithm(epsilon, 0) {}

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

  // Runs the algorithm on (only) the input specified by the iterator,
  // using the epsilon parameter provided in the constructor. Returns the result
  // or an error, if any occurred.
  //
  // Will ignore any data that has already been accumulated. Don't mix this with
  // calls to AddEntry/Entries, PartialResult, Serialize or Merge.
  template <typename Iterator>
  absl::StatusOr<Output> Result(Iterator begin, Iterator end) {
    Reset();
    AddEntries(begin, end);
    return PartialResult();
  }

  // Get the algorithm result on the accumulated data.
  absl::StatusOr<Output> PartialResult() {
    return PartialResult(kDefaultConfidenceLevel);
  }

  // In most cases, override GenerateResult instead to maintain budget checks.
  // Override this PartialResult with caution.
  // Same as above, but allows the user to specify the confidence level that
  // may be returned as part of the Output. Not all Algorithms support
  // confidence levels, for unsupported algorithms the confidence level will not
  // be included. See NoiseConfidenceInterval for more details.
  virtual absl::StatusOr<Output> PartialResult(double noise_interval_level) {
    if (result_returned_) {
      return absl::InvalidArgumentError(
          "The algorithm can only produce results once for a given epsilon, "
          "delta budget.");
    }
    result_returned_ = true;

    return GenerateResult(noise_interval_level);
  }

  // Resets the algorithm to a state in which it has received no input. After
  // Reset is called, the algorithm should only consider input added after the
  // last Reset call when providing output.
  void Reset() {
    result_returned_ = false;
    ResetState();
  }

  // Serializes summary data of current entries into Summary proto. This allows
  // results from distributed aggregation to be recorded and later merged.
  // Returns empty summary for algorithms for which serialize is unimplemented.
  virtual Summary Serialize() const = 0;

  // Merges serialized summary data into this algorithm. The summary proto must
  // represent data from the same algorithm type with identical parameters. The
  // data field must contain the algorithm summary type of the corresponding
  // algorithm used. The summary proto cannot be empty.
  virtual absl::Status Merge(const Summary& summary) = 0;

  // Returns an estimate for the current memory consumption of the algorithm in
  // bytes. Intended to be used for distribution frameworks to prevent
  // out-of-memory errors.
  virtual int64_t MemoryUsed() = 0;

  // Returns the confidence_level confidence interval of noise added within the
  // algorithm, using epsilon and other relevant, algorithm-specific parameters
  // (e.g. bounds) provided by the constructor. This metric may be used to gauge
  // the error rate introduced by the noise.
  //
  // If the returned value is <x,y>, then the noise added has a confidence_level
  // chance of being in the domain [x,y].
  //
  // By default, NoiseConfidenceInterval() returns an error. Algorithms for
  // which a confidence interval can feasibly be calculated override this and
  // output the relevant value.
  // Conservatively, we do not release the error rate for algorithms whose
  // confidence intervals rely on input size.
  virtual absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) {
    return absl::UnimplementedError(
        "NoiseConfidenceInterval() unsupported for this algorithm");
  }

  virtual double GetEpsilon() const { return epsilon_; }

  virtual double GetDelta() const { return delta_; }

 protected:
  // Returns the result of the algorithm when run on all the input that has been
  // provided via AddEntr[y|ies] since the last call to Reset.
  // Apportioning of privacy budget is handled by calls from PartialResult
  // above.
  virtual absl::StatusOr<Output> GenerateResult(
      double noise_interval_level) = 0;

  // Allows child classes to reset their state as part of a global reset.
  virtual void ResetState() = 0;

 private:
  bool result_returned_ = false;
  const double epsilon_;
  const double delta_;
};

template <typename T, class Algorithm, class Builder>
class AlgorithmBuilder {
 public:
  virtual ~AlgorithmBuilder() = default;

  absl::StatusOr<std::unique_ptr<Algorithm>> Build() {
    RETURN_IF_ERROR(ValidateIsFiniteAndPositive(epsilon_, "Epsilon"));

    if (delta_.has_value()) {
      RETURN_IF_ERROR(
          ValidateIsInInclusiveInterval(delta_.value(), 0, 1, "Delta"));
    }  // TODO: Default delta_ to kDefaultDelta?

    if (l0_sensitivity_.has_value()) {
      RETURN_IF_ERROR(
          ValidateIsPositive(l0_sensitivity_.value(),
                             "Maximum number of partitions that can be "
                             "contributed to (i.e., L0 sensitivity)"));
    }  // TODO: Default is set in UpdateAndBuildMechanism() below.

    if (max_contributions_per_partition_.has_value()) {
      RETURN_IF_ERROR(
          ValidateIsPositive(max_contributions_per_partition_.value(),
                             "Maximum number of contributions per partition"));
    }  // TODO: Default is set in UpdateAndBuildMechanism() below.

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
  Builder& SetMaxContributionsPerPartition(int max_contributions) {
    max_contributions_per_partition_ = max_contributions;
    return *static_cast<Builder*>(this);
  }

  Builder& SetLaplaceMechanism(
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder) {
    mechanism_builder_ = std::move(mechanism_builder);
    return *static_cast<Builder*>(this);
  }

 private:
  std::optional<double> epsilon_;
  std::optional<double> delta_;
  std::optional<int> l0_sensitivity_;
  std::optional<int> max_contributions_per_partition_;

  // The mechanism builder is used to interject custom mechanisms for testing.
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_ =
      absl::make_unique<LaplaceMechanism::Builder>();

 protected:
  std::optional<double> GetEpsilon() const { return epsilon_; }
  std::optional<double> GetDelta() const { return delta_; }
  std::optional<int> GetMaxPartitionsContributed() const {
    return l0_sensitivity_;
  }
  std::optional<int> GetMaxContributionsPerPartition() const {
    return max_contributions_per_partition_;
  }

  std::unique_ptr<NumericalMechanismBuilder> GetMechanismBuilderClone() const {
    return mechanism_builder_->Clone();
  }

  virtual absl::StatusOr<std::unique_ptr<Algorithm>> BuildAlgorithm() = 0;

  absl::StatusOr<std::unique_ptr<NumericalMechanism>>
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
    // TODO: Refactor, consolidate, or remove defaults.
    return clone->SetL0Sensitivity(l0_sensitivity_.value_or(1))
        .SetLInfSensitivity(max_contributions_per_partition_.value_or(1))
        .Build();
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_ALGORITHM_H_
