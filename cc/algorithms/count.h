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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_COUNT_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_COUNT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/util.h"
#include "proto/confidence-interval.pb.h"
#include "proto/data.pb.h"
#include "proto/summary.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {

// Count the number of elements in a set, with differentially private noise.
template <typename T>
class Count : public Algorithm<T> {
 public:
  class Builder;

  void AddEntry(const T& v) override { AddMultipleEntries(v, 1); }

  absl::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level) override {
    return mechanism_->NoiseConfidenceInterval(confidence_level);
  }

  // Create and return summary containing the count.
  Summary Serialize() const override {
    // Create CountSummary.
    CountSummary count_summ;
    count_summ.set_count(count_);

    // Create Summary.
    Summary summary;
    summary.mutable_data()->PackFrom(count_summ);
    return summary;
  }

  // Add count from serialized data.
  absl::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return absl::InternalError("Cannot merge summary with no count data.");
    }

    // Add counts.
    CountSummary count_summary;
    if (!summary.data().UnpackTo(&count_summary)) {
      return absl::InternalError("Count summary unable to be unpacked.");
    }
    count_ += count_summary.count();

    return absl::OkStatus();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(Count<T>);
    if (mechanism_) {
      memory += mechanism_->MemoryUsed();
    }
    return memory;
  }

 protected:
  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    Output output;
    int64_t countWithNoise = mechanism_->AddNoise(count_);
    absl::StatusOr<ConfidenceInterval> interval =
        NoiseConfidenceInterval(noise_interval_level);

    if (interval.ok()) {
      output = MakeOutput<int64_t>(countWithNoise, interval.value());
    } else {
      output = MakeOutput<int64_t>(countWithNoise);
    }
    return output;
  }

  void ResetState() override { count_ = 0; }

  int64_t GetCount() const { return count_; }

  // The constructor and count_ are non-private for testing.
  Count(double epsilon, double delta,
        std::unique_ptr<NumericalMechanism> mechanism)
      : Algorithm<T>(epsilon, delta),
        count_(0),
        mechanism_(std::move(mechanism)) {}

 private:
  void AddMultipleEntries(const T& v, int64_t num_of_entries) {
    absl::Status status =
        ValidateIsNonNegative(num_of_entries, "Number of entries");
    if (!status.ok()) {
      return;
    }

    count_ += num_of_entries;
  }

  // Friend class for testing only
  friend class CountTestPeer;

  // Count of the number of entries added.
  int64_t count_;

  std::unique_ptr<NumericalMechanism> mechanism_;
};

template <typename T>
class Count<T>::Builder {
 public:
  Count<T>::Builder& SetEpsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  Count<T>::Builder& SetDelta(double delta) {
    delta_ = delta;
    return *this;
  }

  Count<T>::Builder& SetMaxPartitionsContributed(
      int max_partitions_contributed) {
    max_partitions_contributed_ = max_partitions_contributed;
    return *this;
  }

  Count<T>::Builder& SetMaxContributionsPerPartition(
      int max_contributions_per_partition) {
    max_contributions_per_partition_ = max_contributions_per_partition;
    return *this;
  }

  Count<T>::Builder& SetLaplaceMechanism(
      std::unique_ptr<NumericalMechanismBuilder> mechanism_builder) {
    mechanism_builder_ = std::move(mechanism_builder);
    return *this;
  }

  absl::StatusOr<std::unique_ptr<Count<T>>> Build() {
    RETURN_IF_ERROR(ValidateEpsilon(epsilon_));
    RETURN_IF_ERROR(ValidateDelta(delta_));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(max_partitions_contributed_));
    RETURN_IF_ERROR(
        ValidateMaxContributionsPerPartition(max_contributions_per_partition_));

    ASSIGN_OR_RETURN(std::unique_ptr<NumericalMechanism> count_mechanism,
                     BuildCountMechanism());
    return absl::WrapUnique(
        new Count<T>(epsilon_.value(), delta_, std::move(count_mechanism)));
  }

 private:
  std::optional<double> epsilon_;
  double delta_ = 0;
  int max_partitions_contributed_ = 1;
  int max_contributions_per_partition_ = 1;
  std::unique_ptr<NumericalMechanismBuilder> mechanism_builder_ =
      std::make_unique<LaplaceMechanism::Builder>();

  absl::StatusOr<std::unique_ptr<NumericalMechanism>> BuildCountMechanism() {
    return mechanism_builder_->Clone()
        ->SetEpsilon(epsilon_.value())
        .SetDelta(delta_)
        .SetL0Sensitivity(max_partitions_contributed_)
        .SetLInfSensitivity(max_contributions_per_partition_)
        .Build();
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_COUNT_H_
