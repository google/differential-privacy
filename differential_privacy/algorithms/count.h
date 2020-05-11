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

#include "google/protobuf/any.pb.h"
#include "differential_privacy/base/status.h"
#include "differential_privacy/algorithms/algorithm.h"
#include "differential_privacy/algorithms/numerical-mechanisms.h"
#include "differential_privacy/proto/summary.pb.h"

namespace differential_privacy {

// Count the number of elements in a set, with differentially private noise.
template <typename T>
class Count : public Algorithm<T> {
 public:
  class Builder;

  void AddEntry(const T& v) override { ++count_; }

  base::StatusOr<ConfidenceInterval> NoiseConfidenceInterval(
      double confidence_level, double privacy_budget = 1) override {
    return mechanism_->NoiseConfidenceInterval(confidence_level,
                                               privacy_budget);
  }

  // Create and return summary containing the count.
  Summary Serialize() override {
    // Create CountSummary.
    CountSummary count_summ;
    count_summ.set_count(count_);

    // Create Summary.
    Summary summary;
    summary.mutable_data()->PackFrom(count_summ);
    return summary;
  }

  // Add count from serialized data.
  base::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return base::InvalidArgumentError(
          "Cannot merge summary with no count data.");
    }

    // Add counts.
    CountSummary count_summary;
    if (!summary.data().UnpackTo(&count_summary)) {
      return base::InvalidArgumentError("Count summary unable to be unpacked.");
    }
    count_ += count_summary.count();

    return base::OkStatus();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(Count<T>);
    if (mechanism_) {
      memory += mechanism_->MemoryUsed();
    }
    return memory;
  }

 protected:
  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    Output output;
    std::size_t countWithNoise = std::max<int64_t>(
        std::round(mechanism_->AddNoise(count_, privacy_budget)), 0);
    AddToOutput<int64_t>(&output, countWithNoise);

    base::StatusOr<ConfidenceInterval> interval =
        NoiseConfidenceInterval(noise_interval_level, privacy_budget);
    if (interval.ok()) {
      *(output.mutable_error_report()->mutable_noise_confidence_interval()) =
          interval.ValueOrDie();
    }
    return output;
  }

  void ResetState() override { count_ = 0; }

  // The constructor and count_ are non-private for testing.
  Count(double epsilon, std::unique_ptr<LaplaceMechanism> mechanism)
      : Algorithm<T>(epsilon), count_(0), mechanism_(std::move(mechanism)) {}

  std::size_t count_;

 private:
  std::unique_ptr<LaplaceMechanism> mechanism_;
};

template <typename T>
class Count<T>::Builder
    : public AlgorithmBuilder<T, Count<T>, Count<T>::Builder> {
 private:
  using AlgorithmBuilder =
      differential_privacy::AlgorithmBuilder<T, Count<T>, Count<T>::Builder>;

  base::StatusOr<std::unique_ptr<Count<T>>> BuildAlgorithm() override {
    base::StatusOr<std::unique_ptr<LaplaceMechanism>> mechanism =
        AlgorithmBuilder::laplace_mechanism_builder_
            ->SetEpsilon(AlgorithmBuilder::epsilon_)
            .SetSensitivity(1)
            .Build();
    if (!mechanism.ok()) {
      return mechanism.status();
    }

    return absl::WrapUnique(new Count<T>(AlgorithmBuilder::epsilon_,
                                         std::move(mechanism.ValueOrDie())));
  }
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_COUNT_H_
