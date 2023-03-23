//
// Copyright 2022 Google LLC
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
#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_PARTITION_SELECTION_TESTING_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_PARTITION_SELECTION_TESTING_H_

#include "gmock/gmock.h"
#include "absl/status/statusor.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/partition-selection.h"

namespace differential_privacy {
namespace test_utils {
class MockNearTruncatedStrategy
    : public NearTruncatedGeometricPartitionSelection {
 public:
  class Builder : public NearTruncatedGeometricPartitionSelection::Builder {
   public:
    Builder()
        : NearTruncatedGeometricPartitionSelection::Builder(),
          mock_(absl::make_unique<MockNearTruncatedStrategy>()) {}

    // Can only be called once.
    absl::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      return absl::StatusOr<
          std::unique_ptr<NearTruncatedGeometricPartitionSelection>>(
          std::unique_ptr<NearTruncatedGeometricPartitionSelection>(
              mock_.release()));
    }

    MockNearTruncatedStrategy* mock() { return mock_.get(); }

   private:
    std::unique_ptr<MockNearTruncatedStrategy> mock_;
  };
  MockNearTruncatedStrategy()
      : NearTruncatedGeometricPartitionSelection(0.5, 0.02, 1, 0.02) {}

  MOCK_METHOD(bool, ShouldKeep, (double num_users), (override));
};

class MockLaplaceStrategy : public LaplacePartitionSelection {
 public:
  class Builder : public LaplacePartitionSelection::Builder {
   public:
    Builder()
        : LaplacePartitionSelection::Builder(),
          mock_(absl::make_unique<MockLaplaceStrategy>()) {}

    // Can only be called once.
    absl::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      return absl::StatusOr<std::unique_ptr<LaplacePartitionSelection>>(
          std::unique_ptr<LaplacePartitionSelection>(mock_.release()));
    }

    MockLaplaceStrategy* mock() { return mock_.get(); }

   private:
    std::unique_ptr<MockLaplaceStrategy> mock_;
  };
  MockLaplaceStrategy()
      : LaplacePartitionSelection(
            0.5, 0.02, 1, 0.02, 1,
            std::move(absl::StatusOr<std::unique_ptr<NumericalMechanism>>(
                          absl::make_unique<LaplaceMechanism>(0.5, 1))
                          .value())) {}
  MOCK_METHOD(bool, ShouldKeep, (double num_users), (override));
};

class MockGaussianStrategy : public GaussianPartitionSelection {
 public:
  class Builder : public GaussianPartitionSelection::Builder {
   public:
    Builder()
        : GaussianPartitionSelection::Builder(),
          mock_(absl::make_unique<MockGaussianStrategy>()) {}

    // Can only be called once.
    absl::StatusOr<std::unique_ptr<PartitionSelectionStrategy>> Build()
        override {
      return absl::StatusOr<std::unique_ptr<GaussianPartitionSelection>>(
          std::unique_ptr<GaussianPartitionSelection>(mock_.release()));
    }

    MockGaussianStrategy* mock() { return mock_.get(); }

   private:
    std::unique_ptr<MockGaussianStrategy> mock_;
  };
  MockGaussianStrategy()
      : GaussianPartitionSelection(
            0.5, 0.02, 1, 0.02, 1, 0.5, 1,
            std::move(absl::StatusOr<std::unique_ptr<NumericalMechanism>>(
                          absl::make_unique<GaussianMechanism>(0.5, 1, 1))
                          .value())) {}
  MOCK_METHOD(bool, ShouldKeep, (double num_users), (override));
};
}  // namespace test_utils
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_PARTITION_SELECTION_TESTING_H_
