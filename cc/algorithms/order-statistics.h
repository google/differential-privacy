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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_ORDER_STATISTICS_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_ORDER_STATISTICS_H_

#include "base/percentile.h"
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/binary-search.h"
#include "algorithms/bounded-algorithm.h"
#include "algorithms/numerical-mechanisms.h"

// Old classes for calculating order statistics (aka quantiles, aka
// percentiles). Deprecated, you should use Quantiles instead as it's more
// accurate.

namespace differential_privacy {
namespace continuous {

template <typename T, class Algorithm, class Builder>
class OrderStatisticsBuilder
    : public BoundedAlgorithmBuilder<T, Algorithm, Builder> {
  using AlgorithmBuilder =
      differential_privacy::AlgorithmBuilder<T, Algorithm, Builder>;
  using BoundedBuilder = BoundedAlgorithmBuilder<T, Algorithm, Builder>;

 public:
  OrderStatisticsBuilder() : BoundedBuilder() {
    // Default search bounds are numeric limits.
    BoundedBuilder::SetLower(std::numeric_limits<T>::lowest());
    BoundedBuilder::SetUpper(std::numeric_limits<T>::max());
  }

 protected:
  // Check numeric parameters and construct quantiles and mechanism. Called
  // only at build.
  absl::Status ConstructDependencies() {
    std::unique_ptr<NumericalMechanism> has_to_be_laplace;
    ASSIGN_OR_RETURN(
        has_to_be_laplace,
        AlgorithmBuilder::GetMechanismBuilderClone()
            ->SetEpsilon(AlgorithmBuilder::GetEpsilon().value())
            .SetL0Sensitivity(
                AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1))
            .SetLInfSensitivity(
                AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1))
            .Build());

    // TODO: Remove the following dynamic_cast.
    mechanism_ = absl::WrapUnique<LaplaceMechanism>(
        dynamic_cast<LaplaceMechanism*>(has_to_be_laplace.release()));

    if (mechanism_ == nullptr) {
      return absl::InvalidArgumentError(
          "Order statistics are only supported for Laplace mechanism.");
    }

    quantiles_ = absl::make_unique<base::Percentile<T>>();
    return absl::OkStatus();
  }

  // Constructed when processing parameters.
  std::unique_ptr<LaplaceMechanism> mechanism_;
  std::unique_ptr<base::Percentile<T>> quantiles_;
};

template <typename T>
class ABSL_DEPRECATED("Use Quantiles instead.") Max : public BinarySearch<T> {
 public:
  class ABSL_DEPRECATED("Use Quantiles instead.") Builder
      : public OrderStatisticsBuilder<T, Max<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, Max<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, Max<T>, Builder>;
    using OrderBuilder = OrderStatisticsBuilder<T, Max<T>, Builder>;

   private:
    absl::StatusOr<std::unique_ptr<Max<T>>> BuildBoundedAlgorithm() override {
      RETURN_IF_ERROR(OrderBuilder::ConstructDependencies());
      std::unique_ptr<LaplaceMechanism::Builder> laplace_builder =
          absl::WrapUnique<LaplaceMechanism::Builder>(
              dynamic_cast<LaplaceMechanism::Builder*>(
                  AlgorithmBuilder::GetMechanismBuilderClone().release()));
      return absl::WrapUnique(new Max(
          AlgorithmBuilder::GetEpsilon().value(),
          BoundedBuilder::GetLower().value(),
          BoundedBuilder::GetUpper().value(),
          AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1),
          AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1),
          std::move(laplace_builder), std::move(OrderBuilder::quantiles_)));
    }
  };

 private:
  Max(double epsilon, T lower, T upper, int64_t max_partitions_contributed,
      int64_t max_contributions_per_partition,
      std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
      std::unique_ptr<base::Percentile<T>> quantiles)
      : BinarySearch<T>(epsilon, lower, upper, max_partitions_contributed,
                        max_contributions_per_partition, /*quantile=*/1,
                        std::move(mechanism_builder), std::move(quantiles)) {}
};

template <typename T>
class ABSL_DEPRECATED("Use Quantiles instead.") Min : public BinarySearch<T> {
 public:
  class ABSL_DEPRECATED("Use Quantiles instead.") Builder
      : public OrderStatisticsBuilder<T, Min<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, Min<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, Min<T>, Builder>;
    using OrderBuilder = OrderStatisticsBuilder<T, Min<T>, Builder>;

   private:
    absl::StatusOr<std::unique_ptr<Min<T>>> BuildBoundedAlgorithm() override {
      RETURN_IF_ERROR(OrderBuilder::ConstructDependencies());
      std::unique_ptr<LaplaceMechanism::Builder> laplace_builder =
          absl::WrapUnique<LaplaceMechanism::Builder>(
              dynamic_cast<LaplaceMechanism::Builder*>(
                  AlgorithmBuilder::GetMechanismBuilderClone().release()));
      return absl::WrapUnique(new Min(
          AlgorithmBuilder::GetEpsilon().value(),
          BoundedBuilder::GetLower().value(),
          BoundedBuilder::GetUpper().value(),
          AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1),
          AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1),
          std::move(laplace_builder), std::move(OrderBuilder::quantiles_)));
    }
  };

 private:
  Min(double epsilon, T lower, T upper, int64_t max_partitions_contributed,
      int64_t max_contributions_per_partition,
      std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
      std::unique_ptr<base::Percentile<T>> quantiles)
      : BinarySearch<T>(epsilon, lower, upper, max_partitions_contributed,
                        max_contributions_per_partition, /*quantile=*/0,
                        std::move(mechanism_builder), std::move(quantiles)) {}
};

template <typename T>
class ABSL_DEPRECATED("Use Quantiles instead.") Median
    : public BinarySearch<T> {
 public:
  class ABSL_DEPRECATED("Use Quantiles instead.") Builder
      : public OrderStatisticsBuilder<T, Median<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, Median<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, Median<T>, Builder>;
    using OrderBuilder = OrderStatisticsBuilder<T, Median<T>, Builder>;

   private:
    absl::StatusOr<std::unique_ptr<Median<T>>> BuildBoundedAlgorithm()
        override {
      RETURN_IF_ERROR(OrderBuilder::ConstructDependencies());
      std::unique_ptr<LaplaceMechanism::Builder> laplace_builder =
          absl::WrapUnique<LaplaceMechanism::Builder>(
              dynamic_cast<LaplaceMechanism::Builder*>(
                  AlgorithmBuilder::GetMechanismBuilderClone().release()));
      return absl::WrapUnique(new Median(
          AlgorithmBuilder::GetEpsilon().value(),
          BoundedBuilder::GetLower().value(),
          BoundedBuilder::GetUpper().value(),
          AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1),
          AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1),
          std::move(laplace_builder), std::move(OrderBuilder::quantiles_)));
    }
  };

 private:
  Median(double epsilon, T lower, T upper, int64_t max_partitions_contributed,
         int64_t max_contributions_per_partition,
         std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
         std::unique_ptr<base::Percentile<T>> quantiles)
      : BinarySearch<T>(epsilon, lower, upper, max_partitions_contributed,
                        max_contributions_per_partition, /*quantile=*/0.5,
                        std::move(mechanism_builder), std::move(quantiles)) {}
};

template <typename T>
class ABSL_DEPRECATED("Use Quantiles instead.") Percentile
    : public BinarySearch<T> {
 public:
  class ABSL_DEPRECATED("Use Quantiles instead.") Builder
      : public OrderStatisticsBuilder<T, Percentile<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, Percentile<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, Percentile<T>, Builder>;
    using OrderBuilder = OrderStatisticsBuilder<T, Percentile<T>, Builder>;

   public:
    Builder() = default;
    Builder(Builder&& other) = default;
    Builder& operator=(Builder&& other) = default;
    Builder& SetPercentile(double percentile) {
      percentile_ = percentile;
      return *static_cast<Builder*>(this);
    }

   private:
    absl::StatusOr<std::unique_ptr<Percentile<T>>> BuildBoundedAlgorithm()
        override {
      RETURN_IF_ERROR(OrderBuilder::ConstructDependencies());
      RETURN_IF_ERROR(
          ValidateIsInInclusiveInterval(percentile_, 0, 1, "Percentile"));
      std::unique_ptr<LaplaceMechanism::Builder> laplace_builder =
          absl::WrapUnique<LaplaceMechanism::Builder>(
              dynamic_cast<LaplaceMechanism::Builder*>(
                  AlgorithmBuilder::GetMechanismBuilderClone().release()));
      return absl::WrapUnique(new Percentile(
          percentile_, AlgorithmBuilder::GetEpsilon().value(),
          BoundedBuilder::GetLower().value(),
          BoundedBuilder::GetUpper().value(),
          AlgorithmBuilder::GetMaxPartitionsContributed().value_or(1),
          AlgorithmBuilder::GetMaxContributionsPerPartition().value_or(1),
          std::move(laplace_builder), std::move(OrderBuilder::quantiles_)));
    }

    double percentile_;
  };

  double GetPercentile() const { return percentile_; }

 private:
  Percentile(double percentile, double epsilon, T lower, T upper,
             int64_t max_partitions_contributed,
             int64_t max_contributions_per_partition,
             std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
             std::unique_ptr<base::Percentile<T>> quantiles)
      : BinarySearch<T>(epsilon, lower, upper, max_partitions_contributed,
                        max_contributions_per_partition, percentile,
                        std::move(mechanism_builder), std::move(quantiles)),
        percentile_(percentile) {}

  const double percentile_;
};

}  // namespace continuous
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_ORDER_STATISTICS_H_
