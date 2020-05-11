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

#include "differential_privacy/base/percentile.h"
#include "differential_privacy/base/status.h"
#include "differential_privacy/algorithms/algorithm.h"
#include "differential_privacy/algorithms/binary-search.h"
#include "differential_privacy/algorithms/bounded-algorithm.h"
#include "differential_privacy/algorithms/numerical-mechanisms.h"
#include "differential_privacy/base/status.h"

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
    BoundedBuilder::lower_ = std::numeric_limits<T>::lowest();
    BoundedBuilder::upper_ = std::numeric_limits<T>::max();
  }

 protected:
  // Check numeric parameters and construct quantiles and mechanism. Called
  // only at build.
  base::Status ConstructDependencies() {
    if (BoundedBuilder::upper_ < BoundedBuilder::lower_) {
      return base::InvalidArgumentError(
          "Upper bound cannot be less than lower bound.");
    }
    ASSIGN_OR_RETURN(mechanism_, AlgorithmBuilder::laplace_mechanism_builder_
                                     ->SetEpsilon(AlgorithmBuilder::epsilon_)
                                     .SetSensitivity(1)
                                     .Build());
    quantiles_ = absl::make_unique<base::Percentile<T>>();
    return base::OkStatus();
  }

  // Constructed when processing parameters.
  std::unique_ptr<LaplaceMechanism> mechanism_;
  std::unique_ptr<base::Percentile<T>> quantiles_;
};

template <typename T>
class Max : public BinarySearch<T> {
 public:
  class Builder : public OrderStatisticsBuilder<T, Max<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, Max<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, Max<T>, Builder>;
    using OrderBuilder = OrderStatisticsBuilder<T, Max<T>, Builder>;

   private:
    base::StatusOr<std::unique_ptr<Max<T>>> BuildAlgorithm() override {
      RETURN_IF_ERROR(OrderBuilder::ConstructDependencies());
      return absl::WrapUnique(
          new Max(AlgorithmBuilder::epsilon_, BoundedBuilder::lower_,
                  BoundedBuilder::upper_, std::move(OrderBuilder::mechanism_),
                  std::move(OrderBuilder::quantiles_)));
    }
  };

 private:
  Max(double epsilon, T lower, T upper,
      std::unique_ptr<LaplaceMechanism> mechanism,
      std::unique_ptr<base::Percentile<T>> quantiles)
      : BinarySearch<T>(epsilon, lower, upper, /*quantile=*/1,
                        std::move(mechanism), std::move(quantiles)) {}
};

template <typename T>
class Min : public BinarySearch<T> {
 public:
  class Builder : public OrderStatisticsBuilder<T, Min<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, Min<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, Min<T>, Builder>;
    using OrderBuilder = OrderStatisticsBuilder<T, Min<T>, Builder>;

   private:
    base::StatusOr<std::unique_ptr<Min<T>>> BuildAlgorithm() override {
      RETURN_IF_ERROR(OrderBuilder::ConstructDependencies());
      return absl::WrapUnique(
          new Min(AlgorithmBuilder::epsilon_, BoundedBuilder::lower_,
                  BoundedBuilder::upper_, std::move(OrderBuilder::mechanism_),
                  std::move(OrderBuilder::quantiles_)));
    }
  };

 private:
  Min(double epsilon, T lower, T upper,
      std::unique_ptr<LaplaceMechanism> mechanism,
      std::unique_ptr<base::Percentile<T>> quantiles)
      : BinarySearch<T>(epsilon, lower, upper, /*quantile=*/0,
                        std::move(mechanism), std::move(quantiles)) {}
};

template <typename T>
class Median : public BinarySearch<T> {
 public:
  class Builder : public OrderStatisticsBuilder<T, Median<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, Median<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, Median<T>, Builder>;
    using OrderBuilder = OrderStatisticsBuilder<T, Median<T>, Builder>;

   private:
    base::StatusOr<std::unique_ptr<Median<T>>> BuildAlgorithm() override {
      RETURN_IF_ERROR(OrderBuilder::ConstructDependencies());
      return absl::WrapUnique(new Median(
          AlgorithmBuilder::epsilon_, BoundedBuilder::lower_,
          BoundedBuilder::upper_, std::move(OrderBuilder::mechanism_),
          std::move(OrderBuilder::quantiles_)));
    }
  };

 private:
  Median(double epsilon, T lower, T upper,
         std::unique_ptr<LaplaceMechanism> mechanism,
         std::unique_ptr<base::Percentile<T>> quantiles)
      : BinarySearch<T>(epsilon, lower, upper, /*quantile=*/0.5,
                        std::move(mechanism), std::move(quantiles)) {}
};

template <typename T>
class Percentile : public BinarySearch<T> {
 public:
  class Builder : public OrderStatisticsBuilder<T, Percentile<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, Percentile<T>, Builder>;
    using BoundedBuilder = BoundedAlgorithmBuilder<T, Percentile<T>, Builder>;
    using OrderBuilder = OrderStatisticsBuilder<T, Percentile<T>, Builder>;

   public:
    Builder& SetPercentile(double percentile) {
      percentile_ = percentile;
      return *static_cast<Builder*>(this);
    }

   private:
    base::StatusOr<std::unique_ptr<Percentile<T>>> BuildAlgorithm() override {
      RETURN_IF_ERROR(OrderBuilder::ConstructDependencies());
      if (percentile_ < 0 || percentile_ > 1) {
        return base::InvalidArgumentError(
            "Percentile must be between 0 and 1.");
      }
      return absl::WrapUnique(new Percentile(
          percentile_, AlgorithmBuilder::epsilon_, BoundedBuilder::lower_,
          BoundedBuilder::upper_, std::move(OrderBuilder::mechanism_),
          std::move(OrderBuilder::quantiles_)));
    }

    double percentile_;
  };

  double percentile() { return percentile_; }

 private:
  Percentile(double percentile, double epsilon, T lower, T upper,
             std::unique_ptr<LaplaceMechanism> mechanism,
             std::unique_ptr<base::Percentile<T>> quantiles)
      : BinarySearch<T>(epsilon, lower, upper, percentile, std::move(mechanism),
                        std::move(quantiles)),
        percentile_(percentile) {}

  const double percentile_;
};

}  // namespace continuous
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_ORDER_STATISTICS_H_
