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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_BINARY_SEARCH_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_BINARY_SEARCH_H_

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "base/percentile.h"
#include "google/protobuf/any.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "algorithms/algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "base/status_macros.h"

// Differentially private binary search.
//
// The Bayesian search implementation uses binary search-like iterations. A
// probability map is kept that partitions the search space into intervals and
// assigns a probability that the desired quantile is in that interval. At each
// iteration, we find the value such that we estimate there's a 50% chance the
// target is lower and a 50% change the target is higher. The noisy count of
// number of entries above and below the value is found. Then, based on the
// noisy counts, the probabilities that the desired quantile is below or above
// the current value is found, and used to update the probability map.
//
// Visual Example:
//   e.g., for a dataset {1, 3, 6, 15, 18, 21, 24}, and a search range [0, 32],
//   to find the median we might go through the following steps:
//
//   |   1   3    6    15   18   21  24  |      count_less    count_greater_eq
//   | ----------------------------------|
//   |   0   0    0    0 m  1    1   1   |     4 + Lap(1/e)    3 + Lap(1/e)
//   |   0   0    1 m  1    0    0   1   |     3 + Lap(1/e)    4 + Lap(1/e)
//   |   0   0    1   m1    0    1   0   |     3 + Lap(1/e)    4 + Lap(1/e)
//   |   x   x    x    x    x    x   x   |
//   |   x   x    x    x    x    x   x   |
//
//   Output:
//     Median: 14

namespace differential_privacy {

// Bayesian search creates a map entry for each iteration. Bound this to prevent
// out of memory exception.
const size_t kMaxBayesianIterations = 10000;

// Bayesian search default fraction of the privacy budget used per iteration.
const double kDefaultLocalBudgetFraction = .01;

// Bayesian search maximum fraction of the privacy budget used per iteration.
const double kMaxLocalBudgetFraction = .1;

// If the bayesian update probability is closer than this constant to 50%, then
// we should increase budget use on the next iteration to get a stronger signal
// whether the percentile is above or below the midpoint.
const double kProbabilityTooUncertain = .4;

// If the bayesian update probability is farther than this constant from 50%,
// then we should decrease budget use on the next iteration because we're
// already getting a very strong signal whether the percentile is above or
// below the midpoint.
const double kProbabilityTooCertain = .49;

// Distance from a singularity for which to use the value at the singularity.
const double kSingularityTolerance = std::pow(10, -6);

template <typename T>
class BinarySearch : public Algorithm<T> {
 public:
  void AddEntry(const T& t) override {
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    if (!std::isnan(static_cast<double>(t))) {
      quantiles_->Add(t);
    }
  }

  Summary Serialize() const override {
    BinarySearchSummary bs_summary;
    quantiles_->SerializeToProto(bs_summary.mutable_input());
    Summary summary;
    summary.mutable_data()->PackFrom(bs_summary);
    return summary;
  }

  absl::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return absl::InternalError(
          "Cannot merge summary with no binary search data.");
    }
    BinarySearchSummary bs_summary;
    if (!summary.data().UnpackTo(&bs_summary)) {
      return absl::InternalError(
          "Binary search summary unable to be unpacked.");
    }
    quantiles_->MergeFromProto(bs_summary.input());

    return absl::OkStatus();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(BinarySearch<T>);
    if (mechanism_builder_) {
      memory += sizeof(LaplaceMechanism::Builder);
    }
    if (quantiles_) {
      memory += quantiles_->Memory();
    }
    return memory;
  }

 protected:
  BinarySearch(
      double epsilon, T lower, T upper, int64_t max_partitions_contributed,
      int64_t max_contributions_per_partition, double quantile,
      std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder,
      std::unique_ptr<base::Percentile<T>> input_sketch)
      : Algorithm<T>(epsilon),
        quantile_(quantile),
        upper_(upper),
        lower_(lower),
        max_partitions_contributed_(max_partitions_contributed),
        max_contributions_per_partition_(max_contributions_per_partition),
        mechanism_builder_(std::move(mechanism_builder)),
        quantiles_(std::move(input_sketch)) {
    // TODO: Replace with Builder class & parameter validation
    DCHECK_GE(quantile, 0);
    DCHECK_LE(quantile, 1);
  }

  void ResetState() override { quantiles_->Reset(); }

  absl::StatusOr<Output> GenerateResult(double noise_interval_level) override {
    return BayesianSearch(noise_interval_level);
  }

 private:
  absl::StatusOr<Output> BayesianSearch(double noise_interval_level) {
    // If the bounds are equal, we return the only possible value with total
    // confidence.
    if (lower_ == upper_) {
      ConfidenceInterval ci;
      ci.set_lower_bound(lower_);
      ci.set_upper_bound(lower_);
      ci.set_confidence_level(noise_interval_level);
      Output output = MakeOutput<T>(lower_, ci);
      return output;
    }

    // Start the local_budget at a fraction of the total budget.
    double local_budget = kDefaultLocalBudgetFraction;
    double remaining_budget = 1.0;
    double max_local_budget = kMaxLocalBudgetFraction;
    double min_local_budget = std::nextafter(
        LaplaceMechanism::GetMinEpsilon() / Algorithm<T>::GetEpsilon(), 1.0);

    // Stores probability that the target value is the subrange. Since the map
    // is sorted, it contains a sequence of key-value pairs (k_i, v_i) for
    // i = 1, 2, ..., n, where n is the dictionary size. Then for
    // i = 1, ..., n-1, the subrange [k_i. k_(i+1)) has probability v_i of
    // containing the target value. [k_n, upper_] has probability v_n of
    // containing the target value.
    std::map<double, double> weight;
    double m = lower_ / 2.0 + upper_ / 2.0;
    weight[lower_] = .5;
    weight[m] = .5;

    // Keep doing search iterations while we have enough budget left.
    int iterations = 0;
    while (remaining_budget - local_budget > 0 &&
           iterations < kMaxBayesianIterations) {
      ++iterations;

      // Build laplace mechanism.
      std::unique_ptr<NumericalMechanism> has_to_be_laplace;
      ASSIGN_OR_RETURN(
          has_to_be_laplace,
          mechanism_builder_->Clone()
              ->SetEpsilon(Algorithm<T>::GetEpsilon() * local_budget)
              .SetL0Sensitivity(max_partitions_contributed_)
              .SetLInfSensitivity(max_contributions_per_partition_)
              .Build());

      // TODO: Remove the following dynamic_cast.
      std::unique_ptr<LaplaceMechanism> mechanism =
          absl::WrapUnique<LaplaceMechanism>(
              dynamic_cast<LaplaceMechanism*>(has_to_be_laplace.release()));

      // Find noisy counts for number of values above and below m. A single
      // input only contributes to one of the two counts.
      ASSIGN_OR_RETURN(const double percentile, Percentile(m));
      double noisy_less =
          mechanism->AddNoise(percentile * quantiles_->num_values());
      double noisy_more =
          mechanism->AddNoise((1 - percentile) * quantiles_->num_values());

      double noised_size = noisy_less + noisy_more;
      // For extreme percentiles, we want to push the result toward the range of
      // the input data.
      if (quantile_ < kSingularityTolerance) {
        noisy_less -= GetDatapoints(noised_size);
      } else if ((1 - quantile_) < kSingularityTolerance) {
        noisy_more -= GetDatapoints(noised_size);
      }

      // Calculate update multipliers.
      double update_left =
          BayesianProbabilityLeft(mechanism.get(), noisy_less, noisy_more);

      // Adjust the local budget based on certainty.
      remaining_budget -= local_budget;
      local_budget = std::clamp(UpdateLocalBudget(local_budget, update_left),
                                min_local_budget, max_local_budget);

      // Apply update multipliers.
      UpdateWeight(&weight, m, update_left);

      // Find the subrange to split the bucket and its weight in two.
      double sum_w = 0.0;
      double lower_bound = static_cast<double>(lower_);
      double w = 0;
      auto it = weight.begin();
      for (; it != weight.end(); it++) {
        sum_w += it->second;
        lower_bound = it->first;
        w = it->second;
        if (sum_w >= .5) {
          break;
        }
      }

      double upper_bound = static_cast<double>(upper_);
      if (it != weight.end() && ++it != weight.end()) {
        upper_bound = it->first;
      }

      // Split the bucket into two assuming uniform distribution of probability
      // within the bucket. The bucket starting at lower_bound will retain the
      // weight proportional to its length. The bucket starting at the new
      // split-point will get the remaining weight. Do not split the bucket if
      // m is lower_bound or upper_bound.
      m = (.5 - sum_w + w) / w * (upper_bound - lower_bound) + lower_bound;
      if (lower_bound < m && m < upper_bound) {
        weight[lower_bound] =
            w * (m - lower_bound) / (upper_bound - lower_bound);
        weight[m] = w * (upper_bound - m) / (upper_bound - lower_bound);
      }
    }

    // Round the result instead of truncation.
    if (std::is_integral<T>::value) {
      m = Clamp<double>(std::numeric_limits<T>::lowest(),
                        std::numeric_limits<T>::max(), std::round(m));
    }

    // Return 95% confidence interval of the error.
    Output output = MakeOutput<T>(
        m, ErrorConfidenceInterval(noise_interval_level, weight, m));

    return output;
  }

  // The "datapoints" is used to buffer the noisy less and noisy more
  // count for finding extreme quantiles (at 0 and 1). It approximately can be
  // thought of as you're looking for a value within datapoints away from the
  // desired quantile. If "datapoints" is too small, then we are likely to
  // obtain a very inaccurate result, because values between
  // the search space upper bound and the largest element cannot be
  // discriminated.
  double GetDatapoints(double noised_size) {
    return std::max(2.0, std::min(45.0, 14.0 * noised_size / 100.0));
  }

  // Given a noisy lower L and noisy greater count U for some value in a set,
  // and that the noise of these counts were generated by this mechanism,
  // find the probability that the percentile p element of
  // the set is to the left of the investigated value. The tolerance is the
  // distance from removable singularities to use the value at singularity.
  virtual double BayesianProbabilityLeft(const LaplaceMechanism* mechanism,
                                         double L, double U) {
    double p = quantile_;
    // In the notation we're adding noise drawn from
    // Lap(1 / b). Note that this is (confusingly) the inverse of the usual
    // terminology,
    double b = 1.0 / mechanism->GetDiversity();

    // Removable singularity at p=1/2.
    if (std::abs(p - .5) < kSingularityTolerance) {
      if (L < U) {
        return -.25 * std::exp(b * (L - U)) * (-2 + b * (L - U));
      }
      // L >= U.
      return 1 + std::exp(b * (U - L)) * (-.5 + .25 * b * (U - L));
    }

    // Singularities at p = 0 and p = 1. We use a simplified method.
    if (std::abs(p) < kSingularityTolerance) {
      if (L <= 0) {
        return std::exp(b * L) / 2;
      } else {
        return 1 - std::exp(-b * L) / 2;
      }
    }
    if (std::abs(p - 1) < kSingularityTolerance) {
      if (U <= 0) {
        return 1 - std::exp(b * U) / 2;
      } else {
        return std::exp(-b * U) / 2;
      }
    }

    if (L < p * (L + U)) {
      double num1 = std::exp(b * (L + p * U / (p - 1)));
      double num2 = std::exp(b * (L * (1 / p - 1) - U));
      double denom = 2 * (-1 + 2 * p);
      return (-1 * num1 + 2 * num1 * p - num1 * p * p + num2 * p * p) / denom;
    }
    // L >= p(L+U).
    double num1 = std::exp(-b * (L + p * U / (p - 1)));
    double num2 = std::exp(b * (L - L / p + U));
    double denom = 2 * (-1 + 2 * p);
    return (-2 + num1 * std::pow(-1 + p, 2) + 4 * p - num2 * p * p) / denom;
  }

  void UpdateWeight(std::map<double, double>* weight, double m,
                    double update_left) {
    // Apply the multipliers. For buckets below, apply left update. For buckets
    // above, apply right update. m is always the lower bound of some bucket.
    double sum_w = 0;
    for (std::map<double, double>::iterator it = weight->begin();
         it != weight->end(); it++) {
      // Get bounds and weight of the bucket.
      double lower_bound = it->first;
      double w = it->second;

      // Apply multiplier for the bucket according to position wrt m.
      double new_w = w;
      double update_right = 1 - update_left;
      if (lower_bound < m) {
        new_w *= update_left;
      } else {  // lower_bound >= m
        new_w *= update_right;
      }
      (*weight)[lower_bound] = new_w;
      sum_w += new_w;
    }

    // Normalize so weights sum to 1.
    for (auto const& bucket : *weight) {
      double lower_bound = bucket.first;
      (*weight)[lower_bound] /= sum_w;
    }
  }

  absl::StatusOr<double> Percentile(double m) {
    // If there are no inputs, getting the relative rank will return an error.
    // Arbitrarilty say the percentile is 1/2.
    if (quantiles_->num_values() == 0) {
      return .5;
    }

    std::pair<double, double> percent_pair;
    percent_pair = quantiles_->GetRelativeRank(m);

    // If T is integral, then m is a double between two integers. Take the upper
    // percentile of the nearest lesser integer.
    if (std::is_integral<T>::value) {
      return percent_pair.second;
    }

    // Otherwise, T is a float, get the percentile normally, taking the average
    // between the lower and upper percentiles for the value.
    return (percent_pair.first + percent_pair.second) / 2;
  }

  // Given the local budget probability that the desired value is on the left,
  // return the adjusted local budget for next iteration.
  double UpdateLocalBudget(double local_budget, double update_left) {
    double certainty = std::abs(update_left - .5);

    // If update_left is too close to 1/2, we need to increase the budget.
    if (certainty < kProbabilityTooUncertain) {
      return local_budget * 2;
    }
    // If update_left is too far from 1/2, we can decrease the budget.
    if (certainty > kProbabilityTooCertain) {
      return local_budget / 2;
    }
    return local_budget;
  }

  ConfidenceInterval ErrorConfidenceInterval(
      double confidence_level, const std::map<double, double>& weight,
      double result) {
    ConfidenceInterval interval;
    interval.set_confidence_level(confidence_level);
    double sum_w = 0.0;
    bool found_lower = false;
    std::map<double, double>::const_iterator it;
    for (it = weight.begin(); it != weight.end(); it++) {
      sum_w += it->second;
      if (!found_lower && sum_w >= .5 - confidence_level / 2) {
        interval.set_upper_bound(result - it->first);
        found_lower = true;
      }
      if (sum_w > (.5 + confidence_level / 2)) {
        std::map<double, double>::const_iterator it_next = std::next(it, 1);
        if (it_next == weight.end()) {
          interval.set_lower_bound(result - upper_);
        } else {
          interval.set_lower_bound(result - it_next->first);
        }
        break;
      }
    }
    return interval;
  }

  double quantile_;
  T upper_;
  T lower_;
  int64_t max_contributions_per_partition_;
  int64_t max_partitions_contributed_;

  std::unique_ptr<LaplaceMechanism::Builder> mechanism_builder_;
  std::unique_ptr<base::Percentile<T>> quantiles_;
};
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_BINARY_SEARCH_H_
