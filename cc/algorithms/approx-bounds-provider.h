//
// Copyright 2025 Google LLC
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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_APPROX_BOUNDS_PROVIDER_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_APPROX_BOUNDS_PROVIDER_H_

#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "algorithms/bounds-provider.h"
#include "algorithms/internal/clamped-calculation-without-bounds.h"
#include "algorithms/numerical-mechanisms.h"
#include "algorithms/util.h"
#include "proto/util.h"
#include "proto/data.pb.h"
#include "base/status_macros.h"

namespace differential_privacy {

// This URL is added as payload to the error returned from GenerateResult in
// case there was not enough data to provide approximate bounds.
inline constexpr absl::string_view kApproxBoundsNotEnoughDataUrl =
    "type.googleapis.com/differential_privacy.ApproxBoundsNotEnoughData";

// The minimum allowed success probability when relaxing the success
// probability. If approx bounds fails to find bounds, it will reduce the
// success probability and, if the reduced success probability is still
// greater than kMinSuccessProbability, attempt to find bounds with the
// reduced success probability. Having a minimum success probability ensures
// we fail rather than returning bounds that are just due to noised empty
// bins.
inline constexpr double kMinSuccessProbability = 1 - 1e-6;

// Find the approximate bounds of a set of numbers using logarithmic histogram
// bins. Like other algorithms, ApproxBounds assumes that it only gets one input
// per user.
//
// The algorithm takes as parameters num_bins, scale, and base to construct
// a logarithmic histogram with num_bins number of bins. Scale and base
// determine bin boundaries. Two histograms are created: one for positives and
// another for negatives.
//
// Without loss of generality, bin i contains the number of inputs whose most
// significant bit represents a number that lies in the range
// (scale * base^(i-1), scale * base^i]. There are two exceptions:
//   - Positive bin 0 has boundaries [0, scale * base^0]. Negative bin 0 does
//     not contain 0.
//   - When the highest included positive number in the histogram is the max
//     numeric limit for the data type, the lowest negative bin, instead of
//     containing [-1 * max_numeric_limit, x), will contain
//     [min_numeric_limit, x). This is because the min_numeric_limit is
//     sometimes one greater in magnitude than the max_numeric_limit.
//
// To generate the output, first noise is added to each bin count. Then, the
// success_probability is used to determine a threshold count. The success
// probability is the probability that, given our dataset is empty, all bins
// have noised counts that are less than the threshold count. Therefore,
// increasing success_probability will increase the threshold and the
// probability that bounds are too tight.
//
// For the approx upper bound, we choose the rightmost bin that succeeds the
// threshold count and return its upper boundary. Similarly, for the approx
// lower bound we choose the leftmost bin that succeeds the threshold count and
// return its lower boundary. If the success_probability is too high it is
// possible that no bin is greater than the threshold. In this case, we reduce
// the success probability (and thereby the threshold) and see whether any bins
// exceed the new threshold. We repeat this until a bin exceeds the threshold or
// the success probability becomes small enough that the bounds we would find
// are likely to be due to noise. If we still have not found bounds, we return
// an error status in the output.
//
// For example, if
//   scale = 2, base = 1, num_bins = 4, inputs = {0, 0, 0, 0, 1, 3, 7, 8, 8, 8}
// We have histogram bins and counts
//   [0, 1]: 5
//   (1, 2]: 0
//   (2, 4]: 1
//   (4, 8]: 4
// Then if success_probability=.9 and epsilon=1 we will obtain approximately
// threshold=3.5. Since the count of bin (4, 8] > threshold, we return an
// approx max of 2^3 = 8. Since the count of bin [0,1] > threshold, we return an
// approx min of 0.
template <typename T>
class ApproxBoundsProvider final : public BoundsProvider<T> {
  static_assert(std::is_arithmetic<T>::value,
                "ApproxBoundsProvider can only be used for arithmetic types");

 public:
  struct Options {
    double epsilon;
    int max_partitions_contributed;
    int max_contributions_per_partition;
    double scale = DefaultScaleForT();
    double base = 2.0;
    int64_t num_bins = DefaultNumBins(scale, base);
    double success_probability = 1 - std::pow(10, -9);
    std::unique_ptr<NumericalMechanismBuilder> mechanism_builder =
        std::make_unique<LaplaceMechanism::Builder>();
    std::optional<double> delta;
  };

  static absl::StatusOr<std::unique_ptr<ApproxBoundsProvider<T>>> Create(
      const Options& options) {
    RETURN_IF_ERROR(ValidateEpsilon(options.epsilon));
    RETURN_IF_ERROR(
        ValidateMaxPartitionsContributed(options.max_partitions_contributed));
    RETURN_IF_ERROR(ValidateMaxContributionsPerPartition(
        options.max_contributions_per_partition));

    // Check the validity of the histogram parameters. num_bin and
    // success_probability restrictions prevent undefined threshold
    // calculation.
    RETURN_IF_ERROR(ValidateIsPositive(options.num_bins, "Number of bins"));
    RETURN_IF_ERROR(ValidateIsFiniteAndPositive(options.scale, "Scale"));
    RETURN_IF_ERROR(ValidateIsFinite(options.base, "Base"));
    RETURN_IF_ERROR(ValidateIsGreaterThan(options.base, 1, "Base"));

    // Cache the bin boundary magnitudes for performance. Note that casting
    // numeric limits lead to inconsistencies.
    auto get_boundary = [boundary = options.scale,
                         base = options.base]() mutable {
      if (boundary >= std::numeric_limits<T>::max() / base) {
        return std::numeric_limits<T>::max();
      }
      double this_boundary = boundary;
      boundary *= base;
      return static_cast<T>(this_boundary);
    };
    std::vector<T> bin_boundaries(options.num_bins);
    std::generate(bin_boundaries.begin(), bin_boundaries.end(), get_boundary);

    ASSIGN_OR_RETURN(
        std::unique_ptr<NumericalMechanism> mechanism,
        options.mechanism_builder->SetEpsilon(options.epsilon)
            .SetL0Sensitivity(options.max_partitions_contributed)
            .SetLInfSensitivity(options.max_contributions_per_partition)
            .Build());

    RETURN_IF_ERROR(ValidateIsInExclusiveInterval(options.success_probability,
                                                  0, 1, "Success probability"));

    // Using `new` to access non-public constructor.
    return absl::WrapUnique(new ApproxBoundsProvider<T>(
        options.epsilon, options.num_bins, options.scale, options.base,
        std::move(bin_boundaries), options.success_probability,
        std::move(mechanism)));
  }

  virtual ~ApproxBoundsProvider() = default;

  absl::StatusOr<BoundsResult<T>> FinalizeAndCalculateBounds() override {
    // Populate noisy versions of the histogram bins.
    noisy_pos_bins_ = AddNoise(pos_bins_);
    noisy_neg_bins_ = AddNoise(neg_bins_);

    std::optional<BoundsResult<T>> bounds_result;
    double success_probability = success_probability_;
    int bounding_attempts = 0;
    int max_bounding_attempts = 30;
    do {
      double threshold = mechanism_->Quantile(
          std::pow(success_probability, 1.0 / (2 * pos_bins_.size())));
      bounds_result = FindBounds(threshold);
      double failure_probability = 1 - success_probability;
      success_probability = 1 - 10 * failure_probability;
      bounding_attempts++;
    } while (!bounds_result.has_value() &&
             success_probability > kMinSuccessProbability &&
             bounding_attempts < max_bounding_attempts);

    // Record error status if approx min or max was not found.
    if (!bounds_result.has_value()) {
      absl::Status result = absl::FailedPreconditionError(
          "Bin count threshold was too large to find approximate "
          "bounds. Either run over a larger dataset or decrease "
          "success_probability and try again.");
      result.SetPayload(kApproxBoundsNotEnoughDataUrl, absl::Cord());
      return result;
    }

    return bounds_result.value();
  }

  // Get additional private information in the form of a BoundingReport. Will
  // populate any fields possible and leave the rest blank.
  BoundingReport GetBoundingReport(
      const BoundsResult<T>& bounds_result) override {
    BoundingReport report;
    SetValue<T>(report.mutable_lower_bound(), bounds_result.lower_bound);
    SetValue<T>(report.mutable_upper_bound(), bounds_result.upper_bound);
    absl::StatusOr<double> count = NumInputs();
    absl::StatusOr<double> count_outside =
        NumInputsOutside(bounds_result.lower_bound, bounds_result.upper_bound);
    if (count.ok()) {
      report.set_num_inputs(count.value());
    }
    if (count_outside.ok()) {
      report.set_num_outside(count_outside.value());
    }
    return report;
  }

  void Reset() override {
    std::fill(pos_bins_.begin(), pos_bins_.end(), 0);
    std::fill(neg_bins_.begin(), neg_bins_.end(), 0);
  }

  int64_t MemoryUsed() const override {
    int64_t memory = sizeof(ApproxBoundsProvider<T>) +
                     sizeof(int64_t) * neg_bins_.capacity() +
                     sizeof(int64_t) * pos_bins_.capacity() +
                     sizeof(T) * noisy_neg_bins_.capacity() +
                     sizeof(T) * noisy_pos_bins_.capacity();
    if (mechanism_) {
      memory += mechanism_->MemoryUsed();
    }
    return memory;
  }

  // Return the number of positive bins. This function and the following
  // functions are exposed for calling classes of ApproxBounds to store partial
  // values corresponding to bins. Virtual for testing.
  int64_t NumPositiveBins() { return pos_bins_.size(); }

  double GetEpsilon() const override { return epsilon_; }

  double GetDelta() const override { return delta_; }

  // Serialize the positive and negative bin counts.
  BoundsSummary Serialize() const override {
    BoundsSummary bounds_summary;
    *bounds_summary.mutable_approx_bounds_summary()->mutable_pos_bin_count() = {
        pos_bins_.begin(), pos_bins_.end()};
    *bounds_summary.mutable_approx_bounds_summary()->mutable_neg_bin_count() = {
        neg_bins_.begin(), neg_bins_.end()};
    return bounds_summary;
  }

  // Retrieve positive and negative bin counts from summary and add them.
  absl::Status Merge(const BoundsSummary& bounds_summary) override {
    if (!bounds_summary.has_approx_bounds_summary()) {
      return absl::InternalError(
          "Cannot merge summary with no histogram data.");
    }

    if (pos_bins_.size() !=
            bounds_summary.approx_bounds_summary().pos_bin_count_size() ||
        neg_bins_.size() !=
            bounds_summary.approx_bounds_summary().neg_bin_count_size()) {
      return absl::InternalError(
          "Merged approximate max summary must have the same number of "
          "bin counts as this histogram.");
    }

    // Add bin count from summary to each bin.
    for (int i = 0; i < pos_bins_.size(); ++i) {
      pos_bins_[i] += bounds_summary.approx_bounds_summary().pos_bin_count(i);
      neg_bins_[i] += bounds_summary.approx_bounds_summary().neg_bin_count(i);
    }
    return absl::OkStatus();
  }

  std::unique_ptr<internal::ClampedCalculationWithoutBounds<T>>
  CreateClampedCalculationWithoutBounds() const {
    typename internal::ClampedCalculationWithoutBounds<T>::Options options;
    options.num_bins = pos_bins_.size();
    options.scale = scale_;
    options.base = base_;
    return internal::ClampedCalculationWithoutBounds<T>::Create(options);
  }

  // Adds input num_of_entries times to the bins.
  void AddEntry(const T& input) {
    int64_t num_of_entries = 1;
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    absl::Status status =
        ValidateIsPositive(num_of_entries, "Number of entries");
    if (std::isnan(static_cast<double>(input)) || !status.ok()) {
      return;
    }
    // Place into correct bin according to most significant bit and sign. Note
    // that MostSignificantBit returns 0 for 0.
    int index = MostSignificantBit(input);
    if (input >= 0) {
      pos_bins_[index] += num_of_entries;
    } else {  // value < 0
      neg_bins_[index] += num_of_entries;
    }
  }

  // Splits the value into the elements of the partials vector, each of which
  // corresponds to a bin in ApproxBounds. make_partial is a function that given
  // two numbers, returns the partial value corresponding to if those numbers
  // are the bounds. For example, if we were storing partial sums, make_partials
  // would be the difference function.
  //
  // We split the value into partial sums which corresponds to each bucket of
  // ApproxBounds. For example, consider value = 7 and the bins (0, 1], (1, 2],
  // (2, 4], (4, 8], and the four corresponding negative bins. We would store
  // partial sums:
  // (0, 1]: 1
  // (1, 2]: 1
  // (2, 4]; 2
  // (4, 8]: min(3, 4) = 3
  //
  // Then later, say our bounds are [0, 4]. We can then sum the partial values
  // that lie in bins that are included in the bounds. In our case it is bins
  // (0, 1], (1, 2], (2, 4]. So 1 + 1 + 2 = 4. This is the same result if our
  // value 7 was initially clamped between [0, 4].
  template <typename T2>
  void AddToPartials(std::vector<T2>* partials, T value,
                     std::function<T2(T, T)> make_partial) {
    AddMultipleEntriesToPartials<T2>(partials, value, 1, make_partial);
  }

  template <typename T2>
  void AddToPartialSums(std::vector<T2>* sums, T value) {
    AddMultipleEntriesToPartialSums<T2>(sums, value, 1);
  }

  // Given two vectors of partial values, add the partials in the bins between
  // the boundaries corresponding to lower and upper to get the clamped value.
  // The value_transform and count parameters are used to calculate the
  // contribution of values clamped below lower or above upper, if applicable.
  template <typename T2>
  absl::StatusOr<T2> ComputeFromPartials(const std::vector<T2>& pos_partials,
                                         const std::vector<T2>& neg_partials,
                                         std::function<T2(T)> value_transform,
                                         T lower, T upper, int64_t count) {
    RETURN_IF_ERROR(ValidateIsNonNegative(count, "Count"));

    // Find value by adding the partial values corresponding to bins that are
    // between the lower and upper bound. ApproxBounds will always return a
    // bin boundary as lower and upper bounds.
    int lower_msb = MostSignificantBit(lower);
    int upper_msb = MostSignificantBit(upper);

    // Find value from its per-bin partials.
    T2 value = 0;
    if (lower <= 0 && 0 <= upper) {
      // If 0 is in [lower, upper], then we sum partial values corresponding
      // to 0 to upper, and also from 0 to lower in the negative vectors.
      if (lower < 0) {
        for (int i = 0; i <= lower_msb; ++i) {
          value += neg_partials[i];
        }
      }
      if (upper > 0) {
        for (int i = 0; i <= upper_msb; ++i) {
          value += pos_partials[i];
        }
      }
    } else if (upper < 0) {
      // If lower and upper are negative, each value is clamped so that they
      // contributed at most upper. Anything less they contributed is stored
      // in partial values between lower and upper, which we add.
      value += count * value_transform(upper);
      for (int i = upper_msb + 1; i <= lower_msb; ++i) {
        value += neg_partials[i];
      }
    } else {  // 0 < lower <= upper
      // If lower and upper are both positive, each value is clamped to it
      // contributed at least lower. Anything more contributed is stored
      // between lower and upper in positive vectors, which we add.
      value += count * value_transform(lower);
      for (int i = lower_msb + 1; i <= upper_msb; ++i) {
        value += pos_partials[i];
      }
    }
    return value;
  }

 private:
  ApproxBoundsProvider(double epsilon, int64_t num_bins, double scale,
                       double base, std::vector<T> bin_boundaries,
                       double success_probability,
                       std::unique_ptr<NumericalMechanism> mechanism,
                       double delta = 0.0)
      : BoundsProvider<T>(),
        epsilon_(epsilon),
        delta_(delta),
        pos_bins_(num_bins, 0),
        neg_bins_(num_bins, 0),
        scale_(scale),
        base_(base),
        bin_boundaries_(bin_boundaries),
        success_probability_(success_probability),
        mechanism_(std::move(mechanism)) {}

  // Default scale depends on the input type T.
  static double DefaultScaleForT() {
    return std::is_integral<T>::value ? 1.0 : std::numeric_limits<T>::min();
  }

  // Default number of bins depends on the input type T, scale, and base.
  // Take the subtraction of two logarithms to prevent overflows.
  static int64_t DefaultNumBins(double scale, double base) {
    return std::ceil(
               (std::log(std::numeric_limits<T>::max()) - std::log(scale)) /
               std::log(base)) +
           1;
  }

  // Given a bin index, finds the larger-magnitude boundary of the corresponding
  // bin for negative bin.
  T NegRightBinBoundary(int bin_index) {
    T pos_rbb = PosRightBinBoundary(bin_index);
    if (pos_rbb == std::numeric_limits<T>::max()) {
      return std::numeric_limits<T>::lowest();
    }
    return -pos_rbb;
  }

  // Given a bin index, finds the larger-magnitude boundary of the corresponding
  // bin for positive bin.
  T PosRightBinBoundary(int bin_index) { return bin_boundaries_[bin_index]; }

  std::optional<T> FindUpperBound(double threshold) {
    // Find first bin above threshold for maximum.
    for (int i = pos_bins_.size() - 1; i >= 0; --i) {
      if (noisy_pos_bins_[i] >= threshold) {
        return PosRightBinBoundary(i);
      }
    }

    for (int i = 0; i < neg_bins_.size(); ++i) {
      if (noisy_neg_bins_[i] >= threshold) {
        return NegLeftBinBoundary(i);
      }
    }

    return std::nullopt;
  }

  std::optional<T> FindLowerBound(double threshold) {
    // Find first bin above threshold for minimum.
    for (int i = neg_bins_.size() - 1; i >= 0; --i) {
      if (noisy_neg_bins_[i] >= threshold) {
        return NegRightBinBoundary(i);
      }
    }
    for (int i = 0; i < pos_bins_.size(); ++i) {
      if (noisy_pos_bins_[i] >= threshold) {
        return PosLeftBinBoundary(i);
      }
    }
    return std::nullopt;
  }

  // Finds approximate bounds by comparing noised bin counts to a threshold.
  // This method does not add any noise (it assumes that noisy_pos_bins_ and
  // noisy_neg_bins_ have been initialised) so calling this method multiple
  // times with different thresholds is DP: the noised histogram is itself DP.
  std::optional<BoundsResult<T>> FindBounds(double threshold) {
    std::optional<T> lower_bound = FindLowerBound(threshold);
    if (!lower_bound.has_value()) {
      return std::nullopt;
    }

    std::optional<T> upper_bound = FindUpperBound(threshold);
    if (!upper_bound.has_value()) {
      return std::nullopt;
    }

    return BoundsResult<T>{
        lower_bound.value(),
        upper_bound.value(),
    };
  }

  // Add noise to each member of bins and return noisy vector.
  std::vector<T> AddNoise(const std::vector<int64_t>& bins) {
    std::vector<T> noisy_bins(bins.size());
    for (int i = 0; i < bins.size(); ++i) {
      noisy_bins[i] = mechanism_->AddNoise(bins[i]);
    }
    return noisy_bins;
  }

  // Adds value to partials (as described in comment for AddToPartials())
  // num_of_entries times. This function more efficiently adds multiple entries
  // at once, instead of using AddToPartials() in a for-loop.
  template <typename T2>
  void AddMultipleEntriesToPartials(std::vector<T2>* partials, T value,
                                    int64_t num_of_entries,
                                    std::function<T2(T, T)> make_partial) {
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    absl::Status status =
        ValidateIsPositive(num_of_entries, "Number of entries");
    if (std::isnan(static_cast<double>(value)) || !status.ok()) {
      return;
    }

    int msb = MostSignificantBit(value);

    // Each bin of the logarithmic histograms in ApproxBounds can be a candidate
    // for auto-determined upper and lower bounds. Thus, we store a contribution
    // of the value from the value for each bin.
    for (int i = 0; i <= msb; ++i) {
      // The maximum contribution to the bin is the partial between boundaries.
      T2 partial = 0;
      if (value >= 0) {
        partial = make_partial(PosRightBinBoundary(i), PosLeftBinBoundary(i));
      } else {
        partial = make_partial(NegRightBinBoundary(i), NegLeftBinBoundary(i));
      }

      if (i < msb) {
        // For indices below the msb, add the maximum contribution
        // (num_of_entries times) to the partial.

        (*partials)[i] += partial * num_of_entries;
      } else {
        // For i = msb, add the remaining contribution (num_of_entries times),
        // but not more than the maximum contribution to the partial for this
        // bin. This may occur if the msb was clamped by the ApproxBounds not
        // having enough bins.
        T2 remainder;
        if (value > 0) {
          remainder = make_partial(value, PosLeftBinBoundary(i));
        } else {
          remainder = make_partial(value, NegLeftBinBoundary(i));
        }
        if (std::abs(partial) < std::abs(remainder)) {
          (*partials)[msb] += partial * num_of_entries;
        } else {
          (*partials)[msb] += remainder * num_of_entries;
        }
      }
    }
  }

  // Break value into its partial sums and store it into the sums vector. A
  // specific use case of AddToPartials used in some algorithms.
  template <typename T2>
  void AddMultipleEntriesToPartialSums(std::vector<T2>* sums, T value,
                                       int64_t num_of_entries) {
    AddMultipleEntriesToPartials<T2>(
        sums, value, num_of_entries,
        [](T val1, T val2) { return val1 - val2; });
  }

  // Given a bin index, finds the larger-magnitude boundary of the corresponding
  // bin for positive bin.
  T PosRightBinBoundary(int bin_index) const {
    return bin_boundaries_[bin_index];
  }

  // Given a bin index, finds the smaller-magnitude boundary of the
  // corresponding bin for positive bin.
  T PosLeftBinBoundary(int bin_index) const {
    if (bin_index == 0) {
      return 0;
    }
    return PosRightBinBoundary(bin_index - 1);
  }

  // Given a bin index, finds the smaller-magnitude boundary of the
  // corresponding bin for negative bin.
  T NegLeftBinBoundary(int bin_index) const {
    return -1 * PosLeftBinBoundary(bin_index);
  }

  // Find the most significant bit of the magnitude of the value. For the
  // special case 0, return 0. This is used as the bin index.
  int MostSignificantBit(T value) const {
    // Handle 0 value separately since log(0) is undefined.
    if (value == 0) {
      return 0;
    }

    // Clamp infinities to highest and lowest value.
    value = Clamp(std::numeric_limits<T>::lowest(),
                  std::numeric_limits<T>::max(), value);
    // Sometimes the minimum numeric limit has greater magnitude than the
    // maximum. In this case clamp its magnitude at the maximum numeric limit to
    // find msb. In reality our negative bin will accommodate the value.
    T abs = (value <= -1 * std::numeric_limits<T>::max())
                ? std::numeric_limits<T>::max()
                : std::abs(value);
    // Calculate the most significant bit and clamp to a valid bin index.
    int msb = std::ceil((std::log(abs) - std::log(scale_)) / std::log(base_));
    int bin_index =
        std::max(0, std::min(msb, static_cast<int>(pos_bins_.size() - 1)));
    // Floating-point precision errors mean that for some bin boundaries, we'll
    // end up calculating the larger-magnitude bin rather than the smaller.
    if ((value > 0 && value <= PosLeftBinBoundary(bin_index)) ||
        (value < 0 && value >= NegLeftBinBoundary(bin_index))) {
      return std::max(0, bin_index - 1);
    }
    return bin_index;
  }

  // Calculate the noisy number of inputs outside the two bounds from the
  // most recent result generation. Inputs equal to either bound may or may not
  // be part of the count. Input lower and upper are rounded to the nearest
  // larger-magnitude bin boundary.
  absl::StatusOr<double> NumInputsOutside(T lower, T upper) const {
    // Check that noisy bins have been populated.
    if (noisy_pos_bins_.empty()) {
      return absl::InvalidArgumentError(
          "Noisy histogram bins have not been created. Try generating "
          "results first.");
    }

    int lower_msb = MostSignificantBit(lower);
    int upper_msb = MostSignificantBit(upper);
    double num_outside = 0;

    // Add the count of inputs below lower.
    int pos_i = 0;
    int neg_i = noisy_neg_bins_.size();
    if (lower == 0) {
      neg_i = -1;
    } else if (lower < 0) {
      neg_i = lower_msb;
    } else {  // lower > 0
      neg_i = -1;
      pos_i = lower_msb + 1;
    }
    for (int i = noisy_neg_bins_.size() - 1; i > neg_i; --i) {
      num_outside += noisy_neg_bins_[i];
    }
    for (int i = 0; i < pos_i; ++i) {
      num_outside += noisy_pos_bins_[i];
    }

    // Add the count of inputs above upper.
    pos_i = noisy_pos_bins_.size();
    neg_i = -1;
    if (upper == 0) {
      pos_i = 0;
    } else if (upper < 0) {
      pos_i = 0;
      neg_i = upper_msb;
    } else {  // upper > 0.
      pos_i = upper_msb + 1;
    }
    for (int i = neg_i; i >= 0; --i) {
      num_outside += noisy_neg_bins_[i];
    }
    for (int i = pos_i; i < noisy_pos_bins_.size(); ++i) {
      num_outside += noisy_pos_bins_[i];
    }

    return num_outside;
  }

  // Get the noisy number of total inputs by summing all noisy histogram bins.
  absl::StatusOr<double> NumInputs() const {
    // Number of inputs outside of the empty set.
    return NumInputsOutside(0, 0);
  }

  const double epsilon_;
  const double delta_;
  // Count the values in each logarithmic bin for positives and negatives.
  std::vector<int64_t> pos_bins_;
  std::vector<int64_t> neg_bins_;

  // Noisy DP counts of the positive and negative bins. Populated upon
  // generating the result.
  std::vector<T> noisy_pos_bins_;
  std::vector<T> noisy_neg_bins_;

  // The bin boundary magnitudes, starting from lowest positive magnitude.
  std::vector<T> bin_boundaries_;

  // Multiplicative factor for inputs
  const double scale_;

  // Base of the logarithm.
  const double base_;

  // The desired probability that, when the dataset is empty, no bin counts are
  // above the threshold for determining whether a bin is empty.
  const double success_probability_;

  // Mechanism for adding noise to buckets.
  std::unique_ptr<NumericalMechanism> mechanism_;
};
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_APPROX_BOUNDS_PROVIDER_H_
