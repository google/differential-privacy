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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_APPROX_BOUNDS_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_APPROX_BOUNDS_H_

#include <cmath>
#include <limits>

#include "google/protobuf/any.pb.h"
#include "base/status.h"
#include "algorithms/algorithm.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "base/status_macros.h"

namespace differential_privacy {

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
// probability is the probability that we select a bin that wasn't empty before
// noise addition. Therefore, increasing success_probability will increase the
// threshold. Alternatively, the threshold can be passed as a parameter
// directly.
//
// We chose the bin that succeeds the threshold count and return its greater
// boundary as the approx max. Similarly the leftmost bin that succeeds the
// threshold count is chosen and its smaller boundary is returned as the approx
// min. If the success_probability is too high it is possible that no bin is
// greater than the threshold. In this case we return an error status in the
// output.
//
// For example, if
//   scale = 2, base = 1, num_bins = 4, inputs = {0, 0, 0, 0, 1, 3, 7, 8, 8, 8}
// We have histogram bins and counts
//   [0, 1]: 5
//   (1, 2]: 0
//   (2, 4]: 1
//   (4, 8]: 4
// Then if success_probability=.9 and epsilon=1 we will obtain approximately
// threshold=3.5. Since the count of bin (4, 8] > threshold we return an
// approx max of 2^3 = 8. Since the count of bin [0,1] > threshold we return an
// approx min of 0.
template <typename T>
class ApproxBounds : public Algorithm<T> {
 public:
  class Builder : public AlgorithmBuilder<T, ApproxBounds<T>, Builder> {
    using AlgorithmBuilder =
        differential_privacy::AlgorithmBuilder<T, ApproxBounds<T>, Builder>;

   public:
    // Constructor sets default values depending on the input type. Bins are
    // created to cover entire range of type T.
    Builder()
        : AlgorithmBuilder(),
          base_(2),
          success_probability_(1 - std::pow(10, -9)) {
      if (std::is_integral<T>::value) {
        scale_ = 1.0;
      } else {
        scale_ = std::numeric_limits<T>::min();
      }
      // Take the subtraction of two logarithms to prevent overflow.
      num_bins_ = std::ceil((std::log(std::numeric_limits<T>::max()) -
                             std::log(scale_)) /
                            std::log(base_)) +
                  1;
    }

    Builder& SetNumBins(int64_t num_bins) {
      num_bins_ = num_bins;
      return *static_cast<Builder*>(this);
    }

    Builder& SetScale(double scale) {
      scale_ = scale;
      return *static_cast<Builder*>(this);
    }

    Builder& SetBase(double base) {
      base_ = base;
      return *static_cast<Builder*>(this);
    }

    // Set exactly one of success_probability or k threshold.
    Builder& SetSuccessProbability(double success_probability) {
      success_probability_ = success_probability;
      has_k_ = false;
      return *static_cast<Builder*>(this);
    }

    // Set exactly one of success_probability or k threshold.
    Builder& SetThreshold(double k) {
      k_ = k;
      has_k_ = true;
      return *static_cast<Builder*>(this);
    }

   private:
    base::StatusOr<std::unique_ptr<ApproxBounds<T>>> BuildAlgorithm() override {
      std::unique_ptr<NumericalMechanism> mechanism;
      ASSIGN_OR_RETURN(mechanism, AlgorithmBuilder::UpdateAndBuildMechanism());

      // Check the validity of the histogram parameters. num_bin and
      // success_probability restrictions prevent undefined threshold
      // calculation.
      if (num_bins_ < 1) {
        return base::InvalidArgumentError("Must have one or more bins.");
      }
      if (scale_ <= 0) {
        return base::InvalidArgumentError("Scale must be positive.");
      }
      if (base_ <= 1) {
        return base::InvalidArgumentError("Base must be greater than 1.");
      }
      if (has_k_) {
        if (k_ < 0) {
          return base::InvalidArgumentError("k threshold must be nonnegative.");
        }
      } else {
        if (success_probability_ <= 0 || success_probability_ >= 1) {
          return base::InvalidArgumentError(
              "Success percentage must be between 0 and 1.");
        }
      }

      if (!has_k_) {
        // Calculate minimum bin count threshold given success probability.
        // Given the success probability, find the threshold count needed for a
        // given bin in order for it to be chosen. Note if probability is too
        // high, the threshold will be too high for any bin to be chosen; then
        // we return an error in the output. This is calculated assuming
        // Laplacian noise is added.
        k_ = -log(2 - 2 * std::pow(success_probability_,
                                   1.0 / (2 * num_bins_ - 1))) /
             AlgorithmBuilder::epsilon_.value();
      }

      // Create ApproxBounds.
      return absl::WrapUnique(
          new ApproxBounds(AlgorithmBuilder::epsilon_.value(), num_bins_,
                           scale_, base_, k_, has_k_, std::move(mechanism)));
    }

    // Stores whether threshold k is set.
    bool has_k_ = false;

    double k_;
    double scale_;
    double base_;
    int64_t num_bins_;
    double success_probability_;
  };

  void AddEntry(const T& input) override {
    if (std::isnan(input)) {
      return;
    }

    // Place into correct bin according to most significant bit and sign. Note
    // that MostSignificantBit returns 0 for 0.
    int index = MostSignificantBit(input);
    if (input >= 0) {
      ++pos_bins_[index];
    } else {  // value < 0
      ++neg_bins_[index];
    }
  }

  // Serialize the positive and negative bin counts.
  Summary Serialize() override {
    ApproxBoundsSummary am_summary;
    *am_summary.mutable_pos_bin_count() = {pos_bins_.begin(), pos_bins_.end()};
    *am_summary.mutable_neg_bin_count() = {neg_bins_.begin(), neg_bins_.end()};
    Summary summary;
    summary.mutable_data()->PackFrom(am_summary);
    return summary;
  }

  // Retrieve positive and negative bin counts from summary and add them.
  base::Status Merge(const Summary& summary) override {
    if (!summary.has_data()) {
      return base::InvalidArgumentError(
          "Cannot merge summary with no histogram data.");
    }
    ApproxBoundsSummary am_summary;
    if (!summary.data().UnpackTo(&am_summary)) {
      return base::InvalidArgumentError(
          "Approximate bounds summary unable to be unpacked.");
    }

    if (pos_bins_.size() != am_summary.pos_bin_count_size() ||
        neg_bins_.size() != am_summary.neg_bin_count_size()) {
      return base::InvalidArgumentError(
          "Merged approximate max summary must have the same number of "
          "bin counts as this histogram.");
    }

    // Add bin count from summary to each bin.
    for (int i = 0; i < pos_bins_.size(); ++i) {
      pos_bins_[i] += am_summary.pos_bin_count(i);
      neg_bins_[i] += am_summary.neg_bin_count(i);
    }
    return base::OkStatus();
  }

  int64_t MemoryUsed() override {
    int64_t memory = sizeof(ApproxBounds<T>) +
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

  // Find the most significant bit of the magnitude of the value. For the
  // special case 0, return 0. This is used as the bin index.
  int MostSignificantBit(T value) {
    // Handle 0 value seperately since log(0) is undefined.
    if (value == 0) {
      return 0;
    }

    // Sometimes the minimum numeric limit has greater magnitude than the
    // maximum. In this case clamp its magnitude at the maximum numeric limit to
    // find msb. In reality our negative bin will accommodate the value.
    T abs;
    if (value <= -1 * std::numeric_limits<T>::max()) {
      abs = std::numeric_limits<T>::max();
    } else {
      abs = std::abs(value);
    }

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
        // For indices below the msb, add the maximum contribution to the
        // partial.
        (*partials)[i] += partial;
      } else {
        // For i = msb, add the remaining contribution, but not more than the
        // maximum contribution to the partial for this bin. This may occur if
        // the msb was clamped by the ApproxBounds not having enough bins.
        T2 remainder;
        if (value > 0) {
          remainder = make_partial(value, PosLeftBinBoundary(i));
        } else {
          remainder = make_partial(value, NegLeftBinBoundary(i));
        }
        if (std::abs(partial) < std::abs(remainder)) {
          (*partials)[msb] += partial;
        } else {
          (*partials)[msb] += remainder;
        }
      }
    }
  }

  // Break value into its partial sums and store it into the sums vector. A
  // specific use case of AddToPartials used in some algorithms.
  template <typename T2>
  void AddToPartialSums(std::vector<T2>* sums, T value) {
    AddToPartials<T2>(sums, value, [](T val1, T val2) { return val1 - val2; });
  }

  // Given two vectors of partial values, add the partials in the bins between
  // the boundaries corresponding to lower and upper to get the clamped value.
  // The value_transform and count parameters are used to calculate the
  // contribution of values clamped below lower or above upper, if applicable.
  template <typename T2>
  T2 ComputeFromPartials(const std::vector<T2>& pos_partials,
                         const std::vector<T2>& neg_partials,
                         std::function<T2(T)> value_transform, T lower, T upper,
                         size_t count) {
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

  // Get additional private information in the form of a BoundingReport. Will
  // populate any fields possible and leave the rest blank.
  BoundingReport GetBoundingReport(T lower, T upper) {
    BoundingReport report;
    SetValue<T>(report.mutable_lower_bound(), lower);
    SetValue<T>(report.mutable_upper_bound(), upper);
    base::StatusOr<double> count = NumInputs();
    base::StatusOr<double> count_outside = NumInputsOutside(lower, upper);
    if (count.ok()) {
      report.set_num_inputs(count.ValueOrDie());
    }
    if (count_outside.ok()) {
      report.set_num_outside(count_outside.ValueOrDie());
    }
    return report;
  }

 protected:
  ApproxBounds(double epsilon, int64_t num_bins, double scale, double base,
               double k, bool preset_k,
               std::unique_ptr<NumericalMechanism> mechanism)
      : Algorithm<T>(epsilon),
        pos_bins_(num_bins, 0),
        neg_bins_(num_bins, 0),
        bin_boundaries_(num_bins, 0),
        scale_(scale),
        base_(base),
        k_(k),
        preset_k_(preset_k),
        mechanism_(std::move(mechanism)) {
    // Cache the bin boundary magnitudes for performance. Note that casting
    // numeric limits lead to inconsistencies.
    auto get_boundary = [boundary = scale_, base = base_]() mutable {
      if (boundary >= std::numeric_limits<T>::max() / base) {
        return std::numeric_limits<T>::max();
      }
      double this_boundary = boundary;
      boundary *= base;
      return static_cast<T>(this_boundary);
    };
    std::generate(bin_boundaries_.begin(), bin_boundaries_.end(), get_boundary);
  }

  // Returns an output containing approximate min as the first element and
  // approximate max as the second element. If not enough inputs exist to pass
  // the threshold, populate the output with an error status.
  base::StatusOr<Output> GenerateResult(double privacy_budget,
                                        double noise_interval_level) override {
    DCHECK_GT(privacy_budget, 0.0)
        << "Privacy budget should be greater than zero.";
    if (privacy_budget == 0.0) return Output();

    // If k was not user set, scale it by the privacy_budget to ensure the
    // correct probability of success.
    double threshold = k_;
    if (!preset_k_) {
      threshold /= privacy_budget;
    }

    // Populate noisy versions of the histogram bins.
    noisy_pos_bins_ = AddNoise(privacy_budget, pos_bins_);
    noisy_neg_bins_ = AddNoise(privacy_budget, neg_bins_);

    Output output;

    // Find first bin above threshold for minimum.
    for (int i = neg_bins_.size() - 1; i >= 0; --i) {
      if (noisy_neg_bins_[i] >= threshold) {
        AddToOutput<T>(&output, NegRightBinBoundary(i));
        break;
      }
    }
    if (output.elements_size() == 0) {
      for (int i = 0; i < pos_bins_.size(); ++i) {
        if (noisy_pos_bins_[i] >= threshold) {
          AddToOutput<T>(&output, PosLeftBinBoundary(i));
          break;
        }
      }
    }

    // Find first bin above threshold for maximum.
    for (int i = pos_bins_.size() - 1; i >= 0; --i) {
      if (noisy_pos_bins_[i] >= threshold) {
        AddToOutput<T>(&output, PosRightBinBoundary(i));
        break;
      }
    }
    if (output.elements_size() < 2) {
      for (int i = 0; i < neg_bins_.size(); ++i) {
        if (noisy_neg_bins_[i] >= threshold) {
          AddToOutput<T>(&output, NegLeftBinBoundary(i));
          break;
        }
      }
    }

    // Record error status if approx min or max was not found.
    if (output.elements_size() < 2) {
      return base::InvalidArgumentError(
          "Bin count threshold was too large to find approximate "
          "bounds. Either run over a larger dataset or decrease "
          "success_probability and try again.");
    }

    return output;
  }

  void ResetState() override {
    std::fill(pos_bins_.begin(), pos_bins_.end(), 0);
    std::fill(neg_bins_.begin(), neg_bins_.end(), 0);
  }

  // Given a bin index, finds the larger-magnitude boundary of the corresponding
  // bin for negative bin.
  T NegRightBinBoundary(int bin_index) {
    T pos_rbb = PosRightBinBoundary(bin_index);
    if (pos_rbb == std::numeric_limits<T>::max()) {
      return std::numeric_limits<T>::lowest();
    }
    return -1 * pos_rbb;
  }

  // Given a bin index, finds the larger-magnitude boundary of the corresponding
  // bin for positive bin.
  T PosRightBinBoundary(int bin_index) { return bin_boundaries_[bin_index]; }

 private:
  // Add noise to each member of bins and return noisy vector.
  const std::vector<T> AddNoise(double privacy_budget,
                                const std::vector<int64_t>& bins) {
    std::vector<T> noisy_bins(bins.size());
    for (int i = 0; i < bins.size(); ++i) {
      noisy_bins[i] = mechanism_->AddNoise(bins[i], privacy_budget);
    }
    return noisy_bins;
  }

  // Given a bin index, finds the smaller-magnitude boundary of the
  // corresponding bin for positive bin.
  T PosLeftBinBoundary(int bin_index) {
    if (bin_index == 0) {
      return 0;
    }
    return PosRightBinBoundary(bin_index - 1);
  }

  // Given a bin index, finds the smaller-magnitude boundary of the
  // corresponding bin for negative bin.
  T NegLeftBinBoundary(int bin_index) {
    return -1 * PosLeftBinBoundary(bin_index);
  }

  // Calculate the noisy number of inputs outside the two bounds from the
  // most recent result generation. Inputs equal to either bound may or may not
  // be part of the count. Input lower and upper are rounded to the nearest
  // larger-magnitude bin boundary.
  base::StatusOr<double> NumInputsOutside(T lower, T upper) {
    // Check that noisy bins have been populated.
    if (noisy_pos_bins_.empty()) {
      return base::InvalidArgumentError(
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
  base::StatusOr<double> NumInputs() {
    // Number of inputs outside of the empty set.
    return NumInputsOutside(0, 0);
  }

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
  double scale_;

  // Base of the logarithm.
  double base_;

  // The bin count threshold for choosing a minimum / maximum.
  double k_;

  // Whether k was user-set. If true, then do not scale by privacy budget.
  bool preset_k_;

  // Mechanism for adding noise to buckets.
  std::unique_ptr<NumericalMechanism> mechanism_;
};

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_APPROX_BOUNDS_H_
