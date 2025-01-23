//
// Copyright 2024 Google LLC
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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_CLAMPED_CALCULATION_WITHOUT_BOUNDS_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_CLAMPED_CALCULATION_WITHOUT_BOUNDS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "algorithms/util.h"
#include "base/status_macros.h"

namespace differential_privacy::internal {

template <typename T>
class ClampedCalculationWithoutBounds {
 public:
  struct Options {
    int64_t num_bins;
    double scale;
    double base;
  };

  // Create a new instance of ClampedCalculationWithoutBounds.
  static std::unique_ptr<ClampedCalculationWithoutBounds> Create(
      const Options& options) {
    // Cache the bin boundary magnitudes for performance. Note that casting
    // numeric limits lead to inconsistencies.
    std::vector<T> bin_boundaries(options.num_bins, 0);
    auto get_boundary = [&, next_upper_bound = options.scale]() mutable {
      if (next_upper_bound >= std::numeric_limits<T>::max() / options.base) {
        return std::numeric_limits<T>::max();
      }
      const double this_boundary = next_upper_bound;
      next_upper_bound *= options.base;
      return static_cast<T>(this_boundary);
    };
    std::generate(bin_boundaries.begin(), bin_boundaries.end(), get_boundary);

    return absl::WrapUnique(new ClampedCalculationWithoutBounds<T>(
        options.num_bins, options.scale, options.base, bin_boundaries));
  }

  // Splits the value into the elements of the partials vector, each of which
  // corresponds to a bin. Function `make_partial` is a function that given two
  // numbers, returns the partial value corresponding to if those numbers are
  // the bounds. For example, if we were storing partial sums, make_partials
  // would be the difference function.
  //
  // We split the value into partial sums which corresponds to each bucket. For
  // example, consider value = 7 and the bins (0, 1], (1, 2], (2, 4], (4, 8],
  // and the four corresponding negative bins. We would store partial sums:
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
                     std::function<T2(T, T)> make_partial) const {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(value)) {
        return;
      }
    }

    int msb = MostSignificantBit(value);

    // Each bin of the logarithmic histograms can be a candidate for
    // auto-determined upper and lower bounds. Thus, we store a contribution of
    // the value from the value for each bin.
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

        (*partials)[i] += partial;
      } else {
        // For i = msb, add the remaining contribution (num_of_entries times),
        // but not more than the maximum contribution to the partial for this
        // bin. This may occur if the msb was clamped as not having enough bins.
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

  template <typename T2>
  void AddToPartialSums(std::vector<T2>* sums, T value) const {
    AddToPartials<T2>(sums, value, [](T val1, T val2) { return val1 - val2; });
  }

  // Given two vectors of partial values, add the partials in the bins between
  // the boundaries corresponding to lower and upper to get the clamped value.
  // The value_transform and count parameters are used to calculate the
  // contribution of values clamped below lower or above upper, if applicable.
  template <typename T2>
  absl::StatusOr<T2> ComputeFromPartials(const std::vector<T2>& pos_partials,
                                         const std::vector<T2>& neg_partials,
                                         std::function<T2(T)> value_transform,
                                         T lower, T upper,
                                         int64_t count) const {
    RETURN_IF_ERROR(ValidateIsNonNegative(count, "Count"));

    // Find value by adding the partial values corresponding to bins that are
    // between the lower and upper bound.
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

  int64_t MemoryUsed() const {
    return sizeof(ClampedCalculationWithoutBounds) +
           (sizeof(T) * bin_boundaries_.capacity());
  }

  int64_t GetNumBins() const { return num_bins_; }

  double GetScaleForTesting() const { return scale_; }

  double GetBaseForTesting() const { return base_; }

 private:
  // The number of positive bins.
  const int64_t num_bins_;

  // Multiplicative factor for inputs.
  const double scale_;

  // Base of the logarithm.
  const double base_;

  const std::vector<T> bin_boundaries_;

  ClampedCalculationWithoutBounds(int64_t num_bins, double scale, double base,
                                  const std::vector<T> bin_boundaries)
      : num_bins_(num_bins),
        scale_(scale),
        base_(base),
        bin_boundaries_(bin_boundaries) {}

  // Given a bin index, finds the larger-magnitude boundary of the corresponding
  // bin for negative bin.
  T NegRightBinBoundary(int bin_index) const {
    T pos_rbb = PosRightBinBoundary(bin_index);
    if (pos_rbb == std::numeric_limits<T>::max()) {
      return std::numeric_limits<T>::lowest();
    }
    return -1 * pos_rbb;
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
    T abs;
    if (value <= -1 * std::numeric_limits<T>::max()) {
      abs = std::numeric_limits<T>::max();
    } else {
      abs = std::abs(value);
    }

    // Calculate the most significant bit and clamp to a valid bin index.
    int msb = std::ceil((std::log(abs) - std::log(scale_)) / std::log(base_));
    int bin_index = std::max(0, std::min(msb, static_cast<int>(num_bins_ - 1)));

    // Floating-point precision errors mean that for some bin boundaries, we'll
    // end up calculating the larger-magnitude bin rather than the smaller.
    if ((value > 0 && value <= PosLeftBinBoundary(bin_index)) ||
        (value < 0 && value >= NegLeftBinBoundary(bin_index))) {
      return std::max(0, bin_index - 1);
    }
    return bin_index;
  }
};

}  // namespace differential_privacy::internal

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_INTERNAL_CLAMPED_CALCULATION_WITHOUT_BOUNDS_H_
