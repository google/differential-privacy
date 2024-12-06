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

#ifndef DIFFERENTIAL_PRIVACY_ALGORITHMS_UTIL_H_
#define DIFFERENTIAL_PRIVACY_ALGORITHMS_UTIL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "base/status_macros.h"

namespace differential_privacy {

// Arbitrary default value for epsilon. The algorithm interface falls back on
// this value whenever one is not provided. This value should only be used for
// testing convenience. For any production use case, please set your own epsilon
// based on privacy needs.
ABSL_DEPRECATED("Use your own epsilon based on privacy considerations.")
double DefaultEpsilon();

// Returns the smallest power of 2 greater than or equal to n. n must be > 0.
// Includes negative powers.
double GetNextPowerOfTwo(double n);

// Rounds n to the nearest multiple of base. Ties are broken towards +inf.
// If base is 0, returns n.
double RoundToNearestDoubleMultiple(double n, double base);

int64_t RoundToNearestInt64Multiple(int64_t n, int64_t base);

// Templates are needed for RoundToNearestMultiple(), since without them and
// instead trying to overload RoundToNearestMultiple() causes C++ compiler
// errors stating, for example, RoundToNearestMultiple(5, 3) is ambiguous.
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
T RoundToNearestMultiple(T n, T base) {
  return RoundToNearestInt64Multiple(n, base);
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
T RoundToNearestMultiple(T n, T base) {
  return RoundToNearestDoubleMultiple(n, base);
}

// Return 1 if n > 0, -1 if n < 0, and 0 if n == 0.
template <typename T>
T sign(T n) {
  if (n > 0) return 1;
  if (n < 0) return -1;
  return 0;
}

// Approximate the inverse of the error function.
// Implementation based on Table 5 in Giles' paper
// on approximating the inverse of the error function
// (https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf).
double InverseErrorFunction(double x);

// Estimation of the inverse cdf of the normal distribution centered at mu with
// standard deviation sigma, at probability p. Based on Abramowitz and Stegun
// formula 26.2.23. The error of the estimation is bounded by 4.5 e-4. This
// function will fail if higher accuracy is required.
absl::StatusOr<double> Qnorm(double p, double mu = 0.0, double sigma = 1.0);

template <typename T>
inline const T& Clamp(const T& low, const T& high, const T& value) {
  // Prevents errors in ordering the arguments.
  DCHECK(!(high < low));
  if (high < value) return high;
  if (value < low) return low;
  return value;
}

// Return value for the Safe* operation functions below, including the cast
// resulting value of the operation and whether or not the operation caused an
// overflow.
template <typename T>
struct SafeOpResult {
  T value;
  bool overflow = false;
};

// When T is an integral type, return the addition result if and whether or not
// there would have been an overflow. Otherwise, assign the numeric limit to
// result and signal that there would have been an overflow.
// Note that this should NOT be used to gracefully handle overflows in
// computations on data. See (broken link)
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline SafeOpResult<T> SafeAdd(T lhs, T rhs) {
  if (lhs > 0) {
    // For negative rhs, we will never overflow.
    if (rhs > 0) {
      T safe_distance = std::numeric_limits<T>::max() - lhs;
      if (safe_distance < rhs) {
        return SafeOpResult<T>{std::numeric_limits<T>::max(), true};
      }
    }
  } else if (lhs < 0) {
    // For positive rhs, we will never overflow.
    if (rhs < 0) {
      T safe_distance = std::numeric_limits<T>::lowest() - lhs;
      if (safe_distance > rhs) {
        return SafeOpResult<T>{std::numeric_limits<T>::lowest(), true};
      }
    }
  }
  return SafeOpResult<T>{lhs + rhs, false};
}

// When T is a floating-point type, perform a simple addition, since
// floating-point types don't have the same overflow issues as integral types.
// Note that this should NOT be used to gracefully handle overflows in
// computations on data. See (broken link)
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
inline SafeOpResult<T> SafeAdd(T lhs, T rhs) {
  return SafeOpResult<T>{lhs + rhs, false};
}

// When T is an integral type, assign the subtraction result and whether or not
// there was an overflow. Otherwise, assign the numeric limit to result and
// that there would have been an overflow.
// Note that this should NOT be used to gracefully handle overflows in
// computations on data. See (broken link)
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline SafeOpResult<T> SafeSubtract(T lhs, T rhs) {
  // For integral values, the min numeric limit is larger in magnitude than the
  // max numeric limit, so we cannot negate it. For unsigned types, the lowest
  // numeric limit is 0. For signed types, it is negative.
  if (rhs == std::numeric_limits<T>::lowest() && rhs != 0) {
    if (lhs >= 0) {
      // We use std::numeric_limits<T>::max() here, since we assume that
      // std::numeric_limits<T>::max() <= -(-std::numeric_limits<T>::lowest()).
      return SafeOpResult<T>{std::numeric_limits<T>::max(), true};
    } else {
      return SafeOpResult<T>{lhs - rhs, false};
    }
  }

  // For all other values of rhs, add the negation.
  return SafeAdd(lhs, -rhs);
}

// When T is a floating-point type, perform a simple subtraction, since
// floating-point types don't have the same overflow issues as integral types.
// Note that this should NOT be used to gracefully handle overflows in
// computations on data. See (broken link)
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
inline SafeOpResult<T> SafeSubtract(T lhs, T rhs) {
  return SafeOpResult<T>{lhs - rhs, false};
}

// Return true and assign the square result if squaring will not overflow.
// Note that this should NOT be used to gracefully handle overflows in
// computations on data. See (broken link)
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline SafeOpResult<T> SafeSquare(T num) {
  SafeOpResult<T> safe_op_result;
  double max_root = std::pow(std::numeric_limits<T>::max(), 0.5);
  if ((num > 0 && num > static_cast<T>(max_root)) ||
      (num < 0 && num < -1 * static_cast<T>(max_root))) {
    safe_op_result.overflow = true;
    safe_op_result.value = 0;
  } else {
    safe_op_result.overflow = false;
    safe_op_result.value = num * num;
  }
  return safe_op_result;
}

// Tries to convert a double value to an integral value, manually overflowing
// if necessary to avoid a SIGILL error from a static_cast outside the numeric
// limits of T. Returns a pair containing the the cast (and possibly
// overflowed) value and a boolean indicating whether or not the cast would have
// been successful (i.e., true if the cast would have overflowed).
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline SafeOpResult<T> SafeCastFromDouble(const double in) {
  if (std::isnan(in) || !std::isfinite(in)) {
    // Integral types do not support NaN or infinite values.
    return SafeOpResult<T>{std::numeric_limits<T>::quiet_NaN(), true};
  }
  static const int64_t kTMax = std::numeric_limits<T>::max();
  static const int64_t kTLowest = std::numeric_limits<T>::lowest();
  double t_range_size = 1.0 + kTMax - kTLowest;
  bool overflow = false;
  double d_out = in;

  if (d_out > kTMax) {
    overflow = true;
    // Translate `d_out` into the range of T, where
    // `std::round(d_out / t_range_size)` is the number of times `d_out` would
    // have overflowed outside of the range of T. For example, suppose:
    //   T = int16_t;
    //   d_out = 40000;  which is > MAX_INT16 (== 32767)
    // It follows that:
    //   t_range_size = 32767 - (-32768) + 1 = 65536;
    //   d_out = d_out - t_range_size * std::round(d_out / t_range_size)
    //         = 40000 - 65536 * std::round(40000 / 65536)
    //         = 40000 - 65536 * 1
    //         = 40000 - 65536
    //         = -25536;
    // This result is the same as an overflowed int16_t, such that:
    //   decimal ->  int16_t
    //   -----------------
    //         0 ->      0
    //         1 ->      1
    //         2 ->      2
    //          ...
    //     32766 ->  32766
    //     32767 ->  32767
    //     32768 -> -32768   because of an int16_t overflow
    //     32769 -> -32767
    //     32770 -> -32766
    //          ...
    //     39999 -> -25537
    //     40000 -> -25536
    d_out -= t_range_size * std::round(d_out / t_range_size);
  }

  if (d_out < kTLowest) {
    overflow = true;
    // Translate `d_out` into the range of T, where
    // `std::round(d_out / t_range_size)` is the number of times `d_out` would
    // have underflowed outside of the range of T. For example, suppose:
    //   T = int16_t;
    //   d_out = -40000;  which is < LOWEST_INT16 (== -32768)
    // It follows that:
    //   t_range_size = 32767 - (-32768) + 1 = 65536;
    //   d_out = d_out + t_range_size * std::round(-d_out / t_range_size)
    //         = -40000 + 65536 * std::round(-(-40000) / 65536)
    //         = -40000 + 65536 * std::round(40000 / 65536)
    //         = -40000 + 65536 * 1
    //         = -40000 + 65536
    //         = 25536;
    // This result is the same as an underflowed int16_t, such that:
    //   decimal ->  int16_t
    //   -----------------
    //         0 ->      0
    //        -1 ->     -1
    //        -2 ->     -2
    //          ...
    //    -32766 ->  -32766
    //    -32767 ->  -32767
    //    -32768 ->  -32768
    //    -32769 ->   32767   because of an int16_t overflow
    //    -32770 ->   32766
    //          ...
    //    -39999 ->   25537
    //    -40000 ->   25536
    d_out += t_range_size * std::round(-d_out / t_range_size);
  }
  double d_out_floor = std::trunc(d_out);

  // Since floating-point variables are only approximations of values (and not
  // the precise value itself), they can still have residual decimal values that
  // are outside of the numeric limits of T, which would cause a static_cast to
  // crash with a SIGILL error. To illustrate, if `d_out` == MAX_INT64, then
  // `static_cast<int64_t>(d_out)` will cause a SIGILL error, because the
  // precise value of d_out is actually larger than MAX_INT64 (i.e., at such
  // large magnitudes, doubles are actually exact integers, but many fewer
  // integers can be accurately represented, since the double-precision format
  // can only inaccurately approximate them). To prevent this, we try to simply
  // set `out` to the numerical limit when `d_out` is close enough to the
  // numerical limit.
  T out;
  if (d_out_floor >= kTMax) {
    out = kTMax;
  } else if (d_out_floor <= kTLowest) {
    out = kTLowest;
  } else {
    out = static_cast<T>(d_out_floor);
  }

  return SafeOpResult<T>{out, overflow};
}

// Converts double to other floating points. This should be mostly a no-op since
// we are typically only using doubles.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
inline SafeOpResult<T> SafeCastFromDouble(const double in) {
  return SafeOpResult<T>{static_cast<T>(in), false};
}

template <typename T>
inline double Mean(const std::vector<T>& v) {
  if (v.empty()) {
    return 0.0;
  }
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template <typename T>
inline double Variance(const std::vector<T>& v) {
  if (v.empty()) {
    return 0.0;
  }
  double mean = Mean(v);
  double var = 0;
  for (const T& num : v) {
    var += std::pow(num - mean, 2);
  }
  return var / v.size();
}

template <typename T>
inline double StandardDev(const std::vector<T>& v) {
  return std::pow(Variance(v), .5);
}

// Percentile should be between 0 and 1. Does linear interpolation between
// nearest indices.
template <typename T>
inline T OrderStatistic(double percentile, const std::vector<T>& v) {
  std::vector<T> values = std::vector<T>(v);
  std::sort(values.begin(), values.end());
  const int n = values.size();
  if (n == 0) return 0.0;
  const double pos = n * percentile - 0.5;
  if (pos <= 0.0) return values[0];
  if (pos >= n - 1) return values[n - 1];
  const int index = static_cast<const int>(pos);
  const double fraction = pos - index;
  return (1.0 - fraction) * v[index] + fraction * v[index + 1];
}

// Given two numeric vectors of equal length, returns their linear correlation
// coefficient, or NaN if a variance is zero.  Return NaN for unequal length
// vectors as well.
template <typename T>
double Correlation(const std::vector<T>& x, const std::vector<T>& y) {
  int n = x.size();
  if (n < 2 || n != y.size()) {
    return NAN;
  }

  // First get the means.
  T sum_x = 0.0;
  T sum_y = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_x += x[i];
    sum_y += y[i];
  }
  const double mean_x = sum_x / n;
  const double mean_y = sum_y / n;

  // Then the variances and covariance.
  double sum_xx = 0.0;
  double sum_yy = 0.0;
  double sum_xy = 0.0;
  for (int i = 0; i < n; ++i) {
    const double delta_x = x[i] - mean_x;
    const double delta_y = y[i] - mean_y;
    sum_xx += delta_x * delta_x;
    sum_xy += delta_x * delta_y;
    sum_yy += delta_y * delta_y;
  }

  // Return the correlation coefficient, or NaN if variance in x or y is almost
  // 0.0.
  const double error = std::pow(10, -10);
  if (sum_xx > error && sum_yy > error) {
    return sum_xy / std::sqrt(sum_xx * sum_yy);
  } else {
    return NAN;
  }
}

// Filter a vector v using a selection vector. The selection vector has true
// at an index i if that element is selected. Return a vector of only the
// selected elements in v, preserving order.
template <typename T>
std::vector<T> VectorFilter(const std::vector<T>& v,
                            const std::vector<bool>& selection) {
  std::vector<T> result;
  DCHECK(v.size() == selection.size());
  for (int i = 0; i < std::min(v.size(), selection.size()); ++i) {
    if (selection[i]) {
      result.push_back(v[i]);
    }
  }
  return result;
}

// Transform vector into a pretty std::string.
template <typename T>
std::string VectorToString(const std::vector<T>& v) {
  return absl::StrCat("[", absl::StrJoin(v, ", "), "]");
}

// The functions below provide a common and consistent way for validating
// arguments and formatting error messages.

// Returns absl::OkStatus() if the value of optional `opt` if it is set.
// Otherwise, will return an `error_code` error that includes `name` in the
// error message.
absl::Status ValidateIsSet(
    std::optional<double> opt, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// positive. Otherwise, will return an `error_code` error status that includes
// `name` in the error message.
absl::Status ValidateIsPositive(
    std::optional<double> opt, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// non-negative. Otherwise, will return an `error_code` error status that
// includes `name` in the error message.
absl::Status ValidateIsNonNegative(
    std::optional<double> opt, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// finite. Otherwise, will return an `error_code` error status that includes
// `name` in the error message.
absl::Status ValidateIsFinite(
    std::optional<double> opt, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set, finite,
// and positive. Otherwise, will return an `error_code` error status that
// includes `name` in the error message.
absl::Status ValidateIsFiniteAndPositive(
    std::optional<double> opt, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set, finite,
// and non-negative. Otherwise, will return an `error_code` error status that
// includes `name` in the error message.
absl::Status ValidateIsFiniteAndNonNegative(
    std::optional<double> opt, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// within the inclusive (i.e., closed) interval [`lower_bound`, `upper_bound`].
// Otherwise, will return an `error_code` error status that includes `name` in
// the error message.
absl::Status ValidateIsInInclusiveInterval(
    std::optional<double> opt, double lower_bound, double upper_bound,
    absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// within the exclusive (i.e., open) interval (`lower_bound`, `upper_bound`).
// Otherwise, will return an `error_code` error status that includes `name` in
// the error message.
absl::Status ValidateIsInExclusiveInterval(
    std::optional<double> opt, double lower_bound, double upper_bound,
    absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// strictly lesser than `upper_bound`. Otherwise, will return an `error_code`
// error status that includes `name` in the error message.
absl::Status ValidateIsLesserThan(
    std::optional<double> opt, double upper_bound, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// lesser than or equal to `upper_bound`. Otherwise, will return an `error_code`
// error status that includes `name` in the error message.
absl::Status ValidateIsLesserThanOrEqualTo(
    std::optional<double> opt, double upper_bound, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// strictly greater than `lower_bound`. Otherwise, will return an `error_code`
// error status that includes `name` in the error message.
absl::Status ValidateIsGreaterThan(
    std::optional<double> opt, double lower_bound, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// greater than or equal to `upper_bound`. Otherwise, will return an
// `error_code` error status that includes `name` in the error message.
absl::Status ValidateIsGreaterThanOrEqualTo(
    std::optional<double> opt, double lower_bound, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Returns absl::OkStatus() if the value of optional `opt` if it is set and
// within the interval between `lower_bound` and `upper_bound`, including
// `lower_bound` and/or `upper_bound` if `include_lower` or `include_upper` are
// true, respectively. Otherwise, will return an `error_code` error status that
// includes `name` in the error message.
absl::Status ValidateIsInInterval(
    std::optional<double> opt, double lower_bound, double upper_bound,
    bool include_lower, bool include_upper, absl::string_view name,
    absl::StatusCode error_code = absl::StatusCode::kInvalidArgument);

// Methods for semantical and consistent validation of common parameters.
absl::Status ValidateEpsilon(std::optional<double> epsilon);
absl::Status ValidateDelta(std::optional<double> delta);
absl::Status ValidateMaxPartitionsContributed(
    std::optional<double> max_partitions_contributed);
absl::Status ValidateMaxWindows(std::optional<int> max_windows);
absl::Status ValidateMaxContributionsPerPartition(
    std::optional<double> max_contributions_per_partition);
absl::Status ValidateMaxContributions(std::optional<int> max_contributions);

// Validates common tree parameters.
absl::Status ValidateTreeHeight(std::optional<int> tree_height);
absl::Status ValidateBranchingFactor(std::optional<int> branching_factor);

template <typename T>
absl::Status ValidateBounds(std::optional<T> lower, std::optional<T> upper) {
  if (!lower.has_value() && !upper.has_value()) {
    return absl::OkStatus();
  }
  if (lower.has_value() != upper.has_value()) {
    return absl::InvalidArgumentError(
        "Lower and upper bounds must either both be set or both be unset.");
  }
  RETURN_IF_ERROR(ValidateIsFinite(lower.value(), "Lower bound"));
  RETURN_IF_ERROR(ValidateIsFinite(upper.value(), "Upper bound"));
  if (lower.value() > upper.value()) {
    return absl::InvalidArgumentError(
        "Lower bound cannot be greater than upper bound.");
  }
  if (lower.value() == upper.value()) {
    return absl::InvalidArgumentError(
        "Lower bound cannot be equal to upper bound.");
  }
  return absl::OkStatus();
}

absl::Status ValidatePreThresholdOptional(std::optional<int> pre_threshold);

[[deprecated(
    "This validator is used for a class that is deprecated in favour of the "
    "pre_threshold attribute of other strategies classes.")]] absl::Status
ValidatePreThreshold(std::optional<int64_t> pre_threshold);
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_UTIL_H_
