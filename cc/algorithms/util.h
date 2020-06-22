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
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>

#include "base/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "base/statusor.h"

namespace differential_privacy {

// XORs two strings together character by character. If one std::string is shorter
// than the other it will be repeated until it is the same length as the longer
// std::string before being XORed. If either std::string is empty, the other will be
// returned.
std::string XorStrings(const std::string& longer, const std::string& shorter);

// Arbitrary default value for epsilon. The algorithm interface falls back on
// this value whenever one is not provided. This value should only be used for
// testing convenience. For any production use case, please set your own epsilon
// based on privacy needs.
double DefaultEpsilon();

// Returns the smallest power of 2 greater than or equal to n. n must be > 0.
// Includes negative powers.
double GetNextPowerOfTwo(double n);

// Rounds n to the nearest multiple of base. Ties are broken towards +inf.
// If base is 0, returns n.
double RoundToNearestMultiple(double n, double base);

// Return 1.0 if n > 0, -1.0 if n < 0, and 0 if n == 0.
double sign(double n);

// Approximate the inverse of the error function.
// Implementation based on Table 5 in Giles' paper
// on approximating the inverse of the error function
// (https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf).
double InverseErrorFunction(double x);

// Estimation of the inverse cdf of the normal distribution centered at mu with
// standard deviation sigma, at probability p. Based on Abramowitz and Stegun
// formula 26.2.23. The error of the estimation is bounded by 4.5 e-4. This
// function will fail if higher accuracy is required.
base::StatusOr<double> Qnorm(double p, double mu = 0.0, double sigma = 1.0);

template <typename T>
inline const T& Clamp(const T& low, const T& high, const T& value) {
  // Prevents errors in ordering the arguments.
  DCHECK(!(high < low));
  if (high < value) return high;
  if (value < low) return low;
  return value;
}

// Return true and assign the addition result if the addition will not overflow.
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline bool SafeAdd(T lhs, T rhs, T* result) {
  if (lhs > 0) {
    // For negative rhs, we will never overflow.
    if (rhs > 0) {
      T safe_distance = std::numeric_limits<T>::max() - lhs;
      if (safe_distance < rhs) return false;
    }
  } else if (lhs < 0) {
    // For positive rhs, we will never overflow.
    if (rhs < 0) {
      T safe_distance = std::numeric_limits<T>::lowest() - lhs;
      if (safe_distance > rhs) return false;
    }
  }
  *result = lhs + rhs;
  return true;
}

// Return true and assign the subtraction result if the subtraction will not
// overflow.
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline bool SafeSubtract(T lhs, T rhs, T* result) {
  // For integral values the min numeric limit is larger in magnitude than the
  // max numeric limit, so we cannot negate it. For unsigned types, the lowest
  // numeric limit is 0. FOr signed types, it is negative.
  if (rhs == std::numeric_limits<T>::lowest() && rhs != 0) {
    if (lhs > 0) {
      return false;
    } else {
      *result = lhs - rhs;
      return true;
    }
  }

  // For all other values of rhs, add the negation.
  return SafeAdd(lhs, -rhs, result);
}

// Return true and assign the square result if squaring will not overflow.
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline bool SafeSquare(T num, T* result) {
  double max_root = std::pow(std::numeric_limits<T>::max(), 0.5);
  if (num > 0 && num > static_cast<T>(max_root)) return false;
  if (num < 0 && num < -1 * static_cast<T>(max_root)) return false;
  *result = num * num;
  return true;
}

template <typename T>
inline double Mean(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template <typename T>
inline double Variance(const std::vector<T>& v) {
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
                            const std::vector<bool> selection) {
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

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ALGORITHMS_UTIL_H_
