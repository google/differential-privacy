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

#include "algorithms/util.h"

#include <cmath>
#include <vector>

#include "base/canonical_errors.h"

namespace differential_privacy {

std::string XorStrings(const std::string& longer, const std::string& shorter) {
  if (shorter.size() > longer.size()) {
    return XorStrings(shorter, longer);
  }
  if (shorter.empty()) {
    return longer;
  }
  std::string to_return = longer;
  std::string repeated_shorter = shorter;
  while (repeated_shorter.size() < to_return.size()) {
    repeated_shorter.append(shorter);
  }
  std::transform(longer.begin(), longer.end(), repeated_shorter.begin(),
                 to_return.begin(), std::bit_xor<char>());
  return to_return;
}

double DefaultEpsilon() { return std::log(3); }

double GetNextPowerOfTwo(double n) { return pow(2.0, ceil(log2(n))); }

double InverseErrorFunction(double x) {
  double LESS_THAN_FIVE_CONSTANTS[] = {
      0.0000000281022636, 0.000000343273939, -0.0000035233877,
      -0.00000439150654,  0.00021858087,     -0.00125372503,
      -0.00417768164,     0.246640727,       1.50140941};
  double GREATER_THAN_FIVE_CONSTANTS[] = {
      -0.000200214257, 0.000100950558, 0.00134934322,
      -0.00367342844,  0.00573950773,  -0.0076224613,
      0.00943887047,   1.00167406,     2.83297682};

  double constantArray[9];
  double w = -std::log((1 - x) * (1 + x));
  double ans = 0;

  if (std::abs(x) == 1) {
    return x * std::numeric_limits<double>::infinity();
  }

  if (w < 5) {
    w = w - 2.5;
    std::copy(std::begin(LESS_THAN_FIVE_CONSTANTS),
              std::end(LESS_THAN_FIVE_CONSTANTS), std::begin(constantArray));
  } else {
    w = std::sqrt(w) - 3;
    std::copy(std::begin(GREATER_THAN_FIVE_CONSTANTS),
              std::end(GREATER_THAN_FIVE_CONSTANTS), std::begin(constantArray));
  }

  for (int i = 0; i < 9; i++) {
    double coefficient = constantArray[i];
    ans = coefficient + ans * w;
  }

  return ans * x;
}

base::StatusOr<double> Qnorm(double p, double mu, double sigma) {
  if (p <= 0.0 || p >= 1.0) {
    return base::InvalidArgumentError(
        "Probability must be between 0 and 1, exclusive.");
  }
  double t = std::sqrt(-2.0 * log(std::min(p, 1.0 - p)));
  std::vector<double> c = {2.515517, 0.802853, 0.010328};
  std::vector<double> d = {1.432788, 0.189269, 0.001308};
  double normalized = t - ((c[2] * t + c[1]) * t + c[0]) /
                              (((d[2] * t + d[1]) * t + d[0]) * t + 1.0);
  if (p < .5) {
    normalized *= -1;
  }
  return normalized * sigma + mu;
}

double RoundToNearestMultiple(double n, double base) {
  if (base == 0.0) return n;
  double remainder = fmod(n, base);
  if (fabs(remainder) > base / 2) {
    return n - remainder + sign(remainder) * base;
  }
  if (fabs(remainder) == base / 2) {
    return n + base / 2;
  }
  return n - remainder;
}

double sign(double n) {
  if (n > 0.0) return 1.0;
  if (n < 0.0) return -1.0;
  return 0;
}

}  // namespace differential_privacy
