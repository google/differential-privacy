// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DIFFERENTIAL_PRIVACY_ACCOUNTING_CONVOLUTION_H_
#define DIFFERENTIAL_PRIVACY_ACCOUNTING_CONVOLUTION_H_

#include <vector>

#include "absl/types/optional.h"
#include "accounting/common/common.h"

namespace differential_privacy {
namespace accounting {

// An "unpacked" representation of a probability distribution over integers.
// items[i] = p means that the probability mass at min_key + i is equal to p.
struct UnpackedProbabilityMassFunction {
  int min_key;
  std::vector<double> items;
};

// Unpacks probability mass function to vector + min_key
// For example, for {{5, 2.3}, {3, 3.14}, {1, 1.2}}
// returns [1.2, 0, 3.14, 0, 2.3] with min_key = 1
// vector indexes correspond to the keys in the map with zeros if keys are
// missing. In the above example there are no keys for 2 and 4 entries.
UnpackedProbabilityMassFunction UnpackProbabilityMassFunction(
    const ProbabilityMassFunction& input);

// Creates probability mass function from its unpacked form and an additional
// parameter:
//   |tail_mass_truncation|: an upper bound on the tails of the output
//     probability mass that might be truncated.
ProbabilityMassFunction CreateProbabilityMassFunction(
    const UnpackedProbabilityMassFunction& input,
    double tail_mass_truncation = 0);

// Returns probability mass function produced by convolution of two
// others. Additional parameter:
//   |tail_mass_truncation|: an upper bound on the tails of the output
//     probability mass that might be truncated.
ProbabilityMassFunction Convolve(const ProbabilityMassFunction& x,
                                 const ProbabilityMassFunction& y,
                                 double tail_mass_truncation = 0);

// Representation of bounds for truncation in convolution.
struct ConvolutionTruncationBounds {
  int64_t lower_bound;
  int64_t upper_bound;
};

// Returns bounds such that, when convolving the probability mass function with
// itself num_times, the resulting probability mass outside of the range is at
// most |tail_mass_truncation|. Additional parameter:
//   |orders|: the order for moment generating function used to calculate the
//      bounds. If not given, a default value based on only the size of the
//      vector representing the probability mass function is used.
ConvolutionTruncationBounds ComputeConvolutionTruncationBounds(
    const UnpackedProbabilityMassFunction& x, int num_times,
    double tail_mass_truncation = 0,
    std::optional<std::vector<double>> orders = {});

// Returns convolution of probability mass function with itself num_times.
// Additional parameter:
//   |tail_mass_truncation|: an upper bound on the tails of the output
//     probability mass that might be truncated.
ProbabilityMassFunction Convolve(const ProbabilityMassFunction& x,
                                 int num_times,
                                 double tail_mass_truncation = 0);
}  // namespace accounting
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ACCOUNTING_CONVOLUTION_H_
