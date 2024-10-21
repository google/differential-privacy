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
#ifndef DIFFERENTIAL_PRIVACY_ACCOUNTING_CPP_ACCOUNTANT_H_
#define DIFFERENTIAL_PRIVACY_ACCOUNTING_CPP_ACCOUNTANT_H_

#include <functional>

#include "absl/status/statusor.h"
#include "accounting/common/common.h"
#include "accounting/privacy_loss_mechanism.h"
namespace differential_privacy {
namespace accounting {

// NoiseFunction is any function that constructs and returns
// AdditiveNoisePrivacyLoss subclass (for example LaplacePrivacyLoss)
// for the given parameter and sensitivity.
using NoiseFunction = std::function<std::unique_ptr<AdditiveNoisePrivacyLoss>(
    double parameter, double sensitivity)>;

// Computes the smallest required noise parameter for the given privacy
// parameters, privacy loss of additive noise mechanisms (Laplace or Gaussian)
// and also the number of queries.
absl::StatusOr<double> GetSmallestParameter(EpsilonDelta epsilon_delta,
                                            int num_queries, double sensitivity,
                                            NoiseFunction noise_function,
                                            std::optional<double> upper_bound,
                                            double tolerance = 1e-4);

// Uses the optimal advanced composition theorem, Theorem 3.3 from the paper
// Kairouz, Oh, Viswanath. "The Composition Theorem for Differential Privacy"
// to compute the total DP parameters given that we are applying an algorithm
// with given privacy parameters for a given number of times.
//
// Note that we can compute this alternatively from
// {@link PrivacyLossDistribution} by invoking CreateForPrivacyParameters and
// applying the given number of compositions. When setting
// discretization_interval appropriately, these two approaches should coincide
// but using the advanced composition theorem directly is less computationally
// intensive.
//
// Returns total_epsilon such that, when applying the algorithm the given
// number of  times, the result is still (total_epsilon, total_delta)-DP.
// Arguments are as follows.
//   privacy_parameters: The privacy guarantee of a single query.
//   num_queries: Number of times the algorithm is invoked.
//   total_delta: The target value of total delta of the privacy parameters for
//     the multiple runs of the algorithm.
absl::StatusOr<double> AdvancedComposition(EpsilonDelta privacy_parameters,
                                           int num_queries, double total_delta);

}  // namespace accounting
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_ACCOUNTING_CPP_ACCOUNTANT_H_
