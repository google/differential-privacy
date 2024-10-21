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
#include "accounting/accountant.h"

#include <cmath>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "boost/math/special_functions/binomial.hpp"
#include "accounting/common/common.h"
#include "accounting/privacy_loss_distribution.h"

namespace differential_privacy {
namespace accounting {

absl::StatusOr<double> GetSmallestParameter(EpsilonDelta epsilon_delta,
                                            int num_queries, double sensitivity,
                                            NoiseFunction noise_function,
                                            std::optional<double> upper_bound,
                                            double tolerance) {
  BinarySearchParameters search_parameters = {
      .lower_bound = 0,
      .upper_bound = upper_bound.has_value() ? upper_bound.value()
                                             : 2 * num_queries * sensitivity /
                                                   epsilon_delta.epsilon,
      .initial_guess = std::nullopt,
      .tolerance = tolerance};

  auto compute_delta = [noise_function, sensitivity, num_queries,
                        epsilon_delta](double parameter) {
    std::unique_ptr<PrivacyLossDistribution> pld =
        PrivacyLossDistribution::CreateForAdditiveNoise(
            *noise_function(parameter, sensitivity), EstimateType::kPessimistic,
            5e-5);
    pld->Compose(num_queries,
                 /*tail_mass_truncation=*/0.01 * epsilon_delta.delta);
    return pld->GetDeltaForEpsilon(epsilon_delta.epsilon);
  };

  return InverseMonotoneFunction(compute_delta, epsilon_delta.delta,
                                 search_parameters);
}

absl::StatusOr<double> AdvancedComposition(
    const EpsilonDelta privacy_parameters, const int num_queries,
    const double total_delta) {
  double epsilon = privacy_parameters.epsilon;
  double delta = privacy_parameters.delta;
  int k = num_queries;

  // The calculation follows Theorem 3.3 of https://arxiv.org/pdf/1311.0776.pdf.
  for (int i = k / 2; i >= 0; --i) {
    double delta_i = 0;
    for (int l = 0; l < i; l++) {
      delta_i +=
          boost::math::binomial_coefficient<double>(k, l) *
          (std::exp(epsilon * (k - l)) - std::exp(epsilon * (k - 2 * i + l)));
    }
    delta_i /= std::pow(1 + std::exp(epsilon), k);
    if (1 - std::pow((1 - delta), k) * (1 - delta_i) <= total_delta) {
      return epsilon * (k - 2 * i);
    }
  }

  return absl::NotFoundError(
      absl::StrFormat("Total delta %lf is too small for %d queries each with "
                      "delta equal to %lf.",
                      total_delta, k, delta));
}
}  // namespace accounting
}  // namespace differential_privacy
