// Copyright 2021 Google LLC
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

#include <cstdlib>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "accounting/privacy_loss_distribution.h"
#include "accounting/privacy_loss_mechanism.h"

using absl::PrintF;
using differential_privacy::accounting::GaussianPrivacyLoss;
using differential_privacy::accounting::LaplacePrivacyLoss;
using differential_privacy::accounting::PrivacyLossDistribution;
using ::absl::StatusOr;

int main(int argc, char **argv) {
  constexpr double parameter_laplace = 3;
  constexpr double sensitivity = 1;

  // Create Laplace privacy loss.
  StatusOr<std::unique_ptr<LaplacePrivacyLoss>> laplace_privacy_loss;
  laplace_privacy_loss =
      LaplacePrivacyLoss::Create(parameter_laplace, sensitivity);
  CHECK_EQ(laplace_privacy_loss.status(), absl::OkStatus());

  // Create Privacy Loss Distribution (PLD) for this Laplace privacy loss.
  std::unique_ptr<PrivacyLossDistribution> pld =
      PrivacyLossDistribution::CreateForAdditiveNoise(
          *laplace_privacy_loss.value());

  // Compose it to itself to simulate running multiple queries.
  constexpr int num_laplace = 40;
  pld->Compose(num_laplace);

  // Determine delta for this PLD.
  constexpr double epsilon = 10;
  double delta = pld->GetDeltaForEpsilon(epsilon);

  PrintF(
      "An algorithm that executes the Laplace Mechanism with parameter"
      " %.0lf for a total of %d times is (%.0lf, %.8lf)-DP.\n",
      parameter_laplace, num_laplace, epsilon, delta);

  // PLDs for different mechanisms can also be composed. Below is an example in
  // which we compose PLDs for Laplace Mechanism and Gaussian Mechanism.
  constexpr double standard_deviation = 5;
  StatusOr<std::unique_ptr<GaussianPrivacyLoss>> gaussian_privacy_loss =
      GaussianPrivacyLoss::Create(standard_deviation, sensitivity);
  CHECK_EQ(gaussian_privacy_loss.status(), absl::OkStatus());

  std::unique_ptr<PrivacyLossDistribution> gaussian_pld =
      PrivacyLossDistribution::CreateForAdditiveNoise(
          *gaussian_privacy_loss.value());

  // Compose Gaussian PLD into Laplace.
  CHECK_EQ(pld->Compose(*gaussian_pld), absl::OkStatus());

  // Determine delta after composition.
  delta = pld->GetDeltaForEpsilon(epsilon);

  PrintF(
      "An algorithm that executes the Laplace Mechanism with parameter"
      " %.0lf for a total of %d times and in addition executes once the"
      " Gaussian Mechanism with STD %.0lf is (%.0lf, %.8lf)-DP.\n",
      parameter_laplace, num_laplace, standard_deviation, epsilon, delta);

  return EXIT_SUCCESS;
}
