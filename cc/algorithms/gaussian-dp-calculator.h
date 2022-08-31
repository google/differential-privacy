//
// Copyright 2022 Google LLC
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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_GAUSSIAN_DP_CALCULATOR_H_
#define DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_GAUSSIAN_DP_CALCULATOR_H_

namespace differential_privacy {

// Returns the delta for a Gaussian mechanism with the given standard deviation
// and the given epsilon and l2 sensitivity.
double CalculateDeltaForGaussianStddev(double epsilon, double l2_sensitivity,
                                       double stddev);

}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ALGORITHMS_GAUSSIAN_DP_CALCULATOR_H_
