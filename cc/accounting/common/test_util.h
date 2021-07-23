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

#ifndef DIFFERENTIAL_PRIVACY_CPP_ACCOUNTING_COMMON_TEST_UTIL_H_
#define DIFFERENTIAL_PRIVACY_CPP_ACCOUNTING_COMMON_TEST_UTIL_H_

#include "gmock/gmock.h"
#include "accounting/common/common.h"

namespace differential_privacy {
namespace accounting {

// Matches a given PMF if it is nearly equal to expected. Probabilities masses
// are allowed to be within max_error. (This also means that unexpected outcomes
// can be in PMF as long as its mass is no more than max_error.)
::testing::Matcher<ProbabilityMassFunction> PMFIsNear(
    const ProbabilityMassFunction& expected_pmf, const double max_error);
}  // namespace accounting
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_CPP_ACCOUNTING_COMMON_TEST_UTIL_H_
