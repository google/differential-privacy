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

#include "accounting/common/test_util.h"

namespace differential_privacy {
namespace accounting {

using ::testing::Matcher;

template <typename T>
class ProbabilityMassFunctionMatcher {
 public:
  using is_gtest_matcher = void;

  bool MatchAndExplain(ProbabilityMassFunctionOf<T> pmf,
                       std::ostream* os) const {
    for (auto [outcome, probability_mass] : pmf) {
      auto it = expected_pmf_.find(outcome);
      double expected_probability_mass =
          it == expected_pmf_.end() ? 0 : it->second;
      if (std::abs(probability_mass - expected_probability_mass) > max_error_) {
        if (os != nullptr) {
          *os << outcome << "has mass " << probability_mass
              << " but expected mass is " << expected_probability_mass;
        }
        return false;
      }
    }
    for (auto [outcome, expected_probability_mass] : expected_pmf_) {
      auto it = pmf.find(outcome);
      double probability_mass = it == pmf.end() ? 0 : it->second;
      if (std::abs(probability_mass - expected_probability_mass) > max_error_) {
        if (os != nullptr) {
          *os << outcome << "has mass " << probability_mass
              << " but expected mass is " << expected_probability_mass;
        }
        return false;
      }
    }
    return true;
  }

  void DescribeTo(std::ostream* os) const {
    *os << "is near ";
    ::testing::internal::UniversalPrint(expected_pmf_, os);
    *os << " within error " << max_error_;
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "is not near ";
    ::testing::internal::UniversalPrint(expected_pmf_, os);
    *os << " within error " << max_error_;
  }

  ProbabilityMassFunctionMatcher(
      const ProbabilityMassFunctionOf<T>& expected_pmf, double max_error)
      : expected_pmf_(expected_pmf), max_error_(max_error) {}

  const ProbabilityMassFunctionOf<T>& expected_pmf_;
  double max_error_;
};

Matcher<ProbabilityMassFunction> PMFIsNear(
    const ProbabilityMassFunction& expected_pmf, const double max_error) {
  return ProbabilityMassFunctionMatcher<int>(expected_pmf, max_error);
}
}  // namespace accounting
}  // namespace differential_privacy
