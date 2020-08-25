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

#include "animals_and_carrots.h"

#include <fstream>

#include "base/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "algorithms/bounded-mean.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/count.h"
#include "algorithms/order-statistics.h"

namespace differential_privacy {
namespace example {

CarrotReporter::CarrotReporter(std::string data_filename, double epsilon)
    : epsilon_(epsilon) {
  std::ifstream file(data_filename);
  CHECK(file.is_open()) << "could not open file " << data_filename;
  std::string line;
  while (getline(file, line)) {
    std::vector<std::string> animal_and_count = absl::StrSplit(line, ',');
    CHECK_EQ(animal_and_count.size(), 2);
    int count;
    CHECK(absl::SimpleAtoi(animal_and_count[1], &count));
    carrots_per_animal_[animal_and_count[0]] = count;
  }
}

int CarrotReporter::Sum() {
  int sum = 0;
  for (const auto& pair : carrots_per_animal_) {
    sum += pair.second;
  }
  return sum;
}

double CarrotReporter::Mean() {
  return static_cast<double>(Sum()) / carrots_per_animal_.size();
}

int CarrotReporter::CountAbove(int limit) {
  int count = 0;
  for (const auto& pair : carrots_per_animal_) {
    if (pair.second > limit) {
      ++count;
    }
  }
  return count;
}

int CarrotReporter::Max() {
  int max = 0;
  for (const auto& pair : carrots_per_animal_) {
    max = std::max(pair.second, max);
  }
  return max;
}

double CarrotReporter::PrivacyBudget() { return privacy_budget_; }

base::StatusOr<Output> CarrotReporter::PrivateSum(double privacy_budget) {
  if (privacy_budget_ < privacy_budget) {
    return base::InvalidArgumentError("Not enough privacy budget.");
  }
  privacy_budget_ -= privacy_budget;
  ASSIGN_OR_RETURN(std::unique_ptr<BoundedSum<int>> sum_algorithm,
                   BoundedSum<int>::Builder()
                       .SetEpsilon(epsilon_)
                       .SetLower(0)
                       .SetUpper(150)
                       .Build());
  for (const auto& pair : carrots_per_animal_) {
    sum_algorithm->AddEntry(pair.second);
  }
  return sum_algorithm->PartialResult(privacy_budget);
}

base::StatusOr<Output> CarrotReporter::PrivateMean(double privacy_budget) {
  if (privacy_budget_ < privacy_budget) {
    return base::InvalidArgumentError("Not enough privacy budget.");
  }
  privacy_budget_ -= privacy_budget;
  ASSIGN_OR_RETURN(std::unique_ptr<BoundedMean<int>> mean_algorithm,
                   BoundedMean<int>::Builder().SetEpsilon(epsilon_).Build());
  for (const auto& pair : carrots_per_animal_) {
    mean_algorithm->AddEntry(pair.second);
  }
  return mean_algorithm->PartialResult(privacy_budget);
}

base::StatusOr<Output> CarrotReporter::PrivateCountAbove(double privacy_budget,
                                                         int limit) {
  if (privacy_budget_ < privacy_budget) {
    return base::InvalidArgumentError("Not enough privacy budget.");
  }
  privacy_budget_ -= privacy_budget;
  ASSIGN_OR_RETURN(std::unique_ptr<Count<std::string>> count_algorithm,
                   Count<std::string>::Builder().SetEpsilon(epsilon_).Build());

  for (const auto& pair : carrots_per_animal_) {
    if (pair.second > limit) {
      count_algorithm->AddEntry(pair.first);
    }
  }
  return count_algorithm->PartialResult(privacy_budget);
}

base::StatusOr<Output> CarrotReporter::PrivateMax(double privacy_budget) {
  if (privacy_budget_ < privacy_budget) {
    return base::InvalidArgumentError("Not enough privacy budget.");
  }
  privacy_budget_ -= privacy_budget;
  ASSIGN_OR_RETURN(std::unique_ptr<continuous::Max<int>> max_algorithm,
                   continuous::Max<int>::Builder()
                       .SetEpsilon(epsilon_)
                       .SetLower(0)
                       .SetUpper(150)
                       .Build());
  for (const auto& pair : carrots_per_animal_) {
    max_algorithm->AddEntry(pair.second);
  }
  return max_algorithm->PartialResult(privacy_budget);
}

}  // namespace example
}  // namespace differential_privacy
