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

#ifndef DIFFERENTIAL_PRIVACY_EXAMPLE_ANIMALS_AND_CARROTS_H_
#define DIFFERENTIAL_PRIVACY_EXAMPLE_ANIMALS_AND_CARROTS_H_

#include <map>
#include <string>

#include "absl/status/statusor.h"
#include "proto/data.pb.h"

namespace differential_privacy {
namespace example {

// The CarrotReporter class helps the animals report differentially private (DP)
// aggregate statistics about the number of carrots they have eaten to Farmer
// Fred.
class CarrotReporter {
 public:
  // Loads all the animals and carrots data from the file specified by
  // data_filename into the map carrots_per_animal_. Epsilon is the differential
  // privacy parameter. Epsilon is shared between all private function calls.
  // The fraction of epsilon remaining is tracked by remaining_epsilon_.
  CarrotReporter(std::string data_filename, double total_epsilon);

  // True sum of all the carrots eaten.
  int Sum();

  // True mean of carrots eaten.
  double Mean();

  // True count of the number of animals who ate more than "limit" carrots.
  int CountAbove(int limit);

  // True maximum of the number of carrots eaten by any one animal.
  int Max();

  // Returns the remaining epsilon. Animals should check this to see if
  // they should answer any more of Farmer Fred's questions.
  double RemainingEpsilon();

  // DP sum of all the carrots eaten. Consumes `epsilon` amount of budget.
  // Returns an error is `epsilon` is more than the total remaining epsilon
  // assigned to the CarrotReporter.
  absl::StatusOr<Output> PrivateSum(double epsilon);

  // DP mean of all carrots eaten. Consumes `epsilon` amount of budget. Returns
  // an error is `epsilon` is more than the total remaining epsilon assigned to
  // the CarrotReporter.
  absl::StatusOr<Output> PrivateMean(double epsilon);

  // DP count of the number of animals who ate more than "limit" carrots.
  // Consumes `epsilon` amount of budget. Returns an error is `epsilon` is more
  // than the total remaining epsilon assigned to the CarrotReporter.
  absl::StatusOr<Output> PrivateCountAbove(double epsilon, int limit);

  // DP maximum of the number of carrots eaten by any one animal. Consumes
  // `epsilon` amount of budget. Returns an error is `epsilon` is more than the
  // total remaining epsilon assigned to the CarrotReporter.
  absl::StatusOr<Output> PrivateMax(double epsilon);

 private:
  // Map from the animal name to the number of carrots eaten by that animal.
  std::map<std::string, int> carrots_per_animal_;

  // Differential privacy parameter epsilon. A larger epsilon corresponds to
  // less privacy and more accuracy.
  double total_epsilon_;

  // The privacy budget given to Farmer Fred at the beginning of the day. If
  // this budget depletes, Farmer Fred cannot ask anymore questions about the
  // carrot data. Privacy budget is represented as a fraction on [0, 1].
  double remaining_epsilon_;
};

}  // namespace example
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_EXAMPLE_ANIMALS_AND_CARROTS_H_
