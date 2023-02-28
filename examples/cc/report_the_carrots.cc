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

#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "proto/util.h"
#include "animals_and_carrots.h"
#include "proto/confidence-interval.pb.h"
#include "proto/data.pb.h"

using absl::PrintF;
using differential_privacy::BoundingReport;
using differential_privacy::ConfidenceInterval;
using differential_privacy::GetValue;
using differential_privacy::Output;
using differential_privacy::example::CarrotReporter;
using ::absl::StatusOr;

ABSL_FLAG(
    std::string, CarrotsDataFile,
    "animals_and_carrots.csv",
    "Path to the datafile where the data is stored on the number of "
    "carrots each animal has eaten.");

int main(int argc, char **argv) {
  PrintF(
      "\nIt is a new day. Farmer Fred is ready to ask the animals about their "
      "carrot consumption.\n");

  // Load the carrot data into the CarrotReporter. We use a higher epsilon to
  // obtain a higher accuracy since our dataset is very small.
  const double epsilon = 4;
  CarrotReporter reporter(absl::GetFlag(FLAGS_CarrotsDataFile), epsilon);

  // Query for the total number of carrots. Notice that we explicitly use 25% of
  // our privacy budget.
  PrintF(
      "\nFarmer Fred asks the animals how many total carrots they have "
      "eaten. The animals know the true sum but report the "
      "differentially private sum to Farmer Fred. But first, they ensure "
      "that Farmer Fred still has privacy budget left.\n");
  PrintF("\nPrivacy budget remaining: %.2f\n", reporter.RemainingEpsilon());
  PrintF("True sum: %d\n", reporter.Sum());
  PrintF("DP sum:   %d\n", GetValue<int>(reporter.PrivateSum(1).value()));

  // Query for the mean with a bounding report.
  PrintF(
      "\nFarmer Fred catches on that the animals are giving him DP results. "
      "He asks for the mean number of carrots eaten, but this time, he wants "
      "some additional accuracy information to build his intuition.\n");
  PrintF("\nPrivacy budget remaining: %.2f\n", reporter.RemainingEpsilon());
  PrintF("True mean: %.2f\n", reporter.Mean());
  StatusOr<Output> mean_status = reporter.PrivateMean(1);
  if (!mean_status.ok()) {
    PrintF("Error obtaining mean: %s\n", mean_status.status().message());
    PrintF(
        "The animals were not able to get the private mean with the current "
        "privacy parameters. This is due to the small size of the dataset and "
        "random chance. Please re-run report_the_carrots to try again.\n");
  } else {
    Output mean_output = mean_status.value();
    BoundingReport report = mean_output.error_report().bounding_report();
    double mean = GetValue<double>(mean_output);
    int lower_bound = GetValue<int>(report.lower_bound());
    int upper_bound = GetValue<int>(report.upper_bound());
    double num_inputs = report.num_inputs();
    double num_outside = report.num_outside();
    PrintF("DP mean output:\n%s\n", mean_output.DebugString());
    PrintF(
        "The animals help Fred interpret the results. %.2f is the DP mean. "
        "Since no bounds were set for  the DP mean algorithm, bounds on the "
        "input data were automatically determined. Most of the data fell "
        "between [%d, %d]. Thus, these bounds were used to determine clamping "
        "and global sensitivity. In addition, around %.0f input values fell "
        "inside of these bounds, and around %.0f inputs fell outside of these "
        "bounds. num_inputs and num_outside are themselves DP counts.\n",
        mean, lower_bound, upper_bound, num_inputs, num_outside);
  }

  // Query for the count with a noise confidence interval.
  {
    PrintF(
        "\nFred wonders how many gluttons are in his zoo. How many animals ate "
        "over 90 carrots? And how accurate is the result?\n");
    PrintF("\nPrivacy budget remaining: %.2f\n", reporter.RemainingEpsilon());
    Output count_output = reporter.PrivateCountAbove(1, 90).value();
    int count = GetValue<int>(count_output);
    ConfidenceInterval ci = GetNoiseConfidenceInterval(count_output);
    double confidence_level = ci.confidence_level();
    double lower_bound = ci.lower_bound();
    double upper_bound = ci.upper_bound();
    PrintF("True count: %d\n", reporter.CountAbove(90));
    PrintF("DP count output:\n%s\n", count_output.DebugString());
    PrintF(
        "The animals tell Fred that %d is the DP count. [%.2f, %.2f] is the "
        "%.2f confidence interval of the noise added to the count.\n",
        count, lower_bound, upper_bound, confidence_level);
  }

  // Query for the maximum.
  PrintF(
      "\n'And how gluttonous is the biggest glutton of them all?' Fred "
      "exclaims. He asks for the maximum number of carrots any animal has "
      "eaten.\n");
  PrintF("\nPrivacy budget remaining: %.2f\n", reporter.RemainingEpsilon());
  PrintF("True max: %d\n", reporter.Max());
  PrintF("DP max:   %3.0f\n", GetValue<double>(reporter.PrivateMax(1).value()));

  // Refuse to query for the count of animals who didn't eat carrots.
  PrintF(
      "\nFred also wonders how many animals are not eating any carrots at "
      "all.\n");
  PrintF("\nPrivacy budget remaining: %.2f\n", reporter.RemainingEpsilon());
  PrintF("Error querying for count: %s\n",
         reporter.PrivateCountAbove(1, 0).status().message());
  PrintF(
      "The animals notice that the privacy budget is depleted. They refuse "
      "to answer any more of Fred's questions for risk of violating "
      "privacy.\n");
}
