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

#ifndef DIFFERENTIAL_PRIVACY_TESTING_DENSITY_ESTIMATION_H_
#define DIFFERENTIAL_PRIVACY_TESTING_DENSITY_ESTIMATION_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace differential_privacy {
namespace testing {

// Parameters for pretty printing a histogram.
constexpr int kBoundSpace = 16;
constexpr int kPercentageSpace = 8;
constexpr int kMaxBinBar = 12;
constexpr int kPadding = 8;
constexpr char kNewLine[] = "\n";

// Histogram with fixed width bins. The largest bin counts the number of values
// from the maximum boundary to +infinity.
//
// Example: Make a histogram with the bins   [0, 1), [1, 2), [2, +inf).
//   Histogram<int> hist(0, 1, 3);
//   hist.Add(1);
//   hist.Add(3);
//   hist.BinCount(1).value() == 1 // true
//   hist.BinCount(2).value() == 1 // true
template <typename T>
class Histogram {
 public:
  Histogram(T lowest, double width, int num_bins)
      : lowest_(lowest),
        width_(width),
        bin_counts_(std::vector<int>(num_bins, 0)) {}

  // Increment the count of the bin into which t falls.
  absl::Status Add(T element) {
    double index = (element - lowest_) / width_;
    if (index < 0) {
      return absl::InvalidArgumentError("The element is out of bounds.");
    }
    if (index >= NumBins() - 1) {
      ++bin_counts_[NumBins() - 1];
    } else {
      ++bin_counts_[static_cast<int>(index)];
    }
    return absl::OkStatus();
  }

  // Number of elements in bin index.
  absl::StatusOr<int> BinCount(int index) const {
    if (index < 0 || index >= NumBins()) {
      return absl::InvalidArgumentError("Index is out of bounds.");
    }
    return bin_counts_[index];
  }

  int BinCountOrDie(int index) const {
    absl::StatusOr<int> count = BinCount(index);
    CHECK(count.ok()) << count.status();
    return *count;
  }

  // Number of fixed width bins, including the one to +inf.
  int NumBins() const { return bin_counts_.size(); }

  // Total number of elements in histogram.
  int Total() const {
    return std::accumulate(bin_counts_.begin(), bin_counts_.end(), 0);
  }

  // Maximum count in any bin.
  int MaxBinCount() const {
    return *std::max_element(bin_counts_.begin(), bin_counts_.end());
  }

  // Return the ith bin boundary. Notice that the NumBins()th bin boundary is
  // the upper bound of the highest bin, or +infinity. We use the upper numeric
  // limit to represent this.
  double BinBoundary(int i) const {
    if (i == NumBins()) {
      return std::numeric_limits<T>::max();
    }
    return lowest_ + i * width_;
  }

  // Multi-line pretty print of the histogram displaying non-zero bins.
  std::string ToString() const {
    std::string out;
    int max_count = MaxBinCount();
    int max_count_len = absl::StrCat(max_count).size();
    int total_length = kPadding + 2 * kBoundSpace + max_count_len +
                       kPercentageSpace + kMaxBinBar;
    absl::StrAppend(&out, kNewLine, std::string(total_length, '-'), kNewLine);
    if (max_count == 0) {
      absl::StrAppend(&out, "Empty histogram.", kNewLine);
    } else {
      // For each non-zero bin, we output the bin boundaries, the bin count,
      // the percent bin count, and a bar of variable length representing the
      // percent bin count.
      for (int i = 0; i < NumBins(); ++i) {
        if (bin_counts_[i] == 0) {
          continue;
        }
        std::string lb = BoundToString(i);
        std::string ub = BoundToString(i + 1);
        std::string bin_count = BinCountToString(i, max_count_len);
        std::string percent =
            PercentageToString(bin_counts_[i] * 100.0 / Total());
        int bar_length = std::round(static_cast<double>(bin_counts_[i]) /
                                    max_count * kMaxBinBar);
        absl::StrAppend(&out, "[ ", lb, ", ", ub, ") ", bin_count, " ", percent,
                        "% ", std::string(bar_length, '#'), kNewLine);
      }
    }
    absl::StrAppend(&out, std::string(total_length, '-'));
    return out;
  }

 private:
  // The lowest bin boundary.
  const T lowest_;

  // The width of each bin.
  const double width_;

  // The count in each bin.
  std::vector<int> bin_counts_;

  // Convert the ith bin boundary into std::string with the right format.
  std::string BoundToString(int i) const {
    double boundary = BinBoundary(i);
    if (boundary == std::numeric_limits<T>::max()) {
      return absl::StrCat(std::string(kBoundSpace - 3, ' '), "inf");
    }
    const std::string raw = absl::StrCat(BinBoundary(i));
    if (raw.size() < kBoundSpace) {
      return absl::StrCat(std::string(kBoundSpace - raw.size(), ' '), raw);
    }
    return raw.substr(0, kBoundSpace);
  }

  // Get the ith bin count as a std::string with the right format.
  std::string BinCountToString(int i, int max_count_len) const {
    const std::string raw = absl::StrCat(BinCountOrDie(i));
    if (raw.size() < max_count_len) {
      return absl::StrCat(std::string(max_count_len - raw.size(), ' '), raw);
    }
    return raw;
  }

  // Convert percentage into std::string with the right format.
  std::string PercentageToString(double p) const {
    const std::string with_precision = absl::StrFormat("%.3f", p);
    return absl::StrCat(
        std::string(kPercentageSpace - 1 - with_precision.size(), ' '),
        with_precision);
  }
};

}  // namespace testing
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_TESTING_DENSITY_ESTIMATION_H_
