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

// Defines StochasticTester class and associated utilities.
#ifndef DIFFERENTIAL_PRIVACY_TESTING_STOCHASTIC_TESTER_H_
#define DIFFERENTIAL_PRIVACY_TESTING_STOCHASTIC_TESTER_H_

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <stack>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "algorithms/algorithm.h"
#include "algorithms/util.h"
#include "proto/util.h"
#include "testing/density_estimation.h"
#include "testing/sequence.h"

namespace differential_privacy {
namespace testing {

// The following constants set up the test datasets to be generated on a unit
// hypercube centered on the origin.
constexpr double DefaultDataScale() { return 1.0; }
constexpr double DefaultDataOffset() { return -0.5; }

// Empirically chosen values to allow reasonably stable approximations of the
// distribution while keeping the runtime at most around 15 minutes.
constexpr int DefaultDatasetSize() { return 3; }
constexpr int DefaultNumSamplesPerHistogram() { return 10000; }
constexpr int DefaultNumDatasetsToTest() { return 50; }

constexpr double MinimumRealBinWidth() { return 1e-10; }
constexpr double MinimumIntegralBinWidth() { return 1.0; }
constexpr int MinimumBinCountCombined() { return 2; }
constexpr int MinimumBinCountSingle() { return 1; }

template <typename OutputT>
using AlgorithmResultSamples = std::vector<OutputT>;

using SelectionVector = std::vector<bool>;
using SelectionVectorAndSizePair = std::pair<SelectionVector, size_t>;

// A test framework that tries to prove that an algorithm is not differentially
// private on a number of datasets. The general approach is to generate datasets
// based on a given sequence generator and run the DP algorithm sufficiently
// many times to generate a histogram estimating the probability distribution of
// the output, which is a function of the predicate for differential privacy,
// along with \epsilon.
// Note that an algorithm passing this test does not imply that the algorithm is
// differentially private, but serves to more rigorously test an algorithm by
// providing various inputs.

// This class is templated to support different algorithm output types.
// Therefore, the template parameter should be initialized to be the same as
// that of the output type of the algorithm being tested.
template <typename T, typename OutputT = T,
          typename = std::enable_if_t<std::is_arithmetic<T>::value>>
class StochasticTester;

template <typename T, typename OutputT>
class StochasticTester<T, OutputT> {
 public:
  StochasticTester(
      std::unique_ptr<Algorithm<T>> algorithm,
      std::unique_ptr<Sequence<T>> sequence,
      int64_t num_datasets = DefaultNumDatasetsToTest(),
      int64_t num_samples_per_histogram = DefaultNumSamplesPerHistogram(),
      bool disable_search_branching = false)
      : algorithm_(std::move(algorithm)),
        sequence_(std::move(sequence)),
        num_datasets_(num_datasets),
        num_samples_per_histogram_(num_samples_per_histogram),
        disable_search_branching_(disable_search_branching),
        max_violation_pct_(0.0) {}

  bool Run() {
    Reset();

    // For each dataset, check each member of its powerset for whether it
    // satisfies the dp predicate and record it in class variables. If too
    // many failures are seen, return early.
    const double num_failures_ok = kHistogramPaddingAlpha * num_comparison_;
    for (int i = 0; i < num_datasets_; ++i) {
      std::vector<T> dataset = GenerateDataset();
      CheckDifferentiallyPrivateOnDataset(dataset);
      if (num_comparison_failures_ > num_failures_ok) {
        LOG(INFO)
            << "More than " << kHistogramPaddingAlpha
            << " of comparisons failed so the algorithm is likely not DP.";
        return false;
      }
    }

    LOG(INFO) << "Across all datasets, proportion of comparisons failed: "
              << num_comparison_failures_ << " / " << num_comparison_;
    LOG(INFO) << absl::StrCat(
        "Tested DP over ", num_datasets_,
        " dataset(s). (Maximum violation %: ", max_violation_pct_ * 100, ")");
    return true;
  }

 private:
  struct SelectionVectorHash {
    size_t operator()(const SelectionVector& v) const {
      const std::string serialized_v = absl::StrJoin(v, ".");
      return absl::Hash<std::string>()(serialized_v);
    }
  };

  struct HistogramOptions {
    OutputT lowest;
    double bin_width;
    int num_bins;
  };

  // This computes the optimal bin width as a function of the sample statistics.
  // Namely, the number of samples and order statistics.
  // This is based on the Freedman-Diaconis rule: 2 * \frac{IQR(x)}{n^{1/3}}.
  // See https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule for
  // details.
  // We then determine the number of histogram bins by taking the range of the
  // samples and dividing by the bin size.
  // Ideally this should be applied to each of the histograms to minimize
  // their individual approximation error, but we need the histograms to be
  // comparable, so we take the average of their computed bin widths.
  // When the variation in the data is sufficiently small, the interquartile
  // range is 0.  We fall back to using the range instead here.  If this is
  // also 0, then the distribution of the data is deterministic and we use a
  // small constant bin width and set the number of bins to 1.
  HistogramOptions ComputeHistogramOptions(const std::vector<OutputT>& sample);

  // Computes and returns a HistogramOptions with bin width settings that are
  // suitable (in terms of reducing error) for two sets of samples.  It first
  // computes the optimal bin width and number of bins using
  // ComputeBinWithAndCount above, then takes the average of the bin widths to
  // use as the best bin width for both.  The number of bins is also recomputed
  // based on the newly computed width and the combined range of the samples.
  HistogramOptions ComputeCombinedHistogramOptions(
      const std::vector<OutputT>& sample_lhs,
      const std::vector<OutputT>& sample_rhs);

  // Checks both directions of the DP predicate. This generates histograms based
  // on the samples passed in and compares them based on the DP predicate.
  // We allow for some amount of error that arises from the histogram
  // approximation, thus this still returns true in cases where the predicate
  // is violated within error bounds.
  bool CheckDpPredicate(const std::vector<absl::StatusOr<OutputT>>& dx_samples,
                        const std::vector<absl::StatusOr<OutputT>>& dy_samples);

  // We need to check that all combinations of the input dataset of size 1 to N
  // obey the differential privacy predicate with all datasets that are a
  // distance of 1 away. We cast this as a search problem over a search graph,
  // using DFS with caching. Each state in the search space is represented by a
  // boolean selector vector to select subsets of the dataset. We keep track of
  // the subset sizes directly for efficiency. The DP algorithm is run multiple
  // times to generate a set of samples when successors are generated, which are
  // cached for reuse. Calls CheckDpPredicate for each distance-1 pair to
  // increment counter variables for number of successes of CheckDpPredicate.
  void CheckDifferentiallyPrivateOnDataset(const std::vector<T>& dataset);

  // Given a current selection vector and the number of elements expected in
  // the set of successors, this generates the successors by flipping exactly
  // one "true" value in for each position in selector to false.
  // The expectation here is that succ_selector_size is equal to the number of
  // true values in each selector. Thus, the input selector should have
  // succ_selector_size + 1 true values, since this function generates the
  // successor selectors by flipping exactly 1 true value to false.
  std::vector<SelectionVectorAndSizePair> GenerateSuccessors(
      const SelectionVector& selector, size_t succ_selector_size) const;

  // Runs the DP algorithm over a dataset. In our case, the containers are
  // based on itertools iterators which do not necessarily have size functions,
  // so we provide that interface here.
  template <typename Container>
  std::vector<absl::StatusOr<OutputT>> GenerateSamples(Container* c,
                                                       size_t size) {
    std::vector<absl::StatusOr<OutputT>> samples(num_samples_per_histogram_);
    for (int i = 0; i < num_samples_per_histogram_; ++i) {
      absl::StatusOr<Output> output = algorithm_->Result(c->begin(), c->end());

      // Algorithms such as ApproxBounds may return an error status rather than
      // a value for some datasets on some occasions. In this case we wish to
      // keep testing the dataset instead of throwing it out, treating the error
      // status as a regular value, to be substituted during histogram
      // generation.
      if (output.ok()) {
        samples[i] = GetValue<OutputT>(output.value());
      } else {
        samples[i] = output.status();
      }
    }
    return samples;
  }

  std::vector<T> GenerateDataset() { return sequence_->GetSample(); }

  // Utility function to check a value's relation to boundary_min and
  // boundary_max and collect relative stats about the maximum percent violation
  // over boundary_min relative to the boundary size.
  // For context, we only consider an algorithm as not differentially private
  // if we find an example where the error boundary is also violated
  // (>boundary_max)
  // - boundary_min represents the actual upper bound as stated by the DP
  // predicate
  // - boundary_max includes the histogram approximation error on top of that
  //   upper bound.
  // The return value of this function is true only if the value exceeds
  // boundary_max.
  bool CheckBoundsAndUpdateMaxViolation(double value, double boundary_min,
                                        double boundary_max) {
    if (value <= boundary_min) {
      return false;
    }
    double absolute_violation = value - boundary_min;
    double boundary_size = boundary_max - boundary_min;
    max_violation_pct_ =
        std::max(max_violation_pct_, absolute_violation / boundary_size);
    return value > boundary_max;
  }

  void Reset() {
    max_violation_pct_ = 0.0;
    num_comparison_failures_ = 0;
    num_comparison_ = 0;
    for (const int64_t d : sequence_->NextNDimensions(num_datasets_)) {
      // Without search branching, each subsequence only has 1 child.
      //
      // With search branching, for a sequence of size n, there are (n choose k)
      // unique subsequences of size k. Each of these has k children. Thus there
      // are sum from k={1, 2, ... , n} of (n choose k) * k total comparisons.
      num_comparison_ += disable_search_branching_ ? d : std::pow(2, d - 1) * d;
    }
  }

  // For a pair of vector of samples, find a value that is lower than all
  // non-error values but still close in distance to the distribution of values.
  // Replace all samples that were output error to this error value. Populate
  // the value_samples vectors with replaced values.
  void ReplaceErrorWithValue(
      const std::vector<absl::StatusOr<OutputT>>& dx_samples,
      const std::vector<absl::StatusOr<OutputT>>& dy_samples,
      std::vector<OutputT>* dx_value_samples,
      std::vector<OutputT>* dy_value_samples);

  std::unique_ptr<Algorithm<T>> algorithm_;
  std::unique_ptr<Sequence<T>> sequence_;

  int64_t num_datasets_;
  int64_t num_samples_per_histogram_;

  // This allows control on the amount of the search space to explore by
  // restricting the number of successors generated to 1.
  // It is mainly used to test algorithms that do not depend on the values of
  // the dataset (e.g. Count), where the additional successors are redundant.
  // The default value is false so full search is the standard behavior.
  bool disable_search_branching_;

  // The maximum amount by which any histogram bucket exceeded the differential
  // privacy requirement, expressed as a proportion of the amount the bucket was
  // allowed to exceed the requirement by our error bounds.
  double max_violation_pct_;

  // From Wasserman's All of Nonparametric Statistics, p.130.
  // This is the maximum probability that we get a false negative in any given
  // histograms comparison. 0.05 is chosen arbitrarily.
  static constexpr double kHistogramPaddingAlpha = 0.05;

  // We use these to keep track of what percent of histogram comparisons failed
  // the dp check. We expect at most kHistogramPaddingAlpha of comparisons to
  // fail.
  int64_t num_comparison_ = 0;
  int64_t num_comparison_failures_ = 0;
};

template <typename T, typename OutputT>
typename StochasticTester<T, OutputT>::HistogramOptions
StochasticTester<T, OutputT>::ComputeHistogramOptions(
    const std::vector<OutputT>& sample) {
  HistogramOptions options;
  if (sample.empty()) {
    return options;
  }

  double interquartile_range =
      OrderStatistic(.75, sample) - OrderStatistic(.25, sample);
  // The bin width formula for the rule is 2*IQR / cbrt(n).  When this is zero,
  // we also check if the distribution is actually deterministic by taking
  // max - min.  We use this as an alternative for nearly deterministic
  // distributions to avoid zero bin width.
  double min = *std::min_element(sample.begin(), sample.end());
  double max = *std::max_element(sample.begin(), sample.end());
  double bin_width_numerator =
      interquartile_range > 0 ? 2 * interquartile_range : max - min;
  options.bin_width =
      std::max(bin_width_numerator / cbrt(sample.size()),
               std::is_integral<T>::value ? MinimumIntegralBinWidth()
                                          : MinimumRealBinWidth());
  double num_bins = ceil((max - min) / options.bin_width);
  if (num_bins > std::numeric_limits<int>::max()) {
    num_bins = std::numeric_limits<int>::max();
  }
  options.num_bins =
      std::max(MinimumBinCountSingle(), static_cast<int>(num_bins));
  return options;
}

template <typename T, typename OutputT>
typename StochasticTester<T, OutputT>::HistogramOptions
StochasticTester<T, OutputT>::ComputeCombinedHistogramOptions(
    const std::vector<OutputT>& sample_lhs,
    const std::vector<OutputT>& sample_rhs) {
  HistogramOptions options;

  HistogramOptions options_lhs = ComputeHistogramOptions(sample_lhs);
  HistogramOptions options_rhs = ComputeHistogramOptions(sample_rhs);
  double min_lhs = *std::min_element(sample_lhs.begin(), sample_lhs.end());
  double max_lhs = *std::max_element(sample_lhs.begin(), sample_lhs.end());
  double min_rhs = *std::min_element(sample_rhs.begin(), sample_rhs.end());
  double max_rhs = *std::max_element(sample_rhs.begin(), sample_rhs.end());
  double highest = std::max(max_lhs, max_rhs);
  options.lowest = std::min(min_lhs, min_rhs);

  // We take the average of the two bin widths to use as the combined bin
  // width.  The form of the calculation here is done for overflow protection.
  options.bin_width = (options_lhs.bin_width - options_rhs.bin_width) / 2 +
                      options_rhs.bin_width;

  // For each case below need to specify an extra bucket for the unbounded
  // bucket that goes from the maximum value to +infinity. The first case
  // specially handles when both sets of samples only have 1 bin, which occurs
  // when the samples are deterministic (or near it). In this case,
  // the result should have two bins to provide support for each value (plus
  // one more for the unbounded bin) Otherwise, we calculate the number of
  // bins normally, but still lower bound it to have two bins (and again add
  // the additional unbounded bin).
  if (options_lhs.num_bins == MinimumBinCountSingle() &&
      options_rhs.num_bins == MinimumBinCountSingle()) {
    options.num_bins = MinimumBinCountCombined() + 1;
  } else {
    options.num_bins =
        std::max(MinimumBinCountCombined(),
                 static_cast<int>(
                     ceil((highest - options.lowest) / options.bin_width))) +
        1;
  }
  return options;
}

template <typename T, typename OutputT>
void StochasticTester<T, OutputT>::ReplaceErrorWithValue(
    const std::vector<absl::StatusOr<OutputT>>& dx_samples,
    const std::vector<absl::StatusOr<OutputT>>& dy_samples,
    std::vector<OutputT>* dx_value_samples,
    std::vector<OutputT>* dy_value_samples) {
  // Find the minimum and bin width without error outputs to heuristically chose
  // an error value. If there are no non-error outputs in either sample set, we
  // cannot obtain the stats so default the error value to 0.
  std::vector<OutputT> dx_samples_no_error;
  std::vector<OutputT> dy_samples_no_error;
  for (const absl::StatusOr<OutputT>& e : dx_samples) {
    if (e.ok()) {
      dx_samples_no_error.push_back(e.value());
    }
  }
  for (const absl::StatusOr<OutputT>& e : dy_samples) {
    if (e.ok()) {
      dy_samples_no_error.push_back(e.value());
    }
  }
  OutputT error_value = 0;
  if (!dx_samples_no_error.empty() && !dy_samples_no_error.empty()) {
    HistogramOptions options = ComputeCombinedHistogramOptions(
        dx_samples_no_error, dy_samples_no_error);

    // The error value is the minimum of both samples minus 2x the bin width.
    error_value = options.lowest - 2 * options.bin_width;
  }

  // Copy values in the sample, replacing errors with the error value.
  for (int i = 0; i < dx_value_samples->size(); ++i) {
    if (dx_samples[i].ok()) {
      (*dx_value_samples)[i] = dx_samples[i].value();
    } else {
      (*dx_value_samples)[i] = error_value;
    }
  }
  for (int i = 0; i < dy_value_samples->size(); ++i) {
    if (dy_samples[i].ok()) {
      (*dy_value_samples)[i] = dy_samples[i].value();
    } else {
      (*dy_value_samples)[i] = error_value;
    }
  }
}

template <typename T, typename OutputT>
bool StochasticTester<T, OutputT>::CheckDpPredicate(
    const std::vector<absl::StatusOr<OutputT>>& dx_samples,
    const std::vector<absl::StatusOr<OutputT>>& dy_samples) {
  if (dx_samples.empty() || dy_samples.empty()) {
    return true;
  }
  double epsilon = algorithm_->GetEpsilon();

  // Handle error outputs by replacing them with a default error value. We must
  // replace error values first and include them in the analysis to create the
  // histogram options in order to provide an accurate confidence interval when
  // checking the dp predicate.
  std::vector<OutputT> dx_value_samples(dx_samples.size(), 0);
  std::vector<OutputT> dy_value_samples(dy_samples.size(), 0);
  ReplaceErrorWithValue(dx_samples, dy_samples, &dx_value_samples,
                        &dy_value_samples);
  const HistogramOptions options =
      ComputeCombinedHistogramOptions(dx_value_samples, dy_value_samples);

  // Note that the Histogram class expects double types to be passed into the
  // Add function.  We allow implicit casting here from the templated type
  // where the only non floating point type is an integral type.  The only
  // concern here is if the integral type is large, then the cast to double
  // will lose precision.  In our use cases, the numbers will generally be
  // small and far from this point.
  Histogram<OutputT> dx_hist(options.lowest, options.bin_width,
                             options.num_bins);
  for (const OutputT& e : dx_value_samples) {
    CHECK(dx_hist.Add(e).ok());
  }
  Histogram<OutputT> dy_hist(options.lowest, options.bin_width,
                             options.num_bins);
  for (const OutputT& e : dy_value_samples) {
    CHECK(dy_hist.Add(e).ok());
  }

  // The total number of actual buckets within the bounds is 1 fewer,
  // because there is an extra bucket on the upper extreme to consider values
  // that exceed the boundaries. Note that the error value bucket is also
  // included in the count.
  int actual_num_buckets = options.num_bins - 1;

  // From Wasserman's All of Nonparametric Statistics, p.130.
  // This is the constant used to calculate the 1-\alpha lower and upper
  // confidence interval bounds.
  // c = z_{\alpha/(2m)} * \sqrt{m / n} / 2, where m is the number of bins and
  // n is the number of samples. We choose a 95% confidence interval here,
  // therefore alpha is set to 0.05.
  // This is used to generate an upper bound for each of the histograms.
  // Note that although these intervals are implicitly identical because the
  // number of samples for each set of samples is enforced to be the same in
  // StochasticTester, we don't necessarily make the assumption here and
  // therefore compute an interval for each.
  double dx_size = static_cast<double>(dx_samples.size());
  double dy_size = static_cast<double>(dy_samples.size());
  absl::StatusOr<double> critical_value =
      Qnorm(1 - (kHistogramPaddingAlpha / 2 / actual_num_buckets), /*mu=*/0.0,
            /*sigma=*/1.0);
  CHECK(critical_value.ok()) << critical_value.status();
  double dx_error_interval =
      *critical_value * std::sqrt(actual_num_buckets / dx_size) / 2;
  double dy_error_interval =
      *critical_value * std::sqrt(actual_num_buckets / dy_size) / 2;

  for (int i = 0; i < options.num_bins; ++i) {
    double px = dx_hist.BinCountOrDie(i) / dx_size;
    double py = dy_hist.BinCountOrDie(i) / dy_size;
    double px_differential_privacy_bound = std::exp(epsilon) * px;
    double py_differential_privacy_bound = std::exp(epsilon) * py;

    // The error interval bounds a function over [0, 1] represented by the
    // histogram, which is the probability / (1 / num_bins) at each bucket.
    // Formally the upper bound u = (\sqrt{f} + c)^2 and lower bound l =
    // max(0.0, (\sqrt(f) - c)^2), so in our case, we get f by talking
    // probability * num_bins. To use this bound for probabilities, we
    // divide the result num_bins.
    double px_upper_bound =
        std::pow(std::sqrt(px * actual_num_buckets) + dx_error_interval, 2) /
        actual_num_buckets;
    double py_upper_bound =
        std::pow(std::sqrt(py * actual_num_buckets) + dy_error_interval, 2) /
        actual_num_buckets;
    double px_lower_bound = std::max(
        0.0,
        std::pow(std::sqrt(px * actual_num_buckets) - dx_error_interval, 2) /
            actual_num_buckets);
    double py_lower_bound = std::max(
        0.0,
        std::pow(std::sqrt(py * actual_num_buckets) - dy_error_interval, 2) /
            actual_num_buckets);
    double px_upper_differential_privacy_bound =
        std::exp(epsilon) * px_upper_bound;
    double py_upper_differential_privacy_bound =
        std::exp(epsilon) * py_upper_bound;

    bool bound_exceeded = (dx_hist.BinCountOrDie(i) > 0 &&
                           CheckBoundsAndUpdateMaxViolation(
                               px_lower_bound, py_differential_privacy_bound,
                               py_upper_differential_privacy_bound)) ||
                          (dy_hist.BinCountOrDie(i) > 0 &&
                           CheckBoundsAndUpdateMaxViolation(
                               py_lower_bound, px_differential_privacy_bound,
                               px_upper_differential_privacy_bound));

    // We only report that the predicate is not satisfied if it also exceeds
    // the confidence bounds.
    if (bound_exceeded) {
      LOG(INFO) << "Violation found on histograms ============================";
      LOG(INFO) << dx_hist.ToString();
      LOG(INFO) << dy_hist.ToString();
      LOG(INFO) << absl::StrCat(
          "Bin with violation: ", (i + 1), "/", actual_num_buckets, ", [",
          dx_hist.BinBoundary(i), ", ", dx_hist.BinBoundary(i + 1), "]");
      LOG(INFO) << absl::StrCat("epsilon=", epsilon);
      // The error bin refers to the bin reserved for storing error values.
      LOG(INFO) << absl::StrCat(
          "The bin starting at ", options.lowest,
          " is the number of times the algorithm returned an error.");
      if (px_lower_bound > py_upper_differential_privacy_bound) {
        LOG(INFO) << absl::StrCat("px: ", px);
        LOG(INFO) << absl::StrCat("px_lower_bound: ", px_lower_bound);
        LOG(INFO) << absl::StrCat("py_differential_privacy_bound: ",
                                  py_differential_privacy_bound);
        LOG(INFO) << absl::StrCat("py_upper_differential_privacy_bound: ",
                                  py_upper_differential_privacy_bound);
        LOG(INFO) << absl::StrCat(
            "px_lower_bound > py_upper_differential_privacy_bound: ",
            px_lower_bound, " > ", py_upper_differential_privacy_bound);
        LOG(INFO) << absl::StrCat("px > py_differential_privacy_bound: ", px,
                                  " > ", py_differential_privacy_bound);
      } else {
        LOG(INFO) << absl::StrCat("py: ", py);
        LOG(INFO) << absl::StrCat("py_lower_bound: ", py_lower_bound);
        LOG(INFO) << absl::StrCat("px_differential_privacy_bound: ",
                                  px_differential_privacy_bound);
        LOG(INFO) << absl::StrCat("px_upper_differential_privacy_bound: ",
                                  px_upper_differential_privacy_bound);
        LOG(INFO) << absl::StrCat(
            "py_lower_bound > px_upper_differential_privacy_bound: ",
            py_lower_bound, " > ", px_upper_differential_privacy_bound);
        LOG(INFO) << absl::StrCat("py > px_differential_privacy_bound: ", py,
                                  " > ", px_differential_privacy_bound);
      }
      LOG(INFO) << absl::StrCat("Bounds exceeded by (>100%): ",
                                max_violation_pct_);
      LOG(INFO) << " ";
      return false;
    }
  }
  return true;
}

template <typename T, typename OutputT>
void StochasticTester<T, OutputT>::CheckDifferentiallyPrivateOnDataset(
    const std::vector<T>& dataset) {
  absl::flat_hash_map<SelectionVector,
                      AlgorithmResultSamples<absl::StatusOr<OutputT>>,
                      SelectionVectorHash>
      sample_cache;
  SelectionVector full_set_selector(dataset.size(), true);
  sample_cache[full_set_selector] = GenerateSamples(&dataset, dataset.size());

  std::stack<SelectionVectorAndSizePair> dfs;
  dfs.push(std::make_pair(full_set_selector, dataset.size()));
  while (!dfs.empty()) {
    SelectionVector current_selector = dfs.top().first;
    size_t current_size = dfs.top().second;
    dfs.pop();

    std::vector<SelectionVectorAndSizePair> successors =
        GenerateSuccessors(current_selector, current_size - 1);
    for (const auto& succ_pair : successors) {
      const SelectionVector& succ_selector = succ_pair.first;
      size_t succ_size = succ_pair.second;
      bool is_new_succ = !sample_cache.contains(succ_selector);

      // Generate successor samples if they don't exist.
      if (is_new_succ) {
        std::vector<T> subset;
        for (int i = 0; i < succ_selector.size(); ++i) {
          if (succ_selector[i]) {
            subset.push_back(dataset[i]);
          }
        }
        sample_cache[succ_selector] = GenerateSamples(&subset, succ_size);
      }
      if (!CheckDpPredicate(sample_cache[current_selector],
                            sample_cache[succ_selector])) {
        LOG(INFO) << "Fails DP on: ";
        std::vector<T> c_current = VectorFilter(dataset, current_selector);
        LOG(INFO) << std::setprecision(16) << VectorToString(c_current);
        std::vector<T> c_succ = VectorFilter(dataset, succ_selector);
        LOG(INFO) << std::setprecision(16) << VectorToString(c_succ);
        ++num_comparison_failures_;
      }

      // Only include successors with non-empty subsets and have not been
      // visited.
      if (current_size > 0 && is_new_succ) {
        dfs.push(succ_pair);
      }
    }
  }
}

template <typename T, typename OutputT>
std::vector<SelectionVectorAndSizePair>
StochasticTester<T, OutputT>::GenerateSuccessors(
    const SelectionVector& selector, size_t succ_selector_size) const {
  std::vector<SelectionVectorAndSizePair> successors;
  for (int i = 0; i < selector.size(); ++i) {
    if (selector[i]) {
      successors.emplace_back(
          std::make_pair(SelectionVector(selector), succ_selector_size));
      successors.back().first[i] = false;
      // Done with successor generation if we don't need to branch to any other
      // children.
      if (disable_search_branching_) {
        break;
      }
    }
  }
  return successors;
}

}  // namespace testing
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_TESTING_STOCHASTIC_TESTER_H_
