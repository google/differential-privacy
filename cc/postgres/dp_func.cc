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

#include "dp_func.h"

#include "algorithms/algorithm.h"
#include "algorithms/bounded-mean.h"
#include "algorithms/bounded-standard-deviation.h"
#include "algorithms/bounded-sum.h"
#include "algorithms/bounded-variance.h"
#include "algorithms/count.h"
#include "algorithms/order-statistics.h"
using differential_privacy::Algorithm;
using differential_privacy::BoundedMean;
using differential_privacy::BoundedStandardDeviation;
using differential_privacy::BoundedSum;
using differential_privacy::BoundedVariance;
using differential_privacy::Count;
using differential_privacy::GetValue;
using differential_privacy::continuous::Percentile;

// Construct and return a bounded algorithm. Populate error if unsuccessful.
template <typename Alg>
Alg* BoundedAlgorithm(std::string* err, bool default_epsilon, double epsilon,
                      bool auto_bounds, double lower, double upper) {
  if (default_epsilon) {
    epsilon = std::log(3);
  }
  typename Alg::Builder builder;
  if (!auto_bounds) {
    builder.SetLower(lower).SetUpper(upper);
  }
  auto build_statusor = builder.SetEpsilon(epsilon).Build();
  if (build_statusor.ok()) {
    return build_statusor.value().release();
  }
  *err = std::string(build_statusor.status().message());
  return nullptr;
}

// Delete a bounded algorithm, if it exists.
template <typename Alg>
void DeleteAlgorithm(Alg* alg) {
  if (alg) {
    delete alg;
  }
}

// Add an entry to the algorithm. Return true if the algorithm exists.
bool AlgorithmAddEntry(Algorithm<double>* alg, double entry) {
  if (alg) {
    alg->AddEntry(entry);
    return true;
  }
  return false;
}

// Return the result of return_type from the algorithm, populating error if
// unsuccessful.
template <typename return_type>
double AlgorithmResult(Algorithm<double>* alg, std::string* err) {
  double default_return = 0.0;
  if (!alg) {
    *err = "Underlying algorithm was never constructed.";
    return default_return;
  }
  auto result_statusor = alg->PartialResult();
  if (result_statusor.ok()) {
    return static_cast<double>(GetValue<return_type>(result_statusor.value()));
  } else {
    *err = std::string(result_statusor.status().message());
  }
  return default_return;
}

// DP count.
DpCount::DpCount(std::string* err, bool default_epsilon, double epsilon) {
  if (default_epsilon) {
    epsilon = std::log(3);
  }
  auto count_statusor = Count<double>::Builder().SetEpsilon(epsilon).Build();
  if (count_statusor.ok()) {
    count_ = count_statusor.value().release();
  } else {
    *err = std::string(count_statusor.status().message());
  }
}
DpCount::~DpCount() { DeleteAlgorithm<Count<double>>(count_); }
bool DpCount::AddEntry(double entry) {
  return AlgorithmAddEntry(count_, entry);
}
double DpCount::Result(std::string* err) {
  return AlgorithmResult<int64_t>(count_, err);
}

// DP sum.
DpSum::DpSum(std::string* err, bool default_epsilon, double epsilon,
             bool auto_bounds, double lower, double upper) {
  sum_ = BoundedAlgorithm<BoundedSum<double>>(err, default_epsilon, epsilon,
                                              auto_bounds, lower, upper);
}
DpSum::~DpSum() { DeleteAlgorithm<BoundedSum<double>>(sum_); }
bool DpSum::AddEntry(double entry) { return AlgorithmAddEntry(sum_, entry); }
double DpSum::Result(std::string* err) {
  return AlgorithmResult<double>(sum_, err);
}

// DP mean.
DpMean::DpMean(std::string* err, bool default_epsilon, double epsilon,
               bool auto_bounds, double lower, double upper) {
  mean_ = BoundedAlgorithm<BoundedMean<double>>(err, default_epsilon, epsilon,
                                                auto_bounds, lower, upper);
}
DpMean::~DpMean() { DeleteAlgorithm<BoundedMean<double>>(mean_); }
bool DpMean::AddEntry(double entry) { return AlgorithmAddEntry(mean_, entry); }
double DpMean::Result(std::string* err) {
  return AlgorithmResult<double>(mean_, err);
}

// DP variance.
DpVariance::DpVariance(std::string* err, bool default_epsilon, double epsilon,
                       bool auto_bounds, double lower, double upper) {
  var_ = BoundedAlgorithm<BoundedVariance<double>>(
      err, default_epsilon, epsilon, auto_bounds, lower, upper);
}
DpVariance::~DpVariance() { DeleteAlgorithm<BoundedVariance<double>>(var_); }
bool DpVariance::AddEntry(double entry) {
  return AlgorithmAddEntry(var_, entry);
}
double DpVariance::Result(std::string* err) {
  return AlgorithmResult<double>(var_, err);
}

// DP standard deviation.
DpStandardDeviation::DpStandardDeviation(std::string* err, bool default_epsilon,
                                         double epsilon, bool auto_bounds,
                                         double lower, double upper) {
  sd_ = BoundedAlgorithm<BoundedStandardDeviation<double>>(
      err, default_epsilon, epsilon, auto_bounds, lower, upper);
}
DpStandardDeviation::~DpStandardDeviation() {
  DeleteAlgorithm<BoundedStandardDeviation<double>>(sd_);
}
bool DpStandardDeviation::AddEntry(double entry) {
  return AlgorithmAddEntry(sd_, entry);
}
double DpStandardDeviation::Result(std::string* err) {
  return AlgorithmResult<double>(sd_, err);
}

// DP Ntile.
DpNtile::DpNtile(std::string* err, double percentile, double lower,
                 double upper, bool default_epsilon, double epsilon) {
  if (default_epsilon) {
    epsilon = std::log(3);
  }
  auto build_statusor = Percentile<double>::Builder()
                            .SetPercentile(percentile)
                            .SetEpsilon(epsilon)
                            .SetLower(lower)
                            .SetUpper(upper)
                            .Build();
  if (build_statusor.ok()) {
    perc_ = build_statusor.value().release();
  } else {
    *err = std::string(build_statusor.status().message());
  }
}
DpNtile::~DpNtile() { DeleteAlgorithm<Percentile<double>>(perc_); }
bool DpNtile::AddEntry(double entry) { return AlgorithmAddEntry(perc_, entry); }
double DpNtile::Result(std::string* err) {
  return AlgorithmResult<double>(perc_, err);
}
