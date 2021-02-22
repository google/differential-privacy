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

#ifndef THIRD_PARTY_DIFFERENTIAL_PRIVACY_POSTGRES_DP_FUNC_H
#define THIRD_PARTY_DIFFERENTIAL_PRIVACY_POSTGRES_DP_FUNC_H

#include <inttypes.h>

#include <cmath>
#include <string>
#include <type_traits>

// Forward declare classes from the differential privacy library. We cannot
// include these directly into anon_func.cc.
namespace differential_privacy {

template <typename T>
class Count;

template <typename T>
class BoundedSum;

template <typename T>
class BoundedMean;

template <typename T>
class BoundedVariance;

template <typename T>
class BoundedStandardDeviation;

namespace continuous {

template <typename T>
class Percentile;

}
}  // namespace differential_privacy

// DP functions. Owns an underlying DP algorithm. This wrapping layer is
// neccesary so that C++ dependencies don't conflict with postgres dependencies.
class DpFunc {
 public:
  virtual ~DpFunc() = default;

  // Returns true if adding the entry is successful.
  virtual bool AddEntry(double entry) = 0;
  bool AddEntry(int64_t entry) { return AddEntry(static_cast<double>(entry)); }

  // Result can only be called once per function. Iff grabbing the result fails,
  // the error std::string is populated and we return 0.
  virtual double Result(std::string* err) = 0;

  // Same as result, but the result is rounded to be an integer. Only Result or
  // ResultRounded may be called per function.
  int64_t ResultRounded(std::string* err) { return std::round(Result(err)); }
};

class DpCount : public DpFunc {
 public:
  DpCount(std::string* err, bool default_epsilon = true, double epsilon = 0);
  ~DpCount() override;
  bool AddEntry(double entry) override;
  double Result(std::string* err) override;

 private:
  differential_privacy::Count<double>* count_ = nullptr;
};

class DpSum : public DpFunc {
 public:
  DpSum(std::string* err, bool default_epsilon = true, double epsilon = 0,
        bool auto_bounds = true, double lower = 0, double upper = 0);
  ~DpSum() override;
  bool AddEntry(double entry) override;
  double Result(std::string* err) override;

 private:
  differential_privacy::BoundedSum<double>* sum_ = nullptr;
};

class DpMean : public DpFunc {
 public:
  DpMean(std::string* err, bool default_epsilon = true, double epsilon = 0,
         bool auto_bounds = true, double lower = 0, double upper = 0);
  ~DpMean() override;
  bool AddEntry(double entry) override;
  double Result(std::string* err) override;

 private:
  differential_privacy::BoundedMean<double>* mean_ = nullptr;
};

class DpVariance : public DpFunc {
 public:
  DpVariance(std::string* err, bool default_epsilon = true, double epsilon = 0,
             bool auto_bounds = true, double lower = 0, double upper = 0);
  ~DpVariance() override;
  bool AddEntry(double entry) override;
  double Result(std::string* err) override;

 private:
  differential_privacy::BoundedVariance<double>* var_ = nullptr;
};

class DpStandardDeviation : public DpFunc {
 public:
  DpStandardDeviation(std::string* err, bool default_epsilon = true,
                      double epsilon = 0, bool auto_bounds = true,
                      double lower = 0, double upper = 0);
  ~DpStandardDeviation() override;
  bool AddEntry(double entry) override;
  double Result(std::string* err) override;

 private:
  differential_privacy::BoundedStandardDeviation<double>* sd_ = nullptr;
};

class DpNtile : public DpFunc {
 public:
  // For the ntile function, require bounds because algorithm performs very
  // poorly without them.
  DpNtile(std::string* err, double percentile, double lower, double upper,
          bool default_epsilon = true, double epsilon = 0);
  ~DpNtile() override;
  bool AddEntry(double entry) override;
  double Result(std::string* err) override;

 private:
  differential_privacy::continuous::Percentile<double>* perc_ = nullptr;
};

#endif  // THIRD_PARTY_DIFFERENTIAL_PRIVACY_POSTGRES_DP_FUNC_H
