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

#ifndef DIFFERENTIAL_PRIVACY_TESTING_SEQUENCE_H_
#define DIFFERENTIAL_PRIVACY_TESTING_SEQUENCE_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/random/distributions.h"

namespace differential_privacy {
namespace testing {

// Abstract class to represent an object that generates coordinates to be used
// as datasets.

template <typename T>
class Sequence {
 public:
  virtual std::vector<T> GetSample() = 0;

  // Returns the dimensions of the next `n` samples.
  virtual std::vector<int64_t> NextNDimensions(int n) = 0;

  virtual ~Sequence() = default;
};

// Returns samples from a pre-defined list passed in by the user on
// construction. The samples repeat when the end of the provided vector is
// reached.
// Note that the pre-defined list is copied into this class on instantiation
// and samples are also returned as copies.  Although this is a potential
// performance bottleneck, we generally expect small enough input lists (the
// product of the vector dimensions is on the order of the 100s) for performance
// issues to be negligible.
template <typename T>
class StoredSequence : public Sequence<T> {
 public:
  explicit StoredSequence(const std::vector<std::vector<T>>& stored_sequence)
      : stored_sequence_(stored_sequence), current_index_(0) {
    DCHECK(!stored_sequence_.empty());
  }
  // In addition to returning an element, this advances the current_index,
  // resetting to 0 if we reach the end of stored_sequence.
  std::vector<T> GetSample() override {
    return stored_sequence_[current_index_++ % stored_sequence_.size()];
  }

  std::vector<int64_t> NextNDimensions(int n) override {
    std::vector<int64_t> dimensions;
    int temp_index = current_index_;
    for (int i = 0; i < n; i++) {
      dimensions.push_back(
          stored_sequence_[temp_index++ % stored_sequence_.size()].size());
    }
    return dimensions;
  }

  StoredSequence() = delete;
  ~StoredSequence() override = default;

 private:
  // The pre-defined list passed in by the user.
  std::vector<std::vector<T>> stored_sequence_;

  // The index of the element of stored_sequence that the next call of
  // GetSample() returns.
  int current_index_;
};

// A HypercubeSequence generates a sequence of coordinates on a unit
// hypercube with a mininum coordinate of 0^d by default.
// The domain can be modified by scaling and shifting (in that order).
template <typename T>
class HypercubeSequence : public Sequence<T> {
 public:
  explicit HypercubeSequence(int64_t dimension, double scale = 1.0,
                             double shift = 0.0)
      : dimension_(dimension), scale_(scale), shift_(shift) {}
  std::vector<T> GetSample() override = 0;

  std::vector<int64_t> NextNDimensions(int n) override {
    return std::vector<int64_t>(n, dimension_);
  }

  // In general, the range of the dataset elements is [offset, scale + offset]
  // since Sequence ranges are transformed by scaling first then applying the
  // offset.
  // The following are convenience functions for consistent initialization of
  // the ranges on Bounded* algorithms.
  double RangeMin() { return shift_; }
  double RangeMax() { return scale_ + shift_; }

  HypercubeSequence() = delete;
  ~HypercubeSequence() override = default;

 protected:
  const int64_t dimension_;
  const double scale_;
  const double shift_;
};

// We store the first 12 primes for creating halton sequences.
static const std::vector<int>& GetFirstPrimes() {
  static const std::vector<int>* const first_primes =
      new std::vector<int>({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37});
  return *first_primes;
}

// Halton sequence generator using only one of the first 12 primes.
class Halton {
 public:
  explicit Halton(const int base) : base_(base) {
    CHECK(std::count(GetFirstPrimes().begin(), GetFirstPrimes().end(), base));
  }

  // Returns the i'th value in the Halton sequence. Time complexity is
  // O(log(i)). We satisfy 0 < Get(i) < 1 for all i > 0.
  double Get(int i) const {
    CHECK_GT(i, 0);
    const double ib = 1.0 / base_;  // ib = inverted base
    double cdb = ib;                // cdb = current digit base = ib ^ position
    double h = 0;
    // Iterate through the base 'base_' digits of 'i'.
    for (; i > 0; i /= base_) {
      h += (i % base_) * cdb;
      cdb *= ib;
    }
    return h;
  }

 private:
  int base_;
};

// Low-discrepancy sequence: generates a determinisitic sequence of uniform
// random points that are spread out evenly.
//
// In order for the points across the dimensions to have near zero correlation,
// the bases used to initialize this object should all be different primes.
// The primality of the bases used is enforced by the Halton generators.
// This basic version of the Halton sequence does not perform well past 14
// dimensions (i.e. the points exhibit structure).
//
// In this implementation, we also provide a flag to reject generated samples
// that are not sorted along the dimension, which is relevant for more diverse
// dataset generation.
template <typename T>
class HaltonSequence : public HypercubeSequence<T> {
 public:
  // Bases must be prime as needed by base::math::Halton. They should also be
  // different.
  // The Halton index starts at 1 for the first non-origin point.
  explicit HaltonSequence(const std::vector<int>& bases,
                          bool sorted_only = false, double scale = 1.0,
                          double shift = 0.0)
      : HypercubeSequence<T>(bases.size(), scale, shift),
        current_index_(1),
        sorted_only_(sorted_only) {
    InitializeHaltonGenerators(bases);
  }

  // CHECK fails if dimension > first_primes_.size()
  explicit HaltonSequence(int64_t dimension, bool sorted_only = false,
                          double scale = 1.0, double shift = 0.0)
      : HypercubeSequence<T>(dimension, scale, shift),
        current_index_(1),
        sorted_only_(sorted_only) {
    CHECK(dimension <= GetFirstPrimes().size());
    std::vector<int> bases(GetFirstPrimes().begin(),
                           GetFirstPrimes().begin() + dimension);
    InitializeHaltonGenerators(bases);
  }
  std::vector<T> GetSample() override {
    std::vector<T> result(HypercubeSequence<T>::dimension_);
    do {
      for (int i = 0; i < HypercubeSequence<T>::dimension_; ++i) {
        result[i] = HypercubeSequence<T>::scale_ *
                        halton_generators_[i]->Get(current_index_) +
                    HypercubeSequence<T>::shift_;
      }
      ++current_index_;
    } while (sorted_only_ && !std::is_sorted(result.begin(), result.end()));
    return result;
  }

  HaltonSequence() = delete;
  ~HaltonSequence() override = default;

 private:
  std::vector<std::unique_ptr<Halton>> halton_generators_;
  int64_t current_index_;
  bool sorted_only_;

  void InitializeHaltonGenerators(const std::vector<int>& bases) {
    CHECK(HypercubeSequence<T>::dimension_ == bases.size());
    halton_generators_.resize(bases.size());
    std::transform(bases.begin(), bases.end(), halton_generators_.begin(),
                   [](int b) { return std::make_unique<Halton>(b); });
  }
};

}  // namespace testing
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_TESTING_SEQUENCE_H_
