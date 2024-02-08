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

#ifndef DIFFERENTIAL_PRIVACY_BASE_PERCENTILE_H_
#define DIFFERENTIAL_PRIVACY_BASE_PERCENTILE_H_

#include <cmath>
#include <cstdint>

#include "google/protobuf/repeated_field.h"
#include "proto/util.h"

namespace differential_privacy {
namespace base {

// Percentile contains an underlying vector that stores an input set. Percentile
// retrieves the relative rank of a value with respect to the input set.
//
// Returned values are in [0, 1]. If the value is in the input set, returns:
//    (rank of nearest lesser value, maximum rank of value).
// If the item is not in the set, returns the upper bound rank of the nearest
// lesser item as both values. This is the same value as the lower bound rank
// of the nearest greater item. Ex: Using items with values { 1, 2, 2, 3, 5},
//     Then GetRelativeRank(2) == (.2, .6).
//          GetRelativeRank(3) == (.6, .8).
//          GetRelativeRank(4) == (.8, .8).
//
// In essence, the first rank is what fraction of items are smaller than the
// value. The second rank is what fraction of items are smaller or equal to
// this value. This is useful to ascertain when an input list has many
// instances of the same value, for example.
//
// Adding inputs is an O(1) operation. Retrieving a percentile sorts the
// underlying vector only if there has been an addition since the previous sort.
// Thus, retrieving a percentile is O(nlog n) worst case and O(log n) if no
// additional inputs have been added.
template <typename T>
class Percentile {
 public:
  Percentile() {}

  void Add(const T& t) {
    // REF:
    // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
    if (!std::isnan(static_cast<double>(t))) {
      inputs_.push_back(t);
      sorted_ = false;
    }
  }

  void Reset() {
    inputs_.clear();
    sorted_ = true;
  }

  void SerializeToProto(google::protobuf::RepeatedPtrField<ValueType>* values) {
    for (const T& t : inputs_) {
      values->Add(MakeValueType(t));
    }
  }

  void MergeFromProto(google::protobuf::RepeatedPtrField<ValueType> values) {
    for (const ValueType v : values) {
      inputs_.push_back(GetValue<T>(v));
      sorted_ = false;
    }
  }

  int64_t Memory() {
    return sizeof(Percentile<T>) + sizeof(T) * inputs_.capacity();
  }

  int64_t num_values() { return inputs_.size(); }

  // Obtain the relative rank of value t with respect to the added inputs.
  std::pair<double, double> GetRelativeRank(const T& t) {
    if (num_values() == 0) {
      return std::make_pair(0, 1);
    }

    // If something has been added since the last sort, sort again.
    if (!sorted_) {
      std::sort(inputs_.begin(), inputs_.end());
      sorted_ = true;
    }
    auto lb = std::lower_bound(inputs_.begin(), inputs_.end(), t);
    auto ub = std::upper_bound(lb, inputs_.end(), t);
    double num_lt = std::distance(inputs_.begin(), lb);
    double num_le = std::distance(inputs_.begin(), ub);
    return std::make_pair(num_lt / num_values(), num_le / num_values());
  }

 private:
  std::vector<T> inputs_;
  bool sorted_ = true;
};

}  // namespace base
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_BASE_PERCENTILE_H_
