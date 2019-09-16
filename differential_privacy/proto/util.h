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

#ifndef DIFFERENTIAL_PRIVACY_PROTO_UTIL_H_
#define DIFFERENTIAL_PRIVACY_PROTO_UTIL_H_

#include <limits>

#include "differential_privacy/proto/data.pb.h"

namespace differential_privacy {

template <typename T>
struct is_string
    : public std::integral_constant<
          bool,
          std::is_same<char*, typename std::decay<T>::type>::value ||
              std::is_same<const char*, typename std::decay<T>::type>::value ||
              std::is_same<std::string, typename std::decay<T>::type>::value> {
};
template <>
struct is_string<std::string> : std::true_type {};

template <typename T,
          typename std::enable_if<is_string<T>::value>::type* = nullptr>
T GetValue(const ValueType& value_type) {
  return value_type.string_value();
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T GetValue(const ValueType& value_type) {
  return value_type.int_value();
}

template <typename T, typename std::enable_if<
                          std::is_floating_point<T>::value>::type* = nullptr>
T GetValue(const ValueType& value_type) {
  return value_type.float_value();
}

template <typename T,
          typename std::enable_if<is_string<T>::value>::type* = nullptr>
void SetValue(ValueType* value_type, T value) {
  value_type->set_string_value(value);
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
void SetValue(ValueType* value_type, T value) {
  value_type->set_int_value(value);
}

template <typename T, typename std::enable_if<
                          std::is_floating_point<T>::value>::type* = nullptr>
void SetValue(ValueType* value_type, T value) {
  value_type->set_float_value(value);
}

template <typename T>
ValueType MakeValueType(T value) {
  ValueType value_type;
  SetValue(&value_type, value);
  return value_type;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T GetValue(const Output& output) {
  return output.elements(0).value().int_value();
}

template <typename T, typename std::enable_if<
                          std::is_floating_point<T>::value>::type* = nullptr>
T GetValue(const Output& output) {
  return output.elements(0).value().float_value();
}

template <typename T,
          typename std::enable_if<is_string<T>::value>::type* = nullptr>
Output MakeOutput(T value) {
  Output i;
  auto element = i.add_elements();
  element->mutable_value()->set_string_value(value);
  return i;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
Output MakeOutput(T value) {
  Output i;
  auto element = i.add_elements();
  element->mutable_value()->set_int_value(value);
  return i;
}

template <typename T, typename std::enable_if<
                          std::is_floating_point<T>::value>::type* = nullptr>
Output MakeOutput(T value) {
  Output i;
  auto element = i.add_elements();
  element->mutable_value()->set_float_value(value);
  return i;
}

template <typename T,
          typename std::enable_if<is_string<T>::value>::type* = nullptr>
void AddToOutput(Output* output, T value) {
  auto* element = output->add_elements();
  element->mutable_value()->set_string_value(value);
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
void AddToOutput(Output* output, T value) {
  auto* element = output->add_elements();
  element->mutable_value()->set_int_value(value);
}

template <typename T, typename std::enable_if<
                          std::is_floating_point<T>::value>::type* = nullptr>
void AddToOutput(Output* output, T value) {
  auto* element = output->add_elements();
  element->mutable_value()->set_float_value(value);
}

}  // namespace differential_privacy
#endif  // DIFFERENTIAL_PRIVACY_PROTO_UTIL_H_
