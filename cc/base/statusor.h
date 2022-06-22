//
// Copyright 2019 Google LLC
// Copyright 2018 ZetaSQL Authors
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

#ifndef DIFFERENTIAL_PRIVACY_BASE_STATUSOR_H_
#define DIFFERENTIAL_PRIVACY_BASE_STATUSOR_H_

// StatusOr<T> is the union of a Status object and a T
// object. StatusOr models the concept of an object that is either a
// usable value, or an error Status explaining why such a value is
// not present. To this end, StatusOr<T> does not allow its Status
// value to be OkStatus().
//
// The primary use-case for StatusOr<T> is as the return value of a
// function which may fail.
//
// Example usage of a StatusOr<T>:
//
//  StatusOr<Foo> result = DoBigCalculationThatCouldFail();
//  if (result) {
//    result->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example that is guaranteed crash if the result holds no value:
//
//  StatusOr<Foo> result = DoBigCalculationThatCouldFail();
//  const Foo& foo = result.ValueOrDie();
//  foo.DoSomethingCool();
//
// Example usage of a StatusOr<std::unique_ptr<T>>:
//
//  StatusOr<std::unique_ptr<Foo>> result = FooFactory::MakeNewFoo(arg);
//  if (!result) {
//    LOG(ERROR) << result.status();
//  } else if (*result == nullptr) {
//    LOG(ERROR) << "Unexpected null pointer";
//  } else {
//    (*result)->DoSomethingCool();
//  }
//
// Example factory implementation returning StatusOr<T>:
//
//  StatusOr<Foo> FooFactory::MakeFoo(int arg) {
//    if (arg <= 0) {
//      return
//      ::differential_privacy::base::Status(
//                   ::differential_privacy::base::INVALID_ARGUMENT,
//                   "Arg must be positive");
//    }
//    return Foo(arg);
//  }
//

#include "absl/status/statusor.h"

namespace differential_privacy {
namespace base {

using absl::StatusOr;

}  // namespace base
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_BASE_STATUSOR_H_
