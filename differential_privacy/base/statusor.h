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

#include <new>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "differential_privacy/base/logging.h"
#include "differential_privacy/base/status.h"
#include "differential_privacy/base/statusor_internals.h"

namespace differential_privacy {
namespace base {

// Returned StatusOr objects may not be ignored.
template <typename T>
class ABSL_MUST_USE_RESULT StatusOr;

template <typename T>
class StatusOr : private statusor_internal::StatusOrData<T>,
                 private statusor_internal::TraitsBase<
                     std::is_copy_constructible<T>::value,
                     std::is_move_constructible<T>::value> {
  template <typename U>
  friend class StatusOr;

  typedef statusor_internal::StatusOrData<T> Base;

 public:
  using element_type = T;

  // Constructs a new StatusOr with Status::UNKNOWN status.  This is marked
  // 'explicit' to try to catch cases like 'return {};', where people think
  // StatusOr<std::vector<int>> will be initialized with an empty vector,
  // instead of a Status::UNKNOWN status.
  explicit StatusOr();

  // StatusOr<T> will be copy constructible/assignable if T is copy
  // constructible.
  StatusOr(const StatusOr&) = default;
  StatusOr& operator=(const StatusOr&) = default;

  // StatusOr<T> will be move constructible/assignable if T is move
  // constructible.
  StatusOr(StatusOr&&) = default;
  StatusOr& operator=(StatusOr&&) = default;

  // Conversion copy/move constructor, T must be convertible from U.
  // TODO: These should not participate in overload resolution if U
  // is not convertible to T.
  template <typename U>
  StatusOr(const StatusOr<U>& other);
  template <typename U>
  StatusOr(StatusOr<U>&& other);

  // Conversion copy/move assignment operator, T must be convertible from U.
  template <typename U>
  StatusOr& operator=(const StatusOr<U>& other);
  template <typename U>
  StatusOr& operator=(StatusOr<U>&& other);

  // Constructs a new StatusOr with the given value. After calling this
  // constructor, this->ok() will be true and the contained value may be
  // retrieved with ValueOrDie(), operator*(), or operator->().
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return type
  // so it is convenient and sensible to be able to do 'return T()'
  // when the return type is StatusOr<T>.
  //
  // REQUIRES: T is copy constructible.
  StatusOr(const T& value);

  // Constructs a new StatusOr with the given non-ok status. After calling this
  // constructor, this->ok() will be false and calls to ValueOrDie() will
  // CHECK-fail.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return
  // value, so it is convenient and sensible to be able to do 'return
  // Status()' when the return type is StatusOr<T>.
  //
  // REQUIRES: !status.ok(). This requirement is DCHECKed.
  // In optimized builds, passing OkStatus() here will have the effect
  // of passing INTERNAL as a fallback.
  StatusOr(const Status& status);
  StatusOr& operator=(const Status& status);

  // Similar to the `const T&` overload.
  //
  // REQUIRES: T is move constructible.
  StatusOr(T&& value);

  // RValue versions of the operations declared above.
  StatusOr(Status&& status);
  StatusOr& operator=(Status&& status);

  // Returns this->ok()
  explicit operator bool() const { return ok(); }

  // Returns this->status().ok()
  ABSL_MUST_USE_RESULT bool ok() const { return this->status_.ok(); }

  // Returns a reference to our status. If this contains a T, then
  // returns OkStatus().
  const Status& status() const&;
  Status status() &&;

  // Returns a reference to our current value, or CHECK-fails if !this->ok(). If
  // you have already checked the status using this->ok() or operator bool(),
  // then you probably want to use operator*() or operator->() to access the
  // current value instead of ValueOrDie().
  //
  // Note: for value types that are cheap to copy, prefer simple code:
  //
  //   T value = statusor.ValueOrDie();
  //
  // Otherwise, if the value type is expensive to copy, but can be left
  // in the StatusOr, simply assign to a reference:
  //
  //   T& value = statusor.ValueOrDie();  // or `const T&`
  //
  // Otherwise, if the value type supports an efficient move, it can be
  // used as follows:
  //
  //   T value = std::move(statusor).ValueOrDie();
  //
  // The std::move on statusor instead of on the whole expression enables
  // warnings about possible uses of the statusor object after the move.

  const T& ValueOrDie() const&;
  T& ValueOrDie() &;
  const T&& ValueOrDie() const&&;
  T&& ValueOrDie() &&;

  // Returns a reference to the current value.
  //
  // REQUIRES: this->ok() == true, otherwise the behavior is undefined.
  //
  // Use this->ok() or `operator bool()` to verify that there is a current
  // value. Alternatively, see ValueOrDie() for a similar API that guarantees
  // CHECK-failing if there is no current value.
  //
  const T& operator*() const&;
  T& operator*() &;
  const T&& operator*() const&&;
  T&& operator*() &&;

  // Returns a pointer to the current value.
  //
  // REQUIRES: this->ok() == true, otherwise the behavior is undefined.
  //
  // Use this->ok() or `operator bool()` to verify that there is a current
  // value.
  const T* operator->() const;
  T* operator->();

  // Returns a copy of the current value if this->ok() == true. Otherwise
  // returns a default value.
  template <typename U>
  T value_or(U&& default_value) const&;
  template <typename U>
  T value_or(U&& default_value) &&;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation details for StatusOr<T>

template <typename T>
StatusOr<T>::StatusOr() : Base(Status(UNKNOWN, "")) {}

template <typename T>
StatusOr<T>::StatusOr(const T& value) : Base(value) {}

template <typename T>
StatusOr<T>::StatusOr(const Status& status) : Base(status) {}

template <typename T>
StatusOr<T>& StatusOr<T>::operator=(const Status& status) {
  this->Assign(status);
  return *this;
}

template <typename T>
StatusOr<T>::StatusOr(T&& value) : Base(std::move(value)) {}

template <typename T>
StatusOr<T>::StatusOr(Status&& status) : Base(std::move(status)) {}

template <typename T>
StatusOr<T>& StatusOr<T>::operator=(Status&& status) {
  this->Assign(std::move(status));
  return *this;
}

template <typename T>
template <typename U>
inline StatusOr<T>::StatusOr(const StatusOr<U>& other)
    : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {}

template <typename T>
template <typename U>
inline StatusOr<T>& StatusOr<T>::operator=(const StatusOr<U>& other) {
  if (other.ok())
    this->Assign(other.ValueOrDie());
  else
    this->Assign(other.status());
  return *this;
}

template <typename T>
template <typename U>
inline StatusOr<T>::StatusOr(StatusOr<U>&& other)
    : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {}

template <typename T>
template <typename U>
inline StatusOr<T>& StatusOr<T>::operator=(StatusOr<U>&& other) {
  if (other.ok()) {
    this->Assign(std::move(other).ValueOrDie());
  } else {
    this->Assign(std::move(other).status());
  }
  return *this;
}

template <typename T>
const Status& StatusOr<T>::status() const& {
  return this->status_;
}
template <typename T>
Status StatusOr<T>::status() && {
  return ok() ? OkStatus() : std::move(this->status_);
}

template <typename T>
const T& StatusOr<T>::ValueOrDie() const& {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
T& StatusOr<T>::ValueOrDie() & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::ValueOrDie() const&& {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::ValueOrDie() && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
const T& StatusOr<T>::operator*() const& {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
T& StatusOr<T>::operator*() & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::operator*() const&& {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::operator*() && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
const T* StatusOr<T>::operator->() const {
  this->EnsureOk();
  return &this->data_;
}

template <typename T>
T* StatusOr<T>::operator->() {
  this->EnsureOk();
  return &this->data_;
}

template <typename T>
template <typename U>
T StatusOr<T>::value_or(U&& default_value) const& {
  if (ok()) {
    return this->data_;
  }
  return std::forward<U>(default_value);
}

template <typename T>
template <typename U>
T StatusOr<T>::value_or(U&& default_value) && {
  if (ok()) {
    return std::move(this->data_);
  }
  return std::forward<U>(default_value);
}

template <typename T>
void StatusOr<T>::IgnoreError() const {
  // no-op
}

}  // namespace base
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_BASE_STATUSOR_H_
