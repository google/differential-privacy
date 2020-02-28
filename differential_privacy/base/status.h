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

#ifndef DIFFERENTIAL_PRIVACY_BASE_STATUS_H_
#define DIFFERENTIAL_PRIVACY_BASE_STATUS_H_

#include <ostream>
#include <string>
#include <unordered_map>

#include "absl/base/attributes.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"

namespace differential_privacy {
namespace base {

using StatusCord = std::string;

enum class StatusCode {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
  kUnauthenticated = 16,
  kDoNotUseReservedForFutureExpansionUseDefaultInSwitchInstead_ = 20
};

std::string StatusCodeToString(StatusCode e);

std::ostream& operator<<(std::ostream& os, StatusCode code);

// Handle both for now. This is meant to be _very short lived. Once internal
// code can safely use the new naming, we will switch that that and drop this.
constexpr StatusCode OK = StatusCode::kOk;
constexpr StatusCode CANCELLED = StatusCode::kCancelled;
constexpr StatusCode UNKNOWN = StatusCode::kUnknown;
constexpr StatusCode INVALID_ARGUMENT = StatusCode::kInvalidArgument;
constexpr StatusCode DEADLINE_EXCEEDED = StatusCode::kDeadlineExceeded;
constexpr StatusCode NOT_FOUND = StatusCode::kNotFound;
constexpr StatusCode ALREADY_EXISTS = StatusCode::kAlreadyExists;
constexpr StatusCode PERMISSION_DENIED = StatusCode::kPermissionDenied;
constexpr StatusCode UNAUTHENTICATED = StatusCode::kUnauthenticated;
constexpr StatusCode RESOURCE_EXHAUSTED = StatusCode::kResourceExhausted;
constexpr StatusCode FAILED_PRECONDITION = StatusCode::kFailedPrecondition;
constexpr StatusCode ABORTED = StatusCode::kAborted;
constexpr StatusCode OUT_OF_RANGE = StatusCode::kOutOfRange;
constexpr StatusCode UNIMPLEMENTED = StatusCode::kUnimplemented;
constexpr StatusCode INTERNAL = StatusCode::kInternal;
constexpr StatusCode UNAVAILABLE = StatusCode::kUnavailable;
constexpr StatusCode DATA_LOSS = StatusCode::kDataLoss;

class ABSL_MUST_USE_RESULT Status;

class Status final {
 public:
  // Builds an OK Status.
  Status() = default;

  // Constructs a Status object containing a status code and message.
  // If `code == StatusCode::kOk`, `msg` is ignored and an object identical to
  // an OK status is constructed.
  Status(StatusCode code, absl::string_view message);

  // Return the error message (if any).
  absl::string_view message() const { return message_; }

  // Returns true if the Status is OK.
  ABSL_MUST_USE_RESULT bool ok() const;

  // Deprecated. Use code().
  int error_code() const;

  // Deprecated. Use message().
  std::string error_message() const;

  // Deprecated. Use code().
  StatusCode CanonicalCode() const;

  // If "ok()", does nothing.  Else adds the given `payload` specified, by
  // `type_url` as an additional payload.
  void SetPayload(absl::string_view type_url, const StatusCord& payload);

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  void Update(const Status& rhs) {
    if (ok()) *this = rhs;
  }

  // Return a combination of the error code name and message.
  // Note, no guarantees are made as to the exact nature of the returned std::string.
  // Subject to change at any time.
  std::string ToString() const;

  // Deprecated. Just returns self.
  Status ToCanonical() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  // Returns the stored status code.
  StatusCode code() const { return code_; }

  // Retrieve a single value associated with `type_url`. Returns absl::nullopt
  // if no value is associated with `type_url`.
  absl::optional<StatusCord> GetPayload(const absl::string_view type_url) const;

  // Erase the payload associated with `type_url`, if present.
  void ErasePayload(absl::string_view type_url);

  void ForEachPayload(
      const std::function<void(absl::string_view, const StatusCord&)>& visitor)
      const;

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
  // Structured error payload. String is a 'type_url' for example, a proto
  // descriptor full name.
  absl::node_hash_map<std::string, StatusCord> payload_;
};

inline bool Status::ok() const { return StatusCode::kOk == code_; }

inline int Status::error_code() const { return static_cast<int>(code()); }

inline std::string Status::error_message() const {
  return std::string(message());
}

inline StatusCode Status::CanonicalCode() const { return code(); }

inline Status Status::ToCanonical() const { return *this; }

inline bool Status::operator==(const Status& x) const {
  return (code_ == x.code_) && (message_ == x.message_) &&
         (payload_ == x.payload_);
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

inline void Status::IgnoreError() const {
  // no-op
}

inline void Status::ForEachPayload(
    const std::function<void(absl::string_view, const StatusCord&)>& visitor)
    const {
  for (auto it = payload_.begin(); it != payload_.end(); ++it) {
    visitor(it->first, it->second);
  }
}

// Prints a human-readable representation of 'x' to 'os'.
std::ostream& operator<<(std::ostream& os, const Status& x);

// Constructs an OK status object.
inline Status OkStatus() { return Status(); }

}  // namespace base
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_BASE_STATUS_H_
