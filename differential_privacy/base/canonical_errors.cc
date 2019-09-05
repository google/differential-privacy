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

#include "differential_privacy/base/canonical_errors.h"

namespace differential_privacy {
namespace base {

Status AbortedError(absl::string_view message) {
  return Status(ABORTED, message);
}

Status AlreadyExistsError(absl::string_view message) {
  return Status(ALREADY_EXISTS, message);
}

Status CancelledError(absl::string_view message) {
  return Status(CANCELLED, message);
}

Status DataLossError(absl::string_view message) {
  return Status(DATA_LOSS, message);
}

Status DeadlineExceededError(absl::string_view message) {
  return Status(DEADLINE_EXCEEDED, message);
}

Status FailedPreconditionError(absl::string_view message) {
  return Status(FAILED_PRECONDITION, message);
}

Status InternalError(absl::string_view message) {
  return Status(INTERNAL, message);
}

Status InvalidArgumentError(absl::string_view message) {
  return Status(INVALID_ARGUMENT, message);
}

Status NotFoundError(absl::string_view message) {
  return Status(NOT_FOUND, message);
}

Status OutOfRangeError(absl::string_view message) {
  return Status(OUT_OF_RANGE, message);
}

Status PermissionDeniedError(absl::string_view message) {
  return Status(PERMISSION_DENIED, message);
}

Status ResourceExhaustedError(absl::string_view message) {
  return Status(RESOURCE_EXHAUSTED, message);
}

Status UnauthenticatedError(absl::string_view message) {
  return Status(UNAUTHENTICATED, message);
}

Status UnavailableError(absl::string_view message) {
  return Status(UNAVAILABLE, message);
}

Status UnimplementedError(absl::string_view message) {
  return Status(UNIMPLEMENTED, message);
}

Status UnknownError(absl::string_view message) {
  return Status(UNKNOWN, message);
}

}  // namespace base
}  // namespace differential_privacy
