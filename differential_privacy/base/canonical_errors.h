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

#ifndef DIFFERENTIAL_PRIVACY_BASE_CANONICAL_ERRORS_H_
#define DIFFERENTIAL_PRIVACY_BASE_CANONICAL_ERRORS_H_

// This file declares a set of functions for working with Status objects from
// the canonical errors. There are functions to easily generate such
// status object and function for classifying them.
#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/strings/string_view.h"
#include "differential_privacy/base/status.h"

namespace differential_privacy {
namespace base {

// Each of the functions below creates a canonical error with the given
// message. The error code of the returned status object matches the name of
// the function.
Status AbortedError(absl::string_view message);
Status AlreadyExistsError(absl::string_view message);
Status CancelledError(absl::string_view message);
Status DataLossError(absl::string_view message);
Status DeadlineExceededError(absl::string_view message);
Status FailedPreconditionError(absl::string_view message);
Status InternalError(absl::string_view message);
Status InvalidArgumentError(absl::string_view message);
Status NotFoundError(absl::string_view message);
Status OutOfRangeError(absl::string_view message);
Status PermissionDeniedError(absl::string_view message);
Status ResourceExhaustedError(absl::string_view message);
Status UnauthenticatedError(absl::string_view message);
Status UnavailableError(absl::string_view message);
Status UnimplementedError(absl::string_view message);
Status UnknownError(absl::string_view message);

}  // namespace base
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_BASE_CANONICAL_ERRORS_H_
