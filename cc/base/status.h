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

#include "absl/status/status.h"

namespace differential_privacy::base {

using absl::AbortedError;
using absl::AlreadyExistsError;
using absl::CancelledError;
using absl::DataLossError;
using absl::DeadlineExceededError;
using absl::FailedPreconditionError;
using absl::InternalError;
using absl::InvalidArgumentError;
using absl::NotFoundError;
using absl::OkStatus;
using absl::OutOfRangeError;
using absl::PermissionDeniedError;
using absl::ResourceExhaustedError;
using absl::Status;
using absl::StatusCode;
using absl::UnauthenticatedError;
using absl::UnavailableError;
using absl::UnimplementedError;
using absl::UnknownError;

}  // namespace differential_privacy::base

#endif  // DIFFERENTIAL_PRIVACY_BASE_STATUS_H_
