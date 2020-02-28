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

#include "differential_privacy/base/statusor.h"

#include <ostream>

#include "differential_privacy/base/canonical_errors.h"

namespace differential_privacy {
namespace base {

namespace statusor_internal {

void Helper::HandleInvalidStatusCtorArg(Status* status) {
  const char* kMessage =
      "An OK status is not a valid constructor argument to StatusOr<T>";
  LOG(DFATAL) << kMessage;
  // In optimized builds, we will fall back to absl::INTERNAL.
  *status = InternalError(kMessage);
}

void Helper::Crash(const Status& status) {
  LOG(FATAL) << "Attempting to fetch value instead of handling error "
             << status;
  CHECK(false);
}

}  // namespace statusor_internal
}  // namespace base
}  // namespace differential_privacy
