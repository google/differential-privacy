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

#include "differential_privacy/base/status.h"

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"

namespace differential_privacy {
namespace base {

std::string StatusCodeToString(StatusCode e) {
  if (e == StatusCode::kOk) {
    return "OK";
  } else if (e == StatusCode::kInvalidArgument) {
    return "kInvalidArgument";
  } else if (e == StatusCode::kCancelled) {
    return "kCancelled";
  } else if (e == StatusCode::kUnknown) {
    return "kUnknown";
  } else {
    return absl::StrCat(e);
  }
}

std::ostream& operator<<(std::ostream& os, StatusCode code) {
  return os << StatusCodeToString(code);
}

Status::Status(StatusCode code, absl::string_view message)
    : code_(code), message_(code == StatusCode::kOk ? "" : message) {}

std::string Status::ToString() const {
  return ok() ? "OK" : absl::StrCat(StatusCodeToString(code()), ": ", message_);
}

void Status::SetPayload(absl::string_view type_url, const StatusCord& payload) {
  if (!ok()) {
    payload_[type_url] = payload;
  }
}

absl::optional<StatusCord> Status::GetPayload(
    absl::string_view type_url) const {
  auto it = payload_.find(std::string(type_url));
  if (it == payload_.end()) return absl::nullopt;
  return it->second;
}

void Status::ErasePayload(absl::string_view type_url) {
  payload_.erase(std::string(type_url));
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  return os << x.ToString();
}

}  // namespace base
}  // namespace differential_privacy
