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

#include "base/status_macros.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "base/status.h"
#include "base/statusor.h"

namespace differential_privacy {
namespace base {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Eq;

Status ReturnOk() { return OkStatus(); }

Status ReturnError(absl::string_view msg) {
  return Status(StatusCode::kUnknown, msg);
}

StatusOr<int> ReturnStatusOrValue(int v) { return v; }

StatusOr<int> ReturnStatusOrError(absl::string_view msg) {
  return Status(StatusCode::kUnknown, msg);
}

StatusOr<std::unique_ptr<int>> ReturnStatusOrPtrValue(int v) {
  return absl::make_unique<int>(v);
}

TEST(AssignOrReturn, Works) {
  auto func = []() -> Status {
    ASSIGN_OR_RETURN(int value1, ReturnStatusOrValue(1));
    EXPECT_EQ(1, value1);
    ASSIGN_OR_RETURN(const int value2, ReturnStatusOrValue(2));
    EXPECT_EQ(2, value2);
    ASSIGN_OR_RETURN(const int& value3, ReturnStatusOrValue(3));
    EXPECT_EQ(3, value3);
    ASSIGN_OR_RETURN(ABSL_ATTRIBUTE_UNUSED int value4,
                     ReturnStatusOrError("EXPECTED"));
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, WorksForExistingVariable) {
  auto func = []() -> Status {
    int value = 1;
    ASSIGN_OR_RETURN(value, ReturnStatusOrValue(2));
    EXPECT_EQ(2, value);
    ASSIGN_OR_RETURN(value, ReturnStatusOrValue(3));
    EXPECT_EQ(3, value);
    ASSIGN_OR_RETURN(value, ReturnStatusOrError("EXPECTED"));
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, UniquePtrWorks) {
  auto func = []() -> Status {
    ASSIGN_OR_RETURN(std::unique_ptr<int> ptr, ReturnStatusOrPtrValue(1));
    EXPECT_EQ(*ptr, 1);
    return ReturnError("EXPECTED");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(AssignOrReturn, UniquePtrWorksForExistingVariable) {
  auto func = []() -> Status {
    std::unique_ptr<int> ptr;
    ASSIGN_OR_RETURN(ptr, ReturnStatusOrPtrValue(1));
    EXPECT_EQ(*ptr, 1);

    ASSIGN_OR_RETURN(ptr, ReturnStatusOrPtrValue(2));
    EXPECT_EQ(*ptr, 2);
    return ReturnError("EXPECTED");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(ReturnIfError, Works) {
  auto func = []() -> Status {
    RETURN_IF_ERROR(ReturnOk());
    RETURN_IF_ERROR(ReturnOk());
    RETURN_IF_ERROR(ReturnError("EXPECTED"));
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(ReturnIfError, WorksWithLambda) {
  auto func = []() -> Status {
    RETURN_IF_ERROR([] { return ReturnOk(); }());
    RETURN_IF_ERROR([] { return ReturnError("EXPECTED"); }());
    return ReturnError("ERROR");
  };

  EXPECT_THAT(func().message(), Eq("EXPECTED"));
}

TEST(ReturnIfError, CallsFunctionOnce) {
  auto successFunc = []() -> Status {
    bool calledBefore = false;
    auto successfulCalledOnceFunc = [&calledBefore]() -> Status {
      if (!calledBefore) {
        calledBefore = true;
        return ReturnOk();
      }
      return ReturnError("ERROR");
    };
    RETURN_IF_ERROR(successfulCalledOnceFunc());
    return ReturnOk();
  };
  EXPECT_TRUE(successFunc().ok());

  auto failureFunc = []() -> Status {
    bool calledBefore = false;
    auto successfulCalledOnceFunc = [&calledBefore]() -> Status {
      if (!calledBefore) {
        calledBefore = true;
        return ReturnError("EXPECTED");
      }
      return ReturnError("ERROR");
    };
    RETURN_IF_ERROR(successfulCalledOnceFunc());
    return ReturnOk();
  };
  EXPECT_THAT(failureFunc().message(), Eq("EXPECTED"));
}

}  // namespace
}  // namespace base
}  // namespace differential_privacy
