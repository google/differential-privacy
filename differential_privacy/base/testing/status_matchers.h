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

#ifndef DIFFERENTIAL_PRIVACY_BASE_TESTING_STATUS_MATCHERS_H_
#define DIFFERENTIAL_PRIVACY_BASE_TESTING_STATUS_MATCHERS_H_

// Testing utilities for working with ::differential_privacy::base::Status and
// ::differential_privacy::base::StatusOr.
//
//
// Defines the following utilities:
//
//   =================
//   EXPECT_OK(s)
//
//   ASSERT_OK(s)
//   =================
//   Convenience macros for `EXPECT_THAT(s, IsOk())`, where `s` is either
//   a `Status` or a `StatusOr<T>` or a `StatusProto`.
//
//   There are no EXPECT_NOT_OK/ASSERT_NOT_OK macros since they would not
//   provide much value (when they fail, they would just print the OK status
//   which conveys no more information than EXPECT_FALSE(s.ok());
//   If you want to check for particular errors, better alternatives are:
//   EXPECT_THAT(s, StatusIs(expected_error));
//   EXPECT_THAT(s, StatusIs(_, HasSubstr("expected error")));
//
//   ===============
//   IsOkAndHolds(m)
//   ===============
//
//   This gMock matcher matches a StatusOr<T> value whose status is OK
//   and whose inner value matches matcher m.  Example:
//
//     using ::testing::MatchesRegex;
//     using ::testing::status::IsOkAndHolds;
//     ...
//     StatusOr<std::string> maybe_name = ...;
//     EXPECT_THAT(maybe_name, IsOkAndHolds(MatchesRegex("John .*")));
//
//   ===============================
//   StatusIs(status_code_matcher,
//            error_message_matcher)
//   ===============================
//
//   This gMock matcher matches a Status or StatusOr<T> or StatusProto value if
//   all of the following are true:
//
//     - the status' code() matches status_code_matcher, and
//     - the status' message() matches error_message_matcher.
//
//   Example:
//
//
//     using ::testing::HasSubstr;
//     using ::testing::MatchesRegex;
//     using ::testing::Ne;
//     using ::differential_privacy::base::testing::StatusIs;
//     using ::testing::_;
//     using ::differential_privacy::base::StatusOr;
//     StatusOr<std::string> GetName(int id);
//     ...
//
//     // The status code must be kAborted;
//     // the error message can be anything.
//     EXPECT_THAT(GetName(42),
//                 StatusIs(differential_privacy::base::StatusCode::kAborted,
//                 _));
//     // The status code can be anything; the error message must match the
//     // regex.
//     EXPECT_THAT(GetName(43),
//                 StatusIs(_, MatchesRegex("server.*time-out")));
//
//     // The status code should not be kAborted; the error message can be
//     // anything with "client" in it.
//     EXPECT_CALL(mock_env, HandleStatus(
//         StatusIs(Ne(differential_privacy::base::StatusCode::kAborted),
//                  HasSubstr("client"))));
//
//   ===============================
//   StatusIs(status_code_matcher)
//   ===============================
//
//   This is a shorthand for
//     StatusIs(status_code_matcher, testing::_)
//   In other words, it's like the two-argument StatusIs(), except that it
//   ignores error message.
//
//   ===============
//   IsOk()
//   ===============
//
//   Matches a differential_privacy::base::Status or
//   differential_privacy::base::StatusOr<T> value whose status value is
//   StatusCode::kOK. Equivalent to 'StatusIs(StatusCode::kOK)'. Example:
//     using ::testing::status::IsOk;
//     ...
//     StatusOr<std::string> maybe_name = ...;
//     EXPECT_THAT(maybe_name, IsOk());
//     Status s = ...;
//     EXPECT_THAT(s, IsOk());
//

#include <ostream>
#include <string>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "differential_privacy/base/status.h"
#include "differential_privacy/base/status_macros.h"
#include "differential_privacy/base/statusor.h"

namespace differential_privacy {
namespace base {
namespace testing {
namespace internal_status {

inline const Status& GetStatus(const Status& status) {
  return status;
}

template <typename T>
inline const Status& GetStatus(const StatusOr<T>& status) {
  return status.status();
}

////////////////////////////////////////////////////////////
// Implementation of IsOkAndHolds().

// Monomorphic implementation of matcher IsOkAndHolds(m).  StatusOrType can be
// either StatusOr<T> or a reference to it.
template <typename StatusOrType>
class IsOkAndHoldsMatcherImpl
    : public ::testing::MatcherInterface<StatusOrType> {
 public:
  typedef typename std::remove_reference<StatusOrType>::type::element_type
      value_type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const value_type&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      StatusOrType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    if (!actual_value.ok()) {
      *result_listener << "which has status " << actual_value.status();
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    const bool matches = inner_matcher_.MatchAndExplain(
        actual_value.ValueOrDie(), &inner_listener);
    const std::string inner_explanation = inner_listener.str();
    if (inner_explanation != "") {
      *result_listener << "which contains value "
                       << ::testing::PrintToString(actual_value.ValueOrDie())
                       << ", " << inner_explanation;
    }
    return matches;
  }

 private:
  const ::testing::Matcher<const value_type&> inner_matcher_;
};

// Implements IsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type.  StatusOrType can be either StatusOr<T> or a
  // reference to StatusOr<T>.
  template <typename StatusOrType>
  operator ::testing::Matcher<StatusOrType>() const {  // NOLINT
    return MakeMatcher(
        new IsOkAndHoldsMatcherImpl<StatusOrType>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

////////////////////////////////////////////////////////////
// Implementation of StatusIs().

// StatusIs() is a polymorphic matcher.  This class is the common
// implementation of it shared by all types T where StatusIs() can be
// used as a Matcher<T>.
class StatusIsMatcherCommonImpl {
 public:
  StatusIsMatcherCommonImpl(
      ::testing::Matcher<StatusCode> code_matcher,
      ::testing::Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  void DescribeTo(std::ostream* os) const;

  void DescribeNegationTo(std::ostream* os) const;

  bool MatchAndExplain(const Status& status,
                       ::testing::MatchResultListener* result_listener) const;

 private:
  const ::testing::Matcher<StatusCode> code_matcher_;
  const ::testing::Matcher<const std::string&> message_matcher_;
};

// Monomorphic implementation of matcher StatusIs() for a given type
// T.  T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoStatusIsMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  explicit MonoStatusIsMatcherImpl(StatusIsMatcherCommonImpl common_impl)
      : common_impl_(std::move(common_impl)) {}

  void DescribeTo(std::ostream* os) const override {
    common_impl_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    common_impl_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      T actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    return common_impl_.MatchAndExplain(GetStatus(actual_value),
                                        result_listener);
  }

 private:
  StatusIsMatcherCommonImpl common_impl_;
};

// Implements StatusIs() as a polymorphic matcher.
class StatusIsMatcher {
 public:
  StatusIsMatcher(::testing::Matcher<StatusCode> code_matcher,
                  ::testing::Matcher<const std::string&> message_matcher)
      : common_impl_(std::move(code_matcher), std::move(message_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type.  T can be StatusOr<>, Status, or a reference to either of them.
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::MakeMatcher(new MonoStatusIsMatcherImpl<T>(common_impl_));
  }

 private:
  const StatusIsMatcherCommonImpl common_impl_;
};

// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  void DescribeTo(std::ostream* os) const override { *os << "is OK"; }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not OK";
  }
  bool MatchAndExplain(T actual_value,
                       ::testing::MatchResultListener*) const override {
    return GetStatus(actual_value).ok();
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::MakeMatcher(new MonoIsOkMatcherImpl<T>());
  }
};

}  // namespace internal_status

// Macros for testing the results of functions that return base::Status or
// differential_privacy::base::StatusOr<T> (for any type T).
#define EXPECT_OK(expression) \
  EXPECT_THAT(expression, ::differential_privacy::base::testing::IsOk())
#define ASSERT_OK(expression) \
  ASSERT_THAT(expression, ::differential_privacy::base::testing::IsOk())

// Returns a gMock matcher that matches a StatusOr<> whose status is
// OK and whose value matches the inner matcher.
template <typename InnerMatcher>
internal_status::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
IsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal_status::IsOkAndHoldsMatcher<
      typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

// Returns a gMock matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher, and whose error message matches message_matcher.
template <typename StatusCodeMatcher>
internal_status::StatusIsMatcher StatusIs(
    StatusCodeMatcher&& code_matcher,
    ::testing::Matcher<const std::string&> message_matcher) {
  return internal_status::StatusIsMatcher(
      std::forward<StatusCodeMatcher>(code_matcher),
      std::move(message_matcher));
}

// Returns a gMock matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher.
template <typename StatusCodeMatcher>
internal_status::StatusIsMatcher StatusIs(StatusCodeMatcher&& code_matcher) {
  return StatusIs(std::forward<StatusCodeMatcher>(code_matcher), ::testing::_);
}

// Returns a gMock matcher that matches a Status or StatusOr<> which is OK.
inline internal_status::IsOkMatcher IsOk() {
  return internal_status::IsOkMatcher();
}

}  // namespace testing
}  // namespace base
}  // namespace differential_privacy

#endif  // DIFFERENTIAL_PRIVACY_BASE_TESTING_STATUS_MATCHERS_H_
