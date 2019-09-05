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

// Unit tests for StatusOr

#include "differential_privacy/base/statusor.h"

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "differential_privacy/base/canonical_errors.h"
#include "differential_privacy/base/testing/status_matchers.h"

namespace differential_privacy {
namespace base {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;

class Base1 {
 public:
  virtual ~Base1() {}
  int pad;
};

class Base2 {
 public:
  virtual ~Base2() {}
  int yetotherpad;
};

class Derived : public Base1, public Base2 {
 public:
  virtual ~Derived() {}
  int evenmorepad;
};

class CopyNoAssign {
 public:
  explicit CopyNoAssign(int value) : foo(value) {}
  CopyNoAssign(const CopyNoAssign& other) : foo(other.foo) {}
  int foo;

 private:
  const CopyNoAssign& operator=(const CopyNoAssign&);
};

StatusOr<std::unique_ptr<int>> ReturnUniquePtr() {
  // Uses implicit constructor from T&&
  return absl::make_unique<int>(0);
}

TEST(StatusOr, ElementType) {
  static_assert(std::is_same<StatusOr<int>::element_type, int>(), "");
  static_assert(std::is_same<StatusOr<char>::element_type, char>(), "");
}

TEST(StatusOr, TestMoveOnlyInitialization) {
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.ValueOrDie());
  int* previous = thing.ValueOrDie().get();

  thing = ReturnUniquePtr();
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.ValueOrDie());
  EXPECT_NE(previous, thing.ValueOrDie().get());
}

TEST(StatusOr, TestMoveOnlyInitializationFromTemporaryByValueOrDie) {
  std::unique_ptr<int> ptr(ReturnUniquePtr().ValueOrDie());
  EXPECT_EQ(0, *ptr);
}

TEST(StatusOr, TestValueOrDieOverloadForConstTemporary) {
  static_assert(
      std::is_same<const int&&, decltype(std::declval<const StatusOr<int>&&>()
                                             .ValueOrDie())>(),
      "ValueOrDie() for const temporaries should return const T&&");
}

TEST(StatusOr, TestMoveOnlyConversion) {
  StatusOr<std::unique_ptr<const int>> const_thing(ReturnUniquePtr());
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.ValueOrDie());

  // Test rvalue converting assignment
  const int* const_previous = const_thing.ValueOrDie().get();
  const_thing = ReturnUniquePtr();
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.ValueOrDie());
  EXPECT_NE(const_previous, const_thing.ValueOrDie().get());
}

TEST(StatusOr, TestMoveOnlyVector) {
  // Sanity check that StatusOr<MoveOnly> works in vector.
  std::vector<StatusOr<std::unique_ptr<int>>> vec;
  vec.push_back(ReturnUniquePtr());
  vec.resize(2);
  auto another_vec = std::move(vec);
  EXPECT_EQ(0, *another_vec[0].ValueOrDie());
  EXPECT_EQ(UnknownError(""), another_vec[1].status());
}

TEST(StatusOr, TestDefaultCtor) {
  StatusOr<int> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_TRUE(thing.status().code() == UNKNOWN);
}

TEST(StatusOrDeathTest, TestDefaultCtorValue) {
  StatusOr<int> thing;
  EXPECT_DEATH(thing.ValueOrDie(), "Unknown");

  const StatusOr<int> thing2;
  EXPECT_DEATH(thing.ValueOrDie(), "Unknown");
}

TEST(StatusOr, TestStatusCtor) {
  StatusOr<int> thing(CancelledError(""));
  EXPECT_FALSE(thing.ok());
  EXPECT_TRUE(thing.status().code() == CANCELLED);
}

TEST(StatusOrDeathTest, TestStatusCtorStatusOk) {
  EXPECT_DEBUG_DEATH(
      {
        // This will DCHECK
        StatusOr<int> thing(OkStatus());
        // In optimized mode, we are actually going to get INTERNAL for
        // status here, rather than crashing, so check that.
        EXPECT_FALSE(thing.ok());
        EXPECT_TRUE(thing.status().code() == INTERNAL);
      },
      "An OK status is not a valid constructor argument");
}

TEST(StatusOr, TestValueCtor) {
  const int kI = 4;
  const StatusOr<int> thing(kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusOk) {
  const int kI = 4;
  const StatusOr<int> original(kI);
  const StatusOr<int> copy(original);
  EXPECT_OK(copy.status());
  EXPECT_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusNotOk) {
  StatusOr<int> original(CancelledError(""));
  StatusOr<int> copy(original);
  EXPECT_TRUE(copy.status().code() == CANCELLED);
}

TEST(StatusOr, TestCopyCtorNonAssignable) {
  const int kI = 4;
  CopyNoAssign value(kI);
  StatusOr<CopyNoAssign> original(value);
  StatusOr<CopyNoAssign> copy(original);
  EXPECT_OK(copy.status());
  EXPECT_EQ(original.ValueOrDie().foo, copy.ValueOrDie().foo);
}

TEST(StatusOr, TestCopyCtorStatusOKConverting) {
  const int kI = 4;
  StatusOr<int> original(kI);
  StatusOr<double> copy(original);
  EXPECT_OK(copy.status());
  EXPECT_DOUBLE_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestCopyCtorStatusNotOkConverting) {
  StatusOr<int> original(CancelledError(""));
  StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestAssignmentStatusOk) {
  // Copy assignmment
  {
    const auto p = std::make_shared<int>(17);
    StatusOr<std::shared_ptr<int>> source(p);

    StatusOr<std::shared_ptr<int>> target;
    target = source;

    ASSERT_TRUE(target.ok());
    EXPECT_OK(target.status());
    EXPECT_EQ(p, target.ValueOrDie());

    ASSERT_TRUE(source.ok());
    EXPECT_OK(source.status());
    EXPECT_EQ(p, source.ValueOrDie());
  }

  // Move asssignment
  {
    const auto p = std::make_shared<int>(17);
    StatusOr<std::shared_ptr<int>> source(p);

    StatusOr<std::shared_ptr<int>> target;
    target = std::move(source);

    ASSERT_TRUE(target.ok());
    EXPECT_OK(target.status());
    EXPECT_EQ(p, target.ValueOrDie());

    ASSERT_TRUE(source.ok());
    EXPECT_OK(source.status());
    EXPECT_EQ(nullptr, source.ValueOrDie());
  }
}

TEST(StatusOr, TestAssignmentStatusNotOk) {
  // Copy assignment
  {
    const Status expected = CancelledError("");
    StatusOr<int> source(expected);

    StatusOr<int> target;
    target = source;

    EXPECT_FALSE(target.ok());
    EXPECT_EQ(expected, target.status());

    EXPECT_FALSE(source.ok());
    EXPECT_EQ(expected, source.status());
  }

  // Move assignment
  {
    const Status expected = CancelledError("");
    StatusOr<int> source(expected);

    StatusOr<int> target;
    target = std::move(source);

    EXPECT_FALSE(target.ok());
    EXPECT_EQ(expected, target.status());

    EXPECT_FALSE(source.ok());
    EXPECT_GE(static_cast<int>(source.status().code()), 0);
  }
}

TEST(StatusOr, TestAssignmentStatusOKConverting) {
  // Copy assignment
  {
    const int kI = 4;
    StatusOr<int> source(kI);

    StatusOr<double> target;
    target = source;

    ASSERT_TRUE(target.ok());
    EXPECT_OK(target.status());
    EXPECT_DOUBLE_EQ(kI, target.ValueOrDie());

    ASSERT_TRUE(source.ok());
    EXPECT_OK(source.status());
    EXPECT_DOUBLE_EQ(kI, source.ValueOrDie());
  }

  // Move assignment
  {
    const auto p = new int(17);
    StatusOr<std::unique_ptr<int>> source(absl::WrapUnique(p));

    StatusOr<std::shared_ptr<int>> target;
    target = std::move(source);

    ASSERT_TRUE(target.ok());
    EXPECT_OK(target.status());
    EXPECT_EQ(p, target.ValueOrDie().get());

    ASSERT_TRUE(source.ok());
    EXPECT_OK(source.status());
    EXPECT_EQ(nullptr, source.ValueOrDie().get());
  }
}

TEST(StatusOr, TestAssignmentStatusNotOkConverting) {
  // Copy assignment
  {
    const Status expected = CancelledError("");
    StatusOr<int> source(expected);

    StatusOr<double> target;
    target = source;

    EXPECT_FALSE(target.ok());
    EXPECT_EQ(expected, target.status());

    EXPECT_FALSE(source.ok());
    EXPECT_EQ(expected, source.status());
  }

  // Move assignment
  {
    const Status expected = CancelledError("");
    StatusOr<int> source(expected);

    StatusOr<double> target;
    target = std::move(source);

    EXPECT_FALSE(target.ok());
    EXPECT_EQ(expected, target.status());

    EXPECT_FALSE(source.ok());
    EXPECT_GE(static_cast<int>(source.status().code()), 0);
  }
}

TEST(StatusOr, SelfAssignment) {
  // Copy-assignment, status OK
  {
    // A std::string long enough that it's likely to defeat any inline representation
    // optimization.
    const std::string long_str(128, 'a');

    StatusOr<std::string> so = long_str;
    so = *&so;

    ASSERT_TRUE(so.ok());
    EXPECT_OK(so.status());
    EXPECT_EQ(long_str, so.ValueOrDie());
  }

  // Copy-assignment, error status
  {
    StatusOr<int> so = NotFoundError("taco");
    so = *&so;

    EXPECT_FALSE(so.ok());
    EXPECT_THAT(so, StatusIs(NOT_FOUND, "taco"));
  }

  // Move-assignment with copyable type, status OK
  {
    StatusOr<int> so = 17;

    // Fool the compiler, which otherwise complains.
    auto& same = so;
    so = std::move(same);

    ASSERT_TRUE(so.ok());
    EXPECT_OK(so.status());
    EXPECT_EQ(17, so.ValueOrDie());
  }

  // Move-assignment with copyable type, error status
  {
    StatusOr<int> so = NotFoundError("taco");

    // Fool the compiler, which otherwise complains.
    auto& same = so;
    so = std::move(same);

    EXPECT_FALSE(so.ok());
    EXPECT_THAT(so, StatusIs(NOT_FOUND, "taco"));
  }

  // Move-assignment with non-copyable type, status OK
  {
    const auto raw = new int(17);
    StatusOr<std::unique_ptr<int>> so = absl::WrapUnique(raw);

    // Fool the compiler, which otherwise complains.
    auto& same = so;
    so = std::move(same);

    ASSERT_TRUE(so.ok());
    EXPECT_OK(so.status());
    EXPECT_EQ(raw, so.ValueOrDie().get());
  }

  // Move-assignment with non-copyable type, error status
  {
    StatusOr<std::unique_ptr<int>> so = NotFoundError("taco");

    // Fool the compiler, which otherwise complains.
    auto& same = so;
    so = std::move(same);

    EXPECT_FALSE(so.ok());
    EXPECT_THAT(so, StatusIs(NOT_FOUND, "taco"));
  }
}

TEST(StatusOr, TestStatus) {
  StatusOr<int> good(4);
  EXPECT_TRUE(good.ok());
  StatusOr<int> bad(CancelledError(""));
  EXPECT_FALSE(bad.ok());
  EXPECT_TRUE(bad.status().code() == CANCELLED);
}

TEST(StatusOr, OperatorBoolSugarForOk) {
  StatusOr<int> good(4);
  EXPECT_TRUE(good);
  EXPECT_EQ(good.ok(), static_cast<bool>(good));

  StatusOr<int> bad = CancelledError("");
  EXPECT_FALSE(bad);
  EXPECT_EQ(bad.ok(), static_cast<bool>(bad));
}

TEST(StatusOr, OperatorStarRefQualifiers) {
  static_assert(std::is_same<const int&,
                             decltype(*std::declval<const StatusOr<int>&>())>(),
                "Unexpected ref-qualifiers");
  static_assert(std::is_same<int&, decltype(*std::declval<StatusOr<int>&>())>(),
                "Unexpected ref-qualifiers");
  static_assert(
      std::is_same<const int&&,
                   decltype(*std::declval<const StatusOr<int>&&>())>(),
      "Unexpected ref-qualifiers");
  static_assert(
      std::is_same<int&&, decltype(*std::declval<StatusOr<int>&&>())>(),
      "Unexpected ref-qualifiers");
}

TEST(StatusOr, OperatorStar) {
  const StatusOr<std::string> const_lvalue("hello");
  EXPECT_EQ("hello", *const_lvalue);

  StatusOr<std::string> lvalue("hello");
  EXPECT_EQ("hello", *lvalue);

  // Note: Recall that std::move() is equivalent to a static_cast to an rvalue
  // reference type.
  const StatusOr<std::string> const_rvalue("hello");
  EXPECT_EQ("hello", *std::move(const_rvalue));  // NOLINT

  StatusOr<std::string> rvalue("hello");
  EXPECT_EQ("hello", *std::move(rvalue));
}

TEST(StatusOr, OperatorArrowQualifiers) {
  static_assert(
      std::is_same<const int*,
                   decltype(
                       std::declval<const StatusOr<int>&>().operator->())>(),
      "Unexpected qualifiers");
  static_assert(
      std::is_same<int*,
                   decltype(std::declval<StatusOr<int>&>().operator->())>(),
      "Unexpected qualifiers");
  static_assert(
      std::is_same<const int*,
                   decltype(
                       std::declval<const StatusOr<int>&&>().operator->())>(),
      "Unexpected qualifiers");
  static_assert(
      std::is_same<int*,
                   decltype(std::declval<StatusOr<int>&&>().operator->())>(),
      "Unexpected qualifiers");
}

TEST(StatusOr, OperatorArrow) {
  const StatusOr<std::string> const_lvalue("hello");
  EXPECT_EQ(std::string("hello"), const_lvalue->c_str());

  StatusOr<std::string> lvalue("hello");
  EXPECT_EQ(std::string("hello"), lvalue->c_str());
}

TEST(StatusOr, RValueStatus) {
  StatusOr<int> so(NotFoundError("taco"));
  const Status s = std::move(so).status();

  EXPECT_THAT(s, StatusIs(NOT_FOUND, "taco"));

  // Check that !ok() still implies !status().ok(), even after moving out of the
  // object. See the note on the rvalue ref-qualified status method.
  EXPECT_FALSE(so.ok());  // NOLINT
  EXPECT_FALSE(so.status().ok());
  EXPECT_GE(static_cast<int>(so.status().code()), 0);
  EXPECT_EQ(so.status().message(), "");
}

TEST(StatusOr, TestValue) {
  const int kI = 4;
  StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestValueConst) {
  const int kI = 4;
  const StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.ValueOrDie());
}

TEST(StatusOr, TestPointerStatusCtor) {
  StatusOr<int*> thing(CancelledError(""));
  EXPECT_FALSE(thing.ok());
  EXPECT_TRUE(thing.status().code() == CANCELLED);
}

TEST(StatusOrDeathTest, TestPointerStatusCtorStatusOk) {
  EXPECT_DEBUG_DEATH(
      {
        StatusOr<int*> thing(OkStatus());
        // In optimized mode, we are actually going to get INTERNAL for
        // status here, rather than crashing, so check that.
        EXPECT_FALSE(thing.ok());
        EXPECT_TRUE(thing.status().code() == INTERNAL);
      },
      "An OK status is not a valid constructor argument");
}

TEST(StatusOr, TestPointerValueCtor) {
  const int kI = 4;

  // Construction from a non-null pointer
  {
    StatusOr<const int*> so(&kI);
    EXPECT_TRUE(so.ok());
    EXPECT_OK(so.status());
    EXPECT_EQ(&kI, so.ValueOrDie());
  }

  // Construction from a null pointer constant
  {
    StatusOr<const int*> so(nullptr);
    EXPECT_TRUE(so.ok());
    EXPECT_OK(so.status());
    EXPECT_EQ(nullptr, so.ValueOrDie());
  }

  // Construction from a non-literal null pointer
  {
    const int* const p = nullptr;

    StatusOr<const int*> so(p);
    EXPECT_TRUE(so.ok());
    EXPECT_OK(so.status());
    EXPECT_EQ(nullptr, so.ValueOrDie());
  }
}

TEST(StatusOr, TestPointerCopyCtorStatusOk) {
  const int kI = 0;
  StatusOr<const int*> original(&kI);
  StatusOr<const int*> copy(original);
  EXPECT_OK(copy.status());
  EXPECT_EQ(original.ValueOrDie(), copy.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOk) {
  StatusOr<int*> original(CancelledError(""));
  StatusOr<int*> copy(original);
  EXPECT_TRUE(copy.status().code() == CANCELLED);
}

TEST(StatusOr, TestPointerCopyCtorStatusOKConverting) {
  Derived derived;
  StatusOr<Derived*> original(&derived);
  StatusOr<Base2*> copy(original);
  EXPECT_OK(copy.status());
  EXPECT_EQ(static_cast<const Base2*>(original.ValueOrDie()),
            copy.ValueOrDie());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOkConverting) {
  StatusOr<Derived*> original(CancelledError(""));
  StatusOr<Base2*> copy(original);
  EXPECT_TRUE(copy.status().code() == CANCELLED);
}

TEST(StatusOr, TestPointerAssignmentStatusOk) {
  const int kI = 0;
  StatusOr<const int*> source(&kI);
  StatusOr<const int*> target;
  target = source;
  EXPECT_OK(target.status());
  EXPECT_EQ(source.ValueOrDie(), target.ValueOrDie());
}

TEST(StatusOr, TestPointerAssignmentStatusNotOk) {
  StatusOr<int*> source(CancelledError(""));
  StatusOr<int*> target;
  target = source;
  EXPECT_TRUE(target.status().code() == CANCELLED);
}

TEST(StatusOr, TestPointerAssignmentStatusOKConverting) {
  Derived derived;
  StatusOr<Derived*> source(&derived);
  StatusOr<Base2*> target;
  target = source;
  EXPECT_OK(target.status());
  EXPECT_EQ(static_cast<const Base2*>(source.ValueOrDie()),
            target.ValueOrDie());
}

TEST(StatusOr, TestPointerAssignmentStatusNotOkConverting) {
  StatusOr<Derived*> source(CancelledError(""));
  StatusOr<Base2*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestPointerStatus) {
  const int kI = 0;
  StatusOr<const int*> good(&kI);
  EXPECT_TRUE(good.ok());
  StatusOr<const int*> bad(CancelledError(""));
  EXPECT_TRUE(bad.status().code() == CANCELLED);
}

TEST(StatusOr, TestPointerValue) {
  const int kI = 0;
  StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, TestPointerValueConst) {
  const int kI = 0;
  const StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.ValueOrDie());
}

TEST(StatusOr, StatusOrVectorOfUniquePointerCanReserveAndResize) {
  using EvilType = std::vector<std::unique_ptr<int>>;
  static_assert(std::is_copy_constructible<EvilType>::value, "");
  std::vector<StatusOr<EvilType>> v(5);
  v.reserve(v.capacity() + 10);
  v.resize(v.capacity() + 10);
}

TEST(StatusOr, ConstPayload) {
  // A reduced version of a problematic type found in the wild. All of the
  // operations below should compile.
  StatusOr<const int> a;

  // Copy-construction
  StatusOr<const int> b(a);

  // Copy-assignment
  b = a;

  // Move-construction
  StatusOr<const int> c(std::move(a));

  // Move-assignment
  b = std::move(a);
}

TEST(StatusOr, MapToStatusOrUniquePtr) {
  // A reduced version of a problematic type found in the wild. All of the
  // operations below should compile.
  using MapType = std::map<std::string, StatusOr<std::unique_ptr<int>>>;

  MapType a;

  // Move-construction
  MapType b(std::move(a));

  // Move-assignment
  a = std::move(b);
}

TEST(StatusOrDeathTest, TestPointerValueNotOk) {
  StatusOr<int*> thing(CancelledError(""));
  EXPECT_DEATH(thing.ValueOrDie(), "ancelled");
}

TEST(StatusOrDeathTest, TestPointerValueNotOkConst) {
  const StatusOr<int*> thing(CancelledError(""));
  EXPECT_DEATH(thing.ValueOrDie(), "ancelled");
}

static StatusOr<int> MakeStatus() { return 100; }

TEST(StatusOr, TestIgnoreError) { MakeStatus().IgnoreError(); }

}  // namespace
}  // namespace base
}  // namespace differential_privacy
