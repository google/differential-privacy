//
// Copyright 2021 Google LLC
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

#include "algorithms/internal/count-tree.h"

#include "base/testing/proto_matchers.h"
#include "base/testing/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

namespace differential_privacy {
namespace internal {
namespace {

using ::testing::HasSubstr;
using ::differential_privacy::base::testing::StatusIs;

TEST(CountTreeTest, NumberOfNodes) {
  CountTree test(3, 5);
  EXPECT_EQ(test.GetNumberOfNodes(), 1 + 5 + 25 + 125);
  CountTree test2 = CountTree(4, 9);
  EXPECT_EQ(test2.GetNumberOfNodes(), 1 + 9 + 81 + 729 + 6561);
}

TEST(CountTreeTest, NumberOfLeaves) {
  CountTree test(3, 5);
  EXPECT_EQ(test.GetNumberOfLeaves(), 125);
  CountTree test2 = CountTree(4, 9);
  EXPECT_EQ(test2.GetNumberOfLeaves(), 6561);
}

TEST(CountTreeTest, GetNthLeaf) {
  CountTree test(3, 5);
  EXPECT_EQ(test.GetNthLeaf(0), 31);
  EXPECT_EQ(test.GetNthLeaf(5), 36);
  EXPECT_EQ(test.GetNthLeaf(18), 49);
}

TEST(CountTreeTest, ParentChildInverse) {
  CountTree test(5, 6);
  for (int i = 0; i < test.GetLeftMostLeaf(); ++i) {
    for (int child = test.LeftMostChild(i); child < test.RightMostChild(i);
         ++child) {
      EXPECT_EQ(test.Parent(child), i);
    }
  }
}

TEST(CountTreeTest, ParentChildExamples) {
  CountTree test(3, 5);
  EXPECT_EQ(test.LeftMostChild(0), 1);
  EXPECT_EQ(test.RightMostChild(0), 5);
  EXPECT_EQ(test.LeftMostChild(1), 6);
  EXPECT_EQ(test.RightMostChild(1), 10);
  EXPECT_EQ(test.LeftMostChild(8), 41);
  EXPECT_EQ(test.RightMostChild(8), 45);
  EXPECT_EQ(test.Parent(38), 7);
  EXPECT_EQ(test.Parent(8), 1);
  EXPECT_EQ(test.Parent(2), 0);
}

TEST(CountTreeTest, IsLeaf) {
  CountTree test(3, 5);
  EXPECT_FALSE(test.IsLeaf(0));
  EXPECT_FALSE(test.IsLeaf(1));
  EXPECT_FALSE(test.IsLeaf(6));
  EXPECT_FALSE(test.IsLeaf(30));
  EXPECT_TRUE(test.IsLeaf(31));
  EXPECT_TRUE(test.IsLeaf(155));
}

TEST(CountTreeTest, SubtreeQueries) {
  CountTree test(3, 5);
  EXPECT_EQ(test.LeftMostInSubtree(0), 31);
  EXPECT_EQ(test.RightMostInSubtree(0), 155);
  EXPECT_EQ(test.LeftMostInSubtree(1), 31);
  EXPECT_EQ(test.RightMostInSubtree(1), 55);
  EXPECT_EQ(test.LeftMostInSubtree(3), 81);
  EXPECT_EQ(test.RightMostInSubtree(3), 105);
  EXPECT_EQ(test.LeftMostInSubtree(82), 82);
  EXPECT_EQ(test.RightMostInSubtree(83), 83);
}

TEST(CountTreeTest, IncrementGet) {
  CountTree test(3, 5);
  test.IncrementNode(1);
  EXPECT_EQ(test.GetNodeCount(1), 1);
  EXPECT_EQ(test.GetNodeCount(2), 0);
  test.IncrementNode(8);
  test.IncrementNode(8);
  test.IncrementNode(8);
  EXPECT_EQ(test.GetNodeCount(8), 3);
}

TEST(CountTreeTest, IncrementNodeByGet) {
  CountTree test(3, 5);
  test.IncrementNode(1);
  test.IncrementNodeBy(1, 3);
  EXPECT_EQ(test.GetNodeCount(1), 4);
  test.IncrementNodeBy(1, 5);
  EXPECT_EQ(test.GetNodeCount(1), 9);
  test.IncrementNode(1);
  EXPECT_EQ(test.GetNodeCount(1), 10);
}

TEST(CountTreeTest, SerializeMerge) {
  CountTree test1(3, 5);
  test1.IncrementNode(1);
  test1.IncrementNode(8);
  test1.IncrementNode(8);

  CountTree test2(3, 5);
  EXPECT_OK(test2.Merge(test1.Serialize()));
  test1.IncrementNode(8);
  test2.IncrementNode(8);
  test1.IncrementNode(10);
  test2.IncrementNode(10);

  for (int i = test1.GetRoot(); i < test1.GetNumberOfNodes(); ++i) {
    EXPECT_EQ(test1.GetNodeCount(i), test2.GetNodeCount(i));
  }
}

TEST(CountTreeTest, MisatchMergeFails) {
  CountTree standard(3, 5);
  CountTree shorter(2, 5);
  CountTree wider(3, 6);
  EXPECT_THAT(shorter.Merge(standard.Serialize()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Height")));
  EXPECT_THAT(wider.Merge(standard.Serialize()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Branching")));
}

TEST(CountTreeTest, ClearNodes) {
  CountTree test1(3, 5);
  test1.IncrementNode(1);
  test1.IncrementNode(8);
  test1.IncrementNode(8);

  CountTree test2(3, 5);
  test1.ClearNodes();
  test1.IncrementNode(8);
  test2.IncrementNode(8);
  test1.IncrementNode(10);
  test2.IncrementNode(10);

  for (int i = test1.GetRoot(); i < test1.GetNumberOfNodes(); ++i) {
    EXPECT_EQ(test1.GetNodeCount(i), test2.GetNodeCount(i));
  }
}

}  // namespace
}  // namespace internal
}  // namespace differential_privacy
