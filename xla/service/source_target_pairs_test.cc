/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/source_target_pairs.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"

namespace xla {
namespace {

class SourceTargetPairsTest : public ::testing::Test {
 protected:
  SourceTargetPairs fwd2_ = SourceTargetPairs({{0, 1}, {1, 0}});
  SourceTargetPairs fwd4_ =
      SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 0}});
  SourceTargetPairs bwd4_ =
      SourceTargetPairs({{0, 3}, {1, 0}, {2, 1}, {3, 2}});
};

TEST_F(SourceTargetPairsTest, FromString) {
  ASSERT_TRUE(SourceTargetPairs::FromString("{{0,1},{1,0}}").ok());
  EXPECT_EQ(SourceTargetPairs::FromString("{{0,1},{1,0}}").value(), fwd2_);
  EXPECT_EQ(SourceTargetPairs::FromString("{{0,1},{1,2},{2,3},{3,0}}").value(),
            fwd4_);
  EXPECT_FALSE(SourceTargetPairs::FromString("{{0,1},{1}}").ok());
}

TEST_F(SourceTargetPairsTest, AbslStringify) {
  EXPECT_EQ(
      absl::StrFormat("Source Target Pairs: %v",
                      SourceTargetPairs::FromString("{{0,1},{1,0}}").value()),
      "Source Target Pairs: {{0,1},{1,0}}");
}

TEST_F(SourceTargetPairsTest, ToString) {
  EXPECT_EQ(fwd2_.ToString(), "{{0,1},{1,0}}");
  EXPECT_EQ(fwd4_.ToString(), "{{0,1},{1,2},{2,3},{3,0}}");
  EXPECT_EQ(bwd4_.ToString(), "{{0,3},{1,0},{2,1},{3,2}}");
}

TEST_F(SourceTargetPairsTest, GetMaxDeviceNum) {
  EXPECT_EQ(fwd2_.GetMaxDeviceNum(), 1);
  EXPECT_EQ(fwd4_.GetMaxDeviceNum(), 3);
  EXPECT_EQ(bwd4_.GetMaxDeviceNum(), 3);
  EXPECT_EQ(
      SourceTargetPairs({{0, 1}, {1, 2}, {2, 300}, {3, 4}}).GetMaxDeviceNum(),
      300);
}

}  // namespace
}  // namespace xla
