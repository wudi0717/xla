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

#include "xla/service/collective_permute_cycle.h"

#include <utility>

#include <gtest/gtest.h>
#include "xla/service/source_target_pairs.h"

namespace xla {
namespace collective_permute_cycle {
namespace {

struct CanonicalCycle {
  SourceTargetPairs cycle;
  SourceTargetPairs main_edge;
  SourceTargetPairs back_edge;
};

class CollectivePermuteCycleTest : public ::testing::Test {
 protected:
  CanonicalCycle fwd2_ = {.cycle = SourceTargetPairs({{0, 1}, {1, 0}}),
                          .main_edge = SourceTargetPairs({{0, 1}}),
                          .back_edge = SourceTargetPairs({{1, 0}})};

  CanonicalCycle bwd2_ = {.cycle = SourceTargetPairs({{0, 1}, {1, 0}}),
                          .main_edge = SourceTargetPairs({{1, 0}}),
                          .back_edge = SourceTargetPairs({{0, 1}})};

  CanonicalCycle fwd4_ = {
      .cycle = SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 0}}),
      .main_edge = SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}}),
      .back_edge = SourceTargetPairs({{3, 0}})};

  CanonicalCycle bwd4_ = {
      .cycle = SourceTargetPairs({{0, 3}, {1, 0}, {2, 1}, {3, 2}}),
      .main_edge = SourceTargetPairs({{1, 0}, {2, 1}, {3, 2}}),
      .back_edge = SourceTargetPairs({{0, 3}})};
};

TEST_F(CollectivePermuteCycleTest, HasCycles) {
  EXPECT_TRUE(HasCycles(fwd2_.cycle));
  EXPECT_TRUE(HasCycles(bwd2_.cycle));
  EXPECT_TRUE(HasCycles(fwd4_.cycle));
  EXPECT_TRUE(HasCycles(bwd4_.cycle));
  EXPECT_FALSE(HasCycles(SourceTargetPairs({{1, 2}, {2, 3}, {3, 0}})));
  EXPECT_FALSE(HasCycles(SourceTargetPairs({{1, 2}})));
}

TEST_F(CollectivePermuteCycleTest, IsForwardCycleSplitForm) {
  EXPECT_TRUE(IsForwardCycle(fwd2_.back_edge, fwd2_.main_edge));
  EXPECT_TRUE(IsForwardCycle(fwd4_.back_edge, fwd4_.main_edge));
  EXPECT_FALSE(IsForwardCycle(bwd2_.back_edge, bwd2_.main_edge));
  EXPECT_FALSE(IsForwardCycle(bwd4_.back_edge, bwd4_.main_edge));
}

TEST_F(CollectivePermuteCycleTest, IsBackwardCycleSplitForm) {
  EXPECT_TRUE(IsBackwardCycle(bwd2_.back_edge, bwd2_.main_edge));
  EXPECT_TRUE(IsBackwardCycle(bwd4_.back_edge, bwd4_.main_edge));
  EXPECT_FALSE(IsBackwardCycle(fwd2_.back_edge, fwd2_.main_edge));
  EXPECT_FALSE(IsBackwardCycle(fwd4_.back_edge, fwd4_.main_edge));
}

TEST_F(CollectivePermuteCycleTest, SplitEdges) {
  EXPECT_EQ(SplitEdges(fwd4_.cycle, CycleType::kForward),
            std::make_pair(fwd4_.back_edge, fwd4_.main_edge));
  EXPECT_EQ(SplitEdges(bwd4_.cycle, CycleType::kBackward),
            std::make_pair(bwd4_.back_edge, bwd4_.main_edge));
}

TEST_F(CollectivePermuteCycleTest, DetectCycleTypes) {
  EXPECT_EQ(GetCycleType(fwd2_.cycle), CycleType::kForward);
  EXPECT_EQ(GetCycleType(fwd4_.cycle), CycleType::kForward);
  EXPECT_EQ(GetCycleType(bwd4_.cycle), CycleType::kBackward);
  EXPECT_EQ(GetCycleType(SourceTargetPairs()), CycleType::kNone);
  EXPECT_EQ(GetCycleType(SourceTargetPairs({{0, 1}})), CycleType::kNone);
}

}  // namespace
}  // namespace collective_permute_cycle
}  // namespace xla
