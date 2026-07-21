/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/musa_reduction_chain.h"

#include <memory>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class MusaReductionChainTest : public HloTestBase {
 protected:
  std::optional<MusaReductionChainMatch> MatchEntryFusion(HloModule* module) {
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    return MatchMusaReductionChainFusion(
        fusion, fusion->fused_instructions_computation());
  }
};

TEST_F(MusaReductionChainTest, MatchesDependentMultiplyReductions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule dependent_multiply_chain

multiply_reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,8]{2,1,0} parameter(0)
  p1 = f32[2,3,8]{2,1,0} parameter(1)
  one = f32[] constant(1)
  first = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply_reducer
  first_broadcast = f32[2,3,8]{2,1,0} broadcast(first), dimensions={0,1}
  second_data = f32[2,3,8]{2,1,0} multiply(p1, first_broadcast)
  second = f32[2,3]{1,0} reduce(second_data, one), dimensions={2}, to_apply=multiply_reducer
  second_broadcast = f32[2,3,8]{2,1,0} broadcast(second), dimensions={0,1}
  ROOT result = f32[2,3,8]{2,1,0} add(first_broadcast, second_broadcast)
}

ENTRY main {
  p0 = f32[2,3,8]{2,1,0} parameter(0)
  p1 = f32[2,3,8]{2,1,0} parameter(1)
  ROOT fusion = f32[2,3,8]{2,1,0} fusion(p0, p1), kind=kInput, calls=fused_computation
}
)"));

  std::optional<MusaReductionChainMatch> match = MatchEntryFusion(module.get());
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->rows, 6);
  EXPECT_EQ(match->width, 8);
  EXPECT_EQ(match->first_reduce->name(), "first");
  EXPECT_EQ(match->second_reduce->name(), "second");
}

TEST_F(MusaReductionChainTest, RejectsIndependentReductions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule independent_multiply_reductions

multiply_reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,8]{2,1,0} parameter(0)
  p1 = f32[2,3,8]{2,1,0} parameter(1)
  one = f32[] constant(1)
  first = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply_reducer
  second = f32[2,3]{1,0} reduce(p1, one), dimensions={2}, to_apply=multiply_reducer
  first_broadcast = f32[2,3,8]{2,1,0} broadcast(first), dimensions={0,1}
  second_broadcast = f32[2,3,8]{2,1,0} broadcast(second), dimensions={0,1}
  ROOT result = f32[2,3,8]{2,1,0} add(first_broadcast, second_broadcast)
}

ENTRY main {
  p0 = f32[2,3,8]{2,1,0} parameter(0)
  p1 = f32[2,3,8]{2,1,0} parameter(1)
  ROOT fusion = f32[2,3,8]{2,1,0} fusion(p0, p1), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaReductionChainTest, RejectsRowRemappingBetweenReductions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule row_remapped_multiply_chain

multiply_reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,8]{2,1,0} parameter(0)
  p1 = f32[2,3,8]{2,1,0} parameter(1)
  one = f32[] constant(1)
  first = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply_reducer
  remapped = f32[2,3]{1,0} reverse(first), dimensions={0}
  first_broadcast = f32[2,3,8]{2,1,0} broadcast(remapped), dimensions={0,1}
  second_data = f32[2,3,8]{2,1,0} multiply(p1, first_broadcast)
  second = f32[2,3]{1,0} reduce(second_data, one), dimensions={2}, to_apply=multiply_reducer
  second_broadcast = f32[2,3,8]{2,1,0} broadcast(second), dimensions={0,1}
  ROOT result = f32[2,3,8]{2,1,0} add(first_broadcast, second_broadcast)
}

ENTRY main {
  p0 = f32[2,3,8]{2,1,0} parameter(0)
  p1 = f32[2,3,8]{2,1,0} parameter(1)
  ROOT fusion = f32[2,3,8]{2,1,0} fusion(p0, p1), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaReductionChainTest, RejectsRootThatDoesNotUseBothReductions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule unused_second_reduction

multiply_reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,8]{2,1,0} parameter(0)
  p1 = f32[2,3,8]{2,1,0} parameter(1)
  one = f32[] constant(1)
  first = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply_reducer
  first_broadcast = f32[2,3,8]{2,1,0} broadcast(first), dimensions={0,1}
  second_data = f32[2,3,8]{2,1,0} multiply(p1, first_broadcast)
  second = f32[2,3]{1,0} reduce(second_data, one), dimensions={2}, to_apply=multiply_reducer
  ROOT result = f32[2,3,8]{2,1,0} copy(first_broadcast)
}

ENTRY main {
  p0 = f32[2,3,8]{2,1,0} parameter(0)
  p1 = f32[2,3,8]{2,1,0} parameter(1)
  ROOT fusion = f32[2,3,8]{2,1,0} fusion(p0, p1), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaReductionChainTest, RejectsUnsafeDuplicateFusionOperandPath) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule duplicate_fusion_operand

fused_computation {
  safe = f32[2,3]{1,0} parameter(0)
  unsafe = f32[2,3]{1,0} parameter(1)
  unsafe_reversed = f32[2,3]{1,0} reverse(unsafe), dimensions={0}
  safe_broadcast = f32[2,3,8]{2,1,0} broadcast(safe), dimensions={0,1}
  unsafe_broadcast = f32[2,3,8]{2,1,0} broadcast(unsafe_reversed), dimensions={0,1}
  ROOT result = f32[2,3,8]{2,1,0} add(safe_broadcast, unsafe_broadcast)
}

ENTRY main {
  reduction = f32[2,3]{1,0} parameter(0)
  ROOT fusion = f32[2,3,8]{2,1,0} fusion(reduction, reduction), kind=kLoop, calls=fused_computation
}
)"));

  HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  EXPECT_FALSE(MusaFusionExpressionPreservesRows(
      fusion, fusion->fused_expression_root(), fusion->operand(0), 6));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
