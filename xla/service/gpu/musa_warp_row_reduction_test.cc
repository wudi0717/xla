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

#include "xla/service/gpu/musa_warp_row_reduction.h"

#include <memory>
#include <optional>
#include <string>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class MusaWarpRowReductionTest : public HloTestBase {
 protected:
  auto ParseCanonicalMixedTupleModule() {
    return ParseAndReturnVerifiedModule(R"(
HloModule canonical_mixed_tuple_row_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  full = f32[2,3,17]{2,1,0} multiply(p0, p0)
  sum = f32[2,3]{1,0} reduce(p0, zero), dimensions={2}, to_apply=add
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0})
      tuple(sum, full, product)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0})
      fusion(p0), kind=kInput, calls=fused_computation
}
)"));
  }

  void SetCanonicalMixedFullOutputLayout(HloModule* module,
                                         const Layout& layout) {
    HloInstruction* fusion =
        module->entry_computation()->root_instruction();
    HloInstruction* root = fusion->fused_expression_root();
    HloInstruction* full_output = root->mutable_operand(1);
    *full_output->mutable_shape()->mutable_layout() = layout;
    *root->mutable_shape()->mutable_tuple_shapes(1) = full_output->shape();
    *fusion->mutable_shape()->mutable_tuple_shapes(1) = full_output->shape();
  }

  std::optional<MusaWarpRowReductionMatch> MatchEntryFusion(
      HloModule* module) {
    HloInstruction* fusion =
        module->entry_computation()->root_instruction();
    return MatchMusaWarpRowReductionFusion(
        fusion, fusion->fused_instructions_computation());
  }

  std::optional<MusaTupleWarpRowReductionMatch> MatchEntryTupleFusion(
      HloModule* module) {
    HloInstruction* fusion =
        module->entry_computation()->root_instruction();
    return MatchMusaTupleWarpRowReductionFusion(
        fusion, fusion->fused_instructions_computation());
  }

  std::optional<MusaMixedTupleWarpRowReductionMatch>
  MatchEntryMixedTupleFusion(HloModule* module,
                             std::string* failure_reason = nullptr) {
    HloInstruction* fusion =
        module->entry_computation()->root_instruction();
    return MatchMusaMixedTupleWarpRowReductionFusion(
        fusion, fusion->fused_instructions_computation(), failure_reason);
  }
};

TEST_F(MusaWarpRowReductionTest,
       MatchesMixedTupleAndPreservesOutputIndices) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule mixed_tuple_row_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  squared = f32[2,3,17]{2,1,0} multiply(p0, p0)
  sum = f32[2,3]{1,0} reduce(squared, zero), dimensions={2}, to_apply=add
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0})
      tuple(sum, squared, product)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0})
      fusion(p0), kind=kInput, calls=fused_computation
}
)"));

  auto match = MatchEntryMixedTupleFusion(module.get());
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->rows, 6);
  EXPECT_EQ(match->width, 17);
  ASSERT_EQ(match->reductions.size(), 2);
  EXPECT_EQ(match->reductions[0].tuple_index, 0);
  EXPECT_EQ(match->reductions[1].tuple_index, 2);
  ASSERT_EQ(match->elementwise_outputs.size(), 1);
  EXPECT_EQ(match->elementwise_outputs[0].tuple_index, 1);
  EXPECT_EQ(match->elementwise_outputs[0].instruction->name(), "squared");
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleRootShapeMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseCanonicalMixedTupleModule());
  HloInstruction* root = module->entry_computation()
                             ->root_instruction()
                             ->fused_expression_root();
  root->mutable_shape()
      ->mutable_tuple_shapes(1)
      ->set_dynamic_dimension(0, true);

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest,
       RejectsMixedTupleReductionsWithDifferentWidthsAndDataShapes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule mixed_tuple_mismatched_reduction_width

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  p1 = f32[2,3,9]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  full = f32[2,3,17]{2,1,0} multiply(p0, p0)
  sum0 = f32[2,3]{1,0} reduce(full, zero), dimensions={2}, to_apply=add
  sum1 = f32[2,3]{1,0} reduce(p1, zero), dimensions={2}, to_apply=add
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0})
      tuple(sum0, full, sum1)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  p1 = f32[2,3,9]{2,1,0} parameter(1)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0})
      fusion(p0, p1), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest,
       RejectsMixedTupleReductionsWithDifferentLogicalRowShapes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule mixed_tuple_mismatched_reduction_rows

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  p1 = f32[6,17]{1,0} parameter(1)
  zero = f32[] constant(0)
  full = f32[2,3,17]{2,1,0} multiply(p0, p0)
  sum0 = f32[2,3]{1,0} reduce(full, zero), dimensions={2}, to_apply=add
  sum1 = f32[6]{0} reduce(p1, zero), dimensions={1}, to_apply=add
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[6]{0})
      tuple(sum0, full, sum1)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  p1 = f32[6,17]{1,0} parameter(1)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[6]{0})
      fusion(p0, p1), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleSparseFullOutputLayout) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseCanonicalMixedTupleModule());
  SetCanonicalMixedFullOutputLayout(
      module.get(), LayoutUtil::MakeLayout(
                        {2, 1, 0}, {DIM_DENSE, DIM_DENSE, DIM_COMPRESSED}));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest,
       RejectsMixedTupleNonDim0MajorFullOutputLayout) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseCanonicalMixedTupleModule());
  SetCanonicalMixedFullOutputLayout(module.get(),
                                    LayoutUtil::MakeLayout({0, 1, 2}));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest,
       RejectsMixedTupleNonElementwiseFullOutput) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule non_elementwise_full_output_mixed_tuple

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  sum = f32[2,3]{1,0} reduce(p0, zero), dimensions={2}, to_apply=add
  full = f32[2,3,17]{2,1,0} reverse(p0), dimensions={2}
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0})
      tuple(sum, full, product)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0})
      fusion(p0), kind=kInput, calls=fused_computation
}
)"));

  std::string failure_reason;
  EXPECT_FALSE(
      MatchEntryMixedTupleFusion(module.get(), &failure_reason).has_value());
  EXPECT_EQ(failure_reason, "full_output_not_elementwise");
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleWithoutFullOutput) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule pure_tuple_row_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  sum = f32[2,3]{1,0} reduce(p0, zero), dimensions={2}, to_apply=add
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(sum, product)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3]{1,0})
      fusion(p0), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleWithOneReduction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule one_reduction_mixed_tuple

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  squared = f32[2,3,17]{2,1,0} multiply(p0, p0)
  sum = f32[2,3]{1,0} reduce(squared, zero), dimensions={2}, to_apply=add
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}) tuple(sum, squared)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0})
      fusion(p0), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleWithMismatchedFullShape) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule mismatched_full_shape_mixed_tuple

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  p1 = f32[2,3,9]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  sum = f32[2,3]{1,0} reduce(p0, zero), dimensions={2}, to_apply=add
  full = f32[2,3,9]{2,1,0} multiply(p1, p1)
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3,9]{2,1,0}, f32[2,3]{1,0})
      tuple(sum, full, product)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  p1 = f32[2,3,9]{2,1,0} parameter(1)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3,9]{2,1,0}, f32[2,3]{1,0})
      fusion(p0, p1), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleWithNonF32FullOutput) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule non_f32_full_output_mixed_tuple

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  sum = f32[2,3]{1,0} reduce(p0, zero), dimensions={2}, to_apply=add
  full = f16[2,3,17]{2,1,0} convert(p0)
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f16[2,3,17]{2,1,0}, f32[2,3]{1,0})
      tuple(sum, full, product)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = (f32[2,3]{1,0}, f16[2,3,17]{2,1,0}, f32[2,3]{1,0})
      fusion(p0), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleWithDynamicFullOutput) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseCanonicalMixedTupleModule());

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  HloInstruction* root = fusion->fused_expression_root();
  root->mutable_operand(1)->mutable_shape()
      ->set_dynamic_dimension(0, true);
  root->mutable_shape()
      ->mutable_tuple_shapes(1)
      ->set_dynamic_dimension(0, true);
  fusion->mutable_shape()
      ->mutable_tuple_shapes(1)
      ->set_dynamic_dimension(0, true);

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleWithDuplicateReduction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule duplicate_reduction_mixed_tuple

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  squared = f32[2,3,17]{2,1,0} multiply(p0, p0)
  sum = f32[2,3]{1,0} reduce(squared, zero), dimensions={2}, to_apply=add
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0}, f32[2,3]{1,0})
      tuple(sum, squared, sum, product)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3]{1,0}, f32[2,3]{1,0})
      fusion(p0), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsMixedTupleWithMoreThanSixteenOutputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule excessive_arity_mixed_tuple

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  full = f32[2,3,17]{2,1,0} multiply(p0, p0)
  sum = f32[2,3]{1,0} reduce(full, zero), dimensions={2}, to_apply=add
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0})
      tuple(sum, product, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0}, f32[2,3,17]{2,1,0})
      fusion(p0), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryMixedTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest,
       MatchesTwoOutputAddMultiplyTupleReduction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule tuple_add_multiply_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] parameter(1)
  one = f32[] parameter(2)
  squared = f32[2,3,17]{2,1,0} multiply(p0, p0)
  sum = f32[2,3]{1,0} reduce(squared, zero), dimensions={2}, to_apply=add
  product = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(sum, product)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  one = f32[] constant(1)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3]{1,0}) fusion(p0, zero, one), kind=kInput, calls=fused_computation
}
)"));

  std::optional<MusaTupleWarpRowReductionMatch> match =
      MatchEntryTupleFusion(module.get());
  ASSERT_TRUE(match.has_value());
  ASSERT_EQ(match->reductions.size(), 2);
  EXPECT_EQ(match->rows, 6);
  EXPECT_EQ(match->width, 17);
  EXPECT_EQ(match->reductions[0].kind, MusaWarpRowReductionKind::kAdd);
  EXPECT_EQ(match->reductions[1].kind,
            MusaWarpRowReductionKind::kMultiply);
}

TEST_F(MusaWarpRowReductionTest, RejectsTupleReductionsWithDifferentWidths) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule tuple_mismatched_width_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  p1 = f32[2,3,9]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  sum0 = f32[2,3]{1,0} reduce(p0, zero), dimensions={2}, to_apply=add
  sum1 = f32[2,3]{1,0} reduce(p1, zero), dimensions={2}, to_apply=add
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(sum0, sum1)
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  p1 = f32[2,3,9]{2,1,0} parameter(1)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3]{1,0}) fusion(p0, p1), kind=kInput, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryTupleFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, MatchesF32LastDimensionAddReduction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule add_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,33]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT reduce = f32[2]{0} reduce(p0, zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[2,33]{1,0} parameter(0)
  ROOT fusion = f32[2]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));

  std::optional<MusaWarpRowReductionMatch> match =
      MatchEntryFusion(module.get());
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->rows, 2);
  EXPECT_EQ(match->width, 33);
  EXPECT_EQ(match->kind, MusaWarpRowReductionKind::kAdd);
}

TEST_F(MusaWarpRowReductionTest, MatchesF32LastDimensionMultiplyReduction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule multiply_reduction

multiply {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT product = f32[] multiply(rhs, lhs)
}

fused_computation {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  one = f32[] constant(1)
  ROOT reduce = f32[2,3]{1,0} reduce(p0, one), dimensions={2}, to_apply=multiply
}

ENTRY main {
  p0 = f32[2,3,17]{2,1,0} parameter(0)
  ROOT fusion = f32[2,3]{1,0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));

  std::optional<MusaWarpRowReductionMatch> match =
      MatchEntryFusion(module.get());
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->rows, 6);
  EXPECT_EQ(match->width, 17);
  EXPECT_EQ(match->kind, MusaWarpRowReductionKind::kMultiply);
}

TEST_F(MusaWarpRowReductionTest, RejectsReducerThatReusesOneParameter) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule invalid_reducer_parameters

bad_add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, lhs)
}

fused_computation {
  p0 = f32[2,33]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT reduce = f32[2]{0} reduce(p0, zero), dimensions={1}, to_apply=bad_add
}

ENTRY main {
  p0 = f32[2,33]{1,0} parameter(0)
  ROOT fusion = f32[2]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsMissingLayout) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule missing_layout

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,33]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT reduce = f32[2]{0} reduce(p0, zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[2,33]{1,0} parameter(0)
  ROOT fusion = f32[2]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));
  HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  fusion->fused_expression_root()->mutable_shape()->clear_layout();

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsNegativeZeroAddIdentity) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule negative_zero_identity

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,33]{1,0} parameter(0)
  negative_zero = f32[] constant(-0.0)
  ROOT reduce = f32[2]{0} reduce(p0, negative_zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[2,33]{1,0} parameter(0)
  ROOT fusion = f32[2]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsWrongAddIdentity) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule wrong_add_identity

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,33]{1,0} parameter(0)
  one = f32[] constant(1)
  ROOT reduce = f32[2]{0} reduce(p0, one), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[2,33]{1,0} parameter(0)
  ROOT fusion = f32[2]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsTupleOutput) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule tuple_output

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,33]{1,0} parameter(0)
  zero = f32[] constant(0)
  reduce = f32[2]{0} reduce(p0, zero), dimensions={1}, to_apply=add
  ROOT tuple = (f32[2]{0}) tuple(reduce)
}

ENTRY main {
  p0 = f32[2,33]{1,0} parameter(0)
  ROOT fusion = (f32[2]{0}) fusion(p0), kind=kLoop, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsDynamicShape) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule dynamic_shape

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,33]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT reduce = f32[2]{0} reduce(p0, zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[2,33]{1,0} parameter(0)
  ROOT fusion = f32[2]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));
  HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  fusion->fused_expression_root()->operand(0)->mutable_shape()
      ->set_dynamic_dimension(0, true);

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsNonLastReductionDimension) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule non_last_dimension

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,33]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT reduce = f32[33]{0} reduce(p0, zero), dimensions={0}, to_apply=add
}

ENTRY main {
  p0 = f32[2,33]{1,0} parameter(0)
  ROOT fusion = f32[33]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST_F(MusaWarpRowReductionTest, RejectsWidthAboveKernelLimit) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule excessive_width

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[2,1025]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT reduce = f32[2]{0} reduce(p0, zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[2,1025]{1,0} parameter(0)
  ROOT fusion = f32[2]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)"));

  EXPECT_FALSE(MatchEntryFusion(module.get()).has_value());
}

TEST(MusaWarpRowReductionLaunchConfigTest,
     UsesRequestedThreadsForSegmentedWidth768Reduction) {
  std::optional<MusaWarpRowReductionLaunchConfig> config =
      ResolveMusaWarpRowReductionLaunchConfig(
          /*width=*/768, /*warp_size=*/32, /*threads_per_block_limit=*/1024,
          /*requested_threads_per_block=*/256);

  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->threads_per_block, 256);
  EXPECT_EQ(config->warps_per_block, 8);
  EXPECT_EQ(config->elements_per_thread, 3);
}

TEST(MusaWarpRowReductionLaunchConfigTest,
     PreservesOneElementPerThreadWhenNoOverrideIsRequested) {
  std::optional<MusaWarpRowReductionLaunchConfig> config =
      ResolveMusaWarpRowReductionLaunchConfig(
          /*width=*/768, /*warp_size=*/32, /*threads_per_block_limit=*/1024,
          /*requested_threads_per_block=*/0);

  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->threads_per_block, 768);
  EXPECT_EQ(config->warps_per_block, 24);
  EXPECT_EQ(config->elements_per_thread, 1);
}

TEST(MusaWarpRowReductionLaunchConfigTest,
     RejectsRequestedThreadsThatAreNotWarpAligned) {
  EXPECT_FALSE(ResolveMusaWarpRowReductionLaunchConfig(
                   /*width=*/768, /*warp_size=*/32,
                   /*threads_per_block_limit=*/1024,
                   /*requested_threads_per_block=*/250)
                   .has_value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
