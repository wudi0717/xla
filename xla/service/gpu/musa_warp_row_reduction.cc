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

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

constexpr int64_t kMusaWarpRowReductionMaxWidth = 1024;
constexpr int64_t kMusaWarpRowReductionMaxDataElements =
    32 * 1024 * 1024;
constexpr int64_t kMusaMixedTupleWarpRowReductionMaxOutputs = 16;

bool MusaIsScalarF32Constant(const HloInstruction* fusion,
                             const HloInstruction* instr, double expected) {
  if (instr != nullptr && instr->opcode() == HloOpcode::kParameter) {
    const int64_t parameter_number = instr->parameter_number();
    if (fusion == nullptr || parameter_number < 0 ||
        parameter_number >= fusion->operand_count()) {
      return false;
    }
    instr = fusion->operand(parameter_number);
  }
  if (instr == nullptr || instr->opcode() != HloOpcode::kConstant ||
      !instr->shape().IsArray() || instr->shape().rank() != 0 ||
      instr->shape().element_type() != F32) {
    return false;
  }
  std::optional<double> value = instr->literal().GetAsDouble({});
  if (!value.has_value() || *value != expected) {
    return false;
  }
  return expected != 0.0 || !std::signbit(*value);
}

bool MusaIsScalarF32Parameter(const HloInstruction* instr) {
  return instr != nullptr && instr->opcode() == HloOpcode::kParameter &&
         instr->shape().IsArray() && instr->shape().rank() == 0 &&
         instr->shape().element_type() == F32;
}

std::optional<MusaWarpRowReductionMatch> MatchMusaWarpRowReduction(
    const HloInstruction* fusion, const HloInstruction* reduce) {
  if (reduce == nullptr || reduce->opcode() != HloOpcode::kReduce ||
      reduce->operand_count() != 2 || reduce->dimensions().size() != 1 ||
      !reduce->has_to_apply() || reduce->to_apply() == nullptr) {
    return std::nullopt;
  }

  const Shape& output_shape = reduce->shape();
  const HloInstruction* data = reduce->operand(0);
  const Shape& data_shape = data->shape();
  if (!output_shape.IsArray() || output_shape.element_type() != F32 ||
      output_shape.rank() < 1 || output_shape.rank() > 2 ||
      !data_shape.IsArray() || data_shape.element_type() != F32 ||
      data_shape.rank() != output_shape.rank() + 1 ||
      !output_shape.has_layout() || !data_shape.has_layout() ||
      !LayoutUtil::IsDenseArray(output_shape) ||
      !LayoutUtil::IsDenseArray(data_shape) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(data_shape.layout())) {
    return std::nullopt;
  }

  const int64_t reduce_dim = reduce->dimensions()[0];
  if (reduce_dim != output_shape.rank() ||
      data_shape.layout().minor_to_major(0) != reduce_dim) {
    return std::nullopt;
  }
  for (int64_t dim = 0; dim < output_shape.rank(); ++dim) {
    if (output_shape.dimensions(dim) <= 0 ||
        data_shape.dimensions(dim) != output_shape.dimensions(dim)) {
      return std::nullopt;
    }
  }

  const HloComputation* reducer = reduce->to_apply();
  const HloInstruction* reducer_root = reducer->root_instruction();
  if (reducer_root == nullptr || reducer_root->operand_count() != 2 ||
      reducer->num_parameters() != 2 ||
      !MusaIsScalarF32Parameter(reducer_root->operand(0)) ||
      !MusaIsScalarF32Parameter(reducer_root->operand(1))) {
    return std::nullopt;
  }
  const int64_t lhs_parameter =
      reducer_root->operand(0)->parameter_number();
  const int64_t rhs_parameter =
      reducer_root->operand(1)->parameter_number();
  if (!((lhs_parameter == 0 && rhs_parameter == 1) ||
        (lhs_parameter == 1 && rhs_parameter == 0))) {
    return std::nullopt;
  }

  MusaWarpRowReductionKind kind;
  double identity;
  if (reducer_root->opcode() == HloOpcode::kAdd) {
    kind = MusaWarpRowReductionKind::kAdd;
    identity = 0.0;
  } else if (reducer_root->opcode() == HloOpcode::kMultiply) {
    kind = MusaWarpRowReductionKind::kMultiply;
    identity = 1.0;
  } else {
    return std::nullopt;
  }
  if (!MusaIsScalarF32Constant(fusion, reduce->operand(1), identity)) {
    return std::nullopt;
  }

  const int64_t width = data_shape.dimensions(reduce_dim);
  if (data_shape.is_dynamic_dimension(reduce_dim) || width <= 0 ||
      width > kMusaWarpRowReductionMaxWidth) {
    return std::nullopt;
  }
  int64_t rows = 1;
  for (int64_t dim = 0; dim < output_shape.rank(); ++dim) {
    const int64_t dimension = output_shape.dimensions(dim);
    if (output_shape.is_dynamic_dimension(dim) ||
        data_shape.is_dynamic_dimension(dim) || dimension <= 0 ||
        rows > kMusaWarpRowReductionMaxDataElements / dimension) {
      return std::nullopt;
    }
    rows *= dimension;
  }
  if (rows > kMusaWarpRowReductionMaxDataElements / width) {
    return std::nullopt;
  }

  return MusaWarpRowReductionMatch{reduce, data, rows, width, kind};
}

}  // namespace

std::optional<MusaWarpRowReductionLaunchConfig>
ResolveMusaWarpRowReductionLaunchConfig(
    int64_t width, int64_t warp_size, int64_t threads_per_block_limit,
    int64_t requested_threads_per_block) {
  if (width <= 0 || warp_size <= 0 ||
      (warp_size & (warp_size - 1)) != 0 ||
      threads_per_block_limit < warp_size ||
      requested_threads_per_block < 0) {
    return std::nullopt;
  }

  int64_t threads_per_block = requested_threads_per_block;
  if (threads_per_block == 0) {
    threads_per_block = ((width + warp_size - 1) / warp_size) * warp_size;
  }
  if (threads_per_block < warp_size ||
      threads_per_block % warp_size != 0 ||
      threads_per_block > threads_per_block_limit) {
    return std::nullopt;
  }

  return MusaWarpRowReductionLaunchConfig{
      threads_per_block,
      /*warps_per_block=*/threads_per_block / warp_size,
      /*elements_per_thread=*/
      (width + threads_per_block - 1) / threads_per_block};
}

std::optional<MusaWarpRowReductionMatch> MatchMusaWarpRowReductionFusion(
    const HloInstruction* fusion, const HloComputation* called) {
  if (fusion == nullptr || called == nullptr ||
      fusion->opcode() != HloOpcode::kFusion ||
      fusion->fused_instructions_computation() != called ||
      fusion->shape().IsTuple()) {
    return std::nullopt;
  }
  const HloInstruction* root = called->root_instruction();
  std::optional<MusaWarpRowReductionMatch> match =
      MatchMusaWarpRowReduction(fusion, root);
  if (!match.has_value() ||
      !ShapeUtil::Equal(fusion->shape(), match->reduce->shape())) {
    return std::nullopt;
  }
  return match;
}

std::optional<MusaTupleWarpRowReductionMatch>
MatchMusaTupleWarpRowReductionFusion(const HloInstruction* fusion,
                                     const HloComputation* called) {
  if (fusion == nullptr || called == nullptr ||
      fusion->opcode() != HloOpcode::kFusion ||
      fusion->fused_instructions_computation() != called ||
      !fusion->shape().IsTuple() ||
      fusion->shape().tuple_shapes_size() != 2) {
    return std::nullopt;
  }
  const HloInstruction* root = called->root_instruction();
  if (root == nullptr || root->opcode() != HloOpcode::kTuple ||
      root->operand_count() != 2) {
    return std::nullopt;
  }

  MusaTupleWarpRowReductionMatch tuple_match;
  tuple_match.rows = -1;
  tuple_match.width = -1;
  for (int64_t index = 0; index < root->operand_count(); ++index) {
    std::optional<MusaWarpRowReductionMatch> reduction =
        MatchMusaWarpRowReduction(fusion, root->operand(index));
    if (!reduction.has_value() ||
        !ShapeUtil::Equal(fusion->shape().tuple_shapes(index),
                          reduction->reduce->shape()) ||
        (tuple_match.rows >= 0 && tuple_match.rows != reduction->rows) ||
        (tuple_match.width >= 0 && tuple_match.width != reduction->width)) {
      return std::nullopt;
    }
    tuple_match.rows = reduction->rows;
    tuple_match.width = reduction->width;
    tuple_match.reductions.push_back(*reduction);
  }
  if (tuple_match.reductions[0].reduce == tuple_match.reductions[1].reduce ||
      !ShapeUtil::SameDimensions(tuple_match.reductions[0].reduce->shape(),
                                 tuple_match.reductions[1].reduce->shape())) {
    return std::nullopt;
  }
  return tuple_match;
}

std::optional<MusaMixedTupleWarpRowReductionMatch>
MatchMusaMixedTupleWarpRowReductionFusion(const HloInstruction* fusion,
                                          const HloComputation* called,
                                          std::string* failure_reason) {
  if (failure_reason != nullptr) {
    failure_reason->clear();
  }
  auto fail = [&](const char* reason)
      -> std::optional<MusaMixedTupleWarpRowReductionMatch> {
    if (failure_reason != nullptr) {
      *failure_reason = reason;
    }
    return std::nullopt;
  };
  if (fusion == nullptr || called == nullptr ||
      fusion->opcode() != HloOpcode::kFusion ||
      fusion->fused_instructions_computation() != called) {
    return fail("invalid_fusion_or_called");
  }
  if (!fusion->shape().IsTuple()) {
    return fail("fusion_not_tuple");
  }
  const HloInstruction* root = called->root_instruction();
  if (root == nullptr || root->opcode() != HloOpcode::kTuple) {
    return fail("root_not_tuple");
  }
  if (!ShapeUtil::Equal(fusion->shape(), root->shape())) {
    return fail("root_shape_mismatch");
  }
  if (root->operand_count() != fusion->shape().tuple_shapes_size()) {
    return fail("tuple_arity_mismatch");
  }
  if (root->operand_count() > kMusaMixedTupleWarpRowReductionMaxOutputs) {
    return fail("tuple_arity_exceeds_limit");
  }

  MusaMixedTupleWarpRowReductionMatch mixed_match;
  mixed_match.rows = -1;
  mixed_match.width = -1;
  const Shape* common_reduction_shape = nullptr;
  const Shape* common_data_shape = nullptr;
  for (int64_t index = 0; index < root->operand_count(); ++index) {
    const HloInstruction* output = root->operand(index);
    if (!ShapeUtil::Equal(fusion->shape().tuple_shapes(index),
                          output->shape())) {
      return fail("tuple_element_shape_mismatch");
    }

    std::optional<MusaWarpRowReductionMatch> reduction =
        MatchMusaWarpRowReduction(fusion, output);
    if (!reduction.has_value()) {
      if (output->opcode() == HloOpcode::kReduce) {
        return fail("unsupported_reduction");
      }
      mixed_match.elementwise_outputs.push_back({index, output});
      continue;
    }

    for (const MusaIndexedWarpRowReductionMatch& matched :
         mixed_match.reductions) {
      if (matched.reduction.reduce == reduction->reduce) {
        return fail("duplicate_reduction");
      }
    }
    if (common_reduction_shape != nullptr) {
      if (!ShapeUtil::Equal(*common_reduction_shape,
                            reduction->reduce->shape())) {
        return fail("reduction_shape_mismatch");
      }
      if (!ShapeUtil::Equal(*common_data_shape, reduction->data->shape())) {
        return fail("reduction_data_shape_mismatch");
      }
      if (mixed_match.rows != reduction->rows) {
        return fail("reduction_rows_mismatch");
      }
      if (mixed_match.width != reduction->width) {
        return fail("reduction_width_mismatch");
      }
    }
    common_reduction_shape = &reduction->reduce->shape();
    common_data_shape = &reduction->data->shape();
    mixed_match.rows = reduction->rows;
    mixed_match.width = reduction->width;
    mixed_match.reductions.push_back({index, *reduction});
  }

  if (mixed_match.reductions.size() < 2) {
    return fail("requires_two_reductions");
  }
  if (mixed_match.elementwise_outputs.empty()) {
    return fail("requires_full_output");
  }
  if (common_data_shape == nullptr) {
    return fail("missing_common_data_shape");
  }
  for (const MusaTupleElementwiseOutputMatch& output :
       mixed_match.elementwise_outputs) {
    const Shape& shape = output.instruction->shape();
    if (!output.instruction->IsElementwise()) {
      return fail("full_output_not_elementwise");
    }
    if (!shape.IsArray()) {
      return fail("full_output_not_array");
    }
    if (shape.element_type() != F32) {
      return fail("full_output_not_f32");
    }
    if (!shape.has_layout()) {
      return fail("full_output_missing_layout");
    }
    if (!LayoutUtil::IsDenseArray(shape)) {
      return fail("full_output_not_dense");
    }
    if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
      return fail("full_output_not_dim0_major");
    }
    if (!ShapeUtil::Equal(shape, *common_data_shape)) {
      return fail("full_output_shape_mismatch");
    }
    for (int64_t dim = 0; dim < shape.rank(); ++dim) {
      if (shape.is_dynamic_dimension(dim)) {
        return fail("full_output_dynamic");
      }
    }
  }
  return mixed_match;
}

}  // namespace gpu
}  // namespace xla
