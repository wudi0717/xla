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

#include "xla/service/gpu/musa_hot_tuple_softmax.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

constexpr int64_t kMusaHotTupleSoftmaxMaxDataElements = 262144;

bool MusaIsSupportedOutputShape(const Shape& shape) {
  if (!shape.IsArray() || shape.element_type() != F32 ||
      !(shape.rank() >= 1 && shape.rank() <= 2)) {
    return false;
  }
  for (int64_t dim = 0; dim < shape.rank(); ++dim) {
    if (shape.dimensions(dim) <= 0) {
      return false;
    }
  }
  return true;
}

bool MusaIsScalarF32Constant(const HloInstruction* fusion,
                             const HloInstruction* instr, double expected,
                             bool require_negative_infinity = false) {
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
  if (!value.has_value()) {
    return false;
  }
  if (require_negative_infinity) {
    return std::isinf(*value) && *value < 0.0;
  }
  return *value == expected;
}

bool MusaIsPlainReducer(const HloInstruction* reduce, HloOpcode root_opcode) {
  if (reduce == nullptr || reduce->opcode() != HloOpcode::kReduce ||
      reduce->operand_count() != 2 ||
      !MusaIsSupportedOutputShape(reduce->shape()) ||
      reduce->dimensions().size() != 1 || !reduce->has_to_apply() ||
      reduce->to_apply() == nullptr) {
    return false;
  }
  const Shape& output_shape = reduce->shape();
  const Shape& data_shape = reduce->operand(0)->shape();
  const int64_t reduce_dim = reduce->dimensions()[0];
  if (!data_shape.IsArray() || data_shape.element_type() != F32 ||
      data_shape.rank() != output_shape.rank() + 1 ||
      reduce_dim != output_shape.rank()) {
    return false;
  }
  for (int64_t dim = 0; dim < output_shape.rank(); ++dim) {
    if (data_shape.dimensions(dim) != output_shape.dimensions(dim)) {
      return false;
    }
  }
  const HloInstruction* root = reduce->to_apply()->root_instruction();
  return root != nullptr && root->opcode() == root_opcode &&
         root->operand_count() == 2 &&
         root->operand(0)->opcode() == HloOpcode::kParameter &&
         root->operand(1)->opcode() == HloOpcode::kParameter;
}

}  // namespace

std::optional<MusaHotTupleSoftmaxMatch> MatchMusaHotTupleSoftmaxFusion(
    const HloInstruction* fusion, const HloComputation* called) {
  if (fusion == nullptr || called == nullptr || !fusion->shape().IsTuple()) {
    return std::nullopt;
  }
  const HloInstruction* root = called->root_instruction();
  if (root == nullptr || root->opcode() != HloOpcode::kTuple ||
      root->operand_count() < 2 || root->operand_count() % 2 != 0) {
    return std::nullopt;
  }

  MusaHotTupleSoftmaxMatch match;
  std::string expected_root_order;
  int64_t common_rows = -1;
  int64_t estimated_data_elements = 0;
  const Shape* common_output_shape = nullptr;
  for (int64_t index = 0; index < root->operand_count(); index += 2) {
    const HloInstruction* sum_reduce = root->operand(index);
    const HloInstruction* max_reduce = root->operand(index + 1);
    if (!MusaIsPlainReducer(sum_reduce, HloOpcode::kAdd) ||
        !MusaIsPlainReducer(max_reduce, HloOpcode::kMaximum) ||
        !MusaIsScalarF32Constant(fusion, sum_reduce->operand(1), 0.0) ||
        !MusaIsScalarF32Constant(fusion, max_reduce->operand(1), 0.0,
                                 /*require_negative_infinity=*/true)) {
      return std::nullopt;
    }

    const HloInstruction* exp = sum_reduce->operand(0);
    if (exp->opcode() != HloOpcode::kExp || exp->operand_count() != 1) {
      return std::nullopt;
    }
    const HloInstruction* subtract = exp->operand(0);
    if (subtract->opcode() != HloOpcode::kSubtract ||
        subtract->operand_count() != 2) {
      return std::nullopt;
    }
    const HloInstruction* data = subtract->operand(0);
    const HloInstruction* max_broadcast = subtract->operand(1);
    const Shape& output_shape = sum_reduce->shape();
    const int64_t output_rank = output_shape.rank();
    if (max_broadcast->opcode() != HloOpcode::kBroadcast ||
        max_broadcast->operand_count() != 1 ||
        max_broadcast->operand(0) != max_reduce ||
        max_broadcast->dimensions().size() != output_rank ||
        max_reduce->operand(0) != data ||
        !ShapeUtil::SameDimensions(sum_reduce->shape(),
                                   max_reduce->shape()) ||
        !ShapeUtil::SameDimensions(max_broadcast->shape(), data->shape())) {
      return std::nullopt;
    }
    for (int64_t dim = 0; dim < output_rank; ++dim) {
      if (max_broadcast->dimensions()[dim] != dim) {
        return std::nullopt;
      }
    }

    const int64_t rows = ShapeUtil::ElementsIn(output_shape);
    const int64_t width = data->shape().dimensions(output_rank);
    if (width <= 0 || width > 64 ||
        (common_rows >= 0 && common_rows != rows) ||
        (common_output_shape != nullptr &&
         !ShapeUtil::SameDimensions(*common_output_shape, output_shape))) {
      return std::nullopt;
    }
    if (rows > kMusaHotTupleSoftmaxMaxDataElements / width) {
      return std::nullopt;
    }
    const int64_t group_elements = rows * width;
    if (estimated_data_elements >
        kMusaHotTupleSoftmaxMaxDataElements - group_elements) {
      return std::nullopt;
    }
    estimated_data_elements += group_elements;
    common_rows = rows;
    common_output_shape = &output_shape;
    match.groups.push_back(
        MusaHotTupleSoftmaxGroupMatch{data, sum_reduce, max_reduce, rows, width});
    if (!expected_root_order.empty()) {
      expected_root_order.append(",");
    }
    expected_root_order.append("sum,max");
  }

  match.root_order = expected_root_order;
  if (match.root_order != expected_root_order || match.groups.empty()) {
    return std::nullopt;
  }
  return match;
}

}  // namespace gpu
}  // namespace xla
