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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

constexpr int64_t kMusaReductionChainMaxWidth = 256;
constexpr int64_t kMusaReductionChainMaxDataElements = 32 * 1024 * 1024;

bool MusaInstructionDependsOn(const HloInstruction* instruction,
                              const HloInstruction* dependency) {
  if (instruction == nullptr || dependency == nullptr) {
    return false;
  }
  std::vector<const HloInstruction*> worklist = {instruction};
  absl::flat_hash_set<const HloInstruction*> visited;
  while (!worklist.empty()) {
    const HloInstruction* current = worklist.back();
    worklist.pop_back();
    if (current == dependency) {
      return true;
    }
    if (!visited.insert(current).second) {
      continue;
    }
    for (const HloInstruction* operand : current->operands()) {
      worklist.push_back(operand);
    }
  }
  return false;
}

std::optional<int64_t> MusaRowPrefixRank(const Shape& shape, int64_t rows) {
  if (rows <= 0 || !shape.IsArray() || shape.rank() == 0 ||
      !shape.has_layout() || !LayoutUtil::IsDenseArray(shape) ||
      !LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return std::nullopt;
  }
  int64_t elements = 1;
  for (int64_t dim = 0; dim < shape.rank(); ++dim) {
    if (shape.is_dynamic_dimension(dim) || shape.dimensions(dim) <= 0 ||
        elements > rows / shape.dimensions(dim)) {
      return std::nullopt;
    }
    elements *= shape.dimensions(dim);
    if (elements == rows) {
      return dim + 1;
    }
  }
  return std::nullopt;
}

bool MusaInstructionPreservesRows(const HloInstruction* instruction,
                                  const HloInstruction* dependent_operand,
                                  int64_t rows) {
  const std::optional<int64_t> output_prefix =
      MusaRowPrefixRank(instruction->shape(), rows);
  const std::optional<int64_t> operand_prefix =
      MusaRowPrefixRank(dependent_operand->shape(), rows);
  if (!output_prefix.has_value() || !operand_prefix.has_value()) {
    return false;
  }

  if (instruction->IsElementwise()) {
    return ShapeUtil::SameDimensions(instruction->shape(),
                                     dependent_operand->shape());
  }

  switch (instruction->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
      return ShapeUtil::ElementsIn(instruction->shape()) ==
             ShapeUtil::ElementsIn(dependent_operand->shape());
    case HloOpcode::kBroadcast: {
      if (*output_prefix != *operand_prefix ||
          instruction->dimensions().size() !=
              dependent_operand->shape().rank()) {
        return false;
      }
      for (int64_t dim = 0; dim < *operand_prefix; ++dim) {
        if (instruction->dimensions(dim) != dim) {
          return false;
        }
      }
      for (int64_t dim = *operand_prefix;
           dim < instruction->dimensions().size(); ++dim) {
        if (instruction->dimensions(dim) < *output_prefix) {
          return false;
        }
      }
      return true;
    }
    case HloOpcode::kSlice: {
      if (instruction->shape().rank() !=
              dependent_operand->shape().rank() ||
          *output_prefix != *operand_prefix) {
        return false;
      }
      for (int64_t dim = 0; dim < *operand_prefix; ++dim) {
        if (instruction->slice_starts(dim) != 0 ||
            instruction->slice_limits(dim) !=
                dependent_operand->shape().dimensions(dim) ||
            instruction->slice_strides(dim) != 1) {
          return false;
        }
      }
      return true;
    }
    case HloOpcode::kConcatenate:
      return instruction->concatenate_dimension() >= *output_prefix &&
             *output_prefix == *operand_prefix;
    case HloOpcode::kReduce:
      if (ShapeUtil::ElementsIn(instruction->shape()) != rows) {
        return false;
      }
      for (int64_t dim : instruction->dimensions()) {
        if (dim < *operand_prefix) {
          return false;
        }
      }
      return true;
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
      return false;
    default:
      return false;
  }
}

bool MusaDependencyPreservesRows(
    const HloInstruction* instruction, const HloInstruction* dependency,
    int64_t rows, absl::flat_hash_set<const HloInstruction*>* accepted,
    absl::flat_hash_set<const HloInstruction*>* rejected) {
  if (instruction == dependency) {
    return MusaRowPrefixRank(instruction->shape(), rows).has_value();
  }
  if (accepted->contains(instruction)) {
    return true;
  }
  if (rejected->contains(instruction) ||
      !MusaInstructionDependsOn(instruction, dependency)) {
    return false;
  }

  bool found_dependency_path = false;
  for (const HloInstruction* operand : instruction->operands()) {
    if (!MusaInstructionDependsOn(operand, dependency)) {
      continue;
    }
    found_dependency_path = true;
    if (!MusaInstructionPreservesRows(instruction, operand, rows) ||
        !MusaDependencyPreservesRows(operand, dependency, rows, accepted,
                                     rejected)) {
      rejected->insert(instruction);
      return false;
    }
  }
  if (!found_dependency_path) {
    rejected->insert(instruction);
    return false;
  }
  accepted->insert(instruction);
  return true;
}

bool MusaInstructionDependencyPreservesRows(
    const HloInstruction* instruction, const HloInstruction* dependency,
    int64_t rows) {
  absl::flat_hash_set<const HloInstruction*> accepted;
  absl::flat_hash_set<const HloInstruction*> rejected;
  return MusaDependencyPreservesRows(instruction, dependency, rows, &accepted,
                                     &rejected);
}

bool MusaIsScalarF32One(const HloInstruction* fusion,
                        const HloInstruction* instruction) {
  if (instruction != nullptr && instruction->opcode() == HloOpcode::kParameter) {
    const int64_t parameter_number = instruction->parameter_number();
    if (fusion == nullptr || parameter_number < 0 ||
        parameter_number >= fusion->operand_count()) {
      return false;
    }
    instruction = fusion->operand(parameter_number);
  }
  if (instruction == nullptr || instruction->opcode() != HloOpcode::kConstant ||
      !instruction->shape().IsArray() || instruction->shape().rank() != 0 ||
      instruction->shape().element_type() != F32) {
    return false;
  }
  std::optional<double> value = instruction->literal().GetAsDouble({});
  return value.has_value() && *value == 1.0;
}

bool MusaIsPlainMultiplyReducer(const HloInstruction* reduce) {
  if (reduce == nullptr || reduce->opcode() != HloOpcode::kReduce ||
      reduce->operand_count() != 2 || reduce->dimensions().size() != 1 ||
      !reduce->has_to_apply() || reduce->to_apply() == nullptr) {
    return false;
  }
  const HloComputation* reducer = reduce->to_apply();
  const HloInstruction* root = reducer->root_instruction();
  if (root == nullptr || root->opcode() != HloOpcode::kMultiply ||
      root->operand_count() != 2 || reducer->num_parameters() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter) {
    return false;
  }
  const int64_t lhs = root->operand(0)->parameter_number();
  const int64_t rhs = root->operand(1)->parameter_number();
  return (lhs == 0 && rhs == 1) || (lhs == 1 && rhs == 0);
}

bool MusaMatchReductionShape(const HloInstruction* fusion,
                             const HloInstruction* reduce, int64_t* rows,
                             int64_t* width) {
  if (!MusaIsPlainMultiplyReducer(reduce) ||
      !MusaIsScalarF32One(fusion, reduce->operand(1))) {
    return false;
  }
  const Shape& output = reduce->shape();
  const Shape& data = reduce->operand(0)->shape();
  if (!output.IsArray() || output.element_type() != F32 ||
      output.rank() != 2 || !data.IsArray() || data.element_type() != F32 ||
      data.rank() != 3 || !output.has_layout() || !data.has_layout() ||
      !LayoutUtil::IsDenseArray(output) || !LayoutUtil::IsDenseArray(data) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(data.layout()) ||
      reduce->dimensions()[0] != 2 || data.layout().minor_to_major(0) != 2) {
    return false;
  }
  for (int64_t dim = 0; dim < 2; ++dim) {
    if (output.is_dynamic_dimension(dim) || data.is_dynamic_dimension(dim) ||
        output.dimensions(dim) <= 0 ||
        data.dimensions(dim) != output.dimensions(dim)) {
      return false;
    }
  }
  *width = data.dimensions(2);
  if (data.is_dynamic_dimension(2) || *width <= 0 ||
      *width > kMusaReductionChainMaxWidth) {
    return false;
  }
  *rows = ShapeUtil::ElementsIn(output);
  return *rows > 0 && *rows <= kMusaReductionChainMaxDataElements / *width;
}

}  // namespace

bool MusaFusionExpressionPreservesRows(
    const HloInstruction* fusion, const HloInstruction* expression,
    const HloInstruction* external_operand, int64_t rows) {
  if (fusion == nullptr || expression == nullptr ||
      external_operand == nullptr || fusion->opcode() != HloOpcode::kFusion ||
      fusion->fused_instructions_computation() == nullptr ||
      expression->parent() != fusion->fused_instructions_computation()) {
    return false;
  }
  bool found_dependency_path = false;
  for (int64_t parameter_number = 0;
       parameter_number < fusion->operand_count(); ++parameter_number) {
    if (fusion->operand(parameter_number) != external_operand) {
      continue;
    }
    const HloInstruction* parameter =
        fusion->fused_instructions_computation()->parameter_instruction(
            parameter_number);
    if (!MusaInstructionDependsOn(expression, parameter)) {
      continue;
    }
    found_dependency_path = true;
    if (!MusaInstructionDependencyPreservesRows(expression, parameter, rows)) {
      return false;
    }
  }
  return found_dependency_path;
}

std::optional<MusaReductionChainMatch> MatchMusaReductionChainFusion(
    const HloInstruction* fusion, const HloComputation* called) {
  if (fusion == nullptr || called == nullptr ||
      fusion->opcode() != HloOpcode::kFusion ||
      fusion->fused_instructions_computation() != called ||
      !fusion->shape().IsArray() || fusion->shape().element_type() != F32 ||
      fusion->shape().rank() != 3 || !fusion->shape().has_layout() ||
      !LayoutUtil::IsDenseArray(fusion->shape()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(fusion->shape().layout())) {
    return std::nullopt;
  }
  const HloInstruction* root = called->root_instruction();
  if (root == nullptr || !ShapeUtil::Equal(root->shape(), fusion->shape())) {
    return std::nullopt;
  }

  std::vector<const HloInstruction*> reductions;
  for (const HloInstruction* instruction : called->instructions()) {
    if (instruction->opcode() != HloOpcode::kReduce) {
      continue;
    }
    int64_t rows = 0;
    int64_t width = 0;
    if (!MusaMatchReductionShape(fusion, instruction, &rows, &width)) {
      return std::nullopt;
    }
    reductions.push_back(instruction);
  }
  if (reductions.size() != 2) {
    return std::nullopt;
  }

  const HloInstruction* first_reduce = reductions[0];
  const HloInstruction* second_reduce = reductions[1];
  if (MusaInstructionDependsOn(first_reduce->operand(0), second_reduce)) {
    std::swap(first_reduce, second_reduce);
  }
  const HloInstruction* first_data = first_reduce->operand(0);
  const HloInstruction* second_data = second_reduce->operand(0);
  if (!MusaInstructionDependsOn(second_data, first_reduce) ||
      MusaInstructionDependsOn(first_data, second_reduce) ||
      !MusaInstructionDependsOn(root, first_reduce) ||
      !MusaInstructionDependsOn(root, second_reduce)) {
    return std::nullopt;
  }

  int64_t first_rows = 0;
  int64_t first_width = 0;
  int64_t second_rows = 0;
  int64_t second_width = 0;
  if (!MusaMatchReductionShape(fusion, first_reduce, &first_rows,
                               &first_width) ||
      !MusaMatchReductionShape(fusion, second_reduce, &second_rows,
                               &second_width) ||
      first_rows != second_rows || first_width != second_width ||
      !ShapeUtil::SameDimensions(root->shape(), first_data->shape()) ||
      !ShapeUtil::SameDimensions(root->shape(), second_data->shape()) ||
      ShapeUtil::ElementsIn(root->shape()) != first_rows * first_width) {
    return std::nullopt;
  }
  if (!MusaInstructionDependencyPreservesRows(second_data, first_reduce,
                                               first_rows) ||
      !MusaInstructionDependencyPreservesRows(root, first_reduce,
                                               first_rows) ||
      !MusaInstructionDependencyPreservesRows(root, second_reduce,
                                               first_rows)) {
    return std::nullopt;
  }

  return MusaReductionChainMatch{first_reduce, first_data, second_reduce,
                                 second_data, root, first_rows, first_width};
}

}  // namespace gpu
}  // namespace xla
