/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "musa_layer_norm_rewriter.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "musa_fusion_custom_calls.h"

namespace xla {
namespace gpu {
namespace {

bool IsReduceAdd(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kReduce &&
         instruction->to_apply() != nullptr &&
         instruction->to_apply()->root_instruction()->opcode() == HloOpcode::kAdd;
}

bool IsSingleValueArray(const Shape& shape) {
  return shape.IsArray() && ShapeUtil::ElementsIn(shape) == 1;
}

bool IsScalarLikeShape(const Shape& shape) {
  return ShapeUtil::IsEffectiveScalar(shape) || IsSingleValueArray(shape);
}

bool IsScalarConstant(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kConstant &&
         IsScalarLikeShape(instruction->shape());
}

const HloInstruction* StripTrivialUnaryOps(const HloInstruction* instruction);

bool IsEpsilonLikeConstant(const HloInstruction* instruction) {
  const HloInstruction* stripped = StripTrivialUnaryOps(instruction);
  if (IsScalarConstant(stripped)) {
    return true;
  }
  return stripped->opcode() == HloOpcode::kBroadcast &&
         IsScalarConstant(StripTrivialUnaryOps(stripped->operand(0)));
}

std::optional<double> TryGetScalarConstantAsDouble(
    const HloInstruction* instruction) {
  const HloInstruction* stripped = StripTrivialUnaryOps(instruction);
  if (stripped->opcode() == HloOpcode::kBroadcast) {
    stripped = StripTrivialUnaryOps(stripped->operand(0));
  }
  if (!IsScalarConstant(stripped)) {
    return std::nullopt;
  }
  if (ShapeUtil::IsEffectiveScalar(stripped->shape())) {
    return stripped->literal().GetAsDouble({});
  }
  std::vector<int64_t> zero_index(stripped->shape().rank(), 0);
  return stripped->literal().GetAsDouble(zero_index);
}

const HloInstruction* StripTrivialUnaryOps(const HloInstruction* instruction) {
  const HloInstruction* current = instruction;
  while (current->opcode() == HloOpcode::kBitcast ||
         current->opcode() == HloOpcode::kReshape ||
         current->opcode() == HloOpcode::kConvert ||
         current->opcode() == HloOpcode::kCopy) {
    current = current->operand(0);
  }
  return current;
}

bool ContainsReduceAdd(const HloInstruction* instruction,
                       absl::flat_hash_set<const HloInstruction*>& visited) {
  if (!visited.insert(instruction).second) {
    return false;
  }

  const HloInstruction* stripped = StripTrivialUnaryOps(instruction);
  if (IsReduceAdd(stripped)) {
    return true;
  }

  if (stripped->opcode() == HloOpcode::kBroadcast) {
    return ContainsReduceAdd(stripped->operand(0), visited);
  }

  switch (stripped->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kDivide:
      return ContainsReduceAdd(stripped->operand(0), visited) ||
             ContainsReduceAdd(stripped->operand(1), visited);
    default:
      return false;
  }
}

bool ContainsReduceAdd(const HloInstruction* instruction) {
  absl::flat_hash_set<const HloInstruction*> visited;
  return ContainsReduceAdd(instruction, visited);
}

bool IsScalarParameterAffineLike(
    const HloInstruction* instruction,
    absl::flat_hash_set<const HloInstruction*>& visited) {
  if (!visited.insert(instruction).second) {
    return false;
  }

  const HloInstruction* stripped = StripTrivialUnaryOps(instruction);
  if (stripped->opcode() == HloOpcode::kBroadcast) {
    return IsScalarParameterAffineLike(stripped->operand(0), visited);
  }

  if (stripped->opcode() == HloOpcode::kParameter &&
      IsScalarLikeShape(stripped->shape())) {
    return true;
  }

  auto scalar_const = [](const HloInstruction* value) {
    return IsScalarConstant(StripTrivialUnaryOps(value));
  };

  switch (stripped->opcode()) {
    case HloOpcode::kMultiply:
    case HloOpcode::kAdd:
      return (scalar_const(stripped->operand(0)) &&
              IsScalarParameterAffineLike(stripped->operand(1), visited)) ||
             (scalar_const(stripped->operand(1)) &&
              IsScalarParameterAffineLike(stripped->operand(0), visited));
    case HloOpcode::kSubtract:
      return (scalar_const(stripped->operand(0)) &&
              IsScalarParameterAffineLike(stripped->operand(1), visited)) ||
             (IsScalarParameterAffineLike(stripped->operand(0), visited) &&
              scalar_const(stripped->operand(1)));
    case HloOpcode::kDivide:
      return IsScalarParameterAffineLike(stripped->operand(0), visited) &&
             scalar_const(stripped->operand(1));
    default:
      return false;
  }
}

bool IsScalarParameterLike(const HloInstruction* instruction) {
  absl::flat_hash_set<const HloInstruction*> visited;
  return IsScalarParameterAffineLike(instruction, visited);
}

bool IsRsqrtOfVariancePlusEpsilon(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kRsqrt) {
    return false;
  }

  const HloInstruction* operand = StripTrivialUnaryOps(instruction->operand(0));
  if (operand->opcode() != HloOpcode::kAdd) {
    return false;
  }

  const HloInstruction* lhs = StripTrivialUnaryOps(operand->operand(0));
  const HloInstruction* rhs = StripTrivialUnaryOps(operand->operand(1));
  const bool lhs_is_eps = IsEpsilonLikeConstant(lhs);
  const bool rhs_is_eps = IsEpsilonLikeConstant(rhs);
  if (lhs_is_eps == rhs_is_eps) {
    return false;
  }

  const HloInstruction* variance_term = lhs_is_eps ? rhs : lhs;
  return ContainsReduceAdd(variance_term) ||
         IsScalarParameterLike(variance_term);
}

std::optional<double> ExtractEpsilonFromRsqrt(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kRsqrt) {
    return std::nullopt;
  }

  const HloInstruction* operand = StripTrivialUnaryOps(instruction->operand(0));
  if (operand->opcode() != HloOpcode::kAdd) {
    return std::nullopt;
  }

  const HloInstruction* lhs = StripTrivialUnaryOps(operand->operand(0));
  const HloInstruction* rhs = StripTrivialUnaryOps(operand->operand(1));
  if (IsEpsilonLikeConstant(lhs) && !IsEpsilonLikeConstant(rhs)) {
    return TryGetScalarConstantAsDouble(lhs);
  }
  if (IsEpsilonLikeConstant(rhs) && !IsEpsilonLikeConstant(lhs)) {
    return TryGetScalarConstantAsDouble(rhs);
  }
  return std::nullopt;
}

bool IsInputMinusMeanBroadcast(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kSubtract) {
    return false;
  }

  auto is_data_tensor = [](const HloInstruction* operand) {
    const HloInstruction* stripped = StripTrivialUnaryOps(operand);
    return stripped->shape().IsArray() &&
           !IsScalarLikeShape(stripped->shape()) &&
           stripped->opcode() != HloOpcode::kBroadcast;
  };

  auto is_mean_broadcast = [](const HloInstruction* operand) {
    const HloInstruction* stripped = StripTrivialUnaryOps(operand);
    if (stripped->opcode() != HloOpcode::kBroadcast) {
      return false;
    }
    return ContainsReduceAdd(stripped->operand(0)) ||
           IsScalarParameterLike(stripped->operand(0));
  };

  return (is_data_tensor(instruction->operand(0)) &&
          is_mean_broadcast(instruction->operand(1))) ||
         (is_data_tensor(instruction->operand(1)) &&
          is_mean_broadcast(instruction->operand(0)));
}

const HloInstruction* ExtractInputTensorFromInputMinusMean(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kSubtract) {
    return nullptr;
  }

  auto is_data_tensor = [](const HloInstruction* operand) {
    const HloInstruction* stripped = StripTrivialUnaryOps(operand);
    return stripped->shape().IsArray() &&
           !IsScalarLikeShape(stripped->shape()) &&
           stripped->opcode() != HloOpcode::kBroadcast;
  };

  auto is_mean_broadcast = [](const HloInstruction* operand) {
    const HloInstruction* stripped = StripTrivialUnaryOps(operand);
    if (stripped->opcode() != HloOpcode::kBroadcast) {
      return false;
    }
    return ContainsReduceAdd(stripped->operand(0)) ||
           IsScalarParameterLike(stripped->operand(0));
  };

  const HloInstruction* lhs = StripTrivialUnaryOps(instruction->operand(0));
  const HloInstruction* rhs = StripTrivialUnaryOps(instruction->operand(1));
  if (is_data_tensor(lhs) && is_mean_broadcast(rhs)) {
    return lhs;
  }
  if (is_data_tensor(rhs) && is_mean_broadcast(lhs)) {
    return rhs;
  }
  return nullptr;
}

bool MatchLayerNormCoreMultiply(const HloInstruction* instruction,
                                const HloInstruction** input_tensor,
                                const HloInstruction** rsqrt) {
  if (instruction->opcode() != HloOpcode::kMultiply ||
      !instruction->shape().IsArray()) {
    return false;
  }

  const HloInstruction* lhs = StripTrivialUnaryOps(instruction->operand(0));
  const HloInstruction* rhs = StripTrivialUnaryOps(instruction->operand(1));

  auto match_centered_and_inv_std =
      [&](const HloInstruction* centered, const HloInstruction* other) {
        if (!IsInputMinusMeanBroadcast(centered)) {
          return false;
        }
        if (other->opcode() != HloOpcode::kBroadcast) {
          return false;
        }
        const HloInstruction* rsqrt_like =
            StripTrivialUnaryOps(other->operand(0));
        if (!IsRsqrtOfVariancePlusEpsilon(rsqrt_like)) {
          return false;
        }
        const HloInstruction* input_like =
            ExtractInputTensorFromInputMinusMean(centered);
        if (input_like == nullptr) {
          return false;
        }
        *input_tensor = input_like;
        *rsqrt = rsqrt_like;
        return true;
      };

  return match_centered_and_inv_std(lhs, rhs) ||
         match_centered_and_inv_std(rhs, lhs);
}

bool IsReachableFromRoot(const HloInstruction* root,
                         const HloInstruction* target) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::function<bool(const HloInstruction*)> dfs =
      [&](const HloInstruction* current) -> bool {
    if (current == target) {
      return true;
    }
    if (!visited.insert(current).second) {
      return false;
    }
    for (const HloInstruction* operand : current->operands()) {
      if (dfs(operand)) {
        return true;
      }
    }
    return false;
  };
  return dfs(root);
}

bool LooksLikeStructuredLayerNormFusion(const HloInstruction* fusion) {
  if (fusion->opcode() != HloOpcode::kFusion || !fusion->shape().IsArray()) {
    return false;
  }

  const HloComputation* fused = fusion->fused_instructions_computation();
  const HloInstruction* root = fused->root_instruction();

  std::vector<const HloInstruction*> centered_candidates;
  std::vector<const HloInstruction*> rsqrt_candidates;
  for (const HloInstruction* instruction : fused->instructions()) {
    if (IsInputMinusMeanBroadcast(instruction)) {
      centered_candidates.push_back(instruction);
    }
    if (IsRsqrtOfVariancePlusEpsilon(instruction)) {
      rsqrt_candidates.push_back(instruction);
    }
  }

  if (centered_candidates.empty() || rsqrt_candidates.empty()) {
    return false;
  }

  for (const HloInstruction* centered : centered_candidates) {
    for (const HloInstruction* rsqrt : rsqrt_candidates) {
      for (const HloInstruction* instruction : fused->instructions()) {
        if (instruction->opcode() != HloOpcode::kMultiply) {
          continue;
        }

        const HloInstruction* lhs = StripTrivialUnaryOps(instruction->operand(0));
        const HloInstruction* rhs = StripTrivialUnaryOps(instruction->operand(1));
        const bool lhs_is_centered = lhs == centered;
        const bool rhs_is_centered = rhs == centered;
        if (lhs_is_centered == rhs_is_centered) {
          continue;
        }

        const HloInstruction* other = lhs_is_centered ? rhs : lhs;
        if (other->opcode() != HloOpcode::kBroadcast) {
          continue;
        }
        // Unwrap one level of trivial ops (reshape, bitcast) after broadcast.
        const HloInstruction* rsqrt_like = StripTrivialUnaryOps(other->operand(0));
        if (rsqrt_like != rsqrt) {
          continue;
        }

        if (IsReachableFromRoot(root, instruction)) {
          return true;
        }
      }
    }
  }

  return false;
}

double ExtractEpsilonOrDefault(const HloInstruction* fusion) {
  const HloComputation* fused = fusion->fused_instructions_computation();
  for (const HloInstruction* instruction : fused->instructions()) {
    if (instruction->opcode() != HloOpcode::kRsqrt) {
      continue;
    }
    std::optional<double> epsilon = ExtractEpsilonFromRsqrt(instruction);
    if (epsilon.has_value()) {
      return *epsilon;
    }
  }
  // Keep fallback aligned with TF-side layernorm fusion default.
  return 1e-3;
}

std::string BuildLayerNormContractOpaque(const Shape& shape, double epsilon) {
  const int64_t rank = shape.rank();
  const int64_t axis = rank > 0 ? rank - 1 : 0;

  std::vector<std::string> dims;
  dims.reserve(shape.dimensions_size());
  for (int64_t dim : shape.dimensions()) {
    dims.push_back(absl::StrCat(dim));
  }

  return absl::StrCat("shape=", absl::StrJoin(dims, ","),
                      ";dtype=",
                      primitive_util::LowercasePrimitiveTypeName(
                          shape.element_type()),
                      ";axis=", axis, ";epsilon=", epsilon,
                      ";workspace=0");
}

}  // namespace

StatusOr<bool> MusaLayerNormRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int rewritten_fusions = 0;
  int rewritten_nonfusions = 0;

  VLOG(1) << "[musa-layernorm-rewriter] run on module=" << module->name();

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> fusion_to_rewrite;
    struct NonfusionMatch {
      HloInstruction* multiply;
      const HloInstruction* input_tensor;
      const HloInstruction* rsqrt;
    };
    std::vector<NonfusionMatch> nonfusion_matches;
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (LooksLikeStructuredLayerNormFusion(instruction)) {
        fusion_to_rewrite.push_back(instruction);
        continue;
      }

      if (instruction->opcode() == HloOpcode::kFusion) {
        continue;
      }
      const HloInstruction* unused_input = nullptr;
      const HloInstruction* unused_rsqrt = nullptr;
      if (MatchLayerNormCoreMultiply(instruction, &unused_input, &unused_rsqrt)) {
        nonfusion_matches.push_back(
            NonfusionMatch{instruction, unused_input, unused_rsqrt});
      }
    }

    for (HloInstruction* fusion : fusion_to_rewrite) {
      const HloComputation* fused = fusion->fused_instructions_computation();
      int64_t x_param_number = -1;
      for (const HloInstruction* instruction : fused->instructions()) {
        const HloInstruction* input_param =
            ExtractInputTensorFromInputMinusMean(instruction);
        if (input_param != nullptr) {
          if (input_param->opcode() == HloOpcode::kParameter) {
            x_param_number = input_param->parameter_number();
          }
          break;
        }
      }
      if (x_param_number < 0 || x_param_number >= fusion->operand_count()) {
        VLOG(1) << "[musa-layernorm-rewriter] skip fusion=" << fusion->name()
                << " due to unresolved input parameter";
        continue;
      }
      const Shape& output_shape = fusion->shape();
      if (output_shape.rank() == 0) {
        VLOG(1) << "[musa-layernorm-rewriter] skip fusion=" << fusion->name()
                << " due to scalar output shape";
        continue;
      }

      const int64_t norm_dim = output_shape.dimensions(output_shape.rank() - 1);
      Shape affine_shape =
          ShapeUtil::MakeShape(output_shape.element_type(), {norm_dim});
      HloInstruction* gamma_scalar =
          computation->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::One(output_shape.element_type())));
      HloInstruction* beta_scalar =
          computation->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(output_shape.element_type())));
      HloInstruction* gamma = computation->AddInstruction(
          HloInstruction::CreateBroadcast(affine_shape, gamma_scalar, {}));
      HloInstruction* beta = computation->AddInstruction(
          HloInstruction::CreateBroadcast(affine_shape, beta_scalar, {}));

      std::vector<HloInstruction*> custom_call_operands = {
          fusion->mutable_operand(x_param_number), gamma, beta};
      std::string contract_opaque =
          BuildLayerNormContractOpaque(fusion->shape(),
                                       ExtractEpsilonOrDefault(fusion));
      HloInstruction* custom_call = computation->AddInstruction(
          HloInstruction::CreateCustomCall(fusion->shape(), custom_call_operands,
                                           kMusaLayerNormCustomCallTarget,
                                           std::move(contract_opaque),
                           API_VERSION_STATUS_RETURNING));
      module->SetAndUniquifyInstrName(custom_call, "musa-layernorm-custom-call");

      custom_call->set_metadata(fusion->metadata());
      custom_call->set_frontend_attributes(fusion->frontend_attributes());

      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(fusion, custom_call));
      changed = true;
      ++rewritten_fusions;
    }

    for (const NonfusionMatch& match : nonfusion_matches) {
      HloInstruction* multiply = match.multiply;
      const HloInstruction* input_tensor = match.input_tensor;
      const HloInstruction* rsqrt = match.rsqrt;
      const Shape& output_shape = multiply->shape();
      if (output_shape.rank() == 0) {
        continue;
      }

      const int64_t norm_dim = output_shape.dimensions(output_shape.rank() - 1);
      Shape affine_shape =
          ShapeUtil::MakeShape(output_shape.element_type(), {norm_dim});
      HloInstruction* gamma_scalar =
          computation->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::One(output_shape.element_type())));
      HloInstruction* beta_scalar =
          computation->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(output_shape.element_type())));
      HloInstruction* gamma = computation->AddInstruction(
          HloInstruction::CreateBroadcast(affine_shape, gamma_scalar, {}));
      HloInstruction* beta = computation->AddInstruction(
          HloInstruction::CreateBroadcast(affine_shape, beta_scalar, {}));

      // Keep a conservative default epsilon for non-fusion route-B rewrite.
      (void)rsqrt;
      const double epsilon = 1e-3;
      std::string contract_opaque =
          BuildLayerNormContractOpaque(output_shape, epsilon);
      std::vector<HloInstruction*> custom_call_operands = {
          const_cast<HloInstruction*>(input_tensor), gamma, beta};
      HloInstruction* custom_call = computation->AddInstruction(
          HloInstruction::CreateCustomCall(output_shape, custom_call_operands,
                                           kMusaLayerNormCustomCallTarget,
                                           std::move(contract_opaque),
                                           API_VERSION_STATUS_RETURNING));
      module->SetAndUniquifyInstrName(custom_call, "musa-layernorm-custom-call");

      custom_call->set_metadata(multiply->metadata());
      custom_call->set_frontend_attributes(multiply->frontend_attributes());

      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(multiply, custom_call));
      changed = true;
      ++rewritten_nonfusions;
    }
  }

  VLOG(1) << "[musa-layernorm-rewriter] module=" << module->name()
          << " rewritten_fusions=" << rewritten_fusions
          << " rewritten_nonfusions=" << rewritten_nonfusions;

  return changed;
}

}  // namespace gpu
}  // namespace xla
