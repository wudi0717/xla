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

#include "xla/service/gpu/musa_gemm_epilogue_fusion.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/musa_fusion_custom_calls.h"
#include "xla/service/hlo_dce.h"
#include "xla/status.h"
#include "xla/status_macros.h"

namespace xla {
namespace gpu {
namespace {

bool EnvExplicitlyTrue(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0 ||
         std::strcmp(value, "YES") == 0 || std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "ON") == 0;
}

bool EnvStringEquals(const char* name, const char* expected) {
  const char* value = std::getenv(name);
  return value != nullptr && std::strcmp(value, expected) == 0;
}

std::string EnvStringOrDefault(const char* name, const char* default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  return value;
}

std::set<std::string> ParseShapeSet(const std::string& value) {
  std::set<std::string> shapes;
  std::string current;
  for (char c : value) {
    if (c == ',' || c == ';' || c == '|' || c == ' ') {
      if (!current.empty()) {
        shapes.insert(current);
        current.clear();
      }
      continue;
    }
    current.push_back(c);
  }
  if (!current.empty()) {
    shapes.insert(current);
  }
  return shapes;
}

std::string OpcodeName(const HloInstruction* instr) {
  return instr == nullptr ? "null"
                          : std::string(HloOpcodeString(instr->opcode()));
}

std::string CustomCallTargetName(const HloInstruction* instr) {
  if (instr == nullptr || instr->opcode() != HloOpcode::kCustomCall) {
    return "";
  }
  return std::string(instr->custom_call_target());
}

std::string InstructionPatternKey(const HloInstruction* instr) {
  std::string key = OpcodeName(instr);
  const std::string target = CustomCallTargetName(instr);
  if (!target.empty()) {
    key += "(" + target + ")";
  }
  return key;
}

std::string AddOperandPatternKey(const HloInstruction* add) {
  return InstructionPatternKey(add->operand(0)) + "+" +
         InstructionPatternKey(add->operand(1));
}

std::string RawGemmShapeKey(const HloInstruction* gemm) {
  if (gemm == nullptr || gemm->operand_count() < 2 ||
      gemm->operand(0)->shape().rank() != 2 || gemm->shape().rank() != 2) {
    return "unknown";
  }
  const Shape& lhs = gemm->operand(0)->shape();
  const Shape& out = gemm->shape();
  return std::to_string(out.dimensions(0)) + "x" +
         std::to_string(out.dimensions(1)) + "x" +
         std::to_string(lhs.dimensions(1));
}

std::string TopGroupsString(const std::map<std::string, int64_t>& groups,
                            int64_t limit = 8) {
  std::vector<std::pair<std::string, int64_t>> items(groups.begin(),
                                                     groups.end());
  std::sort(items.begin(), items.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });
  std::string out = "{";
  for (int64_t i = 0; i < static_cast<int64_t>(items.size()) && i < limit;
       ++i) {
    if (i > 0) {
      out += " | ";
    }
    out += std::to_string(items[i].second) + "x:" + items[i].first;
  }
  out += "}";
  return out;
}

std::string UnsupportedGemmReason(const HloInstruction* instr) {
  if (instr == nullptr) {
    return "null";
  }
  if (!IsLegacyCublasMatmul(*instr)) {
    return "not_legacy_gemm:" + InstructionPatternKey(instr);
  }
  if (instr->operand_count() < 2) {
    return "operand_count_lt_2:" + std::to_string(instr->operand_count());
  }
  if (instr->user_count() != 1) {
    return "user_count:" + std::to_string(instr->user_count());
  }
  if (!instr->control_predecessors().empty() ||
      !instr->control_successors().empty()) {
    return "has_control_deps";
  }
  if (instr->shape().rank() != 2) {
    return "rank:" + std::to_string(instr->shape().rank());
  }
  if (instr->shape().element_type() != F32) {
    return "dtype_not_f32";
  }
  auto config_or = instr->backend_config<GemmBackendConfig>();
  if (!config_or.ok()) {
    return "backend_config_error";
  }
  const GemmBackendConfig& config = config_or.value();
  if (config.alpha_real() != 1.0 || config.alpha_imag() != 0.0) {
    return "alpha_not_one";
  }
  if (config.beta() != 0.0) {
    return "beta:" + std::to_string(config.beta());
  }
  if (config.epilogue() != GemmBackendConfig::DEFAULT) {
    return "epilogue:" + std::to_string(config.epilogue());
  }
  return "supported";
}

bool IsSupportedGemmForAddFusion(const HloInstruction* instr) {
  if (instr == nullptr || !IsLegacyCublasMatmul(*instr) ||
      instr->operand_count() < 2 || instr->user_count() != 1 ||
      !instr->control_predecessors().empty() ||
      !instr->control_successors().empty()) {
    return false;
  }
  if (instr->shape().rank() != 2 || instr->shape().element_type() != F32) {
    return false;
  }
  auto config_or = instr->backend_config<GemmBackendConfig>();
  if (!config_or.ok()) {
    return false;
  }
  const GemmBackendConfig& config = config_or.value();
  return config.alpha_real() == 1.0 && config.alpha_imag() == 0.0 &&
         config.beta() == 0.0 &&
         config.epilogue() == GemmBackendConfig::DEFAULT;
}

bool IsSupportedBiasForAddFusion(const HloInstruction* bias,
                                 const Shape& gemm_shape,
                                 bool allow_broadcast_bias) {
  if (bias == nullptr || IsLegacyCublasMatmul(*bias) ||
      IsCublasLtMatmul(*bias) || bias->user_count() != 1 ||
      !bias->control_predecessors().empty() ||
      !bias->control_successors().empty()) {
    return false;
  }
  if (!Shape::Equal()(bias->shape(), gemm_shape)) {
    return false;
  }
  if (allow_broadcast_bias && bias->opcode() == HloOpcode::kBroadcast) {
    return true;
  }
  if (bias->opcode() != HloOpcode::kParameter ||
      !bias->parent()->IsEntryComputation()) {
    return false;
  }
  return bias->GetModule()->input_output_alias_config().ParameterHasAlias(
      bias->parameter_number(), /*param_index=*/{});
}

bool IsSupportedBroadcastBiasForMusaEpilogue(const HloInstruction* bias,
                                             const Shape& gemm_shape) {
  if (bias == nullptr || bias->opcode() != HloOpcode::kBroadcast ||
      bias->operand_count() != 1 || gemm_shape.rank() != 2 ||
      bias->shape().rank() != 2 ||
      !Shape::Equal()(bias->shape(), gemm_shape)) {
    return false;
  }
  const HloBroadcastInstruction* broadcast =
      xla::Cast<HloBroadcastInstruction>(bias);
  const Shape& source_shape = bias->operand(0)->shape();
  if (source_shape.rank() == 0 && broadcast->dimensions().empty()) {
    return true;
  }
  return source_shape.rank() == 1 && broadcast->dimensions().size() == 1 &&
         broadcast->dimensions()[0] == 1 &&
         source_shape.dimensions(0) == gemm_shape.dimensions(1);
}

std::string UnsupportedMusaEpilogueRawGemmShapeReason(
    const HloInstruction* gemm) {
  if (gemm == nullptr) {
    return "null";
  }
  if (gemm->operand_count() < 2) {
    return "operand_count_lt_2:" + std::to_string(gemm->operand_count());
  }
  const Shape& lhs = gemm->operand(0)->shape();
  const Shape& rhs = gemm->operand(1)->shape();
  const Shape& out = gemm->shape();
  if (lhs.rank() != 2) {
    return "lhs_rank:" + std::to_string(lhs.rank());
  }
  if (rhs.rank() != 2) {
    return "rhs_rank:" + std::to_string(rhs.rank());
  }
  if (out.rank() != 2) {
    return "out_rank:" + std::to_string(out.rank());
  }
  if (lhs.element_type() != F32 || rhs.element_type() != F32 ||
      out.element_type() != F32) {
    return "dtype_not_f32";
  }
  if (lhs.dimensions(1) != rhs.dimensions(0)) {
    return "k_mismatch:lhs_cols=" + std::to_string(lhs.dimensions(1)) +
           ",rhs_rows=" + std::to_string(rhs.dimensions(0));
  }
  if (out.dimensions(0) != lhs.dimensions(0) ||
      out.dimensions(1) != rhs.dimensions(1)) {
    return "out_mismatch:out=" + std::to_string(out.dimensions(0)) + "x" +
           std::to_string(out.dimensions(1)) + ",expected=" +
           std::to_string(lhs.dimensions(0)) + "x" +
           std::to_string(rhs.dimensions(1));
  }
  return "";
}

Status FuseGemmAdd(HloComputation* computation, HloInstruction* add,
                   HloInstruction* gemm, HloInstruction* bias) {
  TF_ASSIGN_OR_RETURN(GemmBackendConfig config,
                      gemm->backend_config<GemmBackendConfig>());
  config.set_beta(1.0);

  std::vector<HloInstruction*> operands(gemm->operands().begin(),
                                        gemm->operands().end());
  operands.insert(operands.begin() + 2, bias);

  std::unique_ptr<HloInstruction> fused =
      gemm->CloneWithNewOperands(add->shape(), operands);
  fused->set_metadata(add->metadata());
  fused->set_frontend_attributes(add->frontend_attributes());
  TF_RETURN_IF_ERROR(fused->set_backend_config(config));
  xla::Cast<HloCustomCallInstruction>(fused.get())
      ->set_output_to_operand_aliasing({{{}, {2, {}}}});
  add->GetModule()->SetAndUniquifyInstrName(fused.get(), add->name());
  return computation->ReplaceWithNewInstruction(add, std::move(fused));
}

Status FuseGemmAddToMusaEpilogueCustomCall(HloComputation* computation,
                                           HloInstruction* add,
                                           HloInstruction* gemm,
                                           HloInstruction* bias) {
  if (bias->opcode() != HloOpcode::kBroadcast || bias->operand_count() != 1) {
    return InvalidArgument(
        "MUSA GEMM epilogue custom-call expects a broadcast bias");
  }
  TF_ASSIGN_OR_RETURN(GemmBackendConfig config,
                      gemm->backend_config<GemmBackendConfig>());
  config.set_beta(0.0);
  config.set_epilogue(GemmBackendConfig::BIAS);

  std::vector<HloInstruction*> operands = {
      gemm->mutable_operand(0), gemm->mutable_operand(1),
      bias->mutable_operand(0)};
  std::unique_ptr<HloInstruction> fused = HloInstruction::CreateCustomCall(
      add->shape(), operands, kMusaGemmEpilogueCustomCallTarget,
      "epilogue=broadcast_add", API_VERSION_STATUS_RETURNING);
  fused->set_metadata(add->metadata());
  fused->set_frontend_attributes(add->frontend_attributes());
  TF_RETURN_IF_ERROR(fused->set_backend_config(config));
  add->GetModule()->SetAndUniquifyInstrName(fused.get(), add->name());
  return computation->ReplaceWithNewInstruction(add, std::move(fused));
}

}  // namespace

StatusOr<bool> MusaGemmEpilogueFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!EnvExplicitlyTrue("MUSA_XLA_GEMM_EPILOGUE_FUSION")) {
    return false;
  }

  bool changed = false;
  int64_t add_candidates = 0;
  int64_t gemm_add_candidates = 0;
  int64_t multi_operand_gemm_add_candidates = 0;
  int64_t fused_adds = 0;
  int64_t fused_multi_operand_adds = 0;
  int64_t broadcast_bias_candidates = 0;
  int64_t fused_broadcast_bias_adds = 0;
  int64_t fused_custom_epilogue_adds = 0;
  int64_t skipped_custom_epilogue_runtime = 0;
  int64_t skipped_custom_epilogue_shape = 0;
  int64_t skipped_shape_filter = 0;
  int64_t skipped_broadcast_bias_beta_slow = 0;
  int64_t skipped_gemm = 0;
  int64_t skipped_bias = 0;
  std::map<std::string, int64_t> gemm_operand_count_groups;
  std::map<std::string, int64_t> gemm_shape_groups;
  std::map<std::string, int64_t> broadcast_bias_shape_groups;
  std::map<std::string, int64_t> fused_custom_epilogue_shape_groups;
  std::map<std::string, int64_t> fused_legacy_epilogue_shape_groups;
  std::map<std::string, int64_t> bias_opcode_groups;
  std::map<std::string, int64_t> add_operand_pattern_groups;
  std::map<std::string, int64_t> skipped_gemm_reason_groups;
  std::map<std::string, int64_t> skipped_gemm_operand_pattern_groups;
  std::map<std::string, int64_t> skipped_custom_epilogue_shape_groups;
  std::map<std::string, int64_t> skipped_shape_filter_groups;
  std::map<std::string, int64_t> skipped_broadcast_bias_beta_slow_shape_groups;
  const bool broadcast_bias_requested =
      EnvExplicitlyTrue("MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS");
  const bool force_broadcast_bias_beta = EnvExplicitlyTrue(
      "MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA");
  const bool enable_custom_epilogue =
      EnvExplicitlyTrue("MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL");
  const bool custom_epilogue_runtime_supported =
      EnvStringEquals("MUSA_XLA_GPU_RUNTIME", "classic_thunks");
  const bool allow_broadcast_bias =
      broadcast_bias_requested && force_broadcast_bias_beta;
  const std::string only_shapes_string =
      EnvStringOrDefault("MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES", "");
  const std::set<std::string> only_shapes = ParseShapeSet(only_shapes_string);

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> adds;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kAdd && instr->operand_count() == 2) {
        adds.push_back(instr);
      }
    }

    for (HloInstruction* add : adds) {
      ++add_candidates;
      add_operand_pattern_groups[AddOperandPatternKey(add)] += 1;
      HloInstruction* gemm = nullptr;
      HloInstruction* bias = nullptr;
      if (IsSupportedGemmForAddFusion(add->operand(0))) {
        gemm = add->mutable_operand(0);
        bias = add->mutable_operand(1);
      } else if (IsSupportedGemmForAddFusion(add->operand(1))) {
        gemm = add->mutable_operand(1);
        bias = add->mutable_operand(0);
      } else {
        ++skipped_gemm;
        skipped_gemm_operand_pattern_groups[AddOperandPatternKey(add)] += 1;
        skipped_gemm_reason_groups[UnsupportedGemmReason(add->operand(0)) +
                                   "+" +
                                   UnsupportedGemmReason(add->operand(1))] += 1;
        continue;
      }
      ++gemm_add_candidates;
      const std::string gemm_shape_key = RawGemmShapeKey(gemm);
      gemm_operand_count_groups[std::to_string(gemm->operand_count())] += 1;
      gemm_shape_groups[gemm_shape_key] += 1;
      bias_opcode_groups[OpcodeName(bias)] += 1;
      const bool is_multi_operand_gemm = gemm->operand_count() > 2;
      if (is_multi_operand_gemm) {
        ++multi_operand_gemm_add_candidates;
      }
      const bool is_broadcast_bias = bias->opcode() == HloOpcode::kBroadcast;
      if (is_broadcast_bias) {
        ++broadcast_bias_candidates;
        broadcast_bias_shape_groups[gemm_shape_key] += 1;
      }
      if (!only_shapes.empty() &&
          only_shapes.find(gemm_shape_key) == only_shapes.end()) {
        ++skipped_shape_filter;
        ++skipped_bias;
        skipped_shape_filter_groups[gemm_shape_key] += 1;
        continue;
      }
      if (is_broadcast_bias && broadcast_bias_requested &&
          enable_custom_epilogue) {
        if (!custom_epilogue_runtime_supported) {
          ++skipped_custom_epilogue_runtime;
          ++skipped_bias;
          continue;
        }
        if (!IsSupportedBroadcastBiasForMusaEpilogue(bias, gemm->shape())) {
          ++skipped_bias;
          continue;
        }
        std::string shape_reason =
            UnsupportedMusaEpilogueRawGemmShapeReason(gemm);
        if (!shape_reason.empty()) {
          ++skipped_custom_epilogue_shape;
          ++skipped_bias;
          skipped_custom_epilogue_shape_groups[shape_reason] += 1;
          continue;
        }
        TF_RETURN_IF_ERROR(FuseGemmAddToMusaEpilogueCustomCall(
            computation, add, gemm, bias));
        changed = true;
        ++fused_adds;
        ++fused_custom_epilogue_adds;
        ++fused_broadcast_bias_adds;
        fused_custom_epilogue_shape_groups[gemm_shape_key] += 1;
        continue;
      }
      if (is_broadcast_bias && broadcast_bias_requested &&
          !force_broadcast_bias_beta) {
        ++skipped_broadcast_bias_beta_slow;
        ++skipped_bias;
        skipped_broadcast_bias_beta_slow_shape_groups[gemm_shape_key] += 1;
        continue;
      }
      if (!IsSupportedBiasForAddFusion(bias, gemm->shape(),
                                       allow_broadcast_bias)) {
        ++skipped_bias;
        continue;
      }
      TF_RETURN_IF_ERROR(FuseGemmAdd(computation, add, gemm, bias));
      changed = true;
      ++fused_adds;
      if (is_multi_operand_gemm) {
        ++fused_multi_operand_adds;
      }
      if (is_broadcast_bias) {
        ++fused_broadcast_bias_adds;
      }
      fused_legacy_epilogue_shape_groups[gemm_shape_key] += 1;
    }
  }

  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }
  if (changed || EnvExplicitlyTrue("MUSA_XLA_GEMM_EPILOGUE_FUSION_LOG")) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_FUSION] module=" << module->name()
              << " changed=" << changed
              << " add_candidates=" << add_candidates
              << " gemm_add_candidates=" << gemm_add_candidates
              << " multi_operand_gemm_add_candidates="
              << multi_operand_gemm_add_candidates
              << " fused_adds=" << fused_adds
              << " fused_multi_operand_adds=" << fused_multi_operand_adds
              << " broadcast_bias_candidates=" << broadcast_bias_candidates
              << " fused_broadcast_bias_adds=" << fused_broadcast_bias_adds
              << " fused_custom_epilogue_adds=" << fused_custom_epilogue_adds
              << " skipped_custom_epilogue_runtime="
              << skipped_custom_epilogue_runtime
              << " skipped_custom_epilogue_shape="
              << skipped_custom_epilogue_shape
              << " skipped_shape_filter=" << skipped_shape_filter
              << " skipped_broadcast_bias_beta_slow="
              << skipped_broadcast_bias_beta_slow
              << " skipped_gemm=" << skipped_gemm
              << " skipped_bias=" << skipped_bias
              << " broadcast_bias_requested=" << broadcast_bias_requested
              << " enable_custom_epilogue=" << enable_custom_epilogue
              << " custom_epilogue_runtime_supported="
              << custom_epilogue_runtime_supported
              << " force_broadcast_bias_beta=" << force_broadcast_bias_beta
              << " allow_broadcast_bias=" << allow_broadcast_bias
              << " only_shapes=" << only_shapes_string
              << " broadcast_bias_env="
              << "MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS"
              << " gemm_operand_counts="
              << TopGroupsString(gemm_operand_count_groups)
              << " gemm_shapes=" << TopGroupsString(gemm_shape_groups)
              << " broadcast_bias_shapes="
              << TopGroupsString(broadcast_bias_shape_groups)
              << " fused_custom_epilogue_shapes="
              << TopGroupsString(fused_custom_epilogue_shape_groups)
              << " fused_legacy_epilogue_shapes="
              << TopGroupsString(fused_legacy_epilogue_shape_groups)
              << " bias_opcodes=" << TopGroupsString(bias_opcode_groups)
              << " add_operand_patterns="
              << TopGroupsString(add_operand_pattern_groups)
              << " skipped_gemm_operand_patterns="
              << TopGroupsString(skipped_gemm_operand_pattern_groups)
              << " skipped_gemm_reasons="
              << TopGroupsString(skipped_gemm_reason_groups)
              << " skipped_custom_epilogue_shape_reasons="
              << TopGroupsString(skipped_custom_epilogue_shape_groups)
              << " skipped_broadcast_bias_beta_slow_shapes="
              << TopGroupsString(skipped_broadcast_bias_beta_slow_shape_groups)
              << " skipped_shape_filter_groups="
              << TopGroupsString(skipped_shape_filter_groups);
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
