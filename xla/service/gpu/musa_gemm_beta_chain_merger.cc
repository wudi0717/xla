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

#include "xla/service/gpu/musa_gemm_beta_chain_merger.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

constexpr char kMusaGemmBetaChainCustomCallTarget[] =
    "__musa$gemm_beta_chain";

bool EnvExplicitlyTrue(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
         value[0] == 'y' || value[0] == 'Y' || value[0] == 'o' ||
         value[0] == 'O';
}

int64_t ReadInt64Env(const char* name, int64_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || parsed <= 0) {
    LOG(WARNING) << "Ignoring invalid " << name << "=" << value;
    return default_value;
  }
  return static_cast<int64_t>(parsed);
}

bool HasDenseRowMajorLayout(const Shape& shape) {
  if (!shape.has_layout()) {
    return false;
  }
  const auto& minor_to_major = shape.layout().minor_to_major();
  if (minor_to_major.size() != shape.rank()) {
    return false;
  }
  for (int64_t i = 0; i < shape.rank(); ++i) {
    if (minor_to_major[i] != shape.rank() - 1 - i) {
      return false;
    }
  }
  return true;
}

bool HasStandardGemmDnums(const GemmBackendConfig& config) {
  const DotDimensionNumbers& dnums = config.dot_dimension_numbers();
  return dnums.lhs_batch_dimensions_size() == 0 &&
         dnums.rhs_batch_dimensions_size() == 0 &&
         dnums.lhs_contracting_dimensions_size() == 1 &&
         dnums.rhs_contracting_dimensions_size() == 1 &&
         dnums.lhs_contracting_dimensions(0) == 1 &&
         dnums.rhs_contracting_dimensions(0) == 0;
}

bool IsSupportedGemmNode(const HloInstruction* instr) {
  if (instr == nullptr || !IsLegacyCublasMatmul(*instr) ||
      (instr->operand_count() != 2 && instr->operand_count() != 3) ||
      !instr->control_predecessors().empty() ||
      !instr->control_successors().empty()) {
    return false;
  }
  if (instr->shape().rank() != 2 ||
      instr->shape().element_type() != F32 ||
      !HasDenseRowMajorLayout(instr->shape())) {
    return false;
  }
  const HloInstruction* lhs = instr->operand(0);
  const HloInstruction* rhs = instr->operand(1);
  if (lhs->shape().rank() != 2 || rhs->shape().rank() != 2 ||
      lhs->shape().element_type() != F32 ||
      rhs->shape().element_type() != F32 ||
      !HasDenseRowMajorLayout(lhs->shape()) ||
      !HasDenseRowMajorLayout(rhs->shape())) {
    return false;
  }
  const int64_t m = lhs->shape().dimensions(0);
  const int64_t k = lhs->shape().dimensions(1);
  const int64_t n = rhs->shape().dimensions(1);
  return rhs->shape().dimensions(0) == k &&
         instr->shape().dimensions(0) == m &&
         instr->shape().dimensions(1) == n && m > 0 && k > 0 && n > 0;
}

StatusOr<bool> HasSupportedConfig(const HloInstruction* instr,
                                  double expected_beta) {
  TF_ASSIGN_OR_RETURN(GemmBackendConfig config,
                      instr->backend_config<GemmBackendConfig>());
  return config.alpha_real() == 1.0 && config.alpha_imag() == 0.0 &&
         config.beta() == expected_beta &&
         config.epilogue() == GemmBackendConfig::DEFAULT &&
         HasStandardGemmDnums(config);
}

bool IsUsedAsBetaByGemm(const HloInstruction* instr) {
  for (const HloInstruction* user : instr->users()) {
    if (IsLegacyCublasMatmul(*user) && user->operand_count() == 3 &&
        user->operand(2) == instr) {
      return true;
    }
  }
  return false;
}

StatusOr<std::vector<HloInstruction*>> TryCollectBetaChain(
    HloInstruction* terminal) {
  std::vector<HloInstruction*> chain_from_tail;
  HloInstruction* current = terminal;
  while (current != nullptr) {
    if (!IsSupportedGemmNode(current)) {
      return std::vector<HloInstruction*>{};
    }
    const bool is_head = current->operand_count() == 2;
    TF_ASSIGN_OR_RETURN(bool config_ok,
                        HasSupportedConfig(current, is_head ? 0.0 : 1.0));
    if (!config_ok) {
      return std::vector<HloInstruction*>{};
    }
    chain_from_tail.push_back(current);
    if (is_head) {
      break;
    }
    HloInstruction* previous = current->mutable_operand(2);
    if (previous->user_count() != 1 || previous->users()[0] != current) {
      return std::vector<HloInstruction*>{};
    }
    if (!Shape::Equal().IgnoreElementType()(previous->shape(),
                                            terminal->shape())) {
      return std::vector<HloInstruction*>{};
    }
    current = previous;
  }
  std::reverse(chain_from_tail.begin(), chain_from_tail.end());
  return chain_from_tail;
}

Status MergeBetaChainChunk(HloComputation* computation,
                           const std::vector<HloInstruction*>& chain,
                           HloInstruction* beta_operand) {
  HloInstruction* first = chain.front();
  HloInstruction* terminal = chain.back();
  const int64_t m = terminal->shape().dimensions(0);
  const int64_t n = terminal->shape().dimensions(1);
  int64_t total_k = 0;
  std::vector<HloInstruction*> lhs_operands;
  std::vector<HloInstruction*> rhs_operands;
  lhs_operands.reserve(chain.size());
  rhs_operands.reserve(chain.size());
  for (HloInstruction* gemm : chain) {
    lhs_operands.push_back(gemm->mutable_operand(0));
    rhs_operands.push_back(gemm->mutable_operand(1));
    total_k += gemm->operand(0)->shape().dimensions(1);
  }

  Shape lhs_concat_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {m, total_k}, {1, 0});
  HloInstruction* lhs_concat = computation->AddInstruction(
      HloInstruction::CreateConcatenate(lhs_concat_shape, lhs_operands, 1));
  lhs_concat->set_metadata(terminal->metadata());

  Shape rhs_concat_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {total_k, n}, {1, 0});
  HloInstruction* rhs_concat = computation->AddInstruction(
      HloInstruction::CreateConcatenate(rhs_concat_shape, rhs_operands, 0));
  rhs_concat->set_metadata(terminal->metadata());

  TF_ASSIGN_OR_RETURN(GemmBackendConfig config,
                      first->backend_config<GemmBackendConfig>());
  std::vector<HloInstruction*> gemm_operands = {lhs_concat, rhs_concat};
  if (beta_operand == nullptr) {
    config.set_beta(0.0);
  } else {
    config.set_beta(1.0);
    gemm_operands.push_back(beta_operand);
  }
  HloInstruction* merged_gemm = computation->AddInstruction(
      HloInstruction::CreateCustomCall(terminal->shape(), gemm_operands,
                                       terminal->custom_call_target()));
  merged_gemm->set_metadata(terminal->metadata());
  TF_RETURN_IF_ERROR(merged_gemm->set_backend_config(config));
  if (beta_operand != nullptr) {
    xla::Cast<HloCustomCallInstruction>(merged_gemm)
        ->set_output_to_operand_aliasing({{{}, {2, {}}}});
  }
  return computation->ReplaceInstruction(terminal, merged_gemm);
}

Status RewriteBetaChainToCustomCall(HloComputation* computation,
                                    const std::vector<HloInstruction*>& chain,
                                    HloInstruction* beta_operand) {
  HloInstruction* terminal = chain.back();
  std::vector<HloInstruction*> operands;
  std::vector<std::string> ks;
  operands.reserve(chain.size() * 2 + (beta_operand == nullptr ? 0 : 1));
  ks.reserve(chain.size());
  for (HloInstruction* gemm : chain) {
    operands.push_back(gemm->mutable_operand(0));
    operands.push_back(gemm->mutable_operand(1));
    ks.push_back(absl::StrCat(gemm->operand(0)->shape().dimensions(1)));
  }
  if (beta_operand != nullptr) {
    operands.push_back(beta_operand);
  }

  const std::string opaque =
      absl::StrCat("m=", terminal->shape().dimensions(0),
                   ";n=", terminal->shape().dimensions(1),
                   ";ks=", absl::StrJoin(ks, ","),
                   ";has_beta=", beta_operand == nullptr ? 0 : 1);
  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(
          terminal->shape(), operands, kMusaGemmBetaChainCustomCallTarget,
          opaque, API_VERSION_STATUS_RETURNING));
  custom_call->set_metadata(terminal->metadata());
  custom_call->set_frontend_attributes(terminal->frontend_attributes());
  return computation->ReplaceInstruction(terminal, custom_call);
}

std::vector<std::pair<int64_t, int64_t>> BuildMergeChunks(
    const std::vector<HloInstruction*>& chain, int64_t min_chain_length,
    int64_t max_total_k) {
  std::vector<std::pair<int64_t, int64_t>> chunks;
  int64_t start = 0;
  int64_t total_k = 0;
  for (int64_t i = 0; i < chain.size(); ++i) {
    const int64_t k = chain[i]->operand(0)->shape().dimensions(1);
    if (k > max_total_k) {
      if (i - start >= min_chain_length) {
        chunks.push_back({start, i});
      }
      start = i + 1;
      total_k = 0;
      continue;
    }
    if (i > start && total_k + k > max_total_k) {
      if (i - start >= min_chain_length) {
        chunks.push_back({start, i});
      }
      start = i;
      total_k = 0;
    }
    total_k += k;
  }
  if (chain.size() - start >= min_chain_length) {
    chunks.push_back({start, chain.size()});
  }
  return chunks;
}

}  // namespace

StatusOr<bool> MusaGemmBetaChainMerger::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!EnvExplicitlyTrue("MUSA_XLA_GEMM_BETA_CHAIN_MERGER")) {
    return false;
  }
  const int64_t min_chain_length =
      ReadInt64Env("MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH", 2);
  const int64_t max_chains =
      ReadInt64Env("MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS", 128);
  const int64_t max_total_k =
      ReadInt64Env("MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K", 16);
  const bool rewrite_to_custom_call =
      EnvExplicitlyTrue("MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL");
  if (rewrite_to_custom_call && !allow_custom_call_) {
    LOG(INFO) << "[MUSA_GEMM_BETA_CHAIN_MERGER] module=" << module->name()
              << " skipped_custom_call=true allow_custom_call=false";
    return false;
  }

  bool changed = false;
  int64_t chains_found = 0;
  int64_t chains_rewritten = 0;
  int64_t gemms_removed = 0;
  int64_t max_chain_length = 0;
  int64_t max_chain_total_k = 0;
  absl::flat_hash_set<const HloInstruction*> rewritten;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> terminals;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (IsSupportedGemmNode(instr) && instr->operand_count() == 3 &&
          !IsUsedAsBetaByGemm(instr)) {
        terminals.push_back(instr);
      }
    }

    for (HloInstruction* terminal : terminals) {
      if (chains_rewritten >= max_chains ||
          rewritten.find(terminal) != rewritten.end()) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> chain,
                          TryCollectBetaChain(terminal));
      if (chain.size() < min_chain_length) {
        continue;
      }
      bool overlaps = false;
      for (HloInstruction* gemm : chain) {
        overlaps |= rewritten.find(gemm) != rewritten.end();
      }
      ++chains_found;
      max_chain_length =
          std::max<int64_t>(max_chain_length, chain.size());
      int64_t chain_total_k = 0;
      for (HloInstruction* gemm : chain) {
        chain_total_k += gemm->operand(0)->shape().dimensions(1);
      }
      max_chain_total_k = std::max(max_chain_total_k, chain_total_k);
      if (overlaps) {
        continue;
      }

      if (rewrite_to_custom_call) {
        std::vector<std::pair<int64_t, int64_t>> chunks =
            BuildMergeChunks(chain, min_chain_length, max_total_k);
        for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
          if (chains_rewritten >= max_chains) {
            break;
          }
          std::vector<HloInstruction*> chunk(chain.begin() + it->first,
                                             chain.begin() + it->second);
          HloInstruction* beta_operand =
              it->first == 0 ? nullptr : chain[it->first - 1];
          for (HloInstruction* gemm : chunk) {
            rewritten.insert(gemm);
          }
          TF_RETURN_IF_ERROR(
              RewriteBetaChainToCustomCall(computation, chunk, beta_operand));
          changed = true;
          ++chains_rewritten;
          gemms_removed += chunk.size() - 1;
        }
      } else {
        std::vector<std::pair<int64_t, int64_t>> chunks =
            BuildMergeChunks(chain, min_chain_length, max_total_k);
        for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
          if (chains_rewritten >= max_chains) {
            break;
          }
          std::vector<HloInstruction*> chunk(chain.begin() + it->first,
                                             chain.begin() + it->second);
          HloInstruction* beta_operand =
              it->first == 0 ? nullptr : chain[it->first - 1];
          for (HloInstruction* gemm : chunk) {
            rewritten.insert(gemm);
          }
          TF_RETURN_IF_ERROR(
              MergeBetaChainChunk(computation, chunk, beta_operand));
          changed = true;
          ++chains_rewritten;
          gemms_removed += chunk.size() - 1;
        }
      }
    }
  }

  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }

  if (changed || chains_found > 0 ||
      EnvExplicitlyTrue("MUSA_XLA_GEMM_BETA_CHAIN_LOG_EMPTY")) {
    LOG(INFO) << "[MUSA_GEMM_BETA_CHAIN_MERGER] module=" << module->name()
              << " changed=" << changed
              << " chains_found=" << chains_found
              << " chains_rewritten=" << chains_rewritten
              << " gemms_removed=" << gemms_removed
              << " max_chain_length=" << max_chain_length
              << " max_chain_total_k=" << max_chain_total_k
              << " min_chain_length=" << min_chain_length
              << " max_chains=" << max_chains
              << " max_total_k=" << max_total_k
              << " custom_call=" << rewrite_to_custom_call
              << " allow_custom_call=" << allow_custom_call_;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
