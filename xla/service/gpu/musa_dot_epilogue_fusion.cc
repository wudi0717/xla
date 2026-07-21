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

#include "xla/service/gpu/musa_dot_epilogue_fusion.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
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

std::string EnvStringOrDefault(const char* name, absl::string_view fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return std::string(fallback.data(), fallback.size());
  }
  return std::string(value);
}

int64_t EnvInt64OrDefault(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value) {
    LOG(WARNING) << "Ignoring invalid " << name << "=" << value;
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

bool HasControlDeps(const HloInstruction* instr) {
  return !instr->control_predecessors().empty() ||
         !instr->control_successors().empty();
}

bool IsSupportedDot(const HloInstruction* instr) {
  return instr != nullptr && instr->opcode() == HloOpcode::kDot &&
         instr->operand_count() == 2 && instr->shape().rank() == 2 &&
         instr->shape().element_type() == F32 && !HasControlDeps(instr);
}

bool IsSupportedEpilogueUser(const HloInstruction* user,
                             const HloInstruction* chain_value,
                             const Shape& dot_shape) {
  if (user == nullptr || chain_value == nullptr || HasControlDeps(user) ||
      !user->IsElementwise() || !Shape::Equal()(user->shape(), dot_shape)) {
    return false;
  }
  bool uses_chain_value = false;
  for (const HloInstruction* operand : user->operands()) {
    if (operand == chain_value) {
      uses_chain_value = true;
      continue;
    }
    if (operand->opcode() == HloOpcode::kDot) {
      return false;
    }
  }
  return uses_chain_value;
}

struct DotEpiloguePattern {
  HloInstruction* dot = nullptr;
  std::vector<HloInstruction*> epilogue_chain;
};

DotEpiloguePattern FindDotEpiloguePattern(HloInstruction* dot) {
  DotEpiloguePattern pattern;
  pattern.dot = dot;
  if (!IsSupportedDot(dot)) {
    return pattern;
  }

  HloInstruction* current = dot;
  while (current->user_count() == 1) {
    HloInstruction* user = current->users()[0];
    if (!IsSupportedEpilogueUser(user, current, dot->shape())) {
      break;
    }
    pattern.epilogue_chain.push_back(user);
    current = user;
  }
  return pattern;
}

bool HasAddInEpilogue(const DotEpiloguePattern& pattern) {
  return std::any_of(pattern.epilogue_chain.begin(),
                     pattern.epilogue_chain.end(),
                     [](const HloInstruction* instr) {
                       return instr->opcode() == HloOpcode::kAdd;
                     });
}

std::string OpcodeName(const HloInstruction* instr) {
  if (instr == nullptr) {
    return "null";
  }
  absl::string_view opcode = HloOpcodeString(instr->opcode());
  return std::string(opcode.data(), opcode.size());
}

int64_t DotContractingDimSize(const HloInstruction* dot) {
  if (dot == nullptr || dot->opcode() != HloOpcode::kDot ||
      dot->operand_count() < 1 ||
      dot->dot_dimension_numbers().lhs_contracting_dimensions_size() == 0) {
    return -1;
  }
  const int64_t lhs_k_dim =
      dot->dot_dimension_numbers().lhs_contracting_dimensions(0);
  if (lhs_k_dim < 0 || lhs_k_dim >= dot->operand(0)->shape().rank()) {
    return -1;
  }
  return dot->operand(0)->shape().dimensions(lhs_k_dim);
}

int64_t DotOutputM(const HloInstruction* dot) {
  if (dot == nullptr || dot->opcode() != HloOpcode::kDot ||
      dot->shape().rank() != 2) {
    return -1;
  }
  return dot->shape().dimensions(0);
}

int64_t DotWorkEstimate(const HloInstruction* dot) {
  if (dot == nullptr || dot->opcode() != HloOpcode::kDot ||
      dot->shape().rank() != 2) {
    return -1;
  }
  const int64_t k = DotContractingDimSize(dot);
  if (k < 0) {
    return -1;
  }
  return dot->shape().dimensions(0) * k * dot->shape().dimensions(1);
}

std::string PatternGroupKey(const DotEpiloguePattern& pattern) {
  const HloInstruction* dot = pattern.dot;
  std::string key = "m=";
  if (dot != nullptr && dot->shape().rank() == 2) {
    key += std::to_string(dot->shape().dimensions(0));
    key += " k=" + std::to_string(DotContractingDimSize(dot));
    key += " n=" + std::to_string(dot->shape().dimensions(1));
  } else {
    key += "unknown k=unknown n=unknown";
  }
  key += " lhs=" + OpcodeName(dot != nullptr ? dot->operand(0) : nullptr);
  key += " rhs=" + OpcodeName(dot != nullptr ? dot->operand(1) : nullptr);
  key += " epilogue=[";
  for (int64_t i = 0; i < pattern.epilogue_chain.size(); ++i) {
    if (i > 0) {
      key += ",";
    }
    key += OpcodeName(pattern.epilogue_chain[i]);
  }
  key += "]";
  return key;
}

std::string TopGroupsString(const std::map<std::string, int64_t>& groups,
                            int64_t limit = 8) {
  if (groups.empty()) {
    return "{}";
  }
  std::vector<std::pair<int64_t, std::string>> sorted;
  sorted.reserve(groups.size());
  for (const auto& [key, count] : groups) {
    sorted.push_back({count, key});
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.first != rhs.first) {
                return lhs.first > rhs.first;
              }
              return lhs.second < rhs.second;
            });
  std::string result = "{";
  const int64_t n = std::min<int64_t>(limit, sorted.size());
  for (int64_t i = 0; i < n; ++i) {
    if (i > 0) {
      result += " | ";
    }
    result += std::to_string(sorted[i].first);
    result += "x:";
    result += sorted[i].second;
  }
  result += "}";
  return result;
}

bool PreferLargerDotPattern(const DotEpiloguePattern& lhs,
                            const DotEpiloguePattern& rhs) {
  const int64_t lhs_work = DotWorkEstimate(lhs.dot);
  const int64_t rhs_work = DotWorkEstimate(rhs.dot);
  if (lhs_work != rhs_work) {
    return lhs_work > rhs_work;
  }
  const int64_t lhs_k = DotContractingDimSize(lhs.dot);
  const int64_t rhs_k = DotContractingDimSize(rhs.dot);
  if (lhs_k != rhs_k) {
    return lhs_k > rhs_k;
  }
  const int64_t lhs_n = lhs.dot != nullptr && lhs.dot->shape().rank() == 2
                            ? lhs.dot->shape().dimensions(1)
                            : -1;
  const int64_t rhs_n = rhs.dot != nullptr && rhs.dot->shape().rank() == 2
                            ? rhs.dot->shape().dimensions(1)
                            : -1;
  if (lhs_n != rhs_n) {
    return lhs_n > rhs_n;
  }
  return PatternGroupKey(lhs) < PatternGroupKey(rhs);
}

Status FuseDotEpilogue(HloComputation* computation,
                       const DotEpiloguePattern& pattern,
                       absl::string_view fusion_kind) {
  HloInstruction* root = pattern.epilogue_chain.back();
  HloInstruction* fusion = computation->AddInstruction(
      HloInstruction::CreateFusion(root->shape(),
                                   HloInstruction::FusionKind::kCustom, root));
  root->GetModule()->SetAndUniquifyInstrName(fusion, root->name());

  TF_ASSIGN_OR_RETURN(FusionBackendConfig backend_config,
                      fusion->backend_config<FusionBackendConfig>());
  backend_config.set_kind(std::string(fusion_kind));
  TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(root, fusion));

  for (auto it = pattern.epilogue_chain.rbegin();
       it != pattern.epilogue_chain.rend(); ++it) {
    HloInstruction* instr = *it;
    if (instr == root) {
      continue;
    }
    fusion->FuseInstruction(instr);
    if (instr->user_count() == 0) {
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(instr));
    }
  }

  fusion->FuseInstruction(pattern.dot);
  if (pattern.dot->user_count() == 0) {
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(pattern.dot));
  }
  return OkStatus();
}

}  // namespace

StatusOr<bool> MusaDotEpilogueFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const bool pattern_enabled =
      EnvExplicitlyTrue("MUSA_XLA_DOT_EPILOGUE_PATTERN") ||
      EnvExplicitlyTrue("MUSA_XLA_DOT_EPILOGUE_FUSION");
  const bool rewrite_enabled =
      EnvExplicitlyTrue("MUSA_XLA_DOT_EPILOGUE_FUSION");
  const bool log_enabled =
      EnvExplicitlyTrue("MUSA_XLA_DOT_EPILOGUE_PATTERN_LOG") ||
      EnvExplicitlyTrue("MUSA_XLA_DOT_EPILOGUE_FUSION_LOG");
  const bool log_empty =
      EnvExplicitlyTrue("MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY");
  if (!pattern_enabled) {
    return false;
  }
  const bool require_add =
      EnvExplicitlyTrue("MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD");
  const int64_t max_chain_length_limit = EnvInt64OrDefault(
      "MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH",
      std::numeric_limits<int64_t>::max());
  const int64_t max_fusions_per_module = EnvInt64OrDefault(
      "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE", -1);
  const int64_t min_m = EnvInt64OrDefault(
      "MUSA_XLA_DOT_EPILOGUE_MIN_M", -1);
  const int64_t min_k = EnvInt64OrDefault(
      "MUSA_XLA_DOT_EPILOGUE_MIN_K", -1);
  const int64_t max_fusions_per_pattern = EnvInt64OrDefault(
      "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN", -1);
  const bool sort_by_size =
      EnvExplicitlyTrue("MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE");

  bool changed = false;
  int64_t total_dots = 0;
  int64_t supported_dots = 0;
  int64_t dot_epilogue_patterns = 0;
  int64_t dot_add_patterns = 0;
  int64_t custom_fusions_created = 0;
  int64_t skipped_multi_user = 0;
  int64_t skipped_without_epilogue = 0;
  int64_t max_chain_length = 0;
  int64_t rewrite_candidates = 0;
  int64_t rewrite_filtered_no_add = 0;
  int64_t rewrite_filtered_chain_length = 0;
  int64_t rewrite_filtered_min_m = 0;
  int64_t rewrite_filtered_min_k = 0;
  int64_t rewrite_filtered_pattern_limit = 0;
  int64_t rewrite_filtered_limit = 0;
  std::map<std::string, int64_t> selected_pattern_groups;
  std::map<std::string, int64_t> no_add_filtered_pattern_groups;
  std::map<std::string, int64_t> chain_filtered_pattern_groups;
  std::map<std::string, int64_t> min_m_filtered_pattern_groups;
  std::map<std::string, int64_t> min_k_filtered_pattern_groups;
  std::map<std::string, int64_t> pattern_limit_filtered_pattern_groups;
  std::map<std::string, int64_t> limit_filtered_pattern_groups;
  std::vector<DotEpiloguePattern> patterns_to_fuse;
  std::vector<DotEpiloguePattern> sortable_candidates;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> dots;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kDot) {
        dots.push_back(instr);
      }
    }

    for (HloInstruction* dot : dots) {
      ++total_dots;
      if (!IsSupportedDot(dot)) {
        continue;
      }
      ++supported_dots;
      if (dot->user_count() > 1) {
        ++skipped_multi_user;
        continue;
      }

      DotEpiloguePattern pattern = FindDotEpiloguePattern(dot);
      if (pattern.epilogue_chain.empty()) {
        ++skipped_without_epilogue;
        continue;
      }
      ++dot_epilogue_patterns;
      if (HasAddInEpilogue(pattern)) {
        ++dot_add_patterns;
      }
      max_chain_length = std::max<int64_t>(
          max_chain_length, pattern.epilogue_chain.size());
      if (rewrite_enabled) {
        const std::string pattern_group = PatternGroupKey(pattern);
        if (require_add && !HasAddInEpilogue(pattern)) {
          ++rewrite_filtered_no_add;
          ++no_add_filtered_pattern_groups[pattern_group];
          continue;
        }
        if (static_cast<int64_t>(pattern.epilogue_chain.size()) >
            max_chain_length_limit) {
          ++rewrite_filtered_chain_length;
          ++chain_filtered_pattern_groups[pattern_group];
          continue;
        }
        if (min_m >= 0 && DotOutputM(pattern.dot) < min_m) {
          ++rewrite_filtered_min_m;
          ++min_m_filtered_pattern_groups[pattern_group];
          continue;
        }
        if (min_k >= 0 && DotContractingDimSize(pattern.dot) < min_k) {
          ++rewrite_filtered_min_k;
          ++min_k_filtered_pattern_groups[pattern_group];
          continue;
        }
        if (sort_by_size) {
          sortable_candidates.push_back(std::move(pattern));
          continue;
        }
        if (max_fusions_per_pattern >= 0 &&
            selected_pattern_groups[pattern_group] >=
                max_fusions_per_pattern) {
          ++rewrite_filtered_pattern_limit;
          ++pattern_limit_filtered_pattern_groups[pattern_group];
          continue;
        }
        if (max_fusions_per_module >= 0 &&
            static_cast<int64_t>(patterns_to_fuse.size()) >=
                max_fusions_per_module) {
          ++rewrite_filtered_limit;
          ++limit_filtered_pattern_groups[pattern_group];
          continue;
        }
        ++rewrite_candidates;
        ++selected_pattern_groups[pattern_group];
        patterns_to_fuse.push_back(std::move(pattern));
      }
    }
  }

  if (rewrite_enabled && sort_by_size) {
    std::sort(sortable_candidates.begin(), sortable_candidates.end(),
              PreferLargerDotPattern);
    for (DotEpiloguePattern& pattern : sortable_candidates) {
      const std::string pattern_group = PatternGroupKey(pattern);
      if (max_fusions_per_pattern >= 0 &&
          selected_pattern_groups[pattern_group] >= max_fusions_per_pattern) {
        ++rewrite_filtered_pattern_limit;
        ++pattern_limit_filtered_pattern_groups[pattern_group];
        continue;
      }
      if (max_fusions_per_module >= 0 &&
          static_cast<int64_t>(patterns_to_fuse.size()) >=
              max_fusions_per_module) {
        ++rewrite_filtered_limit;
        ++limit_filtered_pattern_groups[pattern_group];
        continue;
      }
      ++rewrite_candidates;
      ++selected_pattern_groups[pattern_group];
      patterns_to_fuse.push_back(std::move(pattern));
    }
  }

  if (rewrite_enabled) {
    const std::string fusion_kind = EnvStringOrDefault(
        "MUSA_XLA_DOT_EPILOGUE_FUSION_KIND", kTritonGemmFusionKind);
    for (const DotEpiloguePattern& pattern : patterns_to_fuse) {
      if (pattern.dot->parent() == nullptr ||
          pattern.epilogue_chain.empty() ||
          pattern.epilogue_chain.back()->parent() == nullptr) {
        continue;
      }
      TF_RETURN_IF_ERROR(FuseDotEpilogue(pattern.dot->parent(), pattern,
                                         fusion_kind));
      changed = true;
      ++custom_fusions_created;
    }
  }

  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }
  const bool has_interesting_log_content =
      total_dots != 0 || supported_dots != 0 || dot_epilogue_patterns != 0 ||
      dot_add_patterns != 0 || rewrite_candidates != 0 ||
      custom_fusions_created != 0 || skipped_multi_user != 0 ||
      skipped_without_epilogue != 0 || rewrite_filtered_no_add != 0 ||
      rewrite_filtered_chain_length != 0 || rewrite_filtered_min_m != 0 ||
      rewrite_filtered_min_k != 0 || rewrite_filtered_pattern_limit != 0 ||
      rewrite_filtered_limit != 0;
  if (changed || (log_enabled && (log_empty || has_interesting_log_content))) {
    LOG(INFO) << "[MUSA_DOT_EPILOGUE_FUSION] module=" << module->name()
              << " filter_version=6"
              << " changed=" << changed << " rewrite=" << rewrite_enabled
              << " log_empty=" << log_empty
              << " total_dots=" << total_dots
              << " supported_dots=" << supported_dots
              << " dot_epilogue_patterns=" << dot_epilogue_patterns
              << " dot_add_patterns=" << dot_add_patterns
              << " rewrite_candidates=" << rewrite_candidates
              << " custom_fusions_created=" << custom_fusions_created
              << " skipped_multi_user=" << skipped_multi_user
              << " skipped_without_epilogue=" << skipped_without_epilogue
              << " rewrite_filtered_no_add=" << rewrite_filtered_no_add
              << " rewrite_filtered_chain_length="
              << rewrite_filtered_chain_length
              << " rewrite_filtered_min_m=" << rewrite_filtered_min_m
              << " rewrite_filtered_min_k=" << rewrite_filtered_min_k
              << " rewrite_filtered_pattern_limit="
              << rewrite_filtered_pattern_limit
              << " rewrite_filtered_limit=" << rewrite_filtered_limit
              << " require_add=" << require_add
              << " max_chain_length_limit=" << max_chain_length_limit
              << " max_fusions_per_module=" << max_fusions_per_module
              << " min_m=" << min_m
              << " min_k=" << min_k
              << " max_fusions_per_pattern=" << max_fusions_per_pattern
              << " sort_by_size=" << sort_by_size
              << " max_chain_length=" << max_chain_length
              << " top_selected_patterns="
              << TopGroupsString(selected_pattern_groups)
              << " top_no_add_filtered_patterns="
              << TopGroupsString(no_add_filtered_pattern_groups)
              << " top_chain_filtered_patterns="
              << TopGroupsString(chain_filtered_pattern_groups)
              << " top_min_m_filtered_patterns="
              << TopGroupsString(min_m_filtered_pattern_groups)
              << " top_min_k_filtered_patterns="
              << TopGroupsString(min_k_filtered_pattern_groups)
              << " top_pattern_limit_filtered_patterns="
              << TopGroupsString(pattern_limit_filtered_pattern_groups)
              << " top_limit_filtered_patterns="
              << TopGroupsString(limit_filtered_pattern_groups);
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
