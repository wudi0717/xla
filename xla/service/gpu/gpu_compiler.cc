/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gpu_compiler.h"

#include <algorithm>
#include <any>
#include <cctype>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/hlo_constant_splitter.h"
#include "xla/literal_util.h"
#include "xla/layout_util.h"
#include "xla/mlir/backends/gpu/transforms/passes.h"
#include "xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "xla/runtime/jit_executable.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/all_gather_broadcast_reorder.h"
#include "xla/service/all_gather_combiner.h"
#include "xla/service/all_reduce_combiner.h"
#include "xla/service/all_reduce_contiguous.h"
#include "xla/service/all_reduce_folder.h"
#include "xla/service/all_reduce_promotion.h"
#include "xla/service/all_reduce_reassociate.h"
#include "xla/service/async_collective_creator.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/broadcast_canonicalizer.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_inliner.h"
#include "xla/service/collective_permute_decomposer.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/collectives_schedule_linearizer.h"
#include "xla/service/comparison_expander.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/convert_mover.h"
#include "xla/service/convolution_4d_expander.h"
#include "xla/service/convolution_pred_expander.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/dot_merger.h"
#include "xla/service/dump.h"
#include "xla/service/dynamic_dimension_simplifier.h"
#include "xla/service/dynamic_index_splitter.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/eigh_expander.h"
#include "xla/service/executable.h"
#include "xla/service/export_hlo.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gather_simplifier.h"
#include "xla/service/gpu/alias_passthrough_params.h"
#include "xla/service/gpu/all_reduce_blueconnect.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/conv_layout_normalization.h"
#include "xla/service/gpu/copy_fusion.h"
#include "xla/service/gpu/dot_dimension_sorter.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/fusion_wrapper.h"
#include "xla/service/gpu/gemm_broadcast_folding_rewriter.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/gpu/gemm_rewriter_triton.h"
#include "xla/service/gpu/gpu_all_gather_optimizer.h"
#include "xla/service/gpu/gpu_async_collective_annotator.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/gpu_convert_async_collectives_to_sync.h"
#include "xla/service/gpu/gpu_cost_model_stats_collection.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/gpu_layout_assignment.h"
#include "xla/service/gpu/gpu_reduce_scatter_creator.h"
#include "xla/service/gpu/gpu_sanitize_constant_names.h"
#include "xla/service/gpu/gpu_scatter_expander.h"
#include "xla/service/gpu/hlo_fusion_stats.h"
#include "xla/service/gpu/horizontal_loop_fusion.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/loop_double_buffer_transformer.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/musa_dot_epilogue_fusion.h"
#include "xla/service/gpu/musa_fusion_custom_calls.h"
#include "xla/service/gpu/musa_gemm_beta_chain_merger.h"
#include "xla/service/gpu/musa_gemm_epilogue_fusion.h"
#include "xla/service/gpu/musa_hot_tuple_softmax.h"
#include "xla/service/gpu/musa_reduction_chain.h"
#include "xla/service/gpu/musa_warp_row_reduction.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/move_copy_to_users.h"
#include "xla/service/gpu/prepare_hlo_for_ir_emitting_pipeline.h"
#include "xla/service/gpu/reduction_degenerate_dim_remover.h"
#include "xla/service/gpu/reduction_dimension_grouper.h"
#include "xla/service/gpu/reduction_layout_normalizer.h"
#include "xla/service/gpu/reduction_splitter.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/gpu/runtime_intrinsics.h"
#include "xla/service/gpu/scatter_slice_simplifier.h"
#include "xla/service/gpu/softmax_rewriter_triton.h"
#include "xla/service/gpu/topk_specializer.h"
#include "xla/service/gpu/topk_splitter.h"
#include "xla/service/gpu/tree_reduction_rewriter.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_computation_deduplicator.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_rematerialization.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logistic_expander.h"
#include "xla/service/loop_schedule_linearizer.h"
#include "xla/service/operand_upcaster.h"
#include "xla/service/optimization_barrier_expander.h"
#include "xla/service/qr_expander.h"
#include "xla/service/real_imag_expander.h"
#include "xla/service/reduce_decomposer.h"
#include "xla/service/reduce_scatter_combiner.h"
#include "xla/service/reduce_scatter_reassociate.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/result_caster.h"
#include "xla/service/rng_bit_generator_expander.h"
#include "xla/service/rng_expander.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/simplify_fp_conversions.h"
#include "xla/service/slice_sinker.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/spmd/collective_permute_motion.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/stable_sort_expander.h"
#include "xla/service/stochastic_convert_decomposer.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_all_reduce_code_motion.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/while_loop_trip_count_annotator.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/traceme.h"

#ifdef PLATFORM_GOOGLE
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#endif  // PLATFORM_GOOGLE

#ifdef XLA_MUSA_COMPILER_BASE
namespace xla {
namespace gpu {

struct DeviceConfig {
  se::StreamExecutor* stream_exec;
  se::DeviceMemoryAllocator* allocator = nullptr;
};

struct DevicelessConfig {
  std::string model_str;
  se::CudaComputeCapability cuda_compute_capability{0, 0};
};

class AutotuneConfig {
 public:
  AutotuneConfig(const std::variant<DeviceConfig, DevicelessConfig>& config,
                 const DebugOptions& debug_options)
      : config_(config), debug_options_(debug_options) {}

 private:
  std::variant<DeviceConfig, DevicelessConfig> config_;
  DebugOptions debug_options_;
};

struct AutotunerUtil {
  static Status LoadAutotuneResults(const AutotuneResults& results) {
    (void)results;
    return OkStatus();
  }

  static Status LoadAutotuneResultsFromFile(absl::string_view file_path) {
    (void)file_path;
    return OkStatus();
  }

  static Status SerializeAutotuneResultsToFile(absl::string_view file_path) {
    (void)file_path;
    return OkStatus();
  }
};

}  // namespace gpu
}  // namespace xla
#else
#include "xla/service/gpu/autotuner_util.h"
#endif

namespace xla {
namespace gpu {
namespace {
bool ConvIsLowerable(HloInstruction* conv) {
  return GpuConvRewriter::ConvIsLowerable(conv);
}

StatusOr<AutotuneConfig> GetAutotuneConfig(
    se::StreamExecutor* stream_exec, const DebugOptions& debug_options,
    const GpuCompiler::CompileOptions& options,
    const GpuTargetConfig& gpu_target_config,
    const AutotuneResults* autotune_results) {
  if (stream_exec) {
    return AutotuneConfig{DeviceConfig{stream_exec, options.device_allocator},
                          debug_options};
  }
  AutotuneConfig deviceless_config =
      AutotuneConfig{DevicelessConfig{gpu_target_config.device_description_str},
                     debug_options};
  // Deviceless config means we can't run autotuning, and need to rely on saved
  // results.
  TF_RETURN_IF_ERROR(AutotunerUtil::LoadAutotuneResults(*autotune_results));
  return deviceless_config;
}
}  // end anonymous namespace

StatusOr<std::unique_ptr<Executable>>
GpuXlaRuntimeAotCompilationResult::LoadExecutable(
    Compiler* compiler, se::StreamExecutor* executor) const {
  XlaRuntimeExecutableProto xla_runtime_executable =
      xla_runtime_gpu_executable_.xla_runtime_executable();
  TF_ASSIGN_OR_RETURN(HloModuleConfig hlo_module_config,
                      HloModule::CreateModuleConfigFromProto(
                          xla_runtime_executable.hlo_module_proto(),
                          GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(xla_runtime_executable.hlo_module_proto(),
                                 hlo_module_config));
  auto gpu_compiler = tensorflow::down_cast<GpuCompiler*>(compiler);

  std::vector<GpuExecutable::ConstantInfo> constants;
  for (auto& cst : xla_runtime_gpu_executable_.constants()) {
    GpuExecutable::ConstantInfo constant = {
        cst.symbol_name(),
        {cst.content().begin(), cst.content().end()},
        cst.allocation_index()};
    constants.push_back(std::move(constant));
  }

  return GpuExecutable::LoadFromObjFile(
      std::move(hlo_module), xla_runtime_executable.obj_file(),
      xla_runtime_executable.mlir_module(),
      xla_runtime_gpu_executable_.entry_func_attrs(),
      GetDebugOptionsFromFlags(), xla_runtime_gpu_executable_.gpu_asm_text(),
      xla_runtime_gpu_executable_.gpu_binary(), std::move(constants),
      gpu_compiler->GetGpuVersion(executor), executor);
}

GpuCompiler::GpuCompiler(se::Platform::Id platform_id,
                         const char* target_triple, const char* data_layout)
    : platform_id_(platform_id),
      target_triple_(target_triple),
      data_layout_(data_layout),
      pointer_size_(llvm::DataLayout(data_layout)
                        .getPointerSize(0 /* default address space */)) {}

namespace {
// Adds the HloVerifier for GPU to the given pipeline.
void AddHloVerifier(HloPassPipeline* pipeline, HloVerifierOpts&& opts = {},
                    bool debug_only = false) {
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
  if (debug_only) {
    pipeline->AddInvariantCheckerDebug<HloVerifier>(
        std::move(verifier_metadata), "hlo verifier (debug)");
  } else {
    pipeline->AddInvariantChecker<HloVerifier>(std::move(verifier_metadata),
                                               "hlo verifier");
  }
}

void SetInstructionMetadata(HloModule* module) {
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      instruction->set_creation_pass_id(-1);
      instruction->set_logical_creation_pass_id(-1);
    }
  }
}

bool IsMusaCompilation(se::StreamExecutor* stream_exec,
                       se::Platform::Id platform_id) {
  return platform_id == stream_executor::musa::kMusaPlatformId ||
         (stream_exec != nullptr &&
          stream_exec->platform()->id() == stream_executor::musa::kMusaPlatformId);
}

bool MusaGemmBetaChainCustomCallEnabled() {
  const char* value = std::getenv("MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL");
  if (value == nullptr) {
    return false;
  }
  absl::string_view flag(value);
  return flag == "1" || flag == "true" || flag == "TRUE" || flag == "on" ||
         flag == "ON";
}

int64_t MusaDotMergerMaxSizeBytes(int64_t default_bytes) {
  const char* value = std::getenv("MUSA_XLA_DOT_MERGER_MAX_MIB");
  if (value == nullptr || value[0] == '\0') {
    return default_bytes;
  }
  char* end = nullptr;
  const long long mib = std::strtoll(value, &end, 10);
  if (end == value || mib <= 0) {
    LOG(WARNING) << "Ignoring invalid MUSA_XLA_DOT_MERGER_MAX_MIB=" << value;
    return default_bytes;
  }
  return static_cast<int64_t>(mib) << 20;
}

bool MusaPostTransposeDotMergerEnabled() {
  const char* value = std::getenv("MUSA_XLA_POST_TRANSPOSE_DOT_MERGER");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
         value[0] == 'y' || value[0] == 'Y' || value[0] == 'o' ||
         value[0] == 'O';
}

bool MusaPostTransposeDotMergerExplicitlyDisabled() {
  const char* value = std::getenv("MUSA_XLA_POST_TRANSPOSE_DOT_MERGER");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
         value[0] == 'n' || value[0] == 'N';
}

int64_t MusaPostTransposeDotMergerMaxSizeBytes(int64_t default_bytes) {
  const char* value = std::getenv("MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB");
  if (value == nullptr || value[0] == '\0') {
    return default_bytes;
  }
  char* end = nullptr;
  const long long mib = std::strtoll(value, &end, 10);
  if (end == value || mib <= 0) {
    LOG(WARNING) << "Ignoring invalid "
                 << "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB=" << value;
    return default_bytes;
  }
  return static_cast<int64_t>(mib) << 20;
}

bool MusaModuleHasManySmallDots(const HloModule* module,
                                int64_t max_size_to_merge) {
  int64_t dot_count = 0;
  int64_t small_dot_count = 0;
  for (const HloComputation* computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kDot ||
          !instr->control_predecessors().empty() ||
          !instr->control_successors().empty()) {
        continue;
      }
      ++dot_count;
      int64_t bytes = ShapeUtil::ByteSizeOfElements(instr->shape());
      for (const HloInstruction* operand : instr->operands()) {
        bytes += ShapeUtil::ByteSizeOfElements(operand->shape());
      }
      if (bytes <= max_size_to_merge) {
        ++small_dot_count;
      }
    }
  }
  // Keep this auto path targeted at GEMM-heavy graphs like meta_graph_2.  On
  // smaller modules the extra pass can add compile work without reducing enough
  // launches to pay for itself.
  return dot_count >= 128 && small_dot_count >= 64;
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

std::vector<int64_t> ReadPositiveInt64CsvEnv(
    const char* name, std::vector<int64_t> default_values) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_values;
  }
  std::vector<int64_t> result;
  std::string text(value);
  size_t start = 0;
  while (start <= text.size()) {
    size_t end_pos = text.find(',', start);
    if (end_pos == std::string::npos) {
      end_pos = text.size();
    }
    std::string token = text.substr(start, end_pos - start);
    char* end = nullptr;
    const long long parsed = std::strtoll(token.c_str(), &end, 10);
    while (end != nullptr &&
           std::isspace(static_cast<unsigned char>(*end)) != 0) {
      ++end;
    }
    if (!token.empty() &&
        (end == token.c_str() || *end != '\0' || parsed <= 0)) {
      LOG(WARNING) << "Ignoring invalid " << name << "=" << value;
      return default_values;
    }
    if (!token.empty()) {
      result.push_back(static_cast<int64_t>(parsed));
    }
    if (end_pos == text.size()) {
      break;
    }
    start = end_pos + 1;
  }
  return result;
}

bool ContainsInt64(const std::vector<int64_t>& values, int64_t needle) {
  return std::find(values.begin(), values.end(), needle) != values.end();
}

std::string ReadStringEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return "";
  }
  return std::string(value);
}

std::string TrimMusaCsvToken(std::string token) {
  size_t begin = 0;
  while (begin < token.size() &&
         std::isspace(static_cast<unsigned char>(token[begin])) != 0) {
    ++begin;
  }
  size_t end = token.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(token[end - 1])) != 0) {
    --end;
  }
  return token.substr(begin, end - begin);
}

bool MusaNameCsvMatches(const std::string& csv, const HloInstruction* fusion,
                        const HloComputation* called) {
  if (csv.empty()) {
    return true;
  }
  size_t start = 0;
  while (start <= csv.size()) {
    size_t end_pos = csv.find(',', start);
    if (end_pos == std::string::npos) {
      end_pos = csv.size();
    }
    const std::string token =
        TrimMusaCsvToken(csv.substr(start, end_pos - start));
    if (!token.empty() &&
        (token == fusion->name() || token == called->name() ||
         fusion->name().find(token) != std::string::npos ||
         called->name().find(token) != std::string::npos)) {
      return true;
    }
    if (end_pos == csv.size()) {
      break;
    }
    start = end_pos + 1;
  }
  return false;
}

std::string MusaInstructionOperandSummary(const HloInstruction* instr) {
  std::vector<std::string> operands;
  operands.reserve(instr->operand_count());
  for (const HloInstruction* operand : instr->operands()) {
    operands.push_back(absl::StrCat(
        operand->name(), ":", ShapeUtil::HumanString(operand->shape())));
  }
  return absl::StrJoin(operands, ",");
}

std::string MusaInstructionDimensionsSummary(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kReduce:
    case HloOpcode::kTranspose:
      return absl::StrJoin(instr->dimensions(), ",");
    default:
      return "";
  }
}

std::string MusaInstructionConstantValueSummary(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(instr->shape())) {
    return "";
  }
  switch (instr->shape().element_type()) {
    case F16:
    case BF16:
    case F32:
    case F64: {
      std::optional<double> value = instr->literal().GetAsDouble({});
      return value.has_value() ? absl::StrCat(*value) : "";
    }
    case S32:
      return absl::StrCat(instr->literal().GetFirstElement<int32_t>());
    case S64:
      return absl::StrCat(instr->literal().GetFirstElement<int64_t>());
    case U32:
      return absl::StrCat(instr->literal().GetFirstElement<uint32_t>());
    case U64:
      return absl::StrCat(instr->literal().GetFirstElement<uint64_t>());
    default:
      return "";
  }
}

std::string MusaInstructionToApplyRootSummary(const HloInstruction* instr) {
  if (!instr->has_to_apply() || instr->to_apply() == nullptr ||
      instr->to_apply()->root_instruction() == nullptr) {
    return "";
  }
  const HloInstruction* root = instr->to_apply()->root_instruction();
  return absl::StrCat(instr->to_apply()->name(), ":",
                      HloOpcodeString(root->opcode()));
}

bool EnvExplicitlyTrue(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
         value[0] == 'y' || value[0] == 'Y' || value[0] == 'o' ||
         value[0] == 'O';
}

bool EnvExplicitlyFalse(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
         value[0] == 'n' || value[0] == 'N';
}

bool HasDenseRowMajorLayout(const Shape& shape) {
  if (!shape.has_layout() || !shape.IsArray()) {
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

const HloInstruction* StripMusaTrivialDotOperandWrappers(
    const HloInstruction* instr);

struct MusaSameShapeDotFilterStats {
  int64_t skipped_operand_or_control = 0;
  int64_t skipped_rank = 0;
  int64_t skipped_dtype = 0;
  int64_t skipped_layout = 0;
  int64_t skipped_dot_dims = 0;
  int64_t skipped_shape_mismatch = 0;
};

bool IsSupportedMusaSameShapeDot(
    const HloInstruction* instr,
    MusaSameShapeDotFilterStats* filter_stats = nullptr) {
  if (instr->opcode() != HloOpcode::kDot) {
    return false;
  }
  if (instr->operand_count() != 2 || !instr->control_predecessors().empty() ||
      !instr->control_successors().empty()) {
    if (filter_stats != nullptr) {
      ++filter_stats->skipped_operand_or_control;
    }
    return false;
  }
  if (instr->shape().rank() != 2) {
    if (filter_stats != nullptr) {
      ++filter_stats->skipped_rank;
    }
    return false;
  }
  const HloInstruction* lhs = instr->operand(0);
  const HloInstruction* rhs = instr->operand(1);
  if (lhs->shape().rank() != 2 || rhs->shape().rank() != 2) {
    if (filter_stats != nullptr) {
      ++filter_stats->skipped_rank;
    }
    return false;
  }
  if (instr->shape().element_type() != F32 ||
      lhs->shape().element_type() != F32 || rhs->shape().element_type() != F32) {
    if (filter_stats != nullptr) {
      ++filter_stats->skipped_dtype;
    }
    return false;
  }
  if (!HasDenseRowMajorLayout(lhs->shape()) ||
      !HasDenseRowMajorLayout(rhs->shape()) ||
      !HasDenseRowMajorLayout(instr->shape())) {
    if (filter_stats != nullptr) {
      ++filter_stats->skipped_layout;
    }
    return false;
  }
  const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
  if (dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_contracting_dimensions(0) != 1 ||
      dnums.rhs_contracting_dimensions(0) != 0) {
    if (filter_stats != nullptr) {
      ++filter_stats->skipped_dot_dims;
    }
    return false;
  }
  const int64_t m = lhs->shape().dimensions(0);
  const int64_t k = lhs->shape().dimensions(1);
  const int64_t n = rhs->shape().dimensions(1);
  const bool shape_matches =
      rhs->shape().dimensions(0) == k && instr->shape().dimensions(0) == m &&
      instr->shape().dimensions(1) == n && m > 0 && k > 0 && n > 0;
  if (!shape_matches && filter_stats != nullptr) {
    ++filter_stats->skipped_shape_mismatch;
  }
  return shape_matches;
}

std::string MusaPrecisionConfigKey(const PrecisionConfig& precision_config) {
  std::string key = precision_config.ShortDebugString();
  return key.empty() ? "default" : key;
}

std::string MusaSameShapeDotKey(const HloInstruction* instr) {
  const Shape& lhs = instr->operand(0)->shape();
  const Shape& rhs = instr->operand(1)->shape();
  const Shape& out = instr->shape();
  return absl::StrCat(lhs.dimensions(0), "x", lhs.dimensions(1), "_",
                      rhs.dimensions(0), "x", rhs.dimensions(1), "_",
                      out.dimensions(0), "x", out.dimensions(1), "_",
                      MusaPrecisionConfigKey(instr->precision_config()));
}

std::vector<std::string> TopMusaCountGroups(
    const absl::flat_hash_map<std::string, int64_t>& counts,
    int64_t limit = 10) {
  std::vector<std::pair<std::string, int64_t>> sorted(counts.begin(),
                                                      counts.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) {
      return a.second > b.second;
    }
    return a.first < b.first;
  });
  std::vector<std::string> result;
  for (const auto& [key, count] : sorted) {
    if (result.size() >= limit) {
      break;
    }
    result.push_back(absl::StrCat(count, "x:", key));
  }
  return result;
}

std::string MusaOpcodeCountSummary(const HloComputation* computation,
                                   int64_t limit = 8) {
  absl::flat_hash_map<std::string, int64_t> opcode_counts;
  for (const HloInstruction* instr : computation->instructions()) {
    ++opcode_counts[HloOpcodeString(instr->opcode())];
  }
  return absl::StrJoin(TopMusaCountGroups(opcode_counts, limit), ",");
}

std::string MusaHotTupleSoftmaxWidthSummary(
    const std::vector<MusaHotTupleSoftmaxGroupMatch>& groups) {
  std::vector<std::string> widths;
  widths.reserve(groups.size());
  for (const MusaHotTupleSoftmaxGroupMatch& group : groups) {
    widths.push_back(absl::StrCat(group.width));
  }
  return absl::StrJoin(widths, ",");
}

std::string MusaHotTupleSoftmaxReduceSummary(
    const std::vector<MusaHotTupleSoftmaxGroupMatch>& groups,
    bool want_sum_reduce) {
  std::vector<std::string> names;
  names.reserve(groups.size());
  for (const MusaHotTupleSoftmaxGroupMatch& group : groups) {
    names.push_back(want_sum_reduce
                        ? std::string(group.sum_reduce->name())
                        : std::string(group.max_reduce->name()));
  }
  return absl::StrJoin(names, ",");
}

class MusaHotFusionSoftmaxDiag : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "musa-hot-fusion-softmax-diag";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override {
    const bool summary_diag =
        EnvExplicitlyTrue("MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG");
    const bool detail_diag =
        EnvExplicitlyTrue("MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG");
    const bool body_diag =
        EnvExplicitlyTrue("MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG");
    const bool tuple_softmax_match_diag =
        EnvExplicitlyTrue("MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG");
    const std::string body_names =
        ReadStringEnv("MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_NAMES");
    if (!summary_diag && !detail_diag && !body_diag &&
        !tuple_softmax_match_diag) {
      return false;
    }

    int64_t total_fusions = 0;
    int64_t tuple_output_fusions = 0;
    int64_t row_reduce_candidates = 0;
    int64_t softmax_like_candidates = 0;
    int64_t tuple_softmax_detail_count = 0;
    int64_t tuple_softmax_body_candidate_count = 0;
    int64_t tuple_softmax_body_count = 0;
    int64_t tuple_softmax_body_instr_count = 0;
    int64_t matched_tuple_softmax = 0;
    int64_t matched_tuple_softmax_groups = 0;
    absl::flat_hash_map<std::string, int64_t> candidate_shapes;
    absl::flat_hash_map<std::string, int64_t> candidate_opcodes;
    absl::flat_hash_map<std::string, int64_t> candidate_fusions;
    absl::flat_hash_map<std::string, int64_t> tuple_candidate_shapes;
    absl::flat_hash_map<std::string, int64_t> body_candidate_fusions;
    absl::flat_hash_map<std::string, int64_t> body_candidate_shapes;
    absl::flat_hash_map<std::string, int64_t> tuple_softmax_width_pairs;

    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
        if (instr->opcode() != HloOpcode::kFusion) {
          continue;
        }
        ++total_fusions;
        if (instr->shape().IsTuple()) {
          ++tuple_output_fusions;
        }
        if (instr->called_computations().empty()) {
          continue;
        }
        const HloComputation* called = instr->called_computations()[0];
        int64_t reduce_count = 0;
        int64_t exp_count = 0;
        int64_t power_count = 0;
        int64_t log_count = 0;
        int64_t divide_count = 0;
        int64_t subtract_count = 0;
        int64_t add_count = 0;
        int64_t multiply_count = 0;
        int64_t instruction_count = 0;
        for (const HloInstruction* fused_instr : called->instructions()) {
          ++instruction_count;
          switch (fused_instr->opcode()) {
            case HloOpcode::kReduce:
              ++reduce_count;
              break;
            case HloOpcode::kExp:
              ++exp_count;
              break;
            case HloOpcode::kPower:
              ++power_count;
              break;
            case HloOpcode::kLog:
              ++log_count;
              break;
            case HloOpcode::kDivide:
              ++divide_count;
              break;
            case HloOpcode::kSubtract:
              ++subtract_count;
              break;
            case HloOpcode::kAdd:
              ++add_count;
              break;
            case HloOpcode::kMultiply:
              ++multiply_count;
              break;
            default:
              break;
          }
        }
        const bool row_reduce = reduce_count > 0;
        const bool softmax_like =
            reduce_count > 0 &&
            (exp_count > 0 || power_count > 0 || log_count > 0 ||
             divide_count > 0 || subtract_count > 0 || add_count > 0 ||
             multiply_count > 0);
        if (row_reduce) {
          ++row_reduce_candidates;
        }
        if (softmax_like) {
          ++softmax_like_candidates;
        }
        if (!row_reduce && !softmax_like) {
          continue;
        }
        ++candidate_shapes[ShapeUtil::HumanString(instr->shape())];
        ++candidate_opcodes[MusaOpcodeCountSummary(called)];
        ++candidate_fusions[absl::StrCat(instr->name(), "->", called->name())];
        if (tuple_softmax_match_diag && instr->shape().IsTuple()) {
          std::optional<MusaHotTupleSoftmaxMatch> match =
              MatchMusaHotTupleSoftmaxFusion(instr, called);
          if (match.has_value()) {
            ++matched_tuple_softmax;
            matched_tuple_softmax_groups += match->groups.size();
            ++tuple_softmax_width_pairs[MusaHotTupleSoftmaxWidthSummary(
                match->groups)];
            const int64_t rows =
                match->groups.empty() ? 0 : match->groups.front().rows;
            LOG(INFO) << "[MUSA_HOT_TUPLE_SOFTMAX_MATCH] module="
                      << module->name()
                      << " match=" << matched_tuple_softmax
                      << " fusion=" << instr->name()
                      << " callee=" << called->name()
                      << " rows=" << rows
                      << " groups=" << match->groups.size()
                      << " widths="
                      << MusaHotTupleSoftmaxWidthSummary(match->groups)
                      << " root_order=" << match->root_order
                      << " max_reduce="
                      << MusaHotTupleSoftmaxReduceSummary(match->groups,
                                                          false)
                      << " sum_reduce="
                      << MusaHotTupleSoftmaxReduceSummary(match->groups, true);
          }
        }
        if (detail_diag && instr->shape().IsTuple() && softmax_like) {
          ++tuple_softmax_detail_count;
          ++tuple_candidate_shapes[ShapeUtil::HumanString(instr->shape())];
          const HloInstruction* root = called->root_instruction();
          LOG(INFO) << "[MUSA_HOT_FUSION_SOFTMAX_DETAIL_DIAG] module="
                    << module->name()
                    << " tuple_softmax_detail=" << tuple_softmax_detail_count
                    << " fusion=" << instr->name()
                    << " callee=" << called->name()
                    << " shape=" << ShapeUtil::HumanString(instr->shape())
                    << " tuple_arity=" << instr->shape().tuple_shapes_size()
                    << " root="
                    << (root == nullptr ? "null"
                                        : HloOpcodeString(root->opcode()))
                    << " instructions=" << instruction_count
                    << " reduce_count=" << reduce_count
                    << " exp_count=" << exp_count
                    << " power_count=" << power_count
                    << " log_count=" << log_count
                    << " divide_count=" << divide_count
                    << " subtract_count=" << subtract_count
                    << " add_count=" << add_count
                    << " multiply_count=" << multiply_count
                    << " opcode_counts={"
                    << MusaOpcodeCountSummary(called, 12) << "}";
        }
        if (body_diag && instr->shape().IsTuple() && softmax_like) {
          ++tuple_softmax_body_candidate_count;
          ++body_candidate_fusions[absl::StrCat(instr->name(), "->",
                                                called->name())];
          ++body_candidate_shapes[ShapeUtil::HumanString(instr->shape())];
        }
        if (body_diag && instr->shape().IsTuple() && softmax_like &&
            MusaNameCsvMatches(body_names, instr, called)) {
          ++tuple_softmax_body_count;
          int64_t body_instr_index = 0;
          const HloInstruction* root = called->root_instruction();
          for (const HloInstruction* fused_instr : called->instructions()) {
            ++body_instr_index;
            ++tuple_softmax_body_instr_count;
            LOG(INFO) << "[MUSA_HOT_FUSION_BODY_DIAG] module="
                      << module->name()
                      << " body_fusion=" << tuple_softmax_body_count
                      << " fusion=" << instr->name()
                      << " callee=" << called->name()
                      << " shape=" << ShapeUtil::HumanString(instr->shape())
                      << " body_instr=" << body_instr_index
                      << " name=" << fused_instr->name()
                      << " opcode=" << HloOpcodeString(fused_instr->opcode())
                      << " instr_shape="
                      << ShapeUtil::HumanString(fused_instr->shape())
                      << " operands=["
                      << MusaInstructionOperandSummary(fused_instr) << "]"
                      << " dims=["
                      << MusaInstructionDimensionsSummary(fused_instr) << "]"
                      << " constant_value="
                      << MusaInstructionConstantValueSummary(fused_instr)
                      << " to_apply_root="
                      << MusaInstructionToApplyRootSummary(fused_instr)
                      << " parameter_number="
                      << (fused_instr->opcode() == HloOpcode::kParameter
                              ? fused_instr->parameter_number()
                              : -1)
                      << " is_root=" << (fused_instr == root);
          }
        }
      }
    }

    if (summary_diag) {
      LOG(INFO) << "[MUSA_HOT_FUSION_SOFTMAX_DIAG] module=" << module->name()
                << " total_fusions=" << total_fusions
                << " tuple_output_fusions=" << tuple_output_fusions
                << " row_reduce_candidates=" << row_reduce_candidates
                << " softmax_like_candidates=" << softmax_like_candidates
                << " top_candidate_shapes={"
                << absl::StrJoin(TopMusaCountGroups(candidate_shapes), " | ")
                << "} top_candidate_opcodes={"
                << absl::StrJoin(TopMusaCountGroups(candidate_opcodes), " | ")
                << "} top_candidate_fusions={"
                << absl::StrJoin(TopMusaCountGroups(candidate_fusions), " | ")
                << "}";
    }
    if (detail_diag) {
      LOG(INFO) << "[MUSA_HOT_FUSION_SOFTMAX_DETAIL_DIAG] module="
                << module->name()
                << " tuple_softmax_details=" << tuple_softmax_detail_count
                << " top_tuple_candidate_shapes={"
                << absl::StrJoin(TopMusaCountGroups(tuple_candidate_shapes),
                                 " | ")
                << "}";
    }
    if (body_diag) {
      LOG(INFO) << "[MUSA_HOT_FUSION_BODY_DIAG] module=" << module->name()
                << " tuple_softmax_body_candidates="
                << tuple_softmax_body_candidate_count
                << " tuple_softmax_bodies=" << tuple_softmax_body_count
                << " body_instructions=" << tuple_softmax_body_instr_count
                << " body_names=" << body_names
                << " top_body_candidate_fusions={"
                << absl::StrJoin(TopMusaCountGroups(body_candidate_fusions),
                                 " | ")
                << "} top_body_candidate_shapes={"
                << absl::StrJoin(TopMusaCountGroups(body_candidate_shapes),
                                 " | ")
                << "}";
    }
    if (tuple_softmax_match_diag) {
      LOG(INFO) << "[MUSA_HOT_TUPLE_SOFTMAX_MATCH] module=" << module->name()
                << " matched_tuple_softmax=" << matched_tuple_softmax
                << " matched_groups=" << matched_tuple_softmax_groups
                << " top_width_pairs={"
                << absl::StrJoin(TopMusaCountGroups(tuple_softmax_width_pairs),
                                 " | ")
                << "}";
    }
    return false;
  }
};

bool MusaHasDirectOperand(const HloInstruction* user,
                          const HloInstruction* operand) {
  if (user == nullptr || operand == nullptr) {
    return false;
  }
  for (const HloInstruction* candidate : user->operands()) {
    if (candidate == operand) {
      return true;
    }
  }
  return false;
}

absl::string_view MusaWarpRowReductionKindName(
    MusaWarpRowReductionKind kind) {
  switch (kind) {
    case MusaWarpRowReductionKind::kAdd:
      return "add";
    case MusaWarpRowReductionKind::kMultiply:
      return "multiply";
  }
  return "unknown";
}

class MusaReductionChainDiag : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "musa-reduction-chain-diag";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override {
    if (!EnvExplicitlyTrue("MUSA_XLA_REDUCTION_CHAIN_DIAG")) {
      return false;
    }

    int64_t final_fusion_candidates = 0;
    int64_t matched_chains = 0;
    int64_t inline_safe_chains = 0;
    absl::flat_hash_map<std::string, int64_t> chain_shapes;
    absl::flat_hash_map<std::string, int64_t> final_body_opcodes;

    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      for (HloInstruction* final_fusion :
           computation->MakeInstructionPostOrder()) {
        if (final_fusion->opcode() != HloOpcode::kFusion ||
            !final_fusion->shape().IsArray() ||
            final_fusion->shape().element_type() != F32 ||
            final_fusion->called_computations().empty()) {
          continue;
        }

        struct ReductionOperand {
          const HloInstruction* fusion;
          MusaWarpRowReductionMatch match;
        };
        std::vector<ReductionOperand> reductions;
        for (const HloInstruction* operand : final_fusion->operands()) {
          if (operand->opcode() != HloOpcode::kFusion ||
              operand->called_computations().empty()) {
            continue;
          }
          std::optional<MusaWarpRowReductionMatch> match =
              MatchMusaWarpRowReductionFusion(
                  operand, operand->fused_instructions_computation());
          if (match.has_value()) {
            reductions.push_back(ReductionOperand{operand, *match});
          }
        }
        if (reductions.size() < 2) {
          continue;
        }
        ++final_fusion_candidates;

        for (const ReductionOperand& first_operand : reductions) {
          for (const ReductionOperand& second_operand : reductions) {
            const HloInstruction* first = first_operand.fusion;
            const HloInstruction* second = second_operand.fusion;
            if (first == second || !MusaHasDirectOperand(second, first) ||
                !MusaHasDirectOperand(final_fusion, first) ||
                !MusaHasDirectOperand(final_fusion, second)) {
              continue;
            }
            if (first_operand.match.rows != second_operand.match.rows ||
                first_operand.match.width != second_operand.match.width ||
                first_operand.match.kind != second_operand.match.kind) {
              continue;
            }
            const int64_t expected_elements =
                first_operand.match.rows * first_operand.match.width;
            if (ShapeUtil::ElementsIn(final_fusion->shape()) !=
                expected_elements) {
              continue;
            }

            ++matched_chains;
            const bool first_users_are_chain =
                first->users().size() == 2 &&
                MusaHasDirectOperand(second, first) &&
                MusaHasDirectOperand(final_fusion, first);
            const bool second_users_are_chain =
                second->users().size() == 1 &&
                MusaHasDirectOperand(final_fusion, second);
            const bool inline_safe =
                first_users_are_chain && second_users_are_chain;
            if (inline_safe) {
              ++inline_safe_chains;
            }

            const HloComputation* final_body =
                final_fusion->fused_instructions_computation();
            const std::string shape_key = absl::StrCat(
                "rows=", first_operand.match.rows,
                " width=", first_operand.match.width,
                " reducer=",
                MusaWarpRowReductionKindName(first_operand.match.kind),
                " final=", ShapeUtil::HumanString(final_fusion->shape()));
            ++chain_shapes[shape_key];
            ++final_body_opcodes[MusaOpcodeCountSummary(final_body)];
            LOG(INFO) << "[MUSA_REDUCTION_CHAIN_DIAG] module="
                      << module->name() << " match=" << matched_chains
                      << " first=" << first->name()
                      << " second=" << second->name()
                      << " final=" << final_fusion->name()
                      << " rows=" << first_operand.match.rows
                      << " width=" << first_operand.match.width
                      << " reducer="
                      << MusaWarpRowReductionKindName(first_operand.match.kind)
                      << " final_shape="
                      << ShapeUtil::HumanString(final_fusion->shape())
                      << " first_users=" << first->users().size()
                      << " second_users=" << second->users().size()
                      << " final_users=" << final_fusion->users().size()
                      << " first_users_are_chain=" << first_users_are_chain
                      << " second_users_are_chain=" << second_users_are_chain
                      << " inline_safe=" << inline_safe
                      << " final_body_opcodes={"
                      << MusaOpcodeCountSummary(final_body) << "}";
          }
        }
      }
    }

    LOG(INFO) << "[MUSA_REDUCTION_CHAIN_DIAG] module=" << module->name()
              << " final_fusion_candidates=" << final_fusion_candidates
              << " matched_chains=" << matched_chains
              << " inline_safe_chains=" << inline_safe_chains
              << " top_chain_shapes={"
              << absl::StrJoin(TopMusaCountGroups(chain_shapes), " | ")
              << "} top_final_body_opcodes={"
              << absl::StrJoin(TopMusaCountGroups(final_body_opcodes), " | ")
              << "}";
    return false;
  }
};

class MusaReductionChainRewrite : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "musa-reduction-chain-rewrite";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override {
    if (!EnvExplicitlyTrue("MUSA_XLA_REDUCTION_CHAIN_REWRITE")) {
      return false;
    }
    if (!EnvExplicitlyTrue("MUSA_XLA_REDUCTION_CHAIN_KERNEL")) {
      LOG(INFO) << "[MUSA_REDUCTION_CHAIN_REWRITE] module=" << module->name()
                << " changed=false reason=dedicated_kernel_required";
      return false;
    }

    bool changed = false;
    int64_t rewritten_chains = 0;
    auto rewrite_safe = [](HloFusionInstruction* first,
                           HloFusionInstruction* second,
                           HloFusionInstruction* final_fusion,
                           HloComputation* computation,
                           int64_t expected_elements) {
      if (ShapeUtil::ElementsIn(final_fusion->shape()) != expected_elements) {
        return false;
      }
      const bool exclusive_chain =
          first->users().size() == 2 && second->users().size() == 1 &&
          final_fusion->users().size() == 1 &&
          MusaHasDirectOperand(second, first) &&
          MusaHasDirectOperand(final_fusion, first) &&
          MusaHasDirectOperand(final_fusion, second);
      const bool supported_fusion_kinds =
          first->IsInputFusion() && second->IsInputFusion() &&
          final_fusion->IsLoopFusion();
      const bool control_free =
          first->control_predecessors().empty() &&
          first->control_successors().empty() &&
          second->control_predecessors().empty() &&
          second->control_successors().empty() &&
          final_fusion->control_predecessors().empty() &&
          final_fusion->control_successors().empty();
      const bool same_computation =
          first->parent() == computation && second->parent() == computation &&
          final_fusion->parent() == computation;
      return exclusive_chain && supported_fusion_kinds && control_free &&
             same_computation;
    };
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      while (true) {
        HloFusionInstruction* first = nullptr;
        HloFusionInstruction* second = nullptr;
        HloFusionInstruction* final_fusion = nullptr;
        int64_t matched_rows = 0;
        int64_t matched_width = 0;
        MusaWarpRowReductionKind matched_kind =
            MusaWarpRowReductionKind::kAdd;

        for (HloInstruction* final_candidate :
             computation->MakeInstructionPostOrder()) {
          if (final_candidate->opcode() != HloOpcode::kFusion ||
              !final_candidate->shape().IsArray() ||
              final_candidate->shape().element_type() != F32 ||
              final_candidate->called_computations().empty()) {
            continue;
          }

          struct ReductionOperand {
            HloFusionInstruction* fusion;
            MusaWarpRowReductionMatch match;
          };
          std::vector<ReductionOperand> reductions;
          for (HloInstruction* operand : final_candidate->mutable_operands()) {
            if (operand->opcode() != HloOpcode::kFusion ||
                operand->called_computations().empty()) {
              continue;
            }
            std::optional<MusaWarpRowReductionMatch> match =
                MatchMusaWarpRowReductionFusion(
                    operand, operand->fused_instructions_computation());
            if (match.has_value()) {
              reductions.push_back(ReductionOperand{
                  static_cast<HloFusionInstruction*>(operand), *match});
            }
          }
          if (reductions.size() < 2) {
            continue;
          }

          auto* final =
              static_cast<HloFusionInstruction*>(final_candidate);
          for (const ReductionOperand& first_operand : reductions) {
            for (const ReductionOperand& second_operand : reductions) {
              auto* first_candidate = first_operand.fusion;
              auto* second_candidate = second_operand.fusion;
              if (first_candidate == second_candidate ||
                  !MusaHasDirectOperand(second_candidate, first_candidate) ||
                  !MusaHasDirectOperand(final, first_candidate) ||
                  !MusaHasDirectOperand(final, second_candidate)) {
                continue;
              }
              if (first_operand.match.rows != second_operand.match.rows ||
                  first_operand.match.width != second_operand.match.width ||
                  first_operand.match.kind != second_operand.match.kind) {
                continue;
              }
              if (first_operand.match.kind !=
                      MusaWarpRowReductionKind::kMultiply ||
                  first_operand.match.width > 256 ||
                  first_operand.match.reduce->shape().rank() != 2 ||
                  second_operand.match.reduce->shape().rank() != 2 ||
                  final->shape().rank() != 3 ||
                  !final->shape().has_layout() ||
                  !LayoutUtil::IsDenseArray(final->shape()) ||
                  !LayoutUtil::IsMonotonicWithDim0Major(
                      final->shape().layout()) ||
                  !ShapeUtil::SameDimensions(
                      final->shape(), first_operand.match.data->shape()) ||
                  !ShapeUtil::SameDimensions(
                      final->shape(), second_operand.match.data->shape()) ||
                  !MusaFusionExpressionPreservesRows(
                      second_candidate, second_operand.match.data,
                      first_candidate, first_operand.match.rows) ||
                  !MusaFusionExpressionPreservesRows(
                      final,
                      final->fused_instructions_computation()
                          ->root_instruction(),
                      first_candidate, first_operand.match.rows) ||
                  !MusaFusionExpressionPreservesRows(
                      final,
                      final->fused_instructions_computation()
                          ->root_instruction(),
                      second_candidate, first_operand.match.rows)) {
                continue;
              }
              int64_t first_reductions = 0;
              int64_t second_reductions = 0;
              int64_t final_reductions = 0;
              for (const HloInstruction* instruction :
                   first_candidate->fused_instructions_computation()
                       ->instructions()) {
                first_reductions += instruction->opcode() == HloOpcode::kReduce;
              }
              for (const HloInstruction* instruction :
                   second_candidate->fused_instructions_computation()
                       ->instructions()) {
                second_reductions +=
                    instruction->opcode() == HloOpcode::kReduce;
              }
              for (const HloInstruction* instruction :
                   final->fused_instructions_computation()->instructions()) {
                final_reductions += instruction->opcode() == HloOpcode::kReduce;
              }
              if (first_reductions != 1 || second_reductions != 1 ||
                  final_reductions != 0) {
                continue;
              }
              const int64_t expected_elements =
                  first_operand.match.rows * first_operand.match.width;
              if (!rewrite_safe(first_candidate, second_candidate, final,
                                computation, expected_elements)) {
                continue;
              }

              first = first_candidate;
              second = second_candidate;
              final_fusion = final;
              matched_rows = first_operand.match.rows;
              matched_width = first_operand.match.width;
              matched_kind = first_operand.match.kind;
              break;
            }
            if (final_fusion != nullptr) {
              break;
            }
          }
          if (final_fusion != nullptr) {
            break;
          }
        }

        if (final_fusion == nullptr) {
          break;
        }

        const std::string first_name(first->name());
        const std::string second_name(second->name());
        const std::string final_name(final_fusion->name());
        final_fusion->set_fusion_kind(HloInstruction::FusionKind::kInput);

        // Merge the dependent reduction first so the shared first reduction is
        // cloned only once when it is subsequently merged into the final body.
        final_fusion->MergeFusionInstruction(second);
        TF_RET_CHECK(second->user_count() == 0);
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(second));
        final_fusion->MergeFusionInstruction(first);
        TF_RET_CHECK(first->user_count() == 0);
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(first));
        TF_RET_CHECK(MatchMusaReductionChainFusion(
                         final_fusion,
                         final_fusion->fused_instructions_computation())
                         .has_value())
            << "MUSA reduction-chain rewrite produced an unsupported fusion";
        auto frontend_attributes = final_fusion->frontend_attributes();
        (*frontend_attributes.mutable_map())["musa_reduction_chain"] = "1";
        final_fusion->set_frontend_attributes(
            std::move(frontend_attributes));

        changed = true;
        ++rewritten_chains;
        LOG(INFO) << "[MUSA_REDUCTION_CHAIN_REWRITE] module=" << module->name()
                  << " changed=true rewrite=" << rewritten_chains
                  << " first=" << first_name << " second=" << second_name
                  << " final=" << final_name
                  << " rows=" << matched_rows
                  << " width=" << matched_width
                  << " reducer=" << MusaWarpRowReductionKindName(matched_kind)
                  << " final_fused_instructions="
                  << final_fusion->fused_instructions_computation()
                         ->instruction_count();
      }
    }

    LOG(INFO) << "[MUSA_REDUCTION_CHAIN_REWRITE] module=" << module->name()
              << " changed=" << changed
              << " rewritten_chains=" << rewritten_chains;
    return changed;
  }
};

std::string MusaSameShapeDotPostUserPattern(const HloInstruction* dot) {
  if (dot->users().empty()) {
    return "users=0";
  }
  if (dot->users().size() != 1) {
    return absl::StrCat("users=", dot->users().size());
  }
  const HloInstruction* user = dot->users()[0];
  std::string pattern =
      absl::StrCat("user=", HloOpcodeString(user->opcode()));
  if (user->opcode() != HloOpcode::kAdd || user->operand_count() != 2) {
    return pattern;
  }
  const HloInstruction* other =
      user->operand(0) == dot ? user->operand(1) : user->operand(0);
  absl::StrAppend(&pattern, " other=", HloOpcodeString(other->opcode()),
                  " other_shape=", ShapeUtil::HumanString(other->shape()));
  if (other->opcode() == HloOpcode::kBroadcast && other->operand_count() == 1) {
    absl::StrAppend(
        &pattern, " broadcast_operand=",
        HloOpcodeString(other->operand(0)->opcode()), " broadcast_operand_shape=",
        ShapeUtil::HumanString(other->operand(0)->shape()));
  }
  return pattern;
}

bool GetMusaSameShapeDotBroadcastBiasAdd(HloInstruction* dot,
                                         HloInstruction** add_user,
                                         HloInstruction** bias) {
  if (dot->users().size() != 1) {
    return false;
  }
  HloInstruction* user = dot->users()[0];
  if (user->opcode() != HloOpcode::kAdd || user->operand_count() != 2 ||
      !user->control_predecessors().empty() ||
      !user->control_successors().empty() ||
      !ShapeUtil::Compatible(user->shape(), dot->shape())) {
    return false;
  }
  HloInstruction* other =
      user->mutable_operand(0) == dot ? user->mutable_operand(1)
                                      : user->mutable_operand(0);
  if (other->opcode() != HloOpcode::kBroadcast || other->operand_count() != 1 ||
      !ShapeUtil::Compatible(other->shape(), dot->shape()) ||
      other->dimensions().size() != 1 || other->dimensions()[0] != 1) {
    return false;
  }
  HloInstruction* broadcast_operand = other->mutable_operand(0);
  if (broadcast_operand->opcode() != HloOpcode::kConstant ||
      broadcast_operand->shape().rank() != 1 ||
      broadcast_operand->shape().element_type() != dot->shape().element_type() ||
      broadcast_operand->shape().dimensions(0) != dot->shape().dimensions(1)) {
    return false;
  }
  *add_user = user;
  *bias = broadcast_operand;
  return true;
}

struct MusaSameShapeDotAddTreeRootDiag {
  int64_t add_nodes = 0;
  int64_t broadcast_const_leaves = 0;
  int64_t external_dot_leaves = 0;
  int64_t external_supported_dots = 0;
  int64_t external_same_key_dots = 0;
  int64_t external_other_key_dots = 0;
  int64_t external_unsupported_dots = 0;
  int64_t external_leaves = 0;
  int64_t blocked_nodes = 0;
  absl::flat_hash_set<const HloInstruction*> selected_dot_leaves;
  absl::flat_hash_map<std::string, int64_t> leaf_ops;
  absl::flat_hash_map<std::string, int64_t> dot_keys;
  absl::flat_hash_map<std::string, int64_t> external_dot_reasons;
  absl::flat_hash_map<std::string, int64_t> external_dot_keys;
  absl::flat_hash_map<std::string, int64_t> external_leaf_ops;
  absl::flat_hash_map<std::string, int64_t> blocked_reasons;
};

struct MusaSameShapeDotAddTreeChunkDiag {
  int64_t roots = 0;
  int64_t candidate_roots = 0;
  int64_t candidate_dots = 0;
  int64_t candidate_adds = 0;
  int64_t estimated_slice_reduction = 0;
  int64_t estimated_reshape_reduction = 0;
  int64_t estimated_slice_bytes_reduction = 0;
  int64_t broadcast_const_leaves = 0;
  int64_t external_dot_leaves = 0;
  int64_t external_supported_dots = 0;
  int64_t external_same_key_dots = 0;
  int64_t external_other_key_dots = 0;
  int64_t external_unsupported_dots = 0;
  int64_t external_leaves = 0;
  int64_t blocked_nodes = 0;
  int64_t mixed_key_roots = 0;
  int64_t mixed_key_rewritable_roots = 0;
  int64_t mixed_key_dots = 0;
  int64_t mixed_key_external_dots = 0;
  absl::flat_hash_map<std::string, int64_t> root_patterns;
  absl::flat_hash_map<std::string, int64_t> mixed_key_root_patterns;
  absl::flat_hash_map<std::string, int64_t> external_dot_reasons;
  absl::flat_hash_map<std::string, int64_t> external_dot_keys;
  absl::flat_hash_map<std::string, int64_t> external_leaf_ops;
  absl::flat_hash_map<std::string, int64_t> blocked_reasons;
};

struct MusaSameShapeDotAddTreeRewritePlan {
  HloInstruction* root = nullptr;
  std::vector<int64_t> dot_indices;
  std::vector<HloInstruction*> broadcast_leaves;
  int64_t add_nodes = 0;
};

struct MusaSameShapeDotAddTreeRewriteStats {
  int64_t rewritten_roots = 0;
  int64_t rewritten_dots = 0;
  int64_t rewritten_adds = 0;
  int64_t estimated_slice_reduction = 0;
  int64_t estimated_reshape_reduction = 0;
  int64_t skipped_roots = 0;
  absl::flat_hash_map<std::string, int64_t> rewritten_chunks;
  absl::flat_hash_map<std::string, int64_t> skipped_reasons;
};

struct MusaSameShapeDotBatchLaneRef {
  HloInstruction* batch_result = nullptr;
  int64_t index = 0;
  int64_t m = 0;
  int64_t n = 0;
};

struct MusaSameShapeDotMixedKeyAddTreeRewriteStats {
  int64_t rewritten_roots = 0;
  int64_t rewritten_dots = 0;
  int64_t rewritten_adds = 0;
  int64_t estimated_slice_reduction = 0;
  int64_t estimated_reshape_reduction = 0;
  int64_t skipped_roots = 0;
  absl::flat_hash_map<std::string, int64_t> rewritten_patterns;
  absl::flat_hash_map<std::string, int64_t> skipped_reasons;
};

bool IsMusaSameShapeBroadcastConstLeaf(const HloInstruction* instr,
                                       const Shape& output_shape) {
  if (instr->opcode() != HloOpcode::kBroadcast || instr->operand_count() != 1 ||
      !ShapeUtil::Compatible(instr->shape(), output_shape) ||
      instr->dimensions().size() != 1 || instr->dimensions()[0] != 1) {
    return false;
  }
  const HloInstruction* operand = instr->operand(0);
  return operand->opcode() == HloOpcode::kConstant &&
         operand->shape().rank() == 1 &&
         operand->shape().element_type() == output_shape.element_type() &&
         operand->shape().dimensions(0) == output_shape.dimensions(1);
}

std::string MusaSameShapeDotUnsupportedReason(
    const MusaSameShapeDotFilterStats& stats) {
  if (stats.skipped_operand_or_control > 0) {
    return "operand_or_control";
  }
  if (stats.skipped_rank > 0) {
    return "rank";
  }
  if (stats.skipped_dtype > 0) {
    return "dtype";
  }
  if (stats.skipped_layout > 0) {
    return "layout";
  }
  if (stats.skipped_dot_dims > 0) {
    return "dot_dims";
  }
  if (stats.skipped_shape_mismatch > 0) {
    return "shape_mismatch";
  }
  return "not_dot";
}

std::string MusaSameShapeDotAddTreeRootBlockReason(
    const HloInstruction* instr, const Shape& output_shape) {
  if (instr->opcode() != HloOpcode::kAdd) {
    return absl::StrCat("root_not_add:", HloOpcodeString(instr->opcode()));
  }
  if (instr->operand_count() != 2) {
    return "add_operand_count";
  }
  if (!instr->control_predecessors().empty() ||
      !instr->control_successors().empty()) {
    return "add_control_edge";
  }
  if (!ShapeUtil::Compatible(instr->shape(), output_shape)) {
    return "add_shape_mismatch";
  }
  return "";
}

const HloInstruction* MusaSameShapeDotAddTreeRoot(
    const HloInstruction* dot, const Shape& output_shape,
    absl::flat_hash_map<std::string, int64_t>* blocked_reasons) {
  if (dot->users().size() != 1) {
    ++(*blocked_reasons)[absl::StrCat("dot_users=", dot->users().size())];
    return nullptr;
  }
  const HloInstruction* root = dot->users()[0];
  std::string reason =
      MusaSameShapeDotAddTreeRootBlockReason(root, output_shape);
  if (!reason.empty()) {
    ++(*blocked_reasons)[reason];
    return nullptr;
  }

  absl::flat_hash_set<const HloInstruction*> seen;
  seen.insert(root);
  while (root->users().size() == 1) {
    const HloInstruction* next = root->users()[0];
    reason = MusaSameShapeDotAddTreeRootBlockReason(next, output_shape);
    if (!reason.empty()) {
      break;
    }
    if (!seen.insert(next).second) {
      ++(*blocked_reasons)["root_cycle"];
      break;
    }
    root = next;
  }
  return root;
}

void MusaDiagnoseSameShapeDotAddTreeNode(
    const HloInstruction* instr,
    const absl::flat_hash_set<const HloInstruction*>& selected_dots,
    const Shape& output_shape, MusaSameShapeDotAddTreeRootDiag* diag,
    absl::flat_hash_set<const HloInstruction*>* seen_adds,
    bool external_diag, bool mixed_key_diag,
    absl::string_view current_dot_key, int64_t max_depth, int64_t depth) {
  if (depth > max_depth) {
    ++diag->blocked_nodes;
    ++diag->blocked_reasons["depth_limit"];
    return;
  }
  if (selected_dots.find(instr) != selected_dots.end()) {
    diag->selected_dot_leaves.insert(instr);
    if (mixed_key_diag) {
      ++diag->dot_keys[std::string(current_dot_key)];
    }
    return;
  }
  if (IsMusaSameShapeBroadcastConstLeaf(instr, output_shape)) {
    ++diag->broadcast_const_leaves;
    return;
  }
  if (instr->opcode() == HloOpcode::kDot) {
    ++diag->external_dot_leaves;
    ++diag->leaf_ops[ShapeUtil::Compatible(instr->shape(), output_shape)
                         ? "external_dot"
                         : "external_dot_shape_mismatch"];
    if (external_diag || mixed_key_diag) {
      MusaSameShapeDotFilterStats external_filter_stats;
      if (IsSupportedMusaSameShapeDot(instr, &external_filter_stats)) {
        ++diag->external_supported_dots;
        const std::string external_key = MusaSameShapeDotKey(instr);
        if (mixed_key_diag) {
          ++diag->dot_keys[external_key];
        }
        if (external_diag) {
          ++diag->external_dot_keys[external_key];
          if (absl::string_view(external_key) == current_dot_key) {
            ++diag->external_same_key_dots;
            ++diag->external_dot_reasons["supported_same_key_outside_chunk"];
          } else {
            ++diag->external_other_key_dots;
            ++diag->external_dot_reasons["supported_other_key"];
          }
        }
      } else {
        ++diag->external_unsupported_dots;
        if (external_diag) {
          ++diag->external_dot_reasons[absl::StrCat(
              "unsupported:",
              MusaSameShapeDotUnsupportedReason(external_filter_stats))];
        }
      }
    }
    return;
  }
  const std::string reason =
      MusaSameShapeDotAddTreeRootBlockReason(instr, output_shape);
  if (reason.empty()) {
    if (!seen_adds->insert(instr).second) {
      ++diag->blocked_nodes;
      ++diag->blocked_reasons["add_cycle"];
      return;
    }
    ++diag->add_nodes;
    MusaDiagnoseSameShapeDotAddTreeNode(instr->operand(0), selected_dots,
                                        output_shape, diag, seen_adds,
                                        external_diag, mixed_key_diag,
                                        current_dot_key, max_depth,
                                        depth + 1);
    MusaDiagnoseSameShapeDotAddTreeNode(instr->operand(1), selected_dots,
                                        output_shape, diag, seen_adds,
                                        external_diag, mixed_key_diag,
                                        current_dot_key, max_depth,
                                        depth + 1);
    return;
  }
  ++diag->external_leaves;
  ++diag->leaf_ops[HloOpcodeString(instr->opcode())];
  if (external_diag) {
    ++diag->external_leaf_ops[HloOpcodeString(instr->opcode())];
  }
}

MusaSameShapeDotAddTreeChunkDiag DiagnoseMusaSameShapeDotAddTreeChunk(
    absl::Span<HloInstruction* const> dots, bool external_diag,
    bool mixed_key_diag, int64_t max_depth) {
  MusaSameShapeDotAddTreeChunkDiag chunk_diag;
  if (dots.empty()) {
    return chunk_diag;
  }
  const Shape& output_shape = dots.front()->shape();
  const int64_t m = output_shape.dimensions(0);
  const int64_t n = output_shape.dimensions(1);
  const int64_t output_bytes = m * n * 4;
  const std::string current_dot_key = MusaSameShapeDotKey(dots.front());
  absl::flat_hash_set<const HloInstruction*> selected_dots;
  for (const HloInstruction* dot : dots) {
    selected_dots.insert(dot);
  }

  absl::flat_hash_set<const HloInstruction*> roots;
  for (const HloInstruction* dot : dots) {
    const HloInstruction* root = MusaSameShapeDotAddTreeRoot(
        dot, output_shape, &chunk_diag.blocked_reasons);
    if (root != nullptr) {
      roots.insert(root);
    }
  }

  for (const HloInstruction* root : roots) {
    MusaSameShapeDotAddTreeRootDiag root_diag;
    absl::flat_hash_set<const HloInstruction*> seen_adds;
    MusaDiagnoseSameShapeDotAddTreeNode(root, selected_dots, output_shape,
                                        &root_diag, &seen_adds, external_diag,
                                        mixed_key_diag, current_dot_key,
                                        max_depth, 0);
    ++chunk_diag.roots;
    chunk_diag.broadcast_const_leaves += root_diag.broadcast_const_leaves;
    chunk_diag.external_dot_leaves += root_diag.external_dot_leaves;
    chunk_diag.external_supported_dots += root_diag.external_supported_dots;
    chunk_diag.external_same_key_dots += root_diag.external_same_key_dots;
    chunk_diag.external_other_key_dots += root_diag.external_other_key_dots;
    chunk_diag.external_unsupported_dots +=
        root_diag.external_unsupported_dots;
    chunk_diag.external_leaves += root_diag.external_leaves;
    chunk_diag.blocked_nodes += root_diag.blocked_nodes;
    for (const auto& [reason, count] : root_diag.blocked_reasons) {
      chunk_diag.blocked_reasons[reason] += count;
    }
    for (const auto& [reason, count] : root_diag.external_dot_reasons) {
      chunk_diag.external_dot_reasons[reason] += count;
    }
    for (const auto& [key, count] : root_diag.external_dot_keys) {
      chunk_diag.external_dot_keys[key] += count;
    }
    for (const auto& [op, count] : root_diag.external_leaf_ops) {
      chunk_diag.external_leaf_ops[op] += count;
    }
    if (mixed_key_diag && root_diag.dot_keys.size() > 1) {
      int64_t root_dot_count = 0;
      for (const auto& item : root_diag.dot_keys) {
        root_dot_count += item.second;
      }
      const bool rewritable =
          root_diag.blocked_nodes == 0 &&
          root_diag.external_unsupported_dots == 0 &&
          root_diag.external_leaves == 0 && root_dot_count >= 2;
      ++chunk_diag.mixed_key_roots;
      chunk_diag.mixed_key_dots += root_dot_count;
      chunk_diag.mixed_key_external_dots +=
          root_diag.external_supported_dots;
      if (rewritable) {
        ++chunk_diag.mixed_key_rewritable_roots;
      }
      ++chunk_diag.mixed_key_root_patterns[absl::StrCat(
          "rewritable=", rewritable, " keys=", root_diag.dot_keys.size(),
          " dots=", root_dot_count,
          " selected=", root_diag.selected_dot_leaves.size(),
          " ext_dot=", root_diag.external_supported_dots,
          " ext=", root_diag.external_leaves,
          " blocked=", root_diag.blocked_nodes)];
    }
    const int64_t selected_leaf_count =
        root_diag.selected_dot_leaves.size();
    const bool candidate = root_diag.blocked_nodes == 0 &&
                           root_diag.external_dot_leaves == 0 &&
                           root_diag.external_leaves == 0 &&
                           selected_leaf_count >= 2;
    if (candidate) {
      ++chunk_diag.candidate_roots;
      chunk_diag.candidate_dots += selected_leaf_count;
      chunk_diag.candidate_adds += root_diag.add_nodes;
      chunk_diag.estimated_slice_reduction += selected_leaf_count - 1;
      chunk_diag.estimated_reshape_reduction += selected_leaf_count - 1;
      chunk_diag.estimated_slice_bytes_reduction +=
          (selected_leaf_count - 1) * output_bytes;
    }
    ++chunk_diag.root_patterns[absl::StrCat(
        "candidate=", candidate, " dots=", selected_leaf_count,
        " adds=", root_diag.add_nodes,
        " bcast_const=", root_diag.broadcast_const_leaves,
        " ext_dot=", root_diag.external_dot_leaves,
        " ext=", root_diag.external_leaves,
        " blocked=", root_diag.blocked_nodes)];
  }
  return chunk_diag;
}

bool CollectMusaSameShapeDotAddTreeRewriteNode(
    HloInstruction* instr,
    const absl::flat_hash_map<HloInstruction*, int64_t>& selected_dot_indices,
    const Shape& output_shape, MusaSameShapeDotAddTreeRewritePlan* plan,
    absl::flat_hash_set<HloInstruction*>* seen_adds) {
  auto dot_index = selected_dot_indices.find(instr);
  if (dot_index != selected_dot_indices.end()) {
    plan->dot_indices.push_back(dot_index->second);
    return true;
  }
  if (IsMusaSameShapeBroadcastConstLeaf(instr, output_shape)) {
    plan->broadcast_leaves.push_back(instr);
    return true;
  }
  if (!MusaSameShapeDotAddTreeRootBlockReason(instr, output_shape).empty()) {
    return false;
  }
  if (!seen_adds->insert(instr).second) {
    return false;
  }
  ++plan->add_nodes;
  return CollectMusaSameShapeDotAddTreeRewriteNode(
             instr->mutable_operand(0), selected_dot_indices, output_shape,
             plan, seen_adds) &&
         CollectMusaSameShapeDotAddTreeRewriteNode(
             instr->mutable_operand(1), selected_dot_indices, output_shape,
             plan, seen_adds);
}

bool MusaSameShapeDotIndicesAreContiguous(std::vector<int64_t>* indices) {
  std::sort(indices->begin(), indices->end());
  if (indices->size() < 2) {
    return false;
  }
  for (int64_t i = 1; i < indices->size(); ++i) {
    if ((*indices)[i] == (*indices)[i - 1]) {
      return false;
    }
    if ((*indices)[i] != (*indices)[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

std::vector<MusaSameShapeDotAddTreeRewritePlan>
BuildMusaSameShapeDotAddTreeRewritePlans(
    absl::Span<HloInstruction* const> dots,
    MusaSameShapeDotAddTreeRewriteStats* stats) {
  std::vector<MusaSameShapeDotAddTreeRewritePlan> plans;
  if (dots.empty()) {
    return plans;
  }
  const Shape& output_shape = dots.front()->shape();
  absl::flat_hash_map<HloInstruction*, int64_t> selected_dot_indices;
  selected_dot_indices.reserve(dots.size());
  for (int64_t i = 0; i < dots.size(); ++i) {
    selected_dot_indices[dots[i]] = i;
  }

  absl::flat_hash_set<HloInstruction*> roots;
  for (HloInstruction* dot : dots) {
    HloInstruction* root = const_cast<HloInstruction*>(
        MusaSameShapeDotAddTreeRoot(dot, output_shape, &stats->skipped_reasons));
    if (root != nullptr) {
      roots.insert(root);
    }
  }

  absl::flat_hash_set<int64_t> claimed_dot_indices;
  for (HloInstruction* root : roots) {
    MusaSameShapeDotAddTreeRewritePlan plan;
    plan.root = root;
    absl::flat_hash_set<HloInstruction*> seen_adds;
    if (!CollectMusaSameShapeDotAddTreeRewriteNode(
            root, selected_dot_indices, output_shape, &plan, &seen_adds)) {
      ++stats->skipped_roots;
      ++stats->skipped_reasons["external_or_unsupported_leaf"];
      continue;
    }
    if (!MusaSameShapeDotIndicesAreContiguous(&plan.dot_indices)) {
      ++stats->skipped_roots;
      ++stats->skipped_reasons["noncontiguous_or_duplicate_dot_lanes"];
      continue;
    }
    bool overlaps = false;
    for (int64_t index : plan.dot_indices) {
      if (claimed_dot_indices.find(index) != claimed_dot_indices.end()) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) {
      ++stats->skipped_roots;
      ++stats->skipped_reasons["overlapping_dot_lanes"];
      continue;
    }
    for (int64_t index : plan.dot_indices) {
      claimed_dot_indices.insert(index);
    }
    plans.push_back(std::move(plan));
  }
  return plans;
}

HloComputation* GetOrCreateMusaSameShapeAddReduceComputation(
    HloModule* module, HloComputation** add_reduce_computation) {
  if (*add_reduce_computation != nullptr) {
    return *add_reduce_computation;
  }
  HloComputation::Builder builder("musa_same_shape_add_tree_reduce_add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, lhs, rhs));
  *add_reduce_computation = module->AddEmbeddedComputation(builder.Build(add));
  return *add_reduce_computation;
}

struct MusaSameShapeDotMixedKeyAddTreePlan {
  HloInstruction* root = nullptr;
  std::vector<std::pair<HloInstruction*, MusaSameShapeDotBatchLaneRef>> lanes;
  std::vector<HloInstruction*> broadcast_leaves;
  int64_t add_nodes = 0;
};

HloInstruction* MusaSameShapeDotAddTreeRootFromBatchLane(
    HloInstruction* lane, const Shape& output_shape) {
  if (lane->users().size() != 1) {
    return nullptr;
  }
  HloInstruction* root = lane->users()[0];
  if (!MusaSameShapeDotAddTreeRootBlockReason(root, output_shape).empty()) {
    return nullptr;
  }
  absl::flat_hash_set<HloInstruction*> seen;
  seen.insert(root);
  while (root->users().size() == 1) {
    HloInstruction* next = root->users()[0];
    if (!MusaSameShapeDotAddTreeRootBlockReason(next, output_shape).empty() ||
        !seen.insert(next).second) {
      break;
    }
    root = next;
  }
  return root;
}

bool CollectMusaSameShapeDotMixedKeyAddTreeNode(
    HloInstruction* instr,
    const absl::flat_hash_map<HloInstruction*, MusaSameShapeDotBatchLaneRef>&
        lane_refs,
    const Shape& output_shape, MusaSameShapeDotMixedKeyAddTreePlan* plan,
    absl::flat_hash_set<HloInstruction*>* seen_adds) {
  auto lane = lane_refs.find(instr);
  if (lane != lane_refs.end()) {
    plan->lanes.push_back({instr, lane->second});
    return true;
  }
  if (IsMusaSameShapeBroadcastConstLeaf(instr, output_shape)) {
    plan->broadcast_leaves.push_back(instr);
    return true;
  }
  if (!MusaSameShapeDotAddTreeRootBlockReason(instr, output_shape).empty() ||
      !seen_adds->insert(instr).second) {
    return false;
  }
  ++plan->add_nodes;
  return CollectMusaSameShapeDotMixedKeyAddTreeNode(
             instr->mutable_operand(0), lane_refs, output_shape, plan,
             seen_adds) &&
         CollectMusaSameShapeDotMixedKeyAddTreeNode(
             instr->mutable_operand(1), lane_refs, output_shape, plan,
             seen_adds);
}

Status RewriteMusaSameShapeDotMixedKeyAddTrees(
    HloComputation* computation,
    const absl::flat_hash_map<HloInstruction*, MusaSameShapeDotBatchLaneRef>&
        lane_refs,
    HloComputation** add_reduce_computation,
    MusaSameShapeDotMixedKeyAddTreeRewriteStats* stats) {
  absl::flat_hash_set<HloInstruction*> roots;
  for (const auto& item : lane_refs) {
    HloInstruction* lane = item.first;
    HloInstruction* root =
        MusaSameShapeDotAddTreeRootFromBatchLane(lane, lane->shape());
    if (root != nullptr) {
      roots.insert(root);
    }
  }

  for (HloInstruction* root : roots) {
    MusaSameShapeDotMixedKeyAddTreePlan plan;
    plan.root = root;
    absl::flat_hash_set<HloInstruction*> seen_adds;
    if (!CollectMusaSameShapeDotMixedKeyAddTreeNode(
            root, lane_refs, root->shape(), &plan, &seen_adds)) {
      ++stats->skipped_roots;
      ++stats->skipped_reasons["external_or_unsupported_leaf"];
      continue;
    }

    absl::flat_hash_map<
        HloInstruction*,
        std::vector<std::pair<int64_t, HloInstruction*>>>
        lanes_by_batch_result;
    bool shape_mismatch = false;
    for (const auto& [lane, ref] : plan.lanes) {
      if (ref.batch_result == nullptr || ref.m != root->shape().dimensions(0) ||
          ref.n != root->shape().dimensions(1)) {
        shape_mismatch = true;
        break;
      }
      lanes_by_batch_result[ref.batch_result].push_back({ref.index, lane});
    }
    if (shape_mismatch || lanes_by_batch_result.size() < 2 ||
        plan.lanes.size() < 2) {
      ++stats->skipped_roots;
      ++stats->skipped_reasons[shape_mismatch ? "shape_mismatch"
                                               : "not_mixed_batch_result"];
      continue;
    }

    bool duplicate_lane = false;
    int64_t run_count = 0;
    std::vector<HloInstruction*> reduced_terms;
    for (auto& [batch_result, indexed_lanes] : lanes_by_batch_result) {
      std::sort(indexed_lanes.begin(), indexed_lanes.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      for (int64_t i = 1; i < indexed_lanes.size(); ++i) {
        if (indexed_lanes[i - 1].first == indexed_lanes[i].first) {
          duplicate_lane = true;
          break;
        }
      }
      if (duplicate_lane) {
        break;
      }

      for (int64_t begin = 0; begin < indexed_lanes.size();) {
        int64_t end = begin + 1;
        while (end < indexed_lanes.size() &&
               indexed_lanes[end].first == indexed_lanes[end - 1].first + 1) {
          ++end;
        }
        ++run_count;
        const int64_t count = end - begin;
        if (count == 1) {
          reduced_terms.push_back(indexed_lanes[begin].second);
          begin = end;
          continue;
        }

        const int64_t start = indexed_lanes[begin].first;
        const int64_t m = root->shape().dimensions(0);
        const int64_t n = root->shape().dimensions(1);
        Shape slice_shape = ShapeUtil::MakeShapeWithDenseLayout(
            F32, {count, m, n}, {2, 1, 0});
        const std::vector<int64_t> start_indices = {start, 0, 0};
        const std::vector<int64_t> limit_indices = {start + count, m, n};
        const std::vector<int64_t> strides = {1, 1, 1};
        HloInstruction* slice = computation->AddInstruction(
            HloInstruction::CreateSlice(slice_shape, batch_result,
                                        start_indices, limit_indices, strides));
        HloInstruction* zero = computation->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
        HloComputation* reduce_computation =
            GetOrCreateMusaSameShapeAddReduceComputation(
                computation->parent(), add_reduce_computation);
        HloInstruction* reduce = computation->AddInstruction(
            HloInstruction::CreateReduce(root->shape(), slice, zero, {0},
                                         reduce_computation));
        reduce->set_metadata(root->metadata());
        reduced_terms.push_back(reduce);
        begin = end;
      }
    }
    if (duplicate_lane) {
      ++stats->skipped_roots;
      ++stats->skipped_reasons["duplicate_lane"];
      continue;
    }

    reduced_terms.insert(reduced_terms.end(), plan.broadcast_leaves.begin(),
                         plan.broadcast_leaves.end());
    if (reduced_terms.size() < 2) {
      ++stats->skipped_roots;
      ++stats->skipped_reasons["insufficient_reduced_terms"];
      continue;
    }
    HloInstruction* rewritten_root = reduced_terms.front();
    for (int64_t i = 1; i < reduced_terms.size(); ++i) {
      rewritten_root = computation->AddInstruction(
          HloInstruction::CreateBinary(root->shape(), HloOpcode::kAdd,
                                       rewritten_root, reduced_terms[i]));
      rewritten_root->set_metadata(root->metadata());
    }
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(root, rewritten_root));
    ++stats->rewritten_roots;
    stats->rewritten_dots += plan.lanes.size();
    stats->rewritten_adds += plan.add_nodes;
    stats->estimated_slice_reduction += plan.lanes.size() - run_count;
    stats->estimated_reshape_reduction += plan.lanes.size() - run_count;
    ++stats->rewritten_patterns[absl::StrCat(
        "keys=", lanes_by_batch_result.size(), " dots=", plan.lanes.size(),
        " runs=", run_count, " adds=", plan.add_nodes,
        " bcast_const=", plan.broadcast_leaves.size())];
  }
  return OkStatus();
}

bool IsSupportedMusaSameLhsDot(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kDot || instr->operand_count() != 2 ||
      !instr->control_predecessors().empty() ||
      !instr->control_successors().empty() || instr->shape().rank() != 2) {
    return false;
  }
  const HloInstruction* lhs = instr->operand(0);
  const HloInstruction* rhs = instr->operand(1);
  if (lhs->shape().rank() != 2 || rhs->shape().rank() != 2 ||
      instr->shape().element_type() != F32 ||
      lhs->shape().element_type() != F32 || rhs->shape().element_type() != F32 ||
      !HasDenseRowMajorLayout(lhs->shape()) ||
      !HasDenseRowMajorLayout(rhs->shape()) ||
      !HasDenseRowMajorLayout(instr->shape())) {
    return false;
  }
  const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
  if (dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_contracting_dimensions(0) != 1 ||
      dnums.rhs_contracting_dimensions(0) != 0) {
    return false;
  }
  const int64_t m = lhs->shape().dimensions(0);
  const int64_t k = lhs->shape().dimensions(1);
  const int64_t n = rhs->shape().dimensions(1);
  return rhs->shape().dimensions(0) == k && instr->shape().dimensions(0) == m &&
         instr->shape().dimensions(1) == n && m > 0 && k > 0 && n > 0;
}

std::string MusaSameLhsDotKey(const HloInstruction* instr) {
  const Shape& lhs = instr->operand(0)->shape();
  return absl::StrCat(instr->operand(0)->name(), "_",
                      lhs.dimensions(0), "x", lhs.dimensions(1), "_",
                      MusaPrecisionConfigKey(instr->precision_config()));
}

std::string MusaSameRhsDotKey(const HloInstruction* instr) {
  const Shape& rhs = instr->operand(1)->shape();
  return absl::StrCat(instr->operand(1)->name(), "_",
                      rhs.dimensions(0), "x", rhs.dimensions(1), "_",
                      MusaPrecisionConfigKey(instr->precision_config()));
}

const HloInstruction* StripMusaTrivialDotOperandWrappers(
    const HloInstruction* instr) {
  for (int i = 0; i < 8; ++i) {
    if (instr->operand_count() != 1) {
      return instr;
    }
    if (instr->opcode() != HloOpcode::kBitcast &&
        instr->opcode() != HloOpcode::kCopy &&
        instr->opcode() != HloOpcode::kReshape) {
      return instr;
    }
    instr = instr->operand(0);
  }
  return instr;
}

std::string MusaNormalizedLhsDotKey(const HloInstruction* instr) {
  const Shape& lhs = instr->operand(0)->shape();
  const HloInstruction* base =
      StripMusaTrivialDotOperandWrappers(instr->operand(0));
  return absl::StrCat(base->name(), "_", lhs.dimensions(0), "x",
                      lhs.dimensions(1), "_",
                      MusaPrecisionConfigKey(instr->precision_config()));
}

std::string MusaNormalizedRhsDotKey(const HloInstruction* instr) {
  const Shape& rhs = instr->operand(1)->shape();
  const HloInstruction* base =
      StripMusaTrivialDotOperandWrappers(instr->operand(1));
  return absl::StrCat(base->name(), "_", rhs.dimensions(0), "x",
                      rhs.dimensions(1), "_",
                      MusaPrecisionConfigKey(instr->precision_config()));
}

Status MergeSameLhsDotChunk(HloComputation* computation,
                            absl::Span<HloInstruction* const> dots) {
  CHECK_GE(dots.size(), 2);
  HloInstruction* first = dots.front();
  HloInstruction* lhs = first->mutable_operand(0);
  const int64_t m = lhs->shape().dimensions(0);
  const int64_t k = lhs->shape().dimensions(1);

  int64_t total_n = 0;
  std::vector<HloInstruction*> rhs_operands;
  rhs_operands.reserve(dots.size());
  for (HloInstruction* dot : dots) {
    rhs_operands.push_back(dot->mutable_operand(1));
    total_n += dot->operand(1)->shape().dimensions(1);
  }

  Shape rhs_concat_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {k, total_n}, {1, 0});
  HloInstruction* rhs_concat = computation->AddInstruction(
      HloInstruction::CreateConcatenate(rhs_concat_shape, rhs_operands, 1));
  rhs_concat->set_metadata(first->metadata());

  Shape merged_dot_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {m, total_n}, {1, 0});
  HloInstruction* merged_dot = computation->AddInstruction(
      HloInstruction::CreateDot(merged_dot_shape, lhs, rhs_concat,
                                first->dot_dimension_numbers(),
                                first->precision_config()));
  merged_dot->set_metadata(first->metadata());

  int64_t offset = 0;
  for (HloInstruction* old_dot : dots) {
    const int64_t n = old_dot->shape().dimensions(1);
    const std::vector<int64_t> start_indices = {0, offset};
    const std::vector<int64_t> limit_indices = {m, offset + n};
    const std::vector<int64_t> strides = {1, 1};
    HloInstruction* slice = computation->AddInstruction(
        HloInstruction::CreateSlice(old_dot->shape(), merged_dot, start_indices,
                                    limit_indices, strides));
    slice->set_metadata(old_dot->metadata());
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(old_dot, slice));
    offset += n;
  }
  return OkStatus();
}

Status MergeSameRhsDotChunk(HloComputation* computation,
                            absl::Span<HloInstruction* const> dots) {
  CHECK_GE(dots.size(), 2);
  HloInstruction* first = dots.front();
  HloInstruction* rhs = first->mutable_operand(1);
  const int64_t k = rhs->shape().dimensions(0);
  const int64_t n = rhs->shape().dimensions(1);

  int64_t total_m = 0;
  std::vector<HloInstruction*> lhs_operands;
  lhs_operands.reserve(dots.size());
  for (HloInstruction* dot : dots) {
    lhs_operands.push_back(dot->mutable_operand(0));
    total_m += dot->operand(0)->shape().dimensions(0);
  }

  Shape lhs_concat_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {total_m, k}, {1, 0});
  HloInstruction* lhs_concat = computation->AddInstruction(
      HloInstruction::CreateConcatenate(lhs_concat_shape, lhs_operands, 0));
  lhs_concat->set_metadata(first->metadata());

  Shape merged_dot_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {total_m, n}, {1, 0});
  HloInstruction* merged_dot = computation->AddInstruction(
      HloInstruction::CreateDot(merged_dot_shape, lhs_concat, rhs,
                                first->dot_dimension_numbers(),
                                first->precision_config()));
  merged_dot->set_metadata(first->metadata());

  int64_t offset = 0;
  for (HloInstruction* old_dot : dots) {
    const int64_t m = old_dot->shape().dimensions(0);
    const std::vector<int64_t> start_indices = {offset, 0};
    const std::vector<int64_t> limit_indices = {offset + m, n};
    const std::vector<int64_t> strides = {1, 1};
    HloInstruction* slice = computation->AddInstruction(
        HloInstruction::CreateSlice(old_dot->shape(), merged_dot, start_indices,
                                    limit_indices, strides));
    slice->set_metadata(old_dot->metadata());
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(old_dot, slice));
    offset += m;
  }
  return OkStatus();
}

class MusaSameLhsDotMerger : public HloModulePass {
 public:
  absl::string_view name() const override { return "musa-same-lhs-dot-merger"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override {
    const char* enabled = std::getenv("MUSA_XLA_SAME_LHS_DOT_MERGER");
    if (enabled == nullptr || enabled[0] == '\0') {
      return false;
    }
    if (EnvExplicitlyFalse("MUSA_XLA_SAME_LHS_DOT_MERGER")) {
      return false;
    }

    const bool forced = EnvExplicitlyTrue("MUSA_XLA_SAME_LHS_DOT_MERGER");
    const int64_t min_group_size =
        ReadInt64Env("MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_GROUP_SIZE", 2);
    const int64_t max_groups =
        ReadInt64Env("MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUPS", 8);
    const int64_t max_group_size =
        ReadInt64Env("MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUP_SIZE", 16);
    const int64_t max_total_cols =
        ReadInt64Env("MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_TOTAL_COLS", 2048);
    const int64_t min_candidate_dots = ReadInt64Env(
        "MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_CANDIDATE_DOTS", 128);
    const bool normalize_operands = EnvExplicitlyTrue(
        "MUSA_XLA_SAME_LHS_DOT_MERGER_NORMALIZE_OPERANDS");
    if (max_group_size < min_group_size) {
      return false;
    }

    struct Candidate {
      std::vector<HloInstruction*> dots;
      int64_t total_cols = 0;
    };
    struct DotFilterStats {
      int64_t total_dots = 0;
      int64_t skipped_operand_or_control = 0;
      int64_t skipped_rank = 0;
      int64_t skipped_dtype = 0;
      int64_t skipped_layout = 0;
      int64_t skipped_dot_dims = 0;
      int64_t skipped_shape_mismatch = 0;
      int64_t supported_dots = 0;
      int64_t raw_same_lhs_groups = 0;
      int64_t raw_same_lhs_dots = 0;
      int64_t raw_same_rhs_groups = 0;
      int64_t raw_same_rhs_dots = 0;
      int64_t normalized_same_lhs_groups = 0;
      int64_t normalized_same_lhs_dots = 0;
      int64_t normalized_same_rhs_groups = 0;
      int64_t normalized_same_rhs_dots = 0;
    } stats;
    std::vector<std::pair<HloComputation*, Candidate>> candidates;
    int64_t candidate_dot_count = 0;

    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      absl::flat_hash_map<std::string, std::vector<HloInstruction*>> groups;
      absl::flat_hash_map<std::string, std::vector<HloInstruction*>>
          rhs_groups;
      absl::flat_hash_map<std::string, std::vector<HloInstruction*>>
          normalized_lhs_groups;
      absl::flat_hash_map<std::string, std::vector<HloInstruction*>>
          normalized_rhs_groups;
      for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
        if (instr->opcode() != HloOpcode::kDot) {
          continue;
        }
        ++stats.total_dots;
        if (instr->operand_count() != 2 ||
            !instr->control_predecessors().empty() ||
            !instr->control_successors().empty()) {
          ++stats.skipped_operand_or_control;
          continue;
        }
        if (instr->shape().rank() != 2 ||
            instr->operand(0)->shape().rank() != 2 ||
            instr->operand(1)->shape().rank() != 2) {
          ++stats.skipped_rank;
          continue;
        }
        const HloInstruction* lhs = instr->operand(0);
        const HloInstruction* rhs = instr->operand(1);
        if (instr->shape().element_type() != F32 ||
            lhs->shape().element_type() != F32 ||
            rhs->shape().element_type() != F32) {
          ++stats.skipped_dtype;
          continue;
        }
        if (!HasDenseRowMajorLayout(lhs->shape()) ||
            !HasDenseRowMajorLayout(rhs->shape()) ||
            !HasDenseRowMajorLayout(instr->shape())) {
          ++stats.skipped_layout;
          continue;
        }
        const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
        if (dnums.lhs_batch_dimensions_size() != 0 ||
            dnums.rhs_batch_dimensions_size() != 0 ||
            dnums.lhs_contracting_dimensions_size() != 1 ||
            dnums.rhs_contracting_dimensions_size() != 1 ||
            dnums.lhs_contracting_dimensions(0) != 1 ||
            dnums.rhs_contracting_dimensions(0) != 0) {
          ++stats.skipped_dot_dims;
          continue;
        }
        const int64_t m = lhs->shape().dimensions(0);
        const int64_t k = lhs->shape().dimensions(1);
        const int64_t n = rhs->shape().dimensions(1);
        if (rhs->shape().dimensions(0) != k ||
            instr->shape().dimensions(0) != m ||
            instr->shape().dimensions(1) != n || m <= 0 || k <= 0 || n <= 0) {
          ++stats.skipped_shape_mismatch;
          continue;
        }
        ++stats.supported_dots;
        groups[normalize_operands ? MusaNormalizedLhsDotKey(instr)
                                  : MusaSameLhsDotKey(instr)]
            .push_back(instr);
        rhs_groups[MusaSameRhsDotKey(instr)].push_back(instr);
        normalized_lhs_groups[MusaNormalizedLhsDotKey(instr)].push_back(instr);
        normalized_rhs_groups[MusaNormalizedRhsDotKey(instr)].push_back(instr);
      }
      for (const auto& item : rhs_groups) {
        if (item.second.size() >= 2) {
          ++stats.raw_same_rhs_groups;
          stats.raw_same_rhs_dots += item.second.size();
        }
      }
      for (const auto& item : normalized_lhs_groups) {
        if (item.second.size() >= 2) {
          ++stats.normalized_same_lhs_groups;
          stats.normalized_same_lhs_dots += item.second.size();
        }
      }
      for (const auto& item : normalized_rhs_groups) {
        if (item.second.size() >= 2) {
          ++stats.normalized_same_rhs_groups;
          stats.normalized_same_rhs_dots += item.second.size();
        }
      }
      for (auto& item : groups) {
        if (item.second.size() >= 2) {
          ++stats.raw_same_lhs_groups;
          stats.raw_same_lhs_dots += item.second.size();
        }
        if (item.second.size() < min_group_size) {
          continue;
        }
        std::stable_sort(item.second.begin(), item.second.end(),
                         [](const HloInstruction* a, const HloInstruction* b) {
                           return a->shape().dimensions(1) >
                                  b->shape().dimensions(1);
                         });
        Candidate candidate;
        auto maybe_push_candidate = [&]() {
          if (candidate.dots.size() >= min_group_size) {
            candidate_dot_count += candidate.dots.size();
            candidates.push_back(
                std::make_pair(computation, std::move(candidate)));
          }
          candidate = Candidate{};
        };
        for (HloInstruction* dot : item.second) {
          const int64_t n = dot->shape().dimensions(1);
          if (n > max_total_cols) {
            continue;
          }
          if (!candidate.dots.empty() &&
              (candidate.dots.size() >= max_group_size ||
               candidate.total_cols + n > max_total_cols)) {
            maybe_push_candidate();
          }
          candidate.total_cols += n;
          candidate.dots.push_back(dot);
        }
        maybe_push_candidate();
      }
    }

    if (!forced && candidate_dot_count < min_candidate_dots) {
      if (EnvExplicitlyTrue("MUSA_XLA_SAME_LHS_DOT_MERGER_LOG")) {
        LOG(INFO) << "[MUSA_SAME_LHS_DOT_MERGER] module=" << module->name()
                  << " changed=false candidate_dots=" << candidate_dot_count
                  << " total_dots=" << stats.total_dots
                  << " supported_dots=" << stats.supported_dots
                  << " skip_operand_or_control="
                  << stats.skipped_operand_or_control
                  << " skip_rank=" << stats.skipped_rank
                  << " skip_dtype=" << stats.skipped_dtype
                  << " skip_layout=" << stats.skipped_layout
                  << " skip_dot_dims=" << stats.skipped_dot_dims
                  << " skip_shape_mismatch=" << stats.skipped_shape_mismatch
                  << " raw_same_lhs_groups=" << stats.raw_same_lhs_groups
                  << " raw_same_lhs_dots=" << stats.raw_same_lhs_dots
                  << " raw_same_rhs_groups=" << stats.raw_same_rhs_groups
                  << " raw_same_rhs_dots=" << stats.raw_same_rhs_dots
                  << " normalized_same_lhs_groups="
                  << stats.normalized_same_lhs_groups
                  << " normalized_same_lhs_dots="
                  << stats.normalized_same_lhs_dots
                  << " normalized_same_rhs_groups="
                  << stats.normalized_same_rhs_groups
                  << " normalized_same_rhs_dots="
                  << stats.normalized_same_rhs_dots
                  << " normalize_operands=" << normalize_operands
                  << " min_candidate_dots=" << min_candidate_dots;
      }
      return false;
    }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) {
                       if (a.second.dots.size() != b.second.dots.size()) {
                         return a.second.dots.size() > b.second.dots.size();
                       }
                       return a.second.total_cols > b.second.total_cols;
                     });

    bool changed = false;
    int64_t groups_rewritten = 0;
    int64_t dots_merged = 0;
    int64_t columns_merged = 0;
    absl::flat_hash_set<const HloInstruction*> rewritten;
    for (auto& item : candidates) {
      if (groups_rewritten >= max_groups) {
        break;
      }
      std::vector<HloInstruction*> available;
      available.reserve(item.second.dots.size());
      for (HloInstruction* dot : item.second.dots) {
        if (rewritten.find(dot) == rewritten.end()) {
          available.push_back(dot);
        }
      }
      if (available.size() < min_group_size) {
        continue;
      }
      for (HloInstruction* dot : available) {
        rewritten.insert(dot);
      }
      TF_RETURN_IF_ERROR(MergeSameLhsDotChunk(
          item.first, absl::Span<HloInstruction* const>(available.data(),
                                                        available.size())));
      changed = true;
      ++groups_rewritten;
      dots_merged += available.size();
      columns_merged += item.second.total_cols;
    }

    if (changed || EnvExplicitlyTrue("MUSA_XLA_SAME_LHS_DOT_MERGER_LOG")) {
      LOG(INFO) << "[MUSA_SAME_LHS_DOT_MERGER] module=" << module->name()
                << " changed=" << changed
                << " forced=" << forced
                << " candidate_dots=" << candidate_dot_count
                << " total_dots=" << stats.total_dots
                << " supported_dots=" << stats.supported_dots
                << " skip_operand_or_control="
                << stats.skipped_operand_or_control
                << " skip_rank=" << stats.skipped_rank
                << " skip_dtype=" << stats.skipped_dtype
                << " skip_layout=" << stats.skipped_layout
                << " skip_dot_dims=" << stats.skipped_dot_dims
                << " skip_shape_mismatch=" << stats.skipped_shape_mismatch
                << " raw_same_lhs_groups=" << stats.raw_same_lhs_groups
                << " raw_same_lhs_dots=" << stats.raw_same_lhs_dots
                << " raw_same_rhs_groups=" << stats.raw_same_rhs_groups
                << " raw_same_rhs_dots=" << stats.raw_same_rhs_dots
                << " normalized_same_lhs_groups="
                << stats.normalized_same_lhs_groups
                << " normalized_same_lhs_dots="
                << stats.normalized_same_lhs_dots
                << " normalized_same_rhs_groups="
                << stats.normalized_same_rhs_groups
                << " normalized_same_rhs_dots="
                << stats.normalized_same_rhs_dots
                << " groups_rewritten=" << groups_rewritten
                << " dots_merged=" << dots_merged
                << " columns_merged=" << columns_merged
                << " normalize_operands=" << normalize_operands
                << " min_group_size=" << min_group_size
                << " max_group_size=" << max_group_size
                << " max_groups=" << max_groups
                << " max_total_cols=" << max_total_cols
                << " min_candidate_dots=" << min_candidate_dots;
    }
    return changed;
  }
};

class MusaSameRhsDotMerger : public HloModulePass {
 public:
  absl::string_view name() const override { return "musa-same-rhs-dot-merger"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override {
    const char* enabled = std::getenv("MUSA_XLA_SAME_RHS_DOT_MERGER");
    if (enabled == nullptr || enabled[0] == '\0') {
      return false;
    }
    if (EnvExplicitlyFalse("MUSA_XLA_SAME_RHS_DOT_MERGER")) {
      return false;
    }

    const bool forced = EnvExplicitlyTrue("MUSA_XLA_SAME_RHS_DOT_MERGER");
    const int64_t min_group_size =
        ReadInt64Env("MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_GROUP_SIZE", 2);
    const int64_t max_groups =
        ReadInt64Env("MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUPS", 8);
    const int64_t max_group_size =
        ReadInt64Env("MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUP_SIZE", 8);
    const int64_t max_total_rows =
        ReadInt64Env("MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_TOTAL_ROWS", 4096);
    const int64_t min_candidate_dots = ReadInt64Env(
        "MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_CANDIDATE_DOTS", 128);
    if (max_group_size < min_group_size) {
      return false;
    }

    struct Candidate {
      std::vector<HloInstruction*> dots;
      int64_t total_rows = 0;
    };
    std::vector<std::pair<HloComputation*, Candidate>> candidates;
    int64_t total_dots = 0;
    int64_t supported_dots = 0;
    int64_t raw_same_rhs_groups = 0;
    int64_t raw_same_rhs_dots = 0;
    int64_t candidate_dot_count = 0;

    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      absl::flat_hash_map<std::string, std::vector<HloInstruction*>> groups;
      for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
        if (instr->opcode() != HloOpcode::kDot) {
          continue;
        }
        ++total_dots;
        if (!IsSupportedMusaSameLhsDot(instr)) {
          continue;
        }
        ++supported_dots;
        groups[MusaSameRhsDotKey(instr)].push_back(instr);
      }

      for (auto& item : groups) {
        if (item.second.size() >= 2) {
          ++raw_same_rhs_groups;
          raw_same_rhs_dots += item.second.size();
        }
        if (item.second.size() < min_group_size) {
          continue;
        }
        std::stable_sort(item.second.begin(), item.second.end(),
                         [](const HloInstruction* a, const HloInstruction* b) {
                           return a->shape().dimensions(0) >
                                  b->shape().dimensions(0);
                         });
        Candidate candidate;
        auto maybe_push_candidate = [&]() {
          if (candidate.dots.size() >= min_group_size) {
            candidate_dot_count += candidate.dots.size();
            candidates.push_back(
                std::make_pair(computation, std::move(candidate)));
          }
          candidate = Candidate{};
        };
        for (HloInstruction* dot : item.second) {
          const int64_t m = dot->shape().dimensions(0);
          if (m > max_total_rows) {
            continue;
          }
          if (!candidate.dots.empty() &&
              (candidate.dots.size() >= max_group_size ||
               candidate.total_rows + m > max_total_rows)) {
            maybe_push_candidate();
          }
          candidate.total_rows += m;
          candidate.dots.push_back(dot);
        }
        maybe_push_candidate();
      }
    }

    if (!forced && candidate_dot_count < min_candidate_dots) {
      if (EnvExplicitlyTrue("MUSA_XLA_SAME_RHS_DOT_MERGER_LOG")) {
        LOG(INFO) << "[MUSA_SAME_RHS_DOT_MERGER] module=" << module->name()
                  << " changed=false candidate_dots=" << candidate_dot_count
                  << " total_dots=" << total_dots
                  << " supported_dots=" << supported_dots
                  << " raw_same_rhs_groups=" << raw_same_rhs_groups
                  << " raw_same_rhs_dots=" << raw_same_rhs_dots
                  << " min_candidate_dots=" << min_candidate_dots;
      }
      return false;
    }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) {
                       if (a.second.dots.size() != b.second.dots.size()) {
                         return a.second.dots.size() > b.second.dots.size();
                       }
                       return a.second.total_rows > b.second.total_rows;
                     });

    bool changed = false;
    int64_t groups_rewritten = 0;
    int64_t dots_merged = 0;
    int64_t rows_merged = 0;
    absl::flat_hash_set<const HloInstruction*> rewritten;
    for (auto& item : candidates) {
      if (groups_rewritten >= max_groups) {
        break;
      }
      std::vector<HloInstruction*> available;
      available.reserve(item.second.dots.size());
      for (HloInstruction* dot : item.second.dots) {
        if (rewritten.find(dot) == rewritten.end()) {
          available.push_back(dot);
        }
      }
      if (available.size() < min_group_size) {
        continue;
      }
      for (HloInstruction* dot : available) {
        rewritten.insert(dot);
      }
      TF_RETURN_IF_ERROR(MergeSameRhsDotChunk(
          item.first, absl::Span<HloInstruction* const>(available.data(),
                                                        available.size())));
      changed = true;
      ++groups_rewritten;
      dots_merged += available.size();
      rows_merged += item.second.total_rows;
    }

    if (changed || EnvExplicitlyTrue("MUSA_XLA_SAME_RHS_DOT_MERGER_LOG")) {
      LOG(INFO) << "[MUSA_SAME_RHS_DOT_MERGER] module=" << module->name()
                << " changed=" << changed
                << " forced=" << forced
                << " candidate_dots=" << candidate_dot_count
                << " total_dots=" << total_dots
                << " supported_dots=" << supported_dots
                << " raw_same_rhs_groups=" << raw_same_rhs_groups
                << " raw_same_rhs_dots=" << raw_same_rhs_dots
                << " groups_rewritten=" << groups_rewritten
                << " dots_merged=" << dots_merged
                << " rows_merged=" << rows_merged
                << " min_group_size=" << min_group_size
                << " max_group_size=" << max_group_size
                << " max_groups=" << max_groups
                << " max_total_rows=" << max_total_rows
                << " min_candidate_dots=" << min_candidate_dots;
    }
    return changed;
  }
};

Status BatchSameShapeDotChunk(HloComputation* computation,
                              absl::Span<HloInstruction* const> dots,
                              bool pointer_array_output, bool fuse_biasadd,
                              bool rewrite_add_tree,
                              HloComputation** add_reduce_computation,
                              absl::flat_hash_map<
                                  HloInstruction*,
                                  MusaSameShapeDotBatchLaneRef>* lane_refs,
                              bool* biasadd_fused,
                              MusaSameShapeDotAddTreeRewriteStats*
                                  add_tree_rewrite_stats) {
  CHECK_GE(dots.size(), 2);
  *biasadd_fused = false;
  HloInstruction* first = dots.front();
  const int64_t group_size = dots.size();
  const int64_t m = first->operand(0)->shape().dimensions(0);
  const int64_t k = first->operand(0)->shape().dimensions(1);
  const int64_t n = first->operand(1)->shape().dimensions(1);

  std::vector<HloInstruction*> lhs_operands;
  std::vector<HloInstruction*> rhs_operands;
  std::vector<HloInstruction*> add_users;
  std::vector<HloInstruction*> bias_operands;
  lhs_operands.reserve(group_size);
  rhs_operands.reserve(group_size);
  add_users.reserve(group_size);
  bias_operands.reserve(group_size);
  for (HloInstruction* dot : dots) {
    lhs_operands.push_back(dot->mutable_operand(0));
    rhs_operands.push_back(dot->mutable_operand(1));
    HloInstruction* add_user = nullptr;
    HloInstruction* bias = nullptr;
    if (fuse_biasadd &&
        GetMusaSameShapeDotBroadcastBiasAdd(dot, &add_user, &bias)) {
      add_users.push_back(add_user);
      bias_operands.push_back(bias);
    }
  }
  const bool can_fuse_biasadd =
      !pointer_array_output && fuse_biasadd && add_users.size() == dots.size() &&
      bias_operands.size() == dots.size();

  if (pointer_array_output) {
    std::vector<HloInstruction*> operands;
    std::vector<Shape> output_shapes;
    operands.reserve(group_size * 2);
    output_shapes.reserve(group_size);
    for (HloInstruction* dot : dots) {
      operands.push_back(dot->mutable_operand(0));
      operands.push_back(dot->mutable_operand(1));
      output_shapes.push_back(dot->shape());
    }
    Shape tuple_shape = ShapeUtil::MakeTupleShape(output_shapes);
    HloInstruction* custom_call = computation->AddInstruction(
        HloInstruction::CreateCustomCall(
            tuple_shape, operands, kMusaPointerArrayGemmCustomCallTarget,
            absl::StrCat("gemm_count=", group_size),
            API_VERSION_STATUS_RETURNING));
    custom_call->set_metadata(first->metadata());
    for (int64_t i = 0; i < group_size; ++i) {
      HloInstruction* old_dot = dots[i];
      HloInstruction* output = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(old_dot->shape(), custom_call,
                                                i));
      output->set_metadata(old_dot->metadata());
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(old_dot, output));
    }
    return OkStatus();
  }

  Shape lhs_concat_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {group_size * m, k}, {1, 0});
  HloInstruction* lhs_concat = computation->AddInstruction(
      HloInstruction::CreateConcatenate(lhs_concat_shape, lhs_operands, 0));
  Shape lhs_batch_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {group_size, m, k}, {2, 1, 0});
  HloInstruction* lhs_batch = computation->AddInstruction(
      HloInstruction::CreateReshape(lhs_batch_shape, lhs_concat));

  Shape rhs_concat_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {group_size * k, n}, {1, 0});
  HloInstruction* rhs_concat = computation->AddInstruction(
      HloInstruction::CreateConcatenate(rhs_concat_shape, rhs_operands, 0));
  Shape rhs_batch_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {group_size, k, n}, {2, 1, 0});
  HloInstruction* rhs_batch = computation->AddInstruction(
      HloInstruction::CreateReshape(rhs_batch_shape, rhs_concat));

  DotDimensionNumbers batch_dnums;
  batch_dnums.add_lhs_batch_dimensions(0);
  batch_dnums.add_rhs_batch_dimensions(0);
  batch_dnums.add_lhs_contracting_dimensions(2);
  batch_dnums.add_rhs_contracting_dimensions(1);
  Shape batch_dot_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {group_size, m, n}, {2, 1, 0});
  HloInstruction* batch_dot = computation->AddInstruction(
      HloInstruction::CreateDot(batch_dot_shape, lhs_batch, rhs_batch,
                                batch_dnums, first->precision_config()));
  batch_dot->set_metadata(first->metadata());

  HloInstruction* batch_result = batch_dot;
  if (can_fuse_biasadd) {
    Shape bias_concat_shape =
        ShapeUtil::MakeShapeWithDenseLayout(F32, {group_size * n}, {0});
    HloInstruction* bias_concat = computation->AddInstruction(
        HloInstruction::CreateConcatenate(bias_concat_shape, bias_operands, 0));
    Shape bias_batch_shape =
        ShapeUtil::MakeShapeWithDenseLayout(F32, {group_size, n}, {1, 0});
    HloInstruction* bias_batch = computation->AddInstruction(
        HloInstruction::CreateReshape(bias_batch_shape, bias_concat));
    Shape bias_broadcast_shape = ShapeUtil::MakeShapeWithDenseLayout(
        F32, {group_size, m, n}, {2, 1, 0});
    HloInstruction* bias_broadcast = computation->AddInstruction(
        HloInstruction::CreateBroadcast(bias_broadcast_shape, bias_batch,
                                        {0, 2}));
    batch_result = computation->AddInstruction(HloInstruction::CreateBinary(
        batch_dot_shape, HloOpcode::kAdd, batch_dot, bias_broadcast));
    batch_result->set_metadata(add_users.front()->metadata());
    *biasadd_fused = true;
  }

  std::vector<MusaSameShapeDotAddTreeRewritePlan> add_tree_plans;
  absl::flat_hash_set<HloInstruction*> add_tree_consumed_dots;
  if (rewrite_add_tree) {
    add_tree_plans = BuildMusaSameShapeDotAddTreeRewritePlans(
        dots, add_tree_rewrite_stats);
    HloComputation* reduce_computation =
        add_tree_plans.empty()
            ? nullptr
            : GetOrCreateMusaSameShapeAddReduceComputation(
                  computation->parent(), add_reduce_computation);
    for (const MusaSameShapeDotAddTreeRewritePlan& plan : add_tree_plans) {
      const int64_t start = plan.dot_indices.front();
      const int64_t count = plan.dot_indices.back() - start + 1;
      Shape slice_shape =
          ShapeUtil::MakeShapeWithDenseLayout(F32, {count, m, n}, {2, 1, 0});
      const std::vector<int64_t> start_indices = {start, 0, 0};
      const std::vector<int64_t> limit_indices = {start + count, m, n};
      const std::vector<int64_t> strides = {1, 1, 1};
      HloInstruction* slice = computation->AddInstruction(
          HloInstruction::CreateSlice(slice_shape, batch_result, start_indices,
                                      limit_indices, strides));
      HloInstruction* zero = computation->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
      Shape reduce_shape =
          ShapeUtil::MakeShapeWithDenseLayout(F32, {m, n}, {1, 0});
      HloInstruction* reduce = computation->AddInstruction(
          HloInstruction::CreateReduce(reduce_shape, slice, zero, {0},
                                       reduce_computation));
      reduce->set_metadata(plan.root->metadata());
      HloInstruction* rewritten_root = reduce;
      for (HloInstruction* broadcast : plan.broadcast_leaves) {
        rewritten_root = computation->AddInstruction(HloInstruction::CreateBinary(
            reduce_shape, HloOpcode::kAdd, rewritten_root, broadcast));
        rewritten_root->set_metadata(plan.root->metadata());
      }
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(plan.root,
                                                         rewritten_root));
      for (int64_t index : plan.dot_indices) {
        add_tree_consumed_dots.insert(dots[index]);
      }
      ++add_tree_rewrite_stats->rewritten_roots;
      add_tree_rewrite_stats->rewritten_dots += plan.dot_indices.size();
      add_tree_rewrite_stats->rewritten_adds += plan.add_nodes;
      add_tree_rewrite_stats->estimated_slice_reduction +=
          plan.dot_indices.size() - 1;
      add_tree_rewrite_stats->estimated_reshape_reduction +=
          plan.dot_indices.size() - 1;
      ++add_tree_rewrite_stats->rewritten_chunks[absl::StrCat(
          "group=", group_size, " m=", m, " k=", k, " n=", n,
          " roots=", add_tree_plans.size(), " dots=", plan.dot_indices.size(),
          " adds=", plan.add_nodes)];
    }
  }

  for (int64_t i = 0; i < group_size; ++i) {
    HloInstruction* old_dot = dots[i];
    if (add_tree_consumed_dots.find(old_dot) != add_tree_consumed_dots.end()) {
      continue;
    }
    HloInstruction* old_output = can_fuse_biasadd ? add_users[i] : old_dot;
    Shape slice_shape =
        ShapeUtil::MakeShapeWithDenseLayout(F32, {1, m, n}, {2, 1, 0});
    const std::vector<int64_t> start_indices = {i, 0, 0};
    const std::vector<int64_t> limit_indices = {i + 1, m, n};
    const std::vector<int64_t> strides = {1, 1, 1};
    HloInstruction* slice = computation->AddInstruction(
        HloInstruction::CreateSlice(slice_shape, batch_result, start_indices,
                                    limit_indices, strides));
    HloInstruction* reshaped = computation->AddInstruction(
        HloInstruction::CreateReshape(old_output->shape(), slice));
    reshaped->set_metadata(old_output->metadata());
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(old_output, reshaped));
    if (lane_refs != nullptr) {
      (*lane_refs)[reshaped] = MusaSameShapeDotBatchLaneRef{
          batch_result, i, m, n};
    }
  }
  return OkStatus();
}

Status BatchSameShapeSmallKDotChunkAsLoopFusion(
    HloComputation* computation, absl::Span<HloInstruction* const> dots,
    HloComputation** add_reduce_computation) {
  CHECK_GE(dots.size(), 2);
  HloComputation* reduce_computation =
      GetOrCreateMusaSameShapeAddReduceComputation(computation->parent(),
                                                   add_reduce_computation);
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  for (HloInstruction* dot : dots) {
    const int64_t m = dot->operand(0)->shape().dimensions(0);
    const int64_t k = dot->operand(0)->shape().dimensions(1);
    const int64_t n = dot->operand(1)->shape().dimensions(1);
    Shape product_shape =
        ShapeUtil::MakeShapeWithDenseLayout(F32, {m, n, k}, {2, 1, 0});
    HloInstruction* lhs_broadcast = computation->AddInstruction(
        HloInstruction::CreateBroadcast(product_shape, dot->mutable_operand(0),
                                        {0, 2}));
    lhs_broadcast->set_metadata(dot->metadata());
    HloInstruction* rhs_broadcast = computation->AddInstruction(
        HloInstruction::CreateBroadcast(product_shape, dot->mutable_operand(1),
                                        {2, 1}));
    rhs_broadcast->set_metadata(dot->metadata());
    HloInstruction* product = computation->AddInstruction(
        HloInstruction::CreateBinary(product_shape, HloOpcode::kMultiply,
                                     lhs_broadcast, rhs_broadcast));
    product->set_metadata(dot->metadata());
    HloInstruction* reduce = computation->AddInstruction(
        HloInstruction::CreateReduce(dot->shape(), product, zero, {2},
                                     reduce_computation));
    reduce->set_metadata(dot->metadata());
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(dot, reduce));
  }
  return OkStatus();
}

Status BatchSameShapeSmallKDotChunkAsCustomKernel(
    HloComputation* computation, absl::Span<HloInstruction* const> dots) {
  CHECK_GE(dots.size(), 2);
  HloInstruction* first = dots.front();
  const int64_t group_size = dots.size();
  const int64_t m = first->operand(0)->shape().dimensions(0);
  const int64_t k = first->operand(0)->shape().dimensions(1);
  const int64_t n = first->operand(1)->shape().dimensions(1);

  std::vector<HloInstruction*> operands;
  std::vector<Shape> output_shapes;
  operands.reserve(group_size * 2);
  output_shapes.reserve(group_size);
  for (HloInstruction* dot : dots) {
    operands.push_back(dot->mutable_operand(0));
    operands.push_back(dot->mutable_operand(1));
    output_shapes.push_back(dot->shape());
  }

  Shape tuple_shape = ShapeUtil::MakeTupleShape(output_shapes);
  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(
          tuple_shape, operands, kMusaSmallKDotCustomCallTarget,
          absl::StrCat("gemm_count=", group_size, ";m=", m, ";n=", n,
                       ";k=", k),
          API_VERSION_STATUS_RETURNING));
  custom_call->set_metadata(first->metadata());
  for (int64_t i = 0; i < group_size; ++i) {
    HloInstruction* old_dot = dots[i];
    HloInstruction* output = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(old_dot->shape(), custom_call,
                                              i));
    output->set_metadata(old_dot->metadata());
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(old_dot, output));
  }
  return OkStatus();
}

bool IsReachableFromAnySameShapeDot(HloInstruction* dot,
                                    absl::Span<HloInstruction* const> selected,
                                    const HloReachabilityMap& reachability) {
  if (!reachability.IsPresent(dot)) {
    return true;
  }
  for (HloInstruction* selected_dot : selected) {
    if (!reachability.IsPresent(selected_dot)) {
      return true;
    }
    if (reachability.IsReachable(dot, selected_dot) ||
        reachability.IsReachable(selected_dot, dot)) {
      return true;
    }
  }
  return false;
}

class MusaSameShapeDotBatcher : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "musa-same-shape-dot-batcher";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override {
    const bool enabled =
        EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCHER");
    const bool log =
        EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG");
    const bool diagnostic_only =
        EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY");
    if (log) {
      LOG(INFO) << "[MUSA_SAME_SHAPE_DOT_BATCHER_STAGE] module="
                << module->name() << " stage=run enabled=" << enabled
                << " diagnostic_only=" << diagnostic_only;
    }
    if (!enabled) {
      return false;
    }
    const int64_t min_group_size =
        ReadInt64Env("MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE", 8);
    const int64_t max_group_size =
        ReadInt64Env("MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE", 16);
    const int64_t max_groups =
        ReadInt64Env("MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS", 128);
    const int64_t min_candidate_dots = ReadInt64Env(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS", 512);
    const int64_t max_slice_bytes_per_saved_launch = ReadInt64Env(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH",
        2 * 1024 * 1024);
    const int64_t max_output_cols =
        ReadInt64Env("MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS", 0);
    const bool post_dot_diag =
        EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCH_POST_DOT_DIAG");
    const bool add_tree_diag =
        EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_DIAG");
    const bool rewrite_add_tree =
        EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_REWRITE");
    const bool add_tree_external_diag = EnvExplicitlyTrue(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_EXTERNAL_DIAG");
    const bool add_tree_mixed_key_diag = EnvExplicitlyTrue(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_DIAG");
    const int64_t add_tree_max_depth = std::clamp<int64_t>(
        ReadInt64Env("MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MAX_DEPTH", 64),
        1, 4096);
    const bool rewrite_mixed_key_add_tree = EnvExplicitlyTrue(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_REWRITE");
    const bool fuse_biasadd =
        EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCH_BIASADD");
    const bool pointer_array_output = EnvExplicitlyTrue(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POINTER_ARRAY_OUTPUT");
    const bool small_k_diag =
        EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_DIAG");
    const int64_t small_k_max_k =
        ReadInt64Env("MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MAX_K", 8);
    const int64_t small_k_min_group_size = ReadInt64Env(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MIN_GROUP_SIZE", 16);
    const std::vector<int64_t> small_k_output_cols = ReadPositiveInt64CsvEnv(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_OUTPUT_COLS",
        {160, 192, 256});
    const bool small_k_pointer_array_output = EnvExplicitlyTrue(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_POINTER_ARRAY_OUTPUT");
    const bool small_k_loop_fusion = EnvExplicitlyTrue(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_LOOP_FUSION");
    const bool small_k_custom_kernel = EnvExplicitlyTrue(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_KERNEL");
    const int64_t small_k_custom_max_group_size = ReadInt64Env(
        "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_MAX_GROUP_SIZE", 0);
    if (max_group_size < min_group_size) {
      return false;
    }

    bool changed = false;
    int64_t total_dots = 0;
    int64_t supported_dots = 0;
    int64_t raw_same_shape_groups = 0;
    int64_t raw_same_shape_dots = 0;
    MusaSameShapeDotFilterStats filter_stats;
    int64_t groups_rewritten = 0;
    int64_t dots_batched = 0;
    int64_t diagnostic_chunks = 0;
    int64_t diagnostic_dots = 0;
    int64_t diagnostic_launch_reduction = 0;
    int64_t diagnostic_concat_ops = 0;
    int64_t diagnostic_slice_ops = 0;
    int64_t diagnostic_reshape_ops = 0;
    int64_t diagnostic_concat_bytes = 0;
    int64_t diagnostic_slice_bytes = 0;
    int64_t dependency_filtered_dots = 0;
    int64_t dependency_filtered_chunks = 0;
    int64_t estimated_gemm_launch_reduction = 0;
    int64_t estimated_concat_ops = 0;
    int64_t estimated_slice_ops = 0;
    int64_t estimated_reshape_ops = 0;
    int64_t estimated_concat_bytes = 0;
    int64_t estimated_slice_bytes = 0;
    int64_t pointer_array_output_chunks = 0;
    int64_t pointer_array_output_dots = 0;
    int64_t small_k_candidate_chunks = 0;
    int64_t small_k_candidate_dots = 0;
    int64_t small_k_candidate_launch_reduction = 0;
    int64_t small_k_candidate_slice_bytes = 0;
    int64_t small_k_pointer_array_chunks = 0;
    int64_t small_k_pointer_array_dots = 0;
    int64_t small_k_loop_fusion_chunks = 0;
    int64_t small_k_loop_fusion_dots = 0;
    int64_t small_k_custom_kernel_chunks = 0;
    int64_t small_k_custom_kernel_dots = 0;
    int64_t cost_filtered_dots = 0;
    int64_t cost_filtered_chunks = 0;
    int64_t output_cols_filtered_dots = 0;
    int64_t output_cols_filtered_chunks = 0;
    absl::flat_hash_map<std::string, int64_t> batched_chunk_groups;
    absl::flat_hash_map<std::string, int64_t> diagnostic_chunk_groups;
    absl::flat_hash_map<std::string, int64_t> small_k_candidate_chunk_groups;
    absl::flat_hash_map<std::string, int64_t>
        small_k_pointer_array_chunk_groups;
    absl::flat_hash_map<std::string, int64_t> small_k_loop_fusion_chunk_groups;
    absl::flat_hash_map<std::string, int64_t>
        small_k_custom_kernel_chunk_groups;
    absl::flat_hash_map<std::string, int64_t> cost_filtered_chunk_groups;
    int64_t post_dot_diag_chunks = 0;
    int64_t post_dot_diag_dots = 0;
    int64_t post_dot_diag_uniform_chunks = 0;
    absl::flat_hash_map<std::string, int64_t> post_dot_user_patterns;
    absl::flat_hash_map<std::string, int64_t> post_dot_chunk_patterns;
    int64_t add_tree_diag_chunks = 0;
    int64_t add_tree_diag_dots = 0;
    int64_t add_tree_candidate_roots = 0;
    int64_t add_tree_candidate_dots = 0;
    int64_t add_tree_candidate_adds = 0;
    int64_t add_tree_estimated_slice_reduction = 0;
    int64_t add_tree_estimated_reshape_reduction = 0;
    int64_t add_tree_estimated_slice_bytes_reduction = 0;
    int64_t add_tree_external_dot_leaves = 0;
    int64_t add_tree_external_supported_dots = 0;
    int64_t add_tree_external_same_key_dots = 0;
    int64_t add_tree_external_other_key_dots = 0;
    int64_t add_tree_external_unsupported_dots = 0;
    int64_t add_tree_external_leaves = 0;
    int64_t add_tree_blocked_nodes = 0;
    int64_t add_tree_mixed_key_roots = 0;
    int64_t add_tree_mixed_key_rewritable_roots = 0;
    int64_t add_tree_mixed_key_dots = 0;
    int64_t add_tree_mixed_key_external_dots = 0;
    absl::flat_hash_map<std::string, int64_t> add_tree_chunk_patterns;
    absl::flat_hash_map<std::string, int64_t> add_tree_root_patterns;
    absl::flat_hash_map<std::string, int64_t>
        add_tree_mixed_key_root_patterns;
    absl::flat_hash_map<std::string, int64_t> add_tree_blocked_reasons;
    absl::flat_hash_map<std::string, int64_t> add_tree_external_dot_reasons;
    absl::flat_hash_map<std::string, int64_t> add_tree_external_dot_keys;
    absl::flat_hash_map<std::string, int64_t> add_tree_external_leaf_ops;
    MusaSameShapeDotAddTreeRewriteStats add_tree_rewrite_stats;
    MusaSameShapeDotMixedKeyAddTreeRewriteStats
        mixed_key_add_tree_rewrite_stats;
    int64_t biasadd_fused_chunks = 0;
    int64_t biasadd_fused_dots = 0;
    absl::flat_hash_map<std::string, int64_t> biasadd_fused_chunk_groups;
    HloComputation* add_tree_reduce_computation = nullptr;
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      absl::flat_hash_map<HloInstruction*, MusaSameShapeDotBatchLaneRef>
          batch_lane_refs;
      absl::flat_hash_map<std::string, std::vector<HloInstruction*>> groups;
      for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
        if (instr->opcode() == HloOpcode::kDot) {
          ++total_dots;
        }
        if (IsSupportedMusaSameShapeDot(instr, &filter_stats)) {
          ++supported_dots;
          groups[MusaSameShapeDotKey(instr)].push_back(instr);
        }
      }

      std::vector<std::vector<HloInstruction*>> candidates;
      candidates.reserve(groups.size());
      int64_t candidate_dot_count = 0;
      for (auto& item : groups) {
        if (item.second.size() >= min_group_size) {
          ++raw_same_shape_groups;
          raw_same_shape_dots += item.second.size();
          candidate_dot_count += item.second.size();
          candidates.push_back(std::move(item.second));
        }
      }
      if (candidate_dot_count < min_candidate_dots) {
        continue;
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) { return a.size() > b.size(); });
      std::unique_ptr<HloReachabilityMap> reachability =
          HloReachabilityMap::Build(computation);

      for (std::vector<HloInstruction*>& group : candidates) {
        HloInstruction* group_first = group.front();
        const int64_t group_k = group_first->operand(0)->shape().dimensions(1);
        const int64_t group_n = group_first->operand(1)->shape().dimensions(1);
        const bool group_is_small_k_custom_candidate =
            small_k_custom_kernel && group.size() >= small_k_min_group_size &&
            group_k <= small_k_max_k &&
            (small_k_output_cols.empty() ||
             ContainsInt64(small_k_output_cols, group_n));
        const int64_t small_k_chunk_group_cap =
            group_is_small_k_custom_candidate &&
                    small_k_custom_max_group_size > 0
                ? std::min(max_group_size, small_k_custom_max_group_size)
                : max_group_size;
        absl::flat_hash_set<HloInstruction*> consumed;
        while (consumed.size() < group.size()) {
          const int64_t completed_chunks =
              diagnostic_only ? diagnostic_chunks : groups_rewritten;
          if (completed_chunks >= max_groups) {
            break;
          }
          std::vector<HloInstruction*> chunk;
          chunk.reserve(small_k_chunk_group_cap);
          bool skipped_for_dependency = false;
          for (HloInstruction* dot : group) {
            if (consumed.find(dot) != consumed.end()) {
              continue;
            }
            if (IsReachableFromAnySameShapeDot(
                    dot,
                    absl::Span<HloInstruction* const>(chunk.data(),
                                                      chunk.size()),
                    *reachability)) {
              skipped_for_dependency = true;
              continue;
            }
            chunk.push_back(dot);
            consumed.insert(dot);
            if (chunk.size() >= small_k_chunk_group_cap) {
              break;
            }
          }
          if (chunk.size() < min_group_size) {
            for (HloInstruction* dot : chunk) {
              consumed.erase(dot);
            }
            for (HloInstruction* dot : group) {
              consumed.insert(dot);
            }
            dependency_filtered_dots += chunk.size();
            if (skipped_for_dependency) {
              ++dependency_filtered_chunks;
            }
            break;
          }
          HloInstruction* first = chunk.front();
          const int64_t group_size = chunk.size();
          const int64_t m = first->operand(0)->shape().dimensions(0);
          const int64_t k = first->operand(0)->shape().dimensions(1);
          const int64_t n = first->operand(1)->shape().dimensions(1);
          const int64_t saved_launches = group_size - 1;
          const int64_t chunk_slice_bytes = group_size * m * n * 4;
          const int64_t slice_bytes_per_saved_launch =
              saved_launches <= 0 ? chunk_slice_bytes
                                  : chunk_slice_bytes / saved_launches;
          if (max_output_cols > 0 && n > max_output_cols) {
            ++output_cols_filtered_chunks;
            output_cols_filtered_dots += group_size;
            ++cost_filtered_chunk_groups[absl::StrCat(
                "reason=output_cols group=", group_size, " m=", m, " k=", k,
                " n=", n)];
            continue;
          }
          if (slice_bytes_per_saved_launch >
              max_slice_bytes_per_saved_launch) {
            ++cost_filtered_chunks;
            cost_filtered_dots += group_size;
            ++cost_filtered_chunk_groups[absl::StrCat(
                "reason=slice_bytes group=", group_size, " m=", m, " k=", k, " n=", n,
                " slice_per_saved=", slice_bytes_per_saved_launch)];
            continue;
          }
          const bool is_small_k_chunk =
              group_size >= small_k_min_group_size && k <= small_k_max_k &&
              (small_k_output_cols.empty() ||
               ContainsInt64(small_k_output_cols, n));
          const bool chunk_custom_kernel =
              small_k_custom_kernel && is_small_k_chunk;
          const bool chunk_loop_fusion =
              !chunk_custom_kernel && small_k_loop_fusion && is_small_k_chunk;
          if (small_k_diag && is_small_k_chunk) {
            ++small_k_candidate_chunks;
            small_k_candidate_dots += group_size;
            small_k_candidate_launch_reduction += saved_launches;
            small_k_candidate_slice_bytes += chunk_slice_bytes;
            ++small_k_candidate_chunk_groups[absl::StrCat(
                "group=", group_size, " m=", m, " k=", k, " n=", n)];
          }
          const bool chunk_pointer_array_output =
              !chunk_custom_kernel && !chunk_loop_fusion &&
              (pointer_array_output ||
               (small_k_pointer_array_output && is_small_k_chunk));
          if (post_dot_diag) {
            ++post_dot_diag_chunks;
            post_dot_diag_dots += group_size;
            absl::flat_hash_map<std::string, int64_t> chunk_patterns;
            for (HloInstruction* dot : chunk) {
              std::string pattern = MusaSameShapeDotPostUserPattern(dot);
              ++post_dot_user_patterns[pattern];
              ++chunk_patterns[pattern];
            }
            if (chunk_patterns.size() == 1) {
              ++post_dot_diag_uniform_chunks;
            }
            ++post_dot_chunk_patterns[absl::StrJoin(
                TopMusaCountGroups(chunk_patterns, 4), ",")];
          }
          if (add_tree_diag || add_tree_external_diag ||
              add_tree_mixed_key_diag) {
            ++add_tree_diag_chunks;
            add_tree_diag_dots += group_size;
            MusaSameShapeDotAddTreeChunkDiag chunk_diag =
                DiagnoseMusaSameShapeDotAddTreeChunk(
                    absl::Span<HloInstruction* const>(chunk.data(),
                                                      chunk.size()),
                    add_tree_external_diag, add_tree_mixed_key_diag,
                    add_tree_max_depth);
            add_tree_candidate_roots += chunk_diag.candidate_roots;
            add_tree_candidate_dots += chunk_diag.candidate_dots;
            add_tree_candidate_adds += chunk_diag.candidate_adds;
            add_tree_estimated_slice_reduction +=
                chunk_diag.estimated_slice_reduction;
            add_tree_estimated_reshape_reduction +=
                chunk_diag.estimated_reshape_reduction;
            add_tree_estimated_slice_bytes_reduction +=
                chunk_diag.estimated_slice_bytes_reduction;
            add_tree_external_dot_leaves += chunk_diag.external_dot_leaves;
            add_tree_external_supported_dots +=
                chunk_diag.external_supported_dots;
            add_tree_external_same_key_dots +=
                chunk_diag.external_same_key_dots;
            add_tree_external_other_key_dots +=
                chunk_diag.external_other_key_dots;
            add_tree_external_unsupported_dots +=
                chunk_diag.external_unsupported_dots;
            add_tree_external_leaves += chunk_diag.external_leaves;
            add_tree_blocked_nodes += chunk_diag.blocked_nodes;
            add_tree_mixed_key_roots += chunk_diag.mixed_key_roots;
            add_tree_mixed_key_rewritable_roots +=
                chunk_diag.mixed_key_rewritable_roots;
            add_tree_mixed_key_dots += chunk_diag.mixed_key_dots;
            add_tree_mixed_key_external_dots +=
                chunk_diag.mixed_key_external_dots;
            ++add_tree_chunk_patterns[absl::StrCat(
                "group=", group_size, " roots=", chunk_diag.roots,
                " candidate_roots=", chunk_diag.candidate_roots,
                " candidate_dots=", chunk_diag.candidate_dots,
                " candidate_adds=", chunk_diag.candidate_adds,
                " slice_reduction=",
                chunk_diag.estimated_slice_reduction,
                " ext_dot=", chunk_diag.external_dot_leaves,
                " ext=", chunk_diag.external_leaves,
                " blocked=", chunk_diag.blocked_nodes)];
            for (const auto& [pattern, count] : chunk_diag.root_patterns) {
              add_tree_root_patterns[pattern] += count;
            }
            for (const auto& [pattern, count] :
                 chunk_diag.mixed_key_root_patterns) {
              add_tree_mixed_key_root_patterns[pattern] += count;
            }
            for (const auto& [reason, count] : chunk_diag.blocked_reasons) {
              add_tree_blocked_reasons[reason] += count;
            }
            for (const auto& [reason, count] :
                 chunk_diag.external_dot_reasons) {
              add_tree_external_dot_reasons[reason] += count;
            }
            for (const auto& [key, count] : chunk_diag.external_dot_keys) {
              add_tree_external_dot_keys[key] += count;
            }
            for (const auto& [op, count] : chunk_diag.external_leaf_ops) {
              add_tree_external_leaf_ops[op] += count;
            }
          }
          if (diagnostic_only) {
            ++diagnostic_chunks;
            diagnostic_dots += group_size;
            diagnostic_launch_reduction += saved_launches;
            if (!chunk_custom_kernel && !chunk_loop_fusion &&
                !chunk_pointer_array_output) {
              diagnostic_concat_ops += 2;
              diagnostic_slice_ops += group_size;
              diagnostic_reshape_ops += 2 + group_size;
              diagnostic_concat_bytes += group_size * (m * k + k * n) * 4;
              diagnostic_slice_bytes += chunk_slice_bytes;
            }
            ++diagnostic_chunk_groups[absl::StrCat(
                "group=", group_size, " m=", m, " k=", k, " n=", n,
                " slice_per_saved=", slice_bytes_per_saved_launch)];
            continue;
          }
          bool biasadd_fused = false;
          if (chunk_custom_kernel) {
            TF_RETURN_IF_ERROR(BatchSameShapeSmallKDotChunkAsCustomKernel(
                computation,
                absl::Span<HloInstruction* const>(chunk.data(), chunk.size())));
          } else if (chunk_loop_fusion) {
            TF_RETURN_IF_ERROR(BatchSameShapeSmallKDotChunkAsLoopFusion(
                computation,
                absl::Span<HloInstruction* const>(chunk.data(), chunk.size()),
                &add_tree_reduce_computation));
          } else {
            TF_RETURN_IF_ERROR(BatchSameShapeDotChunk(
                computation,
                absl::Span<HloInstruction* const>(chunk.data(), chunk.size()),
                chunk_pointer_array_output, fuse_biasadd,
                rewrite_add_tree && !chunk_pointer_array_output,
                &add_tree_reduce_computation,
                rewrite_mixed_key_add_tree && !chunk_pointer_array_output
                    ? &batch_lane_refs
                    : nullptr,
                &biasadd_fused, &add_tree_rewrite_stats));
          }
          if (biasadd_fused) {
            ++biasadd_fused_chunks;
            biasadd_fused_dots += group_size;
            ++biasadd_fused_chunk_groups[absl::StrCat(
                "group=", group_size, " m=", m, " k=", k, " n=", n)];
          }
          estimated_gemm_launch_reduction += saved_launches;
          if (chunk_custom_kernel) {
            ++small_k_custom_kernel_chunks;
            small_k_custom_kernel_dots += group_size;
            ++small_k_custom_kernel_chunk_groups[absl::StrCat(
                "group=", group_size, " m=", m, " k=", k, " n=", n)];
          } else if (chunk_loop_fusion) {
            ++small_k_loop_fusion_chunks;
            small_k_loop_fusion_dots += group_size;
            ++small_k_loop_fusion_chunk_groups[absl::StrCat(
                "group=", group_size, " m=", m, " k=", k, " n=", n)];
          } else if (chunk_pointer_array_output) {
            ++pointer_array_output_chunks;
            pointer_array_output_dots += group_size;
            if (small_k_pointer_array_output && is_small_k_chunk) {
              ++small_k_pointer_array_chunks;
              small_k_pointer_array_dots += group_size;
              ++small_k_pointer_array_chunk_groups[absl::StrCat(
                  "group=", group_size, " m=", m, " k=", k, " n=", n)];
            }
          } else {
            estimated_concat_ops += 2;
            estimated_slice_ops += group_size;
            estimated_reshape_ops += 2 + group_size;
            estimated_concat_bytes += group_size * (m * k + k * n) * 4;
            estimated_slice_bytes += chunk_slice_bytes;
          }
          ++batched_chunk_groups[absl::StrCat("group=", group_size, " m=", m,
                                              " k=", k, " n=", n)];
          changed = true;
          ++groups_rewritten;
          dots_batched += chunk.size();
        }
        const int64_t completed_chunks =
            diagnostic_only ? diagnostic_chunks : groups_rewritten;
        if (completed_chunks >= max_groups) {
          break;
        }
      }
      if (rewrite_mixed_key_add_tree && !batch_lane_refs.empty()) {
        TF_RETURN_IF_ERROR(RewriteMusaSameShapeDotMixedKeyAddTrees(
            computation, batch_lane_refs, &add_tree_reduce_computation,
            &mixed_key_add_tree_rewrite_stats));
      }
    }
    return changed;
  }
};
}  // namespace

// Runs optimization passes on the given HLO module.
Status GpuCompiler::OptimizeHloModule(HloModule* hlo_module,
                                      se::StreamExecutor* stream_exec,
                                      const CompileOptions& options,
                                      const GpuTargetConfig& gpu_target_config,
                                      const AutotuneResults* autotune_results) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  // By default use an externally provided thread pool.
  tsl::thread::ThreadPool* thread_pool = options.thread_pool;
  std::optional<tsl::thread::ThreadPool> overriding_thread_pool;
  int num_threads = hlo_module->config()
                        .debug_options()
                        .xla_gpu_force_compilation_parallelism();
  // If an external thread pool is provided or single-threaded operation is
  // requested do not create a thread pool.
  if (thread_pool == nullptr && num_threads != 1) {
    // Zero means "default", treat it as "max parallelism" here.
    if (num_threads == 0) {
      num_threads = tsl::port::MaxParallelism();
    }
    overriding_thread_pool.emplace(tsl::Env::Default(), "", num_threads);
    thread_pool = &*overriding_thread_pool;
  }

  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts({},
                                                             ConvIsLowerable);

  // GPU only supports canonical convolutions.
  layout_insensitive_algsimp_opts.set_supports_non_canonical_dots(false);

  // "slow" minmax means we propagate nan.
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(
      !debug_options.xla_gpu_enable_fast_min_max());

  // Always simplify reduce(transpose(x)) and reduce(reshape(x)), even when
  // the transpose/reshape has multiple users.  This helps int8 models, which
  // tend to have lots of transpose+reshape's (converting between NCHW and
  // NCHW_VECT_C).  Without this, those reshape+transposes can get materialized
  // out, which is really bad for perf.
  layout_insensitive_algsimp_opts
      .set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);

  if (gpu_target_config.platform_name == "ROCM") {
    layout_insensitive_algsimp_opts.set_enable_conv_operand_swap(false);
  }
  layout_insensitive_algsimp_opts
      .set_enable_unconditional_reduce_of_concat_replacement(false);

  SetInstructionMetadata(hlo_module);

  HloPassPipeline pre_spmd_pipeline("pre-spmd-partitioner");
  // Run some IR cleanup passes before running the SPMD partitioning
  // passes.
  pre_spmd_pipeline.AddPass<CallInliner>();
  pre_spmd_pipeline.AddPass<ZeroSizedHloElimination>();
  pre_spmd_pipeline.AddPass<ConditionalCanonicalizer>();

  pre_spmd_pipeline.AddPass<TopkDecomposer>([&](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kTopK;
  });

  // The SPMD partitioner would mess up the sort+slice structure, so we need to
  // rewrite Topk before that happens.
  pre_spmd_pipeline.AddPass<TopkRewriter>(
      [](const HloSortInstruction*, int64_t) { return true; });

  TF_RETURN_IF_ERROR(pre_spmd_pipeline.Run(hlo_module).status());

  const int64_t num_partitions = hlo_module->config().num_partitions();
  bool auto_sharding = hlo_module->config().use_auto_spmd_partitioning();

#ifndef PLATFORM_GOOGLE
  if (auto_sharding) {
    LOG(ERROR) << "GPU autosharding is not yet available in open source.";
  }
#endif

  if (num_partitions > 1) {
    if (!hlo_module->config().use_spmd_partitioning()) {
      return InvalidArgument(
          "num_partitions=%d but SPMD partitioning not enabled.",
          num_partitions);
    }
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    HloPassPipeline& spmd_simplify =
        spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

    spmd_simplify.AddPass<AlgebraicSimplifier>(layout_insensitive_algsimp_opts);

    spmd_simplify.AddPass<SortSimplifier>();
    spmd_simplify.AddPass<TupleSimplifier>();
    spmd_simplify.AddPass<ScatterSimplifier>();
    spmd_simplify.AddPass<ScatterExpander>(
        ScatterExpander::kEliminateSimpleScatters);
    spmd_simplify.AddPass<GatherSimplifier>();
    spmd_simplify.AddPass<GatherExpander>(
        GatherExpander::kEliminateSimpleGathers);
    spmd_simplify.AddPass<WhileLoopConstantSinking>();
    spmd_simplify.AddPass<WhileLoopSimplifier>();

    ReshapeMoverOptions reshape_mover_options;
    reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
    spmd_simplify.AddPass<ReshapeMover>(reshape_mover_options);
    spmd_simplify.AddPass<HloConstantFolding>();
    spmd_simplify.AddPass<ConditionalSimplifier>();

    spmd_pipeline.AddPass<HloConstantSplitter>();
    spmd_simplify.AddPass<HloDCE>();

#ifdef PLATFORM_GOOGLE
    if (auto_sharding) {
      AutoShardingOption option;
      option.enable = true;
      if (!hlo_module->config().auto_spmd_partitioning_mesh_shape().empty()) {
        option.device_mesh_shape =
            hlo_module->config().auto_spmd_partitioning_mesh_shape();
      } else {
        // Use a simple mesh shape if not specified.
        option.device_mesh_shape = {
            gpu_target_config.gpu_device_info.core_count(), 1};
      }
      if (!hlo_module->config().auto_spmd_partitioning_mesh_ids().empty()) {
        option.device_mesh_ids =
            hlo_module->config().auto_spmd_partitioning_mesh_ids();
      }
      option.memory_budget_per_device =
          hlo_module->config()
              .debug_options()
              .xla_gpu_auto_spmd_partitioning_memory_budget_gb() *
          1024 * 1024 * 1024;
      option.memory_budget_ratio =
          hlo_module->config()
              .debug_options()
              .xla_gpu_auto_spmd_partitioning_memory_budget_ratio();
      spmd_pipeline.AddPass<AutoSharding>(option);
    }
#endif  // PLATFORM_GOOGLE

    spmd_pipeline.AddPass<ShardingPropagation>(
        /*is_spmd=*/true, /*propagate_metadata=*/false,
        hlo_module->config().allow_spmd_sharding_propagation_to_output());
    spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
        num_partitions, hlo_module->config().replica_count());
    spmd_pipeline.AddPass<CollectivePermuteMotion>();
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  } else {
    HloPassPipeline sharding_removal_pipeline("sharding-removal");
    // Remove redundant sharding ops when partition_count == 1.
    sharding_removal_pipeline.AddPass<ShardingRemover>();
    sharding_removal_pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(sharding_removal_pipeline.Run(hlo_module).status());
  }

  {
    HloPassPipeline pipeline("optimization");
    AddHloVerifier(&pipeline);
    pipeline.AddPass<TopKSplitter>();
    pipeline.AddPass<TopkSpecializer>();
    pipeline.AddPass<TopkDecomposer>();

    HloPredicate upcaster_filter = [&](const HloInstruction* instr) {
      const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
          &gpu_target_config.gpu_device_info.gpu_compute_capability());
      if (cuda_cc != nullptr &&
          !cuda_cc->IsAtLeast(se::CudaComputeCapability::VOLTA)) {
        return true;
      }
      return !gpu::IsMatrixMultiplication(*instr);
    };

    pipeline.AddPass<OperandUpcaster>(upcaster_filter);
    pipeline.AddPass<ResultCaster>(upcaster_filter);

    // Expand random number generation.
    pipeline.AddPass<RngExpander>();
    pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);

    // Comparison total order expander
    pipeline.AddPass<ComparisonExpander>();

    // Remove zero-sized HLO from the input so that other passes don't have to
    // handle it.
    pipeline.AddPass<ZeroSizedHloElimination>();

    if (debug_options.xla_gpu_deterministic_ops()) {
      // Scatter can be indeterministic if indices are not unique or a non
      // associative combiner function is used. Eliminate these Scatter ops.
      pipeline.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateIndeterminisitcScatters);
    }
    // Scatters unsupported on XLA:GPU are eliminated.
    pipeline.AddPass<GpuScatterExpander>();

    // TODO(phawkins): replace QR and Eigh decompositions with calls to
    // cuSOLVER.
    pipeline.AddPass<QrExpander>();
    pipeline.AddPass<EighExpander>();

    pipeline.AddPass<DynamicIndexSplitter>();

    // TODO(b/64094172): make Call work on GPU instead of inlining.
    pipeline.AddPass<CallInliner>();

    pipeline.AddPass<DotDimensionSorter>();
    pipeline.AddPass<DotDecomposer>();

    pipeline.AddPass<StochasticConvertDecomposer>();

    pipeline.AddPass<Convolution4DExpander>();

    // Replace PRED convolutions with F16.
    pipeline.AddPass<ConvolutionPredExpander>();

    // Expand the sort op to support stable sorting if required.
    pipeline.AddPass<StableSortExpander>();

    pipeline.AddPass<BatchNormExpander>(
        /*rewrite_training_op=*/true,
        /*rewrite_inference_op=*/true,
        /*rewrite_grad_op=*/true);

    pipeline.AddPass<LogisticExpander>();
    pipeline.AddPass<ConditionalCanonicalizer>();
    pipeline.AddPass<DynamicDimensionSimplifier>();

    DynamicPadderOptions dynamic_padder_options;

    switch (hlo_module->config().debug_options().xla_gpu_shape_checks()) {
      case DebugOptions::IGNORE:
        dynamic_padder_options.shape_check_mode =
            DynamicDimensionInference::ShapeCheckMode::kIgnore;
        break;
      case DebugOptions::RUNTIME: {
        dynamic_padder_options.shape_check_mode =
            DynamicDimensionInference::ShapeCheckMode::kRuntime;
        dynamic_padder_options.assertion_generator = [&](HloInstruction* inst) {
          auto created = Cast<HloCustomCallInstruction>(
              inst->parent()->AddInstruction(HloInstruction::CreateCustomCall(
                  ShapeUtil::MakeTokenShape(), {inst},
                  kXlaGpuAssertCustomCallTag,
                  "Buffers have different size at runtime",
                  API_VERSION_STATUS_RETURNING)));
          created->set_custom_call_has_side_effect(true);
        };
        break;
      }
      case DebugOptions::COMPILE_TIME:
        dynamic_padder_options.shape_check_mode =
            DynamicDimensionInference::ShapeCheckMode::kCompileTime;
        break;
      default:
        LOG(FATAL) << "Unreachable";
    }

    pipeline.AddPass<DynamicPadder>(dynamic_padder_options);

    // Build simplification pipeline.  The passes in here are run to a fixed
    // point.
    [&, &pipeline =
            pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
      AddHloVerifier(&pipeline, HloVerifierOpts{}, /*debug_only=*/true);

      // BatchNormExpander can create zero-sized ops, so zero-sized HLO
      // elimination has to come after that pass.
      pipeline.AddPass<ZeroSizedHloElimination>();

      pipeline.AddPass<GatherSimplifier>();
      pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
      pipeline.AddPass<ScatterSimplifier>();
      pipeline.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);
      pipeline.AddPass<ScatterSliceSimplifier>();
      pipeline.AddPass<AlgebraicSimplifier>(layout_insensitive_algsimp_opts);
      pipeline.AddPass<BitcastDtypesExpander>();
      // AlgebraicSimplifier may add contracting dimensions to a dot.
      pipeline.AddPass<DotDimensionSorter>();
      pipeline.AddPass<DotDecomposer>();
      const bool is_musa_compilation =
          IsMusaCompilation(stream_exec, platform_id_);
      // Only merge "smallish" dots.  This threshold was not set carefully, but
      // so far we know that 1mb is too small.
      const int64_t dot_merger_max_size =
          is_musa_compilation ? MusaDotMergerMaxSizeBytes(int64_t{16} << 20)
                              : (int64_t{16} << 20);
      const bool musa_has_many_small_dots =
          is_musa_compilation &&
          MusaModuleHasManySmallDots(hlo_module, dot_merger_max_size);
      if (musa_has_many_small_dots) {
        // DotMerger only merges dots that literally share an operand pointer.
        // A CSE pass before merging canonicalizes duplicate reshape/bitcast/
        // slice operands produced by earlier simplification, exposing more
        // same-operand dot groups without changing GEMM backend selection.
        pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
      }
      pipeline.AddPass<DotMerger>(/*max_size_to_merge=*/dot_merger_max_size);
      pipeline.AddPass<SortSimplifier>();
      pipeline.AddPass<TupleSimplifier>();
      pipeline.AddPass<WhileLoopConstantSinking>();
      pipeline.AddPass<WhileLoopSimplifier>();
      pipeline.AddPass<SliceSinker>();

      ReshapeMoverOptions reshape_mover_options;
      reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
      pipeline.AddPass<ReshapeMover>(reshape_mover_options);
      pipeline.AddPass<HloConstantFolding>();
      pipeline.AddPass<ConditionalSimplifier>();
      pipeline.AddPass<RealImagExpander>();
      pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot);
      const int64_t post_transpose_dot_merger_max_size =
          MusaPostTransposeDotMergerMaxSizeBytes(int64_t{64} << 20);
      if (is_musa_compilation &&
          (MusaPostTransposeDotMergerEnabled() ||
           (!MusaPostTransposeDotMergerExplicitlyDisabled() &&
            MusaModuleHasManySmallDots(hlo_module,
                                       post_transpose_dot_merger_max_size)))) {
        pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
        // MUSA meta_graph_2 has thousands of small GEMMs.  Some dots only
        // become mergeable after transpose folding, so run DotMerger once more
        // before CSE/DCE and before GemmRewriter lowers dots to custom calls.
        pipeline.AddPass<DotMerger>(
            /*max_size_to_merge=*/post_transpose_dot_merger_max_size);
      }
      pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
      pipeline.AddPass<HloDCE>();
    }();

    // ConvertMover and ReshapeMover fight with each other: ConvertMover wants
    // to move some converts down the graph, but ReshapeMover wants to move them
    // up the graph.  As a compromise, let ReshapeMover run to a fixed point,
    // and then run ConvertMover + algsimp to a fixed point.
    [&, &pipeline =
            pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification-2")] {
      pipeline.AddPass<ConvertMover>();
      pipeline.AddPass<AlgebraicSimplifier>(layout_insensitive_algsimp_opts);
    }();

    pipeline.AddPass<HloComputationDeduplicator>(
        /*mark_fusion_duplications=*/false);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  const bool enable_all_pipelined =
      debug_options.xla_gpu_enable_pipelined_collectives();

  // Optimize collectives generated by SPMD partitioning. Enable these passes
  // otherwise as well so that all collectives can get these optimizations.
  {
    HloPassPipeline collectives_pipeline("collective-optimizations");
    collectives_pipeline.AddPass<AllReduceFolder>();
    collectives_pipeline.AddPass<ReduceScatterCreator>();
    collectives_pipeline.AddPass<AllGatherOptimizer>();
    collectives_pipeline.AddPass<AllReduceReassociate>(
        debug_options.xla_gpu_enable_reassociation_for_converted_ar());
    collectives_pipeline.AddPass<ReduceScatterReassociate>();
    const DebugOptions& debug_options = hlo_module->config().debug_options();
    collectives_pipeline.AddPass<WhileLoopAllReduceCodeMotion>(
        /*enable_reduce_scatter=*/debug_options
            .xla_gpu_enable_while_loop_reduce_scatter_code_motion());

    if (enable_all_pipelined ||
        debug_options.xla_gpu_enable_pipelined_all_reduce()) {
      CollectivePipeliner::Config config{
          /*level_to_operate_on=*/0,
          /*max_pipelining_per_loop=*/INT64_MAX,
          /*last_run=*/true,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/true,
          /*pipelining_direction=*/
          CollectivePipeliner::PipeliningDirection::kForward,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>};
      collectives_pipeline.AddPass<CollectivePipeliner>(config);
    }
    if (enable_all_pipelined ||
        debug_options.xla_gpu_enable_pipelined_all_gather()) {
      CollectivePipeliner::Config config{
          /*level_to_operate_on=*/0,
          /*max_pipelining_per_loop=*/INT64_MAX,
          /*last_run=*/true,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/true,
          /*pipelining_direction=*/
          CollectivePipeliner::PipeliningDirection::kBackward,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kAllGather>};
      collectives_pipeline.AddPass<CollectivePipeliner>(config);
    }
    if (enable_all_pipelined ||
        debug_options.xla_gpu_enable_pipelined_reduce_scatter()) {
      CollectivePipeliner::Config config{
          /*level_to_operate_on=*/0,
          /*max_pipelining_per_loop=*/INT64_MAX,
          /*last_run=*/true,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/true,
          /*pipelining_direction=*/
          CollectivePipeliner::PipeliningDirection::kForward,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kReduceScatter>};
      collectives_pipeline.AddPass<CollectivePipeliner>(config);
    }

    // Run algebraic simplifier to reshape(broadcast) into a broadcast when
    // the reshape is just adding a unit dimension. This will help with the
    // AllGatherBroadcastReorder pass.
    collectives_pipeline.AddPass<AlgebraicSimplifier>(
        layout_insensitive_algsimp_opts);

    collectives_pipeline.AddPass<AllGatherBroadcastReorder>();

    // promote 16 bit integer all-reduce and reduce-scatter to 32-bit.
    const std::pair<PrimitiveType, PrimitiveType> ar_promoted_types[] = {
        {U16, U32}, {S16, S32}};
    collectives_pipeline.AddPass<AllReducePromotion>(ar_promoted_types);
    // Remove dead computations left over after ar/rs promotion.
    collectives_pipeline.AddPass<HloDCE>();

    // Run WhileLoopTripCountAnnotator after collective pipelining and before
    // layout assignment and fusion.This pass does some pattern-matching on
    // while bodies/conditions, and this is where the HLO is "nicest".
    //
    // It's important that we don't make semantic changes (e.g. unrolling) to
    // any `while` loops after this point, because otherwise the trip-count
    // annotations added by this pass may not be correct after the
    // modifications.
    collectives_pipeline.AddPass<WhileLoopTripCountAnnotator>();

    TF_RETURN_IF_ERROR(collectives_pipeline.Run(hlo_module).status());
  }

  // Run target-specific HLO optimization passes for convolution
  // canonicalization.
  se::GpuComputeCapability gpu_version =
      gpu_target_config.gpu_device_info.gpu_compute_capability();
  se::dnn::VersionInfo dnn_version = gpu_target_config.dnn_version_info;
  if (stream_exec != nullptr) {
    gpu_version = GetGpuVersion(stream_exec);
    se::dnn::DnnSupport* dnn = stream_exec->AsDnn();
    if (dnn == nullptr) {
      if (stream_exec->platform()->id() == stream_executor::musa::kMusaPlatformId) {
        dnn_version = se::dnn::VersionInfo();
      } else {
        return tsl::errors::FailedPrecondition(
            "DNN library initialization failed."
            " Look at the errors above for more details.");
      }
    } else {
      TF_ASSIGN_OR_RETURN(dnn_version, dnn->GetVersion());
    }
  }

  TF_RETURN_IF_ERROR(OptimizeHloConvolutionCanonicalization(
      hlo_module, gpu_version, dnn_version, options.device_allocator));

  {
    // Run layout assignment in a separate pipeline from
    // "post-layout-assignment" because we want everything after layout
    // assignment to have a layout-sensitive invariant-checker, but
    // HloPassPipeline also runs its invariant checker before any passes are
    // run, meaning, the pipeline that contains layout assignment cannot contain
    // a layout-sensitive verifier!
    HloPassPipeline pipeline("layout assignment");
    // Layout assignment uses alias analysis, which requires the call graph to
    // be flattened.
    pipeline.AddPass<FlattenCallGraph>();
    ChannelLayoutConstraints layout_constraints;
    pipeline.AddPass<GpuLayoutAssignment>(
        hlo_module->mutable_entry_computation_layout(), stream_exec,
        &layout_constraints);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  // Run target-specific HLO optimization passes after layout assignment.
  TF_RETURN_IF_ERROR(OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config, autotune_results,
      thread_pool));

  const se::DeviceDescription& gpu_device_info =
      gpu_target_config.gpu_device_info;

  FusionMergerOptions fusion_merger_options;
  if (IsMusaCompilation(stream_exec, platform_id_)) {
    fusion_merger_options.materialize_large_multi_user_reduction_producer =
        EnvExplicitlyTrue(
            "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER");
    fusion_merger_options.min_materialized_producer_elements = ReadInt64Env(
        "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS", 10000000);
    fusion_merger_options.min_materialized_producer_operands = ReadInt64Env(
        "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS", 16);
    fusion_merger_options.log_materialized_reduction_producers =
        EnvExplicitlyTrue("MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG");
  }

  TF_RETURN_IF_ERROR(FusionPipeline(debug_options, ShapeSizeBytesFunction(),
                                    gpu_device_info, fusion_merger_options)
                         .Run(hlo_module)
                         .status());

  if (debug_options.xla_gpu_collect_cost_model_stats()) {
    GpuHloCostAnalysis::Options cost_analysis_options{
        ShapeSizeBytesFunction(),
        /*per_second_rates=*/{},
        /*count_multiple_input_accesses=*/true};

    HloPassPipeline post_fusion_analysis("post_fusion_analysis");
    post_fusion_analysis.AddPass<GpuCostModelStatsCollection>(
        gpu_device_info, cost_analysis_options);
    TF_RETURN_IF_ERROR(post_fusion_analysis.Run(hlo_module).status());
  }

  TF_RETURN_IF_ERROR(
      HorizontalFusionPipeline(gpu_device_info).Run(hlo_module).status());

  if (IsMusaCompilation(stream_exec, platform_id_)) {
    HloPassPipeline pipeline("musa-hot-fusion-softmax-diag");
    pipeline.AddPass<MusaHotFusionSoftmaxDiag>();
    pipeline.AddPass<MusaReductionChainDiag>();
    pipeline.AddPass<MusaReductionChainRewrite>();
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  if (VLOG_IS_ON(2)) {
    HloFusionStatsVisitor stats;
    TF_RETURN_IF_ERROR(hlo_module->entry_computation()->Accept(&stats));
    VLOG(2) << stats.ToString();
  }

  {
    HloPassPipeline pipeline("post-fusion optimization");
    pipeline.AddPass<AllGatherCombiner>(
        debug_options.xla_gpu_all_gather_combine_threshold_bytes(),
        /*combine_threshold_count=*/256,
        debug_options.xla_gpu_enable_all_gather_combine_by_dim());
    pipeline.AddPass<AllReduceCombiner>(
        debug_options.xla_gpu_all_reduce_combine_threshold_bytes(),
        /*combine_threshold_count=*/256);
    pipeline.AddPass<ReduceScatterCombiner>(
        debug_options.xla_gpu_reduce_scatter_combine_threshold_bytes(),
        /*combine_threshold_count=*/256);

    if (debug_options.xla_gpu_all_reduce_contiguous()) {
      pipeline.AddPass<AllReduceContiguous>();
    }

    int32_t blueconnect_num_devices_per_host =
        debug_options.xla_gpu_all_reduce_blueconnect_num_devices_per_host();
    if (blueconnect_num_devices_per_host > 0) {
      pipeline.AddPass<AllReduceBlueConnect>(blueconnect_num_devices_per_host);
    }

    if (debug_options.xla_gpu_enable_while_loop_double_buffering()) {
      pipeline.AddPass<LoopDoubleBufferTransformer>();
      pipeline.AddPass<TupleSimplifier>();
      pipeline.AddPass<HloDCE>();
    }

    {
      // Convert all collectives to their async form, and then annotate the ones
      // that actually need to run asynchronously with a GPU specific backend
      // config.
      AsyncCollectiveCreator::CollectiveCreatorConfig config;
      config.convert_all_reduce = HloPredicateTrue;
      config.convert_collective_permute = HloPredicateTrue;
      config.convert_all_gather = HloPredicateTrue;
      config.convert_reduce_scatter = HloPredicateTrue;
      config.convert_all_to_all = HloPredicateTrue;
      pipeline.AddPass<AsyncCollectiveCreator>(std::move(config));

      auto convert_to_async = [&debug_options](const HloInstruction* inst) {
        const bool enable_all_async =
            debug_options.xla_gpu_enable_async_collectives();
        switch (inst->opcode()) {
          case HloOpcode::kAllReduceStart:
            return enable_all_async ||
                   debug_options.xla_gpu_enable_async_all_reduce();
          case HloOpcode::kAllGatherStart:
            return enable_all_async ||
                   debug_options.xla_gpu_enable_async_all_gather();
          case HloOpcode::kCollectivePermuteStart:
            return enable_all_async ||
                   debug_options.xla_gpu_enable_async_collective_permute();
          case HloOpcode::kAsyncStart: {
            auto async_inst = Cast<HloAsyncInstruction>(inst);
            switch (async_inst->async_wrapped_opcode()) {
              case HloOpcode::kReduceScatter:
                return enable_all_async ||
                       debug_options.xla_gpu_enable_async_reduce_scatter();
              case HloOpcode::kAllToAll:
                return enable_all_async ||
                       debug_options.xla_gpu_enable_async_all_to_all();
              default:
                return false;
            }
          }
          default:
            return false;
        }
      };
      pipeline.AddPass<GpuAsyncCollectiveAnnotator>(convert_to_async);
    }
    pipeline.AddPass<CollectivePermuteDecomposer>(
        debug_options.xla_gpu_collective_permute_decomposer_threshold());

    if (enable_all_pipelined || debug_options.xla_gpu_enable_pipelined_p2p()) {
      auto may_pipeline_p2p = [](const HloInstruction* instruction) {
        const HloRecvDoneInstruction* recv_done =
            DynCast<const HloRecvDoneInstruction>(instruction);
        if (!recv_done || recv_done->is_host_transfer()) return false;
        // Check that the recv-done is used for non-trivial computation, which
        // can also help avoid repeatedly pipelining a loop.
        return recv_done->user_count() == 1 && recv_done->parent() != nullptr &&
               recv_done->users()[0] != recv_done->parent()->root_instruction();
      };
      // We curretly use one asynchronous stream to execute P2P operations,
      // as such, can only support pipelining at most one P2P chain in each
      // loop.
      CollectivePipeliner::Config config{
          /*level_to_operate_on=*/0,
          /*max_pipelining_per_loop=*/1,
          /*last_run=*/true,
          /*pipeline_use_tree=*/false,
          /*process_different_sized_ops=*/true,
          /*pipelining_direction=*/
          CollectivePipeliner::PipeliningDirection::kBackward,
          /*should_process=*/may_pipeline_p2p};
      pipeline.AddPass<CollectivePipeliner>(config);
    }

    AlgebraicSimplifierOptions options = layout_insensitive_algsimp_opts;
    options.set_is_layout_sensitive(true);
    pipeline.AddPass<AlgebraicSimplifier>(options);

    // This invocation is used to populate deduplicated_name for fusions that
    // are considered duplicates according to the comparator in this pass.
    // Currently, the pass doesn't actually deduplicate the fusions.
    pipeline.AddPass<HloComputationDeduplicator>(
        /*mark_fusion_duplications=*/true);

    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  return OkStatus();
}

// Modifies the given HLO module so that it will be accepted by IrEmitter.
// Unlike optimization passes, the passes are necessary for correctness.
Status GpuCompiler::PrepareHloModuleForIrEmitting(HloModule* hlo_module) {
  return PrepareHloModuleForIrEmittingPipeline(*hlo_module, GetCanShareBuffer())
      .Run(hlo_module)
      .status();
}

Status GpuCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const GpuTargetConfig& gpu_target_config,
    const AutotuneResults* autotune_results,
    tsl::thread::ThreadPool* thread_pool) {
  // Constants:
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  const se::GpuComputeCapability gpu_version =
      gpu_target_config.gpu_device_info.gpu_compute_capability();
  const se::DeviceDescription& device_description =
      stream_exec != nullptr ? stream_exec->GetDeviceDescription()
                             : gpu_target_config.gpu_device_info;
  const AlgebraicSimplifierOptions simplifier_options = [&] {
    AlgebraicSimplifierOptions opts;
    opts.set_supports_non_canonical_dots(false);
    opts.set_is_layout_sensitive(true);
    opts.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    opts.set_minmax_propagate_nan(!debug_options.xla_gpu_enable_fast_min_max());
    opts.set_enable_unconditional_reduce_of_concat_replacement(false);
    return opts;
  }();
  TF_ASSIGN_OR_RETURN(AutotuneConfig autotune_config,
                      GetAutotuneConfig(stream_exec, debug_options, options,
                                        gpu_target_config, autotune_results));
  // Lambdas and related constants:
  const GpuFloatSupport bf16_support(BF16);
  const GpuFloatSupport f8e5m2_support(F8E5M2);
  const GpuFloatSupport f8e4m3fn_support(F8E4M3FN);
  const FloatSupport f8e4m3b11fnuz_support(F8E4M3B11FNUZ);
  const FloatSupport f8e5m2fnuz_support(F8E5M2FNUZ);
  const FloatSupport f8e4m3fnuz_support(F8E4M3FNUZ);
  auto add_float_normalization = [&](HloPassPipeline& pipeline) {
    auto& sub_pipeline =
        pipeline.AddPass<HloPassPipeline>("float_normalization");
    sub_pipeline.AddPass<FloatNormalization>(&bf16_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e5m2_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3fn_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3b11fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e5m2fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3fnuz_support);
    // Remove `f32 -> bf16 -> f32` casts inserted by bf16 normalization.
    if (debug_options.xla_gpu_simplify_all_fp_conversions()) {
      sub_pipeline.AddPass<SimplifyFPConversions>(
          SimplifyFPConversions::Scope::kSimplifyAllConversions);
    }
  };

  {
    HloPassPipeline pipeline("hlo normalization");

    // The LayoutAssignment pass may leave behind kCopy instructions which are
    // duplicate or NOPs, so remove them with algebraic simplification and CSE.
    pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(simplifier_options);

    // GemmRewriter assumes that all transposes are folded into gemms, but,
    // since commit 7d529df, this is not always true at this point.
    // Therefore, rerun transpose folding.
    pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot,
                                       TransposeFolding::NeverFoldTranspose);

    pipeline.AddPass<ReshapeDecomposer>();
    pipeline.AddPass<ReduceDecomposer>([&](const HloInstruction* r) {
      return IsReductionFromOrToContiguousDimensions(*r, device_description);
    });
    pipeline.AddPass<HloPassFix<MoveCopyToUsers>>();

    // Rewrite GEMMs into custom calls.
    se::GpuComputeCapability gpu_version =
        gpu_target_config.gpu_device_info.gpu_compute_capability();
    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version);
    if (debug_options.xla_gpu_enable_triton_gemm() && cuda_cc != nullptr &&
        cuda_cc->IsAtLeast(se::CudaComputeCapability::VOLTA)) {
      pipeline.AddPass<GemmRewriterTriton>(gpu_version);
    }
    const bool is_musa_compilation =
        IsMusaCompilation(stream_exec, platform_id_);
    if (EnvExplicitlyTrue("MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG")) {
      LOG(INFO) << "[MUSA_SAME_SHAPE_DOT_BATCHER_STAGE] stage=register"
                << " is_musa=" << is_musa_compilation
                << " stream_exec_present=" << (stream_exec != nullptr)
                << " stream_platform="
                << (stream_exec != nullptr ? stream_exec->platform()->Name()
                                           : "<null>");
    }
    if (is_musa_compilation) {
      pipeline.AddPass<MusaSameRhsDotMerger>();
      pipeline.AddPass<MusaSameLhsDotMerger>();
      pipeline.AddPass<MusaSameShapeDotBatcher>();
      pipeline.AddPass<MusaDotEpilogueFusion>();
    }
    pipeline.AddPass<GemmRewriter>(gpu_version);
    if (IsMusaCompilation(stream_exec, platform_id_)) {
      pipeline.AddPass<MusaGemmEpilogueFusion>();
      pipeline.AddPass<MusaGemmBetaChainMerger>(
          /*allow_custom_call=*/false);
    }

    // Rewrite GEMMs with broadcasted inputs as strided GEMMs.
    pipeline.AddPass<GemmBroadcastFoldingRewriter>();

    if (debug_options.xla_gpu_normalize_layouts()) {
      pipeline.AddPass<LayoutNormalization>(&NormalizeLayoutForGpuCustomCalls);
      pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(simplifier_options);
    }
    pipeline.AddPass<BroadcastCanonicalizer>();

    pipeline.AddPass<ReductionDegenerateDimRemover>();
    pipeline.AddPass<ReductionLayoutNormalizer>();
    // Run Softmax fusion after layout normalization. We expect a default layout
    // in the softmax codegen pipeline. However we should run before
    // ReductionDimensionGrouper, as that makes matching the softmax pattern
    // harder.
    if (debug_options.xla_gpu_enable_triton_softmax_fusion() &&
        cuda_cc != nullptr &&
        cuda_cc->IsAtLeast(se::CudaComputeCapability::VOLTA)) {
      pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(simplifier_options);
      pipeline.AddPass<SoftmaxRewriterTriton>(gpu_version);
    }

    pipeline.AddPass<ReductionDimensionGrouper>();
    pipeline.AddPass<HloPassFix<ReductionSplitter>>(device_description,
                                                    /*ignore_small_dims=*/false);
    pipeline.AddPass<HloPassFix<GpuTreeReductionRewriter>>(device_description);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  HloPassPipeline pipeline("post-layout_assignment");
  AddHloVerifier(&pipeline,
                 HloVerifierOpts{}
                     .MakeLayoutSensitive()
                     .WithInstructionCanChangeLayout(
                         LayoutAssignment::InstructionCanChangeLayout)
                     .VerifyBroadcastDimensionsOrder()
                     .VerifyReshapeIsBitcast(),
                 /*debug_only=*/true);

  // Linearize collective schedule if online autotuning of convolutions is
  // enabled.
  pipeline.AddPass<CollectivesScheduleLinearizer>(
      [this, stream_exec](const HloModule* module) {
        return RequiresCollectiveScheduleLinearizer(module, stream_exec);
      });

  // Triton compilation needs normalized operations on bf16 (i.e. converted to
  // f32).
  add_float_normalization(pipeline);

  TF_RETURN_IF_ERROR(AddTritonGemmAutotuningPasses(
      &pipeline, hlo_module, autotune_config, thread_pool));
  // Inline back the calls which have better performance with cuBLAS.
  pipeline.AddPass<CallInliner>();
  // TODO(tdanyluk): Apply CublasPadForGemms to the cuBLAS GEMMs generated
  // here for possibly better cuBLAS performance.
  if (IsMusaCompilation(stream_exec, platform_id_)) {
    pipeline.AddPass<MusaDotEpilogueFusion>();
  }
  pipeline.AddPass<GemmRewriter>(gpu_version);
  if (IsMusaCompilation(stream_exec, platform_id_)) {
    pipeline.AddPass<MusaGemmEpilogueFusion>();
    pipeline.AddPass<MusaGemmBetaChainMerger>(
        /*allow_custom_call=*/false);
  }
  // Rewrite GEMMs with broadcasted inputs as strided GEMMs.
  pipeline.AddPass<GemmBroadcastFoldingRewriter>();

  TF_RETURN_IF_ERROR(AddConvAndGemmAutotuningPasses(
      &pipeline, hlo_module, autotune_config, thread_pool));

  // The Triton autotuner can insert new bf16 reductions that need to be
  // normalized again.
  add_float_normalization(pipeline);

  // Clean up new_tuple described above.
  pipeline.AddPass<TupleSimplifier>();

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(simplifier_options);

  // Since this CSE runs after collective schedule linearizer which inserts
  // control dependencies, ignore these control deps when replacing instructions
  // with equivalent ones here.
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                           /*only_fusion_computations*/ false,
                           /*ignore_control_dependencies=*/true);
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return OkStatus();
}

StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  const DebugOptions& debug_options = module->config().debug_options();
  TF_RETURN_IF_ERROR(LoadAutotuneResultsFromFile(debug_options));

  MaybeUploadUnoptimizedGpuSymbolsToXSymbol(
      module.get(), GetGpuTargetConfig(stream_exec).ToProto());

  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("GpuCompiler::RunHloPasses for ", module->name()),
      !options.is_autotuning_compilation);
  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tsl::profiler::TraceMeLevel::kInfo);

  GpuTargetConfig gpu_target_config = GetGpuTargetConfig(stream_exec);
  TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(), stream_exec, options,
                                       gpu_target_config,
                                       /*autotune_results=*/nullptr));

  if (IsMusaCompilation(stream_exec, platform_id_) &&
      MusaGemmBetaChainCustomCallEnabled()) {
    HloPassPipeline late_musa_pipeline("late MUSA GEMM beta-chain custom-call");
    late_musa_pipeline.AddPass<MusaGemmBetaChainMerger>(
        /*allow_custom_call=*/true);
    TF_RETURN_IF_ERROR(late_musa_pipeline.Run(module.get()).status());
  }

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

  uint64_t end_usecs = tsl::Env::Default()->NowMicros();

  // This won't record values for calls that error out (because if they error
  // out we have no way of telling how far through the process we got).
  RecordHloPassesDuration(end_usecs - start_usecs);

  TF_RETURN_IF_ERROR(SerializeAutotuneResultsToFile(debug_options));

  return std::move(module);
}

StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPassesWithoutDevice(
    std::unique_ptr<HloModule> module, const CompileOptions& options,
    const GpuTargetConfig& gpu_target_config,
    const AutotuneResults& autotune_results) {
  MaybeUploadUnoptimizedGpuSymbolsToXSymbol(module.get(),
                                            gpu_target_config.ToProto());
  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("GpuCompiler::RunHloPasses for ", module->name()),
      !options.is_autotuning_compilation);
  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tsl::profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(), nullptr, options,
                                       gpu_target_config, &autotune_results));

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

  uint64_t end_usecs = tsl::Env::Default()->NowMicros();

  // This won't record values for calls that error out (because if they error
  // out we have no way of telling how far through the process we got).
  RecordHloPassesDuration(end_usecs - start_usecs);

  return std::move(module);
}

namespace {
Status RunPostSchedulingCopyInsertion(
    HloModule* module,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer) {
  // We run a separate pass of copy elision here because the sequential ordering
  // from the HLO schedule potentially allows for more copies to be eliminated.
  constexpr int64_t kRegionBasedLiveRangeAnalysisLimit = -1;
  const int64_t kUseRegionBasedLiveRangeAnalysis =
      module->config()
              .debug_options()
              .xla_gpu_copy_insertion_use_region_analysis()
          ? kRegionBasedLiveRangeAnalysisLimit
          : 0;
  CopyInsertion copy_insertion(can_share_buffer,
                               kUseRegionBasedLiveRangeAnalysis);
  TF_RETURN_IF_ERROR(copy_insertion.RemoveUnnecessaryCopies(module));

  // Stash away the schedule during copy insertion, to avoid validation failures
  // while the module is in flux.
  HloSchedule saved_schedule = module->schedule();
  module->clear_schedule();

  // RemoveUnnecessaryCopies only considers interference when determining
  // whether it is legal to remove a copy. However, copies in the graph may be
  // necessary for other reason such as preventing a constant from being live
  // out of the graph. So run AddSpecialCaseCopies to re-insert these copies.
  TF_RETURN_IF_ERROR(
      copy_insertion.CopyInsertion::AddSpecialCaseCopies(module));

  TF_RETURN_IF_ERROR(HloDCE().Run(module).status());

  // The passes above can add and remove copies, update the schedule to
  // account for these transformations. Newly added instructions will be
  // placed ASAP in the schedule.

  // Update and restore the schedule. The saved schedule has a reference to the
  // updated HLO module. The saved schedule needs to be updated before restoring
  // it to the module to avoid validation failures.
  TF_RETURN_IF_ERROR(saved_schedule.Update());
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(saved_schedule)));

  return OkStatus();
}
}  // namespace

StatusOr<std::unique_ptr<BufferAssignment>> GpuCompiler::AssignBuffers(
    HloModule* hlo_module, se::StreamExecutor* stream_exec) {
  const se::DeviceDescription& gpu_device_info =
      stream_exec->GetDeviceDescription();
  const int64_t scheduler_mem_limit =
      GetSchedulerMemoryLimit(hlo_module, gpu_device_info, pointer_size_);
  TF_RETURN_IF_ERROR(ScheduleGpuModule(hlo_module, pointer_size_,
                                       scheduler_mem_limit, gpu_device_info));
  TF_RETURN_IF_ERROR(
      RunPostSchedulingCopyInsertion(hlo_module, GetCanShareBuffer()));

  auto buffer_size_bytes_function =
      [this](const BufferValue& buffer_value) -> int64_t {
    return GetSizeOfShape(buffer_value.shape(), pointer_size_);
  };

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          hlo_module,
          std::make_unique<SequentialHloOrdering>(hlo_module->schedule()),
          buffer_size_bytes_function,
          /*color_alignment=*/
          [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
          /*allocate_buffers_for_constants=*/true,
          /*colorer=*/BufferAssigner::DefaultColorer(),
          /*must_not_live_out=*/{}, GetCanShareBuffer()));

  return std::move(assignment);
}

using OutputInfoMap =
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;

static void NullDiagnosticHandler(const llvm::DiagnosticInfo& diag_info,
                                  void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info.print(diagnostic_printer);

  VLOG(5) << error_string;
}

namespace {

std::unique_ptr<llvm::Module> CopyToContext(const llvm::Module& module,
                                            llvm::LLVMContext& context) {
  // We are setting llvm::SmallString's InternalLen to 0, because we want to
  // allocate its buffer on the heap. We use llvm::SmallString instead of
  // std::string, because llvm::raw_svector_ostream is a bit faster than
  // llvm::raw_string_ostream.
  llvm::SmallString<0> bitcode;
  llvm::raw_svector_ostream bitcode_ostream(bitcode);
  llvm::WriteBitcodeToFile(module, bitcode_ostream);

  llvm::Expected<std::unique_ptr<llvm::Module>> new_module =
      llvm::parseBitcodeFile(
          llvm::MemoryBufferRef(llvm::StringRef(bitcode.data(), bitcode.size()),
                                "split_module"),
          context);
  CHECK(new_module) << "Failed to parse bitcode "
                    << llvm::toString(new_module.takeError());

  return std::move(new_module.get());
}

}  // namespace

StatusOr<std::pair<std::string, std::vector<uint8_t>>>
GpuCompiler::CompileToTargetBinary(const HloModuleConfig& module_config,
                                   std::unique_ptr<llvm::Module> llvm_module,
                                   se::GpuComputeCapability gpu_version,
                                   se::StreamExecutor* stream_exec,
                                   const CompileOptions& options,
                                   const HloModule* debug_module) {
  using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;

  const auto compile_single_module =
      [this, gpu_version, &module_config, &options, debug_module](
          llvm::Module* llvm_module, bool relocatable,
          std::optional<int> shard_number) -> StatusOr<BackendCompileResult> {
    {
      // This may print multiple lines per HLO compilation because of the
      // parallelized compilation of LLVM modules.
      XLA_SCOPED_LOGGING_TIMER_IF(
          absl::StrCat(
              "GpuCompiler::RunBackend - Running LLVM verifier for ",
              (debug_module != nullptr ? debug_module->name() : "(unknown)")),
          !options.is_autotuning_compilation);

      llvm_module->getContext().setDiagnosticHandlerCallBack(
          NullDiagnosticHandler, nullptr);

      std::string err;
      llvm::raw_string_ostream err_stream(err);

      // verifyModule() returns true if the module is broken.
      TF_RET_CHECK(!llvm::verifyModule(*llvm_module, &err_stream))
          << "Invalid LLVM IR before optimizations:\n"
          << err_stream.str()
          << "\nThis probably indicates a bug in the HLO -> LLVM IR "
             "lowering. Rerun with --xla_dump_to to get the IR"
          << (debug_module
                  ? absl::StrCat(" and looks for files with name containing: *",
                                 FilenameFor(*debug_module, "", ""), "*")
                  : ".");
    }
    StatusOr<std::pair<std::string, std::vector<uint8_t>>> result =
        CompileTargetBinary(module_config, llvm_module, gpu_version,
                            relocatable, debug_module, options);

    if (!result.ok()) {
      return result;
    }

    const bool should_dump =
        DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                   module_config.debug_options());

    if (should_dump) {
      if (debug_module) {
        if (shard_number.has_value()) {
          llvm_ir::DumpIrIfEnabled(*debug_module, *llvm_module,
                                   /*optimized=*/true,
                                   std::to_string(*shard_number));
        } else {
          llvm_ir::DumpIrIfEnabled(*debug_module, *llvm_module,
                                   /*optimized=*/true);
        }
      } else {
        LOG(ERROR)
            << "Dumping is not implemented since the file name cannot be "
               "inferred. Please implement (potentially MLIR) module -> "
               "filename heuristic.";
      }
    }

    if (user_post_optimization_hook_) {
      user_post_optimization_hook_(*llvm_module);
    }

    // Write PTX to IR dump directory, if IR dumping was requested.
    if (should_dump) {
      absl::string_view ptx = result->first;
      if (debug_module) {
        if (shard_number.has_value()) {
          DumpToFileInDirOrStdout(*debug_module, "",
                                  std::to_string(*shard_number) + ".ptx", ptx);
        } else {
          DumpToFileInDirOrStdout(*debug_module, "", "ptx", ptx);
        }
      } else {
        LOG(ERROR)
            << "Dumping is not implemented since the file name cannot be "
               "inferred. Please implement (potentially MLIR) module -> "
               "filename heuristic.";
      }
    }

    return result;
  };

  // Disable multi-threading during deviceless AOT compilation.
  // TODO(anlunx): Enable multi-threading once deviceless AOT compilation is
  // enabled.
  if (!stream_exec) {
    return compile_single_module(llvm_module.get(), /*relocatable=*/false,
                                 /*shard_number=*/std::nullopt);
  }

  tsl::thread::ThreadPool* thread_pool;
  std::optional<tsl::thread::ThreadPool> overriding_thread_pool;
  switch (
      module_config.debug_options().xla_gpu_force_compilation_parallelism()) {
    case 0:
      thread_pool = options.thread_pool;
      break;
    case 1:
      thread_pool = nullptr;
      break;
    default:
      overriding_thread_pool.emplace(
          tsl::Env::Default(), "",
          module_config.debug_options()
              .xla_gpu_force_compilation_parallelism());
      thread_pool = &*overriding_thread_pool;
      break;
  }

  if (!thread_pool) {
    return compile_single_module(llvm_module.get(), /*relocatable=*/false,
                                 /*shard_number=*/std::nullopt);
  }

  // Test whether LinkModules is supported.
  TF_ASSIGN_OR_RETURN(bool can_use_link_modules,
                      CanUseLinkModules(module_config));
  if (!can_use_link_modules) {
    return compile_single_module(llvm_module.get(), /*relocatable=*/false,
                                 /*shard_number=*/std::nullopt);
  }
  std::vector<std::unique_ptr<llvm::Module>> llvm_modules;
  int num_functions = 0;
  for (llvm::Function& func : llvm_module->functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      num_functions++;
    }
  }

  // Record the name of some constant global variables and their initializers.
  // We'll change the linkage type of these variables from external to internal
  // to ensure constant-folding works properly after calling llvm::SplitModule.
  llvm::DenseMap<llvm::StringRef, llvm::Constant*> const_initializer_map;
  for (llvm::GlobalVariable& gv : llvm_module->globals()) {
    if (gv.hasName() && gv.isConstant() && gv.hasInitializer() &&
        gv.hasExternalLinkage()) {
      llvm::Constant* initializer = gv.getInitializer();
      unsigned int num_elements = 0;
      if (auto* caz =
              llvm::dyn_cast<llvm::ConstantAggregateZero>(initializer)) {
        num_elements = caz->getElementCount().getFixedValue();
      } else if (auto* cds = llvm::dyn_cast<llvm::ConstantDataSequential>(
                     initializer)) {
        num_elements = cds->getNumElements();
      }
      if (num_elements > 0) {
        const_initializer_map[gv.getName()] = initializer;
      }
    }
  }

  llvm::SplitModule(
      *llvm_module,
      std::max<unsigned>(
          1, std::min<unsigned>(thread_pool->NumThreads(), num_functions)),
      [&](std::unique_ptr<llvm::Module> module) {
        // Change the linkage type of some global constant variables to internal
        for (llvm::GlobalVariable& gv : module->globals()) {
          if (gv.hasName() && gv.isConstant() && !gv.hasInitializer() &&
              const_initializer_map.count(gv.getName()) != 0) {
            gv.setInitializer(const_initializer_map[gv.getName()]);
            gv.setLinkage(llvm::GlobalValue::InternalLinkage);
          }
        }
        llvm_modules.push_back(std::move(module));
      },
      /*PreserveLocals=*/true);

  std::vector<StatusOr<BackendCompileResult>> compile_results(
      llvm_modules.size());
  tsl::BlockingCounter counter(llvm_modules.size());
  for (int i = 0; i < llvm_modules.size(); i++) {
    thread_pool->Schedule(
        [&compile_results, compile_single_module, i, &llvm_modules, &counter] {
          // Each thread has its own context to avoid race conditions.
          llvm::LLVMContext new_context;
          std::unique_ptr<llvm::Module> new_module =
              CopyToContext(*llvm_modules.at(i), new_context);
          compile_results.at(i) =
              compile_single_module(new_module.get(),
                                    /*relocatable=*/true, /*shard_number=*/i);
          counter.DecrementCount();
        });
  }
  counter.Wait();

  std::string ptx_snippets;
  std::vector<std::vector<uint8_t>> submodule_compile_results;
  for (auto& maybe_result : compile_results) {
    TF_ASSIGN_OR_RETURN(auto result, maybe_result);
    if (result.second.empty()) {
      continue;
    }
    ptx_snippets += result.first;
    ptx_snippets += "\n";
    submodule_compile_results.push_back(result.second);
  }

  auto maybe_backend_result =
      this->LinkModules(stream_exec, std::move(submodule_compile_results),
                        module_config.debug_options());
  if (!maybe_backend_result.ok()) {
    LOG(ERROR) << "The CUDA linking API did not work. Please use "
                  "XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 to "
                  "bypass it, but expect to get longer compilation time due to "
                  "the lack of multi-threading. Original error: "
               << maybe_backend_result.status();
    return maybe_backend_result.status();
  }

  return std::make_pair(ptx_snippets, std::move(*maybe_backend_result));
}

StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  if (!options.is_autotuning_compilation) {
    VLOG(1) << "Starting to compile HLO module " << module->name();
  }

  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("GpuCompiler::RunBackend for ", module->name()),
      !options.is_autotuning_compilation);
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

  if (options.is_autotuning_compilation) {
    if (module->config()
            .debug_options()
            .xla_gpu_enable_persistent_temp_buffers()) {
      LOG(WARNING) << "Doing autotuning compilations with "
                      "xla_gpu_enable_persistent_temp_buffers wastes memory!";
    }
    if (module->config().debug_options().xla_embed_ir_in_executable()) {
      LOG(WARNING) << "Doing autotuning compilations with "
                      "xla_embed_ir_in_executable wastes memory!";
    }
  }

  TF_RET_CHECK(stream_exec != nullptr);

  llvm::LLVMContext llvm_context;

  const se::DeviceDescription& gpu_device_info =
      stream_exec->GetDeviceDescription();

  if (module->config().hlo_profiling_enabled() || VLOG_IS_ON(1)) {
    HloCostAnalysis::Options cost_analysis_options{ShapeSizeBytesFunction()};
    cost_analysis_options.set_bytes_per_second(
        stream_exec->GetDeviceDescription().memory_bandwidth());
    GpuHloCostAnalysis cost_analysis(cost_analysis_options, &gpu_device_info);
    TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&cost_analysis));
    if (!options.is_autotuning_compilation) {
      VLOG(1) << "HLO memory read+written: "
              << tsl::strings::HumanReadableNumBytes(
                     cost_analysis.bytes_accessed());
    }
    if (module->config().hlo_profiling_enabled()) {
      LOG(ERROR) << "--xla_hlo_profile for GPU is unsupported.";
    }
  }

  const int64_t scheduler_mem_limit =
      GetSchedulerMemoryLimit(module.get(), gpu_device_info, pointer_size_);

  TF_RETURN_IF_ERROR(ScheduleGpuModule(module.get(), pointer_size_,
                                       scheduler_mem_limit, gpu_device_info));

  TF_RETURN_IF_ERROR(
      RunPostSchedulingPipelines(module.get(), scheduler_mem_limit));

  CompileModuleResults compile_module_results;

  TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
      module.get(), &llvm_context, target_triple_, data_layout_,
      stream_exec->platform()->Name(), stream_exec->platform()->id(),
      gpu_device_info, GetCanShareBuffer(), BufferSizeBytesFunction(),
      &compile_module_results, stream_exec));

  if (user_pre_optimization_hook_) {
    user_pre_optimization_hook_(*compile_module_results.llvm_module);
  }
  std::string ir_module_string_before_opt;
  const bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  if (embed_ir_in_executable) {
    ir_module_string_before_opt =
        llvm_ir::DumpToString(compile_module_results.llvm_module.get());
  }

  llvm_ir::DumpIrIfEnabled(*module, *compile_module_results.llvm_module,
                           /*optimized=*/false);

  std::string asm_text;
  std::vector<uint8_t> binary;

  TF_ASSIGN_OR_RETURN(
      std::tie(asm_text, binary),
      CompileToTargetBinary(
          module->config(), std::move(compile_module_results.llvm_module),
          GetGpuVersion(stream_exec), stream_exec, options, module.get()));

  RecordXlaDeviceBinarySize(binary.size());

  if (DumpingEnabledForHloModule(*module) &&
      std::holds_alternative<GpuExecutable::OwnedThunkSequence>(
          compile_module_results.executable)) {
    const ThunkSequence& thunk_sequence =
        *std::get<GpuExecutable::OwnedThunkSequence>(
            compile_module_results.executable);
    DumpToFileInDirOrStdout(*module, "", "thunk_sequence.txt",
                            thunk_sequence.ToString());
  }

  std::shared_ptr<const BufferAssignment> buffer_assignment;
  std::unique_ptr<BufferAssignmentProto> buffer_assignment_proto;
  std::function<std::string()> buffer_assignment_dumper = [] {
    return std::string();
  };
  if (!options.is_autotuning_compilation) {
    // Make it shared to be captured in the later lambda.
    buffer_assignment = std::move(compile_module_results.buffer_assignment);
    buffer_assignment_proto =
        std::make_unique<BufferAssignmentProto>(buffer_assignment->ToProto());
    size_t max_buffers_to_show =
        module->config().debug_options().xla_debug_buffer_assignment_show_max();
    buffer_assignment_dumper = [buffer_assignment, max_buffers_to_show] {
      return buffer_assignment->ToVerboseString(max_buffers_to_show);
    };
  }

  TF_ASSIGN_OR_RETURN(
      auto gpu_executable,
      GpuExecutable::Create(GpuExecutable::Params{
          /*asm_text=*/(options.is_autotuning_compilation && !binary.empty())
              ? std::string()
              : std::move(asm_text),
          /*binary=*/std::move(binary),
          /*gpu_version=*/GetGpuVersion(stream_exec),
          /*executable=*/std::move(compile_module_results.executable),
          /*entry_func_attrs=*/
          std::move(compile_module_results.entry_func_attrs),
          /*constants=*/std::move(compile_module_results.constants),
          /*output_info=*/std::move(compile_module_results.output_info),
          /*module_name=*/std::move(compile_module_results.module_name),
          /*output_shape=*/std::move(compile_module_results.output_shape),
          /*allocations=*/std::move(compile_module_results.allocations),
          /*enable_persistent_temp_buffers=*/
          module->config()
              .debug_options()
              .xla_gpu_enable_persistent_temp_buffers(),
          /*debug_buffer_assignment=*/std::move(buffer_assignment_proto),
          /*verbose_buffer_assignment_string_dumper=*/
          std::move(buffer_assignment_dumper),
          /*debug_module=*/options.is_autotuning_compilation
              ? std::unique_ptr<HloModule>()
              : std::move(module),
          /*enable_debug_info_manager=*/!options.is_autotuning_compilation}));

  if (embed_ir_in_executable) {
    DCHECK_NE("", ir_module_string_before_opt);
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }

  IncrementCompiledProgramsCount();

  if (!options.is_autotuning_compilation && gpu_executable->has_module()) {
    // Dump computation proto state and buffer assignment for
    // CompiledMemoryAnalysis.
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_hlo_module() = gpu_executable->module().ToProto();
    *hlo_proto->mutable_buffer_assignment() = buffer_assignment->ToProto();
    gpu_executable->set_hlo_proto(std::move(hlo_proto));
    gpu_executable->set_debug_info(buffer_assignment->GetStats().ToString());
  }

  return static_cast<std::unique_ptr<Executable>>(std::move(gpu_executable));
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
GpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& options) {
  CHECK(options.PlatformId() == se::cuda::kCudaPlatformId);

  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();
  std::vector<std::unique_ptr<AotCompilationResult>> results;

  std::any target_config = options.target_config();
  auto* gpu_target_config = std::any_cast<GpuTargetConfig>(&target_config);
  CHECK(gpu_target_config != nullptr || options.executor() != nullptr);
  const se::DeviceDescription& gpu_device_info =
      gpu_target_config != nullptr ? gpu_target_config->gpu_device_info
                                   : options.executor()->GetDeviceDescription();
  for (const auto& module : modules) {
    llvm::LLVMContext llvm_context;

    const int64_t scheduler_mem_limit =
        GetSchedulerMemoryLimit(module.get(), gpu_device_info, pointer_size_);
    TF_RETURN_IF_ERROR(ScheduleGpuModule(module.get(), pointer_size_,
                                         scheduler_mem_limit, gpu_device_info));
    TF_RETURN_IF_ERROR(
        RunPostSchedulingPipelines(module.get(), scheduler_mem_limit));

    // Compile the module
    CompileModuleResults compile_module_results;

    if (gpu_target_config) {
      TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
          module.get(), &llvm_context, target_triple_, data_layout_,
          gpu_target_config->platform_name, options.PlatformId(),
          gpu_target_config->gpu_device_info, GetCanShareBuffer(),
          BufferSizeBytesFunction(), &compile_module_results));
    } else {
      CHECK(options.executor() != nullptr);
      auto stream_exec = options.executor();
      TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
          module.get(), &llvm_context, target_triple_, data_layout_,
          stream_exec->platform()->Name(), options.PlatformId(),
          stream_exec->GetDeviceDescription(), GetCanShareBuffer(),
          BufferSizeBytesFunction(), &compile_module_results));
    }
    if (user_pre_optimization_hook_) {
      user_pre_optimization_hook_(*compile_module_results.llvm_module);
    }

    using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;
    BackendCompileResult backend_result;
    if (gpu_target_config) {
      TF_ASSIGN_OR_RETURN(
          backend_result,
          CompileToTargetBinary(
              module->config(), std::move(compile_module_results.llvm_module),
              gpu_target_config->gpu_device_info.gpu_compute_capability(),
              options.executor(), {options.device_allocator()}, module.get()));
    } else {
      TF_ASSIGN_OR_RETURN(
          backend_result,
          CompileToTargetBinary(
              module->config(), std::move(compile_module_results.llvm_module),
              GetGpuVersion(options.executor()), options.executor(),
              {options.device_allocator()}, module.get()));
    }

    auto& compiled_executable = compile_module_results.executable;

    if (!std::holds_alternative<GpuExecutable::OwnedGpuRuntimeProgram>(
            compiled_executable)) {
      return InternalError("Gpu runtime program was not provided");
    }

    // TODO(ezhulenev): Unify AOT compilation with GpuRuntimeExecutable::Create
    // (see `gpu/runtime/executable.h`).

    const auto& program =
        std::get<GpuExecutable::OwnedGpuRuntimeProgram>(compiled_executable);

    // Options for the default XLA runtime compilation pipeline.
    runtime::CompilationPipelineOptions copts;

    // Populate mapping from XLA (SE) enums/structs type id to symbol names.
    copts.populate_type_id_names = RegisterXlaGpuTypeIdNames;

    // For passing LMHLO attributes as XLA (SE) enums/structs to custom calls.
    copts.populate_attr_encodings = RegisterXlaGpuAttrEncoding;

    // Options for constructing XLA runtime JitExecutable.
    runtime::JitExecutable::Options opts;
    opts.specialization = runtime::JitExecutable::Specialization::kDisabled;
    opts.compiler.register_dialects =
        runtime::RegisterDefaultXlaGpuRuntimeDialects;

    // Register XLA Gpu runtime custom calls with the linker.
    opts.compiler.symbols_binding = runtime::ToSymbolsBinding(
        RegisterXlaGpuRuntimeCustomCalls, RegisterXlaGpuTypeIdNames);

    opts.compiler.create_compilation_pipeline =
        [copts](xla::runtime::PassManager& passes) {
          runtime::CreateDefaultXlaGpuRuntimeCompilationPipeline(passes, copts);
        };

    // Instantiate new JitExecutable from the MLIR source.
    auto jit_executable = runtime::JitExecutable::Instantiate(
        program->module, program->entry_point, opts);
    if (!jit_executable.ok())
      return InternalError("Failed to compile XLA program: %s",
                           jit_executable.status().message());

    // For static shapes we can always serialize only the default executable.
    runtime::Executable& executable = jit_executable->DefaultExecutable().get();

    // Check if XLA runtime executable saved the compilation result.
    std::unique_ptr<llvm::MemoryBuffer> obj_file = executable.obj_file();
    if (!obj_file)
      return InternalError("XLA runtime executable didn't save the obj file");

    std::string data(obj_file->getBuffer().data(),
                     obj_file->getBuffer().size());

    results.emplace_back(std::make_unique<GpuXlaRuntimeAotCompilationResult>(
        module->ToProto(), data, program->module,
        compile_module_results.entry_func_attrs, backend_result.first,
        backend_result.second, compile_module_results.constants));
  }
  return std::move(results);
}

HloCostAnalysis::ShapeSizeFunction GpuCompiler::ShapeSizeBytesFunction() const {
  // Capture just the pointer size, not the entire GpuCompiler object.
  return [pointer_size = pointer_size_](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
}

StatusOr<std::unique_ptr<AotCompilationResult>> GpuCompiler::Export(
    Executable* executable) const {
  auto* gpu_executable = tensorflow::down_cast<GpuExecutable*>(executable);
  if (!gpu_executable) return Internal("GpuExecutable is null");
  HloModuleProto module_proto = gpu_executable->module().ToProto();
  TF_ASSIGN_OR_RETURN(auto obj_file, gpu_executable->GetObjFile());
  TF_ASSIGN_OR_RETURN(auto mlir_module, gpu_executable->GetMlirModule());
  xla::EntryFunctionAttributes entry_func_attrs =
      gpu_executable->entry_func_attrs();
  auto text = gpu_executable->text();
  auto binary = gpu_executable->binary();

  std::unique_ptr<AotCompilationResult> result =
      std::make_unique<xla::gpu::GpuXlaRuntimeAotCompilationResult>(
          module_proto, obj_file, mlir_module, entry_func_attrs, text, binary,
          gpu_executable->constants());
  return result;
}

Status GpuCompiler::RunPostSchedulingPipelines(
    HloModule* module, int64_t scheduler_mem_limit) const {
  TF_RETURN_IF_ERROR(
      RunPostSchedulingCopyInsertion(module, GetCanShareBuffer()));
  {
    HloPassPipeline pipeline("post-scheduling-passes");

    HloPredicate is_nop =
        HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant,
                         HloOpcode::kBitcast, HloOpcode::kGetTupleElement>;
    pipeline.AddPass<GpuConvertAsyncCollectivesToSync>(is_nop);
    pipeline.AddPass<OptimizationBarrierExpander>();

    TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  }

  {
    HloPassPipeline pipeline("remat-pipeline");

    HloCostAnalysis hlo_cost_analysis(ShapeSizeBytesFunction());
    HloRematerialization::RematerializationModeConfig
        rematerialization_mode_config(/*recompute=*/true, /*compress=*/true,
                                      /*host_offload=*/false);
    HloRematerialization::Options options(
        hlo_cost_analysis, rematerialization_mode_config,
        // Assume 75% of the total device memory is available for XLA.
        /*memory_limit_bytes=*/scheduler_mem_limit,
        /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
        /*min_remat_size=*/0, /*compact_shape_function=*/nullptr,
        /*host_memory_offload_config=*/std::nullopt);
    HloRematerialization::RematerializationSizes sizes;
    pipeline.AddPass<HloRematerialization>(options, sizes);

    TF_ASSIGN_OR_RETURN(bool changed, pipeline.Run(module));
    if (changed) {
      VLOG(1) << "HloRematerialization saved "
              << sizes.before_bytes - sizes.after_bytes << " bytes";
    }
  }

  {
    HloPassPipeline pipeline("fusion-wrapper");
    pipeline.AddPass<FusionWrapper>();
    // Wrap remaining unfused ops that have no LHLO equivalent in single-op
    // fusions. This needs to happen after rematerialization, because that will
    // insert additional copies.
    TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  }
  return OkStatus();
}

se::GpuComputeCapability GpuCompiler::GetGpuVersion(
    se::StreamExecutor* stream_exec) {
  return stream_exec->GetDeviceDescription().gpu_compute_capability();
}

Status GpuCompiler::LoadAutotuneResultsFromFile(
    const DebugOptions& debug_options) {
  // We are doing this before the timer is started.
  if (absl::string_view file_path =
          debug_options.xla_gpu_load_autotune_results_from();
      !file_path.empty()) {
    static absl::once_flag once;
    Status status = OkStatus();
    absl::call_once(once, [&file_path, &status] {
      status = AutotunerUtil::LoadAutotuneResultsFromFile(file_path);
    });
    TF_RETURN_IF_ERROR(status);
  }
  return OkStatus();
}

Status GpuCompiler::SerializeAutotuneResultsToFile(
    const DebugOptions& debug_options) {
  // We are doing this after the timer is finished.
  if (absl::string_view file_path =
          debug_options.xla_gpu_dump_autotune_results_to();
      !file_path.empty()) {
    // Warning: This writes the autotune results at every compilation, possibly
    // multiple times per process.
    TF_RETURN_IF_ERROR(
        AutotunerUtil::SerializeAutotuneResultsToFile(file_path));
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
