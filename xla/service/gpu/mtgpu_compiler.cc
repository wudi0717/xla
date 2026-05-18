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

#include "xla/service/gpu/mtgpu_compiler.h"

#include <memory>
#include <utility>

#include "xla/service/compiler.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/gpu/gpu_conv_padding_legalization.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/llvm_gpu_backend/mtgpu_backend.h"
#include "xla/service/gpu/musa_layer_norm_rewriter.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu/triangular_solve_rewriter.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/util.h"
#include "tsl/util/env_var.h"

namespace xla {

std::unique_ptr<Compiler> CreateMTGPUCompiler() {
  return std::make_unique<gpu::MTGPUCompiler>();
}

namespace gpu {
namespace {

bool IsMusaCustomFusionEnabled() {
  static const bool enabled = [] {
    bool value = false;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("MUSA_CUSTOM_FUSION",
                                        /*default_val=*/false, &value));
    return value;
  }();
  return enabled;
}

}  // namespace

Status MTGPUCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::GpuComputeCapability gpu_version,
    se::dnn::VersionInfo dnn_version,
    se::DeviceMemoryAllocator* device_allocator) {
  (void)dnn_version;
  (void)device_allocator;

  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);
  pipeline.AddPass<GpuConvRewriter>();
  pipeline.AddPass<GpuConvPaddingLegalization>();
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  AlgebraicSimplifierOptions options;
  options.set_enable_conv_operand_swap(false);
  options.set_enable_unconditional_reduce_of_concat_replacement(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);
  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return OkStatus();
}

Status MTGPUCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const GpuTargetConfig& gpu_target_config,
    const AutotuneResults* autotune_results,
    tsl::thread::ThreadPool* thread_pool) {
  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config, autotune_results,
      thread_pool));

  HloPassPipeline post_pipeline("MTGPU post-layout_assignment");
  VLOG(1) << "[mtgpu] start post-layout pipeline for module="
          << hlo_module->name();
  if (IsMusaCustomFusionEnabled()) {
    VLOG(1) << "[mtgpu] enable MusaLayerNormRewriter by MUSA_CUSTOM_FUSION";
    post_pipeline.AddPass<MusaLayerNormRewriter>();
  } else {
    VLOG(1) << "[mtgpu] disable MusaLayerNormRewriter by MUSA_CUSTOM_FUSION";
  }
  post_pipeline.AddPass<TriangularSolveRewriter>();
  TF_RETURN_IF_ERROR(post_pipeline.Run(hlo_module).status());
  VLOG(1) << "[mtgpu] finish post-layout pipeline for module="
          << hlo_module->name();

  return OkStatus();
}

bool MTGPUCompiler::RequiresCollectiveScheduleLinearizer(
    const HloModule* module, se::StreamExecutor* stream_exec) {
  (void)module;
  (void)stream_exec;
  return false;
}

Status MTGPUCompiler::AddConvAndGemmAutotuningPasses(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool) {
  (void)pipeline;
  (void)hlo_module;
  (void)autotune_config;
  (void)thread_pool;
  return OkStatus();
}

MTGPUCompiler::MTGPUCompiler()
    : GpuCompiler(stream_executor::musa::kMusaPlatformId,
                  mtgpu::TargetTriple(), mtgpu::DataLayout()) {}

StatusOr<std::pair<std::string, std::vector<uint8_t>>>
MTGPUCompiler::CompileTargetBinary(const HloModuleConfig& module_config,
                                   llvm::Module* llvm_module,
                                   se::GpuComputeCapability gpu_version,
                                   bool relocatable,
                                   const HloModule* debug_module,
                                   const CompileOptions& options) {
  (void)debug_module;
  if (relocatable) {
    return Unimplemented("relocatable target binary is not implemented");
  }

  std::vector<uint8_t> hsaco;
  {
    XLA_SCOPED_LOGGING_TIMER_IF(
        "MTGPUCompiler::CompileTargetBinary - CompileToHsaco",
        !options.is_autotuning_compilation);
    TF_ASSIGN_OR_RETURN(
        hsaco, mtgpu::CompileToHsaco(llvm_module, gpu_version,
                                     module_config.debug_options(),
                                     module_config.compilation_cache_key()));
  }

  return std::pair<std::string, std::vector<uint8_t>>("", std::move(hsaco));
}

}  // namespace gpu
}  // namespace xla
