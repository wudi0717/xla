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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/gpu/musa_fusion_custom_calls.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* old_value = std::getenv(name);
    if (old_value != nullptr) {
      old_value_ = old_value;
    }
    Set(value);
  }

  ~ScopedEnvVar() {
    if (old_value_.empty()) {
      Unset();
    } else {
      Set(old_value_.c_str());
    }
  }

 private:
  void Set(const char* value) {
#ifdef _WIN32
    _putenv_s(name_, value);
#else
    setenv(name_, value, /*overwrite=*/1);
#endif
  }

  void Unset() {
#ifdef _WIN32
    _putenv_s(name_, "");
#else
    unsetenv(name_);
#endif
  }

  const char* name_;
  std::string old_value_;
};

using MusaGemmEpilogueFusionTest = HloTestBase;

TEST_F(MusaGemmEpilogueFusionTest, FusesLegacyGemmPlusAliasedMatrixAdd) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "1");

  const std::string hlo_text = R"(
HloModule t, input_output_alias={ {}: (2, {}, must-alias) }

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias = f32[8,4]{1,0} parameter(2)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_cublaslt(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  se::GpuComputeCapability gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  TF_ASSERT_OK_AND_ASSIGN(bool gemm_changed,
                          GemmRewriter(gpu_version).Run(module.get()));
  ASSERT_TRUE(gemm_changed);

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(epilogue_changed);

  int64_t add_count = 0;
  int64_t gemm_count = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kAdd) {
      ++add_count;
    }
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "__cublas$gemm") {
      ++gemm_count;
      EXPECT_EQ(instr->operand_count(), 3);
      TF_ASSERT_OK_AND_ASSIGN(GemmBackendConfig backend_config,
                              instr->backend_config<GemmBackendConfig>());
      EXPECT_EQ(backend_config.beta(), 1.0);
    }
  }
  EXPECT_EQ(add_count, 0);
  EXPECT_EQ(gemm_count, 1);
}

TEST_F(MusaGemmEpilogueFusionTest, SkipsTemporaryBiasBecauseLegacyBetaIsSlow) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias = f32[8,4]{1,0} parameter(2)
  bias.copy = f32[8,4]{1,0} copy(bias)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias.copy)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_cublaslt(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  se::GpuComputeCapability gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  TF_ASSERT_OK_AND_ASSIGN(bool gemm_changed,
                          GemmRewriter(gpu_version).Run(module.get()));
  ASSERT_TRUE(gemm_changed);

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(epilogue_changed);
}

TEST_F(MusaGemmEpilogueFusionTest, FusesLegacyMultiOperandGemmPlusAliasedMatrixAdd) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "1");

  const std::string hlo_text = R"(
HloModule t, input_output_alias={ {}: (6, {}, must-alias) }

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  aux0 = f32[8,4]{1,0} parameter(2)
  aux1 = f32[8,4]{1,0} parameter(3)
  aux2 = f32[8,4]{1,0} parameter(4)
  aux3 = f32[8,4]{1,0} parameter(5)
  bias = f32[8,4]{1,0} parameter(6)
  gemm = f32[8,4]{1,0} custom-call(lhs, rhs, aux0, aux1, aux2, aux3), custom_call_target="__cublas$gemm", backend_config={
    "alpha_real":1,
    "alpha_imag":0,
    "beta":0,
    "dot_dimension_numbers":{
      "lhs_contracting_dimensions":["1"],
      "rhs_contracting_dimensions":["0"]
    }
  }
  ROOT out = f32[8,4]{1,0} add(gemm, bias)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(epilogue_changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), "__cublas$gemm");
  EXPECT_EQ(root->operand_count(), 7);
  EXPECT_EQ(root->operand(2)->name(), "bias");
  TF_ASSERT_OK_AND_ASSIGN(GemmBackendConfig backend_config,
                          root->backend_config<GemmBackendConfig>());
  EXPECT_EQ(backend_config.beta(), 1.0);
}

TEST_F(MusaGemmEpilogueFusionTest, SkipsBroadcastBiasByDefaultBecauseLegacyBetaIsSlow) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias_scalar = f32[] parameter(2)
  bias = f32[8,4]{1,0} broadcast(bias_scalar), dimensions={}
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_cublaslt(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  se::GpuComputeCapability gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  {
    ScopedEnvVar broadcast_bias_off("MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX",
                                    "0");
    TF_ASSERT_OK_AND_ASSIGN(bool gemm_changed,
                            GemmRewriter(gpu_version).Run(module.get()));
    ASSERT_TRUE(gemm_changed);
  }

  ScopedEnvVar broadcast_bias("MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS",
                              "1");

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(epilogue_changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kBroadcast);
}

TEST_F(MusaGemmEpilogueFusionTest,
       RewritesBroadcastBiasToMusaGemmEpilogueCustomCallWhenEnabled) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "1");
  ScopedEnvVar broadcast_bias("MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS",
                              "1");
  ScopedEnvVar custom_call("MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL", "1");
  ScopedEnvVar runtime("MUSA_XLA_GPU_RUNTIME", "classic_thunks");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias_vector = f32[4]{0} parameter(2)
  bias = f32[8,4]{1,0} broadcast(bias_vector), dimensions={1}
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_cublaslt(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  se::GpuComputeCapability gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  {
    ScopedEnvVar broadcast_bias_off("MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX",
                                    "0");
    TF_ASSERT_OK_AND_ASSIGN(bool gemm_changed,
                            GemmRewriter(gpu_version).Run(module.get()));
    ASSERT_TRUE(gemm_changed);
  }

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(epilogue_changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), kMusaGemmEpilogueCustomCallTarget);
  ASSERT_EQ(root->operand_count(), 3);
  EXPECT_EQ(root->operand(0)->name(), "lhs");
  EXPECT_EQ(root->operand(1)->name(), "rhs");
  EXPECT_EQ(root->operand(2)->name(), "bias_vector");
  TF_ASSERT_OK_AND_ASSIGN(GemmBackendConfig backend_config,
                          root->backend_config<GemmBackendConfig>());
  EXPECT_EQ(backend_config.beta(), 0.0);
  EXPECT_EQ(backend_config.epilogue(), GemmBackendConfig::BIAS);
}

TEST_F(MusaGemmEpilogueFusionTest,
       SkipsMusaGemmEpilogueCustomCallOutsideOnlyShapesFilter) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "1");
  ScopedEnvVar broadcast_bias("MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS",
                              "1");
  ScopedEnvVar custom_call("MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL", "1");
  ScopedEnvVar runtime("MUSA_XLA_GPU_RUNTIME", "classic_thunks");
  ScopedEnvVar only_shapes("MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES", "8x8x2");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias_vector = f32[4]{0} parameter(2)
  bias = f32[8,4]{1,0} broadcast(bias_vector), dimensions={1}
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_cublaslt(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  se::GpuComputeCapability gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  {
    ScopedEnvVar broadcast_bias_off("MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX",
                                    "0");
    TF_ASSERT_OK_AND_ASSIGN(bool gemm_changed,
                            GemmRewriter(gpu_version).Run(module.get()));
    ASSERT_TRUE(gemm_changed);
  }

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(epilogue_changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kAdd);
}

TEST_F(MusaGemmEpilogueFusionTest,
       RewritesMusaGemmEpilogueCustomCallInsideOnlyShapesFilter) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "1");
  ScopedEnvVar broadcast_bias("MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS",
                              "1");
  ScopedEnvVar custom_call("MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL", "1");
  ScopedEnvVar runtime("MUSA_XLA_GPU_RUNTIME", "classic_thunks");
  ScopedEnvVar only_shapes("MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES", "8x4x2");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias_vector = f32[4]{0} parameter(2)
  bias = f32[8,4]{1,0} broadcast(bias_vector), dimensions={1}
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_cublaslt(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  se::GpuComputeCapability gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  {
    ScopedEnvVar broadcast_bias_off("MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX",
                                    "0");
    TF_ASSERT_OK_AND_ASSIGN(bool gemm_changed,
                            GemmRewriter(gpu_version).Run(module.get()));
    ASSERT_TRUE(gemm_changed);
  }

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(epilogue_changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), kMusaGemmEpilogueCustomCallTarget);
}

TEST_F(MusaGemmEpilogueFusionTest,
       SkipsMusaGemmEpilogueCustomCallOutsideClassicThunks) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "1");
  ScopedEnvVar broadcast_bias("MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS",
                              "1");
  ScopedEnvVar custom_call("MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL", "1");
  ScopedEnvVar runtime("MUSA_XLA_GPU_RUNTIME", "xla_runtime");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias_vector = f32[4]{0} parameter(2)
  bias = f32[8,4]{1,0} broadcast(bias_vector), dimensions={1}
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_cublaslt(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  se::GpuComputeCapability gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  {
    ScopedEnvVar broadcast_bias_off("MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX",
                                    "0");
    TF_ASSERT_OK_AND_ASSIGN(bool gemm_changed,
                            GemmRewriter(gpu_version).Run(module.get()));
    ASSERT_TRUE(gemm_changed);
  }

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(epilogue_changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kAdd);
}

TEST_F(MusaGemmEpilogueFusionTest, DisabledByDefault) {
  ScopedEnvVar disabled("MUSA_XLA_GEMM_EPILOGUE_FUSION", "0");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias = f32[8,4]{1,0} parameter(2)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_enable_cublaslt(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  se::GpuComputeCapability gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  TF_ASSERT_OK_AND_ASSIGN(bool gemm_changed,
                          GemmRewriter(gpu_version).Run(module.get()));
  ASSERT_TRUE(gemm_changed);

  TF_ASSERT_OK_AND_ASSIGN(bool epilogue_changed,
                          MusaGemmEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(epilogue_changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
