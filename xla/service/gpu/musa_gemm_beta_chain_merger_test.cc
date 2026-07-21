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

#include <cstdlib>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
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

using MusaGemmBetaChainMergerTest = HloTestBase;

TEST_F(MusaGemmBetaChainMergerTest, MergesDotAddBetaChainIntoBlockKGemm) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_BETA_CHAIN_MERGER", "1");
  ScopedEnvVar min_chain("MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH", "3");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  a0 = f32[8,2]{1,0} parameter(0)
  b0 = f32[2,4]{1,0} parameter(1)
  d0 = f32[8,4]{1,0} dot(a0, b0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  a1 = f32[8,3]{1,0} parameter(2)
  b1 = f32[3,4]{1,0} parameter(3)
  d1 = f32[8,4]{1,0} dot(a1, b1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s1 = f32[8,4]{1,0} add(d0, d1)
  a2 = f32[8,5]{1,0} parameter(4)
  b2 = f32[5,4]{1,0} parameter(5)
  d2 = f32[8,4]{1,0} dot(a2, b2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(s1, d2)
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

  TF_ASSERT_OK_AND_ASSIGN(bool chain_changed,
                          MusaGemmBetaChainMerger().Run(module.get()));
  EXPECT_TRUE(chain_changed);

  int64_t gemm_count = 0;
  int64_t concat_count = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "__cublas$gemm") {
      ++gemm_count;
      EXPECT_EQ(instr->operand_count(), 2);
      EXPECT_EQ(instr->operand(0)->shape().dimensions(1), 10);
      EXPECT_EQ(instr->operand(1)->shape().dimensions(0), 10);
    }
    if (instr->opcode() == HloOpcode::kConcatenate) {
      ++concat_count;
    }
  }
  EXPECT_EQ(gemm_count, 1);
  EXPECT_EQ(concat_count, 2);
}

TEST_F(MusaGemmBetaChainMergerTest, SplitsLongChainIntoSmallKChunks) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_BETA_CHAIN_MERGER", "1");
  ScopedEnvVar min_chain("MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH", "2");
  ScopedEnvVar max_total_k("MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K", "5");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  a0 = f32[8,2]{1,0} parameter(0)
  b0 = f32[2,4]{1,0} parameter(1)
  d0 = f32[8,4]{1,0} dot(a0, b0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  a1 = f32[8,3]{1,0} parameter(2)
  b1 = f32[3,4]{1,0} parameter(3)
  d1 = f32[8,4]{1,0} dot(a1, b1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s1 = f32[8,4]{1,0} add(d0, d1)
  a2 = f32[8,2]{1,0} parameter(4)
  b2 = f32[2,4]{1,0} parameter(5)
  d2 = f32[8,4]{1,0} dot(a2, b2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s2 = f32[8,4]{1,0} add(s1, d2)
  a3 = f32[8,3]{1,0} parameter(6)
  b3 = f32[3,4]{1,0} parameter(7)
  d3 = f32[8,4]{1,0} dot(a3, b3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(s2, d3)
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

  TF_ASSERT_OK_AND_ASSIGN(bool chain_changed,
                          MusaGemmBetaChainMerger().Run(module.get()));
  EXPECT_TRUE(chain_changed);

  int64_t gemm_count = 0;
  int64_t concat_count = 0;
  int64_t beta_gemm_count = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "__cublas$gemm") {
      ++gemm_count;
      if (instr->operand_count() == 3) {
        ++beta_gemm_count;
      }
      EXPECT_EQ(instr->operand(0)->shape().dimensions(1), 5);
      EXPECT_EQ(instr->operand(1)->shape().dimensions(0), 5);
    }
    if (instr->opcode() == HloOpcode::kConcatenate) {
      ++concat_count;
    }
  }
  EXPECT_EQ(gemm_count, 2);
  EXPECT_EQ(beta_gemm_count, 1);
  EXPECT_EQ(concat_count, 4);
}

TEST_F(MusaGemmBetaChainMergerTest, RewritesBetaChainToMusaCustomCall) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_BETA_CHAIN_MERGER", "1");
  ScopedEnvVar custom_call("MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL", "1");
  ScopedEnvVar min_chain("MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH", "3");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  a0 = f32[8,2]{1,0} parameter(0)
  b0 = f32[2,4]{1,0} parameter(1)
  d0 = f32[8,4]{1,0} dot(a0, b0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  a1 = f32[8,3]{1,0} parameter(2)
  b1 = f32[3,4]{1,0} parameter(3)
  d1 = f32[8,4]{1,0} dot(a1, b1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s1 = f32[8,4]{1,0} add(d0, d1)
  a2 = f32[8,5]{1,0} parameter(4)
  b2 = f32[5,4]{1,0} parameter(5)
  d2 = f32[8,4]{1,0} dot(a2, b2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(s1, d2)
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

  TF_ASSERT_OK_AND_ASSIGN(
      bool chain_changed,
      MusaGemmBetaChainMerger(/*allow_custom_call=*/true).Run(module.get()));
  EXPECT_TRUE(chain_changed);

  int64_t beta_chain_custom_calls = 0;
  int64_t concat_count = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "__musa$gemm_beta_chain") {
      ++beta_chain_custom_calls;
      EXPECT_EQ(instr->operand_count(), 6);
      EXPECT_EQ(Cast<HloCustomCallInstruction>(instr)->opaque(),
                "m=8;n=4;ks=2,3,5;has_beta=0");
    }
    if (instr->opcode() == HloOpcode::kConcatenate) {
      ++concat_count;
    }
  }
  EXPECT_EQ(beta_chain_custom_calls, 1);
  EXPECT_EQ(concat_count, 0);
}

TEST_F(MusaGemmBetaChainMergerTest, SkipsCustomCallWhenNotAllowed) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_BETA_CHAIN_MERGER", "1");
  ScopedEnvVar custom_call("MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL", "1");
  ScopedEnvVar min_chain("MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH", "3");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  a0 = f32[8,2]{1,0} parameter(0)
  b0 = f32[2,4]{1,0} parameter(1)
  d0 = f32[8,4]{1,0} dot(a0, b0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  a1 = f32[8,3]{1,0} parameter(2)
  b1 = f32[3,4]{1,0} parameter(3)
  d1 = f32[8,4]{1,0} dot(a1, b1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s1 = f32[8,4]{1,0} add(d0, d1)
  a2 = f32[8,5]{1,0} parameter(4)
  b2 = f32[5,4]{1,0} parameter(5)
  d2 = f32[8,4]{1,0} dot(a2, b2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(s1, d2)
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

  TF_ASSERT_OK_AND_ASSIGN(
      bool chain_changed,
      MusaGemmBetaChainMerger(/*allow_custom_call=*/false).Run(module.get()));
  EXPECT_FALSE(chain_changed);
}

TEST_F(MusaGemmBetaChainMergerTest, SplitsLongCustomCallChainIntoSmallKChunks) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_BETA_CHAIN_MERGER", "1");
  ScopedEnvVar custom_call("MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL", "1");
  ScopedEnvVar min_chain("MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH", "2");
  ScopedEnvVar max_total_k("MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K", "5");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  a0 = f32[8,2]{1,0} parameter(0)
  b0 = f32[2,4]{1,0} parameter(1)
  d0 = f32[8,4]{1,0} dot(a0, b0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  a1 = f32[8,3]{1,0} parameter(2)
  b1 = f32[3,4]{1,0} parameter(3)
  d1 = f32[8,4]{1,0} dot(a1, b1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s1 = f32[8,4]{1,0} add(d0, d1)
  a2 = f32[8,2]{1,0} parameter(4)
  b2 = f32[2,4]{1,0} parameter(5)
  d2 = f32[8,4]{1,0} dot(a2, b2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s2 = f32[8,4]{1,0} add(s1, d2)
  a3 = f32[8,3]{1,0} parameter(6)
  b3 = f32[3,4]{1,0} parameter(7)
  d3 = f32[8,4]{1,0} dot(a3, b3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(s2, d3)
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

  TF_ASSERT_OK_AND_ASSIGN(
      bool chain_changed,
      MusaGemmBetaChainMerger(/*allow_custom_call=*/true).Run(module.get()));
  EXPECT_TRUE(chain_changed);

  int64_t beta_chain_custom_calls = 0;
  int64_t concat_count = 0;
  int64_t custom_calls_with_beta = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "__musa$gemm_beta_chain") {
      ++beta_chain_custom_calls;
      if (instr->operand_count() % 2 == 1) {
        ++custom_calls_with_beta;
      }
      EXPECT_LE(instr->operand_count(), 5);
    }
    if (instr->opcode() == HloOpcode::kConcatenate) {
      ++concat_count;
    }
  }
  EXPECT_EQ(beta_chain_custom_calls, 2);
  EXPECT_EQ(custom_calls_with_beta, 1);
  EXPECT_EQ(concat_count, 0);
}

TEST_F(MusaGemmBetaChainMergerTest, LimitsCustomCallChunkRewrites) {
  ScopedEnvVar enabled("MUSA_XLA_GEMM_BETA_CHAIN_MERGER", "1");
  ScopedEnvVar custom_call("MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL", "1");
  ScopedEnvVar min_chain("MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH", "2");
  ScopedEnvVar max_total_k("MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K", "5");
  ScopedEnvVar max_chains("MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  a0 = f32[8,2]{1,0} parameter(0)
  b0 = f32[2,4]{1,0} parameter(1)
  d0 = f32[8,4]{1,0} dot(a0, b0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  a1 = f32[8,3]{1,0} parameter(2)
  b1 = f32[3,4]{1,0} parameter(3)
  d1 = f32[8,4]{1,0} dot(a1, b1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s1 = f32[8,4]{1,0} add(d0, d1)
  a2 = f32[8,2]{1,0} parameter(4)
  b2 = f32[2,4]{1,0} parameter(5)
  d2 = f32[8,4]{1,0} dot(a2, b2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  s2 = f32[8,4]{1,0} add(s1, d2)
  a3 = f32[8,3]{1,0} parameter(6)
  b3 = f32[3,4]{1,0} parameter(7)
  d3 = f32[8,4]{1,0} dot(a3, b3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(s2, d3)
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

  TF_ASSERT_OK_AND_ASSIGN(
      bool chain_changed,
      MusaGemmBetaChainMerger(/*allow_custom_call=*/true).Run(module.get()));
  EXPECT_TRUE(chain_changed);

  int64_t beta_chain_custom_calls = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "__musa$gemm_beta_chain") {
      ++beta_chain_custom_calls;
    }
  }
  EXPECT_EQ(beta_chain_custom_calls, 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
