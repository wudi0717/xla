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

#include <cstdlib>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
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

using MusaDotEpilogueFusionTest = HloTestBase;

TEST_F(MusaDotEpilogueFusionTest, DetectsDotAddPatternWithoutRewriting) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "0");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias = f32[8,4]{1,0} parameter(2)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

TEST_F(MusaDotEpilogueFusionTest, FusesDotAddIntoCustomFusionWhenEnabled) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");
  ScopedEnvVar kind("MUSA_XLA_DOT_EPILOGUE_FUSION_KIND", "__triton_gemm");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias = f32[8,4]{1,0} parameter(2)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);
  TF_ASSERT_OK_AND_ASSIGN(FusionBackendConfig backend_config,
                          root->backend_config<FusionBackendConfig>());
  EXPECT_EQ(backend_config.kind(), "__triton_gemm");

  bool has_dot = false;
  bool has_add = false;
  for (const HloInstruction* instr :
       root->fused_instructions_computation()->instructions()) {
    has_dot |= instr->opcode() == HloOpcode::kDot;
    has_add |= instr->opcode() == HloOpcode::kAdd;
  }
  EXPECT_TRUE(has_dot);
  EXPECT_TRUE(has_add);
}

TEST_F(MusaDotEpilogueFusionTest, FusesDotAddElementwiseChain) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias = f32[8,4]{1,0} parameter(2)
  limit = f32[8,4]{1,0} parameter(3)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add.0 = f32[8,4]{1,0} add(dot.0, bias)
  ROOT out = f32[8,4]{1,0} maximum(add.0, limit)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion);

  bool has_dot = false;
  bool has_add = false;
  bool has_maximum = false;
  for (const HloInstruction* instr :
       root->fused_instructions_computation()->instructions()) {
    has_dot |= instr->opcode() == HloOpcode::kDot;
    has_add |= instr->opcode() == HloOpcode::kAdd;
    has_maximum |= instr->opcode() == HloOpcode::kMaximum;
  }
  EXPECT_TRUE(has_dot);
  EXPECT_TRUE(has_add);
  EXPECT_TRUE(has_maximum);
}

TEST_F(MusaDotEpilogueFusionTest, SkipsLongChainWhenMaxChainLengthIsOne) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");
  ScopedEnvVar max_chain("MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias = f32[8,4]{1,0} parameter(2)
  limit = f32[8,4]{1,0} parameter(3)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add.0 = f32[8,4]{1,0} add(dot.0, bias)
  ROOT out = f32[8,4]{1,0} maximum(add.0, limit)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kMaximum);
}

TEST_F(MusaDotEpilogueFusionTest, RespectsMaxFusionsPerModule) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");
  ScopedEnvVar max_fusions("MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs0 = f32[8,2]{1,0} parameter(0)
  rhs0 = f32[2,4]{1,0} parameter(1)
  bias0 = f32[8,4]{1,0} parameter(2)
  lhs1 = f32[8,2]{1,0} parameter(3)
  rhs1 = f32[2,4]{1,0} parameter(4)
  bias1 = f32[8,4]{1,0} parameter(5)
  dot.0 = f32[8,4]{1,0} dot(lhs0, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.1 = f32[8,4]{1,0} dot(lhs1, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add.0 = f32[8,4]{1,0} add(dot.0, bias0)
  add.1 = f32[8,4]{1,0} add(dot.1, bias1)
  ROOT out = (f32[8,4]{1,0}, f32[8,4]{1,0}) tuple(add.0, add.1)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(changed);

  int64_t fusion_count = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      ++fusion_count;
    }
  }
  EXPECT_EQ(fusion_count, 1);
}

TEST_F(MusaDotEpilogueFusionTest, RequireAddSkipsNonAddEpilogue) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");
  ScopedEnvVar require_add("MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  scale = f32[8,4]{1,0} parameter(2)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} multiply(dot.0, scale)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kMultiply);
}

TEST_F(MusaDotEpilogueFusionTest, MinKSkipsSmallDot) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");
  ScopedEnvVar min_k("MUSA_XLA_DOT_EPILOGUE_MIN_K", "16");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[8,2]{1,0} parameter(0)
  rhs = f32[2,4]{1,0} parameter(1)
  bias = f32[8,4]{1,0} parameter(2)
  dot.0 = f32[8,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, bias)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

TEST_F(MusaDotEpilogueFusionTest, MinMSkipsSmallBatchDot) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");
  ScopedEnvVar min_m("MUSA_XLA_DOT_EPILOGUE_MIN_M", "1024");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs = f32[32,64]{1,0} parameter(0)
  rhs = f32[64,32]{1,0} parameter(1)
  bias = f32[32,32]{1,0} parameter(2)
  dot.0 = f32[32,32]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[32,32]{1,0} add(dot.0, bias)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

TEST_F(MusaDotEpilogueFusionTest, RespectsMaxFusionsPerPattern) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");
  ScopedEnvVar max_per_pattern(
      "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs0 = f32[8,2]{1,0} parameter(0)
  rhs0 = f32[2,4]{1,0} parameter(1)
  bias0 = f32[8,4]{1,0} parameter(2)
  lhs1 = f32[8,2]{1,0} parameter(3)
  rhs1 = f32[2,4]{1,0} parameter(4)
  bias1 = f32[8,4]{1,0} parameter(5)
  dot.0 = f32[8,4]{1,0} dot(lhs0, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.1 = f32[8,4]{1,0} dot(lhs1, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add.0 = f32[8,4]{1,0} add(dot.0, bias0)
  add.1 = f32[8,4]{1,0} add(dot.1, bias1)
  ROOT out = (f32[8,4]{1,0}, f32[8,4]{1,0}) tuple(add.0, add.1)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(changed);

  int64_t fusion_count = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      ++fusion_count;
    }
  }
  EXPECT_EQ(fusion_count, 1);
}

TEST_F(MusaDotEpilogueFusionTest, SortBySizePrefersLargerDot) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");
  ScopedEnvVar max_fusions("MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE", "1");
  ScopedEnvVar sort_by_size("MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  small_lhs = f32[8,2]{1,0} parameter(0)
  small_rhs = f32[2,4]{1,0} parameter(1)
  small_bias = f32[8,4]{1,0} parameter(2)
  large_lhs = f32[8,8]{1,0} parameter(3)
  large_rhs = f32[8,16]{1,0} parameter(4)
  large_bias = f32[8,16]{1,0} parameter(5)
  dot.small = f32[8,4]{1,0} dot(small_lhs, small_rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.large = f32[8,16]{1,0} dot(large_lhs, large_rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add.small = f32[8,4]{1,0} add(dot.small, small_bias)
  add.large = f32[8,16]{1,0} add(dot.large, large_bias)
  ROOT out = (f32[8,4]{1,0}, f32[8,16]{1,0}) tuple(add.small, add.large)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_TRUE(changed);

  int64_t fusion_count = 0;
  bool fused_large_dot = false;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() != HloOpcode::kFusion) {
      continue;
    }
    ++fusion_count;
    fused_large_dot |= instr->shape().rank() == 2 &&
                       instr->shape().dimensions(0) == 8 &&
                       instr->shape().dimensions(1) == 16;
  }
  EXPECT_EQ(fusion_count, 1);
  EXPECT_TRUE(fused_large_dot);
}

TEST_F(MusaDotEpilogueFusionTest, SkipsAddOfTwoDots) {
  ScopedEnvVar pattern("MUSA_XLA_DOT_EPILOGUE_PATTERN", "1");
  ScopedEnvVar rewrite("MUSA_XLA_DOT_EPILOGUE_FUSION", "1");

  const std::string hlo_text = R"(
HloModule t

ENTRY e {
  lhs0 = f32[8,2]{1,0} parameter(0)
  rhs0 = f32[2,4]{1,0} parameter(1)
  lhs1 = f32[8,2]{1,0} parameter(2)
  rhs1 = f32[2,4]{1,0} parameter(3)
  dot.0 = f32[8,4]{1,0} dot(lhs0, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot.1 = f32[8,4]{1,0} dot(lhs1, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[8,4]{1,0} add(dot.0, dot.1)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(
                              hlo_text, GetModuleConfigForTest()));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MusaDotEpilogueFusion().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
