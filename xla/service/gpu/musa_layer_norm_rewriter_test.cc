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

#include "gtest/gtest.h"
#include "musa_fusion_custom_calls.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;
using ::tsl::testing::IsOkAndHolds;

class MusaLayerNormRewriterTest : public HloTestBase {};

TEST_F(MusaLayerNormRewriterTest, RewritesLayerNormNamedFusionToCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule layernorm_rewrite

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

layernorm_fused_computation {
  p0 = f32[2,4]{1,0} parameter(0)
  c0 = f32[] constant(0)
  inv_n = f32[] constant(0.25)
  eps = f32[] constant(0.00001)
  inv_n_vec = f32[2]{0} broadcast(inv_n), dimensions={}
  eps_vec = f32[2]{0} broadcast(eps), dimensions={}

  mean_sum = f32[2]{0} reduce(p0, c0), dimensions={1}, to_apply=add
  mean = f32[2]{0} multiply(mean_sum, inv_n_vec)
  mean_bcast = f32[2,4]{1,0} broadcast(mean), dimensions={0}
  centered = f32[2,4]{1,0} subtract(p0, mean_bcast)

  sq = f32[2,4]{1,0} multiply(centered, centered)
  var_sum = f32[2]{0} reduce(sq, c0), dimensions={1}, to_apply=add
  var = f32[2]{0} multiply(var_sum, inv_n_vec)
  var_eps = f32[2]{0} add(var, eps_vec)
  inv_std = f32[2]{0} rsqrt(var_eps)
  inv_std_bcast = f32[2,4]{1,0} broadcast(inv_std), dimensions={0}
  ROOT norm = f32[2,4]{1,0} multiply(centered, inv_std_bcast)
}

ENTRY main {
  arg0 = f32[2,4]{1,0} parameter(0)
  ROOT layernorm_fusion = f32[2,4]{1,0} fusion(arg0), kind=kLoop, calls=layernorm_fused_computation
}
)"));

  EXPECT_THAT(RunHloPass(MusaLayerNormRewriter(), module.get()),
              IsOkAndHolds(true));
  EXPECT_TRUE(Match(module->entry_computation()->root_instruction(),
                    m::CustomCall({kMusaLayerNormCustomCallTarget})));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloCustomCallInstruction* custom_call =
      DynCast<HloCustomCallInstruction>(root);
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->api_version(), API_VERSION_STATUS_RETURNING);
  EXPECT_NE(custom_call->opaque().find("shape=2,4"), std::string::npos);
  EXPECT_NE(custom_call->opaque().find("dtype=f32"), std::string::npos);
  EXPECT_NE(custom_call->opaque().find("axis=1"), std::string::npos);
  EXPECT_NE(custom_call->opaque().find("epsilon=1e-05"), std::string::npos);
  EXPECT_NE(custom_call->opaque().find("workspace=0"), std::string::npos);
}

TEST_F(MusaLayerNormRewriterTest,
       RewritesLayerNormFusionWithScalarMeanVarianceParams) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule layernorm_rewrite_scalar_params

layernorm_fused_computation {
  p_var = f32[] parameter(0)
  p_x = f32[1,4]{1,0} parameter(1)
  p_mean = f32[] parameter(2)
  eps = f32[] constant(0.00001)
  var_eps = f32[] add(p_var, eps)
  inv_std = f32[] rsqrt(var_eps)
  inv_std_bcast = f32[1,4]{1,0} broadcast(inv_std), dimensions={}
  mean_bcast = f32[1,4]{1,0} broadcast(p_mean), dimensions={}
  centered = f32[1,4]{1,0} subtract(p_x, mean_bcast)
  ROOT norm = f32[1,4]{1,0} multiply(centered, inv_std_bcast)
}

ENTRY main {
  arg_var = f32[] parameter(0)
  arg_x = f32[1,4]{1,0} parameter(1)
  arg_mean = f32[] parameter(2)
  ROOT layernorm_fusion = f32[1,4]{1,0} fusion(arg_var, arg_x, arg_mean), kind=kLoop, calls=layernorm_fused_computation
}
)"));

  EXPECT_THAT(RunHloPass(MusaLayerNormRewriter(), module.get()),
              IsOkAndHolds(true));
  EXPECT_TRUE(Match(module->entry_computation()->root_instruction(),
                    m::CustomCall({kMusaLayerNormCustomCallTarget})));
}

TEST_F(MusaLayerNormRewriterTest,
       RewritesLayerNormFusionWithScaledScalarMeanVarianceParams) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule layernorm_rewrite_scaled_scalar_params

layernorm_fused_computation {
  p_var_sum = f32[] parameter(0)
  p_x = f32[1,4]{1,0} parameter(1)
  p_mean_sum = f32[] parameter(2)
  inv_n = f32[] constant(0.25)
  eps = f32[] constant(0.00001)
  var = f32[] multiply(p_var_sum, inv_n)
  var_eps = f32[] add(var, eps)
  inv_std = f32[] rsqrt(var_eps)
  inv_std_bcast = f32[1,4]{1,0} broadcast(inv_std), dimensions={}
  mean = f32[] multiply(p_mean_sum, inv_n)
  mean_bcast = f32[1,4]{1,0} broadcast(mean), dimensions={}
  centered = f32[1,4]{1,0} subtract(p_x, mean_bcast)
  ROOT norm = f32[1,4]{1,0} multiply(centered, inv_std_bcast)
}

ENTRY main {
  arg_var_sum = f32[] parameter(0)
  arg_x = f32[1,4]{1,0} parameter(1)
  arg_mean_sum = f32[] parameter(2)
  ROOT layernorm_fusion = f32[1,4]{1,0} fusion(arg_var_sum, arg_x, arg_mean_sum), kind=kLoop, calls=layernorm_fused_computation
}
)"));

  EXPECT_THAT(RunHloPass(MusaLayerNormRewriter(), module.get()),
              IsOkAndHolds(true));
  EXPECT_TRUE(Match(module->entry_computation()->root_instruction(),
                    m::CustomCall({kMusaLayerNormCustomCallTarget})));
}

TEST_F(MusaLayerNormRewriterTest,
       RewritesLayerNormFusionWithSingleElementVectorMeanVariance) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule layernorm_rewrite_singleton_vector_params

layernorm_fused_computation {
  p_var_sum = f32[1]{0} parameter(0)
  p_x = f32[1,4]{1,0} parameter(1)
  p_mean_sum = f32[1]{0} parameter(2)
  inv_n = f32[1]{0} constant({0.25})
  eps = f32[1]{0} constant({0.00001})
  var = f32[1]{0} multiply(p_var_sum, inv_n)
  var_eps = f32[1]{0} add(var, eps)
  inv_std = f32[1]{0} rsqrt(var_eps)
  inv_std_bcast = f32[1,4]{1,0} broadcast(inv_std), dimensions={}
  mean = f32[1]{0} multiply(p_mean_sum, inv_n)
  mean_bcast = f32[1,4]{1,0} broadcast(mean), dimensions={}
  centered = f32[1,4]{1,0} subtract(p_x, mean_bcast)
  ROOT norm = f32[1,4]{1,0} multiply(centered, inv_std_bcast)
}

ENTRY main {
  arg_var_sum = f32[1]{0} parameter(0)
  arg_x = f32[1,4]{1,0} parameter(1)
  arg_mean_sum = f32[1]{0} parameter(2)
  ROOT layernorm_fusion = f32[1,4]{1,0} fusion(arg_var_sum, arg_x, arg_mean_sum), kind=kLoop, calls=layernorm_fused_computation
}
)"));

  EXPECT_THAT(RunHloPass(MusaLayerNormRewriter(), module.get()),
              IsOkAndHolds(true));
  EXPECT_TRUE(Match(module->entry_computation()->root_instruction(),
                    m::CustomCall({kMusaLayerNormCustomCallTarget})));
}

TEST_F(MusaLayerNormRewriterTest, RewritesNonFusionLayerNormCoreMultiply) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule layernorm_rewrite_nonfusion

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  x = f32[1,4]{1,0} parameter(0)
  c0 = f32[] constant(0)
  inv_n = f32[] constant(0.25)
  eps = f32[] constant(0.00001)

  mean_sum = f32[] reduce(x, c0), dimensions={0,1}, to_apply=add
  mean = f32[] multiply(mean_sum, inv_n)
  mean_bcast = f32[1,4]{1,0} broadcast(mean), dimensions={}
  centered = f32[1,4]{1,0} subtract(x, mean_bcast)

  sq = f32[1,4]{1,0} multiply(centered, centered)
  var_sum = f32[] reduce(sq, c0), dimensions={0,1}, to_apply=add
  var = f32[] multiply(var_sum, inv_n)
  var_eps = f32[] add(var, eps)
  inv_std = f32[] rsqrt(var_eps)
  inv_std_bcast = f32[1,4]{1,0} broadcast(inv_std), dimensions={}
  ROOT norm = f32[1,4]{1,0} multiply(centered, inv_std_bcast)
}
)"));

  EXPECT_THAT(RunHloPass(MusaLayerNormRewriter(), module.get()),
              IsOkAndHolds(true));
  EXPECT_TRUE(Match(module->entry_computation()->root_instruction(),
                    m::CustomCall({kMusaLayerNormCustomCallTarget})));
}

TEST_F(MusaLayerNormRewriterTest, IgnoresNonLayerNormFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule non_layernorm_rewrite

fused_computation {
  p0 = f32[8,16]{1,0} parameter(0)
  p1 = f32[8,16]{1,0} parameter(1)
  ROOT add = f32[8,16]{1,0} add(p0, p1)
}

ENTRY main {
  arg0 = f32[8,16]{1,0} parameter(0)
  arg1 = f32[8,16]{1,0} parameter(1)
  ROOT layernorm_fusion = f32[8,16]{1,0} fusion(arg0, arg1), kind=kLoop, calls=fused_computation
}
)"));

  EXPECT_THAT(RunHloPass(MusaLayerNormRewriter(), module.get()),
              IsOkAndHolds(false));
  EXPECT_TRUE(Match(module->entry_computation()->root_instruction(),
                    m::Fusion(m::Parameter(0), m::Parameter(1))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
