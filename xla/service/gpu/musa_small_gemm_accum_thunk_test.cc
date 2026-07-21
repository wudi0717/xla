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

#include "xla/service/gpu/musa_small_gemm_accum_thunk.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gemm_thunk.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

GemmConfig MakeF32GemmConfig(double beta = 0.0, int64_t k = 8,
                             int64_t n = 256) {
  MatrixLayout lhs{F32, 1024, k, MatrixLayout::Order::kRowMajor, k, 1, 0};
  MatrixLayout rhs{F32, k, n, MatrixLayout::Order::kRowMajor, n, 1, 0};
  MatrixLayout out{F32, 1024, n, MatrixLayout::Order::kRowMajor, n, 1, 0};
  return GemmConfig{lhs, rhs, out, out, complex128(1.0, 0.0), beta,
                    std::nullopt, 0};
}

std::unique_ptr<GemmThunk> MakeGemmThunk(
    const GemmConfig& config, const BufferAllocation::Slice& lhs,
    const BufferAllocation::Slice& rhs, const BufferAllocation::Slice& out) {
  return std::make_unique<GemmThunk>(Thunk::ThunkInfo(nullptr), config, lhs,
                                    rhs, out, /*deterministic=*/false);
}

TEST(MusaSmallGemmAccumThunkTest, RewritesSameOutputBetaChain) {
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes * 4, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  for (int64_t i = 0; i < 4; ++i) {
    sequence->push_back(MakeGemmThunk(
        MakeF32GemmConfig(i == 0 ? 0.0 : 1.0),
        BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, i * rhs_bytes, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, 0, out_bytes)));
  }

  MusaSmallGemmAccumRewriteOptions options;
  options.min_chain_size = 4;
  options.max_chain_size = 16;
  options.max_k = 16;
  MusaSmallGemmAccumRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaSmallGemmAccumThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.chains_created, 1);
  EXPECT_EQ(stats.gemms_accumulated, 4);
  EXPECT_EQ(stats.estimated_launch_reduction, 3);
  ASSERT_EQ(rewritten->size(), 1);
  EXPECT_EQ((*rewritten)[0]->kind(), Thunk::Kind::kGemm);
}

TEST(MusaSmallGemmAccumThunkTest, SplitsLongSameOutputBetaChain) {
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 10, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes * 10, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  for (int64_t i = 0; i < 10; ++i) {
    sequence->push_back(MakeGemmThunk(
        MakeF32GemmConfig(i == 0 ? 0.0 : 1.0),
        BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, i * rhs_bytes, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, 0, out_bytes)));
  }

  MusaSmallGemmAccumRewriteOptions options;
  options.min_chain_size = 2;
  options.max_chain_size = 4;
  options.max_k = 16;
  MusaSmallGemmAccumRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaSmallGemmAccumThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.chains_created, 3);
  EXPECT_EQ(stats.gemms_accumulated, 10);
  EXPECT_EQ(stats.estimated_launch_reduction, 7);
  EXPECT_EQ(rewritten->size(), 3);
}

TEST(MusaSmallGemmAccumThunkTest, KeepsDifferentOutputsForGroupedGemmPath) {
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes * 4, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes * 4, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  for (int64_t i = 0; i < 4; ++i) {
    sequence->push_back(MakeGemmThunk(
        MakeF32GemmConfig(/*beta=*/0.0),
        BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, i * rhs_bytes, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, i * out_bytes, out_bytes)));
  }

  MusaSmallGemmAccumRewriteOptions options;
  options.min_chain_size = 4;
  options.max_chain_size = 16;
  options.max_k = 16;
  MusaSmallGemmAccumRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaSmallGemmAccumThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.chains_created, 0);
  EXPECT_EQ(stats.gemms_accumulated, 0);
  EXPECT_EQ(rewritten->size(), 4);
}

TEST(MusaSmallGemmAccumThunkTest, RejectsWrongBetaPattern) {
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes * 4, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  for (int64_t i = 0; i < 4; ++i) {
    sequence->push_back(MakeGemmThunk(
        MakeF32GemmConfig(/*beta=*/0.0),
        BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, i * rhs_bytes, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, 0, out_bytes)));
  }

  MusaSmallGemmAccumRewriteOptions options;
  options.min_chain_size = 4;
  options.max_chain_size = 16;
  options.max_k = 16;
  MusaSmallGemmAccumRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaSmallGemmAccumThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.chains_created, 0);
  EXPECT_GT(stats.filtered_beta, 0);
  EXPECT_EQ(rewritten->size(), 4);
}

TEST(MusaSmallGemmAccumThunkTest, RequireCustomKernelSkipsFallbackRewrite) {
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes * 4, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  for (int64_t i = 0; i < 4; ++i) {
    sequence->push_back(MakeGemmThunk(
        MakeF32GemmConfig(i == 0 ? 0.0 : 1.0),
        BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, i * rhs_bytes, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, 0, out_bytes)));
  }

  MusaSmallGemmAccumRewriteOptions options;
  options.min_chain_size = 4;
  options.max_chain_size = 16;
  options.max_k = 16;
  options.require_custom_kernel = true;
  MusaSmallGemmAccumRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaSmallGemmAccumThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.chains_created, 0);
  EXPECT_EQ(stats.gemms_accumulated, 0);
  EXPECT_EQ(stats.filtered_custom_kernel_unavailable, 1);
  EXPECT_EQ(rewritten->size(), 4);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
