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

#include "xla/service/gpu/musa_grouped_gemm_thunk.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gemm_thunk.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

GemmConfig MakeF32GemmConfig(int64_t n = 256) {
  MatrixLayout lhs{F32, 1024, 8, MatrixLayout::Order::kRowMajor, 8, 1, 0};
  MatrixLayout rhs{F32, 8, n, MatrixLayout::Order::kRowMajor, n, 1, 0};
  MatrixLayout out{F32, 1024, n, MatrixLayout::Order::kRowMajor, n, 1, 0};
  return GemmConfig{lhs, rhs, out, out, complex128(1.0, 0.0), 0.0,
                    std::nullopt, 0};
}

std::unique_ptr<GemmThunk> MakeGemmThunk(
    const GemmConfig& config, const BufferAllocation::Slice& lhs,
    const BufferAllocation::Slice& rhs, const BufferAllocation::Slice& out) {
  return std::make_unique<GemmThunk>(Thunk::ThunkInfo(nullptr), config, lhs,
                                    rhs, out, /*deterministic=*/false);
}

TEST(MusaGroupedGemmThunkTest, PlansConsecutiveGroupableRuns) {
  const bool groupable[] = {false, true, true, true, true,
                            true,  false, true, true};
  std::vector<MusaGroupedGemmRun> runs =
      PlanMusaGroupedGemmRuns(groupable, /*min_group_size=*/2,
                              /*max_group_size=*/3);

  ASSERT_EQ(runs.size(), 3);
  EXPECT_EQ(runs[0].start, 1);
  EXPECT_EQ(runs[0].size, 3);
  EXPECT_EQ(runs[1].start, 4);
  EXPECT_EQ(runs[1].size, 2);
  EXPECT_EQ(runs[2].start, 7);
  EXPECT_EQ(runs[2].size, 2);
}

TEST(MusaGroupedGemmThunkTest, RewritesStridedConsecutiveGemms) {
  GemmConfig config = MakeF32GemmConfig();
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes * 4, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  for (int64_t i = 0; i < 4; ++i) {
    sequence->push_back(MakeGemmThunk(
        config, BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, 0, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, i * out_bytes, out_bytes)));
  }

  MusaGroupedGemmRewriteOptions options;
  options.min_group_size = 4;
  options.max_group_size = 16;
  MusaGroupedGemmRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaGroupGemmThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.groups_created, 1);
  EXPECT_EQ(stats.strided_batched_groups, 1);
  EXPECT_EQ(stats.gemms_grouped, 4);
  ASSERT_EQ(rewritten->size(), 1);
  EXPECT_EQ((*rewritten)[0]->kind(), Thunk::Kind::kGemm);
}

TEST(MusaGroupedGemmThunkTest, RewritesIndependentNonAdjacentCompatibleGemms) {
  GemmConfig config = MakeF32GemmConfig();
  GemmConfig other_config = MakeF32GemmConfig(/*n=*/128);
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  const int64_t other_rhs_bytes = 8 * 128 * 4;
  const int64_t other_out_bytes = 1024 * 128 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes * 4, /*color=*/0);
  BufferAllocation other_lhs_alloc(3, lhs_bytes, /*color=*/0);
  BufferAllocation other_rhs_alloc(4, other_rhs_bytes, /*color=*/0);
  BufferAllocation other_out_alloc(5, other_out_bytes, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  sequence->push_back(MakeGemmThunk(
      config, BufferAllocation::Slice(&lhs_alloc, 0, lhs_bytes),
      BufferAllocation::Slice(&rhs_alloc, 0, rhs_bytes),
      BufferAllocation::Slice(&out_alloc, 0, out_bytes)));
  sequence->push_back(MakeGemmThunk(
      other_config, BufferAllocation::Slice(&other_lhs_alloc, 0, lhs_bytes),
      BufferAllocation::Slice(&other_rhs_alloc, 0, other_rhs_bytes),
      BufferAllocation::Slice(&other_out_alloc, 0, other_out_bytes)));
  for (int64_t i = 1; i < 4; ++i) {
    sequence->push_back(MakeGemmThunk(
        config, BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, 0, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, i * out_bytes, out_bytes)));
  }

  MusaGroupedGemmRewriteOptions options;
  options.min_group_size = 4;
  options.max_group_size = 16;
  MusaGroupedGemmRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaGroupGemmThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.groups_created, 1);
  EXPECT_EQ(stats.strided_batched_groups, 1);
  EXPECT_EQ(stats.noncontiguous_groups, 1);
  EXPECT_EQ(stats.gemms_grouped, 4);
  ASSERT_EQ(rewritten->size(), 2);
  EXPECT_EQ((*rewritten)[0]->kind(), Thunk::Kind::kGemm);
  EXPECT_EQ((*rewritten)[1]->kind(), Thunk::Kind::kGemm);
}

TEST(MusaGroupedGemmThunkTest, KeepsDependentGemmsSeparate) {
  GemmConfig config = MakeF32GemmConfig();
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes * 4, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  for (int64_t i = 0; i < 4; ++i) {
    BufferAllocation::Slice lhs =
        i == 1 ? BufferAllocation::Slice(&out_alloc, 0, lhs_bytes)
               : BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes);
    sequence->push_back(MakeGemmThunk(
        config, lhs,
        BufferAllocation::Slice(&rhs_alloc, 0, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, i * out_bytes, out_bytes)));
  }

  MusaGroupedGemmRewriteOptions options;
  options.min_group_size = 4;
  options.max_group_size = 16;
  MusaGroupedGemmRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaGroupGemmThunks(std::move(sequence), options, &stats));

  EXPECT_GT(stats.filtered_dependency, 0);
  EXPECT_GT(stats.filtered_dependency_output_lhs, 0);
  EXPECT_EQ(rewritten->size(), 4);
}

TEST(MusaGroupedGemmThunkTest, ReportsOutputStrideFilter) {
  GemmConfig config = MakeF32GemmConfig();
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes * 8, /*color=*/0);

  const int64_t out_offsets[] = {0, out_bytes, out_bytes * 3, out_bytes * 7};
  auto sequence = std::make_unique<ThunkSequence>();
  for (int64_t i = 0; i < 4; ++i) {
    sequence->push_back(MakeGemmThunk(
        config, BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, 0, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, out_offsets[i], out_bytes)));
  }

  MusaGroupedGemmRewriteOptions options;
  options.min_group_size = 4;
  options.max_group_size = 16;
  MusaGroupedGemmRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaGroupGemmThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.groups_created, 0);
  EXPECT_GT(stats.filtered_not_strided, 0);
  EXPECT_GT(stats.filtered_not_strided_output, 0);
  EXPECT_EQ(rewritten->size(), 4);
}

TEST(MusaGroupedGemmThunkTest, AllowsOutputOverlapWhenWriteOrderIsPreserved) {
  GemmConfig config = MakeF32GemmConfig();
  GemmConfig other_config = MakeF32GemmConfig(/*n=*/128);
  const int64_t lhs_bytes = 1024 * 8 * 4;
  const int64_t rhs_bytes = 8 * 256 * 4;
  const int64_t out_bytes = 1024 * 256 * 4;
  const int64_t other_rhs_bytes = 8 * 128 * 4;
  const int64_t other_out_bytes = 1024 * 128 * 4;
  BufferAllocation lhs_alloc(0, lhs_bytes * 4, /*color=*/0);
  BufferAllocation rhs_alloc(1, rhs_bytes, /*color=*/0);
  BufferAllocation out_alloc(2, out_bytes * 4, /*color=*/0);
  BufferAllocation other_lhs_alloc(3, lhs_bytes, /*color=*/0);
  BufferAllocation other_rhs_alloc(4, other_rhs_bytes, /*color=*/0);

  auto sequence = std::make_unique<ThunkSequence>();
  sequence->push_back(MakeGemmThunk(
      config, BufferAllocation::Slice(&lhs_alloc, 0, lhs_bytes),
      BufferAllocation::Slice(&rhs_alloc, 0, rhs_bytes),
      BufferAllocation::Slice(&out_alloc, 0, out_bytes)));
  sequence->push_back(MakeGemmThunk(
      other_config, BufferAllocation::Slice(&other_lhs_alloc, 0, lhs_bytes),
      BufferAllocation::Slice(&other_rhs_alloc, 0, other_rhs_bytes),
      BufferAllocation::Slice(&out_alloc, 0, other_out_bytes)));
  for (int64_t i = 1; i < 4; ++i) {
    sequence->push_back(MakeGemmThunk(
        config, BufferAllocation::Slice(&lhs_alloc, i * lhs_bytes, lhs_bytes),
        BufferAllocation::Slice(&rhs_alloc, 0, rhs_bytes),
        BufferAllocation::Slice(&out_alloc, i * out_bytes, out_bytes)));
  }

  MusaGroupedGemmRewriteOptions options;
  options.min_group_size = 4;
  options.max_group_size = 16;
  MusaGroupedGemmRewriteStats stats;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<const ThunkSequence> rewritten,
      RewriteMusaGroupGemmThunks(std::move(sequence), options, &stats));

  EXPECT_EQ(stats.groups_created, 1);
  EXPECT_EQ(stats.noncontiguous_groups, 1);
  EXPECT_EQ(stats.gemms_grouped, 4);
  EXPECT_EQ(rewritten->size(), 2);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
