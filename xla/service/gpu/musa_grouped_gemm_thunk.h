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

#ifndef XLA_SERVICE_GPU_MUSA_GROUPED_GEMM_THUNK_H_
#define XLA_SERVICE_GPU_MUSA_GROUPED_GEMM_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

struct MusaGroupedGemmRewriteOptions {
  int64_t min_group_size = 4;
  int64_t max_group_size = 64;
  bool diagnostic_only = false;
  bool log = false;
};

struct MusaGroupedGemmRewriteStats {
  int64_t original_thunks = 0;
  int64_t rewritten_thunks = 0;
  int64_t gemm_thunks = 0;
  int64_t plain_gemm_thunks = 0;
  int64_t candidate_runs = 0;
  int64_t candidate_gemms = 0;
  int64_t groups_created = 0;
  int64_t gemms_grouped = 0;
  int64_t strided_batched_groups = 0;
  int64_t strided_batched_gemms = 0;
  int64_t noncontiguous_groups = 0;
  int64_t noncontiguous_gemms = 0;
  int64_t cross_kernel_candidate_windows = 0;
  int64_t cross_kernel_candidate_gemms = 0;
  int64_t cross_kernel_groupable_windows = 0;
  int64_t cross_kernel_groupable_gemms = 0;
  int64_t cross_kernel_blocked_by_dependency = 0;
  int64_t cross_kernel_blocked_by_config = 0;
  int64_t cross_kernel_blocked_by_stride = 0;
  int64_t filtered_not_strided = 0;
  int64_t filtered_not_strided_lhs = 0;
  int64_t filtered_not_strided_rhs = 0;
  int64_t filtered_not_strided_output = 0;
  int64_t filtered_dependency = 0;
  int64_t filtered_dependency_output_output = 0;
  int64_t filtered_dependency_output_lhs = 0;
  int64_t filtered_dependency_output_rhs = 0;
  int64_t filtered_config = 0;
};

struct MusaGroupedGemmRun {
  int64_t start = 0;
  int64_t size = 0;
};

std::vector<MusaGroupedGemmRun> PlanMusaGroupedGemmRuns(
    absl::Span<const bool> groupable, int64_t min_group_size,
    int64_t max_group_size);

StatusOr<std::unique_ptr<const ThunkSequence>> RewriteMusaGroupGemmThunks(
    std::unique_ptr<const ThunkSequence> sequence,
    const MusaGroupedGemmRewriteOptions& options,
    MusaGroupedGemmRewriteStats* stats);

class MusaGroupedGemmThunk : public Thunk {
 public:
  MusaGroupedGemmThunk(ThunkInfo thunk_info, GemmConfig batched_config,
                       const BufferAllocation::Slice& lhs_buffer,
                       const BufferAllocation::Slice& rhs_buffer,
                       const BufferAllocation::Slice& output_buffer,
                       int64_t gemm_count, bool deterministic);

  MusaGroupedGemmThunk(const MusaGroupedGemmThunk&) = delete;
  MusaGroupedGemmThunk& operator=(const MusaGroupedGemmThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;
  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  std::string ToStringExtra(int indent) const override;

 private:
  const GemmConfig batched_config_;
  const BufferAllocation::Slice lhs_buffer_;
  const BufferAllocation::Slice rhs_buffer_;
  const BufferAllocation::Slice output_buffer_;
  const int64_t gemm_count_;
  const bool deterministic_;
};

class MusaPointerArrayGemmThunk : public Thunk {
 public:
  MusaPointerArrayGemmThunk(
      ThunkInfo thunk_info, GemmConfig config,
      std::vector<BufferAllocation::Slice> lhs_buffers,
      std::vector<BufferAllocation::Slice> rhs_buffers,
      std::vector<BufferAllocation::Slice> output_buffers,
      bool deterministic);

  MusaPointerArrayGemmThunk(const MusaPointerArrayGemmThunk&) = delete;
  MusaPointerArrayGemmThunk& operator=(const MusaPointerArrayGemmThunk&) =
      delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;
  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  std::string ToStringExtra(int indent) const override;

 private:
  const GemmConfig config_;
  const std::vector<BufferAllocation::Slice> lhs_buffers_;
  const std::vector<BufferAllocation::Slice> rhs_buffers_;
  const std::vector<BufferAllocation::Slice> output_buffers_;
  const bool deterministic_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MUSA_GROUPED_GEMM_THUNK_H_
