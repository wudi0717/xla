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

#ifndef XLA_SERVICE_GPU_MUSA_SMALL_GEMM_ACCUM_THUNK_H_
#define XLA_SERVICE_GPU_MUSA_SMALL_GEMM_ACCUM_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

struct MusaSmallGemmAccumRewriteOptions {
  int64_t min_chain_size = 4;
  int64_t max_chain_size = 64;
  int64_t max_k = 64;
  bool require_custom_kernel = false;
  bool log = false;
};

struct MusaSmallGemmAccumRewriteStats {
  int64_t original_thunks = 0;
  int64_t rewritten_thunks = 0;
  int64_t gemm_thunks = 0;
  int64_t candidate_runs = 0;
  int64_t candidate_gemms = 0;
  int64_t chains_created = 0;
  int64_t gemms_accumulated = 0;
  int64_t estimated_launch_reduction = 0;
  int64_t filtered_short_chain = 0;
  int64_t filtered_beta = 0;
  int64_t filtered_config = 0;
  int64_t filtered_layout = 0;
  int64_t filtered_size = 0;
  int64_t filtered_dependency = 0;
  int64_t filtered_custom_kernel_unavailable = 0;
};

StatusOr<std::unique_ptr<const ThunkSequence>> RewriteMusaSmallGemmAccumThunks(
    std::unique_ptr<const ThunkSequence> sequence,
    const MusaSmallGemmAccumRewriteOptions& options,
    MusaSmallGemmAccumRewriteStats* stats);

class MusaSmallGemmAccumThunk : public Thunk {
 public:
  MusaSmallGemmAccumThunk(
      ThunkInfo thunk_info, std::vector<GemmConfig> configs,
      std::vector<BufferAllocation::Slice> lhs_buffers,
      std::vector<BufferAllocation::Slice> rhs_buffers,
      const BufferAllocation::Slice& output_buffer, bool deterministic,
      bool starts_from_existing_output);

  MusaSmallGemmAccumThunk(const MusaSmallGemmAccumThunk&) = delete;
  MusaSmallGemmAccumThunk& operator=(const MusaSmallGemmAccumThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;
  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  std::string ToStringExtra(int indent) const override;

 private:
  const std::vector<GemmConfig> configs_;
  const std::vector<BufferAllocation::Slice> lhs_buffers_;
  const std::vector<BufferAllocation::Slice> rhs_buffers_;
  const BufferAllocation::Slice output_buffer_;
  const bool deterministic_;
  const bool starts_from_existing_output_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MUSA_SMALL_GEMM_ACCUM_THUNK_H_
