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

#ifndef XLA_SERVICE_GPU_MUSA_GEMM_BETA_CHAIN_THUNK_H_
#define XLA_SERVICE_GPU_MUSA_GEMM_BETA_CHAIN_THUNK_H_

#include <optional>
#include <string>
#include <vector>

#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

class MusaGemmBetaChainThunk : public Thunk {
 public:
  MusaGemmBetaChainThunk(
      ThunkInfo thunk_info, std::vector<GemmConfig> configs,
      std::vector<BufferAllocation::Slice> lhs_buffers,
      std::vector<BufferAllocation::Slice> rhs_buffers,
      std::optional<BufferAllocation::Slice> beta_buffer,
      const BufferAllocation::Slice& output_buffer, bool deterministic);

  MusaGemmBetaChainThunk(const MusaGemmBetaChainThunk&) = delete;
  MusaGemmBetaChainThunk& operator=(const MusaGemmBetaChainThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;
  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  std::string ToStringExtra(int indent) const override;

 private:
  const std::vector<GemmConfig> configs_;
  const std::vector<BufferAllocation::Slice> lhs_buffers_;
  const std::vector<BufferAllocation::Slice> rhs_buffers_;
  const std::optional<BufferAllocation::Slice> beta_buffer_;
  const BufferAllocation::Slice output_buffer_;
  const bool deterministic_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MUSA_GEMM_BETA_CHAIN_THUNK_H_
