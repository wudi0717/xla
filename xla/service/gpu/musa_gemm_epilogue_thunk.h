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

#ifndef XLA_SERVICE_GPU_MUSA_GEMM_EPILOGUE_THUNK_H_
#define XLA_SERVICE_GPU_MUSA_GEMM_EPILOGUE_THUNK_H_

#include <memory>
#include <string>
#include <vector>

#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

class MusaGemmEpilogueThunk : public Thunk {
 public:
  MusaGemmEpilogueThunk(ThunkInfo thunk_info, GemmConfig config,
                        BufferAllocation::Slice lhs_buffer,
                        BufferAllocation::Slice rhs_buffer,
                        BufferAllocation::Slice bias_buffer,
                        BufferAllocation::Slice output_buffer);

  MusaGemmEpilogueThunk(const MusaGemmEpilogueThunk&) = delete;
  MusaGemmEpilogueThunk& operator=(const MusaGemmEpilogueThunk&) = delete;
  ~MusaGemmEpilogueThunk() override;

  Status ExecuteOnStream(const ExecuteParams& params) override;
  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  std::string ToStringExtra(int indent) const override;

 private:
  struct MublasLtState;

  const GemmConfig config_;
  const BufferAllocation::Slice lhs_buffer_;
  const BufferAllocation::Slice rhs_buffer_;
  const BufferAllocation::Slice bias_buffer_;
  const BufferAllocation::Slice output_buffer_;
  std::unique_ptr<MublasLtState> mublaslt_state_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MUSA_GEMM_EPILOGUE_THUNK_H_
