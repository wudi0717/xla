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

#include "xla/service/gpu/musa_gemm_beta_chain_thunk.h"

#include <cstddef>
#include <utility>

#include "absl/strings/str_cat.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

MusaGemmBetaChainThunk::MusaGemmBetaChainThunk(
    ThunkInfo thunk_info, std::vector<GemmConfig> configs,
    std::vector<BufferAllocation::Slice> lhs_buffers,
    std::vector<BufferAllocation::Slice> rhs_buffers,
    std::optional<BufferAllocation::Slice> beta_buffer,
    const BufferAllocation::Slice& output_buffer, bool deterministic)
    : Thunk(Kind::kGemm, thunk_info),
      configs_(std::move(configs)),
      lhs_buffers_(std::move(lhs_buffers)),
      rhs_buffers_(std::move(rhs_buffers)),
      beta_buffer_(std::move(beta_buffer)),
      output_buffer_(output_buffer),
      deterministic_(deterministic) {}

Status MusaGemmBetaChainThunk::ExecuteOnStream(const ExecuteParams& params) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  se::DeviceMemoryBase output = allocs.GetDeviceAddress(output_buffer_);
  if (beta_buffer_.has_value()) {
    se::DeviceMemoryBase beta = allocs.GetDeviceAddress(*beta_buffer_);
    params.stream->ThenMemcpy(&output, beta, output_buffer_.size());
  }
  for (size_t i = 0; i < configs_.size(); ++i) {
    Status gemm_status = RunGemm(
        configs_[i], allocs.GetDeviceAddress(lhs_buffers_[i]),
        allocs.GetDeviceAddress(rhs_buffers_[i]), output, deterministic_,
        params.stream);
    if (!gemm_status.ok()) {
      LOG(ERROR) << "[MUSA_GEMM_BETA_CHAIN_THUNK] stage=run_gemm_failed index="
                 << i << " status=" << gemm_status;
    }
    TF_RETURN_IF_ERROR(gemm_status);
  }
  return OkStatus();
}

Status MusaGemmBetaChainThunk::Initialize(const GpuExecutable& /*executable*/,
                                          se::StreamExecutor* executor) {
  if (!executor->AsBlas()) {
    LOG(ERROR) << "[MUSA_GEMM_BETA_CHAIN_THUNK] stage=initialize_failed "
               << "reason=no_blas";
    return absl::InternalError("Failed to initialize BLAS support");
  }
  return OkStatus();
}

std::string MusaGemmBetaChainThunk::ToStringExtra(int /*indent*/) const {
  return absl::StrCat("gemm_count=", configs_.size(), ", has_beta=",
                      beta_buffer_.has_value(), ", output_bytes=",
                      output_buffer_.size());
}

}  // namespace gpu
}  // namespace xla
