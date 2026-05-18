/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/memcpy.h"

#include "absl/strings/str_cat.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/runtime/concurrent_region.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/gpu/runtime/tracing.h"
#include "xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::StridedMemrefView;

enum class MemcpyDirection { kD2D, kD2H, kH2D };

template <MemcpyDirection direction>
absl::Status MemcpyImpl(const ServiceExecutableRunOptions* run_options,
                        ConcurrentRegionStatus* region_status,
                        runtime::StridedMemrefView dst,
                        runtime::StridedMemrefView src, int64_t stream_id) {
  const char* target = direction == MemcpyDirection::kD2D
                           ? "xla.gpu.memcpy.d2d"
                           : (direction == MemcpyDirection::kH2D
                                  ? "xla.gpu.memcpy.h2d"
                                  : "xla.gpu.memcpy.d2h");
  se::Stream* stream = run_options->stream();
  if (stream_id != 0) {
    DCHECK(region_status->IsInConcurrentRegion());
    auto stream_or = region_status->GetStream(stream_id);
    if (!stream_or.ok()) {
      LogMusaRuntimeCustomCallEnd(target, stream_or.status());
      return stream_or.status();
    }
    stream = *stream_or;
  } else if (region_status->IsInConcurrentRegion()) {
    stream = region_status->GetNextStream();
  }

  LogMusaRuntimeCustomCallBegin(
      target, absl::StrCat("stream_attr=", stream_id,
                           " stream=", DebugString(stream),
                           " dst=", DebugString(dst),
                           " src=", DebugString(src)));

  if (dst.sizes != src.sizes) {
    absl::Status status = absl::InvalidArgumentError(
        "Source memref sizes do not match destination memref sizes");
    LogMusaRuntimeCustomCallEnd(target, status);
    return status;
  }

  if (dst.strides != src.strides) {
    absl::Status status = absl::InvalidArgumentError(
        "Source memref strides do not match destination memref strides");
    LogMusaRuntimeCustomCallEnd(target, status);
    return status;
  }

  switch (direction) {
    case MemcpyDirection::kD2D: {
      se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
      se::DeviceMemoryBase src_data = GetDeviceAddress(src);
      stream->ThenMemcpy(&dst_data, src_data, src_data.size());
    } break;
    case MemcpyDirection::kD2H: {
      se::DeviceMemoryBase src_data = GetDeviceAddress(src);
      stream->ThenMemcpy(dst.data, src_data, src_data.size());
    } break;
    case MemcpyDirection::kH2D: {
      se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
      stream->ThenMemcpy(&dst_data, src.data, dst_data.size());
    } break;
  }

  // TODO(jacksonstokes): H2D and D2H memcpy instead of blocking the execution
  // thread should return an async token that will become available when
  // transfer is completed.
  if (direction != MemcpyDirection::kD2D) {
    absl::Status status = stream->BlockHostUntilDone();
    LogMusaRuntimeCustomCallEnd(target, status);
    return status;
  }

  absl::Status status = absl::OkStatus();
  LogMusaRuntimeCustomCallEnd(target, status);
  return status;
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL_TEMPLATE(
    MemcpyDirection direction, Memcpy, FunctionWrapper<MemcpyImpl<direction>>(),
    checks,
    CustomCall::Bind("xla.gpu.memcpy")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<ConcurrentRegionStatus*>()
        .Arg<runtime::StridedMemrefView>()  // dst
        .Arg<runtime::StridedMemrefView>()  // src
        .Attr<int64_t>("stream"));

void RegisterMemcpyCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.memcpy.d2d", Memcpy<MemcpyDirection::kD2D>);
  registry.Register("xla.gpu.memcpy.h2d", Memcpy<MemcpyDirection::kH2D>);
  registry.Register("xla.gpu.memcpy.d2h", Memcpy<MemcpyDirection::kD2H>);
}

}  // namespace gpu
}  // namespace xla
