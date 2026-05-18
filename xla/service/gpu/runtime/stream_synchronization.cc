/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/stream_synchronization.h"

#include "absl/strings/str_cat.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/runtime/concurrent_region.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/gpu/runtime/tracing.h"

namespace xla {
namespace gpu {

static absl::Status AwaitImpl(ConcurrentRegionStatus* region_status,
                              int64_t from, absl::Span<const int64_t> to) {
  auto from_stream_or = region_status->GetStream(from);
  if (!from_stream_or.ok()) {
    LogMusaRuntimeCustomCallEnd("xla.streams.await", from_stream_or.status());
    return from_stream_or.status();
  }
  se::Stream* from_stream = *from_stream_or;
  std::string details = absl::StrCat(
      "from=", from, "@", DebugString(from_stream), " to=");
  absl::StrAppend(&details, DebugString(to));
  for (int64_t to_index : to) {
    auto to_stream_or = region_status->GetStream(to_index);
    if (!to_stream_or.ok()) {
      LogMusaRuntimeCustomCallEnd("xla.streams.await", to_stream_or.status());
      return to_stream_or.status();
    }
    se::Stream* to_stream = *to_stream_or;
    absl::StrAppend(&details, " wait_on=", to_index, "@",
                    DebugString(to_stream));
    from_stream->ThenWaitFor(to_stream);
  }

  LogMusaRuntimeCustomCallBegin("xla.streams.await", details);

  absl::Status status = absl::OkStatus();
  LogMusaRuntimeCustomCallEnd("xla.streams.await", status);
  return status;
}

//===----------------------------------------------------------------------===//
// Define custom calls that mark the concurrent region in CUDA graphs.
//===----------------------------------------------------------------------===//

using xla::runtime::CustomCall;

XLA_RUNTIME_DEFINE_CUSTOM_CALL(Await, FunctionWrapper<AwaitImpl>(), checks,
                               CustomCall::Bind("xla.streams.await")
                                   .UserData<ConcurrentRegionStatus*>()
                                   .Attr<int64_t>("from")
                                   .Attr<absl::Span<const int64_t>>("to"));

void RegisterStreamSynchronizationCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.streams.await", Await);
}

}  // namespace gpu
}  // namespace xla
