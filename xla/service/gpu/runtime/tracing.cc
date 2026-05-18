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

#include "xla/service/gpu/runtime/tracing.h"

#include <cstdlib>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/tracing.h"
#include "xla/service/gpu/runtime/support.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/scoped_annotation_stack.h"

namespace xla {
namespace gpu {

using ::xla::runtime::CustomCall;
using ::xla::runtime::HloTrace;

using ::tsl::profiler::ScopedAnnotationStack;

namespace {

bool GetDebugEnv(absl::string_view name) {
  const char* value = std::getenv(name.data());
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

}  // namespace

//===----------------------------------------------------------------------===//
// Type names for encoded attributes.
//===----------------------------------------------------------------------===//

void RegisterTracingTypeIdNames(runtime::TypeIDNameRegistry& registry) {
  runtime::PopulateTraceTypeIdNames(registry);
}

//===----------------------------------------------------------------------===//
// Tracing custom calls implementation.
//===----------------------------------------------------------------------===//

bool IsMusaDebugRuntimeTraceEnabled() {
  return GetDebugEnv("TF_MUSA_DEBUG_RUNTIME_TRACE") ||
         GetDebugEnv("TF_MUSA_DEBUG_DEALLOC");
}

std::string DebugString(const void* ptr) { return absl::StrFormat("%p", ptr); }

std::string DebugString(const runtime::StridedMemrefView& memref) {
  std::string out = absl::StrCat("ptr=", DebugString(memref.data), " sizes=[");
  for (size_t i = 0; i < memref.sizes.size(); ++i) {
    if (i > 0) absl::StrAppend(&out, ",");
    absl::StrAppend(&out, memref.sizes[i]);
  }
  absl::StrAppend(&out, "] strides=[");
  for (size_t i = 0; i < memref.strides.size(); ++i) {
    if (i > 0) absl::StrAppend(&out, ",");
    absl::StrAppend(&out, memref.strides[i]);
  }
  absl::StrAppend(&out, "]");
  return out;
}

std::string DebugString(absl::Span<const int64_t> values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) absl::StrAppend(&out, ",");
    absl::StrAppend(&out, values[i]);
  }
  absl::StrAppend(&out, "]");
  return out;
}

void LogMusaRuntimeCustomCallBegin(absl::string_view target,
                                   absl::string_view details) {
  if (!IsMusaDebugRuntimeTraceEnabled()) return;
  if (details.empty()) {
    LOG(INFO) << "[MUSA_RUNTIME_TRACE] custom_call_begin target=" << target;
    return;
  }
  LOG(INFO) << "[MUSA_RUNTIME_TRACE] custom_call_begin target=" << target
            << " " << details;
}

void LogMusaRuntimeCustomCallEnd(absl::string_view target,
                                 const absl::Status& status) {
  if (!IsMusaDebugRuntimeTraceEnabled()) return;
  LOG(INFO) << "[MUSA_RUNTIME_TRACE] custom_call_end target=" << target
            << " status=" << status;
}

static absl::StatusOr<int64_t> ActivityStart(runtime::HloTrace annotation) {
  SetCurrentTracingScope(annotation.hlo_op);
  absl::StatusOr<int64_t> activity_id = ScopedAnnotationStack::ActivityStart([&] {
    // We use the same tracing annotation scheme as the ThunkSequence (see
    // implementation of `GetThunkInfo` in `ir_emitter_unnested.cc`).
    return absl::StrFormat("Thunk:#hlo_op=%s#", annotation.hlo_op);
  });
  if (IsMusaDebugRuntimeTraceEnabled()) {
    if (activity_id.ok()) {
      LOG(INFO) << "[MUSA_RUNTIME_TRACE] activity_start id="
                << *activity_id << " hlo_op=" << annotation.hlo_op;
    } else {
      LOG(ERROR) << "[MUSA_RUNTIME_TRACE] activity_start failed hlo_op="
                 << annotation.hlo_op << " status="
                 << activity_id.status();
    }
  }
  return activity_id;
}

static absl::Status ActivityEnd(int64_t activity_id) {
  if (IsMusaDebugRuntimeTraceEnabled()) {
    LOG(INFO) << "[MUSA_RUNTIME_TRACE] activity_end id=" << activity_id;
  }
  ResetCurrentTracingScope();
  ScopedAnnotationStack::ActivityEnd(activity_id);
  return absl::OkStatus();
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(Start, FunctionWrapper<ActivityStart>(), checks,
                               CustomCall::Bind("xla.trace.activity_start")
                                   .Attr<HloTrace>("annotation")
                                   .Ret<int64_t>());

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    End, FunctionWrapper<ActivityEnd>(), checks,
    CustomCall::Bind("xla.trace.activity_end").Arg<int64_t>());

void RegisterTracingCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.trace.activity_start", Start);
  registry.Register("xla.trace.activity_end", End);
}

}  // namespace gpu
}  // namespace xla
