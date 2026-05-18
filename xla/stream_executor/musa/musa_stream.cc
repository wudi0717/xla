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

#include "xla/stream_executor/musa/musa_stream.h"

#include <chrono>
#include <cstdlib>
#include <thread>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/functional/any_invocable.h"
#include "tsl/platform/errors.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"

namespace stream_executor {
namespace musa {
namespace {

bool IsMusaDebugRuntimeTraceEnabled() {
  const char* value = std::getenv("TF_MUSA_DEBUG_RUNTIME_TRACE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

size_t GetDebugLaunchCheckpointWindow() {
  constexpr size_t kDefaultWindow = 16;
  const char* value = std::getenv("TF_MUSA_DEBUG_LAUNCH_CHECKPOINT_WINDOW");
  if (value == nullptr || value[0] == '\0') {
    return kDefaultWindow;
  }
  char* end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || parsed == 0) {
    return kDefaultWindow;
  }
  return static_cast<size_t>(parsed);
}

std::string FormatMuResult(MUresult result) {
  const char* error_name = nullptr;
  const char* error_string = nullptr;
  (void)muGetErrorName(result, &error_name);
  (void)muGetErrorString(result, &error_string);
  std::string formatted =
      error_name != nullptr ? error_name : "UNKNOWN MUSA DRIVER ERROR";
  if (error_string != nullptr && error_string[0] != '\0') {
    absl::StrAppend(&formatted, ": ", error_string);
  }
  return formatted;
}

MusaStream* AsMusaStream(Stream* stream) {
  return static_cast<MusaStream*>(stream);
}

MusaEvent* AsMusaEvent(Event* event) {
  return static_cast<MusaEvent*>(event->implementation());
}

int ResolveStreamPriority(
    std::optional<std::variant<StreamPriority, int>> priority) {
  if (!priority.has_value()) {
    return 0;
  }
  if (std::holds_alternative<int>(*priority)) {
    return std::get<int>(*priority);
  }
  int lowest = 0;
  int highest = 0;
  auto status = musa::ToStatus(muCtxGetStreamPriorityRange(&lowest, &highest));
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 0;
  }
  switch (std::get<StreamPriority>(*priority)) {
    case StreamPriority::Highest:
      return highest;
    case StreamPriority::Lowest:
      return lowest;
    case StreamPriority::Default:
      return 0;
  }
  return 0;
}

void DestroyEvent(MUevent event) {
  if (event != nullptr) {
    auto status = musa::ToStatus(muEventDestroy(event), "Failed to destroy MUSA event");
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }
}

void DestroyStream(MUstream stream) {
  if (stream != nullptr) {
    auto status = musa::ToStatus(muStreamDestroy(stream), "Failed to destroy MUSA stream");
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }
}

void InternalHostCallback(void* data) {
  auto* callback = reinterpret_cast<absl::AnyInvocable<void() &&>*>(data);
  std::move(*callback)();
  delete callback;
}

}  // namespace

tsl::StatusOr<std::unique_ptr<Stream>> MusaStream::Create(
    StreamExecutor* executor, MUcontext context,
    std::optional<std::variant<StreamPriority, int>> priority) {
  ScopedActivateContext activation(context);
  TF_RETURN_IF_ERROR(activation.status());
  MUstream stream = nullptr;
  int stream_priority = ResolveStreamPriority(priority);
  if (stream_priority == 0) {
    TF_RETURN_IF_ERROR(
        musa::ToStatus(muStreamCreate(&stream, MU_STREAM_NON_BLOCKING),
                       "Failed to create MUSA stream"));
  } else {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muStreamCreateWithPriority(&stream, MU_STREAM_NON_BLOCKING,
                                   stream_priority),
        "Failed to create MUSA stream with priority"));
  }

  MUevent completed_event = nullptr;
  auto event_status = musa::ToStatus(
      muEventCreate(&completed_event, MU_EVENT_DISABLE_TIMING),
      "Failed to create MUSA completion event");
  if (!event_status.ok()) {
    DestroyStream(stream);
    return event_status;
  }

  return std::unique_ptr<Stream>(
      new MusaStream(executor, context, priority, stream, completed_event));
}

MusaStream::MusaStream(
    StreamExecutor* executor, MUcontext context,
    std::optional<std::variant<StreamPriority, int>> priority,
    MUstream stream_handle, MUevent completed_event)
    : StreamCommon(executor, priority),
      executor_(executor),
      context_(context),
      stream_handle_(stream_handle),
      completed_event_(completed_event) {
  absl::MutexLock lock(&mu_);
  status_ = ::tsl::OkStatus();
}

MusaStream::~MusaStream() {
  BlockHostUntilDone().IgnoreError();

  ScopedActivateContext activation(context_);
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
  } else {
    ClearDebugLaunchCheckpoints();
    DestroyEvent(completed_event_);
    DestroyStream(stream_handle_);
  }

  stream_handle_ = nullptr;
  completed_event_ = nullptr;
  absl::MutexLock lock(&mu_);
  status_ = tsl::errors::Internal("MUSA stream already destroyed");
}

tsl::Status MusaStream::RecordCompletedEvent() {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  absl::MutexLock submit_lock(&submit_mu_);
  return musa::ToStatus(muEventRecord(completed_event_, stream_handle_),
                        "Failed to record MUSA completion event");
}

void MusaStream::RecordDebugLaunchCheckpoint(const std::string& kernel_name,
                                             const std::string& launch_dims,
                                             uint64_t shared_mem_bytes) {
  if (!IsMusaDebugRuntimeTraceEnabled()) {
    return;
  }
  ScopedActivateContext activation(context_);
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }

  MUevent event = nullptr;
  auto create_status = musa::ToStatus(
      muEventCreate(&event, MU_EVENT_DISABLE_TIMING),
      "Failed to create MUSA debug launch checkpoint event");
  if (!create_status.ok()) {
    LOG(ERROR) << create_status;
    return;
  }

  tsl::Status record_status;
  {
    absl::MutexLock submit_lock(&submit_mu_);
    record_status = musa::ToStatus(
        muEventRecord(event, stream_handle_),
        "Failed to record MUSA debug launch checkpoint event");
  }
  if (!record_status.ok()) {
    LOG(ERROR) << record_status;
    auto destroy_status =
        musa::ToStatus(muEventDestroy(event), "Failed to destroy MUSA event");
    if (!destroy_status.ok()) {
      LOG(ERROR) << destroy_status;
    }
    return;
  }

  const size_t max_debug_launch_checkpoints = GetDebugLaunchCheckpointWindow();
  DebugLaunchCheckpoint evicted;
  bool has_evicted = false;
  uint64_t sequence = 0;
  {
    absl::MutexLock lock(&debug_launch_checkpoints_mu_);
    sequence = ++debug_launch_sequence_;
    debug_launch_checkpoints_.push_back(
        DebugLaunchCheckpoint{sequence, kernel_name, launch_dims,
                              shared_mem_bytes, event});
    if (debug_launch_checkpoints_.size() > max_debug_launch_checkpoints) {
      evicted = debug_launch_checkpoints_.front();
      debug_launch_checkpoints_.pop_front();
      has_evicted = true;
    }
  }

  if (has_evicted && evicted.event != nullptr) {
    auto destroy_status =
        musa::ToStatus(muEventDestroy(evicted.event), "Failed to destroy MUSA event");
    if (!destroy_status.ok()) {
      LOG(ERROR) << destroy_status;
    }
  }

  LOG(INFO) << "[MUSA_RUNTIME_TRACE] launch_event_recorded seq=" << sequence
            << " kernel=" << kernel_name << " " << launch_dims
            << " shmem=" << shared_mem_bytes << " stream=" << stream_handle_
            << " event=" << event;
}

void MusaStream::LogDebugLaunchCheckpointStatuses(int64_t waited_us) {
  if (!IsMusaDebugRuntimeTraceEnabled()) {
    return;
  }
  ScopedActivateContext activation(context_);
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }

  std::deque<DebugLaunchCheckpoint> checkpoints;
  {
    absl::MutexLock lock(&debug_launch_checkpoints_mu_);
    checkpoints = debug_launch_checkpoints_;
  }
  if (checkpoints.empty()) {
    LOG(INFO) << "[MUSA_RUNTIME_TRACE] launch_event_status waited_us="
              << waited_us << " stream=" << stream_handle_
              << " checkpoints=none";
    return;
  }

  for (const DebugLaunchCheckpoint& checkpoint : checkpoints) {
    MUresult result = muEventQuery(checkpoint.event);
    const char* status = "error";
    if (result == MUSA_SUCCESS) {
      status = "complete";
    } else if (result == MUSA_ERROR_NOT_READY) {
      status = "pending";
    }
    LOG(INFO) << "[MUSA_RUNTIME_TRACE] launch_event_status waited_us="
              << waited_us << " stream=" << stream_handle_
              << " seq=" << checkpoint.sequence
              << " kernel=" << checkpoint.kernel_name << " "
              << checkpoint.launch_dims << " shmem="
              << checkpoint.shared_mem_bytes << " event=" << checkpoint.event
              << " status=" << status << " raw=" << FormatMuResult(result);
  }
}

void MusaStream::ClearDebugLaunchCheckpoints() {
  std::deque<DebugLaunchCheckpoint> checkpoints;
  {
    absl::MutexLock lock(&debug_launch_checkpoints_mu_);
    checkpoints.swap(debug_launch_checkpoints_);
  }
  for (const DebugLaunchCheckpoint& checkpoint : checkpoints) {
    if (checkpoint.event == nullptr) {
      continue;
    }
    auto destroy_status =
        musa::ToStatus(muEventDestroy(checkpoint.event), "Failed to destroy MUSA event");
    if (!destroy_status.ok()) {
      LOG(ERROR) << destroy_status;
    }
  }
}

tsl::Status MusaStream::WaitFor(Stream* other) {
  return Stream::WaitFor(other);
}

tsl::Status MusaStream::WaitFor(Event* event) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  absl::MutexLock submit_lock(&submit_mu_);
  return musa::ToStatus(
      muStreamWaitEvent(stream_handle_, AsMusaEvent(event)->handle(), 0),
      "Failed to wait on MUSA event");
}

tsl::Status MusaStream::RecordEvent(Event* event) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  absl::MutexLock submit_lock(&submit_mu_);
  return musa::ToStatus(
      muEventRecord(AsMusaEvent(event)->handle(), stream_handle_),
      "Failed to record MUSA event");
}

tsl::Status MusaStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  absl::MutexLock submit_lock(&submit_mu_);
  return musa::ToStatus(
      muMemsetD8Async(reinterpret_cast<MUdeviceptr>(location->opaque()), 0, size,
                      stream_handle_),
      "Failed to enqueue MUSA memzero");
}

tsl::Status MusaStream::Memset32(DeviceMemoryBase* location, uint32_t pattern,
                                 uint64_t size) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  if (size % sizeof(uint32_t) != 0) {
    return tsl::errors::InvalidArgument("size must be divisible by 4");
  }
  absl::MutexLock submit_lock(&submit_mu_);
  return musa::ToStatus(
      muMemsetD32Async(reinterpret_cast<MUdeviceptr>(location->opaque()),
                       pattern, size / sizeof(uint32_t), stream_handle_),
      "Failed to enqueue MUSA memset32");
}

tsl::Status MusaStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                               uint64_t size) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  absl::MutexLock submit_lock(&submit_mu_);
  return musa::ToStatus(
      muMemcpyHtoDAsync(reinterpret_cast<MUdeviceptr>(gpu_dst->opaque()),
                        host_src, size, stream_handle_),
      "Failed to enqueue MUSA memcpy H2D");
}

tsl::Status MusaStream::Memcpy(void* host_dst,
                               const DeviceMemoryBase& gpu_src, uint64_t size) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  absl::MutexLock submit_lock(&submit_mu_);
  return musa::ToStatus(
      muMemcpyDtoHAsync(host_dst,
                        reinterpret_cast<MUdeviceptr>(gpu_src.opaque()), size,
                        stream_handle_),
      "Failed to enqueue MUSA memcpy D2H");
}

tsl::Status MusaStream::Memcpy(DeviceMemoryBase* gpu_dst,
                               const DeviceMemoryBase& gpu_src,
                               uint64_t size) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  if (size == 0 || gpu_dst->opaque() == nullptr || gpu_src.opaque() == nullptr) {
    absl::MutexLock submit_lock(&submit_mu_);
    return musa::ToStatus(
        muMemcpyDtoDAsync(reinterpret_cast<MUdeviceptr>(gpu_dst->opaque()),
                          reinterpret_cast<MUdeviceptr>(gpu_src.opaque()), size,
                          stream_handle_),
        "Failed to enqueue MUSA memcpy D2D");
  }
  MUcontext dst_context = GetContextForDeviceMemory(gpu_dst->opaque());
  MUcontext src_context =
      GetContextForDeviceMemory(const_cast<void*>(gpu_src.opaque()));
  if (gpu_dst->opaque() != nullptr && gpu_src.opaque() != nullptr &&
      dst_context != nullptr && src_context != nullptr &&
      dst_context != src_context) {
    absl::MutexLock submit_lock(&submit_mu_);
    return musa::ToStatus(
        muMemcpyPeerAsync(reinterpret_cast<MUdeviceptr>(gpu_dst->opaque()),
                          dst_context,
                          reinterpret_cast<MUdeviceptr>(gpu_src.opaque()),
                          src_context, size, stream_handle_),
        "Failed to enqueue MUSA memcpy peer D2D");
  }
  absl::MutexLock submit_lock(&submit_mu_);
  return musa::ToStatus(
      muMemcpyDtoDAsync(reinterpret_cast<MUdeviceptr>(gpu_dst->opaque()),
                        reinterpret_cast<MUdeviceptr>(gpu_src.opaque()), size,
                        stream_handle_),
      "Failed to enqueue MUSA memcpy D2D");
}

tsl::Status MusaStream::BlockHostUntilDone() {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  constexpr int64_t kDefaultSyncTimeoutMs = 30000;
  int64_t sync_timeout_ms = kDefaultSyncTimeoutMs;
  if (const char* env = std::getenv("TF_MUSA_STREAM_SYNC_TIMEOUT_MS")) {
    char* end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end != env && parsed > 0) {
      sync_timeout_ms = parsed;
    }
  }

  const int64_t deadline_us = sync_timeout_ms * 1000;
  constexpr int64_t kPollIntervalUs = 1000;
  constexpr int64_t kTraceIntervalUs = 1000000;
  int64_t waited_us = 0;
  int64_t next_trace_us = 0;
  const bool runtime_trace = IsMusaDebugRuntimeTraceEnabled();
  while (true) {
    MUresult query = muStreamQuery(stream_handle_);
    if (query == MUSA_SUCCESS) {
      if (runtime_trace) {
        LOG(INFO) << "[MUSA_RUNTIME_TRACE] BlockHostUntilDone query stream="
                  << stream_handle_ << " waited_us=" << waited_us
                  << " result=" << FormatMuResult(query);
      }
      return tsl::OkStatus();
    }
    if (query != MUSA_ERROR_NOT_READY) {
      if (runtime_trace) {
        LOG(INFO) << "[MUSA_RUNTIME_TRACE] BlockHostUntilDone query stream="
                  << stream_handle_ << " waited_us=" << waited_us
                  << " result=" << FormatMuResult(query);
      }
      return musa::ToStatus(query, "Failed to query MUSA stream status");
    }
    if (runtime_trace && waited_us >= next_trace_us) {
      LOG(INFO) << "[MUSA_RUNTIME_TRACE] BlockHostUntilDone query stream="
                << stream_handle_ << " waited_us=" << waited_us
                << " result=" << FormatMuResult(query);
      next_trace_us += kTraceIntervalUs;
    }
    if (waited_us >= deadline_us) {
      if (runtime_trace) {
        MUresult final_query = muStreamQuery(stream_handle_);
        LOG(INFO) << "[MUSA_RUNTIME_TRACE] BlockHostUntilDone timeout stream="
                  << stream_handle_ << " waited_us=" << waited_us
                  << " final_result=" << FormatMuResult(final_query);
        LogDebugLaunchCheckpointStatuses(waited_us);
      }
      return tsl::errors::DeadlineExceeded(
          "Timed out while synchronizing MUSA stream");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(kPollIntervalUs));
    waited_us += kPollIntervalUs;
  }
}

tsl::Status MusaStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<tsl::Status() &&> callback) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());

  auto* callback_ptr =
      new absl::AnyInvocable<void() &&>([cb = std::move(callback)]() mutable {
        tsl::Status status = std::move(cb)();
        if (!status.ok()) {
          LOG(WARNING) << "Host callback failed: " << status;
        }
      });

  tsl::Status status;
  {
    absl::MutexLock submit_lock(&submit_mu_);
    status = musa::ToStatus(
        muLaunchHostFunc(stream_handle_, InternalHostCallback, callback_ptr),
        "Failed to enqueue MUSA host callback");
  }
  if (!status.ok()) {
    delete callback_ptr;
  }
  return status;
}

}  // namespace musa
}  // namespace stream_executor
