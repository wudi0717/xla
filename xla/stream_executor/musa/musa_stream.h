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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_

#include <deque>
#include <optional>
#include <string>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "musa.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor {
class Event;
class StreamExecutor;

namespace musa {

class MusaExecutor;

class MusaStream : public StreamCommon {
 public:
  static tsl::StatusOr<std::unique_ptr<Stream>> Create(
      StreamExecutor* executor, MUcontext context,
      std::optional<std::variant<StreamPriority, int>> priority);

  MusaStream(StreamExecutor* executor,
             MUcontext context,
             std::optional<std::variant<StreamPriority, int>> priority,
             MUstream stream_handle, MUevent completed_event);
  ~MusaStream() override;

  tsl::Status WaitFor(Stream* other) override;
  tsl::Status WaitFor(Event* event) override;
  tsl::Status RecordEvent(Event* event) override;
  tsl::Status MemZero(DeviceMemoryBase* location, uint64_t size) override;
  tsl::Status Memset32(DeviceMemoryBase* location, uint32_t pattern,
                       uint64_t size) override;
  tsl::Status Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                     uint64_t size) override;
  tsl::Status Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                     uint64_t size) override;
  tsl::Status Memcpy(DeviceMemoryBase* gpu_dst,
                     const DeviceMemoryBase& gpu_src, uint64_t size) override;
  tsl::Status BlockHostUntilDone() override;
  tsl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<tsl::Status() &&> callback) override;

  PlatformSpecificHandle platform_specific_handle() const override {
    PlatformSpecificHandle handle;
    handle.stream = stream_handle_;
    return handle;
  }

  MUstream stream_handle() const { return stream_handle_; }
  MUevent completed_event() const { return completed_event_; }
  absl::Mutex* submit_mu() { return &submit_mu_; }
  tsl::Status RecordCompletedEvent();
  void RecordDebugLaunchCheckpoint(const std::string& kernel_name,
                                   const std::string& launch_dims,
                                   uint64_t shared_mem_bytes);

 private:
  struct DebugLaunchCheckpoint {
    uint64_t sequence = 0;
    std::string kernel_name;
    std::string launch_dims;
    uint64_t shared_mem_bytes = 0;
    MUevent event = nullptr;
  };

  void LogDebugLaunchCheckpointStatuses(int64_t waited_us);
  void ClearDebugLaunchCheckpoints();

  StreamExecutor* executor_;
  MUcontext context_;
  MUstream stream_handle_;
  MUevent completed_event_;
  absl::Mutex submit_mu_;
  absl::Mutex debug_launch_checkpoints_mu_;
  std::deque<DebugLaunchCheckpoint> debug_launch_checkpoints_
      ABSL_GUARDED_BY(debug_launch_checkpoints_mu_);
  uint64_t debug_launch_sequence_ ABSL_GUARDED_BY(debug_launch_checkpoints_mu_) =
      0;
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_
