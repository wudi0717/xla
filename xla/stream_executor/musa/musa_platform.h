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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_PLATFORM_H_

#include <string>

#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor {
namespace musa {

class MusaPlatform : public Platform {
 public:
  MusaPlatform();
  ~MusaPlatform() override = default;

  Platform::Id id() const override;
  int VisibleDeviceCount() const override;
  const std::string& Name() const override;
  tsl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;
  tsl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;
  tsl::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;
  tsl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

 private:
  std::string name_;
  ExecutorCache executor_cache_;
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_PLATFORM_H_
