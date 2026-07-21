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

#ifndef XLA_STREAM_EXECUTOR_PLATFORM_MANAGER_H_
#define XLA_STREAM_EXECUTOR_PLATFORM_MANAGER_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/multi_platform_manager.h"

namespace stream_executor {

// Compatibility shim for newer OpenXLA callers that expect PlatformManager.
class PlatformManager {
 public:
  static absl::Status RegisterPlatform(std::unique_ptr<Platform> platform) {
    return MultiPlatformManager::RegisterPlatform(std::move(platform));
  }

  static absl::StatusOr<Platform*> PlatformWithName(absl::string_view target) {
    return MultiPlatformManager::PlatformWithName(target);
  }

  static absl::StatusOr<Platform*> PlatformWithId(const Platform::Id& id) {
    return MultiPlatformManager::PlatformWithId(id);
  }

  static absl::StatusOr<Platform*> PlatformWithName(absl::string_view target,
                                                    bool initialize_platform) {
    return MultiPlatformManager::PlatformWithName(target, initialize_platform);
  }

  static absl::StatusOr<Platform*> InitializePlatformWithId(
      const Platform::Id& id) {
    return MultiPlatformManager::InitializePlatformWithId(id, {});
  }

  static absl::StatusOr<std::vector<Platform*>> PlatformsWithFilter(
      const std::function<bool(const Platform*)>& filter) {
    return MultiPlatformManager::PlatformsWithFilter(filter);
  }

  static absl::StatusOr<std::vector<Platform*>> PlatformsWithFilter(
      const std::function<bool(const Platform*)>& filter,
      bool initialize_platform) {
    return MultiPlatformManager::PlatformsWithFilter(filter,
                                                     initialize_platform);
  }
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_PLATFORM_MANAGER_H_
