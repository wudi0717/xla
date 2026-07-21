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

#include "xla/stream_executor/musa/musa_platform.h"

#include <memory>
#include <cstdlib>
#include "absl/memory/memory.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/musa/musa_executor.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"

namespace stream_executor {
namespace musa {

const std::vector<int>& GetMusaVisibleDeviceOrdinals();

namespace {

constexpr char kMusaVisibleDevicesEnv[] = "MUSA_VISIBLE_DEVICES";

tsl::Status PlatformInitialize() {
  return musa::ToStatus(muInit(0), "Failed call to muInit");
}

tsl::StatusOr<int> GetPhysicalDeviceCount() {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  int count = 0;
  auto status = musa::ToStatus(muDeviceGetCount(&count),
                               "Failed to query MUSA device count");
  if (!status.ok()) {
    LOG(ERROR) << status;
    return status;
  }
  return count;
}

tsl::StatusOr<std::vector<int>> BuildVisibleDeviceOrdinals() {
  TF_ASSIGN_OR_RETURN(const int driver_visible_device_count,
                      GetPhysicalDeviceCount());

  std::vector<int> visible_device_ordinals;
  const char* env_value = std::getenv(kMusaVisibleDevicesEnv);
  if (env_value == nullptr) {
    visible_device_ordinals.reserve(driver_visible_device_count);
    for (int ordinal = 0; ordinal < driver_visible_device_count; ++ordinal) {
      visible_device_ordinals.push_back(ordinal);
    }
    return visible_device_ordinals;
  }

  absl::string_view raw_value(env_value);
  raw_value = absl::StripAsciiWhitespace(raw_value);
  if (raw_value.empty() || raw_value == "-1" ||
      absl::EqualsIgnoreCase(raw_value, "none")) {
    return visible_device_ordinals;
  }

  absl::flat_hash_set<int> seen_ordinals;
  for (absl::string_view token : absl::StrSplit(raw_value, ',')) {
    token = absl::StripAsciiWhitespace(token);
    if (token.empty()) {
      return tsl::errors::InvalidArgument(
          "MUSA_VISIBLE_DEVICES contains an empty entry");
    }

    int requested_ordinal = -1;
    if (!absl::SimpleAtoi(token, &requested_ordinal)) {
      return tsl::errors::InvalidArgument(absl::StrCat(
          "MUSA_VISIBLE_DEVICES contains a non-integer entry: ", token));
    }
    if (!seen_ordinals.insert(requested_ordinal).second) {
      return tsl::errors::InvalidArgument(absl::StrCat(
          "MUSA_VISIBLE_DEVICES contains a duplicate entry: ",
          requested_ordinal));
    }

    visible_device_ordinals.push_back(requested_ordinal);
  }

  // Some MUSA runtimes may already honor MUSA_VISIBLE_DEVICES before XLA sees
  // the platform. In that case the remaining visible devices are typically
  // reindexed densely from 0..N-1, so using the raw env ordinals again would
  // double-filter and produce invalid ordinals.
  if (driver_visible_device_count ==
      static_cast<int>(visible_device_ordinals.size())) {
    std::vector<int> remapped_ordinals;
    remapped_ordinals.reserve(driver_visible_device_count);
    for (int ordinal = 0; ordinal < driver_visible_device_count; ++ordinal) {
      remapped_ordinals.push_back(ordinal);
    }
    return remapped_ordinals;
  }

  for (int requested_ordinal : visible_device_ordinals) {
    if (requested_ordinal < 0 ||
        requested_ordinal >= driver_visible_device_count) {
      return tsl::errors::InvalidArgument(absl::StrCat(
          "MUSA_VISIBLE_DEVICES entry ", requested_ordinal,
          " is outside the valid range [0, ", driver_visible_device_count,
          ")"));
    }
  }
  return visible_device_ordinals;
}

}  // namespace

MusaPlatform::MusaPlatform() : name_("MUSA") {}

Platform::Id MusaPlatform::id() const { return kMusaPlatformId; }

int MusaPlatform::VisibleDeviceCount() const {
  return static_cast<int>(GetMusaVisibleDeviceOrdinals().size());
}

const std::string& MusaPlatform::Name() const { return name_; }

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
MusaPlatform::DescriptionForDevice(int ordinal) const {
  return MusaExecutor::CreateDeviceDescription(ordinal);
}

tsl::StatusOr<StreamExecutor*> MusaPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config(ordinal);
  return GetExecutor(config);
}

tsl::StatusOr<StreamExecutor*> MusaPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

tsl::StatusOr<std::unique_ptr<StreamExecutor>>
MusaPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  auto executor = std::make_unique<StreamExecutor>(
      this, std::make_unique<MusaExecutor>(), config.ordinal);
  TF_RETURN_IF_ERROR(executor->Init(config.device_options));
  return std::move(executor);
}

const std::vector<int>& GetMusaVisibleDeviceOrdinals() {
  static const std::vector<int>* visible_device_ordinals = [] {
    auto status_or = BuildVisibleDeviceOrdinals();
    if (!status_or.ok()) {
      LOG(ERROR) << "Failed to parse " << kMusaVisibleDevicesEnv
                 << ": " << status_or.status();
      return new std::vector<int>();
    }
    return new std::vector<int>(std::move(status_or.value()));
  }();
  return *visible_device_ordinals;
}

tsl::StatusOr<int> GetMusaPhysicalDeviceOrdinal(int visible_device_ordinal) {
  const auto& visible_device_ordinals = GetMusaVisibleDeviceOrdinals();
  const int visible_device_count = static_cast<int>(visible_device_ordinals.size());
  if (visible_device_ordinal < 0 ||
      visible_device_ordinal >= visible_device_count) {
    return tsl::errors::InvalidArgument(absl::StrCat(
        "Invalid visible MUSA device ordinal ", visible_device_ordinal,
        "; visible device count is ", visible_device_count));
  }
  return visible_device_ordinals[visible_device_ordinal];
}

}  // namespace musa

static void InitializeMusaPlatform() {
  TF_CHECK_OK(
      MultiPlatformManager::RegisterPlatform(std::make_unique<musa::MusaPlatform>()));
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(musa_platform,
                            stream_executor::InitializeMusaPlatform());
