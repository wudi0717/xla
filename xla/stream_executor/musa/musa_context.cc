#include "xla/stream_executor/musa/musa_context.h"

#include <cstdint>
#include <map>

#include "absl/synchronization/mutex.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace stream_executor {
namespace musa {
namespace {

struct AllocationRecord {
  uint64_t size = 0;
  MUcontext context = nullptr;
};

absl::Mutex& AllocationRegistryMutex() {
  static auto* mu = new absl::Mutex();
  return *mu;
}

std::map<uintptr_t, AllocationRecord>& AllocationRegistry() {
  static auto* registry = new std::map<uintptr_t, AllocationRecord>();
  return *registry;
}

}  // namespace

tsl::StatusOr<std::unique_ptr<MusaContext>> MusaContext::Create(MUdevice device) {
  MUcontext context = nullptr;
  TF_RETURN_IF_ERROR(musa::ToStatus(muDevicePrimaryCtxRetain(&context, device),
                                    "Failed to retain MUSA primary context"));
  return std::unique_ptr<MusaContext>(new MusaContext(device, context));
}

MusaContext::~MusaContext() {
  if (context_ == nullptr) {
    return;
  }

  ScopedActivateContext activation(this);
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }

  auto release_status = musa::ToStatus(muDevicePrimaryCtxRelease(device_),
                                       "Failed to release MUSA primary context");
  if (!release_status.ok()) {
    LOG(ERROR) << release_status;
  }
}

ScopedActivateContext::ScopedActivateContext(const MusaContext* context)
    : status_(::tsl::OkStatus()) {
  Init(context == nullptr ? nullptr : context->context());
}

ScopedActivateContext::ScopedActivateContext(MUcontext context)
    : status_(::tsl::OkStatus()) {
  Init(context);
}

ScopedActivateContext::~ScopedActivateContext() {
  if (!status_.ok() || !restore_previous_) {
    return;
  }

  auto restore_status = musa::ToStatus(muCtxSetCurrent(previous_),
                                       "Failed to restore previous MUSA context");
  if (!restore_status.ok()) {
    LOG(ERROR) << restore_status;
  }
}

void ScopedActivateContext::Init(MUcontext context) {
  if (context == nullptr) {
    status_ = tsl::errors::FailedPrecondition("MUSA context is not initialized");
    return;
  }

  status_ = musa::ToStatus(muCtxGetCurrent(&previous_),
                           "Failed to query current MUSA context");
  if (!status_.ok()) {
    return;
  }

  if (previous_ == context) {
    return;
  }

  status_ =
      musa::ToStatus(muCtxSetCurrent(context), "Failed to activate MUSA context");
  if (!status_.ok()) {
    return;
  }

  restore_previous_ = true;
}

void RegisterDeviceMemoryAllocation(void* ptr, uint64_t size, MUcontext context) {
  if (ptr == nullptr || size == 0 || context == nullptr) {
    return;
  }
  absl::MutexLock lock(&AllocationRegistryMutex());
  AllocationRegistry()[reinterpret_cast<uintptr_t>(ptr)] = {size, context};
}

void UnregisterDeviceMemoryAllocation(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  absl::MutexLock lock(&AllocationRegistryMutex());
  AllocationRegistry().erase(reinterpret_cast<uintptr_t>(ptr));
}

MUcontext GetContextForDeviceMemory(void* ptr) {
  if (ptr == nullptr) {
    return nullptr;
  }

  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  absl::MutexLock lock(&AllocationRegistryMutex());
  auto& registry = AllocationRegistry();
  auto it = registry.upper_bound(address);
  if (it == registry.begin()) {
    return nullptr;
  }
  --it;
  const uintptr_t base = it->first;
  const AllocationRecord& record = it->second;
  if (address < base + record.size) {
    return record.context;
  }
  return nullptr;
}

}  // namespace musa
}  // namespace stream_executor
