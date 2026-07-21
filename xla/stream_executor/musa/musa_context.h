#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_

#include <cstdint>
#include <memory>

#include "musa.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace musa {

class MusaContext {
 public:
  static tsl::StatusOr<std::unique_ptr<MusaContext>> Create(MUdevice device);

  ~MusaContext();

  MUdevice device() const { return device_; }
  MUcontext context() const { return context_; }

 private:
  MusaContext(MUdevice device, MUcontext context)
      : device_(device), context_(context) {}

  MUdevice device_ = 0;
  MUcontext context_ = nullptr;
};

class ScopedActivateContext {
 public:
  explicit ScopedActivateContext(const MusaContext* context);
  explicit ScopedActivateContext(MUcontext context);
  ~ScopedActivateContext();

  bool ok() const { return status_.ok(); }
  const tsl::Status& status() const { return status_; }

 private:
  void Init(MUcontext context);

  MUcontext previous_ = nullptr;
  bool restore_previous_ = false;
  tsl::Status status_;
};

void RegisterDeviceMemoryAllocation(void* ptr, uint64_t size, MUcontext context);
void UnregisterDeviceMemoryAllocation(void* ptr);
MUcontext GetContextForDeviceMemory(void* ptr);

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_
