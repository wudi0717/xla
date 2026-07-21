#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_

#include <cstdint>

#include "musa.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

class MusaKernel : public ::stream_executor::internal::KernelInterface {
 public:
  MusaKernel()
      : gpu_function_(nullptr),
        arity_(0),
        preferred_cache_config_(KernelCacheConfig::kNoPreference) {}

  ~MusaKernel() override = default;

  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  MUfunction AsMusaFunctionHandle() const {
    CHECK(gpu_function_ != nullptr);
    return gpu_function_;
  }

  MUfunction* gpu_function_ptr() { return &gpu_function_; }

  void SetPreferredCacheConfig(KernelCacheConfig config) override {
    preferred_cache_config_ = config;
  }

  KernelCacheConfig GetPreferredCacheConfig() const override {
    return preferred_cache_config_;
  }

  MUfunc_cache GetMusaCacheConfig() const;

  tsl::Status GetKernelMetadata(KernelMetadata* kernel_metadata) const;
  tsl::Status PrepareForLaunch(uint64_t shared_mem_bytes) const;

 private:
  MUfunction gpu_function_;
  unsigned arity_;
  KernelCacheConfig preferred_cache_config_;
};

inline const MusaKernel* AsMusaKernel(const KernelBase* kernel) {
  return static_cast<const MusaKernel*>(kernel->implementation());
}

inline MusaKernel* AsMusaKernel(KernelBase* kernel) {
  return static_cast<MusaKernel*>(kernel->implementation());
}

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_
