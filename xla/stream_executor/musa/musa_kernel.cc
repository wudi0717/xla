#include "xla/stream_executor/musa/musa_kernel.h"

#include "tsl/platform/logging.h"
#include "xla/stream_executor/musa/musa_status.h"

namespace stream_executor {
namespace musa {
namespace {

tsl::Status GetMusaAttribute(MUfunction_attribute attribute, MUfunction func,
                             int* attribute_value) {
  return musa::ToStatus(muFuncGetAttribute(attribute_value, attribute, func),
                        "Failed to query MUSA kernel attribute");
}

}  // namespace

MUfunc_cache MusaKernel::GetMusaCacheConfig() const {
  switch (preferred_cache_config_) {
    case KernelCacheConfig::kNoPreference:
      return MU_FUNC_CACHE_PREFER_NONE;
    case KernelCacheConfig::kPreferShared:
      return MU_FUNC_CACHE_PREFER_SHARED;
    case KernelCacheConfig::kPreferL1:
      return MU_FUNC_CACHE_PREFER_L1;
    case KernelCacheConfig::kPreferEqual:
      return MU_FUNC_CACHE_PREFER_EQUAL;
    default:
      LOG(FATAL) << "Unknown KernelCacheConfig "
                 << static_cast<int32_t>(preferred_cache_config_);
  }
}

tsl::Status MusaKernel::GetKernelMetadata(
    KernelMetadata* kernel_metadata) const {
  int value = 0;
  TF_RETURN_IF_ERROR(
      GetMusaAttribute(MU_FUNC_ATTRIBUTE_NUM_REGS, gpu_function_, &value));
  kernel_metadata->set_registers_per_thread(value);

  TF_RETURN_IF_ERROR(GetMusaAttribute(MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                      gpu_function_, &value));
  kernel_metadata->set_shared_memory_bytes(value);
  return ::tsl::OkStatus();
}

tsl::Status MusaKernel::PrepareForLaunch(uint64_t shared_mem_bytes) const {
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muFuncSetAttribute(gpu_function_,
                           MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set MUSA dynamic shared memory size"));
  }

  MUfunc_cache cache_config = MU_FUNC_CACHE_PREFER_NONE;
  if (preferred_cache_config_ != KernelCacheConfig::kNoPreference) {
    cache_config = GetMusaCacheConfig();
  } else if (shared_mem_bytes != 0) {
    cache_config = MU_FUNC_CACHE_PREFER_SHARED;
  }

  if (cache_config != MU_FUNC_CACHE_PREFER_NONE) {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muFuncSetCacheConfig(gpu_function_, cache_config),
        "Failed to set MUSA kernel cache config"));
  }

  return ::tsl::OkStatus();
}

}  // namespace musa
}  // namespace stream_executor
