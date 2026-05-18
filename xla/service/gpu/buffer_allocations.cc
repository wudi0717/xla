/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/buffer_allocations.h"

#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/map_util.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/lib/gtl/map_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

bool IsMusaDebugDeallocEnabled() {
  const char* value = std::getenv("TF_MUSA_DEBUG_DEALLOC");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

std::string DescribeAllocationForDebug(const BufferAllocation& allocation,
                                       se::DeviceMemoryBase buffer_address,
                                       bool is_live_out) {
  std::vector<std::string> tags;
  if (allocation.is_entry_computation_parameter()) {
    tags.push_back(absl::StrCat("param#", allocation.parameter_number()));
  }
  if (allocation.is_constant()) {
    tags.push_back("constant");
  }
  if (allocation.IsPreallocatedTempBuffer()) {
    tags.push_back("temp");
  }
  if (allocation.maybe_live_out()) {
    tags.push_back(is_live_out ? "live_out" : "not_live_out");
  }
  if (allocation.is_thread_local()) {
    tags.push_back("thread_local");
  }

  std::vector<std::string> assigned_hlos;
  assigned_hlos.reserve(std::min<size_t>(allocation.assigned_buffers().size(), 4));
  size_t emitted = 0;
  for (const auto& it : allocation.assigned_buffers()) {
    if (emitted == 4) {
      assigned_hlos.push_back("...");
      break;
    }
    const HloValue* value = it.first;
    const BufferAllocation::OffsetSize& offset_size = it.second;
    const HloInstruction* instruction = value->instruction();
    assigned_hlos.push_back(absl::StrCat(
        instruction->name(),
        value->index().empty() ? "" : value->index().ToString(), "@offset=",
        offset_size.offset, "/size=", offset_size.size));
    ++emitted;
  }

  return absl::StrCat("alloc#", allocation.index(), " ptr=",
                      reinterpret_cast<uintptr_t>(buffer_address.opaque()),
                      " size=", buffer_address.size(), "B flags=[",
                      absl::StrJoin(tags, ","), "] hlo=[",
                      absl::StrJoin(assigned_hlos, "; "), "]");
}

}  // namespace

Status BufferAllocations::TearDown(
    const std::set<se::DeviceMemoryBase>& live_addresses,
    absl::Span<const BufferAllocation> allocations) {
  // Deallocate temporary buffers, taking care to try to deallocate all of them
  // even if one of the deallocations fails.
  Status status;
  const int64_t num_buffers = allocations.size();
  for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = allocations[i];
    se::DeviceMemoryBase buffer_address = GetDeviceAddress(allocation.index());
    // Deallocate buffers marked "maybe_live_out" but aren't actually live out,
    // and temp buffers.
    const bool is_live_out = live_addresses.count(buffer_address) > 0;
    if ((allocation.maybe_live_out() && !is_live_out) ||
        allocation.IsPreallocatedTempBuffer()) {
      if (IsMusaDebugDeallocEnabled()) {
        LOG(INFO) << "[MUSA_DEALLOC_DEBUG] TearDown deallocate begin "
                  << DescribeAllocationForDebug(allocation, buffer_address,
                                                is_live_out);
      }
      auto dealloc_result =
          memory_allocator_->Deallocate(device_ordinal_, buffer_address);
      if (IsMusaDebugDeallocEnabled()) {
        if (dealloc_result.ok()) {
          LOG(INFO) << "[MUSA_DEALLOC_DEBUG] TearDown deallocate end alloc#"
                    << allocation.index();
        } else {
          LOG(ERROR) << "[MUSA_DEALLOC_DEBUG] TearDown deallocate failed "
                     << DescribeAllocationForDebug(allocation, buffer_address,
                                                   is_live_out)
                     << " status=" << dealloc_result;
        }
      }
      if (!dealloc_result.ok() && status.ok()) {
        status = dealloc_result;
      }
    }
  }
  return status;
}

se::DeviceMemoryBase BufferAllocations::GetDeviceAddress(
    BufferAllocation::Index buffer_index) const {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

se::DeviceMemoryBase& BufferAllocations::GetMutableDeviceAddress(
    BufferAllocation::Index buffer_index) {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

se::DeviceMemoryBase BufferAllocations::GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice) const {
  se::DeviceMemoryBase base = GetDeviceAddress(buffer_slice.index());
  CHECK_LE(buffer_slice.offset(), base.size());
  CHECK_LE(buffer_slice.offset() + buffer_slice.size(), base.size());
  return se::DeviceMemoryBase(
      static_cast<char*>(base.opaque()) + buffer_slice.offset(),
      buffer_slice.size());
}

}  // namespace gpu
}  // namespace xla
