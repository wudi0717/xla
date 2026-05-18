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

#ifndef XLA_STREAM_EXECUTOR_MEMORY_ALLOCATION_H_
#define XLA_STREAM_EXECUTOR_MEMORY_ALLOCATION_H_

#include <cstdint>

namespace stream_executor {

// RAII handle for memory allocated for a device or an attached runtime.
class MemoryAllocation {
 public:
  MemoryAllocation() = default;
  virtual ~MemoryAllocation() = default;

  MemoryAllocation(MemoryAllocation&&) = delete;
  MemoryAllocation& operator=(MemoryAllocation&&) = delete;

  virtual void* opaque() const = 0;
  virtual uint64_t size() const = 0;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MEMORY_ALLOCATION_H_
