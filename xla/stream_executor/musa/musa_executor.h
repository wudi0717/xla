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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/numeric/int128.h"
#include "absl/synchronization/mutex.h"
#include "musa.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

class MusaExecutor : public ::stream_executor::internal::StreamExecutorInterface {
 public:
  MusaExecutor() = default;
  ~MusaExecutor() override = default;

  tsl::Status Init(int device_ordinal, DeviceOptions device_options) override;

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
  void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                     uint64_t size) override;
  void Deallocate(DeviceMemoryBase* mem) override;
  void* UnifiedMemoryAllocate(uint64_t size) override;
  void UnifiedMemoryDeallocate(void* mem) override;

  void* HostMemoryAllocate(uint64_t size) override;
  void HostMemoryDeallocate(void* mem) override;
  bool HostMemoryRegister(void* mem, uint64_t size) override;
  bool HostMemoryUnregister(void* mem) override;

  bool SynchronizeAllActivity() override;
  tsl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                 uint64_t size) override;
  tsl::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) override;
  tsl::Status SynchronousMemcpyDeviceToDevice(
      DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src,
      uint64_t size) override;

  tsl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                      uint64_t size) override;
  tsl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                       uint32_t pattern, uint64_t size) override;
  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64_t size) override;
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64_t size) override;
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override;
  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<tsl::Status() &&> callback) override;

  tsl::Status AllocateEvent(Event* event) override;
  tsl::Status DeallocateEvent(Event* event) override;
  tsl::Status RecordEvent(Stream* stream, Event* event) override;
  tsl::Status WaitForEvent(Stream* stream, Event* event) override;
  tsl::Status WaitForEventOnExternalStream(std::intptr_t stream,
                                           Event* event) override;
  Event::Status PollForEventStatus(Event* event) override;

  bool AllocateStream(Stream* stream) override;
  void DeallocateStream(Stream* stream) override;
  bool CreateStreamDependency(Stream* dependent, Stream* other) override;
  tsl::Status BlockHostUntilDone(Stream* stream) override;

  tsl::Status EnablePeerAccessTo(
      ::stream_executor::internal::StreamExecutorInterface* other) override;
  bool CanEnablePeerAccessTo(
      ::stream_executor::internal::StreamExecutorInterface* other) override;

  blas::BlasSupport* CreateBlas() override;
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;
  tsl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;
  static tsl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);
  tsl::Status GetKernel(const MultiKernelLoaderSpec& spec,
                        KernelBase* kernel) override;
  void UnloadKernel(const KernelBase* kernel) override;
  tsl::Status LoadModule(const MultiModuleLoaderSpec& spec,
                         ModuleHandle* module_handle) override;
  bool UnloadModule(ModuleHandle module_handle) override;
  bool GetSymbol(const std::string& symbol_name, ModuleHandle module_handle,
                 void** mem, size_t* bytes) override;
  tsl::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, const std::vector<uint8_t>& content) override;
  tsl::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                     const BlockDim& block_dims, const KernelBase& kernel,
                     const KernelArgsArrayBase& args) override;
  tsl::Status Submit(Stream* stream,
                     const CommandBuffer& command_buffer) override;

  std::unique_ptr<::stream_executor::internal::EventInterface>
  CreateEventImplementation()
      override;
  std::unique_ptr<::stream_executor::internal::KernelInterface>
  CreateKernelImplementation()
      override;
  std::unique_ptr<::stream_executor::internal::StreamInterface>
  GetStreamImplementation() override;
  tsl::StatusOr<std::unique_ptr<::stream_executor::internal::CommandBufferInterface>>
  GetCommandBufferImplementation(CommandBuffer::Mode mode) override;

  tsl::Status CreateCustomStream(
      StreamExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority,
      std::unique_ptr<Stream>* stream) override;
  Stream* FindAllocatedStream(void* gpu_stream) override;

  ScopedActivateContext ActivateContext() const;
  MUcontext context() const {
    return context_ == nullptr ? nullptr : context_->context();
  }

  void UnregisterStream(MUstream stream);

 private:
  void VlogOccupancyInfo(const KernelBase& kernel, const ThreadDim& thread_dims,
                         size_t dynamic_shared_memory_bytes);

  int device_ordinal_ = -1;
  int physical_device_ordinal_ = -1;
  MUdevice device_ = 0;
  std::unique_ptr<MusaContext> context_;
  mutable absl::Mutex in_memory_modules_mu_;
  absl::flat_hash_map<const KernelBase*, void*> kernel_to_gpu_binary_
      ABSL_GUARDED_BY(in_memory_modules_mu_);
  absl::flat_hash_map<void*, std::pair<MUmodule, uint64_t>> gpu_binary_to_module_
      ABSL_GUARDED_BY(in_memory_modules_mu_);
  mutable absl::Mutex shared_constants_mu_;
  std::map<absl::uint128, std::weak_ptr<DeviceMemoryBase>> shared_constants_
      ABSL_GUARDED_BY(shared_constants_mu_);
  mutable absl::Mutex launched_kernels_mu_;
  std::set<MUfunction> launched_kernels_ ABSL_GUARDED_BY(launched_kernels_mu_);
  mutable absl::Mutex alive_streams_mu_;
  absl::flat_hash_map<void*, Stream*> alive_streams_
      ABSL_GUARDED_BY(alive_streams_mu_);
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_
