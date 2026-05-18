#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_COMMAND_BUFFER_H_

#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "musa.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

class MusaCommandBuffer
    : public ::stream_executor::internal::CommandBufferInterface {
 public:
  static tsl::StatusOr<std::unique_ptr<MusaCommandBuffer>> Create(
      CommandBuffer::Mode mode, MUcontext context);

  ~MusaCommandBuffer() override;

  tsl::Status Trace(Stream* stream,
                    absl::AnyInvocable<tsl::Status()> function) override;

  tsl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                     const KernelBase& kernel,
                     const KernelArgsArrayBase& args) override;

  tsl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                   const DeviceMemoryBase& src,
                                   uint64_t size) override;

  tsl::Status Finalize() override;
  tsl::Status Update() override;

  CommandBuffer::Mode mode() const override { return mode_; }
  CommandBuffer::State state() const override { return state_; }

  MUgraphExec executable() const { return exec_; }

 private:
  MusaCommandBuffer(CommandBuffer::Mode mode, MUcontext context, MUgraph graph);

  tsl::Status CheckNotFinalized();

  static_assert(std::is_pointer_v<MUgraph>, "MUgraph must be a pointer");
  static_assert(std::is_pointer_v<MUgraphExec>,
                "MUgraphExec must be a pointer");
  static_assert(std::is_pointer_v<MUgraphNode>, "MUgraphNode must be a pointer");

  CommandBuffer::Mode mode_;
  CommandBuffer::State state_ = CommandBuffer::State::kCreate;

  MUcontext context_ = nullptr;
  MUgraph graph_ = nullptr;
  MUgraphExec exec_ = nullptr;
  std::vector<MUgraphNode> nodes_;
  int64_t node_update_idx_ = 0;
  bool is_traced_ = false;
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_COMMAND_BUFFER_H_
