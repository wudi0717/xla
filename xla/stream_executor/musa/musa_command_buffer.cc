#include "xla/stream_executor/musa/musa_command_buffer.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "musa.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_kernel.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/stream_executor/musa/musa_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

extern "C" {
MUresult MUSAAPI muStreamBeginCaptureToGraph(
    MUstream hStream, MUgraph hGraph, const MUgraphNode* dependencies,
    const MUgraphEdgeData* dependencyData, size_t numDependencies,
    MUstreamCaptureMode mode);
}

namespace stream_executor {
namespace musa {
namespace {

using Mode = CommandBuffer::Mode;
using State = CommandBuffer::State;

MusaStream* AsMusaStream(Stream* stream) {
  return static_cast<MusaStream*>(stream);
}

tsl::Status UnsupportedStateError(State state) {
  const char* state_str = "unknown";
  switch (state) {
    case State::kCreate:
      state_str = "create";
      break;
    case State::kUpdate:
      state_str = "update";
      break;
    case State::kFinalized:
      state_str = "finalized";
      break;
  }
  return tsl::errors::Internal("Unsupported command buffer state: ", state_str);
}

MUdeviceptr AsDevicePtr(const DeviceMemoryBase& mem) {
  return reinterpret_cast<MUdeviceptr>(const_cast<void*>(mem.opaque()));
}

std::string FormatKernelArgsSummary(const KernelArgsArrayBase& args) {
  std::vector<std::string> parts;
  parts.reserve(args.number_of_arguments());
  KernelArgIterator iter = args.arg_iterator();
  size_t index = 0;
  while (iter.has_next()) {
    KernelArg arg = iter.next();
    if (arg.is_shared) {
      parts.push_back(
          absl::StrCat("shared#", index, "(bytes=", arg.size, ")"));
    } else {
      parts.push_back(
          absl::StrCat("arg#", index, "(size=", arg.size, ", addr=",
                       absl::StrFormat("%p", arg.address), ")"));
    }
    ++index;
  }
  return absl::StrCat("[", absl::StrJoin(parts, ", "), "]");
}

}  // namespace

tsl::StatusOr<std::unique_ptr<MusaCommandBuffer>> MusaCommandBuffer::Create(
    CommandBuffer::Mode mode, MUcontext context) {
  ScopedActivateContext activation(context);
  TF_RETURN_IF_ERROR(activation.status());
  MUgraph graph = nullptr;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muGraphCreate(&graph, /*flags=*/0),
                     "Failed to create MUSA graph"));
  return std::unique_ptr<MusaCommandBuffer>(
      new MusaCommandBuffer(mode, context, graph));
}

MusaCommandBuffer::MusaCommandBuffer(CommandBuffer::Mode mode, MUcontext context,
                                     MUgraph graph)
    : mode_(mode), context_(context), graph_(graph) {}

MusaCommandBuffer::~MusaCommandBuffer() {
  ScopedActivateContext activation(context_);
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }
  if (exec_ != nullptr) {
    auto st =
        musa::ToStatus(muGraphExecDestroy(exec_),
                       "Failed to destroy MUSA executable graph");
    CHECK(st.ok()) << st.message();
  }
  if (graph_ != nullptr) {
    auto st = musa::ToStatus(muGraphDestroy(graph_), "Failed to destroy MUSA graph");
    CHECK(st.ok()) << st.message();
  }
}

tsl::Status MusaCommandBuffer::CheckNotFinalized() {
  if (state_ == State::kFinalized) {
    return tsl::errors::Internal(
        "Command can't be added to a command buffer after it was finalized");
  }
  return ::tsl::OkStatus();
}

tsl::Status MusaCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<tsl::Status()> function) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muStreamBeginCaptureToGraph(AsMusaStream(stream)->stream_handle(), graph_,
                                  /*dependencies=*/nullptr,
                                  /*dependencyData=*/nullptr,
                                  /*numDependencies=*/0,
                                  MU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
      "Failed to begin stream capture to MUSA graph"));

  auto traced = function();

  MUgraph captured_graph = nullptr;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muStreamEndCapture(AsMusaStream(stream)->stream_handle(), &captured_graph),
      "Failed to end MUSA stream capture"));
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();
  (void)start_nanos;
  (void)end_nanos;

  if (!traced.ok()) {
    return tsl::errors::Internal("Failed to capture MUSA graph: ",
                                 traced.message());
  }
  if (captured_graph != graph_) {
    return tsl::errors::Internal("Captured graph does not match command buffer graph");
  }
  size_t num_root_nodes = 0;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muGraphGetRootNodes(captured_graph, nullptr, &num_root_nodes),
      "Failed to inspect captured MUSA graph"));
  if (num_root_nodes == 0) {
    return tsl::errors::Internal(
        "Traced MUSA graph is empty and cannot be instantiated safely");
  }
  is_traced_ = true;
  return ::tsl::OkStatus();
}

tsl::Status MusaCommandBuffer::Launch(const ThreadDim& threads,
                                      const BlockDim& blocks,
                                      const KernelBase& kernel,
                                      const KernelArgsArrayBase& args) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  const MusaKernel* musa_kernel = AsMusaKernel(&kernel);
  MUfunction function = musa_kernel->AsMusaFunctionHandle();
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();
  const std::string arg_summary = FormatKernelArgsSummary(args);
  TF_RETURN_IF_ERROR(musa_kernel->PrepareForLaunch(shared_mem_bytes));
  void** kernel_params = const_cast<void**>(args.argument_addresses().data());

  VLOG(2) << "Record kernel for MUSA command buffer; kernel=" << kernel.name()
          << " gdx=" << blocks.x << " gdy=" << blocks.y << " gdz=" << blocks.z
          << " bdx=" << threads.x << " bdy=" << threads.y
          << " bdz=" << threads.z << "; shmem=" << shared_mem_bytes
          << "; args=" << arg_summary;

  if (state_ == State::kCreate) {
    MUSA_KERNEL_NODE_PARAMS params{};
    params.func = function;
    params.gridDimX = blocks.x;
    params.gridDimY = blocks.y;
    params.gridDimZ = blocks.z;
    params.blockDimX = threads.x;
    params.blockDimY = threads.y;
    params.blockDimZ = threads.z;
    params.sharedMemBytes = shared_mem_bytes;
    params.kernelParams = kernel_params;
    params.extra = nullptr;

    MUgraphNode* node = &nodes_.emplace_back();
    std::string detail = absl::StrCat(
        "Failed to add kernel node to MUSA graph; kernel=", kernel.name(),
        "; grid dims=", blocks.x, "x", blocks.y, "x", blocks.z,
        "; block dims=", threads.x, "x", threads.y, "x", threads.z,
        "; shared memory bytes=", shared_mem_bytes, "; args=", arg_summary);
    return musa::ToStatus(muGraphAddKernelNode(node, graph_, /*dependencies=*/nullptr,
                                               /*numDependencies=*/0, &params),
                          detail.c_str());
  }

  if (state_ == State::kUpdate) {
    if (node_update_idx_ >= static_cast<int64_t>(nodes_.size())) {
      return tsl::errors::Internal("Too many kernel node updates for MUSA graph");
    }
    MUSA_KERNEL_NODE_PARAMS params{};
    params.func = function;
    params.gridDimX = blocks.x;
    params.gridDimY = blocks.y;
    params.gridDimZ = blocks.z;
    params.blockDimX = threads.x;
    params.blockDimY = threads.y;
    params.blockDimZ = threads.z;
    params.sharedMemBytes = shared_mem_bytes;
    params.kernelParams = kernel_params;
    params.extra = nullptr;

    MUgraphNode node = nodes_[node_update_idx_++];
    std::string detail = absl::StrCat(
        "Failed to update MUSA graph kernel node params; kernel=",
        kernel.name(), "; grid dims=", blocks.x, "x", blocks.y, "x", blocks.z,
        "; block dims=", threads.x, "x", threads.y, "x", threads.z,
        "; shared memory bytes=", shared_mem_bytes, "; args=", arg_summary);
    return musa::ToStatus(muGraphExecKernelNodeSetParams(exec_, node, &params),
                          detail.c_str());
  }

  return UnsupportedStateError(state_);
}

tsl::Status MusaCommandBuffer::MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                                    const DeviceMemoryBase& src,
                                                    uint64_t size) {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (state_ == State::kCreate) {
    MUSA_MEMCPY3D params{};
    params.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    params.srcDevice = AsDevicePtr(src);
    params.dstMemoryType = MU_MEMORYTYPE_DEVICE;
    params.dstDevice = AsDevicePtr(*dst);
    params.WidthInBytes = size;
    params.Height = 1;
    params.Depth = 1;

    MUgraphNode* node = &nodes_.emplace_back();
    return musa::ToStatus(
        muGraphAddMemcpyNode(node, graph_, /*dependencies=*/nullptr,
                             /*numDependencies=*/0, &params, context_),
        "Failed to add memcpy node to MUSA graph");
  }

  if (state_ == State::kUpdate) {
    if (node_update_idx_ >= static_cast<int64_t>(nodes_.size())) {
      return tsl::errors::Internal("Too many memcpy node updates for MUSA graph");
    }

    MUSA_MEMCPY3D params{};
    params.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    params.srcDevice = AsDevicePtr(src);
    params.dstMemoryType = MU_MEMORYTYPE_DEVICE;
    params.dstDevice = AsDevicePtr(*dst);
    params.WidthInBytes = size;
    params.Height = 1;
    params.Depth = 1;

    MUgraphNode node = nodes_[node_update_idx_++];
    return musa::ToStatus(
        muGraphExecMemcpyNodeSetParams(exec_, node, &params, context_),
        "Failed to update MUSA graph memcpy node params");
  }

  return UnsupportedStateError(state_);
}

tsl::Status MusaCommandBuffer::Finalize() {
  ScopedActivateContext activation(context_);
  TF_RETURN_IF_ERROR(activation.status());
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (mode_ == Mode::kPrimary && state_ == State::kCreate) {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muGraphInstantiate(&exec_, graph_, /*flags=*/0),
        "Failed to instantiate MUSA graph"));
  } else if (mode_ == Mode::kPrimary && state_ == State::kUpdate) {
    if (node_update_idx_ != static_cast<int64_t>(nodes_.size())) {
      return tsl::errors::Internal(
          absl::StrCat("Incomplete MUSA graph update: expected ", nodes_.size(),
                       " node updates but saw ", node_update_idx_));
    }
  } else if (mode_ == Mode::kNested) {
    // Nested command buffers intentionally do not instantiate executable graphs.
  }

  state_ = State::kFinalized;
  return ::tsl::OkStatus();
}

tsl::Status MusaCommandBuffer::Update() {
  if (state_ != State::kFinalized) {
    return tsl::errors::Internal(
        "Command buffer has to be finalized first before it can be updated");
  }
  if (is_traced_) {
    return tsl::errors::Unimplemented(
        "Traced MUSA command buffers cannot be updated");
  }
  if (exec_ == nullptr) {
    return tsl::errors::Unimplemented(
        "Nested command buffer update is not implemented");
  }

  state_ = State::kUpdate;
  node_update_idx_ = 0;
  return ::tsl::OkStatus();
}

}  // namespace musa
}  // namespace stream_executor
