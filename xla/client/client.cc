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

#include "xla/client/client.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/client/xla_computation.h"
#include "xla/debug_options_flags.h"
#include "xla/execution_options_util.h"
#include "xla/literal.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "tsl/platform/logging.h"

namespace xla {

Client::Client(Service* stub) : stub_(stub) {}

Client::~Client() = default;

StatusOr<Literal> Client::Transfer(const GlobalData& data,
                                   const Shape* shape_with_layout) {
  return stub_->TransferToClient(data, shape_with_layout);
}

StatusOr<std::unique_ptr<GlobalData>> Client::TransferToServer(
    const LiteralSlice& literal, const DeviceHandle* device_handle) {
  return stub_->TransferToServer(literal, device_handle);
}

Status Client::TransferToInfeed(const LiteralSlice& literal, int64_t replica_id,
                                const DeviceHandle* device_handle) {
  return stub_->TransferToInfeed(literal, replica_id, device_handle);
}

StatusOr<Literal> Client::TransferFromOutfeed(
    const Shape* shape_with_layout, int64_t replica_id,
    const DeviceHandle* device_handle) {
  return stub_->TransferFromOutfeed(shape_with_layout, replica_id,
                                    device_handle);
}

Status Client::ResetDevice() { return stub_->ResetDevice(); }

StatusOr<Literal> Client::ExecuteAndTransfer(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const ExecutionOptions* execution_options,
    ExecutionProfile* execution_profile) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GlobalData> data,
      Execute(computation, arguments, execution_options, execution_profile));

  std::optional<Shape> shape_with_output_layout;
  if (execution_options && execution_options->has_shape_with_output_layout()) {
    shape_with_output_layout =
        Shape(execution_options->shape_with_output_layout());
  }
  return Transfer(*data, shape_with_output_layout.has_value()
                             ? &(*shape_with_output_layout)
                             : nullptr);
}

StatusOr<Literal> Client::ComputeConstant(const XlaComputation& computation,
                                          const Layout* output_layout) const {
  return stub_->ComputeConstantGraph(computation, output_layout);
}

StatusOr<XlaComputation> Client::LoadSnapshot(const HloSnapshot& module) {
  TF_RET_CHECK(module.has_hlo() && module.hlo().has_hlo_module());
  return XlaComputation(module.hlo().hlo_module());
}

StatusOr<ExecutionHandle> Client::Compile(
    const XlaComputation& computation, absl::Span<const Shape> argument_shapes,
    const ExecutionOptions* execution_options) {
  std::optional<ExecutionOptions> opts;
  if (execution_options == nullptr) {
    opts = CreateDefaultExecutionOptions();
    execution_options = &*opts;
  }
  if (execution_options->device_handles_size() > 1) {
    return InvalidArgument(
        "Compiling with multiple device handles is not supported. Use "
        "'Execute' instead.");
  }
  return stub_->Compile(computation, argument_shapes, *execution_options);
}

StatusOr<std::unique_ptr<GlobalData>> Client::Execute(
    const ExecutionHandle& handle, absl::Span<GlobalData* const> arguments,
    ExecutionProfile* execution_profile) {
  return stub_->Execute(handle, arguments, execution_profile);
}

StatusOr<std::unique_ptr<GlobalData>> Client::Execute(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const ExecutionOptions* execution_options,
    ExecutionProfile* execution_profile) {
  // Create an ExecutionOptions if necessary, or set its DeviceHandles.
  std::optional<ExecutionOptions> options_storage;
  if (!execution_options || execution_options->device_handles().empty()) {
    if (execution_options) {
      options_storage.emplace(*execution_options);
    } else {
      options_storage.emplace(CreateDefaultExecutionOptions());
    }
    execution_options = &*options_storage;

    TF_ASSIGN_OR_RETURN(auto device_handles,
                        GetDeviceHandles(/*device_count=*/1));
    TF_RET_CHECK(!device_handles.empty());
    *options_storage->add_device_handles() = std::move(device_handles[0]);
  }

  std::vector<XlaComputationInstance> computation_instances = {
      XlaComputationInstance{
          computation,
          std::vector<GlobalData*>(arguments.begin(), arguments.end()),
          *execution_options, execution_profile}};

  // Instead of invoking Compile() and Execute(), invoke
  // Service::ExecuteParallel() to execute our one computation.  Compile()
  // caches the executable forever, which isn't what we want.
  VLOG(1) << "Making ExecuteParallel request: "
          << execution_options->DebugString();
  TF_ASSIGN_OR_RETURN(auto results, ExecuteParallel(computation_instances));
  VLOG(1) << "ExecuteParallel request done.";

  // The result selection is a bit hacky, but better than assuming it is
  // device 0.
  //
  // TODO(b/118493728): Allow Execute to return one result per computation.
  for (int64_t i = 0, end = results.size(); i < end; i++) {
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(*results[i]));
    if (!ShapeUtil::IsEmptyTuple(shape)) {
      VLOG(3) << "Fetching result from device " << i << ": "
              << ShapeUtil::HumanString(shape);
      return std::move(results[i]);
    }
  }
  TF_RET_CHECK(!results.empty());
  VLOG(1) << "Defaulting to device 0 result";
  return std::move(results[0]);
}

StatusOr<std::vector<std::unique_ptr<GlobalData>>> Client::ExecuteParallel(
    absl::Span<const XlaComputationInstance> computations) {
  return stub_->ExecuteGraphParallel(computations);
}

StatusOr<std::vector<DeviceHandle>> Client::GetDeviceHandles(
    int64_t device_count) {
  if (device_count < 1) {
    return InvalidArgument("device_count must be greater than 0");
  }
  return stub_->GetDeviceHandles(device_count);
}

Status Client::Unregister(const GlobalData& data) { return stub_->Unregister(data.handle()); }

StatusOr<std::vector<std::unique_ptr<GlobalData>>> Client::DeconstructTuple(
    const GlobalData& data) {
  return stub_->DeconstructTuple(data);
}

StatusOr<ComputationStats> Client::GetComputationStats(
    const XlaComputation& computation,
    const DebugOptions& debug_options) const {
  ComputationGraphStatsRequest request;

  // TODO(b/74197823): Find a way to avoid the copy of the hlo proto.
  *request.mutable_computation() = computation.proto();
  *request.mutable_debug_options() = debug_options;
  ComputationStatsResponse response;

  VLOG(1) << "making computation graph stats request";
  Status s = stub_->GetComputationGraphStats(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  CHECK(response.has_stats());
  return response.stats();
}

StatusOr<std::unique_ptr<ProgramShape>> Client::GetComputationShape(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(const auto& result, computation.GetProgramShape());
  return std::make_unique<ProgramShape>(result);
}

StatusOr<Shape> Client::GetShape(const GlobalData& data) { return stub_->GetShape(data); }

StatusOr<std::string> Client::ExecutionStatsAsString(
    const XlaComputation& computation, const ExecutionProfile& profile) {
  TF_ASSIGN_OR_RETURN(
      auto computation_stats,
      GetComputationStats(computation, GetDebugOptionsFromFlags()));
  int64_t total_flops =
      computation_stats.flop_count() + computation_stats.transcendental_count();
  if (profile.compute_time_ns() > 0) {
    int64_t nanoseconds = profile.compute_time_ns();
    int64_t cycle_count = profile.compute_cycle_count();
    double gflops = total_flops / nanoseconds;
    return absl::StrCat(
        "[Execution Statistics] flop count: ", computation_stats.flop_count(),
        ", transcendental count: ", computation_stats.transcendental_count(),
        ", compute execution time: ", nanoseconds, " nsec",
        ", compute cycles: ", cycle_count, ", performance: ", gflops,
        "gflop/s");
  }
  return std::string("[Execution Statistics] not available.");
}

StatusOr<ChannelHandle> Client::CreateChannelHandleByType(
    ChannelHandle::ChannelType type) {
  return stub_->CreateChannelHandle(type);
}

StatusOr<ChannelHandle> Client::CreateChannelHandle() {
  return CreateChannelHandleByType(ChannelHandle::DEVICE_TO_DEVICE);
}

StatusOr<ChannelHandle> Client::CreateHostToDeviceChannelHandle() {
  return CreateChannelHandleByType(ChannelHandle::HOST_TO_DEVICE);
}

StatusOr<ChannelHandle> Client::CreateDeviceToHostChannelHandle() {
  return CreateChannelHandleByType(ChannelHandle::DEVICE_TO_HOST);
}

}  // namespace xla
