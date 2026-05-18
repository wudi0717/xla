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

#include "musa_fusion_custom_calls.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "musa_runtime.h"
#include "mudnn.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/runtime/tracing.h"

namespace xla {
namespace gpu {
namespace {

constexpr char kMusaPlatformName[] = "MUSA";

struct LayerNormContract {
  std::string shape;
  std::string dtype;
  int64_t axis = 0;
  double epsilon = 0.0;
  int64_t workspace = 0;
};

absl::Status ParseLayerNormContract(absl::string_view opaque,
                                    LayerNormContract* contract) {
  if (opaque.empty()) {
    return absl::InvalidArgumentError("empty layernorm contract");
  }

  for (absl::string_view item : absl::StrSplit(opaque, ';')) {
    std::pair<absl::string_view, absl::string_view> kv =
        absl::StrSplit(item, absl::MaxSplits('=', 1));
    if (kv.first.empty() || kv.second.empty()) {
      continue;
    }
    if (kv.first == "shape") {
      contract->shape = std::string(kv.second);
    } else if (kv.first == "dtype") {
      contract->dtype = std::string(kv.second);
    } else if (kv.first == "axis") {
      if (!absl::SimpleAtoi(kv.second, &contract->axis)) {
        return absl::InvalidArgumentError("invalid axis in layernorm contract");
      }
    } else if (kv.first == "epsilon") {
      if (!absl::SimpleAtod(kv.second, &contract->epsilon)) {
        return absl::InvalidArgumentError(
            "invalid epsilon in layernorm contract");
      }
    } else if (kv.first == "workspace") {
      if (!absl::SimpleAtoi(kv.second, &contract->workspace)) {
        return absl::InvalidArgumentError(
            "invalid workspace in layernorm contract");
      }
    }
  }

  if (contract->shape.empty() || contract->dtype.empty()) {
    return absl::InvalidArgumentError(
        "missing shape/dtype in layernorm contract");
  }
  if (contract->workspace < 0) {
    return absl::InvalidArgumentError(
        "workspace in layernorm contract must be >= 0");
  }
  return absl::OkStatus();
}

absl::Status ParseShape(absl::string_view shape,
                        std::vector<int64_t>* dims) {
  if (shape.empty()) {
    return absl::InvalidArgumentError("shape must not be empty");
  }
  for (absl::string_view dim : absl::StrSplit(shape, ',')) {
    int64_t value = 0;
    if (!absl::SimpleAtoi(dim, &value) || value <= 0) {
      return absl::InvalidArgumentError("invalid dimension in shape");
    }
    dims->push_back(value);
  }
  return absl::OkStatus();
}

std::vector<int64_t> MakeContiguousStrides(absl::Span<const int64_t> dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

absl::StatusOr<::musa::dnn::Tensor::Type> ParseTensorType(
    absl::string_view dtype) {
  if (dtype == "f32") {
    return ::musa::dnn::Tensor::Type::FLOAT;
  }
  if (dtype == "f16") {
    return ::musa::dnn::Tensor::Type::HALF;
  }
  if (dtype == "bf16") {
    return ::musa::dnn::Tensor::Type::BFLOAT16;
  }
  return absl::UnimplementedError(
      absl::StrCat("unsupported layernorm dtype: ", dtype));
}

size_t TensorTypeByteSize(::musa::dnn::Tensor::Type type) {
  switch (type) {
    case ::musa::dnn::Tensor::Type::HALF:
    case ::musa::dnn::Tensor::Type::BFLOAT16:
      return 2;
    case ::musa::dnn::Tensor::Type::FLOAT:
      return 4;
    case ::musa::dnn::Tensor::Type::DOUBLE:
      return 8;
    default:
      return 0;
  }
}

absl::Status ToStatus(::musa::dnn::Status status, absl::string_view op_name) {
  if (status == ::musa::dnn::Status::SUCCESS) {
    return absl::OkStatus();
  }
  return absl::InternalError(
      absl::StrCat(op_name, " failed with muDNN status ",
                   static_cast<int>(status)));
}

absl::Status BuildTensor(const void* addr, ::musa::dnn::Tensor::Type type,
                         absl::Span<const int64_t> dims,
                         ::musa::dnn::Tensor* tensor) {
  if (dims.empty()) {
    return absl::InvalidArgumentError("tensor dims must not be empty");
  }
  std::vector<int64_t> strides = MakeContiguousStrides(dims);
  if (auto status = ToStatus(tensor->SetAddr(addr), "Tensor::SetAddr");
      !status.ok()) {
    return status;
  }
  if (auto status = ToStatus(tensor->SetType(type), "Tensor::SetType");
      !status.ok()) {
    return status;
  }
  return ToStatus(tensor->SetNdInfo(static_cast<int64_t>(dims.size()),
                                    dims.data(), strides.data()),
                  "Tensor::SetNdInfo");
}

int64_t NumElements(absl::Span<const int64_t> dims) {
  return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

class ScopedMusaAllocation {
 public:
  ScopedMusaAllocation() = default;
  ~ScopedMusaAllocation() {
    if (ptr_ != nullptr) {
      musaFree(ptr_);
    }
  }

  absl::Status Allocate(size_t bytes) {
    if (bytes == 0) {
      ptr_ = nullptr;
      return absl::OkStatus();
    }
    musaError_t err = musaMalloc(&ptr_, bytes);
    if (err != musaSuccess) {
      return absl::InternalError(
          absl::StrCat("musaMalloc failed with code ", static_cast<int>(err)));
    }
    return absl::OkStatus();
  }

  void* ptr() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

void SetFailure(const char* target, XlaCustomCallStatus* status,
                const absl::Status& failure_status) {
  const std::string message = std::string(failure_status.message());
  XlaCustomCallStatusSetFailure(status, message.data(), message.size());
  if (IsMusaDebugRuntimeTraceEnabled()) {
    LogMusaRuntimeCustomCallEnd(target, failure_status);
  }
}

void SetNotImplementedFailure(const char* target, XlaCustomCallStatus* status) {
  const char* message = "MUSA fusion custom call is registered but not implemented yet";
  XlaCustomCallStatusSetFailure(status, message, std::strlen(message));
  if (IsMusaDebugRuntimeTraceEnabled()) {
    LogMusaRuntimeCustomCallEnd(target, absl::UnimplementedError(message));
  }
}

void MusaLayerNormCustomCall(void* stream_handle, void** buffers,
                             const char* opaque, std::size_t opaque_len,
                             XlaCustomCallStatus* status) {
  absl::string_view opaque_view(opaque, opaque_len);
  LayerNormContract contract;
  absl::Status contract_status =
      ParseLayerNormContract(opaque_view, &contract);
  if (!contract_status.ok()) {
    const absl::Status failure_status = absl::InvalidArgumentError(
        absl::StrCat("invalid __musa$layernorm contract: ",
                     contract_status.message()));
    SetFailure(kMusaLayerNormCustomCallTarget, status, failure_status);
    return;
  }

  if (IsMusaDebugRuntimeTraceEnabled()) {
    LogMusaRuntimeCustomCallBegin(kMusaLayerNormCustomCallTarget,
                                  std::string(opaque_view));
  }

  if (stream_handle == nullptr || buffers == nullptr) {
    SetFailure(kMusaLayerNormCustomCallTarget, status,
               absl::InvalidArgumentError("null stream or buffers"));
    return;
  }

  std::vector<int64_t> x_dims;
  if (absl::Status parse_shape_status = ParseShape(contract.shape, &x_dims);
      !parse_shape_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status,
               absl::InvalidArgumentError(
                   absl::StrCat("invalid shape in layernorm contract: ",
                                parse_shape_status.message())));
    return;
  }

  if (contract.axis < 0 || contract.axis >= static_cast<int64_t>(x_dims.size())) {
    SetFailure(kMusaLayerNormCustomCallTarget, status,
               absl::InvalidArgumentError("axis out of range for layernorm shape"));
    return;
  }

  absl::StatusOr<::musa::dnn::Tensor::Type> tensor_type_or =
      ParseTensorType(contract.dtype);
  if (!tensor_type_or.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, tensor_type_or.status());
    return;
  }
  const ::musa::dnn::Tensor::Type tensor_type = *tensor_type_or;
  const size_t element_bytes = TensorTypeByteSize(tensor_type);
  if (element_bytes == 0) {
    SetFailure(kMusaLayerNormCustomCallTarget, status,
               absl::UnimplementedError("unsupported element size for layernorm dtype"));
    return;
  }

  void* x_ptr = buffers[0];
  void* gamma_ptr = buffers[1];
  void* beta_ptr = buffers[2];
  void* y_ptr = buffers[3];
  if (x_ptr == nullptr || gamma_ptr == nullptr || beta_ptr == nullptr ||
      y_ptr == nullptr) {
    SetFailure(kMusaLayerNormCustomCallTarget, status,
               absl::InvalidArgumentError(
                   "layernorm expects non-null x/gamma/beta/y buffers"));
    return;
  }

  std::vector<int64_t> norm_dims(x_dims.begin() + contract.axis, x_dims.end());
  std::vector<int64_t> stats_dims(x_dims.begin(), x_dims.begin() + contract.axis);
  if (stats_dims.empty()) {
    stats_dims.push_back(1);
  }

  const int64_t stats_elements = NumElements(stats_dims);
  ScopedMusaAllocation mean_alloc;
  ScopedMusaAllocation inv_var_alloc;
  if (absl::Status alloc_status = mean_alloc.Allocate(stats_elements * element_bytes);
      !alloc_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, alloc_status);
    return;
  }
  if (absl::Status alloc_status =
          inv_var_alloc.Allocate(stats_elements * element_bytes);
      !alloc_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, alloc_status);
    return;
  }

  int device_id = -1;
  if (musaError_t err = musaGetDevice(&device_id); err != musaSuccess) {
    SetFailure(kMusaLayerNormCustomCallTarget, status,
               absl::InternalError(absl::StrCat(
                   "musaGetDevice failed with code ", static_cast<int>(err))));
    return;
  }

  ::musa::dnn::Handle handle(device_id);
  if (absl::Status set_stream_status = ToStatus(
          handle.SetStream(static_cast<musaStream_t>(stream_handle)),
          "Handle::SetStream");
      !set_stream_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, set_stream_status);
    return;
  }

  ::musa::dnn::Tensor x_tensor;
  ::musa::dnn::Tensor gamma_tensor;
  ::musa::dnn::Tensor beta_tensor;
  ::musa::dnn::Tensor y_tensor;
  ::musa::dnn::Tensor mean_tensor;
  ::musa::dnn::Tensor inv_var_tensor;

  if (absl::Status build_status =
          BuildTensor(x_ptr, tensor_type, x_dims, &x_tensor);
      !build_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, build_status);
    return;
  }
  if (absl::Status build_status =
          BuildTensor(gamma_ptr, tensor_type, norm_dims, &gamma_tensor);
      !build_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, build_status);
    return;
  }
  if (absl::Status build_status =
          BuildTensor(beta_ptr, tensor_type, norm_dims, &beta_tensor);
      !build_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, build_status);
    return;
  }
  if (absl::Status build_status =
          BuildTensor(y_ptr, tensor_type, x_dims, &y_tensor);
      !build_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, build_status);
    return;
  }
  if (absl::Status build_status =
          BuildTensor(mean_alloc.ptr(), tensor_type, stats_dims, &mean_tensor);
      !build_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, build_status);
    return;
  }
  if (absl::Status build_status = BuildTensor(inv_var_alloc.ptr(), tensor_type,
                                              stats_dims, &inv_var_tensor);
      !build_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, build_status);
    return;
  }

  ::musa::dnn::LayerNorm layer_norm;
  if (absl::Status epsilon_status =
          ToStatus(layer_norm.SetEpsilon(contract.epsilon),
                   "LayerNorm::SetEpsilon");
      !epsilon_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, epsilon_status);
    return;
  }
  const int axis = static_cast<int>(contract.axis);
  if (absl::Status axis_status =
          ToStatus(layer_norm.SetAxis(1, &axis), "LayerNorm::SetAxis");
      !axis_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, axis_status);
    return;
  }
  if (absl::Status run_status = ToStatus(
          layer_norm.Run(handle, y_tensor, mean_tensor, inv_var_tensor,
                         x_tensor, gamma_tensor, beta_tensor),
          "LayerNorm::Run");
      !run_status.ok()) {
    SetFailure(kMusaLayerNormCustomCallTarget, status, run_status);
    return;
  }

  XlaCustomCallStatusSetSuccess(status);
  if (IsMusaDebugRuntimeTraceEnabled()) {
    LogMusaRuntimeCustomCallEnd(kMusaLayerNormCustomCallTarget,
                                absl::OkStatus());
  }
}

void MusaMatmulBiasReluCustomCall(void* stream_handle, void** buffers,
                                  const char* opaque, std::size_t opaque_len,
                                  XlaCustomCallStatus* status) {
  (void)stream_handle;
  (void)buffers;
  (void)opaque;
  (void)opaque_len;
  if (IsMusaDebugRuntimeTraceEnabled()) {
    LogMusaRuntimeCustomCallBegin(kMusaMatmulBiasReluCustomCallTarget,
                                  "P1 matmul+bias+relu placeholder implementation");
  }
  SetNotImplementedFailure(kMusaMatmulBiasReluCustomCallTarget, status);
}

}  // namespace

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(std::string(kMusaLayerNormCustomCallTarget),
                                         MusaLayerNormCustomCall,
                                         kMusaPlatformName);

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    std::string(kMusaMatmulBiasReluCustomCallTarget),
    MusaMatmulBiasReluCustomCall, kMusaPlatformName);

}  // namespace gpu
}  // namespace xla
