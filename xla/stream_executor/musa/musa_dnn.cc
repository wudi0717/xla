#include "xla/stream_executor/musa/musa_dnn.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mudnn.h"
#include "mudnn_version.h"
#include "musa_runtime.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "xla/stream_executor/musa/musa_executor.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace musa {
namespace {

tsl::Status ToStatus(::musa::dnn::Status status, const char* operation) {
  if (status == ::musa::dnn::Status::SUCCESS) {
    return tsl::OkStatus();
  }
  return tsl::errors::Internal(operation, " failed with muDNN status ",
                               static_cast<int>(status));
}

tsl::Status Unsupported(const char* operation) {
  return tsl::errors::Unimplemented(operation,
                                    " is not implemented for MUSA DNN");
}

tsl::Status ValidateDescriptorSupport(
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input,
    const dnn::FilterDescriptor& filter, const dnn::BatchDescriptor& output,
    const dnn::ConvolutionDescriptor& convolution) {
  if (kind != dnn::ConvolutionKind::FORWARD) {
    return tsl::errors::Unimplemented(
        "MUSA DNN currently supports forward convolution only");
  }
  if (element_type != dnn::DataType::kFloat ||
      output_type != dnn::DataType::kFloat) {
    return tsl::errors::Unimplemented(
        "MUSA DNN forward convolution currently supports f32 only");
  }
  if (input.ndims() != 2 || filter.ndims() != 2 || output.ndims() != 2 ||
      convolution.ndims() != 2) {
    return tsl::errors::Unimplemented(
        "MUSA DNN forward convolution currently supports 2D tensors only");
  }
  if (input.layout() != dnn::DataLayout::kBatchDepthYX ||
      output.layout() != dnn::DataLayout::kBatchDepthYX ||
      filter.layout() != dnn::FilterLayout::kOutputInputYX) {
    return tsl::errors::Unimplemented(
        "MUSA DNN forward convolution currently supports NCHW/OIHW only");
  }
  if (convolution.convolution_not_crosscorr()) {
    return tsl::errors::Unimplemented(
        "MUSA DNN forward convolution currently supports cross-correlation "
        "only");
  }
  return tsl::OkStatus();
}

std::vector<int> ToIntVector(absl::Span<const int64_t> values) {
  std::vector<int> result;
  result.reserve(values.size());
  for (int64_t value : values) {
    result.push_back(static_cast<int>(value));
  }
  return result;
}

tsl::Status ConfigureTensor(const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            const void* address, ::musa::dnn::Tensor* tensor) {
  TF_RETURN_IF_ERROR(ToStatus(tensor->SetAddr(address), "Tensor::SetAddr"));
  TF_RETURN_IF_ERROR(ToStatus(tensor->SetType(::musa::dnn::Tensor::Type::FLOAT),
                                    "Tensor::SetType"));
  return ToStatus(
      tensor->SetNdInfo(static_cast<int64_t>(dims.size()), dims.data(),
                        strides.data()),
      "Tensor::SetNdInfo");
}

::musa::dnn::Convolution::Algorithm ToMudnnAlgorithm(int64_t algorithm_id) {
  using Algorithm = ::musa::dnn::Convolution::Algorithm;
  switch (algorithm_id) {
    case 1:
      return Algorithm::DIRECT;
    case 2:
      return Algorithm::WINOGRAD_NONFUSED;
    case 3:
      return Algorithm::GEMM;
    default:
      return Algorithm::IMPLICIT_GEMM;
  }
}

class MusaConvRunner : public dnn::ConvRunner {
 public:
  static tsl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> Create(
      MusaExecutor* parent, Stream* stream,
      const dnn::AlgorithmDesc& algorithm_desc, dnn::ConvolutionKind kind,
      dnn::DataType element_type, dnn::DataType output_type,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor) {
    TF_RETURN_IF_ERROR(ValidateDescriptorSupport(
        kind, element_type, output_type, input_descriptor, filter_descriptor,
        output_descriptor, convolution_descriptor));
    auto runner = std::unique_ptr<MusaConvRunner>(new MusaConvRunner(
        parent, algorithm_desc, input_descriptor, filter_descriptor,
        output_descriptor, convolution_descriptor));
    TF_RETURN_IF_ERROR(runner->Init(stream));
    return std::unique_ptr<const dnn::ConvRunner>(std::move(runner));
  }

  ~MusaConvRunner() override {
    for (auto& [stream_handle, workspace] : internal_workspaces_) {
      (void)stream_handle;
      parent_->Deallocate(&workspace);
    }
  }

  std::string ToString() const override {
    return absl::StrCat("muDNN convolution algorithm=", algorithm_desc_.algo_id(),
                        " workspace=", workspace_size_);
  }

  size_t GetWorkspaceSize() const override { return workspace_size_; }

  tsl::StatusOr<dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return dnn::AlgorithmDesc(algorithm_desc_.algo_id(),
                              algorithm_desc_.tensor_ops_enabled(),
                              workspace_size_);
  }

  tsl::Status operator()(Stream* stream, dnn::ProfileResult* profile_result,
                         DeviceMemoryBase scratch_memory,
                         DeviceMemoryBase input_data,
                         DeviceMemoryBase filter_data,
                         DeviceMemoryBase output_data) const override {
    absl::MutexLock lock(&mutex_);
    auto activation = parent_->ActivateContext();
    TF_RETURN_IF_ERROR(activation.status());
    TF_RETURN_IF_ERROR(BindStream(stream));

    ::musa::dnn::Tensor input;
    ::musa::dnn::Tensor filter;
    ::musa::dnn::Tensor output;
    TF_RETURN_IF_ERROR(ConfigureTensor(input_dims_, input_strides_,
                                      input_data.opaque(), &input));
    TF_RETURN_IF_ERROR(ConfigureTensor(filter_dims_, filter_strides_,
                                      filter_data.opaque(), &filter));
    TF_RETURN_IF_ERROR(ConfigureTensor(output_dims_, output_strides_,
                                      output_data.opaque(), &output));

    MUstream musa_stream = GetMusaStreamHandle(stream);
    DeviceMemoryBase* internal_workspace = nullptr;
    bool workspace_fits = true;
    bool workspace_allocation_failed = false;
    size_t requested_workspace_bytes = 0;
    ::musa::dnn::MemoryMaintainer maintainer =
        [&](size_t bytes) -> ::musa::dnn::MemoryHandler {
      requested_workspace_bytes = bytes;
      workspace_size_ = std::max(workspace_size_, bytes);
      if (bytes == 0) {
        return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      }
      if (bytes <= scratch_memory.size() && !scratch_memory.is_null()) {
        return ::musa::dnn::MemoryHandler(scratch_memory.opaque(),
                                          [](void*) {});
      }
      auto [it, inserted] = internal_workspaces_.try_emplace(musa_stream);
      if (inserted) {
        it->second = parent_->Allocate(bytes, /*memory_space=*/0);
        if (it->second.is_null()) {
          internal_workspaces_.erase(it);
          workspace_allocation_failed = true;
          return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
        }
      }
      internal_workspace = &it->second;
      if (bytes <= internal_workspace->size()) {
        return ::musa::dnn::MemoryHandler(internal_workspace->opaque(),
                                          [](void*) {});
      }
      workspace_fits = false;
      return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    };
    const auto run_status = [&] {
      absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
      return convolution_.Run(*handle_, output, input, filter, algorithm_,
                              maintainer);
    }();
    if (workspace_allocation_failed) {
      return tsl::errors::ResourceExhausted(
          "Unable to allocate ", requested_workspace_bytes,
          " bytes for the muDNN convolution workspace");
    }
    if (!workspace_fits) {
      const size_t internal_workspace_size =
          internal_workspace == nullptr ? 0 : internal_workspace->size();
      return tsl::errors::ResourceExhausted(
          "muDNN convolution requested ", requested_workspace_bytes,
          " bytes, exceeding both XLA scratch (", scratch_memory.size(),
          ") and the internal allocation (",
          internal_workspace_size, ") bytes");
    }
    TF_RETURN_IF_ERROR(ToStatus(run_status, "Convolution::Run"));
    if (profile_result != nullptr) {
      profile_result->set_algorithm(
          dnn::AlgorithmDesc(algorithm_desc_.algo_id(),
                             algorithm_desc_.tensor_ops_enabled(),
                             workspace_size_));
      profile_result->set_scratch_size(workspace_size_);
    }
    return tsl::OkStatus();
  }

 private:
  MusaConvRunner(MusaExecutor* parent,
                 const dnn::AlgorithmDesc& algorithm_desc,
                 const dnn::BatchDescriptor& input_descriptor,
                 const dnn::FilterDescriptor& filter_descriptor,
                 const dnn::BatchDescriptor& output_descriptor,
                 const dnn::ConvolutionDescriptor& convolution_descriptor)
      : parent_(parent),
        algorithm_desc_(algorithm_desc),
        algorithm_(ToMudnnAlgorithm(algorithm_desc.algo_id())),
        input_dims_(input_descriptor.full_dims(
            dnn::DataLayout::kBatchDepthYX)),
        input_strides_(input_descriptor.full_strides(
            dnn::DataLayout::kBatchDepthYX)),
        filter_dims_(filter_descriptor.full_dims(
            dnn::FilterLayout::kOutputInputYX)),
        filter_strides_(filter_descriptor.full_strides(
            dnn::FilterLayout::kOutputInputYX)),
        output_dims_(output_descriptor.full_dims(
            dnn::DataLayout::kBatchDepthYX)),
        output_strides_(output_descriptor.full_strides(
            dnn::DataLayout::kBatchDepthYX)),
        padding_(ToIntVector(convolution_descriptor.padding())),
        strides_(ToIntVector(convolution_descriptor.strides())),
        dilations_(ToIntVector(convolution_descriptor.dilations())),
        group_count_(convolution_descriptor.group_count()) {}

  tsl::Status Init(Stream* stream) {
    auto activation = parent_->ActivateContext();
    TF_RETURN_IF_ERROR(activation.status());
    int device = -1;
    const musaError_t device_status = musaGetDevice(&device);
    if (device_status != musaSuccess) {
      return tsl::errors::Internal("musaGetDevice failed with status ",
                                   static_cast<int>(device_status));
    }
    handle_ = std::make_unique<::musa::dnn::Handle>(device);
    TF_RETURN_IF_ERROR(BindStream(stream));
    TF_RETURN_IF_ERROR(ToStatus(
        convolution_.SetNdInfo(static_cast<int>(padding_.size()),
                               padding_.data(), strides_.data(),
                               dilations_.data()),
        "Convolution::SetNdInfo"));
    TF_RETURN_IF_ERROR(ToStatus(convolution_.SetGroups(group_count_),
                               "Convolution::SetGroups"));

    ::musa::dnn::Tensor input;
    ::musa::dnn::Tensor filter;
    ::musa::dnn::Tensor output;
    TF_RETURN_IF_ERROR(
        ConfigureTensor(input_dims_, input_strides_, nullptr, &input));
    TF_RETURN_IF_ERROR(
        ConfigureTensor(filter_dims_, filter_strides_, nullptr, &filter));
    TF_RETURN_IF_ERROR(
        ConfigureTensor(output_dims_, output_strides_, nullptr, &output));
    TF_RETURN_IF_ERROR(ToStatus(convolution_.GetForwardWorkspaceSize(
                                    *handle_, workspace_size_, output, input,
                                    filter, algorithm_),
                                "Convolution::GetForwardWorkspaceSize"));
    return tsl::OkStatus();
  }

  tsl::Status BindStream(Stream* stream) const {
    if (stream == nullptr) {
      return tsl::errors::InvalidArgument("muDNN convolution stream is null");
    }
    if (stream->parent()->implementation() != parent_) {
      return tsl::errors::InvalidArgument(
          "muDNN convolution stream belongs to a different StreamExecutor");
    }
    MUstream musa_stream = GetMusaStreamHandle(stream);
    if (musa_stream == nullptr) {
      return tsl::errors::InvalidArgument(
          "muDNN convolution stream has a null MUSA handle");
    }
    return ToStatus(handle_->SetStream(static_cast<musaStream_t>(musa_stream)),
                    "Handle::SetStream");
  }

  MusaExecutor* parent_;
  dnn::AlgorithmDesc algorithm_desc_;
  ::musa::dnn::Convolution::Algorithm algorithm_;
  std::vector<int64_t> input_dims_;
  std::vector<int64_t> input_strides_;
  std::vector<int64_t> filter_dims_;
  std::vector<int64_t> filter_strides_;
  std::vector<int64_t> output_dims_;
  std::vector<int64_t> output_strides_;
  std::vector<int> padding_;
  std::vector<int> strides_;
  std::vector<int> dilations_;
  int group_count_;
  mutable absl::Mutex mutex_;
  mutable std::unique_ptr<::musa::dnn::Handle> handle_;
  mutable ::musa::dnn::Convolution convolution_;
  mutable std::map<MUstream, DeviceMemoryBase> internal_workspaces_;
  mutable size_t workspace_size_ = 0;
};

}  // namespace

MusaDnnSupport::MusaDnnSupport(MusaExecutor* parent)
    : parent_(CHECK_NOTNULL(parent)) {}

tsl::Status MusaDnnSupport::Init() {
  auto activation = parent_->ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  int device = -1;
  const musaError_t status = musaGetDevice(&device);
  if (status != musaSuccess) {
    return tsl::errors::Internal("musaGetDevice failed with status ",
                                 static_cast<int>(status));
  }
  return tsl::OkStatus();
}

tsl::StatusOr<dnn::VersionInfo> MusaDnnSupport::GetVersion() {
  return dnn::VersionInfo(MUDNN_VERSION_MAJOR, MUDNN_VERSION_MINOR,
                          MUDNN_VERSION_PATCH);
}

tsl::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
MusaDnnSupport::ConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor) {
  return MusaConvRunner::Create(parent_, stream, algorithm_desc, kind,
                                element_type, output_type, input_descriptor,
                                filter_descriptor, output_descriptor,
                                convolution_descriptor);
}

tsl::Status MusaDnnSupport::DoConvolve(
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8_t> scratch_memory,
    dnn::ProfileResult* output_profile_result) {
  TF_ASSIGN_OR_RETURN(
      auto runner,
      ConvolveRunnerFromDesc(stream, algorithm_desc, kind, element_type,
                             output_type, input_descriptor, filter_descriptor,
                             output_descriptor, convolution_descriptor));
  return (*runner)(stream, output_profile_result, scratch_memory, input_data,
                   filter_data, output_data);
}

#define MUSA_DNN_UNSUPPORTED_BOOL(method, signature) \
  bool MusaDnnSupport::method signature { return false; }

MUSA_DNN_UNSUPPORTED_BOOL(
    DoConvolveQuantized,
    (Stream*, const dnn::BatchDescriptor&, const DeviceMemory<float>&,
     const dnn::FilterDescriptor&, const DeviceMemory<int8_t>&,
     const DeviceMemory<float>&, const dnn::ConvolutionDescriptor&,
     const dnn::BatchDescriptor&, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoConvolveQuantized,
    (Stream*, const dnn::BatchDescriptor&, const DeviceMemory<float>&,
     const dnn::FilterDescriptor&, const DeviceMemory<int16>&,
     const DeviceMemory<float>&, const dnn::ConvolutionDescriptor&,
     const dnn::BatchDescriptor&, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoSeparableConvolve,
    (Stream*, const dnn::BatchDescriptor&, const DeviceMemory<float>&,
     const dnn::FilterDescriptor&, int, const DeviceMemory<float>&,
     const DeviceMemory<float>&, const dnn::ConvolutionDescriptor&,
     const dnn::BatchDescriptor&, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoMatMul,
    (Stream*, const DeviceMemory<float>&, const DeviceMemory<float>&,
     const dnn::BatchDescriptor&, const dnn::BatchDescriptor&,
     DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoMatMulQuantized,
    (Stream*, const DeviceMemory<float>&, const DeviceMemory<int8_t>&,
     const DeviceMemory<float>&, const dnn::BatchDescriptor&,
     const dnn::BatchDescriptor&, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoMatMulQuantized,
    (Stream*, const DeviceMemory<float>&, const DeviceMemory<int16>&,
     const DeviceMemory<float>&, const dnn::BatchDescriptor&,
     const dnn::BatchDescriptor&, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoBiasAdd,
    (Stream*, const DeviceMemory<float>&, const DeviceMemory<float>&,
     const dnn::BatchDescriptor&, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoDepthConcatenate,
    (Stream*, absl::Span<const dnn::BatchDescriptor>,
     absl::Span<const DeviceMemory<float>* const>, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoElementwiseOperate,
    (Stream*, dnn::ElementwiseOperation,
     absl::Span<const dnn::BatchDescriptor>,
     absl::Span<const DeviceMemory<float>* const>,
     const dnn::BatchDescriptor&, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoXYPad,
    (Stream*, const dnn::BatchDescriptor&, const DeviceMemory<float>&, int64_t,
     int64_t, int64_t, int64_t, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoXYSlice,
    (Stream*, const dnn::BatchDescriptor&, const DeviceMemory<float>&, int64_t,
     int64_t, int64_t, int64_t, DeviceMemory<float>*))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoMemcpyD2HQuantized,
    (Stream*, const DeviceMemory<float>&, dnn::QuantizedActivationMode, void*,
     int64_t))
MUSA_DNN_UNSUPPORTED_BOOL(
    DoMemcpyH2DQuantized,
    (Stream*, const void*, int64_t, dnn::QuantizedActivationMode,
     DeviceMemory<float>*))

#undef MUSA_DNN_UNSUPPORTED_BOOL

tsl::Status MusaDnnSupport::DoPoolForward(
    dnn::DataType, Stream*, const dnn::PoolingDescriptor&,
    const dnn::BatchDescriptor&, DeviceMemoryBase,
    const dnn::BatchDescriptor&, DeviceMemoryBase, ScratchAllocator*) {
  return Unsupported("DoPoolForward");
}

tsl::Status MusaDnnSupport::DoPoolBackward(
    dnn::DataType, Stream*, const dnn::PoolingDescriptor&,
    const dnn::BatchDescriptor&, DeviceMemoryBase,
    const dnn::BatchDescriptor&, DeviceMemoryBase, DeviceMemoryBase,
    DeviceMemoryBase, ScratchAllocator*) {
  return Unsupported("DoPoolBackward");
}

}  // namespace musa

void initialize_mudnn() {
  tsl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
          musa::kMusaPlatformId, "muDNN",
          [](internal::StreamExecutorInterface* parent) -> dnn::DnnSupport* {
            auto* musa_executor = dynamic_cast<musa::MusaExecutor*>(parent);
            if (musa_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize muDNN with a non-MUSA "
                            "StreamExecutor";
              return nullptr;
            }
            auto dnn = std::make_unique<musa::MusaDnnSupport>(musa_executor);
            const tsl::Status init_status = dnn->Init();
            if (!init_status.ok()) {
              LOG(ERROR) << "Unable to initialize muDNN: "
                         << init_status.message();
              return nullptr;
            }
            return dnn.release();
          });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register muDNN factory: " << status.message();
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_mudnn,
                            { stream_executor::initialize_mudnn(); });
