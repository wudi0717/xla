#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_

#include <cstdint>
#include <memory>

#include "xla/stream_executor/dnn.h"

namespace stream_executor {
namespace musa {

class MusaExecutor;

class MusaDnnSupport : public dnn::DnnSupport {
 public:
  explicit MusaDnnSupport(MusaExecutor* parent);

  tsl::Status Init() override;
  tsl::StatusOr<dnn::VersionInfo> GetVersion() override;

  tsl::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
  ConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType element_type,
      dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor) override;

  tsl::Status DoConvolve(
      dnn::ConvolutionKind kind, dnn::DataType element_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8_t> scratch_memory,
      dnn::ProfileResult* output_profile_result) override;

  bool DoConvolveQuantized(
      Stream*, const dnn::BatchDescriptor&, const DeviceMemory<float>&,
      const dnn::FilterDescriptor&, const DeviceMemory<int8_t>&,
      const DeviceMemory<float>&, const dnn::ConvolutionDescriptor&,
      const dnn::BatchDescriptor&, DeviceMemory<float>*) override;
  bool DoConvolveQuantized(
      Stream*, const dnn::BatchDescriptor&, const DeviceMemory<float>&,
      const dnn::FilterDescriptor&, const DeviceMemory<int16>&,
      const DeviceMemory<float>&, const dnn::ConvolutionDescriptor&,
      const dnn::BatchDescriptor&, DeviceMemory<float>*) override;
  bool DoSeparableConvolve(
      Stream*, const dnn::BatchDescriptor&, const DeviceMemory<float>&,
      const dnn::FilterDescriptor&, int, const DeviceMemory<float>&,
      const DeviceMemory<float>&, const dnn::ConvolutionDescriptor&,
      const dnn::BatchDescriptor&, DeviceMemory<float>*) override;
  bool DoMatMul(Stream*, const DeviceMemory<float>&,
                const DeviceMemory<float>&, const dnn::BatchDescriptor&,
                const dnn::BatchDescriptor&, DeviceMemory<float>*) override;
  bool DoMatMulQuantized(
      Stream*, const DeviceMemory<float>&, const DeviceMemory<int8_t>&,
      const DeviceMemory<float>&, const dnn::BatchDescriptor&,
      const dnn::BatchDescriptor&, DeviceMemory<float>*) override;
  bool DoMatMulQuantized(
      Stream*, const DeviceMemory<float>&, const DeviceMemory<int16>&,
      const DeviceMemory<float>&, const dnn::BatchDescriptor&,
      const dnn::BatchDescriptor&, DeviceMemory<float>*) override;
  bool DoBiasAdd(Stream*, const DeviceMemory<float>&,
                 const DeviceMemory<float>&, const dnn::BatchDescriptor&,
                 DeviceMemory<float>*) override;
  tsl::Status DoPoolForward(
      dnn::DataType, Stream*, const dnn::PoolingDescriptor&,
      const dnn::BatchDescriptor&, DeviceMemoryBase,
      const dnn::BatchDescriptor&, DeviceMemoryBase, ScratchAllocator*) override;
  tsl::Status DoPoolBackward(
      dnn::DataType, Stream*, const dnn::PoolingDescriptor&,
      const dnn::BatchDescriptor&, DeviceMemoryBase,
      const dnn::BatchDescriptor&, DeviceMemoryBase, DeviceMemoryBase,
      DeviceMemoryBase, ScratchAllocator*) override;
  bool DoDepthConcatenate(
      Stream*, absl::Span<const dnn::BatchDescriptor>,
      absl::Span<const DeviceMemory<float>* const>,
      DeviceMemory<float>*) override;
  bool DoElementwiseOperate(
      Stream*, dnn::ElementwiseOperation,
      absl::Span<const dnn::BatchDescriptor>,
      absl::Span<const DeviceMemory<float>* const>,
      const dnn::BatchDescriptor&, DeviceMemory<float>*) override;
  bool DoXYPad(Stream*, const dnn::BatchDescriptor&,
               const DeviceMemory<float>&, int64_t, int64_t, int64_t, int64_t,
               DeviceMemory<float>*) override;
  bool DoXYSlice(Stream*, const dnn::BatchDescriptor&,
                 const DeviceMemory<float>&, int64_t, int64_t, int64_t,
                 int64_t, DeviceMemory<float>*) override;
  bool DoMemcpyD2HQuantized(Stream*, const DeviceMemory<float>&,
                            dnn::QuantizedActivationMode, void*,
                            int64_t) override;
  bool DoMemcpyH2DQuantized(Stream*, const void*, int64_t,
                            dnn::QuantizedActivationMode,
                            DeviceMemory<float>*) override;

 private:
  MusaExecutor* parent_;
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_
