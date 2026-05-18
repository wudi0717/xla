#include "xla/stream_executor/musa/musa_blas.h"

#include <array>
#include <complex>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "musa_bf16.h"
#include "musa_fp16.h"
#include "musa_runtime.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/util/determinism.h"
#include "xla/stream_executor/musa/mublas_wrapper.h"
#include "xla/stream_executor/musa/musa_executor.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/temporary_device_memory.h"

namespace stream_executor {
namespace musa {
namespace {

template <typename T>
struct MuBlasType {
  using type = T;
};

template <>
struct MuBlasType<std::complex<float>> {
  using type = muComplex;
};

template <>
struct MuBlasType<std::complex<double>> {
  using type = muDoubleComplex;
};

template <typename T>
using MuBlasTypeT = typename MuBlasType<T>::type;

template <typename T>
const MuBlasTypeT<T>* ComplexCast(const T* value) {
  return reinterpret_cast<const MuBlasTypeT<T>*>(value);
}

template <typename T>
MuBlasTypeT<T>* ComplexCast(T* value) {
  return reinterpret_cast<MuBlasTypeT<T>*>(value);
}

template <typename T>
const MuBlasTypeT<T>* ComplexCast(const DeviceMemory<T>& value) {
  return reinterpret_cast<const MuBlasTypeT<T>*>(value.opaque());
}

template <typename T>
MuBlasTypeT<T>* ComplexCast(DeviceMemory<T>* value) {
  return reinterpret_cast<MuBlasTypeT<T>*>(value->opaque());
}

std::string ToString(mublasStatus status) {
  switch (status) {
    case MUBLAS_STATUS_SUCCESS:
      return "MUBLAS_STATUS_SUCCESS";
    case MUBLAS_STATUS_INVALID_HANDLE:
      return "MUBLAS_STATUS_INVALID_HANDLE";
    case MUBLAS_STATUS_NOT_IMPLEMENTED:
      return "MUBLAS_STATUS_NOT_IMPLEMENTED";
    case MUBLAS_STATUS_INVALID_POINTER:
      return "MUBLAS_STATUS_INVALID_POINTER";
    case MUBLAS_STATUS_INVALID_SIZE:
      return "MUBLAS_STATUS_INVALID_SIZE";
    case MUBLAS_STATUS_MEMORY_ERROR:
      return "MUBLAS_STATUS_MEMORY_ERROR";
    case MUBLAS_STATUS_INTERNAL_ERROR:
      return "MUBLAS_STATUS_INTERNAL_ERROR";
    case MUBLAS_STATUS_PERF_DEGRADED:
      return "MUBLAS_STATUS_PERF_DEGRADED";
    case MUBLAS_STATUS_SIZE_QUERY_MISMATCH:
      return "MUBLAS_STATUS_SIZE_QUERY_MISMATCH";
    case MUBLAS_STATUS_SIZE_INCREASED:
      return "MUBLAS_STATUS_SIZE_INCREASED";
    case MUBLAS_STATUS_SIZE_UNCHANGED:
      return "MUBLAS_STATUS_SIZE_UNCHANGED";
    case MUBLAS_STATUS_INVALID_VALUE:
      return "MUBLAS_STATUS_INVALID_VALUE";
    case MUBLAS_STATUS_CONTINUE:
      return "MUBLAS_STATUS_CONTINUE";
    case MUBLAS_STATUS_CHECK_NUMERICS_FAIL:
      return "MUBLAS_STATUS_CHECK_NUMERICS_FAIL";
    default:
      return absl::StrCat("<invalid muBLAS status: ", status, ">");
  }
}

tsl::Status ToStatus(::musa::dnn::Status status, const char* op_name) {
  if (status == ::musa::dnn::Status::SUCCESS) {
    return tsl::OkStatus();
  }
  return tsl::errors::Internal(
      absl::StrCat(op_name, " failed with muDNN status ",
                   static_cast<int>(status)));
}

std::string TransposeToString(blas::Transpose transpose) {
  switch (transpose) {
    case blas::Transpose::kNoTranspose:
      return "N";
    case blas::Transpose::kTranspose:
      return "T";
    case blas::Transpose::kConjugateTranspose:
      return "C";
  }
  return "?";
}

std::string BatchedGemmParamsToString(blas::DataType dtype,
                                      blas::Transpose transa,
                                      blas::Transpose transb, uint64_t m,
                                      uint64_t n, uint64_t k, int lda,
                                      int64_t stride_a, int ldb,
                                      int64_t stride_b, int ldc,
                                      int64_t stride_c, int batch_count) {
  return absl::StrCat(
      "batched_gemm_params={dtype=", blas::DataTypeString(dtype),
      ",batch_count=", batch_count, ",m=", m, ",n=", n, ",k=", k,
      ",transa=", TransposeToString(transa), ",transb=",
      TransposeToString(transb), ",lda=", lda, ",stride_a=", stride_a,
      ",ldb=", ldb, ",stride_b=", stride_b, ",ldc=", ldc,
      ",stride_c=", stride_c, "}");
}

bool IsObservedInterleavedB(blas::Transpose transb, uint64_t n, uint64_t k,
                            int ldb, int64_t stride_b, int batch_count) {
  const int64_t b_stored_rows =
      (transb == blas::Transpose::kNoTranspose) ? static_cast<int64_t>(k)
                                                 : static_cast<int64_t>(n);
  return batch_count > 1 && stride_b == b_stored_rows &&
         ldb == b_stored_rows * batch_count;
}

template <size_t N>
std::string ArrayToString(const std::array<int64_t, N>& values) {
  std::string result = "[";
  for (size_t i = 0; i < N; ++i) {
    if (i > 0) {
      absl::StrAppend(&result, ",");
    }
    absl::StrAppend(&result, values[i]);
  }
  absl::StrAppend(&result, "]");
  return result;
}

mublasOperation MUSABlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return MUBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return MUBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return MUBLAS_OP_C;
  }
  return MUBLAS_OP_N;
}

void CheckGemmPreconditions(blas::Transpose transa, blas::Transpose transb,
                            uint64_t m, uint64_t n, uint64_t k, int lda,
                            int ldb) {
  if (transa == blas::Transpose::kNoTranspose) {
    CHECK_GE(lda, static_cast<int64_t>(m));
  } else {
    CHECK_GE(lda, static_cast<int64_t>(k));
  }
  if (transb == blas::Transpose::kNoTranspose) {
    CHECK_GE(ldb, static_cast<int64_t>(k));
  } else {
    CHECK_GE(ldb, static_cast<int64_t>(n));
  }
}

tsl::StatusOr<::musa::dnn::Tensor::Type> AsMudnnTensorType(
    blas::DataType dtype) {
  switch (dtype) {
    case blas::DataType::kFloat:
      return ::musa::dnn::Tensor::Type::FLOAT;
    case blas::DataType::kDouble:
      return ::musa::dnn::Tensor::Type::DOUBLE;
    case blas::DataType::kHalf:
      return ::musa::dnn::Tensor::Type::HALF;
    case blas::DataType::kBF16:
      return ::musa::dnn::Tensor::Type::BFLOAT16;
    default:
      return tsl::errors::Unimplemented(
          absl::StrCat("muDNN BatchMatMul bridge does not support dtype ",
                       blas::DataTypeString(dtype)));
  }
}

tsl::StatusOr<double> HostScalarToMudnnScalar(blas::DataType dtype,
                                              const void* scalar,
                                              const char* scalar_name) {
  switch (dtype) {
    case blas::DataType::kFloat:
      return static_cast<double>(*static_cast<const float*>(scalar));
    case blas::DataType::kDouble:
      return *static_cast<const double*>(scalar);
    // For fp16/bf16 GEMM in StreamExecutor, alpha/beta are passed as float.
    case blas::DataType::kHalf:
    case blas::DataType::kBF16:
      return static_cast<double>(*static_cast<const float*>(scalar));
    default:
      return tsl::errors::Unimplemented(
          absl::StrCat("muDNN BatchMatMul bridge does not support ", scalar_name,
                       " for dtype ", blas::DataTypeString(dtype)));
  }
}

tsl::StatusOr<size_t> MudnnElementSize(blas::DataType dtype) {
  switch (dtype) {
    case blas::DataType::kHalf:
    case blas::DataType::kBF16:
      return 2;
    case blas::DataType::kFloat:
      return 4;
    case blas::DataType::kDouble:
      return 8;
    default:
      return tsl::errors::Unimplemented(
          absl::StrCat("Unsupported datatype for muDNN element size: ",
                       blas::DataTypeString(dtype)));
  }
}

struct MudnnTensorLayout2D {
  std::array<int64_t, 2> dims;
  std::array<int64_t, 2> strides;
};

struct MudnnTensorLayout3D {
  std::array<int64_t, 3> dims;
  std::array<int64_t, 3> strides;
};

std::string LayoutToString(const MudnnTensorLayout2D& layout) {
  return absl::StrCat("dims=", ArrayToString(layout.dims),
                      ",strides=", ArrayToString(layout.strides));
}

std::string LayoutToString(const MudnnTensorLayout3D& layout) {
  return absl::StrCat("dims=", ArrayToString(layout.dims),
                      ",strides=", ArrayToString(layout.strides));
}

int64_t GetTensorFootprint(const MudnnTensorLayout2D& layout) {
  if (layout.dims[0] == 0 || layout.dims[1] == 0) {
    return 0;
  }
  return (layout.dims[0] - 1) * layout.strides[0] +
         (layout.dims[1] - 1) * layout.strides[1] + 1;
}

tsl::StatusOr<MudnnTensorLayout2D> BuildMudnn2DLayoutFromBlas(
    blas::Transpose transpose, int64_t rows, int64_t cols, int64_t leading_dim) {
  const int64_t stored_rows =
      (transpose == blas::Transpose::kNoTranspose) ? rows : cols;
  const int64_t stored_cols =
      (transpose == blas::Transpose::kNoTranspose) ? cols : rows;
  if (leading_dim < stored_rows) {
    return tsl::errors::InvalidArgument(
        absl::StrCat("leading dimension ", leading_dim,
                     " is smaller than required stored rows ", stored_rows));
  }
  // muDNN MatMul prefers row-major tensor descriptors. A BLAS column-major
  // matrix with shape [stored_rows, stored_cols] is the same memory as a
  // row-major tensor with shape [stored_cols, stored_rows].
  return MudnnTensorLayout2D{
      /*dims=*/{stored_cols, stored_rows},
      /*strides=*/{leading_dim, 1},
  };
}

tsl::StatusOr<MudnnTensorLayout3D> BuildMudnn3DLayoutFromBlas(
    blas::Transpose transpose, int64_t batch, int64_t rows, int64_t cols,
    int64_t leading_dim, int64_t batch_stride, bool allow_broadcast_batch) {
  TF_ASSIGN_OR_RETURN(MudnnTensorLayout2D layout_2d,
                      BuildMudnn2DLayoutFromBlas(transpose, rows, cols,
                                                 leading_dim));
  const int64_t min_batch_stride = GetTensorFootprint(layout_2d);
  if (batch > 1 && batch_stride == 0 && !allow_broadcast_batch) {
    return tsl::errors::InvalidArgument(
        "zero batch stride is not allowed for this tensor");
  }
  if (batch > 1 && batch_stride != 0 && batch_stride < min_batch_stride) {
    return tsl::errors::InvalidArgument(
        absl::StrCat("batch stride ", batch_stride,
                     " is smaller than required tensor footprint ",
                     min_batch_stride));
  }
  return MudnnTensorLayout3D{
      /*dims=*/{batch, layout_2d.dims[0], layout_2d.dims[1]},
      /*strides=*/{batch_stride, layout_2d.strides[0], layout_2d.strides[1]},
  };
}

tsl::Status WithMudnnGemmDebugContext(
    const tsl::Status& status, blas::DataType dtype, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, int lda,
    int ldb, int ldc, const MudnnTensorLayout2D& layout_a,
    const MudnnTensorLayout2D& layout_b, const MudnnTensorLayout2D& layout_c) {
  if (status.ok()) {
    return status;
  }
  return tsl::errors::Internal(
      absl::StrCat(status.message(), "; gemm_debug={dtype=",
                   blas::DataTypeString(dtype), ",m=", m, ",n=", n, ",k=", k,
                   ",transa=", TransposeToString(transa), ",transb=",
                   TransposeToString(transb), ",lda=", lda, ",ldb=", ldb,
                   ",ldc=", ldc, ",A{", LayoutToString(layout_a), "},B{",
                   LayoutToString(layout_b), "},C{", LayoutToString(layout_c),
                   "}}"));
}

tsl::Status WithMudnnBatchedGemmDebugContext(
    const tsl::Status& status, blas::DataType dtype, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, int lda,
    int64_t stride_a, int ldb, int64_t stride_b, int ldc, int64_t stride_c,
    int batch_count, const MudnnTensorLayout3D& layout_a,
    const MudnnTensorLayout3D& layout_b, const MudnnTensorLayout3D& layout_c) {
  if (status.ok()) {
    return status;
  }
  return tsl::errors::Internal(
      absl::StrCat(status.message(), "; gemm_debug={dtype=",
                   blas::DataTypeString(dtype), ",batch_count=", batch_count,
                   ",m=", m, ",n=", n, ",k=", k, ",transa=",
                   TransposeToString(transa), ",transb=",
                   TransposeToString(transb), ",lda=", lda, ",stride_a=",
                   stride_a, ",ldb=", ldb, ",stride_b=", stride_b, ",ldc=",
                   ldc, ",stride_c=", stride_c, ",A{", LayoutToString(layout_a),
                   "},B{", LayoutToString(layout_b), "},C{",
                   LayoutToString(layout_c), "}}"));
}

tsl::Status WithMudnnBatchedLayoutBuildDebugContext(
    const tsl::Status& status, const char* operand, blas::DataType dtype,
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64_t n,
    uint64_t k, int lda, int64_t stride_a, int ldb, int64_t stride_b, int ldc,
    int64_t stride_c, int batch_count, blas::Transpose operand_transpose,
    int64_t operand_rows, int64_t operand_cols, int64_t operand_leading_dim,
    int64_t operand_batch_stride, bool allow_broadcast_batch) {
  if (status.ok()) {
    return status;
  }
  return tsl::errors::Internal(
      absl::StrCat(
          status.message(), "; batched_layout_debug={operand=", operand,
          ",dtype=", blas::DataTypeString(dtype), ",batch_count=", batch_count,
          ",m=", m, ",n=", n, ",k=", k, ",transa=",
          TransposeToString(transa), ",transb=", TransposeToString(transb),
          ",lda=", lda, ",stride_a=", stride_a, ",ldb=", ldb,
          ",stride_b=", stride_b, ",ldc=", ldc, ",stride_c=", stride_c,
          ",operand_transpose=", TransposeToString(operand_transpose),
          ",operand_rows=", operand_rows, ",operand_cols=", operand_cols,
          ",operand_leading_dim=", operand_leading_dim,
          ",operand_batch_stride=", operand_batch_stride,
          ",allow_broadcast_batch=", allow_broadcast_batch ? "true" : "false",
          "}"));
}

tsl::Status CreateMudnnTensor2D(const void* addr,
                                ::musa::dnn::Tensor::Type type,
                                const MudnnTensorLayout2D& layout,
                                ::musa::dnn::Tensor* tensor) {
  TF_RETURN_IF_ERROR(ToStatus(tensor->SetAddr(addr), "Tensor::SetAddr"));
  TF_RETURN_IF_ERROR(ToStatus(tensor->SetType(type), "Tensor::SetType"));
  return ToStatus(
      tensor->SetNdInfo(2, layout.dims.data(), layout.strides.data()),
      "Tensor::SetNdInfo");
}

tsl::Status CreateMudnnTensor3D(const void* addr,
                                ::musa::dnn::Tensor::Type type,
                                const MudnnTensorLayout3D& layout,
                                ::musa::dnn::Tensor* tensor) {
  TF_RETURN_IF_ERROR(ToStatus(tensor->SetAddr(addr), "Tensor::SetAddr"));
  TF_RETURN_IF_ERROR(ToStatus(tensor->SetType(type), "Tensor::SetType"));
  return ToStatus(
      tensor->SetNdInfo(3, layout.dims.data(), layout.strides.data()),
      "Tensor::SetNdInfo");
}

tsl::StatusOr<bool> AsMudnnTranspose(blas::DataType dtype,
                                     blas::Transpose transpose) {
  switch (transpose) {
    case blas::Transpose::kNoTranspose:
      return false;
    case blas::Transpose::kTranspose:
      return true;
    case blas::Transpose::kConjugateTranspose:
      if (dtype == blas::DataType::kComplexFloat ||
          dtype == blas::DataType::kComplexDouble) {
        return tsl::errors::Unimplemented(
            "muDNN helper does not support complex conjugate transpose");
      }
      return true;
  }
  return tsl::errors::InvalidArgument("Unknown transpose mode");
}

tsl::Status RunMudnnGemm(::musa::dnn::Handle* handle, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64_t k, blas::DataType dtype, const void* alpha,
                         const DeviceMemoryBase& a, int lda,
                         const DeviceMemoryBase& b, int ldb, const void* beta,
                         DeviceMemoryBase* c, int ldc) {
  TF_ASSIGN_OR_RETURN(auto type, AsMudnnTensorType(dtype));
  TF_ASSIGN_OR_RETURN(bool transpose_a, AsMudnnTranspose(dtype, transa));
  TF_ASSIGN_OR_RETURN(bool transpose_b, AsMudnnTranspose(dtype, transb));
  TF_ASSIGN_OR_RETURN(double alpha_value,
                      HostScalarToMudnnScalar(dtype, alpha, "alpha"));
  TF_ASSIGN_OR_RETURN(double beta_value,
                      HostScalarToMudnnScalar(dtype, beta, "beta"));

  ::musa::dnn::Tensor tensor_a;
  ::musa::dnn::Tensor tensor_b;
  ::musa::dnn::Tensor tensor_c;

  TF_ASSIGN_OR_RETURN(MudnnTensorLayout2D layout_a,
                      BuildMudnn2DLayoutFromBlas(transa, m, k, lda));
  TF_ASSIGN_OR_RETURN(MudnnTensorLayout2D layout_b,
                      BuildMudnn2DLayoutFromBlas(transb, k, n, ldb));
  TF_ASSIGN_OR_RETURN(MudnnTensorLayout2D layout_c,
                      BuildMudnn2DLayoutFromBlas(
                          blas::Transpose::kNoTranspose, m, n, ldc));
  TF_RETURN_IF_ERROR(CreateMudnnTensor2D(a.opaque(), type, layout_a, &tensor_a));
  TF_RETURN_IF_ERROR(CreateMudnnTensor2D(b.opaque(), type, layout_b, &tensor_b));
  TF_RETURN_IF_ERROR(
      CreateMudnnTensor2D(c->opaque(), type, layout_c, &tensor_c));

  ::musa::dnn::MatMul op;
  // Re-express column-major BLAS GEMM as a row-major MatMul:
  //   C_col = op(A) * op(B)
  //   C_row = C_col^T = op(B)^T * op(A)^T
  TF_RETURN_IF_ERROR(
      ToStatus(op.SetTranspose(transpose_b, transpose_a), "MatMul::SetTranspose"));
  TF_RETURN_IF_ERROR(ToStatus(op.SetAlpha(alpha_value), "MatMul::SetAlpha"));
  TF_RETURN_IF_ERROR(ToStatus(op.SetBeta(beta_value), "MatMul::SetBeta"));
  return WithMudnnGemmDebugContext(
      ToStatus(op.Run(*handle, tensor_c, tensor_b, tensor_a), "MatMul::Run"),
      dtype, transa, transb, m, n, k, lda, ldb, ldc, layout_a, layout_b,
      layout_c);
}

tsl::Status RunMudnnBatchedGemm(Stream* stream, ::musa::dnn::Handle* handle,
                                blas::Transpose transa,
                                blas::Transpose transb, uint64_t m, uint64_t n,
                                uint64_t k, blas::DataType dtype,
                                const void* alpha, const DeviceMemoryBase& a,
                                int lda, int64_t stride_a,
                                const DeviceMemoryBase& b, int ldb,
                                int64_t stride_b, const void* beta,
                                DeviceMemoryBase* c, int ldc, int64_t stride_c,
                                int batch_count) {
  TF_ASSIGN_OR_RETURN(auto type, AsMudnnTensorType(dtype));
  TF_ASSIGN_OR_RETURN(bool transpose_a, AsMudnnTranspose(dtype, transa));
  TF_ASSIGN_OR_RETURN(bool transpose_b, AsMudnnTranspose(dtype, transb));
  TF_ASSIGN_OR_RETURN(double alpha_value,
                      HostScalarToMudnnScalar(dtype, alpha, "alpha"));
  TF_ASSIGN_OR_RETURN(double beta_value,
                      HostScalarToMudnnScalar(dtype, beta, "beta"));

  ::musa::dnn::Tensor tensor_a;
  ::musa::dnn::Tensor tensor_b;
  ::musa::dnn::Tensor tensor_c;

  // Observed model case: B has batch dimension interleaved in the middle
  // ([N, B, K] layout with strides [ldb, stride_b, 1]). This is kept here for
  // diagnostics, but the public entry point skips this known non-contiguous
  // layout before reaching muDNN.
  const int64_t b_stored_rows =
      (transb == blas::Transpose::kNoTranspose) ? static_cast<int64_t>(k)
                                                 : static_cast<int64_t>(n);
  const int64_t b_stored_cols =
      (transb == blas::Transpose::kNoTranspose) ? static_cast<int64_t>(n)
                                                 : static_cast<int64_t>(k);
  const bool use_interleaved_b_directly =
      IsObservedInterleavedB(transb, n, k, ldb, stride_b, batch_count);

  if (use_interleaved_b_directly) {
    VLOG(1) << "[OBSERVED_INTERLEAVED_B] trying direct muDNN BatchMatMul path; "
              << BatchedGemmParamsToString(dtype, transa, transb, m, n, k, lda,
                                           stride_a, ldb, stride_b, ldc,
                                           stride_c, batch_count);
  }

  auto build_layout_or_debug =
      [&](const char* operand, blas::Transpose operand_transpose,
          int64_t operand_rows, int64_t operand_cols,
          int64_t operand_leading_dim, int64_t operand_batch_stride,
          bool allow_broadcast_batch)
      -> tsl::StatusOr<MudnnTensorLayout3D> {
    auto layout_or =
        BuildMudnn3DLayoutFromBlas(operand_transpose, batch_count, operand_rows,
                                   operand_cols, operand_leading_dim,
                                   operand_batch_stride, allow_broadcast_batch);
    if (!layout_or.ok()) {
      return WithMudnnBatchedLayoutBuildDebugContext(
          layout_or.status(), operand, dtype, transa, transb, m, n, k, lda,
          stride_a, ldb, stride_b, ldc, stride_c, batch_count,
          operand_transpose, operand_rows, operand_cols, operand_leading_dim,
          operand_batch_stride, allow_broadcast_batch);
    }
    return layout_or;
  };

  TF_ASSIGN_OR_RETURN(MudnnTensorLayout3D layout_a,
                      build_layout_or_debug("A", transa, m, k, lda, stride_a,
                                            /*allow_broadcast_batch=*/true));
  MudnnTensorLayout3D layout_b;
  if (use_interleaved_b_directly) {
    layout_b = MudnnTensorLayout3D{
        /*dims=*/{batch_count, b_stored_cols, b_stored_rows},
        /*strides=*/{stride_b, ldb, 1},
    };
  } else {
    TF_ASSIGN_OR_RETURN(layout_b,
                        build_layout_or_debug("B", transb, k, n, ldb, stride_b,
                                              /*allow_broadcast_batch=*/true));
  }
  TF_ASSIGN_OR_RETURN(MudnnTensorLayout3D layout_c,
                      build_layout_or_debug(
                          "C", blas::Transpose::kNoTranspose, m, n, ldc,
                          stride_c, /*allow_broadcast_batch=*/false));
  TF_RETURN_IF_ERROR(CreateMudnnTensor3D(a.opaque(), type, layout_a, &tensor_a));
  TF_RETURN_IF_ERROR(CreateMudnnTensor3D(b.opaque(), type, layout_b, &tensor_b));
  TF_RETURN_IF_ERROR(CreateMudnnTensor3D(c->opaque(), type, layout_c, &tensor_c));

  ::musa::dnn::BatchMatMul op;
  TF_RETURN_IF_ERROR(ToStatus(op.SetTranspose(transpose_b, transpose_a),
                              "BatchMatMul::SetTranspose"));
  TF_RETURN_IF_ERROR(
      ToStatus(op.SetAlpha(alpha_value), "BatchMatMul::SetAlpha"));
  TF_RETURN_IF_ERROR(ToStatus(op.SetBeta(beta_value), "BatchMatMul::SetBeta"));
  return WithMudnnBatchedGemmDebugContext(
      ToStatus(op.Run(*handle, tensor_c, tensor_b, tensor_a),
               "BatchMatMul::Run"),
      dtype, transa, transb, m, n, k, lda, stride_a, ldb, stride_b, ldc,
      stride_c, batch_count, layout_a, layout_b, layout_c);
}

template <typename T>
bool UnsupportedBool(const char* method) {
  LOG(ERROR) << method << " is not implemented for MUSA BLAS yet";
  return false;
}

tsl::Status UnsupportedStatus(const char* method) {
  return tsl::errors::Unimplemented(
      absl::StrCat(method, " is not implemented for MUSA BLAS yet"));
}

}  // namespace

MUSABlas::MUSABlas(MusaExecutor* parent) : parent_(CHECK_NOTNULL(parent)) {}

bool MUSABlas::Init() {
  auto activation = parent_->ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }

  mublasStatus ret = wrap::mublasCreate(&blas_);
  if (ret != MUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create muBLAS handle: " << ToString(ret);
    return false;
  }

  int device = -1;
  musaError_t runtime_status = musaGetDevice(&device);
  if (runtime_status != musaSuccess) {
    LOG(ERROR) << "failed to query current MUSA runtime device: "
               << static_cast<int>(runtime_status);
    return false;
  }
  dnn_handle_ = std::make_unique<::musa::dnn::Handle>(device);
  return true;
}

MUSABlas::~MUSABlas() {
  if (blas_ == nullptr) {
    return;
  }
  auto activation = parent_->ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }
  wrap::mublasDestroy(blas_);
}

bool MUSABlas::SetStream(Stream* stream) {
  CHECK(blas_ != nullptr);
  CHECK(dnn_handle_ != nullptr);
  auto handle =
      stream == nullptr
          ? nullptr
          : static_cast<musaStream_t>(stream->platform_specific_handle().stream);
  if (handle == current_stream_) {
    return true;
  }
  mublasStatus ret = wrap::mublasSetStream(blas_, handle);
  if (ret != MUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for muBLAS calls: " << ToString(ret);
    return false;
  }
  if (dnn_handle_->SetStream(handle) != ::musa::dnn::Status::SUCCESS) {
    LOG(ERROR) << "failed to set stream for muDNN calls";
    return false;
  }
  current_stream_ = handle;
  return true;
}

template <typename FuncT, typename... Args>
tsl::Status MUSABlas::DoBlasInternalImpl(FuncT func, Stream* stream,
                                         bool err_on_failure, Args... args) {
  absl::MutexLock lock(&mu_);
  CHECK(blas_ != nullptr);

  auto activation = parent_->ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  if (!SetStream(stream)) {
    return tsl::errors::Internal("Failed to bind muBLAS handle to stream");
  }

  if (tsl::OpDeterminismRequired()) {
    mublasStatus ret =
        wrap::mublasSetAtomicsMode(blas_, MUBLAS_ATOMICS_NOT_ALLOWED);
    if (ret != MUBLAS_STATUS_SUCCESS) {
      if (err_on_failure) {
        LOG(ERROR) << "failed to disable muBLAS atomics before "
                   << FuncT::kName << ": " << ToString(ret);
      }
      return tsl::errors::Internal(
          absl::StrCat("Failed to set muBLAS atomics mode: ", ToString(ret)));
    }
  }

  mublasStatus ret = func(blas_, args...);

  if (ret != MUBLAS_STATUS_SUCCESS) {
    std::string message =
        absl::StrFormat("%s failed with: %s", FuncT::kName, ToString(ret));
    if (err_on_failure) {
      LOG(ERROR) << message;
    }
    return tsl::errors::Internal(message);
  }
  return tsl::OkStatus();
}

tsl::Status MUSABlas::DoBlasGemm(Stream* stream, blas::Transpose transa,
                                 blas::Transpose transb, uint64_t m, uint64 n,
                                 uint64_t k, blas::DataType dtype,
                                 const void* alpha, const DeviceMemoryBase& a,
                                 int lda, const DeviceMemoryBase& b, int ldb,
                                 const void* beta, DeviceMemoryBase* c, int ldc,
                                 const NumericOptions& numeric_options) {
  (void)numeric_options;

  auto run_mudnn = [&]() -> tsl::Status {
    absl::MutexLock lock(&mu_);
    CHECK(dnn_handle_ != nullptr);
    auto activation = parent_->ActivateContext();
    TF_RETURN_IF_ERROR(activation.status());
    if (!SetStream(stream)) {
      return tsl::errors::Internal("Failed to bind muDNN handle to stream");
    }
    return RunMudnnGemm(dnn_handle_.get(), transa, transb, m, n, k, dtype,
                        alpha, a, lda, b, ldb, beta, c, ldc);
  };

  switch (dtype) {
    case blas::DataType::kHalf:
    case blas::DataType::kBF16:
    case blas::DataType::kFloat:
    case blas::DataType::kDouble:
      return run_mudnn();
    case blas::DataType::kComplexFloat: {
      CheckGemmPreconditions(transa, transb, m, n, k, lda, ldb);
      auto complex_alpha =
          *static_cast<const std::complex<float>*>(alpha);
      auto complex_beta = *static_cast<const std::complex<float>*>(beta);
      return DoBlasInternalStatus(
          wrap::mublasCgemm, stream, MUSABlasTranspose(transa),
          MUSABlasTranspose(transb), m, n, k, ComplexCast(&complex_alpha),
          reinterpret_cast<const muComplex*>(a.opaque()), lda,
          reinterpret_cast<const muComplex*>(b.opaque()), ldb,
          ComplexCast(&complex_beta), reinterpret_cast<muComplex*>(c->opaque()),
          ldc);
    }
    case blas::DataType::kComplexDouble: {
      CheckGemmPreconditions(transa, transb, m, n, k, lda, ldb);
      auto complex_alpha =
          *static_cast<const std::complex<double>*>(alpha);
      auto complex_beta = *static_cast<const std::complex<double>*>(beta);
      return DoBlasInternalStatus(
          wrap::mublasZgemm, stream, MUSABlasTranspose(transa),
          MUSABlasTranspose(transb), m, n, k, ComplexCast(&complex_alpha),
          reinterpret_cast<const muDoubleComplex*>(a.opaque()), lda,
          reinterpret_cast<const muDoubleComplex*>(b.opaque()), ldb,
          ComplexCast(&complex_beta),
          reinterpret_cast<muDoubleComplex*>(c->opaque()), ldc);
    }
    default:
      return tsl::errors::Unimplemented(
          absl::StrCat("Unsupported datatype for MUSA GEMM: ",
                       blas::DataTypeString(dtype)));
  }
}

bool MUSABlas::GetBlasGemmAlgorithms(
    Stream* stream, std::vector<blas::AlgorithmType>* out_algorithms) {
  (void)stream;
  out_algorithms->clear();
  out_algorithms->push_back(blas::kDefaultAlgorithm);
  return true;
}

tsl::Status MUSABlas::DoBlasGemmWithAlgorithm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, const void* alpha, const DeviceMemoryBase& a,
    blas::DataType type_a, int lda, const DeviceMemoryBase& b,
    blas::DataType type_b, int ldb, const void* beta, DeviceMemoryBase* c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions& numeric_options,
    blas::ProfileResult* output_profile_result) {
  (void)computation_type;
  (void)algorithm;
  if (type_a != type_b || type_a != type_c) {
    return tsl::errors::Unimplemented(
        "MUSA BLAS only supports same-type GEMM without algorithm selection");
  }
  TF_RETURN_IF_ERROR(DoBlasGemm(stream, transa, transb, m, n, k, type_a, alpha,
                                a, lda, b, ldb, beta, c, ldc, numeric_options));
  if (output_profile_result != nullptr) {
    output_profile_result->set_is_valid(true);
  }
  return tsl::OkStatus();
}

tsl::Status MUSABlas::DoBlasGemmStridedBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, blas::DataType dtype, const void* alpha,
    const DeviceMemoryBase& a, int lda, int64_t stride_a,
    const DeviceMemoryBase& b, int ldb, int64_t stride_b, const void* beta,
    DeviceMemoryBase* c, int ldc, int64_t stride_c, int batch_count,
    const NumericOptions& numeric_options) {
  (void)numeric_options;

  auto run_mudnn_batched = [&]() -> tsl::Status {
    absl::MutexLock lock(&mu_);
    CHECK(dnn_handle_ != nullptr);
    auto activation = parent_->ActivateContext();
    TF_RETURN_IF_ERROR(activation.status());
    if (!SetStream(stream)) {
      return tsl::errors::Internal("Failed to bind muDNN handle to stream");
    }
    return RunMudnnBatchedGemm(stream, dnn_handle_.get(), transa, transb, m, n,
                               k, dtype, alpha, a, lda, stride_a, b, ldb,
                               stride_b, beta, c, ldc, stride_c, batch_count);
  };

  auto run_mublas_batched = [&]() -> tsl::Status {
    switch (dtype) {
      case blas::DataType::kHalf: {
        Eigen::half alpha_half = Eigen::half(*static_cast<const float*>(alpha));
        Eigen::half beta_half = Eigen::half(*static_cast<const float*>(beta));
        return DoBlasInternalStatus(
            wrap::mublasGemmStridedBatchedEx, stream, MUSABlasTranspose(transa),
            MUSABlasTranspose(transb), static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(k), &alpha_half, a.opaque(), MUSA_R_16F, lda,
            static_cast<long long int>(stride_a), b.opaque(), MUSA_R_16F, ldb,
            static_cast<long long int>(stride_b), &beta_half, c->opaque(),
            MUSA_R_16F, ldc, static_cast<long long int>(stride_c), batch_count,
            MUBLAS_COMPUTE_16F, MUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      case blas::DataType::kBF16:
        return DoBlasInternalStatus(
            wrap::mublasGemmStridedBatchedEx, stream, MUSABlasTranspose(transa),
            MUSABlasTranspose(transb), static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(k), alpha, a.opaque(), MUSA_R_16BF, lda,
            static_cast<long long int>(stride_a), b.opaque(), MUSA_R_16BF, ldb,
            static_cast<long long int>(stride_b), beta, c->opaque(), MUSA_R_16BF,
            ldc, static_cast<long long int>(stride_c), batch_count,
            MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT_TENSOR_OP);
      case blas::DataType::kFloat:
        return DoBlasInternalStatus(
            wrap::mublasSgemmStridedBatched, stream, MUSABlasTranspose(transa),
            MUSABlasTranspose(transb), m, n, k, static_cast<const float*>(alpha),
            static_cast<const float*>(a.opaque()), lda,
            static_cast<long long int>(stride_a),
            static_cast<const float*>(b.opaque()), ldb,
            static_cast<long long int>(stride_b), static_cast<const float*>(beta),
            static_cast<float*>(c->opaque()), ldc,
            static_cast<long long int>(stride_c), batch_count);
      case blas::DataType::kDouble:
        return DoBlasInternalStatus(
            wrap::mublasDgemmStridedBatched, stream, MUSABlasTranspose(transa),
            MUSABlasTranspose(transb), m, n, k,
            static_cast<const double*>(alpha),
            static_cast<const double*>(a.opaque()), lda,
            static_cast<long long int>(stride_a),
            static_cast<const double*>(b.opaque()), ldb,
            static_cast<long long int>(stride_b),
            static_cast<const double*>(beta), static_cast<double*>(c->opaque()),
            ldc, static_cast<long long int>(stride_c), batch_count);
      default:
        return tsl::errors::Unimplemented(
            absl::StrCat("Unsupported fallback datatype for MUSA strided "
                         "batched GEMM: ",
                         blas::DataTypeString(dtype)));
    }
  };

  switch (dtype) {
    case blas::DataType::kHalf:
    case blas::DataType::kBF16:
    case blas::DataType::kFloat:
    case blas::DataType::kDouble: {
      const bool observed_interleaved_b =
          IsObservedInterleavedB(transb, n, k, ldb, stride_b, batch_count);
      if (observed_interleaved_b) {
        VLOG(1) << "[OBSERVED_INTERLEAVED_B] skipping muDNN BatchMatMul path "
                   "for non-contiguous B; using muBLAS strided batched GEMM; "
                << BatchedGemmParamsToString(dtype, transa, transb, m, n, k,
                                             lda, stride_a, ldb, stride_b, ldc,
                                             stride_c, batch_count);
        return run_mublas_batched();
      }
      tsl::Status mudnn_status = run_mudnn_batched();
      if (mudnn_status.ok()) {
        return mudnn_status;
      }
      LOG(INFO) << "Falling back to muBLAS strided batched GEMM: "
                << mudnn_status.message() << "; "
                << BatchedGemmParamsToString(dtype, transa, transb, m, n, k,
                                             lda, stride_a, ldb, stride_b, ldc,
                                             stride_c, batch_count);
      return run_mublas_batched();
    }
    case blas::DataType::kComplexFloat: {
      CheckGemmPreconditions(transa, transb, m, n, k, lda, ldb);
      auto complex_alpha = *static_cast<const std::complex<float>*>(alpha);
      auto complex_beta = *static_cast<const std::complex<float>*>(beta);
      return DoBlasInternalStatus(
          wrap::mublasCgemmStridedBatched, stream, MUSABlasTranspose(transa),
          MUSABlasTranspose(transb), m, n, k, ComplexCast(&complex_alpha),
          reinterpret_cast<const muComplex*>(a.opaque()), lda,
          static_cast<long long int>(stride_a),
          reinterpret_cast<const muComplex*>(b.opaque()), ldb,
          static_cast<long long int>(stride_b), ComplexCast(&complex_beta),
          reinterpret_cast<muComplex*>(c->opaque()), ldc,
          static_cast<long long int>(stride_c), batch_count);
    }
    case blas::DataType::kComplexDouble: {
      CheckGemmPreconditions(transa, transb, m, n, k, lda, ldb);
      auto complex_alpha = *static_cast<const std::complex<double>*>(alpha);
      auto complex_beta = *static_cast<const std::complex<double>*>(beta);
      return DoBlasInternalStatus(
          wrap::mublasZgemmStridedBatched, stream, MUSABlasTranspose(transa),
          MUSABlasTranspose(transb), m, n, k, ComplexCast(&complex_alpha),
          reinterpret_cast<const muDoubleComplex*>(a.opaque()), lda,
          static_cast<long long int>(stride_a),
          reinterpret_cast<const muDoubleComplex*>(b.opaque()), ldb,
          static_cast<long long int>(stride_b), ComplexCast(&complex_beta),
          reinterpret_cast<muDoubleComplex*>(c->opaque()), ldc,
          static_cast<long long int>(stride_c), batch_count);
    }
    default:
      return tsl::errors::Unimplemented(
          absl::StrCat("Unsupported datatype for MUSA strided batched GEMM: ",
                       blas::DataTypeString(dtype)));
  }
}

tsl::Status MUSABlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, const void* alpha, const DeviceMemoryBase& a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase& b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void* beta,
    DeviceMemoryBase* c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions& numeric_options,
    blas::ProfileResult* output_profile_result) {
  (void)computation_type;
  (void)algorithm;
  if (type_a != type_b || type_a != type_c) {
    return tsl::errors::Unimplemented(
        "MUSA BLAS only supports same-type strided batched GEMM");
  }
  TF_RETURN_IF_ERROR(DoBlasGemmStridedBatched(
      stream, transa, transb, m, n, k, type_a, alpha, a, lda, stride_a, b, ldb,
      stride_b, beta, c, ldc, stride_c, batch_count, numeric_options));
  if (output_profile_result != nullptr) {
    output_profile_result->set_is_valid(true);
  }
  return tsl::OkStatus();
}

#define MUSA_BLAS_STUB_AXPY(type)                                               \
  bool MUSABlas::DoBlasAxpy(Stream* stream, uint64_t elem_count, type alpha,    \
                            const DeviceMemory<type>& x, int incx,              \
                            DeviceMemory<type>* y, int incy) {                  \
    (void)stream;                                                               \
    (void)elem_count;                                                           \
    (void)alpha;                                                                \
    (void)x;                                                                    \
    (void)incx;                                                                 \
    (void)y;                                                                    \
    (void)incy;                                                                 \
    return UnsupportedBool<type>("DoBlasAxpy");                                \
  }

#define MUSA_BLAS_STUB_COPY(type)                                               \
  bool MUSABlas::DoBlasCopy(Stream* stream, uint64_t elem_count,                \
                            const DeviceMemory<type>& x, int incx,              \
                            DeviceMemory<type>* y, int incy) {                  \
    (void)stream;                                                               \
    (void)elem_count;                                                           \
    (void)x;                                                                    \
    (void)incx;                                                                 \
    (void)y;                                                                    \
    (void)incy;                                                                 \
    return UnsupportedBool<type>("DoBlasCopy");                                \
  }

#define MUSA_BLAS_STUB_SCAL(type, alpha_type)                                   \
  bool MUSABlas::DoBlasScal(Stream* stream, uint64_t elem_count,                \
                            alpha_type alpha, DeviceMemory<type>* x, int incx) {\
    (void)stream;                                                               \
    (void)elem_count;                                                           \
    (void)alpha;                                                                \
    (void)x;                                                                    \
    (void)incx;                                                                 \
    return UnsupportedBool<type>("DoBlasScal");                                \
  }

#define MUSA_BLAS_STUB_GEMV(type)                                               \
  bool MUSABlas::DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m,  \
                            uint64 n, type alpha, const DeviceMemory<type>& a,  \
                            int lda, const DeviceMemory<type>& x, int incx,     \
                            type beta, DeviceMemory<type>* y, int incy) {       \
    (void)stream;                                                               \
    (void)trans;                                                                \
    (void)m;                                                                    \
    (void)n;                                                                    \
    (void)alpha;                                                                \
    (void)a;                                                                    \
    (void)lda;                                                                  \
    (void)x;                                                                    \
    (void)incx;                                                                 \
    (void)beta;                                                                 \
    (void)y;                                                                    \
    (void)incy;                                                                 \
    return UnsupportedBool<type>("DoBlasGemv");                                \
  }

#define MUSA_BLAS_STUB_TRSM(type)                                               \
  bool MUSABlas::DoBlasTrsm(Stream* stream, blas::Side side,                    \
                            blas::UpperLower uplo, blas::Transpose transa,      \
                            blas::Diagonal diag, uint64_t m, uint64 n,          \
                            type alpha, const DeviceMemory<type>& a, int lda,   \
                            DeviceMemory<type>* b, int ldb) {                   \
    (void)stream;                                                               \
    (void)side;                                                                 \
    (void)uplo;                                                                 \
    (void)transa;                                                               \
    (void)diag;                                                                 \
    (void)m;                                                                    \
    (void)n;                                                                    \
    (void)alpha;                                                                \
    (void)a;                                                                    \
    (void)lda;                                                                  \
    (void)b;                                                                    \
    (void)ldb;                                                                  \
    return UnsupportedBool<type>("DoBlasTrsm");                                \
  }                                                                             \
  bool MUSABlas::DoBlasTrsmBatched(                                             \
      Stream* stream, blas::Side side, blas::UpperLower uplo,                  \
      blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64 n,       \
      type alpha, const DeviceMemory<type*>& as, int lda,                      \
      DeviceMemory<type*>* bs, int ldb, int batch_count) {                     \
    (void)stream;                                                               \
    (void)side;                                                                 \
    (void)uplo;                                                                 \
    (void)transa;                                                               \
    (void)diag;                                                                 \
    (void)m;                                                                    \
    (void)n;                                                                    \
    (void)alpha;                                                                \
    (void)as;                                                                   \
    (void)lda;                                                                  \
    (void)bs;                                                                   \
    (void)ldb;                                                                  \
    (void)batch_count;                                                          \
    return UnsupportedBool<type>("DoBlasTrsmBatched");                         \
  }

#define MUSA_BLAS_STUB_GEMM_BATCHED(type, alpha_type)                           \
  bool MUSABlas::DoBlasGemmBatched(                                             \
      Stream* stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, alpha_type alpha,                        \
      DeviceMemorySlice<type> a, int lda, DeviceMemorySlice<type> b, int ldb,  \
      alpha_type beta, DeviceMemorySlice<type> c, int ldc, int batch_count,    \
      const NumericOptions& numeric_options,                                    \
      ScratchAllocator* scratch_allocator) {                                    \
    (void)stream;                                                               \
    (void)transa;                                                               \
    (void)transb;                                                               \
    (void)m;                                                                    \
    (void)n;                                                                    \
    (void)k;                                                                    \
    (void)alpha;                                                                \
    (void)a;                                                                    \
    (void)lda;                                                                  \
    (void)b;                                                                    \
    (void)ldb;                                                                  \
    (void)beta;                                                                 \
    (void)c;                                                                    \
    (void)ldc;                                                                  \
    (void)batch_count;                                                          \
    (void)numeric_options;                                                      \
    (void)scratch_allocator;                                                    \
    return UnsupportedBool<type>("DoBlasGemmBatched");                         \
  }

MUSA_BLAS_STUB_AXPY(float)
MUSA_BLAS_STUB_AXPY(double)
MUSA_BLAS_STUB_AXPY(std::complex<float>)
MUSA_BLAS_STUB_AXPY(std::complex<double>)

MUSA_BLAS_STUB_COPY(float)
MUSA_BLAS_STUB_COPY(double)
MUSA_BLAS_STUB_COPY(std::complex<float>)
MUSA_BLAS_STUB_COPY(std::complex<double>)

MUSA_BLAS_STUB_SCAL(float, float)
MUSA_BLAS_STUB_SCAL(double, double)
MUSA_BLAS_STUB_SCAL(std::complex<float>, float)
MUSA_BLAS_STUB_SCAL(std::complex<double>, double)
MUSA_BLAS_STUB_SCAL(std::complex<float>, std::complex<float>)
MUSA_BLAS_STUB_SCAL(std::complex<double>, std::complex<double>)

MUSA_BLAS_STUB_GEMV(float)
MUSA_BLAS_STUB_GEMV(double)
MUSA_BLAS_STUB_GEMV(std::complex<float>)
MUSA_BLAS_STUB_GEMV(std::complex<double>)

bool MUSABlas::DoBlasSbmv(Stream* stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha,
                          const DeviceMemory<float>& a, int lda,
                          const DeviceMemory<float>& x, int incx, float beta,
                          DeviceMemory<float>* y, int incy) {
  (void)stream;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  return UnsupportedBool<float>("DoBlasSbmv");
}

bool MUSABlas::DoBlasSbmv(Stream* stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, double alpha,
                          const DeviceMemory<double>& a, int lda,
                          const DeviceMemory<double>& x, int incx, double beta,
                          DeviceMemory<double>* y, int incy) {
  (void)stream;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  return UnsupportedBool<double>("DoBlasSbmv");
}

MUSA_BLAS_STUB_GEMM_BATCHED(Eigen::half, float)
MUSA_BLAS_STUB_GEMM_BATCHED(Eigen::bfloat16, float)
MUSA_BLAS_STUB_GEMM_BATCHED(float, float)
MUSA_BLAS_STUB_GEMM_BATCHED(double, double)
MUSA_BLAS_STUB_GEMM_BATCHED(std::complex<float>, std::complex<float>)
MUSA_BLAS_STUB_GEMM_BATCHED(std::complex<double>, std::complex<double>)

MUSA_BLAS_STUB_TRSM(float)
MUSA_BLAS_STUB_TRSM(double)
MUSA_BLAS_STUB_TRSM(std::complex<float>)
MUSA_BLAS_STUB_TRSM(std::complex<double>)

#undef MUSA_BLAS_STUB_AXPY
#undef MUSA_BLAS_STUB_COPY
#undef MUSA_BLAS_STUB_SCAL
#undef MUSA_BLAS_STUB_GEMV
#undef MUSA_BLAS_STUB_TRSM
#undef MUSA_BLAS_STUB_GEMM_BATCHED

tsl::Status MUSABlas::GetVersion(std::string* version) {
  if (version != nullptr) {
    *version = "mublas";
  }
  return tsl::OkStatus();
}

}  // namespace musa

void initialize_mublas() {
  tsl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::BlasFactory>(
          musa::kMusaPlatformId, "muBLAS",
          [](internal::StreamExecutorInterface* parent) -> blas::BlasSupport* {
            auto* musa_executor = dynamic_cast<musa::MusaExecutor*>(parent);
            if (musa_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize muBLAS with a non-MUSA "
                            "StreamExecutor";
              return nullptr;
            }
            auto* blas = new musa::MUSABlas(musa_executor);
            if (!blas->Init()) {
              delete blas;
              return nullptr;
            }
            return blas;
          });

  if (!status.ok() && status.code() != absl::StatusCode::kAlreadyExists) {
    LOG(ERROR) << "Unable to register muBLAS factory: " << status.message();
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_mublas,
                            { stream_executor::initialize_mublas(); });
