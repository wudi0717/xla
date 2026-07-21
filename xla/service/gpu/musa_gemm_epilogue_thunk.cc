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

#include "xla/service/gpu/musa_gemm_epilogue_thunk.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/logging.h"

#if __has_include("mublasLt.h") && __has_include("musa_runtime.h")
#define XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT 1
#include "mublasLt.h"
#include "musa_runtime.h"
#else
#define XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT 0
#endif

namespace xla {
namespace gpu {

namespace {

bool EnvExplicitlyTrue(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0' &&
         std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "False") != 0 &&
         std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

bool IsRowMajor(const MatrixLayout& layout) {
  return layout.order == MatrixLayout::Order::kRowMajor;
}

const char* LayoutOrderString(const MatrixLayout& layout) {
  return IsRowMajor(layout) ? "row_major" : "column_major";
}

#if XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT

std::string MublasLtStatusString(mublasStatus_t status) {
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
      return absl::StrCat("<invalid mublasLt status: ",
                          static_cast<int>(status), ">");
  }
}

Status MublasLtStatus(mublasStatus_t status, const char* op_name) {
  if (status == MUBLAS_STATUS_SUCCESS) {
    return OkStatus();
  }
  return absl::InternalError(
      absl::StrCat(op_name, " failed with ", MublasLtStatusString(status)));
}

Status MusaRuntimeStatus(musaError_t status, const char* op_name) {
  if (status == musaSuccess) {
    return OkStatus();
  }
  return absl::InternalError(
      absl::StrCat(op_name, " failed with code ", static_cast<int>(status)));
}

struct MublasLtHandleDeleter {
  void operator()(mublasLtHandle_t handle) const {
    if (handle != nullptr) {
      mublasLtDestroy(handle);
    }
  }
};

struct MublasLtMatmulDescDeleter {
  void operator()(mublasLtMatmulDesc_t desc) const {
    if (desc != nullptr) {
      mublasLtMatmulDescDestroy(desc);
    }
  }
};

struct MublasLtMatrixLayoutDeleter {
  void operator()(mublasLtMatrixLayout_t layout) const {
    if (layout != nullptr) {
      mublasLtMatrixLayoutDestroy(layout);
    }
  }
};

using MublasLtHandlePtr = std::remove_pointer_t<mublasLtHandle_t>;
using SharedMublasLtHandle = std::shared_ptr<MublasLtHandlePtr>;
using WeakMublasLtHandle = std::weak_ptr<MublasLtHandlePtr>;
using OwnedMublasLtMatmulDesc =
    std::unique_ptr<std::remove_pointer_t<mublasLtMatmulDesc_t>,
                    MublasLtMatmulDescDeleter>;
using OwnedMublasLtMatrixLayout =
    std::unique_ptr<std::remove_pointer_t<mublasLtMatrixLayout_t>,
                    MublasLtMatrixLayoutDeleter>;

StatusOr<SharedMublasLtHandle> CreateMublasLtHandle() {
  mublasLtHandle_t handle = nullptr;
  TF_RETURN_IF_ERROR(MublasLtStatus(mublasLtCreate(&handle),
                                    "mublasLtCreate"));
  return SharedMublasLtHandle(handle, MublasLtHandleDeleter());
}

StatusOr<SharedMublasLtHandle> GetSharedMublasLtHandle(int device_ordinal) {
  static absl::Mutex mu(absl::kConstInit);
  static auto* handles = new std::map<int, WeakMublasLtHandle>();

  {
    absl::MutexLock lock(&mu);
    auto it = handles->find(device_ordinal);
    if (it != handles->end()) {
      SharedMublasLtHandle handle = it->second.lock();
      if (handle != nullptr) {
        return handle;
      }
    }
  }

  if (EnvExplicitlyTrue("MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS")) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=shared_handle_create_start"
              << " device_ordinal=" << device_ordinal;
  }
  TF_RETURN_IF_ERROR(
      MusaRuntimeStatus(musaSetDevice(device_ordinal), "musaSetDevice"));
  TF_ASSIGN_OR_RETURN(SharedMublasLtHandle handle, CreateMublasLtHandle());
  if (EnvExplicitlyTrue("MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS")) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=shared_handle_create_done"
              << " device_ordinal=" << device_ordinal;
  }

  {
    absl::MutexLock lock(&mu);
    auto it = handles->find(device_ordinal);
    if (it != handles->end()) {
      SharedMublasLtHandle existing = it->second.lock();
      if (existing != nullptr) {
        return existing;
      }
    }
    (*handles)[device_ordinal] = handle;
  }
  return handle;
}

StatusOr<OwnedMublasLtMatmulDesc> CreateMublasLtMatmulDesc() {
  mublasLtMatmulDesc_t desc = nullptr;
  TF_RETURN_IF_ERROR(MublasLtStatus(
      mublasLtMatmulDescCreate(&desc, MUBLAS_COMPUTE_32F, MUSA_R_32F),
      "mublasLtMatmulDescCreate"));

  mublasOperation op_n = MUBLAS_OP_N;
  TF_RETURN_IF_ERROR(MublasLtStatus(
      mublasLtMatmulDescSetAttribute(desc, MUBLASLT_MATMUL_DESC_TRANSA, &op_n,
                                     sizeof(op_n)),
      "mublasLtMatmulDescSetAttribute(TRANSA)"));
  TF_RETURN_IF_ERROR(MublasLtStatus(
      mublasLtMatmulDescSetAttribute(desc, MUBLASLT_MATMUL_DESC_TRANSB, &op_n,
                                     sizeof(op_n)),
      "mublasLtMatmulDescSetAttribute(TRANSB)"));

  mublasLtEpilogue_t epilogue = MUBLASLT_EPILOGUE_BIAS;
  TF_RETURN_IF_ERROR(MublasLtStatus(
      mublasLtMatmulDescSetAttribute(desc, MUBLASLT_MATMUL_DESC_EPILOGUE,
                                     &epilogue, sizeof(epilogue)),
      "mublasLtMatmulDescSetAttribute(EPILOGUE)"));
  return OwnedMublasLtMatmulDesc(desc);
}

StatusOr<OwnedMublasLtMatrixLayout> CreateColumnMajorF32Layout(int64_t rows,
                                                               int64_t cols,
                                                               int64_t ld) {
  if (rows <= 0 || cols <= 0 || ld < rows) {
    return absl::InvalidArgumentError(
        absl::StrCat("invalid column-major f32 matrix layout rows=", rows,
                     " cols=", cols, " ld=", ld));
  }
  mublasLtMatrixLayout_t layout = nullptr;
  TF_RETURN_IF_ERROR(MublasLtStatus(
      mublasLtMatrixLayoutCreate(&layout, MUSA_R_32F, rows, cols, ld),
      "mublasLtMatrixLayoutCreate"));
  return OwnedMublasLtMatrixLayout(layout);
}

Status RunMublasLtBiasEpilogue(const GemmConfig& config, se::Stream* stream,
                               mublasLtHandle_t handle,
                               mublasLtMatmulDesc_t op_desc,
                               mublasLtMatrixLayout_t a_desc,
                               mublasLtMatrixLayout_t b_desc,
                               mublasLtMatrixLayout_t c_desc,
                               se::DeviceMemoryBase lhs,
                               se::DeviceMemoryBase rhs,
                               se::DeviceMemoryBase bias,
                               se::DeviceMemoryBase output,
                               bool diagnostics) {
  if (config.lhs_layout.dtype != F32 || config.rhs_layout.dtype != F32 ||
      config.output_layout.dtype != F32) {
    return absl::UnimplementedError(
        "MUSA mublasLt GEMM epilogue currently supports f32 only");
  }
  if (!IsRowMajor(config.lhs_layout) || !IsRowMajor(config.rhs_layout) ||
      !IsRowMajor(config.output_layout)) {
    return absl::UnimplementedError(
        "MUSA mublasLt GEMM epilogue currently supports row-major layouts only");
  }

  const int64_t m = config.output_layout.num_rows;
  const int64_t n = config.output_layout.num_cols;
  const int64_t k = config.lhs_layout.num_cols;
  TF_RETURN_IF_ERROR(MusaRuntimeStatus(
      musaSetDevice(stream->parent()->device_ordinal()), "musaSetDevice"));

  void* bias_addr = bias.opaque();
  TF_RETURN_IF_ERROR(MublasLtStatus(
      mublasLtMatmulDescSetAttribute(op_desc,
                                     MUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                     &bias_addr, sizeof(bias_addr)),
      "mublasLtMatmulDescSetAttribute(BIAS_POINTER)"));

  float alpha = 1.0f;
  float beta = 0.0f;
  musaStream_t musa_stream =
      static_cast<musaStream_t>(stream->platform_specific_handle().stream);
  mublasStatus_t status = mublasLtMatmul(
      handle, op_desc, &alpha, rhs.opaque(), a_desc, lhs.opaque(), b_desc,
      &beta, output.opaque(), c_desc, output.opaque(), c_desc, /*algo=*/nullptr,
      /*workspace=*/nullptr, /*workspaceSizeInBytes=*/0, musa_stream);
  if (diagnostics) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=mublaslt_done"
              << " status=" << MublasLtStatusString(status) << " m=" << m
              << " n=" << n << " k=" << k
              << " lhs_ld=" << config.lhs_layout.leading_dim_stride
              << " rhs_ld=" << config.rhs_layout.leading_dim_stride
              << " out_ld=" << config.output_layout.leading_dim_stride
              << " lt_m=" << n << " lt_n=" << m << " lt_k=" << k;
  }
  return MublasLtStatus(status, "mublasLtMatmul");
}

#endif  // XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT

}  // namespace

#if XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT
struct MusaGemmEpilogueThunk::MublasLtState {
  SharedMublasLtHandle handle;
  OwnedMublasLtMatmulDesc op_desc;
  OwnedMublasLtMatrixLayout a_desc;
  OwnedMublasLtMatrixLayout b_desc;
  OwnedMublasLtMatrixLayout c_desc;
};
#else
struct MusaGemmEpilogueThunk::MublasLtState {};
#endif

MusaGemmEpilogueThunk::MusaGemmEpilogueThunk(
    ThunkInfo thunk_info, GemmConfig config,
    BufferAllocation::Slice lhs_buffer, BufferAllocation::Slice rhs_buffer,
    BufferAllocation::Slice bias_buffer, BufferAllocation::Slice output_buffer)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      bias_buffer_(bias_buffer),
      output_buffer_(output_buffer) {}

MusaGemmEpilogueThunk::~MusaGemmEpilogueThunk() = default;

Status MusaGemmEpilogueThunk::ExecuteOnStream(const ExecuteParams& params) {
  const bool exec_diagnostics =
      EnvExplicitlyTrue("MUSA_GEMM_EPILOGUE_THUNK_EXEC_DIAGNOSTICS");
  const bool verbose =
      EnvExplicitlyTrue("MUSA_GEMM_EPILOGUE_THUNK_VERBOSE");
  const bool use_mublaslt =
      !EnvExplicitlyTrue("MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT");
  if (exec_diagnostics || verbose) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=execute_start"
              << " m=" << config_.output_layout.num_rows
              << " n=" << config_.output_layout.num_cols
              << " k=" << config_.lhs_layout.num_cols
              << " lhs_batch=" << config_.lhs_layout.batch_size
              << " rhs_batch=" << config_.rhs_layout.batch_size
              << " out_batch=" << config_.output_layout.batch_size
              << " lhs_order=" << LayoutOrderString(config_.lhs_layout)
              << " rhs_order=" << LayoutOrderString(config_.rhs_layout)
              << " out_order=" << LayoutOrderString(config_.output_layout)
              << " lhs_stride=" << config_.lhs_layout.leading_dim_stride
              << " rhs_stride=" << config_.rhs_layout.leading_dim_stride
              << " out_stride=" << config_.output_layout.leading_dim_stride
              << " alpha=(" << config_.alpha.real() << ","
              << config_.alpha.imag() << ")"
              << " beta=" << config_.beta
              << " lhs_bytes=" << lhs_buffer_.size()
              << " rhs_bytes=" << rhs_buffer_.size()
              << " bias_bytes=" << bias_buffer_.size()
              << " output_bytes=" << output_buffer_.size()
              << " use_mublaslt=" << use_mublaslt;
  }
  if (config_.lhs_layout.batch_size != 1 ||
      config_.rhs_layout.batch_size != 1 ||
      config_.output_layout.batch_size != 1) {
    LOG(ERROR) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=validation_failed "
               << "reason=batched_gemm"
               << " lhs_batch=" << config_.lhs_layout.batch_size
               << " rhs_batch=" << config_.rhs_layout.batch_size
               << " out_batch=" << config_.output_layout.batch_size;
    return absl::InvalidArgumentError(
        "MUSA GEMM epilogue thunk only supports non-batched GEMM for now");
  }
  if (config_.alpha.real() != 1.0 || config_.alpha.imag() != 0.0) {
    LOG(ERROR) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=validation_failed "
               << "reason=alpha"
               << " alpha=(" << config_.alpha.real() << ","
               << config_.alpha.imag() << ")";
    return absl::InvalidArgumentError(
        "MUSA GEMM epilogue thunk only supports alpha=1");
  }
  if (config_.beta != 0.0) {
    LOG(ERROR) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=validation_failed "
               << "reason=beta beta=" << config_.beta;
    return absl::InvalidArgumentError(
        "MUSA GEMM epilogue thunk only supports beta=0");
  }

  const BufferAllocations& allocs = *params.buffer_allocations;
  if (!use_mublaslt) {
    return absl::UnimplementedError(
        "MUSA GEMM epilogue mublasLt path disabled");
  }
#if XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT
  if (mublaslt_state_ == nullptr) {
    return absl::InternalError(
        "MUSA GEMM epilogue mublasLt state was not initialized");
  }
  Status run_status = RunMublasLtBiasEpilogue(
      config_, params.stream, mublaslt_state_->handle.get(),
      mublaslt_state_->op_desc.get(), mublaslt_state_->a_desc.get(),
      mublaslt_state_->b_desc.get(), mublaslt_state_->c_desc.get(),
      allocs.GetDeviceAddress(lhs_buffer_),
      allocs.GetDeviceAddress(rhs_buffer_),
      allocs.GetDeviceAddress(bias_buffer_),
      allocs.GetDeviceAddress(output_buffer_), exec_diagnostics || verbose);
  if (!run_status.ok()) {
    LOG(ERROR) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=mublaslt_failed"
               << " m=" << config_.output_layout.num_rows
               << " n=" << config_.output_layout.num_cols
               << " k=" << config_.lhs_layout.num_cols
               << " lhs_order=" << LayoutOrderString(config_.lhs_layout)
               << " rhs_order=" << LayoutOrderString(config_.rhs_layout)
               << " out_order=" << LayoutOrderString(config_.output_layout)
               << " status=" << run_status;
    return run_status;
  }
#else
  return absl::UnimplementedError(
      "MUSA GEMM epilogue mublasLt path requires mublasLt.h and "
      "musa_runtime.h at compile time");
#endif  // XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT
  if (exec_diagnostics || verbose) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=execute_done";
  }
  return OkStatus();
}

Status MusaGemmEpilogueThunk::Initialize(const GpuExecutable& /*executable*/,
                                         se::StreamExecutor* executor) {
#if XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT
  const bool diagnostics =
      EnvExplicitlyTrue("MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS");
  if (EnvExplicitlyTrue("MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT")) {
    return OkStatus();
  }

  auto state = std::make_unique<MublasLtState>();
  if (diagnostics) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=initialize_start"
              << " device_ordinal=" << executor->device_ordinal()
              << " m=" << config_.output_layout.num_rows
              << " n=" << config_.output_layout.num_cols
              << " k=" << config_.lhs_layout.num_cols;
  }
  TF_ASSIGN_OR_RETURN(SharedMublasLtHandle handle,
                      GetSharedMublasLtHandle(executor->device_ordinal()));
  state->handle = std::move(handle);
  if (diagnostics) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=handle_ready"
              << " device_ordinal=" << executor->device_ordinal();
  }
  TF_ASSIGN_OR_RETURN(OwnedMublasLtMatmulDesc op_desc,
                      CreateMublasLtMatmulDesc());
  state->op_desc = std::move(op_desc);

  const int64_t m = config_.output_layout.num_rows;
  const int64_t n = config_.output_layout.num_cols;
  const int64_t k = config_.lhs_layout.num_cols;
  // XLA gives us row-major C[m, n] = A[m, k] * B[k, n].  Express it as
  // column-major C^T[n, m] = B^T[n, k] * A^T[k, m], matching the path used by
  // RunGemm before handing row-major outputs to BLAS.
  TF_ASSIGN_OR_RETURN(
      OwnedMublasLtMatrixLayout a_desc,
      CreateColumnMajorF32Layout(n, k,
                                 config_.rhs_layout.leading_dim_stride));
  state->a_desc = std::move(a_desc);
  TF_ASSIGN_OR_RETURN(
      OwnedMublasLtMatrixLayout b_desc,
      CreateColumnMajorF32Layout(k, m,
                                 config_.lhs_layout.leading_dim_stride));
  state->b_desc = std::move(b_desc);
  TF_ASSIGN_OR_RETURN(
      OwnedMublasLtMatrixLayout c_desc,
      CreateColumnMajorF32Layout(n, m,
                                 config_.output_layout.leading_dim_stride));
  state->c_desc = std::move(c_desc);
  mublaslt_state_ = std::move(state);
  if (diagnostics) {
    LOG(INFO) << "[MUSA_GEMM_EPILOGUE_THUNK] stage=initialize_done"
              << " m=" << m << " n=" << n << " k=" << k
              << " lhs_ld=" << config_.lhs_layout.leading_dim_stride
              << " rhs_ld=" << config_.rhs_layout.leading_dim_stride
              << " out_ld=" << config_.output_layout.leading_dim_stride
              << " cached_mublaslt_state=1";
  }
#endif  // XLA_MUSA_GEMM_EPILOGUE_HAS_MUBLASLT
  return OkStatus();
}

std::string MusaGemmEpilogueThunk::ToStringExtra(int /*indent*/) const {
  return absl::StrCat("m=", config_.output_layout.num_rows,
                      ", n=", config_.output_layout.num_cols,
                      ", k=", config_.lhs_layout.num_cols,
                      ", output_bytes=", output_buffer_.size());
}

}  // namespace gpu
}  // namespace xla
