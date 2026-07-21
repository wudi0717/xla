#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "Eigen/Core"
#include "tsl/platform/test.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {
namespace {

constexpr int64_t kM = 96;
constexpr int64_t kN = 80;
constexpr int64_t kK = 48;
constexpr int kBatchCount = 3;

tsl::StatusOr<StreamExecutor*> GetExecutor() {
  TF_ASSIGN_OR_RETURN(Platform * platform,
                      MultiPlatformManager::PlatformWithName("MUSA"));
  if (platform->VisibleDeviceCount() <= 0) {
    return tsl::errors::NotFound("No visible MUSA device");
  }
  return platform->ExecutorForDevice(0);
}

template <typename T>
T CastFromDouble(double v) {
  return static_cast<T>(v);
}

template <typename T>
double CastToDouble(T v) {
  return static_cast<double>(v);
}

template <typename T>
void FillMatrix(std::vector<T>* data, int rows, int cols, int ld, int seed) {
  data->assign(ld * cols, CastFromDouble<T>(0.0));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const double val =
          ((seed * 131 + r * 17 + c * 19) % 97 - 48) / 512.0;
      (*data)[r + c * ld] = CastFromDouble<T>(val);
    }
  }
}

template <typename T>
void ComputeReferenceGemm(blas::Transpose transa, blas::Transpose transb,
                          int m, int n, int k, double alpha,
                          const std::vector<T>& a, int lda,
                          const std::vector<T>& b, int ldb, double beta,
                          std::vector<T>* c, int ldc) {
  auto get_a = [&](int row, int col) -> double {
    if (transa == blas::Transpose::kNoTranspose) {
      return CastToDouble(a[row + col * lda]);
    }
    return CastToDouble(a[col + row * lda]);
  };
  auto get_b = [&](int row, int col) -> double {
    if (transb == blas::Transpose::kNoTranspose) {
      return CastToDouble(b[row + col * ldb]);
    }
    return CastToDouble(b[col + row * ldb]);
  };
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double acc = 0.0;
      for (int p = 0; p < k; ++p) {
        acc += get_a(row, p) * get_b(p, col);
      }
      const int idx = row + col * ldc;
      const double prev = CastToDouble((*c)[idx]);
      (*c)[idx] = CastFromDouble<T>(alpha * acc + beta * prev);
    }
  }
}

template <typename T>
double Tolerance() {
  if constexpr (std::is_same_v<T, double>) {
    return 1e-9;
  } else if constexpr (std::is_same_v<T, float>) {
    return 2e-4;
  }
  return 2e-1;
}

template <typename T>
void ExpectNearMatrix(const std::vector<T>& got, const std::vector<T>& expect,
                      int rows, int cols, int ld) {
  const double tol = Tolerance<T>();
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int idx = r + c * ld;
      const double g = CastToDouble(got[idx]);
      const double e = CastToDouble(expect[idx]);
      EXPECT_NEAR(g, e, tol) << "Mismatch at (" << r << ", " << c << ")";
    }
  }
}

template <typename T>
blas::DataType DType();

template <>
blas::DataType DType<float>() {
  return blas::DataType::kFloat;
}
template <>
blas::DataType DType<double>() {
  return blas::DataType::kDouble;
}
template <>
blas::DataType DType<Eigen::half>() {
  return blas::DataType::kHalf;
}
template <>
blas::DataType DType<Eigen::bfloat16>() {
  return blas::DataType::kBF16;
}

template <typename T>
struct AlphaBetaType {
  using Type = T;
};
template <>
struct AlphaBetaType<Eigen::half> {
  using Type = float;
};
template <>
struct AlphaBetaType<Eigen::bfloat16> {
  using Type = float;
};

bool IsKnownMudnnNotSupported(const tsl::Status& status) {
  return !status.ok() &&
         (status.message().find("muDNN status 1") != std::string::npos ||
          status.message().find("muDNN status 4") != std::string::npos);
}

std::string DataTypeName(blas::DataType dtype) {
  return blas::DataTypeString(dtype);
}

std::string TransposeName(blas::Transpose t) {
  switch (t) {
    case blas::Transpose::kNoTranspose:
      return "N";
    case blas::Transpose::kTranspose:
      return "T";
    case blas::Transpose::kConjugateTranspose:
      return "C";
  }
  return "?";
}

std::string TransposePairName(blas::Transpose transa, blas::Transpose transb) {
  return TransposeName(transa) + TransposeName(transb);
}

template <typename T>
void FillInterleavedBatchedMatrix(std::vector<T>* data, int batch_count, int rows,
                                  int cols, int ld, int64_t batch_stride,
                                  int seed) {
  data->assign(ld * cols, CastFromDouble<T>(0.0));
  for (int batch = 0; batch < batch_count; ++batch) {
    for (int c = 0; c < cols; ++c) {
      for (int r = 0; r < rows; ++r) {
        const double val =
            ((seed * 131 + batch * 29 + r * 17 + c * 19) % 97 - 48) / 512.0;
        (*data)[batch * batch_stride + r + c * ld] = CastFromDouble<T>(val);
      }
    }
  }
}

int StoredRows(blas::Transpose transpose, int logical_rows, int logical_cols) {
  return transpose == blas::Transpose::kNoTranspose ? logical_rows
                                                    : logical_cols;
}

int StoredCols(blas::Transpose transpose, int logical_rows, int logical_cols) {
  return transpose == blas::Transpose::kNoTranspose ? logical_cols
                                                    : logical_rows;
}

int LeadingDim(blas::Transpose transpose, int logical_rows, int logical_cols,
               int ld_extra) {
  return StoredRows(transpose, logical_rows, logical_cols) + ld_extra;
}

template <typename T>
void RunOneGemmCase(StreamExecutor* executor, blas::Transpose transa,
                    blas::Transpose transb, int lda_extra = 0,
                    int ldb_extra = 0, int ldc_extra = 0) {
  auto* blas = executor->AsBlas();
  ASSERT_NE(blas, nullptr);
  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());
  ASSERT_TRUE(stream->ok());

  constexpr int m = static_cast<int>(kM);
  constexpr int n = static_cast<int>(kN);
  constexpr int k = static_cast<int>(kK);
  const int a_rows = StoredRows(transa, m, k);
  const int a_cols = StoredCols(transa, m, k);
  const int b_rows = StoredRows(transb, k, n);
  const int b_cols = StoredCols(transb, k, n);
  const int lda = LeadingDim(transa, m, k, lda_extra);
  const int ldb = LeadingDim(transb, k, n, ldb_extra);
  const int ldc = m + ldc_extra;

  using Scalar = typename AlphaBetaType<T>::Type;
  const Scalar alpha = static_cast<Scalar>(1.25);
  const Scalar beta = static_cast<Scalar>(0.0);

  std::vector<T> host_a;
  std::vector<T> host_b;
  std::vector<T> host_c(m * ldc, CastFromDouble<T>(0.0));
  FillMatrix(&host_a, a_rows, a_cols, lda, /*seed=*/7);
  FillMatrix(&host_b, b_rows, b_cols, ldb, /*seed=*/11);

  std::vector<T> expect_c = host_c;
  ComputeReferenceGemm(transa, transb, m, n, k, static_cast<double>(alpha),
                       host_a, lda, host_b, ldb, static_cast<double>(beta),
                       &expect_c, ldc);

  DeviceMemory<T> dev_a = executor->AllocateArray<T>(host_a.size(), 0);
  DeviceMemory<T> dev_b = executor->AllocateArray<T>(host_b.size(), 0);
  DeviceMemory<T> dev_c = executor->AllocateArray<T>(host_c.size(), 0);
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_a, host_a.data(), host_a.size() * sizeof(T)).ok());
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_b, host_b.data(), host_b.size() * sizeof(T)).ok());
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_c, host_c.data(), host_c.size() * sizeof(T)).ok());

  tsl::Status gemm_status = blas->DoBlasGemm(
      stream.get(), transa, transb, m, n, k, DType<T>(), &alpha, dev_a, lda, dev_b,
      ldb, &beta, &dev_c, ldc, NumericOptions());
  if (!gemm_status.ok()) {
    EXPECT_TRUE(IsKnownMudnnNotSupported(gemm_status)) << gemm_status;
    std::cout << "[MUDNN_CASE] kind=gemm mode=" << TransposePairName(transa, transb)
              << " dtype=" << DataTypeName(DType<T>())
              << " result=NOT_SUPPORTED status=\"" << gemm_status.message()
              << "\"\n";
    return;
  }

  std::vector<T> got_c(host_c.size());
  ASSERT_TRUE(
      stream->ThenMemcpy(got_c.data(), dev_c, got_c.size() * sizeof(T)).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  ExpectNearMatrix(got_c, expect_c, m, n, ldc);
  std::cout << "[MUDNN_CASE] kind=gemm mode=" << TransposePairName(transa, transb)
            << " dtype=" << DataTypeName(DType<T>()) << " result=SUPPORTED\n";
}

template <typename T>
void RunOneStridedBatchedCase(StreamExecutor* executor, blas::Transpose transa,
                              blas::Transpose transb, int lda_extra = 0,
                              int ldb_extra = 0, int ldc_extra = 0,
                              int64_t stride_a_extra = 0,
                              int64_t stride_b_extra = 0,
                              int64_t stride_c_extra = 0) {
  auto* blas = executor->AsBlas();
  ASSERT_NE(blas, nullptr);
  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());
  ASSERT_TRUE(stream->ok());

  constexpr int m = static_cast<int>(kM);
  constexpr int n = static_cast<int>(kN);
  constexpr int k = static_cast<int>(kK);
  const int a_rows = StoredRows(transa, m, k);
  const int a_cols = StoredCols(transa, m, k);
  const int b_rows = StoredRows(transb, k, n);
  const int b_cols = StoredCols(transb, k, n);
  const int lda = LeadingDim(transa, m, k, lda_extra);
  const int ldb = LeadingDim(transb, k, n, ldb_extra);
  const int ldc = m + ldc_extra;
  const int64_t stride_a = static_cast<int64_t>(lda) * a_cols + stride_a_extra;
  const int64_t stride_b = static_cast<int64_t>(ldb) * b_cols + stride_b_extra;
  const int64_t stride_c = static_cast<int64_t>(ldc) * n + stride_c_extra;

  using Scalar = typename AlphaBetaType<T>::Type;
  const Scalar alpha = static_cast<Scalar>(0.85);
  const Scalar beta = static_cast<Scalar>(0.0);

  std::vector<T> host_a(stride_a * kBatchCount);
  std::vector<T> host_b(stride_b * kBatchCount);
  std::vector<T> host_c(stride_c * kBatchCount, CastFromDouble<T>(0.0));
  std::vector<T> expect_c = host_c;

  for (int batch = 0; batch < kBatchCount; ++batch) {
    std::vector<T> one_a;
    std::vector<T> one_b;
    FillMatrix(&one_a, a_rows, a_cols, lda, /*seed=*/19 + batch * 3);
    FillMatrix(&one_b, b_rows, b_cols, ldb, /*seed=*/29 + batch * 5);
    std::copy(one_a.begin(), one_a.end(), host_a.begin() + batch * stride_a);
    std::copy(one_b.begin(), one_b.end(), host_b.begin() + batch * stride_b);

    std::vector<T> one_c(stride_c, CastFromDouble<T>(0.0));
    ComputeReferenceGemm(transa, transb, m, n, k, static_cast<double>(alpha),
                         one_a, lda, one_b, ldb, static_cast<double>(beta), &one_c,
                         ldc);
    std::copy(one_c.begin(), one_c.end(), expect_c.begin() + batch * stride_c);
  }

  DeviceMemory<T> dev_a = executor->AllocateArray<T>(host_a.size(), 0);
  DeviceMemory<T> dev_b = executor->AllocateArray<T>(host_b.size(), 0);
  DeviceMemory<T> dev_c = executor->AllocateArray<T>(host_c.size(), 0);
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_a, host_a.data(), host_a.size() * sizeof(T)).ok());
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_b, host_b.data(), host_b.size() * sizeof(T)).ok());
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_c, host_c.data(), host_c.size() * sizeof(T)).ok());

  tsl::Status gemm_batched_status = blas->DoBlasGemmStridedBatched(
      stream.get(), transa, transb, m, n, k, DType<T>(), &alpha, dev_a, lda, stride_a,
      dev_b, ldb, stride_b, &beta, &dev_c, ldc, stride_c, kBatchCount,
      NumericOptions());
  if (!gemm_batched_status.ok()) {
    EXPECT_TRUE(IsKnownMudnnNotSupported(gemm_batched_status))
        << gemm_batched_status;
    std::cout << "[MUDNN_CASE] kind=strided_batched mode="
              << TransposePairName(transa, transb)
              << " dtype=" << DataTypeName(DType<T>())
              << " result=NOT_SUPPORTED status=\""
              << gemm_batched_status.message() << "\"\n";
    return;
  }

  std::vector<T> got_c(host_c.size());
  ASSERT_TRUE(
      stream->ThenMemcpy(got_c.data(), dev_c, got_c.size() * sizeof(T)).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  for (int batch = 0; batch < kBatchCount; ++batch) {
    const std::vector<T> got_one(got_c.begin() + batch * stride_c,
                                 got_c.begin() + (batch + 1) * stride_c);
    const std::vector<T> expect_one(expect_c.begin() + batch * stride_c,
                                    expect_c.begin() + (batch + 1) * stride_c);
    ExpectNearMatrix(got_one, expect_one, m, n, ldc);
  }
  std::cout << "[MUDNN_CASE] kind=strided_batched mode="
            << TransposePairName(transa, transb)
            << " dtype=" << DataTypeName(DType<T>()) << " result=SUPPORTED\n";
}

template <typename T>
void RunAllTransposeCases(StreamExecutor* executor, bool batched,
                          int lda_extra = 0, int ldb_extra = 0,
                          int ldc_extra = 0, int64_t stride_a_extra = 0,
                          int64_t stride_b_extra = 0,
                          int64_t stride_c_extra = 0) {
  constexpr std::array<blas::Transpose, 4> kTransposes = {
      blas::Transpose::kNoTranspose, blas::Transpose::kTranspose,
      blas::Transpose::kNoTranspose, blas::Transpose::kTranspose};
  constexpr std::array<blas::Transpose, 4> kTransposesB = {
      blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
      blas::Transpose::kTranspose, blas::Transpose::kTranspose};
  for (size_t i = 0; i < kTransposes.size(); ++i) {
    if (batched) {
      RunOneStridedBatchedCase<T>(executor, kTransposes[i], kTransposesB[i],
                                  lda_extra, ldb_extra, ldc_extra,
                                  stride_a_extra, stride_b_extra,
                                  stride_c_extra);
    } else {
      RunOneGemmCase<T>(executor, kTransposes[i], kTransposesB[i], lda_extra,
                        ldb_extra, ldc_extra);
    }
  }
}

template <typename T>
void RunObservedModelStridedBatchedCase(StreamExecutor* executor, int m, int n,
                                        int k, int batch_count, int lda,
                                        int64_t stride_a, int ldb,
                                        int64_t stride_b, int ldc,
                                        int64_t stride_c) {
  auto* blas = executor->AsBlas();
  ASSERT_NE(blas, nullptr);
  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());
  ASSERT_TRUE(stream->ok());

  constexpr blas::Transpose kTransA = blas::Transpose::kNoTranspose;
  constexpr blas::Transpose kTransB = blas::Transpose::kNoTranspose;
  using Scalar = typename AlphaBetaType<T>::Type;
  const Scalar alpha = static_cast<Scalar>(1.0);
  const Scalar beta = static_cast<Scalar>(0.0);

  std::vector<T> host_a(stride_a * batch_count, CastFromDouble<T>(0.0));
  std::vector<T> host_b;
  std::vector<T> host_c(stride_c * batch_count, CastFromDouble<T>(0.0));
  std::vector<T> expect_c = host_c;

  FillInterleavedBatchedMatrix(&host_b, batch_count, k, n, ldb, stride_b,
                               /*seed=*/43);

  for (int batch = 0; batch < batch_count; ++batch) {
    std::vector<T> one_a;
    FillMatrix(&one_a, m, k, lda, /*seed=*/19 + batch * 3);
    std::copy(one_a.begin(), one_a.end(), host_a.begin() + batch * stride_a);

    std::vector<T> one_b(ldb * n, CastFromDouble<T>(0.0));
    for (int c = 0; c < n; ++c) {
      for (int r = 0; r < k; ++r) {
        one_b[r + c * ldb] =
            host_b[batch * stride_b + r + c * ldb];
      }
    }

    std::vector<T> one_c(stride_c, CastFromDouble<T>(0.0));
    ComputeReferenceGemm(kTransA, kTransB, m, n, k, static_cast<double>(alpha),
                         one_a, lda, one_b, ldb, static_cast<double>(beta),
                         &one_c, ldc);
    std::copy(one_c.begin(), one_c.end(), expect_c.begin() + batch * stride_c);
  }

  DeviceMemory<T> dev_a = executor->AllocateArray<T>(host_a.size(), 0);
  DeviceMemory<T> dev_b = executor->AllocateArray<T>(host_b.size(), 0);
  DeviceMemory<T> dev_c = executor->AllocateArray<T>(host_c.size(), 0);
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_a, host_a.data(), host_a.size() * sizeof(T)).ok());
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_b, host_b.data(), host_b.size() * sizeof(T)).ok());
  ASSERT_TRUE(
      stream->ThenMemcpy(&dev_c, host_c.data(), host_c.size() * sizeof(T)).ok());

  tsl::Status status = blas->DoBlasGemmStridedBatched(
      stream.get(), kTransA, kTransB, m, n, k, DType<T>(), &alpha, dev_a, lda,
      stride_a, dev_b, ldb, stride_b, &beta, &dev_c, ldc, stride_c,
      batch_count, NumericOptions());
  ASSERT_TRUE(status.ok()) << status;

  std::vector<T> got_c(host_c.size());
  ASSERT_TRUE(
      stream->ThenMemcpy(got_c.data(), dev_c, got_c.size() * sizeof(T)).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  for (int batch = 0; batch < batch_count; ++batch) {
    const std::vector<T> got_one(got_c.begin() + batch * stride_c,
                                 got_c.begin() + (batch + 1) * stride_c);
    const std::vector<T> expect_one(expect_c.begin() + batch * stride_c,
                                    expect_c.begin() + (batch + 1) * stride_c);
    ExpectNearMatrix(got_one, expect_one, m, n, ldc);
  }
}

TEST(MusaBlasMudnnTest, NonBatchedCoversFloatDoubleHalfBf16AndAllTransposes) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  RunAllTransposeCases<float>(executor, /*batched=*/false);
  RunAllTransposeCases<double>(executor, /*batched=*/false);
  RunAllTransposeCases<Eigen::half>(executor, /*batched=*/false);
  RunAllTransposeCases<Eigen::bfloat16>(executor, /*batched=*/false);
}

TEST(MusaBlasMudnnTest,
     StridedBatchedCoversFloatDoubleHalfBf16AndAllTransposes) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  RunAllTransposeCases<float>(executor, /*batched=*/true);
  RunAllTransposeCases<double>(executor, /*batched=*/true);
  RunAllTransposeCases<Eigen::half>(executor, /*batched=*/true);
  RunAllTransposeCases<Eigen::bfloat16>(executor, /*batched=*/true);
}

TEST(MusaBlasMudnnTest,
     NonBatchedNonContiguousLeadingDimsCoverAllTransposes) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  RunAllTransposeCases<float>(executor, /*batched=*/false, /*lda_extra=*/5,
                              /*ldb_extra=*/7, /*ldc_extra=*/9);
  RunAllTransposeCases<double>(executor, /*batched=*/false, /*lda_extra=*/5,
                               /*ldb_extra=*/7, /*ldc_extra=*/9);
  RunAllTransposeCases<Eigen::half>(executor, /*batched=*/false,
                                    /*lda_extra=*/5, /*ldb_extra=*/7,
                                    /*ldc_extra=*/9);
  RunAllTransposeCases<Eigen::bfloat16>(executor, /*batched=*/false,
                                        /*lda_extra=*/5, /*ldb_extra=*/7,
                                        /*ldc_extra=*/9);
}

TEST(MusaBlasMudnnTest,
     DISABLED_StridedBatchedNonContiguousLayoutCoverAllTransposes) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  RunAllTransposeCases<float>(executor, /*batched=*/true, /*lda_extra=*/3,
                              /*ldb_extra=*/5, /*ldc_extra=*/7,
                              /*stride_a_extra=*/11, /*stride_b_extra=*/13,
                              /*stride_c_extra=*/17);
  RunAllTransposeCases<double>(executor, /*batched=*/true, /*lda_extra=*/3,
                               /*ldb_extra=*/5, /*ldc_extra=*/7,
                               /*stride_a_extra=*/11, /*stride_b_extra=*/13,
                               /*stride_c_extra=*/17);
  RunAllTransposeCases<Eigen::half>(executor, /*batched=*/true,
                                    /*lda_extra=*/3, /*ldb_extra=*/5,
                                    /*ldc_extra=*/7, /*stride_a_extra=*/11,
                                    /*stride_b_extra=*/13,
                                    /*stride_c_extra=*/17);
  RunAllTransposeCases<Eigen::bfloat16>(executor, /*batched=*/true,
                                        /*lda_extra=*/3, /*ldb_extra=*/5,
                                        /*ldc_extra=*/7, /*stride_a_extra=*/11,
                                        /*stride_b_extra=*/13,
                                        /*stride_c_extra=*/17);
}

TEST(MusaBlasMudnnTest, ModelObservedInterleavedBOperandFallsBackCorrectly) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  RunObservedModelStridedBatchedCase<float>(
      executor,
      /*m=*/344,
      /*n=*/100,
      /*k=*/172,
      /*batch_count=*/43,
      /*lda=*/344,
      /*stride_a=*/59168,
      /*ldb=*/7396,
      /*stride_b=*/172,
      /*ldc=*/344,
      /*stride_c=*/34400);
  RunObservedModelStridedBatchedCase<float>(
      executor,
      /*m=*/172,
      /*n=*/100,
      /*k=*/344,
      /*batch_count=*/43,
      /*lda=*/172,
      /*stride_a=*/59168,
      /*ldb=*/14792,
      /*stride_b=*/344,
      /*ldc=*/172,
      /*stride_c=*/17200);
}

}  // namespace
}  // namespace stream_executor::musa
