#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_

#include <complex>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "mublas.h"
#include "mudnn.h"
#include "muComplex.h"
#include "xla/stream_executor/blas.h"

namespace stream_executor {
class Stream;

namespace musa {

class MusaExecutor;

class MUSABlas : public blas::BlasSupport {
 public:
  explicit MUSABlas(MusaExecutor* parent);

  bool Init();
  ~MUSABlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

 private:
  bool SetStream(Stream* stream) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  template <typename FuncT, typename... Args>
  tsl::Status DoBlasInternalImpl(FuncT func, Stream* stream,
                                 bool err_on_failure, Args... args);

  template <typename FuncT, typename... Args>
  bool DoBlasInternal(FuncT func, Stream* stream, Args... args) {
    return DoBlasInternalImpl(func, stream, /*err_on_failure=*/true,
                              args...).ok();
  }

  template <typename FuncT, typename... Args>
  tsl::Status DoBlasInternalStatus(FuncT func, Stream* stream, Args... args) {
    return DoBlasInternalImpl(func, stream, /*err_on_failure=*/true, args...);
  }

  MusaExecutor* parent_;
  absl::Mutex mu_;
  mublasHandle_t blas_ ABSL_GUARDED_BY(mu_) = nullptr;
  std::unique_ptr<::musa::dnn::Handle> dnn_handle_ ABSL_GUARDED_BY(mu_);
  musaStream_t current_stream_ ABSL_GUARDED_BY(mu_) = nullptr;
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_
