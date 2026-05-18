#ifndef XLA_STREAM_EXECUTOR_MUSA_MUBLAS_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUBLAS_WRAPPER_H_

#define MUBLAS_BETA_FEATURES_API

#include "mublas.h"

namespace stream_executor {
namespace wrap {

#define MUBLAS_API_WRAPPER(__name)                  \
  struct WrapperShim__##__name {                   \
    constexpr static const char* kName = #__name;  \
    template <typename... Args>                    \
    auto operator()(Args... args) const {          \
      return (::__name)(args...);                  \
    }                                              \
  } __name;

#define FOREACH_MUBLAS_API(__macro)  \
  __macro(mublasCreate)              \
  __macro(mublasDestroy)             \
  __macro(mublasSetStream)           \
  __macro(mublasGetStream)           \
  __macro(mublasSetAtomicsMode)      \
  __macro(mublasGemmEx)              \
  __macro(mublasGemmStridedBatchedEx)\
  __macro(mublasSgemm)               \
  __macro(mublasDgemm)               \
  __macro(mublasCgemm)               \
  __macro(mublasZgemm)               \
  __macro(mublasSgemmStridedBatched) \
  __macro(mublasDgemmStridedBatched) \
  __macro(mublasCgemmStridedBatched) \
  __macro(mublasZgemmStridedBatched)

FOREACH_MUBLAS_API(MUBLAS_API_WRAPPER)

#undef FOREACH_MUBLAS_API
#undef MUBLAS_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUBLAS_WRAPPER_H_
