/* Minimal TensorFlow C API shim for declarations used by pjrt_api.cc. */

#ifndef TENSORFLOW_C_C_API_H_
#define TENSORFLOW_C_C_API_H_

#include <stddef.h>

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TF_StringView {
  const char* data;
  size_t len;
} TF_StringView;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_C_C_API_H_
