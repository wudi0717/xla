/* Minimal TensorFlow status C API shim backed by TSL_Status. */

#ifndef TENSORFLOW_C_TF_STATUS_H_
#define TENSORFLOW_C_TF_STATUS_H_

#include "tsl/c/tsl_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef TSL_Status TF_Status;
typedef TSL_Code TF_Code;

#define TF_OK TSL_OK
#define TF_CANCELLED TSL_CANCELLED
#define TF_UNKNOWN TSL_UNKNOWN
#define TF_INVALID_ARGUMENT TSL_INVALID_ARGUMENT
#define TF_DEADLINE_EXCEEDED TSL_DEADLINE_EXCEEDED
#define TF_NOT_FOUND TSL_NOT_FOUND
#define TF_ALREADY_EXISTS TSL_ALREADY_EXISTS
#define TF_PERMISSION_DENIED TSL_PERMISSION_DENIED
#define TF_UNAUTHENTICATED TSL_UNAUTHENTICATED
#define TF_RESOURCE_EXHAUSTED TSL_RESOURCE_EXHAUSTED
#define TF_FAILED_PRECONDITION TSL_FAILED_PRECONDITION
#define TF_ABORTED TSL_ABORTED
#define TF_OUT_OF_RANGE TSL_OUT_OF_RANGE
#define TF_UNIMPLEMENTED TSL_UNIMPLEMENTED
#define TF_INTERNAL TSL_INTERNAL
#define TF_UNAVAILABLE TSL_UNAVAILABLE
#define TF_DATA_LOSS TSL_DATA_LOSS

static inline TF_Status* TF_NewStatus(void) { return TSL_NewStatus(); }

static inline void TF_DeleteStatus(TF_Status* status) {
  TSL_DeleteStatus(status);
}

static inline void TF_SetStatus(TF_Status* status, TF_Code code,
                                const char* msg) {
  TSL_SetStatus(status, code, msg);
}

static inline TF_Code TF_GetCode(const TF_Status* status) {
  return TSL_GetCode(status);
}

static inline const char* TF_Message(const TF_Status* status) {
  return TSL_Message(status);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_C_TF_STATUS_H_
