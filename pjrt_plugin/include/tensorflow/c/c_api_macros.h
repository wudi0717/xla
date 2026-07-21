/* Minimal TensorFlow C API macro shim for the copied PJRT plugin. */

#ifndef TENSORFLOW_C_C_API_MACROS_H_
#define TENSORFLOW_C_C_API_MACROS_H_

#include <stddef.h>

#ifndef TF_CAPI_EXPORT
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif

#ifndef TF_OFFSET_OF_END
#define TF_OFFSET_OF_END(struct_type, last_field) \
  (offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field))
#endif

#endif  // TENSORFLOW_C_C_API_MACROS_H_
