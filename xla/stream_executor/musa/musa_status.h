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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_STATUS_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_STATUS_H_

#include "absl/base/optimization.h"
#include "tsl/platform/status.h"

#include "musa.h"
#include "musa_runtime.h"

namespace stream_executor {
namespace musa {

namespace internal {
tsl::Status ToStatusSlow(MUresult result, const char* detail);
tsl::Status ToStatusSlow(musaError_t result, const char* detail);
}  // namespace internal

inline tsl::Status ToStatus(MUresult result, const char* detail = "") {
  if (ABSL_PREDICT_TRUE(result == MUSA_SUCCESS)) {
    return ::tsl::OkStatus();
  }
  return internal::ToStatusSlow(result, detail);
}

inline tsl::Status ToStatus(musaError_t result, const char* detail = "") {
  if (ABSL_PREDICT_TRUE(result == musaSuccess)) {
    return ::tsl::OkStatus();
  }
  return internal::ToStatusSlow(result, detail);
}

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_STATUS_H_
