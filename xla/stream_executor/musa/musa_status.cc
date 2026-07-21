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

#include "xla/stream_executor/musa/musa_status.h"

#include "absl/strings/str_cat.h"
#include "tsl/platform/errors.h"

namespace stream_executor {
namespace musa {
namespace internal {

tsl::Status ToStatusSlow(MUresult result, const char* detail) {
  const char* error_name = nullptr;
  const char* error_string = nullptr;
  (void)muGetErrorName(result, &error_name);
  (void)muGetErrorString(result, &error_string);
  std::string message = absl::StrCat(
      detail, detail[0] == '\0' ? "" : ": ",
      error_name != nullptr ? error_name : "UNKNOWN MUSA DRIVER ERROR");
  if (error_string != nullptr) {
    absl::StrAppend(&message, ": ", error_string);
  }
  if (result == MUSA_ERROR_OUT_OF_MEMORY) {
    return tsl::errors::ResourceExhausted(message);
  }
  if (result == MUSA_ERROR_NOT_FOUND) {
    return tsl::errors::NotFound(message);
  }
  return tsl::errors::Internal(message);
}

tsl::Status ToStatusSlow(musaError_t result, const char* detail) {
  const char* error_name = musaGetErrorName(result);
  const char* error_string = musaGetErrorString(result);
  std::string message = absl::StrCat(
      detail, detail[0] == '\0' ? "" : ": ",
      error_name != nullptr ? error_name : "UNKNOWN MUSA RUNTIME ERROR");
  if (error_string != nullptr) {
    absl::StrAppend(&message, ": ", error_string);
  }
  return tsl::errors::Internal(message);
}

}  // namespace internal
}  // namespace musa
}  // namespace stream_executor
