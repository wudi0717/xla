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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_

#include "musa.h"
#include "xla/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

class MusaEvent : public ::stream_executor::internal::EventInterface {
 public:
  MusaEvent() = default;

  MUevent handle() const { return handle_; }
  void set_handle(MUevent handle) { handle_ = handle; }
  void clear_handle() { handle_ = nullptr; }

 private:
  MUevent handle_ = nullptr;
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_
