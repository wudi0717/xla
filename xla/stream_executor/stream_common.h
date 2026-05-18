/* Copyright 2026 The OpenXLA Authors.
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

#ifndef XLA_STREAM_EXECUTOR_STREAM_COMMON_H_
#define XLA_STREAM_EXECUTOR_STREAM_COMMON_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "xla/stream_executor/stream.h"

namespace stream_executor {

// Transitional common stream base for newer StreamExecutor platform streams.
//
// This class reuses the legacy Stream lifecycle, while pulling common stream
// state management behind a dedicated derived type for newer platform streams.
class StreamCommon : public Stream {
 public:
  explicit StreamCommon(StreamExecutor* parent);
  StreamCommon(StreamExecutor* parent,
               std::optional<std::variant<StreamPriority, int>> priority);
  ~StreamCommon() override = default;

  bool ok() const override;
  tsl::StatusOr<Stream*> GetOrCreateSubStream() override;
  void ReturnSubStream(Stream* sub_stream) override;
  StreamExecutor* parent() const override;

  void SetPriority(StreamPriority priority) override;
  void SetPriority(int priority) override;
  const std::string& GetName() const override;
  void SetName(std::string name) override;
  std::variant<StreamPriority, int> priority() const override;

 private:
  std::variant<StreamPriority, int> stream_priority_ =
      StreamPriority::Default;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_COMMON_H_
