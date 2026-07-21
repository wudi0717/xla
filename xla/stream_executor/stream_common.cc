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

#include "xla/stream_executor/stream_common.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "tsl/platform/statusor.h"
namespace stream_executor {

StreamCommon::StreamCommon(StreamExecutor* parent) : Stream(parent) {}

StreamCommon::StreamCommon(
    StreamExecutor* parent,
    std::optional<std::variant<StreamPriority, int>> priority)
    : StreamCommon(parent) {
  if (priority.has_value()) {
    stream_priority_ = *priority;
  }
}

bool StreamCommon::ok() const { return !InErrorState(); }

tsl::StatusOr<Stream*> StreamCommon::GetOrCreateSubStream() {
  std::vector<std::unique_ptr<Stream>> bad_streams;

  absl::MutexLock lock(&mu_);
  for (size_t index = 0; index < sub_streams_.size();) {
    std::pair<std::unique_ptr<Stream>, bool>& pair = sub_streams_[index];
    if (pair.second) {
      Stream* sub_stream = pair.first.get();
      if (sub_stream->ok()) {
        VLOG(1) << DebugStreamPointers() << " reusing sub_stream "
                << sub_stream->DebugStreamPointers();
        pair.second = false;
        return sub_stream;
      }

      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      bad_streams.push_back(std::move(sub_streams_.back().first));
      sub_streams_.pop_back();
      VLOG(1) << DebugStreamPointers() << " dropped !ok sub_stream "
              << sub_stream->DebugStreamPointers();
    } else {
      ++index;
    }
  }

  TF_ASSIGN_OR_RETURN(auto stream, parent_->CreateStream());
  sub_streams_.emplace_back(std::move(stream), false);
  Stream* sub_stream = sub_streams_.back().first.get();
  sub_stream->SetName("Sub-stream of " + GetName());
  VLOG(1) << DebugStreamPointers() << " created new sub_stream "
          << sub_stream->DebugStreamPointers();
  return sub_stream;
}

void StreamCommon::ReturnSubStream(Stream* sub_stream) {
  std::unique_ptr<Stream> bad_stream;

  absl::MutexLock lock(&mu_);
  for (int64_t index = 0, end = sub_streams_.size(); index < end; ++index) {
    std::pair<std::unique_ptr<Stream>, bool>& pair = sub_streams_[index];
    if (pair.first.get() != sub_stream) {
      continue;
    }

    if (sub_stream->ok()) {
      VLOG(1) << DebugStreamPointers() << " returned ok sub_stream "
              << sub_stream->DebugStreamPointers();
      pair.second = true;
    } else {
      VLOG(1) << DebugStreamPointers() << " returned !ok sub_stream "
              << sub_stream->DebugStreamPointers();
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      std::swap(bad_stream, sub_streams_.back().first);
      sub_streams_.pop_back();
    }
    return;
  }

  LOG(FATAL) << DebugStreamPointers()
             << " did not create the returned sub-stream "
             << sub_stream->DebugStreamPointers();
}

StreamExecutor* StreamCommon::parent() const {
  CHECK(parent_ != nullptr);
  return parent_;
}

void StreamCommon::SetPriority(StreamPriority priority) {
  Stream::SetPriority(priority);
  stream_priority_ = priority;
}

void StreamCommon::SetPriority(int priority) {
  Stream::SetPriority(priority);
  stream_priority_ = priority;
}

const std::string& StreamCommon::GetName() const { return name_; }

void StreamCommon::SetName(std::string name) { name_ = std::move(name); }

std::variant<StreamPriority, int> StreamCommon::priority() const {
  return stream_priority_;
}

}  // namespace stream_executor
