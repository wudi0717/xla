/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/sequential_thunk.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace gpu {

using ::tsl::profiler::ScopedAnnotation;

namespace {

bool IsMusaThunkTimingEnabled() {
  const char* value = std::getenv("MUSA_XLA_THUNK_TIMING");
  return value != nullptr && value[0] != '\0' && value[0] != '0' &&
         std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0 &&
         std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

std::string CompactAnnotation(std::string annotation) {
  constexpr size_t kMaxAnnotationLength = 120;
  for (char& ch : annotation) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      ch = '_';
    }
  }
  if (annotation.size() > kMaxAnnotationLength) {
    annotation.resize(kMaxAnnotationLength);
    annotation.append("...");
  }
  return annotation;
}

struct ThunkTimingItem {
  int64_t index = 0;
  std::string kind;
  std::string annotation;
  int64_t elapsed_us = 0;
};

struct KindTimingAggregate {
  std::string kind;
  int64_t count = 0;
  int64_t elapsed_us = 0;
};

std::string FormatKindTotals(
    const absl::flat_hash_map<std::string, KindTimingAggregate>& totals) {
  std::vector<KindTimingAggregate> rows;
  rows.reserve(totals.size());
  for (const auto& entry : totals) {
    rows.push_back(entry.second);
  }
  std::sort(rows.begin(), rows.end(), [](const KindTimingAggregate& a,
                                         const KindTimingAggregate& b) {
    if (a.elapsed_us != b.elapsed_us) return a.elapsed_us > b.elapsed_us;
    return a.kind < b.kind;
  });

  std::string result;
  for (int64_t i = 0; i < rows.size() && i < 16; ++i) {
    if (!result.empty()) absl::StrAppend(&result, " | ");
    absl::StrAppend(&result, rows[i].kind, ":count=", rows[i].count,
                    ",us=", rows[i].elapsed_us);
  }
  return result;
}

std::string FormatTopThunks(std::vector<ThunkTimingItem> timings) {
  std::sort(timings.begin(), timings.end(), [](const ThunkTimingItem& a,
                                               const ThunkTimingItem& b) {
    if (a.elapsed_us != b.elapsed_us) return a.elapsed_us > b.elapsed_us;
    return a.index < b.index;
  });

  std::string result;
  for (int64_t i = 0; i < timings.size() && i < 20; ++i) {
    const ThunkTimingItem& item = timings[i];
    if (!result.empty()) absl::StrAppend(&result, " | ");
    absl::StrAppend(&result, "#", item.index, ":", item.kind,
                    ":us=", item.elapsed_us);
    if (!item.annotation.empty()) {
      absl::StrAppend(&result, ":ann=", item.annotation);
    }
  }
  return result;
}

}  // namespace

SequentialThunk::SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks)
    : Thunk(Kind::kSequential, thunk_info), thunks_(std::move(thunks)) {}

std::string SequentialThunk::ToStringExtra(int indent) const {
  std::string result = "\n";
  absl::StrAppend(&result, thunks().ToString(indent + 1, nullptr));
  return result;
}

Status SequentialThunk::Initialize(const GpuExecutable& executable,
                                   se::StreamExecutor* executor) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(executable, executor));
  }
  return OkStatus();
}

Status SequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  if (IsMusaThunkTimingEnabled()) {
    std::vector<ThunkTimingItem> timings;
    timings.reserve(thunks_.size());
    absl::flat_hash_map<std::string, KindTimingAggregate> kind_totals;
    int64_t total_us = 0;

    for (size_t index = 0; index < thunks_.size(); ++index) {
      const auto& thunk = thunks_[index];
      ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });
      TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
      const int64_t start_us = tsl::Env::Default()->NowMicros();
      Status status = thunk->ExecuteOnStream(params);
      if (!status.ok()) {
        LOG(INFO) << "[MUSA_THUNK_TIMING] failed index=" << index
                  << " kind=" << Thunk::KindToString(thunk->kind())
                  << " status=" << status;
        return status;
      }
      TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
      const int64_t elapsed_us =
          tsl::Env::Default()->NowMicros() - start_us;
      const std::string kind(Thunk::KindToString(thunk->kind()));
      const std::string profile_annotation =
          CompactAnnotation(thunk->profile_annotation());
      timings.push_back(
          {static_cast<int64_t>(index), kind, profile_annotation, elapsed_us});
      KindTimingAggregate& aggregate = kind_totals[kind];
      aggregate.kind = kind;
      aggregate.count += 1;
      aggregate.elapsed_us += elapsed_us;
      total_us += elapsed_us;
    }

    LOG(INFO) << "[MUSA_THUNK_TIMING] total_thunks=" << thunks_.size()
              << " total_us=" << total_us
              << " kind_totals={" << FormatKindTotals(kind_totals) << "}"
              << " top_thunks={" << FormatTopThunks(std::move(timings)) << "}";
    return OkStatus();
  }

  for (const auto& thunk : thunks_) {
    ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
