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

#include "xla/service/gpu/gpu_executable.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/map_util.h"
#include "xla/mlir/runtime/ir/rt_ops.h"
#include "xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "xla/mlir/runtime/transforms/type_converter.h"
#include "xla/runtime/executable.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gemm_thunk.h"
#include "xla/service/gpu/non_atomically_upgradeable_rw_lock.h"
#include "xla/service/gpu/musa_grouped_gemm_thunk.h"
#include "xla/service/gpu/musa_small_gemm_accum_thunk.h"
#include "xla/service/gpu/runtime/executable.h"
#include "xla/service/gpu/runtime2/executable.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

#if TENSORFLOW_USE_ROCM
#include "tsl/platform/random.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_activation.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

// If experimental XLA:GPU runtime is enabled, it automatically disables
// "classic" XLA:GPU runtime which is enabled by default.
bool IsXlaRuntimeExecutableEnabled(const HloModuleConfig& config) {
  bool runtime = config.debug_options().xla_gpu_enable_xla_runtime_executable();
  bool gpu2 = config.debug_options().xla_gpu_enable_gpu2_runtime();
  return runtime && !gpu2;
}

bool IsXlaGpu2RuntimeEnabled(const HloModuleConfig& config) {
  return config.debug_options().xla_gpu_enable_gpu2_runtime();
}

struct MusaClassicThunkGraphCache {
  absl::Mutex mutex;
  std::map<std::vector<uintptr_t>, std::unique_ptr<se::CommandBuffer>> entries
      ABSL_GUARDED_BY(mutex);
  std::set<std::vector<uintptr_t>> failed_keys ABSL_GUARDED_BY(mutex);
  std::set<se::StreamExecutor*> warmed_executors ABSL_GUARDED_BY(mutex);
  std::map<se::StreamExecutor*, std::vector<uintptr_t>> last_keys
      ABSL_GUARDED_BY(mutex);
  std::set<std::vector<uintptr_t>> logged_hit_keys ABSL_GUARDED_BY(mutex);
};

namespace {

using ::tsl::profiler::ScopedAnnotation;
using ::tsl::profiler::ScopedAnnotationAlways;

bool IsMusaDebugDeallocEnabled() {
  const char* value = std::getenv("TF_MUSA_DEBUG_DEALLOC");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool IsTruthyEnv(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool IsMusaThunkDiagnosticsEnabled() {
  return IsTruthyEnv("MUSA_XLA_THUNK_DIAGNOSTICS");
}

bool IsMusaThunkTimingEnabled() {
  const char* value = std::getenv("MUSA_XLA_THUNK_TIMING");
  return value != nullptr && value[0] != '\0' && value[0] != '0' &&
         std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0 &&
         std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

bool IsMusaClassicThunkGraphEnabled() {
  return IsTruthyEnv("MUSA_XLA_CLASSIC_THUNK_GRAPH");
}

bool IsMusaExecutionPathVerboseEnabled() {
  return IsTruthyEnv("MUSA_XLA_EXECUTION_PATH_VERBOSE");
}

int64_t ReadInt64Env(const char* name, int64_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  int64_t parsed = std::strtoll(value, &end, 10);
  if (end == value) {
    return default_value;
  }
  return parsed;
}

std::string DescribeAllocationForDebug(const BufferAllocation& allocation,
                                       se::DeviceMemoryBase buffer_address,
                                       bool is_live_out) {
  std::vector<std::string> tags;
  if (allocation.is_entry_computation_parameter()) {
    tags.push_back(absl::StrCat("param#", allocation.parameter_number()));
  }
  if (allocation.is_constant()) {
    tags.push_back("constant");
  }
  if (allocation.IsPreallocatedTempBuffer()) {
    tags.push_back("temp");
  }
  if (allocation.maybe_live_out()) {
    tags.push_back(is_live_out ? "live_out" : "not_live_out");
  }
  if (allocation.is_thread_local()) {
    tags.push_back("thread_local");
  }
  std::vector<std::string> assigned_hlos;
  assigned_hlos.reserve(std::min<size_t>(allocation.assigned_buffers().size(), 4));
  size_t emitted = 0;
  for (const auto& it : allocation.assigned_buffers()) {
    if (emitted == 4) {
      assigned_hlos.push_back("...");
      break;
    }
    const HloValue* value = it.first;
    const BufferAllocation::OffsetSize& offset_size = it.second;
    const HloInstruction* instruction = value->instruction();
    assigned_hlos.push_back(absl::StrCat(
        instruction->name(), value->index().empty()
                                 ? ""
                                 : absl::StrCat(value->index().ToString()),
        "@offset=", offset_size.offset, "/size=", offset_size.size));
    ++emitted;
  }
  return absl::StrCat("alloc#", allocation.index(), " ptr=",
                      reinterpret_cast<uintptr_t>(buffer_address.opaque()),
                      " size=", buffer_address.size(), "B flags=[",
                      absl::StrJoin(tags, ","), "] hlo=[",
                      absl::StrJoin(assigned_hlos, "; "), "]");
}

bool NeedsAsyncCommsStream(Thunk& thunk) {
  switch (thunk.kind()) {
    case Thunk::Kind::kNcclAllReduceStart:
    case Thunk::Kind::kNcclAllReduceDone:
      return true;
    default:
      return false;
  }
}

std::string ShortMusaThunkAnnotation(const Thunk& thunk) {
  std::string annotation = thunk.profile_annotation();
  if (annotation.empty()) {
    return std::string(Thunk::KindToString(thunk.kind()));
  }
  constexpr size_t kMaxAnnotationLength = 96;
  if (annotation.size() <= kMaxAnnotationLength) {
    return annotation;
  }
  annotation.resize(kMaxAnnotationLength);
  annotation.append("...");
  return annotation;
}

std::string CompactMusaThunkTimingAnnotation(const Thunk& thunk) {
  std::string annotation = ShortMusaThunkAnnotation(thunk);
  for (char& ch : annotation) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      ch = '_';
    }
  }
  return annotation;
}

struct MusaThunkTimingItem {
  int64_t index = 0;
  std::string kind;
  std::string annotation;
  int64_t elapsed_us = 0;
};

struct MusaThunkKindTiming {
  std::string kind;
  int64_t count = 0;
  int64_t elapsed_us = 0;
};

std::string FormatMusaThunkTimingKindTotals(
    const absl::flat_hash_map<std::string, MusaThunkKindTiming>& totals) {
  std::vector<MusaThunkKindTiming> rows;
  rows.reserve(totals.size());
  for (const auto& entry : totals) {
    rows.push_back(entry.second);
  }
  std::sort(rows.begin(), rows.end(), [](const MusaThunkKindTiming& a,
                                         const MusaThunkKindTiming& b) {
    if (a.elapsed_us != b.elapsed_us) return a.elapsed_us > b.elapsed_us;
    return a.kind < b.kind;
  });

  std::vector<std::string> parts;
  for (int64_t i = 0; i < rows.size() && i < 16; ++i) {
    parts.push_back(absl::StrCat(rows[i].kind, ":count=", rows[i].count,
                                 ",us=", rows[i].elapsed_us));
  }
  return absl::StrJoin(parts, " | ");
}

std::string FormatMusaThunkTimingTopThunks(
    std::vector<MusaThunkTimingItem> timings) {
  std::sort(timings.begin(), timings.end(), [](const MusaThunkTimingItem& a,
                                               const MusaThunkTimingItem& b) {
    if (a.elapsed_us != b.elapsed_us) return a.elapsed_us > b.elapsed_us;
    return a.index < b.index;
  });

  std::vector<std::string> parts;
  for (int64_t i = 0; i < timings.size() && i < 20; ++i) {
    const MusaThunkTimingItem& item = timings[i];
    std::string part = absl::StrCat("#", item.index, ":", item.kind,
                                    ":us=", item.elapsed_us);
    if (!item.annotation.empty()) {
      absl::StrAppend(&part, ":ann=", item.annotation);
    }
    parts.push_back(std::move(part));
  }
  return absl::StrJoin(parts, " | ");
}

Status ExecuteThunkWithOptionalTiming(
    int64_t index, Thunk& thunk, const Thunk::ExecuteParams& thunk_params,
    se::Stream* main_stream, std::vector<MusaThunkTimingItem>* timings,
    absl::flat_hash_map<std::string, MusaThunkKindTiming>* kind_totals,
    int64_t* total_us) {
  TF_RETURN_IF_ERROR(main_stream->BlockHostUntilDone());
  const int64_t start_us = tsl::Env::Default()->NowMicros();
  Status status = thunk.ExecuteOnStream(thunk_params);
  if (!status.ok()) {
    LOG(INFO) << "[MUSA_THUNK_TIMING] failed index=" << index
              << " kind=" << Thunk::KindToString(thunk.kind())
              << " status=" << status;
    return status;
  }
  TF_RETURN_IF_ERROR(main_stream->BlockHostUntilDone());
  const int64_t elapsed_us = tsl::Env::Default()->NowMicros() - start_us;
  const std::string kind(Thunk::KindToString(thunk.kind()));
  timings->push_back(
      {index, kind, CompactMusaThunkTimingAnnotation(thunk), elapsed_us});
  MusaThunkKindTiming& aggregate = (*kind_totals)[kind];
  aggregate.kind = kind;
  aggregate.count += 1;
  aggregate.elapsed_us += elapsed_us;
  *total_us += elapsed_us;
  return OkStatus();
}

struct MusaClassicThunkGraphExecution {
  bool executed = false;
};

struct MusaClassicThunkGraphAddressChanges {
  bool signature_changed = false;
  int64_t changed_allocations = 0;
  int64_t changed_params = 0;
  int64_t changed_temp = 0;
  int64_t changed_live_out = 0;
  int64_t changed_constant = 0;
  int64_t changed_other = 0;
  int64_t changed_bytes = 0;
};

void LogMusaClassicThunkGraph(const std::string& module_name,
                              ModuleIdentifier module_id, bool eligible,
                              bool cache_hit, bool captured,
                              absl::string_view fallback_reason,
                              int64_t total_thunks, int64_t kernel_thunks,
                              int64_t gemm_thunks, int64_t cache_entries,
                              MusaClassicThunkGraphAddressChanges changes = {}) {
}

std::vector<uintptr_t> MusaClassicThunkGraphKey(
    se::StreamExecutor* executor,
    const BufferAllocations& buffer_allocations) {
  std::vector<uintptr_t> key;
  key.reserve(2 + buffer_allocations.size() * 2);
  key.push_back(reinterpret_cast<uintptr_t>(executor));
  key.push_back(static_cast<uintptr_t>(buffer_allocations.device_ordinal()));
  for (BufferAllocation::Index i = 0; i < buffer_allocations.size(); ++i) {
    se::DeviceMemoryBase buffer = buffer_allocations.GetDeviceAddress(i);
    key.push_back(reinterpret_cast<uintptr_t>(buffer.opaque()));
    key.push_back(static_cast<uintptr_t>(buffer.size()));
  }
  return key;
}

MusaClassicThunkGraphAddressChanges MusaClassicThunkGraphKeyChanges(
    const std::vector<uintptr_t>& previous_key,
    const std::vector<uintptr_t>& current_key,
    const std::vector<BufferAllocation>& allocations) {
  MusaClassicThunkGraphAddressChanges changes;
  if (previous_key.empty() || previous_key.size() != current_key.size()) {
    return changes;
  }

  changes.signature_changed = previous_key != current_key;
  const size_t allocation_count = std::min(
      allocations.size(), current_key.size() > 2
                              ? (current_key.size() - 2) / 2
                              : static_cast<size_t>(0));
  for (size_t i = 0; i < allocation_count; ++i) {
    const size_t key_index = 2 + i * 2;
    if (previous_key[key_index] == current_key[key_index] &&
        previous_key[key_index + 1] == current_key[key_index + 1]) {
      continue;
    }

    ++changes.changed_allocations;
    changes.changed_bytes += static_cast<int64_t>(current_key[key_index + 1]);
    const BufferAllocation& allocation = allocations[i];
    if (allocation.is_entry_computation_parameter()) {
      ++changes.changed_params;
    } else if (allocation.IsPreallocatedTempBuffer()) {
      ++changes.changed_temp;
    } else if (allocation.is_constant()) {
      ++changes.changed_constant;
    } else if (allocation.maybe_live_out()) {
      ++changes.changed_live_out;
    } else {
      ++changes.changed_other;
    }
  }
  return changes;
}

StatusOr<MusaClassicThunkGraphExecution> TryExecuteMusaClassicThunkGraph(
    const std::string& module_name, ModuleIdentifier module_id,
    const ThunkSequence& thunk_sequence,
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations,
    const std::vector<BufferAllocation>& allocations,
    MusaClassicThunkGraphCache* cache) {
  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();
  int64_t kernel_thunks = 0;
  int64_t gemm_thunks = 0;
  for (const std::unique_ptr<Thunk>& thunk : thunk_sequence) {
    if (thunk->kind() == Thunk::kKernel) {
      ++kernel_thunks;
    } else if (thunk->kind() == Thunk::kGemm) {
      ++gemm_thunks;
    } else {
      LogMusaClassicThunkGraph(
          module_name, module_id, /*eligible=*/false, /*cache_hit=*/false,
          /*captured=*/false,
          absl::StrCat("unsupported_thunk:",
                       Thunk::KindToString(thunk->kind())),
          thunk_sequence.size(), kernel_thunks, gemm_thunks, 0);
      return MusaClassicThunkGraphExecution{};
    }
  }
  if (thunk_sequence.empty()) {
    LogMusaClassicThunkGraph(module_name, module_id, /*eligible=*/false,
                             /*cache_hit=*/false, /*captured=*/false,
                             "empty_sequence", 0, 0, 0, 0);
    return MusaClassicThunkGraphExecution{};
  }

  const int64_t max_cache_entries = std::max<int64_t>(
      1, ReadInt64Env("MUSA_XLA_CLASSIC_THUNK_GRAPH_MAX_CACHE_ENTRIES", 4));
  std::vector<uintptr_t> key =
      MusaClassicThunkGraphKey(executor, buffer_allocations);

  absl::MutexLock lock(&cache->mutex);
  MusaClassicThunkGraphAddressChanges changes;
  auto previous_key = cache->last_keys.find(executor);
  if (previous_key != cache->last_keys.end()) {
    changes = MusaClassicThunkGraphKeyChanges(previous_key->second, key,
                                              allocations);
  }
  cache->last_keys[executor] = key;

  auto existing = cache->entries.find(key);
  if (existing != cache->entries.end()) {
    TF_RETURN_IF_ERROR(executor->Submit(main_stream, *existing->second));
    if (cache->logged_hit_keys.insert(key).second) {
      LogMusaClassicThunkGraph(
          module_name, module_id, /*eligible=*/true, /*cache_hit=*/true,
          /*captured=*/false, "", thunk_sequence.size(), kernel_thunks,
          gemm_thunks, cache->entries.size(), changes);
    }
    return MusaClassicThunkGraphExecution{/*executed=*/true};
  }
  if (cache->failed_keys.find(key) != cache->failed_keys.end()) {
    LogMusaClassicThunkGraph(
        module_name, module_id, /*eligible=*/true, /*cache_hit=*/false,
        /*captured=*/false, "capture_previously_failed",
        thunk_sequence.size(), kernel_thunks, gemm_thunks,
        cache->entries.size(), changes);
    return MusaClassicThunkGraphExecution{};
  }
  if (static_cast<int64_t>(cache->entries.size()) >= max_cache_entries) {
    LogMusaClassicThunkGraph(module_name, module_id, /*eligible=*/true,
                             /*cache_hit=*/false, /*captured=*/false,
                             "cache_full", thunk_sequence.size(),
                             kernel_thunks, gemm_thunks,
                             cache->entries.size(), changes);
    return MusaClassicThunkGraphExecution{};
  }
  if (cache->warmed_executors.insert(executor).second) {
    LogMusaClassicThunkGraph(module_name, module_id, /*eligible=*/true,
                             /*cache_hit=*/false, /*captured=*/false,
                             "warmup", thunk_sequence.size(), kernel_thunks,
                             gemm_thunks, cache->entries.size(), changes);
    return MusaClassicThunkGraphExecution{};
  }

  // Initialize BLAS support before capture so lazy setup is not recorded in
  // the graph trace.
  if (gemm_thunks > 0 && executor->AsBlas() == nullptr) {
    cache->failed_keys.insert(key);
    LogMusaClassicThunkGraph(module_name, module_id, /*eligible=*/true,
                             /*cache_hit=*/false, /*captured=*/false,
                             "blas_unavailable", thunk_sequence.size(),
                             kernel_thunks, gemm_thunks,
                             cache->entries.size(), changes);
    return MusaClassicThunkGraphExecution{};
  }

  StatusOr<se::CommandBuffer> traced = se::CommandBuffer::Trace(
      executor, [&](se::Stream* trace_stream) -> Status {
        absl::InlinedVector<se::Stream*, kAsyncStreamTotal>
            no_async_comms_streams(kAsyncStreamTotal, nullptr);
        for (const std::unique_ptr<Thunk>& thunk : thunk_sequence) {
          Thunk::ExecuteParams thunk_params{*run_options, buffer_allocations,
                                            trace_stream,
                                            no_async_comms_streams};
          TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(thunk_params));
        }
        return OkStatus();
      });
  if (!traced.ok()) {
    cache->failed_keys.insert(key);
    LogMusaClassicThunkGraph(
        module_name, module_id, /*eligible=*/true, /*cache_hit=*/false,
        /*captured=*/false,
        absl::StrCat("capture_failed:", traced.status().message()),
        thunk_sequence.size(), kernel_thunks, gemm_thunks,
        cache->entries.size(), changes);
    return MusaClassicThunkGraphExecution{};
  }

  auto command_buffer =
      std::make_unique<se::CommandBuffer>(std::move(*traced));
  TF_RETURN_IF_ERROR(executor->Submit(main_stream, *command_buffer));
  cache->entries.emplace(std::move(key), std::move(command_buffer));
  LogMusaClassicThunkGraph(module_name, module_id, /*eligible=*/true,
                           /*cache_hit=*/false, /*captured=*/true, "",
                           thunk_sequence.size(), kernel_thunks, gemm_thunks,
                           cache->entries.size(), changes);
  return MusaClassicThunkGraphExecution{/*executed=*/true};
}

void MaybeLogMusaThunkSummary(const std::string& module_name,
                              ModuleIdentifier module_id,
                              const ThunkSequence& thunk_sequence,
                              se::Stream* stream) {
  if (!IsMusaThunkDiagnosticsEnabled() ||
      stream->parent()->platform()->id() !=
          stream_executor::musa::kMusaPlatformId) {
    return;
  }

  std::string module_key =
      absl::StrCat(module_name, "#", module_id, "#", thunk_sequence.size());
  static absl::Mutex logged_mu(absl::kConstInit);
  static auto* logged_modules = new std::set<std::string>();
  {
    absl::MutexLock lock(&logged_mu);
    if (!logged_modules->insert(module_key).second) {
      return;
    }
  }

  absl::flat_hash_map<std::string, int64_t> counts;
  int64_t gemm_thunks = 0;
  int64_t kernel_thunks = 0;
  int64_t current_gemm_run = 0;
  int64_t max_gemm_run = 0;
  int64_t gemm_run_count = 0;
  std::vector<int64_t> gemm_run_lengths;
  absl::flat_hash_map<std::string, int64_t> transition_counts;
  absl::flat_hash_map<std::string, int64_t> gemm_prev_counts;
  absl::flat_hash_map<std::string, int64_t> gemm_next_counts;
  std::vector<std::string> gemm_context_samples;
  auto flush_gemm_run = [&] {
    if (current_gemm_run <= 0) {
      return;
    }
    ++gemm_run_count;
    max_gemm_run = std::max(max_gemm_run, current_gemm_run);
    gemm_run_lengths.push_back(current_gemm_run);
    current_gemm_run = 0;
  };
  auto append_top_counts = [](const absl::flat_hash_map<std::string, int64_t>& map,
                              int64_t limit) {
    std::vector<std::pair<std::string, int64_t>> sorted(map.begin(), map.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) {
        return a.second > b.second;
      }
      return a.first < b.first;
    });
    std::vector<std::string> result;
    for (const auto& [key, count] : sorted) {
      if (result.size() >= limit) {
        break;
      }
      result.push_back(absl::StrCat(key, "=", count));
    }
    return result;
  };
  for (int64_t i = 0; i < thunk_sequence.size(); ++i) {
    const std::unique_ptr<Thunk>& thunk = thunk_sequence[i];
    std::string kind = std::string(Thunk::KindToString(thunk->kind()));
    ++counts[kind];
    if (i > 0) {
      std::string prev_kind =
          std::string(Thunk::KindToString(thunk_sequence[i - 1]->kind()));
      ++transition_counts[absl::StrCat(prev_kind, "->", kind)];
    }
    if (thunk->kind() == Thunk::Kind::kGemm ||
        thunk->kind() == Thunk::Kind::kCublasLtMatmul) {
      ++gemm_thunks;
      ++current_gemm_run;
      std::string prev_kind =
          i == 0 ? "START"
                 : std::string(
                       Thunk::KindToString(thunk_sequence[i - 1]->kind()));
      std::string next_kind =
          i + 1 == thunk_sequence.size()
              ? "END"
              : std::string(Thunk::KindToString(thunk_sequence[i + 1]->kind()));
      ++gemm_prev_counts[prev_kind];
      ++gemm_next_counts[next_kind];
      if (gemm_context_samples.size() < 12) {
        std::string prev_annotation =
            i == 0 ? "START"
                   : ShortMusaThunkAnnotation(*thunk_sequence[i - 1]);
        std::string next_annotation =
            i + 1 == thunk_sequence.size()
                ? "END"
                : ShortMusaThunkAnnotation(*thunk_sequence[i + 1]);
        gemm_context_samples.push_back(absl::StrCat(
            "prev=", prev_kind, "(", prev_annotation, ") gemm=(",
            ShortMusaThunkAnnotation(*thunk), ") next=", next_kind, "(",
            next_annotation, ")"));
      }
    } else if (thunk->kind() == Thunk::Kind::kKernel) {
      ++kernel_thunks;
      flush_gemm_run();
    } else {
      flush_gemm_run();
    }
  }
  flush_gemm_run();

  std::vector<std::pair<std::string, int64_t>> sorted_counts(counts.begin(),
                                                             counts.end());
  std::sort(sorted_counts.begin(), sorted_counts.end(),
            [](const auto& a, const auto& b) {
              if (a.second != b.second) {
                return a.second > b.second;
              }
              return a.first < b.first;
            });
  std::vector<std::string> parts;
  parts.reserve(sorted_counts.size());
  for (const auto& [kind, count] : sorted_counts) {
    parts.push_back(absl::StrCat(kind, "=", count));
  }

  std::sort(gemm_run_lengths.begin(), gemm_run_lengths.end(),
            [](int64_t a, int64_t b) { return a > b; });
  const int64_t min_group_size = std::max<int64_t>(
      1, ReadInt64Env("MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE", 4));
  const int64_t max_group_size = std::max<int64_t>(
      min_group_size,
      ReadInt64Env("MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE", 64));
  int64_t groupable_gemm_thunks = 0;
  int64_t estimated_grouped_gemm_thunks = 0;
  int64_t estimated_launch_reduction = 0;
  for (int64_t run_length : gemm_run_lengths) {
    if (run_length < min_group_size) {
      continue;
    }
    int64_t grouped_launches =
        (run_length + max_group_size - 1) / max_group_size;
    groupable_gemm_thunks += run_length;
    estimated_grouped_gemm_thunks += grouped_launches;
    estimated_launch_reduction += run_length - grouped_launches;
  }
  int64_t nearby_gemm_windows = 0;
  int64_t nearby_gemm_candidates = 0;
  int64_t nearby_gemm_estimated_launch_reduction = 0;
  absl::flat_hash_map<std::string, int64_t> nearby_separator_ops;
  absl::flat_hash_map<std::string, int64_t> nearby_blocked_reasons;
  auto separator_op_name = [](const Thunk& thunk) {
    std::string annotation = thunk.profile_annotation();
    constexpr char kHloOpMarker[] = "#hlo_op=";
    size_t marker = annotation.find(kHloOpMarker);
    if (marker != std::string::npos) {
      size_t begin = marker + sizeof(kHloOpMarker) - 1;
      size_t end = annotation.find('#', begin);
      std::string name = annotation.substr(
          begin, end == std::string::npos ? std::string::npos : end - begin);
      size_t dot = name.find('.');
      if (dot != std::string::npos) {
        name.resize(dot);
      }
      if (!name.empty()) {
        return name;
      }
    }
    if (annotation.empty()) {
      return std::string(Thunk::KindToString(thunk.kind()));
    }
    return annotation;
  };
  auto gemm_shape_key = [](const GemmThunk& gemm) {
    const GemmConfig& config = gemm.config();
    return absl::StrCat(config.output_layout.num_rows, "x",
                        config.output_layout.num_cols, "x",
                        config.lhs_layout.num_cols, ":alpha=(",
                        config.alpha.real(), ",", config.alpha.imag(),
                        "):beta=", config.beta);
  };
  const int64_t nearby_max_separators = std::max<int64_t>(
      1, ReadInt64Env("MUSA_XLA_THUNK_DIAGNOSTICS_NEARBY_GEMM_MAX_SEPARATORS",
                      8));
  for (int64_t start = 0; start < thunk_sequence.size(); ++start) {
    const GemmThunk* seed = thunk_sequence[start]->AsGemmThunk();
    if (seed == nullptr) {
      continue;
    }
    int64_t separator_count = 0;
    int64_t window_gemms = 0;
    absl::flat_hash_map<std::string, int64_t> window_shapes;
    for (int64_t i = start; i < thunk_sequence.size(); ++i) {
      const GemmThunk* gemm = thunk_sequence[i]->AsGemmThunk();
      if (gemm != nullptr) {
        ++window_gemms;
        ++window_shapes[gemm_shape_key(*gemm)];
        if (window_gemms >= max_group_size) {
          break;
        }
        continue;
      }
      if (thunk_sequence[i]->kind() != Thunk::Kind::kKernel) {
        break;
      }
      ++separator_count;
      ++nearby_separator_ops[separator_op_name(*thunk_sequence[i])];
      if (separator_count > nearby_max_separators) {
        ++nearby_blocked_reasons["too_many_separators"];
        break;
      }
    }
    if (window_gemms < min_group_size) {
      continue;
    }
    int64_t best_same_shape = 0;
    for (const auto& [shape, count] : window_shapes) {
      best_same_shape = std::max(best_same_shape, count);
    }
    if (best_same_shape < min_group_size) {
      ++nearby_blocked_reasons["mixed_shapes"];
      continue;
    }
    ++nearby_gemm_windows;
    nearby_gemm_candidates += best_same_shape;
    nearby_gemm_estimated_launch_reduction += best_same_shape - 1;
  }
  std::vector<std::string> top_gemm_runs;
  for (int64_t length : gemm_run_lengths) {
    if (top_gemm_runs.size() >= 8) {
      break;
    }
    top_gemm_runs.push_back(absl::StrCat(length));
  }
  std::vector<std::string> top_transitions =
      append_top_counts(transition_counts, 12);
  std::vector<std::string> top_gemm_prev =
      append_top_counts(gemm_prev_counts, 8);
  std::vector<std::string> top_gemm_next =
      append_top_counts(gemm_next_counts, 8);
  std::vector<std::string> top_nearby_separator_ops =
      append_top_counts(nearby_separator_ops, 12);
  std::vector<std::string> top_nearby_blocked_reasons =
      append_top_counts(nearby_blocked_reasons, 8);

  LOG(INFO) << "[MUSA_XLA_THUNK_DIAGNOSTICS] module=" << module_name
            << " module_id=" << module_id
            << " total_thunks=" << thunk_sequence.size()
            << " gemm_thunks=" << gemm_thunks
            << " kernel_thunks=" << kernel_thunks
            << " counts={" << absl::StrJoin(parts, ",") << "}"
            << " gemm_run_count=" << gemm_run_count
            << " max_gemm_run=" << max_gemm_run
            << " top_gemm_runs={" << absl::StrJoin(top_gemm_runs, ",") << "}"
            << " group_gemm_requested="
            << IsTruthyEnv("MUSA_XLA_GROUP_GEMM_THUNKS")
            << " group_min_size=" << min_group_size
            << " group_max_size=" << max_group_size
            << " groupable_gemm_thunks=" << groupable_gemm_thunks
            << " estimated_grouped_gemm_thunks="
            << estimated_grouped_gemm_thunks
            << " estimated_launch_reduction=" << estimated_launch_reduction
            << " nearby_gemm_windows=" << nearby_gemm_windows
            << " nearby_gemm_candidates=" << nearby_gemm_candidates
            << " nearby_gemm_estimated_launch_reduction="
            << nearby_gemm_estimated_launch_reduction
            << " nearby_gemm_separator_ops={"
            << absl::StrJoin(top_nearby_separator_ops, ",") << "}"
            << " nearby_gemm_blocked_reasons={"
            << absl::StrJoin(top_nearby_blocked_reasons, ",") << "}"
            << " top_transitions={"
            << absl::StrJoin(top_transitions, ",") << "}"
            << " gemm_prev_kinds={" << absl::StrJoin(top_gemm_prev, ",")
            << "}"
            << " gemm_next_kinds={" << absl::StrJoin(top_gemm_next, ",")
            << "}"
            << " gemm_context_samples={"
            << absl::StrJoin(gemm_context_samples, " | ") << "}";
}

void MaybeLogMusaExecutionPath(const std::string& module_name,
                               ModuleIdentifier module_id,
                               se::Stream* stream, bool has_thunks,
                               bool has_xla_runtime, bool has_gpu2_runtime) {
  if (!IsMusaThunkDiagnosticsEnabled() ||
      stream->parent()->platform()->id() !=
          stream_executor::musa::kMusaPlatformId) {
    return;
  }
  if (!IsMusaExecutionPathVerboseEnabled() &&
      module_name.rfind("cluster_", 0) != 0) {
    return;
  }

  const char* path = has_thunks       ? "classic_thunks"
                     : has_xla_runtime ? "xla_runtime"
                     : has_gpu2_runtime ? "gpu2_runtime"
                                        : "none";
  std::string module_key =
      absl::StrCat(module_name, "#", module_id, "#", path);
  static absl::Mutex logged_mu(absl::kConstInit);
  static auto* logged_modules = new std::set<std::string>();
  {
    absl::MutexLock lock(&logged_mu);
    if (!logged_modules->insert(module_key).second) {
      return;
    }
  }

  LOG(INFO) << "[MUSA_XLA_EXECUTION_PATH] module=" << module_name
            << " module_id=" << module_id << " path=" << path
            << " has_thunks=" << has_thunks
            << " has_xla_runtime=" << has_xla_runtime
            << " has_gpu2_runtime=" << has_gpu2_runtime;
}

}  // namespace

StatusOr<std::unique_ptr<GpuExecutable>> GpuExecutable::Create(Params params) {
  auto executable = std::move(params.executable);
  std::unique_ptr<GpuExecutable> result(new GpuExecutable(std::move(params)));

  if (std::holds_alternative<OwnedThunkSequence>(executable)) {
    auto thunks = std::move(std::get<OwnedThunkSequence>(executable));
    if (IsTruthyEnv("MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS")) {
      MusaSmallGemmAccumRewriteOptions options;
      options.min_chain_size = std::max<int64_t>(
          2, ReadInt64Env("MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE", 4));
      options.max_chain_size = std::max<int64_t>(
          options.min_chain_size,
          ReadInt64Env("MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE", 64));
      options.max_k =
          std::max<int64_t>(1, ReadInt64Env("MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K",
                                           64));
      options.require_custom_kernel =
          IsTruthyEnv("MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL");
      options.log = IsTruthyEnv("MUSA_XLA_SMALL_GEMM_ACCUM_LOG") ||
                    IsMusaThunkDiagnosticsEnabled();
      MusaSmallGemmAccumRewriteStats stats;
      TF_ASSIGN_OR_RETURN(
          thunks, RewriteMusaSmallGemmAccumThunks(std::move(thunks), options,
                                                  &stats));
    }
    const bool group_gemm_thunks_requested =
        IsTruthyEnv("MUSA_XLA_GROUP_GEMM_THUNKS");
    const bool group_gemm_cross_kernel_diag_requested =
        IsTruthyEnv("MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG");
    if (group_gemm_thunks_requested || group_gemm_cross_kernel_diag_requested) {
      MusaGroupedGemmRewriteOptions options;
      options.min_group_size = std::max<int64_t>(
          1, ReadInt64Env("MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE", 4));
      options.max_group_size =
          std::max<int64_t>(options.min_group_size,
                            ReadInt64Env("MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE",
                                         64));
      options.diagnostic_only = !group_gemm_thunks_requested;
      options.log = IsTruthyEnv("MUSA_XLA_GROUP_GEMM_THUNKS_LOG") ||
                    IsMusaThunkDiagnosticsEnabled() ||
                    group_gemm_cross_kernel_diag_requested;
      MusaGroupedGemmRewriteStats stats;
      TF_ASSIGN_OR_RETURN(thunks,
                          RewriteMusaGroupGemmThunks(std::move(thunks),
                                                     options, &stats));
    }
    result->thunks_ = std::move(thunks);
    return result;
  }

  if (std::holds_alternative<OwnedGpuRuntimeProgram>(executable)) {
    auto& program = std::get<OwnedGpuRuntimeProgram>(executable);
    TF_ASSIGN_OR_RETURN(
        result->gpu_runtime_executable_,
        GpuRuntimeExecutable::Create(result->module_name_, std::move(program)));
    return result;
  }

  if (std::holds_alternative<OwnedGpu2RuntimeProgram>(executable)) {
    auto& program = std::get<OwnedGpu2RuntimeProgram>(executable);
    TF_ASSIGN_OR_RETURN(
        result->gpu2_runtime_executable_,
        Gpu2RuntimeExecutable::Create(std::move(program), result->text(),
                                      result->binary()));
    return result;
  }

  return InternalError("No XLA gpu executable was provided");
}

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(GpuExecutable::Params params)
    : Executable(std::move(params.debug_module)),
      text_(std::move(params.asm_text)),
      binary_(std::move(params.binary)),
      gpu_version_(params.gpu_version),
      entry_func_attrs_(params.entry_func_attrs),
      module_name_(params.module_name),
      output_shape_(params.output_shape),
      allocations_(std::move(params.allocations)),
      enable_persistent_temp_buffers_(params.enable_persistent_temp_buffers),
      debug_buffer_assignment_(std::move(params.debug_buffer_assignment)),
      verbose_buffer_assignment_string_dumper_(
          params.verbose_buffer_assignment_string_dumper),
      constants_(std::move(params.constants)),
      output_info_(std::move(params.output_info)),
      enable_debug_info_manager_(params.enable_debug_info_manager) {
  musa_classic_thunk_graph_cache_ =
      std::make_unique<MusaClassicThunkGraphCache>();
#if TENSORFLOW_USE_ROCM
  // ROCm uses hsaco hashes to distinguish between modules.
  // Bad things happen if multiple modules with identical code are loaded.
  binary_.resize(binary_.size() + 16);
  *(uint64_t*)(&binary_[binary_.size() - 16]) = tsl::EnvTime::NowNanos();
  *(uint64_t*)(&binary_[binary_.size() - 8]) = tsl::random::New64();
#endif
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(),
                                               debug_buffer_assignment_);
  }
}

GpuExecutable::~GpuExecutable() {
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }

  // Deallocate all persistent buffers.
  for (auto& [executor, map] : persistent_temp_buffers_) {
    for (const auto& alloc_buffer : map) {
      se::DeviceMemoryBase buffer = alloc_buffer.second;
      executor->UnifiedMemoryDeallocate(buffer.opaque());
    }
  }
}

Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  stream_executor::Platform::Id platform_id =
      main_stream->parent()->platform()->id();
  if (platform_id == stream_executor::rocm::kROCmPlatformId) {
    auto cc = main_stream->GetRocmComputeCapability();
    std::string stream_arch = cc.gcn_arch_name();
    std::string gpu_exec_arch =
        std::get<se::RocmComputeCapability>(gpu_version_).gcn_arch_name();
    TF_RET_CHECK(stream_arch == gpu_exec_arch)
        << "AMDGPU GCN ISA version mismatch; expected {" << gpu_exec_arch
        << ", but was " << stream_arch;
  } else if (platform_id == stream_executor::cuda::kCudaPlatformId) {
    se::GpuComputeCapability cc = main_stream->GetCudaComputeCapability();
    TF_RET_CHECK(std::get<se::CudaComputeCapability>(cc) ==
                 std::get<se::CudaComputeCapability>(gpu_version_))
        << "Compute capability mismatch; expected {"
        << std::get<se::CudaComputeCapability>(gpu_version_).ToString()
        << "}, but was {" << std::get<se::CudaComputeCapability>(cc).ToString()
        << "}";
  } else if (platform_id == stream_executor::musa::kMusaPlatformId) {
    // TODO: Add ISA check once MTGPU executable metadata is available.
  } else {
    return InternalError("Unknown platform");
  }

  return OkStatus();
}

namespace {

Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                           uint64_t start_nanos, se::Stream* stream_to_sync);

Status ExecuteThunks(const std::string& module_name, ModuleIdentifier module_id,
                     const ThunkSequence& thunk_sequence,
                     const ServiceExecutableRunOptions* run_options,
                     const BufferAllocations& buffer_allocations,
                     const std::vector<BufferAllocation>& allocations,
                     bool block_host_until_done,
                     bool use_highest_priority_for_async_stream,
                     MusaClassicThunkGraphCache* classic_thunk_graph_cache) {
  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();
  stream_executor::StreamPriority stream_priority =
      stream_executor::StreamPriority::Default;
  if (use_highest_priority_for_async_stream) {
    stream_priority = stream_executor::StreamPriority::Highest;
  }

  // Create the needed streams to support NcclCollectiveThunk.
  absl::InlinedVector<se::Stream*, kAsyncStreamTotal> async_comms_streams(
      kAsyncStreamTotal, nullptr);
  StatusOr<std::vector<StreamPool::Ptr>> streams = run_options->BorrowStreams(
      executor->device_ordinal(), kAsyncStreamTotal, stream_priority);
  if (streams.ok()) {
    for (int64_t i = 0; i < kAsyncStreamTotal; ++i) {
      async_comms_streams[i] = streams->at(i).get();
    }
  }
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();

  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  ScopedAnnotationAlways annotation([&] {
    std::string module_id_str;
    if (module_id >= 0) {
      module_id_str = absl::StrFormat(",program_id=%d", module_id);
    }
    return absl::StrFormat("XlaModule:#hlo_module=%s%s#", module_name,
                           module_id_str);
  });

  MaybeLogMusaThunkSummary(module_name, module_id, thunk_sequence, main_stream);

  const bool thunk_timing_enabled =
      IsMusaThunkTimingEnabled() &&
      main_stream->parent()->platform()->id() ==
          stream_executor::musa::kMusaPlatformId;
  if (IsMusaClassicThunkGraphEnabled()) {
    if (main_stream->parent()->platform()->id() !=
        stream_executor::musa::kMusaPlatformId) {
      LogMusaClassicThunkGraph(module_name, module_id, /*eligible=*/false,
                               /*cache_hit=*/false, /*captured=*/false,
                               "not_musa", thunk_sequence.size(), 0, 0, 0);
    } else if (thunk_timing_enabled) {
      LogMusaClassicThunkGraph(module_name, module_id, /*eligible=*/false,
                               /*cache_hit=*/false, /*captured=*/false,
                               "thunk_timing_enabled",
                               thunk_sequence.size(), 0, 0, 0);
    } else {
      TF_ASSIGN_OR_RETURN(MusaClassicThunkGraphExecution graph_execution,
                          TryExecuteMusaClassicThunkGraph(
                              module_name, module_id, thunk_sequence,
                              run_options, buffer_allocations, allocations,
                              classic_thunk_graph_cache));
      if (graph_execution.executed) {
        return MaybeSyncAndProfile(run_options, start_nanos,
                                   block_host_until_done ? main_stream
                                                         : nullptr);
      }
    }
  }
  std::vector<MusaThunkTimingItem> thunk_timings;
  absl::flat_hash_map<std::string, MusaThunkKindTiming> thunk_kind_totals;
  int64_t thunk_timing_total_us = 0;
  if (thunk_timing_enabled) {
    thunk_timings.reserve(thunk_sequence.size());
  }

  int64_t thunk_index = 0;
  for (const std::unique_ptr<Thunk>& thunk : thunk_sequence) {
    // Annotate execution of this op if tracing was enabled when we started
    // running this module.  If tracing is enabled *while* we're running the
    // module, we won't get any data, but that's probably an OK trade-off.
    ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });
    VLOG(2) << "Executing the thunk for " << thunk->profile_annotation();
    if (NeedsAsyncCommsStream(*thunk)) {
      for (se::Stream* async_stream : async_comms_streams) {
        TF_RET_CHECK(async_stream != nullptr)
            << "`run_options` must have a stream borrower for async thunks.";
      }
    }

    Thunk::ExecuteParams thunk_params{*run_options, buffer_allocations,
                                      main_stream, async_comms_streams};
    if (thunk_timing_enabled) {
      TF_RETURN_IF_ERROR(ExecuteThunkWithOptionalTiming(
          thunk_index, *thunk, thunk_params, main_stream, &thunk_timings,
          &thunk_kind_totals, &thunk_timing_total_us));
    } else {
      TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(thunk_params));
    }
    ++thunk_index;
  }
  if (thunk_timing_enabled) {
    LOG(INFO) << "[MUSA_THUNK_TIMING] module=" << module_name
              << " module_id=" << module_id
              << " total_thunks=" << thunk_sequence.size()
              << " total_us=" << thunk_timing_total_us
              << " kind_totals={"
              << FormatMusaThunkTimingKindTotals(thunk_kind_totals) << "}"
              << " top_thunks={"
              << FormatMusaThunkTimingTopThunks(std::move(thunk_timings))
              << "}";
  }
  return MaybeSyncAndProfile(run_options, start_nanos,
                             block_host_until_done ? main_stream : nullptr);
}

Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                           uint64_t start_nanos,
                           se::Stream* stream_to_sync = nullptr) {
  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (stream_to_sync) {
    if (IsMusaDebugDeallocEnabled() &&
        stream_to_sync->parent()->platform()->id() ==
            stream_executor::musa::kMusaPlatformId) {
      LOG(INFO) << "[MUSA_DEALLOC_DEBUG] BlockHostUntilDone begin stream="
                << stream_to_sync;
    }
    Status block_status = stream_to_sync->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
          "Failed to complete all kernels launched on stream %p: %s",
          stream_to_sync, block_status.message());
    }
    if (IsMusaDebugDeallocEnabled() &&
        stream_to_sync->parent()->platform()->id() ==
            stream_executor::musa::kMusaPlatformId) {
      LOG(INFO) << "[MUSA_DEALLOC_DEBUG] BlockHostUntilDone end stream="
                << stream_to_sync;
    }
  }

  // FinishExecution() blocks until main_stream has completed if profiling is
  // enabled; we therefore do not need to defer profile collection onto a
  // stream.
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (run_options->run_options().execution_profile()) {
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    const double nanoseconds = end_nanos - start_nanos;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return OkStatus();
}

}  // namespace

StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();
  const bool is_musa_executor =
      executor->platform()->id() == stream_executor::musa::kMusaPlatformId;

  absl::MutexLock lock(&module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // The CUDA driver isn't able to load a PTX and a binary which are both empty.
  // It's okay if we skip loading in this case; if the module isn't loaded, all
  // symbol lookups will fail, just as they should for an empty module.
  if (!(executor->platform()->id() == stream_executor::cuda::kCudaPlatformId &&
        binary().empty() && text().empty())) {
    TF_RETURN_IF_ERROR(executor->LoadModule(module_spec, &module_handle));
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const ConstantInfo& info : constants_) {
    StatusOr<stream_executor::DeviceMemoryBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status =
          executor->GetUntypedSymbol(info.symbol_name, module_handle);
    }

    // On MUSA, uploading literals into fatbin module globals (addresses from
    // muModuleGetGlobal) has been observed to wedge or time out for larger
    // constants, while standalone allocations behave like the CUDA fallback path.
    // Fusion kernels pass constant data via i8* parameters, so device memory
    // from CreateOrShareConstant is sufficient; the LLVM global remains unused
    // at runtime for those kernels.
    const bool musa_standalone_constant =
        is_musa_executor && !info.content.empty();

    se::DeviceMemoryBase global;
    if (static_cast<bool>(module_handle) && global_status.ok() &&
        !musa_standalone_constant) {
      // The constant was defined in the PTX and has been allocated by the CUDA
      // driver.
      global = *global_status;
      VLOG(3) << "Resolved global " << info.symbol_name << " to "
              << global.opaque();

      if (!info.content.empty()) {
        // This means the constant did not have an initializer in the PTX and
        // therefore must be initialized by XLA here.
        stream->ThenMemcpy(&global, info.content.data(), info.content.size());
        submitted_mem_copies = true;
      }
    } else {
      // The constant was not defined in the PTX (or MUSA uses a standalone
      // buffer for runtime-initialized literals) and must be allocated and
      // initialized by XLA here.
      CHECK(!info.content.empty());

      std::vector<uint8_t> payload(info.content.begin(), info.content.end());
      if (info.allocation_index >= 0) {
        const auto& allocs = GetAllocations();
        if (info.allocation_index < static_cast<int>(allocs.size())) {
          const int64_t want = allocs[info.allocation_index].size();
          if (want > static_cast<int64_t>(payload.size())) {
            payload.resize(static_cast<size_t>(want), 0);
          }
        }
      }

      TF_ASSIGN_OR_RETURN(
          auto shared, executor->CreateOrShareConstant(stream, payload));
      global = *shared;
      VLOG(3) << "Allocated (or shared) global " << info.symbol_name << " at "
              << global.opaque();
      if (musa_standalone_constant) {
        VLOG(3) << " (MUSA standalone constant buffer; skipped module global)";
      }
      // XLA will continue to own this global at least until this executable is
      // destroyed (longer if another, longer-lived executable shares the same
      // constant).
      shared_constants_.push_back(std::move(shared));
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    TF_CHECK_OK(stream->BlockHostUntilDone());
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return module_globals_.emplace(executor, std::move(globals))
      .first->second.get();
}

StatusOr<se::DeviceMemoryBase> GpuExecutable::BufferForAllocation(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
    int64_t arg_idx) {
  if (allocation.is_thread_local()) {
    return se::DeviceMemoryBase{};
  } else if (allocation.is_entry_computation_parameter()) {
    int64_t param_no = allocation.parameter_number();
    se::DeviceMemoryBase registered_buffer = [&] {
      if (auto unowned_shapedbuffers =
              std::get_if<absl::Span<const ShapedBuffer* const>>(&arguments)) {
        return (*unowned_shapedbuffers)[param_no]->buffers().element(
            allocation.param_shape_index());
      } else {
        return std::get<absl::Span<ExecutionInput>>(arguments)[param_no]
            .Buffer(allocation.param_shape_index())
            .AsDeviceMemoryBase();
      }
    }();
    if (registered_buffer.is_null() && registered_buffer.size() > 0) {
      return FailedPrecondition(
          "Cannot run XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has "
          "zero elements.",
          allocation.param_shape_index().ToString(), param_no);
    }
    return registered_buffer;
  } else if (allocation.is_constant()) {
    auto it = globals->find(arg_idx);
    if (it == globals->end()) {
      return se::DeviceMemoryBase();
    }
    return it->second;
  } else {
    // Allocate each allocation that might escape, or is the temp buffer.
    CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
    const int64_t buffer_size = allocation.size();
    se::DeviceMemoryBase buffer_address;
    if (buffer_size > 0) {
      StatusOr<se::OwningDeviceMemory> buffer =
          memory_allocator->Allocate(device_ordinal, buffer_size);
      if (!buffer.ok()) {
        return ResourceExhausted("%s\n%s\n", buffer.status().message(),
                                 verbose_buffer_assignment_string_dumper_());
      }
      buffer_address = buffer->Release();
    }
    return buffer_address;
  }
}

static Status CheckAlignment(const BufferAllocation& allocation,
                             se::DeviceMemoryBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return kEntryParameterAlignBytes;
    } else if (allocation.is_constant()) {
      return kConstantBufferAlignBytes;
    } else {
      return kXlaAllocatedBufferAlignBytes;
    }
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return InternalError(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return OkStatus();
}

StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
    const BufferAllocToDeviceMemoryMap& buffer_alloc_to_persistent_memory_map) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tsl::profiler::TraceMeLevel::kInfo);

  const int64_t num_buffers = allocations_.size();
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = allocations_[i];
    // Check if the buffer is already stored as a persistent buffer.
    se::DeviceMemoryBase buffer;
    if (buffer_alloc_to_persistent_memory_map.contains(allocation.index())) {
      buffer = buffer_alloc_to_persistent_memory_map.at(allocation.index());
    } else {
      TF_ASSIGN_OR_RETURN(
          buffer, BufferForAllocation(arguments, globals, allocation,
                                      memory_allocator, device_ordinal, i));
    }

    buffers.push_back(buffer);
    TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffer, i));
  }
  return {{buffers, device_ordinal, memory_allocator}};
}

StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return ExecuteAsyncOnStreamImpl(run_options, absl::MakeSpan(arguments));
}

StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  TF_ASSIGN_OR_RETURN(ExecutionOutput out,
                      ExecuteAsyncOnStreamImpl(run_options, arguments));
  return out.ConsumeResult();
}

static Status ExecuteXlaRuntime(const std::string& module_name,
                                ModuleIdentifier module_id,
                                GpuRuntimeExecutable& gpu_runtime_executable,
                                const ServiceExecutableRunOptions* run_options,
                                const std::string& asm_text,
                                const std::vector<uint8_t>& binary,
                                const BufferAllocations& buffer_allocations,
                                const BufferAllocation* temp_buffer,
                                bool block_host_until_done,
                                NonAtomicallyUpgradeableRWLock& gpu_lock) {
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();

  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  ScopedAnnotationAlways annotation([&] {
    std::string module_id_str;
    if (module_id >= 0) {
      module_id_str = absl::StrFormat(",program_id=%d", module_id);
    }
    return absl::StrFormat("XlaModule:#hlo_module=%s%s#", module_name,
                           module_id_str);
  });

  auto executed = gpu_runtime_executable.Execute(
      run_options, asm_text, binary, buffer_allocations, gpu_lock, temp_buffer);
  if (!executed.ok()) return executed;

  return MaybeSyncAndProfile(
      run_options, start_nanos,
      block_host_until_done ? run_options->stream() : nullptr);
}

static Status ExecuteXlaRuntime2(const std::string& module_name,
                                 ModuleIdentifier module_id,
                                 Gpu2RuntimeExecutable& gpu2_executable,
                                 const ServiceExecutableRunOptions* run_options,
                                 const BufferAllocations& buffer_allocations,
                                 const BufferAllocation* temp_buffer,
                                 bool block_host_until_done) {
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();

  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  ScopedAnnotationAlways annotation([&] {
    std::string module_id_str;
    if (module_id >= 0) {
      module_id_str = absl::StrFormat(",program_id=%d", module_id);
    }
    return absl::StrFormat("XlaModule:#hlo_module=%s%s#", module_name,
                           module_id_str);
  });

  auto executed =
      gpu2_executable.Execute(run_options, buffer_allocations, temp_buffer);
  if (!executed.ok()) return executed;

  return MaybeSyncAndProfile(
      run_options, start_nanos,
      block_host_until_done ? run_options->stream() : nullptr);
}

Status GpuExecutable::PopulatePersistentTempBuffers(
    se::StreamExecutor* executor) {
  auto search = persistent_temp_buffers_.find(executor);
  if (search != persistent_temp_buffers_.end()) {
    return OkStatus();
  }

  // Allocate persistent temp buffers.
  BufferAllocToDeviceMemoryMap buffer_alloc_to_device_memory_map;
  for (const BufferAllocation& allocation : allocations_) {
    if (!allocation.IsPreallocatedTempBuffer()) {
      continue;
    }

    const int64_t buffer_size = allocation.size();
    void* ptr = executor->UnifiedMemoryAllocate(buffer_size);
    if (ptr) {
      se::DeviceMemoryBase buffer(ptr, buffer_size);
      buffer_alloc_to_device_memory_map[allocation.index()] = buffer;
    }
  }

  persistent_temp_buffers_[executor] = buffer_alloc_to_device_memory_map;
  return OkStatus();
}

StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options,
    VariantArguments arguments) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuExecutable::ExecuteAsyncOnStreamImpl(", module_name_, ")"));
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  se::StreamExecutor* executor = run_options->stream()->parent();

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // GpuExecutable always bound to a single GpuContext during its execution, so
  // we activate it once to skip expensive context activations later.
  se::gpu::GpuExecutor* gpu_executor = se::gpu::ExtractGpuExecutor(executor);
  se::gpu::ScopedActivateExecutorContext activation(gpu_executor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  // If persistent buffers are enabled, the executable cannot execute
  // concurrently, therefore performance can suffer under contention.
  absl::MutexLockMaybe lock(
      enable_persistent_temp_buffers_ ? &persistent_temp_buffers_mu_ : nullptr);

  // Map from buffer allocation to persistent temp buffers. It is empty if
  // persistent temp buffer is not enabled.
  BufferAllocToDeviceMemoryMap persistent_buffers_map = {};

  if (enable_persistent_temp_buffers_) {
    persistent_temp_buffers_mu_.AssertHeld();
    TF_RETURN_IF_ERROR(PopulatePersistentTempBuffers(executor));
    persistent_buffers_map = persistent_temp_buffers_[executor];
  }

  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously. We do not add locking for
  // "recursive" invocations, which are done when holding a lock already.
  NonAtomicallyUpgradeableRWLock gpu_lock(&GetGpuMutex(executor));
  std::optional<NonAtomicallyUpgradeableRWLock::WriterLock> exclusive_gpu_lock;
  const gpu::GpuExecutableRunOptions* gpu_opts =
      run_options->run_options().gpu_executable_run_options();
  if (gpu_opts && gpu_opts->requires_exclusive_lock_on_gpu()) {
    exclusive_gpu_lock.emplace(&gpu_lock);
  }

  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  auto device_ordinal = executor->device_ordinal();
  ExecutionOutput result(/*on_device_shape=*/output_shape_, memory_allocator,
                         device_ordinal);

  TF_ASSIGN_OR_RETURN(
      BufferAllocations buffer_allocations,
      GenerateBufferAllocations(arguments, globals, memory_allocator,
                                device_ordinal, persistent_buffers_map));
  VLOG(2) << buffer_allocations.ToString();
  std::set<se::DeviceMemoryBase> buffers_in_result;

  const bool is_entire_tuple_contents_aliased = [&] {
    for (auto& p : result.MutableResult()->buffers().leaves()) {
      if (!output_info_.contains(p.first)) {
        continue;
      }
      const OutputInfo& output_info = output_info_.at(p.first);
      if (!output_info.alias_config.has_value()) {
        return false;
      }
    }
    return true;
  }();

  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    if (!output_info_.contains(index)) {
      continue;
    }
    const OutputInfo& output_info = output_info_.at(index);
    const BufferAllocation* allocation =
        &allocations_[output_info.allocation_index];
    se::DeviceMemoryBase& result_buffer = p.second;

    VLOG(4) << "Looking at: allocation " << output_info.allocation_index
            << " @ index: " << index.ToString();

    if (output_info.alias_config) {
      MaybeOwningDeviceMemory* maybe_owning_memory =
          [&]() -> xla::MaybeOwningDeviceMemory* {
        // ScopedBuffer is never an owned buffer.
        if (std::holds_alternative<absl::Span<const ShapedBuffer* const>>(
                arguments)) {
          return nullptr;
        } else {
          auto unowned_execution_input =
              std::get<absl::Span<ExecutionInput>>(arguments);
          ExecutionInput& input =
              unowned_execution_input[allocation->parameter_number()];
          return input.MutableBuffer(allocation->param_shape_index());
        }
      }();
      if (output_info.alias_config->must_alias() && maybe_owning_memory &&
          !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (maybe_owning_memory && maybe_owning_memory->HasOwnership()) {
        std::optional<tensorflow::se::OwningDeviceMemory> owning =
            maybe_owning_memory->Release();
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        // The caller is giving us the
        // input buffer, but in case of error from the execute call, we should
        // not be releasing it as it contains valid data (for example, it is a
        // parameter which the user wants us to alias, in a gradient update
        // computation). So we store the index into the result in the aliased
        // vector, which will be fed to the ExecutionOutput, which will use
        // the indices to drop the addresses from its own ScopedShapedBuffer
        // result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(index);
      } else if (!output_info.passthrough &&
                 !ShapeUtil::GetSubshape(output_shape_, index).IsTuple()) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size =
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(output_shape_, index));
        StatusOr<se::OwningDeviceMemory> allocated_buffer =
            memory_allocator->Allocate(device_ordinal, allocation_size);
        if (!allocated_buffer.ok()) {
          return ResourceExhausted("%s\n%s\n",
                                   allocated_buffer.status().message(),
                                   verbose_buffer_assignment_string_dumper_());
        }
        result_buffer = allocated_buffer->Release();
        se::DeviceMemoryBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        run_options->stream()->ThenMemcpyD2D(&result_buffer, aliased_buffer,
                                             aliased_buffer.size());
        aliased_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);

      // If the entire tuple contents is aliased, the copy insertion will *not*
      // materialize a new tuple, so we mark it as aliased as well.
      if (is_entire_tuple_contents_aliased) {
        result.AddAliasedIndex(index);
      }
    }
    buffers_in_result.insert(result_buffer);
  }

  Status execute_status = ExecuteThunksOrXlaRuntime(
      run_options, buffer_allocations, block_host_until_done, gpu_lock);
  if (IsMusaDebugDeallocEnabled() &&
      executor->platform()->id() == stream_executor::musa::kMusaPlatformId) {
    LOG(INFO) << "[MUSA_DEALLOC_DEBUG] ExecuteThunksOrXlaRuntime status for "
              << module_name_ << ": " << execute_status;
  }
  TF_RETURN_IF_ERROR(execute_status);

  // Free all temporary allocations.
  std::vector<BufferAllocation> non_persistent_allocations;
  for (const BufferAllocation& allocation : allocations_) {
    if (!persistent_buffers_map.contains(allocation.index())) {
      non_persistent_allocations.push_back(allocation);
    }
  }
  if (IsMusaDebugDeallocEnabled() &&
      executor->platform()->id() == stream_executor::musa::kMusaPlatformId) {
    LOG(INFO) << "[MUSA_DEALLOC_DEBUG] Execute finished for module "
              << module_name_ << " block_host_until_done="
              << block_host_until_done << " stream="
              << run_options->stream();
    LOG(INFO) << "[MUSA_DEALLOC_DEBUG] Begin TearDown for module "
              << module_name_ << " on device ordinal " << device_ordinal
              << "; candidate allocations=" << non_persistent_allocations.size();
    for (const BufferAllocation& allocation : non_persistent_allocations) {
      se::DeviceMemoryBase buffer_address =
          buffer_allocations.GetDeviceAddress(allocation.index());
      const bool is_live_out = buffers_in_result.count(buffer_address) > 0;
      if ((allocation.maybe_live_out() && !is_live_out) ||
          allocation.IsPreallocatedTempBuffer()) {
        LOG(INFO) << "[MUSA_DEALLOC_DEBUG] TearDown candidate "
                  << DescribeAllocationForDebug(allocation, buffer_address,
                                                is_live_out);
      }
    }
  }
  TF_RETURN_IF_ERROR(buffer_allocations.TearDown(buffers_in_result,
                                                 non_persistent_allocations));
  if (IsMusaDebugDeallocEnabled() &&
      executor->platform()->id() == stream_executor::musa::kMusaPlatformId) {
    LOG(INFO) << "[MUSA_DEALLOC_DEBUG] TearDown finished for module "
              << module_name_ << " on device ordinal " << device_ordinal;
  }

  // Free allocations for arguments.
  if (auto args = std::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

Status GpuExecutable::ExecuteThunksOrXlaRuntime(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done,
    NonAtomicallyUpgradeableRWLock& gpu_lock) {
  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));

  // There isn't always an HLO module.
  ModuleIdentifier unique_id = -1;
  if (has_module()) {
    unique_id = module().unique_id();
  }

  MaybeLogMusaExecutionPath(module_name_, unique_id, run_options->stream(),
                            static_cast<bool>(thunks_),
                            static_cast<bool>(gpu_runtime_executable_),
                            static_cast<bool>(gpu2_runtime_executable_));

  if (thunks_) {
    se::StreamExecutor* executor = run_options->stream()->parent();
    for (const std::unique_ptr<Thunk>& thunk : *thunks_) {
      TF_RETURN_IF_ERROR(thunk->Initialize(*this, executor));
    }

    return ExecuteThunks(
        module_name_, unique_id, *thunks_, run_options, buffer_allocations,
        allocations_, block_host_until_done,
        /*use_highest_priority_for_async_stream*/
        has_module() ? module_config()
                           .debug_options()
                           .xla_gpu_enable_highest_priority_async_stream()
                     : false,
        musa_classic_thunk_graph_cache_.get());
  }

  // Match IrEmitter's temp buffer allocation for kernel launches. See
  // IrEmitterUnnested::BuildKernelThunkImpl().
  const BufferAllocation* temp_buffer = nullptr;
  for (const BufferAllocation& alloc : allocations_) {
    if (alloc.IsPreallocatedTempBuffer()) {
      // Retrieve the first seen temp buffer.
      if (temp_buffer == nullptr) temp_buffer = &alloc;
    }
  }

  if (gpu_runtime_executable_) {
    return ExecuteXlaRuntime(module_name_, unique_id, *gpu_runtime_executable_,
                             run_options, text_, binary_, buffer_allocations,
                             temp_buffer, block_host_until_done, gpu_lock);
  }

  if (gpu2_runtime_executable_) {
    return ExecuteXlaRuntime2(
        module_name_, unique_id, *gpu2_runtime_executable_, run_options,
        buffer_allocations, temp_buffer, block_host_until_done);
  }

  return FailedPrecondition("Expected XLA gpu executable is not supplied.");
}

int64_t GpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  int64_t size = binary().size();
  for (BufferAllocation::Index i = 0; i < allocations_.size(); ++i) {
    const BufferAllocation& allocation = allocations_[i];
    if (allocation.is_constant()) {
      size += allocation.size();
    }
  }
  return size;
}

Status GpuExecutable::SetUpMlirAllocation(
    mlir::func::FuncOp func, llvm::ArrayRef<int64_t> buffer_sizes,
    std::vector<BufferAllocation>* allocations,
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>* output_info,
    Shape* output_shape) {
  for (int i = 0; i < buffer_sizes.size(); i++) {
    allocations->emplace_back(i, buffer_sizes[i], 0);
  }

  for (int i = 0; i < func.getNumArguments(); i++) {
    if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
      xla::ShapeIndex shape_index;
      if (auto shape_index_attr =
              func.getArgAttrOfType<mlir::DenseIntElementsAttr>(
                  i, "lmhlo.param_shape_index")) {
        for (const llvm::APInt& element : shape_index_attr) {
          shape_index.push_back(element.getSExtValue());
        }
      }
      allocations->at(i).set_entry_computation_parameter(
          param_attr.cast<mlir::IntegerAttr>().getInt(), shape_index,
          static_cast<bool>(func.getArgAttr(i, "lmhlo.output_index")));
    }
    // TODO(timshen): this information is redundant. This is here only for
    // smooth migration to LMHLO. Remove it.
    if (func.getArgAttr(i, "lmhlo.constant_name")) {
      allocations->at(i).set_constant(true);
    }
    if (auto output_index_attr = func.getArgAttr(i, "lmhlo.output_index")) {
      allocations->at(i).set_maybe_live_out(true);

      // Reconstruct a shape index from output_index.
      ShapeIndex shape_index;
      for (const llvm::APInt& element :
           output_index_attr.cast<mlir::DenseIntElementsAttr>()) {
        shape_index.push_back(element.getSExtValue());
      }
      auto& o = (*output_info)[shape_index];
      o.allocation_index = i;
      if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
        HloInputOutputAliasConfig::AliasKind kind =
            HloInputOutputAliasConfig::kMayAlias;
        if (func.getArgAttr(i, "lmhlo.must_alias")) {
          kind = HloInputOutputAliasConfig::kMustAlias;
        }
        o.alias_config.emplace(param_attr.cast<mlir::IntegerAttr>().getInt(),
                               ShapeIndex{}, kind);
      }
      if (func.getArgument(i).use_empty()) {
        o.passthrough = true;
      }
    }
  }
  // Expects result_xla_shape as a XLA shape in string form.
  //
  // The attribute is necessary, because GpuExecutable/ExecutionOutput supports
  // tuples / tree-like shapes, while the LMHLO argument list loses the tree
  // form.
  //
  // The string format is necessary since MLIR doesn't support XLA shape with
  // dynamic_dimension.
  //
  // TODO(timshen): now this field is mandatory. Make it optional for
  // non-GpuExecutable outputs.
  TF_ASSIGN_OR_RETURN(
      *output_shape,
      ParseShape(func->getAttrOfType<mlir::StringAttr>("result_xla_shape")
                     .getValue()
                     .str()));

  return OkStatus();
}

StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment) {
  const HloInstruction* root =
      hlo_module.entry_computation()->root_instruction();

  InstructionValueSet root_value_set =
      assignment.dataflow_analysis().GetInstructionValueSet(root);

  if (root_value_set.IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  using OutputInfoMap =
      absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;
  OutputInfoMap output;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& /*sub_shape*/, const ShapeIndex& index) -> Status {
        const auto& sources = root_value_set.element(index);
        // The points-to set is unambiguous so the set should be a
        // singleton. That is, we know exactly which instruction
        // produced the array at this element.
        CHECK_EQ(1, sources.values().size());
        HloInstruction* src_hlo = sources.values()[0]->instruction();

        GpuExecutable::OutputInfo& info = output[index];
        info.passthrough = src_hlo->opcode() == HloOpcode::kParameter;
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            assignment.GetUniqueSlice(src_hlo, sources.values()[0]->index()));
        CHECK_EQ(slice.offset(), 0) << "Parameter should get its own slice";
        info.allocation_index = slice.index();

        output[index].alias_config =
            hlo_module.input_output_alias_config().GetAliasedParameter(index);

        return OkStatus();
      }));
  return output;
}

GpuExecutable::GpuExecutable(
    std::shared_ptr<HloModule> hlo_module, std::string asm_text,
    std::vector<uint8_t> binary, std::vector<ConstantInfo> constants,
    se::GpuComputeCapability gpu_version,
    xla::EntryFunctionAttributes entry_func_attrs,
    absl::string_view module_name, Shape xla_output_shape,
    std::vector<BufferAllocation> allocations,
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
    std::unique_ptr<GpuRuntimeExecutable> gpu_runtime_executable)
    : Executable(std::move(hlo_module)),
      text_(std::move(asm_text)),
      binary_(std::move(binary)),
      gpu_version_(gpu_version),
      gpu_runtime_executable_(std::move(gpu_runtime_executable)),
      entry_func_attrs_(entry_func_attrs),
      module_name_(module_name),
      output_shape_(xla_output_shape),
      allocations_(std::move(allocations)),
      constants_(std::move(constants)),
      output_info_(std::move(output_info)),
      enable_debug_info_manager_(true) {
  musa_classic_thunk_graph_cache_ =
      std::make_unique<MusaClassicThunkGraphCache>();
  if (has_module()) {
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(),
                                               debug_buffer_assignment_);
  }
}

// Returns a list of functions exported from the `module` that should be loaded
// from the object file. Entrypoint functions always loaded with ordinal 0.
static StatusOr<std::vector<runtime::Executable::LoadFunction>>
GetFunctionsToLoad(mlir::ModuleOp module, std::string_view entry) {
  std::vector<runtime::Executable::LoadFunction> functions;

  // Use canonical type converter because we currently do not support any
  // user-defined types in XLA:GPU executables.
  runtime::TypeConverter type_converter;

  // Converts function type and adds load function metadata. In XLA:GPU exported
  // function runtime signature is the same as regular signature with an extra
  // execution context argument at index 0.
  auto convert = [&](mlir::func::FuncOp func) -> Status {
    auto signature = type_converter.Convert(func.getFunctionType());
    if (!signature.ok())
      return InternalError("Failed to convert entry function type: %s",
                           signature.status().message());

    // TODO(ezhulenev): Copy `signature` once FunctionType is copyable.
    auto rt_signature = type_converter.Convert(func.getFunctionType());
    rt_signature->insert_operand(
        0, std::make_unique<runtime::ExecutionContextOperandType>());

    functions.push_back({func.getName().str(), std::move(*signature),
                         std::move(*rt_signature)});

    return OkStatus();
  };

  mlir::SymbolTable sym_table(module);

  // Load entrypoint function first at ordinal 0.
  TF_CHECK_OK(convert(module.lookupSymbol<mlir::func::FuncOp>(entry)));

  // Load all functions explicitly exported from the module (in XLA:GPU it's
  // always CUDA graph capture functions). We explicitly sort them by ordinal,
  // to make sure they are loaded in correct order.
  auto export_ops = llvm::to_vector(module.getOps<runtime::ExportOp>());
  llvm::sort(export_ops, [](runtime::ExportOp a, runtime::ExportOp b) {
    return b.getOrdinal()->getSExtValue() < b.getOrdinal()->getSExtValue();
  });
  for (runtime::ExportOp exported : export_ops) {
    TF_CHECK_OK(convert(
        sym_table.lookup<mlir::func::FuncOp>(exported.getFunctionRef())));
  }

  return functions;
}

// Get arguments buffer sizes from the entry function signature.
static StatusOr<std::vector<int64_t>> GetBufferSizes(runtime::FunctionType& f) {
  std::vector<int64_t> buffer_sizes;
  for (unsigned i = 0; i < f.num_operands(); ++i) {
    auto* memref = llvm::dyn_cast<runtime::MemrefType>(f.operand(i));

    // Entry function argument must be a statically shaped 1d I8 memref.
    if (memref == nullptr || memref->element_type() != PrimitiveType::S8 ||
        memref->rank() != 1 || runtime::MemrefType::IsDynamic(memref->size(0)))
      return InternalError("Illegal buffer argument type: %s",
                           f.operand(0)->ToString());

    buffer_sizes.push_back(memref->size(0));
  }
  return buffer_sizes;
}

StatusOr<std::unique_ptr<Executable>> GpuExecutable::LoadFromObjFile(
    std::shared_ptr<HloModule> hlo_module, absl::string_view obj_file,
    absl::string_view mlir_module,
    xla::EntryFunctionAttributes entry_func_attrs, DebugOptions debug_options,
    absl::string_view asm_text, absl::string_view binary,
    std::vector<ConstantInfo> constants, se::GpuComputeCapability gpu_version,
    se::StreamExecutor* executor) {
  VLOG(1) << "Load serialized Gpu executable from object file: module="
          << hlo_module->name();

  std::string_view entry = hlo_module->entry_computation()->name();

  // Load MLIR module behind the compiled object file to recover XLA allocations
  // and output info details. Also recover buffer sizes from the entrypoint
  // function signature.
  mlir::MLIRContext context;
  runtime::AppendXlaGpuDialectRegistry(context);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_module, &context);
  if (!module) return InternalError("Failed to parse AOT compiled module");

  // Get the list of functions to be loaded from the object file.
  TF_ASSIGN_OR_RETURN(std::vector<runtime::Executable::LoadFunction> functions,
                      GetFunctionsToLoad(*module, entry));
  VLOG(2) << "Found " << functions.size() << " functions to load";

  // Get the buffer sizes from the entry function signature.
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> buffer_sizes,
                      GetBufferSizes(functions[0].signature));

  // Get the XLA module entrypoint function.
  auto func = mlir::cast<mlir::func::FuncOp>(module->lookupSymbol(entry));

  // Infer XLA allocations and output info from the MLIR module.
  std::vector<BufferAllocation> allocations;
  absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
  Shape result_xla_shape;
  TF_RETURN_IF_ERROR(SetUpMlirAllocation(func, buffer_sizes, &allocations,
                                         &output_info, &result_xla_shape));

  // Create a named buffer from compiled object file.
  llvm::StringRef data(obj_file.data(), obj_file.size());
  auto buffer = llvm::MemoryBuffer::getMemBuffer(data, hlo_module->name());

  auto symbol_map = runtime::ToSymbolsBinding(RegisterXlaGpuRuntimeCustomCalls,
                                              RegisterXlaGpuTypeIdNames);

  // Load XLA Runtime executable from an object file, and link it with Gpu
  // runtime intrinsics implementing Gpu custom calls.
  auto executable = runtime::Executable::LoadFromObjFile(
      hlo_module->name(), std::move(buffer), std::move(functions), symbol_map);

  if (!executable.ok())
    return InternalError("Failed to load XLA Runtime executable: %s",
                         executable.status().message());

  // Move runtime::Executable ownership to the GpuRuntimeExecutable.
  TF_ASSIGN_OR_RETURN(auto gpu_runtime_executable,
                      GpuRuntimeExecutable::Create(
                          hlo_module->name(), buffer_sizes,
                          std::move(*executable), std::move(debug_options)));

  // Construct GpuExecutable for the loaded XLA Runtime executable.
  std::string name = hlo_module->name();
  std::string asm_text_string = std::string(asm_text);
  std::vector<uint8_t> binary_vector(binary.begin(), binary.end());
  return std::unique_ptr<Executable>(new GpuExecutable(
      std::move(hlo_module), std::move(asm_text_string),
      std::move(binary_vector), std::move(constants), gpu_version,
      entry_func_attrs, name, result_xla_shape, std::move(allocations),
      std::move(output_info), std::move(gpu_runtime_executable)));
}

StatusOr<std::string_view> GpuExecutable::GetObjFile() const {
  if (!gpu_runtime_executable_)
    return Internal("gpu_runtime_executable is null");
  return gpu_runtime_executable_->GetObjFile();
}

StatusOr<std::string_view> GpuExecutable::GetMlirModule() const {
  if (!gpu_runtime_executable_)
    return Internal("gpu_runtime_executable is null");
  return gpu_runtime_executable_->GetMlirModule();
}

}  // namespace gpu
}  // namespace xla
