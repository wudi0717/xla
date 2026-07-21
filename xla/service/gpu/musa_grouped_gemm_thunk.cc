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

#include "xla/service/gpu/musa_grouped_gemm_thunk.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gemm_thunk.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {
namespace {

bool IsTruthyEnv(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
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

std::vector<std::string> TopCounts(
    const absl::flat_hash_map<std::string, int64_t>& counts, int64_t limit) {
  std::vector<std::pair<std::string, int64_t>> sorted(counts.begin(),
                                                       counts.end());
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
}

bool SameMatrixLayoutIgnoringBatch(const MatrixLayout& a,
                                   const MatrixLayout& b) {
  return a.dtype == b.dtype && a.num_rows == b.num_rows &&
         a.num_cols == b.num_cols && a.order == b.order &&
         a.leading_dim_stride == b.leading_dim_stride &&
         a.batch_size == 1 && b.batch_size == 1;
}

bool SameGemmConfigForBatching(const GemmConfig& a, const GemmConfig& b) {
  return SameMatrixLayoutIgnoringBatch(a.lhs_layout, b.lhs_layout) &&
         SameMatrixLayoutIgnoringBatch(a.rhs_layout, b.rhs_layout) &&
         SameMatrixLayoutIgnoringBatch(a.c_layout, b.c_layout) &&
         SameMatrixLayoutIgnoringBatch(a.output_layout, b.output_layout) &&
         a.alpha == b.alpha && a.beta == b.beta &&
         a.algorithm == b.algorithm &&
         a.compute_precision == b.compute_precision;
}

enum class GemmDependencyKind {
  kNone,
  kOutputOutput,
  kOutputLhs,
  kOutputRhs,
};

void RecordDependency(GemmDependencyKind kind,
                      MusaGroupedGemmRewriteStats* stats) {
  if (kind == GemmDependencyKind::kNone) {
    return;
  }
  ++stats->filtered_dependency;
  switch (kind) {
    case GemmDependencyKind::kOutputOutput:
      ++stats->filtered_dependency_output_output;
      break;
    case GemmDependencyKind::kOutputLhs:
      ++stats->filtered_dependency_output_lhs;
      break;
    case GemmDependencyKind::kOutputRhs:
      ++stats->filtered_dependency_output_rhs;
      break;
    case GemmDependencyKind::kNone:
      break;
  }
}

bool SlicesOverlap(const BufferAllocation::Slice& a,
                   const BufferAllocation::Slice& b) {
  return a.OverlapsWith(b);
}

GemmDependencyKind FindInterGemmDependency(
    absl::Span<const GemmThunk* const> gemms) {
  for (int64_t i = 0; i < gemms.size(); ++i) {
    const BufferAllocation::Slice& output = gemms[i]->output_buffer();
    for (int64_t j = 0; j < gemms.size(); ++j) {
      if (i != j && output.OverlapsWith(gemms[j]->output_buffer())) {
        return GemmDependencyKind::kOutputOutput;
      }
      if (i != j && SlicesOverlap(output, gemms[j]->lhs_buffer())) {
        return GemmDependencyKind::kOutputLhs;
      }
      if (i != j && SlicesOverlap(output, gemms[j]->rhs_buffer())) {
        return GemmDependencyKind::kOutputRhs;
      }
    }
  }
  return GemmDependencyKind::kNone;
}

bool SameThunkForBatching(const GemmThunk* a, const GemmThunk* b) {
  return a->deterministic() == b->deterministic() &&
         SameGemmConfigForBatching(a->config(), b->config());
}

bool SliceLess(const BufferAllocation::Slice& a,
               const BufferAllocation::Slice& b) {
  if (a.index() != b.index()) {
    return a.index() < b.index();
  }
  if (a.offset() != b.offset()) {
    return a.offset() < b.offset();
  }
  return a.size() < b.size();
}

std::vector<const GemmThunk*> GemmsForPositions(
    absl::Span<GemmThunk* const> run_gemms,
    const std::vector<int64_t>& positions) {
  std::vector<const GemmThunk*> gemms;
  gemms.reserve(positions.size());
  for (int64_t position : positions) {
    gemms.push_back(run_gemms[position]);
  }
  return gemms;
}

bool IsContiguousInOriginalOrder(const std::vector<int64_t>& positions) {
  if (positions.empty()) {
    return true;
  }
  std::vector<int64_t> sorted = positions;
  std::sort(sorted.begin(), sorted.end());
  for (int64_t i = 1; i < sorted.size(); ++i) {
    if (sorted[i] != sorted[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

GemmDependencyKind FindMoveDependencyWithinRun(
    absl::Span<GemmThunk* const> run_gemms,
    const std::vector<int64_t>& positions) {
  if (positions.empty()) {
    return GemmDependencyKind::kOutputOutput;
  }
  std::vector<bool> selected(run_gemms.size(), false);
  int64_t first = run_gemms.size();
  int64_t last = -1;
  for (int64_t position : positions) {
    selected[position] = true;
    first = std::min(first, position);
    last = std::max(last, position);
  }

  std::vector<const GemmThunk*> selected_gemms =
      GemmsForPositions(run_gemms, positions);
  GemmDependencyKind dependency = FindInterGemmDependency(selected_gemms);
  if (dependency != GemmDependencyKind::kNone) {
    return dependency;
  }

  for (int64_t i = first; i <= last; ++i) {
    if (selected[i]) {
      continue;
    }
    const GemmThunk* other = run_gemms[i];
    for (int64_t position : positions) {
      const GemmThunk* gemm = run_gemms[position];
      const BufferAllocation::Slice& selected_output = gemm->output_buffer();
      const BufferAllocation::Slice& other_output = other->output_buffer();
      const bool selected_moves_before_other = position > i;
      if (selected_moves_before_other &&
          selected_output.OverlapsWith(other_output)) {
        return GemmDependencyKind::kOutputOutput;
      }
      if (selected_moves_before_other &&
          selected_output.OverlapsWith(other->lhs_buffer())) {
        return GemmDependencyKind::kOutputLhs;
      }
      if (selected_moves_before_other &&
          selected_output.OverlapsWith(other->rhs_buffer())) {
        return GemmDependencyKind::kOutputRhs;
      }
      if (selected_moves_before_other &&
          other_output.OverlapsWith(gemm->lhs_buffer())) {
        return GemmDependencyKind::kOutputLhs;
      }
      if (selected_moves_before_other &&
          other_output.OverlapsWith(gemm->rhs_buffer())) {
        return GemmDependencyKind::kOutputRhs;
      }
    }
  }
  return GemmDependencyKind::kNone;
}

bool ComputeBatchStride(absl::Span<const GemmThunk* const> gemms,
                        const BufferAllocation::Slice& (GemmThunk::*buffer)()
                            const,
                        PrimitiveType dtype, bool allow_broadcast,
                        int64_t* batch_stride) {
  const BufferAllocation::Slice& first = (gemms[0]->*buffer)();
  const int64_t byte_width = primitive_util::ByteWidth(dtype);
  int64_t stride_bytes = 0;
  bool found_distinct = false;
  for (int64_t i = 1; i < gemms.size(); ++i) {
    const BufferAllocation::Slice& current = (gemms[i]->*buffer)();
    if (current == first) {
      continue;
    }
    if (current.index() != first.index()) {
      return false;
    }
    const int64_t current_stride = current.offset() - first.offset();
    if (current_stride <= 0 || current_stride % i != 0) {
      return false;
    }
    const int64_t per_batch_stride = current_stride / i;
    if (per_batch_stride % byte_width != 0) {
      return false;
    }
    if (!found_distinct) {
      stride_bytes = per_batch_stride;
      found_distinct = true;
    } else if (stride_bytes != per_batch_stride) {
      return false;
    }
    if (current.size() != first.size()) {
      return false;
    }
  }
  if (!found_distinct) {
    if (!allow_broadcast) {
      return false;
    }
    *batch_stride = 0;
    return true;
  }
  for (int64_t i = 1; i < gemms.size(); ++i) {
    const BufferAllocation::Slice& current = (gemms[i]->*buffer)();
    if (current.index() != first.index() ||
        current.offset() != first.offset() + i * stride_bytes ||
        current.size() != first.size()) {
      return false;
    }
  }
  *batch_stride = stride_bytes / byte_width;
  return true;
}

StatusOr<GemmConfig> TryBuildStridedBatchedConfig(
    absl::Span<const GemmThunk* const> gemms,
    MusaGroupedGemmRewriteStats* stats) {
  if (gemms.empty()) {
    return absl::InvalidArgumentError("empty GEMM group");
  }
  const GemmConfig& first = gemms[0]->config();
  const bool deterministic = gemms[0]->deterministic();
  for (int64_t i = 1; i < gemms.size(); ++i) {
    if (gemms[i]->deterministic() != deterministic ||
        !SameGemmConfigForBatching(first, gemms[i]->config())) {
      ++stats->filtered_config;
      return absl::InvalidArgumentError("incompatible GEMM config");
    }
  }
  GemmDependencyKind dependency = FindInterGemmDependency(gemms);
  if (dependency != GemmDependencyKind::kNone) {
    RecordDependency(dependency, stats);
    return absl::InvalidArgumentError("GEMM group has dependency");
  }

  GemmConfig batched = first;
  int64_t lhs_stride = 0;
  int64_t rhs_stride = 0;
  int64_t output_stride = 0;
  bool lhs_strided =
      ComputeBatchStride(gemms, &GemmThunk::lhs_buffer, first.lhs_layout.dtype,
                         /*allow_broadcast=*/true, &lhs_stride);
  bool rhs_strided =
      ComputeBatchStride(gemms, &GemmThunk::rhs_buffer, first.rhs_layout.dtype,
                         /*allow_broadcast=*/true, &rhs_stride);
  bool output_strided = ComputeBatchStride(
      gemms, &GemmThunk::output_buffer, first.output_layout.dtype,
      /*allow_broadcast=*/false, &output_stride);
  if (!lhs_strided || !rhs_strided || !output_strided) {
    ++stats->filtered_not_strided;
    if (!lhs_strided) {
      ++stats->filtered_not_strided_lhs;
    }
    if (!rhs_strided) {
      ++stats->filtered_not_strided_rhs;
    }
    if (!output_strided) {
      ++stats->filtered_not_strided_output;
    }
    return absl::InvalidArgumentError("GEMM group is not strided");
  }

  batched.lhs_layout.batch_size = gemms.size();
  batched.rhs_layout.batch_size = gemms.size();
  batched.output_layout.batch_size = gemms.size();
  batched.c_layout.batch_size = gemms.size();
  batched.lhs_layout.batch_stride = lhs_stride;
  batched.rhs_layout.batch_stride = rhs_stride;
  batched.output_layout.batch_stride = output_stride;
  batched.c_layout.batch_stride = output_stride;
  return batched;
}

struct PlannedGemmGroup {
  std::vector<int64_t> positions;
  GemmConfig batched_config;
  bool noncontiguous = false;
};

std::string SeparatorOpName(const Thunk& thunk) {
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
}

bool ThunkTouchesGemmBuffer(const Thunk& thunk, const GemmThunk& gemm) {
  for (const BufferAllocation::Slice& arg : thunk.buffer_arguments()) {
    if (arg.OverlapsWith(gemm.lhs_buffer()) ||
        arg.OverlapsWith(gemm.rhs_buffer()) ||
        arg.OverlapsWith(gemm.output_buffer())) {
      return true;
    }
  }
  return false;
}

bool HasCrossKernelSeparatorDependency(
    const ThunkSequence& sequence, const std::vector<int64_t>& positions) {
  if (positions.empty()) {
    return true;
  }
  std::vector<int64_t> sorted = positions;
  std::sort(sorted.begin(), sorted.end());
  const int64_t first = sorted.front();
  const int64_t last = sorted.back();
  for (int64_t i = first; i <= last; ++i) {
    if (std::binary_search(sorted.begin(), sorted.end(), i)) {
      continue;
    }
    if (sequence[i]->kind() != Thunk::Kind::kKernel) {
      return true;
    }
    for (int64_t position : sorted) {
      const GemmThunk* gemm = sequence[position]->AsGemmThunk();
      if (gemm != nullptr && ThunkTouchesGemmBuffer(*sequence[i], *gemm)) {
        return true;
      }
    }
  }
  return false;
}

std::vector<const GemmThunk*> GemmsForAbsolutePositions(
    const ThunkSequence& sequence, const std::vector<int64_t>& positions) {
  std::vector<const GemmThunk*> gemms;
  gemms.reserve(positions.size());
  for (int64_t position : positions) {
    gemms.push_back(sequence[position]->AsGemmThunk());
  }
  return gemms;
}

void MaybeLogCrossKernelGemmDiagnostics(
    const ThunkSequence& sequence, int64_t min_group_size,
    int64_t max_group_size, MusaGroupedGemmRewriteStats* stats) {
  if (!IsTruthyEnv("MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG")) {
    return;
  }
  const int64_t max_separator_kernels = std::max<int64_t>(
      1, ReadInt64Env(
             "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_MAX_SEPARATORS", 8));
  absl::flat_hash_map<std::string, int64_t> separator_ops;
  absl::flat_hash_map<std::string, int64_t> blocked_reasons;

  for (int64_t start = 0; start < sequence.size(); ++start) {
    const GemmThunk* seed = sequence[start]->AsGemmThunk();
    if (seed == nullptr) {
      continue;
    }
    std::vector<int64_t> window_gemms;
    int64_t separator_count = 0;
    for (int64_t i = start; i < sequence.size(); ++i) {
      const GemmThunk* gemm = sequence[i]->AsGemmThunk();
      if (gemm != nullptr) {
        window_gemms.push_back(i);
        if (window_gemms.size() >= max_group_size) {
          break;
        }
        continue;
      }
      if (sequence[i]->kind() != Thunk::Kind::kKernel) {
        break;
      }
      ++separator_count;
      ++separator_ops[SeparatorOpName(*sequence[i])];
      if (separator_count > max_separator_kernels) {
        break;
      }
    }
    if (window_gemms.size() < min_group_size) {
      continue;
    }

    std::vector<int64_t> candidates;
    candidates.reserve(window_gemms.size());
    for (int64_t position : window_gemms) {
      const GemmThunk* gemm = sequence[position]->AsGemmThunk();
      if (SameThunkForBatching(seed, gemm)) {
        candidates.push_back(position);
      }
    }
    if (candidates.size() < min_group_size) {
      continue;
    }
    if (candidates.size() > max_group_size) {
      candidates.resize(max_group_size);
    }
    ++stats->cross_kernel_candidate_windows;
    stats->cross_kernel_candidate_gemms += candidates.size();

    if (HasCrossKernelSeparatorDependency(sequence, candidates)) {
      ++stats->cross_kernel_blocked_by_dependency;
      ++blocked_reasons["separator_dependency"];
      continue;
    }

    MusaGroupedGemmRewriteStats scratch;
    std::vector<const GemmThunk*> gemms =
        GemmsForAbsolutePositions(sequence, candidates);
    auto config = TryBuildStridedBatchedConfig(gemms, &scratch);
    if (config.ok()) {
      ++stats->cross_kernel_groupable_windows;
      stats->cross_kernel_groupable_gemms += candidates.size();
      continue;
    }
    if (scratch.filtered_config > 0) {
      ++stats->cross_kernel_blocked_by_config;
      ++blocked_reasons["config"];
    } else if (scratch.filtered_not_strided > 0) {
      ++stats->cross_kernel_blocked_by_stride;
      ++blocked_reasons["stride"];
    } else {
      ++stats->cross_kernel_blocked_by_dependency;
      ++blocked_reasons["gemm_dependency"];
    }
  }

  LOG(INFO) << "[MUSA_XLA_GROUP_GEMM_CROSS_KERNEL_DIAG] "
            << "candidate_windows=" << stats->cross_kernel_candidate_windows
            << " candidate_gemms=" << stats->cross_kernel_candidate_gemms
            << " groupable_windows=" << stats->cross_kernel_groupable_windows
            << " groupable_gemms=" << stats->cross_kernel_groupable_gemms
            << " estimated_launch_reduction="
            << (stats->cross_kernel_groupable_gemms -
                stats->cross_kernel_groupable_windows)
            << " blocked_by_dependency="
            << stats->cross_kernel_blocked_by_dependency
            << " blocked_by_config=" << stats->cross_kernel_blocked_by_config
            << " blocked_by_stride=" << stats->cross_kernel_blocked_by_stride
            << " separator_ops={"
            << absl::StrJoin(TopCounts(separator_ops, 12), ",") << "}"
            << " blocked_reasons={"
            << absl::StrJoin(TopCounts(blocked_reasons, 8), ",") << "}";
}

std::optional<PlannedGemmGroup> TryBuildPlannedGroup(
    absl::Span<GemmThunk* const> run_gemms,
    const std::vector<int64_t>& positions,
    MusaGroupedGemmRewriteStats* stats) {
  std::vector<const GemmThunk*> gemms = GemmsForPositions(run_gemms, positions);
  auto batched_config = TryBuildStridedBatchedConfig(gemms, stats);
  if (!batched_config.ok()) {
    return std::nullopt;
  }
  GemmDependencyKind move_dependency =
      FindMoveDependencyWithinRun(run_gemms, positions);
  if (move_dependency != GemmDependencyKind::kNone) {
    RecordDependency(move_dependency, stats);
    return std::nullopt;
  }
  return PlannedGemmGroup{positions, std::move(batched_config).value(),
                          !IsContiguousInOriginalOrder(positions)};
}

std::optional<PlannedGemmGroup> FindGroupInOrder(
    absl::Span<GemmThunk* const> run_gemms, const std::vector<int64_t>& order,
    int64_t min_group_size, int64_t max_group_size,
    MusaGroupedGemmRewriteStats* stats) {
  for (int64_t start = 0; start + min_group_size <= order.size(); ++start) {
    const int64_t largest =
        std::min<int64_t>(max_group_size, order.size() - start);
    for (int64_t size = largest; size >= min_group_size; --size) {
      std::vector<int64_t> positions(order.begin() + start,
                                     order.begin() + start + size);
      if (auto group = TryBuildPlannedGroup(run_gemms, positions, stats)) {
        return group;
      }
    }
  }
  return std::nullopt;
}

std::optional<PlannedGemmGroup> FindNextPlannedGroup(
    absl::Span<GemmThunk* const> run_gemms, absl::Span<const uint8_t> consumed,
    int64_t min_group_size, int64_t max_group_size,
    MusaGroupedGemmRewriteStats* stats) {
  std::vector<int64_t> processed_config_seeds;
  for (int64_t seed = 0; seed < run_gemms.size(); ++seed) {
    if (consumed[seed]) {
      continue;
    }
    bool config_already_processed = false;
    for (int64_t processed_seed : processed_config_seeds) {
      if (SameThunkForBatching(run_gemms[seed], run_gemms[processed_seed])) {
        config_already_processed = true;
        break;
      }
    }
    if (config_already_processed) {
      continue;
    }
    processed_config_seeds.push_back(seed);

    std::vector<int64_t> candidates;
    candidates.reserve(run_gemms.size());
    for (int64_t i = 0; i < run_gemms.size(); ++i) {
      if (!consumed[i] && SameThunkForBatching(run_gemms[seed], run_gemms[i])) {
        candidates.push_back(i);
      }
    }
    if (candidates.size() < min_group_size) {
      continue;
    }

    if (auto group = FindGroupInOrder(run_gemms, candidates, min_group_size,
                                      max_group_size, stats)) {
      return group;
    }

    std::vector<int64_t> output_order = candidates;
    std::stable_sort(output_order.begin(), output_order.end(),
                     [&](int64_t a, int64_t b) {
                       if (SliceLess(run_gemms[a]->output_buffer(),
                                     run_gemms[b]->output_buffer())) {
                         return true;
                       }
                       if (SliceLess(run_gemms[b]->output_buffer(),
                                     run_gemms[a]->output_buffer())) {
                         return false;
                       }
                       return a < b;
                     });
    if (auto group = FindGroupInOrder(run_gemms, output_order, min_group_size,
                                      max_group_size, stats)) {
      return group;
    }

    std::vector<int64_t> lhs_order = candidates;
    std::stable_sort(lhs_order.begin(), lhs_order.end(), [&](int64_t a,
                                                             int64_t b) {
      if (SliceLess(run_gemms[a]->lhs_buffer(), run_gemms[b]->lhs_buffer())) {
        return true;
      }
      if (SliceLess(run_gemms[b]->lhs_buffer(), run_gemms[a]->lhs_buffer())) {
        return false;
      }
      return a < b;
    });
    if (auto group = FindGroupInOrder(run_gemms, lhs_order, min_group_size,
                                      max_group_size, stats)) {
      return group;
    }
  }
  return std::nullopt;
}

}  // namespace

std::vector<MusaGroupedGemmRun> PlanMusaGroupedGemmRuns(
    absl::Span<const bool> groupable, int64_t min_group_size,
    int64_t max_group_size) {
  std::vector<MusaGroupedGemmRun> runs;
  min_group_size = std::max<int64_t>(1, min_group_size);
  max_group_size = std::max<int64_t>(min_group_size, max_group_size);
  int64_t i = 0;
  while (i < groupable.size()) {
    if (!groupable[i]) {
      ++i;
      continue;
    }
    const int64_t start = i;
    while (i < groupable.size() && groupable[i]) {
      ++i;
    }
    int64_t remaining = i - start;
    int64_t offset = 0;
    while (remaining >= min_group_size) {
      const int64_t size = std::min<int64_t>(remaining, max_group_size);
      runs.push_back({start + offset, size});
      offset += size;
      remaining -= size;
    }
  }
  return runs;
}

StatusOr<std::unique_ptr<const ThunkSequence>> RewriteMusaGroupGemmThunks(
    std::unique_ptr<const ThunkSequence> sequence,
    const MusaGroupedGemmRewriteOptions& options,
    MusaGroupedGemmRewriteStats* stats) {
  stats->original_thunks = sequence == nullptr ? 0 : sequence->size();
  if (sequence == nullptr || sequence->empty()) {
    stats->rewritten_thunks = stats->original_thunks;
    return sequence;
  }

  std::unique_ptr<ThunkSequence> input(
      const_cast<ThunkSequence*>(sequence.release()));
  auto output = std::make_unique<ThunkSequence>();
  const int64_t min_group_size = std::max<int64_t>(1, options.min_group_size);
  const int64_t max_group_size =
      std::max<int64_t>(min_group_size, options.max_group_size);
  MaybeLogCrossKernelGemmDiagnostics(*input, min_group_size, max_group_size,
                                     stats);
  auto log_stats = [&] {
    if (!options.log) {
      return;
    }
    LOG(INFO) << "[MUSA_XLA_GROUP_GEMM_THUNKS] changed="
              << (stats->groups_created > 0)
              << " mode=strided_batched original_thunks="
              << stats->original_thunks
              << " rewritten_thunks=" << stats->rewritten_thunks
              << " diagnostic_only=" << options.diagnostic_only
              << " plain_gemm_thunks=" << stats->plain_gemm_thunks
              << " candidate_runs=" << stats->candidate_runs
              << " candidate_gemms=" << stats->candidate_gemms
              << " groups_created=" << stats->groups_created
              << " gemms_grouped=" << stats->gemms_grouped
              << " strided_batched_groups=" << stats->strided_batched_groups
              << " strided_batched_gemms=" << stats->strided_batched_gemms
              << " noncontiguous_groups=" << stats->noncontiguous_groups
              << " noncontiguous_gemms=" << stats->noncontiguous_gemms
              << " cross_kernel_candidate_windows="
              << stats->cross_kernel_candidate_windows
              << " cross_kernel_candidate_gemms="
              << stats->cross_kernel_candidate_gemms
              << " cross_kernel_groupable_windows="
              << stats->cross_kernel_groupable_windows
              << " cross_kernel_groupable_gemms="
              << stats->cross_kernel_groupable_gemms
              << " cross_kernel_blocked_by_dependency="
              << stats->cross_kernel_blocked_by_dependency
              << " cross_kernel_blocked_by_config="
              << stats->cross_kernel_blocked_by_config
              << " cross_kernel_blocked_by_stride="
              << stats->cross_kernel_blocked_by_stride
              << " estimated_launch_reduction="
              << (stats->gemms_grouped - stats->groups_created)
              << " filtered_not_strided=" << stats->filtered_not_strided
              << " filtered_not_strided_lhs="
              << stats->filtered_not_strided_lhs
              << " filtered_not_strided_rhs="
              << stats->filtered_not_strided_rhs
              << " filtered_not_strided_output="
              << stats->filtered_not_strided_output
              << " filtered_dependency=" << stats->filtered_dependency
              << " filtered_dependency_output_output="
              << stats->filtered_dependency_output_output
              << " filtered_dependency_output_lhs="
              << stats->filtered_dependency_output_lhs
              << " filtered_dependency_output_rhs="
              << stats->filtered_dependency_output_rhs
              << " filtered_config=" << stats->filtered_config;
  };
  if (options.diagnostic_only) {
    stats->rewritten_thunks = input->size();
    log_stats();
    return std::unique_ptr<const ThunkSequence>(input.release());
  }

  for (int64_t i = 0; i < input->size();) {
    GemmThunk* first_gemm = (*input)[i]->AsGemmThunk();
    if (first_gemm == nullptr) {
      output->push_back(std::move((*input)[i]));
      ++i;
      continue;
    }

    const int64_t run_start = i;
    while (i < input->size() && (*input)[i]->AsGemmThunk() != nullptr) {
      ++stats->plain_gemm_thunks;
      ++stats->gemm_thunks;
      ++i;
    }
    int64_t run_end = i;
    const int64_t run_size = run_end - run_start;
    if (run_size < min_group_size) {
      for (int64_t j = run_start; j < run_end; ++j) {
        output->push_back(std::move((*input)[j]));
      }
      continue;
    }
    ++stats->candidate_runs;
    stats->candidate_gemms += run_size;

    std::vector<GemmThunk*> run_gemms;
    run_gemms.reserve(run_size);
    for (int64_t j = run_start; j < run_end; ++j) {
      run_gemms.push_back((*input)[j]->AsGemmThunk());
    }

    std::vector<uint8_t> consumed(run_size, 0);
    std::vector<PlannedGemmGroup> planned_groups;
    while (auto group =
               FindNextPlannedGroup(run_gemms, consumed, min_group_size,
                                    max_group_size, stats)) {
      for (int64_t position : group->positions) {
        consumed[position] = 1;
      }
      planned_groups.push_back(std::move(*group));
    }

    std::vector<int64_t> emit_group(run_size, -1);
    std::vector<int64_t> member_group(run_size, -1);
    for (int64_t group_id = 0; group_id < planned_groups.size(); ++group_id) {
      const std::vector<int64_t>& positions = planned_groups[group_id].positions;
      const int64_t emit_position =
          *std::min_element(positions.begin(), positions.end());
      emit_group[emit_position] = group_id;
      for (int64_t position : positions) {
        member_group[position] = group_id;
      }
    }

    for (int64_t local = 0; local < run_size; ++local) {
      if (emit_group[local] >= 0) {
        PlannedGemmGroup& group = planned_groups[emit_group[local]];
        std::vector<const GemmThunk*> gemms =
            GemmsForPositions(run_gemms, group.positions);
        const int64_t size = gemms.size();
        Thunk::ThunkInfo info((*input)[run_start + local]->op());
        info.profile_annotation =
            absl::StrCat("musa_grouped_strided_gemm(count=", size, ")");
        output->push_back(std::make_unique<MusaGroupedGemmThunk>(
            info, std::move(group.batched_config), gemms[0]->lhs_buffer(),
            gemms[0]->rhs_buffer(), gemms[0]->output_buffer(), size,
            gemms[0]->deterministic()));
        for (int64_t position : group.positions) {
          (*input)[run_start + position].reset();
        }
        ++stats->groups_created;
        ++stats->strided_batched_groups;
        stats->gemms_grouped += size;
        stats->strided_batched_gemms += size;
        if (group.noncontiguous) {
          ++stats->noncontiguous_groups;
          stats->noncontiguous_gemms += size;
        }
        continue;
      }
      if (member_group[local] >= 0) {
        continue;
      }
      output->push_back(std::move((*input)[run_start + local]));
    }
  }

  stats->rewritten_thunks = output->size();
  log_stats();
  return std::unique_ptr<const ThunkSequence>(output.release());
}

MusaGroupedGemmThunk::MusaGroupedGemmThunk(
    ThunkInfo thunk_info, GemmConfig batched_config,
    const BufferAllocation::Slice& lhs_buffer,
    const BufferAllocation::Slice& rhs_buffer,
    const BufferAllocation::Slice& output_buffer, int64_t gemm_count,
    bool deterministic)
    : Thunk(Kind::kGemm, thunk_info),
      batched_config_(std::move(batched_config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      gemm_count_(gemm_count),
      deterministic_(deterministic) {}

Status MusaGroupedGemmThunk::ExecuteOnStream(const ExecuteParams& params) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  Status status =
      RunGemm(batched_config_, allocs.GetDeviceAddress(lhs_buffer_),
              allocs.GetDeviceAddress(rhs_buffer_),
              allocs.GetDeviceAddress(output_buffer_), deterministic_,
              params.stream);
  if (!status.ok()) {
    LOG(ERROR) << "[MUSA_GROUPED_GEMM_THUNK] stage=run_strided_batched_failed "
               << "gemm_count=" << gemm_count_ << " status=" << status;
  }
  return status;
}

Status MusaGroupedGemmThunk::Initialize(const GpuExecutable& /*executable*/,
                                        se::StreamExecutor* executor) {
  if (!executor->AsBlas()) {
    LOG(ERROR) << "[MUSA_GROUPED_GEMM_THUNK] stage=initialize_failed "
               << "reason=no_blas";
    return absl::InternalError("Failed to initialize BLAS support");
  }
  return OkStatus();
}

std::string MusaGroupedGemmThunk::ToStringExtra(int /*indent*/) const {
  return absl::StrCat("mode=strided_batched, gemm_count=", gemm_count_,
                      ", m=", batched_config_.output_layout.num_rows,
                      ", n=", batched_config_.output_layout.num_cols,
                      ", k=", batched_config_.lhs_layout.num_cols,
                      ", lhs_stride=", batched_config_.lhs_layout.batch_stride,
                      ", rhs_stride=", batched_config_.rhs_layout.batch_stride,
                      ", out_stride=",
                      batched_config_.output_layout.batch_stride);
}

MusaPointerArrayGemmThunk::MusaPointerArrayGemmThunk(
    ThunkInfo thunk_info, GemmConfig config,
    std::vector<BufferAllocation::Slice> lhs_buffers,
    std::vector<BufferAllocation::Slice> rhs_buffers,
    std::vector<BufferAllocation::Slice> output_buffers, bool deterministic)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffers_(std::move(lhs_buffers)),
      rhs_buffers_(std::move(rhs_buffers)),
      output_buffers_(std::move(output_buffers)),
      deterministic_(deterministic) {}

Status MusaPointerArrayGemmThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  if (lhs_buffers_.size() != rhs_buffers_.size() ||
      lhs_buffers_.size() != output_buffers_.size() || lhs_buffers_.empty()) {
    return absl::InvalidArgumentError(
        "MUSA pointer-array GEMM requires equal non-empty buffer arrays");
  }
  if (config_.lhs_layout.dtype != F32 || config_.rhs_layout.dtype != F32 ||
      config_.output_layout.dtype != F32 || config_.beta != 0.0 ||
      config_.alpha.imag() != 0.0) {
    return absl::UnimplementedError(
        "MUSA pointer-array GEMM currently supports f32 alpha-real beta-zero");
  }

  MatrixLayout lhs_layout = config_.lhs_layout;
  MatrixLayout rhs_layout = config_.rhs_layout;
  MatrixLayout output_layout = config_.output_layout;
  bool swap_operands = output_layout.order != MatrixLayout::Order::kColumnMajor;
  if (swap_operands) {
    std::swap(lhs_layout, rhs_layout);
    lhs_layout.Transpose();
    rhs_layout.Transpose();
    output_layout.Transpose();
  }

  const auto& lhs_slices = swap_operands ? rhs_buffers_ : lhs_buffers_;
  const auto& rhs_slices = swap_operands ? lhs_buffers_ : rhs_buffers_;
  const BufferAllocations& allocations = *params.buffer_allocations;
  std::vector<se::DeviceMemory<float>> lhs_memory;
  std::vector<se::DeviceMemory<float>> rhs_memory;
  std::vector<se::DeviceMemory<float>> output_memory;
  std::vector<se::DeviceMemory<float>*> lhs_ptrs;
  std::vector<se::DeviceMemory<float>*> rhs_ptrs;
  std::vector<se::DeviceMemory<float>*> output_ptrs;
  const int64_t count = lhs_buffers_.size();
  lhs_memory.reserve(count);
  rhs_memory.reserve(count);
  output_memory.reserve(count);
  lhs_ptrs.reserve(count);
  rhs_ptrs.reserve(count);
  output_ptrs.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    lhs_memory.emplace_back(allocations.GetDeviceAddress(lhs_slices[i]));
    rhs_memory.emplace_back(allocations.GetDeviceAddress(rhs_slices[i]));
    output_memory.emplace_back(
        allocations.GetDeviceAddress(output_buffers_[i]));
  }
  for (int64_t i = 0; i < count; ++i) {
    lhs_ptrs.push_back(&lhs_memory[i]);
    rhs_ptrs.push_back(&rhs_memory[i]);
    output_ptrs.push_back(&output_memory[i]);
  }

  auto as_blas_transpose = [](MatrixLayout::Order order) {
    return order == MatrixLayout::Order::kColumnMajor
               ? se::blas::Transpose::kNoTranspose
               : se::blas::Transpose::kTranspose;
  };
  se::NumericOptions numeric_options{
      deterministic_, /*allow_tf32=*/config_.compute_precision <= 1};
  if (IsTruthyEnv("MUSA_BLAS_GEMM_DIAGNOSTICS")) {
    LOG(INFO) << "[MUSA_POINTER_ARRAY_GEMM_THUNK]"
              << " count=" << count
              << " m=" << output_layout.num_rows
              << " n=" << output_layout.num_cols
              << " k=" << lhs_layout.num_cols
              << " lhs_order=" << static_cast<int>(lhs_layout.order)
              << " rhs_order=" << static_cast<int>(rhs_layout.order)
              << " out_order=" << static_cast<int>(output_layout.order)
              << " lda=" << lhs_layout.leading_dim_stride
              << " ldb=" << rhs_layout.leading_dim_stride
              << " ldc=" << output_layout.leading_dim_stride
              << " allow_tf32=" << numeric_options.allow_tf32;
  }
  params.stream->ThenBlasGemmBatched(
      as_blas_transpose(lhs_layout.order),
      as_blas_transpose(rhs_layout.order), output_layout.num_rows,
      output_layout.num_cols, lhs_layout.num_cols,
      static_cast<float>(config_.alpha.real()), lhs_ptrs,
      lhs_layout.leading_dim_stride, rhs_ptrs, rhs_layout.leading_dim_stride,
      /*beta=*/0.0f, output_ptrs, output_layout.leading_dim_stride, count,
      numeric_options);
  if (!params.stream->ok()) {
    return absl::InternalError("MUSA pointer-array GEMM launch failed");
  }
  return OkStatus();
}

Status MusaPointerArrayGemmThunk::Initialize(
    const GpuExecutable& /*executable*/, se::StreamExecutor* executor) {
  if (!executor->AsBlas()) {
    return absl::InternalError("Failed to initialize MUSA BLAS support");
  }
  return OkStatus();
}

std::string MusaPointerArrayGemmThunk::ToStringExtra(int /*indent*/) const {
  return absl::StrCat("mode=pointer_array, gemm_count=", lhs_buffers_.size(),
                      ", m=", config_.output_layout.num_rows,
                      ", n=", config_.output_layout.num_cols,
                      ", k=", config_.lhs_layout.num_cols);
}

}  // namespace gpu
}  // namespace xla
