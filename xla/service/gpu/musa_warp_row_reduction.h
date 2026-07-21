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

#ifndef XLA_SERVICE_GPU_MUSA_WARP_ROW_REDUCTION_H_
#define XLA_SERVICE_GPU_MUSA_WARP_ROW_REDUCTION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

enum class MusaWarpRowReductionKind { kAdd, kMultiply };

struct MusaWarpRowReductionMatch {
  const HloInstruction* reduce;
  const HloInstruction* data;
  int64_t rows;
  int64_t width;
  MusaWarpRowReductionKind kind;
};

struct MusaTupleWarpRowReductionMatch {
  std::vector<MusaWarpRowReductionMatch> reductions;
  int64_t rows;
  int64_t width;
};

struct MusaIndexedWarpRowReductionMatch {
  int64_t tuple_index;
  MusaWarpRowReductionMatch reduction;
};

struct MusaTupleElementwiseOutputMatch {
  int64_t tuple_index;
  const HloInstruction* instruction;
};

struct MusaMixedTupleWarpRowReductionMatch {
  std::vector<MusaIndexedWarpRowReductionMatch> reductions;
  std::vector<MusaTupleElementwiseOutputMatch> elementwise_outputs;
  int64_t rows;
  int64_t width;
};

struct MusaWarpRowReductionLaunchConfig {
  int64_t threads_per_block;
  int64_t warps_per_block;
  int64_t elements_per_thread;
};

std::optional<MusaWarpRowReductionLaunchConfig>
ResolveMusaWarpRowReductionLaunchConfig(
    int64_t width, int64_t warp_size, int64_t threads_per_block_limit,
    int64_t requested_threads_per_block);

std::optional<MusaWarpRowReductionMatch> MatchMusaWarpRowReductionFusion(
    const HloInstruction* fusion, const HloComputation* called);

std::optional<MusaTupleWarpRowReductionMatch>
MatchMusaTupleWarpRowReductionFusion(const HloInstruction* fusion,
                                     const HloComputation* called);

std::optional<MusaMixedTupleWarpRowReductionMatch>
MatchMusaMixedTupleWarpRowReductionFusion(const HloInstruction* fusion,
                                          const HloComputation* called,
                                          std::string* failure_reason = nullptr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MUSA_WARP_ROW_REDUCTION_H_
