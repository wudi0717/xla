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

#ifndef XLA_SERVICE_GPU_MUSA_REDUCTION_CHAIN_H_
#define XLA_SERVICE_GPU_MUSA_REDUCTION_CHAIN_H_

#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

struct MusaReductionChainMatch {
  const HloInstruction* first_reduce;
  const HloInstruction* first_data;
  const HloInstruction* second_reduce;
  const HloInstruction* second_data;
  const HloInstruction* root;
  int64_t rows;
  int64_t width;
};

// Returns true only when every dependency path from external_operand to
// expression preserves the flattened leading row index.
bool MusaFusionExpressionPreservesRows(
    const HloInstruction* fusion, const HloInstruction* expression,
    const HloInstruction* external_operand, int64_t rows);

std::optional<MusaReductionChainMatch> MatchMusaReductionChainFusion(
    const HloInstruction* fusion, const HloComputation* called);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MUSA_REDUCTION_CHAIN_H_
