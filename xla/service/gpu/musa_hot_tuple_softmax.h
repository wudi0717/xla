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

#ifndef XLA_SERVICE_GPU_MUSA_HOT_TUPLE_SOFTMAX_H_
#define XLA_SERVICE_GPU_MUSA_HOT_TUPLE_SOFTMAX_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

struct MusaHotTupleSoftmaxGroupMatch {
  const HloInstruction* data;
  const HloInstruction* sum_reduce;
  const HloInstruction* max_reduce;
  // Number of logical rows, including every non-reduction dimension.
  int64_t rows;
  int64_t width;
};

struct MusaHotTupleSoftmaxMatch {
  std::vector<MusaHotTupleSoftmaxGroupMatch> groups;
  std::string root_order;
};

std::optional<MusaHotTupleSoftmaxMatch> MatchMusaHotTupleSoftmaxFusion(
    const HloInstruction* fusion, const HloComputation* called);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MUSA_HOT_TUPLE_SOFTMAX_H_
