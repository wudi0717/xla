/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COLLECTIVE_PERMUTE_CYCLE_H_
#define XLA_SERVICE_COLLECTIVE_PERMUTE_CYCLE_H_

#include <utility>

#include "xla/service/source_target_pairs.h"

namespace xla {
namespace collective_permute_cycle {

enum class CycleType { kNone, kForward, kBackward };

std::pair<SourceTargetPairs, SourceTargetPairs> SplitEdges(
    const SourceTargetPairs& pairs, CycleType cycle_type);

CycleType GetCycleType(const SourceTargetPairs& pairs);
bool IsForwardCycle(const SourceTargetPairs& pairs);
bool IsBackwardCycle(const SourceTargetPairs& pairs);
bool HasCycles(const SourceTargetPairs& pairs);

bool IsForwardCycle(const SourceTargetPairs& backedge,
                    const SourceTargetPairs& others);
bool IsBackwardCycle(const SourceTargetPairs& backedge,
                     const SourceTargetPairs& others);

}  // namespace collective_permute_cycle
}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_PERMUTE_CYCLE_H_
