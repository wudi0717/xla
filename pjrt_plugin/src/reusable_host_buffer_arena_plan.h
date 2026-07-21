#ifndef PJRT_PLUGIN_SRC_REUSABLE_HOST_BUFFER_ARENA_PLAN_H_
#define PJRT_PLUGIN_SRC_REUSABLE_HOST_BUFFER_ARENA_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace musa::pjrt {

struct ReusableHostBufferArenaEntryOrder {
    uintptr_t host_key = 0;
    size_t pool_sequence = 0;
    size_t original_index = 0;
};

std::vector<size_t> OrderReusableHostBufferArenaEntries(
    std::vector<ReusableHostBufferArenaEntryOrder> entries,
    bool pool_order);

struct ReusableHostBufferDirtyRange {
    size_t offset = 0;
    size_t bytes = 0;
};

struct ReusableHostBufferArenaCopyPlan {
    bool copy_full_arena = false;
    size_t transferred_bytes = 0;
    std::vector<ReusableHostBufferDirtyRange> ranges;
};

ReusableHostBufferArenaCopyPlan PlanReusableHostBufferArenaCopies(
    size_t arena_bytes,
    std::vector<ReusableHostBufferDirtyRange> dirty_ranges,
    size_t max_ranges, size_t merge_gap_bytes,
    size_t per_copy_overhead_bytes);

}  // namespace musa::pjrt

#endif  // PJRT_PLUGIN_SRC_REUSABLE_HOST_BUFFER_ARENA_PLAN_H_
