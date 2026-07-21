#include "pjrt_plugin/src/reusable_host_buffer_arena_plan.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace musa::pjrt {
namespace {

size_t SaturatingAdd(size_t lhs, size_t rhs) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return std::numeric_limits<size_t>::max();
    }
    return lhs + rhs;
}

ReusableHostBufferArenaCopyPlan FullArenaPlan(size_t arena_bytes) {
    ReusableHostBufferArenaCopyPlan plan;
    plan.copy_full_arena = true;
    plan.transferred_bytes = arena_bytes;
    if (arena_bytes > 0) plan.ranges.push_back({0, arena_bytes});
    return plan;
}

}  // namespace

std::vector<size_t> OrderReusableHostBufferArenaEntries(
    std::vector<ReusableHostBufferArenaEntryOrder> entries,
    bool pool_order) {
    std::sort(entries.begin(), entries.end(),
              [pool_order](const ReusableHostBufferArenaEntryOrder& lhs,
                           const ReusableHostBufferArenaEntryOrder& rhs) {
                  if (pool_order && lhs.pool_sequence != rhs.pool_sequence) {
                      return lhs.pool_sequence < rhs.pool_sequence;
                  }
                  if (lhs.host_key != rhs.host_key) {
                      return lhs.host_key < rhs.host_key;
                  }
                  return lhs.original_index < rhs.original_index;
              });

    std::vector<size_t> order;
    order.reserve(entries.size());
    for (const ReusableHostBufferArenaEntryOrder& entry : entries) {
        order.push_back(entry.original_index);
    }
    return order;
}

ReusableHostBufferArenaCopyPlan PlanReusableHostBufferArenaCopies(
    size_t arena_bytes,
    std::vector<ReusableHostBufferDirtyRange> dirty_ranges,
    size_t max_ranges, size_t merge_gap_bytes,
    size_t per_copy_overhead_bytes) {
    ReusableHostBufferArenaCopyPlan plan;
    if (arena_bytes == 0) return plan;

    std::vector<ReusableHostBufferDirtyRange> normalized;
    normalized.reserve(dirty_ranges.size());
    for (const ReusableHostBufferDirtyRange& range : dirty_ranges) {
        if (range.bytes == 0 || range.offset >= arena_bytes) continue;
        const size_t bytes = std::min(range.bytes, arena_bytes - range.offset);
        normalized.push_back({range.offset, bytes});
    }
    if (normalized.empty()) return plan;

    std::sort(normalized.begin(), normalized.end(),
              [](const ReusableHostBufferDirtyRange& lhs,
                 const ReusableHostBufferDirtyRange& rhs) {
                  return lhs.offset < rhs.offset;
              });

    plan.ranges.reserve(normalized.size());
    for (const ReusableHostBufferDirtyRange& range : normalized) {
        if (plan.ranges.empty()) {
            plan.ranges.push_back(range);
            continue;
        }

        ReusableHostBufferDirtyRange& previous = plan.ranges.back();
        const size_t previous_end = previous.offset + previous.bytes;
        const size_t range_end = range.offset + range.bytes;
        const bool overlaps = range.offset <= previous_end;
        const size_t gap = overlaps ? 0 : range.offset - previous_end;
        if (overlaps || gap <= merge_gap_bytes) {
            previous.bytes = std::max(previous_end, range_end) - previous.offset;
        } else {
            plan.ranges.push_back(range);
        }
    }

    if (max_ranges == 0 || plan.ranges.size() > max_ranges) {
        return FullArenaPlan(arena_bytes);
    }

    for (const ReusableHostBufferDirtyRange& range : plan.ranges) {
        plan.transferred_bytes =
            SaturatingAdd(plan.transferred_bytes, range.bytes);
    }
    const size_t range_overhead =
        per_copy_overhead_bytes != 0 &&
                plan.ranges.size() >
                    std::numeric_limits<size_t>::max() /
                        per_copy_overhead_bytes
            ? std::numeric_limits<size_t>::max()
            : plan.ranges.size() * per_copy_overhead_bytes;
    const size_t range_cost =
        SaturatingAdd(plan.transferred_bytes, range_overhead);
    const size_t full_cost =
        SaturatingAdd(arena_bytes, per_copy_overhead_bytes);
    if (range_cost >= full_cost) return FullArenaPlan(arena_bytes);

    return plan;
}

}  // namespace musa::pjrt
