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

#include "xla/stream_executor/musa/musa_executor.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "absl/base/casts.h"
#include "absl/numeric/int128.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "musa_runtime.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/musa/musa_command_buffer.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_kernel.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/stream_executor/musa/musa_stream.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace musa {

tsl::StatusOr<int> GetMusaPhysicalDeviceOrdinal(int visible_device_ordinal);

// ============================================================================
// MUSA BFC (Best-Fit with Chunks) Memory Allocator
// Reduces muMemAlloc/muMemFree overhead by reusing freed chunks
// ============================================================================

class MusaBFCAllocator {
 public:
  struct Chunk {
    void* ptr;
    size_t size;
    bool in_use;
    Chunk* next;
    Chunk* prev;
  };

  static constexpr size_t kAlignment = 256;
  static constexpr size_t kMinAllocationSize = 4 * 1024;  // 4KB minimum
  static constexpr size_t kDefaultPoolSize = 64 * 1024 * 1024;  // 64MB default

  MusaBFCAllocator() : pool_ptr_(nullptr), pool_size_(0),
                      free_list_(nullptr), allocated_chunks_(nullptr) {}

  ~MusaBFCAllocator() {
    std::lock_guard<std::mutex> lock(mu_);
    std::unordered_set<void*> freed_raw_ptrs;

    // Free all allocated chunks.
    Chunk* chunk = allocated_chunks_;
    while (chunk) {
      Chunk* next = chunk->next;
      auto it = aligned_to_raw_.find(chunk->ptr);
      if (it != aligned_to_raw_.end()) {
        void* raw_ptr = it->second;
        if (freed_raw_ptrs.insert(raw_ptr).second) {
          muMemFree(reinterpret_cast<MUdeviceptr>(raw_ptr));
        }
        aligned_to_raw_.erase(it);
      }
      delete chunk;
      chunk = next;
    }

    // Free all free-list chunks that came from direct allocations.
    chunk = free_list_;
    while (chunk) {
      Chunk* next = chunk->next;
      auto it = aligned_to_raw_.find(chunk->ptr);
      if (it != aligned_to_raw_.end()) {
        void* raw_ptr = it->second;
        if (freed_raw_ptrs.insert(raw_ptr).second) {
          muMemFree(reinterpret_cast<MUdeviceptr>(raw_ptr));
        }
        aligned_to_raw_.erase(it);
      }
      delete chunk;
      chunk = next;
    }

    // Free the main pool
    if (pool_ptr_) {
      muMemFree(reinterpret_cast<MUdeviceptr>(pool_ptr_));
    }
  }

  void* Allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mu_);
    if (size == 0) return nullptr;
    size = (size + kAlignment - 1) & ~(kAlignment - 1);

    // Best-fit search in free list
    Chunk* best_fit = nullptr;
    Chunk* best_prev = nullptr;
    size_t best_size = SIZE_MAX;

    Chunk* curr = free_list_;
    Chunk* curr_prev = nullptr;
    while (curr) {
      if (!curr->in_use && curr->size >= size && curr->size < best_size) {
        best_fit = curr;
        best_prev = curr_prev;
        best_size = curr->size;
      }
      curr_prev = curr;
      curr = curr->next;
    }

    if (best_fit) {
      // Remove from free list
      if (best_prev) {
        best_prev->next = best_fit->next;
      } else {
        free_list_ = best_fit->next;
      }

      // Split if there's remaining space
      size_t remaining = best_fit->size - size;
      if (remaining >= kMinAllocationSize) {
        Chunk* new_chunk = new Chunk;
        new_chunk->ptr = static_cast<char*>(best_fit->ptr) + size;
        new_chunk->size = remaining;
        new_chunk->in_use = false;
        new_chunk->next = free_list_;
        new_chunk->prev = nullptr;
        if (free_list_) free_list_->prev = new_chunk;
        free_list_ = new_chunk;
        best_fit->size = size;
      }

      best_fit->in_use = true;
      best_fit->next = allocated_chunks_;
      best_fit->prev = nullptr;
      if (allocated_chunks_) allocated_chunks_->prev = best_fit;
      allocated_chunks_ = best_fit;

      VLOG(2) << "MusaBFC alloc: " << size << " bytes, ptr=" << best_fit->ptr;
      return best_fit->ptr;
    }

    // No suitable free chunk, allocate new
    return AllocateNew(size);
  }

  bool Deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ptr) return true;

    // Find the chunk
    Chunk* chunk = allocated_chunks_;
    while (chunk) {
      if (chunk->ptr == ptr) {
        chunk->in_use = false;

        // Remove from allocated list
        if (chunk->prev) chunk->prev->next = chunk->next;
        else allocated_chunks_ = chunk->next;
        if (chunk->next) chunk->next->prev = chunk->prev;

        // Add to free list (simple add to head)
        chunk->next = free_list_;
        chunk->prev = nullptr;
        if (free_list_) free_list_->prev = chunk;
        free_list_ = chunk;

        VLOG(2) << "MusaBFC free: ptr=" << ptr << ", size=" << chunk->size;
        return true;
      }
      chunk = chunk->next;
    }
    return false;
  }

  void* ResolveRawPointer(void* ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = aligned_to_raw_.find(ptr);
    return it == aligned_to_raw_.end() ? ptr : it->second;
  }

  void ForgetAlignedPointer(void* ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    aligned_to_raw_.erase(ptr);
  }

 private:
  void AllocatePool() {
    if (pool_ptr_) return;

    auto status = musa::ToStatus(muMemAlloc(
        reinterpret_cast<MUdeviceptr*>(&pool_ptr_), pool_size_),
        "Failed to allocate BFC pool");
    if (!status.ok()) {
      LOG(ERROR) << "MusaBFC: failed to allocate pool of size " << pool_size_;
      pool_size_ = 0;
      return;
    }

    // Create initial free chunk
    Chunk* initial_chunk = new Chunk;
    initial_chunk->ptr = pool_ptr_;
    initial_chunk->size = pool_size_;
    initial_chunk->in_use = false;
    initial_chunk->next = nullptr;
    initial_chunk->prev = nullptr;
    free_list_ = initial_chunk;

    VLOG(2) << "MusaBFC: allocated pool of " << pool_size_ << " bytes at " << pool_ptr_;
  }

  void* AllocateNew(size_t size) {
    // Ensure pool exists
    if (!pool_ptr_ && pool_size_ == 0) {
      pool_size_ = kDefaultPoolSize;
      AllocatePool();
    }

    void* ptr = nullptr;
    size_t alloc_size = size + kAlignment;

    auto status = musa::ToStatus(muMemAlloc(
        reinterpret_cast<MUdeviceptr*>(&ptr), alloc_size),
        "Failed to allocate from BFC");
    if (!status.ok()) {
      LOG(ERROR) << "MusaBFC: failed to allocate " << size << " bytes";
      return nullptr;
    }

    void* aligned_ptr = reinterpret_cast<void*>(
        (reinterpret_cast<size_t>(ptr) + kAlignment - 1) & ~(kAlignment - 1));
    aligned_to_raw_[aligned_ptr] = ptr;

    Chunk* chunk = new Chunk;
    chunk->ptr = aligned_ptr;
    chunk->size = size;
    chunk->in_use = true;
    chunk->next = allocated_chunks_;
    chunk->prev = nullptr;
    if (allocated_chunks_) allocated_chunks_->prev = chunk;
    allocated_chunks_ = chunk;

    VLOG(2) << "MusaBFC: new alloc " << size << " bytes, ptr=" << aligned_ptr;
    return aligned_ptr;
  }

  void* pool_ptr_;
  size_t pool_size_;
  Chunk* free_list_;
  Chunk* allocated_chunks_;
  std::unordered_map<void*, void*> aligned_to_raw_;
  std::mutex mu_;
};

// Global BFC allocator
static std::unique_ptr<MusaBFCAllocator>* g_bfc_allocator = nullptr;
static std::mutex g_bfc_mutex;

inline MusaBFCAllocator* GetBFCAllocator() {
  std::lock_guard<std::mutex> lock(g_bfc_mutex);
  if (!g_bfc_allocator) {
    g_bfc_allocator = new std::unique_ptr<MusaBFCAllocator>(
        new MusaBFCAllocator());
  }
  return g_bfc_allocator->get();
}

// ============================================================================
// Pinned Memory Pool for Host-to-Device Transfers
// Reduces muMemHostAlloc/muMemFreeHost overhead
// ============================================================================

class PinnedMemoryPool {
 public:
  static constexpr size_t kAlignment = 4096;
  static constexpr size_t kMaxPoolSize = 256 * 1024 * 1024;  // 256MB

  struct Buffer {
    void* ptr;
    size_t size;
    bool in_use;
    Buffer* next;
  };

  PinnedMemoryPool() : free_list_(nullptr), in_use_count_(0), total_allocated_(0) {}

  ~PinnedMemoryPool() {
    Buffer* buf = free_list_;
    while (buf) {
      Buffer* next = buf->next;
      muMemFreeHost(buf->ptr);
      delete buf;
      buf = next;
    }
  }

  void* Allocate(size_t size) {
    if (size == 0) return nullptr;
    size = (size + kAlignment - 1) & ~(kAlignment - 1);

    // Search for suitable buffer in free list
    Buffer* best_fit = nullptr;
    Buffer* best_prev = nullptr;
    size_t best_size = SIZE_MAX;

    Buffer* curr = free_list_;
    Buffer* curr_prev = nullptr;
    while (curr) {
      if (curr->size >= size && curr->size < best_size) {
        best_fit = curr;
        best_prev = curr_prev;
        best_size = curr->size;
      }
      curr_prev = curr;
      curr = curr->next;
    }

    if (best_fit) {
      if (best_prev) best_prev->next = best_fit->next;
      else free_list_ = best_fit->next;
      best_fit->in_use = true;
      in_use_count_++;
      VLOG(2) << "PinnedPool: reuse " << size << " bytes, ptr=" << best_fit->ptr;
      return best_fit->ptr;
    }

    // Check pool size limit
    if (total_allocated_ + size > kMaxPoolSize) {
      void* ptr = nullptr;
      auto status = musa::ToStatus(muMemHostAlloc(&ptr, size,
          MU_MEMHOSTALLOC_PORTABLE | MU_MEMHOSTALLOC_DEVICEMAP),
          "Failed to allocate pinned memory");
      if (!status.ok() || !ptr) {
        LOG(ERROR) << "PinnedPool: failed to allocate " << size << " bytes";
        return nullptr;
      }
      VLOG(2) << "PinnedPool: direct alloc " << size << " bytes, ptr=" << ptr;
      return ptr;
    }

    void* ptr = nullptr;
    auto status = musa::ToStatus(muMemHostAlloc(&ptr, size,
        MU_MEMHOSTALLOC_PORTABLE | MU_MEMHOSTALLOC_DEVICEMAP),
        "Failed to allocate pinned memory");
    if (!status.ok() || !ptr) {
      LOG(ERROR) << "PinnedPool: failed to allocate " << size << " bytes";
      return nullptr;
    }

    Buffer* buf = new Buffer;
    buf->ptr = ptr;
    buf->size = size;
    buf->in_use = true;
    buf->next = nullptr;
    in_use_count_++;
    total_allocated_ += size;

    VLOG(2) << "PinnedPool: new " << size << " bytes, ptr=" << ptr
            << ", total=" << total_allocated_;
    return ptr;
  }

  void Deallocate(void* ptr) {
    if (!ptr) return;

    // Check if already in free list
    Buffer* curr = free_list_;
    while (curr) {
      if (curr->ptr == ptr) return;
      curr = curr->next;
    }

    // If pool is large, actually free instead of pooling
    if (total_allocated_ > kMaxPoolSize / 2) {
      VLOG(2) << "PinnedPool: direct free ptr=" << ptr;
      auto status = musa::ToStatus(muMemFreeHost(ptr),
                                   "Failed to free pinned memory");
      if (!status.ok()) {
        LOG(ERROR) << status;
      }
      return;
    }

    // Add to free list
    Buffer* buf = new Buffer;
    buf->ptr = ptr;
    buf->size = 0;
    buf->in_use = false;
    buf->next = free_list_;
    free_list_ = buf;
    in_use_count_--;

    VLOG(2) << "PinnedPool: returned to pool ptr=" << ptr;
  }

 private:
  Buffer* free_list_;
  size_t total_allocated_;
  int in_use_count_;
};

// Global pinned memory pool
static std::unique_ptr<PinnedMemoryPool>* g_pinned_pool = nullptr;
static std::mutex g_pinned_mutex;

inline PinnedMemoryPool* GetPinnedMemoryPool() {
  std::lock_guard<std::mutex> lock(g_pinned_mutex);
  if (!g_pinned_pool) {
    g_pinned_pool = new std::unique_ptr<PinnedMemoryPool>(
        new PinnedMemoryPool());
  }
  return g_pinned_pool->get();
}

namespace {

bool IsMusaDebugDeallocEnabled() {
  const char* value = std::getenv("TF_MUSA_DEBUG_DEALLOC");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool IsMusaDebugRuntimeTraceEnabled() {
  const char* value = std::getenv("TF_MUSA_DEBUG_RUNTIME_TRACE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool IsMusaPollEventTraceEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("TF_MUSA_POLL_EVENT_TRACE");
    if (value == nullptr || value[0] == '\0') {
      value = std::getenv("MUSA_POLL_EVENT_TRACE");
    }
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

uint64_t CurrentThreadIdForTrace() {
#if defined(__linux__)
  return static_cast<uint64_t>(syscall(SYS_gettid));
#else
  return static_cast<uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
}

struct PollEventTraceKey {
  uint64_t thread_id;
  uintptr_t event;

  bool operator==(const PollEventTraceKey& other) const {
    return thread_id == other.thread_id && event == other.event;
  }
};

struct PollEventTraceKeyHash {
  size_t operator()(const PollEventTraceKey& key) const {
    return std::hash<uint64_t>{}(key.thread_id) ^
           (std::hash<uintptr_t>{}(key.event) << 1);
  }
};

struct PollEventTraceStats {
  uint64_t query_count = 0;
  uint64_t not_ready_count = 0;
  std::chrono::steady_clock::time_point first_query;
};

std::string FormatMuResult(MUresult result);

struct PollEventRecordTrace {
  uint64_t thread_id = 0;
  uintptr_t stream = 0;
  uintptr_t mu_stream = 0;
  uint64_t record_count = 0;
  MUresult last_result = MUSA_SUCCESS;
  std::chrono::steady_clock::time_point last_record;
};

std::mutex& PollEventTraceMutex() {
  static auto* mu = new std::mutex;
  return *mu;
}

std::unordered_map<PollEventTraceKey, PollEventTraceStats,
                   PollEventTraceKeyHash>&
PollEventTraceStatsMap() {
  static auto* stats =
      new std::unordered_map<PollEventTraceKey, PollEventTraceStats,
                             PollEventTraceKeyHash>;
  return *stats;
}

std::unordered_map<uintptr_t, PollEventRecordTrace>&
PollEventRecordTraceMap() {
  static auto* records =
      new std::unordered_map<uintptr_t, PollEventRecordTrace>;
  return *records;
}

bool IsPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

double MillisecondsSince(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point now) {
  return std::chrono::duration<double, std::milli>(now - start).count();
}

void TracePollForEventStatus(Event* event, MUevent mu_event, MUresult result) {
  if (!IsMusaPollEventTraceEnabled()) return;

  const uint64_t thread_id = CurrentThreadIdForTrace();
  PollEventTraceKey key{thread_id, reinterpret_cast<uintptr_t>(event)};
  const auto now = std::chrono::steady_clock::now();

  std::lock_guard<std::mutex> lock(PollEventTraceMutex());
  auto& stats = PollEventTraceStatsMap();
  auto& entry = stats[key];
  if (entry.query_count == 0) {
    entry.first_query = now;
  }
  ++entry.query_count;

  const bool not_ready = result == MUSA_ERROR_NOT_READY;
  if (not_ready) {
    ++entry.not_ready_count;
  }

  const bool should_log =
      entry.query_count == 1 ||
      (not_ready && IsPowerOfTwo(entry.not_ready_count)) ||
      result != MUSA_ERROR_NOT_READY;
  if (!should_log) return;

  const double elapsed_ms = MillisecondsSince(entry.first_query, now);
  const double query_rate =
      elapsed_ms > 0.0 ? entry.query_count * 1000.0 / elapsed_ms : 0.0;
  const char* status =
      result == MUSA_SUCCESS
          ? "complete"
          : (result == MUSA_ERROR_NOT_READY ? "not_ready" : "error");
  auto record_it =
      PollEventRecordTraceMap().find(reinterpret_cast<uintptr_t>(event));
  const bool has_record = record_it != PollEventRecordTraceMap().end();
  const double record_age_ms =
      has_record ? MillisecondsSince(record_it->second.last_record, now) : -1.0;

  fprintf(stderr,
          "[musa_poll_event] tid=%llu se_event=%p mu_event=%p status=%s "
          "queries=%llu not_ready=%llu elapsed_ms=%.3f query_rate_hz=%.3f "
          "recorded=%d record_tid=%llu record_count=%llu record_stream=%p "
          "record_mu_stream=%p record_age_ms=%.3f record_result=%s\n",
          static_cast<unsigned long long>(thread_id),
          static_cast<void*>(event), reinterpret_cast<void*>(mu_event), status,
          static_cast<unsigned long long>(entry.query_count),
          static_cast<unsigned long long>(entry.not_ready_count), elapsed_ms,
          query_rate, has_record ? 1 : 0,
          has_record ? static_cast<unsigned long long>(record_it->second.thread_id)
                     : 0ULL,
          has_record
              ? static_cast<unsigned long long>(record_it->second.record_count)
              : 0ULL,
          has_record ? reinterpret_cast<void*>(record_it->second.stream)
                     : nullptr,
          has_record ? reinterpret_cast<void*>(record_it->second.mu_stream)
                     : nullptr,
          record_age_ms,
          has_record ? FormatMuResult(record_it->second.last_result).c_str()
                     : "<none>");
  fflush(stderr);

  if (result != MUSA_ERROR_NOT_READY) {
    stats.erase(key);
  }
}

void TraceRecordEvent(Event* event, MUevent mu_event, Stream* stream,
                      MUstream mu_stream, MUresult result) {
  if (!IsMusaPollEventTraceEnabled()) return;

  const uint64_t thread_id = CurrentThreadIdForTrace();
  const auto now = std::chrono::steady_clock::now();

  std::lock_guard<std::mutex> lock(PollEventTraceMutex());
  auto& record =
      PollEventRecordTraceMap()[reinterpret_cast<uintptr_t>(event)];
  record.thread_id = thread_id;
  record.stream = reinterpret_cast<uintptr_t>(stream);
  record.mu_stream = reinterpret_cast<uintptr_t>(mu_stream);
  record.last_result = result;
  record.last_record = now;
  ++record.record_count;

  fprintf(stderr,
          "[musa_record_event] tid=%llu se_event=%p mu_event=%p stream=%p "
          "mu_stream=%p record_count=%llu result=%s\n",
          static_cast<unsigned long long>(thread_id),
          static_cast<void*>(event), reinterpret_cast<void*>(mu_event),
          static_cast<void*>(stream), reinterpret_cast<void*>(mu_stream),
          static_cast<unsigned long long>(record.record_count),
          FormatMuResult(result).c_str());
  fflush(stderr);
}

std::string FormatMuResult(MUresult result) {
  const char* error_name = nullptr;
  const char* error_string = nullptr;
  (void)muGetErrorName(result, &error_name);
  (void)muGetErrorString(result, &error_string);
  std::string formatted =
      error_name != nullptr ? error_name : "UNKNOWN MUSA DRIVER ERROR";
  if (error_string != nullptr && error_string[0] != '\0') {
    absl::StrAppend(&formatted, ": ", error_string);
  }
  return formatted;
}

MusaEvent* AsMusaEvent(Event* event) {
  return static_cast<MusaEvent*>(event->implementation());
}

MUdeviceptr AsMusaDevicePtr(const DeviceMemoryBase& gpu_mem) {
  return reinterpret_cast<MUdeviceptr>(gpu_mem.opaque());
}

MUdeviceptr AsMusaDevicePtr(DeviceMemoryBase* gpu_mem) {
  return AsMusaDevicePtr(*gpu_mem);
}

int ResolveLegacyStreamPriority(
    std::variant<StreamPriority, int> priority) {
  if (std::holds_alternative<int>(priority)) {
    return std::get<int>(priority);
  }
  int lowest = 0;
  int highest = 0;
  auto status = musa::ToStatus(muCtxGetStreamPriorityRange(&lowest, &highest));
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 0;
  }
  switch (std::get<StreamPriority>(priority)) {
    case StreamPriority::Highest:
      return highest;
    case StreamPriority::Lowest:
      return lowest;
    case StreamPriority::Default:
      return 0;
  }
  return 0;
}

class MusaExecutorStream : public ::stream_executor::internal::StreamInterface {
 public:
  explicit MusaExecutorStream(MUcontext context) : context_(context) {}
  ~MusaExecutorStream() override { Destroy().IgnoreError(); }

  void SetPriority(StreamPriority priority) override { priority_ = priority; }
  void SetPriority(int priority) override { priority_ = priority; }
  std::variant<StreamPriority, int> priority() const override {
    return priority_;
  }

  void* GpuStreamHack() override { return stream_handle_; }
  void** GpuStreamMemberHack() override {
    return reinterpret_cast<void**>(&stream_handle_);
  }

  tsl::Status Init() {
    if (stream_handle_ != nullptr) {
      return tsl::OkStatus();
    }
    ScopedActivateContext activation(context_);
    TF_RETURN_IF_ERROR(activation.status());

    const int stream_priority = ResolveLegacyStreamPriority(priority_);
    if (stream_priority == 0) {
      TF_RETURN_IF_ERROR(musa::ToStatus(
          muStreamCreate(&stream_handle_, MU_STREAM_NON_BLOCKING),
          "Failed to create MUSA stream"));
    } else {
      TF_RETURN_IF_ERROR(musa::ToStatus(
          muStreamCreateWithPriority(&stream_handle_, MU_STREAM_NON_BLOCKING,
                                     stream_priority),
          "Failed to create MUSA stream with priority"));
    }

    auto event_status = musa::ToStatus(
        muEventCreate(&completed_event_, MU_EVENT_DISABLE_TIMING),
        "Failed to create MUSA completion event");
    if (!event_status.ok()) {
      muStreamDestroy(stream_handle_);
      stream_handle_ = nullptr;
      return event_status;
    }
    return tsl::OkStatus();
  }

  tsl::Status Destroy() {
    if (stream_handle_ == nullptr && completed_event_ == nullptr) {
      return tsl::OkStatus();
    }
    ScopedActivateContext activation(context_);
    TF_RETURN_IF_ERROR(activation.status());

    tsl::Status status = tsl::OkStatus();
    if (completed_event_ != nullptr) {
      auto destroy_event_status = musa::ToStatus(
          muEventDestroy(completed_event_), "Failed to destroy MUSA event");
      if (!destroy_event_status.ok()) {
        status = destroy_event_status;
      }
      completed_event_ = nullptr;
    }
    if (stream_handle_ != nullptr) {
      auto destroy_stream_status = musa::ToStatus(
          muStreamDestroy(stream_handle_), "Failed to destroy MUSA stream");
      if (!destroy_stream_status.ok() && status.ok()) {
        status = destroy_stream_status;
      }
      stream_handle_ = nullptr;
    }
    return status;
  }

  tsl::Status RecordCompletedEvent() {
    if (stream_handle_ == nullptr || completed_event_ == nullptr) {
      return tsl::errors::FailedPrecondition("MUSA stream is not initialized");
    }
    absl::MutexLock submit_lock(&submit_mu_);
    return musa::ToStatus(muEventRecord(completed_event_, stream_handle_),
                          "Failed to record MUSA completion event");
  }

  MUstream stream_handle() const { return stream_handle_; }
  MUevent completed_event() const { return completed_event_; }
  absl::Mutex* submit_mu() { return &submit_mu_; }

 private:
  MUcontext context_ = nullptr;
  MUstream stream_handle_ = nullptr;
  MUevent completed_event_ = nullptr;
  absl::Mutex submit_mu_;
  std::variant<StreamPriority, int> priority_ = StreamPriority::Default;
};

MusaExecutorStream* AsMusaExecutorStream(Stream* stream) {
  return static_cast<MusaExecutorStream*>(stream->implementation());
}

MUstream GetMusaStreamHandle(Stream* stream) {
  if (auto* musa_stream = dynamic_cast<MusaStream*>(stream)) {
    return musa_stream->stream_handle();
  }
  return AsMusaExecutorStream(stream)->stream_handle();
}

absl::Mutex* GetMusaStreamSubmitMutex(Stream* stream) {
  if (auto* musa_stream = dynamic_cast<MusaStream*>(stream)) {
    return musa_stream->submit_mu();
  }
  return AsMusaExecutorStream(stream)->submit_mu();
}

MUevent GetMusaCompletedEvent(Stream* stream) {
  if (auto* musa_stream = dynamic_cast<MusaStream*>(stream)) {
    return musa_stream->completed_event();
  }
  return AsMusaExecutorStream(stream)->completed_event();
}

tsl::Status RecordMusaCompletedEvent(Stream* stream) {
  if (auto* musa_stream = dynamic_cast<MusaStream*>(stream)) {
    return musa_stream->RecordCompletedEvent();
  }
  return AsMusaExecutorStream(stream)->RecordCompletedEvent();
}

void RecordMusaDebugLaunchCheckpoint(Stream* stream, const std::string& kernel_name,
                                     const std::string& launch_dims,
                                     uint64_t shared_mem_bytes) {
  if (auto* musa_stream = dynamic_cast<MusaStream*>(stream)) {
    musa_stream->RecordDebugLaunchCheckpoint(kernel_name, launch_dims,
                                             shared_mem_bytes);
  }
}

void ExecutorHostCallback(void* data) {
  auto* callback = reinterpret_cast<absl::AnyInvocable<void() &&>*>(data);
  std::move(*callback)();
  delete callback;
}

absl::uint128 Fingerprint128(absl::string_view s) {
  auto fp = tsl::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}

template <typename T>
tsl::StatusOr<T> GetSimpleAttribute(MUdevice device,
                                    MUdevice_attribute attribute) {
  int value = -1;
  TF_RETURN_IF_ERROR(musa::ToStatus(muDeviceGetAttribute(&value, attribute, device),
                                    "Could not retrieve MUSA device attribute"));
  return static_cast<T>(value);
}

tsl::StatusOr<std::string> GetDeviceName(MUdevice device) {
  constexpr size_t kCharLimit = 64;
  char chars[kCharLimit];
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muDeviceGetName(chars, kCharLimit - 1, device),
                     "Failed to get device name"));
  chars[kCharLimit - 1] = '\0';
  return std::string(chars);
}

tsl::StatusOr<std::pair<int, int>> GetComputeCapability(MUdevice device) {
  TF_ASSIGN_OR_RETURN(
      int cc_major,
      GetSimpleAttribute<int>(device,
                              MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR));
  TF_ASSIGN_OR_RETURN(
      int cc_minor,
      GetSimpleAttribute<int>(device,
                              MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR));
  return std::make_pair(cc_major, cc_minor);
}

bool GetDeviceTotalMemory(MUdevice device, uint64_t* result) {
  size_t value = 0;
  auto status = musa::ToStatus(muDeviceTotalMem(&value, device),
                               "Failed to query total MUSA memory");
  if (!status.ok()) {
    LOG(ERROR) << status;
    return false;
  }
  *result = value;
  return true;
}

std::string FormatKernelArgsSummary(const KernelArgsArrayBase& args) {
  std::vector<std::string> parts;
  parts.reserve(args.number_of_arguments());
  KernelArgIterator iter = args.arg_iterator();
  size_t index = 0;
  while (iter.has_next()) {
    KernelArg arg = iter.next();
    if (arg.is_shared) {
      parts.push_back(
          absl::StrCat("shared#", index, "(bytes=", arg.size, ")"));
    } else {
      parts.push_back(
          absl::StrCat("arg#", index, "(size=", arg.size, ", addr=",
                       absl::StrFormat("%p", arg.address), ")"));
    }
    ++index;
  }
  return absl::StrCat("[", absl::StrJoin(parts, ", "), "]");
}

bool CanEnablePeerAccess(MUdevice from, MUdevice to) {
  if (from == to) {
    return true;
  }
  int can_access_peer = 0;
  auto status = musa::ToStatus(
      muDeviceCanAccessPeer(&can_access_peer, from, to),
      "Failed to query MUSA peer access capability");
  if (!status.ok()) {
    LOG(ERROR) << status;
    return false;
  }
  return can_access_peer != 0;
}

bool CanEnablePeerAccess(const MusaContext* from, const MusaContext* to) {
  if (from == nullptr || to == nullptr) {
    LOG(ERROR) << "Failed to query MUSA peer access capability: null context";
    return false;
  }
  if (from->context() == to->context()) {
    return true;
  }
  return CanEnablePeerAccess(from->device(), to->device());
}

tsl::Status EnablePeerAccess(const MusaContext* from, const MusaContext* to) {
  if (from == nullptr || to == nullptr) {
    return tsl::errors::FailedPrecondition(
        "Failed to enable MUSA peer access: null context");
  }
  if (from->context() == to->context()) {
    return ::tsl::OkStatus();
  }

  ScopedActivateContext activation(from);
  TF_RETURN_IF_ERROR(activation.status());

  MUresult result = muCtxEnablePeerAccess(to->context(), 0);
  if (result != MUSA_SUCCESS &&
      result != MUSA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return tsl::errors::Internal(
        absl::StrCat("Failed to enable MUSA peer access from device ",
                     from->device(), " to device ", to->device(), ": ",
                     musa::ToStatus(result).message()));
  }

  return ::tsl::OkStatus();
}

bool IsEccEnabled(MUdevice device, bool* result) {
  auto status_or =
      GetSimpleAttribute<int>(device, MU_DEVICE_ATTRIBUTE_ECC_ENABLED);
  if (!status_or.ok()) {
    LOG(ERROR) << status_or.status();
    return false;
  }
  *result = status_or.value() != 0;
  return true;
}

std::string GetPCIBusID(MUdevice device) {
  constexpr int kBufferSize = 64;
  char raw_pci_bus_id[kBufferSize];
  auto status = musa::ToStatus(
      muDeviceGetPCIBusId(raw_pci_bus_id, kBufferSize, device),
      "Failed to query PCI bus id");
  if (!status.ok()) {
    LOG(ERROR) << status;
    return "";
  }
  raw_pci_bus_id[kBufferSize - 1] = '\0';
  return absl::AsciiStrToLower(std::string(raw_pci_bus_id));
}

int TryToReadNumaNode(const std::string& pci_bus_id, int device_ordinal) {
  static const int kUnknownNumaNode = -1;
  if (pci_bus_id.empty()) {
    LOG(INFO) << "no PCI bus ID for device ordinal: " << device_ordinal;
    return kUnknownNumaNode;
  }

  std::string filename =
      absl::StrFormat("/sys/bus/pci/devices/%s/numa_node", pci_bus_id);
  FILE* file = fopen(filename.c_str(), "r");
  if (file == nullptr) {
    LOG(INFO) << "could not open file to read NUMA node: " << filename;
    return kUnknownNumaNode;
  }

  char buf[32];
  size_t did_read = fread(buf, sizeof(buf[0]), sizeof(buf) - 1, file);
  buf[did_read] = '\0';
  fclose(file);

  int32_t value = 0;
  if (!absl::SimpleAtoi(buf, &value)) {
    LOG(WARNING) << "could not parse NUMA node value: " << buf;
    return kUnknownNumaNode;
  }
  return value < 0 ? 0 : value;
}

tsl::Status FillBlockDimLimit(MUdevice device, BlockDim* block_dim_limit) {
  int x = 0;
  int y = 0;
  int z = 0;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muDeviceGetAttribute(&x, MU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device),
      "Could not get MUSA max grid dim X"));
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muDeviceGetAttribute(&y, MU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device),
      "Could not get MUSA max grid dim Y"));
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muDeviceGetAttribute(&z, MU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device),
      "Could not get MUSA max grid dim Z"));
  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return ::tsl::OkStatus();
}

}  // namespace

tsl::Status MusaExecutor::Init(int device_ordinal, DeviceOptions device_options) {
  (void)device_options;
  device_ordinal_ = device_ordinal;
  TF_ASSIGN_OR_RETURN(physical_device_ordinal_,
                      GetMusaPhysicalDeviceOrdinal(device_ordinal));
  TF_RETURN_IF_ERROR(musa::ToStatus(muInit(0), "Failed call to muInit"));
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muDeviceGet(&device_, physical_device_ordinal_),
                     "Failed to acquire MUSA device"));
  TF_ASSIGN_OR_RETURN(auto context, MusaContext::Create(device_));
  context_ = std::move(context);
  return ::tsl::OkStatus();
}

ScopedActivateContext MusaExecutor::ActivateContext() const {
  return ScopedActivateContext(context_.get());
}

DeviceMemoryBase MusaExecutor::Allocate(uint64_t size, int64_t memory_space) {
  (void)memory_space;
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return DeviceMemoryBase();
  }

  // Use BFC allocator for better performance with repeated allocations
  void* ptr = GetBFCAllocator()->Allocate(size);
  if (!ptr) {
    LOG(ERROR) << "BFC allocation failed for " << size << " bytes";
    return DeviceMemoryBase();
  }

  RegisterDeviceMemoryAllocation(ptr, size, context());
  return DeviceMemoryBase(ptr, size);
}

void* MusaExecutor::GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                                 uint64_t size) {
  (void)size;
  return reinterpret_cast<char*>(parent->opaque()) + offset;
}

void MusaExecutor::Deallocate(DeviceMemoryBase* mem) {
  if (mem->is_null()) {
    return;
  }
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }
  void* opaque = mem->opaque();
  const uint64_t size = mem->size();
  if (IsMusaDebugDeallocEnabled()) {
    LOG(INFO) << "[MUSA_DEALLOC_DEBUG] muMemFree begin ptr=" << opaque
              << " size=" << size << "B device_ordinal=" << device_ordinal_
              << " physical_device_ordinal=" << physical_device_ordinal_;
  }
  // Prefer BFC fast path; fall back to raw driver free for non-BFC pointers.
  if (!GetBFCAllocator()->Deallocate(opaque)) {
    LOG(ERROR) << "MusaBFC: unknown pointer in Deallocate " << opaque
               << ", skip raw muMemFree to avoid invalid free";
    return;
  }
  if (IsMusaDebugDeallocEnabled()) {
    LOG(INFO) << "[MUSA_DEALLOC_DEBUG] muMemFree end ptr=" << opaque
              << " size=" << size << "B device_ordinal=" << device_ordinal_
              << " physical_device_ordinal=" << physical_device_ordinal_;
  }
  UnregisterDeviceMemoryAllocation(opaque);
}

void* MusaExecutor::UnifiedMemoryAllocate(uint64_t size) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return nullptr;
  }
  MUdeviceptr ptr = 0;
  auto status = musa::ToStatus(
      muMemAllocManaged(&ptr, size, MU_MEM_ATTACH_GLOBAL),
      "Failed to allocate MUSA unified memory");
  if (!status.ok()) {
    LOG(ERROR) << status;
    return nullptr;
  }
  return reinterpret_cast<void*>(ptr);
}

void MusaExecutor::UnifiedMemoryDeallocate(void* mem) {
  if (mem == nullptr) {
    return;
  }
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }
  auto status =
      musa::ToStatus(muMemFree(reinterpret_cast<MUdeviceptr>(mem)),
                     "Failed to free MUSA unified memory");
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

void* MusaExecutor::HostMemoryAllocate(uint64_t size) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return nullptr;
  }
  // Use pinned memory pool for better performance
  void* buffer = GetPinnedMemoryPool()->Allocate(size);
  if (!buffer) {
    LOG(ERROR) << "Pinned memory pool allocation failed for " << size << " bytes";
    return nullptr;
  }
  return buffer;
}

void MusaExecutor::HostMemoryDeallocate(void* mem) {
  if (mem == nullptr) {
    return;
  }
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }
  // Use pinned memory pool for better performance
  GetPinnedMemoryPool()->Deallocate(mem);
}

bool MusaExecutor::HostMemoryRegister(void* mem, uint64_t size) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  return musa::ToStatus(
             muMemHostRegister(mem, size, MU_MEMHOSTREGISTER_PORTABLE),
             "Failed to register MUSA host memory")
      .ok();
}

bool MusaExecutor::HostMemoryUnregister(void* mem) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  return musa::ToStatus(muMemHostUnregister(mem),
                        "Failed to unregister MUSA host memory")
      .ok();
}

bool MusaExecutor::SynchronizeAllActivity() {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  return musa::ToStatus(muCtxSynchronize(), "Failed to synchronize MUSA context")
      .ok();
}

tsl::Status MusaExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                             uint64_t size) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  if (reinterpret_cast<uintptr_t>(location->opaque()) % sizeof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return musa::ToStatus(
        muMemsetD32(AsMusaDevicePtr(location), 0x0, size / sizeof(uint32_t)),
        "Failed synchronous MUSA memset32");
  }
  return musa::ToStatus(muMemsetD8(AsMusaDevicePtr(location), 0, size),
                        "Failed synchronous MUSA memzero");
}

tsl::Status MusaExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                            int value, uint64_t size) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  if (reinterpret_cast<uintptr_t>(location->opaque()) % sizeof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    uint8_t byte_value = static_cast<uint8_t>(value);
    uint32_t pattern = (byte_value << 24) | (byte_value << 16) |
                       (byte_value << 8) | byte_value;
    return musa::ToStatus(
        muMemsetD32(AsMusaDevicePtr(location), pattern, size / sizeof(uint32_t)),
        "Failed synchronous MUSA memset32");
  }
  return musa::ToStatus(muMemsetD8(AsMusaDevicePtr(location), value, size),
                        "Failed synchronous MUSA memset");
}

tsl::Status MusaExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                            const void* host_src,
                                            uint64_t size) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  return musa::ToStatus(muMemcpyHtoD(AsMusaDevicePtr(gpu_dst), host_src, size),
                        "Failed synchronous MUSA memcpy H2D");
}

tsl::Status MusaExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceMemoryBase& gpu_src,
                                            uint64_t size) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  return musa::ToStatus(muMemcpyDtoH(host_dst, AsMusaDevicePtr(gpu_src), size),
                        "Failed synchronous MUSA memcpy D2H");
}

tsl::Status MusaExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  if (size == 0 || gpu_dst->opaque() == nullptr || gpu_src.opaque() == nullptr) {
    return musa::ToStatus(
        muMemcpyDtoD(AsMusaDevicePtr(gpu_dst), AsMusaDevicePtr(gpu_src), size),
        "Failed synchronous MUSA memcpy D2D");
  }
  MUcontext dst_context = GetContextForDeviceMemory(gpu_dst->opaque());
  MUcontext src_context =
      GetContextForDeviceMemory(const_cast<void*>(gpu_src.opaque()));
  if (gpu_dst->opaque() != nullptr && gpu_src.opaque() != nullptr &&
      dst_context != nullptr && src_context != nullptr &&
      dst_context != src_context) {
    return musa::ToStatus(
        muMemcpyPeer(AsMusaDevicePtr(gpu_dst), dst_context,
                     AsMusaDevicePtr(gpu_src), src_context, size),
        "Failed synchronous MUSA memcpy peer D2D");
  }
  return musa::ToStatus(
      muMemcpyDtoD(AsMusaDevicePtr(gpu_dst), AsMusaDevicePtr(gpu_src), size),
      "Failed synchronous MUSA memcpy D2D");
}

tsl::Status MusaExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                  uint64_t size) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  MUstream stream_handle = GetMusaStreamHandle(stream);
  absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
  return musa::ToStatus(
      muMemsetD8Async(reinterpret_cast<MUdeviceptr>(location->opaque()), 0, size,
                      stream_handle),
      "Failed to enqueue MUSA memzero");
}

tsl::Status MusaExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                   uint32_t pattern, uint64_t size) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  if (size % sizeof(uint32_t) != 0) {
    return tsl::errors::InvalidArgument("size must be divisible by 4");
  }
  MUstream stream_handle = GetMusaStreamHandle(stream);
  absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
  return musa::ToStatus(
      muMemsetD32Async(reinterpret_cast<MUdeviceptr>(location->opaque()),
                       pattern, size / sizeof(uint32_t), stream_handle),
      "Failed to enqueue MUSA memset32");
}

bool MusaExecutor::Memcpy(Stream* stream, void* host_dst,
                          const DeviceMemoryBase& gpu_src, uint64_t size) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  MUstream stream_handle = GetMusaStreamHandle(stream);
  absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
  return musa::ToStatus(
             muMemcpyDtoHAsync(host_dst,
                               reinterpret_cast<MUdeviceptr>(gpu_src.opaque()),
                               size, stream_handle),
             "Failed to enqueue MUSA memcpy D2H")
      .ok();
}

bool MusaExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                          const void* host_src, uint64_t size) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  MUstream stream_handle = GetMusaStreamHandle(stream);
  absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
  return musa::ToStatus(
             muMemcpyHtoDAsync(reinterpret_cast<MUdeviceptr>(gpu_dst->opaque()),
                               host_src, size, stream_handle),
             "Failed to enqueue MUSA memcpy H2D")
      .ok();
}

bool MusaExecutor::MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                                        const DeviceMemoryBase& gpu_src,
                                        uint64_t size) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  MUstream stream_handle = GetMusaStreamHandle(stream);
  if (size == 0 || gpu_dst->opaque() == nullptr || gpu_src.opaque() == nullptr) {
    absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
    return musa::ToStatus(
               muMemcpyDtoDAsync(reinterpret_cast<MUdeviceptr>(gpu_dst->opaque()),
                                 reinterpret_cast<MUdeviceptr>(gpu_src.opaque()),
                                 size, stream_handle),
               "Failed to enqueue MUSA memcpy D2D")
        .ok();
  }
  MUcontext dst_context = GetContextForDeviceMemory(gpu_dst->opaque());
  MUcontext src_context =
      GetContextForDeviceMemory(const_cast<void*>(gpu_src.opaque()));
  if (dst_context != nullptr && src_context != nullptr &&
      dst_context != src_context) {
    absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
    return musa::ToStatus(
               muMemcpyPeerAsync(reinterpret_cast<MUdeviceptr>(gpu_dst->opaque()),
                                 dst_context,
                                 reinterpret_cast<MUdeviceptr>(gpu_src.opaque()),
                                 src_context, size, stream_handle),
               "Failed to enqueue MUSA memcpy peer D2D")
        .ok();
  }
  absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
  return musa::ToStatus(
             muMemcpyDtoDAsync(reinterpret_cast<MUdeviceptr>(gpu_dst->opaque()),
                               reinterpret_cast<MUdeviceptr>(gpu_src.opaque()),
                               size, stream_handle),
             "Failed to enqueue MUSA memcpy D2D")
      .ok();
}

bool MusaExecutor::HostCallback(Stream* stream,
                                absl::AnyInvocable<tsl::Status() &&> callback) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }

  auto* callback_ptr =
      new absl::AnyInvocable<void() &&>([cb = std::move(callback)]() mutable {
        tsl::Status status = std::move(cb)();
        if (!status.ok()) {
          LOG(WARNING) << "Host callback failed: " << status;
        }
      });

  MUstream stream_handle = GetMusaStreamHandle(stream);
  tsl::Status status;
  {
    absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
    status = musa::ToStatus(
        muLaunchHostFunc(stream_handle, ExecutorHostCallback, callback_ptr),
        "Failed to enqueue MUSA host callback");
  }
  if (!status.ok()) {
    delete callback_ptr;
  }
  return status.ok();
}

tsl::Status MusaExecutor::AllocateEvent(Event* event) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  MUevent handle = nullptr;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muEventCreate(&handle, MU_EVENT_DISABLE_TIMING),
      "Failed to allocate MUSA event"));
  AsMusaEvent(event)->set_handle(handle);
  return ::tsl::OkStatus();
}

tsl::Status MusaExecutor::DeallocateEvent(Event* event) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  MUevent handle = AsMusaEvent(event)->handle();
  if (handle != nullptr) {
    TF_RETURN_IF_ERROR(
        musa::ToStatus(muEventDestroy(handle), "Failed to destroy MUSA event"));
    AsMusaEvent(event)->clear_handle();
  }
  return ::tsl::OkStatus();
}

tsl::Status MusaExecutor::RecordEvent(Stream* stream, Event* event) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  MUevent mu_event = AsMusaEvent(event)->handle();
  MUstream mu_stream = GetMusaStreamHandle(stream);
  MUresult result;
  {
    absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
    result = muEventRecord(mu_event, mu_stream);
  }
  TraceRecordEvent(event, mu_event, stream, mu_stream, result);
  return musa::ToStatus(result, "Failed to record MUSA event");
}

tsl::Status MusaExecutor::WaitForEvent(Stream* stream, Event* event) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  MUstream stream_handle = GetMusaStreamHandle(stream);
  absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
  return musa::ToStatus(
      muStreamWaitEvent(stream_handle, AsMusaEvent(event)->handle(), 0),
      "Failed to wait on MUSA event");
}

tsl::Status MusaExecutor::WaitForEventOnExternalStream(std::intptr_t stream,
                                                       Event* event) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  return musa::ToStatus(
      muStreamWaitEvent(reinterpret_cast<MUstream>(stream),
                        AsMusaEvent(event)->handle(), 0),
      "Failed waiting on external MUSA stream event");
}

Event::Status MusaExecutor::PollForEventStatus(Event* event) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return Event::Status::kError;
  }
  MUevent mu_event = AsMusaEvent(event)->handle();
  MUresult result = muEventQuery(mu_event);
  TracePollForEventStatus(event, mu_event, result);
  if (result == MUSA_SUCCESS) {
    return Event::Status::kComplete;
  }
  if (result == MUSA_ERROR_NOT_READY) {
    return Event::Status::kPending;
  }
  return Event::Status::kError;
}

bool MusaExecutor::AllocateStream(Stream* stream) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  auto* musa_stream = AsMusaExecutorStream(stream);
  tsl::Status status = musa_stream->Init();
  if (!status.ok()) {
    LOG(ERROR) << status;
    return false;
  }
  absl::MutexLock lock(&alive_streams_mu_);
  alive_streams_[musa_stream->stream_handle()] = stream;
  return true;
}

void MusaExecutor::DeallocateStream(Stream* stream) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }
  auto* musa_stream = AsMusaExecutorStream(stream);
  {
    absl::MutexLock lock(&alive_streams_mu_);
    alive_streams_.erase(musa_stream->stream_handle());
  }
  auto status = musa::ToStatus(muStreamSynchronize(musa_stream->stream_handle()),
                               "Failed to synchronize MUSA stream");
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  status = musa_stream->Destroy();
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

bool MusaExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  tsl::Status record_status = RecordMusaCompletedEvent(other);
  if (!record_status.ok()) {
    LOG(ERROR) << "failed to record completion event; therefore, failed to "
                  "create MUSA stream dependency: "
               << record_status;
    return false;
  }
  tsl::Status wait_status;
  {
    absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(dependent));
    wait_status = musa::ToStatus(
        muStreamWaitEvent(GetMusaStreamHandle(dependent),
                          GetMusaCompletedEvent(other), 0),
        "Failed to create MUSA stream dependency");
  }
  if (!wait_status.ok()) {
    LOG(ERROR) << wait_status;
    return false;
  }
  return true;
}

tsl::Status MusaExecutor::BlockHostUntilDone(Stream* stream) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  return musa::ToStatus(muStreamSynchronize(GetMusaStreamHandle(stream)),
                        "Failed to synchronize MUSA stream");
}

tsl::Status MusaExecutor::EnablePeerAccessTo(
    ::stream_executor::internal::StreamExecutorInterface* other) {
  auto* musa_other = static_cast<MusaExecutor*>(other);
  if (!CanEnablePeerAccessTo(other)) {
    return tsl::errors::FailedPrecondition(
        absl::StrCat("MUSA peer access is unavailable between device ",
                     context_->device(), " and ", musa_other->context_->device(),
                     "."));
  }
  return EnablePeerAccess(context_.get(), musa_other->context_.get());
}

bool MusaExecutor::CanEnablePeerAccessTo(
    ::stream_executor::internal::StreamExecutorInterface* other) {
  auto* musa_other = static_cast<MusaExecutor*>(other);
  return CanEnablePeerAccess(context_.get(), musa_other->context_.get());
}

blas::BlasSupport* MusaExecutor::CreateBlas() {
  PluginRegistry* registry = PluginRegistry::Instance();
  tsl::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(musa::kMusaPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().message();
    return nullptr;
  }

  return status.value()(this);
}

bool MusaExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  auto status = musa::ToStatus(musaMemGetInfo(&free_bytes, &total_bytes),
                               "Failed to query MUSA memory usage");
  if (!status.ok()) {
    LOG(ERROR) << status;
    return false;
  }
  *free = free_bytes;
  *total = total_bytes;
  return true;
}

tsl::Status MusaExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                    KernelBase* kernel) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  MusaKernel* musa_kernel = AsMusaKernel(kernel);
  MUmodule module = nullptr;
  const std::string* kernel_name = nullptr;

  if (spec.has_cuda_cubin_in_memory()) {
    kernel_name = &spec.cuda_cubin_in_memory().kernel_name();
    const char* cubin = spec.cuda_cubin_in_memory().bytes();
    absl::MutexLock lock(&in_memory_modules_mu_);
    void* binary_id = const_cast<char*>(cubin);
    auto& entry = gpu_binary_to_module_[binary_id];
    if (entry.first == nullptr) {
      TF_RETURN_IF_ERROR(musa::ToStatus(
          muModuleLoadFatBinary(&entry.first, cubin),
          "Failed to load MUSA kernel fatbinary"));
      entry.second = 1;
    } else {
      ++entry.second;
    }
    module = entry.first;
    kernel_to_gpu_binary_[kernel] = binary_id;
  } else if (spec.has_cuda_ptx_in_memory()) {
    kernel_name = &spec.cuda_ptx_in_memory().kernel_name();
    TF_ASSIGN_OR_RETURN(auto cc, GetComputeCapability(device_));
    const char* ptx = spec.cuda_ptx_in_memory().text(cc.first, cc.second);
    if (ptx == nullptr) {
      ptx = spec.cuda_ptx_in_memory().default_text();
    }
    if (ptx == nullptr) {
      return tsl::errors::Internal("No PTX found in MUSA kernel loader spec");
    }

    absl::MutexLock lock(&in_memory_modules_mu_);
    void* binary_id = const_cast<char*>(ptx);
    auto& entry = gpu_binary_to_module_[binary_id];
    if (entry.first == nullptr) {
      void* ptx_data = const_cast<char*>(ptx);
      TF_RETURN_IF_ERROR(musa::ToStatus(
          muModuleLoadDataEx(&entry.first, ptx_data, 0, nullptr, nullptr),
          "Failed to load MUSA PTX kernel module"));
      entry.second = 1;
    } else {
      ++entry.second;
    }
    module = entry.first;
    kernel_to_gpu_binary_[kernel] = binary_id;
  } else {
    return tsl::errors::Internal("No method of loading MUSA kernel provided");
  }

  TF_RETURN_IF_ERROR(musa::ToStatus(
      muModuleGetFunction(musa_kernel->gpu_function_ptr(), module,
                          kernel_name->c_str()),
      "Failed to get MUSA module function"));

  musa_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  TF_RETURN_IF_ERROR(musa_kernel->GetKernelMetadata(&kernel_metadata));
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(*kernel_name);
  return ::tsl::OkStatus();
}

void MusaExecutor::UnloadKernel(const KernelBase* kernel) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return;
  }
  absl::MutexLock lock(&in_memory_modules_mu_);
  auto it = kernel_to_gpu_binary_.find(kernel);
  if (it == kernel_to_gpu_binary_.end()) {
    return;
  }

  auto module_it = gpu_binary_to_module_.find(it->second);
  if (module_it != gpu_binary_to_module_.end()) {
    if (--module_it->second.second == 0) {
      auto status = musa::ToStatus(muModuleUnload(module_it->second.first),
                                   "Failed to unload MUSA module");
      if (!status.ok()) {
        LOG(ERROR) << status;
      } else {
        gpu_binary_to_module_.erase(module_it);
      }
    }
  }
  kernel_to_gpu_binary_.erase(it);
}

tsl::Status MusaExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                     ModuleHandle* module_handle) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  absl::MutexLock lock(&in_memory_modules_mu_);
  if (spec.has_cuda_cubin_in_memory()) {
    void* binary_id =
        const_cast<void*>(static_cast<const void*>(spec.cuda_cubin_in_memory().data()));
    auto& entry = gpu_binary_to_module_[binary_id];
    if (entry.first == nullptr) {
      TF_RETURN_IF_ERROR(musa::ToStatus(
          muModuleLoadFatBinary(
              &entry.first,
              reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data())),
          "Failed to load in-memory MUSA fatbinary"));
      entry.second = 1;
    } else {
      ++entry.second;
    }
    *module_handle = ModuleHandle(binary_id);
    return ::tsl::OkStatus();
  }

  if (spec.has_cuda_ptx_in_memory()) {
    const char* ptx = spec.cuda_ptx_in_memory();
    if (ptx == nullptr) {
      return tsl::errors::Internal("PTX not found in spec");
    }
    void* binary_id = const_cast<char*>(ptx);
    auto& entry = gpu_binary_to_module_[binary_id];
    if (entry.first == nullptr) {
      void* ptx_data = const_cast<char*>(ptx);
      TF_RETURN_IF_ERROR(
          musa::ToStatus(muModuleLoadDataEx(&entry.first, ptx_data, 0, nullptr, nullptr),
                         "Failed to load in-memory MUSA PTX"));
      entry.second = 1;
    } else {
      ++entry.second;
    }
    *module_handle = ModuleHandle(binary_id);
    return ::tsl::OkStatus();
  }

  return tsl::errors::Internal("No method of loading MUSA module provided");
}

bool MusaExecutor::UnloadModule(ModuleHandle module_handle) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  absl::MutexLock lock(&in_memory_modules_mu_);
  auto it = gpu_binary_to_module_.find(module_handle.id());
  if (it == gpu_binary_to_module_.end()) {
    return false;
  }
  if (--it->second.second == 0) {
    auto status = musa::ToStatus(muModuleUnload(it->second.first),
                                 "Failed to unload MUSA module");
    if (!status.ok()) {
      LOG(ERROR) << status;
      return false;
    }
    gpu_binary_to_module_.erase(it);
  }
  return true;
}

bool MusaExecutor::GetSymbol(const std::string& symbol_name,
                             ModuleHandle module_handle, void** mem,
                             size_t* bytes) {
  auto activation = ActivateContext();
  if (!activation.ok()) {
    LOG(ERROR) << activation.status();
    return false;
  }
  CHECK(static_cast<bool>(module_handle));
  absl::MutexLock lock(&in_memory_modules_mu_);
  auto it = gpu_binary_to_module_.find(module_handle.id());
  CHECK(it != gpu_binary_to_module_.end());
  MUmodule module = it->second.first;
  CHECK(module != nullptr);
  std::string detail = absl::StrCat("Failed to get symbol '", symbol_name, "'");
  auto status = musa::ToStatus(
      muModuleGetGlobal(reinterpret_cast<MUdeviceptr*>(mem), bytes, module,
                        symbol_name.c_str()),
      detail.c_str());
  if (!status.ok()) {
    LOG(ERROR) << status;
    return false;
  }
  return true;
}

tsl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
MusaExecutor::CreateOrShareConstant(Stream* stream,
                                    const std::vector<uint8_t>& content) {
  absl::MutexLock lock(&shared_constants_mu_);
  absl::uint128 fingerprint = Fingerprint128(absl::string_view(
      reinterpret_cast<const char*>(content.data()), content.size()));
  auto insert_result = shared_constants_.insert(
      {fingerprint, std::weak_ptr<DeviceMemoryBase>()});
  auto it = insert_result.first;

  std::shared_ptr<DeviceMemoryBase> shared_constant = it->second.lock();
  if (shared_constant == nullptr) {
    auto* new_constant =
        new DeviceMemoryBase(Allocate(content.size(), /*memory_space=*/0));
    if (new_constant->opaque() == nullptr) {
      delete new_constant;
      return tsl::errors::Internal(absl::StrFormat(
          "Failed to allocate %d bytes for new constant", content.size()));
    }

    // Avoid coupling constant initialization with the compute stream. Large
    // constant uploads on MUSA can wedge stream synchronization before the
    // first kernel launch completes.
    tsl::Status status =
        SynchronousMemcpy(new_constant, content.data(), content.size());
    if (!status.ok()) {
      Deallocate(new_constant);
      delete new_constant;
      status.Update(tsl::errors::Internal(absl::StrFormat(
          "Memcpy to device address %p failed", new_constant->opaque())));
      return status;
    }

    shared_constant = std::shared_ptr<DeviceMemoryBase>(
        new_constant, [this](DeviceMemoryBase* p) {
          Deallocate(p);
          delete p;
        });
    it->second = shared_constant;
  }
  return shared_constant;
}

tsl::Status MusaExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const KernelBase& kernel,
                                 const KernelArgsArrayBase& args) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  CHECK_EQ(kernel.Arity() + (args.number_of_shared_bytes() > 0),
           args.number_of_arguments());

  const MusaKernel* musa_kernel = AsMusaKernel(&kernel);
  MUfunction function = musa_kernel->AsMusaFunctionHandle();
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();
  const std::string arg_summary = FormatKernelArgsSummary(args);

  VLOG(2) << "Launching kernel " << kernel.name() << "; gdx=" << block_dims.x
          << " gdy=" << block_dims.y << " gdz=" << block_dims.z
          << " bdx=" << thread_dims.x << " bdy=" << thread_dims.y
          << " bdz=" << thread_dims.z << "; shmem=" << shared_mem_bytes
          << "; args=" << arg_summary;

  if (VLOG_IS_ON(2)) {
    absl::MutexLock lock(&launched_kernels_mu_);
    if (!launched_kernels_.count(function)) {
      VlogOccupancyInfo(kernel, thread_dims, shared_mem_bytes);
      launched_kernels_.insert(function);
    }
  }

  TF_RETURN_IF_ERROR(musa_kernel->PrepareForLaunch(shared_mem_bytes));

  void** kernel_params = const_cast<void**>(args.argument_addresses().data());
  MUstream stream_handle = GetMusaStreamHandle(stream);
  const bool runtime_trace = IsMusaDebugRuntimeTraceEnabled();

  std::string detail = absl::StrCat(
      "Failed to launch MUSA kernel ", kernel.name(), "; block dims=",
      thread_dims.x, "x", thread_dims.y, "x", thread_dims.z, "; grid dims=",
      block_dims.x, "x", block_dims.y, "x", block_dims.z,
      "; shared memory bytes=", shared_mem_bytes, "; args=", arg_summary);
  if (runtime_trace) {
    LOG(INFO) << "[MUSA_RUNTIME_TRACE] muLaunchKernel begin kernel="
              << kernel.name() << " grid=" << block_dims.x << "x"
              << block_dims.y << "x" << block_dims.z << " block="
              << thread_dims.x << "x" << thread_dims.y << "x"
              << thread_dims.z << " shmem=" << shared_mem_bytes
              << " stream=" << stream_handle << " args="
              << args.number_of_arguments();
  }
  MUresult launch_result;
  {
    absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
    launch_result =
        muLaunchKernel(function, block_dims.x, block_dims.y, block_dims.z,
                       thread_dims.x, thread_dims.y, thread_dims.z,
                       shared_mem_bytes, stream_handle, kernel_params, nullptr);
  }
  if (runtime_trace) {
    LOG(INFO) << "[MUSA_RUNTIME_TRACE] muLaunchKernel end kernel="
              << kernel.name() << " stream=" << stream_handle
              << " result=" << FormatMuResult(launch_result);
    if (launch_result == MUSA_SUCCESS) {
      MUresult query_result = muStreamQuery(stream_handle);
      LOG(INFO) << "[MUSA_RUNTIME_TRACE] muStreamQuery after launch kernel="
                << kernel.name() << " stream=" << stream_handle
                << " result=" << FormatMuResult(query_result);
    }
  }
  if (launch_result == MUSA_SUCCESS) {
    RecordMusaDebugLaunchCheckpoint(
        stream,
        kernel.name(),
        absl::StrCat("grid=", block_dims.x, "x", block_dims.y, "x",
                     block_dims.z, " block=", thread_dims.x, "x",
                     thread_dims.y, "x", thread_dims.z),
        shared_mem_bytes);
  }
  return musa::ToStatus(launch_result, detail.c_str());
}

void MusaExecutor::VlogOccupancyInfo(const KernelBase& kernel,
                                     const ThreadDim& thread_dims,
                                     size_t dynamic_shared_memory_bytes) {
  VLOG(2) << "Computing kernel occupancy for kernel "
          << kernel.demangled_name();
  VLOG(2) << "Thread dimensions (" << thread_dims.x << ", " << thread_dims.y
          << ", " << thread_dims.z << ")";

  int regs_per_thread = 0;
  if (!kernel.metadata().registers_per_thread(&regs_per_thread)) {
    return;
  }

  int smem_per_block = 0;
  if (!kernel.metadata().shared_memory_bytes(&smem_per_block)) {
    return;
  }

  const MusaKernel* musa_kernel = AsMusaKernel(&kernel);
  int threads_per_block = thread_dims.x * thread_dims.y * thread_dims.z;
  int blocks_per_sm = 0;
  auto status = musa::ToStatus(
      muOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
          &blocks_per_sm, musa_kernel->AsMusaFunctionHandle(), threads_per_block,
          dynamic_shared_memory_bytes, MU_OCCUPANCY_DISABLE_CACHING_OVERRIDE),
      "Failed to calculate MUSA kernel occupancy");
  if (!status.ok()) {
    VLOG(2) << "Unable to compute kernel occupancy: " << status;
    return;
  }

  VLOG(2) << "Kernel metadata: registers/thread=" << regs_per_thread
          << ", static shared memory bytes=" << smem_per_block
          << ", dynamic shared memory bytes=" << dynamic_shared_memory_bytes;
  VLOG(2) << "Resident blocks per SM is " << blocks_per_sm;
}

tsl::Status MusaExecutor::Submit(Stream* stream,
                                 const CommandBuffer& command_buffer) {
  auto activation = ActivateContext();
  TF_RETURN_IF_ERROR(activation.status());
  if (command_buffer.mode() != CommandBuffer::Mode::kPrimary) {
    return tsl::errors::InvalidArgument(
        "Can't submit non-primary command buffer for execution");
  }

  auto* musa_command_buffer =
      static_cast<const MusaCommandBuffer*>(command_buffer.implementation());
  MUstream stream_handle = GetMusaStreamHandle(stream);
  absl::MutexLock submit_lock(GetMusaStreamSubmitMutex(stream));
  return musa::ToStatus(
      muGraphLaunch(musa_command_buffer->executable(), stream_handle),
      "Failed to launch MUSA graph");
}

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
MusaExecutor::CreateDeviceDescription() const {
  return CreateDeviceDescription(device_ordinal_);
}

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
MusaExecutor::CreateDeviceDescription(int device_ordinal) {
  TF_ASSIGN_OR_RETURN(const int physical_device_ordinal,
                      GetMusaPhysicalDeviceOrdinal(device_ordinal));
  MUdevice device = 0;
  TF_RETURN_IF_ERROR(musa::ToStatus(muDeviceGet(&device, physical_device_ordinal),
                                    "Failed call to muDeviceGet"));

  ::stream_executor::internal::DeviceDescriptionBuilder builder;

  {
    int driver_version = 0;
    auto status = musa::ToStatus(muDriverGetVersion(&driver_version),
                                 "Could not get MUSA driver version");
    if (status.ok()) {
      builder.set_driver_version(absl::StrCat(driver_version));
    } else {
      LOG(ERROR) << status;
      builder.set_driver_version("unknown");
    }
  }

  {
    int runtime_version = 0;
    auto status = musa::ToStatus(musaRuntimeGetVersion(&runtime_version),
                                 "Could not get MUSA runtime version");
    if (status.ok()) {
      builder.set_runtime_version(absl::StrCat(runtime_version));
    } else {
      LOG(ERROR) << status;
      builder.set_runtime_version("unknown");
    }
  }

  {
    std::string pci_bus_id = GetPCIBusID(device);
    builder.set_pci_bus_id(pci_bus_id);
    builder.set_numa_node(
        TryToReadNumaNode(pci_bus_id, physical_device_ordinal));
  }

  {
    auto status_or =
        GetSimpleAttribute<int64_t>(device, MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    if (status_or.ok()) {
      builder.set_threads_per_block_limit(status_or.value());
    }

    ThreadDim thread_dim_limit;
    auto x =
        GetSimpleAttribute<int64_t>(device, MU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    auto y =
        GetSimpleAttribute<int64_t>(device, MU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    auto z =
        GetSimpleAttribute<int64_t>(device, MU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
    if (x.ok() && y.ok() && z.ok()) {
      thread_dim_limit.x = x.value();
      thread_dim_limit.y = y.value();
      thread_dim_limit.z = z.value();
      builder.set_thread_dim_limit(thread_dim_limit);
    }
  }

  if (auto clock_rate =
          GetSimpleAttribute<int>(device, MU_DEVICE_ATTRIBUTE_CLOCK_RATE);
      clock_rate.ok()) {
    builder.set_clock_rate_ghz(static_cast<float>(clock_rate.value()) / 1e6f);
  }

  {
    bool ecc_enabled = false;
    if (IsEccEnabled(device, &ecc_enabled)) {
      builder.set_ecc_enabled(ecc_enabled);
    }
  }

  {
    uint64_t device_memory_size = static_cast<uint64_t>(-1);
    if (GetDeviceTotalMemory(device, &device_memory_size)) {
      builder.set_device_memory_size(device_memory_size);
    }
  }

  if (auto l2_cache =
          GetSimpleAttribute<int64_t>(device, MU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
      l2_cache.ok()) {
    builder.set_l2_cache_size(l2_cache.value());
  }

  auto mem_clock_khz =
      GetSimpleAttribute<int>(device, MU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
  auto mem_bus_width_bits = GetSimpleAttribute<int>(
      device, MU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
  if (mem_clock_khz.ok() && mem_bus_width_bits.ok()) {
    builder.set_memory_bandwidth(2LL * mem_clock_khz.value() * 1000LL *
                                 mem_bus_width_bits.value() / 8LL);
  }

  {
    BlockDim block_dim_limit;
    TF_RETURN_IF_ERROR(FillBlockDimLimit(device, &block_dim_limit));
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    TF_ASSIGN_OR_RETURN(std::string device_name, GetDeviceName(device));
    builder.set_name(device_name);
  }

  {
    TF_ASSIGN_OR_RETURN(auto cc, GetComputeCapability(device));
    builder.set_platform_version(
        absl::StrCat("Compute Capability ", cc.first, ".", cc.second));
  }

  builder.set_device_address_bits(64);
  builder.set_device_vendor("Moore Threads");

  if (auto value = GetSimpleAttribute<int64_t>(
          device, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
      value.ok()) {
    builder.set_shared_memory_per_core(value.value());
  }
  if (auto value = GetSimpleAttribute<int64_t>(
          device, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
      value.ok()) {
    builder.set_shared_memory_per_block(value.value());
  }
  if (auto value = GetSimpleAttribute<int64_t>(
          device, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
      value.ok()) {
    builder.set_shared_memory_per_block_optin(value.value());
  }
  if (auto value = GetSimpleAttribute<int>(device,
                                           MU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
      value.ok()) {
    builder.set_core_count(value.value());
  }
  if (auto value =
          GetSimpleAttribute<int64_t>(device, MU_DEVICE_ATTRIBUTE_WARP_SIZE);
      value.ok()) {
    builder.set_threads_per_warp(value.value());
  }
  if (auto value = GetSimpleAttribute<int64_t>(
          device, MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
      value.ok()) {
    builder.set_threads_per_core_limit(value.value());
  }
  if (auto value = GetSimpleAttribute<int64_t>(
          device, MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
      value.ok()) {
    builder.set_registers_per_block_limit(value.value());
  }
  if (auto value = GetSimpleAttribute<int64_t>(
          device, MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
      value.ok()) {
    builder.set_registers_per_core_limit(value.value());
  }

  return builder.Build();
}

std::unique_ptr<::stream_executor::internal::EventInterface>
MusaExecutor::CreateEventImplementation() {
  return std::make_unique<MusaEvent>();
}

std::unique_ptr<::stream_executor::internal::KernelInterface>
MusaExecutor::CreateKernelImplementation() {
  return std::make_unique<MusaKernel>();
}

std::unique_ptr<::stream_executor::internal::StreamInterface>
MusaExecutor::GetStreamImplementation() {
  return std::make_unique<MusaExecutorStream>(context());
}

tsl::StatusOr<std::unique_ptr<::stream_executor::internal::CommandBufferInterface>>
MusaExecutor::GetCommandBufferImplementation(CommandBuffer::Mode mode) {
  return MusaCommandBuffer::Create(mode, context());
}

tsl::Status MusaExecutor::CreateCustomStream(
    StreamExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority,
    std::unique_ptr<Stream>* stream) {
  TF_ASSIGN_OR_RETURN(auto custom_stream,
                      MusaStream::Create(executor, context(), priority));
  auto* musa_stream = static_cast<MusaStream*>(custom_stream.get());
  absl::MutexLock lock(&alive_streams_mu_);
  alive_streams_[musa_stream->stream_handle()] = custom_stream.get();
  *stream = std::move(custom_stream);
  return ::tsl::OkStatus();
}

Stream* MusaExecutor::FindAllocatedStream(void* gpu_stream) {
  absl::MutexLock lock(&alive_streams_mu_);
  auto it = alive_streams_.find(gpu_stream);
  return it == alive_streams_.end() ? nullptr : it->second;
}

void MusaExecutor::UnregisterStream(MUstream stream) {
  absl::MutexLock lock(&alive_streams_mu_);
  alive_streams_.erase(stream);
}

}  // namespace musa
}  // namespace stream_executor
