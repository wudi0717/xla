#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/hlo.pb.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "pjrt_plugin/src/reusable_host_buffer_arena_plan.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {


// =========================================================================
// 🚀 核心垫片层与实现
// =========================================================================

static PJRT_Api base_api;
static bool base_api_initialized = false;
static std::mutex g_execute_submit_mu;
static std::atomic<unsigned long long> g_event_destroy_bypass_count{0};
static std::atomic<unsigned long long> g_buffer_destroy_bypass_count{0};
static std::atomic<unsigned long long> g_client_compile_log_count{0};
static std::atomic<unsigned long long> g_buffer_from_host_log_count{0};
static std::atomic<unsigned long long> g_execute_log_count{0};
static std::atomic<unsigned long long> g_completed_execute_count{0};
static std::atomic<unsigned long long> g_reuse_host_buffer_pool_count{0};
static std::atomic<unsigned long long> g_reuse_host_buffer_hit_count{0};
static std::atomic<size_t> g_reuse_host_buffer_pool_sequence{0};
static constexpr size_t kReusableHostBufferArenaAlignment = 256;
static constexpr size_t kReusableHostBufferArenaMinEntries = 32;
static constexpr size_t kReusableHostBufferArenaMaxDirtyRanges = 64;
static constexpr size_t kReusableHostBufferArenaMergeGapBytes = 1024 * 1024;
static constexpr size_t kReusableHostBufferArenaCopyOverheadBytes =
    1024 * 1024;

struct ReusableHostBufferEntry {
    std::unique_ptr<PJRT_Buffer> owner;
    std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference;
    PJRT_Device* device = nullptr;
    PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
    std::vector<int64_t> dims;
    void* device_ptr = nullptr;
    size_t bytes = 0;
    size_t pool_sequence = 0;
    stream_executor::StreamExecutor* executor = nullptr;
    size_t arena_offset = 0;
    bool arena_member = false;
    bool arena_copy_pending = false;
    bool device_contents_valid = false;
    PJRT_Buffer* cached_view = nullptr;
    bool cached_view_in_use = false;
};

struct ReusableHostBufferArena {
    stream_executor::StreamExecutor* executor = nullptr;
    void* host_ptr = nullptr;
    stream_executor::DeviceMemoryBase device_memory;
    size_t total_bytes = 0;
    size_t entry_count = 0;
    size_t pending_inputs = 0;
    size_t pending_bytes = 0;
    size_t pending_parallel_inputs = 0;
    size_t pending_parallel_bytes = 0;
    double pending_host_pack_ms = 0.0;
    std::vector<musa::pjrt::ReusableHostBufferDirtyRange> pending_ranges;
    std::vector<ReusableHostBufferEntry*> pending_entries;
    size_t content_reuse_inputs = 0;
    size_t content_reuse_bytes = 0;
};

class ReusableHostBufferPackPool {
   public:
    explicit ReusableHostBufferPackPool(size_t total_threads)
        : total_threads_(std::max<size_t>(1, total_threads)) {
        workers_.reserve(total_threads_ - 1);
        for (size_t i = 1; i < total_threads_; ++i) {
            workers_.emplace_back([this]() { WorkerLoop(); });
        }
    }

    ~ReusableHostBufferPackPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            stop_ = true;
        }
        queue_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    void ParallelMemcpy(void* dst, const void* src, size_t bytes) {
        if (bytes == 0) return;
        if (total_threads_ <= 1 || workers_.empty()) {
            memcpy(dst, src, bytes);
            return;
        }

        const size_t chunk_bytes =
            bytes / total_threads_ + (bytes % total_threads_ != 0 ? 1 : 0);
        const size_t task_count =
            bytes / chunk_bytes + (bytes % chunk_bytes != 0 ? 1 : 0);
        if (task_count <= 1) {
            memcpy(dst, src, bytes);
            return;
        }

        auto state = std::make_shared<CopyWaitState>();
        state->remaining = task_count - 1;
        char* dst_bytes = static_cast<char*>(dst);
        const char* src_bytes = static_cast<const char*>(src);
        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            for (size_t i = 1; i < task_count; ++i) {
                const size_t offset = i * chunk_bytes;
                const size_t count = std::min(chunk_bytes, bytes - offset);
                tasks_.emplace_back(
                    [dst_bytes, src_bytes, offset, count, state]() {
                        memcpy(dst_bytes + offset, src_bytes + offset, count);
                        {
                            std::lock_guard<std::mutex> lock(state->mu);
                            --state->remaining;
                        }
                        state->cv.notify_one();
                    });
            }
        }
        queue_cv_.notify_all();

        memcpy(dst_bytes, src_bytes, std::min(chunk_bytes, bytes));
        std::unique_lock<std::mutex> lock(state->mu);
        state->cv.wait(lock, [&state]() { return state->remaining == 0; });
    }

   private:
    struct CopyWaitState {
        std::mutex mu;
        std::condition_variable cv;
        size_t remaining = 0;
    };

    void WorkerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mu_);
                queue_cv_.wait(lock,
                               [this]() { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    size_t total_threads_;
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
};

static std::mutex g_reuse_host_buffer_mu;
static auto* const g_reuse_host_buffers =
    new std::unordered_map<const void*,
                           std::unique_ptr<ReusableHostBufferEntry>>();
static auto* const g_cached_reused_buffer_views =
    new std::unordered_map<PJRT_Buffer*, ReusableHostBufferEntry*>();
static auto* const g_reuse_host_buffer_arenas =
    new std::unordered_map<stream_executor::StreamExecutor*,
                           std::unique_ptr<ReusableHostBufferArena>>();

struct InflightGate {
    std::mutex mu;
    std::condition_variable cv;
    size_t inflight = 0;
};

static InflightGate g_compile_gate;
static InflightGate g_transfer_gate;
static InflightGate g_execute_gate;

static PJRT_Error* WaitForEventViaCallback(PJRT_Event* event);
static bool ReadBoolEnv(const char* name, bool* value);
static PJRT_Error* TryReuseHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args, bool* reused);
static bool MaybePoolHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args);
static xla::Status FlushReusableHostBufferArena();

static int GetPositiveEnvInt(const char* env_name) {
    const char* env = std::getenv(env_name);
    if (env == nullptr || env[0] == '\0') return 0;

    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || (end != nullptr && *end != '\0') || value <= 0) {
        return 0;
    }
    if (value > std::numeric_limits<int>::max()) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(value);
}

static int GetCompileMaxInflight() {
    // MTGPU compilation uses process-global LLVM options and external toolchain
    // state. Avoid concurrent compile requests entering the MUSA backend by
    // default.
    const char* env = std::getenv("MUSA_PJRT_MAX_INFLIGHT_COMPILES");
    if (env == nullptr || env[0] == '\0') return 1;
    return GetPositiveEnvInt("MUSA_PJRT_MAX_INFLIGHT_COMPILES");
}

static int GetTransferMaxInflight() {
    // MUSA runtime can spin in libmusa.so when multiple TF inter-op workers
    // concurrently submit host-to-device transfers. Keep the default
    // conservative; set MUSA_PJRT_MAX_INFLIGHT_TRANSFERS=0 to disable.
    const char* env = std::getenv("MUSA_PJRT_MAX_INFLIGHT_TRANSFERS");
    if (env == nullptr || env[0] == '\0') return 1;
    return GetPositiveEnvInt("MUSA_PJRT_MAX_INFLIGHT_TRANSFERS");
}

static int GetExecuteMaxInflight() {
    // Default TF inter-op parallelism can concurrently enter MUSA execute paths
    // and trigger libmusa.so busy-yield. Serialize by default while allowing an
    // explicit override for runtime validation.
    const char* env = std::getenv("MUSA_PJRT_MAX_INFLIGHT_EXECUTES");
    if (env == nullptr || env[0] == '\0') return 1;
    return GetPositiveEnvInt("MUSA_PJRT_MAX_INFLIGHT_EXECUTES");
}

static double MsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

class ScopedInflightGate {
   public:
    ScopedInflightGate(InflightGate* gate, int max_inflight)
        : gate_(gate), active_(gate != nullptr && max_inflight > 0) {
        if (!active_) return;
        std::unique_lock<std::mutex> lock(gate_->mu);
        gate_->cv.wait(lock, [this, max_inflight]() {
            return gate_->inflight < static_cast<size_t>(max_inflight);
        });
        ++gate_->inflight;
    }

    ~ScopedInflightGate() {
        if (!active_) return;
        {
            std::lock_guard<std::mutex> lock(gate_->mu);
            --gate_->inflight;
        }
        gate_->cv.notify_one();
    }

   private:
    InflightGate* gate_;
    bool active_;
};


// 1. 修正 AddressableMemories 大小错误 (32 -> 40)
PJRT_Error* Proxy_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args) {
    if (!args) return nullptr;
    // Keep the caller-provided ABI size. Rewriting it makes the callee reject
    // the request with "expected 32, got 40" on the TF 2.15 runtime.
    return base_api.PJRT_Device_AddressableMemories(args);
}

static bool ShouldBypassEventDestroy() {
    const char* env = std::getenv("MUSA_PJRT_BYPASS_EVENT_DESTROY");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldBypassBufferDestroy() {
    const char* env = std::getenv("MUSA_PJRT_BYPASS_BUFFER_DESTROY");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldLogProxyDebug() {
    const char* env = std::getenv("MUSA_PJRT_DEBUG_LOG");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}


static bool ShouldSerializeExecuteSubmit() {
    const char* env = std::getenv("MUSA_PJRT_SERIALIZE_EXECUTE_SUBMIT");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldWaitEventBeforeDestroy() {
    const char* env = std::getenv("MUSA_PJRT_WAIT_EVENT_BEFORE_DESTROY");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldWaitBufferReadyBeforeDestroy() {
    const char* env = std::getenv("MUSA_PJRT_WAIT_BUFFER_READY_BEFORE_DESTROY");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldForceHostBufferCopy() {
    const char* env = std::getenv("MUSA_PJRT_FORCE_HOST_BUFFER_COPY");
    if (env == nullptr || env[0] == '\0') return true;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldWaitTransferDoneBeforeReturn() {
    const char* env = std::getenv("MUSA_PJRT_WAIT_TRANSFER_DONE");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldWaitExecuteDoneBeforeReturn() {
    const char* env = std::getenv("MUSA_PJRT_WAIT_EXECUTE_DONE");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldDropExecuteDeviceForCompat() {
    const char* env = std::getenv("MUSA_PJRT_DROP_EXECUTE_DEVICE");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static std::optional<std::string> GetXlaFlagValue(const std::string& flags,
                                                   const std::string& prefix) {
    std::istringstream stream(flags);
    std::string token;
    while (stream >> token) {
        if (token.rfind(prefix, 0) == 0) {
            return token.substr(prefix.size());
        }
    }
    return std::nullopt;
}

static bool HasXlaFlag(const std::string& flags, const std::string& flag) {
    std::istringstream stream(flags);
    std::string token;
    while (stream >> token) {
        if (token == flag) return true;
    }
    return false;
}

static bool SimpleAtoi32(const std::string& value, int* out) {
    if (out == nullptr || value.empty()) return false;
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') return false;
    if (parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(parsed);
    return true;
}

static std::optional<std::string> PatchCompileOptionsWithXlaDumpFlags(
        PJRT_Client_Compile_Args* args,
        bool should_log,
        unsigned long long log_count) {
    const char* xla_flags_env = std::getenv("XLA_FLAGS");
    if (xla_flags_env == nullptr || xla_flags_env[0] == '\0') {
        return std::nullopt;
    }

    const std::string xla_flags(xla_flags_env);
    const std::optional<std::string> dump_to =
        GetXlaFlagValue(xla_flags, "--xla_dump_to=");
    const bool dump_text = HasXlaFlag(xla_flags, "--xla_dump_hlo_as_text");
    const bool dump_long_text =
        HasXlaFlag(xla_flags, "--xla_dump_hlo_as_long_text");
    const std::optional<std::string> dump_pass_re =
        GetXlaFlagValue(xla_flags, "--xla_dump_hlo_pass_re=");
    const std::optional<std::string> dump_max_modules =
        GetXlaFlagValue(xla_flags, "--xla_dump_max_hlo_modules=");

    if (!dump_to.has_value() && !dump_text && !dump_long_text &&
        !dump_pass_re.has_value() && !dump_max_modules.has_value()) {
        return std::nullopt;
    }

    if (args == nullptr || args->compile_options == nullptr ||
        args->compile_options_size == 0) {
        if (should_log) {
            fprintf(stderr,
                    "[musa_pjrt] xla dump patch skipped: count=%llu empty compile_options\n",
                    log_count);
            fflush(stderr);
        }
        return std::nullopt;
    }

    xla::CompileOptionsProto options_proto;
    if (!options_proto.ParseFromArray(args->compile_options,
                                      static_cast<int>(args->compile_options_size))) {
        fprintf(stderr,
                "[musa_pjrt] xla dump patch failed: count=%llu cannot parse CompileOptionsProto size=%zu\n",
                log_count,
                static_cast<size_t>(args->compile_options_size));
        fflush(stderr);
        return std::nullopt;
    }

    xla::DebugOptions* debug_options =
        options_proto.mutable_executable_build_options()->mutable_debug_options();
    bool changed = false;

    if (dump_to.has_value() && debug_options->xla_dump_to() != *dump_to) {
        debug_options->set_xla_dump_to(*dump_to);
        changed = true;
    }
    if (dump_text && !debug_options->xla_dump_hlo_as_text()) {
        debug_options->set_xla_dump_hlo_as_text(true);
        changed = true;
    }
    if (dump_long_text && !debug_options->xla_dump_hlo_as_long_text()) {
        debug_options->set_xla_dump_hlo_as_long_text(true);
        changed = true;
    }
    if (dump_pass_re.has_value() &&
        debug_options->xla_dump_hlo_pass_re() != *dump_pass_re) {
        debug_options->set_xla_dump_hlo_pass_re(*dump_pass_re);
        changed = true;
    }
    if (dump_max_modules.has_value()) {
        int max_modules = 0;
        if (SimpleAtoi32(*dump_max_modules, &max_modules) &&
            debug_options->xla_dump_max_hlo_modules() != max_modules) {
            debug_options->set_xla_dump_max_hlo_modules(max_modules);
            changed = true;
        }
    }

    if (should_log || changed) {
        fprintf(stderr,
                "[musa_pjrt] xla dump options: count=%llu patched=%d dump_to=%s text=%d long_text=%d pass_re=%s max_modules=%d\n",
                log_count,
                changed ? 1 : 0,
                debug_options->xla_dump_to().c_str(),
                debug_options->xla_dump_hlo_as_text() ? 1 : 0,
                debug_options->xla_dump_hlo_as_long_text() ? 1 : 0,
                debug_options->xla_dump_hlo_pass_re().c_str(),
                debug_options->xla_dump_max_hlo_modules());
        fflush(stderr);
    }

    if (!changed) return std::nullopt;

    std::string serialized;
    if (!options_proto.SerializeToString(&serialized)) {
        fprintf(stderr,
                "[musa_pjrt] xla dump patch failed: count=%llu cannot serialize CompileOptionsProto\n",
                log_count);
        fflush(stderr);
        return std::nullopt;
    }
    return serialized;
}

static std::string SanitizeFilenamePart(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '_' || c == '-' || c == '.') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) return "unnamed";
    return out;
}

static void DumpOptimizedProgramAfterCompile(PJRT_Client_Compile_Args* args,
                                             bool should_log,
                                             unsigned long long log_count) {
    const char* xla_flags_env = std::getenv("XLA_FLAGS");
    if (xla_flags_env == nullptr || xla_flags_env[0] == '\0') return;
    const std::optional<std::string> dump_to =
        GetXlaFlagValue(std::string(xla_flags_env), "--xla_dump_to=");
    if (!dump_to.has_value() || dump_to->empty() || *dump_to == "-") return;
    if (args == nullptr || args->executable == nullptr) return;
    if (base_api.PJRT_LoadedExecutable_GetExecutable == nullptr ||
        base_api.PJRT_Executable_OptimizedProgram == nullptr) {
        if (should_log) {
            fprintf(stderr,
                    "[musa_pjrt] optimized HLO dump skipped: count=%llu API unavailable\n",
                    log_count);
            fflush(stderr);
        }
        return;
    }

    PJRT_LoadedExecutable_GetExecutable_Args get_exec_args;
    get_exec_args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get_exec_args.priv = nullptr;
    get_exec_args.loaded_executable = args->executable;
    get_exec_args.executable = nullptr;
    PJRT_Error* get_exec_err =
        base_api.PJRT_LoadedExecutable_GetExecutable(&get_exec_args);
    if (get_exec_err != nullptr || get_exec_args.executable == nullptr) {
        fprintf(stderr,
                "[musa_pjrt] optimized HLO dump skipped: count=%llu get executable err=%p executable=%p\n",
                log_count,
                static_cast<void*>(get_exec_err),
                static_cast<void*>(get_exec_args.executable));
        fflush(stderr);
        return;
    }
    auto destroy_executable = [&]() {
        if (base_api.PJRT_Executable_Destroy == nullptr ||
            get_exec_args.executable == nullptr) {
            return;
        }
        PJRT_Executable_Destroy_Args destroy_args;
        destroy_args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
        destroy_args.priv = nullptr;
        destroy_args.executable = get_exec_args.executable;
        PJRT_Error* destroy_err = base_api.PJRT_Executable_Destroy(&destroy_args);
        if (destroy_err != nullptr && should_log) {
            fprintf(stderr,
                    "[musa_pjrt] optimized HLO dump cleanup: count=%llu destroy err=%p\n",
                    log_count,
                    static_cast<void*>(destroy_err));
            fflush(stderr);
        }
        get_exec_args.executable = nullptr;
    };

    PJRT_Program program;
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.priv = nullptr;
    program.code = nullptr;
    program.code_size = 0;
    program.format = nullptr;
    program.format_size = 0;

    PJRT_Executable_OptimizedProgram_Args optimized_args;
    optimized_args.struct_size = PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE;
    optimized_args.priv = nullptr;
    optimized_args.executable = get_exec_args.executable;
    optimized_args.program = &program;

    PJRT_Error* size_err =
        base_api.PJRT_Executable_OptimizedProgram(&optimized_args);
    if (size_err != nullptr || program.code_size == 0) {
        fprintf(stderr,
                "[musa_pjrt] optimized HLO dump skipped: count=%llu size err=%p code_size=%zu\n",
                log_count,
                static_cast<void*>(size_err),
                static_cast<size_t>(program.code_size));
        fflush(stderr);
        destroy_executable();
        return;
    }

    std::string serialized(program.code_size, '\0');
    program.code = serialized.data();
    PJRT_Error* program_err =
        base_api.PJRT_Executable_OptimizedProgram(&optimized_args);
    if (program_err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] optimized HLO dump skipped: count=%llu program err=%p\n",
                log_count,
                static_cast<void*>(program_err));
        fflush(stderr);
        destroy_executable();
        return;
    }

    std::string format;
    if (program.format != nullptr && program.format_size > 0) {
        format.assign(program.format, program.format_size);
    }

    xla::HloModuleProtoWithConfig proto;
    if (!proto.ParseFromString(serialized)) {
        fprintf(stderr,
                "[musa_pjrt] optimized HLO dump skipped: count=%llu parse HloModuleProtoWithConfig failed format=%s bytes=%zu\n",
                log_count,
                format.c_str(),
                serialized.size());
        fflush(stderr);
        destroy_executable();
        return;
    }

    auto module_or = xla::HloModule::CreateFromProtoWithConfig(proto);
    if (!module_or.ok()) {
        fprintf(stderr,
                "[musa_pjrt] optimized HLO dump skipped: count=%llu create HloModule failed: %s\n",
                log_count,
                module_or.status().ToString().c_str());
        fflush(stderr);
        destroy_executable();
        return;
    }

    const std::string module_name = SanitizeFilenamePart((*module_or)->name());
    char filename[4096];
    snprintf(filename,
             sizeof(filename),
             "%s/module_%04llu.%s.after_optimizations.txt",
             dump_to->c_str(),
             log_count,
             module_name.c_str());

    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out.good()) {
        fprintf(stderr,
                "[musa_pjrt] optimized HLO dump failed: count=%llu cannot open %s\n",
                log_count,
                filename);
        fflush(stderr);
        destroy_executable();
        return;
    }
    out << (*module_or)->ToString();
    out.close();

    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] optimized HLO dumped: count=%llu format=%s bytes=%zu file=%s\n",
                log_count,
                format.c_str(),
                serialized.size(),
                filename);
        fflush(stderr);
    }
    destroy_executable();
}

PJRT_Error* Proxy_Client_Compile(PJRT_Client_Compile_Args* args) {
    const int max_inflight_compiles = GetCompileMaxInflight();
    unsigned long long log_count =
        g_client_compile_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
    bool should_log = ShouldLogProxyDebug();
    const auto wait_start = std::chrono::steady_clock::now();
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] client compile wait: count=%llu max_inflight=%d struct_size=%zu program=%p code_size=%zu options_size=%zu\n",
                log_count,
                max_inflight_compiles,
                args ? static_cast<size_t>(args->struct_size) : 0,
                args ? static_cast<const void*>(args->program) : nullptr,
                (args && args->program) ? static_cast<size_t>(args->program->code_size) : 0,
                args ? static_cast<size_t>(args->compile_options_size) : 0);
        fflush(stderr);
    }
    ScopedInflightGate gate(&g_compile_gate, max_inflight_compiles);
    const double wait_ms = MsSince(wait_start);
    const auto compile_start = std::chrono::steady_clock::now();
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] client compile begin: count=%llu max_inflight=%d wait_ms=%.3f struct_size=%zu program=%p code_size=%zu options_size=%zu\n",
                log_count,
                max_inflight_compiles,
                wait_ms,
                args ? static_cast<size_t>(args->struct_size) : 0,
                args ? static_cast<const void*>(args->program) : nullptr,
                (args && args->program) ? static_cast<size_t>(args->program->code_size) : 0,
                args ? static_cast<size_t>(args->compile_options_size) : 0);
        fflush(stderr);
    }
    std::optional<std::string> patched_compile_options =
        PatchCompileOptionsWithXlaDumpFlags(args, should_log, log_count);
    if (patched_compile_options.has_value()) {
        args->compile_options = patched_compile_options->c_str();
        args->compile_options_size = patched_compile_options->size();
    }
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] client compile call base: count=%llu client=%p program_format=%.*s code=%p code_size=%zu options=%p options_size=%zu\n",
                log_count,
                args ? static_cast<void*>(args->client) : nullptr,
                (args && args->program && args->program->format)
                    ? static_cast<int>(args->program->format_size)
                    : 0,
                (args && args->program && args->program->format)
                    ? args->program->format
                    : "",
                (args && args->program) ? static_cast<void*>(args->program->code)
                                        : nullptr,
                (args && args->program) ? static_cast<size_t>(args->program->code_size)
                                        : 0,
                args ? static_cast<const void*>(args->compile_options) : nullptr,
                args ? static_cast<size_t>(args->compile_options_size) : 0);
        fflush(stderr);
    }
    PJRT_Error* err = base_api.PJRT_Client_Compile(args);
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] client compile base returned: count=%llu err=%p executable=%p\n",
                log_count,
                static_cast<void*>(err),
                args ? static_cast<void*>(args->executable) : nullptr);
        fflush(stderr);
    }
    if (err == nullptr) {
        DumpOptimizedProgramAfterCompile(args, should_log, log_count);
    }
    if (should_log || err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] client compile returned: count=%llu err=%p executable=%p compile_ms=%.3f total_ms=%.3f\n",
                log_count,
                static_cast<void*>(err),
                (args ? static_cast<void*>(args->executable) : nullptr),
                MsSince(compile_start),
                MsSince(wait_start));
        fflush(stderr);
    }
    return err;
}

PJRT_Error* Proxy_Client_BufferFromHostBuffer(
        PJRT_Client_BufferFromHostBuffer_Args* args) {
    const int max_inflight_transfers = GetTransferMaxInflight();
    ScopedInflightGate gate(&g_transfer_gate, max_inflight_transfers);
    const bool should_log = ShouldLogProxyDebug();
    const unsigned long long log_count = should_log
        ? g_buffer_from_host_log_count.fetch_add(
              1, std::memory_order_relaxed) + 1
        : 0;
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] buffer-from-host begin: count=%llu max_inflight=%d struct_size=%zu data=%p type=%d num_dims=%zu semantics=%d device=%p\n",
                log_count, max_inflight_transfers,
                args ? static_cast<size_t>(args->struct_size) : 0,
                args ? args->data : nullptr,
                args ? static_cast<int>(args->type) : -1,
                args ? static_cast<size_t>(args->num_dims) : 0,
                args ? static_cast<int>(args->host_buffer_semantics) : -1,
                args ? static_cast<void*>(args->device) : nullptr);
        fflush(stderr);
    }

    const auto transfer_start = std::chrono::steady_clock::now();
    bool reused = false;
    PJRT_Error* err = TryReuseHostBuffer(args, &reused);
    if (err != nullptr || reused) {
        if (should_log || err != nullptr) {
            fprintf(stderr,
                    "[musa_pjrt] buffer-from-host returned: count=%llu err=%p done_event=%p buffer=%p force_copy=%d wait_done=%d reused=%d pooled=0 transfer_ms=%.3f\n",
                    log_count, static_cast<void*>(err),
                    args ? static_cast<void*>(args->done_with_host_buffer)
                         : nullptr,
                    args ? static_cast<void*>(args->buffer) : nullptr,
                    ShouldForceHostBufferCopy() ? 1 : 0,
                    ShouldWaitTransferDoneBeforeReturn() ? 1 : 0,
                    reused ? 1 : 0, MsSince(transfer_start));
            fflush(stderr);
        }
        return err;
    }

    PJRT_HostBufferSemantics original_semantics =
        args ? args->host_buffer_semantics
             : PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    if (args != nullptr && ShouldForceHostBufferCopy()) {
        args->host_buffer_semantics =
            PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    }

    err = base_api.PJRT_Client_BufferFromHostBuffer(args);
    if (args != nullptr) {
        args->host_buffer_semantics = original_semantics;
    }

    PJRT_Error* wait_err = nullptr;
    if (err == nullptr && args != nullptr &&
        args->done_with_host_buffer != nullptr &&
        ShouldWaitTransferDoneBeforeReturn()) {
        wait_err = WaitForEventViaCallback(args->done_with_host_buffer);
        if (wait_err != nullptr && ShouldLogProxyDebug()) {
            fprintf(stderr,
                    "[musa_pjrt] buffer-from-host wait returned error: count=%llu err=%p done_event=%p\n",
                    log_count, static_cast<void*>(wait_err),
                    static_cast<void*>(args->done_with_host_buffer));
            fflush(stderr);
        }
    }

    bool pooled = false;
    if (err == nullptr && wait_err == nullptr) {
        pooled = MaybePoolHostBuffer(args);
    }

    if (should_log || err != nullptr || wait_err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] buffer-from-host returned: count=%llu err=%p done_event=%p buffer=%p force_copy=%d wait_done=%d reused=0 pooled=%d transfer_ms=%.3f\n",
                log_count, static_cast<void*>(err != nullptr ? err : wait_err),
                args ? static_cast<void*>(args->done_with_host_buffer)
                     : nullptr,
                args ? static_cast<void*>(args->buffer) : nullptr,
                ShouldForceHostBufferCopy() ? 1 : 0,
                ShouldWaitTransferDoneBeforeReturn() ? 1 : 0,
                pooled ? 1 : 0, MsSince(transfer_start));
        fflush(stderr);
    }
    return err != nullptr ? err : wait_err;
}

PJRT_Error* Proxy_Event_Destroy(PJRT_Event_Destroy_Args* args) {
    if (args == nullptr || args->event == nullptr) {
        return nullptr;
    }
    if (ShouldBypassEventDestroy()) {
        const char* bypass_env = std::getenv("MUSA_PJRT_BYPASS_EVENT_DESTROY");
        const char* bypass_env_text =
            (bypass_env != nullptr && bypass_env[0] != '\0') ? bypass_env : "<unset>";
        unsigned long long bypass_count =
            g_event_destroy_bypass_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (ShouldLogProxyDebug() &&
            (bypass_count <= 4 || (bypass_count % 100000) == 0)) {
            fprintf(stderr,
                    "[musa_pjrt] event destroy bypassed: count=%llu event=%p env=%s\n",
                    bypass_count,
                    static_cast<void*>(args->event),
                    bypass_env_text);
            fflush(stderr);
        }
        return nullptr;
    }
    if (ShouldWaitEventBeforeDestroy()) {
        PJRT_Error* wait_err = WaitForEventViaCallback(args->event);
        if (wait_err != nullptr && ShouldLogProxyDebug()) {
            fprintf(stderr,
                    "[musa_pjrt] event destroy wait returned error: err=%p event=%p\n",
                    static_cast<void*>(wait_err),
                    static_cast<void*>(args->event));
            fflush(stderr);
        }
    }
    return base_api.PJRT_Event_Destroy(args);
}

PJRT_Error* Proxy_Buffer_Destroy(
        PJRT_Buffer_Destroy_Args* args) {
    if (args == nullptr || args->buffer == nullptr) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(g_reuse_host_buffer_mu);
        auto cached_it = g_cached_reused_buffer_views->find(args->buffer);
        if (cached_it != g_cached_reused_buffer_views->end() &&
            cached_it->second != nullptr &&
            cached_it->second->cached_view == args->buffer) {
            cached_it->second->cached_view_in_use = false;
            return nullptr;
        }
    }
    if (ShouldBypassBufferDestroy()) {
        unsigned long long bypass_count =
            g_buffer_destroy_bypass_count.fetch_add(
                1, std::memory_order_relaxed) + 1;
        if (ShouldLogProxyDebug() &&
            (bypass_count <= 4 || (bypass_count % 100000) == 0)) {
            fprintf(stderr,
                    "[musa_pjrt] buffer destroy bypassed: count=%llu buffer=%p\n",
                    bypass_count, static_cast<void*>(args->buffer));
            fflush(stderr);
        }
        return nullptr;
    }

    if (ShouldWaitBufferReadyBeforeDestroy()) {
        PJRT_Buffer_ReadyEvent_Args ready_args;
        memset(&ready_args, 0, sizeof(ready_args));
        ready_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        ready_args.buffer = args->buffer;
        PJRT_Error* ready_err = base_api.PJRT_Buffer_ReadyEvent(&ready_args);
        if (ready_err == nullptr && ready_args.event != nullptr) {
            PJRT_Error* wait_err = WaitForEventViaCallback(ready_args.event);
            if (wait_err != nullptr && ShouldLogProxyDebug()) {
                fprintf(stderr,
                        "[musa_pjrt] buffer destroy wait returned error: err=%p buffer=%p ready_event=%p\n",
                        static_cast<void*>(wait_err),
                        static_cast<void*>(args->buffer),
                        static_cast<void*>(ready_args.event));
                fflush(stderr);
            }

            PJRT_Event_Destroy_Args destroy_ready_event_args;
            memset(&destroy_ready_event_args, 0,
                   sizeof(destroy_ready_event_args));
            destroy_ready_event_args.struct_size =
                PJRT_Event_Destroy_Args_STRUCT_SIZE;
            destroy_ready_event_args.event = ready_args.event;
            PJRT_Error* destroy_ready_event_err =
                base_api.PJRT_Event_Destroy(&destroy_ready_event_args);
            if (destroy_ready_event_err != nullptr &&
                ShouldLogProxyDebug()) {
                fprintf(stderr,
                        "[musa_pjrt] buffer destroy ready-event cleanup error: err=%p buffer=%p ready_event=%p\n",
                        static_cast<void*>(destroy_ready_event_err),
                        static_cast<void*>(args->buffer),
                        static_cast<void*>(ready_args.event));
                fflush(stderr);
            }
        } else if (ready_err != nullptr && ShouldLogProxyDebug()) {
            fprintf(stderr,
                    "[musa_pjrt] buffer ready-event query error: err=%p buffer=%p\n",
                    static_cast<void*>(ready_err),
                    static_cast<void*>(args->buffer));
            fflush(stderr);
        }
    }

    return base_api.PJRT_Buffer_Destroy(args);
}

struct BlockingEventState {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    PJRT_Error* error = nullptr;
};

static void OnReadyBlockCallback(PJRT_Error* error, void* user_arg) {
    auto* state = static_cast<BlockingEventState*>(user_arg);
    {
        std::lock_guard<std::mutex> lock(state->mu);
        state->done = true;
        state->error = error;
    }
    state->cv.notify_one();
}

static PJRT_Error* WaitForEventViaCallback(PJRT_Event* event) {
    if (event == nullptr) return nullptr;

    BlockingEventState state;
    PJRT_Event_OnReady_Args onready_args;
    memset(&onready_args, 0, sizeof(onready_args));
    onready_args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
    onready_args.event = event;
    onready_args.callback = OnReadyBlockCallback;
    onready_args.user_arg = &state;

    PJRT_Error* onready_err = base_api.PJRT_Event_OnReady(&onready_args);
    if (onready_err != nullptr) {
        return onready_err;
    }

    std::unique_lock<std::mutex> lock(state.mu);
    state.cv.wait(lock, [&state]() { return state.done; });
    return state.error;
}

static PJRT_Error* GetLoadedExecutableNumOutputs(
    PJRT_LoadedExecutable* loaded_executable,
    size_t* num_outputs) {
    if (loaded_executable == nullptr || num_outputs == nullptr) return nullptr;

    PJRT_LoadedExecutable_GetExecutable_Args get_exec_args;
    memset(&get_exec_args, 0, sizeof(get_exec_args));
    get_exec_args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get_exec_args.loaded_executable = loaded_executable;
    PJRT_Error* get_exec_err = base_api.PJRT_LoadedExecutable_GetExecutable(&get_exec_args);
    if (get_exec_err != nullptr) {
        return get_exec_err;
    }

    PJRT_Executable_NumOutputs_Args num_outputs_args;
    memset(&num_outputs_args, 0, sizeof(num_outputs_args));
    num_outputs_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    num_outputs_args.executable = get_exec_args.executable;
    PJRT_Error* num_outputs_err = base_api.PJRT_Executable_NumOutputs(&num_outputs_args);

    PJRT_Executable_Destroy_Args destroy_exec_args;
    memset(&destroy_exec_args, 0, sizeof(destroy_exec_args));
    destroy_exec_args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    destroy_exec_args.executable = get_exec_args.executable;
    PJRT_Error* destroy_exec_err = base_api.PJRT_Executable_Destroy(&destroy_exec_args);

    if (num_outputs_err != nullptr) {
        return num_outputs_err;
    }
    if (destroy_exec_err != nullptr) {
        return destroy_exec_err;
    }

    *num_outputs = num_outputs_args.num_outputs;
    return nullptr;
}

PJRT_Error* Proxy_LoadedExecutable_Execute(
        PJRT_LoadedExecutable_Execute_Args* args) {
    if (!args) return base_api.PJRT_LoadedExecutable_Execute(args);

    const int max_inflight_executes = GetExecuteMaxInflight();
    ScopedInflightGate gate(&g_execute_gate, max_inflight_executes);

    xla::Status arena_status = FlushReusableHostBufferArena();
    if (!arena_status.ok()) return new PJRT_Error{arena_status};

    if (!args->options) return base_api.PJRT_LoadedExecutable_Execute(args);

    PJRT_ExecuteOptions* original_options = args->options;
    PJRT_Device* original_execute_device = args->execute_device;
    const bool drop_execute_device =
        original_execute_device != nullptr && ShouldDropExecuteDeviceForCompat();
    const bool should_log = ShouldLogProxyDebug();
    const unsigned long long log_count = should_log
        ? g_execute_log_count.fetch_add(1, std::memory_order_relaxed) + 1
        : 0;

    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] execute begin: count=%llu max_inflight=%d args_size=%zu options_size=%zu num_devices=%zu events=%p execute_device=%p drop_execute_device=%d serialize_submit=%d wait_done=%d\n",
                log_count, max_inflight_executes,
                static_cast<size_t>(args->struct_size),
                static_cast<size_t>(original_options->struct_size),
                static_cast<size_t>(args->num_devices),
                static_cast<void*>(args->device_complete_events),
                static_cast<void*>(original_execute_device),
                drop_execute_device ? 1 : 0,
                ShouldSerializeExecuteSubmit() ? 1 : 0,
                ShouldWaitExecuteDoneBeforeReturn() ? 1 : 0);
        fflush(stderr);
    }

    const auto execute_start = std::chrono::steady_clock::now();
    PJRT_Error* err = nullptr;
    if (drop_execute_device) {
        args->execute_device = nullptr;
    }
    if (ShouldSerializeExecuteSubmit()) {
        std::lock_guard<std::mutex> submit_lock(g_execute_submit_mu);
        err = base_api.PJRT_LoadedExecutable_Execute(args);
    } else {
        err = base_api.PJRT_LoadedExecutable_Execute(args);
    }
    if (drop_execute_device) {
        args->execute_device = original_execute_device;
    }

    if (err == nullptr && args->device_complete_events != nullptr &&
        ShouldWaitExecuteDoneBeforeReturn()) {
        for (size_t device_index = 0; device_index < args->num_devices;
             ++device_index) {
            PJRT_Event* event = args->device_complete_events[device_index];
            if (event == nullptr) continue;
            PJRT_Error* wait_err = WaitForEventViaCallback(event);
            if (wait_err != nullptr) {
                err = wait_err;
                if (ShouldLogProxyDebug()) {
                    fprintf(stderr,
                            "[musa_pjrt] execute wait returned error: count=%llu err=%p device_index=%zu event=%p\n",
                            log_count, static_cast<void*>(wait_err),
                            device_index, static_cast<void*>(event));
                    fflush(stderr);
                }
                break;
            }
        }
    }

    if (should_log || err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] execute returned: count=%llu err=%p events=%p first_event=%p execute_ms=%.3f\n",
                log_count, static_cast<void*>(err),
                static_cast<void*>(args->device_complete_events),
                (args->device_complete_events != nullptr &&
                 args->num_devices > 0)
                    ? static_cast<void*>(args->device_complete_events[0])
                    : nullptr,
                MsSince(execute_start));
        fflush(stderr);
    }

    if (err == nullptr) {
        g_completed_execute_count.fetch_add(1, std::memory_order_release);
    }
    return err;
}


static bool ReadBoolEnv(const char* name, bool* value) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return false;

    std::string text(env);
    if (text == "1" || text == "true" || text == "TRUE" ||
        text == "True" || text == "yes" || text == "YES" ||
        text == "on" || text == "ON") {
        *value = true;
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" ||
        text == "False" || text == "no" || text == "NO" ||
        text == "off" || text == "OFF") {
        *value = false;
        return true;
    }

    fprintf(stderr, "[MUSA PJRT] ignoring invalid %s=%s\n", name, env);
    return false;
}

static bool ReadDoubleEnv(const char* name, double* value) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return false;

    char* end = nullptr;
    double parsed = std::strtod(env, &end);
    if (end == env || *end != '\0' || parsed <= 0.0) {
        fprintf(stderr, "[MUSA PJRT] ignoring invalid %s=%s\n", name, env);
        return false;
    }

    *value = parsed;
    return true;
}

static void ApplyMusaAllocatorEnv(xla::GpuAllocatorConfig* allocator_config) {
    // Keep execution concurrency intact. These knobs only control the device
    // memory pool reservation policy used by the PJRT GPU client.
    allocator_config->kind = xla::GpuAllocatorConfig::Kind::kBFC;
    allocator_config->preallocate = false;

    const char* allocator_kind = std::getenv("MUSA_PJRT_ALLOCATOR");
    if (allocator_kind != nullptr && allocator_kind[0] != '\0') {
        std::string kind(allocator_kind);
        for (char& ch : kind) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        if (kind == "platform") {
            allocator_config->kind = xla::GpuAllocatorConfig::Kind::kPlatform;
        } else if (kind == "default") {
            allocator_config->kind = xla::GpuAllocatorConfig::Kind::kDefault;
        } else if (kind == "bfc") {
            allocator_config->kind = xla::GpuAllocatorConfig::Kind::kBFC;
        } else {
            fprintf(stderr, "[MUSA PJRT] ignoring invalid MUSA_PJRT_ALLOCATOR=%s\n",
                    allocator_kind);
        }
    }

    bool preallocate = false;
    if (ReadBoolEnv("MUSA_PJRT_PREALLOCATE", &preallocate)) {
        allocator_config->preallocate = preallocate;
    }

    double memory_fraction = 0.0;
    if (ReadDoubleEnv("MUSA_PJRT_MEMORY_FRACTION", &memory_fraction)) {
        allocator_config->memory_fraction = memory_fraction;
    }

}

static bool ShouldReuseHostBuffers() {
    const char* env = std::getenv("MUSA_PJRT_REUSE_HOST_BUFFERS");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldLogReusableHostBufferDiagnostics() {
    bool enabled = false;
    return ReadBoolEnv("MUSA_PJRT_REUSE_HOST_BUFFERS_DIAGNOSTICS",
                       &enabled) &&
           enabled;
}

static bool ShouldSampleReusableHostBufferCount(unsigned long long count) {
    return count <= 8 || count % 100 == 0;
}

static bool ShouldReuseHostBuffersArena() {
    const char* env = std::getenv("MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA");
    if (env == nullptr || env[0] == '\0') return false;
    const bool enabled = strcmp(env, "0") != 0 &&
                         strcmp(env, "false") != 0 &&
                         strcmp(env, "False") != 0 &&
                         strcmp(env, "FALSE") != 0;
    return enabled && ShouldReuseHostBuffers() &&
           ShouldWaitTransferDoneBeforeReturn() &&
           ShouldWaitExecuteDoneBeforeReturn() &&
           GetTransferMaxInflight() == 1 && GetExecuteMaxInflight() == 1;
}

static bool ShouldCacheReusedBufferViews() {
    const char* env =
        std::getenv("MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS");
    if (env == nullptr || env[0] == '\0') return false;
    const bool enabled = strcmp(env, "0") != 0 &&
                         strcmp(env, "false") != 0 &&
                         strcmp(env, "False") != 0 &&
                         strcmp(env, "FALSE") != 0;
    return enabled && ShouldReuseHostBuffersArena() &&
           ShouldWaitTransferDoneBeforeReturn() &&
           ShouldWaitExecuteDoneBeforeReturn() &&
           GetTransferMaxInflight() == 1 && GetExecuteMaxInflight() == 1;
}

static bool ShouldTrustCachedReusableHostBufferViewLifetime() {
    static const bool enabled = []() {
        bool value = false;
        return ReadBoolEnv(
                   "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS_TRUST_LIFETIME",
                   &value) &&
               value;
    }();
    return enabled;
}

static bool ShouldTrustReusableHostBufferContents() {
    static const bool enabled = []() {
        bool value = false;
        return ReadBoolEnv("MUSA_PJRT_REUSE_HOST_BUFFERS_TRUST_CONTENTS",
                           &value) &&
               value;
    }();
    return enabled && ShouldReuseHostBuffersArena() &&
           ShouldCacheReusedBufferViews() &&
           ShouldTrustCachedReusableHostBufferViewLifetime() &&
           ShouldWaitTransferDoneBeforeReturn() &&
           ShouldWaitExecuteDoneBeforeReturn();
}

static int GetReusableHostBufferArenaPackThreads() {
    const int configured = GetPositiveEnvInt(
        "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_THREADS");
    return std::max(1, std::min(configured > 0 ? configured : 4, 16));
}

static size_t GetReusableHostBufferArenaPackMinBytes() {
    constexpr size_t kDefaultMinBytes = 1024 * 1024;
    const char* env = std::getenv(
        "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_MIN_BYTES");
    if (env == nullptr || env[0] == '\0') return kDefaultMinBytes;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return kDefaultMinBytes;
    }
    return static_cast<size_t>(parsed);
}

static bool ShouldParallelPackReusableHostBuffer(size_t bytes) {
    const char* env = std::getenv(
        "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PARALLEL_PACK");
    if (env == nullptr || env[0] == '\0') return false;
    const bool enabled = strcmp(env, "0") != 0 &&
                         strcmp(env, "false") != 0 &&
                         strcmp(env, "False") != 0 &&
                         strcmp(env, "FALSE") != 0;
    return enabled && ShouldReuseHostBuffersArena() &&
           GetReusableHostBufferArenaPackThreads() > 1 &&
           bytes >= GetReusableHostBufferArenaPackMinBytes();
}

static ReusableHostBufferPackPool* GetReusableHostBufferPackPool() {
    static ReusableHostBufferPackPool* const pool =
        new ReusableHostBufferPackPool(
            static_cast<size_t>(GetReusableHostBufferArenaPackThreads()));
    return pool;
}

static bool AlignReusableHostBufferArenaOffset(size_t value,
                                                size_t* aligned) {
    if (aligned == nullptr) return false;
    constexpr size_t mask = kReusableHostBufferArenaAlignment - 1;
    if (value > std::numeric_limits<size_t>::max() - mask) return false;
    *aligned = (value + mask) & ~mask;
    return true;
}

static bool ShouldUseReusableHostBufferArenaPoolOrderLayout() {
    static const bool enabled = []() {
        bool value = false;
        return ReadBoolEnv(
                   "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_POOL_ORDER_LAYOUT",
                   &value) &&
               value;
    }();
    return enabled && ShouldReuseHostBuffersArena();
}

static ReusableHostBufferArena* GetOrCreateReusableHostBufferArenaLocked(
        stream_executor::StreamExecutor* executor) {
    if (!ShouldReuseHostBuffersArena() || executor == nullptr) return nullptr;
    if (g_completed_execute_count.load(std::memory_order_acquire) == 0) {
        return nullptr;
    }

    auto arena_it = g_reuse_host_buffer_arenas->find(executor);
    if (arena_it != g_reuse_host_buffer_arenas->end()) {
        return arena_it->second.get();
    }

    std::vector<std::pair<uintptr_t, ReusableHostBufferEntry*>> entries;
    entries.reserve(g_reuse_host_buffers->size());
    for (auto& item : *g_reuse_host_buffers) {
        ReusableHostBufferEntry* entry = item.second.get();
        if (entry->executor == executor && entry->bytes > 0 &&
            !entry->arena_member) {
            entries.emplace_back(reinterpret_cast<uintptr_t>(item.first),
                                 entry);
        }
    }
    if (entries.size() < kReusableHostBufferArenaMinEntries) return nullptr;

    const bool pool_order_layout =
        ShouldUseReusableHostBufferArenaPoolOrderLayout();
    std::vector<musa::pjrt::ReusableHostBufferArenaEntryOrder>
        entry_order_inputs;
    entry_order_inputs.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        entry_order_inputs.push_back(
            {entries[i].first, entries[i].second->pool_sequence, i});
    }
    const std::vector<size_t> entry_order =
        musa::pjrt::OrderReusableHostBufferArenaEntries(
            std::move(entry_order_inputs), pool_order_layout);
    if (entry_order.size() != entries.size()) return nullptr;

    std::vector<std::pair<uintptr_t, ReusableHostBufferEntry*>>
        ordered_entries;
    ordered_entries.reserve(entries.size());
    for (const size_t index : entry_order) {
        if (index >= entries.size()) return nullptr;
        ordered_entries.push_back(entries[index]);
    }
    entries = std::move(ordered_entries);

    std::vector<size_t> offsets;
    offsets.reserve(entries.size());
    size_t next_offset = 0;
    for (const auto& item : entries) {
        size_t aligned_offset = 0;
        if (!AlignReusableHostBufferArenaOffset(next_offset,
                                                &aligned_offset) ||
            item.second->bytes >
                std::numeric_limits<size_t>::max() - aligned_offset) {
            return nullptr;
        }
        offsets.push_back(aligned_offset);
        next_offset = aligned_offset + item.second->bytes;
    }

    size_t total_bytes = 0;
    if (!AlignReusableHostBufferArenaOffset(next_offset, &total_bytes) ||
        total_bytes == 0 ||
        total_bytes >
            static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return nullptr;
    }

    void* host_ptr = executor->HostMemoryAllocate(total_bytes);
    if (host_ptr == nullptr) return nullptr;

    stream_executor::DeviceMemory<uint8_t> device_memory =
        executor->AllocateArray<uint8_t>(total_bytes);
    if (device_memory.is_null()) {
        executor->HostMemoryDeallocate(host_ptr);
        return nullptr;
    }

    auto arena = std::make_unique<ReusableHostBufferArena>();
    arena->executor = executor;
    arena->host_ptr = host_ptr;
    arena->device_memory = device_memory;
    arena->total_bytes = total_bytes;
    arena->entry_count = entries.size();

    const uintptr_t device_base =
        reinterpret_cast<uintptr_t>(device_memory.opaque());
    for (size_t i = 0; i < entries.size(); ++i) {
        ReusableHostBufferEntry* entry = entries[i].second;
        entry->arena_offset = offsets[i];
        entry->arena_member = true;
        entry->device_contents_valid = false;
        entry->device_ptr = reinterpret_cast<void*>(device_base + offsets[i]);
    }

    ReusableHostBufferArena* arena_ptr = arena.get();
    g_reuse_host_buffer_arenas->emplace(executor, std::move(arena));
    if (ShouldLogReusableHostBufferDiagnostics()) {
        fprintf(stderr,
                "[MUSA_PJRT_REUSE_HOST_BUFFERS] action=arena_create "
                "entries=%zu arena_mib=%.3f alignment=%zu layout=%s host=%p "
                "device=%p\n",
                arena_ptr->entry_count,
                static_cast<double>(arena_ptr->total_bytes) /
                    (1024.0 * 1024.0),
                kReusableHostBufferArenaAlignment,
                pool_order_layout ? "pool_order" : "host_pointer",
                arena_ptr->host_ptr,
                arena_ptr->device_memory.opaque());
        fflush(stderr);
    }
    return arena_ptr;
}

static bool ShouldCopyReusableHostBufferArenaDirtyRanges() {
    static const bool enabled = []() {
        bool value = false;
        return ReadBoolEnv(
                   "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_DIRTY_RANGES",
                   &value) &&
               value;
    }();
    return enabled && ShouldReuseHostBuffersArena();
}

static xla::Status FlushReusableHostBufferArena() {
    if (!ShouldReuseHostBuffersArena()) return xla::OkStatus();

    std::lock_guard<std::mutex> lock(g_reuse_host_buffer_mu);
    xla::Status flush_status = xla::OkStatus();
    const bool diagnostics = ShouldLogReusableHostBufferDiagnostics();
    size_t flushed_arenas = 0;
    size_t flushed_inputs = 0;
    size_t active_bytes = 0;
    size_t arena_bytes = 0;
    size_t parallel_inputs = 0;
    size_t parallel_bytes = 0;
    size_t copy_ranges = 0;
    size_t transferred_bytes = 0;
    size_t full_copy_arenas = 0;
    size_t content_reuse_inputs = 0;
    size_t content_reuse_bytes = 0;
    double host_pack_ms = 0.0;
    double h2d_ms = 0.0;
    for (auto& item : *g_reuse_host_buffer_arenas) {
        ReusableHostBufferArena& arena = *item.second;
        content_reuse_inputs += arena.content_reuse_inputs;
        content_reuse_bytes += arena.content_reuse_bytes;
        arena.content_reuse_inputs = 0;
        arena.content_reuse_bytes = 0;
        if (arena.pending_inputs == 0) continue;

        musa::pjrt::ReusableHostBufferArenaCopyPlan copy_plan;
        if (ShouldCopyReusableHostBufferArenaDirtyRanges()) {
            copy_plan = musa::pjrt::PlanReusableHostBufferArenaCopies(
                arena.total_bytes, arena.pending_ranges,
                kReusableHostBufferArenaMaxDirtyRanges,
                kReusableHostBufferArenaMergeGapBytes,
                kReusableHostBufferArenaCopyOverheadBytes);
        }
        if (copy_plan.ranges.empty()) {
            copy_plan.copy_full_arena = true;
            copy_plan.transferred_bytes = arena.total_bytes;
            copy_plan.ranges.push_back({0, arena.total_bytes});
        }

        std::chrono::steady_clock::time_point h2d_start;
        if (diagnostics) h2d_start = std::chrono::steady_clock::now();
        xla::Status status = xla::OkStatus();
        for (const musa::pjrt::ReusableHostBufferDirtyRange& range :
             copy_plan.ranges) {
            void* device_ptr = static_cast<char*>(
                                   arena.device_memory.opaque()) +
                               range.offset;
            stream_executor::DeviceMemoryBase dst(device_ptr, range.bytes);
            status = arena.executor->SynchronousMemcpyH2D(
                static_cast<char*>(arena.host_ptr) + range.offset,
                static_cast<int64_t>(range.bytes), &dst);
            if (!status.ok()) break;
        }
        if (diagnostics) h2d_ms += MsSince(h2d_start);
        if (status.ok()) {
            for (ReusableHostBufferEntry* entry : arena.pending_entries) {
                entry->device_contents_valid = true;
            }
        } else {
            for (ReusableHostBufferEntry* entry : arena.pending_entries) {
                entry->device_contents_valid = false;
            }
        }
        ++flushed_arenas;
        flushed_inputs += arena.pending_inputs;
        active_bytes += arena.pending_bytes;
        arena_bytes += arena.total_bytes;
        parallel_inputs += arena.pending_parallel_inputs;
        parallel_bytes += arena.pending_parallel_bytes;
        copy_ranges += copy_plan.ranges.size();
        transferred_bytes += copy_plan.transferred_bytes;
        if (copy_plan.copy_full_arena) ++full_copy_arenas;
        host_pack_ms += arena.pending_host_pack_ms;
        arena.pending_inputs = 0;
        arena.pending_bytes = 0;
        arena.pending_parallel_inputs = 0;
        arena.pending_parallel_bytes = 0;
        arena.pending_host_pack_ms = 0.0;
        arena.pending_ranges.clear();
        arena.pending_entries.clear();
        if (!status.ok() && flush_status.ok()) flush_status = status;
    }

    for (auto& item : *g_reuse_host_buffers) {
        item.second->arena_copy_pending = false;
    }
    if (flushed_arenas > 0 && diagnostics) {
        const unsigned long long execute =
            g_completed_execute_count.load(std::memory_order_acquire) + 1;
        fprintf(stderr,
                "[MUSA_PJRT_REUSE_HOST_BUFFERS] action=arena_copy_summary "
                "execute=%llu arenas=%zu inputs=%zu active_mib=%.3f "
                "arena_mib=%.3f host_pack_ms=%.3f h2d_ms=%.3f "
                "copy_mode=%s ranges=%zu transferred_mib=%.3f "
                "parallel_pack_inputs=%zu parallel_pack_mib=%.3f\n",
                execute, flushed_arenas, flushed_inputs,
                static_cast<double>(active_bytes) / (1024.0 * 1024.0),
                static_cast<double>(arena_bytes) / (1024.0 * 1024.0),
                host_pack_ms, h2d_ms,
                full_copy_arenas > 0 ? "full" : "ranges", copy_ranges,
                static_cast<double>(transferred_bytes) /
                    (1024.0 * 1024.0),
                parallel_inputs,
                static_cast<double>(parallel_bytes) / (1024.0 * 1024.0));
        fflush(stderr);
    }
    if (content_reuse_inputs > 0 && diagnostics) {
        const unsigned long long execute =
            g_completed_execute_count.load(std::memory_order_acquire) + 1;
        fprintf(stderr,
                "[MUSA_PJRT_REUSE_HOST_BUFFERS] "
                "action=arena_content_reuse_summary execute=%llu "
                "skipped_inputs=%zu skipped_mib=%.3f\n",
                execute, content_reuse_inputs,
                static_cast<double>(content_reuse_bytes) /
                    (1024.0 * 1024.0));
        fflush(stderr);
    }
    return flush_status;
}

static bool IsDenseMajorToMinorDeviceLayout(
        const PJRT_Client_BufferFromHostBuffer_Args* args) {
    if (args == nullptr) return false;
    if (args->device_layout == nullptr) return true;
    if (args->device_layout->type != PJRT_Buffer_MemoryLayout_Type_Tiled) {
        return false;
    }

    const PJRT_Buffer_MemoryLayout_Tiled& tiled =
        args->device_layout->tiled;
    if (tiled.minor_to_major_size != args->num_dims ||
        tiled.num_tiles != 0) {
        return false;
    }
    if (args->num_dims > 0 && tiled.minor_to_major == nullptr) {
        return false;
    }
    for (size_t physical_dim = 0; physical_dim < args->num_dims;
         ++physical_dim) {
        const int64_t expected_logical_dim = static_cast<int64_t>(
            args->num_dims - 1 - physical_dim);
        if (tiled.minor_to_major[physical_dim] != expected_logical_dim) {
            return false;
        }
    }
    return true;
}

static bool ReusableHostBufferArgsSupported(
        const PJRT_Client_BufferFromHostBuffer_Args* args) {
    return ShouldReuseHostBuffers() &&
           ShouldWaitTransferDoneBeforeReturn() &&
           ShouldWaitExecuteDoneBeforeReturn() &&
           args != nullptr && args->client != nullptr && args->data != nullptr &&
           args->device != nullptr && args->memory == nullptr &&
           args->num_byte_strides == 0 &&
           IsDenseMajorToMinorDeviceLayout(args);
}

static bool ReusableHostBufferMetadataMatches(
        const ReusableHostBufferEntry& entry,
        const PJRT_Client_BufferFromHostBuffer_Args* args) {
    if (entry.device != args->device || entry.type != args->type ||
        entry.dims.size() != args->num_dims) {
        return false;
    }
    for (size_t i = 0; i < args->num_dims; ++i) {
        if (entry.dims[i] != args->dims[i]) return false;
    }
    return true;
}

static PJRT_Error* CreateReusableHostBufferView(
        PJRT_Client_BufferFromHostBuffer_Args* args,
        const ReusableHostBufferEntry& entry) {
    PJRT_Client_CreateViewOfDeviceBuffer_Args view_args;
    memset(&view_args, 0, sizeof(view_args));
    view_args.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
    view_args.client = args->client;
    view_args.device_buffer_ptr = entry.device_ptr;
    view_args.dims = args->dims;
    view_args.num_dims = args->num_dims;
    view_args.element_type = args->type;
    view_args.layout = args->device_layout;
    view_args.device = args->device;
    view_args.stream = 0;
    PJRT_Error* err = base_api.PJRT_Client_CreateViewOfDeviceBuffer(&view_args);
    if (err == nullptr) args->buffer = view_args.buffer;
    return err;
}

static void DestroyPjrtError(PJRT_Error* error) {
    if (error == nullptr) return;
    PJRT_Error_Destroy_Args destroy_args;
    memset(&destroy_args, 0, sizeof(destroy_args));
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    base_api.PJRT_Error_Destroy(&destroy_args);
}

static bool CachedReusableHostBufferViewIsDeleted(PJRT_Buffer* view) {
    if (view == nullptr || base_api.PJRT_Buffer_IsDeleted == nullptr) {
        return true;
    }
    PJRT_Buffer_IsDeleted_Args is_deleted_args;
    memset(&is_deleted_args, 0, sizeof(is_deleted_args));
    is_deleted_args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
    is_deleted_args.buffer = view;
    PJRT_Error* error = base_api.PJRT_Buffer_IsDeleted(&is_deleted_args);
    if (error != nullptr) {
        DestroyPjrtError(error);
        return true;
    }
    return is_deleted_args.is_deleted;
}

static PJRT_Error* TryReuseHostBuffer(
        PJRT_Client_BufferFromHostBuffer_Args* args, bool* reused) {
    *reused = false;
    if (!ReusableHostBufferArgsSupported(args)) return nullptr;

    std::lock_guard<std::mutex> lock(g_reuse_host_buffer_mu);
    auto it = g_reuse_host_buffers->find(args->data);
    if (it == g_reuse_host_buffers->end() ||
        !ReusableHostBufferMetadataMatches(*it->second, args)) {
        return nullptr;
    }

    ReusableHostBufferEntry& entry = *it->second;
    auto* se_device =
        static_cast<xla::PjRtStreamExecutorDevice*>(entry.device->device);
    if (se_device == nullptr || se_device->local_device_state() == nullptr ||
        se_device->local_device_state()->executor() == nullptr) {
        return nullptr;
    }

    stream_executor::StreamExecutor* executor =
        se_device->local_device_state()->executor();
    entry.executor = executor;

    bool used_arena_copy = false;
    if (ShouldReuseHostBuffersArena()) {
        ReusableHostBufferArena* arena =
            GetOrCreateReusableHostBufferArenaLocked(executor);
        if (arena != nullptr && entry.arena_member) {
            if (entry.arena_copy_pending) return nullptr;

            const bool trust_contents =
                ShouldTrustReusableHostBufferContents();
            const bool skip_content_copy =
                trust_contents && entry.device_contents_valid;
            if (skip_content_copy) {
                ++arena->content_reuse_inputs;
                arena->content_reuse_bytes += entry.bytes;
            } else {
                void* arena_dst = static_cast<char*>(arena->host_ptr) +
                                  entry.arena_offset;
                const bool parallel_pack =
                    ShouldParallelPackReusableHostBuffer(entry.bytes);
                const bool diagnostics =
                    ShouldLogReusableHostBufferDiagnostics();
                std::chrono::steady_clock::time_point pack_start;
                if (diagnostics) pack_start = std::chrono::steady_clock::now();
                if (parallel_pack) {
                    GetReusableHostBufferPackPool()->ParallelMemcpy(
                        arena_dst, args->data, entry.bytes);
                } else {
                    memcpy(arena_dst, args->data, entry.bytes);
                }
                entry.arena_copy_pending = true;
                ++arena->pending_inputs;
                arena->pending_bytes += entry.bytes;
                if (trust_contents) {
                    arena->pending_entries.push_back(&entry);
                }
                if (ShouldCopyReusableHostBufferArenaDirtyRanges()) {
                    arena->pending_ranges.push_back(
                        {entry.arena_offset, entry.bytes});
                }
                if (parallel_pack) {
                    ++arena->pending_parallel_inputs;
                    arena->pending_parallel_bytes += entry.bytes;
                }
                if (diagnostics) {
                    arena->pending_host_pack_ms += MsSince(pack_start);
                }
            }
            used_arena_copy = true;
        }
    }

    if (!used_arena_copy) {
        stream_executor::DeviceMemoryBase dst(entry.device_ptr, entry.bytes);
        xla::Status copy_status = executor->SynchronousMemcpyH2D(
            args->data, static_cast<int64_t>(entry.bytes), &dst);
        if (!copy_status.ok()) return new PJRT_Error{copy_status};
    }

    PJRT_Error* view_err = nullptr;
    bool reused_cached_view = false;
    const bool can_cache_view =
        used_arena_copy && ShouldCacheReusedBufferViews();
    const bool trust_cached_view =
        can_cache_view && ShouldTrustCachedReusableHostBufferViewLifetime();
    if (can_cache_view && entry.cached_view != nullptr) {
        if (entry.cached_view_in_use) {
            // A concurrent caller gets a temporary view.
        } else if (!trust_cached_view &&
                   CachedReusableHostBufferViewIsDeleted(entry.cached_view)) {
            PJRT_Buffer* deleted_view = entry.cached_view;
            g_cached_reused_buffer_views->erase(deleted_view);
            entry.cached_view = nullptr;
            entry.cached_view_in_use = false;

            PJRT_Buffer_Destroy_Args destroy_args;
            memset(&destroy_args, 0, sizeof(destroy_args));
            destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
            destroy_args.buffer = deleted_view;
            DestroyPjrtError(base_api.PJRT_Buffer_Destroy(&destroy_args));
        } else {
            args->buffer = entry.cached_view;
            entry.cached_view_in_use = true;
            reused_cached_view = true;
        }
    }
    if (!reused_cached_view) {
        view_err = CreateReusableHostBufferView(args, entry);
        if (view_err == nullptr && can_cache_view &&
            entry.cached_view == nullptr) {
            entry.cached_view = args->buffer;
            entry.cached_view_in_use = true;
            g_cached_reused_buffer_views->emplace(args->buffer, &entry);
        }
    }
    if (view_err != nullptr) return view_err;

    args->done_with_host_buffer =
        new PJRT_Event{xla::PjRtFuture<xla::Status>(xla::OkStatus())};
    *reused = true;
    if (ShouldLogReusableHostBufferDiagnostics()) {
        const unsigned long long count =
            g_reuse_host_buffer_hit_count.fetch_add(
                1, std::memory_order_relaxed) +
            1;
        if (ShouldSampleReusableHostBufferCount(count)) {
            fprintf(stderr,
                    "[MUSA_PJRT_REUSE_HOST_BUFFERS] action=hit count=%llu "
                    "host=%p device=%p bytes=%zu entries=%zu arena=%d\n",
                    count, args->data, entry.device_ptr, entry.bytes,
                    g_reuse_host_buffers->size(), used_arena_copy ? 1 : 0);
            fflush(stderr);
        }
    }
    return nullptr;
}

static bool MaybePoolHostBuffer(
        PJRT_Client_BufferFromHostBuffer_Args* args) {
    if (!ReusableHostBufferArgsSupported(args) || args->buffer == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_reuse_host_buffer_mu);
    if (g_reuse_host_buffers->find(args->data) !=
        g_reuse_host_buffers->end()) {
        return false;
    }

    PJRT_Buffer* original_buffer = args->buffer;
    auto bytes_or = original_buffer->buffer->GetOnDeviceSizeInBytes();
    if (!bytes_or.ok() || bytes_or.value() == 0) return false;
    auto reference_or = original_buffer->buffer->AcquireExternalReference();
    if (!reference_or.ok()) return false;

    auto entry = std::make_unique<ReusableHostBufferEntry>();
    entry->external_reference = std::move(reference_or.value());
    entry->device = args->device;
    entry->type = args->type;
    if (args->num_dims > 0) {
        entry->dims.assign(args->dims, args->dims + args->num_dims);
    }
    entry->device_ptr =
        entry->external_reference->OpaqueDeviceMemoryDataPointer();
    entry->bytes = bytes_or.value();
    if (entry->device_ptr == nullptr) return false;

    auto* se_device =
        static_cast<xla::PjRtStreamExecutorDevice*>(entry->device->device);
    if (se_device != nullptr && se_device->local_device_state() != nullptr) {
        entry->executor = se_device->local_device_state()->executor();
    }

    PJRT_Error* view_err = CreateReusableHostBufferView(args, *entry);
    if (view_err != nullptr) {
        DestroyPjrtError(view_err);
        args->buffer = original_buffer;
        return false;
    }

    entry->owner.reset(original_buffer);
    entry->pool_sequence = g_reuse_host_buffer_pool_sequence.fetch_add(
        1, std::memory_order_relaxed);
    void* const device_ptr = entry->device_ptr;
    const size_t bytes = entry->bytes;
    g_reuse_host_buffers->emplace(args->data, std::move(entry));
    if (ShouldLogReusableHostBufferDiagnostics()) {
        const unsigned long long count =
            g_reuse_host_buffer_pool_count.fetch_add(
                1, std::memory_order_relaxed) +
            1;
        if (ShouldSampleReusableHostBufferCount(count)) {
            fprintf(stderr,
                    "[MUSA_PJRT_REUSE_HOST_BUFFERS] action=pool "
                    "count=%llu host=%p device=%p bytes=%zu entries=%zu\n",
                    count, args->data, device_ptr, bytes,
                    g_reuse_host_buffers->size());
            fflush(stderr);
        }
    }
    return true;
}

static std::optional<std::set<int>> GetMusaAllowedDevices() {
    const char* env = std::getenv("MUSA_PJRT_ALLOWED_DEVICES");
    std::set<int> devices;

    if (env == nullptr || env[0] == '\0') {
        devices.insert(0);
        return devices;
    }

    std::string text(env);
    std::string lowered(text);
    for (char& ch : lowered) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (lowered == "all") {
        return std::nullopt;
    }

    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) continue;
        char* end = nullptr;
        long value = std::strtol(token.c_str(), &end, 10);
        if (end == token.c_str() || (end != nullptr && *end != '\0') || value < 0 ||
            value > std::numeric_limits<int>::max()) {
            fprintf(stderr,
                    "[MUSA PJRT] invalid MUSA_PJRT_ALLOWED_DEVICES=%s; "
                    "falling back to ordinal 0\n",
                    env);
            fflush(stderr);
            return std::set<int>{0};
        }
        devices.insert(static_cast<int>(value));
    }

    if (devices.empty()) devices.insert(0);
    return devices;
}
PJRT_Error* Musa_Client_Create(PJRT_Client_Create_Args* args) {
    if (std::getenv("MUSA_PJRT_USE_DEFAULT_DEBUG_OPTIONS") == nullptr) {
        setenv("MUSA_PJRT_USE_DEFAULT_DEBUG_OPTIONS", "1", 0);
    }
    xla::GpuAllocatorConfig allocator_config;
    ApplyMusaAllocatorEnv(&allocator_config);
    auto allowed_devices = GetMusaAllowedDevices();
    auto client_or = xla::GetStreamExecutorGpuClient(
        /*asynchronous=*/true, allocator_config, /*node_id=*/0,
        /*num_nodes=*/1, /*allowed_devices=*/allowed_devices,
        /*platform_name=*/std::string("MUSA"));
    if (!client_or.ok()) {
        fprintf(stderr, "[MUSA PJRT] MUSA init failed: %s\n",
                client_or.status().ToString().c_str());
        fflush(stderr);
        return new PJRT_Error{client_or.status()};
    }
    args->client = pjrt::CreateWrapperClient(std::move(client_or.value()));
    return nullptr;
}

// =========================================================================
// 🎯 发现钩子 (API Mounting)
// =========================================================================

__attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
    if (!base_api_initialized) {
        base_api = pjrt::CreatePjrtApi(
            Musa_Client_Create, nullptr, pjrt::PJRT_Plugin_Initialize_NoOp);
        base_api_initialized = true;
    }
    
    static PJRT_Api* truncated_api = nullptr;
    if (!truncated_api) {
        truncated_api = (PJRT_Api*)malloc(792);
        memset(truncated_api, 0, 792);
        size_t local_api_size = sizeof(PJRT_Api);
        size_t copy_size = (local_api_size < 792) ? local_api_size : 792;
        memcpy(truncated_api, &base_api, copy_size);
        truncated_api->struct_size = 792; 

        // 挂载补丁代理函数
        truncated_api->PJRT_Device_AddressableMemories = Proxy_Device_AddressableMemories;
        truncated_api->PJRT_Client_Compile = Proxy_Client_Compile;
        truncated_api->PJRT_Client_BufferFromHostBuffer = Proxy_Client_BufferFromHostBuffer;
        truncated_api->PJRT_LoadedExecutable_Execute = Proxy_LoadedExecutable_Execute;
        truncated_api->PJRT_Event_Destroy = Proxy_Event_Destroy;
        truncated_api->PJRT_Buffer_Destroy = Proxy_Buffer_Destroy;
    }
    return truncated_api;
}


} // extern "C"

