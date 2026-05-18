#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/stream_executor/platform_manager.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/c_api.h"

#include <iostream>
#include <memory>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <dlfcn.h>

extern "C" {

// =========================================================================
// 🏛️ TF 2.15 官方真实 C API 布局还原 (对齐 tpu/c_api_decl.h)
// =========================================================================
#define TPU_C_API_MAX_INLINED 6

struct TF215_IntList { union { int* heap; int inlined[TPU_C_API_MAX_INLINED]; }; int64_t size; };
struct TF215_Int64List { union { int64_t* heap; int64_t inlined[TPU_C_API_MAX_INLINED]; }; int64_t size; };
struct TF215_BoolList { union { bool* heap; bool inlined[TPU_C_API_MAX_INLINED]; }; int64_t size; };
struct TF215_Tile { TF215_Int64List dimensions; };
struct TF215_TileList { union { TF215_Tile* heap; TF215_Tile inlined[TPU_C_API_MAX_INLINED]; }; int64_t size; };

struct TF215_Layout {
    TF215_Int64List minor_to_major;
    TF215_IntList dim_level_types;
    TF215_IntList dim_unique;
    TF215_IntList dim_ordered;
    TF215_TileList tiles;
    int index_primitive_type;
    int pointer_primitive_type;
    int64_t element_size_in_bits;
    int64_t memory_space;
    int64_t dynamic_shape_metadata_prefix_bytes;
};

struct TF215_Shape {
    int element_type;
    TF215_Int64List dimensions;
    TF215_BoolList dynamic_dimensions;
    struct TF215_Shape* tuple_shapes; // owned
    int ntuple_shapes;
    bool has_layout;
    TF215_Layout layout;
};

// 深拷贝拦截器实现
static void DeepCopyIntList(TF215_IntList* dst, const TF215_IntList* src) {
    dst->size = src->size;
    if (src->size > TPU_C_API_MAX_INLINED) {
        dst->heap = new int[src->size]; memcpy(dst->heap, src->heap, src->size * sizeof(int));
    } else { memcpy(dst->inlined, src->inlined, src->size * sizeof(int)); }
}
static void DeepCopyInt64List(TF215_Int64List* dst, const TF215_Int64List* src) {
    dst->size = src->size;
    if (src->size > TPU_C_API_MAX_INLINED) {
        dst->heap = new int64_t[src->size]; memcpy(dst->heap, src->heap, src->size * sizeof(int64_t));
    } else { memcpy(dst->inlined, src->inlined, src->size * sizeof(int64_t)); }
}
static void DeepCopyBoolList(TF215_BoolList* dst, const TF215_BoolList* src) {
    dst->size = src->size;
    if (src->size > TPU_C_API_MAX_INLINED) {
        dst->heap = new bool[src->size]; memcpy(dst->heap, src->heap, src->size * sizeof(bool));
    } else { memcpy(dst->inlined, src->inlined, src->size * sizeof(bool)); }
}
static void DeepCopyTile(TF215_Tile* dst, const TF215_Tile* src) {
    DeepCopyInt64List(&dst->dimensions, &src->dimensions);
}
static void DeepCopyTileList(TF215_TileList* dst, const TF215_TileList* src) {
    dst->size = src->size;
    if (src->size > TPU_C_API_MAX_INLINED) {
        dst->heap = new TF215_Tile[src->size];
        for (int64_t i = 0; i < src->size; ++i) DeepCopyTile(&dst->heap[i], &src->heap[i]);
    } else {
        for (int64_t i = 0; i < src->size; ++i) DeepCopyTile(&dst->inlined[i], &src->inlined[i]);
    }
}
static void DeepCopyLayout(TF215_Layout* dst, const TF215_Layout* src) {
    DeepCopyInt64List(&dst->minor_to_major, &src->minor_to_major);
    DeepCopyIntList(&dst->dim_level_types, &src->dim_level_types);
    DeepCopyIntList(&dst->dim_unique, &src->dim_unique);
    DeepCopyIntList(&dst->dim_ordered, &src->dim_ordered);
    DeepCopyTileList(&dst->tiles, &src->tiles);
    dst->index_primitive_type = src->index_primitive_type;
    dst->pointer_primitive_type = src->pointer_primitive_type;
    dst->element_size_in_bits = src->element_size_in_bits;
    dst->memory_space = src->memory_space;
    dst->dynamic_shape_metadata_prefix_bytes = src->dynamic_shape_metadata_prefix_bytes;
}
static void DeepCopyShape(TF215_Shape* dst, const TF215_Shape* src) {
    dst->element_type = src->element_type;
    DeepCopyInt64List(&dst->dimensions, &src->dimensions);
    DeepCopyBoolList(&dst->dynamic_dimensions, &src->dynamic_dimensions);
    dst->ntuple_shapes = src->ntuple_shapes;
    if (src->ntuple_shapes > 0 && src->tuple_shapes != nullptr) {
        dst->tuple_shapes = new TF215_Shape[src->ntuple_shapes];
        for (int i = 0; i < src->ntuple_shapes; ++i) DeepCopyShape(&dst->tuple_shapes[i], &src->tuple_shapes[i]);
    } else { dst->tuple_shapes = nullptr; }
    dst->has_layout = src->has_layout;
    if (src->has_layout) DeepCopyLayout(&dst->layout, &src->layout);
}

// =========================================================================
// 🚀 核心垫片层与实现
// =========================================================================

static PJRT_Api base_api;
static bool base_api_initialized = false;
static std::mutex g_musa_runtime_mu;
static std::mutex g_execute_submit_mu;
static bool g_musa_runtime_registered = false;
static std::atomic<unsigned long long> g_event_destroy_bypass_count{0};
static std::atomic<unsigned long long> g_buffer_destroy_bypass_count{0};
static std::atomic<unsigned long long> g_client_compile_log_count{0};
static std::atomic<unsigned long long> g_buffer_from_host_log_count{0};
static std::atomic<unsigned long long> g_execute_log_count{0};

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

static int GetEnvIntOrDefault(const char* env_name, int default_value) {
    const char* env = std::getenv(env_name);
    if (env == nullptr || env[0] == '\0') return default_value;

    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || (end != nullptr && *end != '\0')) {
        fprintf(stderr, "[MUSA PJRT] ignoring invalid %s=%s\n", env_name, env);
        return default_value;
    }
    if (value > std::numeric_limits<int>::max()) {
        return std::numeric_limits<int>::max();
    }
    if (value < std::numeric_limits<int>::min()) {
        return std::numeric_limits<int>::min();
    }
    return static_cast<int>(value);
}

static const char* GetEnvOrDefault(const char* env_name, const char* default_value) {
    const char* env = std::getenv(env_name);
    if (env == nullptr || env[0] == '\0') return default_value;
    return env;
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

using TF_CreateAndSetPjRtCApiClient_Fn =
    void (*)(const char*, TF_Status*, void*, int);

TF_CreateAndSetPjRtCApiClient_Fn ResolveTfCreateAndSetPjRtCApiClient() {
    void* symbol = dlsym(RTLD_DEFAULT, "TF_CreateAndSetPjRtCApiClient");
    if (symbol != nullptr) {
        return reinterpret_cast<TF_CreateAndSetPjRtCApiClient_Fn>(symbol);
    }

#ifdef RTLD_NOLOAD
    constexpr const char* kTensorFlowRuntimeLibs[] = {
        "libtensorflow_framework.so.2",
        "libtensorflow_cc.so.2",
    };
    for (const char* lib_name : kTensorFlowRuntimeLibs) {
        void* runtime_handle =
            dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
        if (runtime_handle == nullptr) {
            continue;
        }
        symbol = dlsym(runtime_handle, "TF_CreateAndSetPjRtCApiClient");
        if (symbol != nullptr) {
            return reinterpret_cast<TF_CreateAndSetPjRtCApiClient_Fn>(symbol);
        }
    }
#endif

    void* process_handle = dlopen(nullptr, RTLD_LAZY | RTLD_GLOBAL);
    if (process_handle == nullptr) {
        return nullptr;
    }

    symbol = dlsym(process_handle, "TF_CreateAndSetPjRtCApiClient");
    return reinterpret_cast<TF_CreateAndSetPjRtCApiClient_Fn>(symbol);
}

bool EnsureMusaRuntimeRegistered(TF_Status* tf_status, bool verbose) {
    std::lock_guard<std::mutex> lock(g_musa_runtime_mu);
    if (g_musa_runtime_registered) {
        if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
        return true;
    }

    auto platform_or = stream_executor::PlatformManager::PlatformWithName("MUSA");
    if (!platform_or.ok()) {
        const std::string msg = std::string("failed to initialize MUSA platform: ") +
                                platform_or.status().ToString();
        if (verbose) fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
        if (tf_status) TF_SetStatus(tf_status, TF_INTERNAL, msg.c_str());
        return false;
    }

    auto create_pjrt_client = ResolveTfCreateAndSetPjRtCApiClient();
    if (!create_pjrt_client) {
        const std::string msg =
            "symbol TF_CreateAndSetPjRtCApiClient not found in current TensorFlow runtime";
        if (verbose) fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
        if (tf_status) TF_SetStatus(tf_status, TF_INTERNAL, msg.c_str());
        return false;
    }

    TF_Status* create_status = TF_NewStatus();
    const char* pjrt_device_type = GetEnvOrDefault("MUSA_NPD_DEVICE_TYPE", "MUSA");
    create_pjrt_client(pjrt_device_type, create_status, nullptr, 0);
    if (TF_GetCode(create_status) != TF_OK) {
        const std::string msg = std::string("TF_CreateAndSetPjRtCApiClient failed: ") +
                                TF_Message(create_status);
        if (verbose) fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
        if (tf_status) TF_SetStatus(tf_status, TF_GetCode(create_status), msg.c_str());
        TF_DeleteStatus(create_status);
        return false;
    }
    TF_DeleteStatus(create_status);

    g_musa_runtime_registered = true;
    if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
    if (verbose) {
        printf("✅ [MUSA] Plugin internal device states initialized.\n");
        fflush(stdout);
    }
    return true;
}

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
    PJRT_Error* err = base_api.PJRT_Client_Compile(args);
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

PJRT_Error* Proxy_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
    const int max_inflight_transfers = GetTransferMaxInflight();
    ScopedInflightGate gate(&g_transfer_gate, max_inflight_transfers);
    unsigned long long log_count =
        g_buffer_from_host_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
    bool should_log = ShouldLogProxyDebug();
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] buffer-from-host begin: count=%llu max_inflight=%d struct_size=%zu data=%p type=%d num_dims=%zu semantics=%d device=%p\n",
                log_count,
                max_inflight_transfers,
                args ? static_cast<size_t>(args->struct_size) : 0,
                args ? args->data : nullptr,
                args ? static_cast<int>(args->type) : -1,
                args ? static_cast<size_t>(args->num_dims) : 0,
                args ? static_cast<int>(args->host_buffer_semantics) : -1,
                args ? static_cast<void*>(args->device) : nullptr);
        fflush(stderr);
    }

    PJRT_HostBufferSemantics original_semantics =
        args ? args->host_buffer_semantics
             : PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    if (args != nullptr && ShouldForceHostBufferCopy()) {
        args->host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    }

    const auto transfer_start = std::chrono::steady_clock::now();
    PJRT_Error* err = base_api.PJRT_Client_BufferFromHostBuffer(args);
    if (args != nullptr) {
        args->host_buffer_semantics = original_semantics;
    }

    if (err == nullptr && args != nullptr && args->done_with_host_buffer != nullptr &&
        ShouldWaitTransferDoneBeforeReturn()) {
        PJRT_Error* wait_err =
            WaitForEventViaCallback(args->done_with_host_buffer);
        if (wait_err != nullptr && ShouldLogProxyDebug()) {
            fprintf(stderr,
                    "[musa_pjrt] buffer-from-host wait returned error: count=%llu err=%p done_event=%p\n",
                    log_count,
                    static_cast<void*>(wait_err),
                    static_cast<void*>(args->done_with_host_buffer));
            fflush(stderr);
        }
    }

    if (should_log || err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] buffer-from-host returned: count=%llu err=%p done_event=%p buffer=%p force_copy=%d wait_done=%d transfer_ms=%.3f\n",
                log_count,
                static_cast<void*>(err),
                args ? static_cast<void*>(args->done_with_host_buffer) : nullptr,
                args ? static_cast<void*>(args->buffer) : nullptr,
                ShouldForceHostBufferCopy() ? 1 : 0,
                ShouldWaitTransferDoneBeforeReturn() ? 1 : 0,
                MsSince(transfer_start));
        fflush(stderr);
    }
    return err;
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

PJRT_Error* Proxy_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
    if (args == nullptr || args->buffer == nullptr) {
        return nullptr;
    }
    if (ShouldBypassBufferDestroy()) {
        unsigned long long bypass_count =
            g_buffer_destroy_bypass_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (ShouldLogProxyDebug() &&
            (bypass_count <= 4 || (bypass_count % 100000) == 0)) {
            fprintf(stderr,
                    "[musa_pjrt] buffer destroy bypassed: count=%llu buffer=%p\n",
                    bypass_count,
                    static_cast<void*>(args->buffer));
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
            memset(&destroy_ready_event_args, 0, sizeof(destroy_ready_event_args));
            destroy_ready_event_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
            destroy_ready_event_args.event = ready_args.event;
            PJRT_Error* destroy_ready_event_err =
                base_api.PJRT_Event_Destroy(&destroy_ready_event_args);
            if (destroy_ready_event_err != nullptr && ShouldLogProxyDebug()) {
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

PJRT_Error* Proxy_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args) {
    if (!args || !args->options) return base_api.PJRT_LoadedExecutable_Execute(args);

    const int max_inflight_executes = GetExecuteMaxInflight();
    ScopedInflightGate gate(&g_execute_gate, max_inflight_executes);

    PJRT_ExecuteOptions* original_options = args->options;
    PJRT_Device* original_execute_device = args->execute_device;
    const bool drop_execute_device =
        original_execute_device != nullptr && ShouldDropExecuteDeviceForCompat();
    unsigned long long log_count =
        g_execute_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
    bool should_log = ShouldLogProxyDebug();

    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] execute begin: count=%llu max_inflight=%d args_size=%zu options_size=%zu num_devices=%zu events=%p execute_device=%p drop_execute_device=%d serialize_submit=%d wait_done=%d\n",
                log_count,
                max_inflight_executes,
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
        for (size_t device_index = 0; device_index < args->num_devices; ++device_index) {
            PJRT_Event* event = args->device_complete_events[device_index];
            if (event == nullptr) {
                continue;
            }
            PJRT_Error* wait_err = WaitForEventViaCallback(event);
            if (wait_err != nullptr) {
                err = wait_err;
                if (ShouldLogProxyDebug()) {
                    fprintf(stderr,
                            "[musa_pjrt] execute wait returned error: count=%llu err=%p device_index=%zu event=%p\n",
                            log_count,
                            static_cast<void*>(wait_err),
                            device_index,
                            static_cast<void*>(event));
                    fflush(stderr);
                }
                break;
            }
        }
    }

    if (should_log || err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] execute returned: count=%llu err=%p events=%p first_event=%p execute_ms=%.3f\n",
                log_count,
                static_cast<void*>(err),
                static_cast<void*>(args->device_complete_events),
                (args->device_complete_events != nullptr && args->num_devices > 0)
                    ? static_cast<void*>(args->device_complete_events[0])
                    : nullptr,
                MsSince(execute_start));
        fflush(stderr);
    }

    return err;
}

void Musa_XlaShapeToDeviceShapeRepresentation(
    XLA_Shape* c_xla_shape, int data_type, bool use_fast_memory,
    XLA_LayoutPreference layout_preference, XLA_Shape* c_device_shape, TF_Status* status) {
    if (c_xla_shape && c_device_shape) {
        auto* src = reinterpret_cast<const TF215_Shape*>(c_xla_shape);
        auto* dst = reinterpret_cast<TF215_Shape*>(c_device_shape);
        DeepCopyShape(dst, src);
    }
    if (status) TF_SetStatus(status, TF_OK, "");
}

int32_t Musa_GetDeviceCount(TF_Status* status) {
    if (status) TF_SetStatus(status, TF_OK, "");
    return 8;
}

void Musa_InitPluginInternalDeviceStates(TF_Status* status) {
    EnsureMusaRuntimeRegistered(status, true);
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

    bool preallocate = false;
    if (ReadBoolEnv("MUSA_PJRT_PREALLOCATE", &preallocate)) {
        allocator_config->preallocate = preallocate;
    }

    double memory_fraction = 0.0;
    if (ReadDoubleEnv("MUSA_PJRT_MEMORY_FRACTION", &memory_fraction)) {
        allocator_config->memory_fraction = memory_fraction;
    }

    fprintf(stderr,
            "[MUSA PJRT] allocator preallocate=%s memory_fraction=%.3f",
            allocator_config->preallocate ? "true" : "false",
            allocator_config->memory_fraction);
    fprintf(stderr, "\n");
    fflush(stderr);
}

PJRT_Error* Musa_Client_Create(PJRT_Client_Create_Args* args) {
    xla::GpuAllocatorConfig allocator_config;
    ApplyMusaAllocatorEnv(&allocator_config);
    auto client_or = xla::GetStreamExecutorGpuClient(
        /*asynchronous=*/true, allocator_config, /*node_id=*/0,
        /*num_nodes=*/1, /*allowed_devices=*/std::nullopt,
        /*platform_name=*/std::string("MUSA"));
    if (!client_or.ok()) {
        fprintf(stderr, "🚨 MUSA Init Failed: %s\n", client_or.status().ToString().c_str());
        abort();
    }
    args->client = pjrt::CreateWrapperClient(std::move(client_or.value()));
    printf("✅ MUSA Client Created.\n"); fflush(stdout);
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

__attribute__((visibility("default"))) const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params, TF_Status* tf_status) {
    if (params == nullptr) {
        if (tf_status) {
            TF_SetStatus(tf_status, TF_INVALID_ARGUMENT,
                         "TFNPD_InitPlugin received null params");
        }
        return nullptr;
    }

    const char* dev_type = GetEnvOrDefault("MUSA_NPD_DEVICE_TYPE", "MUSA");
    const char* comp_dev = GetEnvOrDefault("MUSA_NPD_COMPILATION_DEVICE", "XLA_GPU_JIT");
    int priority_val = GetEnvIntOrDefault("MUSA_NPD_PRIORITY", 1000);
    bool is_pluggable = true;
    ReadBoolEnv("MUSA_NPD_IS_PLUGGABLE_DEVICE", &is_pluggable);
    bool use_pjrt = true;
    ReadBoolEnv("MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE", &use_pjrt);

    params->ext = nullptr;
    params->device_type = dev_type;
    params->compilation_device_name = comp_dev;
    params->priority = priority_val;
    params->is_pluggable_device = is_pluggable;
    params->use_pjrt_on_demand_compile = use_pjrt;

    fprintf(stderr,
            "[MUSA PJRT] TFNPD_InitPlugin struct_size=%zu device_type=%s compilation_device=%s priority=%d is_pluggable_device=%s use_pjrt_on_demand_compile=%s\n",
            static_cast<size_t>(params->struct_size),
            dev_type,
            comp_dev,
            priority_val,
            is_pluggable ? "true" : "false",
            use_pjrt ? "true" : "false");
    fflush(stderr);

    static TFNPD_Api npd_api = {};
    npd_api.struct_size = TFNPD_Api_STRUCT_SIZE;
    npd_api.TFNPD_XlaShapeToDeviceShapeRepresentation = Musa_XlaShapeToDeviceShapeRepresentation;
    npd_api.TFNPD_GetDeviceCount = Musa_GetDeviceCount;
    npd_api.TFNPD_InitPluginInternalDeviceStates = Musa_InitPluginInternalDeviceStates;

    if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
    return &npd_api;
}

__attribute__((visibility("default"))) void ForceRegisterMusa() {
    TF_Status* status = TF_NewStatus();
    if (EnsureMusaRuntimeRegistered(status, true) && TF_GetCode(status) == TF_OK) {
        printf("✅ [PYTHON HOOK] MUSA PJRT Factory Registered!\n");
    }
    TF_DeleteStatus(status);
    fflush(stdout);
}

} // extern "C"
