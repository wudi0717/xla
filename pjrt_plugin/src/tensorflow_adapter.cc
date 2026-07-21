#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_status.h"

#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <string>

extern "C" {

#define TPU_C_API_MAX_INLINED 6

struct TF215_IntList {
  union {
    int* heap;
    int inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

struct TF215_Int64List {
  union {
    int64_t* heap;
    int64_t inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

struct TF215_BoolList {
  union {
    bool* heap;
    bool inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

struct TF215_Tile {
  TF215_Int64List dimensions;
};

struct TF215_TileList {
  union {
    TF215_Tile* heap;
    TF215_Tile inlined[TPU_C_API_MAX_INLINED];
  };
  int64_t size;
};

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
  TF215_Shape* tuple_shapes;
  int ntuple_shapes;
  bool has_layout;
  TF215_Layout layout;
};

static std::mutex g_musa_runtime_mu;
static bool g_musa_runtime_registered = false;
static std::mutex g_pjrt_api_mu;
static const PJRT_Api* g_forwarded_pjrt_api = nullptr;

static void DeepCopyIntList(TF215_IntList* dst, const TF215_IntList* src) {
  dst->size = src->size;
  if (src->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new int[src->size];
    std::memcpy(dst->heap, src->heap, src->size * sizeof(int));
  } else {
    std::memcpy(dst->inlined, src->inlined, src->size * sizeof(int));
  }
}

static void DeepCopyInt64List(TF215_Int64List* dst,
                              const TF215_Int64List* src) {
  dst->size = src->size;
  if (src->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new int64_t[src->size];
    std::memcpy(dst->heap, src->heap, src->size * sizeof(int64_t));
  } else {
    std::memcpy(dst->inlined, src->inlined, src->size * sizeof(int64_t));
  }
}

static void DeepCopyBoolList(TF215_BoolList* dst,
                             const TF215_BoolList* src) {
  dst->size = src->size;
  if (src->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new bool[src->size];
    std::memcpy(dst->heap, src->heap, src->size * sizeof(bool));
  } else {
    std::memcpy(dst->inlined, src->inlined, src->size * sizeof(bool));
  }
}

static void DeepCopyTile(TF215_Tile* dst, const TF215_Tile* src) {
  DeepCopyInt64List(&dst->dimensions, &src->dimensions);
}

static void DeepCopyTileList(TF215_TileList* dst,
                             const TF215_TileList* src) {
  dst->size = src->size;
  if (src->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new TF215_Tile[src->size];
    for (int64_t i = 0; i < src->size; ++i) {
      DeepCopyTile(&dst->heap[i], &src->heap[i]);
    }
  } else {
    for (int64_t i = 0; i < src->size; ++i) {
      DeepCopyTile(&dst->inlined[i], &src->inlined[i]);
    }
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
  dst->dynamic_shape_metadata_prefix_bytes =
      src->dynamic_shape_metadata_prefix_bytes;
}

static void DeepCopyShape(TF215_Shape* dst, const TF215_Shape* src) {
  dst->element_type = src->element_type;
  DeepCopyInt64List(&dst->dimensions, &src->dimensions);
  DeepCopyBoolList(&dst->dynamic_dimensions, &src->dynamic_dimensions);
  dst->ntuple_shapes = src->ntuple_shapes;
  if (src->ntuple_shapes > 0 && src->tuple_shapes != nullptr) {
    dst->tuple_shapes = new TF215_Shape[src->ntuple_shapes];
    for (int i = 0; i < src->ntuple_shapes; ++i) {
      DeepCopyShape(&dst->tuple_shapes[i], &src->tuple_shapes[i]);
    }
  } else {
    dst->tuple_shapes = nullptr;
  }
  dst->has_layout = src->has_layout;
  if (src->has_layout) DeepCopyLayout(&dst->layout, &src->layout);
}

static int GetEnvIntOrDefault(const char* env_name, int default_value) {
  const char* env = std::getenv(env_name);
  if (env == nullptr || env[0] == '\0') return default_value;

  char* end = nullptr;
  long value = std::strtol(env, &end, 10);
  if (end == env || (end != nullptr && *end != '\0')) {
    std::fprintf(stderr, "[MUSA TF adapter] ignoring invalid %s=%s\n",
                 env_name, env);
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

static const char* GetEnvOrDefault(const char* env_name,
                                   const char* default_value) {
  const char* env = std::getenv(env_name);
  if (env == nullptr || env[0] == '\0') return default_value;
  return env;
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

  std::fprintf(stderr, "[MUSA TF adapter] ignoring invalid %s=%s\n", name, env);
  return false;
}

static std::string ResolveMusaPjrtPluginPath() {
  const char* explicit_path = std::getenv("MUSA_PJRT_PLUGIN_PATH");
  if (explicit_path != nullptr && explicit_path[0] != '\0') {
    return explicit_path;
  }

  const char* paths = std::getenv("PJRT_NAMES_AND_LIBRARY_PATHS");
  if (paths == nullptr || paths[0] == '\0') return "";

  std::string entries(paths);
  size_t start = 0;
  while (start <= entries.size()) {
    size_t end = entries.find(',', start);
    std::string entry = entries.substr(
        start, end == std::string::npos ? std::string::npos : end - start);
    size_t colon = entry.find(':');
    if (colon != std::string::npos) {
      std::string name = entry.substr(0, colon);
      for (char& ch : name) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      }
      if (name == "musa") return entry.substr(colon + 1);
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return "";
}

using PjrtApiInitFn = const PJRT_Api* (*)();

static void PromoteRuntimeLibraryToGlobal(const char* library_path) {
  if (library_path == nullptr || library_path[0] == '\0') return;

#ifdef RTLD_NOLOAD
  void* loaded = dlopen(library_path, RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
  if (loaded != nullptr) return;
#endif

  void* library = dlopen(library_path, RTLD_LAZY | RTLD_GLOBAL);
  if (library == nullptr) {
    return;
  }
}

static void PreloadTensorFlowRuntimeForCore() {
  const char* override_path = std::getenv("MUSA_TF_RUNTIME_LIBRARY");
  if (override_path != nullptr && override_path[0] != '\0') {
    PromoteRuntimeLibraryToGlobal(override_path);
    return;
  }

  PromoteRuntimeLibraryToGlobal(
      "/workspace/openxla/pjrt_plugin/tf2.15/lib/python3.10/site-packages/"
      "tensorflow/libtensorflow_framework.so.2");
  PromoteRuntimeLibraryToGlobal("libtensorflow_framework.so.2");
}

static const PJRT_Api* ForwardedPjrtApi() {
  std::lock_guard<std::mutex> lock(g_pjrt_api_mu);
  if (g_forwarded_pjrt_api != nullptr) return g_forwarded_pjrt_api;

  std::string library_path = ResolveMusaPjrtPluginPath();
  if (library_path.empty()) {
    std::fprintf(stderr,
                 "[MUSA TF adapter] PJRT core path is empty; set "
                 "PJRT_NAMES_AND_LIBRARY_PATHS=MUSA:/path/to/"
                 "libmusa_pjrt_plugin.so\n");
    std::fflush(stderr);
    return nullptr;
  }

  PreloadTensorFlowRuntimeForCore();

  void* library = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (library == nullptr) {
    std::fprintf(stderr, "[MUSA TF adapter] failed to open PJRT core %s: %s\n",
                 library_path.c_str(), dlerror());
    std::fflush(stderr);
    return nullptr;
  }

  void* symbol = dlsym(library, "GetPjrtApi");
  if (symbol == nullptr) {
    std::fprintf(stderr,
                 "[MUSA TF adapter] GetPjrtApi not found in PJRT core %s\n",
                 library_path.c_str());
    std::fflush(stderr);
    return nullptr;
  }

  auto init_fn = reinterpret_cast<PjrtApiInitFn>(symbol);
  g_forwarded_pjrt_api = init_fn();
  return g_forwarded_pjrt_api;
}

__attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
  const PJRT_Api* api = ForwardedPjrtApi();
  if (api == nullptr) {
    std::fprintf(stderr,
                 "[MUSA TF adapter] fatal: GetPjrtApi forwarding failed; "
                 "cannot register a null PJRT_Api with TensorFlow.\n");
    std::fflush(stderr);
    std::abort();
  }
  return api;
}

using TF_CreateAndSetPjRtCApiClient_Fn =
    void (*)(const char*, TF_Status*, void*, int);

static TF_CreateAndSetPjRtCApiClient_Fn ResolveTfCreateAndSetPjRtCApiClient() {
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
    if (runtime_handle == nullptr) continue;

    symbol = dlsym(runtime_handle, "TF_CreateAndSetPjRtCApiClient");
    if (symbol != nullptr) {
      return reinterpret_cast<TF_CreateAndSetPjRtCApiClient_Fn>(symbol);
    }
  }
#endif

  void* process_handle = dlopen(nullptr, RTLD_LAZY | RTLD_GLOBAL);
  if (process_handle == nullptr) return nullptr;

  symbol = dlsym(process_handle, "TF_CreateAndSetPjRtCApiClient");
  return reinterpret_cast<TF_CreateAndSetPjRtCApiClient_Fn>(symbol);
}

static bool EnsureMusaRuntimeRegistered(TF_Status* tf_status, bool verbose) {
  std::lock_guard<std::mutex> lock(g_musa_runtime_mu);
  if (g_musa_runtime_registered) {
    if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
    return true;
  }

  auto create_pjrt_client = ResolveTfCreateAndSetPjRtCApiClient();
  if (!create_pjrt_client) {
    const std::string msg =
        "symbol TF_CreateAndSetPjRtCApiClient not found in current TensorFlow "
        "runtime";
    if (verbose) std::fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
    if (tf_status) TF_SetStatus(tf_status, TF_INTERNAL, msg.c_str());
    return false;
  }

  TF_Status* create_status = TF_NewStatus();
  const char* pjrt_device_type = GetEnvOrDefault("MUSA_NPD_DEVICE_TYPE", "MUSA");
  const PJRT_Api* pjrt_api = ForwardedPjrtApi();
  if (pjrt_api == nullptr) {
    const std::string msg = "failed to load forwarded PJRT_Api from MUSA core";
    if (verbose) std::fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
    if (tf_status) TF_SetStatus(tf_status, TF_INTERNAL, msg.c_str());
    TF_DeleteStatus(create_status);
    return false;
  }
  create_pjrt_client(
      pjrt_device_type, create_status,
      const_cast<void*>(static_cast<const void*>(pjrt_api)), 0);
  if (TF_GetCode(create_status) != TF_OK) {
    const std::string msg =
        std::string("TF_CreateAndSetPjRtCApiClient failed: ") +
        TF_Message(create_status);
    if (verbose) std::fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
    if (tf_status) {
      TF_SetStatus(tf_status, TF_GetCode(create_status), msg.c_str());
    }
    TF_DeleteStatus(create_status);
    return false;
  }
  TF_DeleteStatus(create_status);

  g_musa_runtime_registered = true;
  if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
  return true;
}

static void Musa_XlaShapeToDeviceShapeRepresentation(
    XLA_Shape* c_xla_shape, int data_type, bool use_fast_memory,
    XLA_LayoutPreference layout_preference, XLA_Shape* c_device_shape,
    TF_Status* status) {
  if (c_xla_shape && c_device_shape) {
    auto* src = reinterpret_cast<const TF215_Shape*>(c_xla_shape);
    auto* dst = reinterpret_cast<TF215_Shape*>(c_device_shape);
    DeepCopyShape(dst, src);
  }
  if (status) TF_SetStatus(status, TF_OK, "");
}

static int32_t Musa_GetDeviceCount(TF_Status* status) {
  if (status) TF_SetStatus(status, TF_OK, "");
  return 8;
}

static void Musa_InitPluginInternalDeviceStates(TF_Status* status) {
  EnsureMusaRuntimeRegistered(status, true);
}

__attribute__((visibility("default"))) const TFNPD_Api* TFNPD_InitPlugin(
    TFNPD_PluginParams* params, TF_Status* tf_status) {
  if (params == nullptr) {
    if (tf_status) {
      TF_SetStatus(tf_status, TF_INVALID_ARGUMENT,
                   "TFNPD_InitPlugin received null params");
    }
    return nullptr;
  }

  const char* dev_type = GetEnvOrDefault("MUSA_NPD_DEVICE_TYPE", "MUSA");
  const char* comp_dev =
      GetEnvOrDefault("MUSA_NPD_COMPILATION_DEVICE", "XLA_GPU_JIT");
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

  static TFNPD_Api npd_api = {};
  npd_api.struct_size = TFNPD_Api_STRUCT_SIZE;
  npd_api.TFNPD_XlaShapeToDeviceShapeRepresentation =
      Musa_XlaShapeToDeviceShapeRepresentation;
  npd_api.TFNPD_GetDeviceCount = Musa_GetDeviceCount;
  npd_api.TFNPD_InitPluginInternalDeviceStates =
      Musa_InitPluginInternalDeviceStates;

  if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
  return &npd_api;
}

__attribute__((visibility("default"))) int ForceRegisterMusa() {
  TF_Status* status = TF_NewStatus();
  if (EnsureMusaRuntimeRegistered(status, true) &&
      TF_GetCode(status) == TF_OK) {
    TF_DeleteStatus(status);
    return 1;
  }
  std::fprintf(stderr, "[MUSA TF adapter] MUSA PJRT factory registration failed: %s\n",
               TF_Message(status));
  TF_DeleteStatus(status);
  std::fflush(stdout);
  std::fflush(stderr);
  return 0;
}

}  // extern "C"
