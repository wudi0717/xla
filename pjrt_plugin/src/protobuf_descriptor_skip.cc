#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

using AddDescriptorsFn = void (*)(const void*);
using InternalAddGeneratedFileFn = void (*)(const void*, int);

bool ShouldSkipDescriptorRegistration() {
  const char* env = std::getenv("MUSA_PJRT_SKIP_PROTO_DESCRIPTOR_REGISTRATION");
  return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0 &&
         std::strcmp(env, "false") != 0 && std::strcmp(env, "False") != 0 &&
         std::strcmp(env, "FALSE") != 0;
}

void LogSkippedRegistration(const char* entrypoint) {
  static int log_count = 0;
  if (log_count < 8) {
    std::fprintf(stderr,
                 "[musa_pjrt] skip protobuf generated descriptor "
                 "registration via %s\n",
                 entrypoint);
    std::fflush(stderr);
    ++log_count;
  }
}

void LogForwardedRegistration(const char* entrypoint) {
  static int log_count = 0;
  if (log_count < 8) {
    std::fprintf(stderr,
                 "[musa_pjrt] forward protobuf generated descriptor "
                 "registration via %s\n",
                 entrypoint);
    std::fflush(stderr);
    ++log_count;
  }
}

template <typename Fn>
Fn ResolveRuntimeSymbol(const char* symbol_name) {
#ifdef RTLD_NEXT
  void* symbol = dlsym(RTLD_NEXT, symbol_name);
  if (symbol != nullptr) {
    return reinterpret_cast<Fn>(symbol);
  }
#else
  void* symbol = nullptr;
#endif

  symbol = dlsym(RTLD_DEFAULT, symbol_name);
  if (symbol != nullptr) {
    return reinterpret_cast<Fn>(symbol);
  }

  return nullptr;
}

AddDescriptorsFn ResolveAddDescriptors() {
  return ResolveRuntimeSymbol<AddDescriptorsFn>(
      "_ZN6google8protobuf8internal14AddDescriptorsEPKNS1_15DescriptorTableE");
}

bool ReadVarint(const unsigned char* data, size_t size, size_t* offset,
                uint64_t* value) {
  uint64_t result = 0;
  int shift = 0;
  while (*offset < size && shift < 64) {
    const unsigned char byte = data[(*offset)++];
    result |= static_cast<uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) {
      *value = result;
      return true;
    }
    shift += 7;
  }
  return false;
}

bool SkipField(const unsigned char* data, size_t size, size_t* offset,
               uint64_t wire_type) {
  uint64_t length = 0;
  switch (wire_type) {
    case 0:
      return ReadVarint(data, size, offset, &length);
    case 1:
      if (size - *offset < 8) return false;
      *offset += 8;
      return true;
    case 2:
      if (!ReadVarint(data, size, offset, &length)) return false;
      if (length > size - *offset) return false;
      *offset += static_cast<size_t>(length);
      return true;
    case 5:
      if (size - *offset < 4) return false;
      *offset += 4;
      return true;
    default:
      return false;
  }
}

bool ReadFileDescriptorName(const void* descriptor, int descriptor_size,
                            const char** name, size_t* name_size) {
  if (descriptor == nullptr || descriptor_size <= 0) return false;

  const auto* data = static_cast<const unsigned char*>(descriptor);
  const size_t size = static_cast<size_t>(descriptor_size);
  size_t offset = 0;
  while (offset < size) {
    uint64_t tag = 0;
    if (!ReadVarint(data, size, &offset, &tag)) return false;

    const uint64_t field_number = tag >> 3;
    const uint64_t wire_type = tag & 0x7;
    if (field_number == 1 && wire_type == 2) {
      uint64_t length = 0;
      if (!ReadVarint(data, size, &offset, &length)) return false;
      if (length > size - offset) return false;
      *name = reinterpret_cast<const char*>(data + offset);
      *name_size = static_cast<size_t>(length);
      return true;
    }

    if (!SkipField(data, size, &offset, wire_type)) return false;
  }
  return false;
}

bool HasPrefix(const char* value, size_t value_size, const char* prefix) {
  const size_t prefix_size = std::strlen(prefix);
  return value_size >= prefix_size &&
         std::memcmp(value, prefix, prefix_size) == 0;
}

void LogDescriptorForward(const void* descriptor, int descriptor_size) {
  const char* name = nullptr;
  size_t name_size = 0;
  if (!ReadFileDescriptorName(descriptor, descriptor_size, &name, &name_size)) {
    return;
  }

  if (HasPrefix(name, name_size, "xla/")) {
    std::fprintf(stderr,
                 "[musa_pjrt] forward generated proto descriptor to active "
                 "protobuf pool: %.*s\n",
                 static_cast<int>(name_size), name);
    std::fflush(stderr);
  }
}

InternalAddGeneratedFileFn ResolveInternalAddGeneratedFile() {
  return ResolveRuntimeSymbol<InternalAddGeneratedFileFn>(
      "_ZN6google8protobuf14DescriptorPool24InternalAddGeneratedFileEPKvi");
}

}  // namespace

extern "C" void
__wrap__ZN6google8protobuf8internal14AddDescriptorsEPKNS1_15DescriptorTableE(
    const void* descriptor_table) {
  if (ShouldSkipDescriptorRegistration()) {
    LogSkippedRegistration("internal::AddDescriptors");
    return;
  }
  LogForwardedRegistration("internal::AddDescriptors");
  AddDescriptorsFn real_fn = ResolveAddDescriptors();
  if (real_fn == nullptr) {
    std::fprintf(stderr,
                 "[musa_pjrt] failed to resolve protobuf "
                 "internal::AddDescriptors\n");
    std::fflush(stderr);
    return;
  }
  real_fn(descriptor_table);
}

extern "C" void
__wrap__ZN6google8protobuf8internal20AddDescriptorsRunnerC1EPKNS1_15DescriptorTableE(
    void* self, const void* descriptor_table) {
  (void)self;
  if (ShouldSkipDescriptorRegistration()) {
    LogSkippedRegistration("internal::AddDescriptorsRunner::C1");
    return;
  }
  LogForwardedRegistration("internal::AddDescriptorsRunner::C1");
  AddDescriptorsFn real_fn = ResolveAddDescriptors();
  if (real_fn == nullptr) {
    std::fprintf(stderr,
                 "[musa_pjrt] failed to resolve protobuf "
                 "internal::AddDescriptors from AddDescriptorsRunner::C1\n");
    std::fflush(stderr);
    return;
  }
  real_fn(descriptor_table);
}

extern "C" void
__wrap__ZN6google8protobuf8internal20AddDescriptorsRunnerC2EPKNS1_15DescriptorTableE(
    void* self, const void* descriptor_table) {
  (void)self;
  if (ShouldSkipDescriptorRegistration()) {
    LogSkippedRegistration("internal::AddDescriptorsRunner::C2");
    return;
  }
  LogForwardedRegistration("internal::AddDescriptorsRunner::C2");
  AddDescriptorsFn real_fn = ResolveAddDescriptors();
  if (real_fn == nullptr) {
    std::fprintf(stderr,
                 "[musa_pjrt] failed to resolve protobuf "
                 "internal::AddDescriptors from AddDescriptorsRunner::C2\n");
    std::fflush(stderr);
    return;
  }
  real_fn(descriptor_table);
}

extern "C" void
__wrap__ZN6google8protobuf14DescriptorPool24InternalAddGeneratedFileEPKvi(
    const void* encoded_file_descriptor, int size) {
  if (ShouldSkipDescriptorRegistration()) {
    LogSkippedRegistration("DescriptorPool::InternalAddGeneratedFile");
    return;
  }
  InternalAddGeneratedFileFn real_fn = ResolveInternalAddGeneratedFile();
  if (real_fn == nullptr) {
    LogSkippedRegistration("DescriptorPool::InternalAddGeneratedFile");
    return;
  }
  LogDescriptorForward(encoded_file_descriptor, size);
  real_fn(encoded_file_descriptor, size);
}
