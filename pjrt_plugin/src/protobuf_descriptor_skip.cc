#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" void
__real__ZN6google8protobuf14DescriptorPool24InternalAddGeneratedFileEPKvi(
    const void* encoded_file_descriptor, int size);

namespace {

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

bool ShouldSkipDescriptor(const void* descriptor, int descriptor_size) {
  const char* name = nullptr;
  size_t name_size = 0;
  if (!ReadFileDescriptorName(descriptor, descriptor_size, &name, &name_size)) {
    return false;
  }

  if (HasPrefix(name, name_size, "xla/")) {
    std::fprintf(stderr,
                 "[musa_pjrt] skip generated proto descriptor already owned "
                 "by TensorFlow: %.*s\n",
                 static_cast<int>(name_size), name);
    std::fflush(stderr);
    return true;
  }
  return false;
}

}  // namespace

extern "C" void
__wrap__ZN6google8protobuf8internal14AddDescriptorsEPKNS1_15DescriptorTableE(
    const void* descriptor_table) {
  (void)descriptor_table;
  LogSkippedRegistration("internal::AddDescriptors");
}

extern "C" void
__wrap__ZN6google8protobuf8internal20AddDescriptorsRunnerC1EPKNS1_15DescriptorTableE(
    void* self, const void* descriptor_table) {
  (void)self;
  (void)descriptor_table;
  LogSkippedRegistration("internal::AddDescriptorsRunner::C1");
}

extern "C" void
__wrap__ZN6google8protobuf8internal20AddDescriptorsRunnerC2EPKNS1_15DescriptorTableE(
    void* self, const void* descriptor_table) {
  (void)self;
  (void)descriptor_table;
  LogSkippedRegistration("internal::AddDescriptorsRunner::C2");
}

extern "C" void
__wrap__ZN6google8protobuf14DescriptorPool24InternalAddGeneratedFileEPKvi(
    const void* encoded_file_descriptor, int size) {
  if (ShouldSkipDescriptor(encoded_file_descriptor, size)) {
    return;
  }
  __real__ZN6google8protobuf14DescriptorPool24InternalAddGeneratedFileEPKvi(
      encoded_file_descriptor, size);
}
