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

#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_MTGPU_BACKEND_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_MTGPU_BACKEND_H_

#include <string>
#include <vector>

#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace llvm {
class Module;
}

namespace xla {
namespace gpu {
namespace mtgpu {

StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module,
    stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& module_config_cache_key);

// Test-only helper that exercises the same toolchain pipeline used by
// `CompileToHsaco` after LLVM IR serialization.
StatusOr<std::vector<uint8_t>> CompileLlvmIrToHsacoForTest(
    const std::string& ir_text, const DebugOptions& debug_options);

}  // namespace mtgpu
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_MTGPU_BACKEND_H_
