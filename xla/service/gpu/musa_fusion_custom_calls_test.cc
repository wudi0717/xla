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

#include "musa_fusion_custom_calls.h"

#include <cstddef>
#include <cstring>
#include <string>

#include "gtest/gtest.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/custom_call_target_registry.h"

namespace xla {
namespace gpu {
namespace {

TEST(MusaFusionCustomCallsTest, LayerNormTargetIsRegisteredForMusa) {
  void* target = CustomCallTargetRegistry::Global()->Lookup(
      std::string(kMusaLayerNormCustomCallTarget), "MUSA");
  EXPECT_NE(target, nullptr);
}

TEST(MusaFusionCustomCallsTest, MatmulBiasReluTargetIsRegisteredForMusa) {
  void* target = CustomCallTargetRegistry::Global()->Lookup(
      std::string(kMusaMatmulBiasReluCustomCallTarget), "MUSA");
  EXPECT_NE(target, nullptr);
}

TEST(MusaFusionCustomCallsTest, LayerNormReturnsFailureForMalformedOpaque) {
  void* target = CustomCallTargetRegistry::Global()->Lookup(
      std::string(kMusaLayerNormCustomCallTarget), "MUSA");
  ASSERT_NE(target, nullptr);

  using StatusReturningCallType =
      void (*)(void* /*stream*/, void** /*buffers*/, const char* /*opaque*/,
               std::size_t /*opaque_len*/, XlaCustomCallStatus* /*status*/);
  auto call_target = reinterpret_cast<StatusReturningCallType>(target);

  const char* malformed_opaque = "dtype=f32;axis=bad;epsilon=1e-5;workspace=0";
  XlaCustomCallStatus status;
  call_target(/*stream=*/nullptr, /*buffers=*/nullptr, malformed_opaque,
              std::strlen(malformed_opaque), &status);

  auto message = xla::CustomCallStatusGetMessage(&status);
  ASSERT_TRUE(message.has_value());
  EXPECT_NE(message->find("invalid __musa$layernorm contract"),
            std::string::npos);
  EXPECT_NE(message->find("invalid axis"), std::string::npos);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
