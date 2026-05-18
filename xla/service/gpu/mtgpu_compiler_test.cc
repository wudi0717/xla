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

#include "xla/client/client_library.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/lib/constants.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_builder.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/test.h"

namespace xla {
namespace gpu {
namespace {

TEST(MTGPUCompilerTest, LocalClientCompileReachesMTGPUBackend) {
  auto platform_or = PlatformUtil::GetPlatform("MUSA");
  if (!platform_or.ok()) {
    GTEST_SKIP() << platform_or.status();
  }
  se::Platform* platform = platform_or.value();

  auto client_or = ClientLibrary::GetOrCreateLocalClient(platform);
  if (!client_or.ok()) {
    GTEST_SKIP() << client_or.status();
  }
  LocalClient* client = client_or.value();

  if (client->device_count() <= 0 || !client->device_ordinal_supported(0)) {
    GTEST_SKIP() << "No usable MUSA device for MTGPU compiler smoke test";
  }

  XlaBuilder builder("mtgpu_compiler_smoke");
  Add(ConstantR0<float>(&builder, 1.0f), ConstantR0<float>(&builder, 2.0f));

  auto computation_or = builder.Build();
  ASSERT_TRUE(computation_or.ok()) << computation_or.status();

  ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(0);

  auto compile_or = client->Compile(computation_or.value(), {}, build_options);
  ASSERT_TRUE(compile_or.ok()) << compile_or.status();
  EXPECT_FALSE(compile_or.value().empty());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
