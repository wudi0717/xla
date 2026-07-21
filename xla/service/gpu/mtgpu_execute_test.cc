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
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/literal_test_util.h"

namespace xla {
namespace gpu {
namespace {

TEST(MTGPUExecuteTest, LocalClientCompileAndRunOnMUSA) {
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

  const int device_ordinal = client->default_device_ordinal();
  if (device_ordinal < 0 || !client->device_ordinal_supported(device_ordinal)) {
    GTEST_SKIP() << "No usable MUSA device for MTGPU execute test";
  }

  XlaBuilder builder("mtgpu_execute_smoke");
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3}), "x");
  Add(x, ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f}));

  auto computation_or = builder.Build();
  ASSERT_TRUE(computation_or.ok()) << computation_or.status();

  auto input_buffer_or = client->LiteralToShapedBuffer(
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f}), device_ordinal);
  ASSERT_TRUE(input_buffer_or.ok()) << input_buffer_or.status();
  ScopedShapedBuffer input_buffer = std::move(input_buffer_or.value());

  const Shape& argument_shape = input_buffer.on_host_shape();
  ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(device_ordinal);

  auto executables_or =
      client->Compile(computation_or.value(), {&argument_shape}, build_options);
  ASSERT_TRUE(executables_or.ok()) << executables_or.status();
  ASSERT_EQ(executables_or->size(), 1);
  std::unique_ptr<LocalExecutable> executable =
      std::move(executables_or.value().front());

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(device_ordinal)
      .set_allocator(client->backend().memory_allocator());

  auto result_buffer_or = executable->Run({&input_buffer}, run_options);
  ASSERT_TRUE(result_buffer_or.ok()) << result_buffer_or.status();

  auto result_literal_or = client->ShapedBufferToLiteral(result_buffer_or.value());
  ASSERT_TRUE(result_literal_or.ok()) << result_literal_or.status();
  LiteralTestUtil::ExpectR1Equal<float>(
      {2.0f, 4.0f, 6.0f}, result_literal_or.value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
