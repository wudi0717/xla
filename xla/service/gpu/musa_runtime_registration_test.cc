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

#include "xla/service/computation_placer.h"
#include "xla/service/platform_util.h"
#include "xla/service/transfer_manager.h"
#include "xla/test.h"

namespace xla {
namespace gpu {
namespace {

TEST(MusaRuntimeRegistrationTest, TransferManagerAndComputationPlacerRegistered) {
  auto platform_or = PlatformUtil::GetPlatform("MUSA");
  if (!platform_or.ok()) {
    GTEST_SKIP() << platform_or.status();
  }
  se::Platform* platform = platform_or.value();

  EXPECT_TRUE(TransferManager::GetForPlatform(platform).ok());
  EXPECT_TRUE(ComputationPlacer::GetForPlatform(platform).ok());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
