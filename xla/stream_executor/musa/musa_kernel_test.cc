#include <cstdint>
#include <memory>
#include <vector>

#include "xla/stream_executor/cuda/cuda_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/test.h"

namespace stream_executor::musa {
namespace {

using AddI32Kernel = TypedKernel<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                                 DeviceMemory<int32_t>>;

tsl::StatusOr<StreamExecutor*> GetExecutor() {
  TF_ASSIGN_OR_RETURN(Platform * platform,
                      MultiPlatformManager::PlatformWithName("MUSA"));
  if (platform->VisibleDeviceCount() <= 0) {
    return tsl::errors::NotFound("No visible MUSA device");
  }
  return platform->ExecutorForDevice(0);
}

TEST(MusaKernelTest, Add) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddCudaPtxInMemory(cuda::internal::kAddI32Kernel, "add");

  AddI32Kernel add(executor);
  auto kernel_status = executor->GetKernel(spec, &add);
  if (!kernel_status.ok()) {
    GTEST_SKIP() << kernel_status;
  }

  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = sizeof(int32_t) * kLength;

  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(kLength, 0);

  ASSERT_TRUE(stream->ThenMemset32(&a, 1, kBytes).ok());
  ASSERT_TRUE(stream->ThenMemset32(&b, 2, kBytes).ok());
  ASSERT_TRUE(stream->ThenMemZero(&c, kBytes).ok());
  ASSERT_TRUE(stream->ThenLaunch(ThreadDim(), BlockDim(4), add, a, b, c).ok());

  std::vector<int32_t> dst(kLength, 42);
  ASSERT_TRUE(stream->ThenMemcpy(dst.data(), c, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);
}

}  // namespace
}  // namespace stream_executor::musa
