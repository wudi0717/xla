#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xla/stream_executor/command_buffer.h"
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

tsl::StatusOr<std::pair<StreamExecutor*, StreamExecutor*>> GetPeerExecutors() {
  TF_ASSIGN_OR_RETURN(Platform * platform,
                      MultiPlatformManager::PlatformWithName("MUSA"));
  if (platform->VisibleDeviceCount() < 2) {
    return tsl::errors::NotFound("Need at least 2 visible MUSA devices");
  }
  TF_ASSIGN_OR_RETURN(StreamExecutor * first, platform->ExecutorForDevice(0));
  TF_ASSIGN_OR_RETURN(StreamExecutor * second, platform->ExecutorForDevice(1));
  return std::make_pair(first, second);
}

TEST(MusaCommandBufferTest, LaunchSingleKernelAndUpdate) {
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
  DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(kLength, 0);

  ASSERT_TRUE(stream->ThenMemset32(&a, 1, kBytes).ok());
  ASSERT_TRUE(stream->ThenMemset32(&b, 2, kBytes).ok());
  ASSERT_TRUE(stream->ThenMemZero(&c, kBytes).ok());
  ASSERT_TRUE(stream->ThenMemZero(&d, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  auto cmd_buffer = CommandBuffer::Create(executor).value();
  ASSERT_TRUE(cmd_buffer.Launch(add, ThreadDim(), BlockDim(4), a, b, c).ok());
  ASSERT_TRUE(cmd_buffer.Finalize().ok());
  ASSERT_TRUE(executor->Submit(stream.get(), cmd_buffer).ok());

  std::vector<int32_t> dst(kLength, 42);
  ASSERT_TRUE(stream->ThenMemcpy(dst.data(), c, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  EXPECT_EQ(dst, (std::vector<int32_t>{3, 3, 3, 3}));

  ASSERT_TRUE(cmd_buffer.Update().ok());
  ASSERT_TRUE(cmd_buffer.Launch(add, ThreadDim(), BlockDim(4), a, b, d).ok());
  ASSERT_TRUE(cmd_buffer.Finalize().ok());
  ASSERT_TRUE(executor->Submit(stream.get(), cmd_buffer).ok());

  std::fill(dst.begin(), dst.end(), 42);
  ASSERT_TRUE(stream->ThenMemcpy(dst.data(), d, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  EXPECT_EQ(dst, (std::vector<int32_t>{3, 3, 3, 3}));
}

TEST(MusaCommandBufferTest, MemcpyDeviceToDeviceAndUpdate) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());

  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = sizeof(int32_t) * kLength;

  DeviceMemory<int32_t> src = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> dst = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> src2 = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> dst2 = executor->AllocateArray<int32_t>(kLength, 0);

  std::vector<int32_t> host_src = {1, 2, 3, 4};
  std::vector<int32_t> host_src2 = {5, 6, 7, 8};
  std::vector<int32_t> host_dst(kLength, 0);

  ASSERT_TRUE(stream->Memcpy(&src, host_src.data(), kBytes).ok());
  ASSERT_TRUE(stream->Memcpy(&src2, host_src2.data(), kBytes).ok());
  ASSERT_TRUE(stream->MemZero(&dst, kBytes).ok());
  ASSERT_TRUE(stream->MemZero(&dst2, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  auto cmd_buffer = CommandBuffer::Create(executor).value();
  ASSERT_TRUE(cmd_buffer.MemcpyDeviceToDevice(&dst, src, kBytes).ok());
  ASSERT_TRUE(cmd_buffer.Finalize().ok());
  ASSERT_TRUE(executor->Submit(stream.get(), cmd_buffer).ok());
  ASSERT_TRUE(stream->Memcpy(host_dst.data(), dst, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  ASSERT_EQ(host_dst, host_src);

  std::fill(host_dst.begin(), host_dst.end(), 0);
  ASSERT_TRUE(cmd_buffer.Update().ok());
  ASSERT_TRUE(cmd_buffer.MemcpyDeviceToDevice(&dst2, src2, kBytes).ok());
  ASSERT_TRUE(cmd_buffer.Finalize().ok());
  ASSERT_TRUE(executor->Submit(stream.get(), cmd_buffer).ok());
  ASSERT_TRUE(stream->Memcpy(host_dst.data(), dst2, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  ASSERT_EQ(host_dst, host_src2);
}

TEST(MusaCommandBufferTest, TraceMemcpyDeviceToDevice) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());

  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = sizeof(int32_t) * kLength;

  DeviceMemory<int32_t> src = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> dst = executor->AllocateArray<int32_t>(kLength, 0);

  std::vector<int32_t> host_src = {9, 10, 11, 12};
  std::vector<int32_t> host_dst(kLength, 0);

  ASSERT_TRUE(stream->Memcpy(&src, host_src.data(), kBytes).ok());
  ASSERT_TRUE(stream->MemZero(&dst, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  auto cmd_buffer = CommandBuffer::Trace(executor, [&](Stream* trace_stream) {
    return trace_stream->Memcpy(&dst, src, kBytes);
  });

  ASSERT_TRUE(cmd_buffer.ok());
  ASSERT_TRUE(executor->Submit(stream.get(), *cmd_buffer).ok());
  ASSERT_TRUE(stream->Memcpy(host_dst.data(), dst, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  ASSERT_EQ(host_dst, host_src);
}

TEST(MusaCommandBufferTest, TraceSingleKernel) {
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
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  auto cmd_buffer = CommandBuffer::Trace(executor, [&](Stream* trace_stream) {
    return trace_stream->ThenLaunch(ThreadDim(), BlockDim(4), add, a, b, c);
  });

  ASSERT_TRUE(cmd_buffer.ok());
  ASSERT_TRUE(executor->Submit(stream.get(), *cmd_buffer).ok());

  std::vector<int32_t> dst(kLength, 42);
  ASSERT_TRUE(stream->ThenMemcpy(dst.data(), c, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  EXPECT_EQ(dst, (std::vector<int32_t>{3, 3, 3, 3}));
}

TEST(MusaCommandBufferTest, TracedCommandBufferCannotUpdate) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());

  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = sizeof(int32_t) * kLength;

  DeviceMemory<int32_t> src = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> dst = executor->AllocateArray<int32_t>(kLength, 0);

  std::vector<int32_t> host_src = {1, 1, 1, 1};

  ASSERT_TRUE(stream->Memcpy(&src, host_src.data(), kBytes).ok());
  ASSERT_TRUE(stream->MemZero(&dst, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  auto cmd_buffer = CommandBuffer::Trace(executor, [&](Stream* trace_stream) {
    return trace_stream->Memcpy(&dst, src, kBytes);
  });

  ASSERT_TRUE(cmd_buffer.ok());
  EXPECT_FALSE(cmd_buffer->Update().ok());
}

TEST(MusaCommandBufferTest, HostCallbackRunsBeforeBlockReturns) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());

  std::atomic<bool> callback_called = false;
  stream->ThenDoHostCallback([&callback_called]() {
    callback_called.store(true, std::memory_order_release);
  });

  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  EXPECT_TRUE(callback_called.load(std::memory_order_acquire));
}

TEST(MusaCommandBufferTest, ZeroSizedDeviceToDeviceMemcpySucceeds) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());

  DeviceMemory<int32_t> src = executor->AllocateArray<int32_t>(1, 0);
  DeviceMemory<int32_t> dst = executor->AllocateArray<int32_t>(1, 0);

  ASSERT_TRUE(executor->SynchronousMemcpy(&dst, src, /*size=*/0));
  ASSERT_TRUE(stream->Memcpy(&dst, src, /*size=*/0).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
}

TEST(MusaCommandBufferTest, EnablePeerAccessIsIdempotent) {
  auto executors_or = GetPeerExecutors();
  if (!executors_or.ok()) {
    GTEST_SKIP() << executors_or.status();
  }

  StreamExecutor* first = executors_or->first;
  StreamExecutor* second = executors_or->second;
  if (!first->CanEnablePeerAccessTo(second) ||
      !second->CanEnablePeerAccessTo(first)) {
    GTEST_SKIP() << "Peer access is not available between visible MUSA devices";
  }

  ASSERT_TRUE(first->EnablePeerAccessTo(second).ok());
  ASSERT_TRUE(first->EnablePeerAccessTo(second).ok());
  ASSERT_TRUE(second->EnablePeerAccessTo(first).ok());
  ASSERT_TRUE(second->EnablePeerAccessTo(first).ok());
}

TEST(MusaCommandBufferTest, PeerMemcpyDeviceToDeviceIfAvailable) {
  auto executors_or = GetPeerExecutors();
  if (!executors_or.ok()) {
    GTEST_SKIP() << executors_or.status();
  }

  StreamExecutor* dst_executor = executors_or->first;
  StreamExecutor* src_executor = executors_or->second;
  if (!dst_executor->CanEnablePeerAccessTo(src_executor) ||
      !src_executor->CanEnablePeerAccessTo(dst_executor)) {
    GTEST_SKIP() << "Peer access is not available between visible MUSA devices";
  }

  ASSERT_TRUE(dst_executor->EnablePeerAccessTo(src_executor).ok());
  ASSERT_TRUE(src_executor->EnablePeerAccessTo(dst_executor).ok());

  auto stream_or = dst_executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());

  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = sizeof(int32_t) * kLength;

  DeviceMemory<int32_t> src = src_executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> dst = dst_executor->AllocateArray<int32_t>(kLength, 0);

  std::vector<int32_t> host_src = {13, 14, 15, 16};
  std::vector<int32_t> host_dst(kLength, 0);

  ASSERT_TRUE(
      src_executor->SynchronousMemcpyH2D(host_src.data(), kBytes, &src).ok());
  ASSERT_TRUE(stream->Memcpy(&dst, src, kBytes).ok());
  ASSERT_TRUE(stream->Memcpy(host_dst.data(), dst, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  ASSERT_EQ(host_dst, host_src);
}

TEST(MusaCommandBufferTest, SynchronousPeerMemcpyDeviceToDeviceIfAvailable) {
  auto executors_or = GetPeerExecutors();
  if (!executors_or.ok()) {
    GTEST_SKIP() << executors_or.status();
  }

  StreamExecutor* dst_executor = executors_or->first;
  StreamExecutor* src_executor = executors_or->second;
  if (!dst_executor->CanEnablePeerAccessTo(src_executor) ||
      !src_executor->CanEnablePeerAccessTo(dst_executor)) {
    GTEST_SKIP() << "Peer access is not available between visible MUSA devices";
  }

  ASSERT_TRUE(dst_executor->EnablePeerAccessTo(src_executor).ok());
  ASSERT_TRUE(src_executor->EnablePeerAccessTo(dst_executor).ok());

  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = sizeof(int32_t) * kLength;

  DeviceMemory<int32_t> src = src_executor->AllocateArray<int32_t>(kLength, 0);
  DeviceMemory<int32_t> dst = dst_executor->AllocateArray<int32_t>(kLength, 0);

  std::vector<int32_t> host_src = {21, 22, 23, 24};
  std::vector<int32_t> host_dst(kLength, 0);

  ASSERT_TRUE(
      src_executor->SynchronousMemcpyH2D(host_src.data(), kBytes, &src).ok());
  ASSERT_TRUE(dst_executor->SynchronousMemcpy(&dst, src, kBytes));
  ASSERT_TRUE(dst_executor->SynchronousMemcpyD2H(dst, kBytes, host_dst.data()).ok());
  ASSERT_EQ(host_dst, host_src);
}

}  // namespace
}  // namespace stream_executor::musa
