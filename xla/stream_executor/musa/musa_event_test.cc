#include <cstdint>
#include <memory>
#include <vector>

#include "xla/stream_executor/event.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/test.h"

namespace stream_executor::musa {
namespace {

tsl::StatusOr<StreamExecutor*> GetExecutor() {
  TF_ASSIGN_OR_RETURN(Platform * platform,
                      MultiPlatformManager::PlatformWithName("MUSA"));
  if (platform->VisibleDeviceCount() <= 0) {
    return tsl::errors::NotFound("No visible MUSA device");
  }
  return platform->ExecutorForDevice(0);
}

TEST(MusaEventTest, RecordWaitAndPoll) {
  auto executor_or = GetExecutor();
  if (!executor_or.ok()) {
    GTEST_SKIP() << executor_or.status();
  }
  StreamExecutor* executor = executor_or.value();

  auto stream_or = executor->CreateStream();
  ASSERT_TRUE(stream_or.ok());
  std::unique_ptr<Stream> stream = std::move(stream_or.value());
  ASSERT_TRUE(stream->ok());

  Event event(executor);
  ASSERT_TRUE(event.Init());

  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = sizeof(int32_t) * kLength;

  DeviceMemory<int32_t> buffer = executor->AllocateArray<int32_t>(kLength, 0);
  std::vector<int32_t> host(kLength, 0);

  ASSERT_TRUE(stream->ThenMemset32(&buffer, 7, kBytes).ok());
  stream->ThenRecordEvent(&event);

  auto status = event.PollForStatus();
  EXPECT_TRUE(status == Event::Status::kPending ||
              status == Event::Status::kComplete);

  stream->ThenWaitFor(&event);
  ASSERT_TRUE(stream->ThenMemcpy(host.data(), buffer, kBytes).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  EXPECT_EQ(host, (std::vector<int32_t>{7, 7, 7, 7}));
  EXPECT_EQ(event.PollForStatus(), Event::Status::kComplete);
}

}  // namespace
}  // namespace stream_executor::musa
