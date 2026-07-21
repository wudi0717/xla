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

#include "xla/service/gpu/musa_small_gemm_accum_thunk.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/gemm_thunk.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {
namespace {

bool EnvExplicitlyTrue(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool MusaSmallGemmAccumCustomKernelAvailable() { return false; }

bool IsRowMajorF32(const MatrixLayout& layout) {
  return layout.dtype == F32 && layout.order == MatrixLayout::Order::kRowMajor;
}

bool IsScalarOne(complex128 value) {
  return value.real() == 1.0 && value.imag() == 0.0;
}

bool IsSmallAccumSupportedConfig(const GemmConfig& config, int64_t max_k,
                                 MusaSmallGemmAccumRewriteStats* stats) {
  if (!IsScalarOne(config.alpha) ||
      (config.beta != 0.0 && config.beta != 1.0) ||
      config.algorithm.has_value()) {
    ++stats->filtered_config;
    return false;
  }
  if (!IsRowMajorF32(config.lhs_layout) || !IsRowMajorF32(config.rhs_layout) ||
      !IsRowMajorF32(config.output_layout) ||
      config.lhs_layout.batch_size != 1 || config.rhs_layout.batch_size != 1 ||
      config.output_layout.batch_size != 1) {
    ++stats->filtered_layout;
    return false;
  }
  if (config.lhs_layout.num_cols != config.rhs_layout.num_rows ||
      config.lhs_layout.num_rows != config.output_layout.num_rows ||
      config.rhs_layout.num_cols != config.output_layout.num_cols ||
      config.lhs_layout.num_cols <= 0 ||
      config.lhs_layout.num_cols > max_k) {
    ++stats->filtered_size;
    return false;
  }
  return true;
}

bool SameOutputShape(const GemmConfig& a, const GemmConfig& b) {
  return a.output_layout.dtype == b.output_layout.dtype &&
         a.output_layout.num_rows == b.output_layout.num_rows &&
         a.output_layout.num_cols == b.output_layout.num_cols &&
         a.output_layout.order == b.output_layout.order &&
         a.output_layout.leading_dim_stride ==
             b.output_layout.leading_dim_stride &&
         a.compute_precision == b.compute_precision;
}

bool HasUnsafeOutputDependency(const std::vector<GemmThunk*>& gemms) {
  for (const GemmThunk* writer : gemms) {
    const BufferAllocation::Slice& output = writer->output_buffer();
    for (const GemmThunk* user : gemms) {
      if (output.OverlapsWith(user->lhs_buffer()) ||
          output.OverlapsWith(user->rhs_buffer())) {
        return true;
      }
    }
  }
  return false;
}

struct AccumChain {
  int64_t start = 0;
  int64_t size = 0;
};

std::vector<AccumChain> FindAccumChains(
    const std::vector<GemmThunk*>& run_gemms, int64_t min_chain_size,
    int64_t max_chain_size, int64_t max_k,
    MusaSmallGemmAccumRewriteStats* stats) {
  std::vector<AccumChain> chains;
  for (int64_t i = 0; i < run_gemms.size();) {
    GemmThunk* first = run_gemms[i];
    if (!IsSmallAccumSupportedConfig(first->config(), max_k, stats) ||
        first->config().beta != 0.0) {
      if (first->config().beta != 0.0) {
        ++stats->filtered_beta;
      }
      ++i;
      continue;
    }

    int64_t j = i + 1;
    while (j < run_gemms.size()) {
      GemmThunk* current = run_gemms[j];
      if (!IsSmallAccumSupportedConfig(current->config(), max_k, stats)) {
        break;
      }
      if (!SameOutputShape(first->config(), current->config()) ||
          first->output_buffer() != current->output_buffer() ||
          first->deterministic() != current->deterministic()) {
        break;
      }
      if (current->config().beta != 1.0) {
        ++stats->filtered_beta;
        break;
      }
      ++j;
    }

    const int64_t size = j - i;
    if (size < min_chain_size) {
      ++stats->filtered_short_chain;
      ++i;
      continue;
    }

    std::vector<GemmThunk*> chain_gemms(run_gemms.begin() + i,
                                        run_gemms.begin() + j);
    if (HasUnsafeOutputDependency(chain_gemms)) {
      ++stats->filtered_dependency;
      ++i;
      continue;
    }

    int64_t remaining = size;
    int64_t chunk_start = i;
    while (remaining >= min_chain_size) {
      const int64_t chunk_size = std::min<int64_t>(remaining, max_chain_size);
      chains.push_back({chunk_start, chunk_size});
      chunk_start += chunk_size;
      remaining -= chunk_size;
    }
    i = j;
  }
  return chains;
}

}  // namespace

StatusOr<std::unique_ptr<const ThunkSequence>> RewriteMusaSmallGemmAccumThunks(
    std::unique_ptr<const ThunkSequence> sequence,
    const MusaSmallGemmAccumRewriteOptions& options,
    MusaSmallGemmAccumRewriteStats* stats) {
  stats->original_thunks = sequence == nullptr ? 0 : sequence->size();
  if (sequence == nullptr || sequence->empty()) {
    stats->rewritten_thunks = stats->original_thunks;
    return sequence;
  }
  if (options.require_custom_kernel &&
      !MusaSmallGemmAccumCustomKernelAvailable()) {
    stats->rewritten_thunks = stats->original_thunks;
    ++stats->filtered_custom_kernel_unavailable;
    if (options.log) {
      LOG(INFO) << "[MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS] changed=false"
                << " original_thunks=" << stats->original_thunks
                << " rewritten_thunks=" << stats->rewritten_thunks
                << " require_custom_kernel=" << options.require_custom_kernel
                << " custom_kernel_available=0"
                << " filtered_custom_kernel_unavailable="
                << stats->filtered_custom_kernel_unavailable;
    }
    return sequence;
  }

  std::unique_ptr<ThunkSequence> input(
      const_cast<ThunkSequence*>(sequence.release()));
  auto output = std::make_unique<ThunkSequence>();
  const int64_t min_chain_size = std::max<int64_t>(2, options.min_chain_size);
  const int64_t max_chain_size =
      std::max<int64_t>(min_chain_size, options.max_chain_size);
  const int64_t max_k = std::max<int64_t>(1, options.max_k);

  for (int64_t i = 0; i < input->size();) {
    if ((*input)[i]->AsGemmThunk() == nullptr) {
      output->push_back(std::move((*input)[i]));
      ++i;
      continue;
    }

    const int64_t run_start = i;
    while (i < input->size() && (*input)[i]->AsGemmThunk() != nullptr) {
      ++stats->gemm_thunks;
      ++i;
    }
    const int64_t run_end = i;
    const int64_t run_size = run_end - run_start;
    if (run_size >= min_chain_size) {
      ++stats->candidate_runs;
      stats->candidate_gemms += run_size;
    }

    std::vector<GemmThunk*> run_gemms;
    run_gemms.reserve(run_size);
    for (int64_t j = run_start; j < run_end; ++j) {
      run_gemms.push_back((*input)[j]->AsGemmThunk());
    }

    std::vector<AccumChain> chains =
        FindAccumChains(run_gemms, min_chain_size, max_chain_size, max_k,
                        stats);
    std::vector<int64_t> chain_at(run_size, -1);
    for (int64_t chain_id = 0; chain_id < chains.size(); ++chain_id) {
      for (int64_t local = chains[chain_id].start;
           local < chains[chain_id].start + chains[chain_id].size; ++local) {
        chain_at[local] = chain_id;
      }
    }

    for (int64_t local = 0; local < run_size;) {
      const int64_t chain_id = chain_at[local];
      if (chain_id < 0) {
        output->push_back(std::move((*input)[run_start + local]));
        ++local;
        continue;
      }
      const AccumChain& chain = chains[chain_id];
      if (local != chain.start) {
        ++local;
        continue;
      }

      std::vector<GemmConfig> configs;
      std::vector<BufferAllocation::Slice> lhs_buffers;
      std::vector<BufferAllocation::Slice> rhs_buffers;
      configs.reserve(chain.size);
      lhs_buffers.reserve(chain.size);
      rhs_buffers.reserve(chain.size);
      for (int64_t offset = 0; offset < chain.size; ++offset) {
        const GemmThunk* gemm = run_gemms[chain.start + offset];
        configs.push_back(gemm->config());
        lhs_buffers.push_back(gemm->lhs_buffer());
        rhs_buffers.push_back(gemm->rhs_buffer());
      }

      const GemmThunk* first = run_gemms[chain.start];
      Thunk::ThunkInfo info((*input)[run_start + chain.start]->op());
      info.profile_annotation =
          absl::StrCat("musa_small_gemm_accum(count=", chain.size, ")");
      output->push_back(std::make_unique<MusaSmallGemmAccumThunk>(
          info, std::move(configs), std::move(lhs_buffers),
          std::move(rhs_buffers), first->output_buffer(), first->deterministic(),
          first->config().beta != 0.0));
      for (int64_t offset = 0; offset < chain.size; ++offset) {
        (*input)[run_start + chain.start + offset].reset();
      }
      ++stats->chains_created;
      stats->gemms_accumulated += chain.size;
      local = chain.start + chain.size;
    }
  }

  stats->rewritten_thunks = output->size();
  stats->estimated_launch_reduction =
      stats->gemms_accumulated - stats->chains_created;
  if (options.log) {
    LOG(INFO) << "[MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS] changed="
              << (stats->chains_created > 0)
              << " original_thunks=" << stats->original_thunks
              << " rewritten_thunks=" << stats->rewritten_thunks
              << " gemm_thunks=" << stats->gemm_thunks
              << " candidate_runs=" << stats->candidate_runs
              << " candidate_gemms=" << stats->candidate_gemms
              << " chains_created=" << stats->chains_created
              << " gemms_accumulated=" << stats->gemms_accumulated
              << " estimated_launch_reduction="
              << stats->estimated_launch_reduction
              << " filtered_short_chain=" << stats->filtered_short_chain
              << " filtered_beta=" << stats->filtered_beta
              << " filtered_config=" << stats->filtered_config
              << " filtered_layout=" << stats->filtered_layout
              << " filtered_size=" << stats->filtered_size
              << " filtered_dependency=" << stats->filtered_dependency
              << " filtered_custom_kernel_unavailable="
              << stats->filtered_custom_kernel_unavailable
              << " require_custom_kernel=" << options.require_custom_kernel
              << " custom_kernel_available="
              << MusaSmallGemmAccumCustomKernelAvailable()
              << " max_k=" << max_k
              << " min_chain_size=" << min_chain_size
              << " max_chain_size=" << max_chain_size;
  }
  return std::unique_ptr<const ThunkSequence>(output.release());
}

MusaSmallGemmAccumThunk::MusaSmallGemmAccumThunk(
    ThunkInfo thunk_info, std::vector<GemmConfig> configs,
    std::vector<BufferAllocation::Slice> lhs_buffers,
    std::vector<BufferAllocation::Slice> rhs_buffers,
    const BufferAllocation::Slice& output_buffer, bool deterministic,
    bool starts_from_existing_output)
    : Thunk(Kind::kGemm, thunk_info),
      configs_(std::move(configs)),
      lhs_buffers_(std::move(lhs_buffers)),
      rhs_buffers_(std::move(rhs_buffers)),
      output_buffer_(output_buffer),
      deterministic_(deterministic),
      starts_from_existing_output_(starts_from_existing_output) {}

Status MusaSmallGemmAccumThunk::ExecuteOnStream(const ExecuteParams& params) {
  if (EnvExplicitlyTrue("MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL")) {
    return absl::UnimplementedError(
        "MUSA small GEMM accum custom kernel launch is not wired yet");
  }

  const BufferAllocations& allocs = *params.buffer_allocations;
  se::DeviceMemoryBase output = allocs.GetDeviceAddress(output_buffer_);
  for (int64_t i = 0; i < static_cast<int64_t>(configs_.size()); ++i) {
    Status gemm_status = RunGemm(
        configs_[i], allocs.GetDeviceAddress(lhs_buffers_[i]),
        allocs.GetDeviceAddress(rhs_buffers_[i]), output, deterministic_,
        params.stream);
    if (!gemm_status.ok()) {
      LOG(ERROR) << "[MUSA_SMALL_GEMM_ACCUM_THUNK] stage=fallback_gemm_failed "
                 << "index=" << i << " status=" << gemm_status;
    }
    TF_RETURN_IF_ERROR(gemm_status);
  }
  return OkStatus();
}

Status MusaSmallGemmAccumThunk::Initialize(
    const GpuExecutable& /*executable*/, se::StreamExecutor* executor) {
  if (!executor->AsBlas() &&
      !EnvExplicitlyTrue("MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL")) {
    return absl::InternalError("Failed to initialize BLAS support");
  }
  return OkStatus();
}

std::string MusaSmallGemmAccumThunk::ToStringExtra(int /*indent*/) const {
  int64_t total_k = 0;
  int64_t max_k = 0;
  for (const GemmConfig& config : configs_) {
    total_k += config.lhs_layout.num_cols;
    max_k = std::max<int64_t>(max_k, config.lhs_layout.num_cols);
  }
  return absl::StrCat("mode=small_gemm_accum, gemm_count=", configs_.size(),
                      ", m=", configs_.empty()
                                  ? 0
                                  : configs_.front().output_layout.num_rows,
                      ", n=", configs_.empty()
                                  ? 0
                                  : configs_.front().output_layout.num_cols,
                      ", total_k=", total_k, ", max_k=", max_k,
                      ", starts_from_existing_output=",
                      starts_from_existing_output_);
}

}  // namespace gpu
}  // namespace xla
