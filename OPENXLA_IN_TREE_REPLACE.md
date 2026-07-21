# OpenXLA In-Tree Replace Notes

## Scope

This repository keeps TensorFlow `compiler/jit` and `tf2xla` as the host side
and vendors newer OpenXLA code into `third_party/xla`.

The intended edit boundary is:

- `third_party/gpus/**`
- `third_party/xla/**`
- minimal glue in `tensorflow/workspace3.bzl`

The intended non-edit boundary is:

- `tensorflow/compiler/jit/**`
- `tensorflow/compiler/tf2xla/**`
- `tensorflow/core/**`

## Frozen Host Boundary

The host side is considered stable as long as the vendored XLA tree continues
to satisfy the contracts below.

### TensorFlow entry points to keep unchanged

- `tensorflow/compiler/jit/xla_device_compiler_client.cc`
- `tensorflow/compiler/jit/xla_launch_util.cc`
- `tensorflow/compiler/jit/xla_compiler_options_util.cc`
- `tensorflow/compiler/jit/kernels/xla_ops.cc`
- `tensorflow/compiler/tf2xla/xla_compiler.h`
- `tensorflow/compiler/tf2xla/xla_compiler.cc`

### XLA symbols that must remain callable

- `xla::LocalClient`
- `xla::LocalExecutable`
- `xla::ExecutableBuildOptions`
- `xla::ExecutableRunOptions`
- `xla::Compiler`
- `xla::Compiler::RunHloPasses`
- `xla::Compiler::RunBackend`

### Required flow

The vendored OpenXLA tree must preserve this host-visible flow:

1. `XlaCompiler` produces `xla::XlaComputation`.
2. `xla::LocalClient::Compile` accepts the computation and argument layouts.
3. XLA lowers through `Compiler::RunHloPasses` and `Compiler::RunBackend`.
4. TensorFlow receives `xla::LocalExecutable` and executes it through
   `ExecutableRunOptions`.

## Vendoring Categories

When importing from `@openxla`, sort files into the buckets below instead of
doing directory-wide blind replacement.

### Usually reusable with minor edits

- `xla/client/**` source files that already exist on both sides
- `xla/service/**` source files that preserve the host ABI above
- `xla/stream_executor/**` platform-independent code
- `xla/stream_executor/musa/**`

### Usually needs label or BUILD adaptation

- files that reference `@tsl//...`
- files that reference `//xla/tsl/...`
- files that load `//xla:xla.default.bzl`
- files that depend on OpenXLA-only package layout under `xla/pjrt/**`
- files that add MUSA conditions in `xla/service/BUILD`

### Usually not copied as-is

- `openxla/WORKSPACE`
- `openxla/.bazelrc`
- OpenXLA root-level toolchain setup
- OpenXLA `third_party/tsl` directory as a full replacement for the vendored
  TSL subtree in TensorFlow

## Repo Label Compatibility

The current compatibility strategy is:

1. Keep TensorFlow's existing vendored repos:
   - `@local_xla`
   - `@local_tsl`
2. Also expose `@tsl` from `tensorflow/workspace3.bzl`.
3. Prefer changing as little imported upstream code as possible.

This allows imported OpenXLA BUILD files to resolve `@tsl//...` while existing
TensorFlow-vendored code continues to use `@local_tsl//...`.

## ABI Checklist

Before syncing any OpenXLA subtree, verify the items below.

### Client ABI

- `xla/client/local_client.h`
- `xla/client/local_client.cc`
- `xla/client/executable_build_options.h`

Checks:

- `LocalClient::Compile`
- `LocalClient::CompileAheadOfTime`
- `LocalClient::Load`
- `default_device_ordinal()`

### Executable ABI

- `xla/service/executable.h`
- `xla/client/local_client.h`

Checks:

- `LocalExecutable`
- `Executable`
- `ExecutableBuildOptions`
- `ExecutableRunOptions`

### Compiler ABI

- `xla/service/compiler.h`
- `xla/service/service.cc`

Checks:

- `Compiler::RunHloPasses`
- `Compiler::RunBackend`
- `Service::BuildExecutable`

### TensorFlow host call sites

- `tensorflow/compiler/jit/xla_device_compiler_client.cc`
- `tensorflow/compiler/jit/xla_launch_util.cc`
- `tensorflow/compiler/tf2xla/xla_compiler.cc`

Checks:

- no signature drift visible to TensorFlow
- no required migration into `tensorflow/compiler/**`
- no new runtime dependency that escapes `third_party/xla`

## MUSA / MTGPU Order On New XLA Baseline

Once the in-tree OpenXLA baseline is stable, reintroduce MUSA/MTGPU in this
order:

1. `third_party/gpus/musa` repository rule and `--config=musa`
2. `xla/stream_executor/musa`
3. `xla/service/gpu/mtgpu_compiler*`
4. `xla/service/gpu/llvm_gpu_backend/mtgpu_backend*`

Reasoning:

- `stream_executor/musa` is the lowest platform layer.
- `mtgpu_compiler` depends on newer GPU compiler and autotuning interfaces.
- `mtgpu_backend` depends on MTGPU LLVM toolchain behavior and device-lib
  plumbing.

## Practical Rule

If an upstream OpenXLA change can be absorbed by:

- editing only `third_party/xla/**`, or
- editing `third_party/xla/**` plus one small workspace glue change,

prefer that over changing TensorFlow host code.
