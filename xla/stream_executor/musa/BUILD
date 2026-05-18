load("//xla:xla.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "musa_status",
    srcs = ["musa_status.cc"],
    hdrs = ["musa_status.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_musa//musa:musa_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "musa_platform_id",
    srcs = ["musa_platform_id.cc"],
    hdrs = ["musa_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "musa_context",
    srcs = ["musa_context.cc"],
    hdrs = ["musa_context.h"],
    deps = [
        ":musa_status",
        "@local_config_musa//musa:musa_headers",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "musa_event",
    hdrs = ["musa_event.h"],
    deps = [
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream_executor_internal",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_stream",
    srcs = ["musa_stream.cc"],
    hdrs = ["musa_stream.h"],
    deps = [
        ":musa_context",
        ":musa_event",
        ":musa_status",
        "//xla/stream_executor:stream_common",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_musa//musa:musa_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "musa_command_buffer",
    srcs = ["musa_command_buffer.cc"],
    hdrs = ["musa_command_buffer.h"],
    deps = [
        ":musa_context",
        ":musa_kernel",
        ":musa_status",
        ":musa_stream",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@local_config_musa//musa:musa_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "musa_kernel",
    srcs = ["musa_kernel.cc"],
    hdrs = ["musa_kernel.h"],
    deps = [
        ":musa_status",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:stream_executor_internal",
        "@local_config_musa//musa:musa_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "mublas_plugin",
    srcs = ["musa_blas.cc"],
    hdrs = [
        "mublas_wrapper.h",
        "musa_blas.h",
    ],
    deps = [
        ":musa_executor",
        ":musa_platform_id",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_musa//musa:musa",
        "@local_config_musa//musa:musa_rpath",
        "@local_config_musa//musa:mublas",
        "@local_config_musa//musa:mudnn",
        "@local_config_musa//musa:musa_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/util:determinism_for_kernels",
    ],
    alwayslink = True,
)

cc_library(
    name = "musa_executor",
    srcs = ["musa_executor.cc"],
    hdrs = ["musa_executor.h"],
    deps = [
        "//xla/stream_executor:blas",
        ":musa_command_buffer",
        ":musa_context",
        ":musa_event",
        ":musa_kernel",
        ":musa_platform_id",
        ":musa_status",
        ":musa_stream",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_musa//musa:musa_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "musa_platform",
    srcs = ["musa_platform.cc"],
    hdrs = ["musa_platform.h"],
    deps = [
        ":musa_executor",
        ":mublas_plugin",
        ":musa_platform_id",
        ":musa_status",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_musa//musa:musa",
        "@local_config_musa//musa:musa_rpath",
        "@local_config_musa//musa:musa_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "musa_kernel_test",
    size = "small",
    srcs = ["musa_kernel_test.cc"],
    linkopts = [
        "-L/usr/local/musa/lib",
        "-lmusa",
        "-lmusart",
    ],
    deps = [
        ":musa_platform",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor/cuda:cuda_test_kernels",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "musa_event_test",
    size = "small",
    srcs = ["musa_event_test.cc"],
    linkopts = [
        "-L/usr/local/musa/lib",
        "-lmusa",
        "-lmusart",
    ],
    deps = [
        ":musa_platform",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "musa_command_buffer_test",
    size = "small",
    srcs = ["musa_command_buffer_test.cc"],
    linkopts = [
        "-L/usr/local/musa/lib",
        "-lmusa",
        "-lmusart",
    ],
    deps = [
        ":musa_platform",
        "//xla/stream_executor",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor/cuda:cuda_test_kernels",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "musa_blas_mudnn_test",
    size = "medium",
    srcs = ["musa_blas_mudnn_test.cc"],
    linkopts = [
        "-L/usr/local/musa/lib",
        "-lmusa",
        "-lmusart",
    ],
    deps = [
        ":musa_platform",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
