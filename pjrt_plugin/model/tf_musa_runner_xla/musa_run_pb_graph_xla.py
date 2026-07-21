#!/usr/bin/env python3
import argparse
import ast
import ctypes
from contextlib import contextmanager
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MUSA_PLUGIN_PATH = (
    REPO_ROOT / "bazel-bin" / "pjrt_plugin" / "libmusa_pjrt_plugin.so"
)
DEFAULT_MUSA_TF_ADAPTER_PATH = (
    REPO_ROOT / "bazel-bin" / "pjrt_plugin" / "libmusa_tf215_npd_adapter.so"
)
DEFAULT_XLA_DUMP_DIR = SCRIPT_DIR / "xla_dump"

GRAPH_DUMP_ENV = "MUSA_DUMP_GRAPHDEF"
GRAPH_DUMP_DIR_ENV = "MUSA_DUMP_GRAPHDEF_DIR"
GRAPH_DUMP_STAGE_SUFFIXES = {
    "after_default_grappler_before_musa_optimizer": "initial",
    "after_custom_fusion_pattern": "after_fusion",
    "final_after_all_musa_passes": "final",
}


def _get_cli_arg(argv, flag, default=None):
    prefix = f"{flag}="
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


def _get_optional_cli_arg(argv, flag, default=None, const="on"):
    prefix = f"{flag}="
    for index, arg in enumerate(argv):
        if arg == flag:
            if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
                return argv[index + 1]
            return const
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


def _has_cli_flag(argv, flag):
    prefix = f"{flag}="
    return flag in argv or any(arg.startswith(prefix) for arg in argv)


def _max_int_from_csv(value, default):
    if value is None:
        return default
    values = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            return default
    return max(values) if values else default


def _early_batch_size_arg(argv):
    return _max_int_from_csv(_get_cli_arg(argv, "--bs"), 1024)


def _early_large_batch_threshold():
    value = os.environ.get("MUSA_XLA_LARGE_BATCH_THRESHOLD", "512")
    try:
        return int(value)
    except ValueError:
        return 512


def _append_cli_option(argv, flag, value):
    argv.extend([flag, str(value)])








def _append_unique_flag(current_value: str, new_flag: str) -> str:
    tokens = current_value.split()
    if new_flag not in tokens:
        tokens.append(new_flag)
    return " ".join(tokens).strip()


def _set_flag_with_prefix(current_value: str, flag_prefix: str, new_flag: str) -> str:
    tokens = [token for token in current_value.split() if not token.startswith(flag_prefix)]
    tokens.append(new_flag)
    return " ".join(tokens).strip()




def _set_tristate_bool_env_value(env_name, value):
    value = str(value).strip().lower()
    if value in ("on", "true", "1"):
        os.environ[env_name] = "1"
    elif value in ("off", "false", "0"):
        os.environ[env_name] = "0"


def _set_tristate_bool_env_from_cli(argv, flag, env_name, default="auto"):
    value = _get_optional_cli_arg(argv, flag, default, const="on")
    _set_tristate_bool_env_value(env_name, value)


def _set_tristate_bool_env_from_cli_or_preserve(
    argv, flag, env_name, default=None
):
    if _has_cli_flag(argv, flag):
        _set_tristate_bool_env_from_cli(argv, flag, env_name)
    elif env_name not in os.environ and default is not None:
        _set_tristate_bool_env_value(env_name, default)


def _set_value_env_from_cli_or_preserve(argv, flag, env_name, default=None):
    if _has_cli_flag(argv, flag):
        value = _get_cli_arg(argv, flag, default)
    elif env_name not in os.environ:
        value = default
    else:
        return
    if value is not None and value != "":
        os.environ[env_name] = str(value)




def _set_xla_gpu_runtime(runtime_mode):
    if runtime_mode not in ("classic_thunks", "xla_runtime"):
        return
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if runtime_mode == "classic_thunks":
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_gpu_enable_xla_runtime_executable=",
            "--xla_gpu_enable_xla_runtime_executable=false",
        )
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_gpu_enable_gpu2_runtime=",
            "--xla_gpu_enable_gpu2_runtime=false",
        )
    else:
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_gpu_enable_xla_runtime_executable=",
            "--xla_gpu_enable_xla_runtime_executable=true",
        )
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_gpu_enable_gpu2_runtime=",
            "--xla_gpu_enable_gpu2_runtime=false",
        )
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["MUSA_XLA_GPU_RUNTIME"] = runtime_mode


















def device_kind(device: Optional[str]) -> str:
    value = (device or "").upper()
    if "MUSA" in value:
        return "MUSA"
    if "GPU" in value or "CUDA" in value:
        return "CUDA"
    return "CPU"


def default_musa_plugin_path() -> str:
    return os.environ.get("MUSA_PJRT_PLUGIN_PATH", str(DEFAULT_MUSA_PLUGIN_PATH))


def default_musa_tf_adapter_path() -> str:
    return os.environ.get(
        "MUSA_TF_NPD_ADAPTER_PATH",
        str(DEFAULT_MUSA_TF_ADAPTER_PATH),
    )


def explicit_cli_flags(argv):
    flags = set()
    for arg in argv:
        text = str(arg)
        if text.startswith("--"):
            flags.add(text.split("=", 1)[0])
    return flags


META2_DEFAULT_PROFILE = {
    "xla_global_jit_level": ("--xla_global_jit_level", "on_2"),
    "rewrite_same_lhs_matmul": ("--rewrite_same_lhs_matmul", "on"),
    "rewrite_same_lhs_matmul_fuse_biasadd": (
        "--rewrite_same_lhs_matmul_fuse_biasadd",
        "on",
    ),
    "rewrite_concat_static_precompute": (
        "--rewrite_concat_static_precompute",
        "on",
    ),
    "pack_slice_feed": ("--pack_slice_feed", "on"),
}


META1_DEFAULT_PROFILE = {
    "musa_xla_gpu_runtime": ("--musa_xla_gpu_runtime", "classic_thunks"),
    "musa_xla_same_shape_dot_batcher": (
        "--musa_xla_same_shape_dot_batcher",
        "on",
    ),
    "musa_xla_same_shape_dot_batch_min_candidate_dots": (
        "--musa_xla_same_shape_dot_batch_min_candidate_dots",
        1,
    ),
    "musa_xla_same_shape_dot_batch_max_output_cols": (
        "--musa_xla_same_shape_dot_batch_max_output_cols",
        256,
    ),
    "musa_xla_same_shape_dot_batch_max_group_size": (
        "--musa_xla_same_shape_dot_batch_max_group_size",
        32,
    ),
    "musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch": (
        "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch",
        2097152,
    ),
    "musa_f32_fast_tf32": ("--musa_f32_fast_tf32", "on"),
    "musa_xla_classic_thunk_graph": (
        "--musa_xla_classic_thunk_graph",
        "on",
    ),
    "musa_xla_hot_tuple_softmax_kernel": (
        "--musa_xla_hot_tuple_softmax_kernel",
        "on",
    ),
    "pjrt_max_inflight_transfers": ("--pjrt_max_inflight_transfers", 1),
    "pjrt_max_inflight_executes": ("--pjrt_max_inflight_executes", 1),
    "pjrt_wait_transfer_done": ("--pjrt_wait_transfer_done", "on"),
    "pjrt_wait_execute_done": ("--pjrt_wait_execute_done", "on"),
    "pjrt_reuse_host_buffers": ("--pjrt_reuse_host_buffers", "on"),
    "pjrt_reuse_host_buffers_arena": (
        "--pjrt_reuse_host_buffers_arena",
        "on",
    ),
    "pjrt_cache_reused_buffer_views": (
        "--pjrt_cache_reused_buffer_views",
        "on",
    ),
    "pjrt_cache_reused_buffer_views_trust_lifetime": (
        "--pjrt_cache_reused_buffer_views_trust_lifetime",
        "on",
    ),
    "pjrt_reuse_host_buffers_arena_parallel_pack": (
        "--pjrt_reuse_host_buffers_arena_parallel_pack",
        "on",
    ),
    "pjrt_reuse_host_buffers_arena_pack_threads": (
        "--pjrt_reuse_host_buffers_arena_pack_threads",
        8,
    ),
    "pjrt_reuse_host_buffers_arena_pack_min_bytes": (
        "--pjrt_reuse_host_buffers_arena_pack_min_bytes",
        131072,
    ),
    "compact_slice_feed": ("--compact_slice_feed", "on"),
    "pack_slice_feed": ("--pack_slice_feed", "on"),
    "pack_slice_feed_ops": ("--pack_slice_feed_ops", "Slice"),
    "pack_slice_feed_min_saved_mib": ("--pack_slice_feed_min_saved_mib", 0.0),
    "pack_slice_feed_min_total_saved_mib": (
        "--pack_slice_feed_min_total_saved_mib",
        0.0,
    ),
    "pack_slice_feed_single_consumer_only": (
        "--pack_slice_feed_single_consumer_only",
        "on",
    ),
    "pack_slice_feed_max_direct_added_inputs": (
        "--pack_slice_feed_max_direct_added_inputs",
        0,
    ),
    "warmup": ("--warmup", 3),
    "run_iters": ("--run_iters", 10),
}


META3_DEFAULT_PROFILE = {
    "musa_xla_gpu_runtime": ("--musa_xla_gpu_runtime", "classic_thunks"),
    "musa_xla_classic_thunk_graph": (
        "--musa_xla_classic_thunk_graph",
        "on",
    ),
    "musa_xla_same_shape_dot_batcher": (
        "--musa_xla_same_shape_dot_batcher",
        "on",
    ),
    "musa_xla_same_shape_dot_batch_min_candidate_dots": (
        "--musa_xla_same_shape_dot_batch_min_candidate_dots",
        1,
    ),
    "musa_xla_same_shape_dot_batch_max_output_cols": (
        "--musa_xla_same_shape_dot_batch_max_output_cols",
        768,
    ),
    "musa_xla_same_shape_dot_batch_max_group_size": (
        "--musa_xla_same_shape_dot_batch_max_group_size",
        96,
    ),
    "musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch": (
        "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch",
        2500000,
    ),
    "musa_xla_warp_row_reduction_kernel": (
        "--musa_xla_warp_row_reduction_kernel",
        "on",
    ),
    "musa_xla_warp_row_reduction_reducers": (
        "--musa_xla_warp_row_reduction_reducers",
        "all",
    ),
    "musa_xla_warp_row_reduction_min_data_elements": (
        "--musa_xla_warp_row_reduction_min_data_elements",
        10000000,
    ),
    "musa_xla_warp_row_reduction_threads_per_block": (
        "--musa_xla_warp_row_reduction_threads_per_block",
        64,
    ),
    "musa_xla_tuple_warp_row_reduction_kernel": (
        "--musa_xla_tuple_warp_row_reduction_kernel",
        "on",
    ),
    "musa_xla_mixed_tuple_warp_row_reduction_kernel": (
        "--musa_xla_mixed_tuple_warp_row_reduction_kernel",
        "on",
    ),
    "musa_xla_mixed_tuple_warp_row_reduction_min_data_elements": (
        "--musa_xla_mixed_tuple_warp_row_reduction_min_data_elements",
        700000,
    ),
    "musa_xla_fusion_merger_materialize_reduction_producer": (
        "--musa_xla_fusion_merger_materialize_reduction_producer",
        "on",
    ),
    "musa_xla_fusion_merger_materialize_min_elements": (
        "--musa_xla_fusion_merger_materialize_min_elements",
        10000000,
    ),
    "musa_xla_fusion_merger_materialize_min_operands": (
        "--musa_xla_fusion_merger_materialize_min_operands",
        16,
    ),
    "pjrt_max_inflight_transfers": ("--pjrt_max_inflight_transfers", 1),
    "pjrt_max_inflight_executes": ("--pjrt_max_inflight_executes", 1),
    "pjrt_wait_transfer_done": ("--pjrt_wait_transfer_done", "on"),
    "pjrt_wait_execute_done": ("--pjrt_wait_execute_done", "on"),
    "pjrt_reuse_host_buffers": ("--pjrt_reuse_host_buffers", "on"),
    "pjrt_reuse_host_buffers_arena": (
        "--pjrt_reuse_host_buffers_arena",
        "on",
    ),
    "pjrt_reuse_host_buffers_arena_parallel_pack": (
        "--pjrt_reuse_host_buffers_arena_parallel_pack",
        "on",
    ),
    "pjrt_reuse_host_buffers_arena_pack_threads": (
        "--pjrt_reuse_host_buffers_arena_pack_threads",
        8,
    ),
    "pjrt_reuse_host_buffers_arena_pack_min_bytes": (
        "--pjrt_reuse_host_buffers_arena_pack_min_bytes",
        131072,
    ),
    "pjrt_reuse_host_buffers_arena_dirty_ranges": (
        "--pjrt_reuse_host_buffers_arena_dirty_ranges",
        "on",
    ),
    "pjrt_reuse_host_buffers_arena_pool_order_layout": (
        "--pjrt_reuse_host_buffers_arena_pool_order_layout",
        "on",
    ),
    "pjrt_reuse_host_buffers_trust_contents": (
        "--pjrt_reuse_host_buffers_trust_contents",
        "on",
    ),
    "pjrt_cache_reused_buffer_views": (
        "--pjrt_cache_reused_buffer_views",
        "on",
    ),
    "pjrt_cache_reused_buffer_views_trust_lifetime": (
        "--pjrt_cache_reused_buffer_views_trust_lifetime",
        "on",
    ),
    "compact_slice_feed": ("--compact_slice_feed", "on"),
    "compact_slice_feed_min_saved_mib": (
        "--compact_slice_feed_min_saved_mib",
        0.0,
    ),
    "compact_slice_feed_min_total_saved_mib": (
        "--compact_slice_feed_min_total_saved_mib",
        0.0,
    ),
    "warmup": ("--warmup", 3),
    "run_iters": ("--run_iters", 10),
}


def select_optimization_profile(mode, specs):
    mode = str(mode or "auto").strip().lower()
    if mode == "off":
        return "off"
    spec_names = [Path(spec).name for spec in specs or []]
    if mode in ("meta1", "meta2", "meta3"):
        expected = {
            "meta1": "meta_graph_1.spec",
            "meta2": "meta_graph_2.spec",
            "meta3": "meta_graph_3.spec",
        }[mode]
        if not spec_names or any(name != expected for name in spec_names):
            raise ValueError(
                f"--optimization_profile {mode} requires only {expected}"
            )
        return mode
    if mode != "auto":
        raise ValueError(f"Unsupported optimization profile: {mode}")
    if spec_names and all(name == "meta_graph_1.spec" for name in spec_names):
        return "meta1"
    if spec_names and all(name == "meta_graph_2.spec" for name in spec_names):
        return "meta2"
    if spec_names and all(name == "meta_graph_3.spec" for name in spec_names):
        return "meta3"
    return "off"


def apply_optimization_profile(args, specs, explicit_flags=None):
    selected = select_optimization_profile(
        getattr(args, "optimization_profile", "auto"),
        specs,
    )
    profile = {
        "meta1": META1_DEFAULT_PROFILE,
        "meta2": META2_DEFAULT_PROFILE,
        "meta3": META3_DEFAULT_PROFILE,
    }.get(selected, {})
    flags = set(explicit_flags or getattr(args, "explicit_cli_flags", []) or [])
    applied = {}
    for attr, (flag, value) in profile.items():
        if flag in flags:
            continue
        if hasattr(args, attr):
            setattr(args, attr, value)
            applied[attr] = value
    args.optimization_profile_selected = selected
    args.optimization_profile_applied = applied
    return selected, applied


def parse_cpu_list(text):
    cpus = set()
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            cpus.add(int(item))
            continue
        start_text, end_text = item.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Invalid CPU range: {item}")
        cpus.update(range(start, end + 1))
    return cpus


def _discover_drm_gpu_pci_bus_ids(sysfs_root=Path("/sys")):
    drm_root = Path(sysfs_root) / "class" / "drm"
    if not drm_root.is_dir():
        return []
    pci_ids = set()
    for card_path in drm_root.glob("card*"):
        if not re.fullmatch(r"card\d+", card_path.name):
            continue
        device_path = card_path / "device"
        try:
            resolved = device_path.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        pci_id = resolved.name
        if re.fullmatch(
            r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",
            pci_id,
        ):
            pci_ids.add(pci_id.lower())
    return sorted(pci_ids)


def _select_musa_gpu_pci_bus_ids(drm_pci_ids, pci_devices):
    drm_candidates = set(drm_pci_ids)
    driver_candidates = set()
    vendor_candidates = set()
    display_candidates = set()
    driver_hints = ("musa", "mtgpu", "mthreads")
    for device in pci_devices:
        pci_id = str(device.get("pci_id", "")).lower()
        uevent = str(device.get("uevent", "")).lower()
        driver_name = str(device.get("driver", "")).lower()
        if any(hint in uevent or hint in driver_name for hint in driver_hints):
            driver_candidates.add(pci_id)
        if str(device.get("vendor", "")).lower() == "0x1ed5":
            vendor_candidates.add(pci_id)
        if str(device.get("class_code", "")).lower().startswith("0x03"):
            display_candidates.add(pci_id)

    for candidates in (driver_candidates, vendor_candidates):
        if len(candidates) == 1:
            return sorted(candidates)
        overlap = candidates & drm_candidates
        if len(overlap) == 1:
            return sorted(overlap)
    if len(drm_candidates) == 1:
        return sorted(drm_candidates)
    if len(display_candidates) == 1:
        return sorted(display_candidates)
    if driver_candidates or vendor_candidates:
        return sorted(driver_candidates | vendor_candidates)
    return sorted(drm_candidates)


def _discover_musa_gpu_pci_bus_ids(sysfs_root=Path("/sys")):
    sysfs_root = Path(sysfs_root)
    drm_candidates = _discover_drm_gpu_pci_bus_ids(sysfs_root)
    devices_root = sysfs_root / "bus" / "pci" / "devices"
    if not devices_root.is_dir():
        return drm_candidates

    pci_pattern = re.compile(
        r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]"
    )
    devices = []
    for device_path in devices_root.iterdir():
        pci_id = device_path.name.lower()
        if not pci_pattern.fullmatch(pci_id):
            continue

        def read_text(name):
            try:
                return (device_path / name).read_text(
                    encoding="ascii", errors="ignore"
                ).strip().lower()
            except OSError:
                return ""

        try:
            driver_name = (device_path / "driver").resolve(strict=True).name.lower()
        except (FileNotFoundError, OSError):
            driver_name = ""
        devices.append(
            {
                "pci_id": pci_id,
                "driver": driver_name,
                "uevent": read_text("uevent"),
                "vendor": read_text("vendor"),
                "class_code": read_text("class"),
            }
        )
    return _select_musa_gpu_pci_bus_ids(drm_candidates, devices)


def _resolve_musa_gpu_pci_bus_id(
    explicit_pci, device, visible_devices, candidates
):
    explicit_pci = str(explicit_pci or "").strip().lower()
    if explicit_pci:
        return explicit_pci, "explicit"

    ordered_candidates = sorted(
        {str(candidate).strip().lower() for candidate in candidates if candidate}
    )
    if not ordered_candidates:
        return "", "gpu_pci_not_found"
    if len(ordered_candidates) == 1:
        return ordered_candidates[0], "single_candidate"

    device_match = re.search(r"(?:^|/)device:MUSA:(\d+)$", str(device), re.I)
    if device_match is None:
        return "", "device_ordinal_unavailable"
    logical_ordinal = int(device_match.group(1))

    visible_devices = str(visible_devices or "").strip()
    if visible_devices:
        visible_ordinals = [item.strip() for item in visible_devices.split(",")]
        if logical_ordinal >= len(visible_ordinals):
            return "", "visible_device_logical_ordinal_out_of_range"
        physical_text = visible_ordinals[logical_ordinal]
        if not re.fullmatch(r"\d+", physical_text):
            return "", "visible_device_not_numeric"
        physical_ordinal = int(physical_text)
        selection = "visible_device_ordinal"
    else:
        physical_ordinal = logical_ordinal
        selection = "device_ordinal"

    if physical_ordinal >= len(ordered_candidates):
        return "", "physical_device_ordinal_out_of_range"
    return ordered_candidates[physical_ordinal], selection


def apply_gpu_local_cpu_affinity(
    mode="auto", pci_bus_id="", device="/device:MUSA:0"
):
    result = {
        "enabled": False,
        "reason": "off",
        "pci_bus_id": "",
        "cpus": [],
        "selection": "",
    }
    mode = str(mode or "auto").strip().lower()
    if mode == "off":
        return result
    if os.name != "posix" or not hasattr(os, "sched_setaffinity"):
        result["reason"] = "unsupported_platform"
        return result
    explicit_pci = pci_bus_id or os.environ.get("MUSA_GPU_PCI_BUS_ID", "")
    selected_pci, selection = _resolve_musa_gpu_pci_bus_id(
        explicit_pci,
        device,
        os.environ.get("MUSA_VISIBLE_DEVICES", ""),
        _discover_musa_gpu_pci_bus_ids(),
    )
    result["selection"] = selection
    if not selected_pci:
        result["reason"] = selection
        return result
    cpu_list_path = Path("/sys/bus/pci/devices") / selected_pci / "local_cpulist"
    try:
        cpus = parse_cpu_list(cpu_list_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError) as exc:
        result["reason"] = f"local_cpulist_error:{type(exc).__name__}"
        result["pci_bus_id"] = selected_pci
        return result
    if not cpus:
        result["reason"] = "empty_local_cpulist"
        result["pci_bus_id"] = selected_pci
        return result
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as exc:
        result["reason"] = f"sched_setaffinity_error:{exc.errno}"
        result["pci_bus_id"] = selected_pci
        result["cpus"] = sorted(cpus)
        return result
    result.update(
        {
            "enabled": True,
            "reason": "ok",
            "pci_bus_id": selected_pci,
            "cpus": sorted(cpus),
        }
    )
    return result






def _early_spec_paths(argv):
    spec = _get_cli_arg(argv, "--spec")
    if spec:
        return [Path(spec)]
    spec_dir = _get_cli_arg(argv, "--spec_dir")
    if not spec_dir:
        return []
    root = Path(spec_dir)
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.spec"))


def _argv_with_optimization_profile(argv):
    effective = list(argv)
    mode = _get_cli_arg(argv, "--optimization_profile", "auto")
    selected = select_optimization_profile(mode, _early_spec_paths(argv))
    profile = {
        "meta1": META1_DEFAULT_PROFILE,
        "meta2": META2_DEFAULT_PROFILE,
        "meta3": META3_DEFAULT_PROFILE,
    }.get(selected, {})
    for _, (flag, value) in profile.items():
        if not _has_cli_flag(effective, flag):
            _append_cli_option(effective, flag, value)
    return effective, selected


def configure_runtime_env_from_argv(argv):
    device = _get_cli_arg(argv, "--device", "/device:MUSA:0")
    kind = device_kind(device)
    enable_xla = _has_cli_flag(argv, "--xla")
    dump_dir = _get_cli_arg(argv, "--xla_dump_dir")
    enable_dump = _has_cli_flag(argv, "--xla_dump") or dump_dir is not None

    if enable_xla:
        tf_xla_flags = os.environ.get("TF_XLA_FLAGS", "")
        tf_xla_flags = _set_flag_with_prefix(
            tf_xla_flags, "--tf_xla_auto_jit=", "--tf_xla_auto_jit=2"
        )
        if kind in ("MUSA", "CUDA"):
            tf_xla_flags = _append_unique_flag(
                tf_xla_flags, "--tf_xla_use_device_api=true"
            )
        os.environ["TF_XLA_FLAGS"] = tf_xla_flags
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    if enable_xla and kind == "MUSA":
        early_batch_size = _early_batch_size_arg(argv)
        plugin_path = _get_cli_arg(argv, "--musa_plugin", default_musa_plugin_path())
        tf_adapter_path = _get_cli_arg(
            argv, "--musa_tf_adapter", default_musa_tf_adapter_path()
        )
        os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = str(tf_adapter_path)
        os.environ["MUSA_PJRT_PLUGIN_PATH"] = str(plugin_path)
        os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{plugin_path}"
        os.environ["MUSA_TF_NPD_ADAPTER_PATH"] = str(tf_adapter_path)
        os.environ["MUSA_NPD_IS_PLUGGABLE_DEVICE"] = "1"
        os.environ["MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE"] = "1"

        bool_options = [
            ("--pjrt_wait_transfer_done", "MUSA_PJRT_WAIT_TRANSFER_DONE"),
            ("--pjrt_wait_execute_done", "MUSA_PJRT_WAIT_EXECUTE_DONE"),
            ("--pjrt_reuse_host_buffers", "MUSA_PJRT_REUSE_HOST_BUFFERS"),
            (
                "--pjrt_reuse_host_buffers_arena",
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA",
            ),
            (
                "--pjrt_reuse_host_buffers_arena_parallel_pack",
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PARALLEL_PACK",
            ),
            (
                "--pjrt_reuse_host_buffers_arena_dirty_ranges",
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_DIRTY_RANGES",
            ),
            (
                "--pjrt_reuse_host_buffers_arena_pool_order_layout",
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_POOL_ORDER_LAYOUT",
            ),
            (
                "--pjrt_reuse_host_buffers_trust_contents",
                "MUSA_PJRT_REUSE_HOST_BUFFERS_TRUST_CONTENTS",
            ),
            (
                "--pjrt_cache_reused_buffer_views",
                "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS",
            ),
            (
                "--pjrt_cache_reused_buffer_views_trust_lifetime",
                "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS_TRUST_LIFETIME",
            ),
            ("--musa_f32_fast_tf32", "MUSA_F32_FAST_TF32"),
            (
                "--musa_xla_same_shape_dot_batcher",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCHER",
            ),
            (
                "--musa_xla_classic_thunk_graph",
                "MUSA_XLA_CLASSIC_THUNK_GRAPH",
            ),
            (
                "--musa_xla_hot_tuple_softmax_kernel",
                "MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL",
            ),
            (
                "--musa_xla_warp_row_reduction_kernel",
                "MUSA_XLA_WARP_ROW_REDUCTION_KERNEL",
            ),
            (
                "--musa_xla_tuple_warp_row_reduction_kernel",
                "MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL",
            ),
            (
                "--musa_xla_mixed_tuple_warp_row_reduction_kernel",
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL",
            ),
            (
                "--musa_xla_fusion_merger_materialize_reduction_producer",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER",
            ),
        ]
        for flag, env_name in bool_options:
            _set_tristate_bool_env_from_cli_or_preserve(
                argv, flag, env_name, default="off"
            )

        for flag, env_name in (
            ("--pjrt_max_inflight_transfers", "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS"),
            ("--pjrt_max_inflight_executes", "MUSA_PJRT_MAX_INFLIGHT_EXECUTES"),
        ):
            os.environ[env_name] = str(_get_cli_arg(argv, flag, "0"))

        value_options = [
            (
                "--pjrt_reuse_host_buffers_arena_pack_threads",
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_THREADS",
            ),
            (
                "--pjrt_reuse_host_buffers_arena_pack_min_bytes",
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_MIN_BYTES",
            ),
            (
                "--musa_xla_same_shape_dot_batch_min_group_size",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE",
            ),
            (
                "--musa_xla_same_shape_dot_batch_max_group_size",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE",
            ),
            (
                "--musa_xla_same_shape_dot_batch_max_groups",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS",
            ),
            (
                "--musa_xla_same_shape_dot_batch_min_candidate_dots",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS",
            ),
            (
                "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH",
            ),
            (
                "--musa_xla_same_shape_dot_batch_max_output_cols",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS",
            ),
            (
                "--musa_xla_classic_thunk_graph_max_cache_entries",
                "MUSA_XLA_CLASSIC_THUNK_GRAPH_MAX_CACHE_ENTRIES",
            ),
            (
                "--musa_xla_warp_row_reduction_reducers",
                "MUSA_XLA_WARP_ROW_REDUCTION_REDUCERS",
            ),
            (
                "--musa_xla_warp_row_reduction_min_data_elements",
                "MUSA_XLA_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS",
            ),
            (
                "--musa_xla_warp_row_reduction_threads_per_block",
                "MUSA_XLA_WARP_ROW_REDUCTION_THREADS_PER_BLOCK",
            ),
            (
                "--musa_xla_mixed_tuple_warp_row_reduction_min_data_elements",
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS",
            ),
            (
                "--musa_xla_fusion_merger_materialize_min_elements",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS",
            ),
            (
                "--musa_xla_fusion_merger_materialize_min_operands",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS",
            ),
        ]
        for flag, env_name in value_options:
            _set_value_env_from_cli_or_preserve(argv, flag, env_name)

        runtime_mode = _get_cli_arg(argv, "--musa_xla_gpu_runtime", "auto")
        if runtime_mode != "auto":
            _set_xla_gpu_runtime(runtime_mode)
        global_jit_level = _get_cli_arg(argv, "--xla_global_jit_level", "off")
        if global_jit_level not in ("", "off", "false", "0", "auto"):
            os.environ["MUSA_XLA_GLOBAL_JIT_LEVEL"] = global_jit_level
        else:
            os.environ.pop("MUSA_XLA_GLOBAL_JIT_LEVEL", None)
        os.environ.setdefault("MUSA_PINNED_H2D_ON_COMPUTE_STREAM", "1")
        if early_batch_size >= _early_large_batch_threshold():
            os.environ.setdefault(
                "MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT", "1"
            )

        for removed_env in (
            "MUSA_PJRT_REUSE_HOST_BUFFERS_DIAGNOSTICS",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ASYNC",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_FAST_PACK",
            "MUSA_PJRT_SOURCE_FEED_ARENA_BASE",
            "MUSA_PJRT_SOURCE_FEED_ARENA_BYTES",
            "MUSA_F32_FAST_TF32_SHAPES",
            "MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG",
            "MUSA_XLA_REDUCTION_CHAIN_DIAG",
            "MUSA_XLA_REDUCTION_CHAIN_REWRITE",
            "MUSA_XLA_REDUCTION_CHAIN_KERNEL",
            "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION",
            "MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS",
            "MUSA_XLA_THUNK_DIAGNOSTICS",
            "MUSA_XLA_THUNK_TIMING",
        ):
            os.environ.pop(removed_env, None)

    if enable_dump:
        xla_dump_dir = os.path.abspath(
            os.path.expanduser(dump_dir or str(DEFAULT_XLA_DUMP_DIR))
        )
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags = _set_flag_with_prefix(
            xla_flags, "--xla_dump_to=", f"--xla_dump_to={xla_dump_dir}"
        )
        xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_text")
        if _get_cli_arg(argv, "--xla_dump_long_text", "false") in (
            "1", "true", "TRUE", "yes", "YES", "on", "ON"
        ):
            xla_flags = _append_unique_flag(
                xla_flags, "--xla_dump_hlo_as_long_text"
            )
        pass_re = _get_cli_arg(argv, "--xla_dump_hlo_pass_re", "^$")
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_dump_hlo_pass_re=",
            f"--xla_dump_hlo_pass_re={pass_re}",
        )
        max_modules = _get_cli_arg(argv, "--xla_dump_max_hlo_modules", "-1")
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_dump_max_hlo_modules=",
            f"--xla_dump_max_hlo_modules={max_modules}",
        )
        module_re = _get_cli_arg(argv, "--xla_dump_hlo_module_re", "")
        if module_re:
            xla_flags = _set_flag_with_prefix(
                xla_flags,
                "--xla_dump_hlo_module_re=",
                f"--xla_dump_hlo_module_re={module_re}",
            )
        os.environ["XLA_FLAGS"] = xla_flags
        os.environ["GRAPH_RUNNER_XLA_DUMP_DIR"] = xla_dump_dir



EARLY_EFFECTIVE_ARGV, EARLY_OPTIMIZATION_PROFILE = _argv_with_optimization_profile(
    sys.argv[1:]
)
_early_cpu_affinity_mode = _get_cli_arg(
    EARLY_EFFECTIVE_ARGV, "--cpu_affinity", "auto"
)
if _early_cpu_affinity_mode == "auto":
    _early_cpu_affinity_mode = (
        "on" if EARLY_OPTIMIZATION_PROFILE == "meta1" else "off"
    )
EARLY_CPU_AFFINITY = apply_gpu_local_cpu_affinity(
    _early_cpu_affinity_mode,
    _get_cli_arg(EARLY_EFFECTIVE_ARGV, "--gpu_pci_bus_id", ""),
    _get_cli_arg(EARLY_EFFECTIVE_ARGV, "--device", "/device:MUSA:0"),
)
configure_runtime_env_from_argv(EARLY_EFFECTIVE_ARGV)













_HLO_INSTRUCTION_RE = re.compile(r"^\s*(?:ROOT\s+)?[A-Za-z0-9_.%-]+ = ")
_HLO_INSTRUCTION_NAME_RE = re.compile(
    r"^\s*(?:ROOT\s+)?%?(?P<name>[A-Za-z0-9_.%-]+)\s*="
)
_HLO_OPCODE_RE = re.compile(r"\s([A-Za-z][A-Za-z0-9_-]*)\(")
_HLO_FUSION_CALL_RE = re.compile(
    r"^\s*(?:ROOT\s+)?%?(?P<name>[A-Za-z0-9_.-]+)\s*=\s*"
    r"(?P<shape>.*?)\s+fusion\(.*?\bcalls=%?(?P<callee>[A-Za-z0-9_.-]+)"
)
_HLO_COMPUTATION_START_RE = re.compile(
    r"^\s*%?(?P<name>[A-Za-z_][A-Za-z0-9_.%-]*)(?:\s+.*?)?\{\s*$"
)
_HLO_ARRAY_SHAPE_RE = re.compile(
    r"\b(?P<dtype>pred|[a-z][0-9]+)\[(?P<dims>[0-9,]*)\]"
)
_HLO_REDUCE_DIMENSIONS_RE = re.compile(r"\bdimensions=\{(?P<dims>[^}]*)\}")
_HLO_TO_APPLY_RE = re.compile(
    r"\bto_apply=%?(?P<callee>[A-Za-z_][A-Za-z0-9_.%-]*)"
)
































_THUNK_DIAG_RE = re.compile(
    r"\[MUSA_XLA_THUNK_DIAGNOSTICS\]\s+"
    r"module=(?P<module>\S+)\s+"
    r"module_id=(?P<module_id>\d+)\s+"
    r"total_thunks=(?P<total_thunks>\d+)\s+"
    r"gemm_thunks=(?P<gemm_thunks>\d+)\s+"
    r"kernel_thunks=(?P<kernel_thunks>\d+)\s+"
    r"counts=\{(?P<counts>[^}]*)\}"
)


























import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.config.run_functions_eagerly(True)


def env_flag_enabled(name):
    value = os.environ.get(name, "")
    return value in ("1", "true", "TRUE", "yes", "YES", "on", "ON")


def env_flag_disabled(name):
    value = os.environ.get(name, "")
    return value in ("0", "false", "FALSE", "no", "NO", "off", "OFF")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def create_session_config(args, musa_loaded: bool):
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = bool(args.allow_soft_placement)
    config.log_device_placement = bool(args.log_device_placement)

    kind = device_kind(args.device)
    if kind == "CUDA":
        config.gpu_options.allow_growth = True
        if args.xla:
            config.graph_options.optimizer_options.global_jit_level = (
                tf.compat.v1.OptimizerOptions.ON_1
            )
    elif kind == "MUSA":
        rewrite_options = config.graph_options.rewrite_options
        if musa_loaded and args.musa_optimizer:
            rewrite_options.custom_optimizers.add().name = "musa_graph_optimizer"
        if args.xla:
            rewrite_options.min_graph_nodes = -1
            jit_level = os.environ.get("MUSA_XLA_GLOBAL_JIT_LEVEL", "").strip().lower()
            if jit_level in ("on", "on_1", "1", "true"):
                config.graph_options.optimizer_options.global_jit_level = (
                    tf.compat.v1.OptimizerOptions.ON_1
                )
            elif jit_level in ("on_2", "2"):
                config.graph_options.optimizer_options.global_jit_level = (
                    getattr(
                        tf.compat.v1.OptimizerOptions,
                        "ON_2",
                        tf.compat.v1.OptimizerOptions.ON_1,
                    )
                )
    elif args.xla:
        config.graph_options.optimizer_options.global_jit_level = (
            tf.compat.v1.OptimizerOptions.ON_1
        )
    return config


@contextmanager
def maybe_xla_jit_scope(args):
    if not (args.xla and args.xla_jit_scope == "on"):
        yield
        return

    experimental = getattr(getattr(tf, "xla", None), "experimental", None)
    jit_scope = getattr(experimental, "jit_scope", None)
    if jit_scope is None:
        compat_xla = getattr(tf.compat.v1, "xla", None)
        compat_experimental = getattr(compat_xla, "experimental", None)
        jit_scope = getattr(compat_experimental, "jit_scope", None)
    if jit_scope is None:
        raise RuntimeError("TensorFlow xla.experimental.jit_scope is unavailable")

    with jit_scope(compile_ops=True):
        yield


def collect_graph_dump_files(dump_dir: Union[str, Path, None]):
    if not dump_dir:
        return {}

    dump_root = Path(dump_dir)
    if not dump_root.exists():
        return {}

    files = {}
    for alias, stage_suffix in GRAPH_DUMP_STAGE_SUFFIXES.items():
        matches = []
        for ext in (".pbtxt", ".pb"):
            matches.extend(dump_root.glob(f"*_{stage_suffix}{ext}"))
        matches = sorted(matches, key=lambda path: (path.stat().st_mtime, str(path)))
        if matches:
            latest = matches[-1]
            files[alias] = {
                "stage": stage_suffix,
                "format": latest.suffix.lstrip("."),
                "path": str(latest.resolve()),
            }
    return files


@contextmanager
def configured_graph_dump_dir(default_dump_dir: Union[Path, None]):
    if not env_flag_enabled(GRAPH_DUMP_ENV):
        yield None
        return

    old_dump_dir = os.environ.get(GRAPH_DUMP_DIR_ENV)
    if old_dump_dir:
        dump_dir = Path(old_dump_dir).resolve()
    elif default_dump_dir is not None:
        dump_dir = Path(default_dump_dir).resolve()
        os.environ[GRAPH_DUMP_DIR_ENV] = str(dump_dir)
    else:
        dump_dir = Path.cwd().resolve()

    dump_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield dump_dir
    finally:
        if old_dump_dir is None:
            os.environ.pop(GRAPH_DUMP_DIR_ENV, None)




def load_meta(spec_path: Path):
    meta = tf.compat.v1.MetaGraphDef()
    meta.ParseFromString(spec_path.read_bytes())
    if not meta.graph_def.node:
        raise ValueError(f"Invalid spec, graph_def is empty: {spec_path}")
    return meta


def read_node_list_collection(meta, key):
    coll = meta.collection_def.get(key)
    if not coll:
        raise ValueError(f"Missing collection_def['{key}'] in spec.")
    if coll.WhichOneof("kind") != "node_list":
        raise ValueError(f"collection_def['{key}'] is not node_list.")
    return list(coll.node_list.value)


def _shape_from_node_attr(node):
    if "_output_shapes" in node.attr and node.attr["_output_shapes"].list.shape:
        shp = node.attr["_output_shapes"].list.shape[0]
        return [d.size if d.size != -1 else None for d in shp.dim]
    if "shape" in node.attr:
        shp = node.attr["shape"].shape
        return [d.size if d.size != -1 else None for d in shp.dim]
    return None


def build_spec_tensor_shape_map(meta):
    out = {}
    for node in meta.graph_def.node:
        shape = _shape_from_node_attr(node)
        if shape is not None:
            out[f"{node.name}:0"] = shape
    return out


def merge_shape(spec_shape, pb_shape):
    if spec_shape is None:
        return list(pb_shape) if pb_shape is not None else []
    if pb_shape is None:
        return list(spec_shape)
    merged = []
    for index in range(max(len(spec_shape), len(pb_shape))):
        spec_dim = spec_shape[index] if index < len(spec_shape) else None
        pb_dim = pb_shape[index] if index < len(pb_shape) else None
        merged.append(spec_dim if spec_dim is not None else pb_dim)
    return merged


def resolve_shape(shape, bs, unknown_dim):
    if shape is None:
        return []
    out = []
    for index, dim in enumerate(shape):
        if dim is None:
            if index != 0:
                out.append(unknown_dim)
            else:
                out.append(bs)
        else:
            out.append(dim)
    return out


def parse_bs_values(bs_arg):
    if isinstance(bs_arg, int):
        return [bs_arg]
    values = []
    seen = set()
    for part in str(bs_arg).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"batch size must be > 0, got {value}")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("bs is empty")
    return values


def parse_index_selection(selection, total):
    if selection in (None, "", "all"):
        return list(range(total))
    indices = []
    seen = set()
    for part in str(selection).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"invalid output index range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for value in values:
            if value < 0 or value >= total:
                raise ValueError(
                    f"output index {value} out of range [0, {total})"
                )
            if value not in seen:
                seen.add(value)
                indices.append(value)
    if not indices:
        raise ValueError("output index selection is empty")
    return indices


def random_array(shape, np_dtype, rng):
    if np_dtype in (np.str_, np.object_, np.bytes_, object):
        total = int(np.prod(shape)) if shape else 1
        vals = np.array(
            [f"s{rng.integers(0, 1_000_000)}".encode("utf-8") for _ in range(total)],
            dtype=object,
        )
        return vals.reshape(shape) if shape else vals.reshape(()).item()
    if np.issubdtype(np_dtype, np.floating):
        return rng.uniform(0.1, 1.0, size=shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.complexfloating):
        real = rng.standard_normal(size=shape)
        imag = rng.standard_normal(size=shape)
        return (real + 1j * imag).astype(np_dtype)
    if np.issubdtype(np_dtype, np.integer):
        return rng.integers(0, 10, size=shape, dtype=np_dtype)
    if np.issubdtype(np_dtype, np.bool_):
        return rng.choice([False, True], size=shape)
    raise TypeError(f"Unsupported dtype for random input: {np_dtype}")


class PinnedHostArray:
    _lib = None

    @classmethod
    def lib(cls):
        if cls._lib is None:
            cls._lib = ctypes.CDLL("/usr/local/musa/lib/libmusart.so")
            cls._lib.musaHostAlloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            cls._lib.musaHostAlloc.restype = ctypes.c_int
            cls._lib.musaFreeHost.argtypes = [ctypes.c_void_p]
            cls._lib.musaFreeHost.restype = ctypes.c_int
        return cls._lib

    def __init__(self, shape, np_dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(np_dtype)
        count = int(np.prod(self.shape)) if self.shape else 1
        self.nbytes = max(1, count * self.dtype.itemsize)
        self.ptr = ctypes.c_void_p()
        err = self.lib().musaHostAlloc(ctypes.byref(self.ptr), self.nbytes, 0)
        if err != 0 or not self.ptr.value:
            raise RuntimeError(f"musaHostAlloc failed: err={err}, bytes={self.nbytes}")
        buf_type = ctypes.c_char * self.nbytes
        self.buffer = buf_type.from_address(self.ptr.value)
        self.array = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self.buffer)

    def __del__(self):
        ptr = getattr(self, "ptr", None)
        if ptr is not None and ptr.value:
            try:
                self.lib().musaFreeHost(ptr)
            except Exception:
                pass
            self.ptr = ctypes.c_void_p()












def pinned_random_array(shape, np_dtype, rng, holders):
    if np_dtype in (np.str_, np.object_, np.bytes_, object):
        return random_array(shape, np_dtype, rng)
    holder = PinnedHostArray(shape, np_dtype)
    holders.append(holder)
    arr = holder.array
    if np.issubdtype(arr.dtype, np.floating):
        arr[...] = rng.uniform(0.1, 1.0, size=shape).astype(arr.dtype, copy=False)
    elif np.issubdtype(arr.dtype, np.complexfloating):
        real = rng.standard_normal(size=shape)
        imag = rng.standard_normal(size=shape)
        arr[...] = (real + 1j * imag).astype(arr.dtype, copy=False)
    elif np.issubdtype(arr.dtype, np.integer):
        arr[...] = rng.integers(0, 10, size=shape, dtype=arr.dtype)
    elif np.issubdtype(arr.dtype, np.bool_):
        arr[...] = rng.choice([False, True], size=shape)
    else:
        raise TypeError(f"Unsupported dtype for pinned random input: {np_dtype}")
    return arr


def _strip_output_port(name):
    return name[1:] if name.startswith("^") else name.split(":")[0]


def should_try_pinned_feed(args):
    if args.pinned_feed == "off":
        return False
    if args.pinned_feed == "on":
        return True
    if env_flag_enabled("MUSA_PINNED_FEED"):
        return True
    if env_flag_disabled("MUSA_PINNED_FEED"):
        return False
    return bool(args.xla and device_kind(args.device) == "MUSA")


def should_allocate_individual_pinned_feed(args):
    return should_try_pinned_feed(args)



def should_use_callable(args):
    if args.use_callable == "on":
        return True
    if args.use_callable == "off":
        return False
    return bool(args.xla and device_kind(args.device) == "MUSA")






def should_pack_concat_feed(args):
    if args.pack_concat_feed == "on":
        return True
    if args.pack_concat_feed == "off":
        return False
    return bool(args.xla and device_kind(args.device) == "MUSA")


def should_pack_slice_feed(args):
    if args.pack_slice_feed == "on":
        return True
    if args.pack_slice_feed == "off":
        return False
    return bool(args.xla and device_kind(args.device) == "MUSA")




def should_bypass_identity(args, graph_def=None):
    if args.bypass_identity == "on":
        return True
    if args.bypass_identity == "off":
        return False
    if not (args.xla and device_kind(args.device) == "MUSA"):
        return False
    if graph_def is None:
        return False
    identity_count = sum(1 for node in graph_def.node if node.op == "Identity")
    return identity_count >= args.bypass_identity_min_nodes


def should_optimize_output_fetches(args):
    if args.optimize_output_fetches == "on":
        return True
    if args.optimize_output_fetches == "off":
        return False
    return bool(args.xla and device_kind(args.device) == "MUSA")


def should_pack_output_fetches(args):
    if args.pack_output_fetches == "on":
        return True
    if args.pack_output_fetches == "off":
        return False
    return bool(args.xla and device_kind(args.device) == "MUSA")










def should_rewrite_pow_square(args):
    if args.rewrite_pow_square == "on":
        return True
    if args.rewrite_pow_square == "off":
        return False
    return bool(args.xla and device_kind(args.device) == "MUSA")


def parse_csv_filter(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("all", "any", "*"):
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def trimmed_mean(lat_ms, trim_ratio=0.1):
    if not lat_ms:
        return 0.0
    values = sorted(lat_ms)
    cut = int(len(values) * trim_ratio)
    trimmed = values[cut : len(values) - cut] if len(values) > 2 * cut else values
    return float(np.mean(trimmed))


def summarize_feed_dict(feed_dict, bs=None):
    items = []
    total_bytes = 0
    by_dtype = Counter()
    by_rank = Counter()
    by_size_bucket = Counter()
    batch_dim_bytes = 0
    non_batch_dim_bytes = 0
    scalar_bytes = 0
    large_input_count = 0
    large_input_bytes = 0

    def bucket_name(nbytes):
        mib = nbytes / (1024.0 * 1024.0)
        if mib >= 32:
            return ">=32MiB"
        if mib >= 16:
            return "16-32MiB"
        if mib >= 4:
            return "4-16MiB"
        if mib >= 1:
            return "1-4MiB"
        if mib >= 0.25:
            return "256KiB-1MiB"
        if mib >= 0.0625:
            return "64-256KiB"
        return "<64KiB"

    for tensor, value in feed_dict.items():
        nbytes = int(getattr(value, "nbytes", 0) or 0)
        shape = list(getattr(value, "shape", []))
        total_bytes += nbytes
        by_dtype[str(getattr(value, "dtype", tensor.dtype.name))] += nbytes
        by_rank[len(shape)] += nbytes
        by_size_bucket[bucket_name(nbytes)] += nbytes
        if not shape:
            scalar_bytes += nbytes
        elif bs is not None and shape[0] == bs:
            batch_dim_bytes += nbytes
        else:
            non_batch_dim_bytes += nbytes
        if nbytes >= 1024 * 1024:
            large_input_count += 1
            large_input_bytes += nbytes
        items.append(
            {
                "name": tensor.name,
                "optype": tensor.op.type,
                "dtype": str(getattr(value, "dtype", tensor.dtype.name)),
                "shape": shape,
                "nbytes": nbytes,
            }
        )
    items.sort(key=lambda item: item["nbytes"], reverse=True)
    return {
        "num_inputs": len(items),
        "total_nbytes": total_bytes,
        "total_mib": total_bytes / (1024.0 * 1024.0),
        "batch_dim_mib": batch_dim_bytes / (1024.0 * 1024.0),
        "non_batch_dim_mib": non_batch_dim_bytes / (1024.0 * 1024.0),
        "scalar_mib": scalar_bytes / (1024.0 * 1024.0),
        "large_input_count": large_input_count,
        "large_input_mib": large_input_bytes / (1024.0 * 1024.0),
        "bytes_by_dtype": dict(by_dtype),
        "bytes_by_rank": {str(rank): bytes_ for rank, bytes_ in by_rank.items()},
        "bytes_by_size_bucket": dict(by_size_bucket),
        "mib_by_size_bucket": {
            name: bytes_ / (1024.0 * 1024.0)
            for name, bytes_ in by_size_bucket.items()
        },
        "largest_inputs": items[:20],
    }
















def _ensure_tensor_name(name):
    if name.startswith("^"):
        name = name[1:]
    if ":" in name:
        return name
    return f"{name}:0"


def bypass_identity_nodes(graph_def, output_spec):
    identity_input = {}
    skipped_with_control = 0
    skipped_control_consumer = 0
    control_consumers = defaultdict(list)
    for node in graph_def.node:
        for input_name in node.input:
            if input_name.startswith("^"):
                control_consumers[_strip_tensor_name(input_name)].append(node.name)

    for node in graph_def.node:
        if node.op != "Identity" or not node.input:
            continue
        data_inputs = [name for name in node.input if not name.startswith("^")]
        control_inputs = [name for name in node.input if name.startswith("^")]
        if len(data_inputs) != 1:
            continue
        if control_inputs:
            skipped_with_control += 1
            continue
        if control_consumers.get(node.name):
            skipped_control_consumer += 1
            continue
        identity_input[node.name] = data_inputs[0]

    def resolve_tensor(name):
        if name.startswith("^"):
            return name
        root = _strip_tensor_name(name)
        if root not in identity_input:
            return name
        suffix = name[len(root):]
        if suffix not in ("", ":0"):
            return name
        seen = set()
        current = identity_input[root]
        while True:
            current_root = _strip_tensor_name(current)
            if current_root in seen or current_root not in identity_input:
                break
            seen.add(current_root)
            current = identity_input[current_root]
        return current

    rewired_edges = 0
    for node in graph_def.node:
        for index, input_name in enumerate(node.input):
            new_input = resolve_tensor(input_name)
            if new_input != input_name:
                node.input[index] = new_input
                rewired_edges += 1

    output_remaps = []
    new_output_spec = []
    for name in output_spec:
        new_name = _ensure_tensor_name(resolve_tensor(name))
        new_output_spec.append(new_name)
        if new_name != name:
            output_remaps.append({"from": name, "to": new_name})

    return new_output_spec, {
        "enabled": True,
        "eligible_identity_nodes": len(identity_input),
        "skipped_with_control": skipped_with_control,
        "skipped_control_consumer": skipped_control_consumer,
        "rewired_edges": rewired_edges,
        "pruned_identity_nodes": 0,
        "output_remaps": output_remaps,
    }


def _const_float_array(node):
    if node is None or node.op != "Const" or "value" not in node.attr:
        return None
    try:
        arr = tf.make_ndarray(node.attr["value"].tensor)
    except Exception:
        return None
    if not np.issubdtype(arr.dtype, np.floating):
        return None
    return np.asarray(arr)


def rewrite_pow_square_nodes(graph_def):
    node_map = {node.name: node for node in graph_def.node}
    rewritten = []
    skipped_non_const = 0
    skipped_non_square = 0
    for node in graph_def.node:
        if node.op != "Pow" or len(node.input) < 2:
            continue
        exponent_node = node_map.get(_strip_tensor_name(node.input[1]))
        exponent = _const_float_array(exponent_node)
        if exponent is None:
            skipped_non_const += 1
            continue
        if exponent.size == 0 or not np.allclose(exponent, 2.0):
            skipped_non_square += 1
            continue
        base_input = node.input[0]
        control_inputs = [
            input_name
            for input_name in node.input[2:]
            if input_name.startswith("^")
        ]
        node.op = "Mul"
        del node.input[:]
        node.input.extend([base_input, base_input] + control_inputs)
        rewritten.append(node.name)
    return {
        "enabled": True,
        "rewritten": len(rewritten),
        "sample": rewritten[:20],
        "skipped_non_const": skipped_non_const,
        "skipped_non_square": skipped_non_square,
    }


def _const_output_dtype(node):
    if "out_type" in node.attr:
        try:
            return tf.as_dtype(node.attr["out_type"].type)
        except Exception:
            return tf.int32
    return tf.int32


def _replace_node_with_const(node, value, dtype):
    node.op = "Const"
    del node.input[:]
    node.attr.clear()
    tensor = tf.make_tensor_proto(value, dtype=dtype)
    node.attr["dtype"].type = dtype.as_datatype_enum
    node.attr["value"].tensor.CopyFrom(tensor)




_RESHAPE_SHAPE_VALUE_OPS = {
    "Add",
    "AddV2",
    "Cast",
    "ConcatV2",
    "Const",
    "ExpandDims",
    "Fill",
    "FloorDiv",
    "FloorMod",
    "GatherV2",
    "Identity",
    "Maximum",
    "Minimum",
    "Mul",
    "Pack",
    "Prod",
    "Range",
    "RealDiv",
    "Reshape",
    "Select",
    "Shape",
    "Size",
    "Slice",
    "Squeeze",
    "StridedSlice",
    "Sub",
    "Unpack",
}


def _static_shape_nodes_feeding_reshape(graph_def, node_map):
    protected = set()
    visited = set()
    stack = []
    for node in graph_def.node:
        if node.op == "Reshape" and len(node.input) >= 2:
            stack.append(_tensor_name_parts(node.input[1])[0])
    while stack:
        node_name = stack.pop()
        if node_name in visited:
            continue
        visited.add(node_name)
        node = node_map.get(node_name)
        if node is None:
            continue
        if node.op in ("Shape", "Rank", "Size"):
            protected.add(node.name)
            continue
        if node.op not in _RESHAPE_SHAPE_VALUE_OPS:
            continue
        for input_name in node.input:
            if input_name.startswith("^"):
                continue
            stack.append(_tensor_name_parts(input_name)[0])
    return protected




def rewrite_concat_static_precompute_nodes(graph_def, concat_pack_state, bs, args):
    packed_inputs = (concat_pack_state or {}).get("packed_inputs") or []
    if not packed_inputs:
        return {
            "enabled": True,
            "reason": "no_concat_pack",
            "replaced_shape": 0,
            "replaced_rank": 0,
            "replaced_size": 0,
            "total_replaced": 0,
            "protected_reshape_shape_nodes": 0,
            "skipped": {},
            "sample_shape": [],
            "sample_rank": [],
            "sample_size": [],
        }

    shape_by_concat = {}
    axis_by_concat = {}
    for item in packed_inputs:
        concat_node = item.get("concat_node")
        shape = item.get("shape")
        if not concat_node or not shape:
            continue
        shape = list(shape)
        axis = item.get("axis")
        if concat_node not in shape_by_concat:
            shape_by_concat[concat_node] = shape
            axis_by_concat[concat_node] = axis
            continue
        previous = shape_by_concat[concat_node]
        if axis is None or len(previous) != len(shape):
            continue
        axis = int(axis)
        if axis < 0:
            axis += len(shape)
        if axis < 0 or axis >= len(shape):
            continue
        merged = list(previous)
        compatible = True
        for index, dim in enumerate(shape):
            if index == axis:
                merged[index] = int(merged[index]) + int(dim)
            elif int(merged[index]) != int(dim):
                compatible = False
                break
        if compatible:
            shape_by_concat[concat_node] = merged
            axis_by_concat[concat_node] = axis

    node_map = {node.name: node for node in graph_def.node}
    shape_by_tensor = {
        (name, 0): list(shape) for name, shape in shape_by_concat.items()
    }
    splitv_shapes = 0
    passthrough_shapes = 0
    changed = True
    while changed:
        changed = False
        for node in graph_def.node:
            if node.op == "SplitV" and len(node.input) >= 3:
                value_base, value_port = _tensor_name_parts(node.input[0])
                input_shape = shape_by_tensor.get((value_base, value_port))
                if input_shape is None:
                    continue
                size_splits = _const_int_list(
                    node_map.get(_strip_tensor_name(node.input[1]))
                )
                axis_values = _const_int_list(
                    node_map.get(_strip_tensor_name(node.input[2]))
                )
                if size_splits is None or not axis_values:
                    continue
                axis = int(axis_values[0])
                rank = len(input_shape)
                if axis < 0:
                    axis += rank
                if axis < 0 or axis >= rank:
                    continue
                if len(size_splits) == 0:
                    continue
                resolved_splits = [int(value) for value in size_splits]
                unknown_positions = [
                    index for index, value in enumerate(resolved_splits) if value < 0
                ]
                if len(unknown_positions) > 1:
                    continue
                axis_dim = input_shape[axis]
                if unknown_positions:
                    if axis_dim is None:
                        continue
                    known_total = sum(
                        value for value in resolved_splits if value >= 0
                    )
                    resolved_splits[unknown_positions[0]] = int(axis_dim) - known_total
                if any(value < 0 for value in resolved_splits):
                    continue
                for output_index, split_size in enumerate(resolved_splits):
                    key = (node.name, output_index)
                    if key in shape_by_tensor:
                        continue
                    output_shape = list(input_shape)
                    output_shape[axis] = int(split_size)
                    shape_by_tensor[key] = output_shape
                    splitv_shapes += 1
                    changed = True
                continue

            if node.op in ("Identity", "Cast") and node.input:
                input_base, input_port = _tensor_name_parts(node.input[0])
                input_shape = shape_by_tensor.get((input_base, input_port))
                if input_shape is None:
                    continue
                key = (node.name, 0)
                if key not in shape_by_tensor:
                    shape_by_tensor[key] = list(input_shape)
                    passthrough_shapes += 1
                    changed = True

    protected_shape_nodes = _static_shape_nodes_feeding_reshape(graph_def, node_map)
    replaced_shape = []
    replaced_rank = []
    replaced_size = []
    skipped = Counter()
    shape_input_ops = Counter()
    unknown_dim = getattr(args, "unknown_dim", 1)
    for node in graph_def.node:
        if node.op not in ("Shape", "Rank", "Size"):
            continue
        if node.name in protected_shape_nodes:
            skipped[f"{node.op}:feeds_reshape"] += 1
            continue
        if not node.input:
            skipped[f"{node.op}:missing_input"] += 1
            continue
        input_name, port = _tensor_name_parts(node.input[0])
        if port != 0:
            pass
        input_node = node_map.get(input_name)
        shape_input_ops[input_node.op if input_node is not None else "<missing>"] += 1
        shape = shape_by_tensor.get((input_name, port))
        if shape is None:
            skipped[f"{node.op}:not_concat_pack"] += 1
            continue
        resolved_shape = resolve_shape(shape, bs, unknown_dim)
        if any(dim is None for dim in resolved_shape):
            skipped[f"{node.op}:unknown_shape"] += 1
            continue
        resolved_shape = [int(dim) for dim in resolved_shape]
        dtype = _const_output_dtype(node)
        if node.op == "Shape":
            _replace_node_with_const(
                node, np.asarray(resolved_shape, dtype=dtype.as_numpy_dtype), dtype
            )
            replaced_shape.append(node.name)
        elif node.op == "Rank":
            _replace_node_with_const(
                node, np.asarray(len(resolved_shape), dtype=dtype.as_numpy_dtype), dtype
            )
            replaced_rank.append(node.name)
        elif node.op == "Size":
            size = int(np.prod(resolved_shape)) if resolved_shape else 1
            _replace_node_with_const(
                node, np.asarray(size, dtype=dtype.as_numpy_dtype), dtype
            )
            replaced_size.append(node.name)

    return {
        "enabled": True,
        "reason": "ok",
        "propagated_shapes": len(shape_by_tensor),
        "splitv_shapes": splitv_shapes,
        "passthrough_shapes": passthrough_shapes,
        "replaced_shape": len(replaced_shape),
        "replaced_rank": len(replaced_rank),
        "replaced_size": len(replaced_size),
        "total_replaced": len(replaced_shape) + len(replaced_rank) + len(replaced_size),
        "protected_reshape_shape_nodes": len(protected_shape_nodes),
        "skipped": dict(skipped),
        "top_shape_input_ops": shape_input_ops.most_common(10),
        "sample_shape": replaced_shape[:20],
        "sample_rank": replaced_rank[:20],
        "sample_size": replaced_size[:20],
    }


def _add_const_node(graph_def, name, value, dtype=tf.int32):
    node = graph_def.node.add()
    node.name = name
    node.op = "Const"
    tensor = tf.make_tensor_proto(value, dtype=dtype)
    node.attr["dtype"].type = dtype.as_datatype_enum
    node.attr["value"].tensor.CopyFrom(tensor)
    return node


def _copy_node_attr(dst_node, src_node, attr_names):
    for name in attr_names:
        if name in src_node.attr:
            dst_node.attr[name].CopyFrom(src_node.attr[name])


def _replace_tensor_input(input_name, replacement_by_node):
    if input_name.startswith("^"):
        return input_name
    base, port = _tensor_name_parts(input_name)
    replacement = replacement_by_node.get(base)
    if replacement is None or port != 0:
        return input_name
    return f"{replacement}:0" if ":" in input_name else replacement


def rewrite_same_lhs_matmul_nodes(graph_def, spec_shape_map, bs, args):
    original_node_count = len(graph_def.node)
    node_map = {node.name: node for node in graph_def.node}
    placeholder_count = sum(1 for node in graph_def.node if node.op == "Placeholder")
    fuse_biasadd = getattr(args, "rewrite_same_lhs_matmul_fuse_biasadd", "off") == "on"
    if (
        args.rewrite_same_lhs_matmul != "on"
        and args.rewrite_same_lhs_matmul_min_placeholders > 0
        and placeholder_count < args.rewrite_same_lhs_matmul_min_placeholders
    ):
        return {
            "enabled": False,
            "reason": "placeholder_count_below_min",
            "placeholder_count": placeholder_count,
            "min_placeholders": args.rewrite_same_lhs_matmul_min_placeholders,
            "groups": 0,
            "original_matmuls": 0,
            "fused_matmuls": 0,
            "estimated_matmul_reduction": 0,
            "biasadd_fusion_enabled": fuse_biasadd,
            "biasadd_groups": 0,
            "fused_biasadds": 0,
            "estimated_biasadd_reduction": 0,
            "skipped_biasadd_fusion": {},
            "rewired_edges": 0,
            "skipped": {
                "placeholder_count_below_min": placeholder_count,
            },
            "sample_groups": [],
        }, {}
    data_consumers = defaultdict(list)
    control_consumers = defaultdict(list)
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            if input_name.startswith("^"):
                control_consumers[base].append(node.name)
            else:
                data_consumers[base].append(node.name)

    groups = defaultdict(list)
    skipped = Counter()
    skipped_biasadd_fusion = Counter()
    allowed_rhs_producer_ops = parse_csv_filter(
        args.rewrite_same_lhs_matmul_rhs_producer_ops
    )
    def _find_unique_biasadd_for_item(item):
        matmul_node = item["node"]
        consumers = data_consumers.get(matmul_node.name, [])
        if len(consumers) != 1:
            return None, "matmul_not_unique_data_consumer"
        biasadd_node = node_map.get(consumers[0])
        if biasadd_node is None or biasadd_node.op != "BiasAdd":
            return None, "consumer_not_biasadd"
        if len(biasadd_node.input) < 2:
            return None, "biasadd_missing_input"
        if any(input_name.startswith("^") for input_name in biasadd_node.input):
            return None, "biasadd_control_input"
        if control_consumers.get(biasadd_node.name):
            return None, "biasadd_control_consumer"
        value_input = biasadd_node.input[0]
        value_base, value_port = _tensor_name_parts(value_input)
        if value_base != matmul_node.name or value_port != 0:
            return None, "biasadd_value_not_matmul"
        if "T" in biasadd_node.attr and (
            biasadd_node.attr["T"].type != item["node"].attr["T"].type
        ):
            return None, "biasadd_dtype_mismatch"
        bias_input = biasadd_node.input[1]
        bias_shape = _tensor_shape_for_analysis(bias_input, node_map, spec_shape_map)
        if bias_shape is None:
            return None, "bias_shape_unknown"
        if len(bias_shape) != 1:
            return None, "bias_not_1d"
        bias_cols = bias_shape[0]
        if not isinstance(bias_cols, int) or bias_cols != item["out_cols"]:
            return None, "bias_cols_mismatch"
        return {
            "node": biasadd_node,
            "bias_input": bias_input,
        }, None

    for node in list(graph_def.node):
        if node.op != "MatMul" or len(node.input) < 2:
            continue
        if any(input_name.startswith("^") for input_name in node.input):
            skipped["control_input"] += 1
            continue
        if control_consumers.get(node.name):
            skipped["control_consumer"] += 1
            continue

        transpose_a, transpose_b = _matmul_transpose_attrs(node)
        if transpose_a or transpose_b:
            skipped["transpose"] += 1
            continue
        if "T" not in node.attr:
            skipped["missing_dtype"] += 1
            continue

        lhs_input = node.input[0]
        rhs_input = node.input[1]
        lhs_producer = node_map.get(_strip_tensor_name(lhs_input))
        rhs_producer = node_map.get(_strip_tensor_name(rhs_input))
        rhs_producer_op = rhs_producer.op if rhs_producer is not None else "missing"
        if (
            allowed_rhs_producer_ops is not None
            and rhs_producer_op not in allowed_rhs_producer_ops
        ):
            skipped["rhs_producer_filter"] += 1
            continue
        if lhs_producer is not None and lhs_producer.op in (
            "MatMul",
            "BatchMatMul",
            "BatchMatMulV2",
        ):
            skipped["matmul_lhs"] += 1
            continue
        if rhs_producer is not None and rhs_producer.op in (
            "MatMul",
            "BatchMatMul",
            "BatchMatMulV2",
        ):
            skipped["matmul_rhs"] += 1
            continue
        lhs_shape = _tensor_shape_for_analysis(lhs_input, node_map, spec_shape_map)
        rhs_shape = _tensor_shape_for_analysis(rhs_input, node_map, spec_shape_map)
        if lhs_shape is None or rhs_shape is None:
            skipped["unknown_shape"] += 1
            continue
        if len(lhs_shape) != 2 or len(rhs_shape) != 2:
            skipped["non_2d"] += 1
            continue
        k_dim, out_cols = _matmul_contract_and_out_dims(
            lhs_shape,
            rhs_shape,
            False,
            False,
        )
        out_rows = lhs_shape[-2]
        if out_rows is None:
            out_rows = bs
        if not isinstance(k_dim, int) or not isinstance(out_cols, int):
            skipped["unknown_contract"] += 1
            continue
        if not isinstance(out_rows, int):
            skipped["unknown_rows"] += 1
            continue
        if k_dim <= 0 or out_cols <= 0:
            skipped["bad_shape"] += 1
            continue
        if args.rewrite_same_lhs_matmul_max_k_dim > 0 and (
            k_dim > args.rewrite_same_lhs_matmul_max_k_dim
        ):
            skipped["k_dim_cap"] += 1
            continue
        if args.rewrite_same_lhs_matmul_max_single_out_cols > 0 and (
            out_cols > args.rewrite_same_lhs_matmul_max_single_out_cols
        ):
            skipped["single_out_cols_cap"] += 1
            continue

        item = {
            "node": node,
            "lhs_input": lhs_input,
            "rhs_input": rhs_input,
            "out_rows": out_rows,
            "out_cols": out_cols,
            "lhs_shape": _resolved_shape_for_report(
                lhs_shape, bs, args.unknown_dim
            ),
        }
        key = (lhs_input, node.attr["T"].type, k_dim)
        groups[key].append(item)

    candidates = []
    for (lhs_input, dtype_enum, k_dim), items in groups.items():
        if len(items) < args.rewrite_same_lhs_matmul_min_group:
            continue
        ordered_items = sorted(items, key=lambda item: item["out_cols"], reverse=True)
        max_total_cols = args.rewrite_same_lhs_matmul_max_total_cols
        chunks = []
        if max_total_cols <= 0:
            chunks = [ordered_items]
        else:
            current = []
            current_cols = 0
            for item in ordered_items:
                item_cols = item["out_cols"]
                if item_cols > max_total_cols:
                    skipped["single_item_total_cols_cap"] += 1
                    continue
                if current and current_cols + item_cols > max_total_cols:
                    chunks.append(current)
                    current = []
                    current_cols = 0
                current.append(item)
                current_cols += item_cols
            if current:
                chunks.append(current)
        for chunk_index, chunk in enumerate(chunks):
            if len(chunk) < args.rewrite_same_lhs_matmul_min_group:
                skipped["chunk_too_small"] += 1
                continue
            candidates.append(
                {
                    "mode": "same_lhs",
                    "lhs_input": lhs_input,
                    "dtype_enum": dtype_enum,
                    "k_dim": k_dim,
                    "total_cols": sum(item["out_cols"] for item in chunk),
                    "total_rows": chunk[0]["out_rows"],
                    "estimated_flops": (
                        2
                        * chunk[0]["out_rows"]
                        * k_dim
                        * sum(item["out_cols"] for item in chunk)
                    ),
                    "items": chunk,
                    "chunk_index": chunk_index,
                    "original_group_size": len(items),
                }
            )

    if args.rewrite_same_lhs_matmul_min_mflops > 0:
        min_flops = int(args.rewrite_same_lhs_matmul_min_mflops * 1_000_000)
        kept_candidates = []
        for group in candidates:
            if group.get("estimated_flops", 0) >= min_flops:
                kept_candidates.append(group)
            else:
                skipped["below_min_mflops"] += 1
        candidates = kept_candidates

    if args.rewrite_same_lhs_matmul_score == "count":
        candidates.sort(
            key=lambda group: (
                len(group["items"]) - 1,
                group["estimated_flops"],
            ),
            reverse=True,
        )
    elif args.rewrite_same_lhs_matmul_score == "small":
        candidates.sort(
            key=lambda group: (
                -(len(group["items"]) - 1),
                group["estimated_flops"],
                group["total_cols"],
                group["k_dim"],
            )
        )
    elif args.rewrite_same_lhs_matmul_score == "hybrid":
        candidates.sort(
            key=lambda group: (
                group["estimated_flops"] * max(1, len(group["items"]) - 1),
                len(group["items"]) - 1,
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda group: (
                group["estimated_flops"],
                len(group["items"]) - 1,
            ),
            reverse=True,
        )
    selected_candidates = []
    selected_matmuls = set()
    for candidate in candidates:
        available_items = [
            item
            for item in candidate["items"]
            if item["node"].name not in selected_matmuls
        ]
        if len(available_items) < args.rewrite_same_lhs_matmul_min_group:
            skipped["overlap_after_selection"] += 1
            continue
        candidate = dict(candidate)
        candidate["items"] = available_items
        candidate["total_cols"] = sum(
            item["out_cols"] for item in available_items
        )
        candidate["estimated_flops"] = (
            2
            * candidate["total_rows"]
            * candidate["k_dim"]
            * candidate["total_cols"]
        )
        selected_candidates.append(candidate)
        selected_matmuls.update(item["node"].name for item in available_items)
    candidates = selected_candidates
    if args.rewrite_same_lhs_matmul_max_groups > 0:
        candidates = candidates[: args.rewrite_same_lhs_matmul_max_groups]

    replacement_by_node = {}
    fused_groups = []
    for group_index, group in enumerate(candidates):
        items = group["items"]
        prefix = f"musa_same_lhs_matmul_group_{group_index}"
        axis_node = _add_const_node(
            graph_def, f"{prefix}_concat_axis", 1
        )

        concat_node = graph_def.node.add()
        concat_node.name = f"{prefix}_concat"
        concat_node.op = "ConcatV2"
        concat_node.input.extend([item["rhs_input"] for item in items])
        concat_node.input.append(axis_node.name)
        concat_node.attr["N"].i = len(items)
        concat_node.attr["T"].type = group["dtype_enum"]
        concat_node.attr["Tidx"].type = tf.int32.as_datatype_enum

        matmul_node = graph_def.node.add()
        matmul_node.name = f"{prefix}_matmul"
        matmul_node.op = "MatMul"
        matmul_node.input.extend([group["lhs_input"], concat_node.name])
        _copy_node_attr(matmul_node, items[0]["node"], ["T"])
        matmul_node.attr["transpose_a"].b = False
        matmul_node.attr["transpose_b"].b = False

        biasadd_by_matmul = {}
        fuse_group_biasadd = False
        fused_biasadd_node = None
        if fuse_biasadd:
            for item in items:
                biasadd_info, reason = _find_unique_biasadd_for_item(item)
                if biasadd_info is None:
                    skipped_biasadd_fusion[reason] += 1
                else:
                    biasadd_by_matmul[item["node"].name] = biasadd_info
            fuse_group_biasadd = len(biasadd_by_matmul) == len(items)
            if fuse_group_biasadd:
                bias_axis_node = _add_const_node(
                    graph_def,
                    f"{prefix}_bias_concat_axis",
                    0,
                )
                bias_concat_node = graph_def.node.add()
                bias_concat_node.name = f"{prefix}_bias_concat"
                bias_concat_node.op = "ConcatV2"
                bias_concat_node.input.extend(
                    [
                        biasadd_by_matmul[item["node"].name]["bias_input"]
                        for item in items
                    ]
                )
                bias_concat_node.input.append(bias_axis_node.name)
                bias_concat_node.attr["N"].i = len(items)
                bias_concat_node.attr["T"].type = group["dtype_enum"]
                bias_concat_node.attr["Tidx"].type = tf.int32.as_datatype_enum

                first_biasadd = biasadd_by_matmul[items[0]["node"].name]["node"]
                fused_biasadd_node = graph_def.node.add()
                fused_biasadd_node.name = f"{prefix}_biasadd"
                fused_biasadd_node.op = "BiasAdd"
                fused_biasadd_node.input.extend([matmul_node.name, bias_concat_node.name])
                _copy_node_attr(fused_biasadd_node, first_biasadd, ["T", "data_format"])
                if "T" not in fused_biasadd_node.attr:
                    fused_biasadd_node.attr["T"].type = group["dtype_enum"]

        offset = 0
        slice_nodes = []
        bias_slice_nodes = []
        for item_index, item in enumerate(items):
            width = item["out_cols"]
            begin = [0, offset]
            size = [-1, width]
            offset += width
            begin_node = _add_const_node(
                graph_def,
                f"{prefix}_slice_{item_index}_begin",
                begin,
            )
            size_node = _add_const_node(
                graph_def,
                f"{prefix}_slice_{item_index}_size",
                size,
            )
            slice_node = graph_def.node.add()
            slice_node.name = f"{prefix}_slice_{item_index}"
            slice_node.op = "Slice"
            slice_node.input.extend([matmul_node.name, begin_node.name, size_node.name])
            slice_node.attr["T"].type = group["dtype_enum"]
            slice_node.attr["Index"].type = tf.int32.as_datatype_enum
            replacement_by_node[item["node"].name] = slice_node.name
            slice_nodes.append(slice_node.name)
            if fuse_group_biasadd and fused_biasadd_node is not None:
                bias_slice_node = graph_def.node.add()
                bias_slice_node.name = f"{prefix}_bias_slice_{item_index}"
                bias_slice_node.op = "Slice"
                bias_slice_node.input.extend(
                    [fused_biasadd_node.name, begin_node.name, size_node.name]
                )
                bias_slice_node.attr["T"].type = group["dtype_enum"]
                bias_slice_node.attr["Index"].type = tf.int32.as_datatype_enum
                biasadd_node = biasadd_by_matmul[item["node"].name]["node"]
                replacement_by_node[biasadd_node.name] = bias_slice_node.name
                bias_slice_nodes.append(bias_slice_node.name)

        fused_groups.append(
            {
                "mode": "same_lhs",
                "lhs": group["lhs_input"],
                "lhs_shape": items[0]["lhs_shape"],
                "matmul_count": len(items),
                "total_cols": group["total_cols"],
                "total_rows": group["total_rows"],
                "k_dim": group["k_dim"],
                "estimated_mflops": group["estimated_flops"] / 1_000_000.0,
                "chunk_index": group["chunk_index"],
                "original_group_size": group["original_group_size"],
                "sample_matmuls": [item["node"].name for item in items[:8]],
                "fused_matmul": matmul_node.name,
                "slice_nodes": slice_nodes[:8],
                "biasadd_fused": bool(fuse_group_biasadd),
                "fused_biasadd": (
                    fused_biasadd_node.name if fused_biasadd_node is not None else ""
                ),
                "bias_slice_nodes": bias_slice_nodes[:8],
            }
        )

    rewired_edges = 0
    if replacement_by_node:
        for node in list(graph_def.node)[:original_node_count]:
            for index, input_name in enumerate(list(node.input)):
                new_input = _replace_tensor_input(input_name, replacement_by_node)
                if new_input != input_name:
                    node.input[index] = new_input
                    rewired_edges += 1

    return {
        "enabled": True,
        "placeholder_count": placeholder_count,
        "min_placeholders": args.rewrite_same_lhs_matmul_min_placeholders,
        "groups": len(fused_groups),
        "original_matmuls": sum(group["matmul_count"] for group in fused_groups),
        "fused_matmuls": len(fused_groups),
        "estimated_matmul_reduction": sum(
            group["matmul_count"] - 1 for group in fused_groups
        ),
        "biasadd_fusion_enabled": fuse_biasadd,
        "biasadd_groups": sum(1 for group in fused_groups if group["biasadd_fused"]),
        "fused_biasadds": sum(
            group["matmul_count"] for group in fused_groups if group["biasadd_fused"]
        ),
        "estimated_biasadd_reduction": sum(
            group["matmul_count"] - 1
            for group in fused_groups
            if group["biasadd_fused"]
        ),
        "skipped_biasadd_fusion": dict(skipped_biasadd_fusion),
        "rewired_edges": rewired_edges,
        "skipped": dict(skipped),
        "sample_groups": fused_groups[:10],
    }, replacement_by_node




def build_output_fetch_plan(outputs, feed_dict, args):
    optimize = should_optimize_output_fetches(args)
    tensor_feed = {tensor.name: value for tensor, value in feed_dict.items()}
    tensor_feed.update({_strip_tensor_name(tensor.name): value for tensor, value in feed_dict.items()})

    def host_value_for_output(tensor):
        if not optimize:
            return None
        if tensor.name in tensor_feed:
            return tensor_feed[tensor.name]
        if _strip_tensor_name(tensor.name) in tensor_feed:
            return tensor_feed[_strip_tensor_name(tensor.name)]
        if tensor.op.type == "Identity" and len(tensor.op.inputs) == 1 and not tensor.op.control_inputs:
            source = tensor.op.inputs[0]
            if source.name in tensor_feed:
                return tensor_feed[source.name]
            source_node = _strip_tensor_name(source.name)
            if source_node in tensor_feed:
                return tensor_feed[source_node]
        return None

    device_fetches = []
    device_fetch_index = {}
    entries = []
    host_output_count = 0
    deduped_output_count = 0
    for tensor in outputs:
        host_value = host_value_for_output(tensor)
        if host_value is not None:
            entries.append({"kind": "host", "value": host_value, "name": tensor.name})
            host_output_count += 1
            continue

        if optimize and tensor.name in device_fetch_index:
            entries.append(
                {
                    "kind": "device",
                    "index": device_fetch_index[tensor.name],
                    "name": tensor.name,
                }
            )
            deduped_output_count += 1
            continue

        device_fetch_index[tensor.name] = len(device_fetches)
        device_fetches.append(tensor)
        entries.append(
            {
                "kind": "device",
                "index": device_fetch_index[tensor.name],
                "name": tensor.name,
            }
        )

    pack_fetches = (
        should_pack_output_fetches(args)
        and len(device_fetches) >= args.pack_output_fetches_min_outputs
    )
    packed_fetches = device_fetches
    packed_plans = {}
    packed_groups = []
    packed_device_fetch_count = 0
    original_device_fetch_count = len(device_fetches)
    if pack_fetches and device_fetches:
        candidates_by_dtype = defaultdict(list)
        individual_indices = set()

        for index, tensor in enumerate(device_fetches):
            try:
                np_dtype = np.dtype(tensor.dtype.as_numpy_dtype)
            except TypeError:
                individual_indices.add(index)
                continue
            shape = tensor.shape.as_list()
            if shape is None or any(dim is None for dim in shape):
                individual_indices.add(index)
                continue
            numel = int(np.prod(shape)) if shape else 1
            nbytes = numel * np_dtype.itemsize
            if numel <= 0:
                individual_indices.add(index)
                continue
            if (
                args.pack_output_fetches_max_total_mib > 0
                and nbytes > args.pack_output_fetches_max_total_mib * 1024 * 1024
            ):
                individual_indices.add(index)
                continue
            candidates_by_dtype[tensor.dtype.name].append(
                {
                    "index": index,
                    "tensor": tensor,
                    "shape": shape,
                    "numel": numel,
                    "nbytes": nbytes,
                    "dtype": tensor.dtype.name,
                }
            )

        new_fetches = []
        assigned_indices = set()
        group_index = 0
        max_total_bytes = int(
            max(0.0, args.pack_output_fetches_max_total_mib) * 1024 * 1024
        )
        for dtype_name, items in sorted(candidates_by_dtype.items()):
            if len(items) < args.pack_output_fetches_min_outputs:
                continue
            total_nbytes = sum(item["nbytes"] for item in items)
            if max_total_bytes > 0 and total_nbytes > max_total_bytes:
                continue

            flat_tensors = [
                tf.reshape(
                    item["tensor"],
                    [-1],
                    name=f"musa_output_fetch_flat_{group_index}_{item['index']}",
                )
                for item in items
            ]
            packed_tensor = tf.concat(
                flat_tensors,
                axis=0,
                name=f"musa_output_fetch_pack_{group_index}",
            )
            fetch_index = len(new_fetches)
            new_fetches.append(packed_tensor)
            offset = 0
            for item in items:
                packed_plans[item["index"]] = {
                    "kind": "packed",
                    "fetch_index": fetch_index,
                    "offset": offset,
                    "numel": item["numel"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                }
                offset += item["numel"]
                assigned_indices.add(item["index"])
            packed_groups.append(
                {
                    "dtype": dtype_name,
                    "outputs": len(items),
                    "nbytes": total_nbytes,
                    "mib": total_nbytes / (1024.0 * 1024.0),
                    "packed_tensor": packed_tensor.name,
                }
            )
            group_index += 1

        for index, tensor in enumerate(device_fetches):
            if index in assigned_indices:
                continue
            packed_plans[index] = {
                "kind": "device",
                "fetch_index": len(new_fetches),
            }
            new_fetches.append(tensor)

        if assigned_indices:
            packed_fetches = new_fetches
            packed_device_fetch_count = len(packed_fetches)
        else:
            packed_plans = {
                index: {"kind": "device", "fetch_index": index}
                for index in range(len(device_fetches))
            }
            packed_fetches = device_fetches
            pack_fetches = False
    else:
        packed_plans = {
            index: {"kind": "device", "fetch_index": index}
            for index in range(len(device_fetches))
        }
        pack_fetches = False

    def reconstruct(device_values):
        values = device_values if isinstance(device_values, (list, tuple)) else [device_values]
        original_device_values = {}
        for index, plan in packed_plans.items():
            value = values[plan["fetch_index"]]
            if plan["kind"] == "packed":
                flat = np.asarray(value).reshape(-1)
                chunk = flat[plan["offset"] : plan["offset"] + plan["numel"]]
                original_device_values[index] = np.asarray(chunk).reshape(plan["shape"])
            else:
                original_device_values[index] = value
        out = []
        for entry in entries:
            if entry["kind"] == "host":
                out.append(entry["value"])
            else:
                out.append(original_device_values[entry["index"]])
        return out

    return {
        "enabled": optimize,
        "entries": entries,
        "device_fetches": packed_fetches,
        "reconstruct": reconstruct,
        "num_outputs": len(outputs),
        "device_fetch_count": len(packed_fetches),
        "original_device_fetch_count": original_device_fetch_count,
        "host_output_count": host_output_count,
        "deduped_output_count": deduped_output_count,
        "pack_output_fetches": pack_fetches,
        "packed_device_fetch_count": packed_device_fetch_count,
        "packed_output_groups": packed_groups,
        "device_fetch_names": [tensor.name for tensor in packed_fetches[:20]],
    }




def collect_reachable_nodes(graph_def, output_spec):
    node_map = {node.name: node for node in graph_def.node}
    reachable = set()
    stack = [_strip_tensor_name(name) for name in output_spec]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        node = node_map.get(name)
        if node is None:
            continue
        reachable.add(name)
        for input_name in node.input:
            stack.append(_strip_tensor_name(input_name))
    return reachable






def _output_shape_from_node(node, output_index=0):
    if node is None:
        return None
    if "_output_shapes" in node.attr:
        shapes = node.attr["_output_shapes"].list.shape
        if output_index < len(shapes):
            return [
                dim.size if dim.size != -1 else None
                for dim in shapes[output_index].dim
            ]
    if output_index == 0:
        return _shape_from_node_attr(node)
    return None


def _tensor_name_parts(name):
    name = name[1:] if name.startswith("^") else name
    if ":" not in name:
        return name, 0
    base, port = name.rsplit(":", 1)
    try:
        return base, int(port)
    except ValueError:
        return name.split(":")[0], 0




def _resolved_shape_for_report(shape, bs, unknown_dim):
    if shape is None:
        return None
    return list(resolve_shape(shape, bs, unknown_dim))


def _tensor_shape_for_analysis(tensor_name, node_map, spec_shape_map):
    if tensor_name in spec_shape_map:
        return spec_shape_map[tensor_name]
    base, port = _tensor_name_parts(tensor_name)
    if port == 0 and f"{base}:0" in spec_shape_map:
        return spec_shape_map[f"{base}:0"]
    node = node_map.get(base)
    shape = _output_shape_from_node(node, port)
    if shape is not None:
        return shape
    if node is not None and node.op == "Const" and "value" in node.attr:
        try:
            return list(tf.make_ndarray(node.attr["value"].tensor).shape)
        except Exception:
            return None
    return None


def _matmul_transpose_attrs(node):
    if node.op == "MatMul":
        return (
            bool(node.attr["transpose_a"].b)
            if "transpose_a" in node.attr
            else False,
            bool(node.attr["transpose_b"].b)
            if "transpose_b" in node.attr
            else False,
        )
    return (
        bool(node.attr["adj_x"].b) if "adj_x" in node.attr else False,
        bool(node.attr["adj_y"].b) if "adj_y" in node.attr else False,
    )


def _matmul_contract_and_out_dims(lhs_shape, rhs_shape, transpose_a, transpose_b):
    if lhs_shape is None or rhs_shape is None or len(lhs_shape) < 2 or len(rhs_shape) < 2:
        return None, None
    lhs_k = lhs_shape[-2] if transpose_a else lhs_shape[-1]
    rhs_k = rhs_shape[-1] if transpose_b else rhs_shape[-2]
    out_cols = rhs_shape[-2] if transpose_b else rhs_shape[-1]
    k_dim = lhs_k if lhs_k is not None else rhs_k
    if k_dim is None:
        k_dim = rhs_k
    return k_dim, out_cols










def build_feed_items(
    meta,
    graph_def,
    input_spec,
    spec_shape_map,
    slice_min_dims,
    bs,
    args,
    rng,
    reachable_nodes=None,
):
    node_map = {node.name: node for node in graph_def.node}
    items = []
    pinned_feed_holders = []
    use_pinned_feed = should_allocate_individual_pinned_feed(args)
    pinned_feed_used = False
    pinned_feed_error = None
    skipped_unreachable = 0

    for name in input_spec:
        node_name = _strip_output_port(name)
        if reachable_nodes is not None and node_name not in reachable_nodes:
            skipped_unreachable += 1
            continue

        node = node_map.get(node_name)
        if node is None or node.op not in ("Placeholder", "PlaceholderWithDefault"):
            continue
        if "dtype" not in node.attr:
            continue

        dtype = tf.as_dtype(node.attr["dtype"].type)
        np_dtype = dtype.as_numpy_dtype
        node_shape = _shape_from_node_attr(node)
        merged_shape = merge_shape(spec_shape_map.get(name), node_shape)
        if name in slice_min_dims:
            merged_shape = list(merged_shape)
            for dim_index, min_dim in slice_min_dims[name].items():
                if dim_index < len(merged_shape):
                    current = merged_shape[dim_index]
                    if current is None or current < min_dim:
                        merged_shape[dim_index] = min_dim
        run_shape = resolve_shape(merged_shape, bs, args.unknown_dim)

        if use_pinned_feed:
            try:
                value = pinned_random_array(
                    run_shape, np_dtype, rng, pinned_feed_holders
                )
                pinned_feed_used = True
            except Exception as exc:
                pinned_feed_error = str(exc)
                if args.pinned_feed == "on":
                    raise
                use_pinned_feed = False
                value = random_array(run_shape, np_dtype, rng)
        else:
            value = random_array(run_shape, np_dtype, rng)

        items.append(
            {
                "name": name,
                "node_name": node_name,
                "optype": node.op,
                "dtype": dtype,
                "np_dtype": np.dtype(np_dtype),
                "shape": list(run_shape),
                "value": value,
            }
        )

    feed_state = {
        "pinned_holders": pinned_feed_holders,
        "pinned_feed_requested": should_try_pinned_feed(args),
        "pinned_feed_used": pinned_feed_used,
        "pinned_feed_error": pinned_feed_error,
        "skipped_unreachable_inputs": skipped_unreachable,
    }
    return items, feed_state


def _num_elements(shape):
    return int(np.prod(shape)) if shape else 1


def _is_packable_dtype(np_dtype):
    dtype = np.dtype(np_dtype)
    return dtype.kind in ("b", "i", "u", "f", "c")


def allocate_packed_array(flat_values, np_dtype, use_pinned, holders):
    total = sum(value.size for value in flat_values)
    dtype = np.dtype(np_dtype)
    if use_pinned:
        holder = PinnedHostArray((total,), dtype)
        holders.append(holder)
        packed = holder.array
    else:
        packed = np.empty((total,), dtype=dtype)

    offset = 0
    for value in flat_values:
        flat = np.asarray(value, dtype=dtype).reshape(-1)
        next_offset = offset + flat.size
        packed[offset:next_offset] = flat
        offset = next_offset
    return packed


def allocate_concat_array(items, axis, out_shape, np_dtype, use_pinned, holders):
    dtype = np.dtype(np_dtype)
    if use_pinned:
        holder = PinnedHostArray(out_shape, dtype)
        holders.append(holder)
        packed = holder.array
    else:
        packed = np.empty(out_shape, dtype=dtype)

    offset = 0
    for item in items:
        value = np.asarray(item["value"], dtype=dtype)
        width = value.shape[axis]
        slices = [slice(None)] * len(out_shape)
        slices[axis] = slice(offset, offset + width)
        packed[tuple(slices)] = value
        offset += width
    return packed


def allocate_value_copy(value, np_dtype, use_pinned, holders):
    dtype = np.dtype(np_dtype)
    arr = np.asarray(value, dtype=dtype)
    if use_pinned:
        holder = PinnedHostArray(arr.shape, dtype)
        holders.append(holder)
        copied = holder.array
        copied[...] = arr
        return copied
    return np.ascontiguousarray(arr, dtype=dtype)


def build_concat_packed_feed(graph, graph_def, feed_items, args):
    item_by_node = {item['node_name']: item for item in feed_items}
    consumer_nodes = {}
    node_map = {node.name: node for node in graph_def.node}
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            if base in item_by_node:
                consumer_nodes.setdefault(base, set()).add(node.name)
    max_total_bytes = int(max(0.0, args.pack_concat_feed_max_total_mib) * 1024 * 1024)
    chunk_max_bytes = int(max(0.0, args.pack_concat_feed_chunk_max_mib) * 1024 * 1024)
    min_group_bytes = int(max(0.0, args.pack_concat_feed_min_total_mib) * 1024 * 1024)
    selected = []
    selected_total = 0
    selected_feed_nodes = set()
    diagnostics = Counter()
    non_feed_input_ops = Counter()
    candidate_bytes = []
    candidate_inputs = []
    candidate_summaries = []
    for node in graph_def.node:
        if node.op != 'ConcatV2' or len(node.input) < 3:
            continue
        diagnostics['concat_nodes'] += 1
        axis_values = _const_int_list(node_map.get(_strip_tensor_name(node.input[-1])))
        if not axis_values or len(axis_values) != 1:
            diagnostics['skip_bad_axis'] += 1
            continue
        data_inputs = node.input[:-1]
        items = []
        skip_reason = None
        for input_name in data_inputs:
            item = item_by_node.get(_strip_tensor_name(input_name))
            if item is None:
                skip_reason = 'skip_non_feed_input'
                producer = node_map.get(_strip_tensor_name(input_name))
                non_feed_input_ops[producer.op if producer is not None else '<missing>'] += 1
                break
            if item['node_name'] in selected_feed_nodes:
                skip_reason = 'skip_already_selected'
                break
            if consumer_nodes.get(item['node_name'], set()) != {node.name}:
                skip_reason = 'skip_multi_consumer_input'
                break
            items.append(item)
        if skip_reason is not None:
            diagnostics[skip_reason] += 1
            continue
        if len(items) < args.pack_concat_feed_min_inputs:
            diagnostics['skip_too_few_inputs'] += 1
            continue
        dtype = np.dtype(items[0]['np_dtype'])
        if not _is_packable_dtype(dtype):
            diagnostics['skip_unpacked_dtype'] += 1
            continue
        if any((np.dtype(item['np_dtype']) != dtype for item in items)):
            diagnostics['skip_mixed_dtype'] += 1
            continue
        shapes = [list(item['shape']) for item in items]
        rank = len(shapes[0])
        if rank == 0 or any((len(shape) != rank for shape in shapes)):
            diagnostics['skip_rank_mismatch'] += 1
            continue
        axis = int(axis_values[0])
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            diagnostics['skip_axis_out_of_range'] += 1
            continue
        out_shape = list(shapes[0])
        out_shape[axis] = 0
        valid = True
        for shape in shapes:
            for dim_index, dim in enumerate(shape):
                if dim_index == axis:
                    continue
                if dim != out_shape[dim_index]:
                    valid = False
                    break
            if not valid:
                break
            out_shape[axis] += int(shape[axis])
        if not valid:
            diagnostics['skip_shape_mismatch'] += 1
            continue
        nbytes = int(np.prod(out_shape)) * dtype.itemsize
        diagnostics['eligible_before_threshold'] += 1
        diagnostics['eligible_before_threshold_nbytes'] += nbytes
        candidate_bytes.append(nbytes)
        candidate_inputs.append(len(items))
        candidate_summaries.append({'concat_node': node.name, 'inputs': len(items), 'nbytes': nbytes, 'mib': nbytes / (1024.0 * 1024.0), 'axis': axis, 'dtype': str(dtype), 'shape': out_shape})
        if nbytes < min_group_bytes:
            diagnostics['skip_too_small_bytes'] += 1
            continue
        use_chunked = chunk_max_bytes > 0 and nbytes > chunk_max_bytes
        if max_total_bytes > 0 and selected_total + nbytes > max_total_bytes and (not use_chunked):
            diagnostics['skip_total_cap'] += 1
            continue
        chunk_groups = None
        if use_chunked:
            chunk_groups = []
            current_items = []
            current_shape = None
            current_nbytes = 0
            for item, shape in zip(items, shapes):
                item_nbytes = int(getattr(item['value'], 'nbytes', 0) or 0)
                if current_items and current_nbytes + item_nbytes > chunk_max_bytes:
                    chunk_groups.append({'items': current_items, 'shape': current_shape, 'nbytes': current_nbytes})
                    current_items = []
                    current_shape = None
                    current_nbytes = 0
                current_items.append(item)
                if current_shape is None:
                    current_shape = list(shape)
                else:
                    current_shape[axis] += int(shape[axis])
                current_nbytes += item_nbytes
            if current_items:
                chunk_groups.append({'items': current_items, 'shape': current_shape, 'nbytes': current_nbytes})
            if len(chunk_groups) <= 1:
                chunk_groups = None
                use_chunked = False
                if max_total_bytes > 0 and selected_total + nbytes > max_total_bytes:
                    diagnostics['skip_total_cap'] += 1
                    continue
        selected.append({'node': node, 'items': items, 'axis': axis, 'shape': out_shape, 'dtype': dtype, 'nbytes': nbytes, 'chunked': use_chunked, 'chunks': chunk_groups})
        selected_total += nbytes
        selected_feed_nodes.update((item['node_name'] for item in items))
        diagnostics['selected_nodes'] += 1
        diagnostics['selected_inputs'] += len(items)
        diagnostics['selected_nbytes'] += nbytes
        if use_chunked:
            diagnostics['selected_chunked_nodes'] += 1
            diagnostics['selected_chunks'] += len(chunk_groups)
    input_map = {}
    feed_dict = {}
    packed_holders = []
    packed_inputs = []
    pinned_error = None
    use_pinned = should_allocate_individual_pinned_feed(args)
    with graph.as_default():
        for index, group in enumerate(selected):
            if group.get('chunked'):
                chunk_tensors = []
                chunks = group.get('chunks') or []
                for chunk_index, chunk in enumerate(chunks):
                    try:
                        packed_value = allocate_concat_array(chunk['items'], group['axis'], chunk['shape'], group['dtype'], use_pinned, packed_holders)
                        packed_pinned = use_pinned
                    except Exception as exc:
                        pinned_error = str(exc)
                        if args.pinned_feed == 'on':
                            raise
                        packed_value = allocate_concat_array(chunk['items'], group['axis'], chunk['shape'], group['dtype'], False, packed_holders)
                        packed_pinned = False
                    packed_tensor = tf.compat.v1.placeholder(tf.as_dtype(group['dtype']), shape=chunk['shape'], name=f'musa_concat_packed_feed_{index}_chunk_{chunk_index}')
                    feed_dict[packed_tensor] = packed_value
                    chunk_tensors.append(packed_tensor)
                    packed_inputs.append({'concat_node': group['node'].name, 'axis': group['axis'], 'shape': list(chunk['shape']), 'dtype': str(group['dtype']), 'num_original_inputs': len(chunk['items']), 'nbytes': int(packed_value.nbytes), 'pinned': packed_pinned, 'chunked': True, 'chunk_index': chunk_index, 'num_chunks': len(chunks)})
                input_map[f"{group['node'].name}:0"] = tf.concat(chunk_tensors, axis=group['axis'], name=f'musa_concat_packed_feed_{index}_concat')
                continue
            try:
                packed_value = allocate_concat_array(group['items'], group['axis'], group['shape'], group['dtype'], use_pinned, packed_holders)
                packed_pinned = use_pinned
            except Exception as exc:
                pinned_error = str(exc)
                if args.pinned_feed == 'on':
                    raise
                packed_value = allocate_concat_array(group['items'], group['axis'], group['shape'], group['dtype'], False, packed_holders)
                packed_pinned = False
            packed_tensor = tf.compat.v1.placeholder(tf.as_dtype(group['dtype']), shape=group['shape'], name=f'musa_concat_packed_feed_{index}')
            feed_dict[packed_tensor] = packed_value
            input_map[f"{group['node'].name}:0"] = packed_tensor
            packed_inputs.append({'concat_node': group['node'].name, 'axis': group['axis'], 'shape': list(group['shape']), 'dtype': str(group['dtype']), 'num_original_inputs': len(group['items']), 'nbytes': int(packed_value.nbytes), 'pinned': packed_pinned, 'chunked': False})
    selected_names = {item['name'] for group in selected for item in group['items']}
    return {'input_map': input_map, 'feed_dict': feed_dict, 'packed_holders': packed_holders, 'unpacked_items': [item for item in feed_items if item['name'] not in selected_names], 'packed_inputs': packed_inputs, 'selected_concat_nodes': len(selected), 'selected_count': sum((len(group['items']) for group in selected)), 'selected_nbytes': selected_total, 'pinned_error': pinned_error, 'diagnostics': {'counts': dict(diagnostics), 'top_non_feed_input_ops': non_feed_input_ops.most_common(20), 'top_concat_candidates': sorted(candidate_summaries, key=lambda item: item['nbytes'], reverse=True)[:10], 'top_candidate_mib': [value / (1024.0 * 1024.0) for value in sorted(candidate_bytes, reverse=True)[:10]], 'top_candidate_inputs': sorted(candidate_inputs, reverse=True)[:10], 'min_inputs': args.pack_concat_feed_min_inputs, 'min_total_mib': args.pack_concat_feed_min_total_mib, 'max_total_mib': args.pack_concat_feed_max_total_mib, 'chunk_max_mib': args.pack_concat_feed_chunk_max_mib}}



def _slice_value_from_begin_size(value, begin, size):
    shape = list(value.shape)
    slices = []
    out_shape = []
    for dim_index, dim in enumerate(shape):
        start = int(begin[dim_index]) if dim_index < len(begin) else 0
        requested = int(size[dim_index]) if dim_index < len(size) else -1
        if start < 0:
            return None
        if requested < 0:
            stop = dim
        else:
            stop = start + requested
        if stop > dim:
            return None
        slices.append(slice(start, stop))
        out_shape.append(stop - start)
    return value[tuple(slices)], out_shape


def _strided_slice_value_from_node(value, node, node_map):
    if len(node.input) < 4:
        return None
    begin = _const_int_list(node_map.get(_strip_tensor_name(node.input[1])))
    end = _const_int_list(node_map.get(_strip_tensor_name(node.input[2])))
    strides = _const_int_list(node_map.get(_strip_tensor_name(node.input[3])))
    if begin is None or end is None or strides is None:
        return None
    ellipsis_mask = int(node.attr.get("ellipsis_mask").i) if "ellipsis_mask" in node.attr else 0
    new_axis_mask = int(node.attr.get("new_axis_mask").i) if "new_axis_mask" in node.attr else 0
    begin_mask = int(node.attr.get("begin_mask").i) if "begin_mask" in node.attr else 0
    end_mask = int(node.attr.get("end_mask").i) if "end_mask" in node.attr else 0
    shrink_axis_mask = (
        int(node.attr.get("shrink_axis_mask").i)
        if "shrink_axis_mask" in node.attr
        else 0
    )
    if ellipsis_mask or new_axis_mask:
        return None

    rank = len(value.shape)
    if len(begin) < rank or len(end) < rank or len(strides) < rank:
        return None

    slices = []
    for dim_index in range(rank):
        stride = int(strides[dim_index])
        if stride == 0:
            return None
        if shrink_axis_mask & (1 << dim_index):
            index = int(begin[dim_index])
            slices.append(index)
            continue
        start = None if begin_mask & (1 << dim_index) else int(begin[dim_index])
        stop = None if end_mask & (1 << dim_index) else int(end[dim_index])
        slices.append(slice(start, stop, stride))
    sliced = value[tuple(slices)]
    return sliced, list(getattr(sliced, "shape", []))


def _slice_like_value_from_node(value, node, node_map):
    if node.op == "Slice":
        if len(node.input) < 3:
            return None
        begin = _const_int_list(node_map.get(_strip_tensor_name(node.input[1])))
        size = _const_int_list(node_map.get(_strip_tensor_name(node.input[2])))
        if begin is None or size is None:
            return None
        return _slice_value_from_begin_size(value, begin, size)
    if node.op == "StridedSlice":
        return _strided_slice_value_from_node(value, node, node_map)
    return None


def _static_slice_bounds_from_node(value, node, node_map):
    if node.op != "Slice" or len(node.input) < 3:
        return None
    begin = _const_int_list(node_map.get(_strip_tensor_name(node.input[1])))
    size = _const_int_list(node_map.get(_strip_tensor_name(node.input[2])))
    shape = [int(dim) for dim in getattr(value, "shape", [])]
    if begin is None or size is None or len(begin) != len(shape) or len(size) != len(shape):
        return None

    resolved_begin = []
    resolved_size = []
    resolved_end = []
    for dim, raw_begin, raw_size in zip(shape, begin, size):
        start = int(raw_begin)
        extent = int(raw_size)
        if start < 0 or start > dim or extent < -1:
            return None
        if extent == -1:
            extent = dim - start
        end = start + extent
        if end < start or end > dim:
            return None
        resolved_begin.append(start)
        resolved_size.append(extent)
        resolved_end.append(end)
    return {
        "begin": resolved_begin,
        "size": resolved_size,
        "end": resolved_end,
    }


def plan_slice_feed_compaction(feed_items, graph_def, args):
    item_by_node = {item["node_name"]: item for item in feed_items}
    node_map = {node.name: node for node in graph_def.node}
    consumer_nodes = {}
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            if base in item_by_node:
                consumer_nodes.setdefault(base, set()).add(node.name)

    min_saved_bytes = int(
        max(0.0, float(getattr(args, "compact_slice_feed_min_saved_mib", 0.0)))
        * 1024
        * 1024
    )
    min_total_saved_bytes = int(
        max(
            0.0,
            float(getattr(args, "compact_slice_feed_min_total_saved_mib", 0.0)),
        )
        * 1024
        * 1024
    )
    groups = []
    skip_reasons = Counter()

    for item in feed_items:
        consumers = sorted(consumer_nodes.get(item["node_name"], set()))
        if not consumers:
            skip_reasons["no_consumers"] += 1
            continue
        slices = []
        for consumer_name in consumers:
            node = node_map.get(consumer_name)
            if node is None:
                skip_reasons["missing_consumer_node"] += 1
                slices = []
                break
            if node.op != "Slice":
                skip_reasons[f"consumer_op_not_supported:{node.op}"] += 1
                slices = []
                break
            if _strip_tensor_name(node.input[0]) != item["node_name"]:
                skip_reasons["consumer_not_primary_input"] += 1
                slices = []
                break
            bounds = _static_slice_bounds_from_node(item["value"], node, node_map)
            if bounds is None:
                skip_reasons["unsupported_or_non_const_slice"] += 1
                slices = []
                break
            slices.append({"node": node, **bounds})
        if not slices:
            continue

        rank = len(slices[0]["begin"])
        bounds_begin = [min(part["begin"][dim] for part in slices) for dim in range(rank)]
        bounds_end = [max(part["end"][dim] for part in slices) for dim in range(rank)]
        compact_shape = [end - begin for begin, end in zip(bounds_begin, bounds_end)]
        compact_view = item["value"][
            tuple(slice(begin, end) for begin, end in zip(bounds_begin, bounds_end))
        ]
        original_nbytes = int(getattr(item["value"], "nbytes", 0) or 0)
        compact_nbytes = int(getattr(compact_view, "nbytes", 0) or 0)
        saved_nbytes = original_nbytes - compact_nbytes
        if saved_nbytes <= 0:
            skip_reasons["no_savings"] += 1
            continue
        if saved_nbytes < min_saved_bytes:
            skip_reasons["saved_below_min"] += 1
            continue

        for part in slices:
            part["adjusted_begin"] = [
                start - base for start, base in zip(part["begin"], bounds_begin)
            ]
        groups.append(
            {
                "item": item,
                "slices": slices,
                "bounds_begin": bounds_begin,
                "bounds_end": bounds_end,
                "compact_shape": compact_shape,
                "compact_view": compact_view,
                "original_nbytes": original_nbytes,
                "compact_nbytes": compact_nbytes,
                "saved_nbytes": saved_nbytes,
            }
        )

    total_saved_nbytes = sum(group["saved_nbytes"] for group in groups)
    if total_saved_nbytes < min_total_saved_bytes:
        if groups:
            skip_reasons["total_saved_below_min"] += len(groups)
        groups = []
        total_saved_nbytes = 0

    return {
        "groups": groups,
        "selected_source_inputs": len(groups),
        "selected_slice_nodes": sum(len(group["slices"]) for group in groups),
        "selected_original_nbytes": sum(
            group["original_nbytes"] for group in groups
        ),
        "selected_nbytes": sum(group["compact_nbytes"] for group in groups),
        "saved_nbytes": total_saved_nbytes,
        "input_delta": 0,
        "skip_reasons": dict(skip_reasons),
    }


def build_slice_compacted_feed(graph, graph_def, feed_items, args):
    plan = plan_slice_feed_compaction(feed_items, graph_def, args)
    input_map = {}
    feed_dict = {}
    packed_holders = []
    compacted_inputs = []
    pinned_error = None
    use_pinned = should_allocate_individual_pinned_feed(args)

    with graph.as_default():
        for group_index, group in enumerate(plan["groups"]):
            item = group["item"]
            try:
                compact_value = allocate_value_copy(
                    group["compact_view"],
                    item["np_dtype"],
                    use_pinned,
                    packed_holders,
                )
                compact_pinned = use_pinned
            except Exception as exc:
                pinned_error = str(exc)
                if args.pinned_feed == "on":
                    raise
                compact_value = allocate_value_copy(
                    group["compact_view"],
                    item["np_dtype"],
                    False,
                    packed_holders,
                )
                compact_pinned = False

            compact_tensor = tf.compat.v1.placeholder(
                item["dtype"],
                shape=group["compact_shape"],
                name=f"musa_slice_compacted_feed_{group_index}",
            )
            feed_dict[compact_tensor] = compact_value
            for slice_index, part in enumerate(group["slices"]):
                replacement = tf.slice(
                    compact_tensor,
                    part["adjusted_begin"],
                    part["size"],
                    name=f"musa_slice_compacted_output_{group_index}_{slice_index}",
                )
                input_map[f"{part['node'].name}:0"] = replacement

            compacted_inputs.append(
                {
                    "source_input": item["name"],
                    "slice_count": len(group["slices"]),
                    "shape": list(compact_value.shape),
                    "dtype": str(np.dtype(item["np_dtype"])),
                    "nbytes": int(compact_value.nbytes),
                    "saved_nbytes": group["saved_nbytes"],
                    "pinned": compact_pinned,
                }
            )

    selected_names = {group["item"]["name"] for group in plan["groups"]}
    return {
        **plan,
        "input_map": input_map,
        "feed_dict": feed_dict,
        "packed_holders": packed_holders,
        "unpacked_items": [
            item for item in feed_items if item["name"] not in selected_names
        ],
        "compacted_inputs": compacted_inputs,
        "pinned_error": pinned_error,
        "diagnostics": {
            "mode": "bounding_box",
            "input_delta": plan["input_delta"],
            "skip_reasons": plan["skip_reasons"],
        },
    }


def slice_pack_added_inputs(groups):
    return sum(max(0, len(group.get("slices", ())) - 1) for group in groups)


def filter_slice_pack_groups(
    groups, *, single_consumer_only=False, max_direct_added_inputs=0
):
    if not single_consumer_only:
        return groups, 0
    selected_indices = {
        index
        for index, group in enumerate(groups)
        if len(group.get("slices", ())) == 1
    }
    remaining_added_inputs = max(0, int(max_direct_added_inputs))
    ranked_multi_consumer = sorted(
        (
            (index, group)
            for index, group in enumerate(groups)
            if len(group.get("slices", ())) > 1
        ),
        key=lambda pair: (
            len(pair[1]["slices"]),
            -int(pair[1].get("saved_nbytes", 0)),
            str(pair[1].get("item", {}).get("name", "")),
        ),
    )
    for index, group in ranked_multi_consumer:
        added_inputs = len(group["slices"]) - 1
        if added_inputs > remaining_added_inputs:
            continue
        selected_indices.add(index)
        remaining_added_inputs -= added_inputs
    selected = [
        group for index, group in enumerate(groups) if index in selected_indices
    ]
    return selected, len(groups) - len(selected)


def build_slice_packed_feed(graph, graph_def, feed_items, args):
    item_by_node = {item["node_name"]: item for item in feed_items}
    node_map = {node.name: node for node in graph_def.node}
    allowed_slice_ops = {
        part.strip()
        for part in str(args.pack_slice_feed_ops).split(",")
        if part.strip()
    }
    consumer_nodes = {}
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            if base in item_by_node:
                consumer_nodes.setdefault(base, set()).add(node.name)

    max_total_bytes = int(max(0.0, args.pack_slice_feed_max_total_mib) * 1024 * 1024)
    min_saved_bytes = int(max(0.0, args.pack_slice_feed_min_saved_mib) * 1024 * 1024)
    min_total_saved_bytes = int(
        max(0.0, args.pack_slice_feed_min_total_saved_mib) * 1024 * 1024
    )
    selected = []
    selected_total = 0
    selected_original_bytes = 0
    selected_feed_nodes = set()
    slice_pack_skip_reasons = Counter()
    disallowed_consumer_ops = Counter()

    for item in feed_items:
        if item["node_name"] in selected_feed_nodes:
            continue
        consumers = sorted(consumer_nodes.get(item["node_name"], set()))
        if not consumers:
            slice_pack_skip_reasons["no_consumers"] += 1
            continue
        slice_groups = []
        valid = True
        for consumer_name in consumers:
            node = node_map.get(consumer_name)
            if node is None:
                slice_pack_skip_reasons["missing_consumer_node"] += 1
                valid = False
                break
            if node.op not in allowed_slice_ops:
                slice_pack_skip_reasons["consumer_op_not_allowed"] += 1
                disallowed_consumer_ops[node.op] += 1
                valid = False
                break
            if _strip_tensor_name(node.input[0]) != item["node_name"]:
                slice_pack_skip_reasons["consumer_not_primary_input"] += 1
                valid = False
                break
            sliced = _slice_like_value_from_node(item["value"], node, node_map)
            if sliced is None:
                slice_pack_skip_reasons["unsupported_or_non_const_slice"] += 1
                valid = False
                break
            sliced_value, sliced_shape = sliced
            slice_groups.append(
                {
                    "node": node,
                    "value": sliced_value,
                    "shape": sliced_shape,
                    "nbytes": int(getattr(sliced_value, "nbytes", 0) or 0),
                }
            )
        if not valid or not slice_groups:
            continue

        original_bytes = int(getattr(item["value"], "nbytes", 0) or 0)
        slice_bytes = sum(group["nbytes"] for group in slice_groups)
        saved_bytes = original_bytes - slice_bytes
        if saved_bytes < min_saved_bytes:
            slice_pack_skip_reasons["saved_below_min"] += 1
            continue
        if max_total_bytes > 0 and selected_total + slice_bytes > max_total_bytes:
            slice_pack_skip_reasons["total_cap"] += 1
            continue

        selected.append(
            {
                "item": item,
                "slices": slice_groups,
                "original_nbytes": original_bytes,
                "slice_nbytes": slice_bytes,
                "saved_nbytes": saved_bytes,
            }
        )
        selected_total += slice_bytes
        selected_original_bytes += original_bytes
        selected_feed_nodes.add(item["node_name"])

    single_consumer_only = (
        getattr(args, "pack_slice_feed_single_consumer_only", "off") == "on"
    )
    single_consumer_candidates = sum(
        1 for group in selected if len(group["slices"]) == 1
    )
    selected, single_consumer_filtered = filter_slice_pack_groups(
        selected,
        single_consumer_only=single_consumer_only,
        max_direct_added_inputs=getattr(
            args, "pack_slice_feed_max_direct_added_inputs", 0
        ),
    )
    if single_consumer_filtered:
        slice_pack_skip_reasons["single_consumer_filtered"] += (
            single_consumer_filtered
        )
    selected_total = sum(group["slice_nbytes"] for group in selected)
    selected_original_bytes = sum(
        group["original_nbytes"] for group in selected
    )
    selected_saved_bytes = sum(group["saved_nbytes"] for group in selected)
    if selected_saved_bytes < min_total_saved_bytes:
        if selected:
            slice_pack_skip_reasons["total_saved_below_min"] += len(selected)
        selected = []
        selected_total = 0
        selected_original_bytes = 0
        selected_saved_bytes = 0

    selected_source_inputs = len(selected)
    selected_slice_nodes = sum(len(group["slices"]) for group in selected)
    added_inputs = slice_pack_added_inputs(selected)
    grouped_flat_mode = False
    if (
        selected
        and args.pack_slice_feed == "auto"
        and args.pack_slice_feed_max_added_inputs >= 0
        and added_inputs > args.pack_slice_feed_max_added_inputs
    ):
        grouped_min_saved_bytes = int(
            max(0.0, args.pack_slice_feed_grouped_min_saved_mib) * 1024 * 1024
        )
        if selected_saved_bytes >= grouped_min_saved_bytes:
            grouped_flat_mode = True
        else:
            return {
                "input_map": {},
                "feed_dict": {},
                "packed_holders": [],
                "unpacked_items": feed_items,
                "packed_inputs": [],
                "selected_source_inputs": 0,
                "selected_slice_nodes": 0,
                "selected_original_nbytes": selected_original_bytes,
                "selected_nbytes": selected_total,
                "saved_nbytes": selected_saved_bytes,
                "pinned_error": None,
                "diagnostics": {
                    "auto_skip_reason": "added_input_cap",
                    "slice_pack_skip_reasons": dict(slice_pack_skip_reasons),
                    "top_disallowed_consumer_ops": disallowed_consumer_ops.most_common(20),
                    "candidate_source_inputs": selected_source_inputs,
                    "candidate_slice_nodes": selected_slice_nodes,
                    "candidate_added_inputs": added_inputs,
                    "max_added_inputs": args.pack_slice_feed_max_added_inputs,
                    "grouped_min_saved_mib": args.pack_slice_feed_grouped_min_saved_mib,
                    "single_consumer_only": single_consumer_only,
                    "single_consumer_candidates": single_consumer_candidates,
                    "single_consumer_filtered": single_consumer_filtered,
                    "max_direct_added_inputs": getattr(
                        args, "pack_slice_feed_max_direct_added_inputs", 0
                    ),
                    "direct_added_inputs": added_inputs,
                },
            }

    input_map = {}
    feed_dict = {}
    packed_holders = []
    packed_inputs = []
    pinned_error = None
    use_pinned = should_allocate_individual_pinned_feed(args)

    with graph.as_default():
        for group_index, group in enumerate(selected):
            item = group["item"]
            if grouped_flat_mode and len(group["slices"]) > 1:
                dtype = np.dtype(item["np_dtype"])
                flats = [
                    np.asarray(slice_group["value"], dtype=dtype).reshape(-1)
                    for slice_group in group["slices"]
                ]
                try:
                    packed_value = allocate_packed_array(
                        flats,
                        dtype,
                        use_pinned,
                        packed_holders,
                    )
                    packed_pinned = use_pinned
                except Exception as exc:
                    pinned_error = str(exc)
                    if args.pinned_feed == "on":
                        raise
                    packed_value = allocate_packed_array(
                        flats,
                        dtype,
                        False,
                        packed_holders,
                    )
                    packed_pinned = False

                packed_tensor = tf.compat.v1.placeholder(
                    item["dtype"],
                    shape=[packed_value.size],
                    name=f"musa_slice_packed_feed_{group_index}",
                )
                feed_dict[packed_tensor] = packed_value

                offset = 0
                for slice_index, slice_group in enumerate(group["slices"]):
                    numel = _num_elements(slice_group["shape"])
                    part = tf.slice(
                        packed_tensor,
                        [offset],
                        [numel],
                        name=(
                            "musa_slice_packed_feed_slice_"
                            f"{group_index}_{slice_index}"
                        ),
                    )
                    replacement = tf.reshape(
                        part,
                        slice_group["shape"],
                        name=(
                            "musa_slice_packed_feed_reshape_"
                            f"{group_index}_{slice_index}"
                        ),
                    )
                    input_map[f"{slice_group['node'].name}:0"] = replacement
                    offset += numel

                packed_inputs.append(
                    {
                        "source_input": item["name"],
                        "slice_node": None,
                        "slice_count": len(group["slices"]),
                        "shape": [int(packed_value.size)],
                        "dtype": str(dtype),
                        "nbytes": int(packed_value.nbytes),
                        "pinned": packed_pinned,
                        "mode": "grouped_flat",
                    }
                )
                continue

            for slice_index, slice_group in enumerate(group["slices"]):
                try:
                    packed_value = allocate_value_copy(
                        slice_group["value"],
                        item["np_dtype"],
                        use_pinned,
                        packed_holders,
                    )
                    packed_pinned = use_pinned
                except Exception as exc:
                    pinned_error = str(exc)
                    if args.pinned_feed == "on":
                        raise
                    packed_value = allocate_value_copy(
                        slice_group["value"],
                        item["np_dtype"],
                        False,
                        packed_holders,
                    )
                    packed_pinned = False

                packed_tensor = tf.compat.v1.placeholder(
                    item["dtype"],
                    shape=list(packed_value.shape),
                    name=(
                        f"musa_slice_packed_feed_{group_index}_{slice_index}"
                    ),
                )
                input_map[f"{slice_group['node'].name}:0"] = packed_tensor
                feed_dict[packed_tensor] = packed_value
                packed_inputs.append(
                    {
                        "source_input": item["name"],
                        "slice_node": slice_group["node"].name,
                        "shape": list(packed_value.shape),
                        "dtype": str(np.dtype(item["np_dtype"])),
                        "nbytes": int(packed_value.nbytes),
                        "pinned": packed_pinned,
                        "mode": "direct",
                    }
                )

    selected_names = {group["item"]["name"] for group in selected}
    return {
        "input_map": input_map,
        "feed_dict": feed_dict,
        "packed_holders": packed_holders,
        "unpacked_items": [
            item for item in feed_items if item["name"] not in selected_names
        ],
        "packed_inputs": packed_inputs,
        "selected_source_inputs": selected_source_inputs,
        "selected_slice_nodes": selected_slice_nodes,
        "selected_original_nbytes": selected_original_bytes,
        "selected_nbytes": selected_total,
        "saved_nbytes": selected_saved_bytes,
        "pinned_error": pinned_error,
        "diagnostics": {
            "mode": (
                "grouped_flat"
                if grouped_flat_mode
                else "direct_capped"
                if single_consumer_only and added_inputs > 0
                else "direct_single_consumer"
                if single_consumer_only
                else "direct"
            ),
            "slice_pack_skip_reasons": dict(slice_pack_skip_reasons),
            "top_disallowed_consumer_ops": disallowed_consumer_ops.most_common(20),
            "candidate_added_inputs": added_inputs,
            "single_consumer_only": single_consumer_only,
            "single_consumer_candidates": single_consumer_candidates,
            "single_consumer_filtered": single_consumer_filtered,
            "max_direct_added_inputs": getattr(
                args, "pack_slice_feed_max_direct_added_inputs", 0
            ),
            "direct_added_inputs": added_inputs,
        },
    }




def parse_spec_id(spec_path: Path):
    match = re.search(r"(\d+)$", spec_path.stem)
    return match.group(1) if match else spec_path.stem


def detect_pb(spec_path: Path, explicit_pb: Union[str, None], extra_search_roots=None):
    if explicit_pb:
        pb = Path(explicit_pb).resolve()
        if not pb.exists():
            raise FileNotFoundError(pb)
        return pb

    spec_id = parse_spec_id(spec_path)
    pb_name = f"frozen_graph_{spec_id}.pb"
    search_roots = [
        spec_path.parent,
        Path.cwd(),
        Path.cwd() / "artifacts",
        Path.cwd() / "frozen_out",
    ]
    if extra_search_roots:
        search_roots.extend(extra_search_roots)

    candidates = []
    seen = set()
    for root in search_roots:
        root = Path(root).resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        candidates.extend(root.glob(f"**/{pb_name}"))
    if not candidates:
        raise FileNotFoundError(f"Auto-detect pb failed, expected file name: {pb_name}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _strip_tensor_name(name):
    base = name[1:] if name.startswith("^") else name
    return base.split(":")[0]


def _const_int_list(node):
    if node is None:
        return None
    if node.op != "Const" or "value" not in node.attr:
        return None
    tensor = node.attr["value"].tensor
    try:
        arr = tf.make_ndarray(tensor)
        if np.issubdtype(arr.dtype, np.integer):
            return [int(x) for x in arr.reshape(-1).tolist()]
    except Exception:
        pass
    if tensor.int_val:
        return list(tensor.int_val)
    if tensor.int64_val:
        return [int(x) for x in tensor.int64_val]
    return None


def infer_placeholder_min_dims(meta):
    node_map = {node.name: node for node in meta.graph_def.node}
    mins = {}
    for node in meta.graph_def.node:
        if node.op != "Slice" or len(node.input) < 3:
            continue
        x_name = _strip_tensor_name(node.input[0])
        x_node = node_map.get(x_name)
        if not x_node or x_node.op not in ("Placeholder", "PlaceholderWithDefault"):
            continue
        begin = _const_int_list(node_map.get(_strip_tensor_name(node.input[1])))
        size = _const_int_list(node_map.get(_strip_tensor_name(node.input[2])))
        if not begin or not size:
            continue
        tname = f"{x_name}:0"
        required = mins.setdefault(tname, {})
        for index, (begin_i, size_i) in enumerate(zip(begin, size)):
            if index == 0 or begin_i < 0:
                continue
            need = begin_i + (size_i if size_i > 0 else 1)
            required[index] = max(required.get(index, 0), int(need))
    return mins


def extract_core_error(stack: Union[str, None]):
    if not stack:
        return None
    lines = [line.strip() for line in stack.strip().splitlines() if line.strip()]
    key_patterns = (
        "ResourceExhaustedError",
        "InvalidArgumentError",
        "NotFoundError",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "ran out of memory",
        "oom",
    )
    for line in reversed(lines):
        low = line.lower()
        if "original stack trace" in low:
            continue
        if any(pattern.lower() in low for pattern in key_patterns):
            return line
    for line in reversed(lines):
        if not line.startswith(("File ", "Traceback")):
            return line
    return lines[-1] if lines else None




def run_single_spec(
    spec_path: Path,
    pb_path: Path,
    args,
    bs: int,
    musa_loaded: bool,
    runner_out: Path,
):
    meta = load_meta(spec_path)
    input_spec = read_node_list_collection(meta, "input_spec")
    output_spec = read_node_list_collection(meta, "output_spec")
    selected_output_indices = parse_index_selection(
        args.output_indices, len(output_spec)
    )
    selected_output_spec = [output_spec[index] for index in selected_output_indices]
    spec_shape_map = build_spec_tensor_shape_map(meta)
    slice_min_dims = infer_placeholder_min_dims(meta)

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(pb_path.read_bytes())

    graph_rewrite_summary = {
        "xla_jit_scope": args.xla_jit_scope,
        "identity_bypass": {
            "enabled": False,
            "eligible_identity_nodes": 0,
            "skipped_with_control": 0,
            "skipped_control_consumer": 0,
            "rewired_edges": 0,
            "pruned_identity_nodes": 0,
            "output_remaps": [],
        },
        "pow_square": {
            "enabled": False,
            "rewritten": 0,
            "sample": [],
            "skipped_non_const": 0,
            "skipped_non_square": 0,
        },
        "same_lhs_matmul": {"enabled": False},
        "concat_static_precompute": None,
    }
    if should_bypass_identity(args, graph_def):
        selected_output_spec, identity_summary = bypass_identity_nodes(
            graph_def, selected_output_spec
        )
        graph_rewrite_summary["identity_bypass"] = identity_summary
    if should_rewrite_pow_square(args):
        graph_rewrite_summary["pow_square"] = rewrite_pow_square_nodes(graph_def)
    if args.rewrite_same_lhs_matmul == "on":
        same_lhs_summary, replacements = rewrite_same_lhs_matmul_nodes(
            graph_def, spec_shape_map, bs, args
        )
        selected_output_spec = [
            _replace_tensor_input(name, replacements)
            for name in selected_output_spec
        ]
        graph_rewrite_summary["same_lhs_matmul"] = same_lhs_summary

    reachable_nodes = (
        collect_reachable_nodes(graph_def, selected_output_spec)
        if args.feed_only_reachable
        else None
    )
    rng = np.random.default_rng(args.seed)
    feed_items, feed_state = build_feed_items(
        meta,
        graph_def,
        input_spec,
        spec_shape_map,
        slice_min_dims,
        bs,
        args,
        rng,
        reachable_nodes=reachable_nodes,
    )

    with tf.Graph().as_default() as graph:
        feed_dict = {}
        input_map = {}
        active_feed_items = feed_items
        concat_pack_state = None
        slice_compact_state = None
        slice_pack_state = None

        if should_pack_concat_feed(args):
            try:
                concat_pack_state = build_concat_packed_feed(
                    graph, graph_def, active_feed_items, args
                )
                if concat_pack_state["selected_count"] > 0:
                    input_map.update(concat_pack_state["input_map"])
                    feed_dict.update(concat_pack_state["feed_dict"])
                    active_feed_items = concat_pack_state["unpacked_items"]
                    if args.rewrite_concat_static_precompute == "on":
                        graph_rewrite_summary["concat_static_precompute"] = (
                            rewrite_concat_static_precompute_nodes(
                                graph_def, concat_pack_state, bs=bs, args=args
                            )
                        )
            except Exception:
                if args.pack_concat_feed == "on":
                    raise
                concat_pack_state = None

        if args.compact_slice_feed == "on":
            slice_compact_state = build_slice_compacted_feed(
                graph, graph_def, active_feed_items, args
            )
            if slice_compact_state["selected_slice_nodes"] > 0:
                input_map.update(slice_compact_state["input_map"])
                feed_dict.update(slice_compact_state["feed_dict"])
                active_feed_items = slice_compact_state["unpacked_items"]

        if should_pack_slice_feed(args):
            slice_pack_state = build_slice_packed_feed(
                graph, graph_def, active_feed_items, args
            )
            if slice_pack_state["selected_slice_nodes"] > 0:
                input_map.update(slice_pack_state["input_map"])
                feed_dict.update(slice_pack_state["feed_dict"])
                active_feed_items = slice_pack_state["unpacked_items"]

        use_device_scope = bool(args.device) and not (
            args.xla
            and device_kind(args.device) == "MUSA"
            and not args.xla_device_scope
        )
        with maybe_xla_jit_scope(args):
            if use_device_scope:
                with tf.device(args.device):
                    tf.import_graph_def(graph_def, name="", input_map=input_map)
            else:
                tf.import_graph_def(graph_def, name="", input_map=input_map)

        outputs = [
            graph.get_tensor_by_name(name) for name in selected_output_spec
        ]

        for item in active_feed_items:
            feed_dict[graph.get_tensor_by_name(item["name"])] = item["value"]

        fetch_mode = args.fetch_mode
        output_fetch_plan = None
        reconstruct_fetches = None
        if fetch_mode == "execute_only":
            with tf.control_dependencies(outputs):
                fetches = tf.no_op(name="musa_execute_outputs_only")
        elif fetch_mode == "barrier_scalar":
            with tf.control_dependencies(outputs):
                fetches = tf.identity(
                    tf.constant(0, dtype=tf.int32),
                    name="musa_execute_outputs_barrier_scalar",
                )
        else:
            output_fetch_plan = build_output_fetch_plan(outputs, feed_dict, args)
            fetches = output_fetch_plan["device_fetches"]
            reconstruct_fetches = output_fetch_plan["reconstruct"]
            if not fetches:
                fetches = tf.no_op(name="musa_no_device_output_fetches")

        graph_dump = {
            "enabled": env_flag_enabled(GRAPH_DUMP_ENV),
            "plugin_loaded": musa_loaded,
            "optimizer_enabled": bool(musa_loaded and args.musa_optimizer),
            "dump_dir": None,
            "files": {},
        }
        default_dump_dir = None
        if graph_dump["optimizer_enabled"] and graph_dump["enabled"]:
            default_dump_dir = runner_out / f"{spec_path.stem}_bs_{bs}"

        config = create_session_config(args, musa_loaded=musa_loaded)
        run_error = None
        lat_ms = []
        last_vals = None
        callable_used = False
        callable_error = None
        with configured_graph_dump_dir(default_dump_dir) as active_dump_dir:
            if active_dump_dir is not None:
                graph_dump["dump_dir"] = str(active_dump_dir)
            with tf.compat.v1.Session(graph=graph, config=config) as sess:
                try:
                    run_once = None
                    if should_use_callable(args):
                        feed_tensors = list(feed_dict)
                        feed_values = [feed_dict[tensor] for tensor in feed_tensors]
                        try:
                            callable_runner = sess.make_callable(
                                fetches, feed_list=feed_tensors
                            )

                            def run_once():
                                return callable_runner(*feed_values)

                            callable_used = True
                        except Exception as exc:
                            callable_error = str(exc)
                            if args.use_callable == "on":
                                raise
                    if run_once is None:

                        def run_once():
                            return sess.run(fetches, feed_dict=feed_dict)

                    if reconstruct_fetches is not None:
                        raw_run_once = run_once

                        def run_once():
                            return reconstruct_fetches(raw_run_once())

                    for _ in range(max(0, args.warmup)):
                        run_once()
                    for _ in range(max(1, args.run_iters)):
                        start = time.perf_counter()
                        last_vals = run_once()
                        lat_ms.append((time.perf_counter() - start) * 1000.0)
                except Exception:
                    run_error = traceback.format_exc()

        if graph_dump["dump_dir"]:
            graph_dump["files"] = collect_graph_dump_files(
                graph_dump["dump_dir"]
            )

    result_values = (
        last_vals if isinstance(last_vals, (list, tuple)) else [last_vals]
    )
    result_shapes = []
    if last_vals is not None:
        result_shapes = [
            {
                "shape": list(getattr(value, "shape", [])),
                "dtype": str(getattr(value, "dtype", "")),
                "nbytes": int(getattr(value, "nbytes", 0) or 0),
            }
            for value in result_values
        ]
    feed_summary = summarize_feed_dict(feed_dict, bs=bs)
    feed_summary.update(
        {
            "original_num_inputs": len(feed_items),
            "skipped_unreachable_inputs": feed_state[
                "skipped_unreachable_inputs"
            ],
            "pinned_feed_used": feed_state["pinned_feed_used"],
            "concat_packed": bool(
                concat_pack_state
                and concat_pack_state.get("selected_count", 0) > 0
            ),
            "slice_compacted": bool(
                slice_compact_state
                and slice_compact_state.get("selected_slice_nodes", 0) > 0
            ),
            "slice_packed": bool(
                slice_pack_state
                and slice_pack_state.get("selected_slice_nodes", 0) > 0
            ),
        }
    )
    return {
        "spec_path": str(spec_path),
        "pb_path": str(pb_path),
        "batch_size": bs,
        "num_outputs": len(selected_output_spec),
        "selected_output_indices": selected_output_indices,
        "status": "ok" if run_error is None else "failed",
        "error_core": extract_core_error(run_error),
        "error": run_error,
        "graph_dump": graph_dump,
        "graph_rewrite": graph_rewrite_summary,
        "feed_summary": feed_summary,
        "output_fetch_plan": (
            {
                "enabled": output_fetch_plan["enabled"],
                "num_outputs": output_fetch_plan["num_outputs"],
                "device_fetch_count": output_fetch_plan["device_fetch_count"],
                "original_device_fetch_count": output_fetch_plan[
                    "original_device_fetch_count"
                ],
                "host_output_count": output_fetch_plan["host_output_count"],
                "deduped_output_count": output_fetch_plan[
                    "deduped_output_count"
                ],
                "pack_output_fetches": output_fetch_plan[
                    "pack_output_fetches"
                ],
                "packed_device_fetch_count": output_fetch_plan[
                    "packed_device_fetch_count"
                ],
                "packed_output_groups": output_fetch_plan[
                    "packed_output_groups"
                ],
                "device_fetch_names": output_fetch_plan["device_fetch_names"],
            }
            if output_fetch_plan is not None
            else None
        ),
        "callable": {
            "requested": should_use_callable(args),
            "used": callable_used,
            "error": callable_error,
        },
        "result_shapes": result_shapes,
        "timing_ms": {
            "warmup": max(0, args.warmup),
            "run_iters": max(1, args.run_iters),
            "average": float(np.mean(lat_ms)) if lat_ms else 0.0,
            "trimmed_avg": trimmed_mean(lat_ms, 0.1),
            "min": float(np.min(lat_ms)) if lat_ms else 0.0,
            "max": float(np.max(lat_ms)) if lat_ms else 0.0,
            "p50": percentile(lat_ms, 50),
            "p90": percentile(lat_ms, 90),
            "p95": percentile(lat_ms, 95),
            "all": lat_ms,
        },
    }



def collect_specs(spec: Union[str, None], spec_dir: Union[str, None]):
    if spec:
        path = Path(spec).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]
    root = Path(spec_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    specs = sorted(root.rglob("*.spec"))
    if not specs:
        raise FileNotFoundError(f"No .spec files found under: {root}")
    return specs


def convert_spec_to_pb(spec_path: Path, convert_script: Path, seed: int, out_root: Path):
    cmd = [
        sys.executable,
        str(convert_script),
        "--spec",
        str(spec_path),
        "--seed",
        str(seed),
        "--out_root",
        str(out_root),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
    if res.returncode != 0:
        core = extract_core_error(res.stderr) or extract_core_error(res.stdout)
        raise RuntimeError(
            f"convert failed for {spec_path}\n"
            f"core_error: {core}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}"
        )

    spec_id = parse_spec_id(spec_path)
    pb_path = out_root / spec_path.stem / f"frozen_graph_{spec_id}.pb"
    if not pb_path.exists():
        raise FileNotFoundError(f"convert succeeded but pb missing: {pb_path}")
    return pb_path.resolve()


def load_runtime_plugins(args):
    kind = device_kind(args.device)
    if kind != "MUSA":
        return False

    if args.xla:
        plugin_path = Path(args.musa_plugin).resolve()
        if not plugin_path.exists():
            raise FileNotFoundError(f"MUSA PJRT plugin not found: {plugin_path}")
        tf_adapter_path = Path(args.musa_tf_adapter).resolve()
        if not tf_adapter_path.exists():
            raise FileNotFoundError(
                f"MUSA TensorFlow adapter not found: {tf_adapter_path}"
            )
        os.environ["MUSA_PJRT_PLUGIN_PATH"] = str(plugin_path)
        os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{plugin_path}"
        lib = ctypes.CDLL(str(tf_adapter_path))
        if hasattr(lib, "ForceRegisterMusa"):
            lib.ForceRegisterMusa.restype = ctypes.c_int
            if lib.ForceRegisterMusa() != 1:
                raise RuntimeError(
                    "MUSA PJRT registration failed in TensorFlow adapter"
                )
        return True

    try:
        import tensorflow_musa  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import tensorflow_musa. Install tensorflow_musa for "
            "non-XLA MUSA runs or use --xla with --musa_plugin."
        ) from exc
    devices = tf.config.list_physical_devices("MUSA")
    if not devices:
        raise RuntimeError(f"requested device {args.device}, but no MUSA devices are visible")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frozen PB from spec I/O tensors, with optional TensorFlow XLA."
    )
    parser.add_argument("--spec", default=None, help="Path to a single *.spec.")
    parser.add_argument(
        "--spec_dir", default=None, help="Directory to scan for *.spec files."
    )
    parser.add_argument(
        "--pb", default=None, help="Path to frozen_graph_*.pb. Only with --spec."
    )
    parser.add_argument(
        "--optimization_profile",
        choices=["auto", "off", "meta1", "meta2", "meta3"],
        default="auto",
        help="Apply verified model-specific optimization defaults.",
    )
    parser.add_argument(
        "--cpu_affinity",
        choices=["auto", "on", "off"],
        default="auto",
        help="Bind to CPU cores local to the MUSA GPU. auto enables it for meta1.",
    )
    parser.add_argument(
        "--gpu_pci_bus_id",
        default="",
        help="Optional PCI bus id for CPU affinity, for example 0000:2a:00.0.",
    )
    parser.add_argument("--bs", default="1024", help="Batch size or CSV list.")
    parser.add_argument("--unknown_dim", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--run_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_root", default="runner_out")
    parser.add_argument("--strict", type=parse_bool, default=True)
    parser.add_argument(
        "--device",
        default="/device:MUSA:0",
        choices=["/device:MUSA:0", "/device:CUDA:0", "/device:CPU:0"],
    )
    parser.add_argument("--allow_soft_placement", type=parse_bool, default=True)
    parser.add_argument("--log_device_placement", type=parse_bool, default=False)
    parser.add_argument("--musa_optimizer", type=parse_bool, default=True)
    parser.add_argument(
        "--pinned_feed", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument("--feed_only_reachable", type=parse_bool, default=True)
    parser.add_argument(
        "--use_callable", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument(
        "--optimize_output_fetches",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument(
        "--pack_output_fetches",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--pack_output_fetches_min_outputs", type=int, default=8)
    parser.add_argument(
        "--pack_output_fetches_max_total_mib", type=float, default=16.0
    )
    parser.add_argument(
        "--bypass_identity", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument("--bypass_identity_min_nodes", type=int, default=1000)
    parser.add_argument(
        "--rewrite_pow_square", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_fuse_biasadd",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument("--rewrite_same_lhs_matmul_min_group", type=int, default=2)
    parser.add_argument(
        "--rewrite_same_lhs_matmul_min_placeholders", type=int, default=1000
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_groups", type=int, default=16
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_total_cols", type=int, default=8192
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_score",
        choices=("flops", "hybrid", "count", "small"),
        default="flops",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_min_mflops", type=float, default=0.0
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_rhs_producer_ops", default=""
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_k_dim", type=int, default=0
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_single_out_cols", type=int, default=0
    )
    parser.add_argument(
        "--pack_concat_feed",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--pack_concat_feed_min_inputs", type=int, default=8)
    parser.add_argument(
        "--pack_concat_feed_min_total_mib", type=float, default=1.0
    )
    parser.add_argument(
        "--pack_concat_feed_max_total_mib", type=float, default=256.0
    )
    parser.add_argument(
        "--pack_concat_feed_chunk_max_mib", type=float, default=0.0
    )
    parser.add_argument(
        "--rewrite_concat_static_precompute",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument(
        "--compact_slice_feed", choices=("on", "off"), default="off"
    )
    parser.add_argument(
        "--compact_slice_feed_min_saved_mib", type=float, default=0.0
    )
    parser.add_argument(
        "--compact_slice_feed_min_total_saved_mib", type=float, default=1.0
    )
    parser.add_argument(
        "--pack_slice_feed",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument("--pack_slice_feed_ops", default="Slice")
    parser.add_argument(
        "--pack_slice_feed_min_saved_mib", type=float, default=0.0
    )
    parser.add_argument(
        "--pack_slice_feed_min_total_saved_mib", type=float, default=1.0
    )
    parser.add_argument(
        "--pack_slice_feed_max_total_mib", type=float, default=0.0
    )
    parser.add_argument(
        "--pack_slice_feed_max_added_inputs", type=int, default=0
    )
    parser.add_argument(
        "--pack_slice_feed_grouped_min_saved_mib", type=float, default=8.0
    )
    parser.add_argument(
        "--pack_slice_feed_single_consumer_only",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pack_slice_feed_max_direct_added_inputs", type=int, default=0
    )
    parser.add_argument(
        "--fetch_mode",
        choices=("outputs", "execute_only", "barrier_scalar"),
        default="outputs",
    )
    parser.add_argument("--output_indices", default="all")
    parser.add_argument("--pjrt_max_inflight_transfers", default="0")
    parser.add_argument("--pjrt_max_inflight_executes", default="0")
    parser.add_argument(
        "--pjrt_wait_transfer_done",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_wait_execute_done",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_parallel_pack",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_pack_threads", type=int, default=4
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_pack_min_bytes",
        type=int,
        default=1048576,
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_dirty_ranges",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_pool_order_layout",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_trust_contents",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_cache_reused_buffer_views",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--pjrt_cache_reused_buffer_views_trust_lifetime",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_f32_fast_tf32",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batcher",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_min_group_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_group_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_groups", type=int, default=128
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_min_candidate_dots",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch",
        type=int,
        default=2097152,
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_output_cols",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--musa_xla_hot_tuple_softmax_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_xla_warp_row_reduction_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_xla_warp_row_reduction_reducers",
        choices=("all", "add", "multiply"),
        default="all",
    )
    parser.add_argument(
        "--musa_xla_warp_row_reduction_min_data_elements",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--musa_xla_warp_row_reduction_threads_per_block",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--musa_xla_tuple_warp_row_reduction_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_xla_mixed_tuple_warp_row_reduction_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_xla_mixed_tuple_warp_row_reduction_min_data_elements",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--musa_xla_fusion_merger_materialize_reduction_producer",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_xla_fusion_merger_materialize_min_elements",
        type=int,
        default=10000000,
    )
    parser.add_argument(
        "--musa_xla_fusion_merger_materialize_min_operands",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--musa_xla_classic_thunk_graph",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
    )
    parser.add_argument(
        "--musa_xla_classic_thunk_graph_max_cache_entries",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--musa_xla_gpu_runtime",
        choices=("auto", "classic_thunks", "xla_runtime"),
        default="auto",
    )
    parser.add_argument(
        "--xla_global_jit_level",
        choices=("off", "auto", "on", "on_1", "on_2"),
        default="off",
    )
    parser.add_argument(
        "--xla_jit_scope", choices=("off", "on"), default="off"
    )
    parser.add_argument(
        "--convert_script",
        default=str(SCRIPT_DIR / "convert_spec_to_pb.py"),
    )
    parser.add_argument("--convert_out_root", default=str(SCRIPT_DIR / "frozen_out"))
    parser.add_argument("--xla", action="store_true")
    parser.add_argument("--xla_device_scope", action="store_true")
    parser.add_argument("--xla_dump", action="store_true")
    parser.add_argument("--xla_dump_dir", default=None)
    parser.add_argument("--xla_dump_hlo_pass_re", default="^$")
    parser.add_argument("--xla_dump_max_hlo_modules", default="-1")
    parser.add_argument("--xla_dump_hlo_module_re", default="")
    parser.add_argument("--xla_dump_long_text", type=parse_bool, default=False)
    parser.add_argument("--musa_plugin", default=default_musa_plugin_path())
    parser.add_argument("--musa_tf_adapter", default=default_musa_tf_adapter_path())
    args = parser.parse_args()
    args.explicit_cli_flags = sorted(explicit_cli_flags(sys.argv[1:]))
    args.optimization_profile_selected = "off"
    args.optimization_profile_applied = {}

    # Internal constants for the retained same-LHS + BiasAdd implementation.
    args.rewrite_same_lhs_matmul_auto_min_reduction = 0
    return args



def main():
    args = parse_args()
    if bool(args.spec) == bool(args.spec_dir):
        raise ValueError("Provide exactly one of --spec or --spec_dir")
    if args.pb and not args.spec:
        raise ValueError("--pb can only be used together with --spec")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    convert_out_root = Path(args.convert_out_root).resolve()
    convert_out_root.mkdir(parents=True, exist_ok=True)
    convert_script = Path(args.convert_script).resolve()
    if not convert_script.exists():
        raise FileNotFoundError(f"convert script not found: {convert_script}")

    specs = collect_specs(args.spec, args.spec_dir)
    selected_profile, applied_profile = apply_optimization_profile(
        args, specs, explicit_flags=args.explicit_cli_flags
    )

    musa_loaded = load_runtime_plugins(args)
    run_root = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    bs_values = parse_bs_values(args.bs)
    all_reports = []
    failures = 0

    for spec_path in specs:
        print(f"[INFO] processing spec: {spec_path}")
        pb_path = None
        detect_error = None
        try:
            pb_path = detect_pb(
                spec_path,
                args.pb,
                extra_search_roots=[out_root, convert_out_root],
            )
            print(f"[INFO] found pb: {pb_path}")
        except Exception as exc:
            detect_error = exc
        if pb_path is None:
            print(f"[INFO] pb missing for {spec_path.name}, auto converting ...")
            try:
                pb_path = convert_spec_to_pb(
                    spec_path, convert_script, args.seed, convert_out_root
                )
                print(f"[INFO] auto-convert success: {pb_path}")
            except Exception:
                error = traceback.format_exc()
                failures += 1
                all_reports.append(
                    {
                        "spec_path": str(spec_path),
                        "pb_path": None,
                        "status": "failed",
                        "error_stage": "detect_or_convert_pb",
                        "error_core": extract_core_error(error)
                        or str(detect_error),
                        "error": error,
                    }
                )
                continue

        for bs in bs_values:
            try:
                report = run_single_spec(
                    spec_path.resolve(),
                    pb_path.resolve(),
                    args,
                    bs,
                    musa_loaded,
                    run_root,
                )
            except Exception:
                error = traceback.format_exc()
                report = {
                    "spec_path": str(spec_path),
                    "pb_path": str(pb_path),
                    "batch_size": bs,
                    "status": "failed",
                    "error_stage": "run_inference",
                    "error_core": extract_core_error(error),
                    "error": error,
                    "timing_ms": {},
                }
            if report["status"] != "ok":
                failures += 1
            all_reports.append(report)
            timing = report.get("timing_ms") or {}
            print(
                f"[INFO] run done: spec={spec_path.name} bs={bs} "
                f"status={report['status']} "
                f"average_time_ms={timing.get('average', 0.0)} "
                f"trimmed_avg_ms={timing.get('trimmed_avg', 0.0)}"
            )
            if report.get("error_core"):
                print(f"[INFO] core error: {report['error_core']}")

    summary = {
        "total_specs": len(specs),
        "bs_values": bs_values,
        "total_runs": len(all_reports),
        "ok": sum(item.get("status") == "ok" for item in all_reports),
        "failed": sum(item.get("status") != "ok" for item in all_reports),
    }
    latency_summary = []
    for report in all_reports:
        timing = report.get("timing_ms") or {}
        latency_summary.append(
            {
                "batch_size": report.get("batch_size"),
                "average_time_ms": timing.get("average"),
                "trimmed_avg_ms": timing.get("trimmed_avg"),
            }
        )
    final_report = {
        "args": vars(args),
        "optimization_profile": selected_profile,
        "cpu_affinity": EARLY_CPU_AFFINITY,
        "summary": summary,
        "results": all_reports,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "run_report.json"
    report_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] latency_summary={latency_summary}")
    if failures and args.strict:
        raise RuntimeError("some specs failed, see run_report.json")



if __name__ == "__main__":
    main()
