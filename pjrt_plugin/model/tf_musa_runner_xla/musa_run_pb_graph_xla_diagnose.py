#!/usr/bin/env python3
import argparse
import ast
import ctypes
from contextlib import contextmanager
import hashlib
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


def _append_unique_flag(current_value: str, new_flag: str) -> str:
    tokens = current_value.split()
    if new_flag not in tokens:
        tokens.append(new_flag)
    return " ".join(tokens).strip()


def _set_flag_with_prefix(current_value: str, flag_prefix: str, new_flag: str) -> str:
    tokens = [token for token in current_value.split() if not token.startswith(flag_prefix)]
    tokens.append(new_flag)
    return " ".join(tokens).strip()


def _set_bool_env_from_cli(argv, flag, env_name, default="off"):
    value = _get_cli_arg(argv, flag, default)
    if value in ("on", "true", "1"):
        os.environ[env_name] = "1"
    elif value in ("off", "false", "0"):
        os.environ[env_name] = "0"


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


def _is_env_enabled(name):
    value = os.environ.get(name, "")
    return value not in ("", "0", "false", "False", "off", "OFF")


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


def _remove_cli_options(argv, option_modes):
    cleaned = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        matched_flag = None
        matched_mode = None
        for flag, mode in option_modes.items():
            if arg == flag or arg.startswith(f"{flag}="):
                matched_flag = flag
                matched_mode = mode
                break
        if matched_flag is None:
            cleaned.append(arg)
            index += 1
            continue
        if "=" in arg:
            index += 1
            continue
        if (
            matched_mode in ("value", "optional")
            and index + 1 < len(argv)
            and not str(argv[index + 1]).startswith("--")
        ):
            index += 2
        else:
            index += 1
    return cleaned


def _append_cli_option(argv, flag, value):
    argv.extend([flag, str(value)])


def _truthy_cli_value(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _sweep_cases_from_arg(value):
    cases = []
    seen = set()
    for raw in str(value).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in ("off", "disable", "disabled", "none"):
            case = {"label": "off", "batcher": "off", "max_output_cols": None}
        elif token in ("on", "all", "0", "none_limit", "nolimit"):
            case = {"label": "cols_all", "batcher": "on", "max_output_cols": 0}
        else:
            try:
                max_output_cols = int(token)
            except ValueError as exc:
                raise ValueError(
                    "--musa_xla_same_shape_dot_batch_sweep accepts comma "
                    f"values from off,on,0,<int>, got {raw!r}"
                ) from exc
            if max_output_cols < 0:
                raise ValueError(
                    "--musa_xla_same_shape_dot_batch_sweep values must be >= 0"
                )
            case = {
                "label": f"cols_{max_output_cols}",
                "batcher": "on",
                "max_output_cols": max_output_cols,
            }
        if case["label"] not in seen:
            seen.add(case["label"])
            cases.append(case)
    if not cases:
        raise ValueError("--musa_xla_same_shape_dot_batch_sweep is empty")
    return cases


def _positive_int_sweep_values_from_arg(value, flag_name):
    if value is None or str(value).strip() == "":
        return [None]
    values = []
    seen = set()
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(f"{flag_name} accepts comma-separated integers") from exc
        if parsed <= 0:
            raise ValueError(f"{flag_name} values must be > 0, got {parsed}")
        if parsed not in seen:
            seen.add(parsed)
            values.append(parsed)
    if not values:
        raise ValueError(f"{flag_name} is empty")
    return values


def _extract_report_path_from_log(lines):
    for line in reversed(lines):
        match = re.search(r"\[OK\]\s+report=(.*)", line)
        if match:
            return match.group(1).strip()
    return None


def _load_sweep_child_summary(report_path):
    if not report_path:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {"report_path": str(path), "report_missing": True}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"report_path": str(path), "report_error": str(exc)}
    timing_items = report.get("average_time_summary") or []
    best_trimmed = None
    best_average = None
    ok = 0
    failed = 0
    for item in timing_items:
        if item.get("status") == "ok":
            ok += 1
            trimmed = item.get("trimmed_avg_ms")
            average = item.get("average_time_ms")
            if trimmed is not None:
                best_trimmed = (
                    float(trimmed)
                    if best_trimmed is None
                    else min(best_trimmed, float(trimmed))
                )
            if average is not None:
                best_average = (
                    float(average)
                    if best_average is None
                    else min(best_average, float(average))
                )
        else:
            failed += 1
    summary = report.get("summary") or {}
    return {
        "report_path": str(path),
        "ok": summary.get("ok", ok),
        "failed": summary.get("failed", failed),
        "best_trimmed_avg_ms": best_trimmed,
        "best_average_time_ms": best_average,
        "latency_summary": timing_items,
    }


def _maybe_run_same_shape_dot_batch_sweep(argv):
    sweep_arg = _get_cli_arg(argv, "--musa_xla_same_shape_dot_batch_sweep")
    if sweep_arg is None:
        return False

    base_cases = _sweep_cases_from_arg(sweep_arg)
    max_group_sizes = _positive_int_sweep_values_from_arg(
        _get_cli_arg(
            argv, "--musa_xla_same_shape_dot_batch_max_group_size_sweep"
        ),
        "--musa_xla_same_shape_dot_batch_max_group_size_sweep",
    )
    max_slice_bytes_values = _positive_int_sweep_values_from_arg(
        _get_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch_sweep",
        ),
        "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch_sweep",
    )
    cases = []
    for case in base_cases:
        group_sizes_for_case = [None] if case["batcher"] == "off" else max_group_sizes
        slice_bytes_for_case = [None] if case["batcher"] == "off" else max_slice_bytes_values
        for max_group_size in group_sizes_for_case:
            for max_slice_bytes in slice_bytes_for_case:
                expanded = dict(case)
                expanded["max_group_size"] = max_group_size
                expanded["max_slice_bytes_per_saved_launch"] = max_slice_bytes
                label_parts = [case["label"]]
                if max_group_size is not None:
                    label_parts.append(f"g{max_group_size}")
                if max_slice_bytes is not None:
                    label_parts.append(f"s{max_slice_bytes}")
                expanded["label"] = "_".join(label_parts)
                cases.append(expanded)
    out_root = Path(_get_cli_arg(argv, "--out_root", "runner_out")).resolve()
    sweep_root = out_root / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    strip_options = {
        "--musa_xla_same_shape_dot_batch_sweep": "value",
        "--musa_xla_same_shape_dot_batch_max_group_size_sweep": "value",
        "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch_sweep": "value",
        "--musa_xla_same_shape_dot_batcher": "value",
        "--musa_xla_same_shape_dot_batch_max_output_cols": "value",
        "--musa_xla_same_shape_dot_batch_max_group_size": "value",
        "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch": "value",
        "--musa_xla_same_shape_dot_batcher_log": "optional",
        "--musa_hlo_pattern_analysis": "optional",
        "--musa_hlo_pattern_analysis_verbose": "optional",
        "--musa_xla_thunk_diagnostics": "optional",
        "--musa_xla_gpu_runtime": "value",
        "--out_root": "value",
        "--xla_dump_dir": "value",
    }
    base_argv = _remove_cli_options(list(argv), strip_options)
    has_xla_dump = _has_cli_flag(argv, "--xla_dump") or _has_cli_flag(
        argv, "--xla_dump_dir"
    )

    print(
        f"[INFO] same-shape dot batch sweep start: cases="
        f"{[case['label'] for case in cases]} root={sweep_root}",
        flush=True,
    )
    results = []
    for index, case in enumerate(cases, start=1):
        case_root = sweep_root / f"{index:02d}_{case['label']}"
        child_out_root = case_root / "runner_out"
        child_log = case_root / "run.log"
        child_argv = list(base_argv)
        _append_cli_option(child_argv, "--out_root", child_out_root)
        _append_cli_option(
            child_argv, "--musa_xla_same_shape_dot_batcher", case["batcher"]
        )
        if case["batcher"] != "off":
            _append_cli_option(child_argv, "--musa_xla_same_shape_dot_batch_min_candidate_dots", "1")
        if case["max_output_cols"] is not None:
            _append_cli_option(
                child_argv,
                "--musa_xla_same_shape_dot_batch_max_output_cols",
                case["max_output_cols"],
            )
        if case.get("max_group_size") is not None:
            _append_cli_option(
                child_argv,
                "--musa_xla_same_shape_dot_batch_max_group_size",
                case["max_group_size"],
            )
        if case.get("max_slice_bytes_per_saved_launch") is not None:
            _append_cli_option(
                child_argv,
                "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch",
                case["max_slice_bytes_per_saved_launch"],
            )
        _append_cli_option(child_argv, "--musa_xla_same_shape_dot_batcher_log", "on")
        _append_cli_option(child_argv, "--musa_hlo_pattern_analysis", "on")
        _append_cli_option(child_argv, "--musa_hlo_pattern_analysis_verbose", "on")
        _append_cli_option(child_argv, "--musa_xla_thunk_diagnostics", "on")
        _append_cli_option(child_argv, "--musa_xla_gpu_runtime", "classic_thunks")
        if has_xla_dump:
            _append_cli_option(child_argv, "--xla_dump_dir", case_root / "xla_dump")

        cmd = [sys.executable, str(Path(__file__).resolve())] + [
            str(item) for item in child_argv
        ]
        case_root.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] sweep case start: label={case['label']} "
            f"batcher={case['batcher']} max_output_cols={case['max_output_cols']} "
            f"max_group_size={case.get('max_group_size')} "
            f"max_slice_bytes_per_saved_launch="
            f"{case.get('max_slice_bytes_per_saved_launch')} "
            f"log={child_log}",
            flush=True,
        )
        lines = []
        with child_log.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=dict(os.environ),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                log_file.write(line + "\n")
                print(f"[SWEEP {case['label']}] {line}", flush=True)
            returncode = proc.wait()

        report_path = _extract_report_path_from_log(lines)
        summary = _load_sweep_child_summary(report_path)
        key_lines = [
            line
            for line in lines
            if (
                "MUSA_SAME_SHAPE_DOT_BATCHER" in line
                or "MUSA_HLO_PATTERN_ANALYSIS" in line
                or "MUSA_XLA_THUNK_DIAGNOSTICS" in line
                or "latency_summary" in line
                or "run done" in line
            )
        ]
        result = {
            "case": case,
            "returncode": returncode,
            "log": str(child_log),
            "report": summary,
            "key_lines": key_lines[-12:],
        }
        results.append(result)
        print(
            f"[INFO] sweep case done: label={case['label']} rc={returncode} "
            f"best_trimmed_ms={summary.get('best_trimmed_avg_ms')} "
            f"best_avg_ms={summary.get('best_average_time_ms')}",
            flush=True,
        )

    final_report = {
        "argv": argv,
        "sweep_root": str(sweep_root),
        "cases": results,
    }
    report_path = sweep_root / "sweep_report.json"
    report_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] sweep_report={report_path}", flush=True)
    for result in sorted(
        results,
        key=lambda item: (
            item["report"].get("best_trimmed_avg_ms")
            if item["report"].get("best_trimmed_avg_ms") is not None
            else float("inf")
        ),
    ):
        report = result["report"]
        print(
            f"[OK] sweep_result label={result['case']['label']} "
            f"rc={result['returncode']} ok={report.get('ok')} "
            f"failed={report.get('failed')} "
            f"trimmed_ms={report.get('best_trimmed_avg_ms')} "
            f"avg_ms={report.get('best_average_time_ms')} "
            f"log={result['log']}",
            flush=True,
        )
    return True


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
    "compact_slice_feed": ("--compact_slice_feed", "on"),    "pack_slice_feed": ("--pack_slice_feed", "on"),
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


def select_optimization_profile(mode, specs):
    mode = str(mode or "auto").strip().lower()
    if mode == "off":
        return "off"
    spec_names = [Path(spec).name for spec in specs or []]
    if mode in ("meta1", "meta2"):
        expected = "meta_graph_1.spec" if mode == "meta1" else "meta_graph_2.spec"
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
    return "off"


def apply_optimization_profile(args, specs, explicit_flags=None):
    selected = select_optimization_profile(
        getattr(args, "optimization_profile", "auto"),
        specs,
    )
    profile = {
        "meta1": META1_DEFAULT_PROFILE,
        "meta2": META2_DEFAULT_PROFILE,
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
        vendor = str(device.get("vendor", "")).lower()
        if vendor == "0x1ed5":
            vendor_candidates.add(pci_id)
        class_code = str(device.get("class_code", "")).lower()
        if class_code.startswith("0x03"):
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
    }.get(selected, {})
    for _, (flag, value) in profile.items():
        if not _has_cli_flag(effective, flag):
            _append_cli_option(effective, flag, str(value))
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
        preserved_diagnostic_env = {
            name: os.environ.get(name)
            for name in (
                "MUSA_BLAS_GEMM_DIAGNOSTICS",
                "MUSA_F32_FAST_TF32",
                "MUSA_F32_FAST_TF32_SHAPES",
                "MUSA_HLO_PATTERN_ANALYSIS",
                "MUSA_HLO_PATTERN_ANALYSIS_VERBOSE",
                "MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY",
                "MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG",
                "MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG",
                "MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG",
                "MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_NAMES",
                "MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG",
                "MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL",
                "MUSA_XLA_REDUCTION_CHAIN_DIAG",
                "MUSA_XLA_REDUCTION_CHAIN_REWRITE",
                "MUSA_XLA_REDUCTION_CHAIN_KERNEL",
                "MUSA_XLA_WARP_ROW_REDUCTION_KERNEL",
                "MUSA_XLA_WARP_ROW_REDUCTION_REDUCERS",
                "MUSA_XLA_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS",
                "MUSA_XLA_WARP_ROW_REDUCTION_THREADS_PER_BLOCK",
                "MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL",
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL",
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS",
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_MAX",
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_THREADS_PER_BLOCK",
                "MUSA_XLA_DIRECT_MT_POW",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG",
                "MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS",
                "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION",
                "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_GROUP_SIZE",
                "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_POSTORDER_DISTANCE",
                "MUSA_XLA_THUNK_DIAGNOSTICS",
                "MUSA_XLA_THUNK_TIMING",
                "MUSA_XLA_CLASSIC_THUNK_GRAPH",
                "MUSA_XLA_CLASSIC_THUNK_GRAPH_MAX_CACHE_ENTRIES",
                "MUSA_XLA_EXECUTION_PATH_VERBOSE",
                "MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS",
                "MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL",
                "MUSA_PJRT_MAX_INFLIGHT_COMPUTATIONS",
                "MUSA_PJRT_NUM_DEVICE_TO_HOST_STREAMS",
                "MUSA_PJRT_NUM_DEVICE_TO_DEVICE_STREAMS",
                "MUSA_XLA_DOT_MERGER_MAX_MIB",
                "MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX",
                "MUSA_XLA_AVOID_GEMM_BETA_CHAIN",
                "MUSA_XLA_DOT_EPILOGUE_PATTERN",
                "MUSA_XLA_DOT_EPILOGUE_PATTERN_LOG",
                "MUSA_XLA_DOT_EPILOGUE_FUSION",
                "MUSA_XLA_DOT_EPILOGUE_FUSION_LOG",
                "MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY",
                "MUSA_XLA_DOT_EPILOGUE_FUSION_KIND",
                "MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD",
                "MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH",
                "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE",
                "MUSA_XLA_DOT_EPILOGUE_MIN_M",
                "MUSA_XLA_DOT_EPILOGUE_MIN_K",
                "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN",
                "MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE",
                "MUSA_XLA_GEMM_EPILOGUE_FUSION",
                "MUSA_XLA_GEMM_EPILOGUE_FUSION_LOG",
                "MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS",
                "MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL",
                "MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA",
                "MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES",
                "MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT",
                "MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS",
                "MUSA_XLA_GEMM_BETA_CHAIN_MERGER",
                "MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH",
                "MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS",
                "MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K",
                "MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL",
                "MUSA_XLA_GEMM_BETA_CHAIN_MERGER_LOG",
                "MUSA_XLA_GEMM_BETA_CHAIN_LOG_EMPTY",
                "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER",
                "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCHER",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POST_DOT_DIAG",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_DIAG",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_REWRITE",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_EXTERNAL_DIAG",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_DIAG",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MAX_DEPTH",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_REWRITE",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_BIASADD",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POINTER_ARRAY_OUTPUT",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_DIAG",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MAX_K",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MIN_GROUP_SIZE",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_OUTPUT_COLS",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_POINTER_ARRAY_OUTPUT",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_LOOP_FUSION",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_KERNEL",
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_MAX_GROUP_SIZE",
                "MUSA_XLA_SAME_LHS_DOT_MERGER",
                "MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_GROUP_SIZE",
                "MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUPS",
                "MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUP_SIZE",
                "MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_TOTAL_COLS",
                "MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_CANDIDATE_DOTS",
                "MUSA_XLA_SAME_LHS_DOT_MERGER_NORMALIZE_OPERANDS",
                "MUSA_XLA_SAME_LHS_DOT_MERGER_LOG",
                "MUSA_XLA_SAME_RHS_DOT_MERGER",
                "MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_GROUP_SIZE",
                "MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUPS",
                "MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUP_SIZE",
                "MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_TOTAL_ROWS",
                "MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_CANDIDATE_DOTS",
                "MUSA_XLA_SAME_RHS_DOT_MERGER_LOG",
                "MUSA_XLA_GROUP_GEMM_THUNKS",
                "MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE",
                "MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE",
                "MUSA_XLA_GROUP_GEMM_THUNKS_LOG",
                "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG",
                "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_MAX_SEPARATORS",
                "MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS",
                "MUSA_XLA_SMALL_GEMM_ACCUM_LOG",
                "MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE",
                "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE",
                "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K",
                "MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL",
            )
        }
        os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = tf_adapter_path
        os.environ["MUSA_PJRT_PLUGIN_PATH"] = plugin_path
        os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{plugin_path}"

        # Clear experiment leftovers, then set the stable PJRT/NPD path.
        for name in (
            "MUSA_XLA_RESPECT_NPD_ENV",
            "MUSA_XLA_GLOBAL_JIT_LEVEL",
            "MUSA_NPD_COMPILATION_DEVICE",
            "MUSA_PJRT_MAX_INFLIGHT_COMPILES",
            "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS",
            "MUSA_PJRT_MAX_INFLIGHT_EXECUTES",
            "MUSA_PJRT_WAIT_TRANSFER_DONE",
            "MUSA_PJRT_WAIT_EXECUTE_DONE",
            "MUSA_PJRT_REUSE_HOST_BUFFERS",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_DIAGNOSTICS",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ASYNC",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PARALLEL_PACK",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_FAST_PACK",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_THREADS",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_MIN_BYTES",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_DIRTY_RANGES",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_POOL_ORDER_LAYOUT",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_TRUST_CONTENTS",
            "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS",
            "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS_TRUST_LIFETIME",
            "MUSA_PJRT_SOURCE_FEED_ARENA_BASE",
            "MUSA_PJRT_SOURCE_FEED_ARENA_BYTES",
            "MUSA_PJRT_SERIALIZE_EXECUTE",
            "MUSA_PJRT_MAX_INFLIGHT_COMPUTATIONS",
            "MUSA_PJRT_NUM_DEVICE_TO_HOST_STREAMS",
            "MUSA_PJRT_NUM_DEVICE_TO_DEVICE_STREAMS",
            "MUSA_PJRT_USE_CALLBACK_STREAM",
            "MUSA_PJRT_DROP_EXECUTE_DEVICE",
            "MUSA_PJRT_FORCE_HOST_BUFFER_COPY",
            "MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT",
            "MUSA_XLA_MAX_FUSION_OPERANDS",
            "MUSA_MUDNN_INTERLEAVED_BATCH_GEMM",
            "MUSA_GEMM_BACKEND",
            "MUSA_F32_FAST_TF32",
            "MUSA_F32_FAST_TF32_SHAPES",
            "MUSA_GEMM_SMALLK_MIN_MAJOR",
            "MUSA_GEMM_SMALLK_MAX_MINOR",
            "MUSA_GEMM_SMALLK_MAX_K",
            "MUSA_GEMM_AUTO_SMALLK_MIN_MAJOR",
            "MUSA_GEMM_AUTO_SMALLK_MAX_MINOR",
            "MUSA_GEMM_AUTO_SMALLK_MAX_K",
            "MUSA_GEMM_AUTO_SKINNY_MIN_MAJOR",
            "MUSA_GEMM_AUTO_SKINNY_MAX_MINOR",
            "MUSA_GEMM_AUTO_SKINNY_MIN_K",
            "MUSA_GEMM_AUTO_SKINNY_MAX_K",
            "MUSA_STRIDED_BATCHED_GEMM_BACKEND",
            "MUSA_BLAS_GEMM_DIAGNOSTICS",
            "MUSA_HLO_PATTERN_ANALYSIS",
            "MUSA_HLO_PATTERN_ANALYSIS_VERBOSE",
            "MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY",
            "MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG",
            "MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL",
            "MUSA_XLA_WARP_ROW_REDUCTION_KERNEL",
            "MUSA_XLA_WARP_ROW_REDUCTION_REDUCERS",
            "MUSA_XLA_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS",
            "MUSA_XLA_WARP_ROW_REDUCTION_THREADS_PER_BLOCK",
            "MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL",
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL",
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS",
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_MAX",
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_THREADS_PER_BLOCK",
            "MUSA_XLA_DIRECT_MT_POW",
            "MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS",
            "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION",
            "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_GROUP_SIZE",
            "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_POSTORDER_DISTANCE",
            "MUSA_XLA_THUNK_DIAGNOSTICS",
            "MUSA_XLA_THUNK_TIMING",
            "MUSA_XLA_CLASSIC_THUNK_GRAPH",
            "MUSA_XLA_CLASSIC_THUNK_GRAPH_MAX_CACHE_ENTRIES",
            "MUSA_XLA_EXECUTION_PATH_VERBOSE",
            "MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS",
            "MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL",
            "MUSA_XLA_DOT_MERGER_MAX_MIB",
            "MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX",
            "MUSA_XLA_AVOID_GEMM_BETA_CHAIN",
            "MUSA_XLA_DOT_EPILOGUE_PATTERN",
            "MUSA_XLA_DOT_EPILOGUE_PATTERN_LOG",
            "MUSA_XLA_DOT_EPILOGUE_FUSION",
            "MUSA_XLA_DOT_EPILOGUE_FUSION_LOG",
            "MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY",
            "MUSA_XLA_DOT_EPILOGUE_FUSION_KIND",
            "MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD",
            "MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH",
            "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE",
            "MUSA_XLA_DOT_EPILOGUE_MIN_M",
            "MUSA_XLA_DOT_EPILOGUE_MIN_K",
            "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN",
            "MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE",
            "MUSA_XLA_GEMM_EPILOGUE_FUSION",
            "MUSA_XLA_GEMM_EPILOGUE_FUSION_LOG",
            "MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS",
            "MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL",
            "MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA",
            "MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES",
            "MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT",
            "MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS",
            "MUSA_XLA_GEMM_BETA_CHAIN_MERGER",
            "MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH",
            "MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS",
            "MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K",
            "MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL",
            "MUSA_XLA_GEMM_BETA_CHAIN_MERGER_LOG",
            "MUSA_XLA_GEMM_BETA_CHAIN_LOG_EMPTY",
            "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER",
            "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCHER",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POST_DOT_DIAG",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_DIAG",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_REWRITE",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_EXTERNAL_DIAG",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_DIAG",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MAX_DEPTH",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_REWRITE",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_BIASADD",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POINTER_ARRAY_OUTPUT",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_DIAG",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MAX_K",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MIN_GROUP_SIZE",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_OUTPUT_COLS",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_POINTER_ARRAY_OUTPUT",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_LOOP_FUSION",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_KERNEL",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_MAX_GROUP_SIZE",
            "MUSA_XLA_SAME_LHS_DOT_MERGER",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_GROUP_SIZE",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUPS",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUP_SIZE",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_TOTAL_COLS",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_CANDIDATE_DOTS",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_NORMALIZE_OPERANDS",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_LOG",
            "MUSA_XLA_SAME_RHS_DOT_MERGER",
            "MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_GROUP_SIZE",
            "MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUPS",
            "MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUP_SIZE",
            "MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_TOTAL_ROWS",
            "MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_CANDIDATE_DOTS",
            "MUSA_XLA_SAME_RHS_DOT_MERGER_LOG",
            "MUSA_XLA_GROUP_GEMM_THUNKS",
            "MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE",
            "MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE",
            "MUSA_XLA_GROUP_GEMM_THUNKS_LOG",
            "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG",
            "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_MAX_SEPARATORS",
            "MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS",
            "MUSA_XLA_SMALL_GEMM_ACCUM_LOG",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K",
            "MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL",
        ):
            os.environ.pop(name, None)
        os.environ["MUSA_NPD_IS_PLUGGABLE_DEVICE"] = "1"
        os.environ["MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE"] = "1"
        force_host_copy = _get_cli_arg(argv, "--pjrt_force_host_copy", "off")
        if force_host_copy in ("on", "true", "1"):
            os.environ["MUSA_PJRT_FORCE_HOST_BUFFER_COPY"] = "1"
        elif force_host_copy in ("off", "false", "0"):
            os.environ["MUSA_PJRT_FORCE_HOST_BUFFER_COPY"] = "0"

        max_inflight_transfers = _get_cli_arg(
            argv, "--pjrt_max_inflight_transfers", "0"
        )
        max_inflight_executes = _get_cli_arg(
            argv, "--pjrt_max_inflight_executes", "0"
        )
        if max_inflight_transfers != "auto":
            os.environ["MUSA_PJRT_MAX_INFLIGHT_TRANSFERS"] = max_inflight_transfers
        if max_inflight_executes != "auto":
            os.environ["MUSA_PJRT_MAX_INFLIGHT_EXECUTES"] = max_inflight_executes
        for name in (
            "MUSA_PJRT_MAX_INFLIGHT_COMPUTATIONS",
            "MUSA_PJRT_NUM_DEVICE_TO_HOST_STREAMS",
            "MUSA_PJRT_NUM_DEVICE_TO_DEVICE_STREAMS",
            "MUSA_F32_FAST_TF32",
            "MUSA_F32_FAST_TF32_SHAPES",
            "MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS",
            "MUSA_XLA_SMALL_GEMM_ACCUM_LOG",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K",
            "MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL",
        ):
            if preserved_diagnostic_env.get(name) is not None:
                os.environ[name] = preserved_diagnostic_env[name]
        wait_transfer_done = _get_cli_arg(argv, "--pjrt_wait_transfer_done", "off")
        if wait_transfer_done in ("on", "true", "1"):
            os.environ["MUSA_PJRT_WAIT_TRANSFER_DONE"] = "1"
        elif wait_transfer_done in ("off", "false", "0"):
            os.environ["MUSA_PJRT_WAIT_TRANSFER_DONE"] = "0"
        _set_bool_env_from_cli(
            argv,
            "--pjrt_wait_execute_done",
            "MUSA_PJRT_WAIT_EXECUTE_DONE",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers",
            "MUSA_PJRT_REUSE_HOST_BUFFERS",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers_diagnostics",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_DIAGNOSTICS",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers_async",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ASYNC",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers_arena",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers_arena_parallel_pack",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PARALLEL_PACK",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers_arena_fast_pack",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_FAST_PACK",
            "off",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--pjrt_reuse_host_buffers_arena_pack_threads",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_THREADS",
            4,
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--pjrt_reuse_host_buffers_arena_pack_min_bytes",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_MIN_BYTES",
            1048576,
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers_arena_dirty_ranges",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_DIRTY_RANGES",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers_arena_pool_order_layout",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_POOL_ORDER_LAYOUT",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_reuse_host_buffers_trust_contents",
            "MUSA_PJRT_REUSE_HOST_BUFFERS_TRUST_CONTENTS",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_cache_reused_buffer_views",
            "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_cache_reused_buffer_views_trust_lifetime",
            "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS_TRUST_LIFETIME",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_bypass_event_destroy",
            "MUSA_PJRT_BYPASS_EVENT_DESTROY",
            "off",
        )
        _set_bool_env_from_cli(
            argv,
            "--pjrt_bypass_buffer_destroy",
            "MUSA_PJRT_BYPASS_BUFFER_DESTROY",
            "off",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_custom_fusion",
            "MUSA_CUSTOM_FUSION",
            "off",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--mudnn_interleaved_batch_gemm",
            "MUSA_MUDNN_INTERLEAVED_BATCH_GEMM",
            "off",
        )
        gemm_backend = _get_cli_arg(argv, "--gemm_backend", "auto")
        if gemm_backend in ("mudnn", "mublas", "smallk_mublas", "auto_mublas"):
            os.environ["MUSA_GEMM_BACKEND"] = gemm_backend
        _set_tristate_bool_env_from_cli_or_preserve(
            argv,
            "--musa_f32_fast_tf32",
            "MUSA_F32_FAST_TF32",
            default="off",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_f32_fast_tf32_shapes",
            "MUSA_F32_FAST_TF32_SHAPES",
            default="",
        )
        if (
            os.environ.get("MUSA_F32_FAST_TF32_SHAPES", "")
            and not _has_cli_flag(argv, "--musa_f32_fast_tf32")
        ):
            os.environ["MUSA_F32_FAST_TF32"] = "1"
        smallk_min_major = _get_cli_arg(argv, "--gemm_smallk_min_major", "512")
        smallk_max_minor = _get_cli_arg(argv, "--gemm_smallk_max_minor", "512")
        smallk_max_k = _get_cli_arg(argv, "--gemm_smallk_max_k", "32")
        if smallk_min_major:
            os.environ["MUSA_GEMM_SMALLK_MIN_MAJOR"] = smallk_min_major
        if smallk_max_minor:
            os.environ["MUSA_GEMM_SMALLK_MAX_MINOR"] = smallk_max_minor
        if smallk_max_k:
            os.environ["MUSA_GEMM_SMALLK_MAX_K"] = smallk_max_k
        auto_smallk_min_major = _get_cli_arg(
            argv, "--gemm_auto_smallk_min_major", "1024"
        )
        auto_smallk_max_minor = _get_cli_arg(
            argv, "--gemm_auto_smallk_max_minor", "128"
        )
        auto_smallk_max_k = _get_cli_arg(argv, "--gemm_auto_smallk_max_k", "16")
        if auto_smallk_min_major:
            os.environ["MUSA_GEMM_AUTO_SMALLK_MIN_MAJOR"] = (
                auto_smallk_min_major
            )
        if auto_smallk_max_minor:
            os.environ["MUSA_GEMM_AUTO_SMALLK_MAX_MINOR"] = (
                auto_smallk_max_minor
            )
        if auto_smallk_max_k:
            os.environ["MUSA_GEMM_AUTO_SMALLK_MAX_K"] = auto_smallk_max_k
        auto_skinny_min_major = _get_cli_arg(
            argv, "--gemm_auto_skinny_min_major", "1024"
        )
        auto_skinny_max_minor = _get_cli_arg(
            argv, "--gemm_auto_skinny_max_minor", "128"
        )
        auto_skinny_min_k = _get_cli_arg(argv, "--gemm_auto_skinny_min_k", "512")
        auto_skinny_max_k = _get_cli_arg(argv, "--gemm_auto_skinny_max_k", "4096")
        if auto_skinny_min_major:
            os.environ["MUSA_GEMM_AUTO_SKINNY_MIN_MAJOR"] = (
                auto_skinny_min_major
            )
        if auto_skinny_max_minor:
            os.environ["MUSA_GEMM_AUTO_SKINNY_MAX_MINOR"] = (
                auto_skinny_max_minor
            )
        if auto_skinny_min_k:
            os.environ["MUSA_GEMM_AUTO_SKINNY_MIN_K"] = auto_skinny_min_k
        if auto_skinny_max_k:
            os.environ["MUSA_GEMM_AUTO_SKINNY_MAX_K"] = auto_skinny_max_k
        strided_batched_backend = _get_cli_arg(
            argv, "--strided_batched_gemm_backend", "auto"
        )
        if strided_batched_backend in ("mudnn", "mublas", "auto_mublas"):
            os.environ["MUSA_STRIDED_BATCHED_GEMM_BACKEND"] = (
                strided_batched_backend
            )
        if _has_cli_flag(argv, "--musa_blas_gemm_diagnostics"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_blas_gemm_diagnostics",
                "MUSA_BLAS_GEMM_DIAGNOSTICS",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_BLAS_GEMM_DIAGNOSTICS"] is not None:
            os.environ["MUSA_BLAS_GEMM_DIAGNOSTICS"] = preserved_diagnostic_env[
                "MUSA_BLAS_GEMM_DIAGNOSTICS"
            ]
        else:
            os.environ["MUSA_BLAS_GEMM_DIAGNOSTICS"] = "0"

        if _has_cli_flag(argv, "--musa_hlo_pattern_analysis"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_hlo_pattern_analysis",
                "MUSA_HLO_PATTERN_ANALYSIS",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_HLO_PATTERN_ANALYSIS"] is not None:
            os.environ["MUSA_HLO_PATTERN_ANALYSIS"] = preserved_diagnostic_env[
                "MUSA_HLO_PATTERN_ANALYSIS"
            ]
        else:
            os.environ["MUSA_HLO_PATTERN_ANALYSIS"] = "0"
        if _has_cli_flag(argv, "--musa_hlo_pattern_analysis_verbose"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_hlo_pattern_analysis_verbose",
                "MUSA_HLO_PATTERN_ANALYSIS_VERBOSE",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_HLO_PATTERN_ANALYSIS_VERBOSE"] is not None:
            os.environ["MUSA_HLO_PATTERN_ANALYSIS_VERBOSE"] = (
                preserved_diagnostic_env["MUSA_HLO_PATTERN_ANALYSIS_VERBOSE"]
            )
        else:
            os.environ["MUSA_HLO_PATTERN_ANALYSIS_VERBOSE"] = "0"
        if _has_cli_flag(argv, "--musa_hlo_pattern_analysis_log_empty"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_hlo_pattern_analysis_log_empty",
                "MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY"] is not None:
            os.environ["MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY"] = (
                preserved_diagnostic_env["MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY"]
            )
        else:
            os.environ["MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY"] = "0"
        if _has_cli_flag(argv, "--musa_xla_thunk_diagnostics"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_thunk_diagnostics",
                "MUSA_XLA_THUNK_DIAGNOSTICS",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_XLA_THUNK_DIAGNOSTICS"] is not None:
            os.environ["MUSA_XLA_THUNK_DIAGNOSTICS"] = preserved_diagnostic_env[
                "MUSA_XLA_THUNK_DIAGNOSTICS"
            ]
        else:
            os.environ["MUSA_XLA_THUNK_DIAGNOSTICS"] = "0"
        if _has_cli_flag(argv, "--musa_xla_thunk_timing"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_thunk_timing",
                "MUSA_XLA_THUNK_TIMING",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_XLA_THUNK_TIMING"] is not None:
            os.environ["MUSA_XLA_THUNK_TIMING"] = preserved_diagnostic_env[
                "MUSA_XLA_THUNK_TIMING"
            ]
        else:
            os.environ["MUSA_XLA_THUNK_TIMING"] = "0"
        if _has_cli_flag(argv, "--musa_xla_classic_thunk_graph"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_classic_thunk_graph",
                "MUSA_XLA_CLASSIC_THUNK_GRAPH",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_XLA_CLASSIC_THUNK_GRAPH"] is not None:
            os.environ["MUSA_XLA_CLASSIC_THUNK_GRAPH"] = preserved_diagnostic_env[
                "MUSA_XLA_CLASSIC_THUNK_GRAPH"
            ]
        else:
            os.environ["MUSA_XLA_CLASSIC_THUNK_GRAPH"] = "0"
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_classic_thunk_graph_max_cache_entries",
            "MUSA_XLA_CLASSIC_THUNK_GRAPH_MAX_CACHE_ENTRIES",
            preserved_diagnostic_env[
                "MUSA_XLA_CLASSIC_THUNK_GRAPH_MAX_CACHE_ENTRIES"
            ]
            or "4",
        )
        if _has_cli_flag(argv, "--musa_xla_horizontal_fusion_diagnostics"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_horizontal_fusion_diagnostics",
                "MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS"
                ]
            )
        else:
            os.environ["MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS"] = "0"
        if _has_cli_flag(argv, "--musa_xla_cross_consumer_horizontal_fusion"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_cross_consumer_horizontal_fusion",
                "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION"
                ]
            )
        else:
            os.environ["MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION"] = "0"
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_cross_consumer_horizontal_fusion_max_group_size",
            "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_GROUP_SIZE",
            preserved_diagnostic_env[
                "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_GROUP_SIZE"
            ]
            or "4",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_cross_consumer_horizontal_fusion_max_postorder_distance",
            "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_POSTORDER_DISTANCE",
            preserved_diagnostic_env[
                "MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_POSTORDER_DISTANCE"
            ]
            or "64",
        )
        if _has_cli_flag(argv, "--musa_xla_hot_fusion_softmax_diag"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_hot_fusion_softmax_diag",
                "MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG"]
            is not None
        ):
            os.environ["MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG"] = (
                preserved_diagnostic_env["MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG"]
            )
        else:
            os.environ["MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG"] = "0"
        if _has_cli_flag(argv, "--musa_xla_hot_fusion_softmax_detail_diag"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_hot_fusion_softmax_detail_diag",
                "MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG"]
            is not None
        ):
            os.environ["MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG"
                ]
            )
        else:
            os.environ["MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG"] = "0"
        if _has_cli_flag(argv, "--musa_xla_hot_fusion_softmax_body_diag"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_hot_fusion_softmax_body_diag",
                "MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG"]
            is not None
        ):
            os.environ["MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG"
                ]
            )
        else:
            os.environ["MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG"] = "0"
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_hot_fusion_softmax_body_names",
            "MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_NAMES",
            preserved_diagnostic_env["MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_NAMES"]
            or "",
        )
        if _has_cli_flag(argv, "--musa_xla_hot_tuple_softmax_match_diag"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_hot_tuple_softmax_match_diag",
                "MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG"]
            is not None
        ):
            os.environ["MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG"
                ]
            )
        else:
            os.environ["MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG"] = "0"
        if _has_cli_flag(argv, "--musa_xla_hot_tuple_softmax_kernel"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_hot_tuple_softmax_kernel",
                "MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL"]
            is not None
        ):
            os.environ["MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL"
                ]
            )
        else:
            os.environ["MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL"] = "0"
        if _has_cli_flag(argv, "--musa_xla_reduction_chain_diag"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_reduction_chain_diag",
                "MUSA_XLA_REDUCTION_CHAIN_DIAG",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_REDUCTION_CHAIN_DIAG"]
            is not None
        ):
            os.environ["MUSA_XLA_REDUCTION_CHAIN_DIAG"] = (
                preserved_diagnostic_env["MUSA_XLA_REDUCTION_CHAIN_DIAG"]
            )
        else:
            os.environ["MUSA_XLA_REDUCTION_CHAIN_DIAG"] = "0"
        if _has_cli_flag(argv, "--musa_xla_reduction_chain_rewrite"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_reduction_chain_rewrite",
                "MUSA_XLA_REDUCTION_CHAIN_REWRITE",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_REDUCTION_CHAIN_REWRITE"]
            is not None
        ):
            os.environ["MUSA_XLA_REDUCTION_CHAIN_REWRITE"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_REDUCTION_CHAIN_REWRITE"
                ]
            )
        else:
            os.environ["MUSA_XLA_REDUCTION_CHAIN_REWRITE"] = "0"
        if _has_cli_flag(argv, "--musa_xla_reduction_chain_kernel"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_reduction_chain_kernel",
                "MUSA_XLA_REDUCTION_CHAIN_KERNEL",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_REDUCTION_CHAIN_KERNEL"]
            is not None
        ):
            os.environ["MUSA_XLA_REDUCTION_CHAIN_KERNEL"] = (
                preserved_diagnostic_env["MUSA_XLA_REDUCTION_CHAIN_KERNEL"]
            )
        else:
            os.environ["MUSA_XLA_REDUCTION_CHAIN_KERNEL"] = "0"
        if os.environ["MUSA_XLA_REDUCTION_CHAIN_KERNEL"] == "1":
            os.environ["MUSA_XLA_REDUCTION_CHAIN_REWRITE"] = "1"
        if _has_cli_flag(argv, "--musa_xla_warp_row_reduction_kernel"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_warp_row_reduction_kernel",
                "MUSA_XLA_WARP_ROW_REDUCTION_KERNEL",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_WARP_ROW_REDUCTION_KERNEL"]
            is not None
        ):
            os.environ["MUSA_XLA_WARP_ROW_REDUCTION_KERNEL"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_WARP_ROW_REDUCTION_KERNEL"
                ]
            )
        else:
            os.environ["MUSA_XLA_WARP_ROW_REDUCTION_KERNEL"] = "0"
        if _has_cli_flag(
            argv, "--musa_xla_tuple_warp_row_reduction_kernel"
        ):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_tuple_warp_row_reduction_kernel",
                "MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL"
                ]
            )
        else:
            os.environ["MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL"] = "0"
        if _has_cli_flag(
            argv, "--musa_xla_mixed_tuple_warp_row_reduction_kernel"
        ):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_mixed_tuple_warp_row_reduction_kernel",
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL"
                ]
            )
        else:
            os.environ["MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL"] = "0"
        if _has_cli_flag(argv, "--musa_xla_direct_mt_pow"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_direct_mt_pow",
                "MUSA_XLA_DIRECT_MT_POW",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_XLA_DIRECT_MT_POW"] is not None:
            os.environ["MUSA_XLA_DIRECT_MT_POW"] = preserved_diagnostic_env[
                "MUSA_XLA_DIRECT_MT_POW"
            ]
        else:
            os.environ["MUSA_XLA_DIRECT_MT_POW"] = "0"
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_warp_row_reduction_reducers",
            "MUSA_XLA_WARP_ROW_REDUCTION_REDUCERS",
            preserved_diagnostic_env[
                "MUSA_XLA_WARP_ROW_REDUCTION_REDUCERS"
            ]
            or "all",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_warp_row_reduction_min_data_elements",
            "MUSA_XLA_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS",
            preserved_diagnostic_env[
                "MUSA_XLA_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS"
            ]
            or "0",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_mixed_tuple_warp_row_reduction_min_data_elements",
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS",
            preserved_diagnostic_env[
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS"
            ],
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_mixed_tuple_warp_row_reduction_small_width_max",
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_MAX",
            preserved_diagnostic_env[
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_MAX"
            ],
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_mixed_tuple_warp_row_reduction_small_width_threads_per_block",
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_THREADS_PER_BLOCK",
            preserved_diagnostic_env[
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_THREADS_PER_BLOCK"
            ],
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_warp_row_reduction_threads_per_block",
            "MUSA_XLA_WARP_ROW_REDUCTION_THREADS_PER_BLOCK",
            preserved_diagnostic_env[
                "MUSA_XLA_WARP_ROW_REDUCTION_THREADS_PER_BLOCK"
            ]
            or "0",
        )
        if _has_cli_flag(
            argv, "--musa_xla_fusion_merger_materialize_reduction_producer"
        ):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_fusion_merger_materialize_reduction_producer",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER"
            ]
            is not None
        ):
            os.environ[
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER"
            ] = preserved_diagnostic_env[
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER"
            ]
        else:
            os.environ[
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER"
            ] = "0"
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_fusion_merger_materialize_min_elements",
            "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS",
            preserved_diagnostic_env[
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS"
            ]
            or "10000000",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_fusion_merger_materialize_min_operands",
            "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS",
            preserved_diagnostic_env[
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS"
            ]
            or "16",
        )
        if _has_cli_flag(argv, "--musa_xla_fusion_merger_materialize_log"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_fusion_merger_materialize_log",
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG"
                ]
            )
        else:
            os.environ["MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG"] = "0"
        if _has_cli_flag(argv, "--musa_xla_execution_path_verbose"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_execution_path_verbose",
                "MUSA_XLA_EXECUTION_PATH_VERBOSE",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_EXECUTION_PATH_VERBOSE"]
            is not None
        ):
            os.environ["MUSA_XLA_EXECUTION_PATH_VERBOSE"] = (
                preserved_diagnostic_env["MUSA_XLA_EXECUTION_PATH_VERBOSE"]
            )
        else:
            os.environ["MUSA_XLA_EXECUTION_PATH_VERBOSE"] = "0"
        if _has_cli_flag(argv, "--musa_xla_gemm_runtime_diagnostics"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_gemm_runtime_diagnostics",
                "MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS"]
            is not None
        ):
            os.environ["MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS"] = (
                preserved_diagnostic_env["MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS"]
            )
        else:
            os.environ["MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS"] = "0"
        gemm_runtime_log_interval = _get_cli_arg(
            argv, "--musa_xla_gemm_runtime_log_interval", ""
        )
        if gemm_runtime_log_interval:
            os.environ["MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL"] = (
                gemm_runtime_log_interval
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL"]
            is not None
        ):
            os.environ["MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL"] = (
                preserved_diagnostic_env["MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL"]
            )
        os.environ.setdefault("MUSA_PINNED_H2D_ON_COMPUTE_STREAM", "1")
        avoid_interleaved_layout = _get_cli_arg(
            argv, "--avoid_interleaved_batch_gemm_layout", "auto"
        )
        if avoid_interleaved_layout in ("on", "true", "1"):
            os.environ["MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT"] = "1"
        elif avoid_interleaved_layout in ("off", "false", "0"):
            os.environ["MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT"] = "0"
        elif early_batch_size >= _early_large_batch_threshold():
            os.environ["MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT"] = "1"
        max_fusion_operands = _get_cli_arg(argv, "--xla_max_fusion_operands", "0")
        try:
            max_fusion_operands_value = int(max_fusion_operands)
        except ValueError:
            max_fusion_operands_value = 0
        if max_fusion_operands_value > 0:
            os.environ["MUSA_XLA_MAX_FUSION_OPERANDS"] = str(
                max_fusion_operands_value
            )
        dot_merger_max_mib = _get_cli_arg(argv, "--musa_xla_dot_merger_max_mib", "0")
        try:
            dot_merger_max_mib_value = int(dot_merger_max_mib)
        except ValueError:
            dot_merger_max_mib_value = 0
        if dot_merger_max_mib_value > 0:
            os.environ["MUSA_XLA_DOT_MERGER_MAX_MIB"] = str(
                dot_merger_max_mib_value
            )
        elif preserved_diagnostic_env["MUSA_XLA_DOT_MERGER_MAX_MIB"] is not None:
            os.environ["MUSA_XLA_DOT_MERGER_MAX_MIB"] = preserved_diagnostic_env[
                "MUSA_XLA_DOT_MERGER_MAX_MIB"
            ]
        if _has_cli_flag(argv, "--musa_xla_fuse_broadcast_bias_as_matrix"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_fuse_broadcast_bias_as_matrix",
                "MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX"]
            is not None
        ):
            os.environ["MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX"] = (
                preserved_diagnostic_env["MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX"]
            )
        else:
            os.environ["MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX"] = "0"
        if _has_cli_flag(argv, "--musa_xla_avoid_gemm_beta_chain"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_avoid_gemm_beta_chain",
                "MUSA_XLA_AVOID_GEMM_BETA_CHAIN",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_AVOID_GEMM_BETA_CHAIN"]
            is not None
        ):
            os.environ["MUSA_XLA_AVOID_GEMM_BETA_CHAIN"] = preserved_diagnostic_env[
                "MUSA_XLA_AVOID_GEMM_BETA_CHAIN"
            ]
        else:
            os.environ["MUSA_XLA_AVOID_GEMM_BETA_CHAIN"] = "0"
        if _has_cli_flag(argv, "--musa_xla_dot_epilogue_pattern"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_dot_epilogue_pattern",
                "MUSA_XLA_DOT_EPILOGUE_PATTERN",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_PATTERN"] is not None:
            os.environ["MUSA_XLA_DOT_EPILOGUE_PATTERN"] = preserved_diagnostic_env[
                "MUSA_XLA_DOT_EPILOGUE_PATTERN"
            ]
        else:
            os.environ["MUSA_XLA_DOT_EPILOGUE_PATTERN"] = "0"
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_dot_epilogue_pattern_log",
            "MUSA_XLA_DOT_EPILOGUE_PATTERN_LOG",
            "off",
        )
        if _has_cli_flag(argv, "--musa_xla_dot_epilogue_fusion"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_dot_epilogue_fusion",
                "MUSA_XLA_DOT_EPILOGUE_FUSION",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_FUSION"] is not None:
            os.environ["MUSA_XLA_DOT_EPILOGUE_FUSION"] = preserved_diagnostic_env[
                "MUSA_XLA_DOT_EPILOGUE_FUSION"
            ]
        else:
            os.environ["MUSA_XLA_DOT_EPILOGUE_FUSION"] = "0"
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_dot_epilogue_fusion_log",
            "MUSA_XLA_DOT_EPILOGUE_FUSION_LOG",
            "off",
        )
        if _has_cli_flag(argv, "--musa_xla_dot_epilogue_log_empty"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_dot_epilogue_log_empty",
                "MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY"] is not None:
            os.environ["MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY"] = preserved_diagnostic_env[
                "MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY"
            ]
        else:
            os.environ["MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY"] = "0"
        dot_epilogue_fusion_kind = _get_cli_arg(
            argv,
            "--musa_xla_dot_epilogue_fusion_kind",
            preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_FUSION_KIND"]
            or "__triton_gemm",
        )
        if dot_epilogue_fusion_kind:
            os.environ["MUSA_XLA_DOT_EPILOGUE_FUSION_KIND"] = (
                dot_epilogue_fusion_kind
            )
        if _has_cli_flag(argv, "--musa_xla_dot_epilogue_require_add"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_dot_epilogue_require_add",
                "MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD"]
            is not None
        ):
            os.environ["MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD"] = (
                preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD"]
            )
        else:
            os.environ["MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD"] = "0"
        dot_epilogue_max_chain_length = _get_cli_arg(
            argv,
            "--musa_xla_dot_epilogue_max_chain_length",
            preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH"]
            or "",
        )
        if dot_epilogue_max_chain_length:
            os.environ["MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH"] = (
                str(dot_epilogue_max_chain_length)
            )
        dot_epilogue_max_fusions = _get_cli_arg(
            argv,
            "--musa_xla_dot_epilogue_max_fusions_per_module",
            preserved_diagnostic_env[
                "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE"
            ]
            or "",
        )
        if dot_epilogue_max_fusions:
            os.environ["MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE"] = (
                str(dot_epilogue_max_fusions)
            )
        dot_epilogue_min_m = _get_cli_arg(
            argv,
            "--musa_xla_dot_epilogue_min_m",
            preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_MIN_M"] or "",
        )
        if dot_epilogue_min_m:
            os.environ["MUSA_XLA_DOT_EPILOGUE_MIN_M"] = str(dot_epilogue_min_m)
        dot_epilogue_min_k = _get_cli_arg(
            argv,
            "--musa_xla_dot_epilogue_min_k",
            preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_MIN_K"] or "",
        )
        if dot_epilogue_min_k:
            os.environ["MUSA_XLA_DOT_EPILOGUE_MIN_K"] = str(dot_epilogue_min_k)
        dot_epilogue_max_per_pattern = _get_cli_arg(
            argv,
            "--musa_xla_dot_epilogue_max_fusions_per_pattern",
            preserved_diagnostic_env[
                "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN"
            ]
            or "",
        )
        if dot_epilogue_max_per_pattern:
            os.environ["MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN"] = (
                str(dot_epilogue_max_per_pattern)
            )
        if _has_cli_flag(argv, "--musa_xla_dot_epilogue_sort_by_size"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_dot_epilogue_sort_by_size",
                "MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE"]
            is not None
        ):
            os.environ["MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE"] = (
                preserved_diagnostic_env["MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE"]
            )
        else:
            os.environ["MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE"] = "0"
        if _has_cli_flag(argv, "--musa_xla_gemm_epilogue_fusion"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_gemm_epilogue_fusion",
                "MUSA_XLA_GEMM_EPILOGUE_FUSION",
                "off",
            )
        elif preserved_diagnostic_env["MUSA_XLA_GEMM_EPILOGUE_FUSION"] is not None:
            os.environ["MUSA_XLA_GEMM_EPILOGUE_FUSION"] = preserved_diagnostic_env[
                "MUSA_XLA_GEMM_EPILOGUE_FUSION"
            ]
        else:
            os.environ["MUSA_XLA_GEMM_EPILOGUE_FUSION"] = "0"
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_gemm_epilogue_fusion_log",
            "MUSA_XLA_GEMM_EPILOGUE_FUSION_LOG",
            "off",
        )
        if _has_cli_flag(argv, "--musa_xla_gemm_epilogue_fuse_broadcast_bias"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_gemm_epilogue_fuse_broadcast_bias",
                "MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS"
                ]
            )
        else:
            os.environ["MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS"] = "0"
        if _has_cli_flag(argv, "--musa_xla_gemm_epilogue_custom_call"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_gemm_epilogue_custom_call",
                "MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL"]
            is not None
        ):
            os.environ["MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL"] = (
                preserved_diagnostic_env["MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL"]
            )
        else:
            os.environ["MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL"] = "0"
        if _has_cli_flag(
            argv, "--musa_xla_gemm_epilogue_force_broadcast_bias_beta"
        ):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_gemm_epilogue_force_broadcast_bias_beta",
                "MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA"
                ]
            )
        else:
            os.environ["MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA"] = "0"
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_gemm_epilogue_only_shapes",
            "MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES",
            preserved_diagnostic_env["MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES"] or "",
        )
        if _has_cli_flag(argv, "--musa_xla_gemm_epilogue_disable_mublaslt"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_gemm_epilogue_disable_mublaslt",
                "MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT",
                "off",
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT"
                ]
            )
        else:
            os.environ["MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT"] = "0"
        if _has_cli_flag(argv, "--musa_gemm_epilogue_thunk_diagnostics"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_gemm_epilogue_thunk_diagnostics",
                "MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS"]
            is not None
        ):
            os.environ["MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS"] = (
                preserved_diagnostic_env["MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS"]
            )
        else:
            os.environ["MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS"] = "0"
        if (
            os.environ.get("MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS") == "1"
            and not _has_cli_flag(argv, "--musa_xla_fuse_broadcast_bias_as_matrix")
        ):
            os.environ["MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX"] = "0"
        if _has_cli_flag(argv, "--musa_xla_gemm_beta_chain_merger"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_gemm_beta_chain_merger",
                "MUSA_XLA_GEMM_BETA_CHAIN_MERGER",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_GEMM_BETA_CHAIN_MERGER"]
            is not None
        ):
            os.environ["MUSA_XLA_GEMM_BETA_CHAIN_MERGER"] = (
                preserved_diagnostic_env["MUSA_XLA_GEMM_BETA_CHAIN_MERGER"]
            )
        else:
            os.environ["MUSA_XLA_GEMM_BETA_CHAIN_MERGER"] = "0"
        gemm_beta_chain_min_length = _get_cli_arg(
            argv, "--musa_xla_gemm_beta_chain_min_chain_length", "2"
        )
        if gemm_beta_chain_min_length:
            os.environ["MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH"] = (
                gemm_beta_chain_min_length
            )
        gemm_beta_chain_max_chains = _get_cli_arg(
            argv, "--musa_xla_gemm_beta_chain_max_chains", "128"
        )
        if gemm_beta_chain_max_chains:
            os.environ["MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS"] = (
                gemm_beta_chain_max_chains
            )
        gemm_beta_chain_max_total_k = _get_cli_arg(
            argv, "--musa_xla_gemm_beta_chain_max_total_k", "16"
        )
        if gemm_beta_chain_max_total_k:
            os.environ["MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K"] = (
                gemm_beta_chain_max_total_k
            )
        if _has_cli_flag(argv, "--musa_xla_gemm_beta_chain_custom_call"):
            _set_tristate_bool_env_from_cli(
                argv,
                "--musa_xla_gemm_beta_chain_custom_call",
                "MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL",
                "off",
            )
        elif (
            preserved_diagnostic_env["MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL"]
            is not None
        ):
            os.environ["MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL"] = (
                preserved_diagnostic_env["MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL"]
            )
        else:
            os.environ["MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL"] = "0"
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_gemm_beta_chain_merger_log",
            "MUSA_XLA_GEMM_BETA_CHAIN_MERGER_LOG",
        )
        post_transpose_dot_merger = _get_cli_arg(
            argv, "--musa_xla_post_transpose_dot_merger", "auto"
        )
        if post_transpose_dot_merger in ("on", "true", "1"):
            os.environ["MUSA_XLA_POST_TRANSPOSE_DOT_MERGER"] = "1"
        elif post_transpose_dot_merger in ("off", "false", "0"):
            os.environ["MUSA_XLA_POST_TRANSPOSE_DOT_MERGER"] = "0"
        elif post_transpose_dot_merger == "auto":
            os.environ["MUSA_XLA_POST_TRANSPOSE_DOT_MERGER"] = "auto"
        elif (
            preserved_diagnostic_env["MUSA_XLA_POST_TRANSPOSE_DOT_MERGER"]
            is not None
        ):
            os.environ["MUSA_XLA_POST_TRANSPOSE_DOT_MERGER"] = (
                preserved_diagnostic_env["MUSA_XLA_POST_TRANSPOSE_DOT_MERGER"]
            )
        post_transpose_dot_merger_max_mib = _get_cli_arg(
            argv, "--musa_xla_post_transpose_dot_merger_max_mib", "0"
        )
        try:
            post_transpose_dot_merger_max_mib_value = int(
                post_transpose_dot_merger_max_mib
            )
        except ValueError:
            post_transpose_dot_merger_max_mib_value = 0
        if post_transpose_dot_merger_max_mib_value > 0:
            os.environ["MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB"] = str(
                post_transpose_dot_merger_max_mib_value
            )
        elif (
            preserved_diagnostic_env[
                "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB"
            ]
            is not None
        ):
            os.environ["MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB"] = (
                preserved_diagnostic_env[
                    "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB"
                ]
            )
        same_shape_diag_only = _get_optional_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_diag_only",
            None,
            const="on",
        )
        if same_shape_diag_only is None:
            same_shape_diag_only = (
                preserved_diagnostic_env[
                    "MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY"
                ]
                or "off"
            )
        same_shape_diag_only_requested = same_shape_diag_only in (
            "on",
            "true",
            "1",
        )
        same_shape_pointer_array_output = _get_optional_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_pointer_array_output",
            "off",
            const="on",
        )
        same_shape_pointer_array_requested = same_shape_pointer_array_output in (
            "on",
            "true",
            "1",
        )
        same_shape_small_k_diag = _get_optional_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_small_k_diag",
            "off",
            const="on",
        )
        same_shape_small_k_diag_requested = same_shape_small_k_diag in (
            "on",
            "true",
            "1",
        )
        same_shape_small_k_pointer_array_output = _get_optional_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_small_k_pointer_array_output",
            "off",
            const="on",
        )
        same_shape_small_k_pointer_array_requested = (
            same_shape_small_k_pointer_array_output in ("on", "true", "1")
        )
        same_shape_small_k_loop_fusion = _get_optional_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_small_k_loop_fusion",
            "off",
            const="on",
        )
        same_shape_small_k_loop_fusion_requested = (
            same_shape_small_k_loop_fusion in ("on", "true", "1")
        )
        same_shape_small_k_custom_kernel = _get_optional_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_small_k_custom_kernel",
            "off",
            const="on",
        )
        same_shape_small_k_custom_kernel_requested = (
            same_shape_small_k_custom_kernel in ("on", "true", "1")
        )
        same_shape_small_k_custom_max_group_size = _get_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_small_k_custom_max_group_size",
            "0",
        )
        same_shape_dot_batcher = _get_cli_arg(
            argv, "--musa_xla_same_shape_dot_batcher", "off"
        )
        if (
            same_shape_diag_only_requested
            or same_shape_pointer_array_requested
            or same_shape_small_k_diag_requested
            or same_shape_small_k_pointer_array_requested
            or same_shape_small_k_loop_fusion_requested
            or same_shape_small_k_custom_kernel_requested
        ):
            same_shape_dot_batcher = "on"
        if same_shape_dot_batcher in ("on", "true", "1", "auto"):
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCHER"] = (
                "1" if same_shape_dot_batcher in ("on", "true", "1") else "auto"
            )
        elif same_shape_dot_batcher in ("off", "false", "0"):
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCHER"] = "0"
        elif (
            preserved_diagnostic_env["MUSA_XLA_SAME_SHAPE_DOT_BATCHER"]
            is not None
        ):
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCHER"] = (
                preserved_diagnostic_env["MUSA_XLA_SAME_SHAPE_DOT_BATCHER"]
            )
        _set_tristate_bool_env_value(
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY",
            same_shape_diag_only,
        )
        same_shape_dot_batch_min_group = _get_cli_arg(
            argv, "--musa_xla_same_shape_dot_batch_min_group_size", "8"
        )
        if same_shape_dot_batch_min_group:
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE"] = (
                same_shape_dot_batch_min_group
            )
        same_shape_dot_batch_max_group = _get_cli_arg(
            argv, "--musa_xla_same_shape_dot_batch_max_group_size", "32"
        )
        if same_shape_dot_batch_max_group:
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE"] = (
                same_shape_dot_batch_max_group
            )
        same_shape_dot_batch_max_groups = _get_cli_arg(
            argv, "--musa_xla_same_shape_dot_batch_max_groups", "128"
        )
        if same_shape_dot_batch_max_groups:
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS"] = (
                same_shape_dot_batch_max_groups
            )
        same_shape_dot_batch_min_candidate_default = (
            "1"
            if (
                same_shape_diag_only_requested
                or same_shape_pointer_array_requested
                or same_shape_small_k_diag_requested
                or same_shape_small_k_pointer_array_requested
                or same_shape_small_k_loop_fusion_requested
                or same_shape_small_k_custom_kernel_requested
            )
            and not _has_cli_flag(
                argv, "--musa_xla_same_shape_dot_batch_min_candidate_dots"
            )
            else "512"
        )
        same_shape_dot_batch_min_candidate_dots = _get_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_min_candidate_dots",
            same_shape_dot_batch_min_candidate_default,
        )
        if same_shape_dot_batch_min_candidate_dots:
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS"] = (
                same_shape_dot_batch_min_candidate_dots
            )
        same_shape_dot_batch_max_slice_bytes_per_saved_launch = _get_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch",
            "2097152",
        )
        if same_shape_dot_batch_max_slice_bytes_per_saved_launch:
            os.environ[
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH"
            ] = same_shape_dot_batch_max_slice_bytes_per_saved_launch
        same_shape_dot_batch_max_output_cols = _get_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_max_output_cols",
            "256",
        )
        if (
            same_shape_dot_batch_max_output_cols
            and str(same_shape_dot_batch_max_output_cols) != "0"
        ):
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS"] = (
                same_shape_dot_batch_max_output_cols
            )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_shape_dot_batcher_log",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_shape_dot_batch_post_dot_diag",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POST_DOT_DIAG",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_shape_dot_batch_add_tree_diag",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_DIAG",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_shape_dot_batch_add_tree_rewrite",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_REWRITE",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_shape_dot_batch_add_tree_external_diag",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_EXTERNAL_DIAG",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_shape_dot_batch_add_tree_mixed_key_diag",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_DIAG",
        )
        add_tree_max_depth = _get_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_add_tree_max_depth",
            "64",
        )
        if add_tree_max_depth:
            os.environ[
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MAX_DEPTH"
            ] = str(add_tree_max_depth)
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_shape_dot_batch_add_tree_mixed_key_rewrite",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_REWRITE",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_shape_dot_batch_biasadd",
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_BIASADD",
        )
        _set_tristate_bool_env_value(
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POINTER_ARRAY_OUTPUT",
            same_shape_pointer_array_output,
        )
        _set_tristate_bool_env_value(
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_DIAG",
            same_shape_small_k_diag,
        )
        same_shape_small_k_max_k = _get_cli_arg(
            argv, "--musa_xla_same_shape_dot_batch_small_k_max_k", "8"
        )
        if same_shape_small_k_max_k:
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MAX_K"] = str(
                same_shape_small_k_max_k
            )
        same_shape_small_k_min_group_size = _get_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_small_k_min_group_size",
            "16",
        )
        if same_shape_small_k_min_group_size:
            os.environ[
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MIN_GROUP_SIZE"
            ] = str(same_shape_small_k_min_group_size)
        same_shape_small_k_output_cols = _get_cli_arg(
            argv,
            "--musa_xla_same_shape_dot_batch_small_k_output_cols",
            "160,192,256",
        )
        if same_shape_small_k_output_cols is not None:
            os.environ["MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_OUTPUT_COLS"] = str(
                same_shape_small_k_output_cols
            )
        _set_tristate_bool_env_value(
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_POINTER_ARRAY_OUTPUT",
            same_shape_small_k_pointer_array_output,
        )
        _set_tristate_bool_env_value(
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_LOOP_FUSION",
            same_shape_small_k_loop_fusion,
        )
        _set_tristate_bool_env_value(
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_KERNEL",
            same_shape_small_k_custom_kernel,
        )
        if (
            same_shape_small_k_custom_max_group_size
            and str(same_shape_small_k_custom_max_group_size) != "0"
        ):
            os.environ[
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_MAX_GROUP_SIZE"
            ] = str(same_shape_small_k_custom_max_group_size)
        same_lhs_dot_merger = _get_cli_arg(
            argv, "--musa_xla_same_lhs_dot_merger", "off"
        )
        if same_lhs_dot_merger in ("on", "true", "1", "auto"):
            os.environ["MUSA_XLA_SAME_LHS_DOT_MERGER"] = (
                "1" if same_lhs_dot_merger in ("on", "true", "1") else "auto"
            )
        elif same_lhs_dot_merger in ("off", "false", "0"):
            os.environ["MUSA_XLA_SAME_LHS_DOT_MERGER"] = "0"
        elif (
            preserved_diagnostic_env["MUSA_XLA_SAME_LHS_DOT_MERGER"]
            is not None
        ):
            os.environ["MUSA_XLA_SAME_LHS_DOT_MERGER"] = (
                preserved_diagnostic_env["MUSA_XLA_SAME_LHS_DOT_MERGER"]
            )
        same_lhs_dot_merger_min_group = _get_cli_arg(
            argv, "--musa_xla_same_lhs_dot_merger_min_group_size", "2"
        )
        if same_lhs_dot_merger_min_group:
            os.environ["MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_GROUP_SIZE"] = (
                same_lhs_dot_merger_min_group
            )
        same_lhs_dot_merger_max_groups = _get_cli_arg(
            argv, "--musa_xla_same_lhs_dot_merger_max_groups", "8"
        )
        if same_lhs_dot_merger_max_groups:
            os.environ["MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUPS"] = (
                same_lhs_dot_merger_max_groups
            )
        same_lhs_dot_merger_max_group = _get_cli_arg(
            argv, "--musa_xla_same_lhs_dot_merger_max_group_size", "16"
        )
        if same_lhs_dot_merger_max_group:
            os.environ["MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUP_SIZE"] = (
                same_lhs_dot_merger_max_group
            )
        same_lhs_dot_merger_max_total_cols = _get_cli_arg(
            argv, "--musa_xla_same_lhs_dot_merger_max_total_cols", "2048"
        )
        if same_lhs_dot_merger_max_total_cols:
            os.environ["MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_TOTAL_COLS"] = (
                same_lhs_dot_merger_max_total_cols
            )
        same_lhs_dot_merger_min_candidate_dots = _get_cli_arg(
            argv, "--musa_xla_same_lhs_dot_merger_min_candidate_dots", "128"
        )
        if same_lhs_dot_merger_min_candidate_dots:
            os.environ["MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_CANDIDATE_DOTS"] = (
                same_lhs_dot_merger_min_candidate_dots
            )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_lhs_dot_merger_normalize_operands",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_NORMALIZE_OPERANDS",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_lhs_dot_merger_log",
            "MUSA_XLA_SAME_LHS_DOT_MERGER_LOG",
        )
        same_rhs_dot_merger = _get_cli_arg(
            argv, "--musa_xla_same_rhs_dot_merger", "off"
        )
        if same_rhs_dot_merger in ("on", "true", "1", "auto"):
            os.environ["MUSA_XLA_SAME_RHS_DOT_MERGER"] = (
                "1" if same_rhs_dot_merger in ("on", "true", "1") else "auto"
            )
        elif same_rhs_dot_merger in ("off", "false", "0"):
            os.environ["MUSA_XLA_SAME_RHS_DOT_MERGER"] = "0"
        elif (
            preserved_diagnostic_env["MUSA_XLA_SAME_RHS_DOT_MERGER"]
            is not None
        ):
            os.environ["MUSA_XLA_SAME_RHS_DOT_MERGER"] = (
                preserved_diagnostic_env["MUSA_XLA_SAME_RHS_DOT_MERGER"]
            )
        same_rhs_dot_merger_min_group = _get_cli_arg(
            argv, "--musa_xla_same_rhs_dot_merger_min_group_size", "2"
        )
        if same_rhs_dot_merger_min_group:
            os.environ["MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_GROUP_SIZE"] = (
                same_rhs_dot_merger_min_group
            )
        same_rhs_dot_merger_max_groups = _get_cli_arg(
            argv, "--musa_xla_same_rhs_dot_merger_max_groups", "8"
        )
        if same_rhs_dot_merger_max_groups:
            os.environ["MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUPS"] = (
                same_rhs_dot_merger_max_groups
            )
        same_rhs_dot_merger_max_group = _get_cli_arg(
            argv, "--musa_xla_same_rhs_dot_merger_max_group_size", "4"
        )
        if same_rhs_dot_merger_max_group:
            os.environ["MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUP_SIZE"] = (
                same_rhs_dot_merger_max_group
            )
        same_rhs_dot_merger_max_total_rows = _get_cli_arg(
            argv, "--musa_xla_same_rhs_dot_merger_max_total_rows", "4096"
        )
        if same_rhs_dot_merger_max_total_rows:
            os.environ["MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_TOTAL_ROWS"] = (
                same_rhs_dot_merger_max_total_rows
            )
        same_rhs_dot_merger_min_candidate_dots = _get_cli_arg(
            argv, "--musa_xla_same_rhs_dot_merger_min_candidate_dots", "128"
        )
        if same_rhs_dot_merger_min_candidate_dots:
            os.environ["MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_CANDIDATE_DOTS"] = (
                same_rhs_dot_merger_min_candidate_dots
            )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_same_rhs_dot_merger_log",
            "MUSA_XLA_SAME_RHS_DOT_MERGER_LOG",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_group_gemm_thunks",
            "MUSA_XLA_GROUP_GEMM_THUNKS",
            default="off",
        )
        group_gemm_thunks_min_group = _get_cli_arg(
            argv, "--musa_xla_group_gemm_thunks_min_group_size", "4"
        )
        if group_gemm_thunks_min_group:
            os.environ["MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE"] = (
                group_gemm_thunks_min_group
            )
        group_gemm_thunks_max_group = _get_cli_arg(
            argv, "--musa_xla_group_gemm_thunks_max_group_size", "64"
        )
        if group_gemm_thunks_max_group:
            os.environ["MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE"] = (
                group_gemm_thunks_max_group
            )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_group_gemm_thunks_log",
            "MUSA_XLA_GROUP_GEMM_THUNKS_LOG",
        )
        _set_tristate_bool_env_from_cli(
            argv,
            "--musa_xla_group_gemm_thunks_cross_kernel_diag",
            "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG",
        )
        group_gemm_cross_kernel_max_separators = _get_cli_arg(
            argv, "--musa_xla_group_gemm_thunks_cross_kernel_max_separators", "8"
        )
        if group_gemm_cross_kernel_max_separators:
            os.environ["MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_MAX_SEPARATORS"] = (
                group_gemm_cross_kernel_max_separators
            )
        _set_tristate_bool_env_from_cli_or_preserve(
            argv,
            "--musa_xla_small_gemm_accum_thunks",
            "MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS",
            default="off",
        )
        _set_tristate_bool_env_from_cli_or_preserve(
            argv,
            "--musa_xla_small_gemm_accum_log",
            "MUSA_XLA_SMALL_GEMM_ACCUM_LOG",
            default="off",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_small_gemm_accum_min_chain_size",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE",
            default="4",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_small_gemm_accum_max_chain_size",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE",
            default="64",
        )
        _set_value_env_from_cli_or_preserve(
            argv,
            "--musa_xla_small_gemm_accum_max_k",
            "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K",
            default="64",
        )
        _set_tristate_bool_env_from_cli_or_preserve(
            argv,
            "--musa_xla_small_gemm_accum_require_custom_kernel",
            "MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL",
            default="off",
        )
        group_gemm_requested = (
            _is_env_enabled("MUSA_XLA_GROUP_GEMM_THUNKS")
            or _is_env_enabled("MUSA_XLA_GROUP_GEMM_THUNKS_LOG")
            or _is_env_enabled("MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG")
        )
        small_gemm_accum_requested = (
            _is_env_enabled("MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS")
            or _is_env_enabled("MUSA_XLA_SMALL_GEMM_ACCUM_LOG")
            or _is_env_enabled("MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL")
        )
        gemm_epilogue_custom_call_requested = _is_env_enabled(
            "MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL"
        )
        musa_xla_gpu_runtime = _get_cli_arg(argv, "--musa_xla_gpu_runtime", "auto")
        if musa_xla_gpu_runtime == "auto" and (
            group_gemm_requested
            or gemm_epilogue_custom_call_requested
        ):
            musa_xla_gpu_runtime = "classic_thunks"
        _set_xla_gpu_runtime(musa_xla_gpu_runtime)
        should_auto_enable_thunk_diagnostics = (
            group_gemm_requested
            or gemm_epilogue_custom_call_requested
            or small_gemm_accum_requested
            or musa_xla_gpu_runtime == "classic_thunks"
        )
        if (
            not _has_cli_flag(argv, "--musa_xla_thunk_diagnostics")
            and preserved_diagnostic_env["MUSA_XLA_THUNK_DIAGNOSTICS"] is None
            and should_auto_enable_thunk_diagnostics
        ):
            os.environ["MUSA_XLA_THUNK_DIAGNOSTICS"] = "1"
        global_jit_level = _get_cli_arg(argv, "--xla_global_jit_level", "off")
        if global_jit_level not in ("", "off", "false", "0", "auto"):
            os.environ["MUSA_XLA_GLOBAL_JIT_LEVEL"] = global_jit_level

    if enable_dump:
        xla_dump_dir = os.path.abspath(
            os.path.expanduser(dump_dir or str(DEFAULT_XLA_DUMP_DIR))
        )
        dump_pass_re = _get_cli_arg(argv, "--xla_dump_hlo_pass_re", "^$")
        dump_max_modules = _get_cli_arg(argv, "--xla_dump_max_hlo_modules", "-1")
        dump_module_re = _get_cli_arg(argv, "--xla_dump_hlo_module_re", "")
        dump_long_text = _get_cli_arg(argv, "--xla_dump_long_text", "false")
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags = _set_flag_with_prefix(
            xla_flags, "--xla_dump_to=", f"--xla_dump_to={xla_dump_dir}"
        )
        xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_text")
        if dump_long_text in ("1", "true", "TRUE", "yes", "YES", "on", "ON"):
            xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_long_text")
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_dump_hlo_pass_re=",
            f"--xla_dump_hlo_pass_re={dump_pass_re}",
        )
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_dump_max_hlo_modules=",
            f"--xla_dump_max_hlo_modules={dump_max_modules}",
        )
        if dump_module_re:
            xla_flags = _set_flag_with_prefix(
                xla_flags,
                "--xla_dump_hlo_module_re=",
                f"--xla_dump_hlo_module_re={dump_module_re}",
            )
        os.environ["XLA_FLAGS"] = xla_flags
        os.environ["GRAPH_RUNNER_XLA_DUMP_DIR"] = xla_dump_dir


if _maybe_run_same_shape_dot_batch_sweep(sys.argv[1:]):
    raise SystemExit(0)

EARLY_EFFECTIVE_ARGV, EARLY_OPTIMIZATION_PROFILE = _argv_with_optimization_profile(
    sys.argv[1:]
)
_early_cpu_affinity_mode = _get_cli_arg(
    EARLY_EFFECTIVE_ARGV,
    "--cpu_affinity",
    "auto",
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


def merge_packed_subset_feed_items(original_items, candidate_items, unpacked_items):
    candidate_names = {item["name"] for item in candidate_items}
    unpacked_by_name = {item["name"]: item for item in unpacked_items}
    merged = []
    for item in original_items:
        if item["name"] not in candidate_names:
            merged.append(item)
            continue
        unpacked_item = unpacked_by_name.get(item["name"])
        if unpacked_item is not None:
            merged.append(unpacked_item)
    return merged


def summarize_large_feed_items(feed_items, graph_def, bs=None, limit=20):
    def base_tensor_name(name):
        text = str(name)
        if text.startswith("^"):
            text = text[1:]
        return text.split(":", 1)[0]

    consumer_by_input = {}
    consumer_names_by_input = {}
    for node in graph_def.node:
        for input_name in node.input:
            base = base_tensor_name(input_name)
            consumer_by_input.setdefault(base, Counter())[node.op] += 1
            consumer_names_by_input.setdefault(base, []).append(node.name)

    items = []
    consumer_hist = Counter()
    total_bytes = 0
    batch_dim_bytes = 0
    non_batch_dim_bytes = 0
    scalar_bytes = 0
    large_input_count = 0
    large_input_bytes = 0
    for item in feed_items:
        value = item["value"]
        nbytes = int(getattr(value, "nbytes", 0) or 0)
        shape = list(getattr(value, "shape", item.get("shape", [])))
        consumers = consumer_by_input.get(item["node_name"], Counter())
        consumer_hist.update(consumers)
        total_bytes += nbytes
        batch_dim = bool(shape and bs is not None and shape[0] == bs)
        if not shape:
            scalar_bytes += nbytes
        elif batch_dim:
            batch_dim_bytes += nbytes
        else:
            non_batch_dim_bytes += nbytes
        if nbytes >= 1024 * 1024:
            large_input_count += 1
            large_input_bytes += nbytes
        items.append(
            {
                "name": item["name"],
                "node_name": item["node_name"],
                "dtype": str(item.get("np_dtype", getattr(value, "dtype", ""))),
                "shape": shape,
                "nbytes": nbytes,
                "mib": nbytes / (1024.0 * 1024.0),
                "batch_dim": batch_dim,
                "consumer_ops": consumers.most_common(10),
                "consumer_nodes": consumer_names_by_input.get(item["node_name"], [])[:8],
            }
        )

    items.sort(key=lambda entry: entry["nbytes"], reverse=True)
    return {
        "num_inputs": len(feed_items),
        "total_mib": total_bytes / (1024.0 * 1024.0),
        "batch_dim_mib": batch_dim_bytes / (1024.0 * 1024.0),
        "non_batch_dim_mib": non_batch_dim_bytes / (1024.0 * 1024.0),
        "scalar_mib": scalar_bytes / (1024.0 * 1024.0),
        "large_input_count": large_input_count,
        "large_input_mib": large_input_bytes / (1024.0 * 1024.0),
        "top_consumer_ops": consumer_hist.most_common(20),
        "largest_inputs": items[:limit],
    }


def _diag_base_tensor_name(name):
    text = str(name)
    if text.startswith("^"):
        text = text[1:]
    return text.split(":", 1)[0]


def summarize_concat_pack_downstream(
    graph_def, concat_pack_state, output_spec=None, limit=20
):
    packed_inputs = (concat_pack_state or {}).get("packed_inputs") or []
    if not packed_inputs:
        return {}

    node_map = {node.name: node for node in graph_def.node}
    consumers_by_input = defaultdict(list)
    for node in graph_def.node:
        for input_name in node.input:
            consumers_by_input[_diag_base_tensor_name(input_name)].append(node)

    output_nodes = {_diag_base_tensor_name(name) for name in (output_spec or [])}
    packed_by_concat = {}
    for item in packed_inputs:
        concat_node = item.get("concat_node")
        if not concat_node:
            continue
        packed = packed_by_concat.setdefault(
            concat_node,
            {
                "concat_node": concat_node,
                "nbytes": 0,
                "num_original_inputs": 0,
                "chunks": 0,
                "shapes": [],
            },
        )
        packed["nbytes"] += int(item.get("nbytes", 0) or 0)
        packed["num_original_inputs"] += int(item.get("num_original_inputs", 0) or 0)
        packed["chunks"] += 1
        if item.get("shape") is not None:
            packed["shapes"].append(list(item.get("shape") or []))

    def collect_source_placeholders(root_name):
        placeholders = set()
        visited = set()
        stack = [root_name]
        while stack:
            name = stack.pop()
            if name in visited:
                continue
            visited.add(name)
            node = node_map.get(name)
            if node is None:
                continue
            if node.op in ("Placeholder", "PlaceholderWithDefault"):
                placeholders.add(name)
                continue
            for input_name in node.input:
                stack.append(_diag_base_tensor_name(input_name))
        return placeholders

    def collect_external_placeholders(downstream_nodes, stop_nodes):
        placeholders = set()
        visited = set()
        stack = []
        for name in downstream_nodes:
            node = node_map.get(name)
            if node is None:
                continue
            for input_name in node.input:
                input_node = _diag_base_tensor_name(input_name)
                if input_node not in stop_nodes:
                    stack.append(input_node)
        while stack:
            name = stack.pop()
            if name in visited or name in stop_nodes:
                continue
            visited.add(name)
            node = node_map.get(name)
            if node is None:
                continue
            if node.op in ("Placeholder", "PlaceholderWithDefault"):
                placeholders.add(name)
                continue
            for input_name in node.input:
                stack.append(_diag_base_tensor_name(input_name))
        return placeholders

    def trace_consumer_chain(concat_node):
        chain = []
        seen = set()
        current = concat_node
        while len(chain) < limit:
            next_nodes = [
                node
                for node in consumers_by_input.get(current, [])
                if node.name not in seen
            ]
            if not next_nodes:
                break
            node = next_nodes[0]
            seen.add(node.name)
            chain.append({"name": node.name, "op": node.op})
            current = node.name
            if node.name in output_nodes:
                break
        return chain

    shape_seed_ops = {"Shape", "ShapeN", "Rank", "Size"}
    shape_transform_ops = {
        "Identity",
        "Cast",
        "Pack",
        "ConcatV2",
        "Slice",
        "StridedSlice",
        "Gather",
        "GatherV2",
        "Add",
        "AddV2",
        "Sub",
        "Mul",
        "Maximum",
        "Minimum",
        "FloorDiv",
        "RealDiv",
        "Prod",
    }

    def concat_static_precompute_candidates(concat_node, downstream_nodes):
        candidates = set()
        changed = True
        while changed:
            changed = False
            for name in downstream_nodes:
                if name in candidates:
                    continue
                node = node_map.get(name)
                if node is None:
                    continue
                data_inputs = [
                    _diag_base_tensor_name(input_name)
                    for input_name in node.input
                    if not str(input_name).startswith("^")
                ]
                if not data_inputs:
                    continue
                if all(
                    input_name == concat_node
                    or input_name in candidates
                    or (
                        node_map.get(input_name) is not None
                        and node_map[input_name].op == "Const"
                    )
                    for input_name in data_inputs
                ):
                    candidates.add(name)
                    changed = True
        boundary = []
        for name in sorted(candidates):
            if name in output_nodes:
                boundary.append(name)
                continue
            if any(
                consumer.name not in candidates
                for consumer in consumers_by_input.get(name, [])
            ):
                boundary.append(name)
        return candidates, boundary

    def value_independent_subgraph(downstream_nodes):
        value_independent = {
            name
            for name in downstream_nodes
            if (node_map.get(name) is not None and node_map[name].op in shape_seed_ops)
        }
        changed = True
        while changed:
            changed = False
            for name in downstream_nodes:
                if name in value_independent:
                    continue
                node = node_map.get(name)
                if node is None or node.op not in shape_transform_ops:
                    continue
                data_inputs = [
                    _diag_base_tensor_name(input_name)
                    for input_name in node.input
                    if not str(input_name).startswith("^")
                ]
                if not data_inputs:
                    continue
                if all(
                    input_name in value_independent
                    or (
                        node_map.get(input_name) is not None
                        and node_map[input_name].op == "Const"
                    )
                    for input_name in data_inputs
                ):
                    value_independent.add(name)
                    changed = True

        supporting_consts = set()
        for name in value_independent:
            node = node_map.get(name)
            if node is None:
                continue
            for input_name in node.input:
                base = _diag_base_tensor_name(input_name)
                input_node = node_map.get(base)
                if input_node is not None and input_node.op == "Const":
                    supporting_consts.add(base)
        return value_independent | supporting_consts

    items = []
    total_downstream_nodes = 0
    total_nbytes = 0
    total_original_inputs = 0
    union_top_ops = Counter()
    union_value_independent_top_ops = Counter()
    union_precompute_candidate_top_ops = Counter()
    total_value_independent_nodes = 0
    total_precompute_candidate_nodes = 0
    for concat_node, packed in packed_by_concat.items():
        direct_consumers = consumers_by_input.get(concat_node, [])
        direct_consumer_ops = Counter(node.op for node in direct_consumers)
        downstream = set()
        stack = [node.name for node in direct_consumers]
        while stack:
            name = stack.pop()
            if name in downstream:
                continue
            node = node_map.get(name)
            if node is None:
                continue
            downstream.add(name)
            for consumer in consumers_by_input.get(name, []):
                stack.append(consumer.name)

        op_hist = Counter()
        output_hits = []
        for name in downstream:
            node = node_map.get(name)
            if node is None:
                continue
            op_hist[node.op] += 1
            union_top_ops[node.op] += 1
            if name in output_nodes:
                output_hits.append(name)

        source_placeholders = collect_source_placeholders(concat_node)
        external_placeholders = collect_external_placeholders(
            downstream, downstream | {concat_node}
        )
        value_independent = value_independent_subgraph(downstream)
        value_independent_hist = Counter(
            node_map[name].op
            for name in value_independent
            if node_map.get(name) is not None
        )
        precompute_candidates, precompute_boundary = (
            concat_static_precompute_candidates(concat_node, downstream)
        )
        precompute_candidate_hist = Counter(
            node_map[name].op
            for name in precompute_candidates
            if node_map.get(name) is not None
        )
        precompute_boundary_hist = Counter(
            node_map[name].op
            for name in precompute_boundary
            if node_map.get(name) is not None
        )
        union_value_independent_top_ops.update(value_independent_hist)
        union_precompute_candidate_top_ops.update(precompute_candidate_hist)
        total_value_independent_nodes += len(value_independent)
        total_precompute_candidate_nodes += len(precompute_candidates)
        total_downstream_nodes += len(downstream)
        total_nbytes += packed["nbytes"]
        total_original_inputs += packed["num_original_inputs"]
        items.append(
            {
                "concat_node": concat_node,
                "nbytes": packed["nbytes"],
                "mib": packed["nbytes"] / (1024.0 * 1024.0),
                "num_original_inputs": packed["num_original_inputs"],
                "chunks": packed["chunks"],
                "shapes": packed["shapes"][:3],
                "direct_consumers": direct_consumer_ops.most_common(limit),
                "direct_consumer_nodes": [node.name for node in direct_consumers[:limit]],
                "downstream_nodes": len(downstream),
                "output_hits": sorted(output_hits)[:limit],
                "source_placeholder_count": len(source_placeholders),
                "external_placeholder_count": len(external_placeholders),
                "external_placeholders": sorted(external_placeholders)[:limit],
                "value_independent_nodes": len(value_independent),
                "value_independent_top_ops": _sorted_counter_items(
                    value_independent_hist, limit
                ),
                "value_independent_samples": sorted(value_independent)[:limit],
                "precompute_candidate_nodes": len(precompute_candidates),
                "precompute_candidate_top_ops": _sorted_counter_items(
                    precompute_candidate_hist, limit
                ),
                "precompute_candidate_samples": sorted(precompute_candidates)[:limit],
                "precompute_boundary_nodes": sorted(precompute_boundary)[:limit],
                "precompute_boundary_ops": _sorted_counter_items(
                    precompute_boundary_hist, limit
                ),
                "precompute_matmul_nodes": sum(
                    1
                    for name in precompute_candidates
                    if node_map.get(name) is not None
                    and node_map[name].op in ("MatMul", "BatchMatMul", "BatchMatMulV2")
                ),
                "top_ops": op_hist.most_common(limit),
                "chain": trace_consumer_chain(concat_node),
            }
        )

    items.sort(key=lambda item: (item["downstream_nodes"], item["nbytes"]), reverse=True)
    return {
        "groups": len(items),
        "total_mib": total_nbytes / (1024.0 * 1024.0),
        "total_original_inputs": total_original_inputs,
        "total_downstream_nodes": total_downstream_nodes,
        "total_value_independent_nodes": total_value_independent_nodes,
        "total_precompute_candidate_nodes": total_precompute_candidate_nodes,
        "value_independent_top_ops": _sorted_counter_items(
            union_value_independent_top_ops, limit
        ),
        "precompute_candidate_top_ops": _sorted_counter_items(
            union_precompute_candidate_top_ops, limit
        ),
        "top_ops": union_top_ops.most_common(limit),
        "items": items[:limit],
    }


def _summarize_xla_dump_dir_for_compare(dump_dir):
    dump_path = Path(dump_dir)
    txt_files = sorted(dump_path.rglob("*.txt")) if dump_path.exists() else []
    after_files = sorted(dump_path.rglob("*after_optimizations*.txt"))
    before_files = sorted(dump_path.rglob("*before_optimizations*.txt"))
    parsed_files = after_files or before_files or txt_files
    instruction_re = re.compile(r"^\s*(?:ROOT\s+)?[A-Za-z0-9_.%-]+ = ")
    opcode_re = re.compile(r"\s([A-Za-z][A-Za-z0-9_-]*)\(")
    opcode_counts = Counter()
    instruction_count = 0
    for path in parsed_files:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not instruction_re.search(line):
                        continue
                    instruction_count += 1
                    rhs = line.split(" = ", 1)[1]
                    match = opcode_re.search(rhs)
                    if match:
                        opcode_counts[match.group(1)] += 1
        except OSError:
            continue
    return {
        "dump_dir": str(dump_path),
        "exists": dump_path.exists(),
        "parsed_files": len(parsed_files),
        "instruction_count": instruction_count,
        "opcode_counts": dict(opcode_counts),
        "top_opcodes": opcode_counts.most_common(20),
    }


def compare_xla_dump_dirs(baseline_dir, freeze_dir):
    baseline = _summarize_xla_dump_dir_for_compare(baseline_dir)
    freeze = _summarize_xla_dump_dir_for_compare(freeze_dir)
    baseline_counts = Counter(baseline.get("opcode_counts", {}))
    freeze_counts = Counter(freeze.get("opcode_counts", {}))
    opcode_removed = baseline_counts - freeze_counts
    opcode_added = freeze_counts - baseline_counts
    return {
        "baseline": baseline,
        "freeze": freeze,
        "instruction_delta": baseline.get("instruction_count", 0)
        - freeze.get("instruction_count", 0),
        "opcode_removed": dict(opcode_removed),
        "opcode_added": dict(opcode_added),
        "top_removed": opcode_removed.most_common(20),
        "top_added": opcode_added.most_common(20),
    }


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
_HLO_DTYPE_BYTES = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "s16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "s32": 4,
    "u32": 4,
    "f32": 4,
    "s64": 8,
    "u64": 8,
    "f64": 8,
}


def _select_xla_hot_fusion_files(dump_path):
    txt_files = sorted(dump_path.rglob("*.txt")) if dump_path.exists() else []
    if not txt_files:
        return [], "none"
    stage_patterns = [
        ("gpu_after_optimizations", "*gpu_after_optimizations*.txt"),
        ("after_optimizations", "*after_optimizations*.txt"),
        ("before_optimizations", "*before_optimizations*.txt"),
    ]
    for stage, pattern in stage_patterns:
        files = sorted(dump_path.rglob(pattern))
        if files:
            return files, stage
    return txt_files, "all_txt"


def _parse_hlo_computation_blocks(lines):
    blocks = {}
    current_name = None
    current_lines = []
    for line in lines:
        if current_name is None:
            match = _HLO_COMPUTATION_START_RE.match(line.strip())
            if match:
                current_name = match.group("name")
                current_lines = [line]
            continue
        current_lines.append(line)
        if line.strip() == "}":
            blocks[current_name] = list(current_lines)
            current_name = None
            current_lines = []
    return blocks


def _xla_dump_file_stage(path):
    name = Path(path).name
    if "gpu_after_optimizations" in name:
        return "gpu_after_optimizations"
    if "after_optimizations" in name:
        return "after_optimizations"
    if "before_optimizations" in name:
        return "before_optimizations"
    return "intermediate"


def _collect_hlo_computation_candidates(paths):
    candidates = {}
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for block_name, block_lines in _parse_hlo_computation_blocks(lines).items():
            candidates.setdefault(block_name, []).append(
                {
                    "name": block_name,
                    "lines": block_lines,
                    "file": str(path),
                    "stage": _xla_dump_file_stage(path),
                }
            )
    for records in candidates.values():
        records.sort(key=lambda record: (record["file"], record["name"]))
    return candidates


def _matching_hlo_computation_names(computation_candidates, callee):
    normalized = str(callee or "").lstrip("%")

    def is_matching_clone(name):
        return (
            name == f"{normalized}.clone"
            or name.startswith(f"{normalized}.clone.")
            or name == f"{normalized}_clone"
            or name.startswith(f"{normalized}_clone.")
        )

    return sorted(
        name
        for name in computation_candidates
        if name == normalized or is_matching_clone(name)
    )


def _resolve_hlo_computation_candidate(
    computation_candidates, callee, preferred_file=None, preferred_stage=None
):
    matching_names = _matching_hlo_computation_names(
        computation_candidates, callee
    )
    records = [
        record
        for name in matching_names
        for record in computation_candidates.get(name, [])
    ]
    if not records:
        return None

    preferred_file = str(preferred_file or "")
    preferred_stage = str(preferred_stage or "")
    normalized_callee = str(callee or "").lstrip("%")
    final_stages = {"gpu_after_optimizations", "after_optimizations"}
    records.sort(
        key=lambda record: (
            record["file"] != preferred_file,
            record["stage"] != preferred_stage,
            record["stage"] not in final_stages,
            record["name"] != normalized_callee,
            record["file"],
            record["name"],
        )
    )
    selected = dict(records[0])
    if selected["file"] == preferred_file:
        tier = "same_file"
    elif selected["stage"] == preferred_stage:
        tier = "same_stage"
    elif selected["stage"] in final_stages:
        tier = "final_stage"
    else:
        tier = "fallback"
    selected["resolution_tier"] = tier
    selected["candidate_count"] = len(records)
    return selected


def _resolve_hlo_computation_block(computation_blocks, callee):
    matching_names = _matching_hlo_computation_names(computation_blocks, callee)
    if matching_names:
        resolved = matching_names[0]
        return resolved, computation_blocks[resolved]
    return None, []


def _hlo_instruction_opcode(line):
    if not _HLO_INSTRUCTION_RE.search(line):
        return None
    try:
        rhs = line.split(" = ", 1)[1]
    except IndexError:
        return None
    match = _HLO_OPCODE_RE.search(rhs)
    return match.group(1) if match else None


def _split_hlo_top_level_operands(operands_text):
    matching_closer = {"[": "]", "{": "}", "(": ")", "<": ">"}
    stack = []
    parts = []
    start = 0
    for index, char in enumerate(str(operands_text or "")):
        if char in matching_closer:
            stack.append(matching_closer[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == "," and not stack:
            parts.append(str(operands_text)[start:index].strip())
            start = index + 1
    parts.append(str(operands_text or "")[start:].strip())
    return [part for part in parts if part]


def _parse_hlo_instruction_record(line):
    name_match = _HLO_INSTRUCTION_NAME_RE.match(str(line or ""))
    if name_match is None or " = " not in str(line or ""):
        return None
    rhs = str(line).split(" = ", 1)[1]
    opcode_match = _HLO_OPCODE_RE.search(rhs)
    if opcode_match is None:
        return None

    operands_start = opcode_match.end()
    operands_end = rhs.find(")", operands_start)
    operands_text = (
        rhs[operands_start:operands_end] if operands_end >= 0 else ""
    )
    operands = []
    for raw_operand in _split_hlo_top_level_operands(operands_text):
        typed_name_match = re.search(
            r"%([A-Za-z_][A-Za-z0-9_.%-]*)\s*$", raw_operand
        )
        if typed_name_match is not None:
            operand = typed_name_match.group(1)
        else:
            operand = raw_operand.strip().lstrip("%")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.%-]*", operand):
            operands.append(operand)

    return {
        "name": name_match.group("name").lstrip("%"),
        "shape": rhs[: opcode_match.start()].strip(),
        "opcode": opcode_match.group(1),
        "operands": operands,
        "line": str(line).strip(),
    }


def _summarize_hlo_operand_chain(
    instruction_map, operand_name, max_depth=4, max_nodes=16
):
    queue = [(str(operand_name or "").lstrip("%"), 0)]
    cursor = 0
    visited = set()
    chain = []
    depends_on_parameter = False
    while cursor < len(queue) and len(chain) < max_nodes:
        name, depth = queue[cursor]
        cursor += 1
        if not name or name in visited:
            continue
        visited.add(name)
        record = instruction_map.get(name)
        if record is None:
            continue
        opcode = record.get("opcode")
        chain.append(
            {
                "name": name,
                "opcode": opcode,
                "shape": record.get("shape"),
                "depth": depth,
            }
        )
        if opcode == "parameter":
            depends_on_parameter = True
        if depth >= max_depth:
            continue
        for child_name in record.get("operands", []):
            if child_name in instruction_map and child_name not in visited:
                queue.append((child_name, depth + 1))
    return {
        "chain": chain,
        "depends_on_parameter": depends_on_parameter,
    }


def _resolve_hlo_scalar_constant(instruction_map, operand_name, max_depth=8):
    passthrough_opcodes = {"bitcast", "broadcast", "convert", "copy", "reshape"}
    name = str(operand_name or "").lstrip("%")
    visited = set()
    for _ in range(max_depth + 1):
        if not name or name in visited:
            return None
        visited.add(name)
        record = instruction_map.get(name)
        if record is None:
            return None
        opcode = record.get("opcode")
        if opcode == "constant":
            if not re.match(r"^[A-Za-z0-9_]+\[\]", record.get("shape", "")):
                return None
            match = re.search(r"\bconstant\(([^()]*)\)", record.get("line", ""))
            return match.group(1).strip() if match is not None else None
        operands = record.get("operands", [])
        if opcode not in passthrough_opcodes or len(operands) != 1:
            return None
        name = operands[0]
    return None


def _summarize_hlo_power_instructions(body_lines, max_depth=4, max_nodes=16):
    instruction_records = []
    instruction_map = {}
    for line in body_lines or []:
        record = _parse_hlo_instruction_record(line)
        if record is None:
            continue
        instruction_records.append(record)
        instruction_map[record["name"]] = record

    user_counts = Counter()
    for record in instruction_records:
        for operand in record.get("operands", []):
            if operand in instruction_map:
                user_counts[operand] += 1

    powers = []
    for record in instruction_records:
        operands = record.get("operands", [])
        if record.get("opcode") != "power" or len(operands) < 2:
            continue
        base_name, exponent_name = operands[:2]
        base_record = instruction_map.get(base_name, {})
        exponent_record = instruction_map.get(exponent_name, {})
        base_summary = _summarize_hlo_operand_chain(
            instruction_map, base_name, max_depth=max_depth, max_nodes=max_nodes
        )
        exponent_summary = _summarize_hlo_operand_chain(
            instruction_map,
            exponent_name,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        powers.append(
            {
                "instruction": record["name"],
                "output_shape": record.get("shape"),
                "base": base_name,
                "exponent": exponent_name,
                "base_opcode": base_record.get("opcode"),
                "exponent_opcode": exponent_record.get("opcode"),
                "base_shape": base_record.get("shape"),
                "exponent_shape": exponent_record.get("shape"),
                "base_constant": _resolve_hlo_scalar_constant(
                    instruction_map, base_name
                ),
                "exponent_constant": _resolve_hlo_scalar_constant(
                    instruction_map, exponent_name
                ),
                "base_chain": base_summary["chain"],
                "exponent_chain": exponent_summary["chain"],
                "base_depends_on_parameter": base_summary[
                    "depends_on_parameter"
                ],
                "exponent_depends_on_parameter": exponent_summary[
                    "depends_on_parameter"
                ],
                "direct_user_count": user_counts[record["name"]],
            }
        )
    return powers


def _hlo_shape_nbytes(shape_text):
    total = 0
    for match in _HLO_ARRAY_SHAPE_RE.finditer(str(shape_text or "")):
        dtype = match.group("dtype")
        dims_text = match.group("dims")
        elem_count = 1
        if dims_text:
            for raw_dim in dims_text.split(","):
                try:
                    elem_count *= int(raw_dim)
                except ValueError:
                    elem_count = 0
                    break
        total += elem_count * _HLO_DTYPE_BYTES.get(dtype, 0)
    return total


def _classify_hlo_hot_fusion(opcode_counts, shape_text):
    opcodes = set(opcode_counts)
    kinds = []
    if "reduce" in opcodes and ("select" in opcodes or "compare" in opcodes):
        kinds.append("masked_softmax_or_row_reduce")
    if "reduce" in opcodes and (
        "exponential" in opcodes or "divide" in opcodes or "subtract" in opcodes
    ):
        kinds.append("softmax_like")
    if "reduce" in opcodes:
        kinds.append("row_reduce")
    if "slice" in opcodes or "dynamic-slice" in opcodes:
        kinds.append("slice_like")
    if "concatenate" in opcodes:
        kinds.append("concat_like")
    if str(shape_text or "").lstrip().startswith("("):
        kinds.append("tuple_output")
    return kinds or ["unknown"]


def _compact_hlo_line(line, max_len=220):
    text = " ".join(str(line or "").strip().split())
    text = re.sub(r"constant\(\{.*?\}\)", "constant({...})", text)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _summarize_hlo_computation_body(body_lines):
    opcode_counts = Counter()
    interesting_lines = []
    for line in body_lines or []:
        opcode = _hlo_instruction_opcode(line)
        if opcode:
            opcode_counts[opcode] += 1
            if opcode not in ("constant", "parameter"):
                interesting_lines.append(_compact_hlo_line(line))
    return opcode_counts, interesting_lines[:12]


def _summarize_hlo_reductions(
    body_lines, computation_candidates, preferred_file=None, preferred_stage=None
):
    reductions = []
    for line in body_lines or []:
        if _hlo_instruction_opcode(line) != "reduce":
            continue
        name_match = _HLO_INSTRUCTION_NAME_RE.match(line)
        dims_match = _HLO_REDUCE_DIMENSIONS_RE.search(line)
        to_apply_match = _HLO_TO_APPLY_RE.search(line)
        dimensions = []
        if dims_match:
            for raw_dim in dims_match.group("dims").split(","):
                raw_dim = raw_dim.strip()
                if raw_dim:
                    try:
                        dimensions.append(int(raw_dim))
                    except ValueError:
                        pass
        to_apply = to_apply_match.group("callee") if to_apply_match else None
        reducer_candidate = (
            _resolve_hlo_computation_candidate(
                computation_candidates,
                to_apply,
                preferred_file=preferred_file,
                preferred_stage=preferred_stage,
            )
            if to_apply
            else None
        )
        reducer_lines = (
            reducer_candidate.get("lines", [])
            if reducer_candidate is not None
            else []
        )
        reducer_root_line = next(
            (
                reducer_line
                for reducer_line in reducer_lines
                if reducer_line.lstrip().startswith("ROOT ")
            ),
            None,
        )
        shape_texts = [
            match.group(0) for match in _HLO_ARRAY_SHAPE_RE.finditer(line)
        ]
        reductions.append(
            {
                "instruction": (
                    name_match.group("name") if name_match is not None else None
                ),
                "line": _compact_hlo_line(line, max_len=360),
                "output_shape": shape_texts[0] if shape_texts else None,
                "input_shapes": shape_texts[1:],
                "dimensions": dimensions,
                "to_apply": to_apply,
                "resolved_reducer_computation": (
                    reducer_candidate.get("name")
                    if reducer_candidate is not None
                    else None
                ),
                "reducer_body_file": (
                    reducer_candidate.get("file")
                    if reducer_candidate is not None
                    else None
                ),
                "reducer_resolution_tier": (
                    reducer_candidate.get("resolution_tier")
                    if reducer_candidate is not None
                    else "missing"
                ),
                "reducer_body_found": bool(reducer_lines),
                "reducer_root_opcode": (
                    _hlo_instruction_opcode(reducer_root_line)
                    if reducer_root_line is not None
                    else None
                ),
            }
        )
    return reductions


def analyze_xla_hot_fusion_dump(dump_dir, hot_fusions="", limit=20):
    dump_path = Path(dump_dir)
    files, parsed_stage = _select_xla_hot_fusion_files(dump_path)
    all_txt_files = sorted(dump_path.rglob("*.txt")) if dump_path.exists() else []
    if not dump_path.exists():
        return {
            "dump_dir": str(dump_path),
            "exists": False,
            "parsed_stage": "none",
            "parsed_files": 0,
            "body_search_files": 0,
            "computation_block_count": 0,
            "fusion_count": 0,
            "missing_callees": [],
            "items": [],
        }

    requested_list = [
        item.strip().lstrip("%")
        for item in str(hot_fusions or "").split(",")
        if item.strip()
    ]
    requested = set(requested_list)
    fusion_calls = {}
    computation_candidates = _collect_hlo_computation_candidates(all_txt_files)
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            match = _HLO_FUSION_CALL_RE.search(line)
            if not match:
                continue
            name = match.group("name")
            callee = match.group("callee")
            if requested and name not in requested and callee not in requested:
                continue
            fusion_calls[name] = {
                "fusion": name,
                "shape": match.group("shape").strip(),
                "called_computation": callee,
                "matched_request": name if name in requested else callee,
                "file": str(path),
                "line": lineno,
                "call_line": _compact_hlo_line(line, max_len=360),
            }

    items = []
    missing_callees = []
    for name, item in sorted(fusion_calls.items()):
        body_candidate = _resolve_hlo_computation_candidate(
            computation_candidates,
            item["called_computation"],
            preferred_file=item["file"],
            preferred_stage=parsed_stage,
        )
        resolved_computation = (
            body_candidate.get("name") if body_candidate is not None else None
        )
        body_lines = (
            body_candidate.get("lines", []) if body_candidate is not None else []
        )
        if not body_lines:
            missing_callees.append(item["called_computation"])
        opcode_counts, body_preview = _summarize_hlo_computation_body(body_lines)
        reductions = _summarize_hlo_reductions(
            body_lines,
            computation_candidates,
            preferred_file=(
                body_candidate.get("file") if body_candidate is not None else None
            ),
            preferred_stage=(
                body_candidate.get("stage") if body_candidate is not None else None
            ),
        )
        powers = _summarize_hlo_power_instructions(body_lines)
        power_patterns = Counter(
            f"{power.get('base_opcode')}->{power.get('exponent_opcode')}"
            for power in powers
        ).most_common(12)
        nbytes = _hlo_shape_nbytes(item["shape"])
        item.update(
            {
                "resolved_computation": resolved_computation,
                "body_file": (
                    body_candidate.get("file") if body_candidate is not None else None
                ),
                "body_resolution_tier": (
                    body_candidate.get("resolution_tier")
                    if body_candidate is not None
                    else "missing"
                ),
                "body_candidate_count": (
                    body_candidate.get("candidate_count", 0)
                    if body_candidate is not None
                    else 0
                ),
                "output_mib": nbytes / (1024.0 * 1024.0),
                "body_found": bool(body_lines),
                "body_line_count": len(body_lines),
                "instruction_count": sum(opcode_counts.values()),
                "top_opcodes": opcode_counts.most_common(12),
                "candidate_kinds": _classify_hlo_hot_fusion(
                    opcode_counts, item["shape"]
                ),
                "body_preview": body_preview,
                "reductions": reductions,
                "powers": powers,
                "power_patterns": power_patterns,
            }
        )
        items.append(item)

    if not requested:
        items.sort(
            key=lambda item: (
                item["output_mib"],
                item["instruction_count"],
                item["fusion"],
            ),
            reverse=True,
        )
    else:
        request_order = {name: index for index, name in enumerate(requested_list)}
        items.sort(
            key=lambda item: request_order.get(
                item.get("matched_request"), len(request_order)
            )
        )

    return {
        "dump_dir": str(dump_path),
        "exists": True,
        "parsed_stage": parsed_stage,
        "parsed_files": len(files),
        "body_search_files": len(all_txt_files),
        "computation_block_count": len(computation_candidates),
        "computation_block_samples": sorted(computation_candidates)[:20],
        "missing_callees": sorted(set(missing_callees)),
        "requested_fusions": sorted(requested),
        "fusion_count": len(items),
        "items": items[: int(limit or 20)],
    }


def print_xla_hot_fusion_summary(summary):
    items = summary.get("items") or []
    print(
        "[INFO] xla_hot_fusion_summary: "
        f"exists={summary.get('exists')} parsed_stage={summary.get('parsed_stage')} "
        f"parsed_files={summary.get('parsed_files')} "
        f"body_search_files={summary.get('body_search_files')} "
        f"computation_block_count={summary.get('computation_block_count')} "
        f"missing_callees={summary.get('missing_callees')} "
        f"fusion_count={summary.get('fusion_count')}"
    )
    for index, item in enumerate(items, start=1):
        top_opcodes = ",".join(
            f"{op}:{count}" for op, count in item.get("top_opcodes", [])[:8]
        )
        candidate_kinds = ",".join(item.get("candidate_kinds", []))
        reducer_summary = ",".join(
            "dims={}:root={}".format(
                ".".join(str(dim) for dim in reduction.get("dimensions", [])),
                reduction.get("reducer_root_opcode"),
            )
            for reduction in item.get("reductions", [])
        )
        power_patterns = ",".join(
            f"{pattern}:{count}"
            for pattern, count in item.get("power_patterns", [])
        )
        print(
            f"[INFO] xla_hot_fusion{index}: fusion={item.get('fusion')} "
            f"shape={item.get('shape')} output_mib={item.get('output_mib', 0.0):.3f} "
            f"callee={item.get('called_computation')} "
            f"resolved={item.get('resolved_computation')} "
            f"body_resolution={item.get('body_resolution_tier')} "
            f"body_candidates={item.get('body_candidate_count')} "
            f"body_found={item.get('body_found')} "
            f"instructions={item.get('instruction_count')} candidate_kinds={candidate_kinds} "
            f"reductions={len(item.get('reductions', []))} reducers={reducer_summary} "
            f"powers={len(item.get('powers', []))} "
            f"power_patterns={power_patterns} "
            f"top_opcodes={top_opcodes} file={Path(item.get('file', '')).name}:{item.get('line')}"
        )
        for power_index, power in enumerate(
            item.get("powers", [])[:12], start=1
        ):
            base_chain = ">".join(
                f"{node.get('opcode')}@{node.get('depth')}"
                for node in power.get("base_chain", [])[:12]
            )
            exponent_chain = ">".join(
                f"{node.get('opcode')}@{node.get('depth')}"
                for node in power.get("exponent_chain", [])[:12]
            )
            print(
                f"[INFO] xla_hot_fusion{index}_power{power_index}: "
                f"instruction={power.get('instruction')} "
                f"shape={power.get('output_shape')} "
                f"base={power.get('base')} "
                f"base_op={power.get('base_opcode')} "
                f"base_shape={power.get('base_shape')} "
                f"base_constant={power.get('base_constant')} "
                f"exponent={power.get('exponent')} "
                f"exponent_op={power.get('exponent_opcode')} "
                f"exponent_shape={power.get('exponent_shape')} "
                f"exponent_constant={power.get('exponent_constant')} "
                f"base_parameter={power.get('base_depends_on_parameter')} "
                f"exponent_parameter="
                f"{power.get('exponent_depends_on_parameter')} "
                f"users={power.get('direct_user_count')} "
                f"base_chain={base_chain} "
                f"exponent_chain={exponent_chain}"
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


def _parse_count_map(value):
    counts = Counter()
    for part in str(value or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, raw_count = part.split("=", 1)
        try:
            counts[key.strip()] += int(raw_count)
        except ValueError:
            continue
    return counts


def _latency_summary_from_log_line(line):
    marker = "latency_summary="
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1].strip()
    try:
        parsed = ast.literal_eval(payload)
    except Exception:
        return None
    if isinstance(parsed, list) and parsed:
        item = parsed[-1]
        if isinstance(item, dict):
            return {
                "average_time_ms": item.get("average_time_ms"),
                "trimmed_avg_ms": item.get("trimmed_avg_ms"),
            }
    return None


def summarize_thunk_diagnostic_log(log_path):
    path = Path(log_path)
    modules = []
    totals = Counter()
    counts = Counter()
    latency = None
    if not path.exists():
        return {
            "log_path": str(path),
            "exists": False,
            "module_count": 0,
            "totals": {"total_thunks": 0, "gemm_thunks": 0, "kernel_thunks": 0},
            "counts": {},
            "main_module": None,
            "top_modules": [],
            "latency": None,
        }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _THUNK_DIAG_RE.search(line)
        if match:
            item_counts = _parse_count_map(match.group("counts"))
            item = {
                "module": match.group("module"),
                "module_id": int(match.group("module_id")),
                "total_thunks": int(match.group("total_thunks")),
                "gemm_thunks": int(match.group("gemm_thunks")),
                "kernel_thunks": int(match.group("kernel_thunks")),
                "counts": dict(item_counts),
            }
            modules.append(item)
            totals["total_thunks"] += item["total_thunks"]
            totals["gemm_thunks"] += item["gemm_thunks"]
            totals["kernel_thunks"] += item["kernel_thunks"]
            counts.update(item_counts)
            continue
        parsed_latency = _latency_summary_from_log_line(line)
        if parsed_latency is not None:
            latency = parsed_latency
    top_modules = sorted(
        modules,
        key=lambda item: (
            item["total_thunks"],
            item["gemm_thunks"],
            item["kernel_thunks"],
        ),
        reverse=True,
    )[:20]
    main_module = top_modules[0] if top_modules else None
    return {
        "log_path": str(path),
        "exists": True,
        "module_count": len(modules),
        "totals": {
            "total_thunks": int(totals["total_thunks"]),
            "gemm_thunks": int(totals["gemm_thunks"]),
            "kernel_thunks": int(totals["kernel_thunks"]),
        },
        "counts": dict(counts),
        "main_module": main_module,
        "top_modules": top_modules,
        "latency": latency,
    }


def _numeric_delta(left, right, keys):
    return {
        key: int((left or {}).get(key, 0) - (right or {}).get(key, 0))
        for key in keys
    }


def compare_thunk_diagnostic_logs(baseline_log, experiment_log):
    baseline = summarize_thunk_diagnostic_log(baseline_log)
    experiment = summarize_thunk_diagnostic_log(experiment_log)
    baseline_counts = Counter(baseline.get("counts", {}))
    experiment_counts = Counter(experiment.get("counts", {}))
    latency_delta = {}
    if baseline.get("latency") and experiment.get("latency"):
        for key in ("average_time_ms", "trimmed_avg_ms"):
            left = baseline["latency"].get(key)
            right = experiment["latency"].get(key)
            if left is not None and right is not None:
                latency_delta[key] = float(left) - float(right)
    return {
        "baseline": baseline,
        "experiment": experiment,
        "total_delta": _numeric_delta(
            baseline.get("totals"),
            experiment.get("totals"),
            ("total_thunks", "gemm_thunks", "kernel_thunks"),
        ),
        "main_module_delta": _numeric_delta(
            baseline.get("main_module"),
            experiment.get("main_module"),
            ("total_thunks", "gemm_thunks", "kernel_thunks"),
        ),
        "counts_removed": dict(baseline_counts - experiment_counts),
        "counts_added": dict(experiment_counts - baseline_counts),
        "latency_delta_ms": latency_delta,
    }


def _sorted_counter_items(counter, limit):
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _diag_attr_int(node, attr_name, default=0):
    attr = getattr(node, "attr", {}) or {}
    if attr_name not in attr:
        return default
    value = attr[attr_name]
    return int(getattr(value, "i", default))


def _diag_const_int_list(node):
    if node is None or getattr(node, "op", None) != "Const":
        return None
    attr = getattr(node, "attr", {}) or {}
    if "value" not in attr:
        return None
    tensor = getattr(attr["value"], "tensor", None)
    if tensor is None:
        return None
    int_val = list(getattr(tensor, "int_val", []) or [])
    if int_val:
        return [int(x) for x in int_val]
    int64_val = list(getattr(tensor, "int64_val", []) or [])
    if int64_val:
        return [int(x) for x in int64_val]
    content = getattr(tensor, "tensor_content", b"") or b""
    if content:
        dtype_code = int(getattr(tensor, "dtype", 0) or 0)
        dtype = np.int64 if dtype_code == 9 else np.int32
        try:
            return [int(x) for x in np.frombuffer(content, dtype=dtype).tolist()]
        except Exception:
            return None
    return None


def _diag_slice_value_from_begin_size(value, begin, size):
    shape = list(value.shape)
    slices = []
    out_shape = []
    for dim_index, dim in enumerate(shape):
        start = int(begin[dim_index]) if dim_index < len(begin) else 0
        requested = int(size[dim_index]) if dim_index < len(size) else -1
        if start < 0:
            return None
        stop = dim if requested < 0 else start + requested
        if stop > dim:
            return None
        slices.append(slice(start, stop))
        out_shape.append(stop - start)
    sliced = value[tuple(slices)]
    return sliced, out_shape


def _diag_strided_slice_value_from_node(value, node, node_map):
    if len(node.input) < 4:
        return None
    begin = _diag_const_int_list(node_map.get(_diag_base_tensor_name(node.input[1])))
    end = _diag_const_int_list(node_map.get(_diag_base_tensor_name(node.input[2])))
    strides = _diag_const_int_list(node_map.get(_diag_base_tensor_name(node.input[3])))
    if begin is None or end is None or strides is None:
        return None
    if _diag_attr_int(node, "ellipsis_mask") or _diag_attr_int(node, "new_axis_mask"):
        return None
    begin_mask = _diag_attr_int(node, "begin_mask")
    end_mask = _diag_attr_int(node, "end_mask")
    shrink_axis_mask = _diag_attr_int(node, "shrink_axis_mask")
    rank = len(value.shape)
    if len(begin) < rank or len(end) < rank or len(strides) < rank:
        return None

    slices = []
    for dim_index in range(rank):
        stride = int(strides[dim_index])
        if stride == 0:
            return None
        if shrink_axis_mask & (1 << dim_index):
            slices.append(int(begin[dim_index]))
            continue
        start = None if begin_mask & (1 << dim_index) else int(begin[dim_index])
        stop = None if end_mask & (1 << dim_index) else int(end[dim_index])
        slices.append(slice(start, stop, stride))
    sliced = value[tuple(slices)]
    return sliced, list(getattr(sliced, "shape", []))


def _diag_slice_like_value_from_node(value, node, node_map):
    if node.op == "Slice":
        if len(node.input) < 3:
            return None
        begin = _diag_const_int_list(node_map.get(_diag_base_tensor_name(node.input[1])))
        size = _diag_const_int_list(node_map.get(_diag_base_tensor_name(node.input[2])))
        if begin is None or size is None:
            return None
        result = _diag_slice_value_from_begin_size(value, begin, size)
        if result is None:
            return None
        sliced_value, sliced_shape = result
        return sliced_value, sliced_shape, {"begin": begin, "size": size}
    if node.op == "StridedSlice":
        result = _diag_strided_slice_value_from_node(value, node, node_map)
        if result is None:
            return None
        begin = _diag_const_int_list(node_map.get(_diag_base_tensor_name(node.input[1])))
        end = _diag_const_int_list(node_map.get(_diag_base_tensor_name(node.input[2])))
        strides = _diag_const_int_list(node_map.get(_diag_base_tensor_name(node.input[3])))
        sliced_value, sliced_shape = result
        return sliced_value, sliced_shape, {
            "begin": begin,
            "end": end,
            "strides": strides,
            "begin_mask": _diag_attr_int(node, "begin_mask"),
            "end_mask": _diag_attr_int(node, "end_mask"),
            "shrink_axis_mask": _diag_attr_int(node, "shrink_axis_mask"),
        }
    return None


def summarize_large_slice_feed_candidates(feed_items, graph_def, args, bs=None, limit=20):
    item_by_node = {item["node_name"]: item for item in feed_items}
    node_map = {node.name: node for node in graph_def.node}
    allowed_slice_ops = {
        part.strip()
        for part in str(getattr(args, "pack_slice_feed_ops", "Slice,StridedSlice")).split(",")
        if part.strip()
    }
    consumer_nodes = {}
    for node in graph_def.node:
        for input_name in node.input:
            base = _diag_base_tensor_name(input_name)
            if base in item_by_node:
                consumer_nodes.setdefault(base, set()).add(node.name)

    min_saved_bytes = int(
        max(0.0, float(getattr(args, "pack_slice_feed_min_saved_mib", 0.0)))
        * 1024
        * 1024
    )
    max_total_bytes = int(
        max(0.0, float(getattr(args, "pack_slice_feed_max_total_mib", 0.0)))
        * 1024
        * 1024
    )
    selected_total = 0
    candidates = []
    status_counts = Counter()
    disallowed_consumer_ops = Counter()

    for item in sorted(
        feed_items,
        key=lambda entry: int(getattr(entry["value"], "nbytes", 0) or 0),
        reverse=True,
    ):
        consumers = sorted(consumer_nodes.get(item["node_name"], set()))
        if not consumers:
            continue
        original_bytes = int(getattr(item["value"], "nbytes", 0) or 0)
        if original_bytes <= 0:
            continue
        slice_nodes = []
        slice_bytes = 0
        status = "ok"
        for consumer_name in consumers:
            node = node_map.get(consumer_name)
            if node is None:
                status = "missing_consumer_node"
                break
            if node.op not in allowed_slice_ops:
                status = "consumer_op_not_allowed"
                disallowed_consumer_ops[node.op] += 1
                break
            if not node.input or _diag_base_tensor_name(node.input[0]) != item["node_name"]:
                status = "consumer_not_primary_input"
                break
            sliced = _diag_slice_like_value_from_node(item["value"], node, node_map)
            if sliced is None:
                status = "unsupported_or_non_const_slice"
                slice_nodes.append({"name": node.name, "op": node.op, "status": status})
                break
            sliced_value, sliced_shape, slice_spec = sliced
            nbytes = int(getattr(sliced_value, "nbytes", 0) or 0)
            slice_bytes += nbytes
            slice_nodes.append(
                {
                    "name": node.name,
                    "op": node.op,
                    "status": "ok",
                    "shape": list(sliced_shape),
                    "mib": nbytes / (1024.0 * 1024.0),
                    "spec": slice_spec,
                }
            )

        saved_bytes = original_bytes - slice_bytes if status == "ok" else 0
        if status == "ok" and saved_bytes < min_saved_bytes:
            status = "saved_below_min"
        would_select = status == "ok"
        if would_select and max_total_bytes > 0 and selected_total + slice_bytes > max_total_bytes:
            status = "total_cap"
            would_select = False
        if would_select:
            selected_total += slice_bytes
        status_counts[status] += 1
        candidates.append(
            {
                "name": item["name"],
                "node_name": item["node_name"],
                "dtype": str(item.get("np_dtype", getattr(item["value"], "dtype", ""))),
                "shape": list(getattr(item["value"], "shape", item.get("shape", []))),
                "batch_dim": bool(
                    bs is not None
                    and len(getattr(item["value"], "shape", [])) > 0
                    and getattr(item["value"], "shape", [None])[0] == bs
                ),
                "consumer_count": len(consumers),
                "consumer_nodes": consumers[:8],
                "status": status,
                "would_select": would_select,
                "original_mib": original_bytes / (1024.0 * 1024.0),
                "slice_mib": slice_bytes / (1024.0 * 1024.0),
                "saved_mib": saved_bytes / (1024.0 * 1024.0),
                "num_slice_nodes": len(slice_nodes),
                "slice_nodes": slice_nodes[:8],
            }
        )

    candidates.sort(
        key=lambda entry: (
            entry["saved_mib"],
            entry["original_mib"],
            1 if entry["status"] == "ok" else 0,
        ),
        reverse=True,
    )
    return {
        "num_inputs": len(feed_items),
        "num_candidates": len(candidates),
        "status_counts": dict(status_counts),
        "top_disallowed_consumer_ops": disallowed_consumer_ops.most_common(20),
        "selected_slice_mib": selected_total / (1024.0 * 1024.0),
        "candidates": candidates[:limit],
    }


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


def summarize_xla_dump_dir(dump_dir: Optional[str]):
    if not dump_dir:
        return {}
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return {"dump_dir": str(dump_path), "exists": False}

    txt_files = sorted(dump_path.rglob("*.txt"))
    after_files = sorted(dump_path.rglob("*after_optimizations*.txt"))
    before_files = sorted(dump_path.rglob("*before_optimizations*.txt"))
    parsed_files = after_files or before_files or txt_files
    opcode_counts = Counter()
    pattern_counts = Counter()
    fusion_count = 0
    instruction_count = 0
    instruction_re = re.compile(r"^\s*(?:ROOT\s+)?[A-Za-z0-9_.%-]+ = ")
    opcode_re = re.compile(r"\s([A-Za-z][A-Za-z0-9_-]*)\(")
    fusion_re = re.compile(r"=\s*.*\bfusion\(")

    def classify_opcode(opcode):
        if opcode in ("dot", "convolution"):
            return "matmul_or_convolution"
        if opcode in ("add", "subtract", "multiply", "divide", "maximum", "minimum"):
            return "elementwise_arithmetic"
        if opcode in ("exponential", "log", "sqrt", "rsqrt", "tanh", "logistic"):
            return "elementwise_activation"
        if opcode in ("compare", "select", "and", "or", "not"):
            return "elementwise_predicate"
        if opcode in ("reshape", "transpose", "bitcast", "copy"):
            return "layout_or_shape"
        if opcode in ("slice", "dynamic-slice", "dynamic-update-slice", "gather"):
            return "slice_or_gather"
        if opcode in ("broadcast", "concatenate", "pad"):
            return "broadcast_or_concat"
        if opcode in ("reduce", "reduce-window"):
            return "reduction"
        if opcode in ("fusion",):
            return "fusion"
        if opcode in ("constant", "parameter", "tuple", "get-tuple-element"):
            return "hlo_plumbing"
        return "other"

    for path in parsed_files:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not instruction_re.search(line):
                        continue
                    instruction_count += 1
                    if fusion_re.search(line):
                        fusion_count += 1
                    rhs = line.split(" = ", 1)[1]
                    match = opcode_re.search(rhs)
                    if match:
                        opcode = match.group(1)
                        opcode_counts[opcode] += 1
                        pattern_counts[classify_opcode(opcode)] += 1
        except OSError:
            continue

    summary = {
        "dump_dir": str(dump_path.resolve()),
        "exists": True,
        "hlo_txt_files": len(txt_files),
        "after_optimizations_files": len(after_files),
        "before_optimizations_files": len(before_files),
        "parsed_stage": (
            "after_optimizations"
            if after_files
            else "before_optimizations"
            if before_files
            else "all_txt"
        ),
        "parsed_files": len(parsed_files),
        "parsed_file_samples": [path.name for path in parsed_files[:10]],
        "instruction_count": instruction_count,
        "fusion_ops": fusion_count,
        "top_opcodes": opcode_counts.most_common(20),
        "top_patterns": pattern_counts.most_common(20),
    }
    if parsed_files:
        (dump_path / "xla_dump_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


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


def fingerprint_feed_value(value):
    array = np.asarray(value)
    pointer = int(array.__array_interface__["data"][0])
    if array.dtype.hasobject:
        payload = repr(array.tolist()).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
    else:
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.blake2b(
            memoryview(contiguous).cast("B"), digest_size=16
        ).hexdigest()
    return {
        "pointer": pointer,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "digest": digest,
    }


def snapshot_feed_values(feed_tensors, feed_values):
    if len(feed_tensors) != len(feed_values):
        raise ValueError("feed tensor/value count mismatch")
    return {
        getattr(tensor, "name", str(index)): fingerprint_feed_value(value)
        for index, (tensor, value) in enumerate(zip(feed_tensors, feed_values))
    }


def compare_feed_snapshots(before, after):
    changed_names = []
    changed_bytes = 0
    pointer_changed_names = []
    for name in sorted(set(before) | set(after)):
        before_value = before.get(name)
        after_value = after.get(name)
        if before_value is None or after_value is None:
            changed_names.append(name)
            changed_bytes += int((after_value or before_value or {}).get("nbytes", 0))
            continue
        if before_value["pointer"] != after_value["pointer"]:
            pointer_changed_names.append(name)
        content_fields = ("dtype", "shape", "nbytes", "digest")
        if any(before_value[field] != after_value[field] for field in content_fields):
            changed_names.append(name)
            changed_bytes += int(after_value["nbytes"])
    return {
        "total_inputs": len(after),
        "total_bytes": sum(int(value["nbytes"]) for value in after.values()),
        "changed_inputs": len(changed_names),
        "changed_bytes": changed_bytes,
        "pointer_changed_inputs": len(pointer_changed_names),
        "changed_names": changed_names,
        "pointer_changed_names": pointer_changed_names,
    }


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


SOURCE_FEED_ARENA_ALIGNMENT = 256


def _align_up(value, alignment):
    value = int(value)
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (value + alignment - 1) // alignment * alignment


def build_pinned_source_feed_arena(
    feed_dict, holders, holder_factory=PinnedHostArray, progress=None
):
    if progress is not None:
        progress("begin", inputs=len(feed_dict))
    if not feed_dict:
        return {
            "enabled": False,
            "reason": "empty_feed",
            "inputs": 0,
            "active_bytes": 0,
            "arena_bytes": 0,
            "alignment": SOURCE_FEED_ARENA_ALIGNMENT,
            "base_ptr": 0,
        }

    layouts = []
    active_bytes = 0
    next_offset = 0
    for key, value in feed_dict.items():
        array = np.asarray(value)
        if array.dtype.kind not in ("b", "i", "u", "f", "c"):
            return {
                "enabled": False,
                "reason": f"unsupported_dtype:{array.dtype}",
                "inputs": 0,
                "active_bytes": 0,
                "arena_bytes": 0,
                "alignment": SOURCE_FEED_ARENA_ALIGNMENT,
                "base_ptr": 0,
            }
        if array.nbytes <= 0:
            return {
                "enabled": False,
                "reason": "empty_value",
                "inputs": 0,
                "active_bytes": 0,
                "arena_bytes": 0,
                "alignment": SOURCE_FEED_ARENA_ALIGNMENT,
                "base_ptr": 0,
            }
        contiguous = np.ascontiguousarray(array)
        offset = _align_up(next_offset, SOURCE_FEED_ARENA_ALIGNMENT)
        layouts.append((key, contiguous, offset))
        next_offset = offset + contiguous.nbytes
        active_bytes += contiguous.nbytes

    arena_bytes = _align_up(next_offset, SOURCE_FEED_ARENA_ALIGNMENT)
    if progress is not None:
        progress(
            "layout_ready",
            inputs=len(layouts),
            active_bytes=int(active_bytes),
            arena_bytes=int(arena_bytes),
        )
        progress("alloc_begin", arena_bytes=int(arena_bytes))
    holder = holder_factory((arena_bytes,), np.uint8)
    base_ptr = int(holder.ptr.value or 0)
    if progress is not None:
        progress(
            "alloc_done", arena_bytes=int(arena_bytes), base_ptr=base_ptr
        )
    if base_ptr == 0:
        return {
            "enabled": False,
            "reason": "null_base",
            "inputs": 0,
            "active_bytes": 0,
            "arena_bytes": 0,
            "alignment": SOURCE_FEED_ARENA_ALIGNMENT,
            "base_ptr": 0,
        }

    replacements = {}
    for key, array, offset in layouts:
        view = np.ndarray(
            shape=array.shape,
            dtype=array.dtype,
            buffer=holder.buffer,
            offset=offset,
            order="C",
        )
        view[...] = array
        replacements[key] = view

    if progress is not None:
        progress("populate_done", inputs=len(replacements))

    feed_dict.clear()
    feed_dict.update(replacements)
    holders.append(holder)
    return {
        "enabled": True,
        "reason": "ok",
        "inputs": len(layouts),
        "active_bytes": int(active_bytes),
        "arena_bytes": int(arena_bytes),
        "alignment": SOURCE_FEED_ARENA_ALIGNMENT,
        "base_ptr": base_ptr,
    }


def log_source_feed_arena_progress(phase, **fields):
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(
        f"[INFO] source_feed_arena_progress: phase={phase}{suffix}",
        flush=True,
    )


def log_run_phase_progress(args, phase, **fields):
    if getattr(args, "run_phase_progress", "off") != "on":
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[INFO] run_phase: phase={phase}{suffix}", flush=True)


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
    return should_try_pinned_feed(args) and (
        getattr(args, "pinned_feed_arena", "off") != "on"
    )


def should_use_callable(args):
    if args.use_callable == "on":
        return True
    if args.use_callable == "off":
        return False
    return bool(args.xla and device_kind(args.device) == "MUSA")


def should_pack_small_feed(args, feed_items):
    if args.pack_small_feed == "on":
        return True
    if args.pack_small_feed == "off":
        return False
    if not (args.xla and device_kind(args.device) == "MUSA"):
        return False
    max_total_bytes = int(max(0.0, args.pack_small_feed_max_total_mib) * 1024 * 1024)
    selected_total = 0
    small_count = 0
    candidates = [
        (int(getattr(item["value"], "nbytes", 0) or 0), item)
        for item in feed_items
        if _is_packable_dtype(item["np_dtype"])
    ]
    candidates.sort(key=lambda pair: pair[0])
    for nbytes, _ in candidates:
        if nbytes > args.pack_small_feed_max_bytes:
            continue
        if max_total_bytes > 0 and selected_total + nbytes > max_total_bytes:
            continue
        selected_total += nbytes
        small_count += 1
    min_total_bytes = int(
        max(0.0, args.pack_small_feed_min_total_mib) * 1024 * 1024
    )
    return (
        small_count >= args.pack_small_feed_min_inputs
        and selected_total >= min_total_bytes
    )


def should_pack_remaining_feed(args, feed_items):
    if args.pack_remaining_feed == "on":
        return True
    if args.pack_remaining_feed == "off":
        return False
    if not (args.xla and device_kind(args.device) == "MUSA"):
        return False
    max_total_bytes = int(
        max(0.0, args.pack_remaining_feed_max_total_mib) * 1024 * 1024
    )
    selected_total = 0
    selected_count = 0
    candidates = [
        (int(getattr(item["value"], "nbytes", 0) or 0), item)
        for item in feed_items
        if _is_packable_dtype(item["np_dtype"])
    ]
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    for nbytes, _ in candidates:
        if nbytes > args.pack_remaining_feed_max_bytes:
            continue
        if max_total_bytes > 0 and selected_total + nbytes > max_total_bytes:
            continue
        selected_total += nbytes
        selected_count += 1
    min_total_bytes = int(
        max(0.0, args.pack_remaining_feed_min_total_mib) * 1024 * 1024
    )
    return (
        selected_count >= args.pack_remaining_feed_min_inputs
        and selected_total >= min_total_bytes
    )


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


def should_rewrite_static_shape_subgraph(args):
    return getattr(args, "rewrite_static_shape_subgraph", "off") == "on"


def same_lhs_matmul_auto_decision(args, matmul_analysis=None):
    mode = getattr(args, "rewrite_same_lhs_matmul", "off")
    placeholder_count = 0
    lhs_reduction = 0
    rhs_reduction = 0
    if matmul_analysis:
        placeholder_count = int(matmul_analysis.get("placeholder_count", 0) or 0)
        lhs_reduction = int(
            matmul_analysis.get("estimated_call_reduction_if_grouped_by_lhs", 0)
            or 0
        )
        rhs_reduction = int(
            matmul_analysis.get("estimated_call_reduction_if_grouped_by_rhs", 0)
            or 0
        )
    include_rhs = getattr(args, "rewrite_same_lhs_matmul_include_rhs", "off")
    active_rhs_reduction = rhs_reduction if include_rhs in ("auto", "on") else 0
    total_reduction = lhs_reduction + active_rhs_reduction
    min_placeholders = int(
        getattr(args, "rewrite_same_lhs_matmul_min_placeholders", 0) or 0
    )
    min_reduction = int(
        getattr(args, "rewrite_same_lhs_matmul_auto_min_reduction", 32) or 0
    )
    decision = {
        "mode": mode,
        "enabled": False,
        "reason": "off",
        "placeholder_count": placeholder_count,
        "min_placeholders": min_placeholders,
        "lhs_reduction": lhs_reduction,
        "rhs_reduction": rhs_reduction,
        "active_rhs_reduction": active_rhs_reduction,
        "total_reduction": total_reduction,
        "min_reduction": min_reduction,
        "include_rhs": include_rhs,
    }
    if mode == "on":
        decision["enabled"] = True
        decision["reason"] = "forced_on"
        return decision
    if mode == "off":
        return decision
    if not (getattr(args, "xla", False) and device_kind(args.device) == "MUSA"):
        decision["reason"] = "not_musa_xla"
        return decision
    if matmul_analysis is None:
        decision["reason"] = "missing_analysis"
        return decision
    if min_placeholders > 0 and placeholder_count < min_placeholders:
        decision["reason"] = "placeholder_threshold"
        return decision
    if total_reduction < min_reduction:
        decision["reason"] = "reduction_threshold"
        return decision
    decision["enabled"] = True
    decision["reason"] = "auto"
    return decision


def should_rewrite_same_lhs_matmul(args, matmul_analysis=None):
    if args.rewrite_same_lhs_matmul == "on":
        return True
    if args.rewrite_same_lhs_matmul == "off":
        return False
    return bool(same_lhs_matmul_auto_decision(args, matmul_analysis).get("enabled"))


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
        "static_shape_rewrite": {
            "enabled": False,
            "replaced_shape": 0,
            "replaced_rank": 0,
            "replaced_size": 0,
            "total_replaced": 0,
            "protected_reshape_shape_nodes": 0,
            "skipped": {},
            "sample_shape": [],
            "sample_rank": [],
            "sample_size": [],
        },
        "same_shape_batch_matmul": {
            "enabled": False,
            "groups": 0,
            "original_matmuls": 0,
            "fused_batch_matmuls": 0,
            "estimated_matmul_reduction": 0,
            "rewired_edges": 0,
            "skipped": {},
            "sample_groups": [],
        },
        "same_lhs_matmul": {
            "enabled": False,
            "groups": 0,
            "original_matmuls": 0,
            "fused_matmuls": 0,
            "estimated_matmul_reduction": 0,
            "biasadd_fusion_enabled": False,
            "biasadd_groups": 0,
            "fused_biasadds": 0,
            "estimated_biasadd_reduction": 0,
            "skipped_biasadd_fusion": {},
            "post_unary_fusion_enabled": False,
            "post_unary_groups": 0,
            "fused_post_unary_ops": 0,
            "estimated_post_unary_reduction": 0,
            "skipped_post_unary_fusion": {},
            "post_binary_fusion_enabled": False,
            "post_binary_groups": 0,
            "fused_post_binary_ops": 0,
            "estimated_post_binary_reduction": 0,
            "skipped_post_binary_fusion": {},
            "post_concat_compaction_enabled": False,
            "post_concat_compacted": 0,
            "skipped_post_concat_compaction": {},
            "rewired_edges": 0,
            "skipped": {},
            "sample_groups": [],
        },
        "matmul_group_analysis": None,
        "same_lhs_matmul_auto_decision": None,
        "largest_inputs": items[:20],
    }


def summarize_feed_items(feed_items, bs=None):
    total_bytes = 0
    batch_dim_bytes = 0
    non_batch_dim_bytes = 0
    scalar_bytes = 0
    large_input_count = 0
    large_input_bytes = 0
    for item in feed_items:
        value = item["value"]
        nbytes = int(getattr(value, "nbytes", 0) or 0)
        shape = list(getattr(value, "shape", item.get("shape", [])))
        total_bytes += nbytes
        if not shape:
            scalar_bytes += nbytes
        elif bs is not None and shape[0] == bs:
            batch_dim_bytes += nbytes
        else:
            non_batch_dim_bytes += nbytes
        if nbytes >= 1024 * 1024:
            large_input_count += 1
            large_input_bytes += nbytes
    return {
        "num_inputs": len(feed_items),
        "total_mib": total_bytes / (1024.0 * 1024.0),
        "batch_dim_mib": batch_dim_bytes / (1024.0 * 1024.0),
        "non_batch_dim_mib": non_batch_dim_bytes / (1024.0 * 1024.0),
        "scalar_mib": scalar_bytes / (1024.0 * 1024.0),
        "large_input_count": large_input_count,
        "large_input_mib": large_input_bytes / (1024.0 * 1024.0),
    }


def summarize_remaining_feed_items(feed_items, graph_def, limit=20):
    consumer_by_input = {}
    consumer_names_by_input = {}
    node_map = {node.name: node for node in graph_def.node}
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            consumer_by_input.setdefault(base, Counter())[node.op] += 1
            consumer_names_by_input.setdefault(base, []).append(node.name)

    items = []
    slice_candidates = []
    consumer_hist = Counter()
    total_bytes = 0
    for item in feed_items:
        value = item["value"]
        nbytes = int(getattr(value, "nbytes", 0) or 0)
        total_bytes += nbytes
        consumers = consumer_by_input.get(item["node_name"], Counter())
        consumer_hist.update(consumers)
        consumer_names = consumer_names_by_input.get(item["node_name"], [])
        if consumer_names and set(consumers.keys()).issubset({"Slice", "StridedSlice"}):
            slice_total_bytes = 0
            slice_nodes = []
            status = "ok"
            for consumer_name in consumer_names:
                node = node_map.get(consumer_name)
                if node is None or node.op not in ("Slice", "StridedSlice"):
                    status = "invalid_slice_node"
                    break
                sliced = _slice_like_value_from_node(value, node, node_map)
                if sliced is None:
                    status = "unsupported_or_non_const_slice"
                    break
                sliced_value, sliced_shape = sliced
                slice_bytes = int(getattr(sliced_value, "nbytes", 0) or 0)
                slice_total_bytes += slice_bytes
                slice_nodes.append(
                    {
                        "name": consumer_name,
                        "shape": list(sliced_shape),
                        "mib": slice_bytes / (1024.0 * 1024.0),
                    }
                )
            saved_bytes = nbytes - slice_total_bytes if status == "ok" else 0
            slice_candidates.append(
                {
                    "name": item["name"],
                    "shape": list(item["shape"]),
                    "original_mib": nbytes / (1024.0 * 1024.0),
                    "slice_mib": slice_total_bytes / (1024.0 * 1024.0),
                    "saved_mib": saved_bytes / (1024.0 * 1024.0),
                    "num_slice_nodes": len(consumer_names),
                    "status": status,
                    "slice_nodes": slice_nodes[:5],
                }
            )
        items.append(
            {
                "name": item["name"],
                "node_name": item["node_name"],
                "dtype": str(item["np_dtype"]),
                "shape": list(item["shape"]),
                "nbytes": nbytes,
                "mib": nbytes / (1024.0 * 1024.0),
                "consumer_ops": consumers.most_common(8),
            }
        )
    items.sort(key=lambda entry: entry["nbytes"], reverse=True)
    return {
        "num_inputs": len(feed_items),
        "total_mib": total_bytes / (1024.0 * 1024.0),
        "top_consumer_ops": consumer_hist.most_common(20),
        "largest_inputs": items[:limit],
        "top_slice_candidates": sorted(
            slice_candidates,
            key=lambda entry: (entry["saved_mib"], entry["original_mib"]),
            reverse=True,
        )[:limit],
    }


def summarize_output_dependencies(graph_def, output_spec, limit=20):
    node_map = {node.name: node for node in graph_def.node}

    def tensor_shape_for_name(tensor_name):
        node_name = _strip_tensor_name(tensor_name)
        node = node_map.get(node_name)
        return _shape_from_node_attr(node) if node is not None else None

    def tensor_dtype_for_name(tensor_name):
        node_name = _strip_tensor_name(tensor_name)
        node = node_map.get(node_name)
        if node is None or "dtype" not in node.attr:
            return None
        try:
            return tf.as_dtype(node.attr["dtype"].type)
        except Exception:
            return None

    def describe_inputs(node):
        if node is None:
            return []
        out = []
        for input_name in node.input:
            if input_name.startswith("^"):
                continue
            input_node_name = _strip_tensor_name(input_name)
            input_node = node_map.get(input_node_name)
            shape = tensor_shape_for_name(input_name)
            dtype = tensor_dtype_for_name(input_name)
            estimated_nbytes = (
                estimate_tensor_nbytes(shape, dtype, bs=None)
                if dtype is not None
                else None
            )
            out.append(
                {
                    "name": _ensure_tensor_name(input_name),
                    "node": input_node_name,
                    "op": input_node.op if input_node is not None else None,
                    "shape": shape,
                    "dtype": dtype.name if dtype is not None else None,
                    "estimated_mib": (
                        estimated_nbytes / (1024.0 * 1024.0)
                        if estimated_nbytes is not None
                        else None
                    ),
                }
            )
        return out

    def describe_input_tree(node, depth=2):
        if node is None or depth <= 0:
            return []
        tree = []
        for item in describe_inputs(node):
            child = node_map.get(item["node"])
            expanded = dict(item)
            if child is not None and depth > 1:
                expanded["inputs"] = describe_input_tree(child, depth - 1)
            tree.append(expanded)
        return tree

    def trace_main_chain(node_name, max_depth=24):
        chain = []
        seen = set()
        current = node_name
        for _ in range(max_depth):
            if current in seen:
                chain.append({"name": current, "op": "cycle"})
                break
            seen.add(current)
            node = node_map.get(current)
            if node is None:
                chain.append({"name": current, "op": "missing"})
                break
            chain.append(
                {
                    "name": node.name,
                    "op": node.op,
                    "num_inputs": len(node.input),
                    "inputs": [_strip_tensor_name(name) for name in node.input[:6]],
                }
            )
            data_inputs = [
                _strip_tensor_name(name)
                for name in node.input
                if not name.startswith("^")
            ]
            if len(data_inputs) != 1:
                break
            current = data_inputs[0]
        return chain

    summaries = []
    for output_name in output_spec:
        root = _strip_tensor_name(output_name)
        reachable = set()
        stack = [root]
        op_hist = Counter()
        edge_count = 0
        placeholder_count = 0
        while stack:
            name = stack.pop()
            if name in reachable:
                continue
            node = node_map.get(name)
            if node is None:
                continue
            reachable.add(name)
            op_hist[node.op] += 1
            if node.op in ("Placeholder", "PlaceholderWithDefault"):
                placeholder_count += 1
            for input_name in node.input:
                edge_count += 1
                stack.append(_strip_tensor_name(input_name))
        root_node = node_map.get(root)
        summaries.append(
            {
                "output": output_name,
                "root_name": root,
                "root_op": root_node.op if root_node is not None else None,
                "reachable_nodes": len(reachable),
                "reachable_edges": edge_count,
                "placeholder_count": placeholder_count,
                "top_ops": op_hist.most_common(limit),
                "root_inputs": describe_inputs(root_node),
                "root_input_tree": describe_input_tree(root_node, depth=3),
                "main_chain": trace_main_chain(root),
            }
        )
    return summaries


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
    non_const_exponent_ops = Counter()
    non_const_exponent_samples = []
    non_square_constant_values = Counter()
    non_square_constant_samples = []
    total_pow = 0
    for node in graph_def.node:
        if node.op != "Pow" or len(node.input) < 2:
            continue
        total_pow += 1
        exponent_input = node.input[1]
        exponent_node = node_map.get(_strip_tensor_name(node.input[1]))
        exponent = _const_float_array(exponent_node)
        if exponent is None:
            skipped_non_const += 1
            exponent_op = exponent_node.op if exponent_node is not None else "<missing>"
            non_const_exponent_ops[exponent_op] += 1
            if len(non_const_exponent_samples) < 20:
                non_const_exponent_samples.append(
                    {
                        "pow": node.name,
                        "exponent_input": exponent_input,
                        "exponent_node": (
                            exponent_node.name if exponent_node is not None else None
                        ),
                        "exponent_op": exponent_op,
                    }
                )
            continue
        if exponent.size == 0 or not np.allclose(exponent, 2.0):
            skipped_non_square += 1
            flat_values = np.asarray(exponent).reshape(-1)
            value_text = ",".join(
                f"{float(value):.9g}" for value in flat_values[:8]
            )
            if flat_values.size > 8:
                value_text += ",..."
            value_key = f"shape={list(exponent.shape)} values=[{value_text}]"
            non_square_constant_values[value_key] += 1
            if len(non_square_constant_samples) < 20:
                non_square_constant_samples.append(
                    {
                        "pow": node.name,
                        "exponent_input": exponent_input,
                        "exponent_node": exponent_node.name,
                        "value": value_key,
                    }
                )
            continue
        base_input = node.input[0]
        control_inputs = [input_name for input_name in node.input[2:] if input_name.startswith("^")]
        node.op = "Mul"
        del node.input[:]
        node.input.extend([base_input, base_input] + control_inputs)
        rewritten.append(node.name)
    return {
        "enabled": True,
        "total_pow": total_pow,
        "rewritten": len(rewritten),
        "sample": rewritten[:20],
        "skipped_non_const": skipped_non_const,
        "skipped_non_square": skipped_non_square,
        "non_const_exponent_ops": non_const_exponent_ops.most_common(20),
        "non_const_exponent_samples": non_const_exponent_samples,
        "non_square_constant_values": non_square_constant_values.most_common(20),
        "non_square_constant_samples": non_square_constant_samples,
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


def _static_shape_for_tensor_name(tensor_name, node_map, spec_shape_map, bs, unknown_dim):
    base, port = _tensor_name_parts(tensor_name)
    if port != 0:
        return None
    spec_shape = spec_shape_map.get(f"{base}:0") or spec_shape_map.get(base)
    node = node_map.get(base)
    node_shape = _shape_from_node_attr(node) if node is not None else None
    shape = merge_shape(spec_shape, node_shape)
    if shape is None:
        return None
    resolved = resolve_shape(shape, bs, unknown_dim)
    if any(dim is None for dim in resolved):
        return None
    return [int(dim) for dim in resolved]


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


def rewrite_static_shape_subgraph_nodes(graph_def, spec_shape_map, bs, args):
    node_map = {node.name: node for node in graph_def.node}
    protected_shape_nodes = _static_shape_nodes_feeding_reshape(graph_def, node_map)
    replaced_shape = []
    replaced_rank = []
    replaced_size = []
    skipped = Counter()
    for node in graph_def.node:
        if node.op not in ("Shape", "Rank", "Size"):
            continue
        if node.name in protected_shape_nodes:
            skipped[f"{node.op}:feeds_reshape"] += 1
            continue
        if not node.input:
            skipped[f"{node.op}:missing_input"] += 1
            continue
        shape = _static_shape_for_tensor_name(
            node.input[0],
            node_map,
            spec_shape_map,
            bs,
            getattr(args, "unknown_dim", 1),
        )
        if shape is None:
            skipped[f"{node.op}:unknown_shape"] += 1
            continue
        dtype = _const_output_dtype(node)
        if node.op == "Shape":
            _replace_node_with_const(node, np.asarray(shape, dtype=dtype.as_numpy_dtype), dtype)
            replaced_shape.append(node.name)
        elif node.op == "Rank":
            _replace_node_with_const(node, np.asarray(len(shape), dtype=dtype.as_numpy_dtype), dtype)
            replaced_rank.append(node.name)
        elif node.op == "Size":
            size = int(np.prod(shape)) if shape else 1
            _replace_node_with_const(node, np.asarray(size, dtype=dtype.as_numpy_dtype), dtype)
            replaced_size.append(node.name)
    return {
        "enabled": True,
        "replaced_shape": len(replaced_shape),
        "replaced_rank": len(replaced_rank),
        "replaced_size": len(replaced_size),
        "total_replaced": len(replaced_shape) + len(replaced_rank) + len(replaced_size),
        "protected_reshape_shape_nodes": len(protected_shape_nodes),
        "skipped": dict(skipped),
        "sample_shape": replaced_shape[:20],
        "sample_rank": replaced_rank[:20],
        "sample_size": replaced_size[:20],
    }


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
    fuse_post_unary = (
        getattr(args, "rewrite_same_lhs_matmul_fuse_post_unary", "off") == "on"
    )
    fuse_post_binary = (
        getattr(args, "rewrite_same_lhs_matmul_fuse_post_binary", "off") == "on"
    )
    compact_post_concat = (
        getattr(args, "rewrite_same_lhs_matmul_compact_post_concat", "off") == "on"
    )
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
            "post_unary_fusion_enabled": fuse_post_unary,
            "post_unary_groups": 0,
            "fused_post_unary_ops": 0,
            "estimated_post_unary_reduction": 0,
            "skipped_post_unary_fusion": {},
            "post_binary_fusion_enabled": fuse_post_binary,
            "post_binary_groups": 0,
            "fused_post_binary_ops": 0,
            "estimated_post_binary_reduction": 0,
            "skipped_post_binary_fusion": {},
            "post_concat_compaction_enabled": compact_post_concat,
            "post_concat_compacted": 0,
            "skipped_post_concat_compaction": {},
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
    rhs_groups = defaultdict(list)
    skipped = Counter()
    skipped_biasadd_fusion = Counter()
    skipped_post_unary_fusion = Counter()
    skipped_post_binary_fusion = Counter()
    skipped_post_concat_compaction = Counter()
    allowed_rhs_producer_ops = parse_csv_filter(
        args.rewrite_same_lhs_matmul_rhs_producer_ops
    )
    post_unary_ops = {
        "Abs",
        "Elu",
        "Exp",
        "Log",
        "Neg",
        "Relu",
        "Relu6",
        "Rsqrt",
        "Sigmoid",
        "Sqrt",
        "Square",
        "Tanh",
    }
    post_binary_ops = {
        "Add",
        "AddV2",
        "Sub",
        "Mul",
        "RealDiv",
        "Maximum",
        "Minimum",
    }
    commutative_post_binary_ops = {"Add", "AddV2", "Mul", "Maximum", "Minimum"}

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

    def _find_unique_post_unary_for_biasadd(biasadd_node, dtype_enum):
        consumers = data_consumers.get(biasadd_node.name, [])
        if len(consumers) != 1:
            return None, "biasadd_not_unique_data_consumer"
        unary_node = node_map.get(consumers[0])
        if unary_node is None:
            return None, "post_unary_missing"
        if unary_node.op not in post_unary_ops:
            return None, f"post_op_not_allowed:{unary_node.op}"
        if any(input_name.startswith("^") for input_name in unary_node.input):
            return None, "post_unary_control_input"
        if control_consumers.get(unary_node.name):
            return None, "post_unary_control_consumer"
        if len(unary_node.input) != 1:
            return None, "post_unary_not_single_input"
        value_base, value_port = _tensor_name_parts(unary_node.input[0])
        if value_base != biasadd_node.name or value_port != 0:
            return None, "post_unary_value_not_biasadd"
        if "T" in unary_node.attr and unary_node.attr["T"].type != dtype_enum:
            return None, "post_unary_dtype_mismatch"
        return unary_node, None

    def _post_binary_other_kind(other_shape, out_rows, out_cols):
        if other_shape is None:
            return None, "post_binary_other_shape_unknown"
        if len(other_shape) == 1:
            cols = other_shape[0]
            if isinstance(cols, int) and cols == out_cols:
                return "vector", None
            return None, "post_binary_other_vector_cols_mismatch"
        if len(other_shape) == 2:
            rows, cols = other_shape
            if not isinstance(cols, int) or cols != out_cols:
                return None, "post_binary_other_matrix_cols_mismatch"
            if rows == 1:
                return "row_matrix", None
            if isinstance(rows, int) and rows == out_rows:
                return "matrix", None
            return None, "post_binary_other_matrix_rows_mismatch"
        return None, "post_binary_other_rank_unsupported"

    def _find_unique_post_binary_for_biasadd(biasadd_node, item, dtype_enum):
        consumers = data_consumers.get(biasadd_node.name, [])
        if len(consumers) != 1:
            return None, "biasadd_not_unique_data_consumer"
        binary_node = node_map.get(consumers[0])
        if binary_node is None:
            return None, "post_binary_missing"
        if binary_node.op not in post_binary_ops:
            return None, f"post_op_not_allowed:{binary_node.op}"
        if any(input_name.startswith("^") for input_name in binary_node.input):
            return None, "post_binary_control_input"
        if control_consumers.get(binary_node.name):
            return None, "post_binary_control_consumer"
        if len(binary_node.input) != 2:
            return None, "post_binary_not_two_inputs"
        biasadd_operand_positions = []
        for position, input_name in enumerate(binary_node.input):
            value_base, value_port = _tensor_name_parts(input_name)
            if value_base == biasadd_node.name and value_port == 0:
                biasadd_operand_positions.append(position)
        if len(biasadd_operand_positions) != 1:
            return None, "post_binary_value_not_biasadd"
        biasadd_position = biasadd_operand_positions[0]
        other_position = 1 - biasadd_position
        post_binary_operand_position = (
            0
            if binary_node.op in commutative_post_binary_ops
            else biasadd_position
        )
        if "T" in binary_node.attr and binary_node.attr["T"].type != dtype_enum:
            return None, "post_binary_dtype_mismatch"
        other_input = binary_node.input[other_position]
        other_shape = _tensor_shape_for_analysis(other_input, node_map, spec_shape_map)
        other_kind, reason = _post_binary_other_kind(
            other_shape, item["out_rows"], item["out_cols"]
        )
        if other_kind is None:
            return None, reason
        return {
            "node": binary_node,
            "other_input": other_input,
            "other_kind": other_kind,
            "post_binary_operand_position": post_binary_operand_position,
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
        if args.rewrite_same_lhs_matmul_include_rhs in ("auto", "on"):
            rhs_key = (rhs_input, node.attr["T"].type, k_dim, out_cols)
            rhs_groups[rhs_key].append(item)

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

    if args.rewrite_same_lhs_matmul_include_rhs in ("auto", "on"):
        for (rhs_input, dtype_enum, k_dim, out_cols), items in rhs_groups.items():
            if len(items) < args.rewrite_same_lhs_matmul_min_group:
                continue
            ordered_items = sorted(items, key=lambda item: item["out_rows"], reverse=True)
            max_total_rows = args.rewrite_same_lhs_matmul_max_total_rows
            chunks = []
            if max_total_rows <= 0:
                chunks = [ordered_items]
            else:
                current = []
                current_rows = 0
                for item in ordered_items:
                    item_rows = item["out_rows"]
                    if item_rows > max_total_rows:
                        skipped["single_item_total_rows_cap"] += 1
                        continue
                    if current and current_rows + item_rows > max_total_rows:
                        chunks.append(current)
                        current = []
                        current_rows = 0
                    current.append(item)
                    current_rows += item_rows
                if current:
                    chunks.append(current)
            for chunk_index, chunk in enumerate(chunks):
                if len(chunk) < args.rewrite_same_lhs_matmul_min_group:
                    skipped["rhs_chunk_too_small"] += 1
                    continue
                candidates.append(
                    {
                        "mode": "same_rhs",
                        "rhs_input": rhs_input,
                        "dtype_enum": dtype_enum,
                        "k_dim": k_dim,
                        "total_cols": out_cols,
                        "total_rows": sum(item["out_rows"] for item in chunk),
                        "estimated_flops": (
                            2
                            * sum(item["out_rows"] for item in chunk)
                            * k_dim
                            * out_cols
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
        if candidate["mode"] == "same_lhs":
            candidate["total_cols"] = sum(item["out_cols"] for item in available_items)
            candidate["estimated_flops"] = (
                2
                * candidate["total_rows"]
                * candidate["k_dim"]
                * candidate["total_cols"]
            )
        else:
            candidate["total_rows"] = sum(item["out_rows"] for item in available_items)
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
    post_concat_replacement_by_node = {}
    fused_groups = []
    for group_index, group in enumerate(candidates):
        items = group["items"]
        prefix = f"musa_same_lhs_matmul_group_{group_index}"
        concat_axis = 1 if group["mode"] == "same_lhs" else 0
        axis_node = _add_const_node(
            graph_def, f"{prefix}_concat_axis", concat_axis
        )

        concat_node = graph_def.node.add()
        concat_node.name = f"{prefix}_concat"
        concat_node.op = "ConcatV2"
        if group["mode"] == "same_lhs":
            concat_node.input.extend([item["rhs_input"] for item in items])
        else:
            concat_node.input.extend([item["lhs_input"] for item in items])
        concat_node.input.append(axis_node.name)
        concat_node.attr["N"].i = len(items)
        concat_node.attr["T"].type = group["dtype_enum"]
        concat_node.attr["Tidx"].type = tf.int32.as_datatype_enum

        matmul_node = graph_def.node.add()
        matmul_node.name = f"{prefix}_matmul"
        matmul_node.op = "MatMul"
        if group["mode"] == "same_lhs":
            matmul_node.input.extend([group["lhs_input"], concat_node.name])
        else:
            matmul_node.input.extend([concat_node.name, group["rhs_input"]])
        _copy_node_attr(matmul_node, items[0]["node"], ["T"])
        matmul_node.attr["transpose_a"].b = False
        matmul_node.attr["transpose_b"].b = False

        biasadd_by_matmul = {}
        fuse_group_biasadd = False
        fused_biasadd_node = None
        post_unary_by_matmul = {}
        post_unary_op = ""
        fuse_group_post_unary = False
        fused_post_unary_node = None
        post_binary_by_matmul = {}
        post_binary_op = ""
        post_binary_operand_position = 0
        post_binary_other_kind = ""
        fuse_group_post_binary = False
        fused_post_binary_node = None
        post_concat_source = ""
        if fuse_biasadd and group["mode"] == "same_lhs":
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
                if fuse_post_unary:
                    for item in items:
                        biasadd_node = biasadd_by_matmul[item["node"].name]["node"]
                        unary_node, reason = _find_unique_post_unary_for_biasadd(
                            biasadd_node, group["dtype_enum"]
                        )
                        if unary_node is None:
                            skipped_post_unary_fusion[reason] += 1
                        else:
                            post_unary_by_matmul[item["node"].name] = unary_node
                    post_unary_ops_in_group = {
                        node.op for node in post_unary_by_matmul.values()
                    }
                    fuse_group_post_unary = (
                        len(post_unary_by_matmul) == len(items)
                        and len(post_unary_ops_in_group) == 1
                    )
                    if (
                        len(post_unary_by_matmul) == len(items)
                        and len(post_unary_ops_in_group) > 1
                    ):
                        skipped_post_unary_fusion["post_unary_op_mismatch"] += len(items)
                    if fuse_group_post_unary:
                        post_unary_op = next(iter(post_unary_ops_in_group))
                        first_unary = post_unary_by_matmul[items[0]["node"].name]
                        fused_post_unary_node = graph_def.node.add()
                        fused_post_unary_node.name = f"{prefix}_post_{post_unary_op.lower()}"
                        fused_post_unary_node.op = post_unary_op
                        fused_post_unary_node.input.append(fused_biasadd_node.name)
                        _copy_node_attr(fused_post_unary_node, first_unary, ["T"])
                if fuse_post_binary:
                    for item in items:
                        biasadd_node = biasadd_by_matmul[item["node"].name]["node"]
                        binary_info, reason = _find_unique_post_binary_for_biasadd(
                            biasadd_node, item, group["dtype_enum"]
                        )
                        if binary_info is None:
                            skipped_post_binary_fusion[reason] += 1
                        else:
                            post_binary_by_matmul[item["node"].name] = binary_info
                    post_binary_ops_in_group = {
                        info["node"].op for info in post_binary_by_matmul.values()
                    }
                    post_binary_positions_in_group = {
                        info["post_binary_operand_position"]
                        for info in post_binary_by_matmul.values()
                    }
                    post_binary_other_kinds_in_group = {
                        info["other_kind"] for info in post_binary_by_matmul.values()
                    }
                    fuse_group_post_binary = (
                        len(post_binary_by_matmul) == len(items)
                        and len(post_binary_ops_in_group) == 1
                        and len(post_binary_positions_in_group) == 1
                        and len(post_binary_other_kinds_in_group) == 1
                    )
                    if (
                        len(post_binary_by_matmul) == len(items)
                        and len(post_binary_ops_in_group) > 1
                    ):
                        skipped_post_binary_fusion["post_binary_op_mismatch"] += len(items)
                    if (
                        len(post_binary_by_matmul) == len(items)
                        and len(post_binary_positions_in_group) > 1
                    ):
                        skipped_post_binary_fusion[
                            "post_binary_operand_position_mismatch"
                        ] += len(items)
                    if (
                        len(post_binary_by_matmul) == len(items)
                        and len(post_binary_other_kinds_in_group) > 1
                    ):
                        skipped_post_binary_fusion[
                            "post_binary_other_kind_mismatch"
                        ] += len(items)
                    if fuse_group_post_binary:
                        post_binary_op = next(iter(post_binary_ops_in_group))
                        post_binary_operand_position = next(
                            iter(post_binary_positions_in_group)
                        )
                        post_binary_other_kind = next(
                            iter(post_binary_other_kinds_in_group)
                        )
                        other_axis_node = _add_const_node(
                            graph_def,
                            f"{prefix}_post_binary_other_concat_axis",
                            0 if post_binary_other_kind == "vector" else 1,
                        )
                        other_concat_node = graph_def.node.add()
                        other_concat_node.name = f"{prefix}_post_binary_other_concat"
                        other_concat_node.op = "ConcatV2"
                        other_concat_node.input.extend(
                            [
                                post_binary_by_matmul[item["node"].name][
                                    "other_input"
                                ]
                                for item in items
                            ]
                        )
                        other_concat_node.input.append(other_axis_node.name)
                        other_concat_node.attr["N"].i = len(items)
                        other_concat_node.attr["T"].type = group["dtype_enum"]
                        other_concat_node.attr["Tidx"].type = tf.int32.as_datatype_enum

                        first_binary = post_binary_by_matmul[items[0]["node"].name][
                            "node"
                        ]
                        fused_post_binary_node = graph_def.node.add()
                        fused_post_binary_node.name = (
                            f"{prefix}_post_{post_binary_op.lower()}"
                        )
                        fused_post_binary_node.op = post_binary_op
                        if post_binary_operand_position == 0:
                            fused_post_binary_node.input.extend(
                                [fused_biasadd_node.name, other_concat_node.name]
                            )
                        else:
                            fused_post_binary_node.input.extend(
                                [other_concat_node.name, fused_biasadd_node.name]
                            )
                        _copy_node_attr(fused_post_binary_node, first_binary, ["T"])
        elif fuse_biasadd:
            skipped_biasadd_fusion["same_rhs_not_supported"] += len(items)
        elif fuse_post_unary:
            skipped_post_unary_fusion["biasadd_fusion_not_enabled"] += len(items)
        if (not fuse_biasadd) and fuse_post_binary:
            skipped_post_binary_fusion["biasadd_fusion_not_enabled"] += len(items)

        offset = 0
        slice_nodes = []
        bias_slice_nodes = []
        post_unary_slice_nodes = []
        post_binary_slice_nodes = []
        item_offsets = {}
        for item_index, item in enumerate(items):
            width = item["out_cols"]
            rows = item["out_rows"]
            if group["mode"] == "same_lhs":
                begin = [0, offset]
                size = [-1, width]
                item_offsets[item["node"].name] = {
                    "index": item_index,
                    "offset": offset,
                    "width": width,
                }
                offset += width
            else:
                begin = [offset, 0]
                size = [rows, -1]
                offset += rows
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
                if fuse_group_post_unary and fused_post_unary_node is not None:
                    post_unary_slice_node = graph_def.node.add()
                    post_unary_slice_node.name = f"{prefix}_post_unary_slice_{item_index}"
                    post_unary_slice_node.op = "Slice"
                    post_unary_slice_node.input.extend(
                        [fused_post_unary_node.name, begin_node.name, size_node.name]
                    )
                    post_unary_slice_node.attr["T"].type = group["dtype_enum"]
                    post_unary_slice_node.attr["Index"].type = tf.int32.as_datatype_enum
                    unary_node = post_unary_by_matmul[item["node"].name]
                    replacement_by_node[unary_node.name] = post_unary_slice_node.name
                    post_unary_slice_nodes.append(post_unary_slice_node.name)
                if fuse_group_post_binary and fused_post_binary_node is not None:
                    post_binary_slice_node = graph_def.node.add()
                    post_binary_slice_node.name = f"{prefix}_post_binary_slice_{item_index}"
                    post_binary_slice_node.op = "Slice"
                    post_binary_slice_node.input.extend(
                        [fused_post_binary_node.name, begin_node.name, size_node.name]
                    )
                    post_binary_slice_node.attr["T"].type = group["dtype_enum"]
                    post_binary_slice_node.attr["Index"].type = tf.int32.as_datatype_enum
                    binary_node = post_binary_by_matmul[item["node"].name]["node"]
                    replacement_by_node[binary_node.name] = post_binary_slice_node.name
                    post_binary_slice_nodes.append(post_binary_slice_node.name)

        post_concat_compacted_nodes = []
        if compact_post_concat:
            concat_source_node = None
            concat_input_to_item = {}
            concat_input_not_group_reason = "concat_input_not_group"
            if fuse_group_post_unary and fused_post_unary_node is not None:
                post_concat_source = "post_unary"
                concat_source_node = fused_post_unary_node
                concat_input_not_group_reason = "concat_input_not_group_post_unary"
                concat_input_to_item = {
                    post_unary_by_matmul[item["node"].name].name: item
                    for item in items
                    if item["node"].name in post_unary_by_matmul
                }
            elif fuse_group_biasadd and fused_biasadd_node is not None:
                post_concat_source = "biasadd"
                concat_source_node = fused_biasadd_node
                concat_input_not_group_reason = "concat_input_not_group_biasadd"
                concat_input_to_item = {
                    biasadd_by_matmul[item["node"].name]["node"].name: item
                    for item in items
                    if item["node"].name in biasadd_by_matmul
                }
            else:
                skipped_post_concat_compaction["biasadd_not_fused"] += len(items)
            if concat_source_node is not None and concat_input_to_item:
                candidate_concat_names = []
                seen_concat_names = set()
                for source_input_name in concat_input_to_item:
                    for consumer_name in data_consumers.get(source_input_name, []):
                        if consumer_name not in seen_concat_names:
                            seen_concat_names.add(consumer_name)
                            candidate_concat_names.append(consumer_name)
                for concat_index, concat_name in enumerate(candidate_concat_names):
                    concat_node = node_map.get(concat_name)
                    if concat_node is None or concat_node.op != "ConcatV2":
                        skipped_post_concat_compaction["consumer_not_concatv2"] += 1
                        continue
                    if len(concat_node.input) < 3:
                        skipped_post_concat_compaction["concat_too_few_inputs"] += 1
                        continue
                    axis_values = _const_int_list(
                        node_map.get(_strip_tensor_name(concat_node.input[-1]))
                    )
                    if not axis_values:
                        skipped_post_concat_compaction["concat_axis_unknown"] += 1
                        continue
                    axis = axis_values[0]
                    if axis == -1:
                        axis = 1
                    if axis != 1:
                        skipped_post_concat_compaction["concat_axis_not_cols"] += 1
                        continue
                    concat_items = []
                    for input_name in concat_node.input[:-1]:
                        if input_name.startswith("^"):
                            concat_items = []
                            skipped_post_concat_compaction["concat_control_input"] += 1
                            break
                        base, port = _tensor_name_parts(input_name)
                        if port != 0 or base not in concat_input_to_item:
                            concat_items = []
                            skipped_post_concat_compaction[
                                concat_input_not_group_reason
                            ] += 1
                            break
                        concat_items.append(concat_input_to_item[base])
                    if not concat_items:
                        continue
                    offsets = [
                        item_offsets[item["node"].name] for item in concat_items
                    ]
                    indices = [item["index"] for item in offsets]
                    if indices != list(range(indices[0], indices[0] + len(indices))):
                        skipped_post_concat_compaction["concat_inputs_not_contiguous"] += 1
                        continue
                    start_offset = offsets[0]["offset"]
                    total_width = sum(item["width"] for item in offsets)
                    compact_begin_node = _add_const_node(
                        graph_def,
                        f"{prefix}_post_concat_{concat_index}_begin",
                        [0, start_offset],
                    )
                    compact_size_node = _add_const_node(
                        graph_def,
                        f"{prefix}_post_concat_{concat_index}_size",
                        [-1, total_width],
                    )
                    compact_slice_node = graph_def.node.add()
                    compact_slice_node.name = f"{prefix}_post_concat_{concat_index}_slice"
                    compact_slice_node.op = "Slice"
                    compact_slice_node.input.extend(
                        [
                            concat_source_node.name,
                            compact_begin_node.name,
                            compact_size_node.name,
                        ]
                    )
                    compact_slice_node.attr["T"].type = group["dtype_enum"]
                    compact_slice_node.attr["Index"].type = tf.int32.as_datatype_enum
                    post_concat_replacement_by_node[concat_node.name] = (
                        compact_slice_node.name
                    )
                    post_concat_compacted_nodes.append(compact_slice_node.name)

        fused_groups.append(
            {
                "mode": group["mode"],
                "lhs": group.get("lhs_input", ""),
                "rhs": group.get("rhs_input", ""),
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
                "post_unary_fused": bool(fuse_group_post_unary),
                "post_unary_op": post_unary_op,
                "fused_post_unary": (
                    fused_post_unary_node.name
                    if fused_post_unary_node is not None
                    else ""
                ),
                "post_unary_slice_nodes": post_unary_slice_nodes[:8],
                "post_binary_fused": bool(fuse_group_post_binary),
                "post_binary_op": post_binary_op,
                "post_binary_operand_position": post_binary_operand_position,
                "post_binary_other_kind": post_binary_other_kind,
                "fused_post_binary": (
                    fused_post_binary_node.name
                    if fused_post_binary_node is not None
                    else ""
                ),
                "post_binary_slice_nodes": post_binary_slice_nodes[:8],
                "post_concat_source": post_concat_source,
                "post_concat_compacted": len(post_concat_compacted_nodes),
                "post_concat_compacted_nodes": post_concat_compacted_nodes[:8],
            }
        )

    rewired_edges = 0
    if post_concat_replacement_by_node:
        replacement_by_node.update(post_concat_replacement_by_node)
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
        "post_unary_fusion_enabled": fuse_post_unary,
        "post_unary_groups": sum(
            1 for group in fused_groups if group["post_unary_fused"]
        ),
        "fused_post_unary_ops": sum(
            group["matmul_count"]
            for group in fused_groups
            if group["post_unary_fused"]
        ),
        "estimated_post_unary_reduction": sum(
            group["matmul_count"] - 1
            for group in fused_groups
            if group["post_unary_fused"]
        ),
        "skipped_post_unary_fusion": dict(skipped_post_unary_fusion),
        "post_binary_fusion_enabled": fuse_post_binary,
        "post_binary_groups": sum(
            1 for group in fused_groups if group["post_binary_fused"]
        ),
        "fused_post_binary_ops": sum(
            group["matmul_count"]
            for group in fused_groups
            if group["post_binary_fused"]
        ),
        "estimated_post_binary_reduction": sum(
            group["matmul_count"] - 1
            for group in fused_groups
            if group["post_binary_fused"]
        ),
        "skipped_post_binary_fusion": dict(skipped_post_binary_fusion),
        "post_concat_compaction_enabled": compact_post_concat,
        "post_concat_compacted": sum(
            group["post_concat_compacted"] for group in fused_groups
        ),
        "post_concat_sources": dict(
            Counter(
                group["post_concat_source"]
                for group in fused_groups
                if group["post_concat_source"]
            )
        ),
        "skipped_post_concat_compaction": dict(skipped_post_concat_compaction),
        "rewired_edges": rewired_edges,
        "skipped": dict(skipped),
        "sample_groups": fused_groups[:10],
    }, replacement_by_node


def rewrite_same_shape_batch_matmul_nodes(graph_def, spec_shape_map, bs, args):
    if getattr(args, "rewrite_same_shape_batch_matmul", "off") != "on":
        return {
            "enabled": False,
            "groups": 0,
            "original_matmuls": 0,
            "fused_batch_matmuls": 0,
            "estimated_matmul_reduction": 0,
            "rewired_edges": 0,
            "skipped": {},
            "sample_groups": [],
        }, {}

    original_node_count = len(graph_def.node)
    node_map = {node.name: node for node in graph_def.node}
    node_index = {node.name: index for index, node in enumerate(graph_def.node)}
    data_consumers = defaultdict(list)
    control_consumers = defaultdict(list)
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            if input_name.startswith("^"):
                control_consumers[base].append(node.name)
            else:
                data_consumers[base].append(node.name)

    skipped = Counter()
    groups = defaultdict(list)
    matmul_ops = ("MatMul", "BatchMatMul", "BatchMatMulV2")

    def _upstream_matmul_names(input_name, max_nodes=4096):
        stack = [_strip_tensor_name(input_name)]
        visited = set()
        matmul_names = set()
        while stack and len(visited) < max_nodes:
            name = stack.pop()
            if name in visited:
                continue
            visited.add(name)
            producer = node_map.get(name)
            if producer is None:
                continue
            if producer.op in matmul_ops:
                matmul_names.add(producer.name)
            for producer_input in producer.input:
                stack.append(_strip_tensor_name(producer_input))
        if len(visited) >= max_nodes:
            matmul_names.add("__upstream_limit_exceeded__")
        return matmul_names

    for node in list(graph_def.node):
        if node.op != "MatMul" or len(node.input) < 2:
            continue
        if node.name.startswith("musa_same_lhs_matmul_group_"):
            skipped["same_lhs_generated_matmul"] += 1
            continue
        if not data_consumers.get(node.name):
            skipped["no_data_consumer"] += 1
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
        direct_matmul_inputs = {
            producer.name
            for producer in (lhs_producer, rhs_producer)
            if producer is not None and producer.op in matmul_ops
        }
        upstream_matmuls = set(direct_matmul_inputs)
        upstream_matmuls.update(_upstream_matmul_names(lhs_input))
        upstream_matmuls.update(_upstream_matmul_names(rhs_input))
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
        if not all(isinstance(dim, int) for dim in (out_rows, k_dim, out_cols)):
            skipped["unknown_contract"] += 1
            continue
        if out_rows <= 0 or k_dim <= 0 or out_cols <= 0:
            skipped["bad_shape"] += 1
            continue
        key = (
            node.attr["T"].type,
            tuple(lhs_shape),
            tuple(rhs_shape),
            out_rows,
            k_dim,
            out_cols,
        )
        groups[key].append(
            {
                "node": node,
                "lhs_input": lhs_input,
                "rhs_input": rhs_input,
                "out_rows": out_rows,
                "out_cols": out_cols,
                "k_dim": k_dim,
                "node_index": node_index.get(node.name, original_node_count),
                "upstream_matmuls": upstream_matmuls,
                "direct_matmul_inputs": direct_matmul_inputs,
                "lhs_shape": _resolved_shape_for_report(
                    lhs_shape, bs, args.unknown_dim
                ),
                "rhs_shape": _resolved_shape_for_report(
                    rhs_shape, bs, args.unknown_dim
                ),
            }
        )

    candidates = []
    min_group = max(2, int(getattr(args, "rewrite_same_shape_batch_matmul_min_group", 4)))
    max_group_size = int(getattr(args, "rewrite_same_shape_batch_matmul_max_group_size", 64))
    for (
        dtype_enum,
        lhs_shape_key,
        rhs_shape_key,
        out_rows,
        k_dim,
        out_cols,
    ), items in groups.items():
        if len(items) < min_group:
            skipped["group_too_small"] += len(items)
            continue
        ordered_items = sorted(items, key=lambda item: item["node_index"])
        chunks = []
        if max_group_size <= 0:
            chunks = [ordered_items]
        else:
            for start in range(0, len(ordered_items), max_group_size):
                chunks.append(ordered_items[start : start + max_group_size])
        for chunk_index, chunk in enumerate(chunks):
            if len(chunk) < min_group:
                skipped["chunk_too_small"] += len(chunk)
                continue
            chunk_node_names = {item["node"].name for item in chunk}
            chunk_upstream_matmuls = set()
            chunk_direct_matmul_inputs = set()
            for item in chunk:
                chunk_upstream_matmuls.update(item["upstream_matmuls"])
                chunk_direct_matmul_inputs.update(item["direct_matmul_inputs"])
            if chunk_upstream_matmuls & chunk_node_names:
                skipped["intra_group_matmul_dependency"] += len(chunk)
                continue
            candidates.append(
                {
                    "dtype_enum": dtype_enum,
                    "out_rows": out_rows,
                    "k_dim": k_dim,
                    "out_cols": out_cols,
                    "items": chunk,
                    "node_names": chunk_node_names,
                    "upstream_matmuls": chunk_upstream_matmuls,
                    "direct_matmul_inputs": chunk_direct_matmul_inputs,
                    "chunk_index": chunk_index,
                    "original_group_size": len(items),
                    "estimated_flops": 2 * len(chunk) * out_rows * k_dim * out_cols,
                }
            )

    candidates.sort(
        key=lambda group: (
            len(group["items"]) - 1,
            group["estimated_flops"],
        ),
        reverse=True,
    )
    max_groups = int(getattr(args, "rewrite_same_shape_batch_matmul_max_groups", 8))
    selected_candidates = []
    selected_matmul_names = set()
    selected_upstream_matmul_names = set()
    for group in candidates:
        group_upstream_matmuls = group["upstream_matmuls"]
        group_node_names = group["node_names"]
        if "__upstream_limit_exceeded__" in group_upstream_matmuls:
            skipped["upstream_limit_exceeded"] += len(group["items"])
            continue
        if group_upstream_matmuls & selected_matmul_names:
            if group["direct_matmul_inputs"] & selected_matmul_names:
                skipped["matmul_input_dependency"] += len(group["items"])
            else:
                skipped["upstream_matmul_dependency"] += len(group["items"])
            continue
        if group_node_names & selected_upstream_matmul_names:
            skipped["downstream_matmul_dependency"] += len(group["items"])
            continue
        selected_candidates.append(group)
        selected_matmul_names.update(group_node_names)
        selected_upstream_matmul_names.update(group_upstream_matmuls)
        if max_groups > 0 and len(selected_candidates) >= max_groups:
            break
    candidates = selected_candidates

    replacement_by_node = {}
    fused_groups = []
    for group_index, group in enumerate(candidates):
        items = group["items"]
        prefix = f"musa_same_shape_batch_matmul_group_{group_index}"

        lhs_pack = graph_def.node.add()
        lhs_pack.name = f"{prefix}_lhs_pack"
        lhs_pack.op = "Pack"
        lhs_pack.input.extend([item["lhs_input"] for item in items])
        lhs_pack.attr["N"].i = len(items)
        lhs_pack.attr["T"].type = group["dtype_enum"]
        lhs_pack.attr["axis"].i = 0

        rhs_pack = graph_def.node.add()
        rhs_pack.name = f"{prefix}_rhs_pack"
        rhs_pack.op = "Pack"
        rhs_pack.input.extend([item["rhs_input"] for item in items])
        rhs_pack.attr["N"].i = len(items)
        rhs_pack.attr["T"].type = group["dtype_enum"]
        rhs_pack.attr["axis"].i = 0

        batch_matmul = graph_def.node.add()
        batch_matmul.name = f"{prefix}_batch_matmul"
        batch_matmul.op = "BatchMatMulV2"
        batch_matmul.input.extend([lhs_pack.name, rhs_pack.name])
        batch_matmul.attr["T"].type = group["dtype_enum"]
        batch_matmul.attr["adj_x"].b = False
        batch_matmul.attr["adj_y"].b = False

        squeeze_nodes = []
        for item_index, item in enumerate(items):
            begin_node = _add_const_node(
                graph_def,
                f"{prefix}_slice_{item_index}_begin",
                [item_index, 0, 0],
            )
            size_node = _add_const_node(
                graph_def,
                f"{prefix}_slice_{item_index}_size",
                [1, -1, -1],
            )
            slice_node = graph_def.node.add()
            slice_node.name = f"{prefix}_slice_{item_index}"
            slice_node.op = "Slice"
            slice_node.input.extend([batch_matmul.name, begin_node.name, size_node.name])
            slice_node.attr["T"].type = group["dtype_enum"]
            slice_node.attr["Index"].type = tf.int32.as_datatype_enum

            squeeze_node = graph_def.node.add()
            squeeze_node.name = f"{prefix}_squeeze_{item_index}"
            squeeze_node.op = "Squeeze"
            squeeze_node.input.append(slice_node.name)
            squeeze_node.attr["T"].type = group["dtype_enum"]
            squeeze_node.attr["squeeze_dims"].list.i.append(0)
            replacement_by_node[item["node"].name] = squeeze_node.name
            squeeze_nodes.append(squeeze_node.name)

        fused_groups.append(
            {
                "matmul_count": len(items),
                "out_rows": group["out_rows"],
                "k_dim": group["k_dim"],
                "out_cols": group["out_cols"],
                "lhs_shape": items[0]["lhs_shape"],
                "rhs_shape": items[0]["rhs_shape"],
                "chunk_index": group["chunk_index"],
                "original_group_size": group["original_group_size"],
                "estimated_mflops": group["estimated_flops"] / 1_000_000.0,
                "fused_batch_matmul": batch_matmul.name,
                "sample_matmuls": [item["node"].name for item in items[:8]],
                "squeeze_nodes": squeeze_nodes[:8],
            }
        )

    rewired_edges = 0
    if replacement_by_node:
        for node in graph_def.node:
            for index, input_name in enumerate(list(node.input)):
                new_input = _replace_tensor_input(input_name, replacement_by_node)
                if new_input != input_name:
                    node.input[index] = new_input
                    rewired_edges += 1

    return {
        "enabled": True,
        "groups": len(fused_groups),
        "original_matmuls": sum(group["matmul_count"] for group in fused_groups),
        "fused_batch_matmuls": len(fused_groups),
        "estimated_matmul_reduction": sum(
            group["matmul_count"] - 1 for group in fused_groups
        ),
        "rewired_edges": rewired_edges,
        "skipped": dict(skipped),
        "sample_groups": fused_groups[:10],
    }, replacement_by_node


def build_output_checksum_scalar(outputs, device):
    if not outputs:
        raise ValueError("checksum_scalar requires at least one output tensor")

    checksum_terms = []
    unsupported = []
    with tf.device(device):
        for index, output in enumerate(outputs):
            dtype = output.dtype.base_dtype
            if not (dtype.is_floating or dtype.is_integer or dtype == tf.bool):
                unsupported.append(f"{output.name}:{dtype.name}")
                continue
            value = tf.cast(output, tf.float32)
            checksum_terms.append(
                tf.reduce_sum(value, name=f"musa_output_checksum_reduce_{index}")
            )

        if unsupported:
            raise TypeError(
                "checksum_scalar has unsupported output dtypes: "
                + ", ".join(unsupported)
            )

        return tf.identity(
            tf.add_n(checksum_terms, name="musa_output_checksum_add"),
            name="musa_output_checksum_scalar",
        )


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


def estimate_tensor_nbytes(shape, dtype, bs=None):
    if shape is None:
        return None
    dims = []
    for index, dim in enumerate(shape):
        if dim is None:
            if index == 0 and bs is not None:
                dims.append(bs)
            else:
                return None
        else:
            dims.append(int(dim))
    try:
        return int(np.prod(dims)) * np.dtype(dtype.as_numpy_dtype).itemsize
    except Exception:
        return None


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


def size_bucket_name(nbytes):
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


def _node_dtype_name(node):
    for key in ("T", "dtype", "DstT"):
        if key in node.attr and node.attr[key].type:
            try:
                return tf.as_dtype(node.attr[key].type).name
            except Exception:
                return str(node.attr[key].type)
    return None


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


def _shape_key(shape):
    if shape is None:
        return None
    return tuple("?" if dim is None else int(dim) for dim in shape)


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


def analyze_matmul_groups(graph_def, spec_shape_map, bs, args):
    node_map = {node.name: node for node in graph_def.node}
    placeholder_count = sum(1 for node in graph_def.node if node.op == "Placeholder")
    consumers_by_node = defaultdict(list)
    for node in graph_def.node:
        for input_name in node.input:
            consumers_by_node[_strip_tensor_name(input_name)].append(node)

    matmul_ops = {"MatMul", "BatchMatMul", "BatchMatMulV2"}
    same_lhs_groups = {}
    same_rhs_groups = {}
    shape_groups = {}
    rhs_producer_hist = Counter()
    dtype_hist = Counter()
    op_hist = Counter()
    biasadd_count = 0
    total = 0

    for node in graph_def.node:
        if node.op not in matmul_ops or len(node.input) < 2:
            continue
        total += 1
        op_hist[node.op] += 1
        dtype = _node_dtype_name(node) or "unknown"
        dtype_hist[dtype] += 1
        lhs_name = node.input[0]
        rhs_name = node.input[1]
        lhs_base, _ = _tensor_name_parts(lhs_name)
        rhs_base, _ = _tensor_name_parts(rhs_name)
        rhs_node = node_map.get(rhs_base)
        rhs_kind = rhs_node.op if rhs_node is not None else "missing"
        rhs_producer_hist[rhs_kind] += 1
        transpose_a, transpose_b = _matmul_transpose_attrs(node)
        lhs_shape = _tensor_shape_for_analysis(lhs_name, node_map, spec_shape_map)
        rhs_shape = _tensor_shape_for_analysis(rhs_name, node_map, spec_shape_map)
        out_shape = _output_shape_from_node(node, 0)
        k_dim, out_cols = _matmul_contract_and_out_dims(
            lhs_shape, rhs_shape, transpose_a, transpose_b
        )
        if out_shape is not None and len(out_shape) >= 2:
            out_rows = out_shape[-2]
        elif lhs_shape is not None and len(lhs_shape) >= 2:
            out_rows = lhs_shape[-1] if transpose_a else lhs_shape[-2]
        else:
            out_rows = None
        has_biasadd = any(
            consumer.op in ("BiasAdd", "Add", "AddV2")
            for consumer in consumers_by_node.get(node.name, [])
        )
        if has_biasadd:
            biasadd_count += 1

        lhs_key = (
            node.op,
            dtype,
            lhs_base,
            _shape_key(lhs_shape),
            bool(transpose_a),
            bool(transpose_b),
            k_dim,
        )
        lhs_group = same_lhs_groups.setdefault(
            lhs_key,
            {
                "op": node.op,
                "dtype": dtype,
                "lhs": lhs_base,
                "lhs_shape": _resolved_shape_for_report(
                    lhs_shape, bs, args.unknown_dim
                ),
                "transpose_a": bool(transpose_a),
                "transpose_b": bool(transpose_b),
                "k_dim": k_dim,
                "count": 0,
                "biasadd_count": 0,
                "rhs_kind_counts": Counter(),
                "rhs_shape_counts": Counter(),
                "out_cols_total": 0,
                "sample_nodes": [],
            },
        )
        lhs_group["count"] += 1
        lhs_group["biasadd_count"] += int(has_biasadd)
        lhs_group["rhs_kind_counts"][rhs_kind] += 1
        lhs_group["rhs_shape_counts"][str(_shape_key(rhs_shape))] += 1
        if isinstance(out_cols, int) and out_cols > 0:
            lhs_group["out_cols_total"] += out_cols
        if len(lhs_group["sample_nodes"]) < 8:
            lhs_group["sample_nodes"].append(node.name)

        rhs_key = (
            node.op,
            dtype,
            rhs_base,
            _shape_key(rhs_shape),
            bool(transpose_a),
            bool(transpose_b),
            k_dim,
            out_cols,
        )
        rhs_group = same_rhs_groups.setdefault(
            rhs_key,
            {
                "op": node.op,
                "dtype": dtype,
                "rhs": rhs_base,
                "rhs_shape": _resolved_shape_for_report(
                    rhs_shape, bs, args.unknown_dim
                ),
                "transpose_a": bool(transpose_a),
                "transpose_b": bool(transpose_b),
                "k_dim": k_dim,
                "out_cols": out_cols,
                "count": 0,
                "biasadd_count": 0,
                "lhs_kind_counts": Counter(),
                "lhs_shape_counts": Counter(),
                "out_rows_total": 0,
                "sample_nodes": [],
            },
        )
        lhs_node = node_map.get(lhs_base)
        lhs_kind = lhs_node.op if lhs_node is not None else "missing"
        rhs_group["count"] += 1
        rhs_group["biasadd_count"] += int(has_biasadd)
        rhs_group["lhs_kind_counts"][lhs_kind] += 1
        rhs_group["lhs_shape_counts"][str(_shape_key(lhs_shape))] += 1
        if isinstance(out_rows, int) and out_rows > 0:
            rhs_group["out_rows_total"] += out_rows
        if len(rhs_group["sample_nodes"]) < 8:
            rhs_group["sample_nodes"].append(node.name)

        shape_key = (
            node.op,
            dtype,
            _shape_key(lhs_shape),
            _shape_key(rhs_shape),
            _shape_key(out_shape),
            bool(transpose_a),
            bool(transpose_b),
        )
        shape_group = shape_groups.setdefault(
            shape_key,
            {
                "op": node.op,
                "dtype": dtype,
                "lhs_shape": _resolved_shape_for_report(
                    lhs_shape, bs, args.unknown_dim
                ),
                "rhs_shape": _resolved_shape_for_report(
                    rhs_shape, bs, args.unknown_dim
                ),
                "output_shape": _resolved_shape_for_report(
                    out_shape, bs, args.unknown_dim
                ),
                "transpose_a": bool(transpose_a),
                "transpose_b": bool(transpose_b),
                "count": 0,
                "rhs_kind_counts": Counter(),
                "sample_nodes": [],
            },
        )
        shape_group["count"] += 1
        shape_group["rhs_kind_counts"][rhs_kind] += 1
        if len(shape_group["sample_nodes"]) < 8:
            shape_group["sample_nodes"].append(node.name)

    candidate_lhs_groups = [
        group for group in same_lhs_groups.values() if group["count"] >= 2
    ]
    candidate_rhs_groups = [
        group for group in same_rhs_groups.values() if group["count"] >= 2
    ]
    candidate_lhs_groups.sort(
        key=lambda group: (group["count"], group["out_cols_total"]), reverse=True
    )
    candidate_rhs_groups.sort(
        key=lambda group: (group["count"], group["out_rows_total"]), reverse=True
    )
    top_shape_groups = sorted(
        shape_groups.values(), key=lambda group: group["count"], reverse=True
    )

    def finalize_group(group):
        out = dict(group)
        if "rhs_kind_counts" in group:
            out["rhs_kind_counts"] = group["rhs_kind_counts"].most_common(8)
        if "lhs_kind_counts" in group:
            out["lhs_kind_counts"] = group["lhs_kind_counts"].most_common(8)
        if "rhs_shape_counts" in group:
            out["rhs_shape_counts"] = group["rhs_shape_counts"].most_common(8)
        if "lhs_shape_counts" in group:
            out["lhs_shape_counts"] = group["lhs_shape_counts"].most_common(8)
        return out

    estimated_lhs_reduction = sum(group["count"] - 1 for group in candidate_lhs_groups)
    estimated_rhs_reduction = sum(group["count"] - 1 for group in candidate_rhs_groups)
    return {
        "placeholder_count": placeholder_count,
        "min_placeholders": getattr(
            args, "rewrite_same_lhs_matmul_min_placeholders", 0
        ),
        "matmul_total": total,
        "matmul_by_op": op_hist.most_common(10),
        "matmul_by_dtype": dtype_hist.most_common(10),
        "matmul_rhs_producer_ops": rhs_producer_hist.most_common(20),
        "matmul_with_biasadd_consumers": biasadd_count,
        "same_lhs_group_count": len(candidate_lhs_groups),
        "same_lhs_matmul_count": sum(group["count"] for group in candidate_lhs_groups),
        "estimated_call_reduction_if_grouped_by_lhs": estimated_lhs_reduction,
        "same_rhs_group_count": len(candidate_rhs_groups),
        "same_rhs_matmul_count": sum(group["count"] for group in candidate_rhs_groups),
        "estimated_call_reduction_if_grouped_by_rhs": estimated_rhs_reduction,
        "top_same_lhs_groups": [
            finalize_group(group) for group in candidate_lhs_groups[:20]
        ],
        "top_same_rhs_groups": [
            finalize_group(group) for group in candidate_rhs_groups[:20]
        ],
        "top_shape_groups": [finalize_group(group) for group in top_shape_groups[:20]],
    }


def analyze_concat_pack_static(graph_def, input_infos, args):
    item_by_node = {
        _strip_output_port(item["name"]): item
        for item in input_infos
    }
    node_map = {node.name: node for node in graph_def.node}
    consumer_nodes = {}
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            if base in item_by_node:
                consumer_nodes.setdefault(base, set()).add(node.name)

    max_total_bytes = int(max(0.0, args.pack_concat_feed_max_total_mib) * 1024 * 1024)
    min_group_bytes = int(
        max(0.0, args.pack_concat_feed_min_total_mib) * 1024 * 1024
    )
    diagnostics = Counter()
    non_feed_input_ops = Counter()
    candidate_summaries = []
    candidate_bytes = []
    candidate_inputs = []
    selected_total = 0
    selected_feed_nodes = set()

    for node in graph_def.node:
        if node.op != "ConcatV2" or len(node.input) < 3:
            continue
        diagnostics["concat_nodes"] += 1
        axis_values = _const_int_list(node_map.get(_strip_tensor_name(node.input[-1])))
        if not axis_values or len(axis_values) != 1:
            diagnostics["skip_bad_axis"] += 1
            continue

        items = []
        skip_reason = None
        for input_name in node.input[:-1]:
            input_node_name = _strip_tensor_name(input_name)
            item = item_by_node.get(input_node_name)
            if item is None:
                producer = node_map.get(input_node_name)
                non_feed_input_ops[producer.op if producer is not None else "<missing>"] += 1
                skip_reason = "skip_non_feed_input"
                break
            if input_node_name in selected_feed_nodes:
                skip_reason = "skip_already_selected"
                break
            if consumer_nodes.get(input_node_name, set()) != {node.name}:
                skip_reason = "skip_multi_consumer_input"
                break
            items.append(item)
        if skip_reason is not None:
            diagnostics[skip_reason] += 1
            continue
        if len(items) < args.pack_concat_feed_min_inputs:
            diagnostics["skip_too_few_inputs"] += 1
            continue

        dtype = np.dtype(items[0]["dtype"])
        if not _is_packable_dtype(dtype):
            diagnostics["skip_unpacked_dtype"] += 1
            continue
        if any(np.dtype(item["dtype"]) != dtype for item in items):
            diagnostics["skip_mixed_dtype"] += 1
            continue

        shapes = [list(item["shape"]) for item in items]
        rank = len(shapes[0])
        if rank == 0 or any(len(shape) != rank for shape in shapes):
            diagnostics["skip_rank_mismatch"] += 1
            continue
        axis = int(axis_values[0])
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            diagnostics["skip_axis_out_of_range"] += 1
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
            diagnostics["skip_shape_mismatch"] += 1
            continue

        nbytes = int(np.prod(out_shape)) * dtype.itemsize
        diagnostics["eligible_before_threshold"] += 1
        diagnostics["eligible_before_threshold_nbytes"] += nbytes
        candidate_bytes.append(nbytes)
        candidate_inputs.append(len(items))
        candidate_summaries.append(
            {
                "concat_node": node.name,
                "inputs": len(items),
                "nbytes": nbytes,
                "mib": nbytes / (1024.0 * 1024.0),
                "axis": axis,
                "dtype": str(dtype),
                "shape": out_shape,
            }
        )
        if nbytes < min_group_bytes:
            diagnostics["skip_too_small_bytes"] += 1
            continue
        if max_total_bytes > 0 and selected_total + nbytes > max_total_bytes:
            diagnostics["skip_total_cap"] += 1
            continue

        selected_total += nbytes
        selected_feed_nodes.update(_strip_output_port(item["name"]) for item in items)
        diagnostics["selected_nodes"] += 1
        diagnostics["selected_inputs"] += len(items)
        diagnostics["selected_nbytes"] += nbytes

    return {
        "counts": dict(diagnostics),
        "top_non_feed_input_ops": non_feed_input_ops.most_common(20),
        "top_concat_candidates": sorted(
            candidate_summaries,
            key=lambda item: item["nbytes"],
            reverse=True,
        )[:10],
        "top_candidate_mib": [
            value / (1024.0 * 1024.0)
            for value in sorted(candidate_bytes, reverse=True)[:10]
        ],
        "top_candidate_inputs": sorted(candidate_inputs, reverse=True)[:10],
        "min_inputs": args.pack_concat_feed_min_inputs,
        "min_total_mib": args.pack_concat_feed_min_total_mib,
        "max_total_mib": args.pack_concat_feed_max_total_mib,
    }


def analyze_graph_static(
    meta,
    graph_def,
    input_spec,
    output_spec,
    spec_shape_map,
    slice_min_dims,
    bs,
    args,
):
    node_map = {node.name: node for node in graph_def.node}
    placeholder_names = {
        _strip_output_port(name)
        for name in input_spec
    }
    reachable_nodes = (
        collect_reachable_nodes(graph_def, output_spec)
        if args.feed_only_reachable
        else None
    )
    op_hist = Counter(node.op for node in graph_def.node)
    consumer_hist = Counter()
    consumer_by_input = {}
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            if base in placeholder_names:
                consumer_hist[node.op] += 1
                consumer_by_input.setdefault(base, Counter())[node.op] += 1

    inputs = []
    bucket_bytes = Counter()
    dtype_bytes = Counter()
    total_bytes = 0
    skipped_unreachable = 0
    large_count = 0
    large_bytes = 0
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
        np_dtype = np.dtype(dtype.as_numpy_dtype)
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
        nbytes = int(np.prod(run_shape)) * np_dtype.itemsize if run_shape else np_dtype.itemsize
        total_bytes += nbytes
        bucket_bytes[size_bucket_name(nbytes)] += nbytes
        dtype_bytes[str(np_dtype)] += nbytes
        if nbytes >= 1024 * 1024:
            large_count += 1
            large_bytes += nbytes
        consumers = consumer_by_input.get(node_name, Counter())
        inputs.append(
            {
                "name": name,
                "dtype": str(np_dtype),
                "shape": list(run_shape),
                "nbytes": nbytes,
                "mib": nbytes / (1024.0 * 1024.0),
                "consumer_ops": consumers.most_common(8),
            }
        )

    inputs.sort(key=lambda item: item["nbytes"], reverse=True)
    return {
        "batch_size": bs,
        "graph_node_count": len(graph_def.node),
        "input_spec_count": len(input_spec),
        "output_spec_count": len(output_spec),
        "reachable_node_count": len(reachable_nodes) if reachable_nodes is not None else None,
        "skipped_unreachable_inputs": skipped_unreachable,
        "input_count": len(inputs),
        "total_mib": total_bytes / (1024.0 * 1024.0),
        "large_input_count": large_count,
        "large_input_mib": large_bytes / (1024.0 * 1024.0),
        "mib_by_size_bucket": {
            name: bytes_ / (1024.0 * 1024.0)
            for name, bytes_ in bucket_bytes.items()
        },
        "mib_by_dtype": {
            name: bytes_ / (1024.0 * 1024.0)
            for name, bytes_ in dtype_bytes.items()
        },
        "largest_inputs": inputs[:30],
        "top_graph_ops": op_hist.most_common(40),
        "top_placeholder_consumer_ops": consumer_hist.most_common(30),
        "concat_pack_analysis": (
            analyze_concat_pack_static(graph_def, inputs, args)
            if should_pack_concat_feed(args)
            else None
        ),
        "matmul_group_analysis": analyze_matmul_groups(
            graph_def, spec_shape_map, bs, args
        ),
    }


def analyze_single_spec(spec_path: Path, pb_path: Path, args, bs: int):
    meta = load_meta(spec_path)
    input_spec = read_node_list_collection(meta, "input_spec")
    output_spec = read_node_list_collection(meta, "output_spec")
    selected_output_indices = parse_index_selection(args.output_indices, len(output_spec))
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
            "total_pow": 0,
            "rewritten": 0,
            "sample": [],
            "skipped_non_const": 0,
            "skipped_non_square": 0,
            "non_const_exponent_ops": [],
            "non_const_exponent_samples": [],
            "non_square_constant_values": [],
            "non_square_constant_samples": [],
        },
    }
    if should_bypass_identity(args, graph_def):
        selected_output_spec, identity_summary = bypass_identity_nodes(
            graph_def, selected_output_spec
        )
        graph_rewrite_summary["identity_bypass"] = identity_summary
    if should_rewrite_pow_square(args):
        graph_rewrite_summary["pow_square"] = rewrite_pow_square_nodes(graph_def)
    if should_rewrite_static_shape_subgraph(args):
        graph_rewrite_summary["static_shape_rewrite"] = (
            rewrite_static_shape_subgraph_nodes(graph_def, spec_shape_map, bs, args)
        )
    matmul_group_analysis = analyze_matmul_groups(graph_def, spec_shape_map, bs, args)
    graph_rewrite_summary["matmul_group_analysis"] = matmul_group_analysis
    graph_rewrite_summary["same_lhs_matmul_auto_decision"] = (
        same_lhs_matmul_auto_decision(args, matmul_group_analysis)
    )
    if should_rewrite_same_lhs_matmul(args, matmul_group_analysis):
        same_lhs_summary, same_lhs_replacements = rewrite_same_lhs_matmul_nodes(
            graph_def,
            spec_shape_map,
            bs,
            args,
        )
        selected_output_spec = [
            _replace_tensor_input(name, same_lhs_replacements)
            for name in selected_output_spec
        ]
        graph_rewrite_summary["same_lhs_matmul"] = same_lhs_summary
    same_shape_summary, same_shape_replacements = (
        rewrite_same_shape_batch_matmul_nodes(
            graph_def,
            spec_shape_map,
            bs,
            args,
        )
    )
    selected_output_spec = [
        _replace_tensor_input(name, same_shape_replacements)
        for name in selected_output_spec
    ]
    graph_rewrite_summary["same_shape_batch_matmul"] = same_shape_summary
    analysis = analyze_graph_static(
        meta,
        graph_def,
        input_spec,
        output_spec,
        spec_shape_map,
        slice_min_dims,
        bs,
        args,
    )
    return {
        "spec_path": str(spec_path),
        "pb_path": str(pb_path),
        "batch_size": bs,
        "status": "ok",
        "mode": "analyze_only",
        "analysis": analysis,
        "graph_rewrite": graph_rewrite_summary,
        "timing_ms": {
            "warmup": 0,
            "run_iters": 0,
            "average": None,
            "trimmed_avg": None,
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "all": [],
        },
    }


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
    item_by_node = {item["node_name"]: item for item in feed_items}
    consumer_nodes = {}
    node_map = {node.name: node for node in graph_def.node}
    for node in graph_def.node:
        for input_name in node.input:
            base = _strip_tensor_name(input_name)
            if base in item_by_node:
                consumer_nodes.setdefault(base, set()).add(node.name)

    max_total_bytes = int(max(0.0, args.pack_concat_feed_max_total_mib) * 1024 * 1024)
    chunk_max_bytes = int(
        max(0.0, args.pack_concat_feed_chunk_max_mib) * 1024 * 1024
    )
    min_group_bytes = int(
        max(0.0, args.pack_concat_feed_min_total_mib) * 1024 * 1024
    )
    selected = []
    selected_total = 0
    selected_feed_nodes = set()
    diagnostics = Counter()
    non_feed_input_ops = Counter()
    candidate_bytes = []
    candidate_inputs = []
    candidate_summaries = []

    for node in graph_def.node:
        if node.op != "ConcatV2" or len(node.input) < 3:
            continue
        diagnostics["concat_nodes"] += 1
        axis_values = _const_int_list(node_map.get(_strip_tensor_name(node.input[-1])))
        if not axis_values or len(axis_values) != 1:
            diagnostics["skip_bad_axis"] += 1
            continue

        data_inputs = node.input[:-1]
        items = []
        skip_reason = None
        for input_name in data_inputs:
            item = item_by_node.get(_strip_tensor_name(input_name))
            if item is None:
                skip_reason = "skip_non_feed_input"
                producer = node_map.get(_strip_tensor_name(input_name))
                non_feed_input_ops[producer.op if producer is not None else "<missing>"] += 1
                break
            if item["node_name"] in selected_feed_nodes:
                skip_reason = "skip_already_selected"
                break
            if consumer_nodes.get(item["node_name"], set()) != {node.name}:
                skip_reason = "skip_multi_consumer_input"
                break
            items.append(item)
        if skip_reason is not None:
            diagnostics[skip_reason] += 1
            continue
        if len(items) < args.pack_concat_feed_min_inputs:
            diagnostics["skip_too_few_inputs"] += 1
            continue

        dtype = np.dtype(items[0]["np_dtype"])
        if not _is_packable_dtype(dtype):
            diagnostics["skip_unpacked_dtype"] += 1
            continue
        if any(np.dtype(item["np_dtype"]) != dtype for item in items):
            diagnostics["skip_mixed_dtype"] += 1
            continue

        shapes = [list(item["shape"]) for item in items]
        rank = len(shapes[0])
        if rank == 0 or any(len(shape) != rank for shape in shapes):
            diagnostics["skip_rank_mismatch"] += 1
            continue
        axis = int(axis_values[0])
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            diagnostics["skip_axis_out_of_range"] += 1
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
            diagnostics["skip_shape_mismatch"] += 1
            continue

        nbytes = int(np.prod(out_shape)) * dtype.itemsize
        diagnostics["eligible_before_threshold"] += 1
        diagnostics["eligible_before_threshold_nbytes"] += nbytes
        candidate_bytes.append(nbytes)
        candidate_inputs.append(len(items))
        candidate_summaries.append(
            {
                "concat_node": node.name,
                "inputs": len(items),
                "nbytes": nbytes,
                "mib": nbytes / (1024.0 * 1024.0),
                "axis": axis,
                "dtype": str(dtype),
                "shape": out_shape,
            }
        )
        if nbytes < min_group_bytes:
            diagnostics["skip_too_small_bytes"] += 1
            continue
        use_chunked = chunk_max_bytes > 0 and nbytes > chunk_max_bytes
        if (
            max_total_bytes > 0
            and selected_total + nbytes > max_total_bytes
            and not use_chunked
        ):
            diagnostics["skip_total_cap"] += 1
            continue

        chunk_groups = None
        if use_chunked:
            chunk_groups = []
            current_items = []
            current_shape = None
            current_nbytes = 0
            for item, shape in zip(items, shapes):
                item_nbytes = int(getattr(item["value"], "nbytes", 0) or 0)
                if (
                    current_items
                    and current_nbytes + item_nbytes > chunk_max_bytes
                ):
                    chunk_groups.append(
                        {
                            "items": current_items,
                            "shape": current_shape,
                            "nbytes": current_nbytes,
                        }
                    )
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
                chunk_groups.append(
                    {
                        "items": current_items,
                        "shape": current_shape,
                        "nbytes": current_nbytes,
                    }
                )
            if len(chunk_groups) <= 1:
                chunk_groups = None
                use_chunked = False
                if max_total_bytes > 0 and selected_total + nbytes > max_total_bytes:
                    diagnostics["skip_total_cap"] += 1
                    continue

        selected.append(
            {
                "node": node,
                "items": items,
                "axis": axis,
                "shape": out_shape,
                "dtype": dtype,
                "nbytes": nbytes,
                "chunked": use_chunked,
                "chunks": chunk_groups,
            }
        )
        selected_total += nbytes
        selected_feed_nodes.update(item["node_name"] for item in items)
        diagnostics["selected_nodes"] += 1
        diagnostics["selected_inputs"] += len(items)
        diagnostics["selected_nbytes"] += nbytes
        if use_chunked:
            diagnostics["selected_chunked_nodes"] += 1
            diagnostics["selected_chunks"] += len(chunk_groups)

    input_map = {}
    feed_dict = {}
    packed_holders = []
    packed_inputs = []
    cache_initializers = []
    pinned_error = None
    use_pinned = should_allocate_individual_pinned_feed(args)
    freeze_concat_packed_feed = args.freeze_concat_packed_feed == "on"
    cache_concat_packed_feed = args.cache_concat_packed_feed == "on"
    if freeze_concat_packed_feed and cache_concat_packed_feed:
        raise ValueError(
            "--freeze_concat_packed_feed and --cache_concat_packed_feed are mutually exclusive"
        )

    with graph.as_default():
        for index, group in enumerate(selected):
            if group.get("chunked"):
                chunk_tensors = []
                chunks = group.get("chunks") or []
                for chunk_index, chunk in enumerate(chunks):
                    try:
                        packed_value = allocate_concat_array(
                            chunk["items"],
                            group["axis"],
                            chunk["shape"],
                            group["dtype"],
                            use_pinned,
                            packed_holders,
                        )
                        packed_pinned = use_pinned
                    except Exception as exc:
                        pinned_error = str(exc)
                        if args.pinned_feed == "on":
                            raise
                        packed_value = allocate_concat_array(
                            chunk["items"],
                            group["axis"],
                            chunk["shape"],
                            group["dtype"],
                            False,
                            packed_holders,
                        )
                        packed_pinned = False

                    if freeze_concat_packed_feed:
                        packed_tensor = tf.constant(
                            packed_value,
                            dtype=tf.as_dtype(group["dtype"]),
                            name=f"musa_concat_packed_feed_{index}_chunk_{chunk_index}",
                        )
                    elif cache_concat_packed_feed:
                        cache_var = tf.Variable(
                            initial_value=tf.zeros(
                                chunk["shape"], dtype=tf.as_dtype(group["dtype"])
                            ),
                            trainable=False,
                            name=f"musa_concat_packed_feed_{index}_chunk_{chunk_index}_cache",
                        )
                        init_feed = tf.compat.v1.placeholder(
                            tf.as_dtype(group["dtype"]),
                            shape=chunk["shape"],
                            name=f"musa_concat_packed_feed_{index}_chunk_{chunk_index}_cache_init",
                        )
                        assign_op = tf.compat.v1.assign(cache_var, init_feed)
                        packed_tensor = tf.identity(
                            cache_var.read_value(),
                            name=f"musa_concat_packed_feed_{index}_chunk_{chunk_index}",
                        )
                        cache_initializers.append(
                            {
                                "assign_op": assign_op,
                                "feed_tensor": init_feed,
                                "value": packed_value,
                                "nbytes": int(packed_value.nbytes),
                            }
                        )
                    else:
                        packed_tensor = tf.compat.v1.placeholder(
                            tf.as_dtype(group["dtype"]),
                            shape=chunk["shape"],
                            name=f"musa_concat_packed_feed_{index}_chunk_{chunk_index}",
                        )
                        feed_dict[packed_tensor] = packed_value
                    chunk_tensors.append(packed_tensor)
                    packed_inputs.append(
                        {
                            "concat_node": group["node"].name,
                            "axis": group["axis"],
                            "shape": list(chunk["shape"]),
                            "dtype": str(group["dtype"]),
                            "num_original_inputs": len(chunk["items"]),
                            "nbytes": int(packed_value.nbytes),
                            "pinned": packed_pinned,
                            "frozen": freeze_concat_packed_feed,
                            "cached": cache_concat_packed_feed,
                            "chunked": True,
                            "chunk_index": chunk_index,
                            "num_chunks": len(chunks),
                        }
                    )

                input_map[f"{group['node'].name}:0"] = tf.concat(
                    chunk_tensors,
                    axis=group["axis"],
                    name=f"musa_concat_packed_feed_{index}_concat",
                )
                continue

            try:
                packed_value = allocate_concat_array(
                    group["items"],
                    group["axis"],
                    group["shape"],
                    group["dtype"],
                    use_pinned,
                    packed_holders,
                )
                packed_pinned = use_pinned
            except Exception as exc:
                pinned_error = str(exc)
                if args.pinned_feed == "on":
                    raise
                packed_value = allocate_concat_array(
                    group["items"],
                    group["axis"],
                    group["shape"],
                    group["dtype"],
                    False,
                    packed_holders,
                )
                packed_pinned = False

            if freeze_concat_packed_feed:
                packed_tensor = tf.constant(
                    packed_value,
                    dtype=tf.as_dtype(group["dtype"]),
                    name=f"musa_concat_packed_feed_{index}",
                )
            elif cache_concat_packed_feed:
                cache_var = tf.Variable(
                    initial_value=tf.zeros(
                        group["shape"], dtype=tf.as_dtype(group["dtype"])
                    ),
                    trainable=False,
                    name=f"musa_concat_packed_feed_{index}_cache",
                )
                init_feed = tf.compat.v1.placeholder(
                    tf.as_dtype(group["dtype"]),
                    shape=group["shape"],
                    name=f"musa_concat_packed_feed_{index}_cache_init",
                )
                assign_op = tf.compat.v1.assign(cache_var, init_feed)
                packed_tensor = tf.identity(
                    cache_var.read_value(),
                    name=f"musa_concat_packed_feed_{index}",
                )
                cache_initializers.append(
                    {
                        "assign_op": assign_op,
                        "feed_tensor": init_feed,
                        "value": packed_value,
                        "nbytes": int(packed_value.nbytes),
                    }
                )
            else:
                packed_tensor = tf.compat.v1.placeholder(
                    tf.as_dtype(group["dtype"]),
                    shape=group["shape"],
                    name=f"musa_concat_packed_feed_{index}",
                )
                feed_dict[packed_tensor] = packed_value
            input_map[f"{group['node'].name}:0"] = packed_tensor
            packed_inputs.append(
                {
                    "concat_node": group["node"].name,
                    "axis": group["axis"],
                    "shape": list(group["shape"]),
                    "dtype": str(group["dtype"]),
                    "num_original_inputs": len(group["items"]),
                    "nbytes": int(packed_value.nbytes),
                    "pinned": packed_pinned,
                    "frozen": freeze_concat_packed_feed,
                    "cached": cache_concat_packed_feed,
                    "chunked": False,
                }
            )

    selected_names = {item["name"] for group in selected for item in group["items"]}
    return {
        "input_map": input_map,
        "feed_dict": feed_dict,
        "packed_holders": packed_holders,
        "cache_initializers": cache_initializers,
        "unpacked_items": [
            item for item in feed_items if item["name"] not in selected_names
        ],
        "packed_inputs": packed_inputs,
        "selected_concat_nodes": len(selected),
        "selected_count": sum(len(group["items"]) for group in selected),
        "selected_nbytes": selected_total,
        "pinned_error": pinned_error,
        "diagnostics": {
            "counts": dict(diagnostics),
            "top_non_feed_input_ops": non_feed_input_ops.most_common(20),
            "top_concat_candidates": sorted(
                candidate_summaries,
                key=lambda item: item["nbytes"],
                reverse=True,
            )[:10],
            "top_candidate_mib": [
                value / (1024.0 * 1024.0)
                for value in sorted(candidate_bytes, reverse=True)[:10]
            ],
            "top_candidate_inputs": sorted(candidate_inputs, reverse=True)[:10],
            "min_inputs": args.pack_concat_feed_min_inputs,
            "min_total_mib": args.pack_concat_feed_min_total_mib,
            "max_total_mib": args.pack_concat_feed_max_total_mib,
            "chunk_max_mib": args.pack_concat_feed_chunk_max_mib,
            "freeze_concat_packed_feed": freeze_concat_packed_feed,
            "cache_concat_packed_feed": cache_concat_packed_feed,
        },
    }


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


def build_small_packed_feed(
    graph,
    feed_items,
    args,
    *,
    name_prefix="musa_small_packed_feed",
    max_bytes=None,
    max_total_mib=None,
    unpack_op=None,
):
    max_bytes = args.pack_small_feed_max_bytes if max_bytes is None else max_bytes
    max_total_mib = (
        args.pack_small_feed_max_total_mib if max_total_mib is None else max_total_mib
    )
    unpack_op = args.pack_small_feed_unpack_op if unpack_op is None else unpack_op
    candidate_groups = {}
    max_total_bytes = int(max(0.0, max_total_mib) * 1024 * 1024)
    selected_total = 0
    candidates = []

    for item in feed_items:
        nbytes = int(getattr(item["value"], "nbytes", 0) or 0)
        if (
            nbytes <= max_bytes
            and _is_packable_dtype(item["np_dtype"])
        ):
            candidates.append((nbytes, item))

    candidates.sort(key=lambda pair: pair[0])
    for nbytes, item in candidates:
        if max_total_bytes > 0 and selected_total + nbytes > max_total_bytes:
            continue
        candidate_groups.setdefault(np.dtype(item["np_dtype"]).str, []).append(item)
        selected_total += nbytes

    groups = {
        dtype_key: items
        for dtype_key, items in candidate_groups.items()
        if len(items) >= 2
    }
    selected_names = {
        item["name"] for items in groups.values() for item in items
    }
    unpacked = [
        item for item in feed_items if item["name"] not in selected_names
    ]
    selected_count = sum(len(items) for items in groups.values())
    selected_bytes = sum(
        int(getattr(item["value"], "nbytes", 0) or 0)
        for items in groups.values()
        for item in items
    )

    input_map = {}
    feed_dict = {}
    packed_holders = []
    use_pinned = should_allocate_individual_pinned_feed(args)
    pinned_error = None
    packed_inputs = []

    with graph.as_default():
        for index, (dtype_key, items) in enumerate(sorted(groups.items())):
            dtype = np.dtype(dtype_key)
            flats = [np.asarray(item["value"], dtype=dtype).reshape(-1) for item in items]
            try:
                packed_value = allocate_packed_array(flats, dtype, use_pinned, packed_holders)
                packed_pinned = use_pinned
            except Exception as exc:
                pinned_error = str(exc)
                if args.pinned_feed == "on":
                    raise
                packed_value = allocate_packed_array(flats, dtype, False, packed_holders)
                packed_pinned = False

            packed_tensor = tf.compat.v1.placeholder(
                tf.as_dtype(dtype),
                shape=[packed_value.size],
                name=f"{name_prefix}_{index}",
            )
            feed_dict[packed_tensor] = packed_value

            sizes = [_num_elements(item["shape"]) for item in items]
            if unpack_op == "split":
                parts = tf.split(
                    packed_tensor,
                    sizes,
                    axis=0,
                    name=f"{name_prefix}_split_{index}",
                )
            else:
                parts = []
                offset = 0
                for item, numel in zip(items, sizes):
                    parts.append(
                        tf.slice(
                            packed_tensor,
                            [offset],
                            [numel],
                            name=f"{name_prefix}_slice_{item['node_name']}",
                        )
                    )
                    offset += numel

            for item, part in zip(items, parts):
                replacement = tf.reshape(
                    part,
                    item["shape"],
                    name=f"{name_prefix}_reshape_{item['node_name']}",
                )
                input_map[item["name"]] = replacement

            packed_inputs.append(
                {
                    "dtype": str(dtype),
                    "num_original_inputs": len(items),
                    "nbytes": int(packed_value.nbytes),
                    "pinned": packed_pinned,
                }
            )

    return {
        "input_map": input_map,
        "feed_dict": feed_dict,
        "packed_holders": packed_holders,
        "unpacked_items": unpacked,
        "packed_inputs": packed_inputs,
        "selected_count": selected_count,
        "selected_nbytes": selected_bytes,
        "pinned_error": pinned_error,
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


def extract_error_tail(stack: Union[str, None], max_lines=80):
    if not stack:
        return []
    lines = [line.rstrip() for line in stack.strip().splitlines()]
    return lines[-max_lines:]


def run_single_spec(spec_path: Path, pb_path: Path, args, bs: int, musa_loaded: bool, runner_out: Path):
    meta = load_meta(spec_path)
    input_spec = read_node_list_collection(meta, "input_spec")
    output_spec = read_node_list_collection(meta, "output_spec")
    selected_output_indices = parse_index_selection(args.output_indices, len(output_spec))
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
            "total_pow": 0,
            "rewritten": 0,
            "sample": [],
            "skipped_non_const": 0,
            "skipped_non_square": 0,
            "non_const_exponent_ops": [],
            "non_const_exponent_samples": [],
            "non_square_constant_values": [],
            "non_square_constant_samples": [],
        },
    }
    if should_bypass_identity(args, graph_def):
        selected_output_spec, identity_summary = bypass_identity_nodes(
            graph_def, selected_output_spec
        )
        graph_rewrite_summary["identity_bypass"] = identity_summary
    if should_rewrite_pow_square(args):
        graph_rewrite_summary["pow_square"] = rewrite_pow_square_nodes(graph_def)
    if should_rewrite_static_shape_subgraph(args):
        graph_rewrite_summary["static_shape_rewrite"] = (
            rewrite_static_shape_subgraph_nodes(
                graph_def, spec_shape_map, bs, args
            )
        )
    matmul_group_analysis = analyze_matmul_groups(
        graph_def, spec_shape_map, bs, args
    )
    graph_rewrite_summary["matmul_group_analysis"] = matmul_group_analysis
    graph_rewrite_summary["same_lhs_matmul_auto_decision"] = (
        same_lhs_matmul_auto_decision(args, matmul_group_analysis)
    )
    if should_rewrite_same_lhs_matmul(args, matmul_group_analysis):
        same_lhs_summary, same_lhs_replacements = (
            rewrite_same_lhs_matmul_nodes(
                graph_def,
                spec_shape_map,
                bs,
                args,
            )
        )
        selected_output_spec = [
            _replace_tensor_input(name, same_lhs_replacements)
            for name in selected_output_spec
        ]
        graph_rewrite_summary["same_lhs_matmul"] = same_lhs_summary
    same_shape_summary, same_shape_replacements = (
        rewrite_same_shape_batch_matmul_nodes(
            graph_def,
            spec_shape_map,
            bs,
            args,
        )
    )
    selected_output_spec = [
        _replace_tensor_input(name, same_shape_replacements)
        for name in selected_output_spec
    ]
    graph_rewrite_summary["same_shape_batch_matmul"] = same_shape_summary

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
    protected_feed_node_names = {
        _strip_tensor_name(name) for name in selected_output_spec
    }
    pack_protected_output_feeds = [
        item
        for item in feed_items
        if item["node_name"] in protected_feed_node_names
    ]

    with tf.Graph().as_default() as graph:
        feed_dict = {}
        concat_pack_state = None
        slice_compact_state = None
        slice_pack_state = None
        small_pack_state = None
        remaining_pack_state = None
        input_map = {}
        active_feed_items = feed_items

        if should_pack_concat_feed(args):
            try:
                concat_pack_state = build_concat_packed_feed(
                    graph, graph_def, feed_items, args
                )
                if concat_pack_state["selected_count"] > 0:
                    input_map.update(concat_pack_state["input_map"])
                    feed_dict.update(concat_pack_state["feed_dict"])
                    active_feed_items = concat_pack_state["unpacked_items"]
                    if args.rewrite_concat_static_precompute == "on":
                        concat_pack_state["concat_static_precompute"] = (
                            rewrite_concat_static_precompute_nodes(
                                graph_def, concat_pack_state, bs=bs, args=args
                            )
                        )
            except Exception as exc:
                if args.pack_concat_feed == "on":
                    raise
                concat_pack_state = {
                    "input_map": {},
                    "feed_dict": {},
                    "packed_holders": [],
                    "unpacked_items": feed_items,
                    "packed_inputs": [],
                    "selected_concat_nodes": 0,
                    "selected_count": 0,
                    "selected_nbytes": 0,
                    "pinned_error": str(exc),
                    "fallback_error": str(exc),
                    "diagnostics": {"error": str(exc)},
                }

        if args.compact_slice_feed == "on":
            try:
                slice_compact_state = build_slice_compacted_feed(
                    graph, graph_def, active_feed_items, args
                )
                if slice_compact_state["selected_slice_nodes"] > 0:
                    input_map.update(slice_compact_state["input_map"])
                    feed_dict.update(slice_compact_state["feed_dict"])
                    active_feed_items = slice_compact_state["unpacked_items"]
            except Exception as exc:
                slice_compact_state = {
                    "input_map": {},
                    "feed_dict": {},
                    "packed_holders": [],
                    "unpacked_items": active_feed_items,
                    "compacted_inputs": [],
                    "selected_source_inputs": 0,
                    "selected_slice_nodes": 0,
                    "selected_original_nbytes": 0,
                    "selected_nbytes": 0,
                    "saved_nbytes": 0,
                    "input_delta": 0,
                    "pinned_error": None,
                    "fallback_error": str(exc),
                    "diagnostics": {"error": str(exc)},
                }

        if should_pack_slice_feed(args):
            try:
                slice_pack_state = build_slice_packed_feed(
                    graph, graph_def, active_feed_items, args
                )
                if slice_pack_state["selected_slice_nodes"] > 0:
                    input_map.update(slice_pack_state["input_map"])
                    feed_dict.update(slice_pack_state["feed_dict"])
                    active_feed_items = slice_pack_state["unpacked_items"]
                elif not slice_pack_state.get("diagnostics"):
                    slice_pack_state = None
            except Exception as exc:
                if args.pack_slice_feed == "on":
                    raise
                slice_pack_state = {
                    "input_map": {},
                    "feed_dict": {},
                    "packed_holders": [],
                    "unpacked_items": active_feed_items,
                    "packed_inputs": [],
                    "selected_source_inputs": 0,
                    "selected_slice_nodes": 0,
                    "selected_original_nbytes": 0,
                    "selected_nbytes": 0,
                    "saved_nbytes": 0,
                    "pinned_error": str(exc),
                    "fallback_error": str(exc),
                    "diagnostics": {"error": str(exc)},
                }

        small_pack_candidate_items = [
            item
            for item in active_feed_items
            if item["node_name"] not in protected_feed_node_names
        ]
        small_pack_requested = should_pack_small_feed(args, small_pack_candidate_items)
        if small_pack_requested:
            try:
                small_pack_state = build_small_packed_feed(
                    graph, small_pack_candidate_items, args
                )
                if small_pack_state["selected_count"] > 0:
                    input_map.update(small_pack_state["input_map"])
                    feed_dict.update(small_pack_state["feed_dict"])
                    active_feed_items = merge_packed_subset_feed_items(
                        active_feed_items,
                        small_pack_candidate_items,
                        small_pack_state["unpacked_items"],
                    )
                else:
                    small_pack_state = None
            except Exception as exc:
                if args.pack_small_feed == "on":
                    raise
                small_pack_state = {
                    "input_map": {},
                    "feed_dict": {},
                    "packed_holders": [],
                    "unpacked_items": feed_items,
                    "packed_inputs": [],
                    "selected_count": 0,
                    "selected_nbytes": 0,
                    "pinned_error": str(exc),
                    "fallback_error": str(exc),
                }

        remaining_pack_candidate_items = [
            item
            for item in active_feed_items
            if item["node_name"] not in protected_feed_node_names
        ]
        remaining_pack_requested = should_pack_remaining_feed(
            args, remaining_pack_candidate_items
        )
        if remaining_pack_requested:
            try:
                remaining_pack_state = build_small_packed_feed(
                    graph,
                    remaining_pack_candidate_items,
                    args,
                    name_prefix="musa_remaining_packed_feed",
                    max_bytes=args.pack_remaining_feed_max_bytes,
                    max_total_mib=args.pack_remaining_feed_max_total_mib,
                    unpack_op=args.pack_remaining_feed_unpack_op,
                )
                if remaining_pack_state["selected_count"] > 0:
                    input_map.update(remaining_pack_state["input_map"])
                    feed_dict.update(remaining_pack_state["feed_dict"])
                    active_feed_items = merge_packed_subset_feed_items(
                        active_feed_items,
                        remaining_pack_candidate_items,
                        remaining_pack_state["unpacked_items"],
                    )
                else:
                    remaining_pack_state = None
            except Exception as exc:
                if args.pack_remaining_feed == "on":
                    raise
                remaining_pack_state = {
                    "input_map": {},
                    "feed_dict": {},
                    "packed_holders": [],
                    "unpacked_items": remaining_pack_candidate_items,
                    "packed_inputs": [],
                    "selected_count": 0,
                    "selected_nbytes": 0,
                    "pinned_error": str(exc),
                    "fallback_error": str(exc),
                }

        use_device_scope = bool(args.device) and not (
            args.xla and device_kind(args.device) == "MUSA" and not args.xla_device_scope
        )
        log_run_phase_progress(
            args,
            "graph_import_begin",
            graph_nodes=len(graph_def.node),
            input_map=len(input_map),
        )
        with maybe_xla_jit_scope(args):
            if use_device_scope:
                with tf.device(args.device):
                    tf.import_graph_def(graph_def, name="", input_map=input_map)
            else:
                tf.import_graph_def(graph_def, name="", input_map=input_map)
        log_run_phase_progress(
            args,
            "graph_import_done",
            graph_ops=len(graph.get_operations()),
        )

        outputs = [graph.get_tensor_by_name(name) for name in selected_output_spec]

        def safe_shape(tensor):
            if tensor.shape.rank is None:
                return None
            return [dim if dim is not None else None for dim in tensor.shape.as_list()]

        output_info = [
            {
                "name": tensor.name,
                "optype": tensor.op.type,
                "dtype": tensor.dtype.name,
                "shape_in_graph": safe_shape(tensor),
                "estimated_nbytes": estimate_tensor_nbytes(
                    safe_shape(tensor), tensor.dtype, bs=bs
                ),
            }
            for tensor in outputs
        ]
        output_nbytes = sum(
            item["estimated_nbytes"] or 0 for item in output_info
        )
        output_dependency_summary = summarize_output_dependencies(
            graph_def, selected_output_spec, limit=20
        )

        for item in active_feed_items:
            tensor = graph.get_tensor_by_name(item["name"])
            feed_dict[tensor] = item["value"]

        source_feed_arena_state = {
            "enabled": False,
            "reason": "off" if args.pinned_feed_arena == "off" else "auto_disabled",
            "inputs": 0,
            "active_bytes": 0,
            "arena_bytes": 0,
            "alignment": SOURCE_FEED_ARENA_ALIGNMENT,
            "base_ptr": 0,
        }
        if args.pinned_feed_arena == "on":
            if (
                os.environ.get("MUSA_PJRT_SOURCE_FEED_ARENA_BASE")
                or os.environ.get("MUSA_PJRT_SOURCE_FEED_ARENA_BYTES")
            ):
                source_feed_arena_state["reason"] = "already_configured"
            else:
                try:
                    source_feed_arena_state = build_pinned_source_feed_arena(
                        feed_dict,
                        feed_state["pinned_holders"],
                        progress=log_source_feed_arena_progress,
                    )
                except Exception as exc:
                    source_feed_arena_state["reason"] = f"allocation_failed:{exc}"
                if source_feed_arena_state["enabled"]:
                    os.environ["MUSA_PJRT_SOURCE_FEED_ARENA_BASE"] = hex(
                        source_feed_arena_state["base_ptr"]
                    )
                    os.environ["MUSA_PJRT_SOURCE_FEED_ARENA_BYTES"] = str(
                        source_feed_arena_state["arena_bytes"]
                    )
                    feed_state["pinned_feed_used"] = True

        if args.pinned_feed_arena != "off":
            print(
                "[INFO] source_feed_arena: "
                f"enabled={source_feed_arena_state['enabled']} "
                f"reason={source_feed_arena_state['reason']} "
                f"inputs={source_feed_arena_state['inputs']} "
                f"active_mib={source_feed_arena_state['active_bytes'] / (1024.0 * 1024.0):.3f} "
                f"arena_mib={source_feed_arena_state['arena_bytes'] / (1024.0 * 1024.0):.3f} "
                f"alignment={source_feed_arena_state['alignment']}"
            )

        fetch_mode = args.fetch_mode
        output_fetch_plan = None
        fetches = outputs
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
        elif fetch_mode == "checksum_scalar":
            fetches = build_output_checksum_scalar(outputs, args.device)
        else:
            output_fetch_plan = build_output_fetch_plan(outputs, feed_dict, args)
            fetches = output_fetch_plan["device_fetches"]
            reconstruct_fetches = output_fetch_plan["reconstruct"]
            if not fetches:
                fetches = tf.no_op(name="musa_no_device_output_fetches")

        original_feed_summary = summarize_feed_items(feed_items, bs=bs)
        feed_summary = summarize_feed_dict(feed_dict, bs=bs)
        feed_summary["remaining_original_feed"] = summarize_remaining_feed_items(
            active_feed_items, graph_def, limit=20
        )
        feed_summary["active_large_feed"] = summarize_large_feed_items(
            active_feed_items, graph_def, bs=bs, limit=20
        )
        feed_summary["original_large_feed"] = summarize_large_feed_items(
            feed_items, graph_def, bs=bs, limit=20
        )
        feed_summary["large_slice_feed_diag"] = summarize_large_slice_feed_candidates(
            feed_items, graph_def, args, bs=bs, limit=20
        )
        feed_summary["concat_pack_downstream"] = summarize_concat_pack_downstream(
            graph_def, concat_pack_state, output_spec=selected_output_spec, limit=20
        )
        feed_summary["original_num_inputs"] = len(feed_items)
        feed_summary["original_total_mib"] = original_feed_summary["total_mib"]
        feed_summary["original_batch_dim_mib"] = original_feed_summary[
            "batch_dim_mib"
        ]
        feed_summary["original_non_batch_dim_mib"] = original_feed_summary[
            "non_batch_dim_mib"
        ]
        feed_summary["original_large_input_count"] = original_feed_summary[
            "large_input_count"
        ]
        feed_summary["original_large_input_mib"] = original_feed_summary[
            "large_input_mib"
        ]
        feed_summary["input_spec_count"] = len(input_spec)
        feed_summary["feed_only_reachable"] = bool(args.feed_only_reachable)
        feed_summary["reachable_node_count"] = (
            len(reachable_nodes) if reachable_nodes is not None else None
        )
        feed_summary["skipped_unreachable_inputs"] = feed_state[
            "skipped_unreachable_inputs"
        ]
        feed_summary["pack_protected_output_feeds"] = len(
            pack_protected_output_feeds
        )
        feed_summary["pack_protected_output_feed_names"] = [
            item["name"] for item in pack_protected_output_feeds[:8]
        ]
        feed_summary["benchmark_mode"] = "dynamic_feed"
        feed_summary["pinned_feed_requested"] = feed_state["pinned_feed_requested"]
        feed_summary["pinned_feed_used"] = feed_state["pinned_feed_used"]
        feed_summary["pinned_feed_error"] = feed_state["pinned_feed_error"]
        feed_summary["source_feed_arena"] = source_feed_arena_state
        concat_pack_used = bool(
            concat_pack_state and concat_pack_state.get("selected_count", 0) > 0
        )
        feed_summary["concat_pack_requested"] = should_pack_concat_feed(args)
        feed_summary["concat_pack_used"] = concat_pack_used
        feed_summary["concat_pack_selected_nodes"] = (
            concat_pack_state.get("selected_concat_nodes", 0)
            if concat_pack_state
            else 0
        )
        feed_summary["concat_pack_selected_inputs"] = (
            concat_pack_state.get("selected_count", 0) if concat_pack_state else 0
        )
        feed_summary["concat_pack_selected_nbytes"] = (
            concat_pack_state.get("selected_nbytes", 0) if concat_pack_state else 0
        )
        feed_summary["concat_pack_selected_mib"] = (
            feed_summary["concat_pack_selected_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["concat_pack_min_inputs"] = args.pack_concat_feed_min_inputs
        feed_summary["concat_pack_min_total_mib"] = (
            args.pack_concat_feed_min_total_mib
        )
        feed_summary["concat_pack_max_total_mib"] = args.pack_concat_feed_max_total_mib
        feed_summary["concat_pack_packed_inputs"] = (
            concat_pack_state.get("packed_inputs", []) if concat_pack_state else []
        )
        feed_summary["concat_pack_frozen"] = bool(
            concat_pack_state
            and any(
                item.get("frozen", False)
                for item in concat_pack_state.get("packed_inputs", [])
            )
        )
        feed_summary["concat_pack_frozen_nbytes"] = (
            sum(
                int(item.get("nbytes", 0) or 0)
                for item in concat_pack_state.get("packed_inputs", [])
                if item.get("frozen", False)
            )
            if concat_pack_state
            else 0
        )
        feed_summary["concat_pack_frozen_mib"] = (
            feed_summary["concat_pack_frozen_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["concat_pack_cached"] = bool(
            concat_pack_state
            and any(
                item.get("cached", False)
                for item in concat_pack_state.get("packed_inputs", [])
            )
        )
        feed_summary["concat_pack_cached_nbytes"] = (
            sum(
                int(item.get("nbytes", 0) or 0)
                for item in concat_pack_state.get("packed_inputs", [])
                if item.get("cached", False)
            )
            if concat_pack_state
            else 0
        )
        feed_summary["concat_pack_cached_mib"] = (
            feed_summary["concat_pack_cached_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["concat_pack_cache_initializers"] = (
            len(concat_pack_state.get("cache_initializers", []))
            if concat_pack_state
            else 0
        )
        feed_summary["concat_pack_error"] = (
            concat_pack_state.get("fallback_error") if concat_pack_state else None
        )
        feed_summary["concat_pack_pinned_error"] = (
            concat_pack_state.get("pinned_error") if concat_pack_state else None
        )
        feed_summary["concat_pack_diagnostics"] = (
            concat_pack_state.get("diagnostics") if concat_pack_state else None
        )
        feed_summary["concat_static_precompute"] = (
            concat_pack_state.get("concat_static_precompute") if concat_pack_state else None
        )
        slice_compact_used = bool(
            slice_compact_state
            and slice_compact_state.get("selected_slice_nodes", 0) > 0
        )
        feed_summary["slice_compact_requested"] = args.compact_slice_feed == "on"
        feed_summary["slice_compact_used"] = slice_compact_used
        feed_summary["slice_compact_source_inputs"] = (
            slice_compact_state.get("selected_source_inputs", 0)
            if slice_compact_state
            else 0
        )
        feed_summary["slice_compact_selected_nodes"] = (
            slice_compact_state.get("selected_slice_nodes", 0)
            if slice_compact_state
            else 0
        )
        feed_summary["slice_compact_input_delta"] = (
            slice_compact_state.get("input_delta", 0) if slice_compact_state else 0
        )
        feed_summary["slice_compact_original_nbytes"] = (
            slice_compact_state.get("selected_original_nbytes", 0)
            if slice_compact_state
            else 0
        )
        feed_summary["slice_compact_selected_nbytes"] = (
            slice_compact_state.get("selected_nbytes", 0)
            if slice_compact_state
            else 0
        )
        feed_summary["slice_compact_saved_nbytes"] = (
            slice_compact_state.get("saved_nbytes", 0) if slice_compact_state else 0
        )
        feed_summary["slice_compact_saved_mib"] = (
            feed_summary["slice_compact_saved_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["slice_compact_inputs"] = (
            slice_compact_state.get("compacted_inputs", [])
            if slice_compact_state
            else []
        )
        feed_summary["slice_compact_error"] = (
            slice_compact_state.get("fallback_error") if slice_compact_state else None
        )
        feed_summary["slice_compact_diagnostics"] = (
            slice_compact_state.get("diagnostics") if slice_compact_state else None
        )
        slice_pack_used = bool(
            slice_pack_state and slice_pack_state.get("selected_slice_nodes", 0) > 0
        )
        feed_summary["slice_pack_requested"] = should_pack_slice_feed(args)
        feed_summary["slice_pack_used"] = slice_pack_used
        feed_summary["slice_pack_source_inputs"] = (
            slice_pack_state.get("selected_source_inputs", 0)
            if slice_pack_state
            else 0
        )
        feed_summary["slice_pack_selected_nodes"] = (
            slice_pack_state.get("selected_slice_nodes", 0)
            if slice_pack_state
            else 0
        )
        feed_summary["slice_pack_original_nbytes"] = (
            slice_pack_state.get("selected_original_nbytes", 0)
            if slice_pack_state
            else 0
        )
        feed_summary["slice_pack_selected_nbytes"] = (
            slice_pack_state.get("selected_nbytes", 0) if slice_pack_state else 0
        )
        feed_summary["slice_pack_saved_nbytes"] = (
            slice_pack_state.get("saved_nbytes", 0) if slice_pack_state else 0
        )
        feed_summary["slice_pack_original_mib"] = (
            feed_summary["slice_pack_original_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["slice_pack_selected_mib"] = (
            feed_summary["slice_pack_selected_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["slice_pack_saved_mib"] = (
            feed_summary["slice_pack_saved_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["slice_pack_min_saved_mib"] = args.pack_slice_feed_min_saved_mib
        feed_summary["slice_pack_min_total_saved_mib"] = (
            args.pack_slice_feed_min_total_saved_mib
        )
        feed_summary["slice_pack_ops"] = args.pack_slice_feed_ops
        feed_summary["slice_pack_max_total_mib"] = args.pack_slice_feed_max_total_mib
        feed_summary["slice_pack_packed_inputs"] = (
            slice_pack_state.get("packed_inputs", []) if slice_pack_state else []
        )
        feed_summary["slice_pack_error"] = (
            slice_pack_state.get("fallback_error") if slice_pack_state else None
        )
        feed_summary["slice_pack_pinned_error"] = (
            slice_pack_state.get("pinned_error") if slice_pack_state else None
        )
        feed_summary["slice_pack_diagnostics"] = (
            slice_pack_state.get("diagnostics") if slice_pack_state else None
        )
        slice_pack_diagnostics = feed_summary["slice_pack_diagnostics"] or {}
        feed_summary["slice_pack_mode"] = slice_pack_diagnostics.get(
            "mode", "off"
        )
        feed_summary["slice_pack_single_consumer_only"] = (
            args.pack_slice_feed_single_consumer_only == "on"
        )
        feed_summary["slice_pack_single_consumer_candidates"] = (
            slice_pack_diagnostics.get("single_consumer_candidates", 0)
        )
        feed_summary["slice_pack_single_consumer_filtered"] = (
            slice_pack_diagnostics.get("single_consumer_filtered", 0)
        )
        feed_summary["slice_pack_direct_added_inputs"] = (
            slice_pack_diagnostics.get("direct_added_inputs", 0)
        )
        small_pack_used = bool(
            small_pack_state and small_pack_state.get("selected_count", 0) > 0
        )
        feed_summary["small_pack_requested"] = small_pack_requested
        feed_summary["small_pack_used"] = small_pack_used
        feed_summary["small_pack_original_inputs"] = len(feed_items)
        feed_summary["small_pack_packed_feed_inputs"] = (
            len(small_pack_state.get("packed_inputs", [])) if small_pack_state else 0
        )
        feed_summary["small_pack_selected_inputs"] = (
            small_pack_state.get("selected_count", 0) if small_pack_state else 0
        )
        feed_summary["small_pack_selected_nbytes"] = (
            small_pack_state.get("selected_nbytes", 0) if small_pack_state else 0
        )
        feed_summary["small_pack_selected_mib"] = (
            feed_summary["small_pack_selected_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["small_pack_max_bytes"] = args.pack_small_feed_max_bytes
        feed_summary["small_pack_max_total_mib"] = args.pack_small_feed_max_total_mib
        feed_summary["small_pack_unpack_op"] = args.pack_small_feed_unpack_op
        feed_summary["small_pack_min_inputs"] = args.pack_small_feed_min_inputs
        feed_summary["small_pack_min_total_mib"] = (
            args.pack_small_feed_min_total_mib
        )
        feed_summary["small_pack_packed_inputs"] = (
            small_pack_state.get("packed_inputs", []) if small_pack_state else []
        )
        feed_summary["small_pack_error"] = (
            small_pack_state.get("fallback_error") if small_pack_state else None
        )
        feed_summary["small_pack_pinned_error"] = (
            small_pack_state.get("pinned_error") if small_pack_state else None
        )
        remaining_pack_used = bool(
            remaining_pack_state
            and remaining_pack_state.get("selected_count", 0) > 0
        )
        feed_summary["remaining_pack_requested"] = remaining_pack_requested
        feed_summary["remaining_pack_used"] = remaining_pack_used
        feed_summary["remaining_pack_candidate_inputs"] = len(
            remaining_pack_candidate_items
        )
        feed_summary["remaining_pack_packed_feed_inputs"] = (
            len(remaining_pack_state.get("packed_inputs", []))
            if remaining_pack_state
            else 0
        )
        feed_summary["remaining_pack_selected_inputs"] = (
            remaining_pack_state.get("selected_count", 0)
            if remaining_pack_state
            else 0
        )
        feed_summary["remaining_pack_selected_nbytes"] = (
            remaining_pack_state.get("selected_nbytes", 0)
            if remaining_pack_state
            else 0
        )
        feed_summary["remaining_pack_selected_mib"] = (
            feed_summary["remaining_pack_selected_nbytes"] / (1024.0 * 1024.0)
        )
        feed_summary["remaining_pack_max_bytes"] = args.pack_remaining_feed_max_bytes
        feed_summary["remaining_pack_max_total_mib"] = (
            args.pack_remaining_feed_max_total_mib
        )
        feed_summary["remaining_pack_unpack_op"] = args.pack_remaining_feed_unpack_op
        feed_summary["remaining_pack_min_inputs"] = args.pack_remaining_feed_min_inputs
        feed_summary["remaining_pack_min_total_mib"] = (
            args.pack_remaining_feed_min_total_mib
        )
        feed_summary["remaining_pack_packed_inputs"] = (
            remaining_pack_state.get("packed_inputs", [])
            if remaining_pack_state
            else []
        )
        feed_summary["remaining_pack_error"] = (
            remaining_pack_state.get("fallback_error")
            if remaining_pack_state
            else None
        )
        feed_summary["remaining_pack_pinned_error"] = (
            remaining_pack_state.get("pinned_error")
            if remaining_pack_state
            else None
        )
        if small_pack_used:
            feed_summary["pinned_feed_used"] = bool(
                feed_state["pinned_feed_used"]
                or any(
                    item.get("pinned", False)
                    for item in small_pack_state.get("packed_inputs", [])
                )
            )
        if remaining_pack_used:
            feed_summary["pinned_feed_used"] = bool(
                feed_summary["pinned_feed_used"]
                or any(
                    item.get("pinned", False)
                    for item in remaining_pack_state.get("packed_inputs", [])
                )
            )
        if concat_pack_used:
            feed_summary["pinned_feed_used"] = bool(
                feed_summary["pinned_feed_used"]
                or any(
                    item.get("pinned", False)
                    for item in concat_pack_state.get("packed_inputs", [])
                )
            )
        if slice_pack_used:
            feed_summary["pinned_feed_used"] = bool(
                feed_summary["pinned_feed_used"]
                or any(
                    item.get("pinned", False)
                    for item in slice_pack_state.get("packed_inputs", [])
                )
            )

        graph_dump = {
            "enabled": env_flag_enabled(GRAPH_DUMP_ENV),
            "plugin_loaded": musa_loaded,
            "optimizer_enabled": False,
            "dump_dir": None,
            "files": {},
        }
        default_dump_dir = None
        if musa_loaded and args.musa_optimizer:
            graph_dump["optimizer_enabled"] = True
            if graph_dump["enabled"]:
                default_dump_dir = runner_out / f"{spec_path.stem}_bs_{bs}"

        config = create_session_config(args, musa_loaded=musa_loaded)
        run_error = None
        lat_ms = []
        last_vals = None
        callable_used = False
        callable_error = None
        feed_immutability = {
            "enabled": args.verify_feed_immutability == "on",
            "verified": False,
            "checked_boundaries": 0,
            "first_changed_boundary": None,
            "total_inputs": len(feed_dict),
            "total_bytes": 0,
            "changed_inputs": 0,
            "changed_bytes": 0,
            "pointer_changed_inputs": 0,
            "changed_names": [],
            "pointer_changed_names": [],
        }
        with configured_graph_dump_dir(default_dump_dir) as active_dump_dir:
            if active_dump_dir is not None:
                graph_dump["dump_dir"] = str(active_dump_dir)
            log_run_phase_progress(
                args,
                "session_create_begin",
                feeds=len(feed_dict),
            )
            with tf.compat.v1.Session(graph=graph, config=config) as sess:
                log_run_phase_progress(args, "session_create_done")
                try:
                    if concat_pack_state:
                        for initializer in concat_pack_state.get(
                            "cache_initializers", []
                        ):
                            sess.run(
                                initializer["assign_op"],
                                feed_dict={
                                    initializer["feed_tensor"]: initializer["value"]
                                },
                            )
                    feed_tensors = list(feed_dict.keys())
                    feed_values = [feed_dict[tensor] for tensor in feed_tensors]
                    feed_snapshot_before = None
                    if args.verify_feed_immutability == "on":
                        feed_snapshot_before = snapshot_feed_values(
                            feed_tensors, feed_values
                        )

                    def verify_feed_boundary(boundary):
                        if feed_snapshot_before is None:
                            return
                        current_snapshot = snapshot_feed_values(
                            feed_tensors, feed_values
                        )
                        comparison = compare_feed_snapshots(
                            feed_snapshot_before, current_snapshot
                        )
                        feed_immutability.update(comparison)
                        feed_immutability["checked_boundaries"] += 1
                        if comparison["changed_inputs"] == 0:
                            return
                        feed_immutability["first_changed_boundary"] = boundary
                        print(
                            "[ERROR] feed_immutability: "
                            f"verified=False boundary={boundary} "
                            f"changed_inputs={comparison['changed_inputs']} "
                            f"changed_mib={comparison['changed_bytes'] / (1024.0 * 1024.0):.3f} "
                            f"changed_names={comparison['changed_names'][:20]}",
                            flush=True,
                        )
                        raise RuntimeError(
                            "feed contents changed at iteration boundary "
                            f"{boundary}: "
                            + ",".join(comparison["changed_names"][:20])
                        )

                    run_once = None
                    if should_use_callable(args):
                        try:
                            log_run_phase_progress(
                                args,
                                "make_callable_begin",
                                feeds=len(feed_tensors),
                            )
                            callable_runner = sess.make_callable(
                                fetches, feed_list=feed_tensors
                            )
                            log_run_phase_progress(
                                args,
                                "make_callable_done",
                                feeds=len(feed_tensors),
                            )

                            def run_once():
                                return callable_runner(*feed_values)

                            callable_used = True
                        except Exception as exc:
                            log_run_phase_progress(
                                args,
                                "make_callable_failed",
                                error=type(exc).__name__,
                            )
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

                    for warmup_index in range(max(0, args.warmup)):
                        log_run_phase_progress(
                            args,
                            "warmup_begin",
                            index=warmup_index + 1,
                            total=max(0, args.warmup),
                        )
                        warmup_start = time.perf_counter()
                        try:
                            run_once()
                        finally:
                            warmup_end = time.perf_counter()
                            verify_feed_boundary(f"warmup_{warmup_index + 1}")
                        log_run_phase_progress(
                            args,
                            "warmup_done",
                            index=warmup_index + 1,
                            elapsed_ms=f"{(warmup_end - warmup_start) * 1000.0:.3f}",
                        )
                    for run_index in range(max(1, args.run_iters)):
                        log_run_phase_progress(
                            args,
                            "run_iter_begin",
                            index=run_index + 1,
                            total=max(1, args.run_iters),
                        )
                        start = time.perf_counter()
                        try:
                            last_vals = run_once()
                        finally:
                            end = time.perf_counter()
                            verify_feed_boundary(f"run_{run_index + 1}")
                        lat_ms.append((end - start) * 1000.0)
                        log_run_phase_progress(
                            args,
                            "run_iter_done",
                            index=run_index + 1,
                            elapsed_ms=f"{lat_ms[-1]:.3f}",
                        )
                    if feed_snapshot_before is not None:
                        feed_immutability["verified"] = True
                        print(
                            "[INFO] feed_immutability: "
                            f"verified=True inputs={feed_immutability['total_inputs']} "
                            f"checked_boundaries={feed_immutability['checked_boundaries']} "
                            f"total_mib={feed_immutability['total_bytes'] / (1024.0 * 1024.0):.3f} "
                            f"changed_inputs={feed_immutability['changed_inputs']} "
                            f"changed_mib={feed_immutability['changed_bytes'] / (1024.0 * 1024.0):.3f} "
                            "pointer_changed_inputs="
                            f"{feed_immutability['pointer_changed_inputs']} "
                            f"changed_names={feed_immutability['changed_names'][:20]}",
                            flush=True,
                        )
                        if feed_immutability["changed_inputs"] > 0:
                            raise RuntimeError(
                                "feed contents changed across warmup/timed iterations: "
                                + ",".join(feed_immutability["changed_names"][:20])
                            )
                except Exception:
                    run_error = traceback.format_exc()

        if graph_dump["dump_dir"]:
            graph_dump["files"] = collect_graph_dump_files(graph_dump["dump_dir"])

    result_shapes = []
    if last_vals is not None:
        result_values = last_vals if isinstance(last_vals, (list, tuple)) else [last_vals]
        result_shapes = [
            {
                "shape": list(getattr(value, "shape", [])),
                "dtype": str(getattr(value, "dtype", "")),
                "nbytes": int(getattr(value, "nbytes", 0) or 0),
                "mib": int(getattr(value, "nbytes", 0) or 0) / (1024.0 * 1024.0),
            }
            for value in result_values
        ]

    return {
        "spec_path": str(spec_path),
        "pb_path": str(pb_path),
        "batch_size": bs,
        "num_outputs": len(output_info),
        "output_spec_count": len(output_spec),
        "selected_output_indices": selected_output_indices,
        "outputs": output_info,
        "output_summary": {
            "fetch_mode": fetch_mode,
            "selected_output_indices": selected_output_indices,
            "estimated_total_nbytes": output_nbytes,
            "estimated_total_mib": output_nbytes / (1024.0 * 1024.0),
        },
        "output_fetch_plan": (
            {
                "enabled": output_fetch_plan["enabled"],
                "num_outputs": output_fetch_plan["num_outputs"],
                "device_fetch_count": output_fetch_plan["device_fetch_count"],
                "original_device_fetch_count": output_fetch_plan.get(
                    "original_device_fetch_count",
                    output_fetch_plan["device_fetch_count"],
                ),
                "host_output_count": output_fetch_plan["host_output_count"],
                "deduped_output_count": output_fetch_plan["deduped_output_count"],
                "pack_output_fetches": output_fetch_plan.get(
                    "pack_output_fetches", False
                ),
                "packed_device_fetch_count": output_fetch_plan.get(
                    "packed_device_fetch_count", 0
                ),
                "packed_output_groups": output_fetch_plan.get(
                    "packed_output_groups", []
                ),
                "device_fetch_names": output_fetch_plan["device_fetch_names"],
            }
            if output_fetch_plan is not None
            else None
        ),
        "output_dependency_summary": output_dependency_summary,
        "feed_summary": feed_summary,
        "callable": {
            "requested": should_use_callable(args),
            "used": callable_used,
            "error": callable_error,
        },
        "feed_immutability": feed_immutability,
        "result_shapes": result_shapes,
        "result_total_mib": sum(item.get("mib", 0.0) for item in result_shapes),
        "status": "ok" if run_error is None else "failed",
        "error_core": extract_core_error(run_error),
        "error_tail": extract_error_tail(run_error),
        "error": run_error,
        "graph_dump": graph_dump,
        "graph_rewrite": graph_rewrite_summary,
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
    parser.add_argument("--spec_dir", default=None, help="Directory to scan for *.spec files.")
    parser.add_argument("--pb", default=None, help="Path to frozen_graph_*.pb. Only with --spec.")
    parser.add_argument(
        "--optimization_profile",
        choices=["auto", "off", "meta1", "meta2"],
        default="auto",
        help="Apply the verified model-specific optimization defaults.",
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
    parser.add_argument(
        "--summarize_xla_dump_dir",
        default=None,
        help="Only summarize an existing XLA dump directory and exit.",
    )
    parser.add_argument(
        "--analyze_xla_hot_fusion_dump_dir",
        default=None,
        help=(
            "Only analyze hot fusion bodies in an existing XLA HLO dump directory "
            "and exit."
        ),
    )
    parser.add_argument(
        "--analyze_xla_hot_fusions",
        default="",
        help=(
            "Comma-separated fusion instruction names to inspect, for example "
            "fusion.292,fusion.220. Empty analyzes largest fusion outputs."
        ),
    )
    parser.add_argument(
        "--analyze_xla_hot_fusion_limit",
        type=int,
        default=20,
        help="Maximum number of hot fusion summary items to print.",
    )
    parser.add_argument(
        "--compare_xla_dump_baseline_dir",
        default=None,
        help="Baseline XLA dump directory for diagnostic opcode diff.",
    )
    parser.add_argument(
        "--compare_xla_dump_freeze_dir",
        default=None,
        help="Freeze/experiment XLA dump directory for diagnostic opcode diff.",
    )
    parser.add_argument(
        "--compare_thunk_log_baseline",
        default=None,
        help="Baseline log containing MUSA_XLA_THUNK_DIAGNOSTICS lines.",
    )
    parser.add_argument(
        "--compare_thunk_log_experiment",
        default=None,
        help="Experiment log containing MUSA_XLA_THUNK_DIAGNOSTICS lines.",
    )
    parser.add_argument("--bs", default="1024", help="Batch size or comma list, e.g. 1,8,32.")
    parser.add_argument("--unknown_dim", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--run_iters", type=int, default=10)
    parser.add_argument(
        "--verify_feed_immutability",
        choices=("on", "off"),
        default="off",
        help=(
            "Hash feed values at every warmup/run boundary outside the measured "
            "interval. Use as a separate correctness check because scanning feeds "
            "perturbs subsequent performance samples."
        ),
    )
    parser.add_argument(
        "--run_phase_progress",
        choices=("on", "off"),
        default="off",
        help="Print graph import, Session, callable, warmup, and timed-run phase boundaries.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_root", default="runner_out")
    parser.add_argument("--strict", type=parse_bool, default=True)
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only parse spec/pb and report static input/op statistics; do not load MUSA runtime or run Session.",
    )
    parser.add_argument(
        "--device",
        default="/device:MUSA:0",
        help="TensorFlow device scope, e.g. /device:MUSA:0, /device:CPU:0.",
    )
    parser.add_argument("--allow_soft_placement", type=parse_bool, default=True)
    parser.add_argument("--log_device_placement", type=parse_bool, default=False)
    parser.add_argument("--musa_optimizer", type=parse_bool, default=True)
    parser.add_argument(
        "--pinned_feed",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use MUSA pinned host memory for generated feed arrays. auto enables it for MUSA XLA and falls back on failure.",
    )
    parser.add_argument(
        "--pinned_feed_arena",
        choices=("auto", "on", "off"),
        default="off",
        help="Store final dense feed values as aligned views of one pinned host allocation for direct PJRT arena H2D. Default off.",
    )
    parser.add_argument(
        "--feed_only_reachable",
        type=parse_bool,
        default=True,
        help="Only feed placeholders that are reachable from requested outputs.",
    )
    parser.add_argument(
        "--use_callable",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use TensorFlow Session.make_callable for repeated dynamic-feed runs.",
    )
    parser.add_argument(
        "--bypass_identity",
        choices=("auto", "on", "off"),
        default="auto",
        help="Bypass no-control Identity nodes before graph import to reduce XLA graph plumbing.",
    )
    parser.add_argument(
        "--bypass_identity_min_nodes",
        type=int,
        default=1000,
        help="In auto mode, prune Identity nodes only when the graph has at least this many Identity nodes.",
    )
    parser.add_argument(
        "--optimize_output_fetches",
        choices=("auto", "on", "off"),
        default="auto",
        help="Deduplicate identical output fetches and satisfy direct feed/Identity(feed) outputs from host values.",
    )
    parser.add_argument(
        "--pack_output_fetches",
        choices=("auto", "on", "off"),
        default="auto",
        help="Pack small device output fetches by dtype into fewer tensors, then reconstruct outputs on host.",
    )
    parser.add_argument(
        "--pack_output_fetches_min_outputs",
        type=int,
        default=8,
        help="Pack output fetches only when at least this many device outputs share a dtype.",
    )
    parser.add_argument(
        "--pack_output_fetches_max_total_mib",
        type=float,
        default=16.0,
        help="Do not pack an output dtype group larger than this many MiB. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--rewrite_pow_square",
        choices=("auto", "on", "off"),
        default="auto",
        help="Rewrite Pow(x, const 2) to Mul(x, x) before graph import.",
    )
    parser.add_argument(
        "--rewrite_static_shape_subgraph",
        choices=("on", "off"),
        default="off",
        help="Replace static Shape/Rank/Size nodes with Const before graph import. Diagnostic opt-in.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul",
        choices=("auto", "on", "off"),
        default="off",
        help="Rewrite groups of MatMul with the same lhs into one wider MatMul plus Slice outputs.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_min_group",
        type=int,
        default=2,
        help="Minimum number of same-lhs MatMul nodes required before graph-level fusion.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_min_placeholders",
        type=int,
        default=1000,
        help="Only run same-lhs MatMul rewrite on large-feed graphs with at least this many Placeholder nodes. Use 0 to force.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_auto_min_reduction",
        type=int,
        default=32,
        help="For --rewrite_same_lhs_matmul auto, require at least this estimated MatMul launch reduction. Use 0 to disable.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_groups",
        type=int,
        default=16,
        help="Maximum same-lhs MatMul groups to rewrite. Use 0 for no cap; higher values may increase XLA compile time.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_total_cols",
        type=int,
        default=8192,
        help="Skip a same-lhs MatMul group if concatenated rhs output columns exceed this cap. Use 0 for no cap.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_score",
        choices=("flops", "hybrid", "count", "small"),
        default="flops",
        help="Rank same-lhs MatMul rewrite candidates by estimated FLOPs, FLOPs times launch reduction, launch-count reduction, or small-GEMM preference.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_min_mflops",
        type=float,
        default=0.0,
        help="Skip same-lhs MatMul rewrite candidates below this estimated MFLOP threshold. Use 0 to disable.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_include_rhs",
        choices=("auto", "on", "off"),
        default="off",
        help="Also rewrite same-rhs MatMul groups by concatenating lhs along rows.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_fuse_biasadd",
        choices=("on", "off"),
        default="off",
        help="For same-lhs MatMul rewrite, also fuse chunks where every MatMul feeds one compatible BiasAdd.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_fuse_post_unary",
        choices=("on", "off"),
        default="off",
        help="For same-lhs MatMul+BiasAdd rewrite, also hoist matching unary elementwise ops before the per-output Slice.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_fuse_post_binary",
        choices=("on", "off"),
        default="off",
        help="For same-lhs MatMul+BiasAdd rewrite, also hoist matching binary elementwise ops before the per-output Slice.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_compact_post_concat",
        choices=("on", "off"),
        default="off",
        help="For same-lhs MatMul+BiasAdd post-op rewrite, replace compatible downstream ConcatV2 nodes with one Slice from the fused post-op tensor.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_rhs_producer_ops",
        default="",
        help="Comma-separated RHS producer op allowlist for graph-level same-LHS MatMul rewrite, e.g. Reshape. Empty means no filter.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_k_dim",
        type=int,
        default=0,
        help="Skip graph-level same-LHS MatMul rewrite candidates whose K dimension exceeds this cap. 0 disables the cap.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_single_out_cols",
        type=int,
        default=0,
        help="Skip graph-level same-LHS MatMul candidates whose individual output columns exceed this cap. 0 disables the cap.",
    )
    parser.add_argument(
        "--rewrite_same_lhs_matmul_max_total_rows",
        type=int,
        default=4096,
        help="For same-rhs rewrite, cap concatenated lhs rows. Use 0 for no cap.",
    )
    parser.add_argument(
        "--rewrite_same_shape_batch_matmul",
        choices=("on", "off"),
        default="off",
        help="Rewrite same-shape MatMul groups into Pack + BatchMatMulV2 + Slice/Squeeze outputs.",
    )
    parser.add_argument(
        "--rewrite_same_shape_batch_matmul_min_group",
        type=int,
        default=4,
        help="Minimum same-shape MatMul count required before BatchMatMulV2 graph rewrite.",
    )
    parser.add_argument(
        "--rewrite_same_shape_batch_matmul_max_groups",
        type=int,
        default=8,
        help="Maximum same-shape BatchMatMulV2 rewrite groups. Use 0 for no cap.",
    )
    parser.add_argument(
        "--rewrite_same_shape_batch_matmul_max_group_size",
        type=int,
        default=64,
        help="Maximum MatMul nodes packed into one BatchMatMulV2 group. Use 0 for no cap.",
    )
    parser.add_argument(
        "--pack_small_feed",
        choices=("auto", "on", "off"),
        default="off",
        help="Dynamically pack small feed tensors by dtype before graph import to reduce PJRT BufferFromHost submissions. Default off keeps MUSA feed allocations small and stable.",
    )
    parser.add_argument(
        "--pack_small_feed_max_bytes",
        type=int,
        default=131072,
        help="Only pack individual feed tensors with nbytes <= this threshold.",
    )
    parser.add_argument(
        "--pack_small_feed_max_total_mib",
        type=float,
        default=12.0,
        help="Cap total bytes selected for small-feed packing. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--pack_small_feed_unpack_op",
        choices=("split", "slice"),
        default="slice",
        help="Unpack packed small feeds with one tf.split per dtype group or many tf.slice ops.",
    )
    parser.add_argument(
        "--pack_small_feed_min_inputs",
        type=int,
        default=32,
        help="auto mode enables small-feed packing only when at least this many small inputs exist.",
    )
    parser.add_argument(
        "--pack_small_feed_min_total_mib",
        type=float,
        default=4.0,
        help="auto mode enables small-feed packing only when selected small inputs total at least this many MiB.",
    )
    parser.add_argument(
        "--pack_remaining_feed",
        choices=("auto", "on", "off"),
        default="off",
        help="Pack remaining feed tensors by dtype after concat/slice/small-feed packing. Default off preserves the validated baseline.",
    )
    parser.add_argument(
        "--pack_remaining_feed_max_bytes",
        type=int,
        default=16777216,
        help="Only pack individual remaining feed tensors with nbytes <= this threshold.",
    )
    parser.add_argument(
        "--pack_remaining_feed_max_total_mib",
        type=float,
        default=192.0,
        help="Cap total bytes selected for remaining-feed packing. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--pack_remaining_feed_unpack_op",
        choices=("split", "slice"),
        default="slice",
        help="Unpack packed remaining feeds with one tf.split per dtype group or many tf.slice ops.",
    )
    parser.add_argument(
        "--pack_remaining_feed_min_inputs",
        type=int,
        default=8,
        help="auto mode enables remaining-feed packing only when at least this many eligible inputs exist.",
    )
    parser.add_argument(
        "--pack_remaining_feed_min_total_mib",
        type=float,
        default=16.0,
        help="auto mode enables remaining-feed packing only when selected remaining inputs total at least this many MiB.",
    )
    parser.add_argument(
        "--pack_concat_feed",
        choices=("auto", "on", "off"),
        default="auto",
        help="Dynamically replace eligible placeholder-only ConcatV2 nodes with one packed feed tensor.",
    )
    parser.add_argument(
        "--pack_concat_feed_min_inputs",
        type=int,
        default=8,
        help="Pack a ConcatV2 feed group only when it has at least this many placeholder inputs.",
    )
    parser.add_argument(
        "--pack_concat_feed_min_total_mib",
        type=float,
        default=1.0,
        help="Pack a ConcatV2 feed group only when the packed tensor is at least this many MiB.",
    )
    parser.add_argument(
        "--pack_concat_feed_max_total_mib",
        type=float,
        default=256.0,
        help="Cap total bytes selected for concat-feed packing. Default 256 keeps the meta2 performance path packed; use 0 to disable the cap.",
    )
    parser.add_argument(
        "--pack_concat_feed_chunk_max_mib",
        type=float,
        default=0.0,
        help="Split an eligible concat-feed pack into chunks no larger than this many MiB. Default 0 disables chunking; set a positive value only as an OOM workaround.",
    )
    parser.add_argument(
        "--freeze_concat_packed_feed",
        choices=("on", "off"),
        default="off",
        help="Upper-bound experiment: materialize concat packed feed tensors as graph constants instead of per-run feeds.",
    )
    parser.add_argument(
        "--cache_concat_packed_feed",
        choices=("on", "off"),
        default="off",
        help="Upload concat packed feed tensors once into graph variables, then reuse them across timed runs.",
    )
    parser.add_argument(
        "--rewrite_concat_static_precompute",
        choices=("on", "off"),
        default="off",
        help="Opt-in experiment: replace shape-only concat packed feed downstream nodes with constants using static packed shapes.",
    )
    parser.add_argument(
        "--compact_slice_feed",
        choices=("on", "off"),
        default="off",
        help="Replace static Slice-only source feeds with one minimal bounding-box feed per source while preserving the feed count.",
    )
    parser.add_argument(
        "--compact_slice_feed_min_saved_mib",
        type=float,
        default=0.0,
        help="Compact a Slice-only source feed only when its bounding box saves at least this many MiB.",
    )
    parser.add_argument(
        "--compact_slice_feed_min_total_saved_mib",
        type=float,
        default=1.0,
        help="Apply bounding-box slice feed compaction only when total selected savings reach this many MiB.",
    )
    parser.add_argument(
        "--pack_slice_feed",
        choices=("auto", "on", "off"),
        default="off",
        help="Replace eligible placeholder-only Slice nodes with smaller dynamic feed tensors. Default off avoids large packed slice feeds on MUSA.",
    )
    parser.add_argument(
        "--pack_slice_feed_ops",
        default="Slice",
        help="Comma-separated slice-like ops eligible for host-side dynamic slicing, e.g. Slice or Slice,StridedSlice.",
    )
    parser.add_argument(
        "--pack_slice_feed_min_saved_mib",
        type=float,
        default=0.0,
        help="Enable slice-feed packing for a source input only when it saves at least this many MiB.",
    )
    parser.add_argument(
        "--pack_slice_feed_min_total_saved_mib",
        type=float,
        default=1.0,
        help="auto mode keeps slice-feed packing only when selected slices save at least this many MiB in total.",
    )
    parser.add_argument(
        "--pack_slice_feed_max_total_mib",
        type=float,
        default=0.0,
        help="Cap total bytes selected for slice-feed packing. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--pack_slice_feed_max_added_inputs",
        type=int,
        default=0,
        help="auto mode keeps slice-feed packing only when it adds at most this many feed tensors. Use -1 to disable this guard.",
    )
    parser.add_argument(
        "--pack_slice_feed_grouped_min_saved_mib",
        type=float,
        default=8.0,
        help="auto mode switches high-fanout slice-feed packing to one flat feed per source input only when it saves at least this many MiB.",
    )
    parser.add_argument(
        "--pack_slice_feed_single_consumer_only",
        choices=("on", "off"),
        default="off",
        help="Materialize only Slice sources with exactly one Slice consumer, preserving the feed count while removing that Slice op.",
    )
    parser.add_argument(
        "--pack_slice_feed_max_direct_added_inputs",
        type=int,
        default=0,
        help="With single-consumer-only mode, additionally materialize multi-consumer Slice groups up to this total added-feed cap.",
    )
    parser.add_argument(
        "--fetch_mode",
        choices=("outputs", "execute_only", "barrier_scalar", "checksum_scalar"),
        default="outputs",
        help="outputs materializes output tensors; execute_only requests output ops without a fetched tensor; barrier_scalar probes control-dependency pruning; checksum_scalar reduces every numeric output on device and fetches one data-dependent scalar.",
    )
    parser.add_argument(
        "--output_indices",
        default="all",
        help="Comma/range selection of output_spec indices to fetch, e.g. all, 0, 0,2, or 0-3.",
    )
    parser.add_argument(
        "--pjrt_force_host_copy",
        choices=("auto", "on", "off"),
        default="off",
        help="Control MUSA_PJRT_FORCE_HOST_BUFFER_COPY for MUSA XLA. off avoids an extra host-side buffer copy.",
    )
    parser.add_argument(
        "--pjrt_max_inflight_transfers",
        default="0",
        help="MUSA_PJRT_MAX_INFLIGHT_TRANSFERS. Use 1 for conservative serialized H2D, 0 to disable the gate, or a positive integer.",
    )
    parser.add_argument(
        "--pjrt_max_inflight_executes",
        default="0",
        help="MUSA_PJRT_MAX_INFLIGHT_EXECUTES. Use 1 for conservative serialized execute, 0 to disable the gate, or a positive integer.",
    )
    parser.add_argument(
        "--pjrt_wait_transfer_done",
        choices=("auto", "on", "off"),
        default="off",
        help="Control MUSA_PJRT_WAIT_TRANSFER_DONE. on is conservative and helps diagnose async H2D transfer failures.",
    )
    parser.add_argument(
        "--pjrt_wait_execute_done",
        choices=("auto", "on", "off"),
        default="off",
        help="Control MUSA_PJRT_WAIT_EXECUTE_DONE. on waits for device execution before returning and is intended for synchronization diagnostics.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers",
        choices=("auto", "on", "off"),
        default="off",
        help="Reuse persistent device buffers for repeated dense host pointers. Requires transfer and execute waits and is intended for graph replay diagnostics.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_diagnostics",
        choices=("auto", "on", "off"),
        default="off",
        help="Collect and print reusable-host-buffer timing and cache diagnostics.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_async",
        choices=("auto", "on", "off"),
        default="off",
        help="Stage reusable host inputs in pinned memory and batch asynchronous H2D copies before execute. Requires --pjrt_reuse_host_buffers on.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena",
        choices=("auto", "on", "off"),
        default="off",
        help="Pack reusable host inputs into one pinned host/device arena and issue one H2D copy before execute. Requires --pjrt_reuse_host_buffers on.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_parallel_pack",
        choices=("auto", "on", "off"),
        default="off",
        help="Use persistent CPU workers to copy large reusable inputs into the host arena. Default off.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_fast_pack",
        choices=("auto", "on", "off"),
        default="off",
        help="Use a fixed-generation worker barrier for parallel arena packing, avoiding per-input task allocations. Default off.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_pack_threads",
        type=int,
        default=4,
        help="Total CPU threads, including the caller, used for each large arena pack copy.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_pack_min_bytes",
        type=int,
        default=1048576,
        help="Minimum input size in bytes for parallel arena packing.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_dirty_ranges",
        choices=("auto", "on", "off"),
        default="off",
        help="Copy only changed reusable arena byte ranges when their estimated cost is below a full-arena H2D. Default off.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_arena_pool_order_layout",
        choices=("auto", "on", "off"),
        default="off",
        help="Lay out reusable arena entries in first-pool order to cluster dirty ranges from nearby child executions. Default off.",
    )
    parser.add_argument(
        "--pjrt_reuse_host_buffers_trust_contents",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Trust repeated feed pointers to retain identical contents and skip "
            "arena host packing/H2D after the first successful upload. "
            "Diagnostic only; unsafe for mutable feeds."
        ),
    )
    parser.add_argument(
        "--pjrt_cache_reused_buffer_views",
        choices=("auto", "on", "off"),
        default="off",
        help="Cache PJRT device-buffer views for stable reusable arena inputs. Requires synchronous single-inflight reusable-buffer arena mode.",
    )
    parser.add_argument(
        "--pjrt_cache_reused_buffer_views_trust_lifetime",
        choices=("auto", "on", "off"),
        default="off",
        help="Skip cached-view deletion queries when the synchronous single-inflight arena path owns the view lifetime.",
    )
    parser.add_argument(
        "--pjrt_bypass_event_destroy",
        choices=("auto", "on", "off"),
        default="off",
        help="Diagnostic switch for MUSA_PJRT_BYPASS_EVENT_DESTROY. off is safe; on may leak event objects.",
    )
    parser.add_argument(
        "--pjrt_bypass_buffer_destroy",
        choices=("auto", "on", "off"),
        default="off",
        help="Diagnostic switch for MUSA_PJRT_BYPASS_BUFFER_DESTROY. off is safe; on leaks device buffers and is only for short profiling runs.",
    )
    parser.add_argument(
        "--musa_custom_fusion",
        choices=("auto", "on", "off"),
        default="off",
        help="Control MUSA_CUSTOM_FUSION for MUSA XLA backend custom-call rewrites.",
    )
    parser.add_argument(
        "--mudnn_interleaved_batch_gemm",
        choices=("auto", "on", "off"),
        default="off",
        help="Force observed interleaved-B strided batched GEMM through muDNN BatchMatMul for diagnosis.",
    )
    parser.add_argument(
        "--gemm_backend",
        choices=("auto", "auto_mublas", "mudnn", "mublas", "smallk_mublas"),
        default="auto",
        help="Select the MUSA regular GEMM backend when supported by the backend.",
    )
    parser.add_argument(
        "--musa_f32_fast_tf32",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Enable opt-in fast TF32 for MUSA float32 GEMM when XLA numeric "
            "options allow TF32. Applies to muDNN and muBLAS regular/batched GEMM."
        ),
    )
    parser.add_argument(
        "--musa_f32_fast_tf32_shapes",
        default="",
        help=(
            "Comma-separated BLAS MxNxK whitelist for MUSA F32 fast-TF32 GEMMs, "
            "for example 256x1024x8,160x1024x4. Providing a non-empty list "
            "enables MUSA_F32_FAST_TF32 for only those shapes unless "
            "--musa_f32_fast_tf32 is explicitly set."
        ),
    )
    parser.add_argument(
        "--gemm_smallk_min_major",
        type=int,
        default=512,
        help="For smallk_mublas, require max(m,n) at least this value.",
    )
    parser.add_argument(
        "--gemm_smallk_max_minor",
        type=int,
        default=512,
        help="For smallk_mublas, require min(m,n) at most this value.",
    )
    parser.add_argument(
        "--gemm_smallk_max_k",
        type=int,
        default=32,
        help="For smallk_mublas, require GEMM k at most this value.",
    )
    parser.add_argument(
        "--gemm_auto_smallk_min_major",
        type=int,
        default=1024,
        help="For default auto GEMM backend, use muBLAS only when max(m,n) is at least this value.",
    )
    parser.add_argument(
        "--gemm_auto_smallk_max_minor",
        type=int,
        default=128,
        help="For default auto GEMM backend, use muBLAS only when min(m,n) is at most this value.",
    )
    parser.add_argument(
        "--gemm_auto_smallk_max_k",
        type=int,
        default=16,
        help="For default auto GEMM backend, use muBLAS only when GEMM k is at most this value.",
    )
    parser.add_argument(
        "--gemm_auto_skinny_min_major",
        type=int,
        default=1024,
        help="For default auto GEMM backend, use muBLAS for skinny GEMMs only when max(m,n) is at least this value.",
    )
    parser.add_argument(
        "--gemm_auto_skinny_max_minor",
        type=int,
        default=128,
        help="For default auto GEMM backend, use muBLAS for skinny GEMMs only when min(m,n) is at most this value.",
    )
    parser.add_argument(
        "--gemm_auto_skinny_min_k",
        type=int,
        default=512,
        help="For default auto GEMM backend, use muBLAS for skinny GEMMs only when k is at least this value.",
    )
    parser.add_argument(
        "--gemm_auto_skinny_max_k",
        type=int,
        default=4096,
        help="For default auto GEMM backend, use muBLAS for skinny GEMMs only when k is at most this value.",
    )
    parser.add_argument(
        "--strided_batched_gemm_backend",
        choices=("auto", "auto_mublas", "mudnn", "mublas"),
        default="auto",
        help="Select the MUSA strided batched GEMM backend when supported by the backend.",
    )
    parser.add_argument(
        "--avoid_interleaved_batch_gemm_layout",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT. auto keeps the large-batch heuristic.",
    )
    parser.add_argument(
        "--xla_max_fusion_operands",
        type=int,
        default=0,
        help="Set MUSA_XLA_MAX_FUSION_OPERANDS for diagnosis. 0 keeps the backend default.",
    )
    parser.add_argument(
        "--musa_xla_dot_merger_max_mib",
        type=int,
        default=0,
        help="Set MUSA_XLA_DOT_MERGER_MAX_MIB. 0 keeps the backend default 16MiB.",
    )
    parser.add_argument(
        "--musa_xla_fuse_broadcast_bias_as_matrix",
        choices=("on", "off"),
        default="off",
        help=(
            "Experimentally fuse add(gemm, broadcast(vector)) through legacy "
            "GEMM matrix bias."
        ),
    )
    parser.add_argument(
        "--musa_xla_avoid_gemm_beta_chain",
        choices=("on", "off"),
        default="off",
        help=(
            "Skip rewriting add(gemm, gemm) into a GEMM beta operand. This "
            "keeps GEMM outputs visible to later fusion passes for meta2 A/B."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_epilogue_fusion",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Fuse supported MUSA legacy GEMM custom-call + matrix add into "
            "a 3-operand GEMM beta epilogue. Default off for manual A/B."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_epilogue_fusion_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print MUSA GEMM epilogue fusion summary even when no rewrite happens.",
    )
    parser.add_argument(
        "--musa_xla_gemm_epilogue_fuse_broadcast_bias",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Allow MusaGemmEpilogueFusion to fuse GEMM + broadcast bias. "
            "This is separate from --musa_xla_fuse_broadcast_bias_as_matrix, "
            "which is consumed earlier by GemmRewriter."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_epilogue_custom_call",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Rewrite GEMM + broadcast bias into the MUSA GEMM epilogue "
            "custom-call target. This requires a backend implementation of "
            "__musa$gemm_epilogue."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_epilogue_force_broadcast_bias_beta",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Force the legacy GEMM beta rewrite for GEMM + broadcast bias. "
            "This is only for reproducing the known slow path on S5000."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_epilogue_only_shapes",
        default="",
        help=(
            "Comma-separated MxNxK whitelist for MUSA GEMM epilogue fusion, "
            "for example 1024x256x512,1024x512x256. Empty keeps all shapes."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_epilogue_disable_mublaslt",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Disable the mublasLt execution path in MusaGemmEpilogueThunk. "
            "Diagnostic only; default off for performance runs."
        ),
    )
    parser.add_argument(
        "--musa_gemm_epilogue_thunk_diagnostics",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print per-thunk MUSA GEMM epilogue runner diagnostics.",
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_pattern",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Recognize pre-GemmRewriter dot + add/bias/elementwise HLO "
            "patterns. Recognition-only unless dot epilogue fusion is enabled."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_pattern_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print pre-GemmRewriter dot epilogue pattern summary.",
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_fusion",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Experimentally wrap supported dot + epilogue chains as custom "
            "fusions before GemmRewriter."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_fusion_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print pre-GemmRewriter dot epilogue fusion rewrite summary.",
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_log_empty",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Also print dot epilogue summaries for modules with no dots, "
            "patterns, filters, or rewrites."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_fusion_kind",
        default="__triton_gemm",
        help=(
            "FusionBackendConfig.kind for experimental dot epilogue custom "
            "fusions. Default mirrors CUDA Triton GEMM fusion dumps."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_require_add",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Only rewrite dot epilogue patterns whose chain contains add.",
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_max_chain_length",
        default="",
        help=(
            "Maximum epilogue chain length to rewrite. Empty keeps backend "
            "default, which does not limit chain length."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_max_fusions_per_module",
        default="",
        help=(
            "Maximum dot epilogue custom fusions rewritten per HLO module. "
            "Empty keeps backend default, which does not limit rewrites."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_min_m",
        default="",
        help=(
            "Minimum dot output rows M to rewrite. Empty keeps backend "
            "default, which does not filter by M."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_min_k",
        default="",
        help=(
            "Minimum dot contracting dimension K to rewrite. Empty keeps "
            "backend default, which does not filter by K."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_max_fusions_per_pattern",
        default="",
        help=(
            "Maximum dot epilogue rewrites per aggregated shape/opcode "
            "pattern. Empty keeps backend default, which does not limit per "
            "pattern."
        ),
    )
    parser.add_argument(
        "--musa_xla_dot_epilogue_sort_by_size",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Sort dot epilogue rewrite candidates by approximate GEMM work "
            "before applying rewrite limits."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_beta_chain_merger",
        choices=("on", "off"),
        default="off",
        help=(
            "Merge legacy GEMM beta accumulation chains into one block-K GEMM. "
            "Default off for manual meta2 experiments."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_beta_chain_min_chain_length",
        type=int,
        default=2,
        help="Minimum GEMM beta-chain length required before block-K merging.",
    )
    parser.add_argument(
        "--musa_xla_gemm_beta_chain_max_chains",
        type=int,
        default=128,
        help="Maximum GEMM beta chains rewritten per module.",
    )
    parser.add_argument(
        "--musa_xla_gemm_beta_chain_max_total_k",
        type=int,
        default=16,
        help=(
            "Cap each merged GEMM beta-chain chunk by concatenated K. "
            "Default 16 preserves the smallk_mublas fast path."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_beta_chain_custom_call",
        choices=("on", "off"),
        default="off",
        help=(
            "Rewrite supported GEMM beta chains to a MUSA accumulator thunk "
            "instead of materializing concat-based block-K GEMMs."
        ),
    )
    parser.add_argument(
        "--musa_xla_gemm_beta_chain_merger_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print MUSA GEMM beta-chain merger summary even when no rewrite happens.",
    )
    parser.add_argument(
        "--musa_xla_post_transpose_dot_merger",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control the MUSA-only second DotMerger after transpose folding.",
    )
    parser.add_argument(
        "--musa_xla_post_transpose_dot_merger_max_mib",
        type=int,
        default=0,
        help="Set MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB. 0 keeps the backend default 64MiB.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batcher",
        choices=("auto", "on", "off"),
        default="off",
        help="Control MUSA same-shape small Dot batching before GemmRewriter. off by default because HLO concat/slice batching can regress meta2.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_diag_only",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Build dependency-safe same-shape Dot chunks and report their "
            "costs without rewriting HLO. Available only in the diagnostic runner."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_min_group_size",
        type=int,
        default=8,
        help="Minimum same-shape Dot count required for MUSA Dot batching.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_group_size",
        type=int,
        default=32,
        help="Maximum Dots folded into one batched Dot chunk.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_groups",
        type=int,
        default=128,
        help="Maximum batched Dot chunks rewritten per module.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_min_candidate_dots",
        type=int,
        default=512,
        help="In auto mode, enable same-shape Dot batching only when at least this many candidate Dots exist in one computation.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch",
        type=int,
        default=2097152,
        help="Skip same-shape Dot batches whose output slice bytes per saved GEMM launch exceed this cap.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_output_cols",
        type=int,
        default=256,
        help=(
            "Skip same-shape Dot batches whose output column count exceeds this cap. "
            "0 disables this filter. Default 256 matches the best S5000 meta_graph_1 sweep."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batcher_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print MUSA same-shape Dot batching summary even when no rewrite happens.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_post_dot_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print post-Dot user pattern diagnostics for rewritten same-shape "
            "Dot batches. Used to choose the next batched epilogue rewrite."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_add_tree_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print add-tree diagnostics for selected same-shape Dot batches. "
            "Used to estimate whether dot+add chains can be combined before slicing."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_add_tree_rewrite",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Fuse safe same-shape Dot add-trees by reducing contiguous batched "
            "lanes before slicing the final add-tree root. off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_add_tree_external_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print details for external leaves in same-shape Dot add-tree "
            "diagnostics, including supported Dot key/reason classification."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_add_tree_mixed_key_rewrite",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Fuse safe add-trees spanning multiple same-shape Dot batch results "
            "by reducing contiguous lane runs before rebuilding the root. off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_add_tree_mixed_key_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print root-level diagnostics for add-trees that span multiple "
            "same-shape Dot batch keys. Used to select the mixed-key rewrite."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_add_tree_max_depth",
        type=int,
        default=64,
        help=(
            "Maximum add-tree recursion depth used by same-shape Dot "
            "diagnostics. The backend clamps the value to 1..4096."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_pointer_array_output",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Use a MUSA pointer-array GEMM custom call with independent outputs "
            "to avoid same-shape concat/slice traffic. off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_small_k_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print selected same-shape small-K Dot batch candidates for the "
            "next dedicated S5000 kernel path. off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_small_k_custom_max_group_size",
        type=int,
        default=0,
        help=(
            "Optional per-chunk max group size for the same-shape small-K "
            "custom kernel path. 0 keeps the global max group size."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_small_k_custom_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Rewrite selected same-shape small-K Dot chunks to a MUSA "
            "custom-call backed by a dedicated fixed-shape kernel. off by "
            "default."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_small_k_loop_fusion",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Rewrite selected same-shape small-K Dot chunks into explicit "
            "broadcast-multiply-reduce HLOs so the GPU fusion pipeline can "
            "lower them as reduction fusions instead of small GEMMs."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_small_k_pointer_array_output",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Use pointer-array output only for selected same-shape small-K Dot "
            "chunks, leaving other chunks on the regular batched-dot path. "
            "This isolates whether removing small-K slice traffic is enough."
        ),
    )
    parser.add_argument(
        "--musa_xla_horizontal_fusion_diagnostics",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print decision-point counters from GpuHorizontalLoopFusion. "
            "This is diagnostic-only and off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_cross_consumer_horizontal_fusion",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Fuse nearby, dependency-independent single-output kLoop fusions "
            "across consumers. Experimental and off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_cross_consumer_horizontal_fusion_max_group_size",
        type=int,
        default=4,
        help="Maximum number of independent kLoop fusions in one group.",
    )
    parser.add_argument(
        "--musa_xla_cross_consumer_horizontal_fusion_max_postorder_distance",
        type=int,
        default=64,
        help=(
            "Maximum HLO postorder distance from a cross-consumer group seed."
        ),
    )
    parser.add_argument(
        "--musa_xla_hot_fusion_softmax_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print post-horizontal-fusion softmax/row-reduce fusion candidates "
            "for the next S5000 hot-fusion optimization path. off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_hot_fusion_softmax_detail_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print tuple-output softmax/row-reduce fusion details for selecting "
            "the first S5000 hot-fusion custom kernel. off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_hot_fusion_softmax_body_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print per-instruction HLO bodies for selected tuple-output "
            "softmax/row-reduce fusions. off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_hot_fusion_softmax_body_names",
        default="",
        help=(
            "Comma-separated fusion or callee names to dump when "
            "--musa_xla_hot_fusion_softmax_body_diag is enabled. Empty dumps "
            "all tuple-output softmax candidates."
        ),
    )
    parser.add_argument(
        "--musa_xla_hot_tuple_softmax_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Enable the opt-in S5000 tuple-output row-softmax kernel. "
            "Unsupported fusions fall back to normal XLA emission."
        ),
    )
    parser.add_argument(
        "--musa_xla_hot_tuple_softmax_match_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Print precise tuple-output row-softmax match groups that are ready "
            "for a dedicated S5000 custom kernel. off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_reduction_chain_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Diagnose two-stage FP32 row-reduction chains followed by a "
            "same-size loop fusion. This is read-only and off by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_reduction_chain_rewrite",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Enable the opt-in MUSA HLO rewrite that merges an exclusive, "
            "control-free two-stage FP32 row-reduction chain into its final "
            "fusion. Unsupported chains are left unchanged."
        ),
    )
    parser.add_argument(
        "--musa_xla_reduction_chain_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Enable the opt-in dedicated MUSA two-stage FP32 multiply "
            "reduction-chain kernel. Enabling it also enables the guarded "
            "HLO rewrite; unsupported chains are left unchanged."
        ),
    )
    parser.add_argument(
        "--musa_xla_warp_row_reduction_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Enable the opt-in MUSA fused FP32 last-dimension warp-row "
            "reduction kernel. Unsupported fusions fall back to normal XLA "
            "emission."
        ),
    )
    parser.add_argument(
        "--musa_xla_warp_row_reduction_reducers",
        choices=("all", "add", "multiply"),
        default="all",
        help=(
            "Restrict the opt-in warp-row reduction kernel to the selected "
            "reducer kind. all preserves the structural matcher behavior."
        ),
    )
    parser.add_argument(
        "--musa_xla_tuple_warp_row_reduction_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Enable the opt-in MUSA kernel for a two-output tuple of same-"
            "shape FP32 last-dimension Add/Multiply reductions. Unsupported "
            "fusions fall back to normal XLA emission."
        ),
    )
    parser.add_argument(
        "--musa_xla_mixed_tuple_warp_row_reduction_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Enable the opt-in MUSA kernel for mixed tuple outputs containing "
            "same-shape FP32 row reductions and full elementwise arrays. "
            "Unsupported fusions fall back to normal XLA emission."
        ),
    )
    parser.add_argument(
        "--musa_xla_direct_mt_pow",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Directly map NVPTX FP32/FP64 pow calls to the corresponding "
            "MUSA device-library entry points during MTGPU lowering."
        ),
    )
    parser.add_argument(
        "--musa_xla_warp_row_reduction_min_data_elements",
        type=int,
        default=0,
        help=(
            "Minimum rows multiplied by reduction width required by the "
            "opt-in warp-row reduction kernel."
        ),
    )
    parser.add_argument(
        "--musa_xla_mixed_tuple_warp_row_reduction_min_data_elements",
        type=int,
        default=None,
        help=(
            "Minimum rows multiplied by reduction width required by the "
            "mixed-tuple warp-row kernel. Unset inherits the common "
            "warp-row reduction threshold."
        ),
    )
    parser.add_argument(
        "--musa_xla_mixed_tuple_warp_row_reduction_small_width_max",
        type=int,
        default=None,
        help=(
            "Maximum reduction width eligible for the mixed-tuple-specific "
            "thread override. Unset disables the override."
        ),
    )
    parser.add_argument(
        "--musa_xla_mixed_tuple_warp_row_reduction_small_width_threads_per_block",
        type=int,
        default=None,
        help=(
            "Threads per block used only by mixed-tuple reductions at or "
            "below the configured small-width maximum."
        ),
    )
    parser.add_argument(
        "--musa_xla_warp_row_reduction_threads_per_block",
        type=int,
        default=0,
        help=(
            "Override threads per block for the opt-in warp-row reduction "
            "kernel. Zero preserves one element per thread; positive values "
            "must be warp-aligned and enable strided per-thread processing."
        ),
    )
    parser.add_argument(
        "--musa_xla_fusion_merger_materialize_reduction_producer",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Keep a large loop-fusion producer materialized when it is shared "
            "by multiple input-fusible reductions. This experimental MUSA "
            "FusionMerger guard is disabled by default."
        ),
    )
    parser.add_argument(
        "--musa_xla_fusion_merger_materialize_min_elements",
        type=int,
        default=10000000,
        help="Minimum producer element count required by the FusionMerger guard.",
    )
    parser.add_argument(
        "--musa_xla_fusion_merger_materialize_min_operands",
        type=int,
        default=16,
        help="Minimum producer operand count required by the FusionMerger guard.",
    )
    parser.add_argument(
        "--musa_xla_fusion_merger_materialize_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Log each producer kept materialized by the FusionMerger guard.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_small_k_max_k",
        type=int,
        default=8,
        help="Maximum K dimension counted by same-shape small-K diagnostics.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_small_k_min_group_size",
        type=int,
        default=16,
        help="Minimum chunk group size counted by same-shape small-K diagnostics.",
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_small_k_output_cols",
        default="160,192,256",
        help=(
            "Comma-separated output column counts counted by same-shape small-K "
            "diagnostics. Empty means all output widths."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_biasadd",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Fuse uniform same-shape Dot + broadcast constant bias Add chunks "
            "inside the batched Dot rewrite. off by default while validating meta_graph_1."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_sweep",
        default=None,
        help=(
            "Run isolated subprocess cases for same-shape Dot batching. "
            "Comma values accept off,on,0,<max_output_cols>, e.g. off,160,192,256,0. "
            "Each case writes a separate log and report under --out_root."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_group_size_sweep",
        default=None,
        help=(
            "Optional comma list for cross-sweeping "
            "--musa_xla_same_shape_dot_batch_max_group_size with "
            "--musa_xla_same_shape_dot_batch_sweep, e.g. 8,16,32,64."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch_sweep",
        default=None,
        help=(
            "Optional comma list for cross-sweeping "
            "--musa_xla_same_shape_dot_batch_max_slice_bytes_per_saved_launch "
            "with --musa_xla_same_shape_dot_batch_sweep, e.g. 262144,524288,1048576."
        ),
    )
    parser.add_argument(
        "--musa_xla_same_lhs_dot_merger",
        choices=("auto", "on", "off"),
        default="off",
        help="Merge same-LHS small HLO Dot groups before GemmRewriter. off by default; use on for manual meta2 experiments.",
    )
    parser.add_argument(
        "--musa_xla_same_lhs_dot_merger_min_group_size",
        type=int,
        default=2,
        help="Minimum same-LHS Dot count required for manual HLO Dot merging.",
    )
    parser.add_argument(
        "--musa_xla_same_lhs_dot_merger_max_groups",
        type=int,
        default=8,
        help="Maximum same-LHS Dot groups rewritten per module.",
    )
    parser.add_argument(
        "--musa_xla_same_lhs_dot_merger_max_group_size",
        type=int,
        default=16,
        help="Maximum Dots folded into one same-LHS merged Dot chunk.",
    )
    parser.add_argument(
        "--musa_xla_same_lhs_dot_merger_max_total_cols",
        type=int,
        default=2048,
        help="Maximum concatenated RHS columns for one same-LHS merged Dot chunk.",
    )
    parser.add_argument(
        "--musa_xla_same_lhs_dot_merger_min_candidate_dots",
        type=int,
        default=128,
        help="In auto mode, enable same-LHS Dot merging only when at least this many candidate Dots exist.",
    )
    parser.add_argument(
        "--musa_xla_same_lhs_dot_merger_normalize_operands",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Group same-LHS HLO Dots after stripping trivial bitcast/copy/reshape operand wrappers.",
    )
    parser.add_argument(
        "--musa_xla_same_lhs_dot_merger_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print same-LHS Dot merger summary even when no rewrite happens.",
    )
    parser.add_argument(
        "--musa_xla_same_rhs_dot_merger",
        choices=("auto", "on", "off"),
        default="off",
        help="Merge same-RHS small HLO Dot groups before GemmRewriter. off by default; use on to test whether meta2 has shared-weight GEMM groups.",
    )
    parser.add_argument(
        "--musa_xla_same_rhs_dot_merger_min_group_size",
        type=int,
        default=2,
        help="Minimum same-RHS Dot count required for manual HLO Dot merging.",
    )
    parser.add_argument(
        "--musa_xla_same_rhs_dot_merger_max_groups",
        type=int,
        default=8,
        help="Maximum same-RHS Dot groups rewritten per module.",
    )
    parser.add_argument(
        "--musa_xla_same_rhs_dot_merger_max_group_size",
        type=int,
        default=4,
        help="Maximum Dots folded into one same-RHS merged Dot chunk.",
    )
    parser.add_argument(
        "--musa_xla_same_rhs_dot_merger_max_total_rows",
        type=int,
        default=4096,
        help="Maximum concatenated LHS rows for one same-RHS merged Dot chunk.",
    )
    parser.add_argument(
        "--musa_xla_same_rhs_dot_merger_min_candidate_dots",
        type=int,
        default=128,
        help="In auto mode, enable same-RHS Dot merging only when at least this many candidate Dots exist.",
    )
    parser.add_argument(
        "--musa_xla_same_rhs_dot_merger_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print same-RHS Dot merger summary even when no rewrite happens.",
    )
    parser.add_argument(
        "--musa_xla_group_gemm_thunks",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Group consecutive MUSA GEMM thunks into one dispatch thunk. off by default; experimental diagnostic for small GEMM overhead.",
    )
    parser.add_argument(
        "--musa_xla_group_gemm_thunks_min_group_size",
        type=int,
        default=4,
        help="Minimum consecutive GEMM thunk count required before grouping.",
    )
    parser.add_argument(
        "--musa_xla_group_gemm_thunks_max_group_size",
        type=int,
        default=64,
        help="Maximum GEMM entries packed into one grouped GEMM thunk.",
    )
    parser.add_argument(
        "--musa_xla_group_gemm_thunks_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print grouped GEMM thunk summary even when no groups are created.",
    )
    parser.add_argument(
        "--musa_xla_group_gemm_thunks_cross_kernel_diag",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Diagnose GEMM groups split by kernel thunks without changing thunk order.",
    )
    parser.add_argument(
        "--musa_xla_group_gemm_thunks_cross_kernel_max_separators",
        type=int,
        default=8,
        help="Maximum intervening kernel thunks scanned by cross-kernel GEMM diagnostics.",
    )
    parser.add_argument(
        "--musa_xla_small_gemm_accum_thunks",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Rewrite consecutive same-output small GEMM beta chains into a MUSA accum thunk. off by default; experimental.",
    )
    parser.add_argument(
        "--musa_xla_small_gemm_accum_log",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Print small GEMM accum thunk rewrite summary even when no chains are created.",
    )
    parser.add_argument(
        "--musa_xla_small_gemm_accum_min_chain_size",
        type=int,
        default=4,
        help="Minimum consecutive same-output GEMM count required for small GEMM accum thunk rewriting.",
    )
    parser.add_argument(
        "--musa_xla_small_gemm_accum_max_chain_size",
        type=int,
        default=64,
        help="Maximum GEMM entries packed into one small GEMM accum thunk.",
    )
    parser.add_argument(
        "--musa_xla_small_gemm_accum_max_k",
        type=int,
        default=64,
        help="Maximum contracting dimension K accepted by small GEMM accum thunk rewriting.",
    )
    parser.add_argument(
        "--musa_xla_small_gemm_accum_require_custom_kernel",
        nargs="?",
        const="on",
        choices=("auto", "on", "off"),
        default="off",
        help="Skip small GEMM accum rewriting unless a real custom kernel is available.",
    )
    parser.add_argument(
        "--musa_xla_gpu_runtime",
        choices=("auto", "classic_thunks", "xla_runtime"),
        default="auto",
        help="Select MUSA XLA GPU executable runtime. auto uses classic_thunks for thunk-only GEMM experiments.",
    )
    parser.add_argument(
        "--xla_global_jit_level",
        choices=("off", "auto", "on", "on_1", "on_2"),
        default="off",
        help="Set ConfigProto global_jit_level on MUSA. off keeps TF auto clustering only.",
    )
    parser.add_argument(
        "--xla_jit_scope",
        choices=("off", "on"),
        default="off",
        help="Import the graph under tf.xla.experimental.jit_scope(compile_ops=True).",
    )
    parser.add_argument(
        "--musa_blas_gemm_diagnostics",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help="Print MUSA GEMM backend decisions for strided batched GEMM.",
    )
    parser.add_argument(
        "--musa_hlo_pattern_analysis",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help="Print lightweight MUSA HLO pattern counts without full HLO dump.",
    )
    parser.add_argument(
        "--musa_hlo_pattern_analysis_verbose",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help="Print verbose MUSA HLO pattern details. off keeps logs concise.",
    )
    parser.add_argument(
        "--musa_hlo_pattern_analysis_log_empty",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help="Print MUSA HLO pattern logs for modules without GEMM/custom calls.",
    )
    parser.add_argument(
        "--musa_xla_thunk_diagnostics",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help="Print per-module MUSA XLA runtime thunk counts. Diagnostic only.",
    )
    parser.add_argument(
        "--musa_xla_thunk_timing",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help=(
            "Synchronize around each sequential thunk and print coarse runtime "
            "timing. Heavy diagnostic only; do not use for benchmark numbers."
        ),
    )
    parser.add_argument(
        "--musa_xla_classic_thunk_graph",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help=(
            "Capture eligible MUSA classic Kernel/GEMM thunk sequences into "
            "one device graph. Experimental and opt-in."
        ),
    )
    parser.add_argument(
        "--musa_xla_classic_thunk_graph_max_cache_entries",
        type=int,
        default=4,
        help=(
            "Maximum pointer-signature graph cache entries per executable "
            "before falling back to ordinary thunk execution."
        ),
    )
    parser.add_argument(
        "--musa_xla_execution_path_verbose",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help="Print execution path for non-cluster MUSA XLA modules too.",
    )
    parser.add_argument(
        "--musa_xla_gemm_runtime_diagnostics",
        choices=("auto", "on", "off"),
        nargs="?",
        const="on",
        default="off",
        help="Print aggregated RunGemm shape counts. Diagnostic only.",
    )
    parser.add_argument(
        "--musa_xla_gemm_runtime_log_interval",
        default="",
        help="Log RunGemm aggregate every N calls. Empty keeps backend default.",
    )
    parser.add_argument(
        "--convert_script",
        default=str(SCRIPT_DIR / "convert_spec_to_pb.py"),
        help="Path to convert spec->pb script.",
    )
    parser.add_argument("--convert_out_root", default="frozen_out")
    parser.add_argument("--xla", action="store_true", help="Enable TensorFlow XLA auto_jit.")
    parser.add_argument(
        "--xla_device_scope",
        action="store_true",
        help="Keep the explicit device scope while importing the graph in MUSA XLA mode.",
    )
    parser.add_argument("--xla_dump", action="store_true", help="Enable HLO dump.")
    parser.add_argument("--xla_dump_dir", default=None, help="Override XLA dump directory.")
    parser.add_argument(
        "--xla_dump_hlo_pass_re",
        default="^$",
        help="Value for XLA --xla_dump_hlo_pass_re when --xla_dump is enabled.",
    )
    parser.add_argument(
        "--xla_dump_max_hlo_modules",
        default="-1",
        help="Value for XLA --xla_dump_max_hlo_modules when --xla_dump is enabled.",
    )
    parser.add_argument(
        "--xla_dump_hlo_module_re",
        default="",
        help="Optional value for XLA --xla_dump_hlo_module_re when --xla_dump is enabled.",
    )
    parser.add_argument(
        "--xla_dump_long_text",
        type=parse_bool,
        default=False,
        help="Enable --xla_dump_hlo_as_long_text. Keep it off for large graphs like meta_graph_2.",
    )
    parser.add_argument(
        "--musa_plugin",
        default=default_musa_plugin_path(),
        help="Path to the MUSA PJRT core plugin for MUSA XLA runs.",
    )
    parser.add_argument(
        "--musa_tf_adapter",
        default=default_musa_tf_adapter_path(),
        help="Path to the TensorFlow NextPluggableDevice adapter.",
    )
    args = parser.parse_args()
    args.explicit_cli_flags = sorted(explicit_cli_flags(sys.argv[1:]))
    args.optimization_profile_selected = "off"
    args.optimization_profile_applied = {}
    return args


def main():
    args = parse_args()
    if args.compare_thunk_log_baseline or args.compare_thunk_log_experiment:
        if not args.compare_thunk_log_baseline or not args.compare_thunk_log_experiment:
            raise ValueError(
                "Provide both --compare_thunk_log_baseline and "
                "--compare_thunk_log_experiment"
            )
        diff = compare_thunk_diagnostic_logs(
            args.compare_thunk_log_baseline,
            args.compare_thunk_log_experiment,
        )
        print(json.dumps(diff, ensure_ascii=False, indent=2))
        return
    if args.compare_xla_dump_baseline_dir or args.compare_xla_dump_freeze_dir:
        if not args.compare_xla_dump_baseline_dir or not args.compare_xla_dump_freeze_dir:
            raise ValueError(
                "Provide both --compare_xla_dump_baseline_dir and "
                "--compare_xla_dump_freeze_dir"
            )
        diff = compare_xla_dump_dirs(
            args.compare_xla_dump_baseline_dir,
            args.compare_xla_dump_freeze_dir,
        )
        print(json.dumps(diff, ensure_ascii=False, indent=2))
        return
    if args.summarize_xla_dump_dir:
        summary = summarize_xla_dump_dir(args.summarize_xla_dump_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.analyze_xla_hot_fusion_dump_dir:
        summary = analyze_xla_hot_fusion_dump(
            args.analyze_xla_hot_fusion_dump_dir,
            hot_fusions=args.analyze_xla_hot_fusions,
            limit=args.analyze_xla_hot_fusion_limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print_xla_hot_fusion_summary(summary)
        return
    if bool(args.spec) == bool(args.spec_dir):
        raise ValueError("Provide exactly one of --spec or --spec_dir")
    if args.pb and not args.spec:
        raise ValueError("--pb can only be used together with --spec")
    if env_flag_enabled("MUSA_PJRT_DEBUG_LOG"):
        print(
            "[WARN] MUSA_PJRT_DEBUG_LOG=1 adds heavy per-transfer logging; "
            "use it only for diagnosis, not performance numbers."
        )
    if args.xla and device_kind(args.device) == "MUSA":
        print(
            f"[INFO] MUSA_CUSTOM_FUSION={os.environ.get('MUSA_CUSTOM_FUSION', '')}"
        )
        print(
            "[INFO] MUSA_MUDNN_INTERLEAVED_BATCH_GEMM="
            f"{os.environ.get('MUSA_MUDNN_INTERLEAVED_BATCH_GEMM', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_BACKEND="
            f"{os.environ.get('MUSA_GEMM_BACKEND', '')}"
        )
        print(
            "[INFO] MUSA_F32_FAST_TF32="
            f"{os.environ.get('MUSA_F32_FAST_TF32', '')}"
        )
        print(
            "[INFO] MUSA_F32_FAST_TF32_SHAPES="
            f"{os.environ.get('MUSA_F32_FAST_TF32_SHAPES', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_SMALLK_MIN_MAJOR="
            f"{os.environ.get('MUSA_GEMM_SMALLK_MIN_MAJOR', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_SMALLK_MAX_MINOR="
            f"{os.environ.get('MUSA_GEMM_SMALLK_MAX_MINOR', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_SMALLK_MAX_K="
            f"{os.environ.get('MUSA_GEMM_SMALLK_MAX_K', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_AUTO_SMALLK_MIN_MAJOR="
            f"{os.environ.get('MUSA_GEMM_AUTO_SMALLK_MIN_MAJOR', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_AUTO_SMALLK_MAX_MINOR="
            f"{os.environ.get('MUSA_GEMM_AUTO_SMALLK_MAX_MINOR', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_AUTO_SMALLK_MAX_K="
            f"{os.environ.get('MUSA_GEMM_AUTO_SMALLK_MAX_K', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_AUTO_SKINNY_MIN_MAJOR="
            f"{os.environ.get('MUSA_GEMM_AUTO_SKINNY_MIN_MAJOR', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_AUTO_SKINNY_MAX_MINOR="
            f"{os.environ.get('MUSA_GEMM_AUTO_SKINNY_MAX_MINOR', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_AUTO_SKINNY_MIN_K="
            f"{os.environ.get('MUSA_GEMM_AUTO_SKINNY_MIN_K', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_AUTO_SKINNY_MAX_K="
            f"{os.environ.get('MUSA_GEMM_AUTO_SKINNY_MAX_K', '')}"
        )
        print(
            "[INFO] MUSA_STRIDED_BATCHED_GEMM_BACKEND="
            f"{os.environ.get('MUSA_STRIDED_BATCHED_GEMM_BACKEND', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_FORCE_HOST_BUFFER_COPY="
            f"{os.environ.get('MUSA_PJRT_FORCE_HOST_BUFFER_COPY', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_MAX_INFLIGHT_TRANSFERS="
            f"{os.environ.get('MUSA_PJRT_MAX_INFLIGHT_TRANSFERS', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_WAIT_TRANSFER_DONE="
            f"{os.environ.get('MUSA_PJRT_WAIT_TRANSFER_DONE', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_WAIT_EXECUTE_DONE="
            f"{os.environ.get('MUSA_PJRT_WAIT_EXECUTE_DONE', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_DIAGNOSTICS="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_DIAGNOSTICS', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_ASYNC="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_ASYNC', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PARALLEL_PACK="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PARALLEL_PACK', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_FAST_PACK="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_FAST_PACK', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_THREADS="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_THREADS', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_MIN_BYTES="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_MIN_BYTES', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_DIRTY_RANGES="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_DIRTY_RANGES', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_POOL_ORDER_LAYOUT="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_POOL_ORDER_LAYOUT', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_REUSE_HOST_BUFFERS_TRUST_CONTENTS="
            f"{os.environ.get('MUSA_PJRT_REUSE_HOST_BUFFERS_TRUST_CONTENTS', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS="
            f"{os.environ.get('MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS', '')}"
        )
        print(
            "[INFO] MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS_TRUST_LIFETIME="
            f"{os.environ.get('MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS_TRUST_LIFETIME', '')}"
        )
        print(
            "[INFO] MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT="
            f"{os.environ.get('MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT', '')}"
        )
        print(
            "[INFO] MUSA_XLA_MAX_FUSION_OPERANDS="
            f"{os.environ.get('MUSA_XLA_MAX_FUSION_OPERANDS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_MERGER_MAX_MIB="
            f"{os.environ.get('MUSA_XLA_DOT_MERGER_MAX_MIB', '')}"
        )
        print(
            "[INFO] MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX="
            f"{os.environ.get('MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX', '')}"
        )
        print(
            "[INFO] MUSA_XLA_AVOID_GEMM_BETA_CHAIN="
            f"{os.environ.get('MUSA_XLA_AVOID_GEMM_BETA_CHAIN', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_EPILOGUE_FUSION="
            f"{os.environ.get('MUSA_XLA_GEMM_EPILOGUE_FUSION', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_EPILOGUE_FUSION_LOG="
            f"{os.environ.get('MUSA_XLA_GEMM_EPILOGUE_FUSION_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS="
            f"{os.environ.get('MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL="
            f"{os.environ.get('MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA="
            f"{os.environ.get('MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES="
            f"{os.environ.get('MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT="
            f"{os.environ.get('MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT', '')}"
        )
        print(
            "[INFO] MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS="
            f"{os.environ.get('MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_PATTERN="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_PATTERN', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_PATTERN_LOG="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_PATTERN_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_FUSION="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_FUSION', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_FUSION_LOG="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_FUSION_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_FUSION_KIND="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_FUSION_KIND', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_MIN_M="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_MIN_M', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_MIN_K="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_MIN_K', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE="
            f"{os.environ.get('MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_BETA_CHAIN_MERGER="
            f"{os.environ.get('MUSA_XLA_GEMM_BETA_CHAIN_MERGER', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH="
            f"{os.environ.get('MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS="
            f"{os.environ.get('MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K="
            f"{os.environ.get('MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL="
            f"{os.environ.get('MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_BETA_CHAIN_MERGER_LOG="
            f"{os.environ.get('MUSA_XLA_GEMM_BETA_CHAIN_MERGER_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_POST_TRANSPOSE_DOT_MERGER="
            f"{os.environ.get('MUSA_XLA_POST_TRANSPOSE_DOT_MERGER', '')}"
        )
        print(
            "[INFO] MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB="
            f"{os.environ.get('MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCHER="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCHER', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_POST_DOT_DIAG="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_POST_DOT_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_DIAG="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_REWRITE="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_REWRITE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_EXTERNAL_DIAG="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_EXTERNAL_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_DIAG="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MAX_DEPTH="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MAX_DEPTH', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_REWRITE="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_REWRITE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_BIASADD="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_BIASADD', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_POINTER_ARRAY_OUTPUT="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_POINTER_ARRAY_OUTPUT', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_DIAG="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS="
            f"{os.environ.get('MUSA_XLA_HORIZONTAL_FUSION_DIAGNOSTICS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION="
            f"{os.environ.get('MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION', '')}"
        )
        print(
            "[INFO] MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_POSTORDER_DISTANCE="
            f"{os.environ.get('MUSA_XLA_CROSS_CONSUMER_HORIZONTAL_FUSION_MAX_POSTORDER_DISTANCE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG="
            f"{os.environ.get('MUSA_XLA_HOT_FUSION_SOFTMAX_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG="
            f"{os.environ.get('MUSA_XLA_HOT_FUSION_SOFTMAX_DETAIL_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG="
            f"{os.environ.get('MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_NAMES="
            f"{os.environ.get('MUSA_XLA_HOT_FUSION_SOFTMAX_BODY_NAMES', '')}"
        )
        print(
            "[INFO] MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG="
            f"{os.environ.get('MUSA_XLA_HOT_TUPLE_SOFTMAX_MATCH_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL="
            f"{os.environ.get('MUSA_XLA_HOT_TUPLE_SOFTMAX_KERNEL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_REDUCTION_CHAIN_DIAG="
            f"{os.environ.get('MUSA_XLA_REDUCTION_CHAIN_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_REDUCTION_CHAIN_REWRITE="
            f"{os.environ.get('MUSA_XLA_REDUCTION_CHAIN_REWRITE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_REDUCTION_CHAIN_KERNEL="
            f"{os.environ.get('MUSA_XLA_REDUCTION_CHAIN_KERNEL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_WARP_ROW_REDUCTION_KERNEL="
            f"{os.environ.get('MUSA_XLA_WARP_ROW_REDUCTION_KERNEL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_WARP_ROW_REDUCTION_REDUCERS="
            f"{os.environ.get('MUSA_XLA_WARP_ROW_REDUCTION_REDUCERS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS="
            f"{os.environ.get('MUSA_XLA_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_WARP_ROW_REDUCTION_THREADS_PER_BLOCK="
            f"{os.environ.get('MUSA_XLA_WARP_ROW_REDUCTION_THREADS_PER_BLOCK', '')}"
        )
        print(
            "[INFO] MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL="
            f"{os.environ.get('MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL="
            f"{os.environ.get('MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS="
            f"{os.environ.get('MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_MAX="
            f"{os.environ.get('MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_MAX', '')}"
        )
        print(
            "[INFO] MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_THREADS_PER_BLOCK="
            f"{os.environ.get('MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_THREADS_PER_BLOCK', '')}"
        )
        print(
            "[INFO] MUSA_XLA_DIRECT_MT_POW="
            f"{os.environ.get('MUSA_XLA_DIRECT_MT_POW', '')}"
        )
        print(
            "[INFO] MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER="
            f"{os.environ.get('MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER', '')}"
        )
        print(
            "[INFO] MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS="
            f"{os.environ.get('MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS="
            f"{os.environ.get('MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG="
            f"{os.environ.get('MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MAX_K="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MAX_K', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MIN_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_MIN_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_OUTPUT_COLS="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_OUTPUT_COLS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_POINTER_ARRAY_OUTPUT="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_POINTER_ARRAY_OUTPUT', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_LOOP_FUSION="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_LOOP_FUSION', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_KERNEL="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_KERNEL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_MAX_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_MAX_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_LHS_DOT_MERGER="
            f"{os.environ.get('MUSA_XLA_SAME_LHS_DOT_MERGER', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUPS="
            f"{os.environ.get('MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUPS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_TOTAL_COLS="
            f"{os.environ.get('MUSA_XLA_SAME_LHS_DOT_MERGER_MAX_TOTAL_COLS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_CANDIDATE_DOTS="
            f"{os.environ.get('MUSA_XLA_SAME_LHS_DOT_MERGER_MIN_CANDIDATE_DOTS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_LHS_DOT_MERGER_NORMALIZE_OPERANDS="
            f"{os.environ.get('MUSA_XLA_SAME_LHS_DOT_MERGER_NORMALIZE_OPERANDS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_LHS_DOT_MERGER_LOG="
            f"{os.environ.get('MUSA_XLA_SAME_LHS_DOT_MERGER_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_RHS_DOT_MERGER="
            f"{os.environ.get('MUSA_XLA_SAME_RHS_DOT_MERGER', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUPS="
            f"{os.environ.get('MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUPS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_TOTAL_ROWS="
            f"{os.environ.get('MUSA_XLA_SAME_RHS_DOT_MERGER_MAX_TOTAL_ROWS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_CANDIDATE_DOTS="
            f"{os.environ.get('MUSA_XLA_SAME_RHS_DOT_MERGER_MIN_CANDIDATE_DOTS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SAME_RHS_DOT_MERGER_LOG="
            f"{os.environ.get('MUSA_XLA_SAME_RHS_DOT_MERGER_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GROUP_GEMM_THUNKS="
            f"{os.environ.get('MUSA_XLA_GROUP_GEMM_THUNKS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE="
            f"{os.environ.get('MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GROUP_GEMM_THUNKS_LOG="
            f"{os.environ.get('MUSA_XLA_GROUP_GEMM_THUNKS_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG="
            f"{os.environ.get('MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_MAX_SEPARATORS="
            f"{os.environ.get('MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_MAX_SEPARATORS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS="
            f"{os.environ.get('MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SMALL_GEMM_ACCUM_LOG="
            f"{os.environ.get('MUSA_XLA_SMALL_GEMM_ACCUM_LOG', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE="
            f"{os.environ.get('MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE="
            f"{os.environ.get('MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K="
            f"{os.environ.get('MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K', '')}"
        )
        print(
            "[INFO] MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL="
            f"{os.environ.get('MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GPU_RUNTIME="
            f"{os.environ.get('MUSA_XLA_GPU_RUNTIME', '')}"
        )
        print(f"[INFO] XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
        print(
            "[INFO] MUSA_XLA_GLOBAL_JIT_LEVEL="
            f"{os.environ.get('MUSA_XLA_GLOBAL_JIT_LEVEL', '')}"
        )
        print(
            "[INFO] MUSA_BLAS_GEMM_DIAGNOSTICS="
            f"{os.environ.get('MUSA_BLAS_GEMM_DIAGNOSTICS', '')}"
        )
        print(
            "[INFO] MUSA_HLO_PATTERN_ANALYSIS="
            f"{os.environ.get('MUSA_HLO_PATTERN_ANALYSIS', '')}"
        )
        print(
            "[INFO] MUSA_HLO_PATTERN_ANALYSIS_VERBOSE="
            f"{os.environ.get('MUSA_HLO_PATTERN_ANALYSIS_VERBOSE', '')}"
        )
        print(
            "[INFO] MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY="
            f"{os.environ.get('MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY', '')}"
        )
        print(
            "[INFO] MUSA_XLA_THUNK_DIAGNOSTICS="
            f"{os.environ.get('MUSA_XLA_THUNK_DIAGNOSTICS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_THUNK_TIMING="
            f"{os.environ.get('MUSA_XLA_THUNK_TIMING', '')}"
        )
        print(
            "[INFO] MUSA_XLA_CLASSIC_THUNK_GRAPH="
            f"{os.environ.get('MUSA_XLA_CLASSIC_THUNK_GRAPH', '')}"
        )
        print(
            "[INFO] MUSA_XLA_CLASSIC_THUNK_GRAPH_MAX_CACHE_ENTRIES="
            f"{os.environ.get('MUSA_XLA_CLASSIC_THUNK_GRAPH_MAX_CACHE_ENTRIES', '')}"
        )
        print(
            "[INFO] MUSA_XLA_EXECUTION_PATH_VERBOSE="
            f"{os.environ.get('MUSA_XLA_EXECUTION_PATH_VERBOSE', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS="
            f"{os.environ.get('MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS', '')}"
        )
        print(
            "[INFO] MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL="
            f"{os.environ.get('MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL', '')}"
        )
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    convert_out_root = Path(args.convert_out_root).resolve()
    convert_out_root.mkdir(parents=True, exist_ok=True)

    bs_values = parse_bs_values(args.bs)
    convert_script = Path(args.convert_script).resolve()
    if not convert_script.exists():
        raise FileNotFoundError(f"convert script not found: {convert_script}")

    specs = collect_specs(args.spec, args.spec_dir)
    selected_profile, applied_profile = apply_optimization_profile(
        args,
        specs,
        explicit_flags=args.explicit_cli_flags,
    )
    if selected_profile != "off":
        print(
            f"[INFO] optimization_profile={selected_profile} "
            f"applied={sorted(applied_profile)}"
        )
    if (
        selected_profile == "meta1"
        or _early_cpu_affinity_mode != "off"
        or EARLY_CPU_AFFINITY.get("enabled")
    ):
        print(
            "[INFO] cpu_affinity: "
            f"enabled={EARLY_CPU_AFFINITY.get('enabled', False)} "
            f"reason={EARLY_CPU_AFFINITY.get('reason', '')} "
            f"selection={EARLY_CPU_AFFINITY.get('selection', '')} "
            f"pci={EARLY_CPU_AFFINITY.get('pci_bus_id', '')} "
            f"cpus={EARLY_CPU_AFFINITY.get('cpus', [])}"
        )
    musa_loaded = False if args.analyze_only else load_runtime_plugins(args)
    run_root = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")

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
                err = traceback.format_exc()
                failures += 1
                all_reports.append(
                    {
                        "spec_path": str(spec_path),
                        "pb_path": None,
                        "status": "failed",
                        "error_stage": "detect_or_convert_pb",
                        "error_core": extract_core_error(err) or str(detect_error),
                        "error": err,
                    }
                )
                continue

        for bs in bs_values:
            try:
                if args.analyze_only:
                    report = analyze_single_spec(
                        spec_path.resolve(), pb_path.resolve(), args, bs
                    )
                else:
                    report = run_single_spec(
                        spec_path.resolve(), pb_path.resolve(), args, bs, musa_loaded, run_root
                    )
            except Exception:
                err = traceback.format_exc()
                report = {
                    "spec_path": str(spec_path),
                    "pb_path": str(pb_path),
                    "batch_size": bs,
                    "status": "failed",
                    "error_stage": "run_inference",
                    "error_core": extract_core_error(err),
                    "error": err,
                }
            if report["status"] != "ok":
                failures += 1
            all_reports.append(report)
            if args.analyze_only:
                analysis = report.get("analysis") or {}
                largest_inputs = analysis.get("largest_inputs") or []
                largest = largest_inputs[0] if largest_inputs else {}
                print(
                    f"[INFO] analyze done: spec={spec_path.name} bs={bs} "
                    f"status={report['status']} "
                    f"inputs={analysis.get('input_count', 0)} "
                    f"input_spec={analysis.get('input_spec_count', 0)} "
                    f"skipped_unreachable={analysis.get('skipped_unreachable_inputs', 0)} "
                    f"total_mib={analysis.get('total_mib', 0.0):.3f} "
                    f"large_inputs={analysis.get('large_input_count', 0)} "
                    f"large_mib={analysis.get('large_input_mib', 0.0):.3f} "
                    f"largest={largest.get('name', 'N/A')} "
                    f"largest_mib={largest.get('mib', 0.0):.3f}"
                )
                print(
                    f"[INFO] analyze buckets: {analysis.get('mib_by_size_bucket', {})}"
                )
                pow_diagnostics = (
                    (report.get("graph_rewrite") or {}).get("pow_square") or {}
                )
                if pow_diagnostics:
                    print(
                        f"[INFO] pow_diagnostics: "
                        f"enabled={pow_diagnostics.get('enabled', False)} "
                        f"total={pow_diagnostics.get('total_pow', 0)} "
                        f"rewritten={pow_diagnostics.get('rewritten', 0)} "
                        f"skipped_non_const="
                        f"{pow_diagnostics.get('skipped_non_const', 0)} "
                        f"skipped_non_square="
                        f"{pow_diagnostics.get('skipped_non_square', 0)} "
                        f"non_const_ops="
                        f"{pow_diagnostics.get('non_const_exponent_ops', [])} "
                        f"non_square_values="
                        f"{pow_diagnostics.get('non_square_constant_values', [])} "
                        f"non_const_samples="
                        f"{pow_diagnostics.get('non_const_exponent_samples', [])[:12]} "
                        f"non_square_samples="
                        f"{pow_diagnostics.get('non_square_constant_samples', [])[:12]}"
                    )
                print(
                    f"[INFO] analyze top_placeholder_consumers: "
                    f"{analysis.get('top_placeholder_consumer_ops', [])[:10]}"
                )
                concat_analysis = analysis.get("concat_pack_analysis") or {}
                if concat_analysis:
                    print(
                        f"[INFO] analyze concat_pack: "
                        f"counts={concat_analysis.get('counts', {})} "
                        f"top_candidate_mib={concat_analysis.get('top_candidate_mib', [])[:5]} "
                        f"top_candidate_inputs={concat_analysis.get('top_candidate_inputs', [])[:5]} "
                        f"top_non_feed_input_ops={concat_analysis.get('top_non_feed_input_ops', [])[:8]}"
                    )
                    for index, item in enumerate(
                        (concat_analysis.get("top_concat_candidates") or [])[:3],
                        start=1,
                    ):
                        print(
                            f"[INFO] analyze concat_candidate{index}: "
                            f"node={item.get('concat_node', 'N/A')} "
                            f"inputs={item.get('inputs', 0)} "
                            f"mib={item.get('mib', 0.0):.3f} "
                            f"axis={item.get('axis', 'N/A')} "
                            f"dtype={item.get('dtype', 'N/A')} "
                            f"shape={item.get('shape', [])}"
                        )
                matmul_analysis = analysis.get("matmul_group_analysis") or {}
                if matmul_analysis:
                    print(
                        f"[INFO] analyze matmul: "
                        f"total={matmul_analysis.get('matmul_total', 0)} "
                        f"same_lhs_groups={matmul_analysis.get('same_lhs_group_count', 0)} "
                        f"same_lhs_matmuls={matmul_analysis.get('same_lhs_matmul_count', 0)} "
                        f"same_lhs_reduction="
                        f"{matmul_analysis.get('estimated_call_reduction_if_grouped_by_lhs', 0)} "
                        f"same_rhs_groups={matmul_analysis.get('same_rhs_group_count', 0)} "
                        f"same_rhs_matmuls={matmul_analysis.get('same_rhs_matmul_count', 0)} "
                        f"same_rhs_reduction="
                        f"{matmul_analysis.get('estimated_call_reduction_if_grouped_by_rhs', 0)} "
                        f"rhs_producers={matmul_analysis.get('matmul_rhs_producer_ops', [])[:8]}"
                    )
                    print(
                        f"[INFO] analyze top_matmul_shapes: "
                        f"{matmul_analysis.get('top_shape_groups', [])[:5]}"
                    )
                    print(
                        f"[INFO] analyze top_same_lhs_matmul_groups: "
                        f"{matmul_analysis.get('top_same_lhs_groups', [])[:5]}"
                    )
                    print(
                        f"[INFO] analyze top_same_rhs_matmul_groups: "
                        f"{matmul_analysis.get('top_same_rhs_groups', [])[:5]}"
                    )
                if report.get("error_core"):
                    print(f"[INFO] core error: {report['error_core']}")
                if report.get("error_tail"):
                    print("[INFO] error tail:")
                    for line in report["error_tail"]:
                        print(f"[INFO]   {line}")
                continue
            avg = (report.get("timing_ms") or {}).get("average")
            trimmed = (report.get("timing_ms") or {}).get("trimmed_avg")
            print(
                f"[INFO] run done: spec={spec_path.name} bs={bs} "
                f"status={report['status']} average_time_ms={avg} "
                f"trimmed_avg_ms={trimmed}"
            )
            feed_summary = report.get("feed_summary") or {}
            largest_inputs = feed_summary.get("largest_inputs") or []
            if feed_summary:
                largest = largest_inputs[0] if largest_inputs else {}
                callable_info = report.get("callable") or {}
                output_summary = report.get("output_summary") or {}
                print(
                    f"[INFO] feed: total_mib={feed_summary.get('total_mib', 0.0):.3f} "
                    f"inputs={feed_summary.get('num_inputs', 0)} "
                    f"input_spec={feed_summary.get('input_spec_count', feed_summary.get('num_inputs', 0))} "
                    f"skipped_unreachable={feed_summary.get('skipped_unreachable_inputs', 0)} "
                    f"protected_outputs={feed_summary.get('pack_protected_output_feeds', 0)} "
                    f"pinned={feed_summary.get('pinned_feed_used', False)} "
                    f"callable={callable_info.get('used', False)} "
                    f"concat_packed={feed_summary.get('concat_pack_used', False)} "
                    f"concat_nodes={feed_summary.get('concat_pack_selected_nodes', 0)} "
                    f"concat_inputs={feed_summary.get('concat_pack_selected_inputs', 0)} "
                    f"concat_mib={feed_summary.get('concat_pack_selected_mib', 0.0):.3f} "
                    f"concat_frozen={feed_summary.get('concat_pack_frozen', False)} "
                    f"concat_frozen_mib={feed_summary.get('concat_pack_frozen_mib', 0.0):.3f} "
                    f"concat_cached={feed_summary.get('concat_pack_cached', False)} "
                    f"concat_cached_mib={feed_summary.get('concat_pack_cached_mib', 0.0):.3f} "
                    f"slice_compacted={feed_summary.get('slice_compact_used', False)} "
                    f"slice_compact_sources={feed_summary.get('slice_compact_source_inputs', 0)} "
                    f"slice_compact_nodes={feed_summary.get('slice_compact_selected_nodes', 0)} "
                    f"slice_compact_input_delta={feed_summary.get('slice_compact_input_delta', 0)} "
                    f"slice_compact_saved_mib={feed_summary.get('slice_compact_saved_mib', 0.0):.3f} "
                    f"slice_packed={feed_summary.get('slice_pack_used', False)} "
                    f"slice_sources={feed_summary.get('slice_pack_source_inputs', 0)} "
                    f"slice_nodes={feed_summary.get('slice_pack_selected_nodes', 0)} "
                    f"slice_saved_mib={feed_summary.get('slice_pack_saved_mib', 0.0):.3f} "
                    f"slice_mode={feed_summary.get('slice_pack_mode', 'off')} "
                    f"slice_single_only={feed_summary.get('slice_pack_single_consumer_only', False)} "
                    f"slice_single_candidates={feed_summary.get('slice_pack_single_consumer_candidates', 0)} "
                    f"slice_single_filtered={feed_summary.get('slice_pack_single_consumer_filtered', 0)} "
                    f"slice_direct_added_inputs={feed_summary.get('slice_pack_direct_added_inputs', 0)} "
                    f"small_packed={feed_summary.get('small_pack_used', False)} "
                    f"small_packed_inputs={feed_summary.get('small_pack_selected_inputs', 0)} "
                    f"small_pack_mib={feed_summary.get('small_pack_selected_mib', 0.0):.3f} "
                    f"small_pack_max_bytes={feed_summary.get('small_pack_max_bytes', 0)} "
                    f"small_pack_unpack={feed_summary.get('small_pack_unpack_op', 'slice')} "
                    f"remaining_packed={feed_summary.get('remaining_pack_used', False)} "
                    f"remaining_candidates={feed_summary.get('remaining_pack_candidate_inputs', 0)} "
                    f"remaining_packed_inputs={feed_summary.get('remaining_pack_selected_inputs', 0)} "
                    f"remaining_pack_mib={feed_summary.get('remaining_pack_selected_mib', 0.0):.3f} "
                    f"remaining_pack_max_bytes={feed_summary.get('remaining_pack_max_bytes', 0)} "
                    f"batch_mib={feed_summary.get('batch_dim_mib', 0.0):.3f} "
                    f"non_batch_mib={feed_summary.get('non_batch_dim_mib', 0.0):.3f} "
                    f"original_batch_mib={feed_summary.get('original_batch_dim_mib', 0.0):.3f} "
                    f"large_inputs={feed_summary.get('large_input_count', 0)} "
                    f"large_mib={feed_summary.get('large_input_mib', 0.0):.3f} "
                    f"largest={largest.get('name', 'N/A')} "
                    f"largest_mib={largest.get('nbytes', 0) / (1024.0 * 1024.0):.3f}"
                )
                for label, prefix in (
                    ("active_large_feed", "active_feed_top"),
                    ("original_large_feed", "original_feed_top"),
                ):
                    large_feed = feed_summary.get(label) or {}
                    if not large_feed:
                        continue
                    print(
                        f"[INFO] {label}: "
                        f"inputs={large_feed.get('num_inputs', 0)} "
                        f"total_mib={large_feed.get('total_mib', 0.0):.3f} "
                        f"large_inputs={large_feed.get('large_input_count', 0)} "
                        f"large_mib={large_feed.get('large_input_mib', 0.0):.3f} "
                        f"batch_mib={large_feed.get('batch_dim_mib', 0.0):.3f} "
                        f"non_batch_mib={large_feed.get('non_batch_dim_mib', 0.0):.3f} "
                        f"top_consumers={large_feed.get('top_consumer_ops', [])[:8]}"
                    )
                    for index, item in enumerate(
                        (large_feed.get("largest_inputs") or [])[:5],
                        start=1,
                    ):
                        print(
                            f"[INFO] {prefix}{index}: "
                            f"name={item.get('name', 'N/A')} "
                            f"mib={item.get('mib', 0.0):.3f} "
                            f"dtype={item.get('dtype', 'N/A')} "
                            f"shape={item.get('shape', [])} "
                            f"batch_dim={item.get('batch_dim', False)} "
                            f"consumers={item.get('consumer_ops', [])} "
                            f"consumer_nodes={item.get('consumer_nodes', [])[:4]}"
                        )
                large_slice_diag = feed_summary.get("large_slice_feed_diag") or {}
                if large_slice_diag:
                    print(
                        f"[INFO] large_slice_feed_diag: "
                        f"inputs={large_slice_diag.get('num_inputs', 0)} "
                        f"candidates={large_slice_diag.get('num_candidates', 0)} "
                        f"status_counts={large_slice_diag.get('status_counts', {})} "
                        f"selected_slice_mib={large_slice_diag.get('selected_slice_mib', 0.0):.3f} "
                        f"top_disallowed={large_slice_diag.get('top_disallowed_consumer_ops', [])[:8]}"
                    )
                    for index, item in enumerate(
                        (large_slice_diag.get("candidates") or [])[:8],
                        start=1,
                    ):
                        first_slice = (item.get("slice_nodes") or [{}])[0]
                        print(
                            f"[INFO] large_slice_candidate{index}: "
                            f"name={item.get('name', 'N/A')} "
                            f"status={item.get('status', 'N/A')} "
                            f"would_select={item.get('would_select', False)} "
                            f"original_mib={item.get('original_mib', 0.0):.3f} "
                            f"slice_mib={item.get('slice_mib', 0.0):.3f} "
                            f"saved_mib={item.get('saved_mib', 0.0):.3f} "
                            f"shape={item.get('shape', [])} "
                            f"num_slices={item.get('num_slice_nodes', 0)} "
                            f"first_slice={first_slice}"
                        )
                concat_diag = feed_summary.get("concat_pack_diagnostics") or {}
                if concat_diag:
                    print(
                        f"[INFO] concat_pack_diag: "
                        f"counts={concat_diag.get('counts', concat_diag)} "
                        f"top_candidate_mib={concat_diag.get('top_candidate_mib', [])[:5]} "
                        f"top_candidate_inputs={concat_diag.get('top_candidate_inputs', [])[:5]} "
                        f"top_non_feed_input_ops={concat_diag.get('top_non_feed_input_ops', [])[:8]}"
                    )
                    for index, item in enumerate(
                        (concat_diag.get("top_concat_candidates") or [])[:3],
                        start=1,
                    ):
                        print(
                            f"[INFO] concat_candidate{index}: "
                            f"node={item.get('concat_node', 'N/A')} "
                            f"inputs={item.get('inputs', 0)} "
                            f"mib={item.get('mib', 0.0):.3f} "
                            f"axis={item.get('axis', 'N/A')} "
                            f"dtype={item.get('dtype', 'N/A')} "
                            f"shape={item.get('shape', [])}"
                        )
                concat_static_precompute = (
                    feed_summary.get("concat_static_precompute") or {}
                )
                if concat_static_precompute:
                    print(
                        f"[INFO] concat_static_precompute: "
                        f"enabled={concat_static_precompute.get('enabled', False)} "
                        f"reason={concat_static_precompute.get('reason', '')} "
                        f"propagated_shapes={concat_static_precompute.get('propagated_shapes', 0)} "
                        f"splitv_shapes={concat_static_precompute.get('splitv_shapes', 0)} "
                        f"passthrough_shapes={concat_static_precompute.get('passthrough_shapes', 0)} "
                        f"replaced_shape={concat_static_precompute.get('replaced_shape', 0)} "
                        f"replaced_rank={concat_static_precompute.get('replaced_rank', 0)} "
                        f"replaced_size={concat_static_precompute.get('replaced_size', 0)} "
                        f"total_replaced={concat_static_precompute.get('total_replaced', 0)} "
                        f"protected_reshape_shape_nodes={concat_static_precompute.get('protected_reshape_shape_nodes', 0)} "
                        f"skipped={concat_static_precompute.get('skipped', {})} "
                        f"top_shape_input_ops={concat_static_precompute.get('top_shape_input_ops', [])[:8]} "
                        f"sample_shape={concat_static_precompute.get('sample_shape', [])[:8]}"
                    )
                concat_downstream = feed_summary.get("concat_pack_downstream") or {}
                if concat_downstream:
                    print(
                        f"[INFO] concat_downstream: "
                        f"groups={concat_downstream.get('groups', 0)} "
                        f"total_mib={concat_downstream.get('total_mib', 0.0):.3f} "
                        f"original_inputs={concat_downstream.get('total_original_inputs', 0)} "
                        f"downstream_nodes={concat_downstream.get('total_downstream_nodes', 0)} "
                        f"value_independent_nodes={concat_downstream.get('total_value_independent_nodes', 0)} "
                        f"value_independent_top_ops={concat_downstream.get('value_independent_top_ops', [])[:10]} "
                        f"precompute_candidate_nodes={concat_downstream.get('total_precompute_candidate_nodes', 0)} "
                        f"precompute_candidate_top_ops={concat_downstream.get('precompute_candidate_top_ops', [])[:10]} "
                        f"top_ops={concat_downstream.get('top_ops', [])[:10]}"
                    )
                    for index, item in enumerate(
                        (concat_downstream.get("items") or [])[:5], start=1
                    ):
                        chain = [
                            f"{node.get('op')}:{node.get('name')}"
                            for node in (item.get("chain") or [])[:8]
                        ]
                        print(
                            f"[INFO] concat_downstream{index}: "
                            f"concat={item.get('concat_node', 'N/A')} "
                            f"mib={item.get('mib', 0.0):.3f} "
                            f"original_inputs={item.get('num_original_inputs', 0)} "
                            f"chunks={item.get('chunks', 0)} "
                            f"downstream_nodes={item.get('downstream_nodes', 0)} "
                            f"value_independent_nodes={item.get('value_independent_nodes', 0)} "
                            f"value_independent_top_ops={item.get('value_independent_top_ops', [])[:10]} "
                            f"precompute_candidate_nodes={item.get('precompute_candidate_nodes', 0)} "
                            f"precompute_candidate_top_ops={item.get('precompute_candidate_top_ops', [])[:10]} "
                            f"precompute_boundary_ops={item.get('precompute_boundary_ops', [])[:10]} "
                            f"precompute_matmul_nodes={item.get('precompute_matmul_nodes', 0)} "
                            f"source_placeholders={item.get('source_placeholder_count', 0)} "
                            f"external_placeholders={item.get('external_placeholder_count', 0)} "
                            f"direct_consumers={item.get('direct_consumers', [])[:8]} "
                            f"output_hits={item.get('output_hits', [])[:8]} "
                            f"top_ops={item.get('top_ops', [])[:10]} "
                            f"chain={chain}"
                        )
                slice_diag = feed_summary.get("slice_pack_diagnostics") or {}
                if slice_diag:
                    print(
                        f"[INFO] slice_pack_diag: "
                        f"mode={slice_diag.get('mode', 'N/A')} "
                        f"skip_reasons={slice_diag.get('slice_pack_skip_reasons', {})} "
                        f"top_disallowed_consumer_ops={slice_diag.get('top_disallowed_consumer_ops', [])[:8]} "
                        f"candidate_added_inputs={slice_diag.get('candidate_added_inputs', 0)} "
                        f"auto_skip_reason={slice_diag.get('auto_skip_reason', '')}"
                    )
                print(
                    f"[INFO] output: fetch_mode={output_summary.get('fetch_mode', 'outputs')} "
                    f"indices={output_summary.get('selected_output_indices', [])} "
                    f"estimated_mib={output_summary.get('estimated_total_mib', 0.0):.3f}"
                )
                for index, item in enumerate((report.get("outputs") or [])[:8]):
                    print(
                        f"[INFO] output_tensor{index}: "
                        f"name={item.get('name', 'N/A')} "
                        f"op={item.get('optype', 'N/A')} "
                        f"dtype={item.get('dtype', 'N/A')} "
                        f"shape={item.get('shape_in_graph', None)} "
                        f"estimated_mib={(item.get('estimated_nbytes') or 0) / (1024.0 * 1024.0):.6f}"
                    )
                result_shapes = report.get("result_shapes") or []
                if result_shapes:
                    print(
                        f"[INFO] result: total_mib={report.get('result_total_mib', 0.0):.6f} "
                        f"shapes={result_shapes[:8]}"
                    )
                output_fetch_plan = report.get("output_fetch_plan") or {}
                if output_fetch_plan:
                    print(
                        f"[INFO] output_fetch_plan: "
                        f"enabled={output_fetch_plan.get('enabled', False)} "
                        f"pack={output_fetch_plan.get('pack_output_fetches', False)} "
                        f"outputs={output_fetch_plan.get('num_outputs', 0)} "
                        f"original_device_fetches={output_fetch_plan.get('original_device_fetch_count', 0)} "
                        f"device_fetches={output_fetch_plan.get('device_fetch_count', 0)} "
                        f"host_outputs={output_fetch_plan.get('host_output_count', 0)} "
                        f"deduped={output_fetch_plan.get('deduped_output_count', 0)} "
                        f"packed_groups={output_fetch_plan.get('packed_output_groups', [])[:4]} "
                        f"device_names={output_fetch_plan.get('device_fetch_names', [])[:8]}"
                    )
                identity_rewrite = (
                    (report.get("graph_rewrite") or {})
                    .get("identity_bypass")
                    or {}
                )
                if identity_rewrite:
                    print(
                        f"[INFO] identity_bypass: "
                        f"enabled={identity_rewrite.get('enabled', False)} "
                        f"eligible={identity_rewrite.get('eligible_identity_nodes', 0)} "
                        f"rewired_edges={identity_rewrite.get('rewired_edges', 0)} "
                        f"pruned={identity_rewrite.get('pruned_identity_nodes', 0)} "
                        f"skipped_control={identity_rewrite.get('skipped_with_control', 0)} "
                        f"skipped_control_consumer={identity_rewrite.get('skipped_control_consumer', 0)} "
                        f"output_remaps={identity_rewrite.get('output_remaps', [])[:4]}"
                    )
                pow_square_rewrite = (
                    (report.get("graph_rewrite") or {})
                    .get("pow_square")
                    or {}
                )
                if pow_square_rewrite:
                    print(
                        f"[INFO] pow_square: "
                        f"enabled={pow_square_rewrite.get('enabled', False)} "
                        f"rewritten={pow_square_rewrite.get('rewritten', 0)} "
                        f"skipped_non_const={pow_square_rewrite.get('skipped_non_const', 0)} "
                        f"skipped_non_square={pow_square_rewrite.get('skipped_non_square', 0)} "
                        f"non_const_ops="
                        f"{pow_square_rewrite.get('non_const_exponent_ops', [])} "
                        f"non_square_values="
                        f"{pow_square_rewrite.get('non_square_constant_values', [])} "
                        f"sample={pow_square_rewrite.get('sample', [])[:8]}"
                    )
                static_shape_rewrite = (
                    (report.get("graph_rewrite") or {})
                    .get("static_shape_rewrite")
                    or {}
                )
                if static_shape_rewrite:
                    print(
                        f"[INFO] static_shape_rewrite: "
                        f"enabled={static_shape_rewrite.get('enabled', False)} "
                        f"replaced_shape={static_shape_rewrite.get('replaced_shape', 0)} "
                        f"replaced_rank={static_shape_rewrite.get('replaced_rank', 0)} "
                        f"replaced_size={static_shape_rewrite.get('replaced_size', 0)} "
                        f"total_replaced={static_shape_rewrite.get('total_replaced', 0)} "
                        f"protected_reshape_shape_nodes={static_shape_rewrite.get('protected_reshape_shape_nodes', 0)} "
                        f"skipped={static_shape_rewrite.get('skipped', {})} "
                        f"sample_shape={static_shape_rewrite.get('sample_shape', [])[:8]}"
                    )
                matmul_analysis = (
                    (report.get("graph_rewrite") or {})
                    .get("matmul_group_analysis")
                    or {}
                )
                if matmul_analysis:
                    print(
                        f"[INFO] matmul_candidates: "
                        f"placeholders={matmul_analysis.get('placeholder_count', 0)} "
                        f"min_placeholders={matmul_analysis.get('min_placeholders', 0)} "
                        f"total={matmul_analysis.get('matmul_total', 0)} "
                        f"same_lhs_groups={matmul_analysis.get('same_lhs_group_count', 0)} "
                        f"same_lhs_matmuls={matmul_analysis.get('same_lhs_matmul_count', 0)} "
                        f"same_lhs_reduction="
                        f"{matmul_analysis.get('estimated_call_reduction_if_grouped_by_lhs', 0)} "
                        f"same_rhs_groups={matmul_analysis.get('same_rhs_group_count', 0)} "
                        f"same_rhs_matmuls={matmul_analysis.get('same_rhs_matmul_count', 0)} "
                        f"same_rhs_reduction="
                        f"{matmul_analysis.get('estimated_call_reduction_if_grouped_by_rhs', 0)} "
                        f"rhs_producers={matmul_analysis.get('matmul_rhs_producer_ops', [])[:8]}"
                    )
                    print(
                        f"[INFO] top_same_lhs_matmul_groups: "
                        f"{matmul_analysis.get('top_same_lhs_groups', [])[:3]}"
                    )
                    print(
                        f"[INFO] top_same_rhs_matmul_groups: "
                        f"{matmul_analysis.get('top_same_rhs_groups', [])[:3]}"
                    )
                same_lhs_auto = (
                    (report.get("graph_rewrite") or {})
                    .get("same_lhs_matmul_auto_decision")
                    or {}
                )
                if same_lhs_auto:
                    print(
                        f"[INFO] same_lhs_matmul_auto: "
                        f"mode={same_lhs_auto.get('mode', 'off')} "
                        f"enabled={same_lhs_auto.get('enabled', False)} "
                        f"reason={same_lhs_auto.get('reason', 'unknown')} "
                        f"placeholders={same_lhs_auto.get('placeholder_count', 0)} "
                        f"min_placeholders={same_lhs_auto.get('min_placeholders', 0)} "
                        f"lhs_reduction={same_lhs_auto.get('lhs_reduction', 0)} "
                        f"rhs_reduction={same_lhs_auto.get('rhs_reduction', 0)} "
                        f"active_rhs_reduction={same_lhs_auto.get('active_rhs_reduction', 0)} "
                        f"total_reduction={same_lhs_auto.get('total_reduction', 0)} "
                        f"min_reduction={same_lhs_auto.get('min_reduction', 0)} "
                        f"include_rhs={same_lhs_auto.get('include_rhs', 'off')}"
                    )
                same_lhs_rewrite = (
                    (report.get("graph_rewrite") or {})
                    .get("same_lhs_matmul")
                    or {}
                )
                if same_lhs_rewrite:
                    print(
                        f"[INFO] same_lhs_matmul: "
                        f"enabled={same_lhs_rewrite.get('enabled', False)} "
                        f"groups={same_lhs_rewrite.get('groups', 0)} "
                        f"original_matmuls={same_lhs_rewrite.get('original_matmuls', 0)} "
                        f"fused_matmuls={same_lhs_rewrite.get('fused_matmuls', 0)} "
                        f"estimated_reduction={same_lhs_rewrite.get('estimated_matmul_reduction', 0)} "
                        f"biasadd_fusion={same_lhs_rewrite.get('biasadd_fusion_enabled', False)} "
                        f"biasadd_groups={same_lhs_rewrite.get('biasadd_groups', 0)} "
                        f"fused_biasadds={same_lhs_rewrite.get('fused_biasadds', 0)} "
                        f"estimated_biasadd_reduction={same_lhs_rewrite.get('estimated_biasadd_reduction', 0)} "
                        f"post_unary_fusion={same_lhs_rewrite.get('post_unary_fusion_enabled', False)} "
                        f"post_unary_groups={same_lhs_rewrite.get('post_unary_groups', 0)} "
                        f"fused_post_unary_ops={same_lhs_rewrite.get('fused_post_unary_ops', 0)} "
                        f"estimated_post_unary_reduction={same_lhs_rewrite.get('estimated_post_unary_reduction', 0)} "
                        f"post_binary_fusion={same_lhs_rewrite.get('post_binary_fusion_enabled', False)} "
                        f"post_binary_groups={same_lhs_rewrite.get('post_binary_groups', 0)} "
                        f"fused_post_binary_ops={same_lhs_rewrite.get('fused_post_binary_ops', 0)} "
                        f"estimated_post_binary_reduction={same_lhs_rewrite.get('estimated_post_binary_reduction', 0)} "
                        f"post_concat_compaction={same_lhs_rewrite.get('post_concat_compaction_enabled', False)} "
                        f"post_concat_compacted={same_lhs_rewrite.get('post_concat_compacted', 0)} "
                        f"post_concat_sources={same_lhs_rewrite.get('post_concat_sources', {})} "
                        f"rewired_edges={same_lhs_rewrite.get('rewired_edges', 0)} "
                        f"skipped={same_lhs_rewrite.get('skipped', {})} "
                        f"skipped_biasadd={same_lhs_rewrite.get('skipped_biasadd_fusion', {})} "
                        f"skipped_post_unary={same_lhs_rewrite.get('skipped_post_unary_fusion', {})} "
                        f"skipped_post_binary={same_lhs_rewrite.get('skipped_post_binary_fusion', {})} "
                        f"skipped_post_concat={same_lhs_rewrite.get('skipped_post_concat_compaction', {})}"
                    )
                    for index, item in enumerate(
                        (same_lhs_rewrite.get("sample_groups") or [])[:3],
                        start=1,
                    ):
                        print(
                            f"[INFO] same_lhs_group{index}: "
                            f"mode={item.get('mode', 'same_lhs')} "
                            f"lhs={item.get('lhs', 'N/A')} "
                            f"rhs={item.get('rhs', 'N/A')} "
                            f"lhs_shape={item.get('lhs_shape', [])} "
                            f"matmuls={item.get('matmul_count', 0)} "
                            f"chunk={item.get('chunk_index', 0)} "
                            f"original_group={item.get('original_group_size', 0)} "
                            f"k={item.get('k_dim', 'N/A')} "
                            f"total_cols={item.get('total_cols', 0)} "
                            f"total_rows={item.get('total_rows', 0)} "
                            f"estimated_mflops={item.get('estimated_mflops', 0):.1f} "
                            f"fused={item.get('fused_matmul', 'N/A')} "
                            f"biasadd_fused={item.get('biasadd_fused', False)} "
                            f"fused_biasadd={item.get('fused_biasadd', '')} "
                            f"post_unary_fused={item.get('post_unary_fused', False)} "
                            f"post_unary_op={item.get('post_unary_op', '')} "
                            f"fused_post_unary={item.get('fused_post_unary', '')} "
                            f"post_binary_fused={item.get('post_binary_fused', False)} "
                            f"post_binary_op={item.get('post_binary_op', '')} "
                            f"post_binary_other_kind={item.get('post_binary_other_kind', '')} "
                            f"fused_post_binary={item.get('fused_post_binary', '')} "
                            f"post_concat_source={item.get('post_concat_source', '')} "
                            f"post_concat_compacted={item.get('post_concat_compacted', 0)} "
                            f"sample={item.get('sample_matmuls', [])[:5]}"
                        )
                same_shape_rewrite = (
                    (report.get("graph_rewrite") or {})
                    .get("same_shape_batch_matmul")
                    or {}
                )
                if same_shape_rewrite:
                    print(
                        f"[INFO] same_shape_batch_matmul: "
                        f"enabled={same_shape_rewrite.get('enabled', False)} "
                        f"groups={same_shape_rewrite.get('groups', 0)} "
                        f"original_matmuls={same_shape_rewrite.get('original_matmuls', 0)} "
                        f"fused_batch_matmuls={same_shape_rewrite.get('fused_batch_matmuls', 0)} "
                        f"estimated_reduction={same_shape_rewrite.get('estimated_matmul_reduction', 0)} "
                        f"rewired_edges={same_shape_rewrite.get('rewired_edges', 0)} "
                        f"skipped={same_shape_rewrite.get('skipped', {})}"
                    )
                    for index, item in enumerate(
                        (same_shape_rewrite.get("sample_groups") or [])[:3],
                        start=1,
                    ):
                        print(
                            f"[INFO] same_shape_batch_group{index}: "
                            f"matmuls={item.get('matmul_count', 0)} "
                            f"chunk={item.get('chunk_index', 0)} "
                            f"original_group={item.get('original_group_size', 0)} "
                            f"shape=({item.get('out_rows', 'N/A')},"
                            f"{item.get('k_dim', 'N/A')},"
                            f"{item.get('out_cols', 'N/A')}) "
                            f"estimated_mflops={item.get('estimated_mflops', 0):.1f} "
                            f"fused={item.get('fused_batch_matmul', 'N/A')} "
                            f"sample={item.get('sample_matmuls', [])[:5]}"
                        )
                for index, item in enumerate(
                    (report.get("output_dependency_summary") or [])[:4]
                ):
                    chain = [
                        f"{node.get('op')}:{node.get('name')}"
                        for node in (item.get("main_chain") or [])[:8]
                    ]
                    print(
                        f"[INFO] output_deps{index}: "
                        f"root={item.get('root_name', 'N/A')} "
                        f"root_op={item.get('root_op', 'N/A')} "
                        f"nodes={item.get('reachable_nodes', 0)} "
                        f"placeholders={item.get('placeholder_count', 0)} "
                        f"top_ops={item.get('top_ops', [])[:10]} "
                        f"root_inputs={item.get('root_inputs', [])[:6]} "
                        f"root_input_tree={item.get('root_input_tree', [])[:3]} "
                        f"chain={chain}"
                    )
                remaining = feed_summary.get("remaining_original_feed") or {}
                if remaining:
                    print(
                        f"[INFO] remaining_original_feed: "
                        f"inputs={remaining.get('num_inputs', 0)} "
                        f"total_mib={remaining.get('total_mib', 0.0):.3f} "
                        f"top_consumers={remaining.get('top_consumer_ops', [])[:8]}"
                    )
                    for index, item in enumerate(
                        (remaining.get("largest_inputs") or [])[:5], start=1
                    ):
                        print(
                            f"[INFO] remaining_top{index}: "
                            f"name={item.get('name', 'N/A')} "
                            f"mib={item.get('mib', 0.0):.3f} "
                            f"shape={item.get('shape', [])} "
                            f"consumers={item.get('consumer_ops', [])}"
                        )
                    for index, item in enumerate(
                        (remaining.get("top_slice_candidates") or [])[:5],
                        start=1,
                    ):
                        print(
                            f"[INFO] slice_candidate{index}: "
                            f"name={item.get('name', 'N/A')} "
                            f"status={item.get('status', 'N/A')} "
                            f"original_mib={item.get('original_mib', 0.0):.3f} "
                            f"slice_mib={item.get('slice_mib', 0.0):.3f} "
                            f"saved_mib={item.get('saved_mib', 0.0):.3f} "
                            f"num_slices={item.get('num_slice_nodes', 0)}"
                        )
            if report.get("error_core"):
                print(f"[INFO] core error: {report['error_core']}")
            if report.get("error_tail"):
                print("[INFO] error tail:")
                for line in report["error_tail"]:
                    print(f"[INFO]   {line}")

    xla_dump_summary = summarize_xla_dump_dir(os.environ.get("GRAPH_RUNNER_XLA_DUMP_DIR"))
    summary = {
        "total_specs": len(specs),
        "bs_values": bs_values,
        "total_runs": len(all_reports),
        "ok": sum(1 for item in all_reports if item.get("status") == "ok"),
        "failed": sum(1 for item in all_reports if item.get("status") != "ok"),
    }
    avg_time_summary = []
    latency_summary = []
    for report in all_reports:
        timing = report.get("timing_ms") or {}
        avg_time_summary.append(
            {
                "spec_path": report.get("spec_path"),
                "pb_path": report.get("pb_path"),
                "batch_size": report.get("batch_size"),
                "status": report.get("status"),
                "average_time_ms": timing.get("average"),
                "trimmed_avg_ms": timing.get("trimmed_avg"),
                "timing_ms": timing,
                "error_core": report.get("error_core"),
            }
        )
        latency_summary.append(
            {
                "batch_size": report.get("batch_size"),
                "average_time_ms": timing.get("average"),
                "trimmed_avg_ms": timing.get("trimmed_avg"),
            }
        )
    avg_time_summary.sort(
        key=lambda item: (str(item.get("pb_path") or ""), int(item.get("batch_size") or 0))
    )

    final_report = {
        "args": vars(args),
        "runtime_env": {
            "TF_XLA_FLAGS": os.environ.get("TF_XLA_FLAGS", ""),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            "TF_PLUGGABLE_DEVICE_LIBRARY_PATH": os.environ.get(
                "TF_PLUGGABLE_DEVICE_LIBRARY_PATH", ""
            ),
            "MUSA_PJRT_PLUGIN_PATH": os.environ.get(
                "MUSA_PJRT_PLUGIN_PATH", ""
            ),
            "PJRT_NAMES_AND_LIBRARY_PATHS": os.environ.get(
                "PJRT_NAMES_AND_LIBRARY_PATHS", ""
            ),
            "MUSA_NPD_IS_PLUGGABLE_DEVICE": os.environ.get(
                "MUSA_NPD_IS_PLUGGABLE_DEVICE", ""
            ),
            "MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE": os.environ.get(
                "MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE", ""
            ),
            "MUSA_PJRT_FORCE_HOST_BUFFER_COPY": os.environ.get(
                "MUSA_PJRT_FORCE_HOST_BUFFER_COPY", ""
            ),
            "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS": os.environ.get(
                "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS", ""
            ),
            "MUSA_PJRT_MAX_INFLIGHT_EXECUTES": os.environ.get(
                "MUSA_PJRT_MAX_INFLIGHT_EXECUTES", ""
            ),
            "MUSA_PJRT_MAX_INFLIGHT_COMPUTATIONS": os.environ.get(
                "MUSA_PJRT_MAX_INFLIGHT_COMPUTATIONS", ""
            ),
            "MUSA_PJRT_NUM_DEVICE_TO_HOST_STREAMS": os.environ.get(
                "MUSA_PJRT_NUM_DEVICE_TO_HOST_STREAMS", ""
            ),
            "MUSA_PJRT_NUM_DEVICE_TO_DEVICE_STREAMS": os.environ.get(
                "MUSA_PJRT_NUM_DEVICE_TO_DEVICE_STREAMS", ""
            ),
            "MUSA_PJRT_WAIT_TRANSFER_DONE": os.environ.get(
                "MUSA_PJRT_WAIT_TRANSFER_DONE", ""
            ),
            "MUSA_PJRT_WAIT_EXECUTE_DONE": os.environ.get(
                "MUSA_PJRT_WAIT_EXECUTE_DONE", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_DIAGNOSTICS": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_DIAGNOSTICS", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ASYNC": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ASYNC", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PARALLEL_PACK": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PARALLEL_PACK", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_THREADS": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_THREADS", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_MIN_BYTES": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_PACK_MIN_BYTES", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_DIRTY_RANGES": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_DIRTY_RANGES", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_POOL_ORDER_LAYOUT": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_ARENA_POOL_ORDER_LAYOUT", ""
            ),
            "MUSA_PJRT_REUSE_HOST_BUFFERS_TRUST_CONTENTS": os.environ.get(
                "MUSA_PJRT_REUSE_HOST_BUFFERS_TRUST_CONTENTS", ""
            ),
            "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS": os.environ.get(
                "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS", ""
            ),
            "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS_TRUST_LIFETIME": os.environ.get(
                "MUSA_PJRT_CACHE_REUSED_BUFFER_VIEWS_TRUST_LIFETIME", ""
            ),
            "MUSA_PJRT_BYPASS_EVENT_DESTROY": os.environ.get(
                "MUSA_PJRT_BYPASS_EVENT_DESTROY", ""
            ),
            "MUSA_PJRT_BYPASS_BUFFER_DESTROY": os.environ.get(
                "MUSA_PJRT_BYPASS_BUFFER_DESTROY", ""
            ),
            "MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT": os.environ.get(
                "MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT", ""
            ),
            "MUSA_XLA_MAX_FUSION_OPERANDS": os.environ.get(
                "MUSA_XLA_MAX_FUSION_OPERANDS", ""
            ),
            "MUSA_XLA_DOT_MERGER_MAX_MIB": os.environ.get(
                "MUSA_XLA_DOT_MERGER_MAX_MIB", ""
            ),
            "MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX": os.environ.get(
                "MUSA_XLA_FUSE_BROADCAST_BIAS_AS_MATRIX", ""
            ),
            "MUSA_XLA_AVOID_GEMM_BETA_CHAIN": os.environ.get(
                "MUSA_XLA_AVOID_GEMM_BETA_CHAIN", ""
            ),
            "MUSA_XLA_GEMM_EPILOGUE_FUSION": os.environ.get(
                "MUSA_XLA_GEMM_EPILOGUE_FUSION", ""
            ),
            "MUSA_XLA_GEMM_EPILOGUE_FUSION_LOG": os.environ.get(
                "MUSA_XLA_GEMM_EPILOGUE_FUSION_LOG", ""
            ),
            "MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS": os.environ.get(
                "MUSA_XLA_GEMM_EPILOGUE_FUSE_BROADCAST_BIAS", ""
            ),
            "MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL": os.environ.get(
                "MUSA_XLA_GEMM_EPILOGUE_CUSTOM_CALL", ""
            ),
            "MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA": os.environ.get(
                "MUSA_XLA_GEMM_EPILOGUE_FORCE_BROADCAST_BIAS_BETA", ""
            ),
            "MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES": os.environ.get(
                "MUSA_XLA_GEMM_EPILOGUE_ONLY_SHAPES", ""
            ),
            "MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT": os.environ.get(
                "MUSA_XLA_GEMM_EPILOGUE_DISABLE_MUBLASLT", ""
            ),
            "MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS": os.environ.get(
                "MUSA_GEMM_EPILOGUE_THUNK_DIAGNOSTICS", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_PATTERN": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_PATTERN", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_PATTERN_LOG": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_PATTERN_LOG", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_FUSION": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_FUSION", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_FUSION_LOG": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_FUSION_LOG", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_LOG_EMPTY", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_FUSION_KIND": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_FUSION_KIND", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_REQUIRE_ADD", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_MAX_CHAIN_LENGTH", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_MODULE", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_MIN_M": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_MIN_M", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_MIN_K": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_MIN_K", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_MAX_FUSIONS_PER_PATTERN", ""
            ),
            "MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE": os.environ.get(
                "MUSA_XLA_DOT_EPILOGUE_SORT_BY_SIZE", ""
            ),
            "MUSA_XLA_GEMM_BETA_CHAIN_MERGER": os.environ.get(
                "MUSA_XLA_GEMM_BETA_CHAIN_MERGER", ""
            ),
            "MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH": os.environ.get(
                "MUSA_XLA_GEMM_BETA_CHAIN_MIN_CHAIN_LENGTH", ""
            ),
            "MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS": os.environ.get(
                "MUSA_XLA_GEMM_BETA_CHAIN_MAX_CHAINS", ""
            ),
            "MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K": os.environ.get(
                "MUSA_XLA_GEMM_BETA_CHAIN_MAX_TOTAL_K", ""
            ),
            "MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL": os.environ.get(
                "MUSA_XLA_GEMM_BETA_CHAIN_CUSTOM_CALL", ""
            ),
            "MUSA_XLA_GEMM_BETA_CHAIN_SKIP_DEBUG_INFO": os.environ.get(
                "MUSA_XLA_GEMM_BETA_CHAIN_SKIP_DEBUG_INFO", ""
            ),
            "MUSA_XLA_GEMM_BETA_CHAIN_MERGER_LOG": os.environ.get(
                "MUSA_XLA_GEMM_BETA_CHAIN_MERGER_LOG", ""
            ),
            "MUSA_XLA_GEMM_BETA_CHAIN_LOG_EMPTY": os.environ.get(
                "MUSA_XLA_GEMM_BETA_CHAIN_LOG_EMPTY", ""
            ),
            "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER": os.environ.get(
                "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER", ""
            ),
            "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB": os.environ.get(
                "MUSA_XLA_POST_TRANSPOSE_DOT_MERGER_MAX_MIB", ""
            ),
            "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER": os.environ.get(
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_REDUCTION_PRODUCER", ""
            ),
            "MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL": os.environ.get(
                "MUSA_XLA_TUPLE_WARP_ROW_REDUCTION_KERNEL", ""
            ),
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL": os.environ.get(
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_KERNEL", ""
            ),
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS": os.environ.get(
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_MIN_DATA_ELEMENTS", ""
            ),
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_MAX": os.environ.get(
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_MAX", ""
            ),
            "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_THREADS_PER_BLOCK": os.environ.get(
                "MUSA_XLA_MIXED_TUPLE_WARP_ROW_REDUCTION_SMALL_WIDTH_THREADS_PER_BLOCK",
                "",
            ),
            "MUSA_XLA_DIRECT_MT_POW": os.environ.get(
                "MUSA_XLA_DIRECT_MT_POW", ""
            ),
            "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS": os.environ.get(
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_ELEMENTS", ""
            ),
            "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS": os.environ.get(
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_MIN_OPERANDS", ""
            ),
            "MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG": os.environ.get(
                "MUSA_XLA_FUSION_MERGER_MATERIALIZE_LOG", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCHER": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCHER", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_DIAG_ONLY", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_GROUP_SIZE", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUP_SIZE", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_GROUPS", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MIN_CANDIDATE_DOTS", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_SLICE_BYTES_PER_SAVED_LAUNCH",
                "",
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_MAX_OUTPUT_COLS", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCHER_LOG", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POST_DOT_DIAG": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POST_DOT_DIAG", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_DIAG": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_DIAG", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_REWRITE": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_REWRITE", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_EXTERNAL_DIAG": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_EXTERNAL_DIAG", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_DIAG": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_DIAG", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MAX_DEPTH": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MAX_DEPTH", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_REWRITE": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_ADD_TREE_MIXED_KEY_REWRITE", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_BIASADD": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_BIASADD", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POINTER_ARRAY_OUTPUT": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_POINTER_ARRAY_OUTPUT", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_KERNEL": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_KERNEL", ""
            ),
            "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_MAX_GROUP_SIZE": os.environ.get(
                "MUSA_XLA_SAME_SHAPE_DOT_BATCH_SMALL_K_CUSTOM_MAX_GROUP_SIZE",
                "",
            ),
            "MUSA_XLA_GROUP_GEMM_THUNKS": os.environ.get(
                "MUSA_XLA_GROUP_GEMM_THUNKS", ""
            ),
            "MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE": os.environ.get(
                "MUSA_XLA_GROUP_GEMM_THUNKS_MIN_GROUP_SIZE", ""
            ),
            "MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE": os.environ.get(
                "MUSA_XLA_GROUP_GEMM_THUNKS_MAX_GROUP_SIZE", ""
            ),
            "MUSA_XLA_GROUP_GEMM_THUNKS_LOG": os.environ.get(
                "MUSA_XLA_GROUP_GEMM_THUNKS_LOG", ""
            ),
            "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG": os.environ.get(
                "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_DIAG", ""
            ),
            "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_MAX_SEPARATORS": os.environ.get(
                "MUSA_XLA_GROUP_GEMM_THUNKS_CROSS_KERNEL_MAX_SEPARATORS", ""
            ),
            "MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS": os.environ.get(
                "MUSA_XLA_SMALL_GEMM_ACCUM_THUNKS", ""
            ),
            "MUSA_XLA_SMALL_GEMM_ACCUM_LOG": os.environ.get(
                "MUSA_XLA_SMALL_GEMM_ACCUM_LOG", ""
            ),
            "MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE": os.environ.get(
                "MUSA_XLA_SMALL_GEMM_ACCUM_MIN_CHAIN_SIZE", ""
            ),
            "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE": os.environ.get(
                "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_CHAIN_SIZE", ""
            ),
            "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K": os.environ.get(
                "MUSA_XLA_SMALL_GEMM_ACCUM_MAX_K", ""
            ),
            "MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL": os.environ.get(
                "MUSA_XLA_SMALL_GEMM_ACCUM_REQUIRE_CUSTOM_KERNEL", ""
            ),
            "MUSA_XLA_GPU_RUNTIME": os.environ.get("MUSA_XLA_GPU_RUNTIME", ""),
            "MUSA_XLA_GLOBAL_JIT_LEVEL": os.environ.get(
                "MUSA_XLA_GLOBAL_JIT_LEVEL", ""
            ),
            "MUSA_CUSTOM_FUSION": os.environ.get("MUSA_CUSTOM_FUSION", ""),
            "MUSA_MUDNN_INTERLEAVED_BATCH_GEMM": os.environ.get(
                "MUSA_MUDNN_INTERLEAVED_BATCH_GEMM", ""
            ),
            "MUSA_GEMM_BACKEND": os.environ.get("MUSA_GEMM_BACKEND", ""),
            "MUSA_F32_FAST_TF32": os.environ.get("MUSA_F32_FAST_TF32", ""),
            "MUSA_F32_FAST_TF32_SHAPES": os.environ.get(
                "MUSA_F32_FAST_TF32_SHAPES", ""
            ),
            "MUSA_GEMM_SMALLK_MIN_MAJOR": os.environ.get(
                "MUSA_GEMM_SMALLK_MIN_MAJOR", ""
            ),
            "MUSA_GEMM_SMALLK_MAX_MINOR": os.environ.get(
                "MUSA_GEMM_SMALLK_MAX_MINOR", ""
            ),
            "MUSA_GEMM_SMALLK_MAX_K": os.environ.get(
                "MUSA_GEMM_SMALLK_MAX_K", ""
            ),
            "MUSA_GEMM_AUTO_SMALLK_MIN_MAJOR": os.environ.get(
                "MUSA_GEMM_AUTO_SMALLK_MIN_MAJOR", ""
            ),
            "MUSA_GEMM_AUTO_SMALLK_MAX_MINOR": os.environ.get(
                "MUSA_GEMM_AUTO_SMALLK_MAX_MINOR", ""
            ),
            "MUSA_GEMM_AUTO_SMALLK_MAX_K": os.environ.get(
                "MUSA_GEMM_AUTO_SMALLK_MAX_K", ""
            ),
            "MUSA_GEMM_AUTO_SKINNY_MIN_MAJOR": os.environ.get(
                "MUSA_GEMM_AUTO_SKINNY_MIN_MAJOR", ""
            ),
            "MUSA_GEMM_AUTO_SKINNY_MAX_MINOR": os.environ.get(
                "MUSA_GEMM_AUTO_SKINNY_MAX_MINOR", ""
            ),
            "MUSA_GEMM_AUTO_SKINNY_MIN_K": os.environ.get(
                "MUSA_GEMM_AUTO_SKINNY_MIN_K", ""
            ),
            "MUSA_GEMM_AUTO_SKINNY_MAX_K": os.environ.get(
                "MUSA_GEMM_AUTO_SKINNY_MAX_K", ""
            ),
            "MUSA_STRIDED_BATCHED_GEMM_BACKEND": os.environ.get(
                "MUSA_STRIDED_BATCHED_GEMM_BACKEND", ""
            ),
            "MUSA_BLAS_GEMM_DIAGNOSTICS": os.environ.get(
                "MUSA_BLAS_GEMM_DIAGNOSTICS", ""
            ),
            "MUSA_HLO_PATTERN_ANALYSIS": os.environ.get(
                "MUSA_HLO_PATTERN_ANALYSIS", ""
            ),
            "MUSA_HLO_PATTERN_ANALYSIS_VERBOSE": os.environ.get(
                "MUSA_HLO_PATTERN_ANALYSIS_VERBOSE", ""
            ),
            "MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY": os.environ.get(
                "MUSA_HLO_PATTERN_ANALYSIS_LOG_EMPTY", ""
            ),
            "MUSA_XLA_THUNK_DIAGNOSTICS": os.environ.get(
                "MUSA_XLA_THUNK_DIAGNOSTICS", ""
            ),
            "MUSA_XLA_THUNK_TIMING": os.environ.get(
                "MUSA_XLA_THUNK_TIMING", ""
            ),
            "MUSA_XLA_EXECUTION_PATH_VERBOSE": os.environ.get(
                "MUSA_XLA_EXECUTION_PATH_VERBOSE", ""
            ),
            "MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS": os.environ.get(
                "MUSA_XLA_GEMM_RUNTIME_DIAGNOSTICS", ""
            ),
            "MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL": os.environ.get(
                "MUSA_XLA_GEMM_RUNTIME_LOG_INTERVAL", ""
            ),
            "MUSA_PINNED_FEED": os.environ.get("MUSA_PINNED_FEED", ""),
            "MUSA_PINNED_H2D_ON_COMPUTE_STREAM": os.environ.get(
                "MUSA_PINNED_H2D_ON_COMPUTE_STREAM", ""
            ),
        },
        "summary": summary,
        "xla_dump_summary": xla_dump_summary,
        "results": all_reports,
        "average_time_summary": avg_time_summary,
    }

    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "run_report.json"
    report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] report={report_path}")
    print(f"[OK] summary={summary}")
    print(f"[OK] latency_summary={latency_summary}")
    if xla_dump_summary:
        print(f"[OK] xla_dump_summary={xla_dump_summary}")

    if failures and args.strict:
        raise RuntimeError("some specs failed, see run_report.json")


if __name__ == "__main__":
    main()
