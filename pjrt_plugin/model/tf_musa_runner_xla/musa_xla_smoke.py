#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "WORKSPACE").exists() and (path / "pjrt_plugin").exists():
            return path
    # Fallback for the checked-in location: <repo>/pjrt_plugin/model[/subdir].
    for path in (start, *start.parents):
        if path.name == "xla":
            return path
    raise RuntimeError(f"cannot find XLA repo root from {start}")


REPO_ROOT = find_repo_root(SCRIPT_DIR)
DEFAULT_MUSA_PLUGIN_PATH = (
    REPO_ROOT / "bazel-bin" / "pjrt_plugin" / "libmusa_pjrt_plugin.so"
)
DEFAULT_MUSA_TF_ADAPTER_PATH = (
    REPO_ROOT / "bazel-bin" / "pjrt_plugin" / "libmusa_tf215_npd_adapter.so"
)


def append_unique_flag(current_value: str, new_flag: str) -> str:
    tokens = current_value.split()
    if new_flag not in tokens:
        tokens.append(new_flag)
    return " ".join(tokens).strip()


def set_flag_with_prefix(current_value: str, flag_prefix: str, new_flag: str) -> str:
    tokens = [token for token in current_value.split() if not token.startswith(flag_prefix)]
    tokens.append(new_flag)
    return " ".join(tokens).strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="MUSA XLA smoke tests for specific ops such as log1p and conv2d."
    )
    parser.add_argument(
        "--case",
        choices=("all", "log1p", "conv2d", "matmul"),
        default="all",
        help="Smoke case to run.",
    )
    parser.add_argument("--device", default="/device:MUSA:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--musa_plugin", default=str(DEFAULT_MUSA_PLUGIN_PATH))
    parser.add_argument("--musa_tf_adapter", default=str(DEFAULT_MUSA_TF_ADAPTER_PATH))
    parser.add_argument("--pjrt_force_host_copy", choices=("on", "off"), default="on")
    parser.add_argument("--pjrt_debug_log", choices=("on", "off"), default="off")
    parser.add_argument("--xla_dump_dir", default=None)
    parser.add_argument("--json", default=None, help="Optional path to write result JSON.")
    return parser.parse_args()


def configure_env(args):
    plugin_path = str(Path(args.musa_plugin).resolve())
    adapter_path = str(Path(args.musa_tf_adapter).resolve())
    os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = adapter_path
    os.environ["MUSA_PJRT_PLUGIN_PATH"] = plugin_path
    os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{plugin_path}"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["MUSA_NPD_IS_PLUGGABLE_DEVICE"] = "1"
    os.environ["MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE"] = "1"
    os.environ["MUSA_PJRT_FORCE_HOST_BUFFER_COPY"] = (
        "1" if args.pjrt_force_host_copy == "on" else "0"
    )
    os.environ["MUSA_PJRT_DEBUG_LOG"] = "1" if args.pjrt_debug_log == "on" else "0"

    tf_xla_flags = os.environ.get("TF_XLA_FLAGS", "")
    tf_xla_flags = set_flag_with_prefix(
        tf_xla_flags, "--tf_xla_auto_jit=", "--tf_xla_auto_jit=2"
    )
    tf_xla_flags = append_unique_flag(tf_xla_flags, "--tf_xla_use_device_api=true")
    os.environ["TF_XLA_FLAGS"] = tf_xla_flags

    if args.xla_dump_dir:
        dump_dir = str(Path(args.xla_dump_dir).resolve())
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags = set_flag_with_prefix(
            xla_flags, "--xla_dump_to=", f"--xla_dump_to={dump_dir}"
        )
        xla_flags = append_unique_flag(xla_flags, "--xla_dump_hlo_as_text")
        os.environ["XLA_FLAGS"] = xla_flags


def load_musa_plugin(tf, adapter_path):
    path = Path(adapter_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"MUSA TensorFlow adapter not found: {path}")
    lib = ctypes.CDLL(str(path))
    if hasattr(lib, "ForceRegisterMusa"):
        lib.ForceRegisterMusa.restype = ctypes.c_int
        if lib.ForceRegisterMusa() != 1:
            raise RuntimeError("MUSA PJRT registration failed in TensorFlow adapter")
    devices = tf.config.list_physical_devices("MUSA")
    if not devices:
        raise RuntimeError("no MUSA physical devices are visible")
    return [str(device) for device in devices]


def max_abs_rel(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    abs_err = np.max(np.abs(actual - expected))
    denom = np.maximum(np.abs(expected), 1e-6)
    rel_err = np.max(np.abs(actual - expected) / denom)
    return float(abs_err), float(rel_err)


def run_log1p(tf, args, rng):
    x_np = rng.uniform(0.001, 16.0, size=(4096,)).astype(np.float32)

    @tf.function(jit_compile=True, input_signature=[tf.TensorSpec([4096], tf.float32)])
    def musa_log1p(x):
        y = tf.math.log1p(x)
        z = tf.math.log1p(x * 0.25 + 0.125)
        return y + z

    with tf.device(args.device):
        x = tf.constant(x_np)
        start = time.perf_counter()
        out = musa_log1p(x).numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    expected = np.log1p(x_np) + np.log1p(x_np * 0.25 + 0.125)
    abs_err, rel_err = max_abs_rel(out, expected)
    return {
        "case": "log1p",
        "status": "ok" if abs_err <= args.atol or rel_err <= args.rtol else "failed",
        "elapsed_ms": elapsed_ms,
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
    }


def run_conv2d(tf, args, rng):
    x_np = rng.normal(0.0, 0.25, size=(1, 16, 16, 8)).astype(np.float32)
    w_np = rng.normal(0.0, 0.25, size=(3, 3, 8, 12)).astype(np.float32)
    b_np = rng.normal(0.0, 0.05, size=(12,)).astype(np.float32)

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec([1, 16, 16, 8], tf.float32),
            tf.TensorSpec([3, 3, 8, 12], tf.float32),
            tf.TensorSpec([12], tf.float32),
        ],
    )
    def musa_conv2d(x, w, b):
        y = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        return tf.nn.bias_add(y, b)

    with tf.device(args.device):
        x = tf.constant(x_np)
        w = tf.constant(w_np)
        b = tf.constant(b_np)
        start = time.perf_counter()
        out = musa_conv2d(x, w, b).numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    with tf.device("/CPU:0"):
        expected = tf.nn.bias_add(
            tf.nn.conv2d(
                tf.constant(x_np),
                tf.constant(w_np),
                strides=[1, 1, 1, 1],
                padding="SAME",
            ),
            tf.constant(b_np),
        ).numpy()
    abs_err, rel_err = max_abs_rel(out, expected)
    return {
        "case": "conv2d",
        "status": "ok" if abs_err <= args.atol or rel_err <= args.rtol else "failed",
        "elapsed_ms": elapsed_ms,
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
    }


def run_matmul(tf, args, rng):
    x_np = rng.normal(size=(128, 128)).astype(np.float32)
    y_np = rng.normal(size=(128, 128)).astype(np.float32)

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec([128, 128], tf.float32),
            tf.TensorSpec([128, 128], tf.float32),
        ],
    )
    def musa_matmul(x, y):
        return tf.nn.relu(tf.matmul(x, y) + 1.0)

    with tf.device(args.device):
        start = time.perf_counter()
        out = musa_matmul(tf.constant(x_np), tf.constant(y_np)).numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    expected = np.maximum(np.matmul(x_np, y_np) + 1.0, 0.0)
    abs_err, rel_err = max_abs_rel(out, expected)
    return {
        "case": "matmul",
        "status": "ok" if abs_err <= args.atol or rel_err <= args.rtol else "failed",
        "elapsed_ms": elapsed_ms,
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
    }


def main():
    args = parse_args()
    configure_env(args)

    import tensorflow as tf

    devices = load_musa_plugin(tf, args.musa_tf_adapter)
    rng = np.random.default_rng(args.seed)
    cases = ["log1p", "conv2d", "matmul"] if args.case == "all" else [args.case]

    results = []
    for _ in range(max(1, args.repeat)):
        for case in cases:
            if case == "log1p":
                result = run_log1p(tf, args, rng)
            elif case == "conv2d":
                result = run_conv2d(tf, args, rng)
            elif case == "matmul":
                result = run_matmul(tf, args, rng)
            else:
                raise ValueError(case)
            results.append(result)
            print(
                "[SMOKE] {case}: status={status} elapsed_ms={elapsed_ms:.3f} "
                "shape={shape} max_abs={max_abs_error:.6g} max_rel={max_rel_error:.6g}".format(
                    **result
                )
            )

    summary = {
        "device": args.device,
        "musa_devices": devices,
        "tf_xla_flags": os.environ.get("TF_XLA_FLAGS", ""),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "results": results,
        "ok": sum(1 for item in results if item["status"] == "ok"),
        "failed": sum(1 for item in results if item["status"] != "ok"),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[SMOKE] summary=" + json.dumps(summary, ensure_ascii=False))
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
