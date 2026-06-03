#!/usr/bin/env python3
import argparse
import ctypes
from contextlib import contextmanager
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MUSA_PLUGIN_PATH = (
    REPO_ROOT / "bazel-bin" / "pjrt_plugin" / "libmusa_pjrt_plugin_zy.so"
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


def _has_cli_flag(argv, flag):
    prefix = f"{flag}="
    return flag in argv or any(arg.startswith(prefix) for arg in argv)


def _append_unique_flag(current_value: str, new_flag: str) -> str:
    tokens = current_value.split()
    if new_flag not in tokens:
        tokens.append(new_flag)
    return " ".join(tokens).strip()


def _set_flag_with_prefix(current_value: str, flag_prefix: str, new_flag: str) -> str:
    tokens = [token for token in current_value.split() if not token.startswith(flag_prefix)]
    tokens.append(new_flag)
    return " ".join(tokens).strip()


def device_kind(device: Optional[str]) -> str:
    value = (device or "").upper()
    if "MUSA" in value:
        return "MUSA"
    if "GPU" in value or "CUDA" in value:
        return "CUDA"
    return "CPU"


def default_musa_plugin_path() -> str:
    return os.environ.get("MUSA_PJRT_PLUGIN_PATH", str(DEFAULT_MUSA_PLUGIN_PATH))


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
        plugin_path = _get_cli_arg(argv, "--musa_plugin", default_musa_plugin_path())
        os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = plugin_path
        os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{plugin_path}"

        # Clear experiment leftovers, then set the stable PJRT/NPD path.
        for name in (
            "MUSA_XLA_RESPECT_NPD_ENV",
            "MUSA_XLA_GLOBAL_JIT_LEVEL",
            "MUSA_NPD_COMPILATION_DEVICE",
            "MUSA_PJRT_MAX_INFLIGHT_COMPILES",
            "MUSA_PJRT_WAIT_TRANSFER_DONE",
            "MUSA_PJRT_WAIT_EXECUTE_DONE",
            "MUSA_PJRT_SERIALIZE_EXECUTE",
            "MUSA_PJRT_DROP_EXECUTE_DEVICE",
        ):
            os.environ.pop(name, None)
        os.environ["MUSA_NPD_IS_PLUGGABLE_DEVICE"] = "1"
        os.environ["MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE"] = "1"

    if enable_dump:
        xla_dump_dir = os.path.abspath(
            os.path.expanduser(dump_dir or str(DEFAULT_XLA_DUMP_DIR))
        )
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags = _set_flag_with_prefix(
            xla_flags, "--xla_dump_to=", f"--xla_dump_to={xla_dump_dir}"
        )
        xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_text")
        xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_long_text")
        xla_flags = _set_flag_with_prefix(
            xla_flags, "--xla_dump_hlo_pass_re=", "--xla_dump_hlo_pass_re=.*"
        )
        xla_flags = _set_flag_with_prefix(
            xla_flags, "--xla_dump_max_hlo_modules=", "--xla_dump_max_hlo_modules=-1"
        )
        os.environ["XLA_FLAGS"] = xla_flags
        os.environ["GRAPH_RUNNER_XLA_DUMP_DIR"] = xla_dump_dir


configure_runtime_env_from_argv(sys.argv[1:])

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.config.run_functions_eagerly(True)


def env_flag_enabled(name):
    value = os.environ.get(name, "")
    return value in ("1", "true", "TRUE", "yes", "YES", "on", "ON")


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
            jit_level = os.environ.get("MUSA_XLA_GLOBAL_JIT_LEVEL", "").strip()
            if jit_level:
                config.graph_options.optimizer_options.global_jit_level = (
                    tf.compat.v1.OptimizerOptions.ON_1
                )
    elif args.xla:
        config.graph_options.optimizer_options.global_jit_level = (
            tf.compat.v1.OptimizerOptions.ON_1
        )
    return config


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

    txt_files = list(dump_path.rglob("*.txt"))
    after_files = sorted(dump_path.rglob("*after_optimizations*.txt"))
    opcode_counts = Counter()
    fusion_count = 0
    instruction_count = 0
    instruction_re = re.compile(r"^\s*(?:ROOT\s+)?[A-Za-z0-9_.%-]+ = ")
    opcode_re = re.compile(r"\s([A-Za-z][A-Za-z0-9_-]*)\(")
    fusion_re = re.compile(r"=\s*.*\bfusion\(")

    for path in after_files:
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
                        opcode_counts[match.group(1)] += 1
        except OSError:
            continue

    summary = {
        "dump_dir": str(dump_path.resolve()),
        "exists": True,
        "hlo_txt_files": len(txt_files),
        "after_optimizations_files": len(after_files),
        "instruction_count": instruction_count,
        "fusion_ops": fusion_count,
        "top_opcodes": opcode_counts.most_common(20),
    }
    if after_files:
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


def run_single_spec(spec_path: Path, pb_path: Path, args, bs: int, musa_loaded: bool, runner_out: Path):
    meta = load_meta(spec_path)
    input_spec = read_node_list_collection(meta, "input_spec")
    output_spec = read_node_list_collection(meta, "output_spec")
    spec_shape_map = build_spec_tensor_shape_map(meta)
    slice_min_dims = infer_placeholder_min_dims(meta)

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(pb_path.read_bytes())

    with tf.Graph().as_default() as graph:
        use_device_scope = bool(args.device) and not (
            args.xla and device_kind(args.device) == "MUSA" and not args.xla_device_scope
        )
        if use_device_scope:
            with tf.device(args.device):
                tf.import_graph_def(graph_def, name="")
        else:
            tf.import_graph_def(graph_def, name="")

        outputs = [graph.get_tensor_by_name(name) for name in output_spec]

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
            }
            for tensor in outputs
        ]

        rng = np.random.default_rng(args.seed)
        feed_dict = {}
        pinned_feed_holders = []
        use_pinned_feed = env_flag_enabled("MUSA_PINNED_FEED")
        for name in input_spec:
            tensor = graph.get_tensor_by_name(name)
            tensor_shape = safe_shape(tensor)
            merged_shape = merge_shape(spec_shape_map.get(name), tensor_shape)
            if name in slice_min_dims:
                merged_shape = list(merged_shape)
                for dim_index, min_dim in slice_min_dims[name].items():
                    if dim_index < len(merged_shape):
                        current = merged_shape[dim_index]
                        if current is None or current < min_dim:
                            merged_shape[dim_index] = min_dim
            run_shape = resolve_shape(merged_shape, bs, args.unknown_dim)
            if tensor.op.type in ("Placeholder", "PlaceholderWithDefault"):
                if use_pinned_feed:
                    value = pinned_random_array(
                        run_shape, tensor.dtype.as_numpy_dtype, rng, pinned_feed_holders
                    )
                else:
                    value = random_array(run_shape, tensor.dtype.as_numpy_dtype, rng)
                feed_dict[tensor] = value

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
        with configured_graph_dump_dir(default_dump_dir) as active_dump_dir:
            if active_dump_dir is not None:
                graph_dump["dump_dir"] = str(active_dump_dir)
            with tf.compat.v1.Session(graph=graph, config=config) as sess:
                try:
                    for _ in range(max(0, args.warmup)):
                        sess.run(outputs, feed_dict=feed_dict)
                    for _ in range(max(1, args.run_iters)):
                        start = time.perf_counter()
                        last_vals = sess.run(outputs, feed_dict=feed_dict)
                        end = time.perf_counter()
                        lat_ms.append((end - start) * 1000.0)
                except Exception:
                    run_error = traceback.format_exc()

        if graph_dump["dump_dir"]:
            graph_dump["files"] = collect_graph_dump_files(graph_dump["dump_dir"])

    result_shapes = []
    if last_vals is not None:
        result_shapes = [
            {
                "shape": list(getattr(value, "shape", [])),
                "dtype": str(getattr(value, "dtype", "")),
            }
            for value in last_vals
        ]

    return {
        "spec_path": str(spec_path),
        "pb_path": str(pb_path),
        "batch_size": bs,
        "num_outputs": len(output_info),
        "outputs": output_info,
        "result_shapes": result_shapes,
        "status": "ok" if run_error is None else "failed",
        "error_core": extract_core_error(run_error),
        "error": run_error,
        "graph_dump": graph_dump,
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
        print("[INFO] MUSA plugin loading skipped.")
        return False

    if args.xla:
        plugin_path = Path(args.musa_plugin).resolve()
        if not plugin_path.exists():
            raise FileNotFoundError(f"MUSA PJRT plugin not found: {plugin_path}")
        lib = ctypes.CDLL(str(plugin_path))
        if hasattr(lib, "ForceRegisterMusa"):
            lib.ForceRegisterMusa()
        print(f">>>> [MUSA/XLA] PJRT plugin loaded: {plugin_path}")
        return True

    try:
        import tensorflow_musa  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import tensorflow_musa. Install tensorflow_musa for "
            "non-XLA MUSA runs or use --xla with --musa_plugin."
        ) from exc
    print(">>>> [MUSA] Plugin loaded by importing tensorflow_musa")
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
    parser.add_argument("--bs", default="1024", help="Batch size or comma list, e.g. 1,8,32.")
    parser.add_argument("--unknown_dim", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--run_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_root", default="runner_out")
    parser.add_argument("--strict", type=parse_bool, default=True)
    parser.add_argument(
        "--device",
        default="/device:MUSA:0",
        help="TensorFlow device scope, e.g. /device:MUSA:0, /device:CPU:0.",
    )
    parser.add_argument("--allow_soft_placement", type=parse_bool, default=True)
    parser.add_argument("--log_device_placement", type=parse_bool, default=False)
    parser.add_argument("--musa_optimizer", type=parse_bool, default=True)
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
        "--musa_plugin",
        default=default_musa_plugin_path(),
        help="Path to libmusa_pjrt_plugin_zy.so for MUSA XLA runs.",
    )
    return parser.parse_args()


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

    bs_values = parse_bs_values(args.bs)
    convert_script = Path(args.convert_script).resolve()
    if not convert_script.exists():
        raise FileNotFoundError(f"convert script not found: {convert_script}")

    print("[INFO] Runtime configuration")
    print(f"  device={args.device}")
    print(f"  xla={args.xla}")
    print(f"  TF_XLA_FLAGS={os.environ.get('TF_XLA_FLAGS', '')}")
    print(f"  XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
    if device_kind(args.device) == "MUSA":
        print(f"  musa_plugin={args.musa_plugin}")
        print(f"  PJRT_NAMES_AND_LIBRARY_PATHS={os.environ.get('PJRT_NAMES_AND_LIBRARY_PATHS', '')}")

    musa_loaded = load_runtime_plugins(args)
    specs = collect_specs(args.spec, args.spec_dir)
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
            avg = (report.get("timing_ms") or {}).get("average")
            trimmed = (report.get("timing_ms") or {}).get("trimmed_avg")
            print(
                f"[INFO] run done: spec={spec_path.name} bs={bs} "
                f"status={report['status']} average_time_ms={avg} "
                f"trimmed_avg_ms={trimmed}"
            )
            if report.get("error_core"):
                print(f"[INFO] core error: {report['error_core']}")

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
            "PJRT_NAMES_AND_LIBRARY_PATHS": os.environ.get(
                "PJRT_NAMES_AND_LIBRARY_PATHS", ""
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
