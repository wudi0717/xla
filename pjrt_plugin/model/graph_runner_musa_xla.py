import os
import sys
import time
import argparse
import ctypes
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional

# ==========================================
# 配置区域
# ==========================================
model_path = "./graph_def.pb"
output_node_name = "predicts"  # 输出节点名称
musa_plugin_path = "/workspace/xla_ref/xla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"
DEFAULT_XLA_DUMP_DIR = "/workspace/xla_ref/xla/pjrt_plugin/model/graph_runner_musa_xla_dump"

def _get_cli_arg_value(argv, flag, default=None):
    if flag not in argv:
        return default
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def _append_unique_flag(current_value: str, new_flag: str) -> str:
    tokens = current_value.split()
    if new_flag not in tokens:
        tokens.append(new_flag)
    return " ".join(tokens).strip()


def _set_flag_with_prefix(current_value: str, flag_prefix: str, new_flag: str) -> str:
    tokens = [token for token in current_value.split() if not token.startswith(flag_prefix)]
    tokens.append(new_flag)
    return " ".join(tokens).strip()


def _resolve_xla_dump_dir_from_argv(argv) -> Optional[str]:
    dump_dir = _get_cli_arg_value(argv, "--xla_dump_dir")
    enable_dump = "--xla_dump" in argv or dump_dir is not None
    if not enable_dump:
        return None
    return os.path.abspath(os.path.expanduser(dump_dir or DEFAULT_XLA_DUMP_DIR))

def _early_device_arg():
    for index, arg in enumerate(sys.argv):
        if arg == "--device" and index + 1 < len(sys.argv):
            return sys.argv[index + 1].lower()
        if arg.startswith("--device="):
            return arg.split("=", 1)[1].lower()
    return "cpu"

def _early_batch_size_arg():
    value = _get_cli_arg_value(sys.argv[1:], "--batchsize")
    if value is None:
        for arg in sys.argv[1:]:
            if arg.startswith("--batchsize="):
                value = arg.split("=", 1)[1]
                break
    if value is None:
        return 100
    try:
        return int(value)
    except ValueError:
        return 100

def _early_large_batch_threshold():
    value = os.environ.get("MUSA_XLA_LARGE_BATCH_THRESHOLD", "512")
    try:
        return int(value)
    except ValueError:
        return 512

use_musa_xla = "--xla" in sys.argv and _early_device_arg() == "musa"

if use_musa_xla:
    early_batch_size = _early_batch_size_arg()
    # Stable MUSA XLA path: let TensorFlow's XLA bridge auto-cluster and route
    # the compiled clusters through the NextPluggableDevice/PJRT plugin.
    os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = musa_plugin_path
    os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{musa_plugin_path}"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    # This combination produced one large cluster_0 HLO module instead of 283
    # tiny on-demand modules. Force it for the normal --device musa --xla path
    # so stale shell exports from experiments cannot silently change behavior.
    os.environ.pop("MUSA_XLA_RESPECT_NPD_ENV", None)
    os.environ.pop("MUSA_XLA_GLOBAL_JIT_LEVEL", None)
    os.environ.pop("MUSA_NPD_COMPILATION_DEVICE", None)
    os.environ.pop("MUSA_PJRT_MAX_INFLIGHT_COMPILES", None)
    os.environ.pop("MUSA_PJRT_MAX_INFLIGHT_TRANSFERS", None)
    os.environ.pop("MUSA_PJRT_MAX_INFLIGHT_EXECUTES", None)
    os.environ.pop("MUSA_PJRT_WAIT_TRANSFER_DONE", None)
    os.environ.pop("MUSA_PJRT_WAIT_EXECUTE_DONE", None)
    os.environ.pop("MUSA_PJRT_SERIALIZE_EXECUTE", None)
    os.environ.pop("MUSA_PJRT_DROP_EXECUTE_DEVICE", None)
    os.environ.pop("MUSA_PJRT_FORCE_HOST_BUFFER_COPY", None)
    os.environ.pop("MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT", None)
    os.environ["MUSA_NPD_IS_PLUGGABLE_DEVICE"] = "1"
    os.environ["MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE"] = "1"
    if early_batch_size >= _early_large_batch_threshold():
        os.environ["MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT"] = "1"
    else:
        # The stable batch=100 path uses fully asynchronous PJRT submission.
        # Keeping these unset falls back to serialized proxy gates and produces
        # very large tail-latency spikes.
        os.environ["MUSA_PJRT_FORCE_HOST_BUFFER_COPY"] = "0"
        os.environ["MUSA_PJRT_MAX_INFLIGHT_TRANSFERS"] = "0"
        os.environ["MUSA_PJRT_MAX_INFLIGHT_EXECUTES"] = "0"
    tf_xla_flags = os.environ.get("TF_XLA_FLAGS", "")
    required_xla_flags = [
        "--tf_xla_auto_jit=2",
        "--tf_xla_use_device_api=true",
    ]
    for flag in required_xla_flags:
        if flag not in tf_xla_flags.split():
            tf_xla_flags = (tf_xla_flags + " " + flag).strip()
    os.environ["TF_XLA_FLAGS"] = tf_xla_flags
elif _early_device_arg() == "musa":
    if os.environ.get("TF_PLUGGABLE_DEVICE_LIBRARY_PATH") == musa_plugin_path:
        os.environ.pop("TF_PLUGGABLE_DEVICE_LIBRARY_PATH", None)
    if os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS") == f"MUSA:{musa_plugin_path}":
        os.environ.pop("PJRT_NAMES_AND_LIBRARY_PATHS", None)
    tf_xla_flags = os.environ.get("TF_XLA_FLAGS", "")
    stripped_flags = [
        flag for flag in tf_xla_flags.split()
        if not flag.startswith("--tf_xla_auto_jit=")
        and flag != "--tf_xla_use_device_api=true"
    ]
    if stripped_flags:
        os.environ["TF_XLA_FLAGS"] = " ".join(stripped_flags)
    else:
        os.environ.pop("TF_XLA_FLAGS", None)

xla_dump_dir = _resolve_xla_dump_dir_from_argv(sys.argv[1:])
if xla_dump_dir:
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags = _set_flag_with_prefix(
        xla_flags,
        "--xla_dump_to=",
        f"--xla_dump_to={xla_dump_dir}",
    )
    xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_text")
    xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_long_text")
    xla_flags = _set_flag_with_prefix(
        xla_flags,
        "--xla_dump_hlo_pass_re=",
        "--xla_dump_hlo_pass_re=.*",
    )
    xla_flags = _set_flag_with_prefix(
        xla_flags,
        "--xla_dump_max_hlo_modules=",
        "--xla_dump_max_hlo_modules=-1",
    )
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["GRAPH_RUNNER_XLA_DUMP_DIR"] = xla_dump_dir

import numpy as np
import tensorflow as tf2
from tensorflow.core.framework import graph_pb2

# 禁用 V2 行为，确保 TF1 图能正常运行
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


# ==========================================
# 1. 加载 MUSA 插件，配置config
# ==========================================
def load_musa_plugin():
    print("\n====================================================")
    print("[PYTHON] Force registering MUSA PJRT plugin...")
    if os.path.exists(musa_plugin_path):
        try:
            lib = ctypes.CDLL(musa_plugin_path)
            lib.ForceRegisterMusa()
            print(f">>>> [MUSA] PJRT plugin hook executed successfully from: {musa_plugin_path}")
            devices = tf2.config.list_physical_devices("MUSA")
            if devices:
                print(f">>>> [MUSA] Physical devices: {devices}")
            else:
                print(">>>> [MUSA] Hook executed, but TensorFlow did not list MUSA devices.")
        except Exception as e:
            print(f"!!!! [MUSA] Failed to register PJRT plugin: {e}")
            sys.exit(1)
    else:
        print(f"!!!! [MUSA] Plugin not found at {musa_plugin_path}")
        sys.exit(1)
    print("====================================================\n")


def prepare_xla_dump_dir(dump_dir: Optional[str]):
    if not dump_dir:
        return
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    print(f"[XLA Dump] 输出目录: {dump_dir}")
    print(f"[XLA Dump] XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")


def _percentile(values: List[int], percentile: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * percentile / 100.0))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _parse_hlo_text_stats(hlo_text_path: Path) -> Dict[str, Any]:
    stats = {
        "file": hlo_text_path.name,
        "instruction_count": 0,
        "parameter_count": 0,
        "fusion_count": 0,
        "root_count": 0,
        "opcode_counts": Counter(),
    }
    instruction_re = re.compile(r"^\s*(?:ROOT\s+)?[A-Za-z0-9_.%-]+ = ")
    opcode_re = re.compile(r"\s([A-Za-z][A-Za-z0-9_-]*)\(")
    fusion_opcode = re.compile(r"=\s*.*\bfusion\(")

    try:
        with hlo_text_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not instruction_re.search(line):
                    continue
                stats["instruction_count"] += 1
                if line.lstrip().startswith("ROOT "):
                    stats["root_count"] += 1
                if " parameter(" in line:
                    stats["parameter_count"] += 1
                if fusion_opcode.search(line):
                    stats["fusion_count"] += 1

                rhs = line.split(" = ", 1)[1]
                opcode_match = opcode_re.search(rhs)
                if opcode_match:
                    stats["opcode_counts"][opcode_match.group(1)] += 1
    except OSError:
        pass
    return stats


def _count_fusion_ops(hlo_text_path: Path) -> int:
    return int(_parse_hlo_text_stats(hlo_text_path)["fusion_count"])


def summarize_xla_dump_dir(dump_dir: Optional[str]):
    if not dump_dir:
        return
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        print(f"[XLA Dump] directory does not exist: {dump_dir}")
        return

    txt_files = list(dump_path.rglob("*.txt"))
    after_files = sorted(dump_path.rglob("*after_optimizations*.txt"))
    module_stats = [_parse_hlo_text_stats(path) for path in after_files]
    fusion_by_file = [
        (path, int(stats["fusion_count"]))
        for path, stats in zip(after_files, module_stats)
    ]
    total_fusions = sum(int(stats["fusion_count"]) for stats in module_stats)
    files_with_fusion = [(path, count) for path, count in fusion_by_file if count > 0]
    instruction_counts = [int(stats["instruction_count"]) for stats in module_stats]
    parameter_counts = [int(stats["parameter_count"]) for stats in module_stats]
    opcode_counts = Counter()
    for stats in module_stats:
        opcode_counts.update(stats["opcode_counts"])

    small_modules = sum(1 for count in instruction_counts if count <= 10)
    zero_fusion_modules = sum(
        1 for stats in module_stats if int(stats["fusion_count"]) == 0
    )
    largest_modules = sorted(
        [
            {
                "file": stats["file"],
                "instruction_count": int(stats["instruction_count"]),
                "parameter_count": int(stats["parameter_count"]),
                "fusion_count": int(stats["fusion_count"]),
            }
            for stats in module_stats
        ],
        key=lambda item: item["instruction_count"],
        reverse=True,
    )

    summary = {
        "dump_dir": str(dump_path),
        "hlo_txt_files": len(txt_files),
        "after_optimizations_files": len(after_files),
        "fusion_ops": total_fusions,
        "modules_with_fusion": len(files_with_fusion),
        "zero_fusion_modules": zero_fusion_modules,
        "small_modules_le_10_instr": small_modules,
        "instruction_count": {
            "total": sum(instruction_counts),
            "min": min(instruction_counts) if instruction_counts else 0,
            "p50": _percentile(instruction_counts, 50),
            "p95": _percentile(instruction_counts, 95),
            "max": max(instruction_counts) if instruction_counts else 0,
        },
        "parameter_count": {
            "total": sum(parameter_counts),
            "min": min(parameter_counts) if parameter_counts else 0,
            "p50": _percentile(parameter_counts, 50),
            "p95": _percentile(parameter_counts, 95),
            "max": max(parameter_counts) if parameter_counts else 0,
        },
        "top_opcodes": opcode_counts.most_common(20),
        "top_fusion_files": [
            {"file": path.name, "fusion_count": count}
            for path, count in sorted(
                fusion_by_file, key=lambda item: item[1], reverse=True
            )[:20]
        ],
        "largest_modules": largest_modules[:20],
    }

    if after_files:
        summary_path = dump_path / "xla_dump_summary.json"
        try:
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            print(f"[XLA Dump] failed to write JSON summary: {exc}")

    print("\n[XLA Dump Summary]")
    print(f"  Directory: {dump_dir}")
    print(f"  HLO txt files: {len(txt_files)}")
    print(f"  after_optimizations files: {len(after_files)}")
    print(f"  fusion ops: {total_fusions}")
    print(f"  modules with fusion: {len(files_with_fusion)}")
    print(f"  modules without fusion: {zero_fusion_modules}")
    print(f"  small modules (<=10 HLO instr): {small_modules}")
    if instruction_counts:
        instr = summary["instruction_count"]
        params = summary["parameter_count"]
        print(
            "  HLO instr/module: "
            f"total={instr['total']} min={instr['min']} p50={instr['p50']} "
            f"p95={instr['p95']} max={instr['max']}"
        )
        print(
            "  parameters/module: "
            f"total={params['total']} min={params['min']} p50={params['p50']} "
            f"p95={params['p95']} max={params['max']}"
        )

    if opcode_counts:
        print("  top HLO opcodes:")
        for opcode, count in opcode_counts.most_common(12):
            print(f"    {count:5d}  {opcode}")

    if fusion_by_file:
        print("  top fusion modules:")
        for path, count in sorted(
            fusion_by_file, key=lambda item: item[1], reverse=True
        )[:10]:
            print(f"    {count:4d}  {path.name}")
        print("  largest modules by HLO instruction count:")
        for item in largest_modules[:10]:
            print(
                f"    instr={item['instruction_count']:4d} "
                f"params={item['parameter_count']:3d} "
                f"fusion={item['fusion_count']:2d}  {item['file']}"
            )
        print(f"  JSON summary: {dump_path / 'xla_dump_summary.json'}")
    else:
        print("  No HLO dump found. This usually means XLA dump flags did not reach the actual PJRT compile options.")
    return
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        print(f"[XLA Dump] 目录不存在: {dump_dir}")
        return

    txt_files = list(dump_path.rglob("*.txt"))
    after_files = sorted(dump_path.rglob("*after_optimizations*.txt"))
    fusion_by_file = [(path, _count_fusion_ops(path)) for path in after_files]
    total_fusions = sum(count for _, count in fusion_by_file)
    files_with_fusion = [(path, count) for path, count in fusion_by_file if count > 0]

    print("\n[XLA Dump 汇总]")
    print(f"  目录: {dump_dir}")
    print(f"  HLO txt 文件数: {len(txt_files)}")
    print(f"  after_optimizations 文件数: {len(after_files)}")
    print(f"  fusion op 总数: {total_fusions}")
    print(f"  包含 fusion 的模块数: {len(files_with_fusion)}")

    if fusion_by_file:
        print("  fusion top 文件:")
        for path, count in sorted(fusion_by_file, key=lambda item: item[1], reverse=True)[:10]:
            print(f"    {count:4d}  {path.name}")
    else:
        print("  未发现 HLO dump；这通常表示 XLA_FLAGS 没有进入实际 PJRT compile options，而不是 fusion 个数为 0。")


def _resolve_musa_global_jit_level() -> Optional[int]:
    value = os.environ.get("MUSA_XLA_GLOBAL_JIT_LEVEL", "").strip().upper()
    if not value or value in ("OFF", "0", "FALSE", "NO"):
        return None
    if value in ("ON_1", "ON1", "1", "TRUE", "YES"):
        return tf.OptimizerOptions.ON_1
    if value in ("ON_2", "ON2", "2"):
        return getattr(tf.OptimizerOptions, "ON_2", tf.OptimizerOptions.ON_1)
    print(f"[MUSA/XLA] Ignore invalid MUSA_XLA_GLOBAL_JIT_LEVEL={value!r}")
    return None


def create_session_config(
    device_type: str = "cpu",
    xla: bool = False,
    log_device_placement: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tf.ConfigProto:
    """Create a Session config with device and XLA settings."""
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = log_device_placement

    device_type_upper = (device_type or "cpu").upper()

    if device_type_upper == "CUDA":
        config.gpu_options.allow_growth = True
        # XLA 仅在 CUDA 设备下有效
        if xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            if logger is not None:
                logger.info("Enabled XLA JIT compilation for CUDA")

    elif device_type_upper == "MUSA":
        rewrite_options = config.graph_options.rewrite_options
        rewrite_options.custom_optimizers.add().name = "musa_graph_optimizer"
        if logger is not None:
            logger.info("Enabled custom optimizer: musa_graph_optimizer")
        if xla:
            # TF_XLA_FLAGS drives auto clustering for MUSA/PJRT.
            rewrite_options.min_graph_nodes = -1
            global_jit_level = _resolve_musa_global_jit_level()
            if global_jit_level is not None:
                config.graph_options.optimizer_options.global_jit_level = global_jit_level
                print(f"[MUSA/XLA] global_jit_level={os.environ.get('MUSA_XLA_GLOBAL_JIT_LEVEL')}")
            if logger is not None:
                logger.info("Enabled XLA JIT compilation for MUSA")

    return config


# ==========================================
# 2. 辅助函数：推断 Placeholder 形状
# ==========================================
def infer_placeholder_shape_from_usage(graph_def, placeholder_name):
    """
    通过分析图中使用该 Placeholder 的节点来推断其形状
    """
    for node in graph_def.node:
        for input_name in node.input:
            clean_input = input_name.split(":")[0].lstrip("^")
            if clean_input == placeholder_name:
                if node.op == "MatMul" or node.op == "Tensordot":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) == 2:
                                # 简单推断逻辑
                                return [64, 32]  # 默认权重矩阵大小
                elif node.op == "BiasAdd":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) >= 1:
                                return [output_shape.dim[-1].size]
    return None


def load_graph_and_get_placeholders(pb_path):
    """
    加载图并获取所有 placeholder 节点信息
    """
    print(f"\n=== 加载图文件: {pb_path} ===")

    if not os.path.exists(pb_path):
        print(f"错误: 文件 {pb_path} 不存在!")
        sys.exit(1)

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

    print(f"图加载成功，总节点数: {len(graph_def.node)}")

    placeholders = {}
    for node in graph_def.node:
        if node.op == "Placeholder":
            # 获取数据类型
            dtype_enum = node.attr["dtype"].type
            dtype_map = {
                tf.float32.as_datatype_enum: np.float32,
                tf.int32.as_datatype_enum: np.int32,
                tf.int64.as_datatype_enum: np.int64,
                tf.bool.as_datatype_enum: np.bool_,
                tf.string.as_datatype_enum: np.str_,
            }
            dtype = dtype_map.get(dtype_enum, np.float32)

            # 获取形状
            shape = []
            shape_found = False

            if "shape" in node.attr:
                shape_proto = node.attr["shape"].shape
                if not shape_proto.unknown_rank:
                    for dim in shape_proto.dim:
                        shape.append(dim.size if dim.size != -1 else None)
                    shape_found = True

            if not shape_found and "_output_shapes" in node.attr:
                output_shapes = node.attr["_output_shapes"].list.shape
                if len(output_shapes) > 0:
                    shape_proto = output_shapes[0]
                    if not shape_proto.unknown_rank:
                        for dim in shape_proto.dim:
                            shape.append(dim.size if dim.size != -1 else None)
                        shape_found = True

            # 兜底推断
            if not shape_found:
                inferred = infer_placeholder_shape_from_usage(graph_def, node.name)
                shape = inferred if inferred else []

            placeholders[node.name] = {"dtype": dtype, "shape": shape}

    print(f"找到 {len(placeholders)} 个 Placeholder 节点")
    return graph_def, placeholders


# ==========================================
# 3. 创建 Mock 数据 (含关键修复)
# ==========================================
def create_mock_data(placeholders, batch_size):
    """
    根据 placeholder 信息创建 mock 数据
    """
    print("\n=== 创建 Mock 数据 ===")

    feed_dict = {}

    for name, info in placeholders.items():
        shape = info["shape"]
        dtype = info["dtype"]

        # 处理形状，将 None 替换为 batch_size
        mock_shape = []
        for dim in shape:
            if dim is None:
                mock_shape.append(batch_size)
            elif dim == 0:
                # 维度为0的情况，保持为0
                mock_shape.append(0)
            else:
                mock_shape.append(dim)

        # 如果形状为空列表，说明是标量，但某些 Placeholder 可能需要特定形状
        # 检查名称中是否包含特定模式来推断正确的形状
        if not mock_shape:
            # 对于 ReadVariableOp/resource 类型的 Placeholder，尝试从名称推断形状
            if "/ReadVariableOp/resource" in name:
                # 这些通常是权重或偏置参数，需要根据上下文推断形状
                # 暂时使用一个合理的默认形状
                if "BiasAdd" in name:
                    # 偏置通常是一维向量
                    mock_shape = [32]  # 默认偏置大小
                elif "MatMul" in name or "Tensordot" in name:
                    # 权重矩阵通常是二维
                    mock_shape = [64, 32]  # 默认权重矩阵大小
                else:
                    # 其他情况使用标量
                    mock_shape = []
            else:
                # 其他标量 Placeholder 保持标量
                mock_shape = []

        # 生成 mock 数据
        if dtype == np.float32:
            # 生成随机浮点数据
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(dtype)
        elif dtype == np.int32:
            # 生成随机整数数据
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.int64:
            # 生成随机长整数数据
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.bool_:
            # 生成随机布尔数据
            mock_data = np.random.choice([True, False], mock_shape).astype(dtype)
        else:
            # 默认生成浮点数据
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(np.float32)

        feed_dict[name + ":0"] = mock_data

        print(f"Mock 数据 - {name}:")
        print(f"  形状: {mock_shape}")
        print(f"  数据类型: {dtype}")
        print(f"  数据范围: [{np.min(mock_data):.4f}, {np.max(mock_data):.4f}]")

    return feed_dict


def summarize_feed_data(feed_dict, warmup_runs, num_runs):
    if not feed_dict:
        return
    total_bytes = sum(getattr(data, "nbytes", 0) for data in feed_dict.values())
    total_runs = warmup_runs + num_runs
    print("\n=== Feed 输入规模 ===")
    print(f"  输入 Tensor 数: {len(feed_dict)}")
    print(f"  每轮输入:       {total_bytes / (1024 ** 2):.2f} MiB")
    print(f"  预热+正式下限: {(total_bytes * total_runs) / (1024 ** 3):.2f} GiB")
    top_inputs = sorted(
        ((name, getattr(data, "nbytes", 0), getattr(data, "shape", None))
         for name, data in feed_dict.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    print("  最大输入 Top5:")
    for name, size, shape in top_inputs:
        print(f"    {size / (1024 ** 2):8.2f} MiB  shape={shape}  {name}")


# ==========================================
# 4. 执行推理
# ==========================================
def run_inference(graph_def, feed_dict, output_node_name, device="cpu", xla=False, num_runs=100, warmup_runs=10):
    print(f"\n=== 执行图推理 ===")
    print(f"输出节点: {output_node_name}")
    print(f"设备: {device.upper()}")
    if device.lower() in ("cuda", "musa"):
        print(f"XLA: {xla}")
    if xla and device.lower() == "musa":
        print(f"TF_XLA_FLAGS: {os.environ.get('TF_XLA_FLAGS', '')}")
        print(f"TF_PLUGGABLE_DEVICE_LIBRARY_PATH: {os.environ.get('TF_PLUGGABLE_DEVICE_LIBRARY_PATH', '')}")
        print(f"PJRT_NAMES_AND_LIBRARY_PATHS: {os.environ.get('PJRT_NAMES_AND_LIBRARY_PATHS', '')}")
        print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', '')}")
        print(f"MUSA_XLA_GLOBAL_JIT_LEVEL: {os.environ.get('MUSA_XLA_GLOBAL_JIT_LEVEL', '')}")
        print(f"MUSA_NPD_COMPILATION_DEVICE: {os.environ.get('MUSA_NPD_COMPILATION_DEVICE', '')}")
        print(f"MUSA_NPD_IS_PLUGGABLE_DEVICE: {os.environ.get('MUSA_NPD_IS_PLUGGABLE_DEVICE', '')}")
        print(f"MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE: {os.environ.get('MUSA_NPD_USE_PJRT_ON_DEMAND_COMPILE', '')}")
        print(f"MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT: {os.environ.get('MUSA_XLA_AVOID_INTERLEAVED_BATCH_GEMM_LAYOUT', '')}")
        print(f"MUSA_PJRT_FORCE_HOST_BUFFER_COPY: {os.environ.get('MUSA_PJRT_FORCE_HOST_BUFFER_COPY', '')}")
        print(f"MUSA_PJRT_MAX_INFLIGHT_TRANSFERS: {os.environ.get('MUSA_PJRT_MAX_INFLIGHT_TRANSFERS', '')}")
        print(f"MUSA_PJRT_MAX_INFLIGHT_EXECUTES: {os.environ.get('MUSA_PJRT_MAX_INFLIGHT_EXECUTES', '')}")
        print(f"预热次数: {warmup_runs}, 正式运行次数: {num_runs}")

    device_name = f"/device:{device.upper()}:0" if device.lower() != "cpu" else "/device:CPU:0"

    with tf.Graph().as_default() as graph:
        # 测量图导入时间
        t_import_start = time.time()
        if xla and device.lower() == "musa":
            print(">>>> [MUSA/XLA] Import graph with TensorFlow auto_jit")
            tf.import_graph_def(graph_def, name="")
        else:
            tf.import_graph_def(graph_def, name="")
        t_import_end = time.time()
        print(f"[时间] 图导入耗时: {(t_import_end - t_import_start)*1000:.2f} ms")

        # 准备 Session Feed
        t_feed_start = time.time()
        session_feed_dict = {}
        for name, data in feed_dict.items():
            try:
                tensor = graph.get_tensor_by_name(name)
                session_feed_dict[tensor] = data
            except KeyError:
                pass

        try:
            output_tensor = graph.get_tensor_by_name(f"{output_node_name}:0")
        except KeyError:
            print(f"Error: cannot find output tensor {output_node_name}:0")
            return None
        t_feed_end = time.time()
        print(f"[时间] Feed Dict 准备耗时: {(t_feed_end - t_feed_start)*1000:.2f} ms")

        # 配置 Session
        config = create_session_config(device_type=device, xla=xla)

        # 测量 Session 创建时间
        t_sess_start = time.time()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            t_sess_end = time.time()
            print(f"[时间] Session 创建耗时: {(t_sess_end - t_sess_start)*1000:.2f} ms")

            try:
                # 预热运行
                print(f">>> 预热运行 {warmup_runs} 次...")
                for _ in range(warmup_runs):
                    _ = sess.run(output_tensor, feed_dict=session_feed_dict)
                print(">>> 预热完成")

                # 正式测量
                print(f">>> 正式运行 {num_runs} 次...")
                run_times = []
                for i in range(num_runs):
                    t_run_start = time.time()
                    result = sess.run(output_tensor, feed_dict=session_feed_dict)
                    t_run_end = time.time()
                    run_times.append((t_run_end - t_run_start) * 1000)  # ms

                # 性能统计
                total_time = sum(run_times)
                avg_time = total_time / num_runs
                min_time = min(run_times)
                max_time = max(run_times)
                p50 = np.percentile(run_times, 50)
                p95 = np.percentile(run_times, 95)
                p99 = np.percentile(run_times, 99)

                print("\n" + "="*50)
                print("[性能统计]")
                print("="*50)
                print(f"  运行次数: {num_runs}")
                print(f"  总耗时:   {total_time:.2f} ms")
                print(f"  平均:     {avg_time:.4f} ms")
                print(f"  最小:     {min_time:.4f} ms")
                print(f"  最大:     {max_time:.4f} ms")
                print(f"  P50:      {p50:.4f} ms")
                print(f"  P95:      {p95:.4f} ms")
                print(f"  P99:      {p99:.4f} ms")
                print(f"  吞吐量:   {1000/avg_time:.2f} 次/秒")
                print("="*50)

                print(f"\n[推理结果统计]")
                print(f"  Shape: {result.shape}")
                print(f"  Dtype: {result.dtype}")
                print(f"  Min:   {np.min(result):.4f}")
                print(f"  Max:   {np.max(result):.4f}")
                print(f"  Mean:  {np.mean(result):.4f}")

                if result.size <= 20:
                    print(f"  Data: {result}")

                return result

            except Exception as e:
                print(f"\n!!!! 推理失败 !!!!")
                print(f"错误信息: {e}")
                import traceback
                traceback.print_exc()
                return None


# ==========================================
# 解析命令行参数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="TensorFlow 图推理性能测试工具")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "musa"],
        help="运行设备: cpu, cuda, musa (默认: cpu)"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=100,
        help="输入数据的 batch size (默认: 100)"
    )
    parser.add_argument(
        "--xla",
        action="store_true",
        help="启用 XLA 加速 (device=cuda/musa 时有效)"
    )
    parser.add_argument(
        "--xla_dump",
        action="store_true",
        help=f"Enable XLA dump, default output dir: {DEFAULT_XLA_DUMP_DIR}"
    )
    parser.add_argument(
        "--xla_dump_dir",
        type=str,
        default=None,
        help="Override XLA dump output dir"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1000,
        help="正式运行次数 (默认: 1000)"
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
        help="预热运行次数 (默认: 10)"
    )
    return parser.parse_args()


# ==========================================
# 主函数
# ==========================================
def main():
    args = parse_args()
    dump_dir = os.environ.get("GRAPH_RUNNER_XLA_DUMP_DIR")

    print("="*50)
    print("参数配置")
    print("="*50)
    print(f"  设备:     {args.device.upper()}")
    print(f"  Batch Size: {args.batchsize}")
    print(f"  XLA:      {args.xla if args.device in ('cuda', 'musa') else 'N/A'}")
    print(f"  XLA Dump: {dump_dir if dump_dir else 'OFF'}")
    print(f"  运行次数: {args.num_runs}")
    print(f"  预热次数: {args.warmup_runs}")
    print("="*50)

    total_start = time.time()
    prepare_xla_dump_dir(dump_dir)

    # 仅当 device=musa 时加载 MUSA 插件
    if args.device.lower() == "musa" and args.xla:
        load_musa_plugin()

    # 1. 分析图
    t0 = time.time()
    graph_def, placeholders = load_graph_and_get_placeholders(model_path)
    print(f"[时间] 图加载与分析耗时: {(time.time() - t0)*1000:.2f} ms")
    if not placeholders:
        print("错误: 未找到 Placeholder")
        return

    # 2. 造数据 (含自动修复)
    t1 = time.time()
    feed_dict = create_mock_data(placeholders, args.batchsize)
    print(f"[时间] Mock 数据创建耗时: {(time.time() - t1)*1000:.2f} ms")
    summarize_feed_data(feed_dict, args.warmup_runs, args.num_runs)

    # 3. 跑推理
    run_inference(
        graph_def,
        feed_dict,
        output_node_name,
        device=args.device,
        xla=args.xla,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )

    total_end = time.time()
    summarize_xla_dump_dir(dump_dir)
    print(f"\n[总耗时] {(total_end - total_start)*1000:.2f} ms")


if __name__ == "__main__":
    main()
