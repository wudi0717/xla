import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

MUSA_ROOT = "/usr/local/musa"
MUSA_DEVICE_LIB_PATH = f"{MUSA_ROOT}/mtgpu/bitcode"
DEFAULT_XLA_DUMP_DIR = "/workspace/xla_ref/xla/pjrt_plugin/model/graph_runner_xla_dump"


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


def _prepend_unique_path(env_name: str, prefix: str):
    current = os.environ.get(env_name, "")
    paths = [path for path in current.split(":") if path]
    if prefix in paths:
        paths = [path for path in paths if path != prefix]
    paths.insert(0, prefix)
    os.environ[env_name] = ":".join(paths)


def _resolve_xla_dump_dir_from_argv(argv) -> Optional[str]:
    dump_dir = _get_cli_arg_value(argv, "--xla_dump_dir")
    enable_dump = "--xla_dump" in argv or dump_dir is not None
    if not enable_dump:
        return None
    return os.path.abspath(os.path.expanduser(dump_dir or DEFAULT_XLA_DUMP_DIR))


def _configure_runtime_env_from_argv(argv):
    device = (_get_cli_arg_value(argv, "--device", "cpu") or "cpu").lower()
    enable_xla = "--xla" in argv and device in {"cuda", "musa"}
    xla_dump_dir = _resolve_xla_dump_dir_from_argv(argv)

    if device == "musa":
        os.environ.setdefault("MUSA_HOME", MUSA_ROOT)
        os.environ.setdefault("MUSA_PATH", MUSA_ROOT)
        os.environ.setdefault("MUSA_DEVICE_LIB_PATH", MUSA_DEVICE_LIB_PATH)
        _prepend_unique_path("PATH", f"{MUSA_ROOT}/bin")
        _prepend_unique_path("LD_LIBRARY_PATH", MUSA_DEVICE_LIB_PATH)

    if enable_xla:
        tf_xla_flags = _append_unique_flag(
            os.environ.get("TF_XLA_FLAGS", ""),
            "--tf_xla_enable_xla_devices",
        )
        if device == "musa":
            tf_xla_flags = _set_flag_with_prefix(
                tf_xla_flags,
                "--tf_xla_min_cluster_size=",
                "--tf_xla_min_cluster_size=5",
            )
        os.environ["TF_XLA_FLAGS"] = tf_xla_flags
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("TF_DISABLE_MKL", "1")

    if xla_dump_dir:
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags = _set_flag_with_prefix(
            xla_flags,
            "--xla_dump_to=",
            f"--xla_dump_to={xla_dump_dir}",
        )
        xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_text")
        xla_flags = _append_unique_flag(xla_flags, "--xla_dump_hlo_as_long_text")
        os.environ["XLA_FLAGS"] = xla_flags
        os.environ["GRAPH_RUNNER_XLA_DUMP_DIR"] = xla_dump_dir


_configure_runtime_env_from_argv(sys.argv[1:])

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib

# ==========================================
# 配置区域
# ==========================================
model_path = "./graph_def.pb"
output_node_name = "predicts"  # 输出节点名称
musa_plugin_path = "/workspace/xla_ref/xla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"

# 禁用 V2 行为，确保 TF1 图能正常运行
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


# ==========================================
# 1. 加载 MUSA 插件，配置config
# ==========================================
def load_musa_plugin():
    if os.path.exists(musa_plugin_path):
        try:
            tf.load_op_library(musa_plugin_path)
            print(f">>>> [MUSA] Plugin loaded successfully from: {musa_plugin_path}")
        except Exception as e:
            print(f"!!!! [MUSA] Failed to load plugin: {e}")
    else:
        print(f"!!!! [MUSA] Plugin not found at {musa_plugin_path}, assuming built-in.")


def print_physical_devices():
    print("\n=== TensorFlow 物理设备 ===")
    try:
        all_devices = tf.config.list_physical_devices()
        if not all_devices:
            print("未发现任何物理设备")
            return
        for device_type in ("CPU", "GPU", "XLA_CPU", "XLA_GPU"):
            devices = tf.config.list_physical_devices(device_type)
            if devices:
                print(f"  {device_type}:")
                for device in devices:
                    print(f"    - {device}")
        print("\n=== TensorFlow 本地设备枚举 ===")
        for device in device_lib.list_local_devices():
            print(f"  - {device.name} ({device.device_type})")
    except Exception as e:
        print(f"!!!! 获取物理设备失败: {e}")


def print_session_devices(sess):
    print("\n=== Session 可见设备 ===")
    try:
        devices = sess.list_devices()
        if not devices:
            print("Session 中未发现任何可见设备")
            return
        for device in devices:
            print(f"  - {device.name} ({device.device_type})")
    except Exception as e:
        print(f"!!!! 获取 Session 设备失败: {e}")


def prepare_xla_dump_dir(dump_dir: Optional[str]):
    if not dump_dir:
        return
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    print(f"[XLA Dump] 输出目录: {dump_dir}")
    print(f"[XLA Dump] XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")


def summarize_xla_dump_dir(dump_dir: Optional[str]):
    if not dump_dir:
        return
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        print(f"[XLA Dump] 目录不存在: {dump_dir}")
        return

    txt_files = list(dump_path.rglob("*.txt"))
    ll_files = list(dump_path.rglob("*.ll"))
    mlir_files = list(dump_path.rglob("*.mlir"))
    print("\n[XLA Dump 汇总]")
    print(f"  目录: {dump_dir}")
    print(f"  .txt: {len(txt_files)}")
    print(f"  .ll:  {len(ll_files)}")
    print(f"  .mlir:{len(mlir_files)}")


def _list_visible_xla_gpu_devices() -> List[str]:
    devices = []
    for device in device_lib.list_local_devices():
        if device.device_type == "XLA_GPU":
            devices.append(device.name)
    return devices


def select_runtime_device(device_type: str, xla: bool) -> Optional[str]:
    normalized_device = (device_type or "cpu").lower()
    if normalized_device == "musa" and xla:
        xla_gpus = _list_visible_xla_gpu_devices()
        if xla_gpus:
            return xla_gpus[0]
        raise RuntimeError(
            "未发现 XLA_GPU 设备；请先确认 TF_XLA_FLAGS 是否在 TensorFlow 导入前生效。"
        )
    if normalized_device == "cuda":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return "/device:GPU:0"
    return None


def assign_default_runtime_device(graph_def, runtime_device: Optional[str]):
    if not runtime_device:
        return graph_def

    assigned_graph_def = graph_pb2.GraphDef()
    assigned_graph_def.CopyFrom(graph_def)
    assigned_count = 0
    for node in assigned_graph_def.node:
        if not node.device:
            node.device = runtime_device
            assigned_count += 1
    print(f"[XLA] 默认将 {assigned_count} 个未指定设备的节点指向 {runtime_device}")
    return assigned_graph_def

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
        if xla:
            config.device_count["XLA_GPU"] = 1
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
            # rewrite_options = config.graph_options.rewrite_options
            # rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
            # rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
            # rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
            if logger is not None:
                logger.info(
                    "Enabled XLA_GPU session for MUSA with TF_XLA_FLAGS=%s",
                    os.environ.get("TF_XLA_FLAGS", ""),
                )
        else:
            rewrite_options = config.graph_options.rewrite_options
            rewrite_options.custom_optimizers.add().name = "musa_graph_optimizer"
            if logger is not None:
                logger.info("Enabled custom optimizer: musa_graph_optimizer")

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

    # print(f"找到 {len(placeholders)} 个 Placeholder 节点")
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

        # print(f"Mock 数据 - {name}:")
        # print(f"  形状: {mock_shape}")
        # print(f"  数据类型: {dtype}")
        # print(f"  数据范围: [{np.min(mock_data):.4f}, {np.max(mock_data):.4f}]")

    return feed_dict


# ==========================================
# 4. 执行推理
# ==========================================
def run_inference(graph_def, feed_dict, output_node_name, device="cpu", xla=False, num_runs=100, warmup_runs=10):
    print(f"\n=== 执行图推理 ===")
    print(f"输出节点: {output_node_name}")
    print(f"设备: {device.upper()}")
    if device.lower() in {"cuda", "musa"}:
        print(f"XLA: {xla}")
    print(f"预热次数: {warmup_runs}, 正式运行次数: {num_runs}")

    runtime_device = select_runtime_device(device, xla)
    if runtime_device:
        print(f"[运行设备] 选中: {runtime_device}")
    prepared_graph_def = assign_default_runtime_device(graph_def, runtime_device)

    with tf.Graph().as_default() as graph:
        # 测量图导入时间
        t_import_start = time.time()
        if runtime_device:
            with graph.device(runtime_device):
                tf.import_graph_def(prepared_graph_def, name="")
        else:
            tf.import_graph_def(prepared_graph_def, name="")
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
                pass  # 忽略图中不存在的 tensor

        # 获取输出 Tensor
        try:
            output_tensor = graph.get_tensor_by_name(f"{output_node_name}:0")
        except KeyError:
            print(f"错误: 找不到输出张量 {output_node_name}:0")
            # 尝试打印所有节点找名字
            # for n in graph.as_graph_def().node: print(n.name)
            return None
        t_feed_end = time.time()
        print(f"[时间] Feed Dict 准备耗时: {(t_feed_end - t_feed_start)*1000:.2f} ms")

        # 打印 op 的最终设备放置，便于确认是否真的落在目标后端上
        config = create_session_config(
            device_type=device,
            xla=xla,
            log_device_placement=False,
        )

        # 测量 Session 创建时间
        t_sess_start = time.time()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            t_sess_end = time.time()
            print(f"[时间] Session 创建耗时: {(t_sess_end - t_sess_start)*1000:.2f} ms")
            print_session_devices(sess)

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
        help="启用 XLA 加速 (当 device=cuda 或 musa 时有效)"
    )
    parser.add_argument(
        "--xla_dump",
        action="store_true",
        help=f"启用 XLA dump，并默认输出到 {DEFAULT_XLA_DUMP_DIR}"
    )
    parser.add_argument(
        "--xla_dump_dir",
        type=str,
        default=None,
        help="指定 XLA dump 输出目录；会自动注入 xla_dump_to/xla_dump_hlo_as_text/xla_dump_hlo_as_long_text"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="正式运行次数 (默认: 2000)"
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
    xla_dump_dir = os.environ.get("GRAPH_RUNNER_XLA_DUMP_DIR")

    print("="*50)
    print("参数配置")
    print("="*50)
    print(f"  设备:     {args.device.upper()}")
    print(f"  Batch Size: {args.batchsize}")
    print(f"  XLA:      {args.xla if args.device in {'cuda', 'musa'} else 'N/A'}")
    print(f"  XLA Dump: {xla_dump_dir if xla_dump_dir else 'OFF'}")
    print(f"  运行次数: {args.num_runs}")
    print(f"  预热次数: {args.warmup_runs}")
    print("="*50)

    if xla_dump_dir and not args.xla:
        print("[提示] 已启用 XLA dump，但当前未显式开启 --xla；只有图实际走到 XLA 编译路径时才会生成 dump。")

    total_start = time.time()
    prepare_xla_dump_dir(xla_dump_dir)

    # 仅当 device=musa 且非 XLA 路径时加载 MUSA 插件
    if args.device.lower() == "musa":
        if args.xla:
            print(">>>> [MUSA/XLA] 跳过外部 plugin，优先走内建 XLA_GPU/MUSA runtime 路径")
        else:
            load_musa_plugin()

    print_physical_devices()

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
    summarize_xla_dump_dir(xla_dump_dir)
    print(f"\n[总耗时] {(total_end - total_start)*1000:.2f} ms")


if __name__ == "__main__":
    main()
