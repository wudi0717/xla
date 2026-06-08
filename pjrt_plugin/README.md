# MUSA PJRT TensorFlow XLA 插件

## 构建

在TensorFlow2.15环境下构建主要的 PJRT 插件共享库：

```bash
bazel build \
  --repo_env=TF_SYSTEM_LIBS=com_github_grpc_grpc \
  --override_repository=local_config_musa=$PWD/third_party/local_config_musa \
  //pjrt_plugin:libmusa_pjrt_plugin.so \
  //pjrt_plugin:libmusa_tf215_registry_bridge.so \
  //pjrt_plugin:libmusa_tf215_npd_adapter.so
```

## 运行

通过 MUSA XLA 运行主要的 GraphDef 演示：

```bash
cd /xla/pjrt_plugin/model
python graph_runner_musa_xla.py --device musa --xla
```

日志中预期出现的关键标志：

```text
TFNPD_InitPlugin ... device_type=MUSA compilation_device=XLA_GPU_JIT
PJRT_Api is set for device type musa
XLA service ... initialized for platform MUSA
Compiled cluster using XLA!
```

对比非 XLA 的 MUSA 执行：

```bash
python graph_runner_musa_xla.py --device musa
```

只运行一次短验证：

```bash
python graph_runner_musa_xla.py --device musa --xla --warmup_runs 1 --num_runs 1
```

当需要确认优化后的 HLO 和 fusion 结果时，可以启用 XLA dump：

```bash
python graph_runner_musa_xla.py \
  --device musa \
  --xla \
  --warmup_runs 1 \
  --num_runs 1 \
  --xla_dump \
  --xla_dump_dir /tmp/musa_xla_dump
```

## 相关参数

`model/graph_runner_musa_xla.py` 支持以下参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --device {cpu,musa} | `cpu` | 运行设备。 |
| `--xla` | 关闭 | 启用 TensorFlow XLA auto_jit 路径。 |
| `--batchsize N` | `100` | 用于实例化动态输入 shape 的 batch size。 |
| `--warmup_runs N` | `10` | 预热阶段的 `Session.run` 迭代次数。 |
| `--num_runs N` | `1000` | 正式计时阶段的 `Session.run` 迭代次数。 |
| `--xla_dump` | 关闭 | 使用默认 dump 目录启用 HLO dump。 |
| `--xla_dump_dir DIR` | 未设置 | 覆盖 HLO dump 输出目录。 |



# TF MUSA Spec Runner

## 文件说明

| 文件                       | 用途                                                    |
| -------------------------- | ------------------------------------------------------- |
| `musa_run_pb_graph_xla.py` | 单个 spec 或 spec 目录的主运行脚本。                    |
| `convert_spec_to_pb.py`    | 将 `.spec` MetaGraphDef 转换为 frozen PB。              |
| `graph_runner_xla.sh`      | 类似 `graph_runner.sh` 的 batch size 基准测试封装脚本。 |

## 示例 ：单个 Spec 启用 XLA

以 batch size 1024 运行一个 spec：

```bash
cd /workspace/xla_ref/xla/pjrt_plugin/model/tf_musa_runner_xla
python3 musa_run_pb_graph_xla.py \
  --spec ./meta_graph/meta_graph_1.spec \
  --bs 1024 \
  --warmup 3 \
  --run_iters 20 \
  --device /device:MUSA:0 \
  --xla
```

## Python Runner 参数

| 参数                          | 默认值                  | 说明                                                         |
| ----------------------------- | ----------------------- | ------------------------------------------------------------ |
| `--spec PATH`                 | unset                   | 运行单个 `.spec` MetaGraphDef。`--spec` 和 `--spec_dir` 必须且只能指定其中一个。 |
| `--spec_dir DIR`              | unset                   | 递归运行目录中的所有 `.spec` 文件。                          |
| `--pb PATH`                   | unset                   | 使用显式指定的 frozen PB。仅在使用 `--spec` 时有效。         |
| `--bs LIST`                   | `1024`                  | 用于填充未解析第一维的 batch size。支持逗号列表，例如 `1,8,1024`。 |
| `--unknown_dim N`             | `1`                     | 用于填充未解析的非 batch 维度。                              |
| `--warmup N`                  | `3`                     | 预热 `Session.run` 迭代次数，不计入计时。                    |
| `--run_iters N`               | `10`                    | 正式计时的 `Session.run` 迭代次数。                          |
| `--seed N`                    | `2026`                  | 随机输入种子。                                               |
| `--out_root DIR`              | `runner_out`            | `run_report.json` 的输出根目录。                             |
| `--strict BOOL`               | `True`                  | 如果任意 spec/batch 用例失败，则抛出错误。                   |
| `--device DEVICE`             | `/device:MUSA:0`        | TensorFlow device scope，例如 `/device:MUSA:0` 或 `/device:CPU:0`。 |
| `--allow_soft_placement BOOL` | `True`                  | 允许 TensorFlow 将不支持的算子放置到其他设备。               |
| `--log_device_placement BOOL` | `False`                 | 打印 TensorFlow 设备放置日志。                               |
| `--musa_optimizer BOOL`       | `True`                  | 对 MUSA 运行启用 `musa_graph_optimizer`。                    |
| `--convert_script PATH`       | `convert_spec_to_pb.py` | 当需要自动转换 PB 时使用的脚本。                             |
| `--convert_out_root DIR`      | `frozen_out`            | 生成 frozen PB 文件的输出根目录。                            |
| `--xla`                       | off                     | 启用 TensorFlow XLA auto JIT。                               |
| `--xla_device_scope`          | off                     | 在 MUSA XLA 模式导入图时保留显式 device scope。              |
| `--xla_dump`                  | off                     | 启用 XLA HLO dump。                                          |
| `--xla_dump_dir DIR`          | unset                   | 覆盖 XLA dump 目录。                                         |
| `--musa_plugin PATH`          | auto                    | `libmusa_pjrt_plugin_zy.so` 的路径。                         |

默认直接运行 Python 脚本时：

```text
warmup=3, run_iters=10
```
