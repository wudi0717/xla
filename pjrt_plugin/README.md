# MUSA PJRT TensorFlow XLA 插件

## 构建

在TensorFlow2.15环境下构建主要的 PJRT 插件共享库：

```bash
bazel build \
  --repo_env=TF_SYSTEM_LIBS=com_github_grpc_grpc \
  --override_repository=local_config_musa=$PWD/third_party/local_config_musa \
  //pjrt_plugin:libmusa_pjrt_plugin_zy.so \
  //pjrt_plugin:libmusa_tf215_registry_bridge.so
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

| 参数                 | 默认值 | 说明                                     |
| -------------------- | ------ | ---------------------------------------- |
| --device {cpu,musa}  | `cpu`  | 运行设备。                               |
| `--xla`              | 关闭   | 启用 TensorFlow XLA auto_jit 路径。      |
| `--batchsize N`      | `100`  | 用于实例化动态输入 shape 的 batch size。 |
| `--warmup_runs N`    | `10`   | 预热阶段的 `Session.run` 迭代次数。      |
| `--num_runs N`       | `1000` | 正式计时阶段的 `Session.run` 迭代次数。  |
| `--xla_dump`         | 关闭   | 使用默认 dump 目录启用 HLO dump。        |
| `--xla_dump_dir DIR` | 未设置 | 覆盖 HLO dump 输出目录。                 |
