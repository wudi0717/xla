import ctypes
import os
import sys

import numpy as np

musa_plugin_path = "/workspace/xla_ref/xla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"

os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = musa_plugin_path
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{musa_plugin_path}"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = (
    os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_use_device_api=true"
).strip()

import tensorflow as tf


def load_musa_plugin():
    if not os.path.exists(musa_plugin_path):
        print(f"!!!! [MUSA] Plugin not found at {musa_plugin_path}")
        sys.exit(1)

    lib = ctypes.CDLL(musa_plugin_path)
    lib.ForceRegisterMusa()

    devices = tf.config.list_physical_devices("MUSA")
    print(f">>>> [MUSA] Physical devices: {devices}")
    if not devices:
        sys.exit(1)


@tf.function(
    jit_compile=True,
    input_signature=[
        tf.TensorSpec([128, 128], tf.float32),
        tf.TensorSpec([128, 128], tf.float32),
    ],
)
def xla_matmul(x, y):
    z = tf.matmul(x, y)
    return tf.nn.relu(z + 1.0)


def main():
    load_musa_plugin()

    x = tf.constant(np.random.normal(size=(128, 128)).astype(np.float32))
    y = tf.constant(np.random.normal(size=(128, 128)).astype(np.float32))
    print(f">>>> [MUSA/XLA] input shapes: {x.shape}, {y.shape}")

    with tf.device("/device:MUSA:0"):
        result = xla_matmul(x, y)

    print(f">>>> [MUSA/XLA] result shape: {result.shape}")
    result_np = result.numpy()
    print(f">>>> [MUSA/XLA] result mean: {float(np.mean(result_np)):.6f}")


if __name__ == "__main__":
    main()
