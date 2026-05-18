"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_vendored")
load("//third_party/gpus:musa_configure.bzl", "musa_configure")

# buildifier: disable=function-docstring
# buildifier: disable=unnamed-macro
def workspace():
    # TensorFlow-vendored code still uses @local_xla and @local_tsl, while
    # imported OpenXLA code can refer to @tsl directly.
    tf_vendored(name = "local_xla", relpath = ".")
    tf_vendored(name = "local_tsl", relpath = "third_party/tsl")
    tf_vendored(name = "tsl", relpath = "third_party/tsl")

    # MUSA/MTGPU code depends on @local_config_musa in both XLA and TSL.
    musa_configure(name = "local_config_musa")

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace4 = workspace
