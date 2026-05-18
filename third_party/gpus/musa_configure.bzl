"""Repository rule for local MUSA toolkit configuration."""

def _get_env(repository_ctx, names, default):
    for name in names:
        value = repository_ctx.os.environ.get(name)
        if value:
            return value
    return default

def _join(a, b):
    if a.endswith("/"):
        return a + b
    return a + "/" + b

def _musa_configure_impl(repository_ctx):
    musa_root = _get_env(
        repository_ctx,
        ["MUSA_TOOLKIT_PATH", "MUSA_HOME", "MUSA_PATH"],
        "/usr/local/musa",
    )
    include_dir = _join(musa_root, "include")
    lib_dir = _join(musa_root, "lib")
    if repository_ctx.path(_join(musa_root, "lib64")).exists:
        lib_dir = _join(musa_root, "lib64")

    repository_ctx.file("BUILD", "")
    repository_ctx.symlink(repository_ctx.path(include_dir), "musa/include")
    repository_ctx.symlink(repository_ctx.path(lib_dir), "musa/lib")
    repository_ctx.template(
        "musa/BUILD",
        Label("//third_party/gpus:musa.BUILD.tpl"),
        {
            "%{musa_lib_dir}": lib_dir,
        },
    )

musa_configure = repository_rule(
    implementation = _musa_configure_impl,
    environ = [
        "MUSA_TOOLKIT_PATH",
        "MUSA_HOME",
        "MUSA_PATH",
    ],
)
