package(default_visibility = ["//visibility:public"])

cc_library(
    name = "musa_headers",
    hdrs = glob(["include/**"]),
    includes = ["include"],
)

cc_library(
    name = "musa_rpath",
    linkopts = ["-Wl,-rpath,/usr/local/musa/lib"],
)

cc_library(
    name = "musa",
    linkopts = ["-L/usr/local/musa/lib", "-lmusa", "-lmusart"],
    deps = [":musa_headers"],
)

cc_library(
    name = "mublas",
    linkopts = ["-L/usr/local/musa/lib", "-lmublas"],
    deps = [":musa_headers"],
)

cc_library(
    name = "mudnn",
    linkopts = ["-L/usr/local/musa/lib", "-lmudnn"],
    deps = [":musa_headers"],
)
