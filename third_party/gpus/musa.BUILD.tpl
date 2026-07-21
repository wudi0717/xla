licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "musa_headers",
    hdrs = glob(["include/**"]),
    includes = ["include"],
)

cc_library(
    name = "musa_rpath",
    linkopts = ["-Wl,-rpath,%{musa_lib_dir}"],
)

cc_library(
    name = "musa",
    linkopts = [
        "-L%{musa_lib_dir}",
        "-lmusa",
        "-lmusart",
    ],
    deps = [":musa_headers"],
)

cc_library(
    name = "mublas",
    linkopts = [
        "-L%{musa_lib_dir}",
        "-lmublas",
    ],
    deps = [":musa_headers"],
)

cc_library(
    name = "mudnn",
    linkopts = [
        "-L%{musa_lib_dir}",
        "-lmudnn",
    ],
    deps = [":musa_headers"],
)
