licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_seal_ops.so',
    srcs = [
        "cc/kernels/seal_kernels.cc",
        "cc/kernels/seal_tensors.h",
        "cc/kernels/seal_tensors.cc",
        "cc/kernels/key_variants.h",
        "cc/kernels/key_variants.cc",
        "cc/ops/seal_ops.cc",
        "cc/kernels/seal_helpers.h"
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@seal//:lib"
    ],
    copts = ["-pthread", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1", "-fPIC"],
)

py_library(
    name = "seal_ops_py",
    srcs = ([
        "python/ops/seal_ops.py",
    ]),
    data = [
        ":python/ops/_seal_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "seal_ops_py_test",
    srcs = [
        "python/ops/seal_ops_test.py",
    ],
    main = "python/ops/seal_ops_test.py",
    deps = [
        ":seal_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "tf_seal_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":seal_ops_py"
    ],
    srcs_version = "PY2AND3",
)
