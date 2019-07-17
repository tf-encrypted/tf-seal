# On Bazel CI Mac machines, we are using cmake built from sources
toolchain(
    name = "built_cmake_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:osx",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = "@rules_foreign_cc//tools/build_defs/native_tools:built_cmake",
    toolchain_type = "@rules_foreign_cc//tools/build_defs:cmake_toolchain",
)
