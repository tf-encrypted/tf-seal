load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_google_googletest",
    url = "https://github.com/google/googletest/archive/release-1.8.1.zip",
    strip_prefix = "googletest-release-1.8.1",
    sha256 = "927827c183d01734cc5cfef85e0ff3f5a92ffe6188e0d18e909c5efebf28a0c7",
)

#
# CMake rules
#

http_archive(
   name = "rules_foreign_cc",
   url = "https://github.com/bazelbuild/rules_foreign_cc/archive/master.zip",
   strip_prefix = "rules_foreign_cc-master",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies([
    # "//:built_cmake_toolchain",
])

#
# TensorFlow
#

load("//external/tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

#
# Protobuf
#

# Note that we have to use the specific version used by TensorFlow.

http_archive(
    name = "com_google_protobuf",
    sha256 = "b9e92f9af8819bbbc514e2902aec860415b70209f31dfc8c4fa72515a5df9d59",
    strip_prefix = "protobuf-310ba5ee72661c081129eb878c1bbcec936b20f0",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz"],
)

http_archive(
    name = "com_google_protobuf_cc",
    sha256 = "b9e92f9af8819bbbc514e2902aec860415b70209f31dfc8c4fa72515a5df9d59",
    strip_prefix = "protobuf-310ba5ee72661c081129eb878c1bbcec936b20f0",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

#
# SEAL
#

# compile SEAL with Bazel
http_archive(
    name = "seal",
    build_file = "seal/BUILD",
    url = "https://github.com/microsoft/SEAL/archive/3.3.1.zip",
    sha256 = "83b8748f2f342b0e90eb81c73cfe162a86cc5ca6f7facfa75fa84b9f82a2b74d",
    strip_prefix = "SEAL-3.3.1",
)

# compile SEAL manually
# new_local_repository(
#     name = "seal",
#     path = "/usr/local/",
#     build_file = "third_party/seal/BUILD",
# )
