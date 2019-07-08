load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

load("//tf:tf_configure.bzl", "tf_configure")

http_archive(
    name = "com_google_googletest",
    url = "https://github.com/google/googletest/archive/release-1.8.1.zip",
    strip_prefix = "googletest-release-1.8.1",
    sha256 = "927827c183d01734cc5cfef85e0ff3f5a92ffe6188e0d18e909c5efebf28a0c7",
)

tf_configure(name = "local_config_tf")

new_local_repository(
    name = "seal",
    path = "/usr/local/",
    build_file = "external/seal.BUILD"
)