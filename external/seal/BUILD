#
# compile SEAL with Bazel
#

load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

cmake_external(
    name = "lib",
    cache_entries = {
        "MAKE_INSTALL_PREFIX": "$EXT_BUILD_DEPS/seal",
        "SEAL_LIB_BUILD_TYPE": "Static_PIC",
    },
    lib_source = "@seal//:native/src",
    make_commands = ["make", "make install"],
    static_libraries = ["libseal.a"],
    visibility = ["//visibility:public"],
)

# cmake_external(
#     name = "examples",
#     cache_entries = {
#         "MAKE_PREFIX_PATH": "$EXT_BUILD_DEPS/seal",
#     },
#     lib_source = "@seal//:native/examples",
#     make_commands = ["make"],
#     visibility = ["//visibility:public"],
#     deps = [":lib"],
# )

#
# compile SEAL manually
#

# cc_library(
#     name = "lib",
#     srcs = [
#         "lib/libseal.a",
#     ],
#     hdrs = [
#         "include/seal/seal.h",
#         "include/seal/biguint.h",
#         "include/seal/memorymanager.h",
#         "include/seal/util/mempool.h",
#         "include/seal/util/defines.h",
#         "include/seal/util/msvc.h",
#         "include/seal/util/clang.h",
#         "include/seal/util/config.h",
#         "include/seal/util/gcc.h",
#         "include/seal/util/globals.h",
#         "include/seal/util/hestdparms.h",
#         "include/seal/util/common.h",
#         "include/seal/util/locks.h",
#         "include/seal/util/pointer.h",
#         "include/seal/util/uintcore.h",
#         "include/seal/util/uintarith.h",
#         "include/seal/util/uintarithmod.h",
#         "include/seal/ciphertext.h",
#         "include/seal/context.h",
#         "include/seal/encryptionparams.h",
#         "include/seal/randomgen.h",
#         "include/seal/util/aes.h",
#         "include/seal/smallmodulus.h",
#         "include/seal/util/hash.h",
#         "include/seal/modulus.h",
#         "include/seal/util/smallntt.h",
#         "include/seal/util/baseconverter.h",
#         "include/seal/intarray.h",
#         "include/seal/valcheck.h",
#         "include/seal/ckks.h",
#         "include/seal/plaintext.h",
#         "include/seal/util/polycore.h",
#         "include/seal/util/uintarithsmallmod.h",
#         "include/seal/util/numth.h",
#         "include/seal/decryptor.h",
#         "include/seal/secretkey.h",
#         "include/seal/intencoder.h",
#         "include/seal/encryptor.h",
#         "include/seal/publickey.h",
#         "include/seal/evaluator.h",
#         "include/seal/relinkeys.h",
#         "include/seal/kswitchkeys.h",
#         "include/seal/galoiskeys.h",
#         "include/seal/keygenerator.h",
#         "include/seal/batchencoder.h",
#         "include/seal/randomtostd.h"
#     ],
#     visibility = ["//visibility:public"],
#     strip_include_prefix = "include",
# )