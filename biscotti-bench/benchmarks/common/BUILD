load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    # Dependents get this dir on their include path so they can write
    # `#include "types.h"` bare, not the workspace-relative absolute.
    includes = ["."],
    deps = [
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
