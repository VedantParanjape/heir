load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Suite-level shared code: workload hooks (bench_hooks::*) and reference
# implementation (reference_matmul + gen_random_matrix). Every mm bench
# binary depends on this.
cc_library(
    name = "hooks",
    hdrs = [
        "bench_hooks.h",
        "reference.h",
    ],
    # Dependents get this dir on their include path so the emitted
    # benchmark.cpp can `#include "bench_hooks.h"` and reference.h can
    # `#include "types.h"` (via common:types's own includes).
    includes = ["."],
    deps = [
        "//biscotti-bench/benchmarks/common:types",
    ],
)

# One cc_binary per (kernel, size, variant). Variants:
#   ""          — hoisted / coyote-optimized (default from build.sh)
#   "_baseline" — biscotti inline=-1, fully unrolled (build.sh --baseline)
# The three source files (kernel.{cpp,h}, benchmark.cpp) live in
# output/<kernel><variant>/, produced by scripts/build.sh.
#
# Add a new (kernel, size) tuple below to register both variants of a
# new benchmark.
[
    cc_binary(
        name = "bench_" + kernel + variant,
        srcs = [
            "output/" + kernel + variant + "/benchmark.cpp",
            "output/" + kernel + variant + "/kernel.cpp",
            "output/" + kernel + variant + "/kernel.h",
        ],
        copts = [
            "-DBENCH_HOOKS_SIZE=" + size,
            # HEIR PKE emitter currently emits `std::vector<size_t> = {-1, 1}`
            # for rotation shifts; the narrowing is well-defined and the
            # runtime does mod-ring reduction, so silence the warning until
            # the emitter is fixed upstream.
            "-Wno-c++11-narrowing",
        ],
        # Put the per-kernel output dir on the include path so the
        # emitted benchmark.cpp can `#include "kernel.h"` bare.
        includes = ["output/" + kernel + variant],
        deps = [
            ":hooks",
            "//biscotti-bench/benchmarks/common:types",
            "@google_benchmark//:benchmark_main",
            "@openfhe//:core",
            "@openfhe//:pke",
        ],
    )
    for kernel, size in [
        ("mm4x4", "4"),
        ("mm6x6_3x3", "6"),
        ("mm6x6_2x2", "6"),
        ("mm8x8", "8"),
    ]
    for variant in [
        "",
        "_baseline",
    ]
]
