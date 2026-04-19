load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RecursiveCallVectorization",
    srcs = [
        "MergeSchedules.cpp",
        "RecursiveCallVectorization.cpp",
    ],
    hdrs = [
        "CoyoteCaller.h",
        "MergeSchedules.h",
        "RecursiveCallVectorization.h",
        "RecursiveProgramInfo.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Transforms/CoyoteVectorizer",
        "@heir//lib/Utils:AttributeUtils",
        "@heir//lib/Utils/Graph",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "RecursiveCallVectorization",
)
