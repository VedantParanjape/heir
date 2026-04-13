//===- CoyoteVectorizer.h - Coyote Vectorization Pass -----------*- C++ -*-===//
//
// Header file for the Coyote vectorization pass based on ASPLOS '23 paper.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_H_
#define LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_H_

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/EquivalenceClasses.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/RegionUtils.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h.inc"

void coyoteVectorizer(func::FuncOp& func);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_H_
