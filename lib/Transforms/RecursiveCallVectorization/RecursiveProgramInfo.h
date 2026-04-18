#ifndef RECURSIVE_PROGRAM_INFO_H
#define RECURSIVE_PROGRAM_INFO_H

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/CallGraph.h"      // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Inliner.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/RegionUtils.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

typedef struct recursiveProgramNode_ {
  recursiveProgramNode_ *parent = nullptr;
  bool vectorizationCandidate = false;
  func::FuncOp function;
  func::CallOp caller;
  SmallVector<std::pair<TypedAttr, int>> staticArgumentValues;
  SmallVector<recursiveProgramNode_ *> children;
} recursiveProgramNode;

typedef struct recursiveProgramInfo_ {
  SmallVector<std::pair<Operation *, int>> baseConditions;
  SmallVector<std::pair<Operation *, int>> recursiveCalls;
  SmallVector<std::pair<Value *, int>> progressArguments;
  SmallVector<std::pair<TypedAttr, int>> staticArgumentValues;
  Operation *call;
  recursiveProgramNode *root;
} recursiveProgramInfo;
static DenseMap<Operation *, recursiveProgramInfo> biscottiCalls;

}  // namespace heir
}  // namespace mlir

#endif  // RECURSIVE_PROGRAM_INFO_H
