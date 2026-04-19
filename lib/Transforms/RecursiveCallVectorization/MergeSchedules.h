//===- NeedlemanWunschMerge.h - NW-based MLIR function merge ----*- C++ -*-===//
//
// Merges two func::FuncOp with identical signatures using Needleman-Wunsch
// sequence alignment on topologically sorted operations. Matched operations
// are shared (emitted once); unmatched operations from both functions are
// preserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_COYOTEVECTORIZER_NEEDLEMANWUNSCHMERGE_H_
#define LIB_TRANSFORMS_COYOTEVECTORIZER_NEEDLEMANWUNSCHMERGE_H_

#include "lib/Transforms/RecursiveCallVectorization/RecursiveProgramInfo.h"
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

/// One entry in the NW alignment.
struct AlignmentEntry {
  enum Kind {
    Match,  // Both opA and opB present; opA is emitted, opB reuses its results
    GapA,   // Only opB present (gap in sequence A)
    GapB,   // Only opA present (gap in sequence B)
  };
  Kind kind;
  Operation *opA = nullptr;
  Operation *opB = nullptr;
};

/// Scoring constants for the NW alignment.
struct NWScoreConfig {
  int matchExact = 4;   // same opcode + same operand structure
  int matchOpcode = 2;  // same opcode, different operand structure
  int matchClass = 1;   // same dialect, different opcode
  int mismatch = -1;    // different dialect
  int gapPenalty = -2;  // insertion/deletion
};

/// Merge two func::FuncOp with identical function types using NW alignment.
///
/// Both functions must have the same FunctionType. The merged function:
/// - Shares the same argument types (same inputs)
/// - Has concatenated return types (retA, retB)
/// - Shares matched operations (emitted once for both)
///
/// The merged function is inserted before funcA in the parent module.
/// On success, `result` is set to the newly created function.
LogicalResult mergeWithNeedlemanWunsch(
    func::FuncOp funcA, func::FuncOp funcB, func::FuncOp &result,
    const NWScoreConfig &config = NWScoreConfig());

/// Extract operations from a func's secret.generic body in topological order.
/// Exposed for testing.
llvm::SmallVector<Operation *> extractSortedOps(func::FuncOp func);

/// Run NW alignment on two operation sequences. Exposed for testing.
llvm::SmallVector<AlignmentEntry> runNeedlemanWunsch(
    llvm::ArrayRef<Operation *> seqA, llvm::ArrayRef<Operation *> seqB,
    const NWScoreConfig &config = NWScoreConfig());

void findScheduleMergingCandidates(
    recursiveProgramNode *node,
    DenseMap<recursiveProgramNode *, SmallVector<recursiveProgramNode *>>
        &candidates,
    DenseSet<func::CallOp> &visited);

typedef struct cipherTextSlot_ {
  Operation *op;
  int index;
} cipherTextSlot;

SmallVector<cipherTextSlot> processTensorOpsAfterMerging(
    RankedTensorType mergedType, SmallVector<Value> subArgs, OpBuilder builder);
Value mergeTensorOps(SmallVector<cipherTextSlot> &ctxt,
                     RankedTensorType mergedType, OpBuilder builder);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_COYOTEVECTORIZER_NEEDLEMANWUNSCHMERGE_H_
