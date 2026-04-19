//===- NeedlemanWunschMerge.h - NW-based MLIR function merge ----*- C++ -*-===//
//
// Merges two func::FuncOp with identical signatures using Needleman-Wunsch
// sequence alignment on topologically sorted operations. Matched operations
// are shared (emitted once); unmatched operations from both functions are
// preserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_RECURSIVECALLVECTORIZATION_MERGESCHEDULES_H_
#define LIB_TRANSFORMS_RECURSIVECALLVECTORIZATION_MERGESCHEDULES_H_

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
/// This is pure alignment — no type widening, no signature changes. The merged
/// function has the **same signature as funcA (== funcB)** and yields A's
/// values. The body contains:
/// - Matched operations emitted once (A's version); B's downstream uses are
///   remapped to share A's results.
/// - Gap operations from both sides emitted as-is.
///
/// The widening / re-wiring for SIMD packing is left to a separate pass.
///
/// The merged function is created detached; the caller must insert it into
/// the module (e.g. `module.push_back(merged)`).
LogicalResult mergeWithNeedlemanWunsch(
    func::FuncOp funcA, func::FuncOp funcB, func::FuncOp &result,
    const NWScoreConfig &config = NWScoreConfig());

/// Merge N pre-vectorized kernels at the **Schedule level** using NW.
///
/// Internally performs pairwise NW alignment from left to right:
/// `funcs[0] + funcs[1] → m01`, then `m01 + funcs[2] → m012`, etc. Each
/// pairwise step produces a merged function and tracks which original
/// kernel indices each cloned op represents.
///
/// After all pairwise merges, the **mod-N data layout** is applied as a
/// final step using all input schedules:
///
///     merged_lane(op) = original_lane(op, kernel_k) * N + k
///
/// Matched ops (representing multiple kernels) take kernel 0's lane as
/// canonical; gap ops use their owning kernel's lane translated into the
/// kernel's stripe.
///
/// The merged function is created detached (caller must `module.push_back`).
/// The merged Schedule references **cloned** ops in the merged function
/// body, and stays valid until canonicalization erases the scalar ops.
LogicalResult mergeSchedulesWithNW(
    llvm::ArrayRef<func::FuncOp> funcs, llvm::ArrayRef<Schedule> schedules,
    func::FuncOp &mergedFunc, Schedule &mergedSchedule,
    const NWScoreConfig &config = NWScoreConfig());

/// Extract operations from a func's secret.generic body in topological order.
/// Exposed for testing.
llvm::SmallVector<Operation *> extractSortedOps(func::FuncOp func);

/// Run NW alignment on two operation sequences. Exposed for testing.
llvm::SmallVector<AlignmentEntry> runNeedlemanWunsch(
    llvm::ArrayRef<Operation *> seqA, llvm::ArrayRef<Operation *> seqB,
    const NWScoreConfig &config = NWScoreConfig());

/// Merge multiple tensor insert chains into a single wider chain.
///
/// Each element of \p chainEnds is the final tensor.insert result Value of an
/// independent insert chain (a sequence of tensor.extract + tensor.insert pairs
/// that builds up a tensor from a zero constant). The chains are combined into
/// one wider tensor, with chain i's last-dimension insert indices offset by the
/// sum of preceding chains' widths.
///
/// \p builder must be positioned inside the block containing the chains (e.g.,
/// before secret.yield). Returns the final Value of the merged chain, or
/// nullptr on failure.
Value mergeInsertChains(llvm::ArrayRef<Value> chainEnds, OpBuilder &builder);

/// Retype a single function argument and forward-propagate the type change.
///
/// Sets argument \p argIdx of \p func to \p newType, then walks the use-def
/// graph forward, updating types as it goes:
///  - Element-wise body ops: result tensor shape is recomputed from the first
///    ranked-tensor operand (preserving each result's element type).
///  - secret.generic operand: the matching block arg in the body is retyped
///    (unwrapping the secret), and propagation continues into the body.
///  - secret.yield operand: the parent secret.generic's matching result is
///    retyped (preserving secret wrapping), and propagation continues outside.
///  - func.return operand: the function signature's matching result type is
///    updated at the end.
///
/// Finally updates the function signature to reflect the new arg and any
/// changed result types. Plaintext / non-tensor uses are left untouched.
void widenFunctionArgAndPropagate(func::FuncOp func, unsigned argIdx,
                                  Type newType);

void findScheduleMergingCandidates(
    recursiveProgramNode *node,
    DenseMap<recursiveProgramNode *, SmallVector<recursiveProgramNode *>>
        &candidates,
    DenseSet<func::CallOp> &visited);

typedef struct cipherTextSlot_ {
  Operation *op;
  int index;
  int parentDim;
} cipherTextSlot;

SmallVector<cipherTextSlot> createMergedCipherTextMappings(
    RankedTensorType mergedType, SmallVector<Value> subArgs, OpBuilder builder);
Value createNewInsertOpsFromSeedOps(SmallVector<cipherTextSlot> &ctxt,
                                    RankedTensorType mergedType,
                                    OpBuilder builder);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_RECURSIVECALLVECTORIZATION_MERGESCHEDULES_H_
