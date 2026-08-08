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
///
/// **Signature layout (kernel-major, in `funcs` order).**
/// With per-kernel input arity R and per-kernel result arity R_out:
///
///     merged inputs:  [call0_arg0, ..., call0_arg{R-1},
///                      call1_arg0, ..., call1_arg{R-1},
///                      ...,
///                      call{N-1}_arg0, ..., call{N-1}_arg{R-1}]
///                     // merged_input[k*R + i]  ==  funcs[k]'s input i
///
/// Outputs are **scalar SSA values** (one `secret<elemType>` per leaf), not
/// tensors. For each kernel, we walk its yield operands' tensor.insert
/// chains and emit each leaf scalar in original program order. So the
/// result layout is:
///
///     merged results: [k0_leaf0, k0_leaf1, ..., k0_leaf{L0-1},
///                      k1_leaf0, k1_leaf1, ..., k1_leaf{L1-1},
///                      ...,
///                      k{N-1}_leaf0, ..., k{N-1}_leaf{LN-1-1}]
///
/// where Lk is kernel k's total leaf-scalar count (sum over its yield
/// operands). All kernels share the same Lk (because signatures match), so
/// `Lk == L_out` and merged_result[k * L_out + l] == funcs[k]'s leaf l.
///
/// Caller-side rewrite for inputs is purely positional: iterate `funcs` (or
/// the equivalent original-call list) in order and concatenate each call's
/// `getArgOperands()` to form the merged call's arg list. For outputs,
/// downstream consumers (e.g., a reduction kernel) take these scalars
/// directly — no tensor extracts at the boundary.
LogicalResult mergeSchedulesWithNW(
    llvm::ArrayRef<func::FuncOp> funcs, llvm::ArrayRef<Schedule> schedules,
    func::FuncOp &mergedFunc, Schedule &mergedSchedule,
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

/// Pretty-print a Schedule organized by (cycle, lane) for debugging.
///
/// Groups ops by their `alignment` (cycle), then within each cycle sorts by
/// `lanes`. Prints the warp size, total op count, and per-cycle listing of
/// (lane → op) pairs.
void prettyPrintSchedule(const Schedule &schedule,
                         llvm::raw_ostream &os = llvm::outs());

void findScheduleMergingCandidates(
    recursiveProgramNode *node,
    DenseMap<recursiveProgramNode *, SmallVector<recursiveProgramNode *>>
        &candidates,
    DenseSet<recursiveProgramNode *> &visited);

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

LogicalResult mergeSchedulesVertically(llvm::ArrayRef<func::CallOp> funcs,
                                       llvm::ArrayRef<Schedule> schedules,
                                       Schedule &mergedSchedule);

Schedule buildNaiveReductionSchedule(func::FuncOp reductionKernel,
                                     unsigned warpSize);
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_RECURSIVECALLVECTORIZATION_MERGESCHEDULES_H_
