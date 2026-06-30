#ifndef SCALARIZE_CALL_TREE_H
#define SCALARIZE_CALL_TREE_H

//===- ScalarizeFunctionBoundaries.h - Tensor → scalar across calls ----*- C++
//-*-===//
//
// Utility that scalarizes secret<tensor<NxT>> inputs across function call
// boundaries, traversing the call tree from a root function.
//
// Pattern handled:
//
//   Caller (root or any rewritten callee):
//     %src : secret<tensor<MxT>>
//     %ta = secret.generic(%src) ({
//       ^body(%t : tensor<MxT>):
//         %x0 = tensor.extract %t[%c0]
//         %x1 = tensor.extract %t[%c1]
//         %x2 = tensor.extract %t[%c2]
//         %x3 = tensor.extract %t[%c3]
//         %tmp0 = tensor.insert %x0 into %dense_zero[%c0]
//         %tmp1 = tensor.insert %x1 into %tmp0[%c1]
//         %tmp2 = tensor.insert %x2 into %tmp1[%c2]
//         %ta   = tensor.insert %x3 into %tmp2[%c3]
//         secret.yield %ta
//     }) -> secret<tensor<4xT>>
//     %r = call @callee(%ta) : (secret<tensor<4xT>>) -> ...
//
//   Becomes:
//     %src : secret<tensor<MxT>>
//     %s0, %s1, %s2, %s3 = secret.generic(%src) ({
//       ^body(%t : tensor<MxT>):
//         %x0 = tensor.extract %t[%c0]
//         %x1 = tensor.extract %t[%c1]
//         %x2 = tensor.extract %t[%c2]
//         %x3 = tensor.extract %t[%c3]
//         secret.yield %x0, %x1, %x2, %x3
//     }) -> (secret<T>, secret<T>, secret<T>, secret<T>)
//     %r = call @callee(%s0, %s1, %s2, %s3) : (secret<T>, secret<T>, secret<T>,
//     secret<T>) -> ...
//
//   And @callee is rewritten:
//     func @callee(%a0 : secret<T>, %a1 : secret<T>, %a2 : secret<T>, %a3 :
//     secret<T>)
//       %r = secret.generic(%a0, %a1, %a2, %a3) ({
//         ^body(%i0 : T, %i1 : T, %i2 : T, %i3 : T):
//           // uses %i0..%i3 directly, no tensor.extract on a tensor block arg
//       }) -> ...
//
// The root function's signature is never modified. All callees reachable from
// the root via the call tree are scalarized in BFS order. Insert chains that
// become dead are left in place — run --canonicalize / --cse afterward.
//
//===----------------------------------------------------------------------

#include <optional>
#include <queue>
#include <utility>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// Internal helpers (anonymous namespace at use site)
//===----------------------------------------------------------------------===//

/// Walk back through a tensor.insert chain. Returns (slot_index, scalar)
inline llvm::SmallVector<std::pair<int64_t, mlir::Value>> traceInsertChain(
    mlir::Value tensorVal) {
  llvm::SmallVector<std::pair<int64_t, mlir::Value>> entries;
  mlir::Value cur = tensorVal;
  while (auto ins = cur.getDefiningOp<mlir::tensor::InsertOp>()) {
    if (ins.getIndices().size() != 1) return {};
    auto cst =
        ins.getIndices()[0].getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (!cst) return {};
    entries.push_back({cst.value(), ins.getScalar()});
    cur = ins.getDest();
  }
  llvm::sort(entries,
             [](const auto &a, const auto &b) { return a.first < b.first; });
  return entries;
}

struct ProducerInfo {
  mlir::heir::secret::GenericOp generic;
  unsigned resultIdx;
};

/// If `tensorVal` is a result of a secret.generic, return the op and which
/// result index it is. Otherwise nullopt.
inline std::optional<ProducerInfo> findProducingGeneric(mlir::Value tensorVal) {
  auto opResult = llvm::dyn_cast_if_present<mlir::OpResult>(tensorVal);
  if (!opResult) return std::nullopt;
  auto generic =
      llvm::dyn_cast<mlir::heir::secret::GenericOp>(opResult.getOwner());
  if (!generic) return std::nullopt;
  return ProducerInfo{generic, opResult.getResultNumber()};
}

/// Append `newScalars` to the yield of `generic`, rebuilding it via
/// `secret::GenericOp::addNewYieldedValues`. Updates `generic` to the new op
/// and erases the old one. Returns the result indices in the new generic that
/// correspond to the appended values.
inline llvm::SmallVector<unsigned> appendYieldedValues(
    mlir::heir::secret::GenericOp &generic,
    llvm::ArrayRef<mlir::Value> newScalars, mlir::PatternRewriter &rewriter) {
  if (newScalars.empty()) return {};
  unsigned oldCount = generic->getNumResults();
  mlir::heir::secret::GenericOp oldGeneric = generic;
  // Set insertion point at the OLD generic so the new generic is created
  // at the same position. Without this, addNewYieldedValues may place the
  // new generic at the rewriter's default insertion point — which can be
  // AFTER consumers of the generic's results, causing dominance errors.
  rewriter.setInsertionPoint(oldGeneric);
  auto [newGeneric, newResults] =
      oldGeneric.addNewYieldedValues(mlir::ValueRange(newScalars), rewriter);
  for (auto [oldRes, newRes] :
       llvm::zip(oldGeneric->getResults(), newGeneric->getResults()))
    oldRes.replaceAllUsesWith(newRes);
  rewriter.eraseOp(oldGeneric);
  generic = newGeneric;

  llvm::SmallVector<unsigned> indices;
  indices.reserve(newScalars.size());
  for (unsigned i = 0; i < newScalars.size(); ++i)
    indices.push_back(oldCount + i);
  return indices;
}

/// Info captured by rewriteCallSite about which positions of the call were
/// scalarized.
struct ScalarizedCallInfo {
  /// Old arg positions that ended up scalarized in the new call.
  llvm::SmallVector<unsigned> scalarizedOldPositions;
  /// For each old arg position, count of scalar args replacing it in the new
  /// call. 1 = unchanged. >1 = scalarized into that many scalars.
  llvm::SmallVector<unsigned> newCounts;
};

/// Rewrite a single call site so each scalarizable tensor operand is replaced
/// by N scalar operands. Mutates `call` to point at the new call op.
/// Producer-side secret.generic ops are batched per producer to avoid
/// stale references when multiple operands share a producer.
inline ScalarizedCallInfo rewriteCallSite(mlir::func::CallOp &call,
                                          mlir::PatternRewriter &rewriter) {
  ScalarizedCallInfo info;
  info.newCounts.assign(call.getNumOperands(), 1);

  // 1. Plan per producer (no IR mods yet).
  struct OperandPlan {
    unsigned operandIdx;
    llvm::SmallVector<mlir::Value> scalars;
  };
  llvm::DenseMap<mlir::heir::secret::GenericOp, llvm::SmallVector<OperandPlan>>
      byProducer;
  for (unsigned i = 0; i < call.getNumOperands(); ++i) {
    mlir::Value operand = call.getOperand(i);
    auto secretTy =
        llvm::dyn_cast<mlir::heir::secret::SecretType>(operand.getType());
    if (!secretTy ||
        !llvm::isa<mlir::RankedTensorType>(secretTy.getValueType()))
      continue;
    auto producer = findProducingGeneric(operand);
    if (!producer) continue;
    auto yieldOp = llvm::cast<mlir::heir::secret::YieldOp>(
        producer->generic.getRegion().front().getTerminator());
    mlir::Value yieldVal = yieldOp.getOperand(producer->resultIdx);
    auto chain = traceInsertChain(yieldVal);
    if (chain.empty()) continue;
    llvm::SmallVector<mlir::Value> chainScalars;
    chainScalars.reserve(chain.size());
    for (auto &e : chain) chainScalars.push_back(e.second);
    byProducer[producer->generic].push_back({i, std::move(chainScalars)});
  }

  // 2. Process each producer once; record per-operand new-result Values.
  llvm::DenseMap<unsigned, llvm::SmallVector<mlir::Value>> operandToScalars;
  for (auto &kv : byProducer) {
    mlir::heir::secret::GenericOp gen = kv.first;
    llvm::SmallVector<mlir::Value> allScalars;
    llvm::SmallVector<std::pair<unsigned, std::pair<unsigned, unsigned>>>
        spans;  // (operandIdx, (start, count)) in allScalars
    for (auto &plan : kv.second) {
      spans.push_back(
          {plan.operandIdx,
           {(unsigned)allScalars.size(), (unsigned)plan.scalars.size()}});
      for (mlir::Value s : plan.scalars) allScalars.push_back(s);
    }
    auto newIndices = appendYieldedValues(gen, allScalars, rewriter);
    for (auto &span : spans) {
      auto [start, count] = span.second;
      llvm::SmallVector<mlir::Value> scalarResults;
      scalarResults.reserve(count);
      for (unsigned k = 0; k < count; ++k)
        scalarResults.push_back(gen->getResult(newIndices[start + k]));
      operandToScalars[span.first] = std::move(scalarResults);
    }
  }

  // 3. Build the new call's operand list.
  llvm::SmallVector<mlir::Value> newOperands;
  for (unsigned i = 0; i < call.getNumOperands(); ++i) {
    auto it = operandToScalars.find(i);
    if (it == operandToScalars.end()) {
      newOperands.push_back(call.getOperand(i));
    } else {
      for (mlir::Value s : it->second) newOperands.push_back(s);
      info.newCounts[i] = it->second.size();
      info.scalarizedOldPositions.push_back(i);
    }
  }

  if (info.scalarizedOldPositions.empty()) return info;

  // Mutate the existing call's operand list in place rather than creating
  // a new call and erasing the old one. This keeps any external handles
  // to `call` (e.g. stored in caller's data structures) alive. Result
  // types stay the same since this is an input-only scalarization.
  call->setOperands(newOperands);
  return info;
}

/// Rewrite `callee`'s signature so each tensor arg at the given positions is
/// replaced with N scalar args (N = numElements of the tensor). Updates the
/// inner secret.generic operand list, body block args, and rewires
/// tensor.extract ops inside the body to use the corresponding scalar block
/// arg directly. Returns true on success, false if a non-trivial use of an
/// old arg blocks the rewrite.
///
/// Assumptions:
///  - The callee has exactly one secret.generic in its body.
///  - The secret.generic's operand i corresponds to func entry arg i
///    (positional pass-through).
///  - For scalarized tensor args, all uses inside the body are
///    tensor.extract with a single constant index.
inline bool scalarizeCalleeSignature(
    mlir::func::FuncOp callee, llvm::ArrayRef<unsigned> tensorArgPositions) {
  if (tensorArgPositions.empty()) return false;

  mlir::MLIRContext *ctx = callee.getContext();
  mlir::Location loc = callee.getLoc();
  mlir::Block &entry = callee.front();

  // -----------------------------------------------------------------------
  // Commit-on-success rewrite. Phase 1 plans and validates without touching
  // the IR. Phase 2 applies the mutations atomically. If any precondition
  // fails in Phase 1, we return false with the IR unchanged.
  // -----------------------------------------------------------------------

  llvm::DenseSet<unsigned> scalarizeSet(tensorArgPositions.begin(),
                                        tensorArgPositions.end());

  // 1. Compute new function input types and a per-old-arg map (newStart,
  // count).
  llvm::SmallVector<mlir::Type> newInputs;
  llvm::SmallVector<std::pair<unsigned, unsigned>> oldToNew(
      entry.getNumArguments());
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    mlir::BlockArgument oldArg = entry.getArgument(i);
    mlir::Type t = oldArg.getType();
    if (scalarizeSet.count(i)) {
      auto secretTy = llvm::dyn_cast<mlir::heir::secret::SecretType>(t);
      if (!secretTy) {
        llvm::errs() << "scalarize: arg " << i << " of " << callee.getName()
                     << " is not secret-typed\n";
        return false;
      }
      auto tensorTy =
          llvm::dyn_cast<mlir::RankedTensorType>(secretTy.getValueType());
      if (!tensorTy || !tensorTy.hasStaticShape()) {
        llvm::errs() << "scalarize: arg " << i << " of " << callee.getName()
                     << " is not a statically-shaped tensor\n";
        return false;
      }
      unsigned N = tensorTy.getNumElements();
      mlir::Type elemTy = tensorTy.getElementType();
      mlir::Type newSecretTy = mlir::heir::secret::SecretType::get(elemTy);
      oldToNew[i] = {(unsigned)newInputs.size(), N};
      for (unsigned j = 0; j < N; ++j) newInputs.push_back(newSecretTy);
    } else {
      oldToNew[i] = {(unsigned)newInputs.size(), 1};
      newInputs.push_back(t);
    }
  }

  // ---------------- Phase 1: plan + validate (no IR mutations) -----------

  // 1a. Find the inner secret.generic to rewrite. Pick the FIRST generic
  //     (by walk order) whose operands include any of the scalarizable
  //     outer args. A callee may legitimately contain multiple generics
  //     (e.g. a "splitter" at the top consuming outer args, and an
  //     "accumulator" at the bottom consuming call results) — the original
  //     `walk-and-overwrite` would pick the last one, which is wrong.
  mlir::heir::secret::GenericOp innerGeneric;
  callee.walk([&](mlir::heir::secret::GenericOp g) -> mlir::WalkResult {
    for (mlir::Value op : g->getOperands()) {
      if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(op)) {
        if (ba.getOwner() == &entry && scalarizeSet.count(ba.getArgNumber())) {
          innerGeneric = g;
          return mlir::WalkResult::interrupt();
        }
      }
    }
    return mlir::WalkResult::advance();
  });

  // 1b. Validate scalarizable outer args have only "safe" uses
  //     (= used solely by `innerGeneric` as an operand, or no uses at all
  //      if there is no inner generic). If something else references them,
  //      we can't safely scalarize without leaving dangling refs.
  for (unsigned i : tensorArgPositions) {
    mlir::BlockArgument arg = entry.getArgument(i);
    for (mlir::OpOperand &use : arg.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (innerGeneric && user == innerGeneric.getOperation()) continue;
      llvm::errs() << "scalarize: callee " << callee.getName()
                   << " has non-trivial use of outer arg " << i
                   << " (user op: " << user->getName()
                   << ") — refusing to scalarize\n";
      return false;
    }
  }

  // 1c. Plan inner-side rewrites: for each operand of the inner generic
  //     that came from a scalarizable outer arg, validate the corresponding
  //     inner block arg's uses are only constant-indexed tensor.extracts,
  //     and record which extract feeds which slot for Phase 2 to rewire.
  struct InnerOperandPlan {
    unsigned newStart;  // index into newInnerArgTypes / newGenericOperands.
    unsigned count;     // 1 if not scalarized, N otherwise.
    // For scalarized (count > 1): per-slot extract op to RAUW-and-erase.
    llvm::SmallVector<mlir::Operation *> extractOps;
  };

  mlir::Block *innerBody =
      innerGeneric ? &innerGeneric.getRegion().front() : nullptr;
  unsigned origInnerCount = innerBody ? innerBody->getNumArguments() : 0;
  llvm::SmallVector<InnerOperandPlan> innerPlans(origInnerCount);
  llvm::SmallVector<mlir::Type> newInnerArgTypes;

  if (innerGeneric) {
    for (unsigned i = 0; i < innerGeneric->getNumOperands(); ++i) {
      mlir::Value oldOperand = innerGeneric->getOperand(i);
      auto oldArg = llvm::dyn_cast<mlir::BlockArgument>(oldOperand);
      if (!oldArg || oldArg.getOwner() != &entry ||
          !scalarizeSet.count(oldArg.getArgNumber())) {
        // Operand not scalarized — passes through unchanged.
        innerPlans[i].newStart = (unsigned)newInnerArgTypes.size();
        innerPlans[i].count = 1;
        newInnerArgTypes.push_back(innerBody->getArgument(i).getType());
        continue;
      }
      auto [newStart, count] = oldToNew[oldArg.getArgNumber()];
      innerPlans[i].newStart = (unsigned)newInnerArgTypes.size();
      innerPlans[i].count = count;
      auto innerTensorTy = llvm::dyn_cast<mlir::RankedTensorType>(
          innerBody->getArgument(i).getType());
      if (!innerTensorTy) {
        llvm::errs() << "scalarize: inner arg " << i << " of "
                     << callee.getName() << " is not a ranked tensor\n";
        return false;
      }
      mlir::Type elemTy = innerTensorTy.getElementType();

      // Validate uses of this inner arg + map each extract to its slot.
      llvm::SmallVector<mlir::Operation *> perSlot(count, nullptr);
      mlir::BlockArgument innerArg = innerBody->getArgument(i);
      for (mlir::OpOperand &use : innerArg.getUses()) {
        auto ext = llvm::dyn_cast<mlir::tensor::ExtractOp>(use.getOwner());
        if (!ext) {
          llvm::errs() << "scalarize: callee " << callee.getName()
                       << " has non-extract use of inner arg " << i
                       << " (user: " << use.getOwner()->getName()
                       << ") — refusing to scalarize\n";
          return false;
        }
        auto cst =
            ext.getIndices().front().getDefiningOp<mlir::arith::ConstantOp>();
        if (!cst) {
          llvm::errs() << "scalarize: callee " << callee.getName()
                       << " has dynamic-index extract on inner arg " << i
                       << " — refusing to scalarize\n";
          return false;
        }
        unsigned slot = llvm::cast<mlir::IntegerAttr>(cst.getValue()).getInt();
        if (slot >= count) {
          llvm::errs() << "scalarize: callee " << callee.getName()
                       << " extract slot " << slot
                       << " out of range on inner arg " << i << "\n";
          return false;
        }
        perSlot[slot] = ext.getOperation();
      }
      innerPlans[i].extractOps = std::move(perSlot);
      for (unsigned j = 0; j < count; ++j) newInnerArgTypes.push_back(elemTy);
    }
  }

  // ---------------- Phase 2: apply (no validation failures past here) ----

  // 2a. Append new outer args to entry block.
  unsigned origCount = entry.getNumArguments();
  for (mlir::Type t : newInputs) entry.addArgument(t, loc);

  // 2b. No-inner-generic short path: RAUW unchanged args, erase originals,
  //     update signature.
  if (!innerGeneric) {
    for (unsigned i = 0; i < origCount; ++i) {
      mlir::BlockArgument oldArg = entry.getArgument(i);
      auto [newStart, count] = oldToNew[i];
      if (count == 1)
        oldArg.replaceAllUsesWith(entry.getArgument(origCount + newStart));
      // Scalarized args have no uses (Phase 1 checked) — nothing to do.
    }
    for (unsigned i = origCount; i-- > 0;) entry.eraseArgument(i);
    callee.setType(mlir::FunctionType::get(
        ctx, newInputs, callee.getFunctionType().getResults()));
    return true;
  }

  // 2c. Build the new generic operand list using the newly-added entry args.
  llvm::SmallVector<mlir::Value> newGenericOperands;
  newGenericOperands.reserve(newInnerArgTypes.size());
  for (unsigned i = 0; i < innerGeneric->getNumOperands(); ++i) {
    mlir::Value oldOperand = innerGeneric->getOperand(i);
    auto oldArg = llvm::dyn_cast<mlir::BlockArgument>(oldOperand);
    if (!oldArg || oldArg.getOwner() != &entry ||
        !scalarizeSet.count(oldArg.getArgNumber())) {
      // Unchanged operand. If it's a non-scalarized entry arg, redirect to
      // its new position; otherwise keep as-is.
      if (oldArg && oldArg.getOwner() == &entry) {
        auto [newStart, _count] = oldToNew[oldArg.getArgNumber()];
        newGenericOperands.push_back(entry.getArgument(origCount + newStart));
      } else {
        newGenericOperands.push_back(oldOperand);
      }
      continue;
    }
    auto [newStart, count] = oldToNew[oldArg.getArgNumber()];
    for (unsigned j = 0; j < count; ++j)
      newGenericOperands.push_back(entry.getArgument(origCount + newStart + j));
  }

  // 2d. Add new inner block args.
  llvm::SmallVector<mlir::BlockArgument> newInnerArgs;
  newInnerArgs.reserve(newInnerArgTypes.size());
  for (mlir::Type t : newInnerArgTypes)
    newInnerArgs.push_back(innerBody->addArgument(t, loc));

  // 2e. Rewire inner extracts / replace passthrough inner block args.
  llvm::SmallVector<mlir::Operation *> toErase;
  for (unsigned i = 0; i < origInnerCount; ++i) {
    mlir::BlockArgument oldArg = innerBody->getArgument(i);
    auto &plan = innerPlans[i];
    if (plan.count == 1) {
      oldArg.replaceAllUsesWith(newInnerArgs[plan.newStart]);
      continue;
    }
    for (unsigned slot = 0; slot < plan.count; ++slot) {
      mlir::Operation *ext = plan.extractOps[slot];
      if (!ext) continue;  // some slots may have no extract (unused element).
      ext->getResult(0).replaceAllUsesWith(newInnerArgs[plan.newStart + slot]);
      toErase.push_back(ext);
    }
  }
  for (mlir::Operation *op : toErase) op->erase();

  // 2f. Update secret.generic operand list — drops references to old outer
  //     args.
  innerGeneric->setOperands(newGenericOperands);

  // 2g. Erase old inner block args. Phase 1 validated only extracts use
  //     them, and 2e converted those, so all should be use-empty.
  for (unsigned i = origInnerCount; i-- > 0;) {
    mlir::BlockArgument arg = innerBody->getArgument(i);
    if (!arg.use_empty()) {
      llvm::errs() << "scalarize: BUG — inner arg " << i << " of "
                   << callee.getName() << " has residual uses after planning\n";
      return false;
    }
    innerBody->eraseArgument(i);
  }

  // 2h. Erase old outer block args. Phase 1 validated only the inner generic
  //     uses them; 2f rewired the generic, so all should be use-empty.
  for (unsigned i = origCount; i-- > 0;) {
    mlir::BlockArgument arg = entry.getArgument(i);
    if (!arg.use_empty()) {
      llvm::errs() << "scalarize: BUG — outer arg " << i << " of "
                   << callee.getName() << " has residual uses after planning\n";
      return false;
    }
    entry.eraseArgument(i);
  }

  // 2i. Update function signature.
  callee.setType(mlir::FunctionType::get(
      ctx, newInputs, callee.getFunctionType().getResults()));
  return true;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

/// Scalarize tensor inputs of all functions reachable from `root`, leaving
/// `root`'s signature untouched. For each callee, every
/// secret<tensor<NxT>> argument whose source is a constant-indexed
/// tensor.insert chain is replaced with N secret<T> arguments. Producer-side
/// secret.generic ops are updated to yield the underlying scalars; call sites
/// are updated to pass scalars; callees are rewritten to consume scalars.
///
/// Propagates through the call tree in BFS order. Visits each function at
/// most once. If a callee receives multiple tensor args from the same
/// producer, all are batched into a single producer rewrite.
///
/// This function must be called with a `PatternRewriter &` (the kind handed
/// out by `applyPatternsGreedily` etc.) because the underlying
/// `secret::GenericOp::addNewYieldedValues` API requires one. See
/// `ScalarizeBoundariesPattern` below for a one-shot OpRewritePattern
/// wrapper if you want to call it from `runOnOperation`.
///
/// Run --canonicalize / --cse afterward to clean up the now-dead
/// tensor.insert chains.
inline void scalarizeBoundariesFromRoot(mlir::func::FuncOp root,
                                        mlir::PatternRewriter &rewriter) {
  if (!root || root.empty()) return;

  llvm::DenseSet<mlir::func::FuncOp> visited;
  // Callees whose signature has already been scalarized this pass. After
  // the first scalarization, subsequent call sites to the same callee
  // still get their operands rewritten (so they match the new signature),
  // but the callee body itself isn't re-touched.
  llvm::DenseSet<mlir::func::FuncOp> scalarizedCallees;
  std::queue<mlir::func::FuncOp> work;
  work.push(root);

  while (!work.empty()) {
    mlir::func::FuncOp current = work.front();
    work.pop();
    if (!visited.insert(current).second) continue;
    if (current.empty()) continue;

    llvm::SmallVector<mlir::func::CallOp> calls;
    current.walk([&](mlir::func::CallOp c) { calls.push_back(c); });

    for (mlir::func::CallOp call : calls) {
      auto calleeSym = call.getCalleeAttr();
      auto callee = llvm::dyn_cast_if_present<mlir::func::FuncOp>(
          mlir::SymbolTable::lookupNearestSymbolFrom(call, calleeSym));
      if (!callee) continue;
      if (callee == current) continue;
      if (callee.isExternal()) continue;
      if (callee.getName().starts_with("__")) continue;

      auto info = rewriteCallSite(call, rewriter);
      if (info.scalarizedOldPositions.empty()) {
        // Nothing scalarized at this site; still enqueue callee so its own
        // child calls get visited.
        work.push(callee);
        continue;
      }

      // Only rewrite the callee signature once per pass. Subsequent call
      // sites to the same callee have already had their operands fixed up
      // by rewriteCallSite above; the callee's signature is already scalar.
      if (scalarizedCallees.insert(callee).second) {
        if (!scalarizeCalleeSignature(callee, info.scalarizedOldPositions)) {
          llvm::errs() << "scalarize: callee " << callee.getName()
                       << " signature rewrite failed; call site already "
                          "rewritten — IR is in an inconsistent state\n";
        }
      }
      work.push(callee);
    }
  }
}

//===----------------------------------------------------------------------===//
// Internal driver pattern (not part of the public API).
//===----------------------------------------------------------------------===//

/// One-shot pattern that scalarizes the configured target function. Reports
/// success only when an actual change happens, so the greedy driver knows
/// to stop after one effective pass.
class ScalarizeBoundariesPattern
    : public mlir::OpRewritePattern<mlir::func::FuncOp> {
  mlir::func::FuncOp target;

 public:
  ScalarizeBoundariesPattern(mlir::MLIRContext *ctx, mlir::func::FuncOp target)
      : mlir::OpRewritePattern<mlir::func::FuncOp>(ctx), target(target) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::func::FuncOp func, mlir::PatternRewriter &rewriter) const override {
    if (func != target) return mlir::failure();
    auto countScalarizableTensorOperands = [&]() {
      unsigned n = 0;
      func.walk([&](mlir::func::CallOp c) {
        for (mlir::Value op : c.getOperands()) {
          auto secretTy =
              llvm::dyn_cast<mlir::heir::secret::SecretType>(op.getType());
          if (secretTy &&
              llvm::isa<mlir::RankedTensorType>(secretTy.getValueType()))
            ++n;
        }
      });
      return n;
    };
    unsigned before = countScalarizableTensorOperands();
    scalarizeBoundariesFromRoot(func, rewriter);
    unsigned after = countScalarizableTensorOperands();
    return (before != after) ? mlir::success() : mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Single-function entry point.
//===----------------------------------------------------------------------===//

/// Scalarize tensor inputs across the call tree rooted at `root`. The root
/// signature is preserved; everything reachable from it via func.call is
/// rewritten. Internally spins up a one-shot pattern + greedy driver so the
/// underlying PatternRewriter requirement of secret::GenericOp is satisfied.
///
/// Just call this from your pass:
///
///   scalarizeBoundariesFromRoot(rootFunc);
///
/// Run --canonicalize / --cse afterward to clean up dead tensor.insert chains.
inline void scalarizeBoundariesFromRoot(mlir::func::FuncOp root) {
  if (!root || root.empty()) return;
  mlir::MLIRContext *ctx = root.getContext();
  mlir::ModuleOp module = root->getParentOfType<mlir::ModuleOp>();
  if (!module) return;

  mlir::RewritePatternSet patterns(ctx);
  patterns.add<ScalarizeBoundariesPattern>(ctx, root);
  (void)mlir::applyPatternsGreedily(module, std::move(patterns));
}

}  // namespace heir
}  // namespace mlir

#endif  // SCALARIZE_CALL_TREE_H
