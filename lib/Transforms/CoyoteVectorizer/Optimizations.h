//===- OptimizationPasses.cpp - Blend and rotation optimization ----------===//
//
// Port of Python's relax_blends (SA-based) and better_rotations.
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "GraphUtils.h"
#include "GreedyAlign.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
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
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// BlendOptimizer: SA-based blend minimization
// Python: blend_alignment.py:52-128 (relax_blends)
//===----------------------------------------------------------------------===//

class BlendOptimizer {
 public:
  BlendOptimizer() : rng(std::random_device{}()) {}

  /// Optimize schedule to reduce blend operations using simulated annealing.
  /// Python: relax_blends(schedule, rounds=1000, beta=0.05, t=10)
  void optimize(Schedule &schedule, unsigned rounds = 1000, double beta = 0.05,
                double t = 10.0) {
    buildDependences(schedule);

    double current = countBlends(schedule.instructions, schedule.alignment,
                                 schedule.maxStep());

    std::uniform_int_distribution<int64_t> stepDist(0, schedule.maxStep());
    std::uniform_real_distribution<double> realDist(0.0, 1.0);

    for (unsigned round = 0; round < rounds; ++round) {
      if (current == 0.0) break;

      // Cool temperature: t /= (1 + t * beta)
      t /= (1.0 + t * beta);

      // Pick a random step
      int64_t step = stepDist(rng);
      auto opsAtStep = schedule.getStep(step);
      if (opsAtStep.empty()) continue;

      // With 50% probability take lhs operand producers, else rhs.
      // Python: if random() < 0.5: lhs else rhs
      bool useLhs = realDist(rng) < 0.5;
      unsigned operandIdx = useLhs ? 0 : 1;

      // Collect producer ops for the chosen operand slot across opsAtStep.
      llvm::SmallVector<Operation *> operandProducers;
      for (auto *op : opsAtStep) {
        if (op->getNumOperands() <= operandIdx) continue;
        Value v = op->getOperand(operandIdx);
        Operation *prod = v.getDefiningOp();
        if (prod && schedule.alignment.count(prod))
          operandProducers.push_back(prod);
      }

      if (operandProducers.empty()) continue;

      // Skip if any producer transitively depends on another in the list.
      // Python: if len(operations) and independent(operations):
      if (!isIndependent(operandProducers)) continue;

      // Build candidate alignment (copy of current).
      llvm::DenseMap<Operation *, int64_t> candAlign = schedule.alignment;

      // Group producers by opcode; for each group try to move all to same step.
      llvm::DenseMap<llvm::StringRef, llvm::SmallVector<Operation *>> grouped;
      for (auto *op : operandProducers)
        grouped[op->getName().getStringRef()].push_back(op);

      for (auto &[opcode, group] : grouped) {
        // Pick a random target step from within the group.
        // Python: new_step = choice([schedule.alignment[g] for g in group])
        std::uniform_int_distribution<size_t> groupDist(0, group.size() - 1);
        int64_t newStep = candAlign.lookup(group[groupDist(rng)]);

        for (auto *o : group) {
          int64_t oLane = schedule.lanes.lookup(o);
          int64_t oOldStep = candAlign.lookup(o);
          if (oOldStep == newStep) continue;

          // Find the incumbent at (newStep, oLane) — the op displaced by o.
          // Python: incumbents = candidate.at_step(new_step)
          //                        .intersection(candidate.at_lane(lanes[o]))
          Operation *incumbent = nullptr;
          for (auto *other : schedule.instructions) {
            if (other == o) continue;
            if (candAlign.lookup(other) == newStep &&
                schedule.lanes.lookup(other) == oLane) {
              incumbent = other;
              break;
            }
          }

          // Constraint checks (Python: lines 96-103 in blend_alignment.py):
          // Incumbent (if any) must be movable to oOldStep.
          if (incumbent) {
            if (!allProducersBefore(incumbent, oOldStep, candAlign)) continue;
            if (!allConsumersAfter(incumbent, oOldStep, candAlign)) continue;
          }
          // o must be movable to newStep.
          if (!allProducersBefore(o, newStep, candAlign)) continue;
          if (!allConsumersAfter(o, newStep, candAlign)) continue;

          // Apply the swap in the candidate.
          if (incumbent) candAlign[incumbent] = oOldStep;
          candAlign[o] = newStep;
        }
      }

      // Evaluate candidate cost and SA accept/reject.
      double newCost =
          countBlends(schedule.instructions, candAlign, schedule.maxStep());
      if (newCost < current ||
          realDist(rng) < std::exp((current - newCost) / t)) {
        schedule.alignment = candAlign;
        current = newCost;
      }
    }
  }

 private:
  std::mt19937 rng;

  // Transitive producers: op -> all ops it (transitively) depends on.
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> producers;
  // Direct+transitive consumers: op -> all ops that (transitively) use it.
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> consumers;

  /// Build transitive producer/consumer sets in program (topological) order.
  /// Python: blend_alignment.py:23-38 (get_dependences)
  ///
  /// Because schedule.instructions is in program order, when we process op i
  /// all its dependencies have already been processed — so we can inherit
  /// their transitive sets by union.
  void buildDependences(const Schedule &schedule) {
    producers.clear();
    consumers.clear();

    // Temporary sets for fast membership queries during construction.
    llvm::DenseMap<Operation *, llvm::DenseSet<Operation *>> prodSet;

    for (auto *op : schedule.instructions) {
      for (Value operand : op->getOperands()) {
        Operation *prod = operand.getDefiningOp();
        if (!prod || !schedule.alignment.count(prod)) continue;

        prodSet[op].insert(prod);
        for (auto *trans : prodSet[prod]) prodSet[op].insert(trans);
      }
    }

    // Materialise into SmallVector for cheap iteration later.
    for (auto &[op, deps] : prodSet) {
      producers[op].assign(deps.begin(), deps.end());
      for (auto *dep : deps) consumers[dep].push_back(op);
    }
  }

  /// Count total blends needed across all steps.
  /// Python: blend_alignment.py:10-20 (count_blends)
  ///
  /// For each step, collect the set of source steps for lhs operands and rhs
  /// operands separately. Blend cost = max(0, |lhs_steps|-1)
  ///                                 + max(0, |rhs_steps|-1).
  double countBlends(llvm::ArrayRef<Operation *> instructions,
                     const llvm::DenseMap<Operation *, int64_t> &align,
                     int64_t maxStep) const {
    double blends = 0;
    for (int64_t step = 0; step <= maxStep; ++step) {
      llvm::DenseSet<int64_t> lhsSrcs, rhsSrcs;
      for (auto *op : instructions) {
        if (align.lookup(op) != step) continue;
        if (op->getNumOperands() > 0) {
          if (auto *prod = op->getOperand(0).getDefiningOp())
            if (align.count(prod)) lhsSrcs.insert(align.lookup(prod));
        }
        if (op->getNumOperands() > 1) {
          if (auto *prod = op->getOperand(1).getDefiningOp())
            if (align.count(prod)) rhsSrcs.insert(align.lookup(prod));
        }
      }
      if (lhsSrcs.size() > 1) blends += lhsSrcs.size() - 1;
      if (rhsSrcs.size() > 1) blends += rhsSrcs.size() - 1;
    }
    return blends;
  }

  /// Check that no op in the list is a transitive producer of any other.
  /// Python: independent(ops) — blend_alignment.py:55-56
  bool isIndependent(llvm::ArrayRef<Operation *> ops) const {
    llvm::DenseSet<Operation *> opSet(ops.begin(), ops.end());
    for (auto *op : ops) {
      auto it = producers.find(op);
      if (it == producers.end()) continue;
      for (auto *dep : it->second)
        if (opSet.count(dep)) return false;
    }
    return true;
  }

  /// All transitive producers of op must be at steps strictly before `step`.
  bool allProducersBefore(
      Operation *op, int64_t step,
      const llvm::DenseMap<Operation *, int64_t> &align) const {
    auto it = producers.find(op);
    if (it == producers.end()) return true;
    for (auto *prod : it->second)
      if (align.lookup(prod) >= step) return false;
    return true;
  }

  /// All consumers of op must be at steps strictly after `step`.
  bool allConsumersAfter(
      Operation *op, int64_t step,
      const llvm::DenseMap<Operation *, int64_t> &align) const {
    auto it = consumers.find(op);
    if (it == consumers.end()) return true;
    for (auto *consumer : it->second)
      if (align.lookup(consumer) <= step) return false;
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Standalone wrappers
//===----------------------------------------------------------------------===//

inline void optimizeBlends(Schedule &schedule, unsigned rounds) {
  BlendOptimizer optimizer;
  optimizer.optimize(schedule, rounds);
}

//===----------------------------------------------------------------------===//
// MLIR Code Generation directly from Schedule
//===----------------------------------------------------------------------===//

/// Lower the schedule directly to MLIR tensor operations.
///
/// Algorithm:
///   For each time step (in order):
///     1. Collect ops at this step. Skip pure-extract steps (inputs are already
///        tensors — the block-arg IS the vector).
///     2. For each operand slot (lhs, rhs):
///          a. Group ops by source vector (opVec[producer]).
///          b. Single source  → tensor_ext.rotate(srcVec, shift) for cross-lane
///          alignment. c. Multiple sources → blend: mask1*vec1 + mask2*vec2 (no
///          rotation inside blend).
///             Masks are dense integer tensor constants.
///     3. Emit the element-wise arith op on tensors.
///   After all steps, replace each scalar op's uses with
///   tensor.extract %vecAtStep[step][lane] and erase the now-dead scalar ops.
///
/// Rotation convention: tensor_ext.rotate is a LEFT rotation.
///   result[i] = src[(i + shift) % W]
/// To bring prodLane's value to consLane:
///   result[consLane] = src[consLane + shift] = src[prodLane]
///   => shift = prodLane - consLane
/// (Negative shift = right rotation, which is equivalent.)
void lowerToMLIR(func::FuncOp func, const Schedule &schedule) {
  if (schedule.instructions.empty()) return;

  unsigned W = schedule.warpSize;
  MLIRContext *ctx = func.getContext();
  Location loc = func.getLoc();
  OpBuilder builder(ctx);

  // Insert new ops just before the terminator of the block that contains the
  // scheduled ops. Scheduled ops may live inside a nested region (e.g.
  // secret.generic), so we must insert there — not at the outer func
  // terminator.
  Block *scheduleBlock = schedule.instructions.front()->getBlock();
  builder.setInsertionPoint(scheduleBlock->getTerminator());

  // --- opVec: op -> the tensor<1xW x T> that holds op's result at its lane ---
  llvm::DenseMap<Operation *, Value> opVec;

  // Determine element type from the first scheduled op.
  Type elemType;
  for (auto *op : schedule.instructions) {
    if (!op->getResults().empty()) {
      elemType = op->getResult(0).getType();
      break;
    }
  }
  auto vecType = RankedTensorType::get({1, (int64_t)W}, elemType);

  // Helpers for __coyote_load identification.
  auto isCoyoteLoad = [](Operation *op) -> bool {
    auto call = dyn_cast_if_present<func::CallOp>(op);
    return call && call.getCallee() == "__coyote_load";
  };
  // If `op` is a __coyote_load that wraps a tensor.extract, return the extract.
  auto loadInputExtract = [&](Operation *op) -> tensor::ExtractOp {
    if (!isCoyoteLoad(op)) return {};
    auto call = cast<func::CallOp>(op);
    return call->getOperand(0).getDefiningOp<tensor::ExtractOp>();
  };

  // --- Group scheduled "input load" ops by (source tensor, cycle). ---
  // Splitting by cycle is essential: a single lane may receive different
  // scalars at different cycles, so each cycle needs its own packed input
  // ciphertext. An input load is either a tensor.extract or a __coyote_load
  // wrapping an extract; both are treated uniformly.
  llvm::DenseMap<std::pair<Value, int64_t>, llvm::SmallVector<Operation *>>
      srcCycleToExtracts;
  llvm::SetVector<Value> distinctSources;
  for (auto *op : schedule.instructions) {
    Value source;
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
      source = extractOp.getTensor();
    } else if (auto ext = loadInputExtract(op)) {
      source = ext.getTensor();
    } else {
      continue;
    }
    int64_t cycle = schedule.alignment.lookup(op);
    srcCycleToExtracts[{source, cycle}].push_back(op);
    distinctSources.insert(source);
  }

  auto secretVecType = secret::SecretType::get(vecType);
  Operation *genericOp = scheduleBlock->getParentOp();

  // --- Capture original per-source info BEFORE any mutation ---
  // Each distinct source is a body block arg of secret.generic, mapped 1:1
  // to an outer func arg via genericOp's operand at the same index.
  struct OrigSourceInfo {
    BlockArgument bodyArg;   // body block arg (typed as the bare tensor)
    Type origPlaintextType;  // body block arg type (bare, no `secret`)
    unsigned operandIdx;     // operand index in genericOp / func arg idx
  };
  llvm::SmallVector<OrigSourceInfo> origInfos;
  for (Value src : distinctSources) {
    auto bodyArg = dyn_cast<BlockArgument>(src);
    if (!bodyArg) continue;
    OrigSourceInfo info;
    info.bodyArg = bodyArg;
    info.origPlaintextType = bodyArg.getType();
    info.operandIdx = bodyArg.getArgNumber();
    origInfos.push_back(info);
  }
  llvm::DenseMap<BlockArgument, OrigSourceInfo *> bodyArgToInfo;
  for (auto &info : origInfos) bodyArgToInfo[info.bodyArg] = &info;

  // --- Plan buckets, one per (source, cycle), in stable order ---
  struct Bucket {
    OrigSourceInfo *info;
    int64_t cycle;
    llvm::SmallVector<Operation *> inputLoadOps;
    DenseIntElementsAttr layoutAttr;
  };
  llvm::SmallVector<Bucket> buckets;
  for (auto &[key, ops] : srcCycleToExtracts) {
    auto bodyArg = dyn_cast<BlockArgument>(key.first);
    if (!bodyArg || !bodyArgToInfo.count(bodyArg)) continue;
    Bucket b;
    b.info = bodyArgToInfo[bodyArg];
    b.cycle = key.second;
    b.inputLoadOps = ops;
    // Build the slot→lane permutation attribute for this bucket.
    SmallVector<int64_t> layoutData;
    int64_t numMappings = 0;
    for (auto *op : ops) {
      tensor::ExtractOp extractOp;
      if (auto e = dyn_cast<tensor::ExtractOp>(op))
        extractOp = e;
      else
        extractOp = loadInputExtract(op);
      if (!extractOp) continue;
      auto indices = extractOp.getIndices();
      auto constIdx = indices.back().getDefiningOp<arith::ConstantIndexOp>();
      int64_t slot = constIdx ? constIdx.value() : 0;
      int64_t lane = schedule.lanes.lookup(op);
      layoutData.push_back(0);
      layoutData.push_back(slot);
      layoutData.push_back(0);
      layoutData.push_back(lane);
      ++numMappings;
    }
    auto layoutAttrType =
        RankedTensorType::get({numMappings, 4}, builder.getI64Type());
    b.layoutAttr = DenseIntElementsAttr::get(layoutAttrType, layoutData);
    buckets.push_back(b);
  }
  llvm::sort(buckets, [](const Bucket &a, const Bucket &b) {
    if (a.info->operandIdx != b.info->operandIdx)
      return a.info->operandIdx < b.info->operandIdx;
    return a.cycle < b.cycle;
  });

  // --- Refactor kernel signature: one secret-wide arg per bucket ---
  // For each bucket: append a new func arg of secretVecType annotated with
  // tensor_ext.original_type<originalBareType, bucketLayout>, append the
  // matching new operand to secret.generic, and append a new body block arg
  // of vecType. Wire opVec for each input-load op to its bucket's body arg.
  // AddClientInterface will read the OriginalTypeAttrs and emit one
  // encryption helper per bucket — the boundary layout work moves to the
  // client interface, and the kernel body sees pre-packed ciphertexts.
  llvm::SmallVector<BlockArgument> bucketBodyArgs;
  bucketBodyArgs.reserve(buckets.size());
  StringAttr origTypeAttrName =
      StringAttr::get(ctx, "tensor_ext.original_type");
  // Attach only `tensor_ext.original_type`. We deliberately do NOT attach
  // `tensor_ext.layout` here — LayoutPropagation will fill that with an
  // identity LayoutAttr by default, which is fine (body propagation is
  // lane-wise). The boundary permutation lives entirely in original_type
  // and is consumed only by AddClientInterface, which generates per-bucket
  // encrypt helpers from it. Requires the upstream ConvertFunc patch that
  // skips the original_type overwrite when one is already set.
  for (const Bucket &bucket : buckets) {
    auto origTypeAttr = tensor_ext::OriginalTypeAttr::get(
        ctx, bucket.info->origPlaintextType, bucket.layoutAttr);
    auto argAttrs = DictionaryAttr::get(
        ctx, {NamedAttribute(origTypeAttrName, origTypeAttr)});

    unsigned newFuncIdx = func.getNumArguments();
    func.insertArgument(newFuncIdx, secretVecType, argAttrs, loc);
    Value newFuncArg = func.getArgument(newFuncIdx);

    // Append as new operand to the secret.generic (variadic $inputs).
    SmallVector<Value> newOperands(genericOp->getOperands().begin(),
                                   genericOp->getOperands().end());
    newOperands.push_back(newFuncArg);
    genericOp->setOperands(newOperands);

    BlockArgument newBodyArg = scheduleBlock->addArgument(vecType, loc);
    bucketBodyArgs.push_back(newBodyArg);
  }

  // Update secret.generic result type and the func return type.
  genericOp->getResult(0).setType(secretVecType);
  SmallVector<Type> newFuncArgTypes;
  for (unsigned i = 0; i < func.getNumArguments(); ++i)
    newFuncArgTypes.push_back(func.getArgument(i).getType());
  func.setType(FunctionType::get(ctx, newFuncArgTypes, {secretVecType}));

  // Wire opVec: each input-load op resolves to its bucket's body block arg.
  llvm::DenseMap<Operation *, int64_t> extractLayoutLane;
  for (size_t i = 0; i < buckets.size(); ++i) {
    BlockArgument bodyArg = bucketBodyArgs[i];
    for (Operation *op : buckets[i].inputLoadOps) {
      opVec[op] = bodyArg;
      extractLayoutLane[op] = schedule.lanes.lookup(op);
    }
  }

  // Stash the count of original args so we can erase them after the rest of
  // the lowering finishes (the original tensor.extracts / __coyote_load calls
  // that still reference them get DCE'd by the existing dead-IR sweep below).
  unsigned origArgCount = origInfos.size();

  // --- Rotation cache ---
  using RotKey = std::pair<void *, int64_t>;
  std::map<RotKey, Value> rotCache;

  auto valuePtr = [](Value v) -> void * { return v.getAsOpaquePointer(); };

  auto getRotated = [&](Value vec, int64_t shift) -> Value {
    if (shift == 0) return vec;
    RotKey key{valuePtr(vec), shift};
    auto it = rotCache.find(key);
    if (it != rotCache.end()) return it->second;

    // Emit the shift as `index` (not i32). tensor_ext::RotateOp attaches the
    // IndexTypesNeedNoLayoutImpl interface which tells layout-propagation
    // that an `index`-typed shift does not need a layout. Without a layout,
    // convert-to-ciphertext-semantics leaves the scalar shift alone instead
    // of widening it to tensor<1x64xi32>, which would break the eventual
    // bgv.rotate_cols verifier (it expects a scalar dynamic_shift).
    Value shiftVal =
        arith::ConstantIndexOp::create(builder, loc, shift).getResult();
    Value rotated =
        tensor_ext::RotateOp::create(builder, loc, vec, shiftVal).getResult();
    rotCache[key] = rotated;
    return rotated;
  };

  // --- buildOperandVec ---
  // Group by (source_ptr, shift) so that multiple consumers needing different
  // rotations of the same source vector are handled correctly via blending.
  using ShiftKey = std::pair<void *, int64_t>;
  auto buildOperandVec = [&](llvm::ArrayRef<Operation *> opsAtStep,
                             unsigned opIdx) -> Value {
    std::map<ShiftKey, llvm::SmallVector<int64_t>> keyToLanes;
    std::map<void *, Value> ptrToVec;

    for (auto *op : opsAtStep) {
      if (opIdx >= op->getNumOperands()) continue;
      Operation *prod = op->getOperand(opIdx).getDefiningOp();
      if (!prod || !opVec.count(prod)) continue;

      int64_t consLane = schedule.lanes.lookup(op);
      // For extract producers, use the layout lane (= slot index) since that's
      // where assign_layout physically placed the value.
      int64_t prodLane = isa<tensor::ExtractOp>(prod)
                             ? extractLayoutLane.lookup(prod)
                             : schedule.lanes.lookup(prod);
      int64_t shift = prodLane - consLane;

      Value src = opVec[prod];
      void *ptr = valuePtr(src);
      ShiftKey key{ptr, shift};
      keyToLanes[key].push_back(consLane);
      ptrToVec[ptr] = src;
    }

    if (keyToLanes.empty()) return nullptr;

    // Single (source, shift) — no blend needed
    if (keyToLanes.size() == 1) {
      auto &[key, lanes] = *keyToLanes.begin();
      auto &[ptr, shift] = key;
      (void)lanes;
      return getRotated(ptrToVec[ptr], shift);
    }

    // Multiple (source, shift) combinations — blend with masks
    Value blendResult;
    auto zeroAttr = builder.getIntegerAttr(elemType, 0);
    auto oneAttr = builder.getIntegerAttr(elemType, 1);
    for (auto &[key, lanes] : keyToLanes) {
      auto &[ptr, shift] = key;
      Value rotated = getRotated(ptrToVec[ptr], shift);

      SmallVector<Attribute> maskAttrs(W, zeroAttr);
      for (int64_t lane : lanes) maskAttrs[lane] = oneAttr;

      auto maskAttr = DenseElementsAttr::get(vecType, maskAttrs);
      Value mask = arith::ConstantOp::create(builder, loc, maskAttr);

      Value masked = arith::MulIOp::create(builder, loc, rotated, mask);
      blendResult =
          blendResult ? arith::AddIOp::create(builder, loc, blendResult, masked)
                            .getResult()
                      : masked;
    }
    return blendResult;
  };

  // --- Emit vector ops step by step ---
  for (int64_t step = 0; step <= schedule.maxStep(); ++step) {
    auto opsAtStep = schedule.getStep(step);
    if (opsAtStep.empty()) continue;

    // Skip steps that are entirely input loads (either tensor.extract or
    // __coyote_load wrapping one) — they were handled by the assign_layout
    // pass above and already have opVec entries.
    if (llvm::all_of(opsAtStep, [&](Operation *op) {
          return isa<tensor::ExtractOp>(op) || (bool)loadInputExtract(op);
        }))
      continue;

    // Special case: a step of __coyote_load calls wrapping arith results.
    // These are SIMD-level identity: opVec[call] = rotated source vector.
    if (llvm::all_of(opsAtStep, isCoyoteLoad)) {
      Value lhsVec = buildOperandVec(opsAtStep, 0);
      if (!lhsVec) continue;
      for (auto *op : opsAtStep) opVec[op] = lhsVec;
      continue;
    }

    Value lhsVec = buildOperandVec(opsAtStep, 0);
    Value rhsVec = buildOperandVec(opsAtStep, 1);
    if (!lhsVec || !rhsVec) continue;

    Operation *refOp = opsAtStep.front();
    Value result;

    if (isa<arith::AddIOp>(refOp))
      result = arith::AddIOp::create(builder, loc, lhsVec, rhsVec);
    else if (isa<arith::MulIOp>(refOp))
      result = arith::MulIOp::create(builder, loc, lhsVec, rhsVec);
    else if (isa<arith::SubIOp>(refOp))
      result = arith::SubIOp::create(builder, loc, lhsVec, rhsVec);
    else if (isa<arith::AddFOp>(refOp))
      result = arith::AddFOp::create(builder, loc, lhsVec, rhsVec);
    else if (isa<arith::MulFOp>(refOp))
      result = arith::MulFOp::create(builder, loc, lhsVec, rhsVec);
    else if (isa<arith::SubFOp>(refOp))
      result = arith::SubFOp::create(builder, loc, lhsVec, rhsVec);
    else {
      llvm::errs() << "[lowerToMLIR] Unhandled op kind: " << refOp->getName()
                   << " — skipping step " << step << "\n";
      continue;
    }

    for (auto *op : opsAtStep)
      if (!isa<tensor::ExtractOp>(op)) opVec[op] = result;
  }

  // --- Output handling via assign_layout (inverse permutation) ---
  // Scan ALL insert ops in the block — don't depend on opVec, since not all
  // output producers may have opVec entries (the vectorized accumulation
  // structure differs from the original scalar data flow).
  //
  // Multi-ciphertext output support: Coyote may legitimately materialize
  // different output-carrying scalar values in DIFFERENT vector ops (e.g.
  // 63 outputs live in the "bulk" op, one orphan lives in a singleton chain
  // scheduled at a non-canonical cycle). If so, the packed result becomes a
  // rank-2 secret tensor `!secret.secret<tensor<N x W x T>>` where each row
  // of the leading dim is one output ciphertext. The single
  // `tensor_ext.layout` attribute uses the `src_ct` column to indicate which
  // physical ciphertext each logical output lives in.

  // Step 1: walk leaf inserts to discover distinct output-carrying vector
  // ops. A "leaf" is the last insert in a chain (nothing else uses its
  // result as a destination). For each such chain, walk backwards through
  // the inserts to collect every scalar producer's `opVec` Value.
  llvm::SetVector<Value> outputVecs;
  scheduleBlock->walk([&](tensor::InsertOp insertOp) {
    bool isLast = true;
    for (auto &use : insertOp.getResult().getUses()) {
      if (isa<tensor::InsertOp>(use.getOwner())) {
        isLast = false;
        break;
      }
    }
    if (!isLast) return;

    // Walk back through the insert chain and record each scalar's vector op.
    Value cursor = insertOp.getResult();
    while (auto in =
               dyn_cast_if_present<tensor::InsertOp>(cursor.getDefiningOp())) {
      Operation *producer = in.getScalar().getDefiningOp();
      if (producer && opVec.count(producer)) {
        outputVecs.insert(opVec[producer]);
      }
      cursor = in.getDest();
    }
  });

  // Step 2: build the layout data. Each row is
  // [src_ct=ctIdx, src_slot=lane, dst_ct=0, dst_slot=logical_slot].
  // ctIdx is the index of the producer's vector op in `outputVecs`.
  SmallVector<int64_t> invLayoutData;
  int64_t numOutputMappings = 0;

  scheduleBlock->walk([&](tensor::InsertOp insertOp) {
    Value scalar = insertOp.getScalar();
    Operation *producer = scalar.getDefiningOp();
    if (!producer || !schedule.lanes.count(producer)) return;

    int64_t lane = schedule.lanes.lookup(producer);

    auto indices = insertOp.getIndices();
    auto constIdx = indices.back().getDefiningOp<arith::ConstantIndexOp>();
    int64_t slot = constIdx ? constIdx.value() : 0;

    // Which output ciphertext does this scalar's value live in?
    int64_t ctIdx = 0;
    if (opVec.count(producer)) {
      Value producerVec = opVec[producer];
      auto it = std::find(outputVecs.begin(), outputVecs.end(), producerVec);
      if (it != outputVecs.end()) ctIdx = std::distance(outputVecs.begin(), it);
    }

    invLayoutData.push_back(ctIdx);
    invLayoutData.push_back(lane);
    invLayoutData.push_back(0);
    invLayoutData.push_back(slot);
    ++numOutputMappings;
  });

  if (numOutputMappings > 0) {
    auto invLayoutAttrType =
        RankedTensorType::get({numOutputMappings, 4}, builder.getI64Type());
    auto invLayoutAttr =
        DenseIntElementsAttr::get(invLayoutAttrType, invLayoutData);

    // Attach output layout only to the secret.generic op. Do NOT propagate
    // it as a top-level `tensor_ext.layout` on the func result — that slot
    // is what LayoutPropagation stamps its synthesized Presburger relation
    // into, and having our dense permutation there just creates a race
    // (LayoutPropagation overwrites it, then ConvertFunc reads Presburger
    // and would rebuild original_type around it, absent the HEIR-side
    // preservation guard). The `tensor_ext.original_type` we set at the
    // end of this block carries the same dense permutation in the
    // canonical place for AddClientInterface — one source of truth is
    // enough.
    genericOp->setAttr("tensor_ext.layout", invLayoutAttr);

    int64_t N = static_cast<int64_t>(outputVecs.size());
    if (N == 0) {
      // Fallback: no output-carrying vec ops detected via the leaf-insert
      // walk (unusual, but possible if the schedule has inserts whose
      // producers aren't in opVec). Use the pre-existing "last non-extract
      // vec op in program order" heuristic and RAUW leaf inserts. This
      // preserves the pre-refactor behavior for edge cases.
      Value finalVec;
      for (auto *op : schedule.instructions)
        if (!isa<tensor::ExtractOp>(op) && opVec.count(op))
          finalVec = opVec[op];
      if (finalVec) {
        scheduleBlock->walk([&](tensor::InsertOp insertOp) {
          bool isLast = true;
          for (auto &use : insertOp.getResult().getUses()) {
            if (isa<tensor::InsertOp>(use.getOwner())) {
              isLast = false;
              break;
            }
          }
          if (isLast) insertOp.getResult().replaceAllUsesWith(finalVec);
        });
      }
    } else if (N == 1) {
      // Single-ct case: identical to previous behavior — RAUW leaf inserts
      // with the sole output vec. Result type stays `tensor<1 x W x T>`
      // (already set at line 911), so no signature change needed.
      Value finalVec = outputVecs[0];
      scheduleBlock->walk([&](tensor::InsertOp insertOp) {
        bool isLast = true;
        for (auto &use : insertOp.getResult().getUses()) {
          if (isa<tensor::InsertOp>(use.getOwner())) {
            isLast = false;
            break;
          }
        }
        if (isLast) insertOp.getResult().replaceAllUsesWith(finalVec);
      });
    } else {
      // Multi-ct case: pack the N output vecs (each `tensor<1 x W x T>`)
      // into a rank-2 `tensor<N x W x T>` via a chain of
      // `tensor.insert_slice` ops, then redirect the existing `secret.yield`
      // terminator to yield the packed value. The old scalar-insert chain
      // becomes dead code and is cleaned up by the canonicalize + DCE
      // sweep further below.
      auto packedType = RankedTensorType::get({N, (int64_t)W}, elemType);
      auto secretPackedType = secret::SecretType::get(packedType);

      // Concat the N output vecs (each `tensor<1 x W x T>`) along dim 0
      // into the packed `tensor<N x W x T>`.
      SmallVector<Value> vecs(outputVecs.begin(), outputVecs.end());
      Operation *concatOp =
          tensor::ConcatOp::create(builder, loc, packedType, /*dim=*/0, vecs);
      Value packed = concatOp->getResult(0);

      // Stamp the multi-ct output layout directly on the concat op via
      // its `attr-dict` slot. This gives LayoutPropagation an explicit
      // layout to honor at this boundary instead of synthesizing a default
      // Presburger relation. Unlike wrapping in `tensor_ext.assign_layout`,
      // this doesn't emit any data-movement ops during ciphertext-semantics
      // materialization — the concat's semantics are unchanged; only its
      // metadata carries the layout forward.
      concatOp->setAttr("tensor_ext.layout", invLayoutAttr);

      // Redirect the existing secret.yield to yield the packed tensor.
      // The old tensor.insert chain rooted at the original yield operand
      // now has no users and will be swept by the DCE pass below.
      Operation *term = scheduleBlock->getTerminator();
      term->setOperands({packed});

      // Update the secret.generic result type and the func return type to
      // the packed rank-2 form.
      genericOp->getResult(0).setType(secretPackedType);
      SmallVector<Type> newFuncArgTypes;
      for (unsigned i = 0; i < func.getNumArguments(); ++i)
        newFuncArgTypes.push_back(func.getArgument(i).getType());
      func.setType(FunctionType::get(ctx, newFuncArgTypes, {secretPackedType}));
    }

    // Stamp `tensor_ext.original_type` on the func result AFTER any type
    // mutation above. `func.setType` in the multi-ct branch replaces the
    // FuncOp's FunctionType, which may reset result-attr indexing in some
    // MLIR versions. Setting the attr as the last step guarantees it
    // survives on the final result index 0. Downstream (AddClientInterface,
    // ConvertToCiphertextSemantics) requires this attr with
    // `enable-layout-assignment=true` — without it, encrypt/decrypt helpers
    // can't be generated. Also prevents an upstream pass from synthesizing
    // a stale Presburger `original_type` from outer context that would fail
    // AssignLayoutOp's rank check.
    //
    // originalType = data-semantic output shape: one row per logical output
    // scalar (numOutputMappings). Layout is our dense permutation attr,
    // whose rows already carry the correct src_ct/src_slot/dst_slot routing.
    auto originalOutputType =
        RankedTensorType::get({numOutputMappings}, elemType);
    auto originalTypeAttr = tensor_ext::OriginalTypeAttr::get(
        ctx, originalOutputType, invLayoutAttr);
    func.setResultAttr(0, "tensor_ext.original_type", originalTypeAttr);
  }

  // --- Dead scalar IR sweep ---
  // The scheduled scalar ops, their feeding tensor.extract/__coyote_load chain,
  // and the consuming tensor.insert chain are now superseded by the SIMD ops
  // we just emitted. They form chains where each op is the only user of its
  // predecessor, so a single use_empty() pass won't unwind them — iterate to
  // fixpoint.
  auto isDeadCandidate = [&](Operation *op) {
    if (schedule.lanes.contains(op)) return true;
    if (isa<tensor::ExtractOp, tensor::InsertOp>(op)) return true;
    if (auto call = dyn_cast<func::CallOp>(op))
      if (call.getCallee() == "__coyote_load") return true;
    return false;
  };
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    scheduleBlock->walk([&](Operation *op) {
      if (isDeadCandidate(op) && op->use_empty()) dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  }

  // --- Erase the original (pre-bucket-refactor) kernel args ---
  // After the DCE sweep above, the original tensor.extract / __coyote_load
  // chains that referenced the original body block args are gone. We can
  // now drop the original block args (positions 0..origArgCount-1) along
  // with their matching secret.generic operands and outer func args.
  if (origArgCount > 0) {
    // Drop original operands of secret.generic at positions 0..origArgCount-1.
    SmallVector<Value> survivingOperands;
    for (unsigned i = origArgCount; i < genericOp->getNumOperands(); ++i)
      survivingOperands.push_back(genericOp->getOperand(i));
    genericOp->setOperands(survivingOperands);

    // Drop original body block args at the same positions.
    for (unsigned i = 0; i < origArgCount; ++i) scheduleBlock->eraseArgument(0);

    // Drop original func args (positions 0..origArgCount-1).
    llvm::BitVector funcArgsToErase(func.getNumArguments());
    for (unsigned i = 0; i < origArgCount; ++i) funcArgsToErase.set(i);
    func.eraseArguments(funcArgsToErase);
  }
}

}  // namespace heir
}  // namespace mlir

#endif  // OPTIMIZATION_H
