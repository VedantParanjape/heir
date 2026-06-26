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
inline void lowerToMLIR(func::FuncOp func, const Schedule &schedule) {
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

  // --- Expand argument types from tensor<1xN> to tensor<1xW> ---
  // Group scheduled "input load" ops by (source tensor, cycle). Splitting by
  // cycle is essential: a single lane may receive different scalars at
  // different cycles, so each cycle needs its own packed input vector. An
  // input load is either a tensor.extract or a __coyote_load(extract) wrap;
  // both are treated uniformly here, with the call op as the schedule key.
  llvm::DenseMap<std::pair<Value, int64_t>, llvm::SmallVector<Operation *>>
      srcCycleToExtracts;
  llvm::SetVector<Value> distinctSources;
  for (auto *op : schedule.instructions) {
    Value source;
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
      source = extractOp.getTensor();
    } else if (auto ext = loadInputExtract(op)) {
      // __coyote_load wrapping an extract: use the call op as the schedule
      // key, source from the underlying extract's tensor.
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

  // Track which block args we widened so we know which extracts to retarget.
  // Widening is per source tensor (not per cycle): each input is widened once.
  llvm::DenseSet<BlockArgument> widenedBlockArgs;
  for (Value source : distinctSources) {
    if (auto blockArg = dyn_cast<BlockArgument>(source)) {
      blockArg.setType(vecType);
      widenedBlockArgs.insert(blockArg);

      // Expand corresponding outer func arg type
      unsigned innerIdx = blockArg.getArgNumber();
      Value outerOperand = genericOp->getOperand(innerIdx);
      if (auto funcArg = dyn_cast<BlockArgument>(outerOperand))
        funcArg.setType(secretVecType);
    }
  }

  // Rewrite any tensor.extract that reads from a widened block arg to use
  // 2-D indices ([0, origIdx]) — the original IR has 1-D extracts because
  // the input was tensor<NxT>; after widening to tensor<1xWxT> they need to
  // address the extra leading dim.
  if (!widenedBlockArgs.empty()) {
    OpBuilder ib(scheduleBlock, scheduleBlock->begin());
    Value zeroIdx = arith::ConstantIndexOp::create(ib, loc, 0);

    SmallVector<tensor::ExtractOp> toRewrite;
    scheduleBlock->walk([&](tensor::ExtractOp ext) {
      auto ba = dyn_cast<BlockArgument>(ext.getTensor());
      if (!ba || !widenedBlockArgs.count(ba)) return;
      if (ext.getIndices().size() != 1) return;  // already multi-dim, skip
      toRewrite.push_back(ext);
    });

    for (tensor::ExtractOp ext : toRewrite) {
      OpBuilder eb(ext);
      SmallVector<Value> newIndices{zeroIdx};
      for (Value idx : ext.getIndices()) newIndices.push_back(idx);
      auto newExt = tensor::ExtractOp::create(eb, ext.getLoc(), ext.getTensor(),
                                              newIndices);
      ext.getResult().replaceAllUsesWith(newExt.getResult());
      ext.erase();
    }
  }

  // Update secret.generic result type
  genericOp->getResult(0).setType(secretVecType);

  // Update func signature
  SmallVector<Type> newArgTypes;
  for (unsigned i = 0; i < func.getNumArguments(); ++i)
    newArgTypes.push_back(func.getArgument(i).getType());
  func.setType(FunctionType::get(ctx, newArgTypes, {secretVecType}));

  // --- Input handling via assign_layout (same-type: 1xW → 1xW) ---
  // One assign_layout per (source tensor, cycle) bucket. Splitting by cycle
  // ensures each lane can receive a different scalar at each cycle, instead
  // of colliding into a single packed vector — which would force consumers
  // at different cycles to read identical operands.
  llvm::DenseMap<Operation *, int64_t> extractLayoutLane;

  for (auto &[key, extractOps] : srcCycleToExtracts) {
    if (extractOps.empty()) continue;
    Value source = key.first;

    // Build layout: for each input-load op, map its slot to its scheduled
    // lane. The op may be a tensor.extract directly, or a __coyote_load that
    // wraps an extract — in the latter case we read the slot from the wrapped
    // extract but use the call op's scheduled lane.
    SmallVector<int64_t> layoutData;
    int64_t numMappings = 0;
    for (auto *op : extractOps) {
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

      extractLayoutLane[op] = lane;

      layoutData.push_back(0);
      layoutData.push_back(slot);
      layoutData.push_back(0);
      layoutData.push_back(lane);
      ++numMappings;
    }
    auto layoutAttrType =
        RankedTensorType::get({numMappings, 4}, builder.getI64Type());
    auto layoutAttr = DenseIntElementsAttr::get(layoutAttrType, layoutData);

    Value permuted =
        tensor_ext::AssignLayoutOp::create(builder, loc, source, layoutAttr);

    for (auto *op : extractOps) opVec[op] = permuted;
  }

  // --- Rotation cache ---
  using RotKey = std::pair<void *, int64_t>;
  std::map<RotKey, Value> rotCache;

  auto valuePtr = [](Value v) -> void * { return v.getAsOpaquePointer(); };

  auto getRotated = [&](Value vec, int64_t shift) -> Value {
    if (shift == 0) return vec;
    RotKey key{valuePtr(vec), shift};
    auto it = rotCache.find(key);
    if (it != rotCache.end()) return it->second;

    Value shiftVal = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(builder.getI32Type(), shift));
    Value rotated = tensor_ext::RotateOp::create(builder, loc, vec, shiftVal);
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
  SmallVector<int64_t> invLayoutData;
  int64_t numOutputMappings = 0;
  tensor::InsertOp lastInsert;

  scheduleBlock->walk([&](tensor::InsertOp insertOp) {
    Value scalar = insertOp.getScalar();
    Operation *producer = scalar.getDefiningOp();
    if (!producer || !schedule.lanes.count(producer)) return;

    int64_t lane = schedule.lanes.lookup(producer);

    auto indices = insertOp.getIndices();
    auto constIdx = indices.back().getDefiningOp<arith::ConstantIndexOp>();
    int64_t slot = constIdx ? constIdx.value() : 0;

    // [src_ct=0, src_slot=lane, dst_ct=0, dst_slot=slot]
    invLayoutData.push_back(0);
    invLayoutData.push_back(lane);
    invLayoutData.push_back(0);
    invLayoutData.push_back(slot);
    ++numOutputMappings;

    // Track the last insert in the chain (the one yielded).
    lastInsert = insertOp;
  });

  if (numOutputMappings > 0) {
    auto invLayoutAttrType =
        RankedTensorType::get({numOutputMappings, 4}, builder.getI64Type());
    auto invLayoutAttr =
        DenseIntElementsAttr::get(invLayoutAttrType, invLayoutData);

    // Attach output layout to the secret.generic op result, then propagate
    // to the func return type via HEIR's AttributeUtils.
    genericOp->setAttr("tensor_ext.layout", invLayoutAttr);
    copyReturnOperandAttrsToFuncResultAttrs(func, "tensor_ext.layout");

    // Find the last insert in the chain (its result is yielded) and replace
    // with the final vectorized result.
    Value finalVec;
    for (auto *op : schedule.instructions)
      if (!isa<tensor::ExtractOp>(op) && opVec.count(op)) finalVec = opVec[op];
    if (finalVec) {
      // Walk to find the actual last insert (the one whose result feeds yield).
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
}

}  // namespace heir
}  // namespace mlir

#endif  // OPTIMIZATION_H
