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

void optimizeBlends(Schedule &schedule, unsigned rounds) {
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

  // --- Expand argument types from tensor<1xN> to tensor<1xW> ---
  llvm::DenseMap<Value, llvm::SmallVector<Operation *>> srcToExtracts;
  for (auto *op : schedule.instructions) {
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
      Value source = extractOp.getTensor();
      srcToExtracts[source].push_back(op);
    }
  }

  auto secretVecType = secret::SecretType::get(vecType);
  Operation *genericOp = scheduleBlock->getParentOp();

  for (auto &[source, extractOps] : srcToExtracts) {
    if (extractOps.empty()) continue;

    // Expand inner block arg type to vecType
    if (auto blockArg = dyn_cast<BlockArgument>(source)) {
      blockArg.setType(vecType);

      // Expand corresponding outer func arg type
      unsigned innerIdx = blockArg.getArgNumber();
      Value outerOperand = genericOp->getOperand(innerIdx);
      if (auto funcArg = dyn_cast<BlockArgument>(outerOperand))
        funcArg.setType(secretVecType);
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
  for (auto &[source, extractOps] : srcToExtracts) {
    if (extractOps.empty()) continue;

    // Build layout: map each source slot to ALL consumer lanes (with
    // duplication).  This matches Python Coyote's load vectors where each
    // element appears at every lane that needs it, eliminating input rotations.
    SmallVector<int64_t> layoutData;
    int64_t numMappings = 0;
    for (auto *op : extractOps) {
      auto extractOp = cast<tensor::ExtractOp>(op);

      auto indices = extractOp.getIndices();
      auto constIdx = indices.back().getDefiningOp<arith::ConstantIndexOp>();
      int64_t slot = constIdx ? constIdx.value() : 0;

      // Map to each consumer's lane
      for (auto &use : op->getResult(0).getUses()) {
        Operation *consumer = use.getOwner();
        int64_t consLane = schedule.lanes.lookup(consumer);
        layoutData.push_back(0);
        layoutData.push_back(slot);
        layoutData.push_back(0);
        layoutData.push_back(consLane);
        ++numMappings;
      }
    }
    auto layoutAttrType =
        RankedTensorType::get({numMappings, 4}, builder.getI64Type());
    auto layoutAttr = DenseIntElementsAttr::get(layoutAttrType, layoutData);

    // Same-type assign_layout: tensor<1xW> → tensor<1xW>
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
      // Extract ops: assign_layout already placed the value at the consumer's
      // lane (via duplication), so no rotation is needed.
      int64_t shift = 0;
      if (!isa<tensor::ExtractOp>(prod)) {
        int64_t prodLane = schedule.lanes.lookup(prod);
        shift = prodLane - consLane;
      }

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

    if (llvm::all_of(opsAtStep,
                     [](Operation *op) { return isa<tensor::ExtractOp>(op); }))
      continue;

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

    for (auto *op : opsAtStep) opVec[op] = result;
  }

  // --- Output handling via assign_layout (inverse permutation) ---
  llvm::SmallVector<tensor::InsertOp> allInsertOps;
  for (auto *op : schedule.instructions) {
    if (isa<tensor::ExtractOp>(op)) continue;
    if (!opVec.count(op)) continue;
    if (op->getResults().empty()) continue;

    for (auto &use : op->getResult(0).getUses()) {
      if (auto insertOp = dyn_cast<tensor::InsertOp>(use.getOwner()))
        allInsertOps.push_back(insertOp);
    }
  }

  if (!allInsertOps.empty()) {
    llvm::DenseMap<Value, llvm::SmallVector<tensor::InsertOp>> vecToInserts;
    for (auto insertOp : allInsertOps) {
      Value scalar = insertOp.getScalar();
      Operation *producer = scalar.getDefiningOp();
      if (producer && opVec.count(producer))
        vecToInserts[opVec[producer]].push_back(insertOp);
    }

    for (auto &[vecResult, insertOps] : vecToInserts) {
      if (insertOps.empty()) continue;

      // Build inverse permutation: [lane -> slot]
      // Dest is tensor<1xN>: index [0, slot] -> slot is the flat index.
      SmallVector<int64_t> invLayoutData;
      for (auto insertOp : insertOps) {
        Value scalar = insertOp.getScalar();
        Operation *producer = scalar.getDefiningOp();
        int64_t lane = schedule.lanes.lookup(producer);

        auto indices = insertOp.getIndices();
        auto constIdx = indices.back().getDefiningOp<arith::ConstantIndexOp>();
        int64_t slot = constIdx ? constIdx.value() : 0;

        // [src_ct=0, src_slot=lane, dst_ct=0, dst_slot=slot]
        invLayoutData.push_back(0);
        invLayoutData.push_back(lane);
        invLayoutData.push_back(0);
        invLayoutData.push_back(slot);
      }

      int64_t numMappings = insertOps.size();
      auto invLayoutAttrType =
          RankedTensorType::get({numMappings, 4}, builder.getI64Type());
      auto invLayoutAttr =
          DenseIntElementsAttr::get(invLayoutAttrType, invLayoutData);

      // Annotate the yield (or its parent) with the output layout.
      // This is metadata only — the ciphertext is already in this layout,
      // we're just recording it so the client knows how to unpack after
      // decryption.
      Operation *terminator = scheduleBlock->getTerminator();
      terminator->setAttr("coyote.output_layout", invLayoutAttr);

      // Replace the insert chain result with the vectorized result directly.
      Value result = vecResult;

      // Find the last insert in the chain and replace its result.
      tensor::InsertOp lastInsert = insertOps.back();
      for (auto insertOp : insertOps) {
        bool isLast = true;
        for (auto &use : insertOp.getResult().getUses()) {
          if (isa<tensor::InsertOp>(use.getOwner())) {
            isLast = false;
            break;
          }
        }
        if (isLast) {
          lastInsert = insertOp;
          break;
        }
      }
      lastInsert.getResult().replaceAllUsesWith(result);
    }
  }

  // --- Erase now-dead scalar ops (reverse order for SSA validity) ---
  for (auto it = schedule.instructions.rbegin();
       it != schedule.instructions.rend(); ++it) {
    auto *op = *it;
    if (op->use_empty()) op->erase();
  }
}

}  // namespace heir
}  // namespace mlir

#endif  // OPTIMIZATION_H
