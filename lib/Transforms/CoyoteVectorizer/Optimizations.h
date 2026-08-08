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
#include "lib/Dialect/Secret/IR/SecretOps.h"
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
#include "llvm/include/llvm/Support/MathExtras.h"        // from @llvm-project
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

  // Coyote schedules at warpSize `W` (the logical lane count). Emit
  // tensors at `R = nextPow2(3*W)` instead so that the rest of the
  // pipeline sees a physical tensor already sized for the FHE ring, with
  // W-slot data tiled `R/W` times to fill it. Rotations stay canonical
  // in [0, W) — mod-R cyclic rotate on tiled data is equivalent to
  // mod-W cyclic rotate on the logical view.
  //
  // This avoids the downstream padding-vs-tiling issue in HEIR's
  // ConvertToCiphertextSemantics: with data already ring-sized and
  // tiled, no widening pass fills the tail with zeros.
  unsigned W = schedule.warpSize;
  unsigned R = static_cast<unsigned>(llvm::PowerOf2Ceil(3 * W));
  unsigned tileFactor = R / W;
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
  auto vecType = RankedTensorType::get({1, (int64_t)R}, elemType);

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
    // Each raw-input scalar (identified by `slot` in the source arg)
    // maps to `lane` in the ciphertext. To tile the coyote schedule
    // across the full ring, replicate each (slot, lane) entry
    // `tileFactor` times with `lane += k*W` — same source scalar lives
    // at the same offset within each tile, so mod-R cyclic rotations
    // behave as mod-W rotations on the tiled data.
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
      for (unsigned k = 0; k < tileFactor; ++k) {
        layoutData.push_back(0);
        layoutData.push_back(slot);
        layoutData.push_back(0);
        layoutData.push_back(lane + static_cast<int64_t>(k * W));
        ++numMappings;
      }
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
    // Canonicalize: rewrite every rotation as a positive left shift in
    // the range `[0, warpSize)`. tensor_ext.rotate is defined as a left
    // rotation for positive shifts (docs: "This op represents a
    // left-rotation of a tensor by given number of indices. Negative
    // shift values are interpreted as right-rotations."), so mapping
    // -K to (N - K) gives a uniform positive-left-shift form.
    //
    // Downstream benefits:
    //   - `rotCache` keys collapse — two schedule ops that differ only in
    //     sign convention share the same rotate.
    //   - Wrap-safety analysis (future) sees a uniform `+K` form and
    //     doesn't have to case-split on sign.
    //   - Emit stays semantically identical: cyclic shift by K mod N is
    //     the same operation regardless of how we write the constant.
    int64_t warpSize = static_cast<int64_t>(schedule.warpSize);
    int64_t normalized =
        warpSize > 0 ? ((shift % warpSize) + warpSize) % warpSize : shift;
    if (normalized == 0) return vec;
    RotKey key{valuePtr(vec), normalized};
    auto it = rotCache.find(key);
    if (it != rotCache.end()) return it->second;

    // Emit the shift as `index` (not i32). tensor_ext::RotateOp attaches the
    // IndexTypesNeedNoLayoutImpl interface which tells layout-propagation
    // that an `index`-typed shift does not need a layout. Without a layout,
    // convert-to-ciphertext-semantics leaves the scalar shift alone instead
    // of widening it to tensor<1x64xi32>, which would break the eventual
    // bgv.rotate_cols verifier (it expects a scalar dynamic_shift).
    Value shiftVal =
        arith::ConstantIndexOp::create(builder, loc, normalized).getResult();
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

    // Multiple (source, shift) combinations — blend with masks.
    // Masks are built at logical warp size W (one entry per lane) and
    // then tiled `tileFactor` times to match the R-sized physical tensor.
    // This keeps the mask aligned with the tiled ciphertext data so that
    // ct * mask preserves the tiling invariant across the full ring.
    Value blendResult;
    auto zeroAttr = builder.getIntegerAttr(elemType, 0);
    auto oneAttr = builder.getIntegerAttr(elemType, 1);
    for (auto &[key, lanes] : keyToLanes) {
      auto &[ptr, shift] = key;
      Value rotated = getRotated(ptrToVec[ptr], shift);

      SmallVector<Attribute> baseMask(W, zeroAttr);
      for (int64_t lane : lanes) baseMask[lane] = oneAttr;

      SmallVector<Attribute> maskAttrs;
      maskAttrs.reserve(R);
      for (unsigned k = 0; k < tileFactor; ++k)
        maskAttrs.append(baseMask.begin(), baseMask.end());
      // Pad to exactly R with zeros. `tileFactor * W` matches R only when
      // R is an integer multiple of W (true when W is a power of 2, since
      // R = nextPow2(3*W) = 4*W there). For non-power-of-2 W, tileFactor
      // rounds down and we'd emit a short mask, which asserts inside
      // DenseElementsAttr::get. Zeros in the trailing pad are safe: the
      // extra R-tileFactor*W slots don't hold meaningful data (coyote's
      // rotations stay within tileFactor*W).
      while (maskAttrs.size() < R) maskAttrs.push_back(zeroAttr);

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

  // Step 1: discover distinct output-carrying vector ops.
  //
  // Two input-side patterns are handled:
  //   (a) Chains of `tensor.insert` — a "leaf" insert (no other insert
  //       uses its result) roots a chain; walk it backwards and record
  //       each scalar producer's `opVec` value.
  //   (b) A single `tensor.from_elements` — its operands are the logical
  //       output scalars in row-major position order. This is the
  //       pattern coyote's recursive templates emit for their base case:
  //           %out = tensor.from_elements %a, %b, %c : tensor<3xT>
  //           %out_c = tensor.cast %out : tensor<3xT> to tensor<?xT>
  //           secret.yield %out_c
  //       Without this branch neither the RAUW below nor the DCE sweep
  //       finds anything to work on, the scalar chain stays live, and
  //       the block-arg erase asserts on live use-lists.
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
  scheduleBlock->walk([&](tensor::FromElementsOp feOp) {
    for (Value scalar : feOp.getElements()) {
      Operation *producer = scalar.getDefiningOp();
      if (producer && opVec.count(producer)) {
        outputVecs.insert(opVec[producer]);
      }
    }
  });

  // Step 2: build the layout data. Each row is
  // [src_ct=ctIdx, src_slot=lane, dst_ct=0, dst_slot=logical_slot].
  // ctIdx is the index of the producer's vector op in `outputVecs`.
  SmallVector<int64_t> invLayoutData;
  int64_t numOutputMappings = 0;

  auto recordLayoutRow = [&](Operation *producer, int64_t slot) {
    if (!producer || !schedule.lanes.count(producer)) return;
    int64_t lane = schedule.lanes.lookup(producer);
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
  };

  scheduleBlock->walk([&](tensor::FromElementsOp feOp) {
    // Row-major operand position = logical output slot.
    for (auto [slotIdx, scalar] : llvm::enumerate(feOp.getElements())) {
      recordLayoutRow(scalar.getDefiningOp(), static_cast<int64_t>(slotIdx));
    }
  });

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
      // Single-ct case: retarget the secret.yield's operand to the sole
      // output vec. Result type is `tensor<1xWxT>` (already set on the
      // generic result at line 911), and yield must match the generic's
      // result type, so this is type-correct atomically.
      //
      // Also RAUW the leaf `tensor.insert` for the insert-chain output
      // pattern — same as before — so downstream DCE can clean the chain.
      // For the `tensor.from_elements` + `tensor.cast` + `secret.yield`
      // pattern (used by coyote's recursive-template base case) the
      // yield-retarget above is what actually reconnects the output;
      // the leaf-insert RAUW below is a no-op because there are no
      // inserts, but keeping it preserves the historical behavior on
      // insert-based outputs.
      Value finalVec = outputVecs[0];
      Operation *term = scheduleBlock->getTerminator();
      if (term && term->getNumOperands() == 1) term->setOperand(0, finalVec);

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
      // Multi-ct case (V1): yield N separate `tensor<1 x W x T>` values from
      // `secret.generic` and return them as N separate SSA values from the
      // enclosing func. Each result carries a *partial* dense-permutation
      // `tensor_ext.original_type` describing the subset of the user's
      // logical output tensor that this ct owns.
      //
      // Prior versions here packed the N vecs into a single `tensor<N x W x T>`
      // via `tensor.concat` and (later) `tensor.insert_slice`. Both routes
      // ended up funneled through a scheme-to-openfhe wrap-scalar-into-size-1-
      // buffer pattern that, combined with an `add_inplace` liveness bug in
      // HEIR's AllocToInPlace pass and an `InsertOp` C++ emission alias,
      // corrupted the multi-ct output at runtime. Yielding N separate values
      // eliminates the wrap entirely.
      SmallVector<Value> vecs(outputVecs.begin(), outputVecs.end());
      auto singleCtType = RankedTensorType::get({1, (int64_t)R}, elemType);
      auto secretSingleCtType = secret::SecretType::get(singleCtType);
      SmallVector<Type> newResultTypes(N, secretSingleCtType);

      // Recreate the secret.generic with N result types (result count is
      // fixed at op-creation time, so we can't just mutate the existing op).
      // Build via OperationState so we can add a bare (blockless) region
      // that `takeBody` can then move blocks into without conflicting with
      // an auto-inserted entry block. `genericOp` is `Operation*`, so use
      // `->` throughout.
      OpBuilder outerBuilder(genericOp);
      SmallVector<Value> genericOperands(genericOp->getOperands().begin(),
                                         genericOp->getOperands().end());
      OperationState state(genericOp->getLoc(),
                           secret::GenericOp::getOperationName());
      state.addOperands(genericOperands);
      state.addTypes(newResultTypes);
      state.addRegion();
      for (NamedAttribute na : genericOp->getAttrs()) {
        // The combined-layout `tensor_ext.layout` was an artifact of the
        // packed representation; each result now carries its own partial
        // layout via `tensor_ext.original_type` set below.
        if (na.getName() == "tensor_ext.layout") continue;
        // Skip attrs MLIR maintains automatically (operand segment sizes).
        if (na.getName().strref().starts_with("operandSegmentSizes")) continue;
        state.addAttribute(na.getName(), na.getValue());
      }
      Operation *newGenericOp = outerBuilder.create(state);
      auto newGeneric = cast<secret::GenericOp>(newGenericOp);

      // Move the body region from the old generic to the new one.
      newGeneric.getRegion().takeBody(genericOp->getRegion(0));

      // The moved body's terminator was the old `secret.yield` yielding the
      // combined-packed tensor. Rewrite it to yield the N per-ct vecs.
      Block *newBody = &newGeneric.getRegion().front();
      Operation *newTerm = newBody->getTerminator();
      newTerm->setOperands(vecs);

      // Replace the enclosing func's return op to return the N new results.
      // The old generic had one use — the func.return of its single result.
      func::ReturnOp existingReturn;
      func.walk([&](func::ReturnOp op) { existingReturn = op; });
      assert(existingReturn && "expected a func.return in the enclosing func");
      OpBuilder retBuilder(existingReturn);
      func::ReturnOp::create(retBuilder, existingReturn.getLoc(),
                             newGenericOp->getResults());
      existingReturn.erase();

      // Erase the now-unused old generic and rebind the local `genericOp`
      // reference so the code below (dead-scalar sweep, layout-stamping)
      // operates on the new op.
      genericOp->erase();
      genericOp = newGenericOp;

      // Update the func signature to N result types.
      SmallVector<Type> newFuncArgTypes;
      for (unsigned i = 0; i < func.getNumArguments(); ++i)
        newFuncArgTypes.push_back(func.getArgument(i).getType());
      func.setType(FunctionType::get(ctx, newFuncArgTypes, newResultTypes));
    }

    // Stamp `tensor_ext.original_type` on each func result AFTER any type
    // mutation above. Downstream (`AddClientInterface`,
    // `ConvertToCiphertextSemantics`) requires this attr with
    // `enable-layout-assignment=true` — without it, encrypt/decrypt helpers
    // can't be generated. For the V1 multi-return branch, each of the N
    // results carries a PARTIAL layout describing the subset of the user's
    // logical output tensor that this ct owns; the union of the N partial
    // layouts equals the full `invLayoutAttr` we would have stamped on a
    // single packed result.
    //
    // Both single-ct and multi-ct branches share `originalType =
    // tensor<numOutputMappings x elemT>` (the full user-facing shape).
    auto originalOutputType =
        RankedTensorType::get({numOutputMappings}, elemType);

    if (N <= 1) {
      auto originalTypeAttr = tensor_ext::OriginalTypeAttr::get(
          ctx, originalOutputType, invLayoutAttr);
      func.setResultAttr(0, "tensor_ext.original_type", originalTypeAttr);
    } else {
      // Split `invLayoutData` (rows of [src_ct, src_slot, dst_ct, dst_slot])
      // into N per-ct tables. In each per-result table, `src_ct` becomes 0
      // (the result now carries a single ct) while `src_slot`, `dst_ct`,
      // `dst_slot` are copied verbatim.
      SmallVector<SmallVector<int64_t>> perCtRows(N);
      for (int64_t row = 0; row < numOutputMappings; ++row) {
        int64_t srcCt = invLayoutData[4 * row + 0];
        int64_t srcSlot = invLayoutData[4 * row + 1];
        int64_t dstCt = invLayoutData[4 * row + 2];
        int64_t dstSlot = invLayoutData[4 * row + 3];
        assert(srcCt >= 0 && srcCt < N && "src_ct out of range");
        perCtRows[srcCt].push_back(0);
        perCtRows[srcCt].push_back(srcSlot);
        perCtRows[srcCt].push_back(dstCt);
        perCtRows[srcCt].push_back(dstSlot);
      }
      for (int64_t i = 0; i < N; ++i) {
        int64_t nRows = static_cast<int64_t>(perCtRows[i].size()) / 4;
        auto perLayoutTy =
            RankedTensorType::get({nRows, 4}, builder.getI64Type());
        auto perLayoutAttr =
            DenseIntElementsAttr::get(perLayoutTy, perCtRows[i]);
        auto perOriginalTypeAttr = tensor_ext::OriginalTypeAttr::get(
            ctx, originalOutputType, perLayoutAttr);
        func.setResultAttr(i, "tensor_ext.original_type", perOriginalTypeAttr);
      }
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
    // Also sweep the from_elements/cast chain used by coyote's recursive-
    // template base case for its output. After the N==1 yield-retarget
    // above, those ops are use-less but they anchor the transitive chain
    // that keeps the original scalar arith and the original block-arg
    // extracts alive. Without adding them here, DCE stops at the
    // still-live from_elements, the original block args stay in use, and
    // the erase-args step below asserts on a live use-list.
    if (isa<tensor::ExtractOp, tensor::InsertOp, tensor::FromElementsOp,
            tensor::CastOp>(op))
      return true;
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
  // Normally the vectorization above (a) RAUW-replaces the leaf
  // tensor.insert with the final vec, (b) DCE'd the scalar arith chain
  // that fed it, and (c) DCE'd the scalar extracts referencing the
  // original body block args — leaving those args use-less and safe to
  // drop along with their matching secret.generic operands and outer
  // func args.
  //
  // In practice this isn't always clean — e.g. some small / base-case
  // kernels reach here with the leaf-insert RAUW skipped or one of the
  // DCE waves missing an op. When that happens the original block arg
  // is still transitively used and calling Block::eraseArgument asserts
  // inside the BlockArgument destructor's use-list invariant check.
  //
  // Be defensive: check use_empty per original arg and only erase the
  // ones that are actually dead. Any leftover live args indicate a
  // missed RAUW/DCE upstream that should be fixed separately, but they
  // shouldn't crash the compile.
  if (origArgCount > 0) {
    llvm::BitVector origArgsStillLive(origArgCount);
    for (unsigned i = 0; i < origArgCount; ++i) {
      if (!scheduleBlock->getArgument(i).use_empty()) origArgsStillLive.set(i);
    }

    // Drop only the dead original block args and their matching generic
    // operands / func args. We iterate high-to-low on positions so index
    // math doesn't shift under us mid-erase.
    SmallVector<Value> survivingOperands(genericOp->getOperands().begin(),
                                         genericOp->getOperands().end());
    llvm::BitVector funcArgsToErase(func.getNumArguments());
    for (int i = static_cast<int>(origArgCount) - 1; i >= 0; --i) {
      if (origArgsStillLive.test(i)) continue;
      scheduleBlock->eraseArgument(i);
      survivingOperands.erase(survivingOperands.begin() + i);
      funcArgsToErase.set(i);
    }
    genericOp->setOperands(survivingOperands);
    if (funcArgsToErase.any()) func.eraseArguments(funcArgsToErase);
  }

  // --- Sever now-incompatible callers of the rewritten function ---
  //
  // The signature mutation above changed `func`'s inputs/outputs from
  // data-semantic (`!secret<tensor<Nxi32>>` etc.) to ct-semantic
  // (`!secret<tensor<1xRxi32>>` + one arg per bucket, N outputs, etc.).
  // Any pre-existing `func.call` targeting `func` was written against the
  // old signature and is now type-incompatible — the MLIR verifier that
  // runs at the end of RecursiveCallVectorization refuses to accept the
  // module, and the pipeline aborts before strip_scaffold has a chance
  // to prune the scaffolding caller (e.g. `@main`).
  //
  // For the biscotti scaffold shape ("outer wrapper calls the kernel we
  // just vectorized"), the correct action is to drop the caller entirely
  // — the downstream `AddClientInterface` pass synthesizes proper
  // encrypt/decrypt shims that consume the new signature directly. So we
  // walk the enclosing module for any `func::CallOp` targeting our symbol
  // and erase its containing FuncOp. We never erase `func` itself.
  //
  // If a caller legitimately needed to keep functioning, this is where
  // you'd inject an encrypt-and-repack shim instead — but no coyote flow
  // currently produces that shape.
  if (auto module = func->getParentOfType<ModuleOp>()) {
    StringRef funcSym = func.getSymName();
    llvm::SmallDenseSet<func::FuncOp> deadCallers;
    module.walk([&](func::CallOp callOp) {
      if (callOp.getCallee() != funcSym) return;
      auto callerFunc = callOp->getParentOfType<func::FuncOp>();
      if (callerFunc && callerFunc != func) deadCallers.insert(callerFunc);
    });
    for (func::FuncOp caller : deadCallers) caller.erase();
  }
}

//===- HoistInputLoads.h - Schedule-level input-load hoisting ---*- C++ -*-===//
//
// Rewrites a Schedule so that input-side __coyote_load ops (loads whose
// source is a func block argument) are collapsed into aggregate virtual loads
// keyed by (consumer cycle, consumer operand position). The client is expected
// to precompute a ciphertext for each aggregate load, so the FHE server no
// longer emits the rotations / blends / pt-ct muls that would otherwise
// materialize the load's SIMD layout at runtime.
//
// Runs before lowerToMLIR; consumes and mutates a Schedule in place.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Input-side classification
//===----------------------------------------------------------------------===//

/// True iff `op` is an input-side load. Two shapes are recognized:
///   (a) `tensor.extract %arg[...]` where %arg is directly a func
///       BlockArgument — the tensor-block-arg case. Tensor block args are
///       *not* wrapped by wrapBlockArgsWithVirtualLoads (which guards on
///       `arg.getType().isIntOrFloat()`), so the extract itself is the
///       input-side op.
///   (b) `func.call @__coyote_load(x)` where x is a func BlockArgument, or
///       is a tensor.extract feeding from one — the scalar-block-arg case
///       that wrapBlockArgsWithVirtualLoads produces.
static bool isInputSideLoad(Operation *op) {
  if (auto ext = dyn_cast<tensor::ExtractOp>(op))
    return isa<BlockArgument>(ext.getTensor());
  auto call = dyn_cast<func::CallOp>(op);
  if (!call || call.getCallee() != "__coyote_load" ||
      call.getNumOperands() != 1)
    return false;
  Value src = call.getOperand(0);
  if (isa<BlockArgument>(src)) return true;
  if (auto ext = src.getDefiningOp<tensor::ExtractOp>())
    return isa<BlockArgument>(ext.getTensor());
  return false;
}

//===----------------------------------------------------------------------===//
// Main entry point
//===----------------------------------------------------------------------===//
/// Hoist input-side __coyote_load ops in `schedule` into aggregate virtual
/// loads. Mutates `schedule` in place; each hoisted load is rewritten so its
/// (lane, alignment) equals its consumer's SIMD slot. The layout spec
/// (which scalar source goes into which lane of which ciphertext) is fully
/// recoverable by regrouping the rewritten __coyote_load ops in the
/// resulting schedule by (alignment, consumer operand position).
///
/// Returns the number of aggregate virtual loads created — i.e. how many
/// client-side ciphertexts / func args the caller must provision.
///
/// Asserts:
///   (1) no input-side producer belongs to more than one (cycle, operandPos)
///       group (otherwise the in-place rewrite would need to duplicate it);
///   (2) the number of distinct frontier alignments is <= the number of
///       aggregate loads produced (every hoisted alignment gets at least one
///       layout arg).
inline unsigned hoistInputSideLoads(Schedule &schedule) {
  // Fast lookup for "does this op belong to the schedule?".
  llvm::DenseSet<Operation *> inSchedule;
  for (Operation *op : schedule.instructions) inSchedule.insert(op);

  // -------- Steps 1 & 2: classify. Input-side = tensor.extract from a
  // BlockArgument, or __coyote_load wrapping one; everything else is
  // runtime. Store just the input-side set — the runtime set is its
  // complement within `inSchedule`.
  llvm::DenseSet<Operation *> inputSideOps;
  for (Operation *op : schedule.instructions)
    if (isInputSideLoad(op)) inputSideOps.insert(op);

  // -------- Step 3: find the hoist frontier. -------------------------------
  // A frontier entry is (producer, consumer, operandPos) where the consumer
  // is runtime-mandated and the producer is hoistable (in the schedule and
  // not runtime).
  struct FrontierEntry {
    Operation *producer;
    Operation *consumer;
    unsigned operandPos;
  };
  llvm::SmallVector<FrontierEntry> frontier;
  for (Operation *consumer : schedule.instructions) {
    if (inputSideOps.contains(consumer)) continue;  // runtime consumers only
    for (unsigned p = 0, e = consumer->getNumOperands(); p < e; ++p) {
      Operation *producer = consumer->getOperand(p).getDefiningOp();
      if (!producer) continue;                         // external block arg
      if (!inSchedule.contains(producer)) continue;    // not part of schedule
      if (!inputSideOps.contains(producer)) continue;  // runtime -> runtime
      frontier.push_back({producer, consumer, p});
    }
  }

  // -------- Step 4: group frontier by (consumer cycle, operand pos). -------
  using GroupKey = std::pair<int64_t, unsigned>;
  llvm::DenseMap<GroupKey, llvm::SmallVector<FrontierEntry>> stages;
  for (const FrontierEntry &fe : frontier) {
    int64_t cycle = schedule.alignment.lookup(fe.consumer);
    stages[{cycle, fe.operandPos}].push_back(fe);
  }

  // -------- Step 5: rebuild the load prefix. -------------------------------
  // Each (consumer_cycle, opPos) group becomes one dedicated aggregate-load
  // cycle at the head of the schedule. Producers in a group are placed at
  // (group_index, consumer_lane). Non-hoisted ops keep their original
  // alignment; the pass only works when the schedule already has enough
  // head-room, which is what Assert (2) below verifies before any mutation.
  int64_t numLayoutArgs = stages.size();

  // Collect the hoisted-producer set up front so Assert (2) can distinguish
  // the ops that will *become* the load prefix from the ops that must sit
  // after it.
  llvm::DenseSet<Operation *> hoistedProducers;
  llvm::DenseSet<int64_t> frontierAlignments;
  for (auto &[key, entries] : stages) {
    frontierAlignments.insert(key.first);
    for (const FrontierEntry &fe : entries)
      hoistedProducers.insert(fe.producer);
  }

  // Compute the shift needed so the load prefix (cycles 0..G-1) fits
  // before every non-hoisted op. This used to be an assertion, but the
  // baseline (non-recursive) variant packs cycle 0 tightly with parallel
  // loads and starts compute at cycle 2, while the frontier can produce
  // more stages than that (G > 2). Rather than fail, shift non-hoisted
  // ops down by the required amount — semantically neutral (compute
  // order preserved, just delayed by head-room), and results in a
  // schedule the rest of the pipeline can lower.
  int64_t minNonHoistedCycle = INT64_MAX;
  for (Operation *op : schedule.instructions) {
    if (hoistedProducers.contains(op)) continue;
    int64_t c = schedule.alignment.lookup(op);
    if (c < minNonHoistedCycle) minNonHoistedCycle = c;
  }
  // If no non-hoisted ops exist (schedule is pure loads), no shift needed.
  if (minNonHoistedCycle != INT64_MAX) {
    int64_t shift = numLayoutArgs - minNonHoistedCycle;
    if (shift > 0) {
      for (Operation *op : schedule.instructions) {
        if (hoistedProducers.contains(op)) continue;
        schedule.alignment[op] += shift;
      }
      minNonHoistedCycle += shift;
    }
  }
  assert(numLayoutArgs <= minNonHoistedCycle &&
         "aggregate-load prefix would overrun a non-hoisted op — "
         "shift logic above should have already ensured this");

  // Assert (3): every frontier alignment received at least one layout arg.
  assert((int64_t)frontierAlignments.size() <= numLayoutArgs &&
         "fewer aggregate loads than distinct frontier alignments");
  (void)frontierAlignments;
  (void)minNonHoistedCycle;

  // Now perform the rewrite.
  int64_t g = 0;
  llvm::DenseSet<Operation *> seenProducers;
  for (auto &[key, entries] : stages) {
    for (const FrontierEntry &fe : entries) {
      // Assert (1): a producer may only be rewritten into one target slot.
      assert(!seenProducers.contains(fe.producer) &&
             "producer belongs to multiple stage-groups — in-place rewrite "
             "would require duplication");
      seenProducers.insert(fe.producer);
      schedule.lanes[fe.producer] = schedule.lanes.lookup(fe.consumer);
      schedule.alignment[fe.producer] = g;
    }
    ++g;
  }

  return static_cast<unsigned>(numLayoutArgs);
}

}  // namespace heir
}  // namespace mlir

#endif  // OPTIMIZATION_H
