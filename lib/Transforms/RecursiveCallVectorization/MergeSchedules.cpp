//===- NeedlemanWunschMerge.cpp - NW-based MLIR function merge --*- C++ -*-===//
//
// Implementation of Needleman-Wunsch sequence alignment for merging two
// MLIR func::FuncOp with identical signatures.
//
//===----------------------------------------------------------------------===//

#include "lib/Transforms/RecursiveCallVectorization/MergeSchedules.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"             // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// Helper: find the secret.generic body block inside a func
//===----------------------------------------------------------------------===//

static Block *getSecretGenericBody(func::FuncOp func) {
  Block *result = nullptr;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "secret.generic") {
      // The body is the first block of the first region
      if (!op->getRegions().empty() && !op->getRegion(0).empty()) {
        result = &op->getRegion(0).front();
      }
    }
  });
  return result;
}

/// Get the secret.generic op inside a func.
static Operation *getSecretGenericOp(func::FuncOp func) {
  Operation *result = nullptr;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "secret.generic") {
      result = op;
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Step 1: Extract operations in topological order using CircuitGraph
//===----------------------------------------------------------------------===//

llvm::SmallVector<Operation *> extractSortedOps(func::FuncOp func) {
  Block *body = getSecretGenericBody(func);
  if (!body) return {};

  // Collect schedulable ops (skip secret.yield and other terminators)
  llvm::SetVector<Operation *> opSet;
  for (Operation &op : *body) {
    if (op.getName().getStringRef() == "secret.yield") continue;
    opSet.insert(&op);
  }

  if (opSet.empty()) return {};

  // Use MLIR's built-in topological sort
  auto sorted = mlir::topologicalSort(opSet);

  return llvm::SmallVector<Operation *>(sorted.begin(), sorted.end());
}

//===----------------------------------------------------------------------===//
// Step 2: NW scoring
//===----------------------------------------------------------------------===//

/// Trace an operand back to its "origin": if it's a block argument of the
/// body (ciphertext), return its arg number; if it's defined by an op in the
/// sorted sequence, return that op's index. Returns -1000 for
/// external/plaintext values (func args used via implicit capture, etc.).
static int64_t traceOrigin(Value val,
                           const llvm::DenseMap<Operation *, int64_t> &opIndex,
                           Block *bodyBlock) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    // Only body block args (ciphertext) get a stable origin index.
    // Func-level block args (plaintext, captured implicitly) are external.
    if (blockArg.getOwner() == bodyBlock) return -(blockArg.getArgNumber() + 1);
    return -1000;  // plaintext func arg
  }

  Operation *defOp = val.getDefiningOp();
  if (defOp) {
    auto it = opIndex.find(defOp);
    if (it != opIndex.end()) return it->second;
  }
  return -1000;  // external / unknown
}

/// Check if two constants have identical values.
static bool sameConstantValue(Operation *a, Operation *b) {
  auto constA = dyn_cast<arith::ConstantOp>(a);
  auto constB = dyn_cast<arith::ConstantOp>(b);
  if (!constA || !constB) return false;
  return constA.getValue() == constB.getValue();
}

/// Score two operations for NW alignment.
/// bodyBlockA/bodyBlockB are the secret.generic body blocks, used to
/// distinguish ciphertext block args from plaintext implicit captures.
static int scoreOps(Operation *a, Operation *b,
                    const llvm::DenseMap<Operation *, int64_t> &indexA,
                    const llvm::DenseMap<Operation *, int64_t> &indexB,
                    Block *bodyBlockA, Block *bodyBlockB,
                    const NWScoreConfig &config) {
  llvm::StringRef nameA = a->getName().getStringRef();
  llvm::StringRef nameB = b->getName().getStringRef();

  // Different dialect entirely
  auto dialectA = nameA.split('.').first;
  auto dialectB = nameB.split('.').first;

  if (dialectA != dialectB) return config.mismatch;

  // Same dialect, different opcode
  if (nameA != nameB) return config.matchClass;

  // Same opcode — check operand structure for exact match
  // Special case: constants match exactly if they have the same value
  if (isa<arith::ConstantOp>(a)) {
    return sameConstantValue(a, b) ? config.matchExact : config.matchOpcode;
  }

  // For other ops: check operand origins AND plaintext constraints.
  // If any operand is plaintext (external to the body), the ops can only merge
  // if the plaintext values are compile-time constants with equal values.
  if (a->getNumOperands() == b->getNumOperands()) {
    bool allMatch = true;
    for (unsigned i = 0; i < a->getNumOperands(); ++i) {
      Value opndA = a->getOperand(i);
      Value opndB = b->getOperand(i);
      int64_t originA = traceOrigin(opndA, indexA, bodyBlockA);
      int64_t originB = traceOrigin(opndB, indexB, bodyBlockB);

      // If either operand is external (plaintext), check if both are the
      // same compile-time constant. If not, can't merge.
      if (originA == -1000 || originB == -1000) {
        Operation *defA = opndA.getDefiningOp();
        Operation *defB = opndB.getDefiningOp();
        if (!defA || !defB || !sameConstantValue(defA, defB)) {
          return config.mismatch;  // different or unknown plaintext
        }
        // Same constant value — this operand is fine, continue checking
        continue;
      }

      if (originA != originB) {
        allMatch = false;
        break;
      }
    }
    if (allMatch) return config.matchExact;
  }

  return config.matchOpcode;
}

//===----------------------------------------------------------------------===//
// Step 2: Needleman-Wunsch DP + traceback
//===----------------------------------------------------------------------===//

llvm::SmallVector<AlignmentEntry> runNeedlemanWunsch(
    llvm::ArrayRef<Operation *> seqA, llvm::ArrayRef<Operation *> seqB,
    const NWScoreConfig &config) {
  int M = seqA.size();
  int N = seqB.size();

  // Infer body blocks from the ops themselves
  Block *bodyBlockA = M > 0 ? seqA[0]->getBlock() : nullptr;
  Block *bodyBlockB = N > 0 ? seqB[0]->getBlock() : nullptr;

  // Build op->index maps for origin tracing
  llvm::DenseMap<Operation *, int64_t> indexA, indexB;
  for (int i = 0; i < M; ++i) indexA[seqA[i]] = i;
  for (int j = 0; j < N; ++j) indexB[seqB[j]] = j;

  // DP matrix
  std::vector<std::vector<int>> dp(M + 1, std::vector<int>(N + 1, 0));
  for (int i = 1; i <= M; ++i) dp[i][0] = i * config.gapPenalty;
  for (int j = 1; j <= N; ++j) dp[0][j] = j * config.gapPenalty;

  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      int matchScore =
          dp[i - 1][j - 1] + scoreOps(seqA[i - 1], seqB[j - 1], indexA, indexB,
                                      bodyBlockA, bodyBlockB, config);
      int gapA = dp[i - 1][j] + config.gapPenalty;
      int gapB = dp[i][j - 1] + config.gapPenalty;
      dp[i][j] = std::max({matchScore, gapA, gapB});
    }
  }

  // Traceback
  llvm::SmallVector<AlignmentEntry> alignment;
  int i = M, j = N;

  while (i > 0 || j > 0) {
    AlignmentEntry entry;

    if (i > 0 && j > 0) {
      int diagScore = scoreOps(seqA[i - 1], seqB[j - 1], indexA, indexB,
                               bodyBlockA, bodyBlockB, config);
      if (dp[i][j] == dp[i - 1][j - 1] + diagScore) {
        // Diagonal move — only treat as Match if score >= matchOpcode
        if (diagScore >= config.matchOpcode) {
          entry.kind = AlignmentEntry::Match;
          entry.opA = seqA[i - 1];
          entry.opB = seqB[j - 1];
          alignment.push_back(entry);
        } else {
          // Downgrade: emit both as gaps
          entry.kind = AlignmentEntry::GapB;
          entry.opA = seqA[i - 1];
          alignment.push_back(entry);
          AlignmentEntry entryB;
          entryB.kind = AlignmentEntry::GapA;
          entryB.opB = seqB[j - 1];
          alignment.push_back(entryB);
        }
        --i;
        --j;
        continue;
      }
    }

    if (i > 0 && dp[i][j] == dp[i - 1][j] + config.gapPenalty) {
      entry.kind = AlignmentEntry::GapB;
      entry.opA = seqA[i - 1];
      alignment.push_back(entry);
      --i;
    } else {
      entry.kind = AlignmentEntry::GapA;
      entry.opB = seqB[j - 1];
      alignment.push_back(entry);
      --j;
    }
  }

  // Reverse: traceback produces entries in reverse order
  std::reverse(alignment.begin(), alignment.end());
  return alignment;
}

//===----------------------------------------------------------------------===//
// Step 3: Build merged function (pure alignment — no type changes)
//===----------------------------------------------------------------------===//

LogicalResult mergeWithNeedlemanWunsch(func::FuncOp funcA, func::FuncOp funcB,
                                       func::FuncOp &result,
                                       const NWScoreConfig &config) {
  // --- Validate: signatures must be identical (pure alignment, no widening)
  // ---
  if (funcA.getFunctionType() != funcB.getFunctionType()) {
    llvm::errs() << "NW Merge: function types do not match\n";
    return failure();
  }

  Block *bodyA = getSecretGenericBody(funcA);
  Block *bodyB = getSecretGenericBody(funcB);
  if (!bodyA || !bodyB) {
    llvm::errs() << "NW Merge: could not find secret.generic body\n";
    return failure();
  }

  Operation *genericA = getSecretGenericOp(funcA);
  Operation *genericB = getSecretGenericOp(funcB);
  if (!genericA || !genericB) return failure();

  // --- Step 1: Extract + topo sort ---
  auto seqA = extractSortedOps(funcA);
  auto seqB = extractSortedOps(funcB);

  if (seqA.empty() && seqB.empty()) return failure();

  // --- Step 2: Run NW alignment ---
  auto alignment = runNeedlemanWunsch(seqA, seqB, config);

  // --- Step 3: Build merged function (same signature as A; yield A's values)
  // ---
  MLIRContext *ctx = funcA.getContext();
  Location loc = funcA.getLoc();
  OpBuilder builder(ctx);

  FunctionType mergedFuncType = funcA.getFunctionType();
  std::string mergedName =
      (funcA.getName() + "_nw_merged_" + funcB.getName()).str();
  auto mergedFunc = func::FuncOp::create(loc, mergedName, mergedFuncType);

  // Copy attributes from A (signature-dependent attrs are unchanged here, but
  // we still skip them defensively in case A and B differ).
  for (auto attr : funcA->getAttrs()) {
    if (attr.getName() == "sym_name" || attr.getName() == "function_type")
      continue;
    mergedFunc->setAttr(attr.getName(), attr.getValue());
  }

  Block *funcBody = mergedFunc.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  // Map both A's and B's func args 1:1 to the merged func args (same types).
  IRMapping outerRemapA, outerRemapB;
  Block &funcEntryA = funcA.front();
  Block &funcEntryB = funcB.front();
  for (unsigned i = 0; i < funcEntryA.getNumArguments(); ++i) {
    outerRemapA.map(funcEntryA.getArgument(i), funcBody->getArgument(i));
    outerRemapB.map(funcEntryB.getArgument(i), funcBody->getArgument(i));
  }

  // Clone any non-func-arg operands of A's/B's secret.generic into the merged
  // func body so they can be passed as merged-generic operands.
  for (Value origOperand : genericA->getOperands()) {
    if (!outerRemapA.contains(origOperand)) {
      if (Operation *defOp = origOperand.getDefiningOp()) {
        Operation *cloned = builder.clone(*defOp);
        outerRemapA.map(origOperand, cloned->getResult(0));
      }
    }
  }
  for (Value origOperand : genericB->getOperands()) {
    if (!outerRemapB.contains(origOperand)) {
      if (Operation *defOp = origOperand.getDefiningOp()) {
        Operation *cloned = builder.clone(*defOp);
        outerRemapB.map(origOperand, cloned->getResult(0));
      }
    }
  }

  // Build the merged secret.generic. Operands = A's, then any extra from B.
  // Block arg types come straight from the originals (no widening).
  SmallVector<Value> genericOperands;
  auto *genericBlock = new Block();

  for (unsigned i = 0; i < genericA->getNumOperands(); ++i) {
    genericOperands.push_back(
        outerRemapA.lookupOrDefault(genericA->getOperand(i)));
    genericBlock->addArgument(bodyA->getArgument(i).getType(), loc);
  }
  unsigned numArgsFromA = genericA->getNumOperands();
  for (unsigned i = numArgsFromA; i < genericB->getNumOperands(); ++i) {
    genericOperands.push_back(
        outerRemapB.lookupOrDefault(genericB->getOperand(i)));
    genericBlock->addArgument(bodyB->getArgument(i).getType(), loc);
  }

  // Merged generic result types = A's generic result types (same signature).
  SmallVector<Type> mergedGenericResultTypes(genericA->getResultTypes().begin(),
                                             genericA->getResultTypes().end());

  OperationState genericState(loc, "secret.generic");
  genericState.addOperands(genericOperands);
  genericState.addTypes(mergedGenericResultTypes);
  genericState.addRegion()->push_back(genericBlock);
  Operation *mergedGeneric = builder.create(genericState);

  builder.setInsertionPointToStart(genericBlock);

  // Map A's and B's inner block args to the merged block args.
  IRMapping remapA, remapB;
  for (unsigned i = 0; i < bodyA->getNumArguments(); ++i)
    remapA.map(bodyA->getArgument(i), genericBlock->getArgument(i));
  for (unsigned i = 0; i < bodyB->getNumArguments(); ++i) {
    // B shares the leading args with A (identical signatures imply identical
    // block arg counts and types); any extras go to B's appended block args.
    unsigned mergedIdx = (i < numArgsFromA) ? i : i;
    remapB.map(bodyB->getArgument(i), genericBlock->getArgument(mergedIdx));
  }

  // Handle implicit captures of func-level values inside the body.
  auto cloneOuterDeps = [&](Block *origBody, IRMapping &remap,
                            IRMapping &outerRemap) {
    OpBuilder outerBuilder(ctx);
    outerBuilder.setInsertionPoint(mergedGeneric);
    for (Operation &op : *origBody) {
      for (Value operand : op.getOperands()) {
        if (remap.contains(operand)) continue;
        if (isa<BlockArgument>(operand)) {
          if (outerRemap.contains(operand))
            remap.map(operand, outerRemap.lookup(operand));
          continue;
        }
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() == origBody) continue;
        Operation *cloned = outerBuilder.clone(*defOp, outerRemap);
        remap.map(operand, cloned->getResult(0));
      }
    }
  };

  cloneOuterDeps(bodyA, remapA, outerRemapA);
  cloneOuterDeps(bodyB, remapB, outerRemapB);

  // Walk alignment and clone operations as-is (no type changes).
  Operation *yieldA = bodyA->getTerminator();
  for (const auto &entry : alignment) {
    switch (entry.kind) {
      case AlignmentEntry::Match: {
        // Emit opA once; B's downstream uses share A's results.
        Operation *cloned = builder.clone(*entry.opA, remapA);
        for (unsigned k = 0; k < entry.opB->getNumResults(); ++k)
          remapB.map(entry.opB->getResult(k), cloned->getResult(k));
        break;
      }
      case AlignmentEntry::GapA:
        builder.clone(*entry.opB, remapB);
        break;
      case AlignmentEntry::GapB:
        builder.clone(*entry.opA, remapA);
        break;
    }
  }

  // Build secret.yield from A's yield values (same signature as A).
  SmallVector<Value> yieldOperands;
  for (Value v : yieldA->getOperands())
    yieldOperands.push_back(remapA.lookupOrDefault(v));

  OperationState yieldState(loc, "secret.yield");
  yieldState.addOperands(yieldOperands);
  builder.create(yieldState);

  // Build func.return.
  builder.setInsertionPointAfter(mergedGeneric);
  SmallVector<Value> returnOperands(mergedGeneric->getResults().begin(),
                                    mergedGeneric->getResults().end());
  func::ReturnOp::create(builder, loc, returnOperands);

  result = mergedFunc;
  return success();
}

//===----------------------------------------------------------------------===//
// Step 3b: Schedule-level merge — N-way with pairwise NW + final interleave
//===----------------------------------------------------------------------===//

namespace {
/// One step's worth of pairwise NW merge: takes a "running" func + its
/// origins (which original kernels each running op represents) and merges
/// with a fresh kernel `funcB` at kernel index `kernelIdxB`.
///
/// Outputs a new merged function whose body contains cloned ops. Populates
/// `newOrigins` with a list of (kernelIdx, originalOpInThatKernel) per cloned
/// op, representing which kernels' computations the merged op corresponds to.
LogicalResult pairwiseScheduleMergeStep(
    func::FuncOp runningFunc,
    const llvm::DenseMap<Operation *,
                         llvm::SmallVector<std::pair<unsigned, Operation *>, 8>>
        &runningOrigins,
    llvm::ArrayRef<Operation *> runningSeq, func::FuncOp funcB,
    llvm::ArrayRef<Operation *> seqB, unsigned kernelIdxB,
    const NWScoreConfig &config, func::FuncOp &newMergedFunc,
    llvm::SmallVector<Operation *> &newSeq,
    llvm::DenseMap<Operation *,
                   llvm::SmallVector<std::pair<unsigned, Operation *>, 8>>
        &newOrigins) {
  using Origin = std::pair<unsigned, Operation *>;
  using OriginList = llvm::SmallVector<Origin, 8>;

  Block *bodyA = getSecretGenericBody(runningFunc);
  Block *bodyB = getSecretGenericBody(funcB);
  if (!bodyA || !bodyB) {
    llvm::errs() << "NW Merge: could not find secret.generic body\n";
    return failure();
  }

  Operation *genericA = getSecretGenericOp(runningFunc);
  Operation *genericB = getSecretGenericOp(funcB);
  if (!genericA || !genericB) return failure();

  auto alignment = runNeedlemanWunsch(runningSeq, seqB, config);

  MLIRContext *ctx = runningFunc.getContext();
  Location loc = runningFunc.getLoc();
  OpBuilder builder(ctx);

  FunctionType mergedFuncType = runningFunc.getFunctionType();
  std::string mergedName =
      (runningFunc.getName() + "_nw_" + funcB.getName()).str();
  newMergedFunc = func::FuncOp::create(loc, mergedName, mergedFuncType);

  for (auto attr : runningFunc->getAttrs()) {
    if (attr.getName() == "sym_name" || attr.getName() == "function_type")
      continue;
    newMergedFunc->setAttr(attr.getName(), attr.getValue());
  }

  Block *funcBody = newMergedFunc.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  IRMapping outerRemapA, outerRemapB;
  Block &funcEntryA = runningFunc.front();
  Block &funcEntryB = funcB.front();
  for (unsigned i = 0; i < funcEntryA.getNumArguments(); ++i) {
    outerRemapA.map(funcEntryA.getArgument(i), funcBody->getArgument(i));
    outerRemapB.map(funcEntryB.getArgument(i), funcBody->getArgument(i));
  }

  for (Value origOperand : genericA->getOperands()) {
    if (!outerRemapA.contains(origOperand)) {
      if (Operation *defOp = origOperand.getDefiningOp()) {
        Operation *cloned = builder.clone(*defOp);
        outerRemapA.map(origOperand, cloned->getResult(0));
      }
    }
  }
  for (Value origOperand : genericB->getOperands()) {
    if (!outerRemapB.contains(origOperand)) {
      if (Operation *defOp = origOperand.getDefiningOp()) {
        Operation *cloned = builder.clone(*defOp);
        outerRemapB.map(origOperand, cloned->getResult(0));
      }
    }
  }

  SmallVector<Value> genericOperands;
  auto *genericBlock = new Block();
  for (unsigned i = 0; i < genericA->getNumOperands(); ++i) {
    genericOperands.push_back(
        outerRemapA.lookupOrDefault(genericA->getOperand(i)));
    genericBlock->addArgument(bodyA->getArgument(i).getType(), loc);
  }
  unsigned numArgsFromA = genericA->getNumOperands();
  for (unsigned i = numArgsFromA; i < genericB->getNumOperands(); ++i) {
    genericOperands.push_back(
        outerRemapB.lookupOrDefault(genericB->getOperand(i)));
    genericBlock->addArgument(bodyB->getArgument(i).getType(), loc);
  }

  SmallVector<Type> mergedGenericResultTypes(genericA->getResultTypes().begin(),
                                             genericA->getResultTypes().end());

  OperationState genericState(loc, "secret.generic");
  genericState.addOperands(genericOperands);
  genericState.addTypes(mergedGenericResultTypes);
  genericState.addRegion()->push_back(genericBlock);
  Operation *mergedGeneric = builder.create(genericState);

  builder.setInsertionPointToStart(genericBlock);

  IRMapping remapA, remapB;
  for (unsigned i = 0; i < bodyA->getNumArguments(); ++i)
    remapA.map(bodyA->getArgument(i), genericBlock->getArgument(i));
  for (unsigned i = 0; i < bodyB->getNumArguments(); ++i) {
    unsigned mergedIdx = (i < numArgsFromA) ? i : i;
    remapB.map(bodyB->getArgument(i), genericBlock->getArgument(mergedIdx));
  }

  auto cloneOuterDeps = [&](Block *origBody, IRMapping &remap,
                            IRMapping &outerRemap) {
    OpBuilder outerBuilder(ctx);
    outerBuilder.setInsertionPoint(mergedGeneric);
    for (Operation &op : *origBody) {
      for (Value operand : op.getOperands()) {
        if (remap.contains(operand)) continue;
        if (isa<BlockArgument>(operand)) {
          if (outerRemap.contains(operand))
            remap.map(operand, outerRemap.lookup(operand));
          continue;
        }
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() == origBody) continue;
        Operation *cloned = outerBuilder.clone(*defOp, outerRemap);
        remap.map(operand, cloned->getResult(0));
      }
    }
  };
  cloneOuterDeps(bodyA, remapA, outerRemapA);
  cloneOuterDeps(bodyB, remapB, outerRemapB);

  // Walk alignment: clone ops + update origins.
  Operation *yieldA = bodyA->getTerminator();
  newSeq.clear();
  for (const auto &entry : alignment) {
    Operation *cloned = nullptr;
    OriginList ol;
    switch (entry.kind) {
      case AlignmentEntry::Match: {
        cloned = builder.clone(*entry.opA, remapA);
        for (unsigned k = 0; k < entry.opB->getNumResults(); ++k)
          remapB.map(entry.opB->getResult(k), cloned->getResult(k));
        // origins[cloned] = origins[opA] ∪ {(kernelIdxB, opB)}
        auto it = runningOrigins.find(entry.opA);
        if (it != runningOrigins.end())
          ol.append(it->second.begin(), it->second.end());
        ol.push_back({kernelIdxB, entry.opB});
        break;
      }
      case AlignmentEntry::GapA: {
        cloned = builder.clone(*entry.opB, remapB);
        ol.push_back({kernelIdxB, entry.opB});
        break;
      }
      case AlignmentEntry::GapB: {
        cloned = builder.clone(*entry.opA, remapA);
        auto it = runningOrigins.find(entry.opA);
        if (it != runningOrigins.end())
          ol.append(it->second.begin(), it->second.end());
        break;
      }
    }
    if (cloned) {
      newOrigins[cloned] = std::move(ol);
      newSeq.push_back(cloned);
    }
  }

  // Build secret.yield from A's yield values.
  SmallVector<Value> yieldOperands;
  for (Value v : yieldA->getOperands())
    yieldOperands.push_back(remapA.lookupOrDefault(v));

  OperationState yieldState(loc, "secret.yield");
  yieldState.addOperands(yieldOperands);
  builder.create(yieldState);

  builder.setInsertionPointAfter(mergedGeneric);
  SmallVector<Value> returnOperands(mergedGeneric->getResults().begin(),
                                    mergedGeneric->getResults().end());
  func::ReturnOp::create(builder, loc, returnOperands);

  return success();
}
}  // namespace

//===----------------------------------------------------------------------===//
// Step 3b: Schedule-level merge — NW on (cycle, op_type) sequences + mod-N
//
// Algorithm:
//   Phase 1: For each kernel, build a cycle-type sequence — a sorted list of
//            (origCycle, opType) pairs by walking the kernel's alignment map.
//            All ops at the same cycle must share an opcode (asserted).
//   Phase 2: NW-align cycle-type sequences pairwise across N kernels.
//   Phase 3: Build a new merged FuncOp + secret.generic body. For each merge
//            step, clone each participating kernel's ops at that step's
//            original cycle into the merged body. Each clone has a fresh
//            Operation* — no shallow-sharing.
//   Phase 4: Populate the merged Schedule keyed by the clones. Apply mod-N
//            lanes: merged_lane = origLane * N + kernel_idx. merged_cycle is
//            the step's sequential index. warpSize = max * N.
//===----------------------------------------------------------------------===//

namespace {

struct CycleEntry {
  int64_t origCycle;
  OperationName opType;
};

struct MergeStep {
  // OperationName has no default constructor, so wrap in optional. It's
  // always set before use.
  std::optional<OperationName> opType;
  // Per kernel 0..N-1: original cycle in that kernel, or -1 if kernel
  // does not participate at this step.
  llvm::SmallVector<int64_t, 8> kernelCycles;
};

struct PairAlignEntry {
  enum Kind { Match, GapA, GapB } kind;
  int posA = -1;
  int posB = -1;
};

/// Build a sorted (cycle, opType) sequence from a Schedule. Asserts all ops
/// at the same cycle share an opcode.
llvm::SmallVector<CycleEntry> buildCycleTypeSeq(const Schedule &s) {
  llvm::DenseMap<int64_t, OperationName> typeAtCycle;
  for (const auto &kv : s.alignment) {
    Operation *op = kv.first;
    int64_t cycle = kv.second;
    auto it = typeAtCycle.find(cycle);
    if (it != typeAtCycle.end()) {
      assert(it->second == op->getName() &&
             "NW Merge: all ops at the same cycle must share an opcode");
    } else {
      typeAtCycle.insert({cycle, op->getName()});
    }
  }
  llvm::SmallVector<int64_t> cycles;
  for (const auto &kv : typeAtCycle) cycles.push_back(kv.first);
  llvm::sort(cycles);
  llvm::SmallVector<CycleEntry> result;
  result.reserve(cycles.size());
  for (int64_t c : cycles) {
    auto it = typeAtCycle.find(c);
    assert(it != typeAtCycle.end());
    result.push_back({c, it->second});
  }
  return result;
}

/// Standard NW alignment on cycle-type sequences. Score: +4 match on opcode,
/// -1 mismatch, -2 gap. Returns alignment entries indexing into seqA / seqB.
llvm::SmallVector<PairAlignEntry> nwAlignCycleSeqs(
    llvm::ArrayRef<CycleEntry> seqA, llvm::ArrayRef<CycleEntry> seqB) {
  int M = (int)seqA.size();
  int N = (int)seqB.size();
  constexpr int kMatch = 4, kMismatch = -1, kGap = -2;

  std::vector<std::vector<int>> dp(M + 1, std::vector<int>(N + 1, 0));
  for (int i = 1; i <= M; ++i) dp[i][0] = i * kGap;
  for (int j = 1; j <= N; ++j) dp[0][j] = j * kGap;
  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      int sc = (seqA[i - 1].opType == seqB[j - 1].opType) ? kMatch : kMismatch;
      dp[i][j] = std::max(
          {dp[i - 1][j - 1] + sc, dp[i - 1][j] + kGap, dp[i][j - 1] + kGap});
    }
  }

  llvm::SmallVector<PairAlignEntry> result;
  int i = M, j = N;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0) {
      int sc = (seqA[i - 1].opType == seqB[j - 1].opType) ? kMatch : kMismatch;
      if (dp[i][j] == dp[i - 1][j - 1] + sc && sc == kMatch) {
        result.push_back({PairAlignEntry::Match, i - 1, j - 1});
        --i;
        --j;
        continue;
      }
    }
    if (i > 0 && dp[i][j] == dp[i - 1][j] + kGap) {
      result.push_back({PairAlignEntry::GapB, i - 1, -1});
      --i;
    } else {
      result.push_back({PairAlignEntry::GapA, -1, j - 1});
      --j;
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

/// Pairwise-reduce N cycle-type sequences into a single ordered list of
/// MergeSteps. Each step records which kernels participate (kernelCycles[k]
/// >= 0) and what their original cycle was in that kernel.
llvm::SmallVector<MergeStep> nwReduceNWay(
    llvm::ArrayRef<llvm::SmallVector<CycleEntry>> cycleSeqs) {
  unsigned N = cycleSeqs.size();
  llvm::SmallVector<MergeStep> running;
  // Initialize from kernel 0.
  for (const auto &e : cycleSeqs[0]) {
    MergeStep s{};
    s.opType = e.opType;
    s.kernelCycles.assign(N, -1);
    s.kernelCycles[0] = e.origCycle;
    running.push_back(std::move(s));
  }

  for (unsigned k = 1; k < N; ++k) {
    // Build a CycleEntry seq from running (positional index serves as
    // "cycle" for the NW input).
    llvm::SmallVector<CycleEntry> runSeq;
    runSeq.reserve(running.size());
    for (size_t i = 0; i < running.size(); ++i)
      runSeq.push_back({(int64_t)i, *running[i].opType});

    auto align = nwAlignCycleSeqs(runSeq, cycleSeqs[k]);
    llvm::SmallVector<MergeStep> next;
    for (const auto &e : align) {
      if (e.kind == PairAlignEntry::Match) {
        MergeStep s = running[e.posA];
        s.kernelCycles[k] = cycleSeqs[k][e.posB].origCycle;
        next.push_back(std::move(s));
      } else if (e.kind == PairAlignEntry::GapA) {
        MergeStep s{};
        s.opType = cycleSeqs[k][e.posB].opType;
        s.kernelCycles.assign(N, -1);
        s.kernelCycles[k] = cycleSeqs[k][e.posB].origCycle;
        next.push_back(std::move(s));
      } else {  // GapB
        next.push_back(running[e.posA]);
      }
    }
    running = std::move(next);
  }
  return running;
}

/// Collect all ops at `cycle` in `schedule`, sorted by lane.
llvm::SmallVector<Operation *> opsAtCycle(const Schedule &schedule,
                                          int64_t cycle) {
  llvm::SmallVector<Operation *> ops;
  for (const auto &kv : schedule.alignment) {
    if (kv.second == cycle) ops.push_back(kv.first);
  }
  llvm::sort(ops, [&](Operation *a, Operation *b) {
    return schedule.lanes.lookup(a) < schedule.lanes.lookup(b);
  });
  return ops;
}

}  // namespace

LogicalResult mergeSchedulesWithNW(llvm::ArrayRef<func::FuncOp> funcs,
                                   llvm::ArrayRef<Schedule> schedules,
                                   func::FuncOp &mergedFunc,
                                   Schedule &mergedSchedule,
                                   const NWScoreConfig &config) {
  unsigned N = funcs.size();
  if (N < 2 || N != schedules.size()) {
    llvm::errs() << "NW Merge: need >= 2 funcs with matching schedules\n";
    return failure();
  }

  // Validate signatures match across all funcs.
  func::FuncOp f0 = funcs[0];
  FunctionType ft0 = f0.getFunctionType();
  for (unsigned k = 1; k < N; ++k) {
    func::FuncOp fk = funcs[k];
    if (fk.getFunctionType() != ft0) {
      llvm::errs() << "NW Merge: function types differ at idx " << k << "\n";
      return failure();
    }
  }

  // --- Phase 1: Cycle-type sequences per kernel ---
  llvm::SmallVector<llvm::SmallVector<CycleEntry>> cycleSeqs;
  cycleSeqs.reserve(N);
  for (unsigned k = 0; k < N; ++k)
    cycleSeqs.push_back(buildCycleTypeSeq(schedules[k]));

  // --- Phase 2: NW reduction ---
  auto mergeSteps = nwReduceNWay(cycleSeqs);

  // --- Phase 3: Build merged FuncOp and clone ops per merge step ---
  MLIRContext *ctx = f0.getContext();
  Location loc = f0.getLoc();
  OpBuilder builder(ctx);

  // Build the merged function with N*R inputs (kernel-major). Result types
  // are placeholders here; they'll be replaced once Phase 5 assembles the
  // per-kernel result tensors.
  unsigned R = ft0.getNumInputs();
  llvm::SmallVector<Type> wideInputs;
  wideInputs.reserve(N * R);
  for (unsigned k = 0; k < N; ++k)
    for (Type t : ft0.getInputs()) wideInputs.push_back(t);
  FunctionType mergedFuncType =
      FunctionType::get(ctx, wideInputs, ft0.getResults());

  std::string mergedName = (f0.getName() + "_nw_merged").str();
  mergedFunc = func::FuncOp::create(loc, mergedName, mergedFuncType);
  for (auto attr : f0->getAttrs()) {
    if (attr.getName() == "sym_name" || attr.getName() == "function_type" ||
        attr.getName() == "arg_attrs" || attr.getName() == "res_attrs")
      continue;
    mergedFunc->setAttr(attr.getName(), attr.getValue());
  }

  // Gather each kernel's secret.generic + body.
  llvm::SmallVector<Operation *> origGenerics(N, nullptr);
  llvm::SmallVector<Block *> origBodies(N, nullptr);
  for (unsigned k = 0; k < N; ++k) {
    origGenerics[k] = getSecretGenericOp(funcs[k]);
    origBodies[k] = getSecretGenericBody(funcs[k]);
    if (!origGenerics[k] || !origBodies[k]) {
      llvm::errs() << "NW Merge: kernel " << k << " missing secret.generic\n";
      return failure();
    }
  }

  Block *mergedBody = mergedFunc.addEntryBlock();
  builder.setInsertionPointToStart(mergedBody);

  // Per-kernel outer mappings: each kernel k's func args map to merged
  // func args [k*R, k*R+R).
  llvm::SmallVector<IRMapping> outerMappings(N);
  for (unsigned k = 0; k < N; ++k) {
    func::FuncOp fk = funcs[k];
    Block &entry = fk.front();
    for (unsigned i = 0; i < entry.getNumArguments(); ++i)
      outerMappings[k].map(entry.getArgument(i),
                           mergedBody->getArgument(k * R + i));
  }

  // Clone non-arg operands of each kernel's secret.generic into the merged
  // func body. Each kernel gets its own clone routed via its outerMappings.
  for (unsigned k = 0; k < N; ++k) {
    for (Value origOperand : origGenerics[k]->getOperands()) {
      if (!outerMappings[k].contains(origOperand)) {
        if (Operation *defOp = origOperand.getDefiningOp()) {
          Operation *cloned = builder.clone(*defOp);
          outerMappings[k].map(origOperand, cloned->getResult(0));
        }
      }
    }
  }

  // Build the merged secret.generic's body block. The op itself is created
  // at the end, once we know the assembled per-kernel result types. Operands
  // and body block args are kernel-major: kernel k contributes operands
  // [k*RG, k*RG+RG) and body args of the same range, where RG is the
  // per-kernel secret.generic operand count.
  unsigned RG = origGenerics[0]->getNumOperands();
  llvm::SmallVector<Value> mergedGenericOperands;
  mergedGenericOperands.reserve(N * RG);
  auto *mergedGenericBlock = new Block();
  for (unsigned k = 0; k < N; ++k) {
    Operation *genericK = origGenerics[k];
    Block *bodyK = origBodies[k];
    for (unsigned i = 0; i < genericK->getNumOperands(); ++i) {
      mergedGenericOperands.push_back(
          outerMappings[k].lookupOrDefault(genericK->getOperand(i)));
      mergedGenericBlock->addArgument(bodyK->getArgument(i).getType(), loc);
    }
  }

  builder.setInsertionPointToStart(mergedGenericBlock);

  // Per-kernel inner mappings: each kernel k's body block args map to merged
  // body block args [k*RG, k*RG+RG).
  llvm::SmallVector<IRMapping> innerMappings(N);
  for (unsigned k = 0; k < N; ++k) {
    Block *bodyK = origBodies[k];
    for (unsigned i = 0; i < bodyK->getNumArguments(); ++i)
      innerMappings[k].map(bodyK->getArgument(i),
                           mergedGenericBlock->getArgument(k * RG + i));
  }

  // Clone implicit captures (constants and other outer values referenced
  // inside each kernel's body) into the merged func body. The generic op
  // doesn't exist yet; captures sit at the end of the func body and will
  // dominate the generic when it's created below.
  auto cloneCaptures = [&](unsigned k) {
    OpBuilder outerBuilder(ctx);
    outerBuilder.setInsertionPointToEnd(mergedBody);
    for (Operation &op : *origBodies[k]) {
      for (Value operand : op.getOperands()) {
        if (innerMappings[k].contains(operand)) continue;
        if (isa<BlockArgument>(operand)) {
          if (outerMappings[k].contains(operand))
            innerMappings[k].map(operand, outerMappings[k].lookup(operand));
          continue;
        }
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() == origBodies[k]) continue;
        Operation *cloned = outerBuilder.clone(*defOp, outerMappings[k]);
        innerMappings[k].map(operand, cloned->getResult(0));
      }
    }
  };
  for (unsigned k = 0; k < N; ++k) cloneCaptures(k);

  // --- Phase 4: Walk merge steps, clone ops per kernel, populate Schedule ---
  mergedSchedule.lanes.clear();
  mergedSchedule.alignment.clear();
  mergedSchedule.instructions.clear();
  unsigned maxOrigWarp = 0;
  for (const Schedule &s : schedules)
    maxOrigWarp = std::max(maxOrigWarp, s.warpSize);
  mergedSchedule.warpSize = maxOrigWarp * N;

  int64_t mergedCycleCounter = 0;
  for (const MergeStep &step : mergeSteps) {
    int64_t mergedCycle = mergedCycleCounter++;
    for (unsigned k = 0; k < N; ++k) {
      if (step.kernelCycles[k] < 0) continue;
      int64_t origCycle = step.kernelCycles[k];
      auto ops = opsAtCycle(schedules[k], origCycle);
      for (Operation *origOp : ops) {
        Operation *clonedOp = builder.clone(*origOp, innerMappings[k]);
        int64_t origLane = schedules[k].lanes.lookup(origOp);
        mergedSchedule.lanes[clonedOp] = origLane * (int64_t)N + (int64_t)k;
        mergedSchedule.alignment[clonedOp] = mergedCycle;
        mergedSchedule.instructions.push_back(clonedOp);
      }
    }
  }

  // --- Phase 5: Per-kernel result collection (scalar interface) ---
  // For each kernel k (in functionsToMerge order), walk each original yield
  // operand's tensor.insert chain and collect the leaf scalar SSA values
  // (looked up via innerMappings[k]). No insert chain is rebuilt — the
  // merged secret.generic returns scalars directly. Order is kernel-major,
  // with scalars in original program order within each kernel.
  llvm::SmallVector<Value> assembledResults;
  llvm::SmallVector<Type> assembledTypes;
  for (unsigned k = 0; k < N; ++k) {
    Operation *yieldK = origBodies[k]->getTerminator();
    for (Value yieldOperand : yieldK->getOperands()) {
      llvm::SmallVector<Value> scalars;
      Value cur = yieldOperand;
      while (auto ins = cur.getDefiningOp<tensor::InsertOp>()) {
        scalars.push_back(ins.getScalar());
        cur = ins.getDest();
      }
      // Walk above collected scalars in reverse chain order; emit them in
      // original program order.
      for (auto it = scalars.rbegin(); it != scalars.rend(); ++it) {
        Value scalar = innerMappings[k].lookupOrDefault(*it);
        assembledResults.push_back(scalar);
        assembledTypes.push_back(scalar.getType());
      }
    }
  }

  // secret.yield carries the per-kernel scalar leaves.
  OperationState yieldState(loc, "secret.yield");
  yieldState.addOperands(assembledResults);
  builder.create(yieldState);

  // Now create the secret.generic with its proper result types and attach
  // the populated body block.
  OpBuilder outerBuilder(ctx);
  outerBuilder.setInsertionPointToEnd(mergedBody);
  llvm::SmallVector<Type> secretResultTypes;
  for (Type t : assembledTypes)
    secretResultTypes.push_back(secret::SecretType::get(t));
  OperationState genericState(loc, "secret.generic");
  genericState.addOperands(mergedGenericOperands);
  genericState.addTypes(secretResultTypes);
  genericState.addRegion()->push_back(mergedGenericBlock);
  Operation *mergedGeneric = outerBuilder.create(genericState);

  // Update the merged function's signature to return all assembled results.
  llvm::SmallVector<Type> funcInputs(wideInputs.begin(), wideInputs.end());
  mergedFunc.setType(FunctionType::get(ctx, funcInputs, secretResultTypes));

  // Build func.return with all generic results.
  outerBuilder.setInsertionPointToEnd(mergedBody);
  llvm::SmallVector<Value> returnOperands(mergedGeneric->getResults().begin(),
                                          mergedGeneric->getResults().end());
  func::ReturnOp::create(outerBuilder, loc, returnOperands);

  return success();
}

//===----------------------------------------------------------------------===//
// mergeInsertChains — combine multiple insert chains into one wider chain
//===----------------------------------------------------------------------===//

Value mergeInsertChains(llvm::ArrayRef<Value> chainEnds, OpBuilder &builder) {
  if (chainEnds.empty()) return nullptr;

  // --- Phase 1: Walk each chain backward to collect entries ---
  struct InsertEntry {
    Value scalar;                  // the extracted value being inserted
    SmallVector<int64_t> indices;  // constant index values per dimension
  };

  SmallVector<SmallVector<InsertEntry>> allChains;
  SmallVector<int64_t> chainWidths;

  for (Value chainEnd : chainEnds) {
    auto insertOp = chainEnd.getDefiningOp<tensor::InsertOp>();
    if (!insertOp) {
      llvm::errs() << "mergeInsertChains: chainEnd is not a tensor.insert\n";
      return nullptr;
    }

    // Get width from the result tensor type's last dimension
    auto tensorType = cast<RankedTensorType>(insertOp.getResult().getType());
    chainWidths.push_back(tensorType.getShape().back());

    // Walk backward through the chain
    SmallVector<InsertEntry> entries;
    Operation *current = insertOp;
    while (auto curInsert = dyn_cast_or_null<tensor::InsertOp>(current)) {
      InsertEntry entry;
      entry.scalar = curInsert.getScalar();

      // Extract constant index values
      for (Value idx : curInsert.getIndices()) {
        if (auto constIdx = idx.getDefiningOp<arith::ConstantIndexOp>()) {
          entry.indices.push_back(constIdx.value());
        } else {
          llvm::errs() << "mergeInsertChains: non-constant index\n";
          return nullptr;
        }
      }

      entries.push_back(entry);

      // Move to predecessor: the "dest" operand
      Value dest = curInsert.getDest();
      current =
          dest.getDefiningOp();  // nullptr or arith.constant = end of chain
    }

    std::reverse(entries.begin(), entries.end());
    allChains.push_back(std::move(entries));
  }

  // --- Phase 2: Compute offsets and create merged zero tensor ---
  SmallVector<int64_t> offsets;
  int64_t totalWidth = 0;
  for (int64_t w : chainWidths) {
    offsets.push_back(totalWidth);
    totalWidth += w;
  }

  // Get reference shape from first chain, replace last dim with totalWidth
  auto refInsert = chainEnds[0].getDefiningOp<tensor::InsertOp>();
  auto refType = cast<RankedTensorType>(refInsert.getResult().getType());
  SmallVector<int64_t> mergedShape(refType.getShape());
  mergedShape.back() = totalWidth;
  auto mergedTensorType =
      RankedTensorType::get(mergedShape, refType.getElementType());

  Location loc = refInsert.getLoc();
  auto zeroAttr = DenseElementsAttr::get(
      mergedTensorType, builder.getIntegerAttr(refType.getElementType(), 0));
  Value currentDest = arith::ConstantOp::create(builder, loc, zeroAttr);

  // --- Phase 3: Build merged insert chain ---
  for (unsigned i = 0; i < allChains.size(); ++i) {
    int64_t offset = offsets[i];
    for (auto &entry : allChains[i]) {
      // Build index values, offsetting the last dimension
      SmallVector<Value> newIndices;
      for (unsigned d = 0; d < entry.indices.size(); ++d) {
        int64_t idx = entry.indices[d];
        if (d == entry.indices.size() - 1)
          idx += offset;  // offset the last dimension
        newIndices.push_back(arith::ConstantIndexOp::create(builder, loc, idx));
      }

      currentDest = tensor::InsertOp::create(builder, loc, entry.scalar,
                                             currentDest, newIndices);
    }
  }

  return currentDest;
}

//===----------------------------------------------------------------------===//
// widenFunctionArgAndPropagate — retype a single arg and propagate forward
//===----------------------------------------------------------------------===//

void widenFunctionArgAndPropagate(func::FuncOp func, unsigned argIdx,
                                  Type newType) {
  Block &entry = func.front();
  if (argIdx >= entry.getNumArguments()) return;

  // Set the arg type and seed the worklist.
  BlockArgument arg = entry.getArgument(argIdx);
  if (arg.getType() == newType) return;
  arg.setType(newType);

  SmallVector<Value> worklist;
  worklist.push_back(arg);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();

    for (OpOperand &use : v.getUses()) {
      Operation *op = use.getOwner();
      llvm::StringRef name = op->getName().getStringRef();

      // --- secret.generic: retype the matching body block arg ---
      if (name == "secret.generic") {
        unsigned idx = use.getOperandNumber();
        if (op->getNumRegions() == 0 || op->getRegion(0).empty()) continue;
        Block &body = op->getRegion(0).front();
        if (idx >= body.getNumArguments()) continue;

        BlockArgument blockArg = body.getArgument(idx);
        // The body block arg has the unwrapped type of the operand.
        Type innerType = v.getType();
        if (auto secretTy = dyn_cast<secret::SecretType>(innerType))
          innerType = secretTy.getValueType();
        if (blockArg.getType() != innerType) {
          blockArg.setType(innerType);
          worklist.push_back(blockArg);
        }
        continue;
      }

      // --- secret.yield: retype the parent generic's matching result ---
      if (name == "secret.yield") {
        Operation *parent = op->getParentOp();
        if (!parent || parent->getName().getStringRef() != "secret.generic")
          continue;
        unsigned idx = use.getOperandNumber();
        if (idx >= parent->getNumResults()) continue;

        Value parentResult = parent->getResult(idx);
        Type newResType = v.getType();
        if (isa<secret::SecretType>(parentResult.getType()))
          newResType = secret::SecretType::get(v.getType());
        if (parentResult.getType() != newResType) {
          parentResult.setType(newResType);
          worklist.push_back(parentResult);
        }
        continue;
      }

      // --- func.return: handled at the end via signature update ---
      if (name == "func.return") continue;

      // --- Element-wise op: recompute result types from first tensor operand
      // ---
      RankedTensorType srcType;
      for (Value operand : op->getOperands()) {
        if (auto rt = dyn_cast<RankedTensorType>(operand.getType())) {
          srcType = rt;
          break;
        }
      }
      if (!srcType) continue;

      for (Value res : op->getResults()) {
        auto resType = dyn_cast<RankedTensorType>(res.getType());
        if (!resType) continue;
        Type newResType =
            RankedTensorType::get(srcType.getShape(), resType.getElementType());
        if (res.getType() != newResType) {
          res.setType(newResType);
          worklist.push_back(res);
        }
      }
    }
  }

  // Update the function signature: inputs from entry block args, results
  // from func.return operand types (which got updated via propagation).
  SmallVector<Type> newInputs;
  for (BlockArgument a : entry.getArguments()) newInputs.push_back(a.getType());

  SmallVector<Type> newResults(func.getFunctionType().getResults().begin(),
                               func.getFunctionType().getResults().end());
  func.walk([&](func::ReturnOp ret) {
    newResults.clear();
    for (Value v : ret.getOperands()) newResults.push_back(v.getType());
  });

  func.setType(FunctionType::get(func.getContext(), newInputs, newResults));
}

//===----------------------------------------------------------------------===//
// prettyPrintSchedule — debug-friendly dump of a Schedule by (cycle, lane)
//===----------------------------------------------------------------------===//

void prettyPrintSchedule(const Schedule &schedule, llvm::raw_ostream &os) {
  if (schedule.instructions.empty()) {
    os << "=== Schedule (empty) ===\n";
    return;
  }

  // Find max cycle.
  int64_t maxCycle = 0;
  for (Operation *op : schedule.instructions) {
    auto it = schedule.alignment.find(op);
    if (it != schedule.alignment.end() && it->second > maxCycle)
      maxCycle = it->second;
  }

  os << "=== Schedule (warpSize=" << schedule.warpSize
     << ", ops=" << schedule.instructions.size() << ", depth=" << (maxCycle + 1)
     << ") ===\n";

  // Group ops by cycle.
  std::map<int64_t, llvm::SmallVector<Operation *>> opsByCycle;
  llvm::SmallVector<Operation *> unscheduled;
  for (Operation *op : schedule.instructions) {
    auto it = schedule.alignment.find(op);
    if (it == schedule.alignment.end()) {
      unscheduled.push_back(op);
      continue;
    }
    opsByCycle[it->second].push_back(op);
  }

  // For each cycle, sort by lane and print.
  for (auto &[cycle, ops] : opsByCycle) {
    llvm::sort(ops, [&](Operation *a, Operation *b) {
      int64_t la = schedule.lanes.lookup(a);
      int64_t lb = schedule.lanes.lookup(b);
      if (la != lb) return la < lb;
      return a < b;
    });

    os << "\nCycle " << cycle << " (" << ops.size() << " ops):\n";
    for (Operation *op : ops) {
      int64_t lane = schedule.lanes.lookup(op);
      // Right-align lane in a 4-char field for readability.
      os << "  lane ";
      if (lane < 10)
        os << "  ";
      else if (lane < 100)
        os << " ";
      os << lane << ": ";
      op->print(os);
      os << "\n";
    }
  }

  // List any ops that weren't in the alignment map (structural ops like
  // tensor.insert / arith.constant that aren't scheduled).
  if (!unscheduled.empty()) {
    os << "\nUnscheduled (" << unscheduled.size() << " ops, structural):\n";
    for (Operation *op : unscheduled) {
      os << "  ";
      op->print(os);
      os << "\n";
    }
  }
  os << "\n";
}

void findScheduleMergingCandidates(
    recursiveProgramNode *node,
    DenseMap<recursiveProgramNode *, SmallVector<recursiveProgramNode *>>
        &candidates,
    DenseSet<func::CallOp> &visited) {
  if (!node) return;

  for (recursiveProgramNode *child : node->children)
    findScheduleMergingCandidates(child, candidates, visited);

  if (node->children.empty() && !visited.count(node->caller))
    candidates[node->parent].push_back(node);
  visited.insert(node->caller);
}

static void collectTensorInsertChain(Value inputTensor,
                                     SmallVector<Value> &insertChain) {
  if (auto defOp = inputTensor.getDefiningOp()) {
    if (auto genericOp = dyn_cast<secret::GenericOp>(defOp)) {
      // value is a result of secret.generic
      auto resultIdx = cast<OpResult>(inputTensor).getResultNumber();
      // Find what was yielded at that index
      auto yieldOp =
          cast<secret::YieldOp>(genericOp.getBody()->getTerminator());
      Value yieldedVal = yieldOp->getOperand(resultIdx);
      collectTensorInsertChain(yieldedVal, insertChain);
    } else if (auto constantOp = dyn_cast<arith::ConstantOp>(defOp)) {
      insertChain.push_back(constantOp.getResult());
      return;
    } else {
      if (auto insOp = dyn_cast<tensor::InsertOp>(defOp)) {
        insertChain.push_back(insOp.getResult());
        collectTensorInsertChain(insOp.getDest(), insertChain);
      }
    }
  } else if (auto blockArg = dyn_cast<BlockArgument>(inputTensor)) {
    llvm::errs() << "Reached block argument during insert chain expansion. "
                    "Cannot expand further.\n";
    assert(1);
  }
}

static RankedTensorType extractUnderlyingTensor(Value in) {
  return mlir::cast<RankedTensorType>(
      mlir::cast<secret::SecretType>(in.getType()).getValueType());
}

SmallVector<cipherTextSlot> createMergedCipherTextMappings(
    RankedTensorType mergedType, SmallVector<Value> subArgs,
    OpBuilder builder) {
  if (!mergedType.hasStaticShape()) return {};

  llvm::outs() << "Rank: " << mergedType.getRank() << "\n";
  // do a sanity check that summation of dim-1 of all subArgs = dim-1 of
  // mergedType
  int mergedDim1 = mergedType.getDimSize(1);
  for (auto subarg : subArgs) {
    auto tensorType = extractUnderlyingTensor(subarg);
    mergedDim1 -= tensorType.getDimSize(1);
  }

  if (mergedDim1 != 0)
    assert(false && "merged dim size doesn't match up with the subargs");

  SmallVector<cipherTextSlot> ctxt;
  for (int i = 0; i < mergedType.getDimSize(1); i++) {
    ctxt.push_back({nullptr, i, 0});
  }

  DenseMap<Value, SmallVector<Value>> insertChains;
  for (auto subarg : subArgs) {
    if (insertChains.contains(subarg)) continue;

    SmallVector<Value> chain;
    collectTensorInsertChain(subarg, chain);
    if (isa<arith::ConstantOp>(chain.back().getDefiningOp())) chain.pop_back();
    std::reverse(chain.begin(), chain.end());
    insertChains[subarg] = chain;
  }

  for (auto a : insertChains) {
    for (auto c : a.second) c.dump();

    llvm::outs() << "\n";
  }

  int offset = 0;
  for (auto subarg : subArgs) {
    for (auto insertVal : insertChains[subarg]) {
      auto insertOp = cast<tensor::InsertOp>(insertVal.getDefiningOp());
      int index = offset + cast<arith::ConstantIndexOp>(
                               insertOp.getIndices()[1].getDefiningOp())
                               .value();
      ctxt[index].op = insertOp;
      ctxt[index].parentDim = offset;
    }
    offset += extractUnderlyingTensor(subarg).getDimSize(1);
  }

  return ctxt;
}

// auto zero = builder.getZeroAttr(mergedType.getElementType());
// auto attr = DenseElementsAttr::get(mergedType, zero);
// auto prevOp = arith::ConstantOp::create(builder, builder.getUnknownLoc(),
// attr); prevOp.dump(); auto newInsertOp = tensor::InsertOp::create(builder,
// insertOp.getLoc(), ins.getScalar(), prevOp.getResult(), ins.getIndices());
// newInsertOp.dump();

Value createNewInsertOpsFromSeedOps(SmallVector<cipherTextSlot> &ctxt,
                                    RankedTensorType mergedType,
                                    OpBuilder builder) {
  auto zero = builder.getZeroAttr(mergedType.getElementType());
  auto attr = DenseElementsAttr::get(mergedType, zero);
  Operation *seedOp =
      arith::ConstantOp::create(builder, builder.getUnknownLoc(), attr);

  llvm::outs() << "Creating new insert ops from seed op:\n";
  seedOp->dump();
  for (int i = 0; i < ctxt.size(); i++) {
    auto &slot = ctxt[i];
    if (slot.op) {
      auto insertOp = cast<tensor::InsertOp>(slot.op);
      builder.setInsertionPoint(insertOp->getBlock()->getTerminator());
      SmallVector<Value> newIndices(insertOp.getIndices().begin(),
                                    insertOp.getIndices().end());
      newIndices.back() = arith::ConstantIndexOp::create(
          builder, insertOp.getLoc(), slot.index);

      auto newInsertOp = tensor::InsertOp::create(
          builder, insertOp.getLoc(), insertOp.getScalar(),
          seedOp->getResult(0), newIndices);
      slot.op = newInsertOp;
      seedOp = slot.op;
      // newInsertOp.dump();
    }
  }

  return seedOp->getResult(0);
}

}  // namespace heir
}  // namespace mlir
