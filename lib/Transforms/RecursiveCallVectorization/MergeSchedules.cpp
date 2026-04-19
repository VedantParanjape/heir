//===- NeedlemanWunschMerge.cpp - NW-based MLIR function merge --*- C++ -*-===//
//
// Implementation of Needleman-Wunsch sequence alignment for merging two
// MLIR func::FuncOp with identical signatures.
//
//===----------------------------------------------------------------------===//

#include "lib/Transforms/RecursiveCallVectorization/MergeSchedules.h"

#include <algorithm>
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

LogicalResult mergeSchedulesWithNW(llvm::ArrayRef<func::FuncOp> funcs,
                                   llvm::ArrayRef<Schedule> schedules,
                                   func::FuncOp &mergedFunc,
                                   Schedule &mergedSchedule,
                                   const NWScoreConfig &config) {
  using Origin = std::pair<unsigned, Operation *>;
  using OriginList = llvm::SmallVector<Origin, 8>;

  if (funcs.size() < 2 || funcs.size() != schedules.size()) {
    llvm::errs() << "NW Merge: need at least 2 funcs with matching schedules\n";
    return failure();
  }
  unsigned N = funcs.size();

  // ArrayRef gives const elements; copy out to call non-const methods on
  // FuncOp.
  func::FuncOp f0 = funcs[0];
  FunctionType ft0 = f0.getFunctionType();
  for (unsigned i = 1; i < N; ++i) {
    func::FuncOp fi = funcs[i];
    if (fi.getFunctionType() != ft0) {
      llvm::errs() << "NW Merge: function types do not match at idx " << i
                   << "\n";
      return failure();
    }
  }

  // Initialize: running side starts as funcs[0]; each of its ops has origin
  // [(0, itself)].
  llvm::SmallVector<Operation *> runningSeq(schedules[0].instructions.begin(),
                                            schedules[0].instructions.end());
  if (runningSeq.empty()) runningSeq = extractSortedOps(f0);
  if (runningSeq.empty()) return failure();

  llvm::DenseMap<Operation *, OriginList> runningOrigins;
  for (Operation *op : runningSeq) runningOrigins[op] = {{0u, op}};

  func::FuncOp runningFunc = f0;

  // Pairwise reduce with funcs[1..N-1].
  for (unsigned k = 1; k < N; ++k) {
    func::FuncOp fk = funcs[k];
    llvm::SmallVector<Operation *> seqK(schedules[k].instructions.begin(),
                                        schedules[k].instructions.end());
    if (seqK.empty()) seqK = extractSortedOps(fk);

    func::FuncOp newMerged;
    llvm::SmallVector<Operation *> newSeq;
    llvm::DenseMap<Operation *, OriginList> newOrigins;

    if (failed(pairwiseScheduleMergeStep(runningFunc, runningOrigins,
                                         runningSeq, fk, seqK, k, config,
                                         newMerged, newSeq, newOrigins))) {
      return failure();
    }

    runningFunc = newMerged;
    runningSeq = std::move(newSeq);
    runningOrigins = std::move(newOrigins);
  }

  mergedFunc = runningFunc;

  // --- Final mod-N interleave using all original schedules ---
  // For each cloned op in the final merged schedule:
  //   - If it represents kernels {k_0, k_1, ...}: matched op
  //     merged_lane = schedules[k_0].lanes[orig_in_k_0] * N + k_0
  //     (kernel 0 chosen as canonical; lowering aligns other stripes via
  //     rotations) merged_cycle = max over all kernels' cycles
  //   - If it represents single kernel {k}: gap op for that kernel
  //     merged_lane = schedules[k].lanes[orig] * N + k
  //     merged_cycle = schedules[k].alignment[orig]
  unsigned maxOrigWarp = 0;
  for (const Schedule &s : schedules)
    maxOrigWarp = std::max(maxOrigWarp, s.warpSize);

  mergedSchedule.lanes.clear();
  mergedSchedule.alignment.clear();
  mergedSchedule.instructions.clear();
  mergedSchedule.warpSize = maxOrigWarp * N;
  mergedSchedule.instructions = runningSeq;

  for (Operation *op : runningSeq) {
    auto it = runningOrigins.find(op);
    if (it == runningOrigins.end()) continue;
    const OriginList &ol = it->second;
    if (ol.empty()) continue;

    // Canonical: first origin in the list (kernel 0 if matched).
    unsigned canonK = ol[0].first;
    Operation *canonOrig = ol[0].second;
    int64_t canonLane = schedules[canonK].lanes.lookup(canonOrig);
    mergedSchedule.lanes[op] = canonLane * N + canonK;

    // Cycle: max across all kernels the op represents.
    int64_t maxCycle = 0;
    for (const auto &[k, origOp] : ol) {
      int64_t c = schedules[k].alignment.lookup(origOp);
      if (c > maxCycle) maxCycle = c;
    }
    mergedSchedule.alignment[op] = maxCycle;
  }

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
