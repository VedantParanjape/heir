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
// Step 3: Build merged function
//===----------------------------------------------------------------------===//

/// Sum last dimension of two tensor types.
/// e.g. tensor<1x3xi32> + tensor<1x3xi32> -> tensor<1x6xi32>
static Type sumLastDim(Type typeA, Type typeB) {
  auto rankedA = dyn_cast<RankedTensorType>(typeA);
  auto rankedB = dyn_cast<RankedTensorType>(typeB);
  if (!rankedA || !rankedB) return typeA;

  SmallVector<int64_t> newShape(rankedA.getShape());
  newShape.back() = rankedA.getShape().back() + rankedB.getShape().back();
  return RankedTensorType::get(newShape, rankedA.getElementType());
}

LogicalResult mergeWithNeedlemanWunsch(func::FuncOp funcA, func::FuncOp funcB,
                                       func::FuncOp &result,
                                       const NWScoreConfig &config) {
  // --- Validate structural compatibility ---
  // Functions must have the same number of args/results with compatible types:
  // same rank, same element type, same wrapping (secret or not). The last
  // tensor dimension can differ (it gets summed during merge).
  FunctionType ftA = funcA.getFunctionType();
  FunctionType ftB = funcB.getFunctionType();
  if (ftA.getNumInputs() != ftB.getNumInputs() ||
      ftA.getNumResults() != ftB.getNumResults()) {
    llvm::errs() << "NW Merge: function arg/result count mismatch\n";
    return failure();
  }
  for (unsigned i = 0; i < ftA.getNumInputs(); ++i) {
    Type tA = ftA.getInput(i), tB = ftB.getInput(i);
    bool secA = isa<secret::SecretType>(tA), secB = isa<secret::SecretType>(tB);
    if (secA != secB) {
      llvm::errs() << "NW Merge: secret wrapping mismatch at arg " << i << "\n";
      return failure();
    }
    Type innerA = secA ? cast<secret::SecretType>(tA).getValueType() : tA;
    Type innerB = secB ? cast<secret::SecretType>(tB).getValueType() : tB;
    auto rA = dyn_cast<RankedTensorType>(innerA);
    auto rB = dyn_cast<RankedTensorType>(innerB);
    if ((rA != nullptr) != (rB != nullptr)) {
      llvm::errs() << "NW Merge: tensor/scalar mismatch at arg " << i << "\n";
      return failure();
    }
    if (rA && rB) {
      if (rA.getRank() != rB.getRank() ||
          rA.getElementType() != rB.getElementType()) {
        llvm::errs() << "NW Merge: incompatible tensor structure at arg " << i
                     << "\n";
        return failure();
      }
    } else if (innerA != innerB) {
      llvm::errs() << "NW Merge: incompatible scalar types at arg " << i
                   << "\n";
      return failure();
    }
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

  // --- Step 3: Build merged function ---
  MLIRContext *ctx = funcA.getContext();
  Location loc = funcA.getLoc();
  OpBuilder builder(ctx);

  // --- Build merged func-level arg types ---
  // Secret-wrapped tensor args: widen (sum last dims), shared between A and B.
  // All other args (plain scalars, plain tensors): duplicate — A's copy then
  // B's.
  SmallVector<Type> mergedFuncArgTypes;
  // Track where each original arg ends up in the merged func
  SmallVector<unsigned> argMapA, argMapB;
  for (unsigned i = 0; i < ftA.getNumInputs(); ++i) {
    Type argTypeA = ftA.getInput(i);
    Type argTypeB = ftB.getInput(i);
    bool isSecret = isa<secret::SecretType>(argTypeA);
    Type innerA =
        isSecret ? cast<secret::SecretType>(argTypeA).getValueType() : argTypeA;
    Type innerB =
        isSecret ? cast<secret::SecretType>(argTypeB).getValueType() : argTypeB;

    if (isSecret && isa<RankedTensorType>(innerA)) {
      // Secret tensor: widen by summing last dims of A and B
      Type widened = sumLastDim(innerA, innerB);
      argMapA.push_back(mergedFuncArgTypes.size());
      argMapB.push_back(mergedFuncArgTypes.size());  // same index
      mergedFuncArgTypes.push_back(secret::SecretType::get(widened));
    } else {
      // Non-secret: duplicate (A's copy, then B's copy)
      argMapA.push_back(mergedFuncArgTypes.size());
      mergedFuncArgTypes.push_back(argTypeA);
      argMapB.push_back(mergedFuncArgTypes.size());
      mergedFuncArgTypes.push_back(argTypeB);
    }
  }

  // --- Build merged inner arg types (secret.generic block arg types) ---
  SmallVector<Type> mergedInnerArgTypes;
  for (unsigned i = 0; i < bodyA->getNumArguments(); ++i) {
    Type innerA = bodyA->getArgument(i).getType();
    Type innerB = bodyB->getArgument(i).getType();
    mergedInnerArgTypes.push_back(sumLastDim(innerA, innerB));
  }

  // Merged inner result types: sum last dimension
  Operation *yieldA = bodyA->getTerminator();
  Operation *yieldB = bodyB->getTerminator();
  SmallVector<Type> mergedInnerRetTypes;
  for (unsigned i = 0; i < yieldA->getNumOperands(); ++i) {
    Type innerA = yieldA->getOperand(i).getType();
    Type innerB = yieldB->getOperand(i).getType();
    mergedInnerRetTypes.push_back(sumLastDim(innerA, innerB));
  }

  // Outer result types: wrap merged inner ret types in secret if needed
  bool isSecretWrapped =
      (genericA->getNumResults() > 0 &&
       genericA->getResult(0).getType() != yieldA->getOperand(0).getType());
  SmallVector<Type> outerRetTypes;
  for (Type innerRet : mergedInnerRetTypes) {
    if (isSecretWrapped)
      outerRetTypes.push_back(secret::SecretType::get(innerRet));
    else
      outerRetTypes.push_back(innerRet);
  }

  // Create the merged function
  FunctionType mergedFuncType =
      FunctionType::get(ctx, mergedFuncArgTypes, outerRetTypes);

  builder.setInsertionPoint(funcA);
  std::string mergedName =
      (funcA.getName() + "_nw_merged_" + funcB.getName()).str();
  auto mergedFunc =
      func::FuncOp::create(builder, loc, mergedName, mergedFuncType);

  // Copy attributes, but skip arg_attrs since the arg count/types changed
  for (auto attr : funcA->getAttrs()) {
    if (attr.getName() == "sym_name" || attr.getName() == "function_type" ||
        attr.getName() == "arg_attrs" || attr.getName() == "res_attrs")
      continue;
    mergedFunc->setAttr(attr.getName(), attr.getValue());
  }

  // Create the function body
  Block *funcBody = mergedFunc.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  // Build secret.generic carrying over ALL operands from the original.
  // The original secret.generic may have extra operands beyond the func args
  // (e.g., rotation constants, index values). We must preserve them.
  //
  // Strategy: start with the merged func args (widened tensors), then append
  // any extra operands from genericA that aren't func args. For extra operands,
  // we recreate them in the merged func body before passing to the generic.

  // Map original func args to merged func args using the arg maps.
  // Secret args share the same merged arg; non-secret args get separate copies.
  IRMapping outerRemapA, outerRemapB;
  Block &funcEntryA = funcA.front();
  Block &funcEntryB = funcB.front();
  for (unsigned i = 0; i < funcEntryA.getNumArguments(); ++i)
    outerRemapA.map(funcEntryA.getArgument(i),
                    funcBody->getArgument(argMapA[i]));
  for (unsigned i = 0; i < funcEntryB.getNumArguments(); ++i)
    outerRemapB.map(funcEntryB.getArgument(i),
                    funcBody->getArgument(argMapB[i]));

  // Clone any non-func-arg operands from genericA into the merged func body
  // (e.g., constants, index_cast results) so they can be passed to the generic
  for (unsigned i = 0; i < genericA->getNumOperands(); ++i) {
    Value origOperand = genericA->getOperand(i);
    if (!outerRemapA.contains(origOperand)) {
      Operation *defOp = origOperand.getDefiningOp();
      if (defOp) {
        Operation *cloned = builder.clone(*defOp);
        outerRemapA.map(origOperand, cloned->getResult(0));
      }
    }
  }
  for (unsigned i = 0; i < genericB->getNumOperands(); ++i) {
    Value origOperand = genericB->getOperand(i);
    if (!outerRemapB.contains(origOperand)) {
      Operation *defOp = origOperand.getDefiningOp();
      if (defOp) {
        Operation *cloned = builder.clone(*defOp);
        outerRemapB.map(origOperand, cloned->getResult(0));
      }
    }
  }

  // Build the generic operands list and inner block args.
  // We take all of genericA's operands (remapped), then append any extra
  // operands from genericB that aren't already covered.
  SmallVector<Value> genericOperands;
  auto *genericBlock = new Block();

  // Add genericA's operands (with widened types for tensor args)
  for (unsigned i = 0; i < genericA->getNumOperands(); ++i) {
    Value outerVal = outerRemapA.lookupOrDefault(genericA->getOperand(i));
    genericOperands.push_back(outerVal);

    // Inner block arg type: widen if it's a tensor and has a counterpart in B
    Type innerTypeA = bodyA->getArgument(i).getType();
    if (i < bodyB->getNumArguments()) {
      Type innerTypeB = bodyB->getArgument(i).getType();
      genericBlock->addArgument(sumLastDim(innerTypeA, innerTypeB), loc);
    } else {
      genericBlock->addArgument(innerTypeA, loc);
    }
  }

  // Track how many args we took from A
  unsigned numArgsFromA = genericA->getNumOperands();

  // Add any extra operands from genericB beyond what A has
  for (unsigned i = numArgsFromA; i < genericB->getNumOperands(); ++i) {
    Value outerVal = outerRemapB.lookupOrDefault(genericB->getOperand(i));
    genericOperands.push_back(outerVal);
    genericBlock->addArgument(bodyB->getArgument(i).getType(), loc);
  }

  OperationState genericState(loc, "secret.generic");
  genericState.addOperands(genericOperands);
  genericState.addTypes(outerRetTypes);
  genericState.addRegion()->push_back(genericBlock);
  Operation *mergedGeneric = builder.create(genericState);

  // Populate the generic body
  builder.setInsertionPointToStart(genericBlock);

  // --- SSA remapping ---
  // Map all of A's and B's inner block args to the merged block args.
  IRMapping remapA, remapB;

  for (unsigned i = 0; i < bodyA->getNumArguments(); ++i)
    remapA.map(bodyA->getArgument(i), genericBlock->getArgument(i));
  for (unsigned i = 0; i < bodyB->getNumArguments(); ++i) {
    if (i < numArgsFromA)
      remapB.map(bodyB->getArgument(i), genericBlock->getArgument(i));
    else
      remapB.map(bodyB->getArgument(i),
                 genericBlock->getArgument(numArgsFromA + (i - numArgsFromA)));
  }

  // Clone outer-scope values used inside the body (e.g., rotation constants
  // like %c-4_i32 that are defined in the func body but used inside
  // secret.generic without being passed as explicit operands).
  // We clone them into the merged func body (before the generic) and add
  // mappings so the inner body clones can find them.
  auto cloneOuterDeps = [&](Block *origBody, IRMapping &remap,
                            IRMapping &outerRemap) {
    OpBuilder outerBuilder(ctx);
    outerBuilder.setInsertionPoint(mergedGeneric);
    for (Operation &op : *origBody) {
      for (Value operand : op.getOperands()) {
        if (remap.contains(operand)) continue;
        // If it's a func-level block arg used via implicit capture,
        // map it through the outer remap (which knows about arg duplication)
        if (isa<BlockArgument>(operand)) {
          if (outerRemap.contains(operand))
            remap.map(operand, outerRemap.lookup(operand));
          continue;
        }
        // Check if this operand is defined outside the body
        Operation *defOp = operand.getDefiningOp();
        if (!defOp) continue;
        if (defOp->getBlock() == origBody)
          continue;  // defined inside body, will be cloned later
        // It's an outer-scope value — clone its defining op
        Operation *cloned = outerBuilder.clone(*defOp, outerRemap);
        remap.map(operand, cloned->getResult(0));
      }
    }
  };

  cloneOuterDeps(bodyA, remapA, outerRemapA);
  cloneOuterDeps(bodyB, remapB, outerRemapB);

  // Helper: widen all tensor result types of a cloned op to match the merged
  // tensor size. Since the operands are already widened (via remapping), the
  // result types must be widened to match.
  auto widenResultTypes = [](Operation *cloned, Operation *origA,
                             Operation *origB) {
    for (unsigned k = 0; k < cloned->getNumResults(); ++k) {
      auto origTypeA =
          dyn_cast<RankedTensorType>(origA->getResult(k).getType());
      auto origTypeB =
          dyn_cast<RankedTensorType>(origB->getResult(k).getType());
      if (origTypeA && origTypeB) {
        cloned->getResult(k).setType(sumLastDim(origTypeA, origTypeB));
      }
    }
  };

  auto widenResultTypesSingle = [](Operation *cloned, Operation *orig) {
    // For gap ops, the operands are widened but the clone has the original
    // result types. Update result types to match operand types.
    for (unsigned k = 0; k < cloned->getNumResults(); ++k) {
      // Infer result type from first operand's type (element-wise ops
      // produce the same shape as their operands)
      if (cloned->getNumOperands() > 0) {
        auto opType =
            dyn_cast<RankedTensorType>(cloned->getOperand(0).getType());
        auto resType =
            dyn_cast<RankedTensorType>(cloned->getResult(k).getType());
        if (opType && resType) {
          cloned->getResult(k).setType(RankedTensorType::get(
              opType.getShape(), resType.getElementType()));
        }
      }
    }
  };

  // --- Walk alignment and clone operations ---
  for (const auto &entry : alignment) {
    switch (entry.kind) {
      case AlignmentEntry::Match: {
        // Clone opA with remapped (widened) operands.
        // The cloned op operates on the wider tensor — first half is A's data,
        // second half is B's data. Both computed in one instruction.
        Operation *cloned = builder.clone(*entry.opA, remapA);
        widenResultTypes(cloned, entry.opA, entry.opB);
        // B's downstream uses point to the same widened result
        for (unsigned k = 0; k < entry.opB->getNumResults(); ++k)
          remapB.map(entry.opB->getResult(k), cloned->getResult(k));
        break;
      }
      case AlignmentEntry::GapA: {
        // Only opB — clone with widened operands, widen result types
        Operation *cloned = builder.clone(*entry.opB, remapB);
        widenResultTypesSingle(cloned, entry.opB);
        break;
      }
      case AlignmentEntry::GapB: {
        // Only opA — clone with widened operands, widen result types
        Operation *cloned = builder.clone(*entry.opA, remapA);
        widenResultTypesSingle(cloned, entry.opA);
        break;
      }
    }
  }

  // --- Build secret.yield ---
  // The yield operands are already widened tensors flowing through the
  // computation. Just remap the original yield values.
  SmallVector<Value> yieldOperands;
  for (unsigned i = 0; i < yieldA->getNumOperands(); ++i)
    yieldOperands.push_back(remapA.lookupOrDefault(yieldA->getOperand(i)));

  OperationState yieldState(loc, "secret.yield");
  yieldState.addOperands(yieldOperands);
  builder.create(yieldState);

  // --- Build func.return ---
  builder.setInsertionPointAfter(mergedGeneric);
  SmallVector<Value> returnOperands;
  for (unsigned i = 0; i < mergedGeneric->getNumResults(); ++i)
    returnOperands.push_back(mergedGeneric->getResult(i));
  func::ReturnOp::create(builder, loc, returnOperands);

  result = mergedFunc;
  return success();
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

SmallVector<cipherTextSlot> processTensorOpsAfterMerging(
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
    ctxt.push_back({nullptr, i});
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

Value mergeTensorOps(SmallVector<cipherTextSlot> &ctxt,
                     RankedTensorType mergedType, OpBuilder builder) {
  auto zero = builder.getZeroAttr(mergedType.getElementType());
  auto attr = DenseElementsAttr::get(mergedType, zero);
  Operation *seedOp =
      arith::ConstantOp::create(builder, builder.getUnknownLoc(), attr);

  for (auto &slot : ctxt) {
    if (slot.op) {
      auto insertOp = cast<tensor::InsertOp>(slot.op);
      builder.setInsertionPoint(insertOp->getBlock()->getTerminator());
      auto newInsertOp = tensor::InsertOp::create(
          builder, insertOp.getLoc(), insertOp.getScalar(),
          seedOp->getResult(0), insertOp.getIndices());
      slot.op = newInsertOp;
      seedOp = slot.op;

      newInsertOp.dump();
    }
  }

  return seedOp->getResult(0);
}

}  // namespace heir
}  // namespace mlir
