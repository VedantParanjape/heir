#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/Utils.h"
#include "lib/Transforms/RecursiveCallVectorization/CoyoteCaller.h"
#include "lib/Transforms/RecursiveCallVectorization/MergeSchedules.h"
#include "lib/Transforms/RecursiveCallVectorization/RecursiveProgramInfo.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/CallGraph.h"      // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"         // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Inliner.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/RegionUtils.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "recursive-call-vectorization"
#define NODE_SIZE_THRESHOLD 100

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RECURSIVECALLVECTORIZATION
#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h.inc"

DenseSet<func::FuncOp> functionDeleteList;
std::map<std::set<std::pair<int, int>>, func::FuncOp> functionCache;

static func::FuncOp getEnclosingFunction(Operation *op, ModuleOp &module) {
  auto callOp = dyn_cast<func::CallOp>(op);
  if (!callOp) {
    llvm::errs() << "Error: Operation is not a func::CallOp\n";
    return nullptr;
  }

  SymbolTable symTab(module);
  auto callee = callOp.getCallee();
  auto funcOp = symTab.lookup<func::FuncOp>(callee);

  return funcOp;
}

static bool findBiscottiAttribute(Value op, StringRef attrName, int &outValue) {
  FailureOr<Attribute> attr = findAttributeAssociatedWith(op, attrName);

  if (succeeded(attr)) {
    // llvm::outs() << "Operand: " << op << "\n"
    //              << " Found attribute: " << attrName << ": " << attr.value()
    //              << "\n";

    if (auto intAttr = dyn_cast<IntegerAttr>(attr.value()))
      outValue = intAttr.getInt();

    return true;
  }

  outValue = -1;
  return false;
}

static bool findBiscottiAttributeOnOps(Operation *op, StringRef attrName,
                                       int &outValue) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName)) {
    int outValue = attr.getInt();
    // llvm::outs() << "Operation: " << *op << "\n"
    //              << " Found attribute on Ops: " << attrName << ": " <<
    //              outValue
    //              << "\n";

    return true;
  }

  outValue = -1;
  return false;
}

static bool findBiscottiArrayAttribute(Value op, StringRef attrName,
                                       SmallVector<int64_t> &outValue) {
  FailureOr<Attribute> attr = findAttributeAssociatedWith(op, attrName);
  if (succeeded(attr)) {
    // llvm::outs() << "Operand: " << op << "\n"
    //              << " Found Array in attributes: " << attrName << ": "
    //              << attr.value() << "\n";
    if (auto intAttr = dyn_cast<DenseI64ArrayAttr>(attr.value())) {
      outValue = SmallVector<int64_t>(intAttr.asArrayRef());
      return true;
    }
  }
  return false;
}

static void printRecursiveAttributes(recursiveProgramInfo *rpi) {
  llvm::outs() << "Recursive Call Info for call: " << *(rpi->call) << "\n";

  // TODO: make sure the argument numbers match up with static values.
  llvm::outs() << " Progress Arguments:\n";
  for (auto pa : rpi->progressArguments) {
    llvm::outs() << "  Arg: " << *(pa.first) << " at index " << pa.second
                 << "\n";
  }

  llvm::outs() << " Static Argument Values:\n";
  for (auto sa : rpi->staticArgumentValues) {
    llvm::outs() << "  Value: " << sa.first << " at index " << sa.second
                 << "\n";
  }

  llvm::outs() << " Recursive Calls:\n";
  for (auto rc : rpi->recursiveCalls) {
    llvm::outs() << "  Call: " << *(rc.first) << " at index " << rc.second
                 << "\n";
  }

  llvm::outs() << " Base Conditions:\n";
  for (auto bc : rpi->baseConditions) {
    llvm::outs() << "  Op: " << *(bc.first) << " at index " << bc.second
                 << "\n";
  }
}

static void indent(unsigned level) {
  for (unsigned i = 0; i < level; ++i) llvm::outs() << "  ";
}

bool leaf = false;
// TODO: Rework this function (mainly clean it up)
static void prettyPrintRecursiveProgramTree(recursiveProgramNode *node,
                                            unsigned indentLevel = 0) {
  if (!node) return;

  indent(indentLevel);

  // Print function name
  if (node->function) {
    llvm::outs() << "func @" << node->function.getSymName();
  } else {
    llvm::outs() << "<null func>";
  }

  // Print static arguments
  if (!node->staticArgumentValues.empty()) {
    llvm::outs() << " [static args: ";
    bool first = true;
    for (auto &[op, _] : node->staticArgumentValues) {
      if (!first) llvm::outs() << ", ";
      first = false;

      if (op) {
        llvm::outs() << op;
      } else {
        llvm::outs() << "<unknown-op>";
      }
    }
    llvm::outs() << "]";
  }

  llvm::outs() << "\n";

  // Recurse into children
  for (recursiveProgramNode *child : node->children) {
    prettyPrintRecursiveProgramTree(child, indentLevel + 1);
  }

  // if (node->children.size() == 8 && !leaf) {
  //   leaf = true;
  //   indent(indentLevel + 1);
  //   llvm::outs() << "(leaf node)\n";
  //   node->function.dump();
  // }
}

static Operation *insertConstantAtTop(func::FuncOp &funcOp, TypedAttr attr) {
  Block &entryBlock = funcOp.front();
  OpBuilder builder(&entryBlock, entryBlock.begin());
  return arith::ConstantOp::create(builder, funcOp.getLoc(), attr);
}

class MergeTensorInsertChains final
    : public OpRewritePattern<secret::GenericOp> {
 public:
  SmallVector<Value> collectMergedTensorArgs;

  MergeTensorInsertChains(MLIRContext *context, SmallVector<Value> args)
      : OpRewritePattern<secret::GenericOp>(context),
        collectMergedTensorArgs(std::move(args)) {}

  LogicalResult matchAndRewrite(secret::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto [modifiedGeneric, newResults] = genericOp.addNewYieldedValues(
        ValueRange(collectMergedTensorArgs), rewriter);

    for (auto [oldRes, newRes] :
         llvm::zip(genericOp->getResults(), modifiedGeneric->getResults()))
      oldRes.replaceAllUsesWith(newRes);

    rewriter.eraseOp(genericOp);
    return success();
  }
};

struct ScalarizeAnyElementwise : public RewritePattern {
  ScalarizeAnyElementwise(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit*/ 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only match ops marked Elementwise that operate on tensors
    if (!OpTrait::hasElementwiseMappableTraits(op)) return failure();
    if (op->getNumResults() != 1) return failure();

    auto tensorTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorTy || !tensorTy.hasStaticShape()) return failure();

    // All operands must also be tensors of the same shape
    for (Value operand : op->getOperands()) {
      auto opTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!opTy || opTy.getShape() != tensorTy.getShape()) return failure();
    }

    Location loc = op->getLoc();
    ArrayRef<int64_t> shape = tensorTy.getShape();
    int64_t numElems = tensorTy.getNumElements();
    SmallVector<Value> scalars;
    scalars.reserve(numElems);

    for (int64_t linear = 0; linear < numElems; ++linear) {
      // linear → multi-dim indices (row-major)
      SmallVector<Value> idx;
      int64_t rem = linear;
      for (int d = shape.size() - 1; d >= 0; --d) {
        idx.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, rem % shape[d]));
        rem /= shape[d];
      }
      std::reverse(idx.begin(), idx.end());

      // Extract one scalar from each operand at this position
      SmallVector<Value> scalarOperands;
      for (Value operand : op->getOperands())
        scalarOperands.push_back(
            tensor::ExtractOp::create(rewriter, loc, operand, idx));

      // Clone the op with scalar types
      OperationState state(loc, op->getName());
      state.addOperands(scalarOperands);
      state.addTypes(tensorTy.getElementType());
      state.addAttributes(op->getAttrs());
      Operation *scalarOp = rewriter.create(state);
      scalars.push_back(scalarOp->getResult(0));
    }

    Value result =
        tensor::FromElementsOp::create(rewriter, loc, tensorTy, scalars);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct RecursiveCallVectorization
    : impl::RecursiveCallVectorizationBase<RecursiveCallVectorization> {
  using RecursiveCallVectorizationBase::RecursiveCallVectorizationBase;

  struct SecretInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation *call, Operation *callable,
                         bool wouldBeCloned) const final {
      return true;
    }

    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
      return true;
    }

    bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
      return true;
    }
  };

  void unrollTensorGenerates(Operation *op);
  void removeRedundantTensorCasts(Operation *Op);
  void removeDuplicateFunctions(recursiveProgramNode *node);
  void findMergeableRecursiveCallNodes(
      recursiveProgramNode *node,
      SmallVector<recursiveProgramNode *> &mergeableNodes);
  int countNodeFunctionSize(recursiveProgramNode *node);
  void mergeRecursiveCallNodes(
      SmallVector<recursiveProgramNode *> &mergeableNodes);
  void foldAllOpsInFunc(func::FuncOp &funcOp, MLIRContext *ctx);
  void buildRecursiveAttributes(Block *block, Dialect *dialect);
  void buildRecursiveCallTree(Operation *op,
                              recursiveProgramInfo &recursiveProgramInfo);
  void refreshRecursiveCallTree(Operation *op,
                                recursiveProgramInfo &recursiveProgramInfo);
  bool tryUnrollingRecursiveBlock(Block *block, Dialect *dialect);
  /// Carve a secret.generic into a fresh func::FuncOp.
  /// Returns the new func; the original generic is replaced with a func.call.
  func::FuncOp outlineSecretGeneric(
      secret::GenericOp genericOp,
      std::string funcName = "outlined_reduction_generic") {
    MLIRContext *ctx = genericOp.getContext();
    ModuleOp module = genericOp->getParentOfType<ModuleOp>();
    Location loc = genericOp.getLoc();
    static int uniqueId = 0;
    funcName = funcName + std::to_string(uniqueId++);

    // 1. Collect everything referenced from outside the generic op:
    //    - The generic's own explicit operands
    //    - Values used inside the body via implicit capture
    SetVector<Value> rawCaptures;
    for (Value operand : genericOp->getOperands()) rawCaptures.insert(operand);
    getUsedValuesDefinedAbove(genericOp->getRegions(), rawCaptures);

    // 2. Partition captures: clone constants inline, pass the rest as args.
    SetVector<Value> argCaptures;
    SmallVector<std::pair<Value, Operation *>> constantsToClone;
    for (Value v : rawCaptures) {
      Operation *defOp = v.getDefiningOp();
      if (defOp && defOp->hasTrait<OpTrait::ConstantLike>())
        constantsToClone.push_back({v, defOp});
      else
        argCaptures.insert(v);
    }

    // 3. Build the new function type from non-constant captures only.
    SmallVector<Type> argTypes;
    argTypes.reserve(argCaptures.size());
    for (Value v : argCaptures) argTypes.push_back(v.getType());

    SmallVector<Type> resultTypes(genericOp.getResultTypes().begin(),
                                  genericOp.getResultTypes().end());
    auto funcType = FunctionType::get(ctx, argTypes, resultTypes);

    // 4. Create the function at module top.
    OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
    auto outlinedFunc =
        func::FuncOp::create(moduleBuilder, loc, funcName, funcType);
    outlinedFunc.setPrivate();
    Block *entry = outlinedFunc.addEntryBlock();

    // 5. Build a single mapping that covers both arg captures and inlined
    // constants.
    IRMapping mapping;
    for (auto [origVal, blockArg] :
         llvm::zip(argCaptures, entry->getArguments()))
      mapping.map(origVal, blockArg);

    OpBuilder bodyBuilder(entry, entry->begin());
    for (auto [origVal, defOp] : constantsToClone) {
      Operation *clonedConst = bodyBuilder.clone(*defOp);
      mapping.map(origVal, clonedConst->getResult(0));
    }

    // 6. Clone the secret.generic into the function body using the mapping.
    //    The clone's operands and captured values will resolve via `mapping`
    //    to either the new func args or the inlined constants.
    Operation *clonedGeneric = bodyBuilder.clone(*genericOp, mapping);

    // 7. func.return the cloned generic's results.
    func::ReturnOp::create(bodyBuilder, loc, clonedGeneric->getResults());

    // 8. Replace the original generic with a call to the outlined function.
    //    Pass only the non-constant capture values as call args.
    OpBuilder callBuilder(genericOp);
    SmallVector<Value> callArgs(argCaptures.begin(), argCaptures.end());
    auto callOp =
        func::CallOp::create(callBuilder, loc, outlinedFunc, callArgs);
    genericOp->replaceAllUsesWith(callOp.getResults());
    genericOp->erase();

    return outlinedFunc;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<secret::SecretDialect>();
    registry.insert<tensor_ext::TensorExtDialect>();
    registry.addExtension(
        +[](MLIRContext *ctx, secret::SecretDialect *dialect) {
          dialect->addInterfaces<SecretInlinerInterface>();
        });
  }

  DenseMap<BlockArgument, int64_t> buildForcedLanesFromMerge(
      func::FuncOp mergedFunc, const Schedule &mergedSchedule,
      func::FuncOp outlinedReductionFunc) {
    DenseMap<BlockArgument, int64_t> forcedLanes;
    if (!mergedFunc || !outlinedReductionFunc) return forcedLanes;

    // 1. Find the merged function's secret.generic + yield.
    secret::GenericOp mergedGeneric;
    mergedFunc.walk([&](secret::GenericOp g) { mergedGeneric = g; });
    if (!mergedGeneric) return forcedLanes;
    auto mergedYield = cast<secret::YieldOp>(
        mergedGeneric.getRegion().front().getTerminator());

    // 2. result-index → upstream lane.
    DenseMap<unsigned, int64_t> resultLane;
    for (unsigned r = 0; r < mergedYield->getNumOperands(); ++r) {
      Operation *producer = mergedYield->getOperand(r).getDefiningOp();
      if (!producer) continue;
      auto it = mergedSchedule.lanes.find(producer);
      if (it == mergedSchedule.lanes.end()) continue;
      resultLane[r] = it->second;
    }

    // 3. Find the outlined function's secret.generic.
    secret::GenericOp reductionGeneric;
    outlinedReductionFunc.walk(
        [&](secret::GenericOp g) { reductionGeneric = g; });
    if (!reductionGeneric) return forcedLanes;

    // 4. Pin each body block arg using its positional index.
    //    Body block arg i corresponds to outlined func arg i, which
    //    corresponds to merged call result i.
    Block &gbody = reductionGeneric.getRegion().front();
    for (unsigned i = 0; i < gbody.getNumArguments(); ++i) {
      BlockArgument arg = gbody.getArgument(i);
      if (!arg.getType().isIntOrFloat()) continue;
      auto laneIt = resultLane.find(i);
      if (laneIt == resultLane.end()) continue;
      forcedLanes[arg] = laneIt->second;
    }

    return forcedLanes;
  }

  void runOnOperation() override {
    Dialect *mlirDialect = getContext().getLoadedDialect(dialect);

    getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      if (funcOp.empty()) return;
      if (tryUnrollingRecursiveBlock(&funcOp.getBlocks().front(),
                                     mlirDialect)) {
        sortTopologically(&funcOp.getBlocks().front());
      }
    });

    // Second pass: now safe to erase, walk is done
    for (auto funcs : functionDeleteList) {
      llvm::outs() << "Erasing function: " << funcs.getName() << "\n";
      for (auto &block : funcs.getBody().getBlocks())
        block.dropAllDefinedValueUses();
      funcs.getBody().dropAllReferences();
      funcs.erase();
    }

    prettyPrintRecursiveProgramTree(biscottiCalls.begin()->second.root);
    removeRedundantTensorCasts(getOperation());
    unrollTensorGenerates(getOperation());

    getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      if (funcOp.empty()) return;
      foldAllOpsInFunc(funcOp, funcOp.getContext());
    });

    getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      if (funcOp.empty()) return;
      // Step 1: build the BitVector
      llvm::BitVector eraseArgs(funcOp.getNumArguments());

      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        if (funcOp.getArgument(i).use_empty()) {
          eraseArgs.set(i);  // mark for deletion
        }
      }

      auto oldType = funcOp.getFunctionType();

      llvm::SmallVector<Type> newInputs;
      for (unsigned i = 0; i < oldType.getNumInputs(); ++i) {
        if (!eraseArgs.test(i)) newInputs.push_back(oldType.getInput(i));
      }
      auto newType = mlir::FunctionType::get(funcOp.getContext(), newInputs,
                                             oldType.getResults());

      // Step 2: erase them
      mlir::function_interface_impl::eraseFunctionArguments(funcOp, eraseArgs,
                                                            newType);
      auto uses =
          mlir::SymbolTable::getSymbolUses(funcOp, funcOp->getParentOp());
      if (uses) {
        for (auto use : *uses) {
          if (auto call = llvm::dyn_cast<func::CallOp>(use.getUser())) {
            llvm::SmallVector<Value> newOperands;

            for (auto [i, operand] : llvm::enumerate(call.getOperands())) {
              if (!eraseArgs.test(i)) {
                newOperands.push_back(operand);
              }
            }

            call->setOperands(newOperands);
          }
        }
      }
    });

    for (auto &calls : biscottiCalls) {
      processVectorizationCandidates(calls.second.root);
    }

    for (auto &calls : biscottiCalls) {
      // continue;
      DenseMap<recursiveProgramNode *, SmallVector<recursiveProgramNode *>>
          mergeableNodes;
      DenseSet<func::CallOp> visited;
      findScheduleMergingCandidates(calls.second.root, mergeableNodes, visited);

      // print mergeable nodes function name
      llvm::outs() << "Schedule merging candidates:\n";
      for (auto node : mergeableNodes) {
        llvm::outs() << "  " << node.first->function.getName() << "\n";
        for (int i = 1; i < node.second.size(); i++) {
          llvm::outs() << "    " << node.second[i]->function.getName() << "\n";
        }
      }

      for (auto node : mergeableNodes) {
        func::FuncOp merged;
        // DenseMap<int, SmallVector<Value>> argumentsToExpand;
        // DenseMap<int, SmallVector<Value>> resultsToExpand;
        // for (int i = 0; i < node.second.size(); i++) {
        //   for (int j = 0; j < node.second[i]->caller.getArgOperands().size();
        //        j++)
        //     if (isa<secret::SecretType>(
        //             node.second[i]->caller.getArgOperands()[j].getType()))
        //       argumentsToExpand[j].push_back(
        //           node.second[i]->caller.getArgOperands()[j]);

        //   for (int j = 0; j < node.second[i]->caller.getResults().size();
        //   j++)
        //     if (isa<secret::SecretType>(
        //             node.second[i]->caller.getResults()[j].getType()))
        //       resultsToExpand[j].push_back(
        //           node.second[i]->caller.getResults()[j]);
        // }

        OpBuilder builder(&node.first->function.getBody().front(),
                          node.first->function.getBody().front().begin());
        // This part builds a ciphertext model based on the offsets
        // of the insert chains.
        // SmallVector<Value> collectMergedTensorArgs;
        // for (auto args : argumentsToExpand) {
        //   auto tensorType = expandedArgTypes[args.first];
        //   auto ctxt = createMergedCipherTextMappings(tensorType, args.second,
        //   builder); collectMergedTensorArgs.push_back(
        //       createNewInsertOpsFromSeedOps(ctxt, tensorType, builder));
        //   for (auto slot: ctxt) {
        //     llvm::outs() << slot.index << ": " << slot.op << "\n";
        //   }
        // }

        SmallVector<func::FuncOp> functionsToMerge;
        SmallVector<Schedule> schedulesToMerge;
        Schedule finalSchedule;
        for (int i = 0; i < node.second.size(); i++) {
          functionsToMerge.push_back(node.second[i]->function);
          schedulesToMerge.push_back(node.second[i]->coyoteSchedule);
        }
        mergeSchedulesWithNW(functionsToMerge, schedulesToMerge, merged,
                             finalSchedule);

        merged.dump();
        prettyPrintSchedule(finalSchedule);
        for (auto &lanes : finalSchedule.lanes) {
          llvm::outs() << "Op: " << *lanes.first << "\n";
          llvm::outs() << "Lane: " << lanes.second << "\n";
        }

        // Determine the expanded argument/result types based on the number of
        // merged functions The idea is that if we N functions to be merged, we
        // get the max (N dim sizes) and the new expanded type is N * max (N dim
        // sizes)
        // DenseMap<int, RankedTensorType> expandedArgTypes;
        // for (auto args : argumentsToExpand) {
        // int maxDimSize = 0;

        // for (auto arg : args.second) {
        //   auto tensorType = mlir::cast<RankedTensorType>(
        //       mlir::cast<secret::SecretType>(arg.getType()).getValueType());
        //   maxDimSize = std::max(maxDimSize, (int)tensorType.getDimSize(1));
        // }

        // auto tensorType = mlir::cast<RankedTensorType>(
        //     mlir::cast<secret::SecretType>(args.second[0].getType()).getValueType());
        // int mergedDimSize = maxDimSize * args.second.size();
        // SmallVector<int64_t> newShape(tensorType.getShape());
        // newShape.back() = mergedDimSize;
        // expandedArgTypes[args.first] = RankedTensorType::get(newShape,
        // tensorType.getElementType());
        // }

        // for (auto expandedArg : expandedArgTypes) {
        //   llvm::outs() << "Expanding argument " << expandedArg.first
        //                << " to type " << expandedArg.second << "\n";
        // }

        // DenseMap<int, RankedTensorType> expandedResultTypes;
        // for (auto args : resultsToExpand) {
        //   int maxDimSize = 0;

        //   for (auto arg : args.second) {
        //     auto tensorType = mlir::cast<RankedTensorType>(
        //         mlir::cast<secret::SecretType>(arg.getType()).getValueType());
        //     maxDimSize = std::max(maxDimSize, (int)tensorType.getDimSize(1));
        //   }

        //   auto tensorType = mlir::cast<RankedTensorType>(
        //       mlir::cast<secret::SecretType>(args.second[0].getType()).getValueType());
        //   int mergedDimSize = maxDimSize * args.second.size();
        //   SmallVector<int64_t> newShape(tensorType.getShape());
        //   newShape.back() = mergedDimSize;
        //   expandedResultTypes[args.first] = RankedTensorType::get(newShape,
        //   tensorType.getElementType());
        // }

        // for (auto expandedResult : expandedResultTypes) {
        //   llvm::outs() << "Expanding result " << expandedResult.first
        //                << " to type " << expandedResult.second << "\n";
        // }

        // for (int i = 1; i < node.second.size(); i++) {
        //   auto current = node.second[i]->function;
        //   auto result = mergeWithNeedlemanWunsch(merged, current, merged);

        //   if (succeeded(result)) {
        //     merged.dump();
        //   } else {
        //     assert(false && "Merging failed");
        //   }
        // }
        // Change the function definition of the merged function to have the new
        // expanded types. Widen each secret arg of the function individually
        // for (unsigned i = 0; i < merged.getNumArguments(); ++i) {
        //   auto targetType = expandedArgTypes[i];
        //   Type secretTargetType = secret::SecretType::get(targetType); 
        //   Type oldSecretArgType = merged.getArgument(i).getType();
        //   if (isa<secret::SecretType>(oldSecretArgType) && secretTargetType
        //   != oldSecretArgType) {
        //       widenFunctionArgAndPropagate(merged, i, secretTargetType);
        //   }
        // }
        ModuleOp module = node.second[0]->function->getParentOfType<ModuleOp>();
        module.push_back(merged);

        // reduction steps
        // TODO: check if the merged funcs have more args than the base
        // functions ideally we should handle this, but it is a waste of time to
        // handle all of these edge cases right now. Ideally after staging all
        // the progress arguments should disappear and won't cause any issues.
        // But we should add an assert which checks for this, would be easier to
        // debug these self-sabotage issues if they come up.

        // We make an assumption that all the new merged tensor argument are in
        // the same defining op. assert(llvm::all_of(collectMergedTensorArgs,
        // [&](Value v) {
        //   return v.getDefiningOp()->getParentOp() ==
        //   collectMergedTensorArgs[0].getDefiningOp()->getParentOp();
        // }) && "All merged tensor arguments should be defined by the same
        // op"); Operation *op = collectMergedTensorArgs[0].getDefiningOp();
        // auto genericOp = cast<secret::GenericOp>(op->getParentOp());

        // // Update yield first
        // auto yieldOp = genericOp.getYieldOp();
        // yieldOp.getValuesMutable().append(collectMergedTensorArgs);

        // // Build new result types from yield
        // auto newTypes = llvm::to_vector<4>(llvm::map_range(
        //     yieldOp.getValues().getTypes(),
        //     [](Type t) -> Type { return secret::SecretType::get(t); }));

        // // Clone with new types
        // auto *newOp =
        //     cloneWithNewResultTypes(genericOp.getOperation(), newTypes);
        // // insert the new generic op inside the same block as the older
        // // generic op.
        // genericOp->getBlock()->getOperations().insert(
        //     std::next(Block::iterator(genericOp)), newOp);
        // // Insert after old generic
        // newOp->moveAfter(genericOp);

        // // Replace old results
        // for (auto [oldRes, newRes] :
        //      llvm::zip(genericOp->getResults(), newOp->getResults()))
        //   oldRes.replaceAllUsesWith(newRes);

        // genericOp->erase();

        // New results are the last N
        // auto newGenericOp = cast<secret::GenericOp>(newOp);
        // auto newResultStartIter = newGenericOp.getResults().drop_front(
        //     newGenericOp.getNumResults() - collectMergedTensorArgs.size());

        // // for (auto caller : node.second)
        // //   caller->caller.setCalleeAttr(mlir::SymbolRefAttr::get(merged));

        SmallVector<Value> callArgs;
        for (auto n : node.second)
          for (auto arg : n->caller.getArgOperands()) callArgs.push_back(arg);

        builder.setInsertionPoint(node.second[0]->caller);
        auto callOp = func::CallOp::create(
            builder, node.second[0]->caller.getLoc(), merged.getName(),
            merged.getFunctionType().getResults(), callArgs);

        for (int i = 0; i < node.second.size(); i++) {
          if (node.second[0]->caller.getNumResults() != 1)
            llvm::report_fatal_error("expected single-result calls");
          node.second[i]->caller.getResult(0).replaceAllUsesWith(
              callOp.getResults()[i]);
          node.second[i]->caller.erase();
        }
        node.first->children.clear();

        auto findCommonGeneric = [&]() -> secret::GenericOp {
          secret::GenericOp first = nullptr;
          for (auto res : callOp.getResults()) {
            for (auto &use : res.getUses()) {
              auto owner = dyn_cast<secret::GenericOp>(use.getOwner());
              if (!owner) return nullptr;
              if (!first)
                first = owner;
              else if (first != owner)
                return nullptr;
            }
          }
          return first;
        };

        // Now we need to add the merged function result to the generic block
        // args that use the old function results. For simplicity we check that
        // all uses are in the same generic op, otherwise it asserts.
        auto commonGeneric = findCommonGeneric();
        assert(commonGeneric &&
               "All results of merged functions must be used by the same "
               "secret.generic");

        Block &body = commonGeneric.getRegion().front();
        Location loc = commonGeneric.getLoc();
        builder.setInsertionPointToStart(&body);

        unsigned numOldArgs = body.getNumArguments();

        // 1. New operand list (callOp's results) + add new scalar block args.
        SmallVector<Value> newOperands;
        SmallVector<BlockArgument> newArgs;
        for (unsigned k = 0; k < numOldArgs; ++k) {
          auto tensorTy = cast<RankedTensorType>(body.getArgument(k).getType());
          unsigned L = tensorTy.getNumElements();
          Type elemTy = tensorTy.getElementType();
          for (unsigned j = 0; j < L; ++j) {
            newOperands.push_back(callOp.getResult(k * L + j));
            newArgs.push_back(body.addArgument(elemTy, loc));
          }
        }

        // 2. For each old tensor arg, rebuild a tensor<L x elemTy> from its L
        //    new scalar args and replace the old arg's uses with it.
        unsigned cursor = 0;
        for (unsigned k = 0; k < numOldArgs; ++k) {
          BlockArgument oldArg = body.getArgument(k);
          auto tensorTy = cast<RankedTensorType>(oldArg.getType());
          unsigned L = tensorTy.getNumElements();
          SmallVector<Value> slice(newArgs.begin() + cursor,
                                   newArgs.begin() + cursor + L);
          Value rebuilt =
              tensor::FromElementsOp::create(builder, loc, tensorTy, slice);
          oldArg.replaceAllUsesWith(rebuilt);
          cursor += L;
        }

        // 3. Erase the (now unused) old tensor block args.
        for (unsigned i = numOldArgs; i-- > 0;) body.eraseArgument(i);

        // 4. Update the generic's operand list.
        commonGeneric->setOperands(newOperands);

        // Outline the secret.generic into a new function, then scalarize the
        // body to be fed into coyote.
        func::FuncOp reductionKernel = outlineSecretGeneric(commonGeneric);
        MLIRContext *ctx = &getContext();
        RewritePatternSet patterns(ctx);
        patterns.add<ScalarizeAnyElementwise>(ctx);
        tensor::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
        tensor::FromElementsOp::getCanonicalizationPatterns(patterns, ctx);
        (void)applyPatternsGreedily(reductionKernel, std::move(patterns));
        foldAllOpsInFunc(reductionKernel, ctx);

        // --- Wrap scalar secret.generic block args with @__coyote_load so the
        //     Coyote vectorizer sees them as input loads (otherwise they fall
        //     through to "const" in the converter). ---
        {
          ModuleOp module = reductionKernel->getParentOfType<ModuleOp>();
          OpBuilder modBuilder(ctx);

          // Find the inner secret.generic.
          secret::GenericOp innerGeneric;
          reductionKernel.walk([&](secret::GenericOp g) { innerGeneric = g; });
          if (innerGeneric) {
            Block &gbody = innerGeneric.getRegion().front();
            OpBuilder bodyBuilder(ctx);
            bodyBuilder.setInsertionPointToStart(&gbody);

            // Ensure one @__coyote_load stub per element type used. The stub is
            // an empty (declaration-only) private func: (T) -> T.
            DenseMap<Type, func::FuncOp> stubByType;
            auto getStub = [&](Type elemTy) -> func::FuncOp {
              auto it = stubByType.find(elemTy);
              if (it != stubByType.end()) return it->second;
              std::string sym =
                  "__coyote_load";  // single name (i32-only for now);
                                    // if mixing types, add a suffix
              if (auto existing = module.lookupSymbol<func::FuncOp>(sym)) {
                // Verify type matches; otherwise rename per-type. For i32-only
                // this is fine.
                stubByType[elemTy] = existing;
                return existing;
              }
              OpBuilder::InsertionGuard g(modBuilder);
              modBuilder.setInsertionPointToStart(module.getBody());
              auto fnType = modBuilder.getFunctionType({elemTy}, {elemTy});
              auto stub = func::FuncOp::create(modBuilder, module.getLoc(), sym,
                                               fnType);
              stub.setPrivate();  // declaration-only — no entry block added.
              stubByType[elemTy] = stub;
              return stub;
            };

            for (BlockArgument arg : gbody.getArguments()) {
              if (!arg.getType().isIntOrFloat()) continue;
              func::FuncOp stub = getStub(arg.getType());
              auto call = func::CallOp::create(bodyBuilder, arg.getLoc(), stub,
                                               ValueRange{arg});
              // Redirect every use of arg EXCEPT the call op itself.
              arg.replaceAllUsesExcept(call.getResult(0), call);
            }
          }
        }

        auto forcedLanes =
            buildForcedLanesFromMerge(merged, finalSchedule, reductionKernel);
        auto reductionSchedule =
            runCoyoteVectorizer(reductionKernel, forcedLanes);
        prettyPrintSchedule(reductionSchedule);
      }
    }

    getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      if (funcOp.empty()) return;
      foldAllOpsInFunc(funcOp, funcOp.getContext());
    });
  }
};

void RecursiveCallVectorization::unrollTensorGenerates(Operation *op) {
  op->walk([](tensor::GenerateOp generateOp) {
    auto resultType = mlir::dyn_cast<RankedTensorType>(generateOp.getType());
    if (!resultType || !resultType.hasStaticShape()) return;

    int64_t numElements = resultType.getNumElements();
    OpBuilder builder(generateOp);
    Location loc = generateOp.getLoc();

    // Start with a zero tensor
    Value result = arith::ConstantOp::create(builder, loc, resultType,
                                             builder.getZeroAttr(resultType));

    // For each element, inline the body with a constant index
    for (int64_t i = 0; i < numElements; i++) {
      IRMapping mapping;
      Value idx = arith::ConstantIndexOp::create(builder, loc, i);
      mapping.map(generateOp.getBody().getArgument(0), idx);

      // Clone all ops in the body except the yield
      for (auto &bodyOp : generateOp.getBody().front().without_terminator()) {
        auto *cloned = builder.clone(bodyOp, mapping);
        for (auto [oldResult, newResult] :
             llvm::zip(bodyOp.getResults(), cloned->getResults()))
          mapping.map(oldResult, newResult);
      }

      // Get the yielded value
      auto yieldOp =
          cast<tensor::YieldOp>(generateOp.getBody().front().getTerminator());
      Value yieldedVal = mapping.lookup(yieldOp.getValue());

      // Insert into result tensor
      Value idxVal = arith::ConstantIndexOp::create(builder, loc, i);
      result = tensor::InsertOp::create(builder, loc, yieldedVal, result,
                                        ValueRange{idxVal});
    }

    generateOp.replaceAllUsesWith(result);
    generateOp.erase();
  });
}

void RecursiveCallVectorization::removeRedundantTensorCasts(Operation *Op) {
  bool changed = true;
  while (changed) {
    changed = false;

    // Step 1: Remove static->dynamic tensor.cast in secret.generic yields
    // and propagate static types through generic body
    Op->walk<WalkOrder::PostOrder>([&](secret::GenericOp genericOp) {
      auto yieldOp =
          cast<secret::YieldOp>(genericOp.getBody()->getTerminator());

      // Remove redundant casts in yield
      for (auto [idx, yieldedVal] : llvm::enumerate(yieldOp->getOperands())) {
        auto castOp = yieldedVal.getDefiningOp<tensor::CastOp>();
        if (!castOp) continue;
        auto srcType =
            mlir::dyn_cast<RankedTensorType>(castOp.getSource().getType());
        auto dstType = mlir::dyn_cast<RankedTensorType>(castOp.getType());
        if (!srcType || !dstType) continue;
        if (!srcType.hasStaticShape() || dstType.hasStaticShape()) continue;
        yieldOp->setOperand(idx, castOp.getSource());
        changed = true;
      }

      // Update block arg types from operand types
      for (auto [arg, operand] : llvm::zip(genericOp.getBody()->getArguments(),
                                           genericOp.getOperands())) {
        auto secretType = mlir::dyn_cast<secret::SecretType>(operand.getType());
        if (secretType && arg.getType() != secretType.getValueType()) {
          arg.setType(secretType.getValueType());
          changed = true;
        }
      }

      // Update op result types inside body
      genericOp.getBody()->walk([&](Operation *op) {
        for (auto result : op->getResults()) {
          auto tensorType = mlir::dyn_cast<RankedTensorType>(result.getType());
          if (!tensorType || tensorType.hasStaticShape()) continue;
          for (auto operand : op->getOperands()) {
            auto operandType =
                mlir::dyn_cast<RankedTensorType>(operand.getType());
            if (operandType && operandType.hasStaticShape()) {
              result.setType(operandType);
              changed = true;
              break;
            }
          }
        }
      });

      // Sync generic result types with yield operand types
      for (auto [idx, yieldedVal] : llvm::enumerate(yieldOp->getOperands())) {
        auto staticType =
            mlir::dyn_cast<RankedTensorType>(yieldedVal.getType());
        if (staticType && staticType.hasStaticShape()) {
          auto newType = secret::SecretType::get(staticType);
          if (genericOp.getResult(idx).getType() != newType) {
            genericOp.getResult(idx).setType(newType);
            changed = true;
          }
        }
      }
    });

    // Step 2: Update call ops and callee signatures together
    Op->walk<WalkOrder::PostOrder>([&](func::CallOp callOp) {
      auto *callee =
          SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr());
      auto calleeFunc = dyn_cast<func::FuncOp>(callee);
      if (!calleeFunc) return;

      // Update callee input types from call operands
      SmallVector<Type> newInputTypes;
      bool inputNeedsUpdate = false;
      for (auto [operand, inputType] :
           llvm::zip(callOp->getOperands(),
                     calleeFunc.getFunctionType().getInputs())) {
        newInputTypes.push_back(operand.getType());
        if (operand.getType() != inputType) inputNeedsUpdate = true;
      }
      if (inputNeedsUpdate) {
        calleeFunc.setType(
            FunctionType::get(calleeFunc.getContext(), newInputTypes,
                              calleeFunc.getFunctionType().getResults()));
        for (auto [arg, newType] :
             llvm::zip(calleeFunc.getArguments(), newInputTypes))
          arg.setType(newType);
        changed = true;
      }

      // Update call result types from callee return types
      for (auto [result, newType] :
           llvm::zip(callOp->getResults(),
                     calleeFunc.getFunctionType().getResults())) {
        if (result.getType() != newType) {
          result.setType(newType);
          changed = true;
        }
      }
    });

    // Step 3: Update function return types
    Op->walk<WalkOrder::PostOrder>([&](func::FuncOp funcOp) {
      auto returnOp =
          cast<func::ReturnOp>(funcOp.getBody().back().getTerminator());
      SmallVector<Type> newResultTypes;
      for (auto val : returnOp->getOperands())
        newResultTypes.push_back(val.getType());
      if (newResultTypes != funcOp.getFunctionType().getResults()) {
        funcOp.setType(FunctionType::get(funcOp.getContext(),
                                         funcOp.getFunctionType().getInputs(),
                                         newResultTypes));
        changed = true;
      }
    });
  }
}

// Makes the recursive tree data structure stale. Need to refresh
// the tree with new attributes and call ops after using this.
void RecursiveCallVectorization::removeDuplicateFunctions(
    recursiveProgramNode *node) {
  if (!node) return;

  for (recursiveProgramNode *child : node->children) {
    removeDuplicateFunctions(child);
  }

  std::set<std::pair<int, int>> staticArgsKey;
  for (auto &[op, idx] : node->staticArgumentValues)
    staticArgsKey.insert(
        {cast<mlir::IntegerAttr>(op).getValue().getSExtValue(), idx});

  if (functionCache.find(staticArgsKey) != functionCache.end()) {
    func::FuncOp cachedFunc = functionCache[staticArgsKey];
    llvm::outs() << "Removing duplicate function: " << node->function.getName()
                 << " (reusing " << cachedFunc.getName() << ")\n";
    node->caller.setCallee(cachedFunc.getName());
    functionDeleteList.insert(node->function);
    node->function = cachedFunc;
  } else {
    functionCache[staticArgsKey] = node->function;
    llvm::outs() << "Caching function: " << node->function.getName() << "\n";
  }
}

void RecursiveCallVectorization::findMergeableRecursiveCallNodes(
    recursiveProgramNode *node,
    SmallVector<recursiveProgramNode *> &mergeableNodes) {
  if (!node) return;

  for (recursiveProgramNode *child : node->children) {
    findMergeableRecursiveCallNodes(child, mergeableNodes);
  }

  bool areAllChildrenLeaves = std::all_of(
      node->children.begin(), node->children.end(),
      [](recursiveProgramNode *child) { return child->children.empty(); });

  if (areAllChildrenLeaves && !node->children.empty()) {
    mergeableNodes.push_back(node);
  }
}

int RecursiveCallVectorization::countNodeFunctionSize(
    recursiveProgramNode *node) {
  if (!node) return 0;

  int size = 0;
  node->function.walk([&](secret::GenericOp genOp) {
    genOp.getBody()->walk([&](Operation *op) {
      if (isa<tensor::ExtractOp>(op) ||
          ((op->getDialect()->getNamespace() == "arith") &&
           !isa<arith::ConstantOp>(op)))
        ++size;
    });
  });

  return size;
}

void RecursiveCallVectorization::mergeRecursiveCallNodes(
    SmallVector<recursiveProgramNode *> &mergeableNodes) {
  std::queue<recursiveProgramNode *> workQueue;
  for (auto *node : mergeableNodes) workQueue.push(node);

  DenseSet<StringRef> mergedFunctions;
  while (!workQueue.empty()) {
    recursiveProgramNode *node = workQueue.front();
    workQueue.pop();

    assert(node && "Node should not be null");

    if (mergedFunctions.contains(node->function.getName())) {
      llvm::outs() << "Skipping already merged function: "
                   << node->function.getName() << "\n";
      node->children.clear();
      continue;
    }

    int nodeSize = countNodeFunctionSize(node);
    llvm::outs() << "Trying to merge node with function: "
                 << node->function.getName() << "\n";

    for (recursiveProgramNode *child : node->children) {
      int childSize = countNodeFunctionSize(child);
      nodeSize += childSize;
      llvm::outs() << "   Child node with function: "
                   << child->function.getName() << ", size: " << childSize
                   << "\n";
    }

    llvm::outs() << "Node function size: " << nodeSize << "\n";
    // TODO: Tune this threshold.
    if (nodeSize > NODE_SIZE_THRESHOLD) {
      llvm::outs() << "   Skipping merge due to large node size.\n";
      continue;
    }

    mergedFunctions.insert(node->function.getName());
    for (recursiveProgramNode *child : node->children) {
      ModuleOp parentModule = node->function->getParentOfType<ModuleOp>();
      if (!parentModule)
        llvm::errs() << "Error: Parent module not found for function "
                     << node->function.getName() << "\n";

      auto &ChildFunction = child->function;
      // Collect call ops first, only within node->function
      SmallVector<func::CallOp> callsToInline;
      node->function.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == ChildFunction.getName()) {
          callsToInline.push_back(callOp);
        }
      });

      // Now inline them
      InlinerInterface interface(&getContext());
      InlinerConfig config;
      for (auto callOp : callsToInline) {
        llvm::outs() << "Inlining " << ChildFunction.getName() << " into "
                     << node->function.getName() << "\n";
        if (failed(inlineCall(interface, config.getCloneCallback(), callOp,
                              ChildFunction,
                              ChildFunction.getCallableRegion()))) {
          llvm::errs() << "Failed to inline " << ChildFunction.getName()
                       << "\n";
        } else {
          // If the call still has uses, something went wrong with replacement
          // For now, don't erase — just leave it
          if (callOp.use_empty()) {
            callOp.erase();
          }
        }
      }
      // Erase if no remaining calls
      if (ChildFunction.symbolKnownUseEmpty(parentModule))
        ChildFunction.erase();
    }

    if (countNodeFunctionSize(node) < NODE_SIZE_THRESHOLD && node->parent)
      workQueue.push(node->parent);

    // TODO: Clear children after merging, since they've been inlined into the
    // parent. This assumes that the merge step above didn't fail. It it failed,
    // we might generate incorrect code. But there is a weak guarantee that this
    // will not happen. Look at this in future, if there are any bugs.
    node->children.clear();
  }
}

// Pattern 1: arith.addi (tensor.from_elements %a), (tensor.from_elements %b)
//         -> tensor.from_elements (arith.addi %a, %b)
// Only for tensor<1xi32>
class FoldAddOfFromElements final : public OpRewritePattern<arith::AddIOp> {
 public:
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  FoldAddOfFromElements(MLIRContext *context)
      : OpRewritePattern<arith::AddIOp>(context) {}

  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override {
    // Check result is tensor<1x...>
    auto resultType = mlir::dyn_cast<RankedTensorType>(addOp.getType());
    if (!resultType || !resultType.hasStaticShape()) return failure();
    if (resultType.getNumElements() != 1) return failure();

    // Check both operands are tensor.from_elements
    auto lhsFromElements =
        addOp.getLhs().getDefiningOp<tensor::FromElementsOp>();
    auto rhsFromElements =
        addOp.getRhs().getDefiningOp<tensor::FromElementsOp>();
    if (!lhsFromElements || !rhsFromElements) return failure();
    if (lhsFromElements.getElements().size() != 1) return failure();
    if (rhsFromElements.getElements().size() != 1) return failure();
    // Create scalar add and wrap in from_elements
    Value scalarAdd = arith::AddIOp::create(rewriter, addOp.getLoc(),
                                            lhsFromElements.getElements()[0],
                                            rhsFromElements.getElements()[0]);
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(addOp, resultType,
                                                        ValueRange{scalarAdd});
    return success();
  }
};

// Pattern 2: tensor.extract (tensor.from_elements %x)[0] -> %x
// Only for tensor<1x...>
class FoldExtractFromFromElements final
    : public OpRewritePattern<tensor::ExtractOp> {
 public:
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  FoldExtractFromFromElements(MLIRContext *context)
      : OpRewritePattern<tensor::ExtractOp>(context) {}

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Check source is a size-1 tensor
    auto tensorType =
        mlir::dyn_cast<RankedTensorType>(extractOp.getTensor().getType());
    if (!tensorType || !tensorType.hasStaticShape() ||
        tensorType.getNumElements() != 1)
      return failure();

    // Check source is tensor.from_elements with single element
    auto fromElements =
        extractOp.getTensor().getDefiningOp<tensor::FromElementsOp>();
    if (!fromElements || fromElements.getElements().size() != 1)
      return failure();

    rewriter.replaceOp(extractOp, fromElements.getElements()[0]);
    return success();
  }
};

void RecursiveCallVectorization::foldAllOpsInFunc(func::FuncOp &funcOp,
                                                  MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);
  // for (auto *dialect : ctx->getLoadedDialects())
  //   llvm::outs() << dialect->getNamespace() << "\n";
  for (auto *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);
  patterns.add<secret::MergeAdjacentGenerics>(ctx);
  patterns.add<FoldAddOfFromElements, FoldExtractFromFromElements>(ctx);

  // fold constants and apply canonicalization patterns
  GreedyRewriteConfig config;
  // Makes compilation faster, but may miss some patterns.
  config.setUseTopDownTraversal();
  (void)applyPatternsGreedily(funcOp, std::move(patterns), config);

  // Call DCE for the simplification
  IRRewriter rewriter(funcOp.getContext());
  (void)mlir::eraseUnreachableBlocks(rewriter,
                                     funcOp.getOperation()->getRegions());
}

void RecursiveCallVectorization::buildRecursiveAttributes(Block *block,
                                                          Dialect *dialect) {
  for (auto &op : block->getOperations()) {
    if (dialect && op.getDialect() != dialect) continue;

    if (!isa<func::CallOp>(op)) continue;

    int attrValue;
    if (findBiscottiAttribute(op.getResult(0), "biscotti.call", attrValue) &&
        biscottiCalls.find(&op) == biscottiCalls.end()) {
      recursiveProgramInfo call;
      call.call = &op;
      SmallVector<int64_t> attrValueArray;
      biscottiCalls[&op] = call;
    }
  }

  for (auto &calls : biscottiCalls) {
    Operation *op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    func::FuncOp funcOp = getEnclosingFunction(op, parentModule);
    if (!funcOp) continue;

    for (auto argOps : funcOp.getArguments()) {
      int attrValue;
      if (findBiscottiAttribute(argOps, "biscotti.progress_argument",
                                attrValue)) {
        recursiveProgramInfo.progressArguments.push_back({&argOps, attrValue});
        Operation *defOp = op->getOperand(attrValue).getDefiningOp();
        assert(defOp && "static arg operand must have a defining op");
        // arith.constant stores its constant under the "value" attribute
        auto attr = cast<TypedAttr>(defOp->getAttr("value"));
        recursiveProgramInfo.staticArgumentValues.push_back({attr, attrValue});
      }
    }
  }

  for (auto &calls : biscottiCalls) {
    Operation *op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    func::FuncOp funcOp = getEnclosingFunction(op, parentModule);
    if (!funcOp) continue;

    funcOp.walk([&](Operation *calledOp) {
      auto call = dyn_cast<func::CallOp>(calledOp);
      if (!call) return;

      int attrValue;
      if (findBiscottiAttribute(call->getResult(0), "biscotti.recursive_call",
                                attrValue)) {
        recursiveProgramInfo.recursiveCalls.push_back({calledOp, attrValue});
      }
    });

    funcOp.walk([&](Operation *baseOp) {
      int attrValue;
      if (findBiscottiAttributeOnOps(baseOp, "biscotti.base_condition",
                                     attrValue)) {
        recursiveProgramInfo.baseConditions.push_back({baseOp, attrValue});
      }
    });
  }

  for (auto &calls : biscottiCalls) {
    printRecursiveAttributes(&calls.second);
  }
}

static int functionCounter = 0;
void RecursiveCallVectorization::buildRecursiveCallTree(
    Operation *rootOp, recursiveProgramInfo &recursiveProgramInfo) {
  std::queue<std::pair<Operation *, recursiveProgramNode *>> workQueue;

  recursiveProgramNode *root = new recursiveProgramNode();
  root->staticArgumentValues = recursiveProgramInfo.staticArgumentValues;
  recursiveProgramInfo.root = root;
  workQueue.push({rootOp, root});
  int recursiveCallCounter = 0;

  // Essentially we have a recursive function callOp here.
  ModuleOp parentModule = rootOp->getParentOfType<ModuleOp>();
  func::FuncOp funcOp = getEnclosingFunction(rootOp, parentModule);
  if (!funcOp) {
    llvm::outs() << "Error: Could not find enclosing function for operation.\n";
    return;
  }
  functionDeleteList.insert(funcOp);

  while (!workQueue.empty()) {
    // llvm::outs() << "== Processing node in recursive call tree...\n";
    // Pop a node to be processed.
    Operation *op = workQueue.front().first;
    recursiveProgramNode *currentNode = workQueue.front().second;
    workQueue.pop();
    // llvm::outs() << *op << "\n";

    func::FuncOp funcOpCloned = funcOp.clone();
    // Set these clones to private, so they can be safely deleted later.
    funcOpCloned.setPrivate();
    funcOpCloned.setName(funcOp.getName().str() + "_clone_" +
                         std::to_string(functionCounter) + "_" +
                         std::to_string(recursiveCallCounter++));

    // Insert the constant arguments into the cloned function.
    // Then replace uses of the arguments with these constants.
    for (auto knownValue : currentNode->staticArgumentValues) {
      Operation *newConstant =
          insertConstantAtTop(funcOpCloned, knownValue.first);
      funcOpCloned.getArgument(knownValue.second)
          .replaceAllUsesWith(newConstant->getResult(0));
    }

    // Perform Constant Op propagation + Op folding + DCE.
    foldAllOpsInFunc(funcOpCloned, funcOp.getContext());
    // funcOpCloned.dump();
    parentModule.push_back(funcOpCloned);

    currentNode->function = funcOpCloned;
    dyn_cast<func::CallOp>(op).setCallee(funcOpCloned.getName());
    currentNode->caller = dyn_cast<func::CallOp>(op);

    // analyse the cloned function for further recursive calls.
    // find static argument values for each recursive call.
    // add the recursive calls as children to the current node and add to
    // process queue.
    funcOpCloned.walk([&](Operation *calledOp) {
      auto call = dyn_cast<func::CallOp>(calledOp);
      if (!call) return;

      int attrValue;
      if (findBiscottiAttribute(call->getResult(0), "biscotti.recursive_call",
                                attrValue)) {
        recursiveProgramNode *childNode = new recursiveProgramNode();

        for (auto progressArg : recursiveProgramInfo.progressArguments) {
          int attrValue = progressArg.second;

          Operation *defOp = calledOp->getOperand(attrValue).getDefiningOp();
          assert(defOp && defOp->hasTrait<OpTrait::ConstantLike>() &&
                 "progress argument must be a constant");

          // arith.constant stores its value under the "value" attribute.
          auto attr = cast<TypedAttr>(defOp->getAttr("value"));
          childNode->staticArgumentValues.push_back({attr, attrValue});
        }
        childNode->parent = currentNode;
        currentNode->children.push_back(childNode);
        workQueue.push({calledOp, childNode});
      }
    });
  }
  prettyPrintRecursiveProgramTree(root);
}

// Rebuilds the tree structure with updated call ops and attributes,
// after merging duplicate functions. This is necessary because the
// merge step can change the call ops and static argument values,
// which are used to build the tree structure.
void RecursiveCallVectorization::refreshRecursiveCallTree(
    Operation *rootOp, recursiveProgramInfo &recursiveProgramInfo) {
  std::queue<std::pair<Operation *, recursiveProgramNode *>> workQueue;

  recursiveProgramNode *root = new recursiveProgramNode();
  root->staticArgumentValues = recursiveProgramInfo.staticArgumentValues;
  recursiveProgramInfo.root = root;
  workQueue.push({rootOp, root});
  int recursiveCallCounter = 0;

  // Essentially we have a recursive function callOp here.
  ModuleOp parentModule = rootOp->getParentOfType<ModuleOp>();
  while (!workQueue.empty()) {
    llvm::outs() << "== Processing node in recursive call tree...\n";
    // Pop a node to be processed.
    Operation *op = workQueue.front().first;
    recursiveProgramNode *currentNode = workQueue.front().second;
    workQueue.pop();
    // llvm::outs() << *op << "\n";

    func::FuncOp funcOp = getEnclosingFunction(op, parentModule);
    if (!funcOp) {
      llvm::outs()
          << "Error: Could not find enclosing function for operation.\n";
      return;
    }

    currentNode->function = funcOp;
    currentNode->caller = dyn_cast<func::CallOp>(op);

    // analyse the cloned function for further recursive calls.
    // find static argument values for each recursive call.
    // add the recursive calls as children to the current node and add to
    // process queue.
    funcOp.walk([&](Operation *calledOp) {
      auto call = dyn_cast<func::CallOp>(calledOp);
      if (!call) return;

      int attrValue;
      if (findBiscottiAttribute(call->getResult(0), "biscotti.recursive_call",
                                attrValue)) {
        recursiveProgramNode *childNode = new recursiveProgramNode();

        for (auto progressArg : recursiveProgramInfo.progressArguments) {
          int attrValue = progressArg.second;

          Operation *defOp = calledOp->getOperand(attrValue).getDefiningOp();
          assert(defOp && defOp->hasTrait<OpTrait::ConstantLike>() &&
                 "progress argument must be a constant");

          // arith.constant stores its value under the "value" attribute.
          auto attr = cast<TypedAttr>(defOp->getAttr("value"));
          childNode->staticArgumentValues.push_back({attr, attrValue});
        }
        childNode->parent = currentNode;
        currentNode->children.push_back(childNode);
        workQueue.push({calledOp, childNode});
      }
    });
  }
  prettyPrintRecursiveProgramTree(root);
}

bool RecursiveCallVectorization::tryUnrollingRecursiveBlock(Block *block,
                                                            Dialect *dialect) {
  if (auto funcOp = dyn_cast<func::FuncOp>(block->getParentOp()))
    if (funcOp.getName().contains("clone")) return false;

  llvm::outs() << "Analyzing block for recursive call vectorization: \n";

  buildRecursiveAttributes(block, dialect);
  for (auto &calls : biscottiCalls) {
    Operation *op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    buildRecursiveCallTree(op, recursiveProgramInfo);
    functionCounter++;
  }

  for (auto &calls : biscottiCalls) {
    removeDuplicateFunctions(calls.second.root);
  }
  // The inlining and specialization above has potentially poisoned the tree
  // structure. Instead of making correct inplace updates, just rebuild the tree
  // from scratch.
  biscottiCalls.clear();
  buildRecursiveAttributes(block, dialect);
  for (auto &calls : biscottiCalls) {
    Operation *op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    refreshRecursiveCallTree(op, recursiveProgramInfo);
  }

  for (auto &calls : biscottiCalls) {
    recursiveProgramNode *root = calls.second.root;
    SmallVector<recursiveProgramNode *> mergeableNodes;
    findMergeableRecursiveCallNodes(root, mergeableNodes);
    mergeRecursiveCallNodes(mergeableNodes);

    for (recursiveProgramNode *node : mergeableNodes) {
      llvm::outs() << "Found mergeable node with parent function: "
                   << node->function.getName() << "\n";
      llvm::outs() << "Static argument values for this node:\n";
      for (auto &[attr, idx] : node->staticArgumentValues) {
        llvm::outs() << "  Arg index: " << idx << ", Value: " << attr << "\n";
      }
    }
    prettyPrintRecursiveProgramTree(root);
  }

  // llvm::outs() << "Checking dialect inliner interfaces:\n";
  // for (auto *dialect : getContext().getLoadedDialects()) {
  //   auto *inlinerInterface =
  //       dialect->getRegisteredInterface<DialectInlinerInterface>();
  //   llvm::outs() << "  " << dialect->getNamespace() << ": "
  //                << (inlinerInterface ? "has inliner" : "NO inliner") <<
  //                "\n";
  // }

  return false;
}

}  // namespace heir
}  // namespace mlir
