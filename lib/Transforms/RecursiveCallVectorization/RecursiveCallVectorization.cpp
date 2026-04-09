#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h"

#include <cassert>
#include <cstdint>

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
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
#include "mlir/include/mlir/Pass/PassManager.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Inliner.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/RegionUtils.h"    // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "recursive-call-vectorization"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RECURSIVECALLVECTORIZATION
#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h.inc"

typedef struct recursiveProgramInfo_ {
  SmallVector<std::pair<Operation *, int>> baseConditions;
  SmallVector<std::pair<Operation *, int>> recursiveCalls;
  SmallVector<std::pair<Value *, int>> progressArguments;
  SmallVector<std::pair<Operation *, int>> staticArgumentValues;
  Operation *call;
} recursiveProgramInfo;
DenseMap<Operation *, recursiveProgramInfo> biscottiCalls;

typedef struct recursiveProgramNode_ {
  func::FuncOp parent;
  SmallVector<std::pair<Operation *, int>> staticArgumentValues;
  SmallVector<recursiveProgramNode_ *> children;
} recursiveProgramNode;

DenseSet<func::FuncOp> functionDeleteList;

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
    llvm::outs() << "Operand: " << op << "\n"
                 << " Found attribute: " << attrName << ": " << attr.value()
                 << "\n";

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
    llvm::outs() << "Operation: " << *op << "\n"
                 << " Found attribute on Ops: " << attrName << ": " << outValue
                 << "\n";

    return true;
  }

  outValue = -1;
  return false;
}

static bool findBiscottiArrayAttribute(Value op, StringRef attrName,
                                       SmallVector<int64_t> &outValue) {
  FailureOr<Attribute> attr = findAttributeAssociatedWith(op, attrName);
  if (succeeded(attr)) {
    llvm::outs() << "Operand: " << op << "\n"
                 << " Found Array in attributes: " << attrName << ": "
                 << attr.value() << "\n";
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
    llvm::outs() << "  Value: " << *(sa.first) << " at index " << sa.second
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
  if (node->parent) {
    llvm::outs() << "func @" << node->parent.getSymName();
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
        llvm::outs() << *op;
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
  //   node->parent.dump();
  // }
}

static Operation *insertConstantAtTop(func::FuncOp &funcOp,
                                      Operation *ConstantOp) {
  Block &entryBlock = funcOp.front();
  OpBuilder builder(&entryBlock, entryBlock.begin());

  Operation *clonedConstantOp = builder.clone(*ConstantOp);
  return clonedConstantOp;
}

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
  bool tryVectorizeRecursiveBlock(Block *block, Dialect *dialect);
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<secret::SecretDialect>();
    registry.addExtension(
        +[](MLIRContext *ctx, secret::SecretDialect *dialect) {
          dialect->addInterfaces<SecretInlinerInterface>();
        });
  }

  void runOnOperation() override {
    Dialect *mlirDialect = getContext().getLoadedDialect(dialect);

    getOperation()->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      if (funcOp.empty()) return;
      if (tryVectorizeRecursiveBlock(&funcOp.getBlocks().front(),
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

    removeRedundantTensorCasts(getOperation());
    unrollTensorGenerates(getOperation());

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
  node->parent.walk([&](secret::GenericOp genOp) {
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
  for (recursiveProgramNode *node : mergeableNodes) {
    int nodeSize = countNodeFunctionSize(node);
    llvm::outs() << "Merging node with function: " << node->parent.getName()
                 << "\n";

    for (recursiveProgramNode *child : node->children) {
      int childSize = countNodeFunctionSize(child);
      nodeSize += childSize;
      llvm::outs() << "   Child node with function: " << child->parent.getName()
                   << ", size: " << childSize << "\n";
    }

    llvm::outs() << "Node function size: " << nodeSize << "\n";
    // TODO: Tune this threshold.
    if (nodeSize > 100) {
      llvm::outs() << "   Skipping merge due to large node size.\n";
      continue;
    }

    for (recursiveProgramNode *child : node->children) {
      ModuleOp parentModule = node->parent->getParentOfType<ModuleOp>();
      if (!parentModule)
        llvm::errs() << "Error: Parent module not found for function "
                     << node->parent.getName() << "\n";

      auto &ChildFunction = child->parent;
      // Collect call ops first, only within node->parent
      SmallVector<func::CallOp> callsToInline;
      node->parent.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == ChildFunction.getName()) {
          callsToInline.push_back(callOp);
        }
      });

      // Now inline them
      InlinerInterface interface(&getContext());
      InlinerConfig config;
      for (auto callOp : callsToInline) {
        llvm::outs() << "Inlining " << ChildFunction.getName() << " into "
                     << node->parent.getName() << "\n";
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
    // llvm::outs() << "Trying to fold extract from from_elements at: " <<
    // extractOp.getLoc() << "\n";
    auto tensorType =
        mlir::dyn_cast<RankedTensorType>(extractOp.getTensor().getType());
    if (!tensorType || !tensorType.hasStaticShape() ||
        tensorType.getNumElements() != 1)
      return failure();

    // llvm::outs() << "Tensor type: " << tensorType << "\n";
    // Check source is tensor.from_elements with single element
    auto fromElements =
        extractOp.getTensor().getDefiningOp<tensor::FromElementsOp>();
    if (!fromElements || fromElements.getElements().size() != 1)
      return failure();

    // llvm::outs() << "Folding extract from from_elements at: " <<
    // extractOp.getLoc() << "\n";
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
        recursiveProgramInfo.staticArgumentValues.push_back(
            {op->getOperand(attrValue).getDefiningOp(), attrValue});
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
  workQueue.push({rootOp, root});
  int recursiveCallCounter = 0;

  // Essentially we have a recursive function callOp here.
  ModuleOp parentModule = rootOp->getParentOfType<ModuleOp>();
  func::FuncOp funcOp = getEnclosingFunction(rootOp, parentModule);
  if (!funcOp) {
    llvm::outs() << "Error: Could not find enclosing function for operation.\n";
    return;
  }

  while (!workQueue.empty()) {
    llvm::outs() << "== Processing node in recursive call tree...\n";
    // Pop a node to be processed.
    Operation *op = workQueue.front().first;
    recursiveProgramNode *currentNode = workQueue.front().second;
    workQueue.pop();
    llvm::outs() << *op << "\n";

    // llvm::outs() << "== Processing recursive function: " << funcOp.getName()
    // << "\n"; Clone the funcOp to specialise it.
    func::FuncOp funcOpCloned = funcOp.clone();
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

    currentNode->parent = funcOpCloned;
    parentModule.push_back(funcOpCloned);
    dyn_cast<func::CallOp>(op).setCallee(funcOpCloned.getName());
    functionDeleteList.insert(funcOp);

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
          // TODO: Add a check calledOp->getOperand(attrValue).getDefiningOp()
          // is actually a constant.
          childNode->staticArgumentValues.push_back(
              {calledOp->getOperand(attrValue).getDefiningOp(), attrValue});
        }
        currentNode->children.push_back(childNode);
        workQueue.push({calledOp, childNode});
      }
    });
  }
  prettyPrintRecursiveProgramTree(root);

  SmallVector<recursiveProgramNode *> mergeableNodes;
  findMergeableRecursiveCallNodes(root, mergeableNodes);
  mergeRecursiveCallNodes(mergeableNodes);

  for (recursiveProgramNode *node : mergeableNodes) {
    llvm::outs() << "Found mergeable node with parent function: "
                 << node->parent.getName() << "\n";
    llvm::outs() << "Static argument values for this node:\n";
    for (auto &[op, idx] : node->staticArgumentValues) {
      llvm::outs() << "  Arg index: " << idx << ", Value: " << *op << "\n";
    }
  }
}

bool RecursiveCallVectorization::tryVectorizeRecursiveBlock(Block *block,
                                                            Dialect *dialect) {
  if (auto funcOp = dyn_cast<func::FuncOp>(block->getParentOp()))
    if (funcOp.getName().contains("clone")) return false;

  llvm::outs() << "Analyzing block for recursive call vectorization: \n";
  block->dump();

  buildRecursiveAttributes(block, dialect);
  for (auto &calls : biscottiCalls) {
    Operation *op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    buildRecursiveCallTree(op, recursiveProgramInfo);
    functionCounter++;
  }

  llvm::outs() << "Checking dialect inliner interfaces:\n";
  for (auto *dialect : getContext().getLoadedDialects()) {
    auto *inlinerInterface =
        dialect->getRegisteredInterface<DialectInlinerInterface>();
    llvm::outs() << "  " << dialect->getNamespace() << ": "
                 << (inlinerInterface ? "has inliner" : "NO inliner") << "\n";
  }

  return false;
}

}  // namespace heir
}  // namespace mlir
