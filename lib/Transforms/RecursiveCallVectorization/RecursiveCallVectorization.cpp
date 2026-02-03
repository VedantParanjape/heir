#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h"

#include <cassert>
#include <cstdint>

#include "lib/Utils/Graph/Graph.h"
#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Analysis/CallGraph.h"        // from @llvm-project
#include "mlir/include/mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlow.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Transforms/Inliner.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/RegionUtils.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/InliningUtils.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h" // from @llvm-project

#define DEBUG_TYPE "recursive-call-vectorization"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RECURSIVECALLVECTORIZATION
#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h.inc"

typedef struct recursiveProgramInfo_ {
  SmallVector<std::pair<Operation*, int>> baseConditions;
  SmallVector<std::pair<Operation*, int>> recursiveCalls;
  SmallVector<std::pair<Value*, int>> progressArguments;
  SmallVector<std::pair<Operation*, int>> staticArgumentValues;
  Operation* call;
} recursiveProgramInfo;
DenseMap<Operation*, recursiveProgramInfo> biscottiCalls;

typedef struct recursiveProgramNode_ {
  func::FuncOp parent;
  SmallVector<std::pair<Operation*, int>> staticArgumentValues;
  SmallVector<recursiveProgramNode_*> children;
} recursiveProgramNode;
std::queue<std::pair<Operation*, recursiveProgramNode*>> workQueue;

static func::FuncOp getEnclosingFunction(Operation* op, ModuleOp &module) {
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
  FailureOr<Attribute> attr =
      findAttributeAssociatedWith(op, attrName);

  if (succeeded(attr)) {
    llvm::outs() << "Operand: " << op << "\n"
                 << " Found attribute: " << attrName 
                 << ": " << attr.value() << "\n";

    if (auto intAttr = dyn_cast<IntegerAttr>(attr.value()))
      outValue = intAttr.getInt();

    return true;
  }

  outValue = -1;
  return false;
}

static bool findBiscottiAttributeOnOps(Operation* op, StringRef attrName, int &outValue) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName)) {
    int outValue = attr.getInt();
    llvm::outs() << "Operation: " << *op << "\n"
                 << " Found attribute on Ops: " << attrName 
                 << ": " << outValue << "\n";

    return true;
  }

  outValue = -1;
  return false;
}

static void printRecursiveAttributes(recursiveProgramInfo *rpi) {
  llvm::outs() << "Recursive Call Info for call: " << *(rpi->call) << "\n";

// TODO: make sure the argument numbers match up with static values.
  llvm::outs() << " Progress Arguments:\n";
  for (auto pa : rpi->progressArguments) {
    llvm::outs() << "  Arg: " << *(pa.first) << " at index " << pa.second << "\n";
  }

  llvm::outs() << " Static Argument Values:\n";
  for (auto sa : rpi->staticArgumentValues) {
    llvm::outs() << "  Value: " << *(sa.first) << " at index " << sa.second << "\n";
  }

  llvm::outs() << " Recursive Calls:\n";
  for (auto rc : rpi->recursiveCalls) {
    llvm::outs() << "  Call: " << *(rc.first) << " at index " << rc.second << "\n";
  }

  llvm::outs() << " Base Conditions:\n";
  for (auto bc : rpi->baseConditions) {
    llvm::outs() << "  Op: " << *(bc.first) << " at index " << bc.second << "\n";
  }
}

static void indent(unsigned level) {
  for (unsigned i = 0; i < level; ++i)
    llvm::outs() << "  ";
}

// TODO: Rework this function (mainly clean it up)
static void prettyPrintRecursiveProgramTree(
  recursiveProgramNode *node,
  unsigned indentLevel = 0) {

  if (!node)
    return;

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
      if (!first)
        llvm::outs() << ", ";
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
}

static Operation* insertConstantAtTop(func::FuncOp &funcOp, Operation *ConstantOp) {
  Block &entryBlock = funcOp.front();
  OpBuilder builder(&entryBlock, entryBlock.begin());

  Operation *clonedConstantOp = builder.clone(*ConstantOp);
  return clonedConstantOp;
}

struct RecursiveCallVectorization
    : impl::RecursiveCallVectorizationBase<RecursiveCallVectorization> {
  using RecursiveCallVectorizationBase::RecursiveCallVectorizationBase;

  void foldAllOpsInFunc(func::FuncOp &funcOp, MLIRContext *ctx);
  void buildRecursiveAttributes(Block* block, Dialect* dialect);
  void buildRecursiveCallTree(Operation *op, recursiveProgramInfo &recursiveProgramInfo);
  bool tryVectorizeRecursiveBlock(Block* block, Dialect* dialect);

  void runOnOperation() override {
    Dialect* mlirDialect = getContext().getLoadedDialect(dialect);

    getOperation()->walk<WalkOrder::PreOrder>([&](Block* block) {
      if (tryVectorizeRecursiveBlock(block, mlirDialect)) {
        sortTopologically(block);
      }
    });
  }
};

void RecursiveCallVectorization::foldAllOpsInFunc(func::FuncOp &funcOp, MLIRContext *ctx) {
    RewritePatternSet patterns(ctx);
    // for (auto *dialect : ctx->getLoadedDialects())
    //   llvm::outs() << dialect->getNamespace() << "\n";

    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, ctx);

    // fold constants and apply canonicalization patterns
    GreedyRewriteConfig config;
    // Makes compilation faster, but may miss some patterns.
    config.setUseTopDownTraversal();
    (void)applyPatternsGreedily(funcOp, std::move(patterns), config);

    // Call DCE for the simplification
    IRRewriter rewriter(funcOp.getContext());
    (void)mlir::eraseUnreachableBlocks(rewriter, funcOp.getOperation()->getRegions());
}

void RecursiveCallVectorization::buildRecursiveAttributes(Block* block, Dialect* dialect) {
  for (auto &op : block->getOperations()) {
    if (dialect && op.getDialect() != dialect)
      continue;

    if (!isa<func::CallOp>(op))
      continue;

    int attrValue;
    if (findBiscottiAttribute(op.getResult(0), "biscotti.call", attrValue) && biscottiCalls.find(&op) == biscottiCalls.end()) {
      recursiveProgramInfo call;
      call.call = &op;
      biscottiCalls[&op] = call;
    }
  }

  for (auto &calls: biscottiCalls) {
    Operation* op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();    
    func::FuncOp funcOp = getEnclosingFunction(op, parentModule);
    if (!funcOp)
      continue;
    
    for (auto argOps: funcOp.getArguments()) {
      int attrValue;
      if (findBiscottiAttribute(argOps, "biscotti.progress_argument", attrValue)) {
        recursiveProgramInfo.progressArguments.push_back({&argOps, attrValue});
        recursiveProgramInfo.staticArgumentValues.push_back({op->getOperand(attrValue).getDefiningOp(), attrValue});
      }
    }
  }

  for (auto &calls: biscottiCalls) {
    Operation *op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();    
    func::FuncOp funcOp = getEnclosingFunction(op, parentModule);
    if (!funcOp)
      continue;
    
    funcOp.walk([&](Operation *calledOp) {
      auto call = dyn_cast<func::CallOp>(calledOp);
      if (!call)
        return;

      int attrValue;
      if (findBiscottiAttribute(call->getResult(0), "biscotti.recursive_call", attrValue)) {
        recursiveProgramInfo.recursiveCalls.push_back({calledOp, attrValue});
      }
    });

    funcOp.walk([&](Operation *baseOp) {
      int attrValue;
      if (findBiscottiAttributeOnOps(baseOp, "biscotti.base_condition", attrValue)) {
        recursiveProgramInfo.baseConditions.push_back({baseOp, attrValue});
      }
    });
  }

  for (auto &calls: biscottiCalls) {
    printRecursiveAttributes(&calls.second);
  }
}

void RecursiveCallVectorization::buildRecursiveCallTree(Operation *rootOp, recursiveProgramInfo &recursiveProgramInfo) {
  recursiveProgramNode *root = new recursiveProgramNode();
  root->staticArgumentValues = recursiveProgramInfo.staticArgumentValues;
  workQueue.push({rootOp, root});

  while (!workQueue.empty()) {
    llvm::outs() << "== Processing node in recursive call tree...\n";
    // Pop a node to be processed.
    Operation *op = workQueue.front().first;
    recursiveProgramNode *currentNode = workQueue.front().second;
    workQueue.pop();
    llvm::outs() << *op << "\n";
    
    // Essentially we have a recursive function callOp here.
    ModuleOp parentModule = rootOp->getParentOfType<ModuleOp>();
    func::FuncOp funcOp = getEnclosingFunction(op, parentModule);
    if (!funcOp) {
      llvm::outs() << "Error: Could not find enclosing function for operation.\n";
      return;
    }

    // llvm::outs() << "== Processing recursive function: " << funcOp.getName() << "\n";
    // Clone the funcOp to specialise it.
    func::FuncOp funcOpCloned = funcOp.clone();
    // funcOpCloned.setName(funcOp.getName().str() + "_clone");

    // Insert the constant arguments into the cloned function.
    // Then replace uses of the arguments with these constants.
    for (auto knownValue: currentNode->staticArgumentValues) {
      Operation *newConstant = insertConstantAtTop(funcOpCloned, knownValue.first);
      funcOpCloned.getArgument(knownValue.second).replaceAllUsesWith(newConstant->getResult(0));
    }

    // Perform Constant Op propagation + Op folding + DCE.
    foldAllOpsInFunc(funcOpCloned, funcOp.getContext());
    // funcOpCloned.dump();

    currentNode->parent = funcOpCloned;

    // analyse the cloned function for further recursive calls.
    // find static argument values for each recursive call.
    // add the recursive calls as children to the current node and add to process queue.
    funcOpCloned.walk([&](Operation *calledOp) {
      auto call = dyn_cast<func::CallOp>(calledOp);
      if (!call)
        return;

      int attrValue;
      if (findBiscottiAttribute(call->getResult(0), "biscotti.recursive_call", attrValue)) {
        recursiveProgramNode *childNode = new recursiveProgramNode();
        
        for (auto progressArg : recursiveProgramInfo.progressArguments) {
          int attrValue = progressArg.second;
          // TODO: Add a check calledOp->getOperand(attrValue).getDefiningOp() is actually a constant.
          childNode->staticArgumentValues.push_back({calledOp->getOperand(attrValue).getDefiningOp(), attrValue});
        }
        currentNode->children.push_back(childNode);
        workQueue.push({calledOp, childNode});
      }
    });
  }

  prettyPrintRecursiveProgramTree(root);
}

bool RecursiveCallVectorization::tryVectorizeRecursiveBlock(Block* block, Dialect* dialect) {
  buildRecursiveAttributes(block, dialect);
  for (auto &calls: biscottiCalls) {
    Operation *op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    buildRecursiveCallTree(op, recursiveProgramInfo);
  }

  return false;
}

}  // namespace heir
}  // namespace mlir
