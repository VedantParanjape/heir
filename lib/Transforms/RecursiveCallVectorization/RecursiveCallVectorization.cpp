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
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Transforms/Inliner.h"         // from @llvm-project
#include "mlir/include/mlir/Transforms/InliningUtils.h"   // from @llvm-project

#define DEBUG_TYPE "recursive-call-vectorization"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_RECURSIVECALLVECTORIZATION
#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h.inc"

typedef struct recursiveProgramInfo_ {
  SmallVector<std::pair<Operation*, int>> baseConditions;
  SmallVector<std::pair<Operation*, int>> recursiveCalls;
  SmallVector<std::pair<Value*, int>> progressArguments;
  SmallVector<std::pair<Value, int>> staticArgumentValues;
  Operation* call;
} recursiveProgramInfo;
DenseMap<Operation*, recursiveProgramInfo> biscottiCalls;

static func::FuncOp getEnclosingFunction(Operation* op) {
  auto callOp = dyn_cast<func::CallOp>(op);
  if (!callOp)
    return nullptr;

  auto callee = callOp.getCalleeAttr();
  auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callOp, callee);

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

  llvm::outs() << " Progress Arguments:\n";
  for (auto pa : rpi->progressArguments) {
    llvm::outs() << "  Arg: " << *(pa.first) << " at index " << pa.second << "\n";
  }

  llvm::outs() << " Static Argument Values:\n";
  for (auto sa : rpi->staticArgumentValues) {
    llvm::outs() << "  Value: " << sa.first << " at index " << sa.second << "\n";
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

static void buildRecursiveAttributes(Block* block, Dialect* dialect) {
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
    
    func::FuncOp funcOp = getEnclosingFunction(op);
    if (!funcOp)
      continue;
    
    for (auto argOps: funcOp.getArguments()) {
      int attrValue;
      if (findBiscottiAttribute(argOps, "biscotti.progress_argument", attrValue)) {
        recursiveProgramInfo.progressArguments.push_back({&argOps, attrValue});
        recursiveProgramInfo.staticArgumentValues.push_back({op->getOperand(attrValue), attrValue});
      }
    }
  }

  for (auto &calls: biscottiCalls) {
    Operation *op = calls.first;
    recursiveProgramInfo &recursiveProgramInfo = calls.second;

    func::FuncOp funcOp = getEnclosingFunction(op);
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


bool tryVectorizeRecursiveBlock(Block* block, Dialect* dialect) {
  buildRecursiveAttributes(block, dialect);

  return false;
}

struct RecursiveCallVectorization
    : impl::RecursiveCallVectorizationBase<RecursiveCallVectorization> {
  using RecursiveCallVectorizationBase::RecursiveCallVectorizationBase;
  
  void runOnOperation() override {
    Dialect* mlirDialect = getContext().getLoadedDialect(dialect);

    getOperation()->walk<WalkOrder::PreOrder>([&](Block* block) {
      if (tryVectorizeRecursiveBlock(block, mlirDialect)) {
        sortTopologically(block);
      }
    });
  }
};

}  // namespace heir
}  // namespace mlir
