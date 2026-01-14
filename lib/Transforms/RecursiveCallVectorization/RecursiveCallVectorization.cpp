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

static void findBiscottiAttributes(Value operand, StringRef attrName) {
  FailureOr<Attribute> attr =
      findAttributeAssociatedWith(operand, attrName);

  if (succeeded(attr)) {
    llvm::outs() << "Operand: " << operand << "\n";
    llvm::outs() << "Found attribute: " << attrName << ": " << attr.value() << "\n";

    if (auto intAttr = dyn_cast<IntegerAttr>(attr.value())) {
      int64_t value = intAttr.getInt();
    }
  } else {
    // llvm::outs() << "No attribute found\n";
  }
}

bool tryVectorizeRecursiveBlock(Block* block, Dialect* dialect) {
  llvm::outs() << "Analyzing block for recursive functions...\n";
  graph::Graph<Operation*> graph;
  for (auto& op : block->getOperations()) {
    // if (!op.hasTrait<OpTrait::Elementwise>()) {
    //   continue;
    // }

    if (dialect && op.getDialect() != dialect) {
      continue;
    }
  
    for (auto operand : op.getOperands()) {
      findBiscottiAttributes(operand, "biscotti.call");
      findBiscottiAttributes(operand, "biscotti.progress_args");
      findBiscottiAttributes(operand, "biscotti.base_condition");
      findBiscottiAttributes(operand, "biscotti.recursive_call");
    }

    // for (auto result : op.getResults()) {
    //   FailureOr<Attribute> attr =
    //       findAttributeAssociatedWith(result, "my_dialect.my_attr");
    //   if (succeeded(attr)) {
    //   }
    // }
  }

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
