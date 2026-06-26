#ifndef UTILS_H
#define UTILS_H

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Dialect/Utils.h"
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

namespace mlir {
namespace heir {

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

inline void foldAllOpsInFunc(func::FuncOp &funcOp, MLIRContext *ctx) {
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

inline func::FuncOp getEnclosingFunction(Operation *op, ModuleOp &module) {
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

/// Redirect \p call to a freshly-created private dummy function whose
/// signature matches the call op's *current* operand/result types, so the
/// verifier accepts the call even after the original callee's signature has
/// drifted (e.g. the callee was widened from `tensor<16xi32>` to
/// `tensor<1x64xi32>` by CoyoteVectorizer).
///
/// The dummy body is a single-block trivial pass-through: for each result we
/// prefer to forward an argument of the same type; if none matches, we emit
/// an `arith.constant` zero of the result type. Bails if any result type
/// cannot be synthesized that way (e.g. non-numeric, non-passthrough types).
///
/// The original callee is left untouched — this only inserts a new function
/// and rewrites the call's `callee` attribute to point at it.
///
/// Use case: after CoyoteVectorizer widens `@kernel_clone_0_0` to a SIMD
/// signature, the test-scaffolding `main` still calls it with the pre-widen
/// types. Redirecting the call to a stub keeps `main` verifier-clean while
/// leaving the real vectorized kernel intact.
static LogicalResult redirectCallToDummy(func::CallOp call) {
  auto module = call->getParentOfType<ModuleOp>();
  if (!module) return failure();

  // Compose signature from the call op itself (not from the current callee,
  // which may have drifted).
  SmallVector<Type> argTypes(call.getOperandTypes());
  SmallVector<Type> resTypes(call.getResultTypes());

  // Plan result materialization first, so we can bail before mutating.
  SmallVector<int64_t> plan;  // -1 = constant zero, >=0 = argument index
  for (Type rt : resTypes) {
    int64_t pick = -1;
    for (unsigned i = 0; i < argTypes.size(); ++i) {
      if (argTypes[i] == rt) {
        pick = (int64_t)i;
        break;
      }
    }
    if (pick < 0) {
      Type elt = rt;
      if (auto rtt = dyn_cast<RankedTensorType>(rt)) elt = rtt.getElementType();
      if (!isa<IntegerType, FloatType, IndexType>(elt)) return failure();
    }
    plan.push_back(pick);
  }

  // Synthesize a unique symbol name based on the callee.
  std::string base = (call.getCallee() + "_dummy").str();
  std::string sym = base;
  unsigned counter = 0;
  while (module.lookupSymbol(sym)) sym = base + "_" + std::to_string(counter++);

  OpBuilder b(module.getContext());
  b.setInsertionPointToEnd(module.getBody());
  auto fnType = FunctionType::get(module.getContext(), argTypes, resTypes);
  auto fn = func::FuncOp::create(b, call.getLoc(), sym, fnType);
  fn.setPrivate();

  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);

  SmallVector<Value> retVals;
  for (auto [rt, pick] : llvm::zip(resTypes, plan)) {
    if (pick >= 0) {
      retVals.push_back(entry->getArgument(pick));
      continue;
    }
    Attribute zeroAttr;
    if (auto rtt = dyn_cast<RankedTensorType>(rt)) {
      Type elt = rtt.getElementType();
      Attribute eltZero = isa<FloatType>(elt)
                              ? (Attribute)b.getFloatAttr(elt, 0.0)
                              : (Attribute)b.getIntegerAttr(elt, 0);
      zeroAttr = DenseElementsAttr::get(rtt, eltZero);
    } else if (isa<FloatType>(rt)) {
      zeroAttr = b.getFloatAttr(rt, 0.0);
    } else {
      zeroAttr = b.getIntegerAttr(rt, 0);
    }
    auto cst =
        arith::ConstantOp::create(b, call.getLoc(), cast<TypedAttr>(zeroAttr));
    retVals.push_back(cst.getResult());
  }
  func::ReturnOp::create(b, call.getLoc(), retVals);

  // Redirect the call to the new dummy.
  call.setCalleeAttr(SymbolRefAttr::get(b.getContext(), sym));
  return success();
}

}  // namespace heir
}  // namespace mlir

#endif
