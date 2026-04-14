#ifndef COYOTE_CALLER_H
#define COYOTE_CALLER_H

#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h"
#include "lib/Transforms/RecursiveCallVectorization/RecursiveProgramInfo.h"
#include "mlir/include/mlir/Analysis/CallGraph.h"  // from @llvm-project

namespace mlir {
namespace heir {

void expandTensorShapeAcrossDefChainHelper(Value inputTensor,
                                           SmallVector<Value> &insertChain) {
  if (auto defOp = inputTensor.getDefiningOp()) {
    if (auto genericOp = dyn_cast<secret::GenericOp>(defOp)) {
      // value is a result of secret.generic
      auto resultIdx = cast<OpResult>(inputTensor).getResultNumber();
      // Find what was yielded at that index
      auto yieldOp =
          cast<secret::YieldOp>(genericOp.getBody()->getTerminator());
      Value yieldedVal = yieldOp->getOperand(resultIdx);
      insertChain.push_back(yieldedVal);
      expandTensorShapeAcrossDefChainHelper(yieldedVal, insertChain);
    } else if (auto constantOp = dyn_cast<arith::ConstantOp>(defOp)) {
      insertChain.push_back(constantOp.getResult());
      return;
    } else {
      if (auto insOp = dyn_cast<tensor::InsertOp>(defOp)) {
        insertChain.push_back(insOp.getResult());
        expandTensorShapeAcrossDefChainHelper(insOp.getDest(), insertChain);
      }
    }
  } else if (auto blockArg = dyn_cast<BlockArgument>(inputTensor)) {
    llvm::errs() << "Reached block argument during insert chain expansion. "
                    "Cannot expand further.\n";
    assert(1);
  }
}

// If a tensor is constructed by a series of tensor.insert ops
// starting from an empty tensor, this function walks the insertion
// chain, and expands to the required tensor shape.
//
// Works only if we want to expand <NxT> to <1xMxT> where M > N.
// Here T = type of the tensor element, and N is a known constant.
void expandTensorShapeAcrossDefChain(Value inputTensor,
                                     RankedTensorType targetType,
                                     OpBuilder &builder) {
  if (inputTensor.getType() == secret::SecretType::get(targetType) ||
      inputTensor.getType() == targetType)
    return;

  SmallVector<Value> insertChain;
  insertChain.push_back(inputTensor);
  expandTensorShapeAcrossDefChainHelper(inputTensor, insertChain);

  for (int i = insertChain.size() - 1; i >= 0; i--) {
    Value val = insertChain[i];
    if (val.getType() == secret::SecretType::get(targetType) ||
        val.getType() == targetType)
      continue;

    llvm::outs() << "Insert chain value: " << val << "\n";
    if (auto defOp = val.getDefiningOp()) {
      if (auto genericOp = dyn_cast<secret::GenericOp>(defOp)) {
        auto resultIdx = cast<OpResult>(val).getResultNumber();
        genericOp.getResult(resultIdx).setType(
            secret::SecretType::get(targetType));

        // Find what was yielded at that index
        auto yieldOp =
            cast<secret::YieldOp>(genericOp.getBody()->getTerminator());

        Value yieldedVal = yieldOp->getOperand(resultIdx);
      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(defOp)) {
        OpBuilder builder(constantOp);
        auto newConst =
            arith::ConstantOp::create(builder, constantOp.getLoc(), targetType,
                                      builder.getZeroAttr(targetType));
        constantOp.getResult().replaceUsesWithIf(
            newConst.getResult(), [&](OpOperand &use) {
              // Only replace use in the insert op that's part of this chain
              return use.getOwner() == insertChain[i - 1].getDefiningOp();
            });
      } else {
        if (auto insOp = dyn_cast<tensor::InsertOp>(defOp)) {
          insOp.getResult().setType(targetType);
          Value c0 =
              arith::ConstantIndexOp::create(builder, insOp->getLoc(), 0);
          insOp->insertOperands(2, {c0});
        }
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      llvm::errs() << "Reached block argument during insert chain expansion. "
                      "Cannot expand further.\n";
      assert(1);
    }
  }
}

void expandTensorShapeAcrossUseChain(Value inputTensor,
                                     RankedTensorType targetType,
                                     OpBuilder &builder) {
  if (inputTensor.getType() == secret::SecretType::get(targetType)) return;

  if (isa<secret::SecretType>(inputTensor.getType()))
    inputTensor.setType(secret::SecretType::get(targetType));
  else
    inputTensor.setType(targetType);

  for (auto &use : inputTensor.getUses()) {
    if (auto secretGenericOp = dyn_cast<secret::GenericOp>(use.getOwner())) {
      auto argIdx = use.getOperandNumber();
      expandTensorShapeAcrossUseChain(
          secretGenericOp.getBody()->getArgument(argIdx), targetType, builder);
    } else if (use.getOwner()->getDialect()->getNamespace() == "arith") {
      for (auto result : use.getOwner()->getResults()) {
        expandTensorShapeAcrossUseChain(result, targetType, builder);
      }
    } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(use.getOwner())) {
      if (extractOp->getNumOperands() == 3) continue;
      auto c0 = arith::ConstantIndexOp::create(builder, extractOp.getLoc(), 0);
      extractOp->insertOperands(1, {c0});
    }
  }
}

void runCoyoteVectorizer(func::FuncOp func) { coyoteVectorizer(func); }

void findVectorizationCandidates(
    recursiveProgramNode *node,
    SmallVector<recursiveProgramNode *> &vectorizationCandidates) {
  if (!node) return;

  for (recursiveProgramNode *child : node->children)
    findVectorizationCandidates(child, vectorizationCandidates);

  if (node->children.empty()) vectorizationCandidates.push_back(node);
}

void processVectorizationCandidates(recursiveProgramNode *root) {
  SmallVector<recursiveProgramNode *> vectorizationCandidates;
  findVectorizationCandidates(root, vectorizationCandidates);

  for (recursiveProgramNode *candidate : vectorizationCandidates) {
    SmallVector<Type> oldArgTypes;
    SmallVector<Type> oldResultTypes;
    for (auto arg : candidate->function.getArguments())
      oldArgTypes.push_back(arg.getType());
    for (auto result : candidate->function.getFunctionType().getResults())
      oldResultTypes.push_back(result);

    // oldArgTypes.push_back(candidate->function.getArgument(0).getType());
    // oldResultTypes.push_back(candidate->function.getFunctionType().getResult(0));
    // auto tensorType = mlir::RankedTensorType::get({1, 8},
    // IntegerType::get(candidate->function.getContext(), 32));

    runCoyoteVectorizer(candidate->function);
    auto funcOp = candidate->caller->getParentOfType<func::FuncOp>();
    OpBuilder builder(&funcOp.getBody().front(),
                      funcOp.getBody().front().begin());
    llvm::outs() << "Processing vectorization candidate: "
                 << candidate->function.getName() << "\n";
    for (int i = 0; i < oldArgTypes.size(); i++) {
      llvm::outs() << "Checking argument " << i << " of type " << oldArgTypes[i]
                   << "\n";
      llvm::outs() << "Candidate function argument type: "
                   << candidate->function.getArgument(i).getType() << "\n";
      if (isa<secret::SecretType>(oldArgTypes[i]) &&
          oldArgTypes[i] != candidate->function.getArgument(i).getType()) {
        llvm::outs() << "Expanding tensor shape of argument " << i << "\n";
        auto tensorType = mlir::cast<RankedTensorType>(
            mlir::cast<secret::SecretType>(
                candidate->function.getArgument(i).getType())
                .getValueType());
        expandTensorShapeAcrossDefChain(candidate->caller.getArgOperands()[i],
                                        tensorType, builder);
      }

      for (int i = 0; i < oldResultTypes.size(); i++) {
        if (isa<secret::SecretType>(oldResultTypes[i]) &&
            oldResultTypes[i] !=
                candidate->function.getFunctionType().getResult(i)) {
          llvm::outs() << "Expanding tensor shape of result " << i << "\n";
          auto tensorType = mlir::cast<RankedTensorType>(
              mlir::cast<secret::SecretType>(
                  candidate->function.getFunctionType().getResult(i))
                  .getValueType());
          expandTensorShapeAcrossUseChain(candidate->caller.getResults()[i],
                                          tensorType, builder);
        }
      }
    }
  }

  // vectorizationCandidates[0]->caller->getParentOfType<func::FuncOp>().dump();
  // assert(0);
}

}  // namespace heir
}  // namespace mlir

#endif  // COYOTE_CALLER_H
