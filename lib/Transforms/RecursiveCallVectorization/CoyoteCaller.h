#ifndef COYOTE_CALLER_H
#define COYOTE_CALLER_H

#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h"
#include "lib/Transforms/RecursiveCallVectorization/RecursiveProgramInfo.h"

namespace mlir {
namespace heir {

void expandTensorShapeAcrossInsertChainHelper(Value inputTensor,
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
      expandTensorShapeAcrossInsertChainHelper(yieldedVal, insertChain);
    } else if (auto constantOp = dyn_cast<arith::ConstantOp>(defOp)) {
      insertChain.push_back(constantOp.getResult());
      return;
    } else {
      if (auto insOp = dyn_cast<tensor::InsertOp>(defOp)) {
        insertChain.push_back(insOp.getResult());
        expandTensorShapeAcrossInsertChainHelper(insOp.getDest(), insertChain);
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
void expandTensorShapeAcrossInsertChain(Value inputTensor,
                                        RankedTensorType targetType,
                                        OpBuilder &builder) {
  if (inputTensor.getType() == targetType) return;

  SmallVector<Value> insertChain;
  insertChain.push_back(inputTensor);
  expandTensorShapeAcrossInsertChainHelper(inputTensor, insertChain);

  for (int i = insertChain.size() - 1; i >= 0; i--) {
    if (insertChain[i].getType() == targetType) continue;

    Value val = insertChain[i];
    llvm::outs() << "Insert chain value: " << val << "\n";
    if (auto defOp = val.getDefiningOp()) {
      if (auto genericOp = dyn_cast<secret::GenericOp>(defOp)) {
        auto resultIdx = cast<OpResult>(inputTensor).getResultNumber();
        genericOp.getResult(resultIdx).setType(
            secret::SecretType::get(targetType));

        // Find what was yielded at that index
        auto yieldOp =
            cast<secret::YieldOp>(genericOp.getBody()->getTerminator());

        Value yieldedVal = yieldOp->getOperand(resultIdx);
      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(defOp)) {
        OpBuilder builder(constantOp);
        auto newConst = builder.create<arith::ConstantOp>(
            constantOp.getLoc(), targetType, builder.getZeroAttr(targetType));
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
    } else if (auto blockArg = dyn_cast<BlockArgument>(inputTensor)) {
      llvm::errs() << "Reached block argument during insert chain expansion. "
                      "Cannot expand further.\n";
      assert(1);
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
    for (auto arg : candidate->function.getArguments())
      oldArgTypes.push_back(arg.getType());

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
        expandTensorShapeAcrossInsertChain(
            candidate->caller.getArgOperands()[i], tensorType, builder);
      }
    }
  }
}

}  // namespace heir
}  // namespace mlir

#endif  // COYOTE_CALLER_H
