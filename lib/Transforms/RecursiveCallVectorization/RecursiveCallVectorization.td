#ifndef LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_TD_
#define LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_TD_

include "mlir/Pass/PassBase.td"

def RecursiveCallVectorization : Pass<"recursive-call-vectorization", "ModuleOp"> {
  let summary = "A vectorizer for bounded recursive programs.";
  let description = [{
  This pass resolves a bounded recursive call tree and vectorizes the tree levels that
  satisfy the vectorization criteria.
  }];

  let dependentDialects = [
    "tensor::TensorDialect",
  ];

  let options = [
    Option<"dialect", "dialect", "std::string", "", "Use this to restrict the "
      "dialect whose ops should be vectorized.">,
    Option<"nodeSizeThreshold", "node-size-threshold", "int64_t", "-1",
      "Use this to restrict the size of the node function that can be vectorized. "
      "If the node function size is greater than this threshold, it will not be "
      "vectorized. A value of -1 means no restriction on the node function size.">
  ];
}

#endif  // LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_TD_
