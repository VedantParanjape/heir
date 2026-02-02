#ifndef LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_TD_
#define LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_TD_

include "mlir/Pass/PassBase.td"

def RecursiveCallVectorization : Pass<"recursive-call-vectorization", "ModuleOp"> {
  let summary = "A vectorizer for bounded recursive programs.";
  let description = [{
  This pass resolves a bounded recursive call tree and vectorizes the tree levels that
  satistfy the vectorization criteria.
  }];

  let dependentDialects = [
    "tensor::TensorDialect",
  ];

  let options = [
    Option<"dialect", "dialect", "std::string", "", "Use this to restrict the "
      "dialect whose ops should be vectorized.">,
  ];
}

#endif  // LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_TD_
