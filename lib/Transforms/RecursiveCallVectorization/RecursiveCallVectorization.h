#ifndef LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_H_
#define LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/RecursiveCallVectorization/RecursiveCallVectorization.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_RECURSIVE_CALL_VECTORIZATION_RECURSIVE_CALL_VECTORIZATION_H_
