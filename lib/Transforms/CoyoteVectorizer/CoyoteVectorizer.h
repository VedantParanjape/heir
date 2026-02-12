//===- CoyoteVectorizer.h - Coyote Vectorization Pass -----------*- C++ -*-===//
//
// Header file for the Coyote vectorization pass based on ASPLOS '23 paper.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_H_
#define LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_H_
