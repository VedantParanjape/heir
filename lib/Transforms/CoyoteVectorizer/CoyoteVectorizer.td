//===- CoyoteVectorizer.td - Coyote Vectorization Pass ----*- tablegen -*-===//
//
// This file defines the Coyote vectorization pass based on the ASPLOS '23 paper:
// "Coyote: A Compiler for Vectorizing Encrypted Arithmetic Circuits"
//
// The pass implements rotation-aware vectorization for FHE circuits, using:
// 1. Simulated annealing for lane placement
// 2. Best-first search for circuit quotient optimization
// 3. ILP-based instruction alignment
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_TD_
#define LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_TD_

include "mlir/Pass/PassBase.td"

def CoyoteVectorizer : Pass<"coyote-vectorize", "mlir::func::FuncOp"> {
  let summary = "Vectorize FHE circuits using the Coyote algorithm";
  let description = [{
    This pass implements the Coyote vectorization algorithm from ASPLOS '23,
    which performs rotation-aware vectorization of encrypted arithmetic circuits.

    The key insight is that vectorization and data layout must be co-optimized
    because rotations (the only way to move data between vector lanes in FHE)
    are extremely expensive. The pass:

    1. Constructs a circuit quotient graph by identifying subcircuits that
       should be computed in scalar on a single lane to avoid excessive rotations

    2. Uses simulated annealing to find optimal lane placements that minimize
       the number of distinct rotation amounts needed at each epoch

    3. Aligns isomorphic instructions at each epoch using ILP to create
       efficient vector schedules

    4. Generates blend and rotation instructions to realize the data movement

    The pass operates on Secret dialect operations and lowers to tensor_ext
    operations with appropriate rotate and extract operations.
  }];

  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];

  let options = [
    Option<"searchRounds", "search-rounds", "unsigned", /*default=*/"200",
           "Number of rounds for circuit quotient search">,
    Option<"annealingRounds", "annealing-rounds", "unsigned", /*default=*/"20000",
           "Number of simulated annealing rounds for lane placement">,
    Option<"initialTemp", "initial-temp", "double", /*default=*/"50.0",
           "Initial temperature for simulated annealing">,
    Option<"coolingRate", "cooling-rate", "double", /*default=*/"0.001",
           "Cooling rate (beta) for simulated annealing">,
    Option<"rotationCost", "rotation-cost", "double", /*default=*/"1.0",
           "Cost weight for rotations (multiplies and rotates)">,
    Option<"addCost", "add-cost", "double", /*default=*/"0.1",
           "Cost weight for additions/subtractions">,
    Option<"vectorWidth", "vector-width", "unsigned", /*default=*/"8192",
           "Target vector width (ciphertext slots)">
  ];
}

#endif // LIB_TRANSFORMS_COYOTEVECTORIZER_COYOTEVECTORIZER_TD_
