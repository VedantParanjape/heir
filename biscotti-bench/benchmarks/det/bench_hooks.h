// Workload-specific hooks consumed by the harness emitted from
// `heir-translate --emit-openfhe-pke-harness`. Determinant variant:
//   - `SIZE` == N (matrix dimension), baked in per benchmark via
//     -DBENCH_HOOKS_SIZE=<N>. The flattened input is N*N elements.
//   - Single-input workload: the kernel's several ciphertext args are all
//     packings of the SAME N*N matrix. The harness fills args from A and B
//     (first ceil(N/2), then floor(N/2)); we return the SAME matrix for
//     both (mirror trick) so every arg sees one coherent matrix.
//   - `reference` returns a length-1 vector holding det(matrix).
//
// Required members of the `bench_hooks` namespace:
//   constexpr int SIZE;
//   std::vector<int32_t> gen_input_A(int size, uint32_t seed);
//   std::vector<int32_t> gen_input_B(int size, uint32_t seed);
//   std::vector<int32_t> reference(const std::vector<int32_t>& A,
//                                  const std::vector<int32_t>& B);

#pragma once

#include <cstdint>
#include <vector>

#include "reference.h"

namespace bench_hooks {

// Matrix dimension N. Override per benchmark via -DBENCH_HOOKS_SIZE.
#ifndef BENCH_HOOKS_SIZE
#define BENCH_HOOKS_SIZE 3
#endif
constexpr int SIZE = BENCH_HOOKS_SIZE;

// A = the flattened row-major N*N matrix (seed from the harness).
inline std::vector<int32_t> gen_input_A(int size, uint32_t seed) {
  return det_bench::gen_matrix(size, seed);
}

// Determinant is single-input from the kernel's POV; mirror A EXACTLY
// (ignore the caller's seed, reuse A's seed=42) so every ciphertext arg
// the harness fills is a packing of the SAME matrix.
inline std::vector<int32_t> gen_input_B(int size, uint32_t /*seed*/) {
  return det_bench::gen_matrix(size, /*seed=*/42);
}

// A is the matrix that produced every arg (seed 42); det(A) is the answer.
inline std::vector<int32_t> reference(const std::vector<int32_t>& A,
                                      const std::vector<int32_t>& B) {
  (void)B;
  return det_bench::reference_det(A, SIZE);
}

}  // namespace bench_hooks
