// Workload-specific hooks consumed by the main.cpp emitted from
// `heir-translate --emit-openfhe-pke-harness`. Dot-product variant:
//   - `SIZE` == N (vector length, baked in per benchmark via
//     -DBENCH_HOOKS_SIZE=<N>).
//   - inputs are length-N int32 vectors.
//   - `reference` == reference_dot (returned as a 1-element vector).
//
// The emitted benchmark.cpp inlines the split of (A, B) into K encrypt
// inputs (⌈K/2⌉ A + ⌊K/2⌋ B, biscotti convention). K is baked in at
// emit time from the count of __encrypt__argK helpers, so per-variant
// K differences are handled without any per-workload code here.
//
// Required members of the `bench_hooks` namespace:
//   constexpr int SIZE = <value>;
//   std::vector<int32_t> gen_input_A(int size, uint32_t seed);
//   std::vector<int32_t> gen_input_B(int size, uint32_t seed);
//   std::vector<int32_t> reference(const std::vector<int32_t>& A,
//                                  const std::vector<int32_t>& B);

#pragma once

#include <cstdint>
#include <vector>

#include "reference.h"

namespace bench_hooks {

// Vector length. Override per benchmark before including this header if
// you want a different N without editing this file.
#ifndef BENCH_HOOKS_SIZE
#define BENCH_HOOKS_SIZE 3
#endif
constexpr int SIZE = BENCH_HOOKS_SIZE;

inline std::vector<int32_t> gen_input_A(int size, uint32_t seed) {
  return dot_bench::gen_random_vector(size, seed);
}

inline std::vector<int32_t> gen_input_B(int size, uint32_t seed) {
  return dot_bench::gen_random_vector(size, seed);
}

inline std::vector<int32_t> reference(const std::vector<int32_t>& A,
                                      const std::vector<int32_t>& B) {
  return dot_bench::reference_dot(A, B, SIZE);
}

}  // namespace bench_hooks
