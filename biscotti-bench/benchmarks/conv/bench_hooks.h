// Workload-specific hooks consumed by the main.cpp emitted from
// `heir-translate --emit-openfhe-pke-harness`. Convolution variant:
//   - `SIZE` == N (image side length; image is NxN row-major), baked in
//     per benchmark via -DBENCH_HOOKS_SIZE=<N>.
//   - Two encrypted inputs: image (A, N*N elements) and filter
//     (B, 9 elements). Both are secret at the MLIR level → the base
//     case does ct-ct multiplications, matching coyote's benchmark
//     setup for a fair perf comparison.
//   - The harness A/B split (⌈N/2⌉ args to A, ⌊N/2⌋ args to B) works
//     for two-input workloads the same way as mm/dot: the framework
//     scalarizes inputs in order, image args come first and land in
//     the A bucket, filter args come after and land in B. Symmetry
//     between mm's two matrices and conv's image+filter.
//   - `reference` == reference_conv (used for correctness check).
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

// Image side length. Override per benchmark before including this
// header if you want a different N without editing this file.
#ifndef BENCH_HOOKS_SIZE
#define BENCH_HOOKS_SIZE 3
#endif
constexpr int SIZE = BENCH_HOOKS_SIZE;

// A = image (NxN random pixels).
inline std::vector<int32_t> gen_input_A(int size, uint32_t seed) {
  return conv_bench::gen_random_image(size, seed);
}

// B = filter (fixed 9 elements regardless of image size).
// `size` parameter is ignored — filter is always 3x3.
inline std::vector<int32_t> gen_input_B(int /*size*/, uint32_t seed) {
  return conv_bench::gen_random_filter(seed);
}

inline std::vector<int32_t> reference(const std::vector<int32_t>& A,
                                      const std::vector<int32_t>& B) {
  // A = image, B = filter.
  return conv_bench::reference_conv(A, B, SIZE);
}

}  // namespace bench_hooks
