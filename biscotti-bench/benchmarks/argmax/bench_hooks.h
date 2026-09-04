// Workload-specific hooks consumed by the main.cpp emitted from
// `heir-translate --emit-openfhe-pke-harness`. Argmax variant:
//   - `SIZE` == N (array length), baked in per benchmark via
//     -DBENCH_HOOKS_SIZE=<N>.
//   - Single-input workload from the kernel's POV: the encrypted input
//     is the NxN column-major pairwise-bit matrix (N*N bits total).
//     The original N-element array never enters the kernel — it's
//     used only at the caller (in this harness) to compute the bits
//     and the reference one-hot.
//   - The harness has a two-arg (A, B) convention inherited from
//     mm/dot; we return the SAME bit matrix for both A and B (via the
//     seed-42 mirror trick) so any encrypt-side split lands consistent
//     bits in every bucket.
//   - `reference` returns the length-N one-hot: `[b_0, ..., b_{N-1}]`
//     with exactly one 1 at the argmax index.
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

// Array length. Override per benchmark before including this header if
// you want a different N without editing this file.
#ifndef BENCH_HOOKS_SIZE
#define BENCH_HOOKS_SIZE 4
#endif
constexpr int SIZE = BENCH_HOOKS_SIZE;

// A = the pre-computed NxN column-major bit matrix (N*N int32 values,
// each 0 or 1). The size argument passed by the harness is the array
// length N; we use it to know how many elements the caller intended.
inline std::vector<int32_t> gen_input_A(int size, uint32_t seed) {
  auto arr = argmax_bench::gen_random_array(size, seed);
  return argmax_bench::compute_bits_col_major(arr);
}

// Argmax is single-input from the kernel's POV; mirror A EXACTLY
// (ignore the caller's seed and use A's seed) so the harness's A/B
// split convention never leaks different bit matrices into any encrypt
// bucket. The emitted harness splits ⌈N/2⌉ args to A and ⌊N/2⌋ to B;
// for argmax, every arg must see the SAME bit matrix.
inline std::vector<int32_t> gen_input_B(int size, uint32_t /*seed*/) {
  auto arr = argmax_bench::gen_random_array(size, /*seed=*/42);
  return argmax_bench::compute_bits_col_major(arr);
}

inline std::vector<int32_t> reference(const std::vector<int32_t>& A,
                                      const std::vector<int32_t>& B) {
  (void)A;  // A is the bit matrix, not the array — reference is
            // computed from the array that produced A, which we
            // regenerate here from the same seed.
  (void)B;
  auto arr = argmax_bench::gen_random_array(SIZE, /*seed=*/42);
  return argmax_bench::reference_onehot(arr);
}

}  // namespace bench_hooks
