// Workload-specific hooks consumed by the main.cpp emitted from
// `heir-translate --emit-openfhe-pke-harness`. Matmul variant:
//   - `SIZE` == K (matrix side, baked in per benchmark via
//     -DBENCH_HOOKS_SIZE=<K>).
//   - inputs are K*K row-major int32 matrices.
//   - `reference` == reference_matmul (used for correctness check).
//
// The emitted benchmark.cpp inlines the split of (A, B) into N encrypt
// inputs (⌈N/2⌉ A + ⌊N/2⌋ B, biscotti convention). N is baked in at
// emit time from the count of __encrypt__argK helpers, so per-variant
// N differences are handled without any per-workload code here.
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

// Matrix side. Override per benchmark before including this header if
// you want a different K without editing this file.
#ifndef BENCH_HOOKS_SIZE
#define BENCH_HOOKS_SIZE 4
#endif
constexpr int SIZE = BENCH_HOOKS_SIZE;

inline std::vector<int32_t> gen_input_A(int size, uint32_t /*seed*/) {
  std::vector<int32_t> v(size * size);
  for (int i = 0; i < int(v.size()); ++i) v[i] = i + 1;  // [1, 2, ..., 16]
  return v;
}

inline std::vector<int32_t> gen_input_B(int size, uint32_t /*seed*/) {
  std::vector<int32_t> v(size * size);
  for (int i = 0; i < int(v.size()); ++i)
    v[i] = 100 + i;  // [100, 101, ..., 115]
  return v;
}

inline std::vector<int32_t> reference(const std::vector<int32_t>& A,
                                      const std::vector<int32_t>& B) {
  return mm_bench::reference_matmul(A, B, SIZE);
}

}  // namespace bench_hooks
