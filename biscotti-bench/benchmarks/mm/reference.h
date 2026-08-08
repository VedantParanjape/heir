// Matmul-specific reference implementation and input generator.
// Consumed by benchmarks/mm/bench_hooks.h — everything workload-shaped
// stays here so bench_hooks stays thin.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "types.h"

namespace mm_bench {

// C = A * B on K*K int32 row-major matrices.
inline std::vector<int32_t> reference_matmul(const std::vector<int32_t>& A,
                                             const std::vector<int32_t>& B,
                                             int K) {
  std::vector<int32_t> C(K * K, 0);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      int64_t acc = 0;
      for (int k = 0; k < K; ++k)
        acc += int64_t(A[i * K + k]) * int64_t(B[k * K + j]);
      C[i * K + j] = int32_t(acc);
    }
  }
  return C;
}

// Small values (0..8) so partial products fit in int32 without
// approaching plaintextModulus.
inline std::vector<int32_t> gen_random_matrix(int K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int32_t> dist(0, 8);
  std::vector<int32_t> M(K * K);
  for (auto& v : M) v = dist(rng);
  return M;
}

}  // namespace mm_bench
