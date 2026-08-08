// Dot-product-specific reference implementation and input generator.
// Consumed by benchmarks/dot/bench_hooks.h — everything workload-shaped
// stays here so bench_hooks stays thin.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "types.h"

namespace dot_bench {

// Reference: scalar dot product of two length-N int32 vectors.
// The bench harness compares `got` against this — returned as a
// 1-element vector to keep the same "vector-of-int32" contract that
// bench_hooks::reference uses across all suites.
inline std::vector<int32_t> reference_dot(const std::vector<int32_t>& A,
                                          const std::vector<int32_t>& B,
                                          int N) {
  int64_t acc = 0;
  for (int i = 0; i < N; ++i) acc += int64_t(A[i]) * int64_t(B[i]);
  return {int32_t(acc)};
}

// Small values (0..8) so partial products fit in int32 without approaching
// the FHE plaintext modulus. Matches the mm suite for consistency.
inline std::vector<int32_t> gen_random_vector(int N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int32_t> dist(0, 8);
  std::vector<int32_t> V(N);
  for (auto& v : V) v = dist(rng);
  return V;
}

}  // namespace dot_bench
