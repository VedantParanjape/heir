// Determinant reference implementation and input generator.
// Consumed by benchmarks/det/bench_hooks.h.
//
// The kernel input is a flattened row-major N*N matrix (encrypted); the
// kernel computes det via cofactor (Laplace) expansion along row 0 and
// returns a length-1 result. This header regenerates the same matrix in
// plaintext and computes its determinant so the harness can verify.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "types.h"

namespace det_bench {

// Random N*N matrix, row-major, small entries so the determinant stays
// well inside the BFV plaintext modulus (a 4x4 with entries in [0,3] has
// |det| <= 24*81 ~= 1944).
inline std::vector<int32_t> gen_matrix(int N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int32_t> dist(0, 3);
  std::vector<int32_t> m(static_cast<size_t>(N) * N);
  for (auto& v : m) v = dist(rng);
  return m;
}

// Plaintext determinant by cofactor expansion along row 0 (matches the
// kernel's algorithm). int64 accumulation avoids intermediate overflow.
inline int64_t det_cofactor(const std::vector<int32_t>& m, int n) {
  if (n == 1) return m[0];
  if (n == 2)
    return static_cast<int64_t>(m[0]) * m[3] -
           static_cast<int64_t>(m[1]) * m[2];
  int64_t d = 0;
  for (int j = 0; j < n; ++j) {
    std::vector<int32_t> minor(static_cast<size_t>(n - 1) * (n - 1));
    int idx = 0;
    for (int r = 1; r < n; ++r)
      for (int c = 0; c < n; ++c)
        if (c != j) minor[idx++] = m[r * n + c];
    int64_t sign = (j % 2 == 0) ? 1 : -1;
    d += sign * static_cast<int64_t>(m[j]) * det_cofactor(minor, n - 1);
  }
  return d;
}

// Reference output: length-1 vector holding det(matrix).
inline std::vector<int32_t> reference_det(const std::vector<int32_t>& m,
                                          int N) {
  return {static_cast<int32_t>(det_cofactor(m, N))};
}

}  // namespace det_bench
