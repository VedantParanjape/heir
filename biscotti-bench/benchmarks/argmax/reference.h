// Argmax-of-array reference implementation and input generator.
// Consumed by benchmarks/argmax/bench_hooks.h — everything workload-shaped
// stays here so bench_hooks stays thin.
//
// Two-phase argmax:
//   1. Caller pre-computes all NxN pairwise comparison bits in plaintext:
//      bits[i][j] = 1 if arr[i] > arr[j] else 0, with bits[i][i] = 1 so
//      row-products don't trivially zero out on the self-slot.
//   2. Kernel receives the NxN bit matrix (encrypted, column-major
//      layout) and computes, for each row i, the product of all N bits
//      in row i. b_i = 1 iff arr[i] beat every other element = argmax.
//   Output: length-N one-hot vector `[b_0, ..., b_{N-1}]`, encrypted.
//   Caller decrypts and reads which slot is 1.
//
// Column-major layout matters: the recursive kernel splits the input
// tensor into halves of `k/2` columns each, so columns must be
// contiguous. Rows sit within each column at slots 0..N-1.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "types.h"

namespace argmax_bench {

// Random distinct-ish array (0..1e6). Wide range makes ties vanishingly
// unlikely for a benchmark; tie-breaking isn't defined by this kernel
// (would produce a one-hot with two 1s, or zero if strict inequality
// missed both directions).
inline std::vector<int32_t> gen_random_array(int N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int32_t> dist(0, 1'000'000);
  std::vector<int32_t> arr(N);
  for (auto& v : arr) v = dist(rng);
  return arr;
}

// Compute the NxN pairwise bit matrix, stored COLUMN-MAJOR:
// output[j * N + i] = 1 if arr[i] > arr[j] else 0    (i.e. bits[i][j])
// with bits[i][i] = 1 on the diagonal.
//
// Column-major means column j occupies indices [j*N, (j+1)*N). This is
// what the recursive kernel expects: each recursive call receives a
// contiguous span of `k` columns via tensor slicing.
inline std::vector<int32_t> compute_bits_col_major(
    const std::vector<int32_t>& arr) {
  int N = static_cast<int>(arr.size());
  std::vector<int32_t> bits(N * N);
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      int32_t v = (i == j) ? 1 : ((arr[i] > arr[j]) ? 1 : 0);
      bits[j * N + i] = v;
    }
  }
  return bits;
}

// Reference one-hot: `[b_0, ..., b_{N-1}]` with exactly one 1 at the
// argmax index. Computed directly from the array (no bit matrix), used
// only to verify kernel output.
inline std::vector<int32_t> reference_onehot(const std::vector<int32_t>& arr) {
  int N = static_cast<int>(arr.size());
  int argmax = 0;
  for (int i = 1; i < N; ++i)
    if (arr[i] > arr[argmax]) argmax = i;
  std::vector<int32_t> onehot(N, 0);
  onehot[argmax] = 1;
  return onehot;
}

}  // namespace argmax_bench
