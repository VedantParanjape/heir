// Convolution-specific reference implementation and input generators.
// Consumed by benchmarks/conv/bench_hooks.h — everything workload-shaped
// stays here so bench_hooks stays thin.
//
// Both image AND filter are runtime inputs (both encrypted in the FHE
// kernel). This matches coyote's benchmark setup where the filter is
// secret, producing ct-ct multiplications instead of pt-ct.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "types.h"

namespace conv_bench {

constexpr int FILTER_SIZE = 3;
constexpr int FILTER_ELEMS = FILTER_SIZE * FILTER_SIZE;

// Valid convolution: output[i,j] = sum_{du,dv} filter[du,dv] * image[i+du,
// j+dv] for i,j in [0, N-2). Output stored row-major in a (N-2)*(N-2) vector.
inline std::vector<int32_t> reference_conv(const std::vector<int32_t>& image,
                                           const std::vector<int32_t>& filter,
                                           int N) {
  int out_side = N - 2;
  std::vector<int32_t> out(out_side * out_side, 0);
  for (int i = 0; i < out_side; ++i) {
    for (int j = 0; j < out_side; ++j) {
      int32_t sum = 0;
      for (int du = 0; du < FILTER_SIZE; ++du) {
        for (int dv = 0; dv < FILTER_SIZE; ++dv) {
          sum += filter[du * FILTER_SIZE + dv] * image[(i + du) * N + (j + dv)];
        }
      }
      out[i * out_side + j] = sum;
    }
  }
  return out;
}

// Small image values (0..8) so per-window partial sums stay comfortably
// below the FHE plaintext modulus.
inline std::vector<int32_t> gen_random_image(int N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int32_t> dist(0, 8);
  std::vector<int32_t> img(N * N);
  for (auto& v : img) v = dist(rng);
  return img;
}

// Small filter coefficients (0..3). Combined with image values (0..8),
// max single sum-of-products is 9 * 8 * 3 = 216 — well within int32 and
// the plaintext modulus (65537).
inline std::vector<int32_t> gen_random_filter(uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int32_t> dist(0, 3);
  std::vector<int32_t> filter(FILTER_ELEMS);
  for (auto& v : filter) v = dist(rng);
  return filter;
}

}  // namespace conv_bench
