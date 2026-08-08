// Shared OpenFHE type aliases used by every benchmark suite. Kept
// separate from workload-specific reference implementations so that
// per-suite headers can pull in just what they need.

#pragma once

// Match the include style used by HEIR-emitted kernels when built
// under Bazel (via --openfhe-include-type=source-relative). Under a
// cmake-installed OpenFHE these would be "openfhe/pke/openfhe.h" etc.
#include "src/pke/include/cryptocontext.h"      // from @openfhe
#include "src/pke/include/gen-cryptocontext.h"  // from @openfhe
#include "src/pke/include/scheme/bfvrns/gen-cryptocontext-bfvrns.h"  // from @openfhe

namespace mm_bench {

using CiphertextT = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using PlaintextT = lbcrypto::Plaintext;
using CryptoContextT = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using PublicKeyT = lbcrypto::PublicKey<lbcrypto::DCRTPoly>;
using PrivateKeyT = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;

}  // namespace mm_bench
