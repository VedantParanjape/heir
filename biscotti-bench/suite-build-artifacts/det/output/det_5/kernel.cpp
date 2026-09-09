
#include <cassert>

#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using ConstCiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextBFVRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

std::vector<Plaintext> det_clone_0_0__preprocessing(CryptoContextT cc) {
  [[maybe_unused]] size_t v0 = 0;
  [[maybe_unused]] size_t v1 = 1;
  [[maybe_unused]] size_t v2 = 2;
  std::vector<int64_t> v3 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v4 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v5 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<Plaintext> v6(3);
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v3;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v3[i % v3.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  v6[0] = pt;
  auto pt1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v4;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (unsigned i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v4[i % v4.size()]);
  }
  auto pt1 = cc->MakePackedPlaintext(pt1_filled);
  v6[1] = pt1;
  auto pt2_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt2_filled = v5;
  pt2_filled.clear();
  pt2_filled.reserve(pt2_filled_n);
  for (unsigned i = 0; i < pt2_filled_n; ++i) {
    pt2_filled.push_back(v5[i % v5.size()]);
  }
  auto pt2 = cc->MakePackedPlaintext(pt2_filled);
  v6[2] = pt2;
  return v6;
}
std::vector<CiphertextT> det_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5,
    std::vector<CiphertextT> v6, std::vector<CiphertextT> v7,
    const std::vector<Plaintext>& v8) {
  std::vector<size_t> v9 = {255, 73};
  std::vector<size_t> v10 = {5, 251, 497};
  std::vector<size_t> v11 = {25, 437, 206, 231};
  [[maybe_unused]] size_t v12 = 3;
  [[maybe_unused]] size_t v13 = 0;
  [[maybe_unused]] size_t v14 = 191;
  [[maybe_unused]] size_t v15 = 246;
  [[maybe_unused]] size_t v16 = 510;
  [[maybe_unused]] size_t v17 = 1;
  [[maybe_unused]] size_t v18 = 2;
  const auto& ct = v6[0];
  const auto& ct1 = v2[0];
  const auto& ct2 = cc->EvalMultNoRelin(ct, ct1);
  const auto& ct3 = v5[0];
  const auto& ct4 = v3[0];
  auto ct5 = cc->EvalMultNoRelin(ct3, ct4);
  cc->EvalSubInPlace(ct5, ct2);
  cc->RelinearizeInPlace(ct5);
  const auto& ct8 = v0[0];
  auto ct9 = cc->EvalMultNoRelin(ct8, ct5);
  cc->RelinearizeInPlace(ct9);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct9);
  std::vector<CiphertextT> v19(4);
#pragma omp parallel for
  for (auto v21 = 0; v21 < 4; ++v21) {
    size_t v23 = v11[v21];
    const auto& ct11 = cc->EvalFastRotation(
        ct9, v23, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v24 = {ct11};
    v19[v21] = v24[0];
  }
  const auto& ct12 = v19[0];
  auto ct13 = v19[1];
  const auto& ct14 = v19[2];
  const auto& ct15 = v19[3];
  const auto& ct16 = v4[0];
  const auto& ct17 = v1[0];
  std::vector<CiphertextT> v25(3);
  Plaintext pt = v8[0];
  Plaintext pt1 = v8[1];
  Plaintext pt2 = v8[2];
  const auto& ct18 = v7[0];
  std::vector<CiphertextT> v26(2);
  std::vector<CiphertextT> v27(1);
  cc->EvalSubInPlace(ct13, ct14);
  cc->EvalAddInPlace(ct13, ct15);
  cc->EvalSubInPlace(ct13, ct9);
  cc->EvalAddInPlace(ct13, ct12);
  auto ct23 = cc->EvalMultNoRelin(ct16, ct13);
  cc->RelinearizeInPlace(ct23);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct23);
  auto ct25 = cc->EvalMultNoRelin(ct17, ct13);
  cc->RelinearizeInPlace(ct25);
#pragma omp parallel for
  for (auto v29 = 0; v29 < 3; ++v29) {
    size_t v31 = v10[v29];
    const auto& ct27 = cc->EvalFastRotation(
        ct23, v31, 2 * cc->GetRingDimension(), digit_decomp1);
    const std::vector<CiphertextT> v32 = {ct27};
    v25[v29] = v32[0];
  }
  const auto& ct28 = v25[0];
  const auto& ct29 = v25[1];
  auto ct30 = v25[2];
  const auto& ct31 = cc->EvalRotate(ct25, 246);
  cc->EvalSubInPlace(ct30, ct31);
  cc->EvalAddInPlace(ct30, ct29);
  cc->EvalSubInPlace(ct30, ct23);
  cc->EvalAddInPlace(ct30, ct28);
  auto ct36 = cc->EvalMult(ct30, pt);
  const auto& ct37 = cc->EvalRotate(ct30, 191);
  const auto& ct38 = cc->EvalMult(ct37, pt1);
  cc->EvalAddInPlace(ct36, ct38);
  const auto& ct40 = cc->EvalRotate(ct36, 246);
  auto ct41 = cc->EvalMult(ct40, pt2);
  const auto& ct42 = cc->EvalMult(ct36, pt);
  cc->EvalAddInPlace(ct41, ct42);
  auto ct44 = cc->EvalMultNoRelin(ct18, ct41);
  cc->RelinearizeInPlace(ct44);
  const auto& digit_decomp2 = cc->EvalFastRotationPrecompute(ct44);
#pragma omp parallel for
  for (auto v34 = 0; v34 < 2; ++v34) {
    size_t v36 = v9[v34];
    const auto& ct46 = cc->EvalFastRotation(
        ct44, v36, 2 * cc->GetRingDimension(), digit_decomp2);
    const std::vector<CiphertextT> v37 = {ct46};
    v26[v34] = v37[0];
  }
  const auto& ct47 = v26[0];
  auto ct48 = v26[1];
  cc->EvalSubInPlace(ct48, ct47);
  cc->EvalAddInPlace(ct48, ct44);
  auto ct51 = cc->EvalRotate(ct48, 510);
  cc->EvalSubInPlace(ct51, ct47);
  cc->EvalAddInPlace(ct51, ct44);
  std::vector<CiphertextT> v38(v27);
  v38[0] = ct51;
  return v38;
}
std::vector<CiphertextT> det_clone_0_0(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5,
    std::vector<CiphertextT> v6, std::vector<CiphertextT> v7) {
  const auto& v8 = det_clone_0_0__preprocessing(cc);
  const auto& v9 =
      det_clone_0_0__preprocessed(cc, v0, v1, v2, v3, v4, v5, v6, v7, v8);
  return v9;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 10;
  [[maybe_unused]] size_t v3 = 14;
  [[maybe_unused]] size_t v4 = 13;
  [[maybe_unused]] size_t v5 = 11;
  [[maybe_unused]] size_t v6 = 12;
  int32_t v7 = v0[12];
  int32_t v8 = v0[11];
  int32_t v9 = v0[13];
  int32_t v10 = v0[14];
  int32_t v11 = v0[10];
  const std::vector<int32_t> v12 = {
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v7,  v7,  v8,  v8,  v8,  v8,  v11, v11, v11, v11,
      v8,  v11, v11, v11, v11, v8,  v11, v11, v11, v11, v8,  v11, v11, v11, v11,
      v9,  v9,  v9,  v7,  v7,  v9,  v9,  v9,  v7,  v7,  v7,  v7,  v8,  v8,  v8,
      v7,  v7,  v8,  v8,  v8,  v7,  v7,  v8,  v8,  v8,  v10, v10, v10, v10, v9,
      v10, v10, v10, v10, v9,  v10, v10, v10, v10, v9,  v9,  v9,  v9,  v7,  v7,
      v9,  v9,  v9,  v7,  v7,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v7,  v7,  v8,  v8,  v8,  v8,  v11, v11, v11,
      v11, v8,  v11, v11, v11, v11, v8,  v11, v11, v11, v11, v8,  v11, v11, v11,
      v11, v9,  v9,  v9,  v7,  v7,  v9,  v9,  v9,  v7,  v7,  v7,  v7,  v8,  v8,
      v8,  v7,  v7,  v8,  v8,  v8,  v7,  v7,  v8,  v8,  v8,  v10, v10, v10, v10,
      v9,  v10, v10, v10, v10, v9,  v10, v10, v10, v10, v9,  v9,  v9,  v9,  v7,
      v7,  v9,  v9,  v9,  v7,  v7,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v7,  v7,  v8,  v8,  v8,  v8,  v11, v11,
      v11, v11, v8,  v11, v11, v11, v11, v8,  v11, v11, v11, v11, v8,  v11, v11,
      v11, v11, v9,  v9,  v9,  v7,  v7,  v9,  v9,  v9,  v7,  v7,  v7,  v7,  v8,
      v8,  v8,  v7,  v7,  v8,  v8,  v8,  v7,  v7,  v8,  v8,  v8,  v10, v10, v10,
      v10, v9,  v10, v10, v10, v10, v9,  v10, v10, v10, v10, v9,  v9,  v9,  v9,
      v7,  v7,  v9,  v9,  v9,  v7,  v7,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v7,  v7,  v8,  v8,  v8,  v8,  v11,
      v11, v11, v11, v8,  v11, v11, v11, v11, v8,  v11, v11, v11, v11, v8,  v11,
      v11, v11, v11, v9,  v9,  v9,  v7,  v7,  v9,  v9,  v9,  v7,  v7,  v7,  v7,
      v8,  v8,  v8,  v7,  v7,  v8,  v8,  v8,  v7,  v7,  v8,  v8,  v8,  v10, v10,
      v10, v10, v9,  v10, v10, v10, v10, v9,  v10, v10, v10, v10, v9,  v9,  v9,
      v9,  v7,  v7,  v9,  v9,  v9,  v7,  v7,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1};
  std::vector<int32_t> v13(1 * 1024);
  for (int64_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
    for (int64_t v13_i1 = 0; v13_i1 < 1024; ++v13_i1) {
      v13[v13_i1 + 1024 * (v13_i0)] =
          v12[0 + v13_i1 * 1 + 1024 * (0 + v13_i0 * 1)];
    }
  }
  std::vector<int64_t> v14(std::begin(v13), std::end(v13));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v14;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v14[i % v14.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v15 = {ct};
  return v15;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 6;
  [[maybe_unused]] size_t v3 = 7;
  int32_t v4 = v0[7];
  int32_t v5 = v0[6];
  const std::vector<int32_t> v6 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v4, v4, v5, v5, v5, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v4, v4, v5, v5, v5, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v4, v4, v5,
      v5, v5, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v4, v4, v5, v5, v5, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v7(1 * 1024);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 1024; ++v7_i1) {
      v7[v7_i1 + 1024 * (v7_i0)] = v6[0 + v7_i1 * 1 + 1024 * (0 + v7_i0 * 1)];
    }
  }
  std::vector<int64_t> v8(std::begin(v7), std::end(v7));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v8;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v8[i % v8.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v9 = {ct};
  return v9;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 23;
  [[maybe_unused]] size_t v3 = 21;
  [[maybe_unused]] size_t v4 = 22;
  [[maybe_unused]] size_t v5 = 20;
  int32_t v6 = v0[20];
  int32_t v7 = v0[22];
  int32_t v8 = v0[21];
  int32_t v9 = v0[23];
  const std::vector<int32_t> v10 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v9,
      v9, v9, v7, v7, v9, v9, v9, v7, v7, v7, v7, v8, v8, v8, v7, v7, v8, v8,
      v8, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v8, v6, v6, v6, v6, v8, v6,
      v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v7, v7, v8, v8, v8,
      v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6,
      v6, v6, v7, v7, v8, v8, v8, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8,
      v6, v6, v6, v6, v8, v6, v6, v6, v6, v7, v7, v8, v8, v8, v8, v6, v6, v6,
      v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v9, v9, v9, v7, v7, v9, v9, v9, v7, v7, v7, v7, v8, v8, v8,
      v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v8, v6, v6,
      v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v7,
      v7, v8, v8, v8, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6,
      v6, v8, v6, v6, v6, v6, v7, v7, v8, v8, v8, v8, v6, v6, v6, v6, v8, v6,
      v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v7, v7, v8, v8, v8,
      v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6,
      v6, v6, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v9, v9, v9, v7, v7, v9, v9, v9, v7, v7, v7,
      v7, v8, v8, v8, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v7, v7, v8, v8,
      v8, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6,
      v6, v6, v6, v7, v7, v8, v8, v8, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6,
      v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v7, v7, v8, v8, v8, v8, v6, v6,
      v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v7,
      v7, v8, v8, v8, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6,
      v6, v8, v6, v6, v6, v6, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v9, v9, v9, v7, v7, v9, v9,
      v9, v7, v7, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8,
      v7, v7, v8, v8, v8, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6,
      v6, v6, v8, v6, v6, v6, v6, v7, v7, v8, v8, v8, v8, v6, v6, v6, v6, v8,
      v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v7, v7, v8, v8,
      v8, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v8, v6,
      v6, v6, v6, v7, v7, v8, v8, v8, v8, v6, v6, v6, v6, v8, v6, v6, v6, v6,
      v8, v6, v6, v6, v6, v8, v6, v6, v6, v6, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 1024);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 1024; ++v11_i1) {
      v11[v11_i1 + 1024 * (v11_i0)] =
          v10[0 + v11_i1 * 1 + 1024 * (0 + v11_i0 * 1)];
    }
  }
  std::vector<int64_t> v12(std::begin(v11), std::end(v11));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v12;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v12[i % v12.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v13 = {ct};
  return v13;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 22;
  [[maybe_unused]] size_t v3 = 23;
  [[maybe_unused]] size_t v4 = 24;
  [[maybe_unused]] size_t v5 = 21;
  int32_t v6 = v0[21];
  int32_t v7 = v0[24];
  int32_t v8 = v0[23];
  int32_t v9 = v0[22];
  const std::vector<int32_t> v10 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v7,
      v7, v7, v7, v8, v7, v7, v7, v7, v8, v7, v7, v7, v7, v8, v8, v8, v8, v9,
      v9, v8, v8, v8, v9, v9, v7, v7, v7, v7, v8, v7, v7, v7, v7, v8, v7, v7,
      v7, v7, v8, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9,
      v8, v8, v8, v9, v9, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v9, v9, v6,
      v6, v6, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v9, v9, v6, v6, v6, v9,
      v9, v6, v6, v6, v9, v9, v6, v6, v6, v8, v8, v8, v9, v9, v8, v8, v8, v9,
      v9, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v7, v7, v7, v7, v8, v7, v7, v7, v7, v8, v7, v7, v7, v7, v8,
      v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v7, v7, v7, v7, v8, v7, v7, v7,
      v7, v8, v7, v7, v7, v7, v8, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v8,
      v8, v8, v9, v9, v8, v8, v8, v9, v9, v9, v9, v6, v6, v6, v9, v9, v6, v6,
      v6, v9, v9, v6, v6, v6, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v9, v9,
      v6, v6, v6, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v8, v8, v8, v9, v9,
      v8, v8, v8, v9, v9, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v9, v9, v6,
      v6, v6, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v7, v7, v7, v7, v8, v7, v7, v7, v7, v8, v7,
      v7, v7, v7, v8, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v7, v7, v7, v7,
      v8, v7, v7, v7, v7, v8, v7, v7, v7, v7, v8, v8, v8, v8, v9, v9, v8, v8,
      v8, v9, v9, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v9, v9, v6, v6, v6,
      v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v8, v8, v8, v9, v9, v8, v8, v8,
      v9, v9, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v8,
      v8, v8, v9, v9, v8, v8, v8, v9, v9, v9, v9, v6, v6, v6, v9, v9, v6, v6,
      v6, v9, v9, v6, v6, v6, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v7, v7, v7, v7, v8, v7, v7,
      v7, v7, v8, v7, v7, v7, v7, v8, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9,
      v7, v7, v7, v7, v8, v7, v7, v7, v7, v8, v7, v7, v7, v7, v8, v8, v8, v8,
      v9, v9, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v9,
      v9, v6, v6, v6, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v8, v8, v8, v9,
      v9, v8, v8, v8, v9, v9, v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v9, v9,
      v6, v6, v6, v8, v8, v8, v9, v9, v8, v8, v8, v9, v9, v9, v9, v6, v6, v6,
      v9, v9, v6, v6, v6, v9, v9, v6, v6, v6, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 1024);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 1024; ++v11_i1) {
      v11[v11_i1 + 1024 * (v11_i0)] =
          v10[0 + v11_i1 * 1 + 1024 * (0 + v11_i0 * 1)];
    }
  }
  std::vector<int64_t> v12(std::begin(v11), std::end(v11));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v12;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v12[i % v12.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v13 = {ct};
  return v13;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg4(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 6;
  [[maybe_unused]] size_t v3 = 7;
  [[maybe_unused]] size_t v4 = 5;
  [[maybe_unused]] size_t v5 = 9;
  [[maybe_unused]] size_t v6 = 8;
  int32_t v7 = v0[8];
  int32_t v8 = v0[9];
  int32_t v9 = v0[5];
  int32_t v10 = v0[7];
  int32_t v11 = v0[6];
  const std::vector<int32_t> v12 = {
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v11, v9, v9, v9, v9, v1, v1, v1,
      v1, v1, v7, v7, v7, v10, v10, v8, v8,  v8, v8, v7, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v11, v9, v9, v9, v9, v1, v1, v1,
      v1, v1, v7, v7, v7, v10, v10, v8, v8,  v8, v8, v7, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v11, v9, v9, v9, v9, v1, v1, v1,
      v1, v1, v7, v7, v7, v10, v10, v8, v8,  v8, v8, v7, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v11, v9, v9, v9, v9, v1, v1, v1,
      v1, v1, v7, v7, v7, v10, v10, v8, v8,  v8, v8, v7, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v13(1 * 1024);
  for (int64_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
    for (int64_t v13_i1 = 0; v13_i1 < 1024; ++v13_i1) {
      v13[v13_i1 + 1024 * (v13_i0)] =
          v12[0 + v13_i1 * 1 + 1024 * (0 + v13_i0 * 1)];
    }
  }
  std::vector<int64_t> v14(std::begin(v13), std::end(v13));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v14;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v14[i % v14.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v15 = {ct};
  return v15;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg5(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 15;
  [[maybe_unused]] size_t v3 = 16;
  [[maybe_unused]] size_t v4 = 17;
  [[maybe_unused]] size_t v5 = 18;
  int32_t v6 = v0[18];
  int32_t v7 = v0[17];
  int32_t v8 = v0[16];
  int32_t v9 = v0[15];
  const std::vector<int32_t> v10 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6,
      v6, v6, v7, v7, v6, v6, v6, v7, v7, v7, v7, v8, v8, v8, v7, v7, v8, v8,
      v8, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v8, v9,
      v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v7, v7, v8, v8, v8,
      v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9,
      v9, v9, v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8,
      v9, v9, v9, v9, v8, v9, v9, v9, v9, v7, v7, v8, v8, v8, v8, v9, v9, v9,
      v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v6, v6, v6, v7, v7, v6, v6, v6, v7, v7, v7, v7, v8, v8, v8,
      v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v8, v9, v9,
      v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v7,
      v7, v8, v8, v8, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9,
      v9, v8, v9, v9, v9, v9, v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v8, v9,
      v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v7, v7, v8, v8, v8,
      v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9,
      v9, v9, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v6, v6, v6, v7, v7, v6, v6, v6, v7, v7, v7,
      v7, v8, v8, v8, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v7, v7, v8, v8,
      v8, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9,
      v9, v9, v9, v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9,
      v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v7, v7, v8, v8, v8, v8, v9, v9,
      v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v7,
      v7, v8, v8, v8, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9,
      v9, v8, v9, v9, v9, v9, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6, v6, v6, v7, v7, v6, v6,
      v6, v7, v7, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8, v7, v7, v8, v8, v8,
      v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9,
      v9, v9, v8, v9, v9, v9, v9, v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v8,
      v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v7, v7, v8, v8,
      v8, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v8, v9,
      v9, v9, v9, v7, v7, v8, v8, v8, v8, v9, v9, v9, v9, v8, v9, v9, v9, v9,
      v8, v9, v9, v9, v9, v8, v9, v9, v9, v9, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 1024);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 1024; ++v11_i1) {
      v11[v11_i1 + 1024 * (v11_i0)] =
          v10[0 + v11_i1 * 1 + 1024 * (0 + v11_i0 * 1)];
    }
  }
  std::vector<int64_t> v12(std::begin(v11), std::end(v11));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v12;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v12[i % v12.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v13 = {ct};
  return v13;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg6(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 16;
  [[maybe_unused]] size_t v3 = 17;
  [[maybe_unused]] size_t v4 = 18;
  [[maybe_unused]] size_t v5 = 19;
  int32_t v6 = v0[19];
  int32_t v7 = v0[18];
  int32_t v8 = v0[17];
  int32_t v9 = v0[16];
  const std::vector<int32_t> v10 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6,
      v6, v6, v6, v7, v6, v6, v6, v6, v7, v6, v6, v6, v6, v7, v7, v7, v7, v8,
      v8, v7, v7, v7, v8, v8, v6, v6, v6, v6, v7, v6, v6, v6, v6, v7, v6, v6,
      v6, v6, v7, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8,
      v7, v7, v7, v8, v8, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v8, v8, v9,
      v9, v9, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v8, v8, v9, v9, v9, v8,
      v8, v9, v9, v9, v8, v8, v9, v9, v9, v7, v7, v7, v8, v8, v7, v7, v7, v8,
      v8, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v6, v6, v6, v6, v7, v6, v6, v6, v6, v7, v6, v6, v6, v6, v7,
      v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v6, v6, v6, v6, v7, v6, v6, v6,
      v6, v7, v6, v6, v6, v6, v7, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v7,
      v7, v7, v8, v8, v7, v7, v7, v8, v8, v8, v8, v9, v9, v9, v8, v8, v9, v9,
      v9, v8, v8, v9, v9, v9, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v8, v8,
      v9, v9, v9, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v7, v7, v7, v8, v8,
      v7, v7, v7, v8, v8, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v8, v8, v9,
      v9, v9, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v6, v6, v6, v6, v7, v6, v6, v6, v6, v7, v6,
      v6, v6, v6, v7, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v6, v6, v6, v6,
      v7, v6, v6, v6, v6, v7, v6, v6, v6, v6, v7, v7, v7, v7, v8, v8, v7, v7,
      v7, v8, v8, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v8, v8, v9, v9, v9,
      v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v7, v7, v7, v8, v8, v7, v7, v7,
      v8, v8, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v7,
      v7, v7, v8, v8, v7, v7, v7, v8, v8, v8, v8, v9, v9, v9, v8, v8, v9, v9,
      v9, v8, v8, v9, v9, v9, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6, v6, v6, v6, v7, v6, v6,
      v6, v6, v7, v6, v6, v6, v6, v7, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8,
      v6, v6, v6, v6, v7, v6, v6, v6, v6, v7, v6, v6, v6, v6, v7, v7, v7, v7,
      v8, v8, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v8,
      v8, v9, v9, v9, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v7, v7, v7, v8,
      v8, v7, v7, v7, v8, v8, v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v8, v8,
      v9, v9, v9, v7, v7, v7, v8, v8, v7, v7, v7, v8, v8, v8, v8, v9, v9, v9,
      v8, v8, v9, v9, v9, v8, v8, v9, v9, v9, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 1024);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 1024; ++v11_i1) {
      v11[v11_i1 + 1024 * (v11_i0)] =
          v10[0 + v11_i1 * 1 + 1024 * (0 + v11_i0 * 1)];
    }
  }
  std::vector<int64_t> v12(std::begin(v11), std::end(v11));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v12;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v12[i % v12.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v13 = {ct};
  return v13;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg7(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 4;
  [[maybe_unused]] size_t v3 = 3;
  [[maybe_unused]] size_t v4 = 1;
  [[maybe_unused]] size_t v5 = 0;
  [[maybe_unused]] size_t v6 = 2;
  int32_t v7 = v0[2];
  int32_t v8 = v0[1];
  int32_t v9 = v0[0];
  int32_t v10 = v0[3];
  int32_t v11 = v0[4];
  const std::vector<int32_t> v12 = {
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v9,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v8, v7, v10, v11, v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v9,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v8, v7,  v10, v11,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v9, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v8,  v7,  v10,
      v11, v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v9, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v8,  v7,
      v10, v11, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1,
      v1,  v1,  v1, v1};
  std::vector<int32_t> v13(1 * 1024);
  for (int64_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
    for (int64_t v13_i1 = 0; v13_i1 < 1024; ++v13_i1) {
      v13[v13_i1 + 1024 * (v13_i0)] =
          v12[0 + v13_i1 * 1 + 1024 * (0 + v13_i0 * 1)];
    }
  }
  std::vector<int64_t> v14(std::begin(v13), std::end(v13));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v14;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v14[i % v14.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v15 = {ct};
  return v15;
}
std::vector<int32_t> det_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 219;
  [[maybe_unused]] size_t v2 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(1024);
  const auto& v3_cast = pt->GetPackedValue();
  std::vector<int32_t> v3(std::begin(v3_cast), std::end(v3_cast));
  int32_t v4 = v3[219 + 1024 * (0)];
  const std::vector<int32_t> v5 = {v4};
  return v5;
}
CryptoContextT det_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(6);
  params.SetPlaintextModulus(65537);
  params.SetKeySwitchTechnique(HYBRID);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT det_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                       PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(
      sk, {206, 251, 73, 246, 497, 191, 255, 25, 231, 437, 510, 5});
  return cc;
}
