
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

std::vector<Plaintext> argmax_clone_0_0__preprocessing(CryptoContextT cc) {
  [[maybe_unused]] size_t v0 = 0;
  [[maybe_unused]] size_t v1 = 1;
  [[maybe_unused]] size_t v2 = 2;
  [[maybe_unused]] size_t v3 = 3;
  [[maybe_unused]] size_t v4 = 4;
  [[maybe_unused]] size_t v5 = 5;
  [[maybe_unused]] size_t v6 = 6;
  [[maybe_unused]] size_t v7 = 7;
  [[maybe_unused]] size_t v8 = 8;
  [[maybe_unused]] size_t v9 = 9;
  [[maybe_unused]] size_t v10 = 10;
  [[maybe_unused]] size_t v11 = 11;
  [[maybe_unused]] size_t v12 = 12;
  [[maybe_unused]] size_t v13 = 13;
  [[maybe_unused]] size_t v14 = 14;
  [[maybe_unused]] size_t v15 = 15;
  [[maybe_unused]] size_t v16 = 16;
  [[maybe_unused]] size_t v17 = 17;
  [[maybe_unused]] size_t v18 = 18;
  [[maybe_unused]] size_t v19 = 19;
  [[maybe_unused]] size_t v20 = 20;
  [[maybe_unused]] size_t v21 = 21;
  [[maybe_unused]] size_t v22 = 22;
  [[maybe_unused]] size_t v23 = 23;
  std::vector<int64_t> v24 = {
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  std::vector<int64_t> v25 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v26 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v27 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v28 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v29 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v30 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v31 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v32 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v33 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v34 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v35 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v36 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v37 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v38 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v39 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v40 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v41 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v42 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v43 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v44 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v45 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v46 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v47 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<Plaintext> v48(24);
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v24;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v24[i % v24.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  v48[0] = pt;
  auto pt1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v25;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (unsigned i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v25[i % v25.size()]);
  }
  auto pt1 = cc->MakePackedPlaintext(pt1_filled);
  v48[1] = pt1;
  auto pt2_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt2_filled = v26;
  pt2_filled.clear();
  pt2_filled.reserve(pt2_filled_n);
  for (unsigned i = 0; i < pt2_filled_n; ++i) {
    pt2_filled.push_back(v26[i % v26.size()]);
  }
  auto pt2 = cc->MakePackedPlaintext(pt2_filled);
  v48[2] = pt2;
  auto pt3_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt3_filled = v27;
  pt3_filled.clear();
  pt3_filled.reserve(pt3_filled_n);
  for (unsigned i = 0; i < pt3_filled_n; ++i) {
    pt3_filled.push_back(v27[i % v27.size()]);
  }
  auto pt3 = cc->MakePackedPlaintext(pt3_filled);
  v48[3] = pt3;
  auto pt4_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt4_filled = v28;
  pt4_filled.clear();
  pt4_filled.reserve(pt4_filled_n);
  for (unsigned i = 0; i < pt4_filled_n; ++i) {
    pt4_filled.push_back(v28[i % v28.size()]);
  }
  auto pt4 = cc->MakePackedPlaintext(pt4_filled);
  v48[4] = pt4;
  auto pt5_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt5_filled = v29;
  pt5_filled.clear();
  pt5_filled.reserve(pt5_filled_n);
  for (unsigned i = 0; i < pt5_filled_n; ++i) {
    pt5_filled.push_back(v29[i % v29.size()]);
  }
  auto pt5 = cc->MakePackedPlaintext(pt5_filled);
  v48[5] = pt5;
  auto pt6_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt6_filled = v30;
  pt6_filled.clear();
  pt6_filled.reserve(pt6_filled_n);
  for (unsigned i = 0; i < pt6_filled_n; ++i) {
    pt6_filled.push_back(v30[i % v30.size()]);
  }
  auto pt6 = cc->MakePackedPlaintext(pt6_filled);
  v48[6] = pt6;
  auto pt7_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt7_filled = v31;
  pt7_filled.clear();
  pt7_filled.reserve(pt7_filled_n);
  for (unsigned i = 0; i < pt7_filled_n; ++i) {
    pt7_filled.push_back(v31[i % v31.size()]);
  }
  auto pt7 = cc->MakePackedPlaintext(pt7_filled);
  v48[7] = pt7;
  auto pt8_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt8_filled = v32;
  pt8_filled.clear();
  pt8_filled.reserve(pt8_filled_n);
  for (unsigned i = 0; i < pt8_filled_n; ++i) {
    pt8_filled.push_back(v32[i % v32.size()]);
  }
  auto pt8 = cc->MakePackedPlaintext(pt8_filled);
  v48[8] = pt8;
  auto pt9_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt9_filled = v33;
  pt9_filled.clear();
  pt9_filled.reserve(pt9_filled_n);
  for (unsigned i = 0; i < pt9_filled_n; ++i) {
    pt9_filled.push_back(v33[i % v33.size()]);
  }
  auto pt9 = cc->MakePackedPlaintext(pt9_filled);
  v48[9] = pt9;
  auto pt10_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt10_filled = v34;
  pt10_filled.clear();
  pt10_filled.reserve(pt10_filled_n);
  for (unsigned i = 0; i < pt10_filled_n; ++i) {
    pt10_filled.push_back(v34[i % v34.size()]);
  }
  auto pt10 = cc->MakePackedPlaintext(pt10_filled);
  v48[10] = pt10;
  auto pt11_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt11_filled = v35;
  pt11_filled.clear();
  pt11_filled.reserve(pt11_filled_n);
  for (unsigned i = 0; i < pt11_filled_n; ++i) {
    pt11_filled.push_back(v35[i % v35.size()]);
  }
  auto pt11 = cc->MakePackedPlaintext(pt11_filled);
  v48[11] = pt11;
  auto pt12_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt12_filled = v36;
  pt12_filled.clear();
  pt12_filled.reserve(pt12_filled_n);
  for (unsigned i = 0; i < pt12_filled_n; ++i) {
    pt12_filled.push_back(v36[i % v36.size()]);
  }
  auto pt12 = cc->MakePackedPlaintext(pt12_filled);
  v48[12] = pt12;
  auto pt13_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt13_filled = v37;
  pt13_filled.clear();
  pt13_filled.reserve(pt13_filled_n);
  for (unsigned i = 0; i < pt13_filled_n; ++i) {
    pt13_filled.push_back(v37[i % v37.size()]);
  }
  auto pt13 = cc->MakePackedPlaintext(pt13_filled);
  v48[13] = pt13;
  auto pt14_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt14_filled = v38;
  pt14_filled.clear();
  pt14_filled.reserve(pt14_filled_n);
  for (unsigned i = 0; i < pt14_filled_n; ++i) {
    pt14_filled.push_back(v38[i % v38.size()]);
  }
  auto pt14 = cc->MakePackedPlaintext(pt14_filled);
  v48[14] = pt14;
  auto pt15_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt15_filled = v39;
  pt15_filled.clear();
  pt15_filled.reserve(pt15_filled_n);
  for (unsigned i = 0; i < pt15_filled_n; ++i) {
    pt15_filled.push_back(v39[i % v39.size()]);
  }
  auto pt15 = cc->MakePackedPlaintext(pt15_filled);
  v48[15] = pt15;
  auto pt16_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt16_filled = v40;
  pt16_filled.clear();
  pt16_filled.reserve(pt16_filled_n);
  for (unsigned i = 0; i < pt16_filled_n; ++i) {
    pt16_filled.push_back(v40[i % v40.size()]);
  }
  auto pt16 = cc->MakePackedPlaintext(pt16_filled);
  v48[16] = pt16;
  auto pt17_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt17_filled = v41;
  pt17_filled.clear();
  pt17_filled.reserve(pt17_filled_n);
  for (unsigned i = 0; i < pt17_filled_n; ++i) {
    pt17_filled.push_back(v41[i % v41.size()]);
  }
  auto pt17 = cc->MakePackedPlaintext(pt17_filled);
  v48[17] = pt17;
  auto pt18_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt18_filled = v42;
  pt18_filled.clear();
  pt18_filled.reserve(pt18_filled_n);
  for (unsigned i = 0; i < pt18_filled_n; ++i) {
    pt18_filled.push_back(v42[i % v42.size()]);
  }
  auto pt18 = cc->MakePackedPlaintext(pt18_filled);
  v48[18] = pt18;
  auto pt19_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt19_filled = v43;
  pt19_filled.clear();
  pt19_filled.reserve(pt19_filled_n);
  for (unsigned i = 0; i < pt19_filled_n; ++i) {
    pt19_filled.push_back(v43[i % v43.size()]);
  }
  auto pt19 = cc->MakePackedPlaintext(pt19_filled);
  v48[19] = pt19;
  auto pt20_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt20_filled = v44;
  pt20_filled.clear();
  pt20_filled.reserve(pt20_filled_n);
  for (unsigned i = 0; i < pt20_filled_n; ++i) {
    pt20_filled.push_back(v44[i % v44.size()]);
  }
  auto pt20 = cc->MakePackedPlaintext(pt20_filled);
  v48[20] = pt20;
  auto pt21_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt21_filled = v45;
  pt21_filled.clear();
  pt21_filled.reserve(pt21_filled_n);
  for (unsigned i = 0; i < pt21_filled_n; ++i) {
    pt21_filled.push_back(v45[i % v45.size()]);
  }
  auto pt21 = cc->MakePackedPlaintext(pt21_filled);
  v48[21] = pt21;
  auto pt22_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt22_filled = v46;
  pt22_filled.clear();
  pt22_filled.reserve(pt22_filled_n);
  for (unsigned i = 0; i < pt22_filled_n; ++i) {
    pt22_filled.push_back(v46[i % v46.size()]);
  }
  auto pt22 = cc->MakePackedPlaintext(pt22_filled);
  v48[22] = pt22;
  auto pt23_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt23_filled = v47;
  pt23_filled.clear();
  pt23_filled.reserve(pt23_filled_n);
  for (unsigned i = 0; i < pt23_filled_n; ++i) {
    pt23_filled.push_back(v47[i % v47.size()]);
  }
  auto pt23 = cc->MakePackedPlaintext(pt23_filled);
  v48[23] = pt23;
  return v48;
}
struct argmax_clone_0_0__preprocessedStruct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
};
argmax_clone_0_0__preprocessedStruct argmax_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    const std::vector<Plaintext>& v4) {
  std::vector<size_t> v5 = {56, 10, 5, 35, 24, 37, 55, 48};
  std::vector<size_t> v6 = {61, 22};
  std::vector<size_t> v7 = {61, 59, 35, 22, 56, 3, 10};
  [[maybe_unused]] size_t v8 = 0;
  [[maybe_unused]] size_t v9 = 22;
  [[maybe_unused]] size_t v10 = 10;
  [[maybe_unused]] size_t v11 = 5;
  [[maybe_unused]] size_t v12 = 3;
  [[maybe_unused]] size_t v13 = 1;
  [[maybe_unused]] size_t v14 = 2;
  [[maybe_unused]] size_t v15 = 4;
  [[maybe_unused]] size_t v16 = 6;
  [[maybe_unused]] size_t v17 = 7;
  [[maybe_unused]] size_t v18 = 8;
  [[maybe_unused]] size_t v19 = 9;
  [[maybe_unused]] size_t v20 = 11;
  [[maybe_unused]] size_t v21 = 12;
  [[maybe_unused]] size_t v22 = 13;
  [[maybe_unused]] size_t v23 = 14;
  [[maybe_unused]] size_t v24 = 15;
  [[maybe_unused]] size_t v25 = 16;
  [[maybe_unused]] size_t v26 = 17;
  [[maybe_unused]] size_t v27 = 18;
  [[maybe_unused]] size_t v28 = 19;
  [[maybe_unused]] size_t v29 = 20;
  [[maybe_unused]] size_t v30 = 21;
  [[maybe_unused]] size_t v31 = 23;
  const auto& ct = v3[0];
  const auto& ct1 = v2[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  Plaintext pt = v4[0];
  const auto& ct4 = v1[0];
  auto ct5 = cc->EvalMult(ct4, pt);
  Plaintext pt1 = v4[1];
  Plaintext pt2 = v4[2];
  Plaintext pt3 = v4[3];
  std::vector<CiphertextT> v32(8);
  Plaintext pt4 = v4[4];
  Plaintext pt5 = v4[5];
  Plaintext pt6 = v4[6];
  Plaintext pt7 = v4[7];
  Plaintext pt8 = v4[8];
  const auto& ct6 = v0[0];
  auto ct7 = cc->EvalMult(ct6, pt);
  Plaintext pt9 = v4[9];
  Plaintext pt10 = v4[10];
  Plaintext pt11 = v4[11];
  Plaintext pt12 = v4[12];
  Plaintext pt13 = v4[13];
  Plaintext pt14 = v4[14];
  Plaintext pt15 = v4[15];
  Plaintext pt16 = v4[16];
  Plaintext pt17 = v4[17];
  Plaintext pt18 = v4[18];
  const auto& ct8 =
      cc->EvalFastRotation(ct2, 3, 2 * cc->GetRingDimension(), digit_decomp);
  Plaintext pt19 = v4[19];
  const auto& ct9 = cc->EvalMult(ct8, pt19);
  Plaintext pt20 = v4[20];
  Plaintext pt21 = v4[21];
  Plaintext pt22 = v4[22];
  Plaintext pt23 = v4[23];
  std::vector<CiphertextT> v33(7);
  std::vector<CiphertextT> v34(1);
  std::vector<CiphertextT> v35(2);
#pragma omp parallel for
  for (auto v37 = 0; v37 < 8; ++v37) {
    size_t v39 = v5[v37];
    const auto& ct10 = cc->EvalFastRotation(
        ct2, v39, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v40 = {ct10};
    v32[v37] = v40[0];
  }
  const auto& ct11 = v32[0];
  const auto& ct12 = v32[1];
  const auto& ct13 = v32[2];
  const auto& ct14 = v32[3];
  const auto& ct15 = v32[4];
  const auto& ct16 = v32[5];
  const auto& ct17 = v32[6];
  const auto& ct18 = v32[7];
  const auto& ct19 = cc->EvalMult(ct18, pt4);
  const auto& ct20 = cc->EvalMult(ct17, pt5);
  const auto& ct21 = cc->EvalMult(ct11, pt6);
  const auto& ct22 = cc->EvalMult(ct13, pt7);
  const auto& ct23 = cc->EvalMult(ct12, pt8);
  const auto& ct24 = cc->EvalMult(ct15, pt6);
  cc->EvalAddInPlace(ct7, ct24);
  const auto& ct26 = cc->EvalMult(ct14, pt9);
  cc->EvalAddInPlace(ct7, ct26);
  const auto& ct28 = cc->EvalMult(ct16, pt10);
  cc->EvalAddInPlace(ct7, ct28);
  const auto& ct30 = cc->EvalMult(ct17, pt11);
  cc->EvalAddInPlace(ct7, ct30);
  const auto& ct32 = cc->EvalMult(ct11, pt12);
  cc->EvalAddInPlace(ct7, ct32);
  const auto& ct34 = cc->EvalMult(ct13, pt13);
  cc->EvalAddInPlace(ct7, ct34);
  const auto& ct36 = cc->EvalMult(ct13, pt20);
  const auto& ct37 = cc->EvalMult(ct15, pt1);
  cc->EvalAddInPlace(ct5, ct37);
  const auto& ct39 = cc->EvalMult(ct14, pt2);
  cc->EvalAddInPlace(ct5, ct39);
  const auto& ct41 = cc->EvalMult(ct16, pt3);
  cc->EvalAddInPlace(ct5, ct41);
  cc->EvalAddInPlace(ct5, ct19);
  cc->EvalAddInPlace(ct5, ct20);
  cc->EvalAddInPlace(ct5, ct21);
  cc->EvalAddInPlace(ct5, ct22);
  cc->EvalAddInPlace(ct5, ct23);
  auto ct48 = cc->EvalMultNoRelin(ct5, ct7);
  cc->RelinearizeInPlace(ct48);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct48);
  const auto& ct50 = cc->EvalMult(ct48, pt17);
#pragma omp parallel for
  for (auto v42 = 0; v42 < 7; ++v42) {
    size_t v44 = v7[v42];
    const auto& ct51 = cc->EvalFastRotation(
        ct48, v44, 2 * cc->GetRingDimension(), digit_decomp1);
    const std::vector<CiphertextT> v45 = {ct51};
    v33[v42] = v45[0];
  }
  const auto& ct52 = v33[0];
  const auto& ct53 = v33[1];
  const auto& ct54 = v33[2];
  const auto& ct55 = v33[3];
  const auto& ct56 = v33[4];
  const auto& ct57 = v33[5];
  const auto& ct58 = v33[6];
  const auto& ct59 = cc->EvalMult(ct58, pt19);
  auto ct60 = cc->EvalMult(ct54, pt14);
  const auto& ct61 = cc->EvalMult(ct53, pt15);
  cc->EvalAddInPlace(ct60, ct61);
  const auto& ct63 = cc->EvalMult(ct52, pt16);
  cc->EvalAddInPlace(ct60, ct63);
  cc->EvalAddInPlace(ct60, ct50);
  const auto& ct66 = cc->EvalMult(ct55, pt18);
  cc->EvalAddInPlace(ct60, ct66);
  cc->EvalAddInPlace(ct60, ct9);
  cc->EvalAddInPlace(ct60, ct36);
  auto ct70 = cc->EvalMult(ct54, pt20);
  const auto& ct71 = cc->EvalMult(ct56, pt21);
  cc->EvalAddInPlace(ct70, ct71);
  const auto& ct73 = cc->EvalMult(ct53, pt22);
  cc->EvalAddInPlace(ct70, ct73);
  const auto& ct75 = cc->EvalMult(ct52, pt23);
  cc->EvalAddInPlace(ct70, ct75);
  const auto& ct77 = cc->EvalMult(ct57, pt14);
  cc->EvalAddInPlace(ct70, ct77);
  cc->EvalAddInPlace(ct70, ct59);
  auto ct80 = cc->EvalMultNoRelin(ct60, ct70);
  cc->RelinearizeInPlace(ct80);
  const auto& digit_decomp2 = cc->EvalFastRotationPrecompute(ct80);
  std::vector<CiphertextT> v46(v34);
  v46[0] = ct80;
  const auto& ct82 = cc->EvalMult(ct80, pt20);
  auto ct83 = cc->EvalMult(ct80, pt19);
#pragma omp parallel for
  for (auto v48 = 0; v48 < 2; ++v48) {
    size_t v50 = v6[v48];
    const auto& ct84 = cc->EvalFastRotation(
        ct80, v50, 2 * cc->GetRingDimension(), digit_decomp2);
    const std::vector<CiphertextT> v51 = {ct84};
    v35[v48] = v51[0];
  }
  const auto& ct85 = v35[0];
  const auto& ct86 = v35[1];
  const auto& ct87 = cc->EvalMult(ct86, pt20);
  cc->EvalAddInPlace(ct83, ct87);
  auto ct89 = cc->EvalMult(ct85, pt19);
  cc->EvalAddInPlace(ct89, ct82);
  auto ct91 = cc->EvalMultNoRelin(ct89, ct83);
  cc->RelinearizeInPlace(ct91);
  std::vector<CiphertextT> v52(v34);
  v52[0] = ct91;
  return {v46, v52};
}
struct argmax_clone_0_0Struct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
};
argmax_clone_0_0Struct argmax_clone_0_0(CryptoContextT cc,
                                        std::vector<CiphertextT> v0,
                                        std::vector<CiphertextT> v1,
                                        std::vector<CiphertextT> v2,
                                        std::vector<CiphertextT> v3) {
  const auto& v4 = argmax_clone_0_0__preprocessing(cc);
  auto v5Struct = argmax_clone_0_0__preprocessed(cc, v0, v1, v2, v3, v4);
  const auto& v5 = v5Struct.arg0;
  const auto& v6 = v5Struct.arg1;
  return {v5, v6};
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg0(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 62;
  [[maybe_unused]] size_t v3 = 59;
  [[maybe_unused]] size_t v4 = 46;
  [[maybe_unused]] size_t v5 = 30;
  [[maybe_unused]] size_t v6 = 27;
  [[maybe_unused]] size_t v7 = 11;
  int32_t v8 = v0[11];
  int32_t v9 = v0[27];
  int32_t v10 = v0[30];
  int32_t v11 = v0[46];
  int32_t v12 = v0[59];
  int32_t v13 = v0[62];
  const std::vector<int32_t> v14 = {
      v1, v1, v1, v1, v10, v1, v1, v1, v1, v1,  v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v9,  v1, v1, v1, v1, v1,  v11, v1, v8, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v12, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,  v13, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v10, v1, v1, v1, v1, v1,  v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v9,  v1, v1, v1, v1, v1,  v11, v1, v8, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v12, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,  v13, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v10, v1, v1, v1, v1, v1,  v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v9,  v1, v1, v1, v1, v1,  v11, v1, v8, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v12, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,  v13, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v10, v1, v1, v1, v1, v1,  v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v9,  v1, v1, v1, v1, v1,  v11, v1, v8, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v12, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,  v13, v1, v1, v1, v1, v1};
  std::vector<int32_t> v15(1 * 256);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 256; ++v15_i1) {
      v15[v15_i1 + 256 * (v15_i0)] =
          v14[0 + v15_i1 * 1 + 256 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg1(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 54;
  [[maybe_unused]] size_t v3 = 51;
  [[maybe_unused]] size_t v4 = 38;
  [[maybe_unused]] size_t v5 = 22;
  [[maybe_unused]] size_t v6 = 19;
  [[maybe_unused]] size_t v7 = 3;
  int32_t v8 = v0[3];
  int32_t v9 = v0[19];
  int32_t v10 = v0[22];
  int32_t v11 = v0[38];
  int32_t v12 = v0[51];
  int32_t v13 = v0[54];
  const std::vector<int32_t> v14 = {
      v1, v1, v1, v1, v10, v1, v1, v1, v1, v1,  v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v9,  v1, v1, v1, v1, v1,  v11, v1, v8, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v12, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,  v13, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v10, v1, v1, v1, v1, v1,  v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v9,  v1, v1, v1, v1, v1,  v11, v1, v8, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v12, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,  v13, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v10, v1, v1, v1, v1, v1,  v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v9,  v1, v1, v1, v1, v1,  v11, v1, v8, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v12, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,  v13, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v10, v1, v1, v1, v1, v1,  v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v9,  v1, v1, v1, v1, v1,  v11, v1, v8, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v12, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,  v13, v1, v1, v1, v1, v1};
  std::vector<int32_t> v15(1 * 256);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 256; ++v15_i1) {
      v15[v15_i1 + 256 * (v15_i0)] =
          v14[0 + v15_i1 * 1 + 256 * (0 + v15_i0 * 1)];
    }
  }
  std::vector<int64_t> v16(std::begin(v15), std::end(v15));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v17 = {ct};
  return v17;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg2(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 63;
  [[maybe_unused]] size_t v3 = 61;
  [[maybe_unused]] size_t v4 = 60;
  [[maybe_unused]] size_t v5 = 58;
  [[maybe_unused]] size_t v6 = 57;
  [[maybe_unused]] size_t v7 = 47;
  [[maybe_unused]] size_t v8 = 45;
  [[maybe_unused]] size_t v9 = 44;
  [[maybe_unused]] size_t v10 = 43;
  [[maybe_unused]] size_t v11 = 42;
  [[maybe_unused]] size_t v12 = 41;
  [[maybe_unused]] size_t v13 = 40;
  [[maybe_unused]] size_t v14 = 31;
  [[maybe_unused]] size_t v15 = 29;
  [[maybe_unused]] size_t v16 = 28;
  [[maybe_unused]] size_t v17 = 26;
  [[maybe_unused]] size_t v18 = 24;
  [[maybe_unused]] size_t v19 = 15;
  [[maybe_unused]] size_t v20 = 14;
  [[maybe_unused]] size_t v21 = 13;
  [[maybe_unused]] size_t v22 = 12;
  [[maybe_unused]] size_t v23 = 10;
  [[maybe_unused]] size_t v24 = 56;
  [[maybe_unused]] size_t v25 = 9;
  [[maybe_unused]] size_t v26 = 25;
  [[maybe_unused]] size_t v27 = 8;
  int32_t v28 = v0[8];
  int32_t v29 = v0[9];
  int32_t v30 = v0[10];
  int32_t v31 = v0[12];
  int32_t v32 = v0[13];
  int32_t v33 = v0[14];
  int32_t v34 = v0[15];
  int32_t v35 = v0[24];
  int32_t v36 = v0[25];
  int32_t v37 = v0[26];
  int32_t v38 = v0[28];
  int32_t v39 = v0[29];
  int32_t v40 = v0[31];
  int32_t v41 = v0[40];
  int32_t v42 = v0[41];
  int32_t v43 = v0[42];
  int32_t v44 = v0[43];
  int32_t v45 = v0[44];
  int32_t v46 = v0[45];
  int32_t v47 = v0[47];
  int32_t v48 = v0[56];
  int32_t v49 = v0[57];
  int32_t v50 = v0[58];
  int32_t v51 = v0[60];
  int32_t v52 = v0[61];
  int32_t v53 = v0[63];
  const std::vector<int32_t> v54 = {
      v1,  v1,  v41, v45, v1,  v38, v1,  v50, v1,  v30, v1,  v37, v1,  v1,  v51,
      v1,  v1,  v36, v39, v49, v1,  v42, v53, v48, v1,  v28, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v46, v44, v1,  v1,  v31, v33, v35, v47, v1,  v1,  v1,  v1,
      v1,  v43, v1,  v40, v1,  v1,  v1,  v32, v1,  v52, v1,  v29, v1,  v1,  v1,
      v1,  v1,  v34, v1,  v1,  v1,  v41, v45, v1,  v38, v1,  v50, v1,  v30, v1,
      v37, v1,  v1,  v51, v1,  v1,  v36, v39, v49, v1,  v42, v53, v48, v1,  v28,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v46, v44, v1,  v1,  v31, v33, v35, v47,
      v1,  v1,  v1,  v1,  v1,  v43, v1,  v40, v1,  v1,  v1,  v32, v1,  v52, v1,
      v29, v1,  v1,  v1,  v1,  v1,  v34, v1,  v1,  v1,  v41, v45, v1,  v38, v1,
      v50, v1,  v30, v1,  v37, v1,  v1,  v51, v1,  v1,  v36, v39, v49, v1,  v42,
      v53, v48, v1,  v28, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v46, v44, v1,  v1,
      v31, v33, v35, v47, v1,  v1,  v1,  v1,  v1,  v43, v1,  v40, v1,  v1,  v1,
      v32, v1,  v52, v1,  v29, v1,  v1,  v1,  v1,  v1,  v34, v1,  v1,  v1,  v41,
      v45, v1,  v38, v1,  v50, v1,  v30, v1,  v37, v1,  v1,  v51, v1,  v1,  v36,
      v39, v49, v1,  v42, v53, v48, v1,  v28, v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v46, v44, v1,  v1,  v31, v33, v35, v47, v1,  v1,  v1,  v1,  v1,  v43, v1,
      v40, v1,  v1,  v1,  v32, v1,  v52, v1,  v29, v1,  v1,  v1,  v1,  v1,  v34,
      v1};
  std::vector<int32_t> v55(1 * 256);
  for (int64_t v55_i0 = 0; v55_i0 < 1; ++v55_i0) {
    for (int64_t v55_i1 = 0; v55_i1 < 256; ++v55_i1) {
      v55[v55_i1 + 256 * (v55_i0)] =
          v54[0 + v55_i1 * 1 + 256 * (0 + v55_i0 * 1)];
    }
  }
  std::vector<int64_t> v56(std::begin(v55), std::end(v55));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v56;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v56[i % v56.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v57 = {ct};
  return v57;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg3(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 55;
  [[maybe_unused]] size_t v3 = 53;
  [[maybe_unused]] size_t v4 = 50;
  [[maybe_unused]] size_t v5 = 49;
  [[maybe_unused]] size_t v6 = 36;
  [[maybe_unused]] size_t v7 = 35;
  [[maybe_unused]] size_t v8 = 34;
  [[maybe_unused]] size_t v9 = 33;
  [[maybe_unused]] size_t v10 = 32;
  [[maybe_unused]] size_t v11 = 48;
  [[maybe_unused]] size_t v12 = 23;
  [[maybe_unused]] size_t v13 = 21;
  [[maybe_unused]] size_t v14 = 20;
  [[maybe_unused]] size_t v15 = 18;
  [[maybe_unused]] size_t v16 = 17;
  [[maybe_unused]] size_t v17 = 39;
  [[maybe_unused]] size_t v18 = 16;
  [[maybe_unused]] size_t v19 = 7;
  [[maybe_unused]] size_t v20 = 6;
  [[maybe_unused]] size_t v21 = 52;
  [[maybe_unused]] size_t v22 = 5;
  [[maybe_unused]] size_t v23 = 37;
  [[maybe_unused]] size_t v24 = 4;
  [[maybe_unused]] size_t v25 = 2;
  [[maybe_unused]] size_t v26 = 1;
  [[maybe_unused]] size_t v27 = 0;
  int32_t v28 = v0[0];
  int32_t v29 = v0[1];
  int32_t v30 = v0[2];
  int32_t v31 = v0[4];
  int32_t v32 = v0[5];
  int32_t v33 = v0[6];
  int32_t v34 = v0[7];
  int32_t v35 = v0[16];
  int32_t v36 = v0[17];
  int32_t v37 = v0[18];
  int32_t v38 = v0[20];
  int32_t v39 = v0[21];
  int32_t v40 = v0[23];
  int32_t v41 = v0[32];
  int32_t v42 = v0[33];
  int32_t v43 = v0[34];
  int32_t v44 = v0[35];
  int32_t v45 = v0[36];
  int32_t v46 = v0[37];
  int32_t v47 = v0[39];
  int32_t v48 = v0[48];
  int32_t v49 = v0[49];
  int32_t v50 = v0[50];
  int32_t v51 = v0[52];
  int32_t v52 = v0[53];
  int32_t v53 = v0[55];
  const std::vector<int32_t> v54 = {
      v1,  v1,  v41, v45, v1,  v38, v1,  v50, v1,  v30, v1,  v37, v1,  v1,  v51,
      v1,  v1,  v36, v39, v49, v1,  v42, v53, v48, v1,  v28, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v46, v44, v1,  v1,  v31, v33, v35, v47, v1,  v1,  v1,  v1,
      v1,  v43, v1,  v40, v1,  v1,  v1,  v32, v1,  v52, v1,  v29, v1,  v1,  v1,
      v1,  v1,  v34, v1,  v1,  v1,  v41, v45, v1,  v38, v1,  v50, v1,  v30, v1,
      v37, v1,  v1,  v51, v1,  v1,  v36, v39, v49, v1,  v42, v53, v48, v1,  v28,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v46, v44, v1,  v1,  v31, v33, v35, v47,
      v1,  v1,  v1,  v1,  v1,  v43, v1,  v40, v1,  v1,  v1,  v32, v1,  v52, v1,
      v29, v1,  v1,  v1,  v1,  v1,  v34, v1,  v1,  v1,  v41, v45, v1,  v38, v1,
      v50, v1,  v30, v1,  v37, v1,  v1,  v51, v1,  v1,  v36, v39, v49, v1,  v42,
      v53, v48, v1,  v28, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v46, v44, v1,  v1,
      v31, v33, v35, v47, v1,  v1,  v1,  v1,  v1,  v43, v1,  v40, v1,  v1,  v1,
      v32, v1,  v52, v1,  v29, v1,  v1,  v1,  v1,  v1,  v34, v1,  v1,  v1,  v41,
      v45, v1,  v38, v1,  v50, v1,  v30, v1,  v37, v1,  v1,  v51, v1,  v1,  v36,
      v39, v49, v1,  v42, v53, v48, v1,  v28, v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v46, v44, v1,  v1,  v31, v33, v35, v47, v1,  v1,  v1,  v1,  v1,  v43, v1,
      v40, v1,  v1,  v1,  v32, v1,  v52, v1,  v29, v1,  v1,  v1,  v1,  v1,  v34,
      v1};
  std::vector<int32_t> v55(1 * 256);
  for (int64_t v55_i0 = 0; v55_i0 < 1; ++v55_i0) {
    for (int64_t v55_i1 = 0; v55_i1 < 256; ++v55_i1) {
      v55[v55_i1 + 256 * (v55_i0)] =
          v54[0 + v55_i1 * 1 + 256 * (0 + v55_i0 * 1)];
    }
  }
  std::vector<int64_t> v56(std::begin(v55), std::end(v55));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v56;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v56[i % v56.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v57 = {ct};
  return v57;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 35;
  [[maybe_unused]] size_t v3 = 52;
  [[maybe_unused]] size_t v4 = 48;
  [[maybe_unused]] size_t v5 = 41;
  [[maybe_unused]] size_t v6 = 51;
  [[maybe_unused]] size_t v7 = 34;
  [[maybe_unused]] size_t v8 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v9_cast = pt->GetPackedValue();
  std::vector<int32_t> v9(std::begin(v9_cast), std::end(v9_cast));
  int32_t v10 = v9[34 + 256 * (0)];
  int32_t v11 = v9[51 + 256 * (0)];
  int32_t v12 = v9[41 + 256 * (0)];
  int32_t v13 = v9[48 + 256 * (0)];
  int32_t v14 = v9[52 + 256 * (0)];
  int32_t v15 = v9[35 + 256 * (0)];
  const std::vector<int32_t> v16 = {v10, v11, v12, v1, v13, v14, v1, v15};
  return v16;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result1(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 33;
  [[maybe_unused]] size_t v3 = 31;
  [[maybe_unused]] size_t v4 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v5_cast = pt->GetPackedValue();
  std::vector<int32_t> v5(std::begin(v5_cast), std::end(v5_cast));
  int32_t v6 = v5[31 + 256 * (0)];
  int32_t v7 = v5[33 + 256 * (0)];
  const std::vector<int32_t> v8 = {v1, v1, v1, v6, v1, v1, v7, v1};
  return v8;
}
CryptoContextT argmax_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(7);
  params.SetPlaintextModulus(65537);
  params.SetKeySwitchTechnique(HYBRID);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT argmax_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                          PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {59, 35, 61, 56, 37, 22, 3, 48, 10, 55, 24, 5});
  return cc;
}
