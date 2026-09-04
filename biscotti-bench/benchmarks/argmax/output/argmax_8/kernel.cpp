
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
  std::vector<int64_t> v20 = {
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
  std::vector<int64_t> v21 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  std::vector<int64_t> v22 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0};
  std::vector<int64_t> v23 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  std::vector<int64_t> v24 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v25 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v26 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0};
  std::vector<int64_t> v27 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v28 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v29 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v30 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0};
  std::vector<int64_t> v31 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  std::vector<int64_t> v32 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0};
  std::vector<int64_t> v33 = {
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
  std::vector<int64_t> v34 = {
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
  std::vector<int64_t> v35 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0};
  std::vector<int64_t> v36 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  std::vector<int64_t> v37 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
  std::vector<int64_t> v38 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v39 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0};
  std::vector<Plaintext> v40(20);
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v20;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v20[i % v20.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  v40[0] = pt;
  auto pt1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v21;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (unsigned i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v21[i % v21.size()]);
  }
  auto pt1 = cc->MakePackedPlaintext(pt1_filled);
  v40[1] = pt1;
  auto pt2_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt2_filled = v22;
  pt2_filled.clear();
  pt2_filled.reserve(pt2_filled_n);
  for (unsigned i = 0; i < pt2_filled_n; ++i) {
    pt2_filled.push_back(v22[i % v22.size()]);
  }
  auto pt2 = cc->MakePackedPlaintext(pt2_filled);
  v40[2] = pt2;
  auto pt3_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt3_filled = v23;
  pt3_filled.clear();
  pt3_filled.reserve(pt3_filled_n);
  for (unsigned i = 0; i < pt3_filled_n; ++i) {
    pt3_filled.push_back(v23[i % v23.size()]);
  }
  auto pt3 = cc->MakePackedPlaintext(pt3_filled);
  v40[3] = pt3;
  auto pt4_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt4_filled = v24;
  pt4_filled.clear();
  pt4_filled.reserve(pt4_filled_n);
  for (unsigned i = 0; i < pt4_filled_n; ++i) {
    pt4_filled.push_back(v24[i % v24.size()]);
  }
  auto pt4 = cc->MakePackedPlaintext(pt4_filled);
  v40[4] = pt4;
  auto pt5_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt5_filled = v25;
  pt5_filled.clear();
  pt5_filled.reserve(pt5_filled_n);
  for (unsigned i = 0; i < pt5_filled_n; ++i) {
    pt5_filled.push_back(v25[i % v25.size()]);
  }
  auto pt5 = cc->MakePackedPlaintext(pt5_filled);
  v40[5] = pt5;
  auto pt6_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt6_filled = v26;
  pt6_filled.clear();
  pt6_filled.reserve(pt6_filled_n);
  for (unsigned i = 0; i < pt6_filled_n; ++i) {
    pt6_filled.push_back(v26[i % v26.size()]);
  }
  auto pt6 = cc->MakePackedPlaintext(pt6_filled);
  v40[6] = pt6;
  auto pt7_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt7_filled = v27;
  pt7_filled.clear();
  pt7_filled.reserve(pt7_filled_n);
  for (unsigned i = 0; i < pt7_filled_n; ++i) {
    pt7_filled.push_back(v27[i % v27.size()]);
  }
  auto pt7 = cc->MakePackedPlaintext(pt7_filled);
  v40[7] = pt7;
  auto pt8_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt8_filled = v28;
  pt8_filled.clear();
  pt8_filled.reserve(pt8_filled_n);
  for (unsigned i = 0; i < pt8_filled_n; ++i) {
    pt8_filled.push_back(v28[i % v28.size()]);
  }
  auto pt8 = cc->MakePackedPlaintext(pt8_filled);
  v40[8] = pt8;
  auto pt9_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt9_filled = v29;
  pt9_filled.clear();
  pt9_filled.reserve(pt9_filled_n);
  for (unsigned i = 0; i < pt9_filled_n; ++i) {
    pt9_filled.push_back(v29[i % v29.size()]);
  }
  auto pt9 = cc->MakePackedPlaintext(pt9_filled);
  v40[9] = pt9;
  auto pt10_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt10_filled = v30;
  pt10_filled.clear();
  pt10_filled.reserve(pt10_filled_n);
  for (unsigned i = 0; i < pt10_filled_n; ++i) {
    pt10_filled.push_back(v30[i % v30.size()]);
  }
  auto pt10 = cc->MakePackedPlaintext(pt10_filled);
  v40[10] = pt10;
  auto pt11_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt11_filled = v31;
  pt11_filled.clear();
  pt11_filled.reserve(pt11_filled_n);
  for (unsigned i = 0; i < pt11_filled_n; ++i) {
    pt11_filled.push_back(v31[i % v31.size()]);
  }
  auto pt11 = cc->MakePackedPlaintext(pt11_filled);
  v40[11] = pt11;
  auto pt12_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt12_filled = v32;
  pt12_filled.clear();
  pt12_filled.reserve(pt12_filled_n);
  for (unsigned i = 0; i < pt12_filled_n; ++i) {
    pt12_filled.push_back(v32[i % v32.size()]);
  }
  auto pt12 = cc->MakePackedPlaintext(pt12_filled);
  v40[12] = pt12;
  auto pt13_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt13_filled = v33;
  pt13_filled.clear();
  pt13_filled.reserve(pt13_filled_n);
  for (unsigned i = 0; i < pt13_filled_n; ++i) {
    pt13_filled.push_back(v33[i % v33.size()]);
  }
  auto pt13 = cc->MakePackedPlaintext(pt13_filled);
  v40[13] = pt13;
  auto pt14_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt14_filled = v34;
  pt14_filled.clear();
  pt14_filled.reserve(pt14_filled_n);
  for (unsigned i = 0; i < pt14_filled_n; ++i) {
    pt14_filled.push_back(v34[i % v34.size()]);
  }
  auto pt14 = cc->MakePackedPlaintext(pt14_filled);
  v40[14] = pt14;
  auto pt15_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt15_filled = v35;
  pt15_filled.clear();
  pt15_filled.reserve(pt15_filled_n);
  for (unsigned i = 0; i < pt15_filled_n; ++i) {
    pt15_filled.push_back(v35[i % v35.size()]);
  }
  auto pt15 = cc->MakePackedPlaintext(pt15_filled);
  v40[15] = pt15;
  auto pt16_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt16_filled = v36;
  pt16_filled.clear();
  pt16_filled.reserve(pt16_filled_n);
  for (unsigned i = 0; i < pt16_filled_n; ++i) {
    pt16_filled.push_back(v36[i % v36.size()]);
  }
  auto pt16 = cc->MakePackedPlaintext(pt16_filled);
  v40[16] = pt16;
  auto pt17_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt17_filled = v37;
  pt17_filled.clear();
  pt17_filled.reserve(pt17_filled_n);
  for (unsigned i = 0; i < pt17_filled_n; ++i) {
    pt17_filled.push_back(v37[i % v37.size()]);
  }
  auto pt17 = cc->MakePackedPlaintext(pt17_filled);
  v40[17] = pt17;
  auto pt18_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt18_filled = v38;
  pt18_filled.clear();
  pt18_filled.reserve(pt18_filled_n);
  for (unsigned i = 0; i < pt18_filled_n; ++i) {
    pt18_filled.push_back(v38[i % v38.size()]);
  }
  auto pt18 = cc->MakePackedPlaintext(pt18_filled);
  v40[18] = pt18;
  auto pt19_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt19_filled = v39;
  pt19_filled.clear();
  pt19_filled.reserve(pt19_filled_n);
  for (unsigned i = 0; i < pt19_filled_n; ++i) {
    pt19_filled.push_back(v39[i % v39.size()]);
  }
  auto pt19 = cc->MakePackedPlaintext(pt19_filled);
  v40[19] = pt19;
  return v40;
}
struct argmax_clone_0_0__preprocessedStruct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
};
argmax_clone_0_0__preprocessedStruct argmax_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5,
    std::vector<CiphertextT> v6, std::vector<CiphertextT> v7,
    std::vector<CiphertextT> v8, std::vector<CiphertextT> v9,
    const std::vector<Plaintext>& v10) {
  std::vector<size_t> v11 = {40, 57};
  std::vector<size_t> v12 = {33, 40};
  std::vector<size_t> v13 = {57, 9};
  [[maybe_unused]] size_t v14 = 0;
  [[maybe_unused]] size_t v15 = 9;
  [[maybe_unused]] size_t v16 = 33;
  [[maybe_unused]] size_t v17 = 57;
  [[maybe_unused]] size_t v18 = 1;
  [[maybe_unused]] size_t v19 = 2;
  [[maybe_unused]] size_t v20 = 3;
  [[maybe_unused]] size_t v21 = 4;
  [[maybe_unused]] size_t v22 = 5;
  [[maybe_unused]] size_t v23 = 6;
  [[maybe_unused]] size_t v24 = 7;
  [[maybe_unused]] size_t v25 = 8;
  [[maybe_unused]] size_t v26 = 10;
  [[maybe_unused]] size_t v27 = 11;
  [[maybe_unused]] size_t v28 = 12;
  [[maybe_unused]] size_t v29 = 13;
  [[maybe_unused]] size_t v30 = 14;
  [[maybe_unused]] size_t v31 = 15;
  [[maybe_unused]] size_t v32 = 16;
  [[maybe_unused]] size_t v33 = 17;
  [[maybe_unused]] size_t v34 = 18;
  [[maybe_unused]] size_t v35 = 19;
  const auto& ct = v9[0];
  const auto& ct1 = v8[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  const auto& ct4 = v7[0];
  const auto& ct5 = v0[0];
  auto ct6 = cc->EvalMultNoRelin(ct4, ct5);
  cc->RelinearizeInPlace(ct6);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct6);
  const auto& ct8 =
      cc->EvalFastRotation(ct6, 57, 2 * cc->GetRingDimension(), digit_decomp1);
  Plaintext pt = v10[0];
  auto ct9 = cc->EvalMult(ct8, pt);
  Plaintext pt1 = v10[1];
  const auto& ct10 = cc->EvalMult(ct6, pt1);
  cc->EvalAddInPlace(ct9, ct10);
  Plaintext pt2 = v10[2];
  const auto& ct12 = v6[0];
  const auto& ct13 = cc->EvalMult(ct12, pt2);
  cc->EvalAddInPlace(ct9, ct13);
  const auto& ct15 = v1[0];
  auto ct16 = cc->EvalMult(ct15, pt2);
  Plaintext pt3 = v10[3];
  Plaintext pt4 = v10[4];
  const auto& ct17 = cc->EvalMult(ct2, pt4);
  std::vector<CiphertextT> v36(2);
  auto v37 = v36;
#pragma omp parallel for
  for (auto v38 = 0; v38 < 2; ++v38) {
    size_t v40 = v13[v38];
    const auto& ct18 = cc->EvalFastRotation(
        ct2, v40, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v41 = {ct18};
    v37[v38] = v41[0];
  }
  const auto& ct19 = v37[0];
  const auto& ct20 = v37[1];
  Plaintext pt5 = v10[5];
  const auto& ct21 = cc->EvalMult(ct20, pt5);
  Plaintext pt6 = v10[6];
  const auto& ct22 = v5[0];
  auto ct23 = cc->EvalMult(ct22, pt6);
  Plaintext pt7 = v10[7];
  const auto& ct24 = cc->EvalMult(ct2, pt7);
  cc->EvalAddInPlace(ct23, ct24);
  auto ct26 = cc->EvalMult(ct6, pt7);
  const auto& ct27 = v2[0];
  const auto& ct28 = cc->EvalMult(ct27, pt6);
  cc->EvalAddInPlace(ct26, ct28);
  auto ct30 = cc->EvalMultNoRelin(ct23, ct26);
  cc->RelinearizeInPlace(ct30);
  const auto& ct32 = v4[0];
  auto ct33 = cc->EvalMult(ct32, pt7);
  const auto& ct34 =
      cc->EvalFastRotation(ct6, 9, 2 * cc->GetRingDimension(), digit_decomp1);
  const auto& ct35 = cc->EvalMult(ct34, pt);
  cc->EvalAddInPlace(ct33, ct35);
  const auto& ct37 = cc->EvalMult(ct30, pt3);
  cc->EvalAddInPlace(ct33, ct37);
  Plaintext pt8 = v10[8];
  auto v42 = v36;
#pragma omp parallel for
  for (auto v43 = 0; v43 < 2; ++v43) {
    size_t v45 = v12[v43];
    const auto& ct39 = cc->EvalFastRotation(
        ct2, v45, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v46 = {ct39};
    v42[v43] = v46[0];
  }
  const auto& ct40 = v42[0];
  const auto& ct41 = v42[1];
  Plaintext pt9 = v10[9];
  const auto& ct42 = cc->EvalMult(ct41, pt9);
  Plaintext pt10 = v10[10];
  const auto& ct43 = cc->EvalMult(ct2, pt10);
  const auto& ct44 = cc->EvalRotate(ct30, 33);
  Plaintext pt11 = v10[11];
  auto ct45 = cc->EvalMult(ct44, pt11);
  Plaintext pt12 = v10[12];
  const auto& ct46 = cc->EvalMult(ct30, pt12);
  cc->EvalAddInPlace(ct45, ct46);
  const auto& ct48 = v3[0];
  const auto& ct49 = cc->EvalMult(ct48, pt7);
  cc->EvalAddInPlace(ct45, ct49);
  Plaintext pt13 = v10[13];
  Plaintext pt14 = v10[14];
  Plaintext pt15 = v10[15];
  Plaintext pt16 = v10[16];
  Plaintext pt17 = v10[17];
  Plaintext pt18 = v10[18];
  std::vector<CiphertextT> v47(1);
  Plaintext pt19 = v10[19];
  auto ct51 = cc->EvalMult(ct30, pt7);
  const auto& ct52 = cc->EvalMult(ct19, pt3);
  cc->EvalAddInPlace(ct16, ct52);
  cc->EvalAddInPlace(ct16, ct17);
  cc->EvalAddInPlace(ct16, ct21);
  auto ct56 = cc->EvalMultNoRelin(ct9, ct16);
  cc->RelinearizeInPlace(ct56);
  const auto& digit_decomp2 = cc->EvalFastRotationPrecompute(ct56);
  const auto& ct58 = cc->EvalMult(ct40, pt8);
  cc->EvalAddInPlace(ct33, ct58);
  cc->EvalAddInPlace(ct33, ct42);
  cc->EvalAddInPlace(ct33, ct43);
  auto ct62 = cc->EvalMultNoRelin(ct33, ct45);
  cc->RelinearizeInPlace(ct62);
  auto ct64 = cc->EvalMult(ct62, pt13);
  const auto& ct65 = cc->EvalMult(ct56, pt15);
  auto ct66 = cc->EvalMult(ct62, pt16);
  auto v48 = v36;
#pragma omp parallel for
  for (auto v49 = 0; v49 < 2; ++v49) {
    size_t v51 = v11[v49];
    const auto& ct67 = cc->EvalFastRotation(
        ct56, v51, 2 * cc->GetRingDimension(), digit_decomp2);
    const std::vector<CiphertextT> v52 = {ct67};
    v48[v49] = v52[0];
  }
  const auto& ct68 = v48[0];
  const auto& ct69 = v48[1];
  const auto& ct70 = cc->EvalMult(ct69, pt17);
  cc->EvalAddInPlace(ct66, ct70);
  const auto& ct72 = cc->EvalMult(ct56, pt18);
  cc->EvalAddInPlace(ct66, ct72);
  auto ct74 = cc->EvalMult(ct62, pt19);
  const auto& ct75 = cc->EvalMult(ct68, pt14);
  cc->EvalAddInPlace(ct64, ct75);
  cc->EvalAddInPlace(ct64, ct65);
  auto ct78 = cc->EvalMultNoRelin(ct64, ct66);
  cc->RelinearizeInPlace(ct78);
  std::vector<CiphertextT> v53(v47);
  v53[0] = ct78;
  const auto& ct80 = cc->EvalMult(ct78, pt7);
  cc->EvalAddInPlace(ct74, ct80);
  const auto& ct82 = cc->EvalMult(ct78, pt19);
  cc->EvalAddInPlace(ct51, ct82);
  auto ct84 = cc->EvalMultNoRelin(ct74, ct51);
  cc->RelinearizeInPlace(ct84);
  std::vector<CiphertextT> v54(v47);
  v54[0] = ct84;
  return {v53, v54};
}
struct argmax_clone_0_0Struct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
};
argmax_clone_0_0Struct argmax_clone_0_0(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5,
    std::vector<CiphertextT> v6, std::vector<CiphertextT> v7,
    std::vector<CiphertextT> v8, std::vector<CiphertextT> v9) {
  const auto& v10 = argmax_clone_0_0__preprocessing(cc);
  auto v11Struct = argmax_clone_0_0__preprocessed(cc, v0, v1, v2, v3, v4, v5,
                                                  v6, v7, v8, v9, v10);
  const auto& v11 = v11Struct.arg0;
  const auto& v12 = v11Struct.arg1;
  return {v11, v12};
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg0(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 56;
  [[maybe_unused]] size_t v3 = 42;
  [[maybe_unused]] size_t v4 = 15;
  [[maybe_unused]] size_t v5 = 44;
  [[maybe_unused]] size_t v6 = 14;
  [[maybe_unused]] size_t v7 = 13;
  [[maybe_unused]] size_t v8 = 10;
  int32_t v9 = v0[10];
  int32_t v10 = v0[13];
  int32_t v11 = v0[14];
  int32_t v12 = v0[15];
  int32_t v13 = v0[42];
  int32_t v14 = v0[44];
  int32_t v15 = v0[56];
  const std::vector<int32_t> v16 = {
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v15, v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v9,  v1,  v1, v11, v1,  v12, v1,
      v1, v14, v1, v1, v1, v1, v1, v1, v1, v13, v10, v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v15, v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v9,  v1,  v1, v11, v1,  v12, v1,
      v1, v14, v1, v1, v1, v1, v1, v1, v1, v13, v10, v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v15, v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v9,  v1,  v1, v11, v1,  v12, v1,
      v1, v14, v1, v1, v1, v1, v1, v1, v1, v13, v10, v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v15, v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v9,  v1,  v1, v11, v1,  v12, v1,
      v1, v14, v1, v1, v1, v1, v1, v1, v1, v13, v10, v1, v1,  v1,  v1,  v1};
  std::vector<int32_t> v17(1 * 256);
  for (int64_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
    for (int64_t v17_i1 = 0; v17_i1 < 256; ++v17_i1) {
      v17[v17_i1 + 256 * (v17_i0)] =
          v16[0 + v17_i1 * 1 + 256 * (0 + v17_i0 * 1)];
    }
  }
  std::vector<int64_t> v18(std::begin(v17), std::end(v17));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v18;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v18[i % v18.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v19 = {ct};
  return v19;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg1(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 59;
  [[maybe_unused]] size_t v3 = 57;
  [[maybe_unused]] size_t v4 = 43;
  [[maybe_unused]] size_t v5 = 41;
  [[maybe_unused]] size_t v6 = 8;
  int32_t v7 = v0[8];
  int32_t v8 = v0[41];
  int32_t v9 = v0[43];
  int32_t v10 = v0[57];
  int32_t v11 = v0[59];
  const std::vector<int32_t> v12 = {
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v7, v1, v1,
      v1, v8, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v11, v1, v1, v1, v10, v1, v1, v9, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v7, v1, v1,
      v1, v8, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v11, v1, v1, v1, v10, v1, v1, v9, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v7, v1, v1,
      v1, v8, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v11, v1, v1, v1, v10, v1, v1, v9, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v7, v1, v1,
      v1, v8, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v11, v1, v1, v1, v10, v1, v1, v9, v1, v1, v1};
  std::vector<int32_t> v13(1 * 256);
  for (int64_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
    for (int64_t v13_i1 = 0; v13_i1 < 256; ++v13_i1) {
      v13[v13_i1 + 256 * (v13_i0)] =
          v12[0 + v13_i1 * 1 + 256 * (0 + v13_i0 * 1)];
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg2(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 63;
  [[maybe_unused]] size_t v3 = 62;
  [[maybe_unused]] size_t v4 = 61;
  [[maybe_unused]] size_t v5 = 58;
  [[maybe_unused]] size_t v6 = 45;
  [[maybe_unused]] size_t v7 = 28;
  [[maybe_unused]] size_t v8 = 27;
  [[maybe_unused]] size_t v9 = 25;
  int32_t v10 = v0[25];
  int32_t v11 = v0[27];
  int32_t v12 = v0[28];
  int32_t v13 = v0[45];
  int32_t v14 = v0[58];
  int32_t v15 = v0[61];
  int32_t v16 = v0[62];
  int32_t v17 = v0[63];
  const std::vector<int32_t> v18 = {
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v1,  v1, v1,  v1,
      v1,  v1, v12, v1, v1, v1, v1, v1, v1, v1,  v1,  v15, v1,  v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v16, v1, v17, v1,
      v14, v1, v1,  v1, v1, v1, v1, v1, v1, v10, v13, v1,  v11, v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v1,  v1, v1,  v1,
      v1,  v1, v12, v1, v1, v1, v1, v1, v1, v1,  v1,  v15, v1,  v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v16, v1, v17, v1,
      v14, v1, v1,  v1, v1, v1, v1, v1, v1, v10, v13, v1,  v11, v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v1,  v1, v1,  v1,
      v1,  v1, v12, v1, v1, v1, v1, v1, v1, v1,  v1,  v15, v1,  v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v16, v1, v17, v1,
      v14, v1, v1,  v1, v1, v1, v1, v1, v1, v10, v13, v1,  v11, v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v1,  v1, v1,  v1,
      v1,  v1, v12, v1, v1, v1, v1, v1, v1, v1,  v1,  v15, v1,  v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v16, v1, v17, v1,
      v14, v1, v1,  v1, v1, v1, v1, v1, v1, v10, v13, v1,  v11, v1, v1,  v1};
  std::vector<int32_t> v19(1 * 256);
  for (int64_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
    for (int64_t v19_i1 = 0; v19_i1 < 256; ++v19_i1) {
      v19[v19_i1 + 256 * (v19_i0)] =
          v18[0 + v19_i1 * 1 + 256 * (0 + v19_i0 * 1)];
    }
  }
  std::vector<int64_t> v20(std::begin(v19), std::end(v19));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v20;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v20[i % v20.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v21 = {ct};
  return v21;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg3(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 24;
  int32_t v3 = v0[24];
  const std::vector<int32_t> v4 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1};
  std::vector<int32_t> v5(1 * 256);
  for (int64_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
    for (int64_t v5_i1 = 0; v5_i1 < 256; ++v5_i1) {
      v5[v5_i1 + 256 * (v5_i0)] = v4[0 + v5_i1 * 1 + 256 * (0 + v5_i0 * 1)];
    }
  }
  std::vector<int64_t> v6(std::begin(v5), std::end(v5));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v6;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v6[i % v6.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v7 = {ct};
  return v7;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg4(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 16;
  int32_t v3 = v0[16];
  const std::vector<int32_t> v4 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1};
  std::vector<int32_t> v5(1 * 256);
  for (int64_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
    for (int64_t v5_i1 = 0; v5_i1 < 256; ++v5_i1) {
      v5[v5_i1 + 256 * (v5_i0)] = v4[0 + v5_i1 * 1 + 256 * (0 + v5_i0 * 1)];
    }
  }
  std::vector<int64_t> v6(std::begin(v5), std::end(v5));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v6;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v6[i % v6.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v7 = {ct};
  return v7;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg5(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 55;
  [[maybe_unused]] size_t v3 = 54;
  [[maybe_unused]] size_t v4 = 53;
  [[maybe_unused]] size_t v5 = 50;
  [[maybe_unused]] size_t v6 = 37;
  [[maybe_unused]] size_t v7 = 20;
  [[maybe_unused]] size_t v8 = 19;
  [[maybe_unused]] size_t v9 = 17;
  int32_t v10 = v0[17];
  int32_t v11 = v0[19];
  int32_t v12 = v0[20];
  int32_t v13 = v0[37];
  int32_t v14 = v0[50];
  int32_t v15 = v0[53];
  int32_t v16 = v0[54];
  int32_t v17 = v0[55];
  const std::vector<int32_t> v18 = {
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v1,  v1, v1,  v1,
      v1,  v1, v12, v1, v1, v1, v1, v1, v1, v1,  v1,  v15, v1,  v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v16, v1, v17, v1,
      v14, v1, v1,  v1, v1, v1, v1, v1, v1, v10, v13, v1,  v11, v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v1,  v1, v1,  v1,
      v1,  v1, v12, v1, v1, v1, v1, v1, v1, v1,  v1,  v15, v1,  v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v16, v1, v17, v1,
      v14, v1, v1,  v1, v1, v1, v1, v1, v1, v10, v13, v1,  v11, v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v1,  v1, v1,  v1,
      v1,  v1, v12, v1, v1, v1, v1, v1, v1, v1,  v1,  v15, v1,  v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v16, v1, v17, v1,
      v14, v1, v1,  v1, v1, v1, v1, v1, v1, v10, v13, v1,  v11, v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v1,  v1, v1,  v1,
      v1,  v1, v12, v1, v1, v1, v1, v1, v1, v1,  v1,  v15, v1,  v1, v1,  v1,
      v1,  v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1,  v1,  v16, v1, v17, v1,
      v14, v1, v1,  v1, v1, v1, v1, v1, v1, v10, v13, v1,  v11, v1, v1,  v1};
  std::vector<int32_t> v19(1 * 256);
  for (int64_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
    for (int64_t v19_i1 = 0; v19_i1 < 256; ++v19_i1) {
      v19[v19_i1 + 256 * (v19_i0)] =
          v18[0 + v19_i1 * 1 + 256 * (0 + v19_i0 * 1)];
    }
  }
  std::vector<int64_t> v20(std::begin(v19), std::end(v19));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v20;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v20[i % v20.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v21 = {ct};
  return v21;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg6(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 51;
  [[maybe_unused]] size_t v3 = 49;
  [[maybe_unused]] size_t v4 = 35;
  [[maybe_unused]] size_t v5 = 33;
  [[maybe_unused]] size_t v6 = 0;
  int32_t v7 = v0[0];
  int32_t v8 = v0[33];
  int32_t v9 = v0[35];
  int32_t v10 = v0[49];
  int32_t v11 = v0[51];
  const std::vector<int32_t> v12 = {
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v7, v1, v1,
      v1, v8, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v11, v1, v1, v1, v10, v1, v1, v9, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v7, v1, v1,
      v1, v8, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v11, v1, v1, v1, v10, v1, v1, v9, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v7, v1, v1,
      v1, v8, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v11, v1, v1, v1, v10, v1, v1, v9, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v7, v1, v1,
      v1, v8, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v11, v1, v1, v1, v10, v1, v1, v9, v1, v1, v1};
  std::vector<int32_t> v13(1 * 256);
  for (int64_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
    for (int64_t v13_i1 = 0; v13_i1 < 256; ++v13_i1) {
      v13[v13_i1 + 256 * (v13_i0)] =
          v12[0 + v13_i1 * 1 + 256 * (0 + v13_i0 * 1)];
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg7(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 48;
  [[maybe_unused]] size_t v3 = 36;
  [[maybe_unused]] size_t v4 = 34;
  [[maybe_unused]] size_t v5 = 7;
  [[maybe_unused]] size_t v6 = 6;
  [[maybe_unused]] size_t v7 = 5;
  [[maybe_unused]] size_t v8 = 2;
  int32_t v9 = v0[2];
  int32_t v10 = v0[5];
  int32_t v11 = v0[6];
  int32_t v12 = v0[7];
  int32_t v13 = v0[34];
  int32_t v14 = v0[36];
  int32_t v15 = v0[48];
  const std::vector<int32_t> v16 = {
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v15, v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v9,  v1,  v1, v11, v1,  v12, v1,
      v1, v14, v1, v1, v1, v1, v1, v1, v1, v13, v10, v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v15, v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v9,  v1,  v1, v11, v1,  v12, v1,
      v1, v14, v1, v1, v1, v1, v1, v1, v1, v13, v10, v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v15, v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v9,  v1,  v1, v11, v1,  v12, v1,
      v1, v14, v1, v1, v1, v1, v1, v1, v1, v13, v10, v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v1,  v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v1,  v1,  v1, v1,  v15, v1,  v1,
      v1, v1,  v1, v1, v1, v1, v1, v1, v1, v9,  v1,  v1, v11, v1,  v12, v1,
      v1, v14, v1, v1, v1, v1, v1, v1, v1, v13, v10, v1, v1,  v1,  v1,  v1};
  std::vector<int32_t> v17(1 * 256);
  for (int64_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
    for (int64_t v17_i1 = 0; v17_i1 < 256; ++v17_i1) {
      v17[v17_i1 + 256 * (v17_i0)] =
          v16[0 + v17_i1 * 1 + 256 * (0 + v17_i0 * 1)];
    }
  }
  std::vector<int64_t> v18(std::begin(v17), std::end(v17));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v18;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v18[i % v18.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v19 = {ct};
  return v19;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg8(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 47;
  [[maybe_unused]] size_t v3 = 46;
  [[maybe_unused]] size_t v4 = 40;
  [[maybe_unused]] size_t v5 = 31;
  [[maybe_unused]] size_t v6 = 30;
  [[maybe_unused]] size_t v7 = 29;
  [[maybe_unused]] size_t v8 = 26;
  [[maybe_unused]] size_t v9 = 12;
  [[maybe_unused]] size_t v10 = 60;
  [[maybe_unused]] size_t v11 = 11;
  [[maybe_unused]] size_t v12 = 9;
  int32_t v13 = v0[9];
  int32_t v14 = v0[11];
  int32_t v15 = v0[12];
  int32_t v16 = v0[26];
  int32_t v17 = v0[29];
  int32_t v18 = v0[30];
  int32_t v19 = v0[31];
  int32_t v20 = v0[40];
  int32_t v21 = v0[46];
  int32_t v22 = v0[47];
  int32_t v23 = v0[60];
  const std::vector<int32_t> v24 = {
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v1,  v21, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v22, v1,  v1, v1,  v1,  v1, v1,  v20, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v18, v1,  v1, v1,
      v16, v15, v1, v17, v1, v1, v1,  v19, v1, v13, v23, v1, v14, v1,  v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v1,  v21, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v22, v1,  v1, v1,  v1,  v1, v1,  v20, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v18, v1,  v1, v1,
      v16, v15, v1, v17, v1, v1, v1,  v19, v1, v13, v23, v1, v14, v1,  v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v1,  v21, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v22, v1,  v1, v1,  v1,  v1, v1,  v20, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v18, v1,  v1, v1,
      v16, v15, v1, v17, v1, v1, v1,  v19, v1, v13, v23, v1, v14, v1,  v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v1,  v21, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v22, v1,  v1, v1,  v1,  v1, v1,  v20, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v18, v1,  v1, v1,
      v16, v15, v1, v17, v1, v1, v1,  v19, v1, v13, v23, v1, v14, v1,  v1, v1};
  std::vector<int32_t> v25(1 * 256);
  for (int64_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
    for (int64_t v25_i1 = 0; v25_i1 < 256; ++v25_i1) {
      v25[v25_i1 + 256 * (v25_i0)] =
          v24[0 + v25_i1 * 1 + 256 * (0 + v25_i0 * 1)];
    }
  }
  std::vector<int64_t> v26(std::begin(v25), std::end(v25));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v26;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v26[i % v26.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v27 = {ct};
  return v27;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg9(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 52;
  [[maybe_unused]] size_t v3 = 39;
  [[maybe_unused]] size_t v4 = 38;
  [[maybe_unused]] size_t v5 = 32;
  [[maybe_unused]] size_t v6 = 23;
  [[maybe_unused]] size_t v7 = 22;
  [[maybe_unused]] size_t v8 = 21;
  [[maybe_unused]] size_t v9 = 18;
  [[maybe_unused]] size_t v10 = 4;
  [[maybe_unused]] size_t v11 = 3;
  [[maybe_unused]] size_t v12 = 1;
  int32_t v13 = v0[1];
  int32_t v14 = v0[3];
  int32_t v15 = v0[4];
  int32_t v16 = v0[18];
  int32_t v17 = v0[21];
  int32_t v18 = v0[22];
  int32_t v19 = v0[23];
  int32_t v20 = v0[32];
  int32_t v21 = v0[38];
  int32_t v22 = v0[39];
  int32_t v23 = v0[52];
  const std::vector<int32_t> v24 = {
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v1,  v21, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v22, v1,  v1, v1,  v1,  v1, v1,  v20, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v18, v1,  v1, v1,
      v16, v15, v1, v17, v1, v1, v1,  v19, v1, v13, v23, v1, v14, v1,  v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v1,  v21, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v22, v1,  v1, v1,  v1,  v1, v1,  v20, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v18, v1,  v1, v1,
      v16, v15, v1, v17, v1, v1, v1,  v19, v1, v13, v23, v1, v14, v1,  v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v1,  v21, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v22, v1,  v1, v1,  v1,  v1, v1,  v20, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v18, v1,  v1, v1,
      v16, v15, v1, v17, v1, v1, v1,  v19, v1, v13, v23, v1, v14, v1,  v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v1,  v21, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v22, v1,  v1, v1,  v1,  v1, v1,  v20, v1, v1,
      v1,  v1,  v1, v1,  v1, v1, v1,  v1,  v1, v1,  v1,  v1, v18, v1,  v1, v1,
      v16, v15, v1, v17, v1, v1, v1,  v19, v1, v13, v23, v1, v14, v1,  v1, v1};
  std::vector<int32_t> v25(1 * 256);
  for (int64_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
    for (int64_t v25_i1 = 0; v25_i1 < 256; ++v25_i1) {
      v25[v25_i1 + 256 * (v25_i0)] =
          v24[0 + v25_i1 * 1 + 256 * (0 + v25_i0 * 1)];
    }
  }
  std::vector<int64_t> v26(std::begin(v25), std::end(v25));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v26;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v26[i % v26.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v27 = {ct};
  return v27;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 46;
  [[maybe_unused]] size_t v3 = 44;
  [[maybe_unused]] size_t v4 = 58;
  [[maybe_unused]] size_t v5 = 49;
  [[maybe_unused]] size_t v6 = 48;
  [[maybe_unused]] size_t v7 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v8_cast = pt->GetPackedValue();
  std::vector<int32_t> v8(std::begin(v8_cast), std::end(v8_cast));
  int32_t v9 = v8[48 + 256 * (0)];
  int32_t v10 = v8[49 + 256 * (0)];
  int32_t v11 = v8[58 + 256 * (0)];
  int32_t v12 = v8[44 + 256 * (0)];
  int32_t v13 = v8[46 + 256 * (0)];
  const std::vector<int32_t> v14 = {v1, v1, v9, v1, v10, v11, v12, v13};
  return v14;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result1(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 60;
  [[maybe_unused]] size_t v3 = 57;
  [[maybe_unused]] size_t v4 = 29;
  [[maybe_unused]] size_t v5 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v6_cast = pt->GetPackedValue();
  std::vector<int32_t> v6(std::begin(v6_cast), std::end(v6_cast));
  int32_t v7 = v6[29 + 256 * (0)];
  int32_t v8 = v6[57 + 256 * (0)];
  int32_t v9 = v6[60 + 256 * (0)];
  const std::vector<int32_t> v10 = {v7, v8, v1, v9, v1, v1, v1, v1};
  return v10;
}
CryptoContextT argmax_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(9);
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
  cc->EvalRotateKeyGen(sk, {33, 40, 9, 57});
  return cc;
}
