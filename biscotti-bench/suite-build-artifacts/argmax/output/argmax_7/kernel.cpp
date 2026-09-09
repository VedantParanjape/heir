
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
  std::vector<int64_t> v19 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v20 = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v21 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v22 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v23 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v24 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v25 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v26 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v27 = {
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v28 = {
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v29 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v30 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v31 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v32 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v33 = {
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
  std::vector<int64_t> v34 = {
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
  std::vector<int64_t> v35 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v36 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v37 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<Plaintext> v38(19);
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v19;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v19[i % v19.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  v38[0] = pt;
  auto pt1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v20;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (unsigned i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v20[i % v20.size()]);
  }
  auto pt1 = cc->MakePackedPlaintext(pt1_filled);
  v38[1] = pt1;
  auto pt2_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt2_filled = v21;
  pt2_filled.clear();
  pt2_filled.reserve(pt2_filled_n);
  for (unsigned i = 0; i < pt2_filled_n; ++i) {
    pt2_filled.push_back(v21[i % v21.size()]);
  }
  auto pt2 = cc->MakePackedPlaintext(pt2_filled);
  v38[2] = pt2;
  auto pt3_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt3_filled = v22;
  pt3_filled.clear();
  pt3_filled.reserve(pt3_filled_n);
  for (unsigned i = 0; i < pt3_filled_n; ++i) {
    pt3_filled.push_back(v22[i % v22.size()]);
  }
  auto pt3 = cc->MakePackedPlaintext(pt3_filled);
  v38[3] = pt3;
  auto pt4_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt4_filled = v23;
  pt4_filled.clear();
  pt4_filled.reserve(pt4_filled_n);
  for (unsigned i = 0; i < pt4_filled_n; ++i) {
    pt4_filled.push_back(v23[i % v23.size()]);
  }
  auto pt4 = cc->MakePackedPlaintext(pt4_filled);
  v38[4] = pt4;
  auto pt5_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt5_filled = v24;
  pt5_filled.clear();
  pt5_filled.reserve(pt5_filled_n);
  for (unsigned i = 0; i < pt5_filled_n; ++i) {
    pt5_filled.push_back(v24[i % v24.size()]);
  }
  auto pt5 = cc->MakePackedPlaintext(pt5_filled);
  v38[5] = pt5;
  auto pt6_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt6_filled = v25;
  pt6_filled.clear();
  pt6_filled.reserve(pt6_filled_n);
  for (unsigned i = 0; i < pt6_filled_n; ++i) {
    pt6_filled.push_back(v25[i % v25.size()]);
  }
  auto pt6 = cc->MakePackedPlaintext(pt6_filled);
  v38[6] = pt6;
  auto pt7_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt7_filled = v26;
  pt7_filled.clear();
  pt7_filled.reserve(pt7_filled_n);
  for (unsigned i = 0; i < pt7_filled_n; ++i) {
    pt7_filled.push_back(v26[i % v26.size()]);
  }
  auto pt7 = cc->MakePackedPlaintext(pt7_filled);
  v38[7] = pt7;
  auto pt8_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt8_filled = v27;
  pt8_filled.clear();
  pt8_filled.reserve(pt8_filled_n);
  for (unsigned i = 0; i < pt8_filled_n; ++i) {
    pt8_filled.push_back(v27[i % v27.size()]);
  }
  auto pt8 = cc->MakePackedPlaintext(pt8_filled);
  v38[8] = pt8;
  auto pt9_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt9_filled = v28;
  pt9_filled.clear();
  pt9_filled.reserve(pt9_filled_n);
  for (unsigned i = 0; i < pt9_filled_n; ++i) {
    pt9_filled.push_back(v28[i % v28.size()]);
  }
  auto pt9 = cc->MakePackedPlaintext(pt9_filled);
  v38[9] = pt9;
  auto pt10_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt10_filled = v29;
  pt10_filled.clear();
  pt10_filled.reserve(pt10_filled_n);
  for (unsigned i = 0; i < pt10_filled_n; ++i) {
    pt10_filled.push_back(v29[i % v29.size()]);
  }
  auto pt10 = cc->MakePackedPlaintext(pt10_filled);
  v38[10] = pt10;
  auto pt11_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt11_filled = v30;
  pt11_filled.clear();
  pt11_filled.reserve(pt11_filled_n);
  for (unsigned i = 0; i < pt11_filled_n; ++i) {
    pt11_filled.push_back(v30[i % v30.size()]);
  }
  auto pt11 = cc->MakePackedPlaintext(pt11_filled);
  v38[11] = pt11;
  auto pt12_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt12_filled = v31;
  pt12_filled.clear();
  pt12_filled.reserve(pt12_filled_n);
  for (unsigned i = 0; i < pt12_filled_n; ++i) {
    pt12_filled.push_back(v31[i % v31.size()]);
  }
  auto pt12 = cc->MakePackedPlaintext(pt12_filled);
  v38[12] = pt12;
  auto pt13_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt13_filled = v32;
  pt13_filled.clear();
  pt13_filled.reserve(pt13_filled_n);
  for (unsigned i = 0; i < pt13_filled_n; ++i) {
    pt13_filled.push_back(v32[i % v32.size()]);
  }
  auto pt13 = cc->MakePackedPlaintext(pt13_filled);
  v38[13] = pt13;
  auto pt14_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt14_filled = v33;
  pt14_filled.clear();
  pt14_filled.reserve(pt14_filled_n);
  for (unsigned i = 0; i < pt14_filled_n; ++i) {
    pt14_filled.push_back(v33[i % v33.size()]);
  }
  auto pt14 = cc->MakePackedPlaintext(pt14_filled);
  v38[14] = pt14;
  auto pt15_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt15_filled = v34;
  pt15_filled.clear();
  pt15_filled.reserve(pt15_filled_n);
  for (unsigned i = 0; i < pt15_filled_n; ++i) {
    pt15_filled.push_back(v34[i % v34.size()]);
  }
  auto pt15 = cc->MakePackedPlaintext(pt15_filled);
  v38[15] = pt15;
  auto pt16_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt16_filled = v35;
  pt16_filled.clear();
  pt16_filled.reserve(pt16_filled_n);
  for (unsigned i = 0; i < pt16_filled_n; ++i) {
    pt16_filled.push_back(v35[i % v35.size()]);
  }
  auto pt16 = cc->MakePackedPlaintext(pt16_filled);
  v38[16] = pt16;
  auto pt17_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt17_filled = v36;
  pt17_filled.clear();
  pt17_filled.reserve(pt17_filled_n);
  for (unsigned i = 0; i < pt17_filled_n; ++i) {
    pt17_filled.push_back(v36[i % v36.size()]);
  }
  auto pt17 = cc->MakePackedPlaintext(pt17_filled);
  v38[17] = pt17;
  auto pt18_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt18_filled = v37;
  pt18_filled.clear();
  pt18_filled.reserve(pt18_filled_n);
  for (unsigned i = 0; i < pt18_filled_n; ++i) {
    pt18_filled.push_back(v37[i % v37.size()]);
  }
  auto pt18 = cc->MakePackedPlaintext(pt18_filled);
  v38[18] = pt18;
  return v38;
}
struct argmax_clone_0_0__preprocessedStruct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
  std::vector<CiphertextT> arg2;
  std::vector<CiphertextT> arg3;
};
argmax_clone_0_0__preprocessedStruct argmax_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5,
    std::vector<CiphertextT> v6, std::vector<CiphertextT> v7,
    std::vector<CiphertextT> v8, const std::vector<Plaintext>& v9) {
  std::vector<size_t> v10 = {53, 35, 48};
  std::vector<size_t> v11 = {48, 3};
  std::vector<size_t> v12 = {35, 53};
  std::vector<size_t> v13 = {3, 35};
  [[maybe_unused]] size_t v14 = 0;
  [[maybe_unused]] size_t v15 = 3;
  [[maybe_unused]] size_t v16 = 1;
  [[maybe_unused]] size_t v17 = 2;
  [[maybe_unused]] size_t v18 = 4;
  [[maybe_unused]] size_t v19 = 5;
  [[maybe_unused]] size_t v20 = 6;
  [[maybe_unused]] size_t v21 = 7;
  [[maybe_unused]] size_t v22 = 8;
  [[maybe_unused]] size_t v23 = 9;
  [[maybe_unused]] size_t v24 = 10;
  [[maybe_unused]] size_t v25 = 11;
  [[maybe_unused]] size_t v26 = 12;
  [[maybe_unused]] size_t v27 = 13;
  [[maybe_unused]] size_t v28 = 14;
  [[maybe_unused]] size_t v29 = 15;
  [[maybe_unused]] size_t v30 = 16;
  [[maybe_unused]] size_t v31 = 17;
  [[maybe_unused]] size_t v32 = 18;
  const auto& ct = v8[0];
  const auto& ct1 = v7[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  Plaintext pt = v9[0];
  auto ct4 = cc->EvalMult(ct2, pt);
  Plaintext pt1 = v9[1];
  const auto& ct5 = v0[0];
  const auto& ct6 = cc->EvalMult(ct5, pt1);
  cc->EvalAddInPlace(ct4, ct6);
  const auto& ct8 = v6[0];
  auto ct9 = cc->EvalMultNoRelin(ct8, ct4);
  cc->RelinearizeInPlace(ct9);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct9);
  Plaintext pt2 = v9[2];
  const auto& ct11 = v5[0];
  auto ct12 = cc->EvalMult(ct11, pt2);
  Plaintext pt3 = v9[3];
  Plaintext pt4 = v9[4];
  const auto& ct13 = v1[0];
  auto ct14 = cc->EvalMult(ct13, pt4);
  std::vector<CiphertextT> v33(2);
  auto v34 = v33;
#pragma omp parallel for
  for (auto v35 = 0; v35 < 2; ++v35) {
    size_t v37 = v13[v35];
    const auto& ct15 = cc->EvalFastRotation(
        ct2, v37, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v38 = {ct15};
    v34[v35] = v38[0];
  }
  const auto& ct16 = v34[0];
  const auto& ct17 = v34[1];
  const auto& ct18 = cc->EvalMult(ct17, pt3);
  cc->EvalAddInPlace(ct14, ct18);
  Plaintext pt5 = v9[5];
  const auto& ct20 = cc->EvalMult(ct2, pt5);
  cc->EvalAddInPlace(ct14, ct20);
  Plaintext pt6 = v9[6];
  auto v39 = v33;
#pragma omp parallel for
  for (auto v40 = 0; v40 < 2; ++v40) {
    size_t v42 = v12[v40];
    const auto& ct22 = cc->EvalFastRotation(
        ct9, v42, 2 * cc->GetRingDimension(), digit_decomp1);
    const std::vector<CiphertextT> v43 = {ct22};
    v39[v40] = v43[0];
  }
  const auto& ct23 = v39[0];
  const auto& ct24 = v39[1];
  Plaintext pt7 = v9[7];
  const auto& ct25 = cc->EvalMult(ct24, pt7);
  Plaintext pt8 = v9[8];
  const auto& ct26 = v4[0];
  const auto& ct27 = cc->EvalMult(ct26, pt8);
  Plaintext pt9 = v9[9];
  const auto& ct28 = v2[0];
  auto ct29 = cc->EvalMult(ct28, pt9);
  Plaintext pt10 = v9[10];
  std::vector<CiphertextT> v44(1);
  Plaintext pt11 = v9[11];
  const auto& ct30 = v3[0];
  auto ct31 = cc->EvalMult(ct30, pt11);
  Plaintext pt12 = v9[12];
  const auto& ct32 = cc->EvalMult(ct9, pt12);
  auto v45 = v33;
#pragma omp parallel for
  for (auto v46 = 0; v46 < 2; ++v46) {
    size_t v48 = v11[v46];
    const auto& ct33 = cc->EvalFastRotation(
        ct9, v48, 2 * cc->GetRingDimension(), digit_decomp1);
    const std::vector<CiphertextT> v49 = {ct33};
    v45[v46] = v49[0];
  }
  const auto& ct34 = v45[0];
  const auto& ct35 = v45[1];
  Plaintext pt13 = v9[13];
  const auto& ct36 = cc->EvalMult(ct35, pt13);
  Plaintext pt14 = v9[14];
  std::vector<CiphertextT> v50(3);
  Plaintext pt15 = v9[15];
  Plaintext pt16 = v9[16];
  Plaintext pt17 = v9[17];
  Plaintext pt18 = v9[18];
  const auto& ct37 = cc->EvalMult(ct16, pt3);
  cc->EvalAddInPlace(ct12, ct37);
  auto ct39 = cc->EvalMultNoRelin(ct12, ct14);
  cc->RelinearizeInPlace(ct39);
  auto ct41 = cc->EvalMult(ct23, pt6);
  cc->EvalAddInPlace(ct41, ct25);
  cc->EvalAddInPlace(ct41, ct27);
  const auto& ct44 = cc->EvalMult(ct39, pt10);
  cc->EvalAddInPlace(ct29, ct44);
  auto ct46 = cc->EvalMultNoRelin(ct41, ct29);
  cc->RelinearizeInPlace(ct46);
  const auto& digit_decomp2 = cc->EvalFastRotationPrecompute(ct46);
  std::vector<CiphertextT> v51(v44);
  v51[0] = ct46;
  const auto& ct48 = cc->EvalMult(ct34, pt6);
  cc->EvalAddInPlace(ct31, ct48);
  cc->EvalAddInPlace(ct31, ct32);
  cc->EvalAddInPlace(ct31, ct36);
#pragma omp parallel for
  for (auto v53 = 0; v53 < 3; ++v53) {
    size_t v55 = v10[v53];
    const auto& ct52 = cc->EvalFastRotation(
        ct46, v55, 2 * cc->GetRingDimension(), digit_decomp2);
    const std::vector<CiphertextT> v56 = {ct52};
    v50[v53] = v56[0];
  }
  const auto& ct53 = v50[0];
  const auto& ct54 = v50[1];
  const auto& ct55 = v50[2];
  const auto& ct56 = cc->EvalMult(ct55, pt15);
  const auto& ct57 = cc->EvalMult(ct46, pt16);
  auto ct58 = cc->EvalMult(ct46, pt15);
  const auto& ct59 = cc->EvalMult(ct39, pt17);
  cc->EvalAddInPlace(ct58, ct59);
  auto ct61 = cc->EvalMult(ct53, pt18);
  const auto& ct62 = cc->EvalMult(ct46, pt14);
  cc->EvalAddInPlace(ct61, ct62);
  auto ct64 = cc->EvalMult(ct54, pt14);
  cc->EvalAddInPlace(ct64, ct56);
  cc->EvalAddInPlace(ct64, ct57);
  auto ct67 = cc->EvalMultNoRelin(ct31, ct64);
  cc->RelinearizeInPlace(ct67);
  std::vector<CiphertextT> v57(v44);
  v57[0] = ct67;
  const auto& ct69 = cc->EvalMult(ct67, pt12);
  cc->EvalAddInPlace(ct61, ct69);
  auto ct71 = cc->EvalMultNoRelin(ct58, ct61);
  cc->RelinearizeInPlace(ct71);
  std::vector<CiphertextT> v58(v44);
  v58[0] = ct71;
  auto ct73 = cc->EvalMultNoRelin(ct67, ct71);
  cc->RelinearizeInPlace(ct73);
  std::vector<CiphertextT> v59(v44);
  v59[0] = ct73;
  return {v59, v57, v58, v51};
}
struct argmax_clone_0_0Struct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
  std::vector<CiphertextT> arg2;
  std::vector<CiphertextT> arg3;
};
argmax_clone_0_0Struct argmax_clone_0_0(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5,
    std::vector<CiphertextT> v6, std::vector<CiphertextT> v7,
    std::vector<CiphertextT> v8) {
  const auto& v9 = argmax_clone_0_0__preprocessing(cc);
  auto v10Struct = argmax_clone_0_0__preprocessed(cc, v0, v1, v2, v3, v4, v5,
                                                  v6, v7, v8, v9);
  const auto& v10 = v10Struct.arg0;
  const auto& v11 = v10Struct.arg1;
  const auto& v12 = v10Struct.arg2;
  const auto& v13 = v10Struct.arg3;
  return {v10, v11, v12, v13};
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg0(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 32;
  [[maybe_unused]] size_t v3 = 31;
  [[maybe_unused]] size_t v4 = 30;
  [[maybe_unused]] size_t v5 = 29;
  int32_t v6 = v0[29];
  int32_t v7 = v0[30];
  int32_t v8 = v0[31];
  int32_t v9 = v0[32];
  const std::vector<int32_t> v10 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v7, v1, v1, v1, v1, v9, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v6, v1, v1, v8, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v7, v1, v1, v1, v1, v9, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6,
      v1, v1, v8, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v7, v1, v1, v1, v1, v9, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v6, v1, v1, v8, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v7, v1, v1, v1, v1, v9, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v6, v1, v1, v8, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 256);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 256; ++v11_i1) {
      v11[v11_i1 + 256 * (v11_i0)] =
          v10[0 + v11_i1 * 1 + 256 * (0 + v11_i0 * 1)];
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg1(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 46;
  [[maybe_unused]] size_t v3 = 44;
  [[maybe_unused]] size_t v4 = 34;
  [[maybe_unused]] size_t v5 = 33;
  [[maybe_unused]] size_t v6 = 17;
  int32_t v7 = v0[17];
  int32_t v8 = v0[33];
  int32_t v9 = v0[34];
  int32_t v10 = v0[44];
  int32_t v11 = v0[46];
  const std::vector<int32_t> v12 = {
      v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v10, v1, v1, v1, v1, v1, v1, v1,  v1, v7, v1, v1, v1,
      v9, v1, v1, v8,  v1, v1, v1, v1, v1, v1, v11, v1, v1, v1, v1, v1,
      v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v10, v1, v1, v1, v1, v1, v1, v1,  v1, v7, v1, v1, v1,
      v9, v1, v1, v8,  v1, v1, v1, v1, v1, v1, v11, v1, v1, v1, v1, v1,
      v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v10, v1, v1, v1, v1, v1, v1, v1,  v1, v7, v1, v1, v1,
      v9, v1, v1, v8,  v1, v1, v1, v1, v1, v1, v11, v1, v1, v1, v1, v1,
      v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1, v1, v10, v1, v1, v1, v1, v1, v1, v1,  v1, v7, v1, v1, v1,
      v9, v1, v1, v8,  v1, v1, v1, v1, v1, v1, v11, v1, v1, v1, v1, v1,
      v1, v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1};
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
  [[maybe_unused]] size_t v2 = 48;
  [[maybe_unused]] size_t v3 = 47;
  [[maybe_unused]] size_t v4 = 45;
  [[maybe_unused]] size_t v5 = 43;
  [[maybe_unused]] size_t v6 = 20;
  [[maybe_unused]] size_t v7 = 19;
  int32_t v8 = v0[19];
  int32_t v9 = v0[20];
  int32_t v10 = v0[43];
  int32_t v11 = v0[45];
  int32_t v12 = v0[47];
  int32_t v13 = v0[48];
  const std::vector<int32_t> v14 = {
      v1, v1, v1, v1,  v1, v1,  v8, v1, v1, v1,  v1, v1, v11, v1, v1, v1,
      v1, v1, v1, v1,  v1, v13, v1, v1, v1, v10, v1, v1, v1,  v1, v1, v1,
      v9, v1, v1, v12, v1, v1,  v1, v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1, v1, v1, v1,  v1, v1,  v1, v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1, v1, v1, v1,  v1, v1,  v8, v1, v1, v1,  v1, v1, v11, v1, v1, v1,
      v1, v1, v1, v1,  v1, v13, v1, v1, v1, v10, v1, v1, v1,  v1, v1, v1,
      v9, v1, v1, v12, v1, v1,  v1, v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1, v1, v1, v1,  v1, v1,  v1, v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1, v1, v1, v1,  v1, v1,  v8, v1, v1, v1,  v1, v1, v11, v1, v1, v1,
      v1, v1, v1, v1,  v1, v13, v1, v1, v1, v10, v1, v1, v1,  v1, v1, v1,
      v9, v1, v1, v12, v1, v1,  v1, v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1, v1, v1, v1,  v1, v1,  v1, v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1, v1, v1, v1,  v1, v1,  v8, v1, v1, v1,  v1, v1, v11, v1, v1, v1,
      v1, v1, v1, v1,  v1, v13, v1, v1, v1, v10, v1, v1, v1,  v1, v1, v1,
      v9, v1, v1, v12, v1, v1,  v1, v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1, v1, v1, v1,  v1, v1,  v1, v1, v1, v1,  v1, v1, v1,  v1, v1, v1};
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg3(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 6;
  [[maybe_unused]] size_t v3 = 5;
  int32_t v4 = v0[5];
  int32_t v5 = v0[6];
  const std::vector<int32_t> v6 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v5, v1, v1, v4,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v5, v1, v1, v4, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v5, v1,
      v1, v4, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v5, v1, v1, v4, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1};
  std::vector<int32_t> v7(1 * 256);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 256; ++v7_i1) {
      v7[v7_i1 + 256 * (v7_i0)] = v6[0 + v7_i1 * 1 + 256 * (0 + v7_i0 * 1)];
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg4(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 41;
  [[maybe_unused]] size_t v3 = 40;
  [[maybe_unused]] size_t v4 = 38;
  [[maybe_unused]] size_t v5 = 36;
  [[maybe_unused]] size_t v6 = 13;
  [[maybe_unused]] size_t v7 = 12;
  [[maybe_unused]] size_t v8 = 3;
  int32_t v9 = v0[3];
  int32_t v10 = v0[12];
  int32_t v11 = v0[13];
  int32_t v12 = v0[36];
  int32_t v13 = v0[38];
  int32_t v14 = v0[40];
  int32_t v15 = v0[41];
  const std::vector<int32_t> v16 = {
      v1,  v1, v1, v1,  v1, v1,  v10, v1, v1, v1,  v1, v1, v13, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v15, v1,  v1, v1, v12, v1, v1, v9,  v1, v1, v1,
      v11, v1, v1, v14, v1, v1,  v1,  v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1,  v1,  v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1,  v10, v1, v1, v1,  v1, v1, v13, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v15, v1,  v1, v1, v12, v1, v1, v9,  v1, v1, v1,
      v11, v1, v1, v14, v1, v1,  v1,  v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1,  v1,  v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1,  v10, v1, v1, v1,  v1, v1, v13, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v15, v1,  v1, v1, v12, v1, v1, v9,  v1, v1, v1,
      v11, v1, v1, v14, v1, v1,  v1,  v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1,  v1,  v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1,  v10, v1, v1, v1,  v1, v1, v13, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v15, v1,  v1, v1, v12, v1, v1, v9,  v1, v1, v1,
      v11, v1, v1, v14, v1, v1,  v1,  v1, v1, v1,  v1, v1, v1,  v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1,  v1,  v1, v1, v1,  v1, v1, v1,  v1, v1, v1};
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg5(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 39;
  [[maybe_unused]] size_t v3 = 37;
  [[maybe_unused]] size_t v4 = 27;
  [[maybe_unused]] size_t v5 = 26;
  [[maybe_unused]] size_t v6 = 10;
  [[maybe_unused]] size_t v7 = 1;
  int32_t v8 = v0[1];
  int32_t v9 = v0[10];
  int32_t v10 = v0[26];
  int32_t v11 = v0[27];
  int32_t v12 = v0[37];
  int32_t v13 = v0[39];
  const std::vector<int32_t> v14 = {
      v1,  v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1,  v1, v1, v12, v1, v1, v1, v1, v1, v8, v1,  v1, v9, v1, v1, v1,
      v11, v1, v1, v10, v1, v1, v1, v1, v1, v1, v13, v1, v1, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1,  v1, v1, v12, v1, v1, v1, v1, v1, v8, v1,  v1, v9, v1, v1, v1,
      v11, v1, v1, v10, v1, v1, v1, v1, v1, v1, v13, v1, v1, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1,  v1, v1, v12, v1, v1, v1, v1, v1, v8, v1,  v1, v9, v1, v1, v1,
      v11, v1, v1, v10, v1, v1, v1, v1, v1, v1, v13, v1, v1, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1,  v1, v1, v12, v1, v1, v1, v1, v1, v8, v1,  v1, v9, v1, v1, v1,
      v11, v1, v1, v10, v1, v1, v1, v1, v1, v1, v13, v1, v1, v1, v1, v1,
      v1,  v1, v1, v1,  v1, v1, v1, v1, v1, v1, v1,  v1, v1, v1, v1, v1};
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg6(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 24;
  [[maybe_unused]] size_t v3 = 23;
  [[maybe_unused]] size_t v4 = 25;
  [[maybe_unused]] size_t v5 = 4;
  [[maybe_unused]] size_t v6 = 22;
  [[maybe_unused]] size_t v7 = 2;
  [[maybe_unused]] size_t v8 = 0;
  int32_t v9 = v0[0];
  int32_t v10 = v0[2];
  int32_t v11 = v0[4];
  int32_t v12 = v0[22];
  int32_t v13 = v0[23];
  int32_t v14 = v0[24];
  int32_t v15 = v0[25];
  const std::vector<int32_t> v16 = {
      v1, v1, v1, v1, v1, v1, v1,  v1, v13, v1,  v1,  v1, v1,  v15, v1, v1,
      v1, v1, v1, v9, v1, v1, v10, v1, v1,  v12, v11, v1, v14, v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v1,  v1,  v1,  v1, v1,  v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v1,  v1,  v1,  v1, v1,  v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v13, v1,  v1,  v1, v1,  v15, v1, v1,
      v1, v1, v1, v9, v1, v1, v10, v1, v1,  v12, v11, v1, v14, v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v1,  v1,  v1,  v1, v1,  v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v1,  v1,  v1,  v1, v1,  v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v13, v1,  v1,  v1, v1,  v15, v1, v1,
      v1, v1, v1, v9, v1, v1, v10, v1, v1,  v12, v11, v1, v14, v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v1,  v1,  v1,  v1, v1,  v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v1,  v1,  v1,  v1, v1,  v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v13, v1,  v1,  v1, v1,  v15, v1, v1,
      v1, v1, v1, v9, v1, v1, v10, v1, v1,  v12, v11, v1, v14, v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v1,  v1,  v1,  v1, v1,  v1,  v1, v1,
      v1, v1, v1, v1, v1, v1, v1,  v1, v1,  v1,  v1,  v1, v1,  v1,  v1, v1};
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg7(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 42;
  [[maybe_unused]] size_t v3 = 28;
  [[maybe_unused]] size_t v4 = 18;
  [[maybe_unused]] size_t v5 = 16;
  [[maybe_unused]] size_t v6 = 15;
  [[maybe_unused]] size_t v7 = 14;
  int32_t v8 = v0[14];
  int32_t v9 = v0[15];
  int32_t v10 = v0[16];
  int32_t v11 = v0[18];
  int32_t v12 = v0[28];
  int32_t v13 = v0[42];
  const std::vector<int32_t> v14 = {
      v1, v13, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v8, v1, v1, v10, v1, v1, v9, v11, v1, v1, v1, v1, v1,
      v1, v12, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v13, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v8, v1, v1, v10, v1, v1, v9, v11, v1, v1, v1, v1, v1,
      v1, v12, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v13, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v8, v1, v1, v10, v1, v1, v9, v11, v1, v1, v1, v1, v1,
      v1, v12, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v13, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v8, v1, v1, v10, v1, v1, v9, v11, v1, v1, v1, v1, v1,
      v1, v12, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1};
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg8(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 35;
  [[maybe_unused]] size_t v3 = 21;
  [[maybe_unused]] size_t v4 = 11;
  [[maybe_unused]] size_t v5 = 9;
  [[maybe_unused]] size_t v6 = 8;
  [[maybe_unused]] size_t v7 = 7;
  int32_t v8 = v0[7];
  int32_t v9 = v0[8];
  int32_t v10 = v0[9];
  int32_t v11 = v0[11];
  int32_t v12 = v0[21];
  int32_t v13 = v0[35];
  const std::vector<int32_t> v14 = {
      v1, v13, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v8, v1, v1, v10, v1, v1, v9, v11, v1, v1, v1, v1, v1,
      v1, v12, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v13, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v8, v1, v1, v10, v1, v1, v9, v11, v1, v1, v1, v1, v1,
      v1, v12, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v13, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v8, v1, v1, v10, v1, v1, v9, v11, v1, v1, v1, v1, v1,
      v1, v12, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v13, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v8, v1, v1, v10, v1, v1, v9, v11, v1, v1, v1, v1, v1,
      v1, v12, v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1,
      v1, v1,  v1, v1, v1, v1, v1,  v1, v1, v1, v1,  v1, v1, v1, v1, v1};
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
std::vector<int32_t> argmax_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 32;
  [[maybe_unused]] size_t v3 = 35;
  [[maybe_unused]] size_t v4 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v5_cast = pt->GetPackedValue();
  std::vector<int32_t> v5(std::begin(v5_cast), std::end(v5_cast));
  int32_t v6 = v5[35 + 256 * (0)];
  int32_t v7 = v5[32 + 256 * (0)];
  const std::vector<int32_t> v8 = {v1, v1, v1, v1, v1, v6, v7};
  return v8;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result1(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 42;
  [[maybe_unused]] size_t v3 = 19;
  [[maybe_unused]] size_t v4 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v5_cast = pt->GetPackedValue();
  std::vector<int32_t> v5(std::begin(v5_cast), std::end(v5_cast));
  int32_t v6 = v5[19 + 256 * (0)];
  int32_t v7 = v5[42 + 256 * (0)];
  const std::vector<int32_t> v8 = {v1, v1, v6, v1, v7, v1, v1};
  return v8;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result2(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 28;
  [[maybe_unused]] size_t v3 = 25;
  [[maybe_unused]] size_t v4 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v5_cast = pt->GetPackedValue();
  std::vector<int32_t> v5(std::begin(v5_cast), std::end(v5_cast));
  int32_t v6 = v5[25 + 256 * (0)];
  int32_t v7 = v5[28 + 256 * (0)];
  const std::vector<int32_t> v8 = {v1, v6, v1, v7, v1, v1, v1};
  return v8;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result3(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 30;
  [[maybe_unused]] size_t v3 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v4_cast = pt->GetPackedValue();
  std::vector<int32_t> v4(std::begin(v4_cast), std::end(v4_cast));
  int32_t v5 = v4[30 + 256 * (0)];
  const std::vector<int32_t> v6 = {v5, v1, v1, v1, v1, v1, v1};
  return v6;
}
CryptoContextT argmax_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(10);
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
  cc->EvalRotateKeyGen(sk, {35, 53, 3, 48});
  return cc;
}
