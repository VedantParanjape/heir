
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
  std::vector<int64_t> v16 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v17 = {
      0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v18 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v19 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v20 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v21 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v22 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v23 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v24 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v25 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v26 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v27 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v28 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v29 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v30 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v31 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<Plaintext> v32(16);
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v16;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v16[i % v16.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  v32[0] = pt;
  auto pt1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v17;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (unsigned i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v17[i % v17.size()]);
  }
  auto pt1 = cc->MakePackedPlaintext(pt1_filled);
  v32[1] = pt1;
  auto pt2_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt2_filled = v18;
  pt2_filled.clear();
  pt2_filled.reserve(pt2_filled_n);
  for (unsigned i = 0; i < pt2_filled_n; ++i) {
    pt2_filled.push_back(v18[i % v18.size()]);
  }
  auto pt2 = cc->MakePackedPlaintext(pt2_filled);
  v32[2] = pt2;
  auto pt3_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt3_filled = v19;
  pt3_filled.clear();
  pt3_filled.reserve(pt3_filled_n);
  for (unsigned i = 0; i < pt3_filled_n; ++i) {
    pt3_filled.push_back(v19[i % v19.size()]);
  }
  auto pt3 = cc->MakePackedPlaintext(pt3_filled);
  v32[3] = pt3;
  auto pt4_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt4_filled = v20;
  pt4_filled.clear();
  pt4_filled.reserve(pt4_filled_n);
  for (unsigned i = 0; i < pt4_filled_n; ++i) {
    pt4_filled.push_back(v20[i % v20.size()]);
  }
  auto pt4 = cc->MakePackedPlaintext(pt4_filled);
  v32[4] = pt4;
  auto pt5_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt5_filled = v21;
  pt5_filled.clear();
  pt5_filled.reserve(pt5_filled_n);
  for (unsigned i = 0; i < pt5_filled_n; ++i) {
    pt5_filled.push_back(v21[i % v21.size()]);
  }
  auto pt5 = cc->MakePackedPlaintext(pt5_filled);
  v32[5] = pt5;
  auto pt6_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt6_filled = v22;
  pt6_filled.clear();
  pt6_filled.reserve(pt6_filled_n);
  for (unsigned i = 0; i < pt6_filled_n; ++i) {
    pt6_filled.push_back(v22[i % v22.size()]);
  }
  auto pt6 = cc->MakePackedPlaintext(pt6_filled);
  v32[6] = pt6;
  auto pt7_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt7_filled = v23;
  pt7_filled.clear();
  pt7_filled.reserve(pt7_filled_n);
  for (unsigned i = 0; i < pt7_filled_n; ++i) {
    pt7_filled.push_back(v23[i % v23.size()]);
  }
  auto pt7 = cc->MakePackedPlaintext(pt7_filled);
  v32[7] = pt7;
  auto pt8_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt8_filled = v24;
  pt8_filled.clear();
  pt8_filled.reserve(pt8_filled_n);
  for (unsigned i = 0; i < pt8_filled_n; ++i) {
    pt8_filled.push_back(v24[i % v24.size()]);
  }
  auto pt8 = cc->MakePackedPlaintext(pt8_filled);
  v32[8] = pt8;
  auto pt9_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt9_filled = v25;
  pt9_filled.clear();
  pt9_filled.reserve(pt9_filled_n);
  for (unsigned i = 0; i < pt9_filled_n; ++i) {
    pt9_filled.push_back(v25[i % v25.size()]);
  }
  auto pt9 = cc->MakePackedPlaintext(pt9_filled);
  v32[9] = pt9;
  auto pt10_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt10_filled = v26;
  pt10_filled.clear();
  pt10_filled.reserve(pt10_filled_n);
  for (unsigned i = 0; i < pt10_filled_n; ++i) {
    pt10_filled.push_back(v26[i % v26.size()]);
  }
  auto pt10 = cc->MakePackedPlaintext(pt10_filled);
  v32[10] = pt10;
  auto pt11_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt11_filled = v27;
  pt11_filled.clear();
  pt11_filled.reserve(pt11_filled_n);
  for (unsigned i = 0; i < pt11_filled_n; ++i) {
    pt11_filled.push_back(v27[i % v27.size()]);
  }
  auto pt11 = cc->MakePackedPlaintext(pt11_filled);
  v32[11] = pt11;
  auto pt12_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt12_filled = v28;
  pt12_filled.clear();
  pt12_filled.reserve(pt12_filled_n);
  for (unsigned i = 0; i < pt12_filled_n; ++i) {
    pt12_filled.push_back(v28[i % v28.size()]);
  }
  auto pt12 = cc->MakePackedPlaintext(pt12_filled);
  v32[12] = pt12;
  auto pt13_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt13_filled = v29;
  pt13_filled.clear();
  pt13_filled.reserve(pt13_filled_n);
  for (unsigned i = 0; i < pt13_filled_n; ++i) {
    pt13_filled.push_back(v29[i % v29.size()]);
  }
  auto pt13 = cc->MakePackedPlaintext(pt13_filled);
  v32[13] = pt13;
  auto pt14_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt14_filled = v30;
  pt14_filled.clear();
  pt14_filled.reserve(pt14_filled_n);
  for (unsigned i = 0; i < pt14_filled_n; ++i) {
    pt14_filled.push_back(v30[i % v30.size()]);
  }
  auto pt14 = cc->MakePackedPlaintext(pt14_filled);
  v32[14] = pt14;
  auto pt15_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt15_filled = v31;
  pt15_filled.clear();
  pt15_filled.reserve(pt15_filled_n);
  for (unsigned i = 0; i < pt15_filled_n; ++i) {
    pt15_filled.push_back(v31[i % v31.size()]);
  }
  auto pt15 = cc->MakePackedPlaintext(pt15_filled);
  v32[15] = pt15;
  return v32;
}
struct argmax_clone_0_0__preprocessedStruct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
};
argmax_clone_0_0__preprocessedStruct argmax_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    const std::vector<Plaintext>& v4) {
  std::vector<size_t> v5 = {8, 37};
  std::vector<size_t> v6 = {47, 29, 6, 32, 37};
  std::vector<size_t> v7 = {29, 32, 1, 47, 16};
  [[maybe_unused]] size_t v8 = 0;
  [[maybe_unused]] size_t v9 = 8;
  [[maybe_unused]] size_t v10 = 6;
  [[maybe_unused]] size_t v11 = 1;
  [[maybe_unused]] size_t v12 = 2;
  [[maybe_unused]] size_t v13 = 3;
  [[maybe_unused]] size_t v14 = 4;
  [[maybe_unused]] size_t v15 = 5;
  [[maybe_unused]] size_t v16 = 7;
  [[maybe_unused]] size_t v17 = 9;
  [[maybe_unused]] size_t v18 = 10;
  [[maybe_unused]] size_t v19 = 11;
  [[maybe_unused]] size_t v20 = 12;
  [[maybe_unused]] size_t v21 = 13;
  [[maybe_unused]] size_t v22 = 14;
  [[maybe_unused]] size_t v23 = 15;
  const auto& ct = v3[0];
  const auto& ct1 = v2[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  Plaintext pt = v4[0];
  Plaintext pt1 = v4[1];
  const auto& ct4 = cc->EvalMult(ct2, pt1);
  Plaintext pt2 = v4[2];
  const auto& ct5 = v1[0];
  Plaintext pt3 = v4[3];
  Plaintext pt4 = v4[4];
  Plaintext pt5 = v4[5];
  Plaintext pt6 = v4[6];
  Plaintext pt7 = v4[7];
  Plaintext pt8 = v4[8];
  std::vector<CiphertextT> v24(5);
  auto v25 = v24;
#pragma omp parallel for
  for (auto v26 = 0; v26 < 5; ++v26) {
    size_t v28 = v7[v26];
    const auto& ct6 = cc->EvalFastRotation(ct2, v28, 2 * cc->GetRingDimension(),
                                           digit_decomp);
    const std::vector<CiphertextT> v29 = {ct6};
    v25[v26] = v29[0];
  }
  const auto& ct7 = v25[0];
  const auto& ct8 = v25[1];
  const auto& ct9 = v25[2];
  const auto& ct10 = v25[3];
  const auto& ct11 = v25[4];
  Plaintext pt9 = v4[9];
  const auto& ct12 = cc->EvalMult(ct11, pt9);
  const auto& ct13 = v0[0];
  auto ct14 = cc->EvalMult(ct13, pt2);
  Plaintext pt10 = v4[10];
  Plaintext pt11 = v4[11];
  Plaintext pt12 = v4[12];
  Plaintext pt13 = v4[13];
  std::vector<CiphertextT> v30(2);
  Plaintext pt14 = v4[14];
  Plaintext pt15 = v4[15];
  const auto& ct15 = cc->EvalMult(ct7, pt2);
  std::vector<CiphertextT> v31(1);
  auto ct16 = cc->EvalMult(ct8, pt);
  cc->EvalAddInPlace(ct16, ct4);
  const auto& ct18 = cc->EvalMult(ct9, pt2);
  cc->EvalAddInPlace(ct16, ct18);
  auto ct20 = cc->EvalMultNoRelin(ct5, ct16);
  cc->RelinearizeInPlace(ct20);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct20);
  auto ct22 = cc->EvalMult(ct7, pt3);
  const auto& ct23 = cc->EvalMult(ct8, pt4);
  cc->EvalAddInPlace(ct22, ct23);
  const auto& ct25 = cc->EvalMult(ct9, pt5);
  cc->EvalAddInPlace(ct22, ct25);
  auto ct27 = cc->EvalMult(ct7, pt6);
  const auto& ct28 = cc->EvalMult(ct8, pt7);
  cc->EvalAddInPlace(ct27, ct28);
  const auto& ct30 = cc->EvalMult(ct10, pt8);
  cc->EvalAddInPlace(ct27, ct30);
  cc->EvalAddInPlace(ct27, ct12);
  auto ct33 = cc->EvalMultNoRelin(ct22, ct27);
  cc->RelinearizeInPlace(ct33);
  const auto& digit_decomp2 = cc->EvalFastRotationPrecompute(ct33);
  auto v32 = v24;
#pragma omp parallel for
  for (auto v33 = 0; v33 < 5; ++v33) {
    size_t v35 = v6[v33];
    const auto& ct35 = cc->EvalFastRotation(
        ct20, v35, 2 * cc->GetRingDimension(), digit_decomp1);
    const std::vector<CiphertextT> v36 = {ct35};
    v32[v33] = v36[0];
  }
  const auto& ct36 = v32[0];
  const auto& ct37 = v32[1];
  const auto& ct38 = v32[2];
  const auto& ct39 = v32[3];
  const auto& ct40 = v32[4];
  const auto& ct41 = cc->EvalMult(ct40, pt8);
  const auto& ct42 = cc->EvalMult(ct36, pt12);
  const auto& ct43 = cc->EvalMult(ct38, pt13);
#pragma omp parallel for
  for (auto v38 = 0; v38 < 2; ++v38) {
    size_t v40 = v5[v38];
    const auto& ct44 = cc->EvalFastRotation(
        ct33, v40, 2 * cc->GetRingDimension(), digit_decomp2);
    const std::vector<CiphertextT> v41 = {ct44};
    v30[v38] = v41[0];
  }
  const auto& ct45 = v30[0];
  const auto& ct46 = v30[1];
  auto ct47 = cc->EvalMult(ct46, pt13);
  const auto& ct48 = cc->EvalMult(ct33, pt14);
  cc->EvalAddInPlace(ct47, ct48);
  const auto& ct50 = cc->EvalMult(ct45, pt15);
  cc->EvalAddInPlace(ct47, ct50);
  cc->EvalAddInPlace(ct47, ct15);
  const auto& ct53 = cc->EvalMult(ct37, pt10);
  cc->EvalAddInPlace(ct14, ct53);
  const auto& ct55 = cc->EvalMult(ct39, pt11);
  cc->EvalAddInPlace(ct14, ct55);
  cc->EvalAddInPlace(ct14, ct41);
  cc->EvalAddInPlace(ct14, ct42);
  cc->EvalAddInPlace(ct14, ct43);
  auto ct60 = cc->EvalMultNoRelin(ct14, ct47);
  cc->RelinearizeInPlace(ct60);
  std::vector<CiphertextT> v42(v31);
  v42[0] = ct60;
  auto ct62 = cc->EvalMultNoRelin(ct60, ct45);
  cc->RelinearizeInPlace(ct62);
  std::vector<CiphertextT> v43(v31);
  v43[0] = ct62;
  return {v42, v43};
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
  [[maybe_unused]] size_t v2 = 5;
  int32_t v3 = v0[5];
  const std::vector<int32_t> v4 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v3,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v3, v1, v1,
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg1(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 6;
  [[maybe_unused]] size_t v3 = 4;
  [[maybe_unused]] size_t v4 = 3;
  [[maybe_unused]] size_t v5 = 2;
  [[maybe_unused]] size_t v6 = 1;
  [[maybe_unused]] size_t v7 = 0;
  int32_t v8 = v0[0];
  int32_t v9 = v0[1];
  int32_t v10 = v0[2];
  int32_t v11 = v0[3];
  int32_t v12 = v0[4];
  int32_t v13 = v0[6];
  const std::vector<int32_t> v14 = {
      v1,  v10, v1,  v1,  v1,  v1,  v1,  v1,  v9, v1, v1, v1, v1, v1, v1, v1,
      v1,  v1,  v1,  v12, v1,  v1,  v1,  v1,  v1, v1, v1, v1, v1, v1, v8, v1,
      v1,  v1,  v1,  v13, v1,  v1,  v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1,
      v11, v1,  v10, v1,  v1,  v1,  v1,  v1,  v1, v9, v1, v1, v1, v1, v1, v1,
      v1,  v1,  v1,  v1,  v12, v1,  v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v8,
      v1,  v1,  v1,  v1,  v13, v1,  v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1,
      v1,  v11, v1,  v10, v1,  v1,  v1,  v1,  v1, v1, v9, v1, v1, v1, v1, v1,
      v1,  v1,  v1,  v1,  v1,  v12, v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1,
      v8,  v1,  v1,  v1,  v1,  v13, v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1,
      v1,  v1,  v11, v1,  v10, v1,  v1,  v1,  v1, v1, v1, v9, v1, v1, v1, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v12, v1,  v1, v1, v1, v1, v1, v1, v1, v1,
      v1,  v8,  v1,  v1,  v1,  v1,  v13, v1,  v1, v1, v1, v1, v1, v1, v1, v1,
      v1,  v1,  v1,  v11, v1,  v10, v1,  v1,  v1, v1, v1, v1, v9, v1, v1, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v12, v1, v1, v1, v1, v1, v1, v1, v1,
      v1,  v1,  v8,  v1,  v1,  v1,  v1,  v13, v1, v1, v1, v1, v1, v1, v1, v1,
      v1,  v1,  v1,  v1,  v11, v1,  v1,  v1,  v1, v1, v1, v1, v1, v1, v1, v1};
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
  [[maybe_unused]] size_t v2 = 48;
  [[maybe_unused]] size_t v3 = 47;
  [[maybe_unused]] size_t v4 = 46;
  [[maybe_unused]] size_t v5 = 45;
  [[maybe_unused]] size_t v6 = 43;
  [[maybe_unused]] size_t v7 = 42;
  [[maybe_unused]] size_t v8 = 34;
  [[maybe_unused]] size_t v9 = 44;
  [[maybe_unused]] size_t v10 = 33;
  [[maybe_unused]] size_t v11 = 32;
  [[maybe_unused]] size_t v12 = 29;
  [[maybe_unused]] size_t v13 = 28;
  [[maybe_unused]] size_t v14 = 20;
  [[maybe_unused]] size_t v15 = 19;
  [[maybe_unused]] size_t v16 = 18;
  [[maybe_unused]] size_t v17 = 31;
  [[maybe_unused]] size_t v18 = 17;
  [[maybe_unused]] size_t v19 = 16;
  [[maybe_unused]] size_t v20 = 15;
  [[maybe_unused]] size_t v21 = 30;
  [[maybe_unused]] size_t v22 = 14;
  int32_t v23 = v0[14];
  int32_t v24 = v0[15];
  int32_t v25 = v0[16];
  int32_t v26 = v0[17];
  int32_t v27 = v0[18];
  int32_t v28 = v0[19];
  int32_t v29 = v0[20];
  int32_t v30 = v0[28];
  int32_t v31 = v0[29];
  int32_t v32 = v0[30];
  int32_t v33 = v0[31];
  int32_t v34 = v0[32];
  int32_t v35 = v0[33];
  int32_t v36 = v0[34];
  int32_t v37 = v0[42];
  int32_t v38 = v0[43];
  int32_t v39 = v0[44];
  int32_t v40 = v0[45];
  int32_t v41 = v0[46];
  int32_t v42 = v0[47];
  int32_t v43 = v0[48];
  const std::vector<int32_t> v44 = {
      v1,  v25, v1,  v1,  v1,  v1,  v32, v1,  v24, v1,  v33, v1,  v1,  v40, v1,
      v28, v38, v1,  v1,  v27, v37, v1,  v41, v30, v1,  v1,  v42, v36, v1,  v1,
      v23, v26, v1,  v1,  v1,  v1,  v29, v31, v1,  v1,  v34, v1,  v39, v1,  v35,
      v43, v1,  v1,  v1,  v1,  v25, v1,  v1,  v1,  v1,  v32, v1,  v24, v1,  v33,
      v1,  v1,  v40, v1,  v28, v38, v1,  v1,  v27, v37, v1,  v41, v30, v1,  v1,
      v42, v36, v1,  v1,  v23, v26, v1,  v1,  v1,  v1,  v29, v31, v1,  v1,  v34,
      v1,  v39, v1,  v35, v43, v1,  v1,  v1,  v1,  v25, v1,  v1,  v1,  v1,  v32,
      v1,  v24, v1,  v33, v1,  v1,  v40, v1,  v28, v38, v1,  v1,  v27, v37, v1,
      v41, v30, v1,  v1,  v42, v36, v1,  v1,  v23, v26, v1,  v1,  v1,  v1,  v29,
      v31, v1,  v1,  v34, v1,  v39, v1,  v35, v43, v1,  v1,  v1,  v1,  v25, v1,
      v1,  v1,  v1,  v32, v1,  v24, v1,  v33, v1,  v1,  v40, v1,  v28, v38, v1,
      v1,  v27, v37, v1,  v41, v30, v1,  v1,  v42, v36, v1,  v1,  v23, v26, v1,
      v1,  v1,  v1,  v29, v31, v1,  v1,  v34, v1,  v39, v1,  v35, v43, v1,  v1,
      v1,  v1,  v25, v1,  v1,  v1,  v1,  v32, v1,  v24, v1,  v33, v1,  v1,  v40,
      v1,  v28, v38, v1,  v1,  v27, v37, v1,  v41, v30, v1,  v1,  v42, v36, v1,
      v1,  v23, v26, v1,  v1,  v1,  v1,  v29, v31, v1,  v1,  v34, v1,  v39, v1,
      v35, v43, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
  std::vector<int32_t> v45(1 * 256);
  for (int64_t v45_i0 = 0; v45_i0 < 1; ++v45_i0) {
    for (int64_t v45_i1 = 0; v45_i1 < 256; ++v45_i1) {
      v45[v45_i1 + 256 * (v45_i0)] =
          v44[0 + v45_i1 * 1 + 256 * (0 + v45_i0 * 1)];
    }
  }
  std::vector<int64_t> v46(std::begin(v45), std::end(v45));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v46;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v46[i % v46.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v47 = {ct};
  return v47;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg3(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 41;
  [[maybe_unused]] size_t v3 = 39;
  [[maybe_unused]] size_t v4 = 38;
  [[maybe_unused]] size_t v5 = 35;
  [[maybe_unused]] size_t v6 = 27;
  [[maybe_unused]] size_t v7 = 26;
  [[maybe_unused]] size_t v8 = 40;
  [[maybe_unused]] size_t v9 = 25;
  [[maybe_unused]] size_t v10 = 24;
  [[maybe_unused]] size_t v11 = 37;
  [[maybe_unused]] size_t v12 = 22;
  [[maybe_unused]] size_t v13 = 23;
  [[maybe_unused]] size_t v14 = 21;
  [[maybe_unused]] size_t v15 = 36;
  [[maybe_unused]] size_t v16 = 13;
  [[maybe_unused]] size_t v17 = 12;
  [[maybe_unused]] size_t v18 = 11;
  [[maybe_unused]] size_t v19 = 10;
  [[maybe_unused]] size_t v20 = 9;
  [[maybe_unused]] size_t v21 = 8;
  [[maybe_unused]] size_t v22 = 7;
  int32_t v23 = v0[7];
  int32_t v24 = v0[8];
  int32_t v25 = v0[9];
  int32_t v26 = v0[10];
  int32_t v27 = v0[11];
  int32_t v28 = v0[12];
  int32_t v29 = v0[13];
  int32_t v30 = v0[21];
  int32_t v31 = v0[22];
  int32_t v32 = v0[23];
  int32_t v33 = v0[24];
  int32_t v34 = v0[25];
  int32_t v35 = v0[26];
  int32_t v36 = v0[27];
  int32_t v37 = v0[35];
  int32_t v38 = v0[36];
  int32_t v39 = v0[37];
  int32_t v40 = v0[38];
  int32_t v41 = v0[39];
  int32_t v42 = v0[40];
  int32_t v43 = v0[41];
  const std::vector<int32_t> v44 = {
      v1,  v25, v1,  v1,  v1,  v1,  v32, v1,  v24, v1,  v33, v1,  v1,  v40, v1,
      v28, v38, v1,  v1,  v27, v37, v1,  v41, v30, v1,  v1,  v42, v36, v1,  v1,
      v23, v26, v1,  v1,  v1,  v1,  v29, v31, v1,  v1,  v34, v1,  v39, v1,  v35,
      v43, v1,  v1,  v1,  v1,  v25, v1,  v1,  v1,  v1,  v32, v1,  v24, v1,  v33,
      v1,  v1,  v40, v1,  v28, v38, v1,  v1,  v27, v37, v1,  v41, v30, v1,  v1,
      v42, v36, v1,  v1,  v23, v26, v1,  v1,  v1,  v1,  v29, v31, v1,  v1,  v34,
      v1,  v39, v1,  v35, v43, v1,  v1,  v1,  v1,  v25, v1,  v1,  v1,  v1,  v32,
      v1,  v24, v1,  v33, v1,  v1,  v40, v1,  v28, v38, v1,  v1,  v27, v37, v1,
      v41, v30, v1,  v1,  v42, v36, v1,  v1,  v23, v26, v1,  v1,  v1,  v1,  v29,
      v31, v1,  v1,  v34, v1,  v39, v1,  v35, v43, v1,  v1,  v1,  v1,  v25, v1,
      v1,  v1,  v1,  v32, v1,  v24, v1,  v33, v1,  v1,  v40, v1,  v28, v38, v1,
      v1,  v27, v37, v1,  v41, v30, v1,  v1,  v42, v36, v1,  v1,  v23, v26, v1,
      v1,  v1,  v1,  v29, v31, v1,  v1,  v34, v1,  v39, v1,  v35, v43, v1,  v1,
      v1,  v1,  v25, v1,  v1,  v1,  v1,  v32, v1,  v24, v1,  v33, v1,  v1,  v40,
      v1,  v28, v38, v1,  v1,  v27, v37, v1,  v41, v30, v1,  v1,  v42, v36, v1,
      v1,  v23, v26, v1,  v1,  v1,  v1,  v29, v31, v1,  v1,  v34, v1,  v39, v1,
      v35, v43, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
  std::vector<int32_t> v45(1 * 256);
  for (int64_t v45_i0 = 0; v45_i0 < 1; ++v45_i0) {
    for (int64_t v45_i1 = 0; v45_i1 < 256; ++v45_i1) {
      v45[v45_i1 + 256 * (v45_i0)] =
          v44[0 + v45_i1 * 1 + 256 * (0 + v45_i0 * 1)];
    }
  }
  std::vector<int64_t> v46(std::begin(v45), std::end(v45));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v46;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v46[i % v46.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v47 = {ct};
  return v47;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 47;
  [[maybe_unused]] size_t v3 = 39;
  [[maybe_unused]] size_t v4 = 42;
  [[maybe_unused]] size_t v5 = 18;
  [[maybe_unused]] size_t v6 = 28;
  [[maybe_unused]] size_t v7 = 32;
  [[maybe_unused]] size_t v8 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v9_cast = pt->GetPackedValue();
  std::vector<int32_t> v9(std::begin(v9_cast), std::end(v9_cast));
  int32_t v10 = v9[32 + 256 * (0)];
  int32_t v11 = v9[28 + 256 * (0)];
  int32_t v12 = v9[18 + 256 * (0)];
  int32_t v13 = v9[42 + 256 * (0)];
  int32_t v14 = v9[39 + 256 * (0)];
  int32_t v15 = v9[47 + 256 * (0)];
  const std::vector<int32_t> v16 = {v10, v11, v12, v13, v14, v1, v15};
  return v16;
}
std::vector<int32_t> argmax_clone_0_0__decrypt__result1(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 35;
  [[maybe_unused]] size_t v3 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v4_cast = pt->GetPackedValue();
  std::vector<int32_t> v4(std::begin(v4_cast), std::end(v4_cast));
  int32_t v5 = v4[35 + 256 * (0)];
  const std::vector<int32_t> v6 = {v1, v1, v1, v1, v1, v5, v1};
  return v6;
}
CryptoContextT argmax_clone_0_0__generate_crypto_context() {
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
CryptoContextT argmax_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                          PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {47, 16, 37, 6, 32, 1, 8, 29});
  return cc;
}
