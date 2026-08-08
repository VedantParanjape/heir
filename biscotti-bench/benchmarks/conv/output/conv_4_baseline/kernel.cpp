
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

std::vector<Plaintext> conv_clone_0_0__preprocessing(CryptoContextT cc) {
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
  [[maybe_unused]] size_t v24 = 24;
  [[maybe_unused]] size_t v25 = 25;
  [[maybe_unused]] size_t v26 = 26;
  [[maybe_unused]] size_t v27 = 27;
  [[maybe_unused]] size_t v28 = 28;
  std::vector<int64_t> v29 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v30 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v31 = {
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v32 = {
      1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v33 = {
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v34 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v35 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v36 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v37 = {
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v38 = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v39 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v40 = {
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v41 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v42 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v43 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v44 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v45 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v46 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v47 = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v48 = {
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v49 = {
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v50 = {
      0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v51 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v52 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v53 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v54 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v55 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v56 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v57 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<Plaintext> v58(29);
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v29;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v29[i % v29.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  v58[0] = pt;
  auto pt1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v30;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (unsigned i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v30[i % v30.size()]);
  }
  auto pt1 = cc->MakePackedPlaintext(pt1_filled);
  v58[1] = pt1;
  auto pt2_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt2_filled = v31;
  pt2_filled.clear();
  pt2_filled.reserve(pt2_filled_n);
  for (unsigned i = 0; i < pt2_filled_n; ++i) {
    pt2_filled.push_back(v31[i % v31.size()]);
  }
  auto pt2 = cc->MakePackedPlaintext(pt2_filled);
  v58[2] = pt2;
  auto pt3_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt3_filled = v32;
  pt3_filled.clear();
  pt3_filled.reserve(pt3_filled_n);
  for (unsigned i = 0; i < pt3_filled_n; ++i) {
    pt3_filled.push_back(v32[i % v32.size()]);
  }
  auto pt3 = cc->MakePackedPlaintext(pt3_filled);
  v58[3] = pt3;
  auto pt4_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt4_filled = v33;
  pt4_filled.clear();
  pt4_filled.reserve(pt4_filled_n);
  for (unsigned i = 0; i < pt4_filled_n; ++i) {
    pt4_filled.push_back(v33[i % v33.size()]);
  }
  auto pt4 = cc->MakePackedPlaintext(pt4_filled);
  v58[4] = pt4;
  auto pt5_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt5_filled = v34;
  pt5_filled.clear();
  pt5_filled.reserve(pt5_filled_n);
  for (unsigned i = 0; i < pt5_filled_n; ++i) {
    pt5_filled.push_back(v34[i % v34.size()]);
  }
  auto pt5 = cc->MakePackedPlaintext(pt5_filled);
  v58[5] = pt5;
  auto pt6_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt6_filled = v35;
  pt6_filled.clear();
  pt6_filled.reserve(pt6_filled_n);
  for (unsigned i = 0; i < pt6_filled_n; ++i) {
    pt6_filled.push_back(v35[i % v35.size()]);
  }
  auto pt6 = cc->MakePackedPlaintext(pt6_filled);
  v58[6] = pt6;
  auto pt7_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt7_filled = v36;
  pt7_filled.clear();
  pt7_filled.reserve(pt7_filled_n);
  for (unsigned i = 0; i < pt7_filled_n; ++i) {
    pt7_filled.push_back(v36[i % v36.size()]);
  }
  auto pt7 = cc->MakePackedPlaintext(pt7_filled);
  v58[7] = pt7;
  auto pt8_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt8_filled = v37;
  pt8_filled.clear();
  pt8_filled.reserve(pt8_filled_n);
  for (unsigned i = 0; i < pt8_filled_n; ++i) {
    pt8_filled.push_back(v37[i % v37.size()]);
  }
  auto pt8 = cc->MakePackedPlaintext(pt8_filled);
  v58[8] = pt8;
  auto pt9_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt9_filled = v38;
  pt9_filled.clear();
  pt9_filled.reserve(pt9_filled_n);
  for (unsigned i = 0; i < pt9_filled_n; ++i) {
    pt9_filled.push_back(v38[i % v38.size()]);
  }
  auto pt9 = cc->MakePackedPlaintext(pt9_filled);
  v58[9] = pt9;
  auto pt10_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt10_filled = v39;
  pt10_filled.clear();
  pt10_filled.reserve(pt10_filled_n);
  for (unsigned i = 0; i < pt10_filled_n; ++i) {
    pt10_filled.push_back(v39[i % v39.size()]);
  }
  auto pt10 = cc->MakePackedPlaintext(pt10_filled);
  v58[10] = pt10;
  auto pt11_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt11_filled = v40;
  pt11_filled.clear();
  pt11_filled.reserve(pt11_filled_n);
  for (unsigned i = 0; i < pt11_filled_n; ++i) {
    pt11_filled.push_back(v40[i % v40.size()]);
  }
  auto pt11 = cc->MakePackedPlaintext(pt11_filled);
  v58[11] = pt11;
  auto pt12_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt12_filled = v41;
  pt12_filled.clear();
  pt12_filled.reserve(pt12_filled_n);
  for (unsigned i = 0; i < pt12_filled_n; ++i) {
    pt12_filled.push_back(v41[i % v41.size()]);
  }
  auto pt12 = cc->MakePackedPlaintext(pt12_filled);
  v58[12] = pt12;
  auto pt13_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt13_filled = v42;
  pt13_filled.clear();
  pt13_filled.reserve(pt13_filled_n);
  for (unsigned i = 0; i < pt13_filled_n; ++i) {
    pt13_filled.push_back(v42[i % v42.size()]);
  }
  auto pt13 = cc->MakePackedPlaintext(pt13_filled);
  v58[13] = pt13;
  auto pt14_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt14_filled = v43;
  pt14_filled.clear();
  pt14_filled.reserve(pt14_filled_n);
  for (unsigned i = 0; i < pt14_filled_n; ++i) {
    pt14_filled.push_back(v43[i % v43.size()]);
  }
  auto pt14 = cc->MakePackedPlaintext(pt14_filled);
  v58[14] = pt14;
  auto pt15_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt15_filled = v44;
  pt15_filled.clear();
  pt15_filled.reserve(pt15_filled_n);
  for (unsigned i = 0; i < pt15_filled_n; ++i) {
    pt15_filled.push_back(v44[i % v44.size()]);
  }
  auto pt15 = cc->MakePackedPlaintext(pt15_filled);
  v58[15] = pt15;
  auto pt16_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt16_filled = v45;
  pt16_filled.clear();
  pt16_filled.reserve(pt16_filled_n);
  for (unsigned i = 0; i < pt16_filled_n; ++i) {
    pt16_filled.push_back(v45[i % v45.size()]);
  }
  auto pt16 = cc->MakePackedPlaintext(pt16_filled);
  v58[16] = pt16;
  auto pt17_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt17_filled = v46;
  pt17_filled.clear();
  pt17_filled.reserve(pt17_filled_n);
  for (unsigned i = 0; i < pt17_filled_n; ++i) {
    pt17_filled.push_back(v46[i % v46.size()]);
  }
  auto pt17 = cc->MakePackedPlaintext(pt17_filled);
  v58[17] = pt17;
  auto pt18_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt18_filled = v47;
  pt18_filled.clear();
  pt18_filled.reserve(pt18_filled_n);
  for (unsigned i = 0; i < pt18_filled_n; ++i) {
    pt18_filled.push_back(v47[i % v47.size()]);
  }
  auto pt18 = cc->MakePackedPlaintext(pt18_filled);
  v58[18] = pt18;
  auto pt19_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt19_filled = v48;
  pt19_filled.clear();
  pt19_filled.reserve(pt19_filled_n);
  for (unsigned i = 0; i < pt19_filled_n; ++i) {
    pt19_filled.push_back(v48[i % v48.size()]);
  }
  auto pt19 = cc->MakePackedPlaintext(pt19_filled);
  v58[19] = pt19;
  auto pt20_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt20_filled = v49;
  pt20_filled.clear();
  pt20_filled.reserve(pt20_filled_n);
  for (unsigned i = 0; i < pt20_filled_n; ++i) {
    pt20_filled.push_back(v49[i % v49.size()]);
  }
  auto pt20 = cc->MakePackedPlaintext(pt20_filled);
  v58[20] = pt20;
  auto pt21_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt21_filled = v50;
  pt21_filled.clear();
  pt21_filled.reserve(pt21_filled_n);
  for (unsigned i = 0; i < pt21_filled_n; ++i) {
    pt21_filled.push_back(v50[i % v50.size()]);
  }
  auto pt21 = cc->MakePackedPlaintext(pt21_filled);
  v58[21] = pt21;
  auto pt22_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt22_filled = v51;
  pt22_filled.clear();
  pt22_filled.reserve(pt22_filled_n);
  for (unsigned i = 0; i < pt22_filled_n; ++i) {
    pt22_filled.push_back(v51[i % v51.size()]);
  }
  auto pt22 = cc->MakePackedPlaintext(pt22_filled);
  v58[22] = pt22;
  auto pt23_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt23_filled = v52;
  pt23_filled.clear();
  pt23_filled.reserve(pt23_filled_n);
  for (unsigned i = 0; i < pt23_filled_n; ++i) {
    pt23_filled.push_back(v52[i % v52.size()]);
  }
  auto pt23 = cc->MakePackedPlaintext(pt23_filled);
  v58[23] = pt23;
  auto pt24_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt24_filled = v53;
  pt24_filled.clear();
  pt24_filled.reserve(pt24_filled_n);
  for (unsigned i = 0; i < pt24_filled_n; ++i) {
    pt24_filled.push_back(v53[i % v53.size()]);
  }
  auto pt24 = cc->MakePackedPlaintext(pt24_filled);
  v58[24] = pt24;
  auto pt25_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt25_filled = v54;
  pt25_filled.clear();
  pt25_filled.reserve(pt25_filled_n);
  for (unsigned i = 0; i < pt25_filled_n; ++i) {
    pt25_filled.push_back(v54[i % v54.size()]);
  }
  auto pt25 = cc->MakePackedPlaintext(pt25_filled);
  v58[25] = pt25;
  auto pt26_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt26_filled = v55;
  pt26_filled.clear();
  pt26_filled.reserve(pt26_filled_n);
  for (unsigned i = 0; i < pt26_filled_n; ++i) {
    pt26_filled.push_back(v55[i % v55.size()]);
  }
  auto pt26 = cc->MakePackedPlaintext(pt26_filled);
  v58[26] = pt26;
  auto pt27_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt27_filled = v56;
  pt27_filled.clear();
  pt27_filled.reserve(pt27_filled_n);
  for (unsigned i = 0; i < pt27_filled_n; ++i) {
    pt27_filled.push_back(v56[i % v56.size()]);
  }
  auto pt27 = cc->MakePackedPlaintext(pt27_filled);
  v58[27] = pt27;
  auto pt28_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt28_filled = v57;
  pt28_filled.clear();
  pt28_filled.reserve(pt28_filled_n);
  for (unsigned i = 0; i < pt28_filled_n; ++i) {
    pt28_filled.push_back(v57[i % v57.size()]);
  }
  auto pt28 = cc->MakePackedPlaintext(pt28_filled);
  v58[28] = pt28;
  return v58;
}
std::vector<CiphertextT> conv_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    const std::vector<Plaintext>& v4) {
  std::vector<size_t> v5 = {7, 6};
  std::vector<size_t> v6 = {33, 9};
  std::vector<size_t> v7 = {2, 22, 3};
  std::vector<size_t> v8 = {16, 7, 11, 6, 32, 3, 20, 34, 24};
  std::vector<size_t> v9 = {33, 2, 3};
  std::vector<size_t> v10 = {34, 6};
  [[maybe_unused]] size_t v11 = 0;
  [[maybe_unused]] size_t v12 = 25;
  [[maybe_unused]] size_t v13 = 22;
  [[maybe_unused]] size_t v14 = 20;
  [[maybe_unused]] size_t v15 = 16;
  [[maybe_unused]] size_t v16 = 24;
  [[maybe_unused]] size_t v17 = 7;
  [[maybe_unused]] size_t v18 = 6;
  [[maybe_unused]] size_t v19 = 4;
  [[maybe_unused]] size_t v20 = 3;
  [[maybe_unused]] size_t v21 = 2;
  [[maybe_unused]] size_t v22 = 9;
  [[maybe_unused]] size_t v23 = 11;
  [[maybe_unused]] size_t v24 = 1;
  [[maybe_unused]] size_t v25 = 5;
  [[maybe_unused]] size_t v26 = 8;
  [[maybe_unused]] size_t v27 = 10;
  [[maybe_unused]] size_t v28 = 12;
  [[maybe_unused]] size_t v29 = 13;
  [[maybe_unused]] size_t v30 = 14;
  [[maybe_unused]] size_t v31 = 15;
  [[maybe_unused]] size_t v32 = 17;
  [[maybe_unused]] size_t v33 = 18;
  [[maybe_unused]] size_t v34 = 19;
  [[maybe_unused]] size_t v35 = 21;
  [[maybe_unused]] size_t v36 = 23;
  [[maybe_unused]] size_t v37 = 26;
  [[maybe_unused]] size_t v38 = 27;
  [[maybe_unused]] size_t v39 = 28;
  const auto& ct = v1[0];
  const auto& ct1 = v2[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  std::vector<CiphertextT> v40(9);
  Plaintext pt = v4[0];
  Plaintext pt1 = v4[1];
  Plaintext pt2 = v4[2];
  Plaintext pt3 = v4[3];
  Plaintext pt4 = v4[4];
  Plaintext pt5 = v4[5];
  Plaintext pt6 = v4[6];
  Plaintext pt7 = v4[7];
  Plaintext pt8 = v4[8];
  Plaintext pt9 = v4[9];
  Plaintext pt10 = v4[10];
  Plaintext pt11 = v4[11];
  Plaintext pt12 = v4[12];
  const auto& ct4 = v0[0];
  const auto& ct5 = v3[0];
  auto ct6 = cc->EvalMultNoRelin(ct4, ct5);
  cc->RelinearizeInPlace(ct6);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct6);
  Plaintext pt13 = v4[13];
  Plaintext pt14 = v4[14];
  const auto& ct8 = cc->EvalMult(ct6, pt14);
  std::vector<CiphertextT> v41(2);
  auto v42 = v41;
#pragma omp parallel for
  for (auto v43 = 0; v43 < 2; ++v43) {
    size_t v45 = v10[v43];
    const auto& ct9 = cc->EvalFastRotation(ct6, v45, 2 * cc->GetRingDimension(),
                                           digit_decomp1);
    const std::vector<CiphertextT> v46 = {ct9};
    v42[v43] = v46[0];
  }
  const auto& ct10 = v42[0];
  const auto& ct11 = v42[1];
  Plaintext pt15 = v4[15];
  const auto& ct12 = cc->EvalMult(ct11, pt15);
  Plaintext pt16 = v4[16];
  Plaintext pt17 = v4[17];
  Plaintext pt18 = v4[18];
  std::vector<CiphertextT> v47(3);
  Plaintext pt19 = v4[19];
  Plaintext pt20 = v4[20];
  Plaintext pt21 = v4[21];
  Plaintext pt22 = v4[22];
  Plaintext pt23 = v4[23];
  Plaintext pt24 = v4[24];
  Plaintext pt25 = v4[25];
  Plaintext pt26 = v4[26];
  Plaintext pt27 = v4[27];
  Plaintext pt28 = v4[28];
  const auto& ct13 = cc->EvalMult(ct2, pt28);
  std::vector<CiphertextT> v48(1);
#pragma omp parallel for
  for (auto v50 = 0; v50 < 9; ++v50) {
    size_t v52 = v8[v50];
    const auto& ct14 = cc->EvalFastRotation(
        ct2, v52, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v53 = {ct14};
    v40[v50] = v53[0];
  }
  const auto& ct15 = v40[0];
  const auto& ct16 = v40[1];
  const auto& ct17 = v40[2];
  const auto& ct18 = v40[3];
  const auto& ct19 = v40[4];
  const auto& ct20 = v40[5];
  const auto& ct21 = v40[6];
  const auto& ct22 = v40[7];
  const auto& ct23 = v40[8];
  auto ct24 = cc->EvalMult(ct23, pt);
  const auto& ct25 = cc->EvalMult(ct19, pt1);
  cc->EvalAddInPlace(ct24, ct25);
  const auto& ct27 = cc->EvalMult(ct22, pt2);
  cc->EvalAddInPlace(ct24, ct27);
  const auto& ct29 = cc->EvalMult(ct20, pt3);
  cc->EvalAddInPlace(ct24, ct29);
  const auto& ct31 = cc->EvalMult(ct18, pt4);
  cc->EvalAddInPlace(ct24, ct31);
  const auto& ct33 = cc->EvalMult(ct16, pt5);
  cc->EvalAddInPlace(ct24, ct33);
  const auto& ct35 = cc->EvalMult(ct15, pt6);
  cc->EvalAddInPlace(ct24, ct35);
  const auto& ct37 = cc->EvalMult(ct21, pt7);
  cc->EvalAddInPlace(ct24, ct37);
  auto ct39 = cc->EvalMult(ct23, pt6);
  const auto& ct40 = cc->EvalMult(ct19, pt5);
  cc->EvalAddInPlace(ct39, ct40);
  const auto& ct42 = cc->EvalMult(ct22, pt8);
  cc->EvalAddInPlace(ct39, ct42);
  const auto& ct44 = cc->EvalMult(ct18, pt9);
  cc->EvalAddInPlace(ct39, ct44);
  const auto& ct46 = cc->EvalMult(ct16, pt10);
  cc->EvalAddInPlace(ct39, ct46);
  const auto& ct48 = cc->EvalMult(ct15, pt11);
  cc->EvalAddInPlace(ct39, ct48);
  const auto& ct50 = cc->EvalMult(ct21, pt12);
  cc->EvalAddInPlace(ct39, ct50);
  cc->EvalAddInPlace(ct24, ct39);
  const auto& digit_decomp2 = cc->EvalFastRotationPrecompute(ct24);
  auto ct53 = cc->EvalMult(ct10, pt13);
  cc->EvalAddInPlace(ct53, ct8);
  cc->EvalAddInPlace(ct53, ct12);
  auto ct56 = cc->EvalMult(ct23, pt16);
  const auto& ct57 = cc->EvalMult(ct19, pt17);
  cc->EvalAddInPlace(ct56, ct57);
  const auto& ct59 = cc->EvalMult(ct18, pt13);
  cc->EvalAddInPlace(ct56, ct59);
  const auto& ct61 = cc->EvalMult(ct17, pt15);
  cc->EvalAddInPlace(ct56, ct61);
  cc->EvalAddInPlace(ct53, ct56);
  const auto& digit_decomp3 = cc->EvalFastRotationPrecompute(ct53);
  auto ct64 = cc->EvalMult(ct24, pt18);
  auto v54 = v47;
#pragma omp parallel for
  for (auto v55 = 0; v55 < 3; ++v55) {
    size_t v57 = v7[v55];
    const auto& ct65 = cc->EvalFastRotation(
        ct24, v57, 2 * cc->GetRingDimension(), digit_decomp2);
    const std::vector<CiphertextT> v58 = {ct65};
    v54[v55] = v58[0];
  }
  const auto& ct66 = v54[0];
  const auto& ct67 = v54[1];
  const auto& ct68 = v54[2];
  const auto& ct69 = cc->EvalMult(ct68, pt6);
  cc->EvalAddInPlace(ct64, ct69);
  auto v59 = v41;
#pragma omp parallel for
  for (auto v60 = 0; v60 < 2; ++v60) {
    size_t v62 = v6[v60];
    const auto& ct71 = cc->EvalFastRotation(
        ct24, v62, 2 * cc->GetRingDimension(), digit_decomp2);
    const std::vector<CiphertextT> v63 = {ct71};
    v59[v60] = v63[0];
  }
  const auto& ct72 = v59[0];
  const auto& ct73 = v59[1];
  const auto& ct74 = cc->EvalMult(ct73, pt19);
  cc->EvalAddInPlace(ct64, ct74);
  const auto& ct76 = cc->EvalMult(ct67, pt20);
  cc->EvalAddInPlace(ct64, ct76);
  auto ct78 = cc->EvalMult(ct24, pt6);
  const auto& ct79 = cc->EvalMult(ct66, pt18);
  cc->EvalAddInPlace(ct78, ct79);
  const auto& ct81 = cc->EvalMult(ct68, pt21);
  cc->EvalAddInPlace(ct78, ct81);
  cc->EvalAddInPlace(ct64, ct78);
  auto ct84 = cc->EvalMult(ct72, pt15);
  const auto& ct85 = cc->EvalMult(ct24, pt22);
  cc->EvalAddInPlace(ct84, ct85);
  const auto& ct87 = cc->EvalMult(ct73, pt23);
  cc->EvalAddInPlace(ct84, ct87);
  auto v64 = v47;
#pragma omp parallel for
  for (auto v65 = 0; v65 < 3; ++v65) {
    size_t v67 = v9[v65];
    const auto& ct89 = cc->EvalFastRotation(
        ct53, v67, 2 * cc->GetRingDimension(), digit_decomp3);
    const std::vector<CiphertextT> v68 = {ct89};
    v64[v65] = v68[0];
  }
  const auto& ct90 = v64[0];
  const auto& ct91 = v64[1];
  const auto& ct92 = v64[2];
  const auto& ct93 = cc->EvalMult(ct92, pt23);
  const auto& ct94 = cc->EvalRotate(ct64, 25);
  auto ct95 = cc->EvalMult(ct94, pt24);
  const auto& ct96 = cc->EvalMult(ct64, pt6);
  cc->EvalAddInPlace(ct95, ct96);
  auto ct98 = cc->EvalMult(ct19, pt27);
  cc->EvalAddInPlace(ct98, ct13);
  const auto& ct100 = cc->EvalMult(ct18, pt12);
  cc->EvalAddInPlace(ct98, ct100);
  const auto& ct102 = cc->EvalMult(ct17, pt7);
  cc->EvalAddInPlace(ct98, ct102);
  auto ct104 = cc->EvalMult(ct90, pt22);
  const auto& ct105 = cc->EvalMult(ct91, pt15);
  cc->EvalAddInPlace(ct104, ct105);
  cc->EvalAddInPlace(ct104, ct93);
  cc->EvalAddInPlace(ct84, ct104);
  const auto& digit_decomp4 = cc->EvalFastRotationPrecompute(ct84);
  auto ct109 = cc->EvalMult(ct84, pt15);
  auto v69 = v41;
#pragma omp parallel for
  for (auto v70 = 0; v70 < 2; ++v70) {
    size_t v72 = v5[v70];
    const auto& ct110 = cc->EvalFastRotation(
        ct84, v72, 2 * cc->GetRingDimension(), digit_decomp4);
    const std::vector<CiphertextT> v73 = {ct110};
    v69[v70] = v73[0];
  }
  const auto& ct111 = v69[0];
  const auto& ct112 = v69[1];
  const auto& ct113 = cc->EvalMult(ct112, pt25);
  cc->EvalAddInPlace(ct109, ct113);
  const auto& ct115 = cc->EvalMult(ct111, pt6);
  cc->EvalAddInPlace(ct109, ct115);
  cc->EvalAddInPlace(ct95, ct109);
  auto ct118 = cc->EvalMult(ct95, pt25);
  const auto& ct119 = cc->EvalRotate(ct95, 4);
  const auto& ct120 = cc->EvalMult(ct119, pt26);
  cc->EvalAddInPlace(ct118, ct120);
  cc->EvalAddInPlace(ct118, ct98);
  std::vector<CiphertextT> v74(v48);
  v74[0] = ct118;
  return v74;
}
std::vector<CiphertextT> conv_clone_0_0(CryptoContextT cc,
                                        std::vector<CiphertextT> v0,
                                        std::vector<CiphertextT> v1,
                                        std::vector<CiphertextT> v2,
                                        std::vector<CiphertextT> v3) {
  const auto& v4 = conv_clone_0_0__preprocessing(cc);
  const auto& v5 = conv_clone_0_0__preprocessed(cc, v0, v1, v2, v3, v4);
  return v5;
}
std::vector<CiphertextT> conv_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                       std::vector<int32_t> v0,
                                                       PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 13;
  [[maybe_unused]] size_t v3 = 12;
  [[maybe_unused]] size_t v4 = 9;
  [[maybe_unused]] size_t v5 = 8;
  int32_t v6 = v0[8];
  int32_t v7 = v0[9];
  int32_t v8 = v0[12];
  int32_t v9 = v0[13];
  const std::vector<int32_t> v10 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v7, v1, v1, v8, v9, v1, v1, v6, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v7, v1, v1, v8, v9, v1, v1, v6, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v7, v1, v1, v8, v9, v1, v1, v6,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 128);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 128; ++v11_i1) {
      v11[v11_i1 + 128 * (v11_i0)] =
          v10[0 + v11_i1 * 1 + 128 * (0 + v11_i0 * 1)];
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
std::vector<CiphertextT> conv_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                       std::vector<int32_t> v0,
                                                       PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 13;
  [[maybe_unused]] size_t v3 = 8;
  [[maybe_unused]] size_t v4 = 3;
  [[maybe_unused]] size_t v5 = 15;
  [[maybe_unused]] size_t v6 = 10;
  [[maybe_unused]] size_t v7 = 11;
  [[maybe_unused]] size_t v8 = 9;
  [[maybe_unused]] size_t v9 = 6;
  [[maybe_unused]] size_t v10 = 14;
  [[maybe_unused]] size_t v11 = 5;
  [[maybe_unused]] size_t v12 = 7;
  [[maybe_unused]] size_t v13 = 4;
  [[maybe_unused]] size_t v14 = 2;
  [[maybe_unused]] size_t v15 = 1;
  [[maybe_unused]] size_t v16 = 0;
  int32_t v17 = v0[0];
  int32_t v18 = v0[1];
  int32_t v19 = v0[2];
  int32_t v20 = v0[4];
  int32_t v21 = v0[5];
  int32_t v22 = v0[6];
  int32_t v23 = v0[9];
  int32_t v24 = v0[10];
  int32_t v25 = v0[3];
  int32_t v26 = v0[7];
  int32_t v27 = v0[11];
  int32_t v28 = v0[8];
  int32_t v29 = v0[13];
  int32_t v30 = v0[14];
  int32_t v31 = v0[15];
  const std::vector<int32_t> v32 = {
      v23, v1,  v28, v21, v25, v26, v22, v20, v30, v22, v22, v23, v24, v24, v21,
      v24, v1,  v31, v18, v1,  v1,  v27, v21, v19, v27, v30, v21, v29, v18, v17,
      v20, v26, v22, v23, v24, v19, v23, v1,  v28, v21, v25, v26, v22, v20, v30,
      v22, v22, v23, v24, v24, v21, v24, v1,  v31, v18, v1,  v1,  v27, v21, v19,
      v27, v30, v21, v29, v18, v17, v20, v26, v22, v23, v24, v19, v23, v1,  v28,
      v21, v25, v26, v22, v20, v30, v22, v22, v23, v24, v24, v21, v24, v1,  v31,
      v18, v1,  v1,  v27, v21, v19, v27, v30, v21, v29, v18, v17, v20, v26, v22,
      v23, v24, v19, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1};
  std::vector<int32_t> v33(1 * 128);
  for (int64_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
    for (int64_t v33_i1 = 0; v33_i1 < 128; ++v33_i1) {
      v33[v33_i1 + 128 * (v33_i0)] =
          v32[0 + v33_i1 * 1 + 128 * (0 + v33_i0 * 1)];
    }
  }
  std::vector<int64_t> v34(std::begin(v33), std::end(v33));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v34;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v34[i % v34.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v35 = {ct};
  return v35;
}
std::vector<CiphertextT> conv_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                       std::vector<int32_t> v0,
                                                       PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 8;
  [[maybe_unused]] size_t v3 = 5;
  [[maybe_unused]] size_t v4 = 4;
  [[maybe_unused]] size_t v5 = 7;
  [[maybe_unused]] size_t v6 = 3;
  [[maybe_unused]] size_t v7 = 2;
  [[maybe_unused]] size_t v8 = 1;
  [[maybe_unused]] size_t v9 = 0;
  int32_t v10 = v0[0];
  int32_t v11 = v0[1];
  int32_t v12 = v0[2];
  int32_t v13 = v0[3];
  int32_t v14 = v0[4];
  int32_t v15 = v0[5];
  int32_t v16 = v0[7];
  int32_t v17 = v0[8];
  const std::vector<int32_t> v18 = {
      v13, v1,  v13, v10, v12, v12, v11, v13, v17, v14, v12, v16, v16, v14, v14,
      v17, v1,  v17, v11, v1,  v1,  v17, v13, v11, v15, v16, v11, v16, v10, v10,
      v10, v15, v15, v14, v15, v12, v13, v1,  v13, v10, v12, v12, v11, v13, v17,
      v14, v12, v16, v16, v14, v14, v17, v1,  v17, v11, v1,  v1,  v17, v13, v11,
      v15, v16, v11, v16, v10, v10, v10, v15, v15, v14, v15, v12, v13, v1,  v13,
      v10, v12, v12, v11, v13, v17, v14, v12, v16, v16, v14, v14, v17, v1,  v17,
      v11, v1,  v1,  v17, v13, v11, v15, v16, v11, v16, v10, v10, v10, v15, v15,
      v14, v15, v12, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1};
  std::vector<int32_t> v19(1 * 128);
  for (int64_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
    for (int64_t v19_i1 = 0; v19_i1 < 128; ++v19_i1) {
      v19[v19_i1 + 128 * (v19_i0)] =
          v18[0 + v19_i1 * 1 + 128 * (0 + v19_i0 * 1)];
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
std::vector<CiphertextT> conv_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                       std::vector<int32_t> v0,
                                                       PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 6;
  int32_t v3 = v0[6];
  const std::vector<int32_t> v4 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v3, v1, v1, v3, v3, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v3, v1, v1, v3, v3, v1, v1, v3, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v3, v3, v1, v1, v3,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v5(1 * 128);
  for (int64_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
    for (int64_t v5_i1 = 0; v5_i1 < 128; ++v5_i1) {
      v5[v5_i1 + 128 * (v5_i0)] = v4[0 + v5_i1 * 1 + 128 * (0 + v5_i0 * 1)];
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
std::vector<int32_t> conv_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 11;
  [[maybe_unused]] size_t v2 = 12;
  [[maybe_unused]] size_t v3 = 10;
  [[maybe_unused]] size_t v4 = 15;
  [[maybe_unused]] size_t v5 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(128);
  const auto& v6_cast = pt->GetPackedValue();
  std::vector<int32_t> v6(std::begin(v6_cast), std::end(v6_cast));
  int32_t v7 = v6[15 + 128 * (0)];
  int32_t v8 = v6[10 + 128 * (0)];
  int32_t v9 = v6[12 + 128 * (0)];
  int32_t v10 = v6[11 + 128 * (0)];
  const std::vector<int32_t> v11 = {v7, v8, v9, v10};
  return v11;
}
CryptoContextT conv_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(5);
  params.SetPlaintextModulus(65537);
  params.SetKeySwitchTechnique(HYBRID);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT conv_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                        PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk,
                       {7, 33, 2, 9, 16, 4, 11, 25, 6, 32, 20, 34, 22, 3, 24});
  return cc;
}
