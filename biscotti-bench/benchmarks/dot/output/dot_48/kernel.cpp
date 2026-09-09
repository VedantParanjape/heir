
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

std::vector<Plaintext> dot_clone_0_0__preprocessing(CryptoContextT cc) {
  [[maybe_unused]] size_t v0 = 0;
  [[maybe_unused]] size_t v1 = 1;
  [[maybe_unused]] size_t v2 = 2;
  [[maybe_unused]] size_t v3 = 3;
  [[maybe_unused]] size_t v4 = 4;
  [[maybe_unused]] size_t v5 = 5;
  [[maybe_unused]] size_t v6 = 6;
  [[maybe_unused]] size_t v7 = 7;
  std::vector<int64_t> v8 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v9 = {
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v10 = {
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v11 = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v12 = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v13 = {
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v14 = {
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v15 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<Plaintext> v16(8);
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v8;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v8[i % v8.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  v16[0] = pt;
  auto pt1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v9;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (unsigned i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v9[i % v9.size()]);
  }
  auto pt1 = cc->MakePackedPlaintext(pt1_filled);
  v16[1] = pt1;
  auto pt2_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt2_filled = v10;
  pt2_filled.clear();
  pt2_filled.reserve(pt2_filled_n);
  for (unsigned i = 0; i < pt2_filled_n; ++i) {
    pt2_filled.push_back(v10[i % v10.size()]);
  }
  auto pt2 = cc->MakePackedPlaintext(pt2_filled);
  v16[2] = pt2;
  auto pt3_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt3_filled = v11;
  pt3_filled.clear();
  pt3_filled.reserve(pt3_filled_n);
  for (unsigned i = 0; i < pt3_filled_n; ++i) {
    pt3_filled.push_back(v11[i % v11.size()]);
  }
  auto pt3 = cc->MakePackedPlaintext(pt3_filled);
  v16[3] = pt3;
  auto pt4_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt4_filled = v12;
  pt4_filled.clear();
  pt4_filled.reserve(pt4_filled_n);
  for (unsigned i = 0; i < pt4_filled_n; ++i) {
    pt4_filled.push_back(v12[i % v12.size()]);
  }
  auto pt4 = cc->MakePackedPlaintext(pt4_filled);
  v16[4] = pt4;
  auto pt5_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt5_filled = v13;
  pt5_filled.clear();
  pt5_filled.reserve(pt5_filled_n);
  for (unsigned i = 0; i < pt5_filled_n; ++i) {
    pt5_filled.push_back(v13[i % v13.size()]);
  }
  auto pt5 = cc->MakePackedPlaintext(pt5_filled);
  v16[5] = pt5;
  auto pt6_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt6_filled = v14;
  pt6_filled.clear();
  pt6_filled.reserve(pt6_filled_n);
  for (unsigned i = 0; i < pt6_filled_n; ++i) {
    pt6_filled.push_back(v14[i % v14.size()]);
  }
  auto pt6 = cc->MakePackedPlaintext(pt6_filled);
  v16[6] = pt6;
  auto pt7_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt7_filled = v15;
  pt7_filled.clear();
  pt7_filled.reserve(pt7_filled_n);
  for (unsigned i = 0; i < pt7_filled_n; ++i) {
    pt7_filled.push_back(v15[i % v15.size()]);
  }
  auto pt7 = cc->MakePackedPlaintext(pt7_filled);
  v16[7] = pt7;
  return v16;
}
std::vector<CiphertextT> dot_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    const std::vector<Plaintext>& v4) {
  [[maybe_unused]] size_t v5 = 0;
  [[maybe_unused]] size_t v6 = 4;
  [[maybe_unused]] size_t v7 = 2;
  [[maybe_unused]] size_t v8 = 1;
  [[maybe_unused]] size_t v9 = 24;
  [[maybe_unused]] size_t v10 = 36;
  [[maybe_unused]] size_t v11 = 3;
  [[maybe_unused]] size_t v12 = 5;
  [[maybe_unused]] size_t v13 = 6;
  [[maybe_unused]] size_t v14 = 7;
  const auto& ct = v1[0];
  const auto& ct1 = v2[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& ct4 = v0[0];
  const auto& ct5 = v3[0];
  auto ct6 = cc->EvalMultNoRelin(ct4, ct5);
  cc->RelinearizeInPlace(ct6);
  Plaintext pt = v4[0];
  auto ct8 = cc->EvalMult(ct2, pt);
  const auto& ct9 = cc->EvalRotate(ct2, 24);
  Plaintext pt1 = v4[1];
  const auto& ct10 = cc->EvalMult(ct9, pt1);
  cc->EvalAddInPlace(ct8, ct10);
  Plaintext pt2 = v4[2];
  auto ct12 = cc->EvalMult(ct2, pt2);
  const auto& ct13 = cc->EvalMult(ct9, pt);
  cc->EvalAddInPlace(ct12, ct13);
  Plaintext pt3 = v4[3];
  const auto& ct15 = cc->EvalMult(ct6, pt3);
  cc->EvalAddInPlace(ct12, ct15);
  cc->EvalAddInPlace(ct8, ct12);
  Plaintext pt4 = v4[4];
  auto ct18 = cc->EvalMult(ct2, pt4);
  Plaintext pt5 = v4[5];
  const auto& ct19 = cc->EvalMult(ct6, pt5);
  cc->EvalAddInPlace(ct18, ct19);
  Plaintext pt6 = v4[6];
  auto ct21 = cc->EvalMult(ct8, pt6);
  Plaintext pt7 = v4[7];
  const auto& ct22 = cc->EvalMult(ct6, pt7);
  cc->EvalAddInPlace(ct21, ct22);
  cc->EvalAddInPlace(ct18, ct21);
  auto ct25 = cc->EvalMult(ct18, pt);
  const auto& ct26 = cc->EvalMult(ct9, pt7);
  cc->EvalAddInPlace(ct25, ct26);
  const auto& ct28 = cc->EvalRotate(ct18, 36);
  auto ct29 = cc->EvalMult(ct28, pt);
  const auto& ct30 = cc->EvalMult(ct18, pt7);
  cc->EvalAddInPlace(ct29, ct30);
  cc->EvalAddInPlace(ct25, ct29);
  const auto& ct33 = cc->EvalAdd(ct25, ct28);
  auto ct34 = cc->EvalRotate(ct25, 4);
  cc->EvalAddInPlace(ct34, ct33);
  const auto& ct36 = cc->EvalRotate(ct34, 2);
  cc->EvalAddInPlace(ct34, ct36);
  const auto& ct38 = cc->EvalRotate(ct34, 1);
  std::vector<CiphertextT> v15(1);
  cc->EvalAddInPlace(ct34, ct38);
  std::vector<CiphertextT> v16(v15);
  v16[0] = ct34;
  return v16;
}
std::vector<CiphertextT> dot_clone_0_0(CryptoContextT cc,
                                       std::vector<CiphertextT> v0,
                                       std::vector<CiphertextT> v1,
                                       std::vector<CiphertextT> v2,
                                       std::vector<CiphertextT> v3) {
  const auto& v4 = dot_clone_0_0__preprocessing(cc);
  const auto& v5 = dot_clone_0_0__preprocessed(cc, v0, v1, v2, v3, v4);
  return v5;
}
std::vector<CiphertextT> dot_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 36;
  [[maybe_unused]] size_t v3 = 24;
  [[maybe_unused]] size_t v4 = 44;
  [[maybe_unused]] size_t v5 = 32;
  [[maybe_unused]] size_t v6 = 41;
  [[maybe_unused]] size_t v7 = 29;
  [[maybe_unused]] size_t v8 = 45;
  [[maybe_unused]] size_t v9 = 33;
  [[maybe_unused]] size_t v10 = 12;
  [[maybe_unused]] size_t v11 = 20;
  [[maybe_unused]] size_t v12 = 17;
  [[maybe_unused]] size_t v13 = 8;
  [[maybe_unused]] size_t v14 = 5;
  [[maybe_unused]] size_t v15 = 21;
  [[maybe_unused]] size_t v16 = 0;
  [[maybe_unused]] size_t v17 = 9;
  int32_t v18 = v0[9];
  int32_t v19 = v0[21];
  int32_t v20 = v0[5];
  int32_t v21 = v0[17];
  int32_t v22 = v0[8];
  int32_t v23 = v0[20];
  int32_t v24 = v0[0];
  int32_t v25 = v0[12];
  int32_t v26 = v0[33];
  int32_t v27 = v0[45];
  int32_t v28 = v0[29];
  int32_t v29 = v0[41];
  int32_t v30 = v0[32];
  int32_t v31 = v0[44];
  int32_t v32 = v0[24];
  int32_t v33 = v0[36];
  const std::vector<int32_t> v34 = {
      v1,  v1,  v1,  v1,  v18, v26, v19, v27, v20, v28, v21, v29, v1,  v1,  v1,
      v1,  v22, v30, v23, v31, v24, v32, v25, v33, v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v18, v26, v19, v27, v20, v28, v21, v29,
      v1,  v1,  v1,  v1,  v22, v30, v23, v31, v24, v32, v25, v33, v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v18, v26, v19, v27, v20,
      v28, v21, v29, v1,  v1,  v1,  v1,  v22, v30, v23, v31, v24, v32, v25, v33,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v18, v26,
      v19, v27, v20, v28, v21, v29, v1,  v1,  v1,  v1,  v22, v30, v23, v31, v24,
      v32, v25, v33, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v18, v26, v19, v27, v20, v28, v21, v29, v1,  v1,  v1,  v1,  v22, v30,
      v23, v31, v24, v32, v25, v33, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
  std::vector<int32_t> v35(1 * 256);
  for (int64_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
    for (int64_t v35_i1 = 0; v35_i1 < 256; ++v35_i1) {
      v35[v35_i1 + 256 * (v35_i0)] =
          v34[0 + v35_i1 * 1 + 256 * (0 + v35_i0 * 1)];
    }
  }
  std::vector<int64_t> v36(std::begin(v35), std::end(v35));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v36;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v36[i % v36.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v37 = {ct};
  return v37;
}
std::vector<CiphertextT> dot_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 38;
  [[maybe_unused]] size_t v3 = 26;
  [[maybe_unused]] size_t v4 = 37;
  [[maybe_unused]] size_t v5 = 25;
  [[maybe_unused]] size_t v6 = 43;
  [[maybe_unused]] size_t v7 = 31;
  [[maybe_unused]] size_t v8 = 39;
  [[maybe_unused]] size_t v9 = 27;
  [[maybe_unused]] size_t v10 = 47;
  [[maybe_unused]] size_t v11 = 35;
  [[maybe_unused]] size_t v12 = 46;
  [[maybe_unused]] size_t v13 = 14;
  [[maybe_unused]] size_t v14 = 2;
  [[maybe_unused]] size_t v15 = 42;
  [[maybe_unused]] size_t v16 = 40;
  [[maybe_unused]] size_t v17 = 34;
  [[maybe_unused]] size_t v18 = 30;
  [[maybe_unused]] size_t v19 = 28;
  [[maybe_unused]] size_t v20 = 22;
  [[maybe_unused]] size_t v21 = 13;
  [[maybe_unused]] size_t v22 = 1;
  [[maybe_unused]] size_t v23 = 18;
  [[maybe_unused]] size_t v24 = 19;
  [[maybe_unused]] size_t v25 = 16;
  [[maybe_unused]] size_t v26 = 7;
  [[maybe_unused]] size_t v27 = 10;
  [[maybe_unused]] size_t v28 = 15;
  [[maybe_unused]] size_t v29 = 3;
  [[maybe_unused]] size_t v30 = 6;
  [[maybe_unused]] size_t v31 = 23;
  [[maybe_unused]] size_t v32 = 4;
  [[maybe_unused]] size_t v33 = 11;
  int32_t v34 = v0[11];
  int32_t v35 = v0[23];
  int32_t v36 = v0[3];
  int32_t v37 = v0[15];
  int32_t v38 = v0[7];
  int32_t v39 = v0[19];
  int32_t v40 = v0[1];
  int32_t v41 = v0[13];
  int32_t v42 = v0[10];
  int32_t v43 = v0[22];
  int32_t v44 = v0[4];
  int32_t v45 = v0[16];
  int32_t v46 = v0[6];
  int32_t v47 = v0[18];
  int32_t v48 = v0[2];
  int32_t v49 = v0[14];
  int32_t v50 = v0[35];
  int32_t v51 = v0[47];
  int32_t v52 = v0[27];
  int32_t v53 = v0[39];
  int32_t v54 = v0[31];
  int32_t v55 = v0[43];
  int32_t v56 = v0[25];
  int32_t v57 = v0[37];
  int32_t v58 = v0[34];
  int32_t v59 = v0[46];
  int32_t v60 = v0[28];
  int32_t v61 = v0[40];
  int32_t v62 = v0[30];
  int32_t v63 = v0[42];
  int32_t v64 = v0[26];
  int32_t v65 = v0[38];
  const std::vector<int32_t> v66 = {
      v1,  v1,  v1,  v1,  v34, v50, v35, v51, v36, v52, v37, v53, v1,  v1,  v1,
      v1,  v38, v54, v39, v55, v40, v56, v41, v57, v1,  v1,  v1,  v1,  v42, v58,
      v43, v59, v44, v60, v45, v61, v1,  v1,  v1,  v1,  v46, v62, v47, v63, v48,
      v64, v49, v65, v1,  v1,  v1,  v1,  v34, v50, v35, v51, v36, v52, v37, v53,
      v1,  v1,  v1,  v1,  v38, v54, v39, v55, v40, v56, v41, v57, v1,  v1,  v1,
      v1,  v42, v58, v43, v59, v44, v60, v45, v61, v1,  v1,  v1,  v1,  v46, v62,
      v47, v63, v48, v64, v49, v65, v1,  v1,  v1,  v1,  v34, v50, v35, v51, v36,
      v52, v37, v53, v1,  v1,  v1,  v1,  v38, v54, v39, v55, v40, v56, v41, v57,
      v1,  v1,  v1,  v1,  v42, v58, v43, v59, v44, v60, v45, v61, v1,  v1,  v1,
      v1,  v46, v62, v47, v63, v48, v64, v49, v65, v1,  v1,  v1,  v1,  v34, v50,
      v35, v51, v36, v52, v37, v53, v1,  v1,  v1,  v1,  v38, v54, v39, v55, v40,
      v56, v41, v57, v1,  v1,  v1,  v1,  v42, v58, v43, v59, v44, v60, v45, v61,
      v1,  v1,  v1,  v1,  v46, v62, v47, v63, v48, v64, v49, v65, v1,  v1,  v1,
      v1,  v34, v50, v35, v51, v36, v52, v37, v53, v1,  v1,  v1,  v1,  v38, v54,
      v39, v55, v40, v56, v41, v57, v1,  v1,  v1,  v1,  v42, v58, v43, v59, v44,
      v60, v45, v61, v1,  v1,  v1,  v1,  v46, v62, v47, v63, v48, v64, v49, v65,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
  std::vector<int32_t> v67(1 * 256);
  for (int64_t v67_i0 = 0; v67_i0 < 1; ++v67_i0) {
    for (int64_t v67_i1 = 0; v67_i1 < 256; ++v67_i1) {
      v67[v67_i1 + 256 * (v67_i0)] =
          v66[0 + v67_i1 * 1 + 256 * (0 + v67_i0 * 1)];
    }
  }
  std::vector<int64_t> v68(std::begin(v67), std::end(v67));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v68;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v68[i % v68.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v69 = {ct};
  return v69;
}
std::vector<CiphertextT> dot_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 38;
  [[maybe_unused]] size_t v3 = 26;
  [[maybe_unused]] size_t v4 = 37;
  [[maybe_unused]] size_t v5 = 25;
  [[maybe_unused]] size_t v6 = 43;
  [[maybe_unused]] size_t v7 = 31;
  [[maybe_unused]] size_t v8 = 39;
  [[maybe_unused]] size_t v9 = 27;
  [[maybe_unused]] size_t v10 = 47;
  [[maybe_unused]] size_t v11 = 35;
  [[maybe_unused]] size_t v12 = 46;
  [[maybe_unused]] size_t v13 = 14;
  [[maybe_unused]] size_t v14 = 2;
  [[maybe_unused]] size_t v15 = 42;
  [[maybe_unused]] size_t v16 = 40;
  [[maybe_unused]] size_t v17 = 34;
  [[maybe_unused]] size_t v18 = 30;
  [[maybe_unused]] size_t v19 = 28;
  [[maybe_unused]] size_t v20 = 22;
  [[maybe_unused]] size_t v21 = 13;
  [[maybe_unused]] size_t v22 = 1;
  [[maybe_unused]] size_t v23 = 18;
  [[maybe_unused]] size_t v24 = 19;
  [[maybe_unused]] size_t v25 = 16;
  [[maybe_unused]] size_t v26 = 7;
  [[maybe_unused]] size_t v27 = 10;
  [[maybe_unused]] size_t v28 = 15;
  [[maybe_unused]] size_t v29 = 3;
  [[maybe_unused]] size_t v30 = 6;
  [[maybe_unused]] size_t v31 = 23;
  [[maybe_unused]] size_t v32 = 4;
  [[maybe_unused]] size_t v33 = 11;
  int32_t v34 = v0[11];
  int32_t v35 = v0[23];
  int32_t v36 = v0[3];
  int32_t v37 = v0[15];
  int32_t v38 = v0[7];
  int32_t v39 = v0[19];
  int32_t v40 = v0[1];
  int32_t v41 = v0[13];
  int32_t v42 = v0[10];
  int32_t v43 = v0[22];
  int32_t v44 = v0[4];
  int32_t v45 = v0[16];
  int32_t v46 = v0[6];
  int32_t v47 = v0[18];
  int32_t v48 = v0[2];
  int32_t v49 = v0[14];
  int32_t v50 = v0[35];
  int32_t v51 = v0[47];
  int32_t v52 = v0[27];
  int32_t v53 = v0[39];
  int32_t v54 = v0[31];
  int32_t v55 = v0[43];
  int32_t v56 = v0[25];
  int32_t v57 = v0[37];
  int32_t v58 = v0[34];
  int32_t v59 = v0[46];
  int32_t v60 = v0[28];
  int32_t v61 = v0[40];
  int32_t v62 = v0[30];
  int32_t v63 = v0[42];
  int32_t v64 = v0[26];
  int32_t v65 = v0[38];
  const std::vector<int32_t> v66 = {
      v1,  v1,  v1,  v1,  v34, v50, v35, v51, v36, v52, v37, v53, v1,  v1,  v1,
      v1,  v38, v54, v39, v55, v40, v56, v41, v57, v1,  v1,  v1,  v1,  v42, v58,
      v43, v59, v44, v60, v45, v61, v1,  v1,  v1,  v1,  v46, v62, v47, v63, v48,
      v64, v49, v65, v1,  v1,  v1,  v1,  v34, v50, v35, v51, v36, v52, v37, v53,
      v1,  v1,  v1,  v1,  v38, v54, v39, v55, v40, v56, v41, v57, v1,  v1,  v1,
      v1,  v42, v58, v43, v59, v44, v60, v45, v61, v1,  v1,  v1,  v1,  v46, v62,
      v47, v63, v48, v64, v49, v65, v1,  v1,  v1,  v1,  v34, v50, v35, v51, v36,
      v52, v37, v53, v1,  v1,  v1,  v1,  v38, v54, v39, v55, v40, v56, v41, v57,
      v1,  v1,  v1,  v1,  v42, v58, v43, v59, v44, v60, v45, v61, v1,  v1,  v1,
      v1,  v46, v62, v47, v63, v48, v64, v49, v65, v1,  v1,  v1,  v1,  v34, v50,
      v35, v51, v36, v52, v37, v53, v1,  v1,  v1,  v1,  v38, v54, v39, v55, v40,
      v56, v41, v57, v1,  v1,  v1,  v1,  v42, v58, v43, v59, v44, v60, v45, v61,
      v1,  v1,  v1,  v1,  v46, v62, v47, v63, v48, v64, v49, v65, v1,  v1,  v1,
      v1,  v34, v50, v35, v51, v36, v52, v37, v53, v1,  v1,  v1,  v1,  v38, v54,
      v39, v55, v40, v56, v41, v57, v1,  v1,  v1,  v1,  v42, v58, v43, v59, v44,
      v60, v45, v61, v1,  v1,  v1,  v1,  v46, v62, v47, v63, v48, v64, v49, v65,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
  std::vector<int32_t> v67(1 * 256);
  for (int64_t v67_i0 = 0; v67_i0 < 1; ++v67_i0) {
    for (int64_t v67_i1 = 0; v67_i1 < 256; ++v67_i1) {
      v67[v67_i1 + 256 * (v67_i0)] =
          v66[0 + v67_i1 * 1 + 256 * (0 + v67_i0 * 1)];
    }
  }
  std::vector<int64_t> v68(std::begin(v67), std::end(v67));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v68;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v68[i % v68.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v69 = {ct};
  return v69;
}
std::vector<CiphertextT> dot_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 36;
  [[maybe_unused]] size_t v3 = 24;
  [[maybe_unused]] size_t v4 = 44;
  [[maybe_unused]] size_t v5 = 32;
  [[maybe_unused]] size_t v6 = 41;
  [[maybe_unused]] size_t v7 = 29;
  [[maybe_unused]] size_t v8 = 45;
  [[maybe_unused]] size_t v9 = 33;
  [[maybe_unused]] size_t v10 = 12;
  [[maybe_unused]] size_t v11 = 20;
  [[maybe_unused]] size_t v12 = 17;
  [[maybe_unused]] size_t v13 = 8;
  [[maybe_unused]] size_t v14 = 5;
  [[maybe_unused]] size_t v15 = 21;
  [[maybe_unused]] size_t v16 = 0;
  [[maybe_unused]] size_t v17 = 9;
  int32_t v18 = v0[9];
  int32_t v19 = v0[21];
  int32_t v20 = v0[5];
  int32_t v21 = v0[17];
  int32_t v22 = v0[8];
  int32_t v23 = v0[20];
  int32_t v24 = v0[0];
  int32_t v25 = v0[12];
  int32_t v26 = v0[33];
  int32_t v27 = v0[45];
  int32_t v28 = v0[29];
  int32_t v29 = v0[41];
  int32_t v30 = v0[32];
  int32_t v31 = v0[44];
  int32_t v32 = v0[24];
  int32_t v33 = v0[36];
  const std::vector<int32_t> v34 = {
      v1,  v1,  v1,  v1,  v18, v26, v19, v27, v20, v28, v21, v29, v1,  v1,  v1,
      v1,  v22, v30, v23, v31, v24, v32, v25, v33, v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v18, v26, v19, v27, v20, v28, v21, v29,
      v1,  v1,  v1,  v1,  v22, v30, v23, v31, v24, v32, v25, v33, v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v18, v26, v19, v27, v20,
      v28, v21, v29, v1,  v1,  v1,  v1,  v22, v30, v23, v31, v24, v32, v25, v33,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v18, v26,
      v19, v27, v20, v28, v21, v29, v1,  v1,  v1,  v1,  v22, v30, v23, v31, v24,
      v32, v25, v33, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v18, v26, v19, v27, v20, v28, v21, v29, v1,  v1,  v1,  v1,  v22, v30,
      v23, v31, v24, v32, v25, v33, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
  std::vector<int32_t> v35(1 * 256);
  for (int64_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
    for (int64_t v35_i1 = 0; v35_i1 < 256; ++v35_i1) {
      v35[v35_i1 + 256 * (v35_i0)] =
          v34[0 + v35_i1 * 1 + 256 * (0 + v35_i0 * 1)];
    }
  }
  std::vector<int64_t> v36(std::begin(v35), std::end(v35));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v36;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v36[i % v36.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v37 = {ct};
  return v37;
}
std::vector<int32_t> dot_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 16;
  [[maybe_unused]] size_t v2 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v3_cast = pt->GetPackedValue();
  std::vector<int32_t> v3(std::begin(v3_cast), std::end(v3_cast));
  int32_t v4 = v3[16 + 256 * (0)];
  const std::vector<int32_t> v5 = {v4};
  return v5;
}
CryptoContextT dot_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(4);
  params.SetPlaintextModulus(65537);
  params.SetKeySwitchTechnique(HYBRID);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT dot_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                       PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {2, 4, 1, 36, 24});
  return cc;
}
