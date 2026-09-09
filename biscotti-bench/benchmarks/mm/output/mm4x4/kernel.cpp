
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

std::vector<Plaintext> mm_clone_0_0__preprocessing(CryptoContextT cc) {
  [[maybe_unused]] size_t v0 = 0;
  [[maybe_unused]] size_t v1 = 1;
  std::vector<int64_t> v2 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v3 = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
      1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<Plaintext> v4(2);
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v2;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v2[i % v2.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  v4[0] = pt;
  auto pt1_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt1_filled = v3;
  pt1_filled.clear();
  pt1_filled.reserve(pt1_filled_n);
  for (unsigned i = 0; i < pt1_filled_n; ++i) {
    pt1_filled.push_back(v3[i % v3.size()]);
  }
  auto pt1 = cc->MakePackedPlaintext(pt1_filled);
  v4[1] = pt1;
  return v4;
}
std::vector<CiphertextT> mm_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    const std::vector<Plaintext>& v4) {
  std::vector<size_t> v5 = {55, 1};
  [[maybe_unused]] size_t v6 = 0;
  [[maybe_unused]] size_t v7 = 1;
  const auto& ct = v1[0];
  const auto& ct1 = v2[0];
  const auto& ct2 = cc->EvalMultNoRelin(ct, ct1);
  const auto& ct3 = v0[0];
  const auto& ct4 = v3[0];
  auto ct5 = cc->EvalMultNoRelin(ct3, ct4);
  cc->EvalAddInPlace(ct5, ct2);
  cc->RelinearizeInPlace(ct5);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct5);
  Plaintext pt = v4[0];
  Plaintext pt1 = v4[1];
  const auto& ct8 = cc->EvalMult(ct5, pt1);
  auto ct9 = cc->EvalMult(ct5, pt);
  std::vector<CiphertextT> v8(2);
#pragma omp parallel for
  for (auto v10 = 0; v10 < 2; ++v10) {
    size_t v12 = v5[v10];
    const auto& ct10 = cc->EvalFastRotation(
        ct5, v12, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v13 = {ct10};
    v8[v10] = v13[0];
  }
  const auto& ct11 = v8[0];
  const auto& ct12 = v8[1];
  const auto& ct13 = cc->EvalMult(ct12, pt1);
  cc->EvalAddInPlace(ct9, ct13);
  std::vector<CiphertextT> v14(1);
  auto ct15 = cc->EvalMult(ct11, pt);
  cc->EvalAddInPlace(ct15, ct8);
  cc->EvalAddInPlace(ct15, ct9);
  std::vector<CiphertextT> v15(v14);
  v15[0] = ct15;
  return v15;
}
std::vector<CiphertextT> mm_clone_0_0(CryptoContextT cc,
                                      std::vector<CiphertextT> v0,
                                      std::vector<CiphertextT> v1,
                                      std::vector<CiphertextT> v2,
                                      std::vector<CiphertextT> v3) {
  const auto& v4 = mm_clone_0_0__preprocessing(cc);
  const auto& v5 = mm_clone_0_0__preprocessed(cc, v0, v1, v2, v3, v4);
  return v5;
}
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 14;
  [[maybe_unused]] size_t v3 = 12;
  [[maybe_unused]] size_t v4 = 10;
  [[maybe_unused]] size_t v5 = 6;
  [[maybe_unused]] size_t v6 = 2;
  [[maybe_unused]] size_t v7 = 4;
  [[maybe_unused]] size_t v8 = 8;
  [[maybe_unused]] size_t v9 = 0;
  int32_t v10 = v0[0];
  int32_t v11 = v0[4];
  int32_t v12 = v0[2];
  int32_t v13 = v0[6];
  int32_t v14 = v0[8];
  int32_t v15 = v0[12];
  int32_t v16 = v0[10];
  int32_t v17 = v0[14];
  const std::vector<int32_t> v18 = {
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v10, v12, v14, v16, v14,
      v16, v11, v13, v11, v13, v15, v17, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v11, v13, v11, v13, v15,
      v17, v15, v17, v10, v12, v10, v12, v14, v16, v14, v16, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v10, v12, v10, v12, v14, v16, v14, v16, v11, v13, v11,
      v13, v15, v17, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v11, v13, v11, v13, v15, v17, v15, v17, v10,
      v12, v10, v12, v14, v16, v14, v16, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v10, v12, v10, v12, v14, v16, v14, v16, v11, v13, v11, v13, v15, v17, v15,
      v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v11, v13, v11, v13, v15, v17, v15, v17, v10, v12, v10, v12, v14,
      v16, v14, v16, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v10, v12,
      v14, v16, v14, v16, v11, v13, v11, v13, v15, v17, v15, v17, v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v11, v13,
      v11, v13, v15, v17, v15, v17, v10, v12, v10, v12, v14, v16, v14, v16, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
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
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 15;
  [[maybe_unused]] size_t v3 = 13;
  [[maybe_unused]] size_t v4 = 11;
  [[maybe_unused]] size_t v5 = 7;
  [[maybe_unused]] size_t v6 = 9;
  [[maybe_unused]] size_t v7 = 3;
  [[maybe_unused]] size_t v8 = 5;
  [[maybe_unused]] size_t v9 = 1;
  int32_t v10 = v0[1];
  int32_t v11 = v0[5];
  int32_t v12 = v0[3];
  int32_t v13 = v0[7];
  int32_t v14 = v0[9];
  int32_t v15 = v0[13];
  int32_t v16 = v0[11];
  int32_t v17 = v0[15];
  const std::vector<int32_t> v18 = {
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v10, v12, v14, v16, v14,
      v16, v11, v13, v11, v13, v15, v17, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v11, v13, v11, v13, v15,
      v17, v15, v17, v10, v12, v10, v12, v14, v16, v14, v16, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v10, v12, v10, v12, v14, v16, v14, v16, v11, v13, v11,
      v13, v15, v17, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v11, v13, v11, v13, v15, v17, v15, v17, v10,
      v12, v10, v12, v14, v16, v14, v16, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v10, v12, v10, v12, v14, v16, v14, v16, v11, v13, v11, v13, v15, v17, v15,
      v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v11, v13, v11, v13, v15, v17, v15, v17, v10, v12, v10, v12, v14,
      v16, v14, v16, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v10, v12,
      v14, v16, v14, v16, v11, v13, v11, v13, v15, v17, v15, v17, v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v11, v13,
      v11, v13, v15, v17, v15, v17, v10, v12, v10, v12, v14, v16, v14, v16, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
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
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 14;
  [[maybe_unused]] size_t v3 = 15;
  [[maybe_unused]] size_t v4 = 6;
  [[maybe_unused]] size_t v5 = 7;
  [[maybe_unused]] size_t v6 = 12;
  [[maybe_unused]] size_t v7 = 13;
  [[maybe_unused]] size_t v8 = 4;
  [[maybe_unused]] size_t v9 = 5;
  int32_t v10 = v0[5];
  int32_t v11 = v0[4];
  int32_t v12 = v0[13];
  int32_t v13 = v0[12];
  int32_t v14 = v0[7];
  int32_t v15 = v0[6];
  int32_t v16 = v0[15];
  int32_t v17 = v0[14];
  const std::vector<int32_t> v18 = {
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v14, v16, v10, v12, v14,
      v16, v11, v13, v15, v17, v11, v13, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v14, v16, v10,
      v12, v14, v16, v11, v13, v15, v17, v11, v13, v15, v17, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v10, v12, v14, v16, v10, v12, v14, v16, v11, v13, v15,
      v17, v11, v13, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v14, v16, v10, v12, v14, v16, v11,
      v13, v15, v17, v11, v13, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v10, v12, v14, v16, v10, v12, v14, v16, v11, v13, v15, v17, v11, v13, v15,
      v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v10, v12, v14, v16, v10, v12, v14, v16, v11, v13, v15, v17, v11,
      v13, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v14, v16,
      v10, v12, v14, v16, v11, v13, v15, v17, v11, v13, v15, v17, v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12,
      v14, v16, v10, v12, v14, v16, v11, v13, v15, v17, v11, v13, v15, v17, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
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
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 11;
  [[maybe_unused]] size_t v3 = 2;
  [[maybe_unused]] size_t v4 = 10;
  [[maybe_unused]] size_t v5 = 3;
  [[maybe_unused]] size_t v6 = 9;
  [[maybe_unused]] size_t v7 = 8;
  [[maybe_unused]] size_t v8 = 0;
  [[maybe_unused]] size_t v9 = 1;
  int32_t v10 = v0[1];
  int32_t v11 = v0[0];
  int32_t v12 = v0[9];
  int32_t v13 = v0[8];
  int32_t v14 = v0[3];
  int32_t v15 = v0[2];
  int32_t v16 = v0[11];
  int32_t v17 = v0[10];
  const std::vector<int32_t> v18 = {
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v14, v16, v10, v12, v14,
      v16, v11, v13, v15, v17, v11, v13, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v14, v16, v10,
      v12, v14, v16, v11, v13, v15, v17, v11, v13, v15, v17, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v10, v12, v14, v16, v10, v12, v14, v16, v11, v13, v15,
      v17, v11, v13, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v14, v16, v10, v12, v14, v16, v11,
      v13, v15, v17, v11, v13, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v10, v12, v14, v16, v10, v12, v14, v16, v11, v13, v15, v17, v11, v13, v15,
      v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v10, v12, v14, v16, v10, v12, v14, v16, v11, v13, v15, v17, v11,
      v13, v15, v17, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12, v14, v16,
      v10, v12, v14, v16, v11, v13, v15, v17, v11, v13, v15, v17, v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v10, v12,
      v14, v16, v10, v12, v14, v16, v11, v13, v15, v17, v11, v13, v15, v17, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1};
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
std::vector<int32_t> mm_clone_0_0__decrypt__result0(CryptoContextT cc,
                                                    std::vector<CiphertextT> v0,
                                                    PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 47;
  [[maybe_unused]] size_t v2 = 22;
  [[maybe_unused]] size_t v3 = 44;
  [[maybe_unused]] size_t v4 = 20;
  [[maybe_unused]] size_t v5 = 14;
  [[maybe_unused]] size_t v6 = 54;
  [[maybe_unused]] size_t v7 = 12;
  [[maybe_unused]] size_t v8 = 52;
  [[maybe_unused]] size_t v9 = 42;
  [[maybe_unused]] size_t v10 = 19;
  [[maybe_unused]] size_t v11 = 40;
  [[maybe_unused]] size_t v12 = 16;
  [[maybe_unused]] size_t v13 = 11;
  [[maybe_unused]] size_t v14 = 50;
  [[maybe_unused]] size_t v15 = 8;
  [[maybe_unused]] size_t v16 = 49;
  [[maybe_unused]] size_t v17 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v18_cast = pt->GetPackedValue();
  std::vector<int32_t> v18(std::begin(v18_cast), std::end(v18_cast));
  int32_t v19 = v18[49 + 256 * (0)];
  int32_t v20 = v18[8 + 256 * (0)];
  int32_t v21 = v18[50 + 256 * (0)];
  int32_t v22 = v18[11 + 256 * (0)];
  int32_t v23 = v18[16 + 256 * (0)];
  int32_t v24 = v18[40 + 256 * (0)];
  int32_t v25 = v18[19 + 256 * (0)];
  int32_t v26 = v18[42 + 256 * (0)];
  int32_t v27 = v18[52 + 256 * (0)];
  int32_t v28 = v18[12 + 256 * (0)];
  int32_t v29 = v18[54 + 256 * (0)];
  int32_t v30 = v18[14 + 256 * (0)];
  int32_t v31 = v18[20 + 256 * (0)];
  int32_t v32 = v18[44 + 256 * (0)];
  int32_t v33 = v18[22 + 256 * (0)];
  int32_t v34 = v18[47 + 256 * (0)];
  const std::vector<int32_t> v35 = {v19, v20, v21, v22, v23, v24, v25, v26,
                                    v27, v28, v29, v30, v31, v32, v33, v34};
  return v35;
}
CryptoContextT mm_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(2);
  params.SetPlaintextModulus(65537);
  params.SetKeySwitchTechnique(HYBRID);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT mm_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                      PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {1, 55});
  return cc;
}
