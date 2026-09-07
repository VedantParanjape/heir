
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
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v3 = {
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
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
  [[maybe_unused]] size_t v5 = 0;
  [[maybe_unused]] size_t v6 = 1;
  const auto& ct = v1[0];
  const auto& ct1 = v2[0];
  const auto& ct2 = cc->EvalMultNoRelin(ct, ct1);
  const auto& ct3 = v0[0];
  const auto& ct4 = v3[0];
  const auto& ct5 = cc->EvalMultNoRelin(ct3, ct4);
  Plaintext pt = v4[0];
  auto ct6 = cc->EvalMult(ct2, pt);
  Plaintext pt1 = v4[1];
  const auto& ct7 = cc->EvalMult(ct5, pt1);
  cc->EvalAddInPlace(ct6, ct7);
  auto ct9 = cc->EvalMult(ct2, pt1);
  const auto& ct10 = cc->EvalMult(ct5, pt);
  cc->EvalAddInPlace(ct9, ct10);
  cc->EvalAddInPlace(ct6, ct9);
  cc->RelinearizeInPlace(ct6);
  const auto& ct14 = cc->EvalRotate(ct6, 1);
  std::vector<CiphertextT> v7(1);
  cc->EvalAddInPlace(ct6, ct14);
  std::vector<CiphertextT> v8(v7);
  v8[0] = ct6;
  return v8;
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
  [[maybe_unused]] size_t v2 = 15;
  [[maybe_unused]] size_t v3 = 10;
  [[maybe_unused]] size_t v4 = 11;
  [[maybe_unused]] size_t v5 = 14;
  [[maybe_unused]] size_t v6 = 13;
  [[maybe_unused]] size_t v7 = 8;
  [[maybe_unused]] size_t v8 = 9;
  [[maybe_unused]] size_t v9 = 12;
  [[maybe_unused]] size_t v10 = 7;
  [[maybe_unused]] size_t v11 = 2;
  [[maybe_unused]] size_t v12 = 3;
  [[maybe_unused]] size_t v13 = 6;
  [[maybe_unused]] size_t v14 = 5;
  [[maybe_unused]] size_t v15 = 1;
  [[maybe_unused]] size_t v16 = 0;
  [[maybe_unused]] size_t v17 = 4;
  int32_t v18 = v0[4];
  int32_t v19 = v0[1];
  int32_t v20 = v0[0];
  int32_t v21 = v0[5];
  int32_t v22 = v0[6];
  int32_t v23 = v0[3];
  int32_t v24 = v0[2];
  int32_t v25 = v0[7];
  int32_t v26 = v0[12];
  int32_t v27 = v0[9];
  int32_t v28 = v0[8];
  int32_t v29 = v0[13];
  int32_t v30 = v0[14];
  int32_t v31 = v0[11];
  int32_t v32 = v0[10];
  int32_t v33 = v0[15];
  const std::vector<int32_t> v34 = {
      v18, v22, v18, v22, v26, v30, v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v19, v23, v27, v31,
      v27, v31, v20, v24, v20, v24, v28, v32, v28, v32, v21, v25, v21, v25, v29,
      v33, v29, v33, v18, v22, v18, v22, v26, v30, v26, v30, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v19,
      v23, v27, v31, v27, v31, v20, v24, v20, v24, v28, v32, v28, v32, v21, v25,
      v21, v25, v29, v33, v29, v33, v18, v22, v18, v22, v26, v30, v26, v30, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v19, v23, v19, v23, v27, v31, v27, v31, v20, v24, v20, v24, v28, v32, v28,
      v32, v21, v25, v21, v25, v29, v33, v29, v33, v18, v22, v18, v22, v26, v30,
      v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v19, v23, v19, v23, v27, v31, v27, v31, v20, v24, v20, v24,
      v28, v32, v28, v32, v21, v25, v21, v25, v29, v33, v29, v33, v18, v22, v18,
      v22, v26, v30, v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v19, v23, v27, v31, v27, v31, v20,
      v24, v20, v24, v28, v32, v28, v32, v21, v25, v21, v25, v29, v33, v29, v33,
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
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 14;
  [[maybe_unused]] size_t v3 = 11;
  [[maybe_unused]] size_t v4 = 10;
  [[maybe_unused]] size_t v5 = 15;
  [[maybe_unused]] size_t v6 = 12;
  [[maybe_unused]] size_t v7 = 9;
  [[maybe_unused]] size_t v8 = 8;
  [[maybe_unused]] size_t v9 = 13;
  [[maybe_unused]] size_t v10 = 6;
  [[maybe_unused]] size_t v11 = 3;
  [[maybe_unused]] size_t v12 = 2;
  [[maybe_unused]] size_t v13 = 7;
  [[maybe_unused]] size_t v14 = 4;
  [[maybe_unused]] size_t v15 = 1;
  [[maybe_unused]] size_t v16 = 0;
  [[maybe_unused]] size_t v17 = 5;
  int32_t v18 = v0[5];
  int32_t v19 = v0[0];
  int32_t v20 = v0[1];
  int32_t v21 = v0[4];
  int32_t v22 = v0[7];
  int32_t v23 = v0[2];
  int32_t v24 = v0[3];
  int32_t v25 = v0[6];
  int32_t v26 = v0[13];
  int32_t v27 = v0[8];
  int32_t v28 = v0[9];
  int32_t v29 = v0[12];
  int32_t v30 = v0[15];
  int32_t v31 = v0[10];
  int32_t v32 = v0[11];
  int32_t v33 = v0[14];
  const std::vector<int32_t> v34 = {
      v18, v22, v18, v22, v26, v30, v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v19, v23, v27, v31,
      v27, v31, v20, v24, v20, v24, v28, v32, v28, v32, v21, v25, v21, v25, v29,
      v33, v29, v33, v18, v22, v18, v22, v26, v30, v26, v30, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v19,
      v23, v27, v31, v27, v31, v20, v24, v20, v24, v28, v32, v28, v32, v21, v25,
      v21, v25, v29, v33, v29, v33, v18, v22, v18, v22, v26, v30, v26, v30, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v19, v23, v19, v23, v27, v31, v27, v31, v20, v24, v20, v24, v28, v32, v28,
      v32, v21, v25, v21, v25, v29, v33, v29, v33, v18, v22, v18, v22, v26, v30,
      v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v19, v23, v19, v23, v27, v31, v27, v31, v20, v24, v20, v24,
      v28, v32, v28, v32, v21, v25, v21, v25, v29, v33, v29, v33, v18, v22, v18,
      v22, v26, v30, v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v19, v23, v27, v31, v27, v31, v20,
      v24, v20, v24, v28, v32, v28, v32, v21, v25, v21, v25, v29, v33, v29, v33,
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
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 11;
  [[maybe_unused]] size_t v3 = 15;
  [[maybe_unused]] size_t v4 = 10;
  [[maybe_unused]] size_t v5 = 14;
  [[maybe_unused]] size_t v6 = 3;
  [[maybe_unused]] size_t v7 = 7;
  [[maybe_unused]] size_t v8 = 2;
  [[maybe_unused]] size_t v9 = 6;
  [[maybe_unused]] size_t v10 = 9;
  [[maybe_unused]] size_t v11 = 13;
  [[maybe_unused]] size_t v12 = 8;
  [[maybe_unused]] size_t v13 = 12;
  [[maybe_unused]] size_t v14 = 1;
  [[maybe_unused]] size_t v15 = 5;
  [[maybe_unused]] size_t v16 = 0;
  [[maybe_unused]] size_t v17 = 4;
  int32_t v18 = v0[4];
  int32_t v19 = v0[0];
  int32_t v20 = v0[5];
  int32_t v21 = v0[1];
  int32_t v22 = v0[12];
  int32_t v23 = v0[8];
  int32_t v24 = v0[13];
  int32_t v25 = v0[9];
  int32_t v26 = v0[6];
  int32_t v27 = v0[2];
  int32_t v28 = v0[7];
  int32_t v29 = v0[3];
  int32_t v30 = v0[14];
  int32_t v31 = v0[10];
  int32_t v32 = v0[15];
  int32_t v33 = v0[11];
  const std::vector<int32_t> v34 = {
      v18, v22, v26, v30, v18, v22, v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v27, v31, v19, v23,
      v27, v31, v20, v24, v28, v32, v20, v24, v28, v32, v21, v25, v29, v33, v21,
      v25, v29, v33, v18, v22, v26, v30, v18, v22, v26, v30, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v27,
      v31, v19, v23, v27, v31, v20, v24, v28, v32, v20, v24, v28, v32, v21, v25,
      v29, v33, v21, v25, v29, v33, v18, v22, v26, v30, v18, v22, v26, v30, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v19, v23, v27, v31, v19, v23, v27, v31, v20, v24, v28, v32, v20, v24, v28,
      v32, v21, v25, v29, v33, v21, v25, v29, v33, v18, v22, v26, v30, v18, v22,
      v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v19, v23, v27, v31, v19, v23, v27, v31, v20, v24, v28, v32,
      v20, v24, v28, v32, v21, v25, v29, v33, v21, v25, v29, v33, v18, v22, v26,
      v30, v18, v22, v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v27, v31, v19, v23, v27, v31, v20,
      v24, v28, v32, v20, v24, v28, v32, v21, v25, v29, v33, v21, v25, v29, v33,
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
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 15;
  [[maybe_unused]] size_t v3 = 11;
  [[maybe_unused]] size_t v4 = 14;
  [[maybe_unused]] size_t v5 = 10;
  [[maybe_unused]] size_t v6 = 7;
  [[maybe_unused]] size_t v7 = 3;
  [[maybe_unused]] size_t v8 = 6;
  [[maybe_unused]] size_t v9 = 2;
  [[maybe_unused]] size_t v10 = 13;
  [[maybe_unused]] size_t v11 = 9;
  [[maybe_unused]] size_t v12 = 12;
  [[maybe_unused]] size_t v13 = 8;
  [[maybe_unused]] size_t v14 = 5;
  [[maybe_unused]] size_t v15 = 1;
  [[maybe_unused]] size_t v16 = 4;
  [[maybe_unused]] size_t v17 = 0;
  int32_t v18 = v0[0];
  int32_t v19 = v0[4];
  int32_t v20 = v0[1];
  int32_t v21 = v0[5];
  int32_t v22 = v0[8];
  int32_t v23 = v0[12];
  int32_t v24 = v0[9];
  int32_t v25 = v0[13];
  int32_t v26 = v0[2];
  int32_t v27 = v0[6];
  int32_t v28 = v0[3];
  int32_t v29 = v0[7];
  int32_t v30 = v0[10];
  int32_t v31 = v0[14];
  int32_t v32 = v0[11];
  int32_t v33 = v0[15];
  const std::vector<int32_t> v34 = {
      v18, v22, v26, v30, v18, v22, v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v27, v31, v19, v23,
      v27, v31, v20, v24, v28, v32, v20, v24, v28, v32, v21, v25, v29, v33, v21,
      v25, v29, v33, v18, v22, v26, v30, v18, v22, v26, v30, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v27,
      v31, v19, v23, v27, v31, v20, v24, v28, v32, v20, v24, v28, v32, v21, v25,
      v29, v33, v21, v25, v29, v33, v18, v22, v26, v30, v18, v22, v26, v30, v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v19, v23, v27, v31, v19, v23, v27, v31, v20, v24, v28, v32, v20, v24, v28,
      v32, v21, v25, v29, v33, v21, v25, v29, v33, v18, v22, v26, v30, v18, v22,
      v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v19, v23, v27, v31, v19, v23, v27, v31, v20, v24, v28, v32,
      v20, v24, v28, v32, v21, v25, v29, v33, v21, v25, v29, v33, v18, v22, v26,
      v30, v18, v22, v26, v30, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v19, v23, v27, v31, v19, v23, v27, v31, v20,
      v24, v28, v32, v20, v24, v28, v32, v21, v25, v29, v33, v21, v25, v29, v33,
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
std::vector<int32_t> mm_clone_0_0__decrypt__result0(CryptoContextT cc,
                                                    std::vector<CiphertextT> v0,
                                                    PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 46;
  [[maybe_unused]] size_t v2 = 44;
  [[maybe_unused]] size_t v3 = 38;
  [[maybe_unused]] size_t v4 = 30;
  [[maybe_unused]] size_t v5 = 36;
  [[maybe_unused]] size_t v6 = 28;
  [[maybe_unused]] size_t v7 = 42;
  [[maybe_unused]] size_t v8 = 6;
  [[maybe_unused]] size_t v9 = 40;
  [[maybe_unused]] size_t v10 = 4;
  [[maybe_unused]] size_t v11 = 34;
  [[maybe_unused]] size_t v12 = 2;
  [[maybe_unused]] size_t v13 = 26;
  [[maybe_unused]] size_t v14 = 32;
  [[maybe_unused]] size_t v15 = 24;
  [[maybe_unused]] size_t v16 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(256);
  const auto& v17_cast = pt->GetPackedValue();
  std::vector<int32_t> v17(std::begin(v17_cast), std::end(v17_cast));
  int32_t v18 = v17[24 + 256 * (0)];
  int32_t v19 = v17[32 + 256 * (0)];
  int32_t v20 = v17[26 + 256 * (0)];
  int32_t v21 = v17[34 + 256 * (0)];
  int32_t v22 = v17[0 + 256 * (0)];
  int32_t v23 = v17[40 + 256 * (0)];
  int32_t v24 = v17[2 + 256 * (0)];
  int32_t v25 = v17[42 + 256 * (0)];
  int32_t v26 = v17[28 + 256 * (0)];
  int32_t v27 = v17[36 + 256 * (0)];
  int32_t v28 = v17[30 + 256 * (0)];
  int32_t v29 = v17[38 + 256 * (0)];
  int32_t v30 = v17[4 + 256 * (0)];
  int32_t v31 = v17[44 + 256 * (0)];
  int32_t v32 = v17[6 + 256 * (0)];
  int32_t v33 = v17[46 + 256 * (0)];
  const std::vector<int32_t> v34 = {v18, v19, v20, v21, v22, v23, v24, v25,
                                    v26, v27, v28, v29, v30, v31, v32, v33};
  return v34;
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
  cc->EvalRotateKeyGen(sk, {1});
  return cc;
}
