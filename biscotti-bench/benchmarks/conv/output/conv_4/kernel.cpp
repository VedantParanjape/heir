
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
  std::vector<int64_t> v2 = {
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v3 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
std::vector<CiphertextT> conv_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    const std::vector<Plaintext>& v2) {
  std::vector<size_t> v3 = {16, 48, 32};
  std::vector<size_t> v4 = {12, 4};
  [[maybe_unused]] size_t v5 = 2;
  [[maybe_unused]] size_t v6 = 0;
  [[maybe_unused]] size_t v7 = 1;
  const auto& ct = v0[0];
  const auto& ct1 = v1[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  Plaintext pt = v2[0];
  auto ct4 = cc->EvalMult(ct2, pt);
  std::vector<CiphertextT> v8(2);
#pragma omp parallel for
  for (auto v10 = 0; v10 < 2; ++v10) {
    size_t v12 = v4[v10];
    const auto& ct5 = cc->EvalFastRotation(ct2, v12, 2 * cc->GetRingDimension(),
                                           digit_decomp);
    const std::vector<CiphertextT> v13 = {ct5};
    v8[v10] = v13[0];
  }
  auto ct6 = v8[0];
  const auto& ct7 = v8[1];
  Plaintext pt1 = v2[1];
  const auto& ct8 = cc->EvalMult(ct7, pt1);
  cc->EvalAddInPlace(ct4, ct8);
  cc->EvalAddInPlace(ct6, ct4);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct6);
  std::vector<CiphertextT> v14(3);
  std::vector<CiphertextT> v15(1);
#pragma omp parallel for
  for (auto v17 = 0; v17 < 3; ++v17) {
    size_t v19 = v3[v17];
    const auto& ct11 = cc->EvalFastRotation(
        ct6, v19, 2 * cc->GetRingDimension(), digit_decomp1);
    const std::vector<CiphertextT> v20 = {ct11};
    v14[v17] = v20[0];
  }
  auto ct12 = v14[0];
  const auto& ct13 = v14[1];
  const auto& ct14 = v14[2];
  cc->EvalAddInPlace(ct6, ct14);
  cc->EvalAddInPlace(ct12, ct13);
  cc->EvalAddInPlace(ct12, ct6);
  cc->EvalAddInPlace(ct12, ct7);
  std::vector<CiphertextT> v21(v15);
  v21[0] = ct12;
  return v21;
}
std::vector<CiphertextT> conv_clone_0_0(CryptoContextT cc,
                                        std::vector<CiphertextT> v0,
                                        std::vector<CiphertextT> v1) {
  const auto& v2 = conv_clone_0_0__preprocessing(cc);
  const auto& v3 = conv_clone_0_0__preprocessed(cc, v0, v1, v2);
  return v3;
}
std::vector<CiphertextT> conv_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                       std::vector<int32_t> v0,
                                                       PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 15;
  [[maybe_unused]] size_t v3 = 14;
  [[maybe_unused]] size_t v4 = 3;
  [[maybe_unused]] size_t v5 = 13;
  [[maybe_unused]] size_t v6 = 11;
  [[maybe_unused]] size_t v7 = 7;
  [[maybe_unused]] size_t v8 = 2;
  [[maybe_unused]] size_t v9 = 1;
  [[maybe_unused]] size_t v10 = 5;
  [[maybe_unused]] size_t v11 = 12;
  [[maybe_unused]] size_t v12 = 8;
  [[maybe_unused]] size_t v13 = 10;
  [[maybe_unused]] size_t v14 = 4;
  [[maybe_unused]] size_t v15 = 6;
  [[maybe_unused]] size_t v16 = 0;
  [[maybe_unused]] size_t v17 = 9;
  int32_t v18 = v0[9];
  int32_t v19 = v0[6];
  int32_t v20 = v0[10];
  int32_t v21 = v0[8];
  int32_t v22 = v0[5];
  int32_t v23 = v0[4];
  int32_t v24 = v0[1];
  int32_t v25 = v0[2];
  int32_t v26 = v0[0];
  int32_t v27 = v0[7];
  int32_t v28 = v0[11];
  int32_t v29 = v0[3];
  int32_t v30 = v0[13];
  int32_t v31 = v0[14];
  int32_t v32 = v0[12];
  int32_t v33 = v0[15];
  const std::vector<int32_t> v34 = {
      v18, v20, v30, v31, v19, v27, v20, v28, v20, v28, v31, v33, v21, v18, v32,
      v30, v22, v19, v18, v20, v23, v22, v21, v18, v24, v25, v22, v19, v25, v29,
      v19, v27, v26, v24, v23, v22, v18, v20, v30, v31, v19, v27, v20, v28, v20,
      v28, v31, v33, v21, v18, v32, v30, v22, v19, v18, v20, v23, v22, v21, v18,
      v24, v25, v22, v19, v25, v29, v19, v27, v26, v24, v23, v22, v18, v20, v30,
      v31, v19, v27, v20, v28, v20, v28, v31, v33, v21, v18, v32, v30, v22, v19,
      v18, v20, v23, v22, v21, v18, v24, v25, v22, v19, v25, v29, v19, v27, v26,
      v24, v23, v22, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1};
  std::vector<int32_t> v35(1 * 128);
  for (int64_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
    for (int64_t v35_i1 = 0; v35_i1 < 128; ++v35_i1) {
      v35[v35_i1 + 128 * (v35_i0)] =
          v34[0 + v35_i1 * 1 + 128 * (0 + v35_i0 * 1)];
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
std::vector<CiphertextT> conv_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                       std::vector<int32_t> v0,
                                                       PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 2;
  [[maybe_unused]] size_t v3 = 1;
  [[maybe_unused]] size_t v4 = 3;
  [[maybe_unused]] size_t v5 = 6;
  [[maybe_unused]] size_t v6 = 8;
  [[maybe_unused]] size_t v7 = 4;
  [[maybe_unused]] size_t v8 = 5;
  [[maybe_unused]] size_t v9 = 0;
  [[maybe_unused]] size_t v10 = 7;
  int32_t v11 = v0[7];
  int32_t v12 = v0[5];
  int32_t v13 = v0[8];
  int32_t v14 = v0[6];
  int32_t v15 = v0[4];
  int32_t v16 = v0[3];
  int32_t v17 = v0[1];
  int32_t v18 = v0[2];
  int32_t v19 = v0[0];
  const std::vector<int32_t> v20 = {
      v11, v11, v11, v11, v12, v12, v12, v12, v13, v13, v13, v13, v14, v14, v14,
      v14, v15, v15, v15, v15, v16, v16, v16, v16, v17, v17, v17, v17, v18, v18,
      v18, v18, v19, v19, v19, v19, v11, v11, v11, v11, v12, v12, v12, v12, v13,
      v13, v13, v13, v14, v14, v14, v14, v15, v15, v15, v15, v16, v16, v16, v16,
      v17, v17, v17, v17, v18, v18, v18, v18, v19, v19, v19, v19, v11, v11, v11,
      v11, v12, v12, v12, v12, v13, v13, v13, v13, v14, v14, v14, v14, v15, v15,
      v15, v15, v16, v16, v16, v16, v17, v17, v17, v17, v18, v18, v18, v18, v19,
      v19, v19, v19, v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1};
  std::vector<int32_t> v21(1 * 128);
  for (int64_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
    for (int64_t v21_i1 = 0; v21_i1 < 128; ++v21_i1) {
      v21[v21_i1 + 128 * (v21_i0)] =
          v20[0 + v21_i1 * 1 + 128 * (0 + v21_i0 * 1)];
    }
  }
  std::vector<int64_t> v22(std::begin(v21), std::end(v21));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v22;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v22[i % v22.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v23 = {ct};
  return v23;
}
std::vector<int32_t> conv_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 7;
  [[maybe_unused]] size_t v2 = 6;
  [[maybe_unused]] size_t v3 = 5;
  [[maybe_unused]] size_t v4 = 4;
  [[maybe_unused]] size_t v5 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(128);
  const auto& v6_cast = pt->GetPackedValue();
  std::vector<int32_t> v6(std::begin(v6_cast), std::end(v6_cast));
  int32_t v7 = v6[4 + 128 * (0)];
  int32_t v8 = v6[5 + 128 * (0)];
  int32_t v9 = v6[6 + 128 * (0)];
  int32_t v10 = v6[7 + 128 * (0)];
  const std::vector<int32_t> v11 = {v7, v8, v9, v10};
  return v11;
}
CryptoContextT conv_clone_0_0__generate_crypto_context() {
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
CryptoContextT conv_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                        PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {16, 4, 32, 48, 12});
  return cc;
}
