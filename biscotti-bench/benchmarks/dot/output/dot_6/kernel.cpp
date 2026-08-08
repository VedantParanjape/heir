
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
  std::vector<int64_t> v2 = {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  std::vector<int64_t> v3 = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
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
std::vector<CiphertextT> dot_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    const std::vector<Plaintext>& v2) {
  std::vector<size_t> v3 = {2, 1};
  [[maybe_unused]] size_t v4 = 0;
  [[maybe_unused]] size_t v5 = 1;
  [[maybe_unused]] size_t v6 = 3;
  const auto& ct = v0[0];
  const auto& ct1 = v1[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  std::vector<CiphertextT> v7(2);
  Plaintext pt = v2[0];
  Plaintext pt1 = v2[1];
  std::vector<CiphertextT> v8(1);
#pragma omp parallel for
  for (auto v10 = 0; v10 < 2; ++v10) {
    size_t v12 = v3[v10];
    const auto& ct4 = cc->EvalFastRotation(ct2, v12, 2 * cc->GetRingDimension(),
                                           digit_decomp);
    const std::vector<CiphertextT> v13 = {ct4};
    v7[v10] = v13[0];
  }
  const auto& ct5 = v7[0];
  const auto& ct6 = v7[1];
  auto ct7 = cc->EvalMult(ct6, pt);
  const auto& ct8 = cc->EvalMult(ct5, pt1);
  cc->EvalAddInPlace(ct7, ct8);
  cc->EvalAddInPlace(ct7, ct2);
  auto ct11 = cc->EvalMult(ct6, pt1);
  const auto& ct12 = cc->EvalMult(ct5, pt);
  cc->EvalAddInPlace(ct11, ct12);
  cc->EvalAddInPlace(ct11, ct7);
  const auto& ct15 = cc->EvalRotate(ct11, 3);
  cc->EvalAddInPlace(ct11, ct15);
  std::vector<CiphertextT> v14(v8);
  v14[0] = ct11;
  return v14;
}
std::vector<CiphertextT> dot_clone_0_0(CryptoContextT cc,
                                       std::vector<CiphertextT> v0,
                                       std::vector<CiphertextT> v1) {
  const auto& v2 = dot_clone_0_0__preprocessing(cc);
  const auto& v3 = dot_clone_0_0__preprocessed(cc, v0, v1, v2);
  return v3;
}
std::vector<CiphertextT> dot_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 3;
  [[maybe_unused]] size_t v3 = 2;
  [[maybe_unused]] size_t v4 = 4;
  [[maybe_unused]] size_t v5 = 1;
  [[maybe_unused]] size_t v6 = 5;
  [[maybe_unused]] size_t v7 = 0;
  int32_t v8 = v0[0];
  int32_t v9 = v0[1];
  int32_t v10 = v0[2];
  int32_t v11 = v0[3];
  int32_t v12 = v0[4];
  int32_t v13 = v0[5];
  const std::vector<int32_t> v14 = {v13, v11, v12, v10, v9,  v8,  v13, v11,
                                    v12, v10, v9,  v8,  v13, v11, v12, v10,
                                    v9,  v8,  v13, v11, v12, v10, v9,  v8,
                                    v13, v11, v12, v10, v9,  v8,  v1,  v1};
  std::vector<int32_t> v15(1 * 32);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 32; ++v15_i1) {
      v15[v15_i1 + 32 * (v15_i0)] = v14[0 + v15_i1 * 1 + 32 * (0 + v15_i0 * 1)];
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
std::vector<CiphertextT> dot_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 3;
  [[maybe_unused]] size_t v3 = 2;
  [[maybe_unused]] size_t v4 = 4;
  [[maybe_unused]] size_t v5 = 1;
  [[maybe_unused]] size_t v6 = 5;
  [[maybe_unused]] size_t v7 = 0;
  int32_t v8 = v0[0];
  int32_t v9 = v0[1];
  int32_t v10 = v0[2];
  int32_t v11 = v0[3];
  int32_t v12 = v0[4];
  int32_t v13 = v0[5];
  const std::vector<int32_t> v14 = {v13, v11, v12, v10, v9,  v8,  v13, v11,
                                    v12, v10, v9,  v8,  v13, v11, v12, v10,
                                    v9,  v8,  v13, v11, v12, v10, v9,  v8,
                                    v13, v11, v12, v10, v9,  v8,  v1,  v1};
  std::vector<int32_t> v15(1 * 32);
  for (int64_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
    for (int64_t v15_i1 = 0; v15_i1 < 32; ++v15_i1) {
      v15[v15_i1 + 32 * (v15_i0)] = v14[0 + v15_i1 * 1 + 32 * (0 + v15_i0 * 1)];
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
std::vector<int32_t> dot_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 3;
  [[maybe_unused]] size_t v2 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(32);
  const auto& v3_cast = pt->GetPackedValue();
  std::vector<int32_t> v3(std::begin(v3_cast), std::end(v3_cast));
  int32_t v4 = v3[3 + 32 * (0)];
  const std::vector<int32_t> v5 = {v4};
  return v5;
}
CryptoContextT dot_clone_0_0__generate_crypto_context() {
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
CryptoContextT dot_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                       PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {2, 1, 3});
  return cc;
}
