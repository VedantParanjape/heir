
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

std::vector<CiphertextT> det_clone_0_0(CryptoContextT cc,
                                       std::vector<CiphertextT> v0,
                                       std::vector<CiphertextT> v1,
                                       std::vector<CiphertextT> v2,
                                       std::vector<CiphertextT> v3,
                                       std::vector<CiphertextT> v4) {
  std::vector<size_t> v5 = {4, 11};
  [[maybe_unused]] size_t v6 = 0;
  [[maybe_unused]] size_t v7 = 1;
  const auto& ct = v4[0];
  const auto& ct1 = v3[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& ct4 = v2[0];
  const auto& ct5 = v0[0];
  auto ct6 = cc->EvalMultNoRelin(ct4, ct5);
  cc->RelinearizeInPlace(ct6);
  auto ct8 = cc->EvalRotate(ct2, 1);
  cc->EvalSubInPlace(ct8, ct6);
  const auto& ct10 = v1[0];
  auto ct11 = cc->EvalMultNoRelin(ct10, ct8);
  cc->RelinearizeInPlace(ct11);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct11);
  std::vector<CiphertextT> v8(2);
#pragma omp parallel for
  for (auto v10 = 0; v10 < 2; ++v10) {
    size_t v12 = v5[v10];
    const auto& ct13 = cc->EvalFastRotation(
        ct11, v12, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v13 = {ct13};
    v8[v10] = v13[0];
  }
  const auto& ct14 = v8[0];
  const auto& ct15 = v8[1];
  std::vector<CiphertextT> v14(1);
  cc->EvalSubInPlace(ct11, ct14);
  cc->EvalAddInPlace(ct11, ct15);
  std::vector<CiphertextT> v15(v14);
  v15[0] = ct11;
  return v15;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 6;
  [[maybe_unused]] size_t v3 = 7;
  int32_t v4 = v0[7];
  int32_t v5 = v0[6];
  const std::vector<int32_t> v6 = {
      v5, v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v5,
      v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v5, v1,
      v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v5, v1, v1,
      v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v7(1 * 64);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 64; ++v7_i1) {
      v7[v7_i1 + 64 * (v7_i0)] = v6[0 + v7_i1 * 1 + 64 * (0 + v7_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 2;
  [[maybe_unused]] size_t v3 = 1;
  [[maybe_unused]] size_t v4 = 0;
  int32_t v5 = v0[0];
  int32_t v6 = v0[1];
  int32_t v7 = v0[2];
  const std::vector<int32_t> v8 = {
      v6, v1, v1, v1, v1, v1, v1, v7, v1, v1, v1, v5, v1, v1, v1, v6,
      v1, v1, v1, v1, v1, v1, v7, v1, v1, v1, v5, v1, v1, v1, v6, v1,
      v1, v1, v1, v1, v1, v7, v1, v1, v1, v5, v1, v1, v1, v6, v1, v1,
      v1, v1, v1, v1, v7, v1, v1, v1, v5, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v9(1 * 64);
  for (int64_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
    for (int64_t v9_i1 = 0; v9_i1 < 64; ++v9_i1) {
      v9[v9_i1 + 64 * (v9_i0)] = v8[0 + v9_i1 * 1 + 64 * (0 + v9_i0 * 1)];
    }
  }
  std::vector<int64_t> v10(std::begin(v9), std::end(v9));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v10;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v10[i % v10.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v11 = {ct};
  return v11;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 4;
  [[maybe_unused]] size_t v3 = 5;
  int32_t v4 = v0[5];
  int32_t v5 = v0[4];
  const std::vector<int32_t> v6 = {
      v4, v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v4,
      v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v4, v1,
      v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v4, v1, v1,
      v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v7(1 * 64);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 64; ++v7_i1) {
      v7[v7_i1 + 64 * (v7_i0)] = v6[0 + v7_i1 * 1 + 64 * (0 + v7_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 7;
  [[maybe_unused]] size_t v3 = 8;
  int32_t v4 = v0[8];
  int32_t v5 = v0[7];
  const std::vector<int32_t> v6 = {
      v1, v4, v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1,
      v4, v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v4,
      v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v4, v1,
      v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v7(1 * 64);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 64; ++v7_i1) {
      v7[v7_i1 + 64 * (v7_i0)] = v6[0 + v7_i1 * 1 + 64 * (0 + v7_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg4(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 3;
  [[maybe_unused]] size_t v3 = 4;
  int32_t v4 = v0[4];
  int32_t v5 = v0[3];
  const std::vector<int32_t> v6 = {
      v1, v5, v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1,
      v5, v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v5,
      v1, v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v5, v1,
      v1, v1, v1, v1, v1, v5, v1, v1, v1, v4, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v7(1 * 64);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 64; ++v7_i1) {
      v7[v7_i1 + 64 * (v7_i0)] = v6[0 + v7_i1 * 1 + 64 * (0 + v7_i0 * 1)];
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
std::vector<int32_t> det_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 11;
  [[maybe_unused]] size_t v2 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(64);
  const auto& v3_cast = pt->GetPackedValue();
  std::vector<int32_t> v3(std::begin(v3_cast), std::end(v3_cast));
  int32_t v4 = v3[11 + 64 * (0)];
  const std::vector<int32_t> v5 = {v4};
  return v5;
}
CryptoContextT det_clone_0_0__generate_crypto_context() {
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
CryptoContextT det_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                       PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {4, 11, 1});
  return cc;
}
