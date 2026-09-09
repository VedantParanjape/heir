
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
  std::vector<size_t> v5 = {6, 7};
  [[maybe_unused]] size_t v6 = 1;
  [[maybe_unused]] size_t v7 = 0;
  const auto& ct = v3[0];
  const auto& ct1 = v0[0];
  const auto& ct2 = cc->EvalMultNoRelin(ct, ct1);
  const auto& ct3 = v2[0];
  const auto& ct4 = v1[0];
  auto ct5 = cc->EvalMultNoRelin(ct3, ct4);
  cc->EvalSubInPlace(ct5, ct2);
  cc->RelinearizeInPlace(ct5);
  const auto& ct8 = v4[0];
  auto ct9 = cc->EvalMultNoRelin(ct8, ct5);
  cc->RelinearizeInPlace(ct9);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct9);
  std::vector<CiphertextT> v8(2);
#pragma omp parallel for
  for (auto v10 = 0; v10 < 2; ++v10) {
    size_t v12 = v5[v10];
    const auto& ct11 = cc->EvalFastRotation(
        ct9, v12, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v13 = {ct11};
    v8[v10] = v13[0];
  }
  auto ct12 = v8[0];
  const auto& ct13 = v8[1];
  cc->EvalSubInPlace(ct12, ct13);
  std::vector<CiphertextT> v14(1);
  cc->EvalAddInPlace(ct12, ct9);
  std::vector<CiphertextT> v15(v14);
  v15[0] = ct12;
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
  const std::vector<int32_t> v6 = {v1, v1, v1, v4, v5, v5, v1, v1, v1, v1, v1,
                                   v4, v5, v5, v1, v1, v1, v1, v1, v4, v5, v5,
                                   v1, v1, v1, v1, v1, v4, v5, v5, v1, v1};
  std::vector<int32_t> v7(1 * 32);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 32; ++v7_i1) {
      v7[v7_i1 + 32 * (v7_i0)] = v6[0 + v7_i1 * 1 + 32 * (0 + v7_i0 * 1)];
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
  [[maybe_unused]] size_t v2 = 7;
  [[maybe_unused]] size_t v3 = 8;
  int32_t v4 = v0[8];
  int32_t v5 = v0[7];
  const std::vector<int32_t> v6 = {v1, v1, v1, v4, v4, v5, v1, v1, v1, v1, v1,
                                   v4, v4, v5, v1, v1, v1, v1, v1, v4, v4, v5,
                                   v1, v1, v1, v1, v1, v4, v4, v5, v1, v1};
  std::vector<int32_t> v7(1 * 32);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 32; ++v7_i1) {
      v7[v7_i1 + 32 * (v7_i0)] = v6[0 + v7_i1 * 1 + 32 * (0 + v7_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 3;
  [[maybe_unused]] size_t v3 = 4;
  int32_t v4 = v0[4];
  int32_t v5 = v0[3];
  const std::vector<int32_t> v6 = {v1, v1, v1, v4, v5, v5, v1, v1, v1, v1, v1,
                                   v4, v5, v5, v1, v1, v1, v1, v1, v4, v5, v5,
                                   v1, v1, v1, v1, v1, v4, v5, v5, v1, v1};
  std::vector<int32_t> v7(1 * 32);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 32; ++v7_i1) {
      v7[v7_i1 + 32 * (v7_i0)] = v6[0 + v7_i1 * 1 + 32 * (0 + v7_i0 * 1)];
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
  [[maybe_unused]] size_t v2 = 4;
  [[maybe_unused]] size_t v3 = 5;
  int32_t v4 = v0[5];
  int32_t v5 = v0[4];
  const std::vector<int32_t> v6 = {v1, v1, v1, v4, v4, v5, v1, v1, v1, v1, v1,
                                   v4, v4, v5, v1, v1, v1, v1, v1, v4, v4, v5,
                                   v1, v1, v1, v1, v1, v4, v4, v5, v1, v1};
  std::vector<int32_t> v7(1 * 32);
  for (int64_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
    for (int64_t v7_i1 = 0; v7_i1 < 32; ++v7_i1) {
      v7[v7_i1 + 32 * (v7_i0)] = v6[0 + v7_i1 * 1 + 32 * (0 + v7_i0 * 1)];
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
  [[maybe_unused]] size_t v2 = 2;
  [[maybe_unused]] size_t v3 = 1;
  [[maybe_unused]] size_t v4 = 0;
  int32_t v5 = v0[0];
  int32_t v6 = v0[1];
  int32_t v7 = v0[2];
  const std::vector<int32_t> v8 = {v1, v1, v1, v5, v6, v7, v1, v1, v1, v1, v1,
                                   v5, v6, v7, v1, v1, v1, v1, v1, v5, v6, v7,
                                   v1, v1, v1, v1, v1, v5, v6, v7, v1, v1};
  std::vector<int32_t> v9(1 * 32);
  for (int64_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
    for (int64_t v9_i1 = 0; v9_i1 < 32; ++v9_i1) {
      v9[v9_i1 + 32 * (v9_i0)] = v8[0 + v9_i1 * 1 + 32 * (0 + v9_i0 * 1)];
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
std::vector<int32_t> det_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 5;
  [[maybe_unused]] size_t v2 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(32);
  const auto& v3_cast = pt->GetPackedValue();
  std::vector<int32_t> v3(std::begin(v3_cast), std::end(v3_cast));
  int32_t v4 = v3[5 + 32 * (0)];
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
  cc->EvalRotateKeyGen(sk, {7, 6});
  return cc;
}
