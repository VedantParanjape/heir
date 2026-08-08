
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

std::vector<CiphertextT> dot_clone_0_0(CryptoContextT cc,
                                       std::vector<CiphertextT> v0,
                                       std::vector<CiphertextT> v1) {
  std::vector<size_t> v2 = {2, 4};
  [[maybe_unused]] size_t v3 = 1;
  [[maybe_unused]] size_t v4 = 0;
  const auto& ct = v0[0];
  const auto& ct1 = v1[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  std::vector<CiphertextT> v5(2);
#pragma omp parallel for
  for (auto v7 = 0; v7 < 2; ++v7) {
    size_t v9 = v2[v7];
    const auto& ct4 =
        cc->EvalFastRotation(ct2, v9, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v10 = {ct4};
    v5[v7] = v10[0];
  }
  auto ct5 = v5[0];
  const auto& ct6 = v5[1];
  cc->EvalAddInPlace(ct5, ct6);
  std::vector<CiphertextT> v11(1);
  cc->EvalAddInPlace(ct2, ct5);
  std::vector<CiphertextT> v12(v11);
  v12[0] = ct2;
  return v12;
}
std::vector<CiphertextT> dot_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 2;
  [[maybe_unused]] size_t v3 = 1;
  [[maybe_unused]] size_t v4 = 0;
  int32_t v5 = v0[0];
  int32_t v6 = v0[1];
  int32_t v7 = v0[2];
  const std::vector<int32_t> v8 = {v5, v7, v6, v5, v7, v6, v5, v7,
                                   v6, v5, v7, v6, v5, v7, v6, v1};
  std::vector<int32_t> v9(1 * 16);
  for (int64_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
    for (int64_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
      v9[v9_i1 + 16 * (v9_i0)] = v8[0 + v9_i1 * 1 + 16 * (0 + v9_i0 * 1)];
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
std::vector<CiphertextT> dot_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 2;
  [[maybe_unused]] size_t v3 = 1;
  [[maybe_unused]] size_t v4 = 0;
  int32_t v5 = v0[0];
  int32_t v6 = v0[1];
  int32_t v7 = v0[2];
  const std::vector<int32_t> v8 = {v5, v7, v6, v5, v7, v6, v5, v7,
                                   v6, v5, v7, v6, v5, v7, v6, v1};
  std::vector<int32_t> v9(1 * 16);
  for (int64_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
    for (int64_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
      v9[v9_i1 + 16 * (v9_i0)] = v8[0 + v9_i1 * 1 + 16 * (0 + v9_i0 * 1)];
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
std::vector<int32_t> dot_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(16);
  const auto& v2_cast = pt->GetPackedValue();
  std::vector<int32_t> v2(std::begin(v2_cast), std::end(v2_cast));
  int32_t v3 = v2[0 + 16 * (0)];
  const std::vector<int32_t> v4 = {v3};
  return v4;
}
CryptoContextT dot_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(1);
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
  cc->EvalRotateKeyGen(sk, {2, 4});
  return cc;
}
