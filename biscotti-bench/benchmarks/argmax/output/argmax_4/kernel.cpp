
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

std::vector<CiphertextT> argmax_clone_0_0(CryptoContextT cc,
                                          std::vector<CiphertextT> v0,
                                          std::vector<CiphertextT> v1,
                                          std::vector<CiphertextT> v2,
                                          std::vector<CiphertextT> v3) {
  [[maybe_unused]] size_t v4 = 0;
  [[maybe_unused]] size_t v5 = 1;
  const auto& ct = v3[0];
  const auto& ct1 = v2[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& ct4 = v1[0];
  const auto& ct5 = v0[0];
  auto ct6 = cc->EvalMultNoRelin(ct4, ct5);
  cc->RelinearizeInPlace(ct6);
  const auto& ct8 = cc->EvalRotate(ct2, 1);
  auto ct9 = cc->EvalMultNoRelin(ct6, ct8);
  std::vector<CiphertextT> v6(1);
  cc->RelinearizeInPlace(ct9);
  std::vector<CiphertextT> v7(v6);
  v7[0] = ct9;
  return v7;
}
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg0(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 7;
  [[maybe_unused]] size_t v3 = 6;
  [[maybe_unused]] size_t v4 = 5;
  [[maybe_unused]] size_t v5 = 4;
  int32_t v6 = v0[4];
  int32_t v7 = v0[5];
  int32_t v8 = v0[6];
  int32_t v9 = v0[7];
  const std::vector<int32_t> v10 = {
      v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1, v1,
      v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1, v1,
      v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1, v1,
      v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 64);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 64; ++v11_i1) {
      v11[v11_i1 + 64 * (v11_i0)] = v10[0 + v11_i1 * 1 + 64 * (0 + v11_i0 * 1)];
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg1(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 3;
  [[maybe_unused]] size_t v3 = 2;
  [[maybe_unused]] size_t v4 = 1;
  [[maybe_unused]] size_t v5 = 0;
  int32_t v6 = v0[0];
  int32_t v7 = v0[1];
  int32_t v8 = v0[2];
  int32_t v9 = v0[3];
  const std::vector<int32_t> v10 = {
      v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1, v1,
      v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1, v1,
      v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1, v1,
      v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 64);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 64; ++v11_i1) {
      v11[v11_i1 + 64 * (v11_i0)] = v10[0 + v11_i1 * 1 + 64 * (0 + v11_i0 * 1)];
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg2(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 15;
  [[maybe_unused]] size_t v3 = 14;
  [[maybe_unused]] size_t v4 = 13;
  [[maybe_unused]] size_t v5 = 12;
  int32_t v6 = v0[12];
  int32_t v7 = v0[13];
  int32_t v8 = v0[14];
  int32_t v9 = v0[15];
  const std::vector<int32_t> v10 = {
      v1, v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1,
      v1, v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1,
      v1, v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1,
      v1, v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1};
  std::vector<int32_t> v11(1 * 64);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 64; ++v11_i1) {
      v11[v11_i1 + 64 * (v11_i0)] = v10[0 + v11_i1 * 1 + 64 * (0 + v11_i0 * 1)];
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
std::vector<CiphertextT> argmax_clone_0_0__encrypt__arg3(
    CryptoContextT cc, std::vector<int32_t> v0, PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 11;
  [[maybe_unused]] size_t v3 = 10;
  [[maybe_unused]] size_t v4 = 9;
  [[maybe_unused]] size_t v5 = 8;
  int32_t v6 = v0[8];
  int32_t v7 = v0[9];
  int32_t v8 = v0[10];
  int32_t v9 = v0[11];
  const std::vector<int32_t> v10 = {
      v1, v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1,
      v1, v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1,
      v1, v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1,
      v1, v6, v1, v1, v1, v7, v1, v1, v8, v1, v1, v1, v9, v1, v1, v1};
  std::vector<int32_t> v11(1 * 64);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 64; ++v11_i1) {
      v11[v11_i1 + 64 * (v11_i0)] = v10[0 + v11_i1 * 1 + 64 * (0 + v11_i0 * 1)];
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
std::vector<int32_t> argmax_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 11;
  [[maybe_unused]] size_t v2 = 7;
  [[maybe_unused]] size_t v3 = 4;
  [[maybe_unused]] size_t v4 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(64);
  const auto& v5_cast = pt->GetPackedValue();
  std::vector<int32_t> v5(std::begin(v5_cast), std::end(v5_cast));
  int32_t v6 = v5[0 + 64 * (0)];
  int32_t v7 = v5[4 + 64 * (0)];
  int32_t v8 = v5[7 + 64 * (0)];
  int32_t v9 = v5[11 + 64 * (0)];
  const std::vector<int32_t> v10 = {v6, v7, v8, v9};
  return v10;
}
CryptoContextT argmax_clone_0_0__generate_crypto_context() {
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
CryptoContextT argmax_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                          PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {1});
  return cc;
}
