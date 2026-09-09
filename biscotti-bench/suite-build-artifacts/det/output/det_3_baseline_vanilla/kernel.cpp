
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

std::vector<Plaintext> det_clone_0_0__preprocessing(CryptoContextT cc) {
  [[maybe_unused]] size_t v0 = 0;
  [[maybe_unused]] size_t v1 = 1;
  std::vector<int64_t> v2 = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v3 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
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
std::vector<CiphertextT> det_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5,
    const std::vector<Plaintext>& v6) {
  [[maybe_unused]] size_t v7 = 0;
  [[maybe_unused]] size_t v8 = 15;
  [[maybe_unused]] size_t v9 = 13;
  [[maybe_unused]] size_t v10 = 17;
  [[maybe_unused]] size_t v11 = 1;
  const auto& ct = v5[0];
  const auto& ct1 = v4[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& ct4 = cc->EvalRotate(ct2, 13);
  const auto& ct5 = cc->EvalSub(ct4, ct2);
  Plaintext pt = v6[0];
  const auto& ct6 = v0[0];
  auto ct7 = cc->EvalMult(ct6, pt);
  Plaintext pt1 = v6[1];
  const auto& ct8 = cc->EvalMult(ct5, pt1);
  cc->EvalAddInPlace(ct7, ct8);
  const auto& ct10 = v3[0];
  auto ct11 = cc->EvalMultNoRelin(ct10, ct7);
  cc->RelinearizeInPlace(ct11);
  const auto& ct13 = cc->EvalSub(ct11, ct4);
  const auto& ct14 = v2[0];
  auto ct15 = cc->EvalMultNoRelin(ct14, ct13);
  cc->RelinearizeInPlace(ct15);
  const auto& ct17 = v1[0];
  auto ct18 = cc->EvalMultNoRelin(ct17, ct5);
  cc->RelinearizeInPlace(ct18);
  auto ct20 = cc->EvalRotate(ct11, 17);
  const auto& ct21 = cc->EvalRotate(ct15, 15);
  cc->EvalSubInPlace(ct20, ct21);
  std::vector<CiphertextT> v12(1);
  cc->EvalAddInPlace(ct20, ct18);
  std::vector<CiphertextT> v13(v12);
  v13[0] = ct20;
  return v13;
}
std::vector<CiphertextT> det_clone_0_0(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5) {
  const auto& v6 = det_clone_0_0__preprocessing(cc);
  const auto& v7 = det_clone_0_0__preprocessed(cc, v0, v1, v2, v3, v4, v5, v6);
  return v7;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 8;
  int32_t v3 = v0[8];
  const std::vector<int32_t> v4 = {
      v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v5(1 * 64);
  for (int64_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
    for (int64_t v5_i1 = 0; v5_i1 < 64; ++v5_i1) {
      v5[v5_i1 + 64 * (v5_i0)] = v4[0 + v5_i1 * 1 + 64 * (0 + v5_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 2;
  int32_t v3 = v0[2];
  const std::vector<int32_t> v4 = {
      v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v5(1 * 64);
  for (int64_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
    for (int64_t v5_i1 = 0; v5_i1 < 64; ++v5_i1) {
      v5[v5_i1 + 64 * (v5_i0)] = v4[0 + v5_i1 * 1 + 64 * (0 + v5_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 1;
  int32_t v3 = v0[1];
  const std::vector<int32_t> v4 = {
      v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v3, v1, v1, v1, v1, v1, v1, v1, v1, v1};
  std::vector<int32_t> v5(1 * 64);
  for (int64_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
    for (int64_t v5_i1 = 0; v5_i1 < 64; ++v5_i1) {
      v5[v5_i1 + 64 * (v5_i0)] = v4[0 + v5_i1 * 1 + 64 * (0 + v5_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 3;
  [[maybe_unused]] size_t v3 = 0;
  int32_t v4 = v0[0];
  int32_t v5 = v0[3];
  const std::vector<int32_t> v6 = {
      v1, v1, v1, v1, v1, v1, v5, v1, v4, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v5, v1, v4, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v5, v1, v4, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v5, v1, v4, v1, v1, v1, v1, v1, v1, v1};
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
  [[maybe_unused]] size_t v2 = 6;
  [[maybe_unused]] size_t v3 = 8;
  [[maybe_unused]] size_t v4 = 7;
  int32_t v5 = v0[7];
  int32_t v6 = v0[8];
  int32_t v7 = v0[6];
  const std::vector<int32_t> v8 = {
      v1, v1, v1, v7, v5, v6, v1, v7, v5, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v7, v5, v6, v1, v7, v5, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v7, v5, v6, v1, v7, v5, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v7, v5, v6, v1, v7, v5, v1, v1, v1, v1, v1, v1, v1};
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg5(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 3;
  [[maybe_unused]] size_t v3 = 5;
  [[maybe_unused]] size_t v4 = 4;
  int32_t v5 = v0[4];
  int32_t v6 = v0[5];
  int32_t v7 = v0[3];
  const std::vector<int32_t> v8 = {
      v1, v1, v1, v6, v7, v5, v1, v5, v6, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v6, v7, v5, v1, v5, v6, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v6, v7, v5, v1, v5, v6, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v6, v7, v5, v1, v5, v6, v1, v1, v1, v1, v1, v1, v1};
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
std::vector<int32_t> det_clone_0_0__decrypt__result0(
    CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 7;
  [[maybe_unused]] size_t v2 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(64);
  const auto& v3_cast = pt->GetPackedValue();
  std::vector<int32_t> v3(std::begin(v3_cast), std::end(v3_cast));
  int32_t v4 = v3[7 + 64 * (0)];
  const std::vector<int32_t> v5 = {v4};
  return v5;
}
CryptoContextT det_clone_0_0__generate_crypto_context() {
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
CryptoContextT det_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                       PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {13, 15, 17});
  return cc;
}
