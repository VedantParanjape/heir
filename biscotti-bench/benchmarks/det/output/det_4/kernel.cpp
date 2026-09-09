
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

std::vector<CiphertextT> det_clone_0_0(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    std::vector<CiphertextT> v4, std::vector<CiphertextT> v5) {
  std::vector<size_t> v6 = {21, 22, 52, 51};
  std::vector<size_t> v7 = {28, 52, 24};
  [[maybe_unused]] size_t v8 = 3;
  [[maybe_unused]] size_t v9 = 2;
  [[maybe_unused]] size_t v10 = 1;
  [[maybe_unused]] size_t v11 = 0;
  const auto& ct = v5[0];
  const auto& ct1 = v2[0];
  const auto& ct2 = cc->EvalMultNoRelin(ct, ct1);
  const auto& ct3 = v4[0];
  const auto& ct4 = v3[0];
  auto ct5 = cc->EvalMultNoRelin(ct3, ct4);
  cc->EvalSubInPlace(ct5, ct2);
  cc->RelinearizeInPlace(ct5);
  const auto& ct8 = v1[0];
  auto ct9 = cc->EvalMultNoRelin(ct8, ct5);
  cc->RelinearizeInPlace(ct9);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct9);
  std::vector<CiphertextT> v12(3);
#pragma omp parallel for
  for (auto v14 = 0; v14 < 3; ++v14) {
    size_t v16 = v7[v14];
    const auto& ct11 = cc->EvalFastRotation(
        ct9, v16, 2 * cc->GetRingDimension(), digit_decomp);
    const std::vector<CiphertextT> v17 = {ct11};
    v12[v14] = v17[0];
  }
  const auto& ct12 = v12[0];
  auto ct13 = v12[1];
  const auto& ct14 = v12[2];
  cc->EvalSubInPlace(ct13, ct14);
  cc->EvalAddInPlace(ct13, ct12);
  cc->EvalSubInPlace(ct13, ct9);
  const auto& ct18 = v0[0];
  auto ct19 = cc->EvalMultNoRelin(ct18, ct13);
  cc->RelinearizeInPlace(ct19);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct19);
  std::vector<CiphertextT> v18(4);
  std::vector<CiphertextT> v19(1);
#pragma omp parallel for
  for (auto v21 = 0; v21 < 4; ++v21) {
    size_t v23 = v6[v21];
    const auto& ct21 = cc->EvalFastRotation(
        ct19, v23, 2 * cc->GetRingDimension(), digit_decomp1);
    const std::vector<CiphertextT> v24 = {ct21};
    v18[v21] = v24[0];
  }
  const auto& ct22 = v18[0];
  const auto& ct23 = v18[1];
  const auto& ct24 = v18[2];
  auto ct25 = v18[3];
  cc->EvalSubInPlace(ct25, ct24);
  cc->EvalAddInPlace(ct25, ct22);
  cc->EvalSubInPlace(ct25, ct23);
  std::vector<CiphertextT> v25(v19);
  v25[0] = ct25;
  return v25;
}
std::vector<CiphertextT> det_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
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
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6, v7, v8, v9,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6, v7, v8, v9,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6, v7, v8, v9,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v6, v7, v8, v9};
  std::vector<int32_t> v11(1 * 128);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 128; ++v11_i1) {
      v11[v11_i1 + 128 * (v11_i0)] =
          v10[0 + v11_i1 * 1 + 128 * (0 + v11_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 4;
  [[maybe_unused]] size_t v3 = 7;
  [[maybe_unused]] size_t v4 = 6;
  [[maybe_unused]] size_t v5 = 5;
  int32_t v6 = v0[5];
  int32_t v7 = v0[6];
  int32_t v8 = v0[7];
  int32_t v9 = v0[4];
  const std::vector<int32_t> v10 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v6, v9, v9, v9, v7, v7, v6, v6, v8, v8, v8, v7, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v6, v9, v9, v9, v7, v7, v6, v6, v8, v8, v8, v7, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v6, v9, v9, v9, v7, v7, v6, v6, v8, v8, v8, v7, v1, v1, v1, v1,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v6, v9, v9, v9, v7, v7, v6, v6, v8, v8, v8, v7, v1, v1, v1, v1};
  std::vector<int32_t> v11(1 * 128);
  for (int64_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
    for (int64_t v11_i1 = 0; v11_i1 < 128; ++v11_i1) {
      v11[v11_i1 + 128 * (v11_i0)] =
          v10[0 + v11_i1 * 1 + 128 * (0 + v11_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 12;
  [[maybe_unused]] size_t v3 = 14;
  [[maybe_unused]] size_t v4 = 13;
  int32_t v5 = v0[13];
  int32_t v6 = v0[14];
  int32_t v7 = v0[12];
  const std::vector<int32_t> v8 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v6, v6, v5, v5, v5, v7, v7, v7, v5, v7, v7, v7, v5, v7, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v6, v6, v5, v5, v5, v7, v7, v7, v5, v7, v7, v7, v5, v7, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v6, v6, v5, v5, v5, v7, v7, v7, v5, v7, v7, v7, v5, v7, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v6, v6, v5, v5, v5, v7, v7, v7, v5, v7, v7, v7, v5, v7, v7, v7};
  std::vector<int32_t> v9(1 * 128);
  for (int64_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
    for (int64_t v9_i1 = 0; v9_i1 < 128; ++v9_i1) {
      v9[v9_i1 + 128 * (v9_i0)] = v8[0 + v9_i1 * 1 + 128 * (0 + v9_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 13;
  [[maybe_unused]] size_t v3 = 14;
  [[maybe_unused]] size_t v4 = 15;
  int32_t v5 = v0[15];
  int32_t v6 = v0[14];
  int32_t v7 = v0[13];
  const std::vector<int32_t> v8 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v5, v6, v5, v5, v5, v6, v6, v6, v7, v7, v6, v6, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v5, v6, v5, v5, v5, v6, v6, v6, v7, v7, v6, v6, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v5, v6, v5, v5, v5, v6, v6, v6, v7, v7, v6, v6, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v5, v6, v5, v5, v5, v6, v6, v6, v7, v7, v6, v6, v7, v7};
  std::vector<int32_t> v9(1 * 128);
  for (int64_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
    for (int64_t v9_i1 = 0; v9_i1 < 128; ++v9_i1) {
      v9[v9_i1 + 128 * (v9_i0)] = v8[0 + v9_i1 * 1 + 128 * (0 + v9_i0 * 1)];
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
std::vector<CiphertextT> det_clone_0_0__encrypt__arg4(CryptoContextT cc,
                                                      std::vector<int32_t> v0,
                                                      PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 8;
  [[maybe_unused]] size_t v3 = 9;
  [[maybe_unused]] size_t v4 = 10;
  int32_t v5 = v0[10];
  int32_t v6 = v0[9];
  int32_t v7 = v0[8];
  const std::vector<int32_t> v8 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v6, v6, v6, v7, v7, v7, v6, v7, v7, v7, v6, v7, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v6, v6, v6, v7, v7, v7, v6, v7, v7, v7, v6, v7, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v6, v6, v6, v7, v7, v7, v6, v7, v7, v7, v6, v7, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v6, v6, v6, v7, v7, v7, v6, v7, v7, v7, v6, v7, v7, v7};
  std::vector<int32_t> v9(1 * 128);
  for (int64_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
    for (int64_t v9_i1 = 0; v9_i1 < 128; ++v9_i1) {
      v9[v9_i1 + 128 * (v9_i0)] = v8[0 + v9_i1 * 1 + 128 * (0 + v9_i0 * 1)];
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
  [[maybe_unused]] size_t v2 = 9;
  [[maybe_unused]] size_t v3 = 10;
  [[maybe_unused]] size_t v4 = 11;
  int32_t v5 = v0[11];
  int32_t v6 = v0[10];
  int32_t v7 = v0[9];
  const std::vector<int32_t> v8 = {
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v5, v6, v5, v5, v5, v6, v6, v6, v7, v7, v6, v6, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v5, v6, v5, v5, v5, v6, v6, v6, v7, v7, v6, v6, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v5, v6, v5, v5, v5, v6, v6, v6, v7, v7, v6, v6, v7, v7,
      v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1,
      v5, v5, v5, v6, v5, v5, v5, v6, v6, v6, v7, v7, v6, v6, v7, v7};
  std::vector<int32_t> v9(1 * 128);
  for (int64_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
    for (int64_t v9_i1 = 0; v9_i1 < 128; ++v9_i1) {
      v9[v9_i1 + 128 * (v9_i0)] = v8[0 + v9_i1 * 1 + 128 * (0 + v9_i0 * 1)];
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
  [[maybe_unused]] size_t v1 = 9;
  [[maybe_unused]] size_t v2 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(128);
  const auto& v3_cast = pt->GetPackedValue();
  std::vector<int32_t> v3(std::begin(v3_cast), std::end(v3_cast));
  int32_t v4 = v3[9 + 128 * (0)];
  const std::vector<int32_t> v5 = {v4};
  return v5;
}
CryptoContextT det_clone_0_0__generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(3);
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
  cc->EvalRotateKeyGen(sk, {52, 21, 28, 51, 22, 24});
  return cc;
}
