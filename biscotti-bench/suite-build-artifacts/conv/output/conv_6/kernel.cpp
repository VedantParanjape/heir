
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
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> v3 = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
  std::vector<size_t> v3 = {64, 304, 240};
  std::vector<size_t> v4 = {16, 48};
  [[maybe_unused]] size_t v5 = 2;
  [[maybe_unused]] size_t v6 = 0;
  [[maybe_unused]] size_t v7 = 1;
  const auto& ct = v0[0];
  const auto& ct1 = v1[0];
  auto ct2 = cc->EvalMultNoRelin(ct, ct1);
  cc->RelinearizeInPlace(ct2);
  const auto& digit_decomp = cc->EvalFastRotationPrecompute(ct2);
  std::vector<CiphertextT> v8(2);
  Plaintext pt = v2[0];
  auto ct4 = cc->EvalMult(ct2, pt);
  Plaintext pt1 = v2[1];
  std::vector<CiphertextT> v9(3);
  std::vector<CiphertextT> v10(1);
#pragma omp parallel for
  for (auto v12 = 0; v12 < 2; ++v12) {
    size_t v14 = v4[v12];
    const auto& ct5 = cc->EvalFastRotation(ct2, v14, 2 * cc->GetRingDimension(),
                                           digit_decomp);
    const std::vector<CiphertextT> v15 = {ct5};
    v8[v12] = v15[0];
  }
  const auto& ct6 = v8[0];
  auto ct7 = v8[1];
  const auto& ct8 = cc->EvalMult(ct6, pt1);
  cc->EvalAddInPlace(ct4, ct8);
  cc->EvalAddInPlace(ct7, ct4);
  const auto& digit_decomp1 = cc->EvalFastRotationPrecompute(ct7);
#pragma omp parallel for
  for (auto v17 = 0; v17 < 3; ++v17) {
    size_t v19 = v3[v17];
    const auto& ct11 = cc->EvalFastRotation(
        ct7, v19, 2 * cc->GetRingDimension(), digit_decomp1);
    const std::vector<CiphertextT> v20 = {ct11};
    v9[v17] = v20[0];
  }
  auto ct12 = v9[0];
  const auto& ct13 = v9[1];
  const auto& ct14 = v9[2];
  cc->EvalAddInPlace(ct7, ct14);
  cc->EvalAddInPlace(ct12, ct13);
  cc->EvalAddInPlace(ct12, ct7);
  cc->EvalAddInPlace(ct12, ct6);
  std::vector<CiphertextT> v21(v10);
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
  [[maybe_unused]] size_t v2 = 35;
  [[maybe_unused]] size_t v3 = 34;
  [[maybe_unused]] size_t v4 = 30;
  [[maybe_unused]] size_t v5 = 27;
  [[maybe_unused]] size_t v6 = 31;
  [[maybe_unused]] size_t v7 = 26;
  [[maybe_unused]] size_t v8 = 23;
  [[maybe_unused]] size_t v9 = 33;
  [[maybe_unused]] size_t v10 = 29;
  [[maybe_unused]] size_t v11 = 25;
  [[maybe_unused]] size_t v12 = 11;
  [[maybe_unused]] size_t v13 = 17;
  [[maybe_unused]] size_t v14 = 10;
  [[maybe_unused]] size_t v15 = 22;
  [[maybe_unused]] size_t v16 = 5;
  [[maybe_unused]] size_t v17 = 3;
  [[maybe_unused]] size_t v18 = 2;
  [[maybe_unused]] size_t v19 = 1;
  [[maybe_unused]] size_t v20 = 6;
  [[maybe_unused]] size_t v21 = 7;
  [[maybe_unused]] size_t v22 = 18;
  [[maybe_unused]] size_t v23 = 21;
  [[maybe_unused]] size_t v24 = 32;
  [[maybe_unused]] size_t v25 = 28;
  [[maybe_unused]] size_t v26 = 15;
  [[maybe_unused]] size_t v27 = 24;
  [[maybe_unused]] size_t v28 = 9;
  [[maybe_unused]] size_t v29 = 16;
  [[maybe_unused]] size_t v30 = 12;
  [[maybe_unused]] size_t v31 = 20;
  [[maybe_unused]] size_t v32 = 8;
  [[maybe_unused]] size_t v33 = 19;
  [[maybe_unused]] size_t v34 = 4;
  [[maybe_unused]] size_t v35 = 14;
  [[maybe_unused]] size_t v36 = 0;
  [[maybe_unused]] size_t v37 = 13;
  int32_t v38 = v0[13];
  int32_t v39 = v0[14];
  int32_t v40 = v0[19];
  int32_t v41 = v0[20];
  int32_t v42 = v0[8];
  int32_t v43 = v0[9];
  int32_t v44 = v0[15];
  int32_t v45 = v0[21];
  int32_t v46 = v0[12];
  int32_t v47 = v0[18];
  int32_t v48 = v0[7];
  int32_t v49 = v0[6];
  int32_t v50 = v0[1];
  int32_t v51 = v0[2];
  int32_t v52 = v0[3];
  int32_t v53 = v0[0];
  int32_t v54 = v0[16];
  int32_t v55 = v0[22];
  int32_t v56 = v0[10];
  int32_t v57 = v0[11];
  int32_t v58 = v0[17];
  int32_t v59 = v0[23];
  int32_t v60 = v0[4];
  int32_t v61 = v0[5];
  int32_t v62 = v0[25];
  int32_t v63 = v0[26];
  int32_t v64 = v0[31];
  int32_t v65 = v0[32];
  int32_t v66 = v0[27];
  int32_t v67 = v0[33];
  int32_t v68 = v0[24];
  int32_t v69 = v0[30];
  int32_t v70 = v0[28];
  int32_t v71 = v0[34];
  int32_t v72 = v0[29];
  int32_t v73 = v0[35];
  const std::vector<int32_t> v74 = {
      v38, v44, v62, v66, v39, v54, v63, v70, v40, v45, v64, v67, v41, v55, v65,
      v71, v42, v56, v41, v55, v43, v57, v45, v59, v39, v54, v63, v70, v44, v58,
      v66, v72, v39, v54, v63, v70, v44, v58, v66, v72, v41, v55, v65, v71, v45,
      v59, v67, v73, v46, v39, v68, v63, v38, v44, v62, v66, v47, v41, v69, v65,
      v40, v45, v64, v67, v48, v43, v40, v45, v42, v56, v41, v55, v38, v44, v62,
      v66, v39, v54, v63, v70, v49, v42, v47, v41, v48, v43, v40, v45, v46, v39,
      v68, v63, v38, v44, v62, v66, v50, v52, v38, v44, v51, v60, v39, v54, v48,
      v43, v40, v45, v42, v56, v41, v55, v51, v60, v39, v54, v52, v61, v44, v58,
      v42, v56, v41, v55, v43, v57, v45, v59, v53, v51, v46, v39, v50, v52, v38,
      v44, v49, v42, v47, v41, v48, v43, v40, v45, v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v38, v44, v62, v66, v39, v54, v63, v70, v40, v45, v64, v67, v41, v55,
      v65, v71, v42, v56, v41, v55, v43, v57, v45, v59, v39, v54, v63, v70, v44,
      v58, v66, v72, v39, v54, v63, v70, v44, v58, v66, v72, v41, v55, v65, v71,
      v45, v59, v67, v73, v46, v39, v68, v63, v38, v44, v62, v66, v47, v41, v69,
      v65, v40, v45, v64, v67, v48, v43, v40, v45, v42, v56, v41, v55, v38, v44,
      v62, v66, v39, v54, v63, v70, v49, v42, v47, v41, v48, v43, v40, v45, v46,
      v39, v68, v63, v38, v44, v62, v66, v50, v52, v38, v44, v51, v60, v39, v54,
      v48, v43, v40, v45, v42, v56, v41, v55, v51, v60, v39, v54, v52, v61, v44,
      v58, v42, v56, v41, v55, v43, v57, v45, v59, v53, v51, v46, v39, v50, v52,
      v38, v44, v49, v42, v47, v41, v48, v43, v40, v45, v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v38, v44, v62, v66, v39, v54, v63, v70, v40, v45, v64, v67, v41,
      v55, v65, v71, v42, v56, v41, v55, v43, v57, v45, v59, v39, v54, v63, v70,
      v44, v58, v66, v72, v39, v54, v63, v70, v44, v58, v66, v72, v41, v55, v65,
      v71, v45, v59, v67, v73, v46, v39, v68, v63, v38, v44, v62, v66, v47, v41,
      v69, v65, v40, v45, v64, v67, v48, v43, v40, v45, v42, v56, v41, v55, v38,
      v44, v62, v66, v39, v54, v63, v70, v49, v42, v47, v41, v48, v43, v40, v45,
      v46, v39, v68, v63, v38, v44, v62, v66, v50, v52, v38, v44, v51, v60, v39,
      v54, v48, v43, v40, v45, v42, v56, v41, v55, v51, v60, v39, v54, v52, v61,
      v44, v58, v42, v56, v41, v55, v43, v57, v45, v59, v53, v51, v46, v39, v50,
      v52, v38, v44, v49, v42, v47, v41, v48, v43, v40, v45, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v38, v44, v62, v66, v39, v54, v63, v70, v40, v45, v64, v67,
      v41, v55, v65, v71, v42, v56, v41, v55, v43, v57, v45, v59, v39, v54, v63,
      v70, v44, v58, v66, v72, v39, v54, v63, v70, v44, v58, v66, v72, v41, v55,
      v65, v71, v45, v59, v67, v73, v46, v39, v68, v63, v38, v44, v62, v66, v47,
      v41, v69, v65, v40, v45, v64, v67, v48, v43, v40, v45, v42, v56, v41, v55,
      v38, v44, v62, v66, v39, v54, v63, v70, v49, v42, v47, v41, v48, v43, v40,
      v45, v46, v39, v68, v63, v38, v44, v62, v66, v50, v52, v38, v44, v51, v60,
      v39, v54, v48, v43, v40, v45, v42, v56, v41, v55, v51, v60, v39, v54, v52,
      v61, v44, v58, v42, v56, v41, v55, v43, v57, v45, v59, v53, v51, v46, v39,
      v50, v52, v38, v44, v49, v42, v47, v41, v48, v43, v40, v45, v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1};
  std::vector<int32_t> v75(1 * 1024);
  for (int64_t v75_i0 = 0; v75_i0 < 1; ++v75_i0) {
    for (int64_t v75_i1 = 0; v75_i1 < 1024; ++v75_i1) {
      v75[v75_i1 + 1024 * (v75_i0)] =
          v74[0 + v75_i1 * 1 + 1024 * (0 + v75_i0 * 1)];
    }
  }
  std::vector<int64_t> v76(std::begin(v75), std::end(v75));
  auto pt_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v76;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (unsigned i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v76[i % v76.size()]);
  }
  auto pt = cc->MakePackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  const std::vector<CiphertextT> v77 = {ct};
  return v77;
}
std::vector<CiphertextT> conv_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                       std::vector<int32_t> v0,
                                                       PublicKeyT pk) {
  [[maybe_unused]] int32_t v1 = 0;
  [[maybe_unused]] size_t v2 = 2;
  [[maybe_unused]] size_t v3 = 1;
  [[maybe_unused]] size_t v4 = 3;
  [[maybe_unused]] size_t v5 = 6;
  [[maybe_unused]] size_t v6 = 5;
  [[maybe_unused]] size_t v7 = 8;
  [[maybe_unused]] size_t v8 = 4;
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
      v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11,
      v11, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12,
      v12, v12, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13,
      v13, v13, v13, v14, v14, v14, v14, v14, v14, v14, v14, v14, v14, v14, v14,
      v14, v14, v14, v14, v15, v15, v15, v15, v15, v15, v15, v15, v15, v15, v15,
      v15, v15, v15, v15, v15, v16, v16, v16, v16, v16, v16, v16, v16, v16, v16,
      v16, v16, v16, v16, v16, v16, v17, v17, v17, v17, v17, v17, v17, v17, v17,
      v17, v17, v17, v17, v17, v17, v17, v18, v18, v18, v18, v18, v18, v18, v18,
      v18, v18, v18, v18, v18, v18, v18, v18, v19, v19, v19, v19, v19, v19, v19,
      v19, v19, v19, v19, v19, v19, v19, v19, v19, v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11,
      v11, v11, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12,
      v12, v12, v12, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13,
      v13, v13, v13, v13, v14, v14, v14, v14, v14, v14, v14, v14, v14, v14, v14,
      v14, v14, v14, v14, v14, v15, v15, v15, v15, v15, v15, v15, v15, v15, v15,
      v15, v15, v15, v15, v15, v15, v16, v16, v16, v16, v16, v16, v16, v16, v16,
      v16, v16, v16, v16, v16, v16, v16, v17, v17, v17, v17, v17, v17, v17, v17,
      v17, v17, v17, v17, v17, v17, v17, v17, v18, v18, v18, v18, v18, v18, v18,
      v18, v18, v18, v18, v18, v18, v18, v18, v18, v19, v19, v19, v19, v19, v19,
      v19, v19, v19, v19, v19, v19, v19, v19, v19, v19, v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11,
      v11, v11, v11, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12,
      v12, v12, v12, v12, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13,
      v13, v13, v13, v13, v13, v14, v14, v14, v14, v14, v14, v14, v14, v14, v14,
      v14, v14, v14, v14, v14, v14, v15, v15, v15, v15, v15, v15, v15, v15, v15,
      v15, v15, v15, v15, v15, v15, v15, v16, v16, v16, v16, v16, v16, v16, v16,
      v16, v16, v16, v16, v16, v16, v16, v16, v17, v17, v17, v17, v17, v17, v17,
      v17, v17, v17, v17, v17, v17, v17, v17, v17, v18, v18, v18, v18, v18, v18,
      v18, v18, v18, v18, v18, v18, v18, v18, v18, v18, v19, v19, v19, v19, v19,
      v19, v19, v19, v19, v19, v19, v19, v19, v19, v19, v19, v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11, v11,
      v11, v11, v11, v11, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12, v12,
      v12, v12, v12, v12, v12, v13, v13, v13, v13, v13, v13, v13, v13, v13, v13,
      v13, v13, v13, v13, v13, v13, v14, v14, v14, v14, v14, v14, v14, v14, v14,
      v14, v14, v14, v14, v14, v14, v14, v15, v15, v15, v15, v15, v15, v15, v15,
      v15, v15, v15, v15, v15, v15, v15, v15, v16, v16, v16, v16, v16, v16, v16,
      v16, v16, v16, v16, v16, v16, v16, v16, v16, v17, v17, v17, v17, v17, v17,
      v17, v17, v17, v17, v17, v17, v17, v17, v17, v17, v18, v18, v18, v18, v18,
      v18, v18, v18, v18, v18, v18, v18, v18, v18, v18, v18, v19, v19, v19, v19,
      v19, v19, v19, v19, v19, v19, v19, v19, v19, v19, v19, v19, v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,  v1,
      v1,  v1,  v1,  v1};
  std::vector<int32_t> v21(1 * 1024);
  for (int64_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
    for (int64_t v21_i1 = 0; v21_i1 < 1024; ++v21_i1) {
      v21[v21_i1 + 1024 * (v21_i0)] =
          v20[0 + v21_i1 * 1 + 1024 * (0 + v21_i0 * 1)];
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
  [[maybe_unused]] size_t v1 = 31;
  [[maybe_unused]] size_t v2 = 27;
  [[maybe_unused]] size_t v3 = 30;
  [[maybe_unused]] size_t v4 = 26;
  [[maybe_unused]] size_t v5 = 23;
  [[maybe_unused]] size_t v6 = 19;
  [[maybe_unused]] size_t v7 = 22;
  [[maybe_unused]] size_t v8 = 18;
  [[maybe_unused]] size_t v9 = 29;
  [[maybe_unused]] size_t v10 = 25;
  [[maybe_unused]] size_t v11 = 28;
  [[maybe_unused]] size_t v12 = 24;
  [[maybe_unused]] size_t v13 = 21;
  [[maybe_unused]] size_t v14 = 17;
  [[maybe_unused]] size_t v15 = 20;
  [[maybe_unused]] size_t v16 = 16;
  [[maybe_unused]] size_t v17 = 0;
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(1024);
  const auto& v18_cast = pt->GetPackedValue();
  std::vector<int32_t> v18(std::begin(v18_cast), std::end(v18_cast));
  int32_t v19 = v18[16 + 1024 * (0)];
  int32_t v20 = v18[20 + 1024 * (0)];
  int32_t v21 = v18[17 + 1024 * (0)];
  int32_t v22 = v18[21 + 1024 * (0)];
  int32_t v23 = v18[24 + 1024 * (0)];
  int32_t v24 = v18[28 + 1024 * (0)];
  int32_t v25 = v18[25 + 1024 * (0)];
  int32_t v26 = v18[29 + 1024 * (0)];
  int32_t v27 = v18[18 + 1024 * (0)];
  int32_t v28 = v18[22 + 1024 * (0)];
  int32_t v29 = v18[19 + 1024 * (0)];
  int32_t v30 = v18[23 + 1024 * (0)];
  int32_t v31 = v18[26 + 1024 * (0)];
  int32_t v32 = v18[30 + 1024 * (0)];
  int32_t v33 = v18[27 + 1024 * (0)];
  int32_t v34 = v18[31 + 1024 * (0)];
  const std::vector<int32_t> v35 = {v19, v20, v21, v22, v23, v24, v25, v26,
                                    v27, v28, v29, v30, v31, v32, v33, v34};
  return v35;
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
  cc->EvalRotateKeyGen(sk, {64, 16, 48, 240, 304});
  return cc;
}
