
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

std::vector<Plaintext> mm_clone_0_0__preprocessing(CryptoContextT cc);
struct mm_clone_0_0__preprocessedStruct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
};
mm_clone_0_0__preprocessedStruct mm_clone_0_0__preprocessed(
    CryptoContextT cc, std::vector<CiphertextT> v0, std::vector<CiphertextT> v1,
    std::vector<CiphertextT> v2, std::vector<CiphertextT> v3,
    const std::vector<Plaintext>& v4);
struct mm_clone_0_0Struct {
  std::vector<CiphertextT> arg0;
  std::vector<CiphertextT> arg1;
};
mm_clone_0_0Struct mm_clone_0_0(CryptoContextT cc, std::vector<CiphertextT> v0,
                                std::vector<CiphertextT> v1,
                                std::vector<CiphertextT> v2,
                                std::vector<CiphertextT> v3);
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg0(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk);
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg1(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk);
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg2(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk);
std::vector<CiphertextT> mm_clone_0_0__encrypt__arg3(CryptoContextT cc,
                                                     std::vector<int32_t> v0,
                                                     PublicKeyT pk);
std::vector<int32_t> mm_clone_0_0__decrypt__result0(CryptoContextT cc,
                                                    std::vector<CiphertextT> v0,
                                                    PrivateKeyT sk);
std::vector<int32_t> mm_clone_0_0__decrypt__result1(CryptoContextT cc,
                                                    std::vector<CiphertextT> v0,
                                                    PrivateKeyT sk);
CryptoContextT mm_clone_0_0__generate_crypto_context();
CryptoContextT mm_clone_0_0__configure_crypto_context(CryptoContextT cc,
                                                      PrivateKeyT sk);
