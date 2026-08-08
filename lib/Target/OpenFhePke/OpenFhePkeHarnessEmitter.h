#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKEHARNESSEMITTER_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKEHARNESSEMITTER_H_

#include <string>

#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

/// Emit a benchmarking driver `main.cpp` for the HEIR-emitted OpenFHE PKE
/// kernel found in `op`.
///
/// The emitted main.cpp is workload-agnostic: everything matmul- or
/// dot-specific (input dimension, generators, reference computation, split)
/// is delegated to a user-maintained `bench_hooks.h` that declares
/// functions in the `bench_hooks` namespace:
///
///   int workload_size(int argc, char** argv);
///   std::vector<int32_t> gen_input_A(int size, uint32_t seed);
///   std::vector<int32_t> gen_input_B(int size, uint32_t seed);
///   std::vector<int32_t> baseline(const std::vector<int32_t>& A,
///                                 const std::vector<int32_t>& B);
///   std::vector<std::vector<int32_t>> split(
///       const std::vector<int32_t>& A, const std::vector<int32_t>& B,
///       int size);
///
/// The emitter derives from MLIR:
///   - the target kernel: first non-`main`, non-`*_dummy` `func.func`;
///   - N (encrypt-arg count) = kernel's arg count;
///   - the symbol prefix = kernel's name (used to name
///     <prefix>__encrypt__argK, <prefix>__decrypt__result0, etc).
///
/// `headerInclude` is the #include filename for the header emitted by
/// --emit-openfhe-pke-header. The google_benchmark name defaults to the
/// kernel's function symbol.
::mlir::LogicalResult translateToOpenFhePkeHarness(
    ::mlir::Operation* op, llvm::raw_ostream& os,
    const std::string& headerInclude);

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKEHARNESSEMITTER_H_
