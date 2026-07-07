bazel run //tools:heir-opt -- \
  /local/scratch/a/paranjav/biscotti/heir/kernel-mm4-baseline.mlir \
  --mlir-to-bfv='ciphertext-degree=1024 plaintext-modulus=65537 enable-split-preprocessing=1' \
  --scheme-to-openfhe \
  -o /local/scratch/a/paranjav/biscotti/heir/output-openfhe.mlir

# Step 2: Translate to C++ implementation
bazel run //tools:heir-translate -- \
  /local/scratch/a/paranjav/biscotti/heir/output-openfhe.mlir \
  --openfhe-include-type=source-relative \
  --emit-openfhe-pke \
  -o /local/scratch/a/paranjav/biscotti/heir/mm_bench/kernel.cpp

# Step 3: Translate to C++ header
bazel run //tools:heir-translate -- \
  /local/scratch/a/paranjav/biscotti/heir/output-openfhe.mlir \
  --openfhe-include-type=source-relative \
  --emit-openfhe-pke-header \
  -o /local/scratch/a/paranjav/biscotti/heir/mm_bench/kernel.h

# (Optional) Step 4: pybind11 bindings for calling from Python
# bazel run //tools:heir-translate -- \
#   /local/scratch/a/paranjav/biscotti/heir/output-openfhe.mlir \
#   --openfhe-include-type=source-relative \
#   --emit-openfhe-pke-pybind \
#   --pybind-header-include=kernel.h \
#   --pybind-module-name=mm_kernel \
#   -o /local/scratch/a/paranjav/biscotti/heir/mm_bench/kernel_pybind.cpp
