#!/bin/bash
# Compile an MLIR kernel through HEIR to an OpenFHE C++ kernel pair
# (kernel.cpp + kernel.h) placed in an output folder.
#
# Usage:
#   ./build_kernel.sh <input.mlir> <output-folder>
#
# Example:
#   ./build_kernel.sh ../kernel-mm4-baseline.mlir ../mm_bench/
#
# Optional env vars:
#   HEIR_ROOT    - HEIR checkout root (default: parent of this script's dir)
#   CT_DEGREE    - ciphertext-degree for --mlir-to-bfv (default: 1024)

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <input.mlir> <output-folder>" >&2
  exit 1
fi

INPUT="$1"
OUT_DIR="$2"

if [[ ! -f "$INPUT" ]]; then
  echo "error: input file not found: $INPUT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
INPUT_ABS="$(readlink -f "$INPUT")"
OUT_ABS="$(readlink -f "$OUT_DIR")"

HEIR_ROOT="${HEIR_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
CT_DEGREE="${CT_DEGREE:-1024}"

# Intermediate lives next to the output for easier debugging.
STEM="$(basename "${INPUT_ABS%.mlir}")"
OFHE_MLIR="${OUT_ABS}/${STEM}-openfhe.mlir"

echo "=== HEIR pipeline ==="
echo "  input:      $INPUT_ABS"
echo "  output dir: $OUT_ABS"
echo "  heir root:  $HEIR_ROOT"
echo "  ct_degree=$CT_DEGREE"
echo

cd "$HEIR_ROOT"

echo "=== Step 1: heir-opt (--mlir-to-bfv, --scheme-to-openfhe) ==="
bazel run //tools:heir-opt -- \
  "$INPUT_ABS" \
  --mlir-to-bfv="ciphertext-degree=${CT_DEGREE} plaintext-modulus=65537 enable-split-preprocessing=1" \
  --scheme-to-openfhe \
  -o "$OFHE_MLIR"

echo "=== Step 2: heir-translate --emit-openfhe-pke → kernel.cpp ==="
bazel run //tools:heir-translate -- \
  "$OFHE_MLIR" \
  --openfhe-include-type=source-relative \
  --emit-openfhe-pke \
  -o "${OUT_ABS}/kernel.cpp"

echo "=== Step 3: heir-translate --emit-openfhe-pke-header → kernel.h ==="
bazel run //tools:heir-translate -- \
  "$OFHE_MLIR" \
  --openfhe-include-type=source-relative \
  --emit-openfhe-pke-header \
  -o "${OUT_ABS}/kernel.h"

echo
echo "=== Done ==="
echo "  $OFHE_MLIR"
echo "  ${OUT_ABS}/kernel.cpp"
echo "  ${OUT_ABS}/kernel.h"
