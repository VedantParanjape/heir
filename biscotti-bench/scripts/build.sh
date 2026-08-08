#!/bin/bash
# Compile an MLIR kernel through HEIR to an OpenFHE C++ kernel triple
# (kernel.cpp + kernel.h + benchmark.cpp) placed under the suite's
# output/ subdirectory.
#
# Layout (relative to WORKSPACE root):
#   benchmarks/<suite>/src/<kernel>.mlir             ← authored input
#   benchmarks/<suite>/build/<kernel>/               ← intermediate MLIRs
#   benchmarks/<suite>/output/<kernel>/kernel.cpp    ← emitted
#   benchmarks/<suite>/output/<kernel>/kernel.h      ← emitted
#   benchmarks/<suite>/output/<kernel>/benchmark.cpp ← emitted
#
# Always runs recursive-call-vectorization + canonicalize + cse + symbol-dce
# followed by strip_scaffold.py as Step 0.
#
# Usage:
#   ./scripts/build.sh [flags] <suite> <kernel>
#
# Flags:
#   --baseline           Shortcut for --threshold=-1 (fully unroll recursive
#                        calls).
#   --threshold=N        node-size-threshold for recursive-call-vectorization.
#                        -1 means unlimited. Default: 100.
#
# Example:
#   ./scripts/build.sh --baseline mm mm4x4
#   ./scripts/build.sh --threshold=8 mm mm6x6
#   ./scripts/build.sh dot dot_100
#
# Optional env vars:
#   HEIR_ROOT     - HEIR checkout root (default: workspace root inferred
#                   from this script's location: <script dir>/..)
#   WORKSPACE     - path to the repo root that holds benchmarks/ (default:
#                   $HEIR_ROOT)
#   CT_DEGREE     - ciphertext-degree for --mlir-to-bfv. Default `-1` means
#                   auto-detect from the recursed MLIR (uses the max
#                   inner-dim of any `!secret.secret<tensor<NxKx...>>` type
#                   found on any func arg/result). Set to a concrete number
#                   (e.g. 1024) to override.

set -euo pipefail

# ---- flag parsing ----
THRESHOLD="100"
BASELINE=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      THRESHOLD="-1"; BASELINE=1; shift ;;
    --threshold=*)
      THRESHOLD="${1#*=}"; shift ;;
    --threshold)
      THRESHOLD="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    -*)
      echo "error: unknown flag $1" >&2; exit 1 ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
  echo "usage: $0 [--baseline | --threshold=N] <suite> <kernel>" >&2
  exit 1
fi

SUITE="${POSITIONAL[0]}"
KERNEL="${POSITIONAL[1]}"

# ---- path derivation ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# HEIR workspace root is two levels up: scripts/ → biscotti-bench/ → heir/.
HEIR_ROOT="${HEIR_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
# biscotti-bench/ is the repo that holds the benchmarks/ tree.
BENCH_REPO="${BENCH_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CT_DEGREE="${CT_DEGREE:--1}"

SUITE_DIR="${BENCH_REPO}/benchmarks/${SUITE}"
INPUT="${SUITE_DIR}/src/${KERNEL}.mlir"

# --baseline emits into <kernel>_baseline/ subdirs so the hoisted and
# baseline variants can coexist and be benchmarked side-by-side without
# clobbering each other.
VARIANT_SUFFIX=""
if [[ $BASELINE -eq 1 ]]; then VARIANT_SUFFIX="_baseline"; fi
BUILD_DIR="${SUITE_DIR}/build/${KERNEL}${VARIANT_SUFFIX}"
OUT_DIR="${SUITE_DIR}/output/${KERNEL}${VARIANT_SUFFIX}"

if [[ ! -f "$INPUT" ]]; then
  echo "error: input file not found: $INPUT" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR" "$OUT_DIR"
INPUT_ABS="$(readlink -f "$INPUT")"
BUILD_ABS="$(readlink -f "$BUILD_DIR")"
OUT_ABS="$(readlink -f "$OUT_DIR")"

STEM="${KERNEL}"
RECURSED_MLIR="${BUILD_ABS}/${STEM}-recursed.mlir"
OFHE_MLIR="${BUILD_ABS}/${STEM}-openfhe.mlir"
STRIP_SCAFFOLD="${SCRIPT_DIR}/strip_scaffold.py"

echo "=== HEIR pipeline ==="
echo "  suite:       $SUITE"
echo "  kernel:      $KERNEL"
echo "  input:       $INPUT_ABS"
echo "  build dir:   $BUILD_ABS"
echo "  output dir:  $OUT_ABS"
echo "  heir root:   $HEIR_ROOT"
if [[ "$CT_DEGREE" == "-1" ]]; then
  echo "  ct_degree:   auto-detect (from post-recursed MLIR)"
else
  echo "  ct_degree:   $CT_DEGREE"
fi
echo "  recursive-call-vectorization: node-size-threshold=$THRESHOLD"
echo

cd "$HEIR_ROOT"

# All bazel/heir output for this build is captured in a single log
# file. Step banners stay on the console so you can see progress; the
# noisy output goes to the log. On failure we dump the tail of the log
# so the error is still visible without opening the file.
LOG_FILE="${BUILD_ABS}/build.log"
: > "$LOG_FILE"  # truncate
echo "  log:         $LOG_FILE"
echo

run_step() {
  local label="$1"; shift
  echo "=== $label ==="
  echo >> "$LOG_FILE"
  echo "=== $label ===" >> "$LOG_FILE"
  if ! "$@" >> "$LOG_FILE" 2>&1; then
    echo "error: step failed; tail of log:" >&2
    tail -n 60 "$LOG_FILE" >&2
    exit 1
  fi
}

run_step "Step 0: heir-opt --recursive-call-vectorization + strip scaffold" \
  bazel run //tools:heir-opt -- \
    "$INPUT_ABS" \
    "--recursive-call-vectorization=node-size-threshold=${THRESHOLD}" \
    --canonicalize --cse --symbol-dce \
    -o "$RECURSED_MLIR"
run_step "Step 0b: strip biscotti scaffold" \
  python3 "$STRIP_SCAFFOLD" "$RECURSED_MLIR" -o "$RECURSED_MLIR"

# Auto-detect ciphertext-degree from the recursed MLIR when CT_DEGREE == -1.
# Coyote emits secret ciphertext-semantic types like
# `!secret.secret<tensor<NxKxi32>>`; we pick the maximum inner-dim K across
# all such types found in the file so that heir-opt's --mlir-to-bfv sees a
# ciphertext-degree big enough to hold every arg/result. Fallback: 1024 if
# no such types are present in the file (shouldn't happen for a real
# secret compute).
if [[ "$CT_DEGREE" == "-1" ]]; then
  DETECTED=$(python3 -c '
import re, sys
txt = open(sys.argv[1]).read()
dims = re.findall(r"!secret\.secret<tensor<\d+x(\d+)x", txt)
print(max((int(d) for d in dims), default=1024))
' "$RECURSED_MLIR")
  echo "=== Auto-detected ciphertext-degree=${DETECTED} from $(basename "$RECURSED_MLIR") ==="
  CT_DEGREE="$DETECTED"
fi

run_step "Step 1: heir-opt (--mlir-to-bfv, --scheme-to-openfhe)" \
  bazel run //tools:heir-opt -- \
    "$RECURSED_MLIR" \
    --mlir-to-bfv="enable-arithmetization=false ciphertext-degree=${CT_DEGREE} plaintext-modulus=65537 enable-split-preprocessing=1" \
    --scheme-to-openfhe \
    -o "$OFHE_MLIR"

run_step "Step 2: heir-translate --emit-openfhe-pke → kernel.cpp" \
  bazel run //tools:heir-translate -- \
    "$OFHE_MLIR" \
    --openfhe-include-type=source-relative \
    --emit-openfhe-pke \
    -o "${OUT_ABS}/kernel.cpp"

run_step "Step 3: heir-translate --emit-openfhe-pke-header → kernel.h" \
  bazel run //tools:heir-translate -- \
    "$OFHE_MLIR" \
    --openfhe-include-type=source-relative \
    --emit-openfhe-pke-header \
    -o "${OUT_ABS}/kernel.h"

run_step "Step 4: heir-translate --emit-openfhe-pke-harness → benchmark.cpp" \
  bazel run //tools:heir-translate -- \
    "$OFHE_MLIR" \
    --emit-openfhe-pke-harness \
    --harness-header-include="kernel.h" \
    -o "${OUT_ABS}/benchmark.cpp"

echo
echo "=== Done ==="
echo "  $RECURSED_MLIR"
echo "  $OFHE_MLIR"
echo "  ${OUT_ABS}/kernel.cpp"
echo "  ${OUT_ABS}/kernel.h"
echo "  ${OUT_ABS}/benchmark.cpp"
echo "  ${LOG_FILE}"
