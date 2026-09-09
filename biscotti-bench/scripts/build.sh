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
#                        calls). Schedules with our modified (biscotti) Coyote.
#   --baseline-vanilla   Like --baseline, but schedules with the unmodified
#                        upstream Coyote (COYOTE_VANILLA_DIR) instead of the
#                        modified one. Emits into <kernel>_baseline_vanilla/.
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
#   COYOTE_BISCOTTI_DIR - dir holding run_coyote_from_circuit.py for the
#                   modified (biscotti) Coyote. Used by every variant except
#                   --baseline-vanilla.
#   COYOTE_VANILLA_DIR  - same, for the unmodified upstream Coyote. Used only
#                   by --baseline-vanilla.
#   CT_DEGREE     - ciphertext-degree for --mlir-to-bfv. Default `-1` means
#                   auto-detect from the recursed MLIR (uses the max
#                   inner-dim of any `!secret.secret<tensor<NxKx...>>` type
#                   found on any func arg/result). Set to a concrete number
#                   (e.g. 1024) to override.

set -euo pipefail

# ---- flag parsing ----
THRESHOLD="100"
BASELINE=0
VANILLA=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      THRESHOLD="-1"; BASELINE=1; shift ;;
    --baseline-vanilla)
      THRESHOLD="-1"; BASELINE=1; VANILLA=1; shift ;;
    --threshold=*)
      THRESHOLD="${1#*=}"; shift ;;
    --threshold)
      THRESHOLD="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,47p' "$0"; exit 0 ;;
    -*)
      echo "error: unknown flag $1" >&2; exit 1 ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
  echo "usage: $0 [--baseline | --baseline-vanilla | --threshold=N] <suite> <kernel>" >&2
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
# Compilation mode for the heir tools. Default `opt` (release) so heir-opt /
# heir-translate are optimized -- important for meaningful compile-time numbers
# and to enable NDEBUG. Override with BAZEL_MODE=dbg for debugging.
BAZEL_MODE="${BAZEL_MODE:-opt}"

# ---- Coyote scheduler location ----
# The recursive-call-vectorization pass locates Coyote's bridge script
# (run_coyote_from_circuit.py) via the COYOTE_PYTHON_PATH env var. We keep two
# checkouts so --baseline-vanilla can be scheduled by an unmodified upstream
# Coyote while every other variant uses our modified (biscotti) Coyote.
# Set these to the directories that contain run_coyote_from_circuit.py.
COYOTE_BISCOTTI_DIR="${COYOTE_BISCOTTI_DIR:-/path/to/coyote}"          # modified
COYOTE_VANILLA_DIR="${COYOTE_VANILLA_DIR:-/path/to/coyote-vanilla}"    # upstream

if [[ $VANILLA -eq 1 ]]; then
  COYOTE_PYTHON_PATH="$COYOTE_VANILLA_DIR"
else
  COYOTE_PYTHON_PATH="$COYOTE_BISCOTTI_DIR"
fi
export COYOTE_PYTHON_PATH

# Build the heir tools ONCE with bazel, BEFORE activating the venv, and then
# call the resulting binaries directly (never `bazel run` again). This keeps the
# venv-modified PATH out of bazel entirely: bazel's toolchain repo rules read
# the live PATH, so activating the venv before a `bazel run` re-resolves them
# and rebuilds the whole LLVM/MLIR exec-config from scratch on every invocation.
# Building here (clean PATH) once, then exec'ing bazel-bin binaries, avoids that.
( cd "$HEIR_ROOT" && bazel build -c "${BAZEL_MODE}" //tools:heir-opt //tools:heir-translate )
HEIR_OPT="${HEIR_ROOT}/bazel-bin/tools/heir-opt"
HEIR_TRANSLATE="${HEIR_ROOT}/bazel-bin/tools/heir-translate"

# Activate the Coyote venv (networkx, z3-solver) that sits next to the bridge,
# so the pass's `python3` invocation has the deps regardless of the caller's
# environment. Each variant activates its own venv (biscotti vs vanilla).
# Activated AFTER the bazel build above so the venv PATH never reaches bazel.
if [[ -f "${COYOTE_PYTHON_PATH}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${COYOTE_PYTHON_PATH}/.venv/bin/activate"
fi

SUITE_DIR="${BENCH_REPO}/benchmarks/${SUITE}"
INPUT="${SUITE_DIR}/src/${KERNEL}.mlir"

# --baseline emits into <kernel>_baseline/ subdirs so the hoisted and
# baseline variants can coexist and be benchmarked side-by-side without
# clobbering each other.
VARIANT_SUFFIX=""
if [[ $BASELINE -eq 1 ]]; then VARIANT_SUFFIX="_baseline"; fi
if [[ $VANILLA -eq 1 ]]; then VARIANT_SUFFIX="_baseline_vanilla"; fi
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
echo "  coyote:      $COYOTE_PYTHON_PATH$([[ $VANILLA -eq 1 ]] && echo '  (vanilla)' || echo '  (biscotti)')"
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
# Dedicated compile-timing report for the vectorization step (Step 0), where
# the Coyote scheduler runs. One file per (kernel, variant), so the three
# configs' compile times can be diffed directly.
COMPILE_TIMING="${BUILD_ABS}/compile-timing.txt"
{
  echo "# compile-timing: ${SUITE}/${KERNEL}${VARIANT_SUFFIX}"
  echo "# coyote:          ${COYOTE_PYTHON_PATH} ($([[ $VANILLA -eq 1 ]] && echo vanilla || echo biscotti))"
  echo "# node-threshold:  ${THRESHOLD}"
  echo "# date:            $(date -Is 2>/dev/null || date)"
  echo
} > "$COMPILE_TIMING"
echo "  log:         $LOG_FILE"
echo "  timing:      $COMPILE_TIMING"
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

# Like run_step, but also records this step's wall time -- and any MLIR
# pass-timing table it emitted -- into COMPILE_TIMING. Used for every pipeline
# step so the full compile time (vectorization + lowering + heir-translate
# emits) is captured per (kernel, variant). heir-opt steps additionally pass
# --mlir-timing for an in-process per-pass breakdown; heir-translate is not a
# pass pipeline, so only its wall time is available.
timed_step() {
  local label="$1"; shift
  local pre t0 t1
  pre=$(wc -l < "$LOG_FILE")
  t0=$(date +%s.%N)
  run_step "$label" "$@"
  t1=$(date +%s.%N)
  {
    echo "== ${label} =="
    echo "wall_seconds (incl. bazel launch): $(awk "BEGIN{printf \"%.3f\", ${t1}-${t0}}")"
    # Extract just this step's MLIR pass-timing table (the lines it appended).
    tail -n +"$((pre + 1))" "$LOG_FILE" \
      | awk 'tolower($0) ~ /timing report|execution time report/{p=1} p'
    echo
  } >> "$COMPILE_TIMING"
}

# The whole pipeline is timed via timed_step; each step's wall time (and pass
# timing, for heir-opt steps) is appended to COMPILE_TIMING.
PIPELINE_START=$(date +%s.%N)

# Step 0 (recursive-call-vectorization) invokes the Coyote scheduler. --mlir-
# timing makes heir-opt emit a per-pass table whose RecursiveCallVectorization
# row includes the synchronous Coyote subprocess -- the scheduler compile time.
timed_step "Step 0: heir-opt --recursive-call-vectorization + strip scaffold" \
  "$HEIR_OPT" \
    "$INPUT_ABS" \
    "--recursive-call-vectorization=node-size-threshold=${THRESHOLD}" \
    --canonicalize --cse --symbol-dce \
    --mlir-timing \
    -o "$RECURSED_MLIR"
timed_step "Step 0b: strip biscotti scaffold" \
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

timed_step "Step 1: heir-opt (--mlir-to-bfv, --scheme-to-openfhe)" \
  "$HEIR_OPT" \
    "$RECURSED_MLIR" \
    --mlir-to-bfv="enable-arithmetization=false ciphertext-degree=${CT_DEGREE} plaintext-modulus=65537 enable-split-preprocessing=1" \
    --scheme-to-openfhe \
    --mlir-timing \
    -o "$OFHE_MLIR"

timed_step "Step 2: heir-translate --emit-openfhe-pke → kernel.cpp" \
  "$HEIR_TRANSLATE" \
    "$OFHE_MLIR" \
    --openfhe-include-type=source-relative \
    --emit-openfhe-pke \
    -o "${OUT_ABS}/kernel.cpp"

timed_step "Step 3: heir-translate --emit-openfhe-pke-header → kernel.h" \
  "$HEIR_TRANSLATE" \
    "$OFHE_MLIR" \
    --openfhe-include-type=source-relative \
    --emit-openfhe-pke-header \
    -o "${OUT_ABS}/kernel.h"

timed_step "Step 4: heir-translate --emit-openfhe-pke-harness → benchmark.cpp" \
  "$HEIR_TRANSLATE" \
    "$OFHE_MLIR" \
    --emit-openfhe-pke-harness \
    --harness-header-include="kernel.h" \
    -o "${OUT_ABS}/benchmark.cpp"

# Total pipeline wall time (includes the ct-degree detection between steps).
{
  echo "== TOTAL pipeline =="
  echo "wall_seconds: $(awk "BEGIN{printf \"%.3f\", $(date +%s.%N)-${PIPELINE_START}}")"
} >> "$COMPILE_TIMING"

echo
echo "=== Done ==="
echo "  $RECURSED_MLIR"
echo "  $OFHE_MLIR"
echo "  ${OUT_ABS}/kernel.cpp"
echo "  ${OUT_ABS}/kernel.h"
echo "  ${OUT_ABS}/benchmark.cpp"
echo "  ${LOG_FILE}"
echo "  ${COMPILE_TIMING}"
