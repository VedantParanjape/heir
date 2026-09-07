#!/bin/bash
# Build (bazel) + run one benchmark. The suite+kernel form derives the
# bazel target as //biscotti-bench/benchmarks/<suite>:bench_<kernel>.
# Everything after `--` is forwarded to the benchmark binary
# (google_benchmark flags).
#
# There are three build variants of every kernel, all emitted by
# scripts/build.sh (see its --baseline / --baseline-vanilla flags):
#   (default)          hoisted / coyote-optimized       -> bench_<kernel>
#   --baseline         biscotti Coyote, fully unrolled   (inline=-1)
#                                                        -> bench_<kernel>_baseline
#   --baseline-vanilla vanilla upstream Coyote, unrolled
#                                                -> bench_<kernel>_baseline_vanilla
#
# Usage:
#   ./scripts/run.sh [variant] <suite> <kernel> [-- <benchmark args>]
#   ./scripts/run.sh --compare  <suite> <kernel> [-- <benchmark args>]
#
# Single-run variant selector (run one config on its own):
#   (none)             run the hoisted target
#   --baseline         run the biscotti-baseline target
#   --baseline-vanilla run the vanilla-baseline target
#
# --compare runs all three targets back-to-back and diffs them pairwise
# (vanilla-baseline vs biscotti-baseline, biscotti-baseline vs hoisted, and
# vanilla-baseline vs hoisted). All three must already be emitted via
# build.sh <suite> <kernel>, build.sh --baseline <suite> <kernel>, and
# build.sh --baseline-vanilla <suite> <kernel>.
#
# Examples:
#   ./scripts/run.sh mm mm4x4
#   ./scripts/run.sh --baseline-vanilla mm mm4x4 -- --benchmark_repetitions=5
#   ./scripts/run.sh --compare mm mm4x4 -- --benchmark_repetitions=5
#
# Optional env vars:
#   HEIR_ROOT     - HEIR checkout root (default: <script dir>/../..)
#   BAZEL_MODE    - compilation mode passed to bazel (default: opt)
#   LOG_FILE      - if set, tee output here as well as stdout
#                   (single-target mode only)
#   COMPARE_PY    - path to google_benchmark's tools/compare.py.
#                   If found (via env or bazel's external cache),
#                   --compare invokes it on the JSON outputs for a
#                   side-by-side stat diff. Fallback: dump raw JSON.

set -euo pipefail

# ---- leading-flag parsing ----
COMPARE=0
VARIANT_SUFFIX=""     # "" | _baseline | _baseline_vanilla (single-run mode)
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --compare)          COMPARE=1; shift ;;
    --baseline)         VARIANT_SUFFIX="_baseline"; shift ;;
    --baseline-vanilla) VARIANT_SUFFIX="_baseline_vanilla"; shift ;;
    --)                 break ;;
    *) echo "error: unknown flag $1" >&2; exit 1 ;;
  esac
done

if [[ $# -lt 2 ]]; then
  echo "usage: $0 [--compare | --baseline | --baseline-vanilla] <suite> <kernel> [-- <benchmark args>]" >&2
  exit 1
fi

SUITE="$1"
KERNEL="$2"
shift 2

BENCH_ARGS=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then shift; fi
  BENCH_ARGS=("$@")
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
HEIR_ROOT="${HEIR_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BAZEL_MODE="${BAZEL_MODE:-opt}"

TARGET_BASE="//biscotti-bench/benchmarks/${SUITE}:bench_${KERNEL}"

cd "$HEIR_ROOT"

# If the caller passed --benchmark_repetitions but not
# --benchmark_report_aggregates_only, auto-add the latter so the JSON (and
# compare.py) gets a clean mean/median/stddev-only table.
EXTRA_ARGS=()
if [[ ${#BENCH_ARGS[@]} -gt 0 ]]; then
  wants_reps=0
  has_aggr=0
  for a in "${BENCH_ARGS[@]}"; do
    case "$a" in
      --benchmark_repetitions*)             wants_reps=1 ;;
      --benchmark_report_aggregates_only*)  has_aggr=1 ;;
    esac
  done
  if [[ $wants_reps -eq 1 && $has_aggr -eq 0 ]]; then
    EXTRA_ARGS+=("--benchmark_report_aggregates_only=true")
  fi
fi

# Run one target to the console (optionally tee'd to LOG_FILE).
run_one() {
  local target="$1"
  echo
  echo "=== bazel build (-c ${BAZEL_MODE}) ${target} ==="
  bazel build "-c" "${BAZEL_MODE}" "${target}"
  echo
  echo "=== running ${target} ==="
  if [[ -n "${LOG_FILE:-}" ]]; then
    bazel run "-c" "${BAZEL_MODE}" "${target}" -- \
      "${BENCH_ARGS[@]}" "${EXTRA_ARGS[@]}" | tee "${LOG_FILE}"
  else
    bazel run "-c" "${BAZEL_MODE}" "${target}" -- \
      "${BENCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
  fi
}

# Build one target and run it, emitting google_benchmark JSON to $2.
build_run_json() {
  local target="$1" out="$2"
  echo
  echo "=== bazel build (-c ${BAZEL_MODE}) ${target} ==="
  bazel build "-c" "${BAZEL_MODE}" "${target}"
  echo "=== running ${target} → ${out} ==="
  bazel run "-c" "${BAZEL_MODE}" "${target}" -- \
    --benchmark_format=json --benchmark_out="${out}" \
    "${BENCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
}

# Diff two JSON runs with compare.py (contender relative to ref).
compare_pair() {
  local ref_label="$1" ref_json="$2" con_label="$3" con_json="$4"
  echo
  echo "=== compare: ${ref_label} (ref) vs ${con_label} ==="
  python3 "$COMPARE_PY" benchmarks "${ref_json}" "${con_json}"
}

if [[ $COMPARE -eq 1 ]]; then
  # Locate google_benchmark's compare.py. Env override first, then the
  # bazel external-fetch cache. If neither, fall back to raw JSON output.
  if [[ -z "${COMPARE_PY:-}" ]]; then
    COMPARE_PY="$(find "${HOME}/.cache/bazel" -path '*google_benchmark*/tools/compare.py' -print -quit 2>/dev/null || true)"
  fi

  VANILLA_TARGET="${TARGET_BASE}_baseline_vanilla"
  BASELINE_TARGET="${TARGET_BASE}_baseline"
  HOISTED_TARGET="${TARGET_BASE}"

  VANILLA_JSON=$(mktemp --suffix=.json)
  BASELINE_JSON=$(mktemp --suffix=.json)
  HOISTED_JSON=$(mktemp --suffix=.json)
  # shellcheck disable=SC2064
  trap "rm -f '${VANILLA_JSON}' '${BASELINE_JSON}' '${HOISTED_JSON}'" EXIT

  # Run all three configs, each to its own JSON.
  build_run_json "${VANILLA_TARGET}"  "${VANILLA_JSON}"
  build_run_json "${BASELINE_TARGET}" "${BASELINE_JSON}"
  build_run_json "${HOISTED_TARGET}"  "${HOISTED_JSON}"

  echo
  if [[ -n "${COMPARE_PY:-}" && -f "$COMPARE_PY" ]]; then
    # compare.py is pairwise, so diff the three configs as three pairs:
    #   1. vanilla vs biscotti baseline  — effect of our Coyote changes alone
    #   2. biscotti baseline vs hoisted  — effect of decomposition + hoisting
    #   3. vanilla vs hoisted            — end-to-end, ours vs upstream Coyote
    compare_pair "vanilla-baseline"  "${VANILLA_JSON}"  "biscotti-baseline" "${BASELINE_JSON}"
    compare_pair "biscotti-baseline" "${BASELINE_JSON}" "hoisted"           "${HOISTED_JSON}"
    compare_pair "vanilla-baseline"  "${VANILLA_JSON}"  "hoisted"           "${HOISTED_JSON}"
  else
    echo "=== compare.py not found — dumping raw JSON outputs ==="
    echo "--- vanilla-baseline (${VANILLA_TARGET}) ---";  cat "${VANILLA_JSON}";  echo
    echo "--- biscotti-baseline (${BASELINE_TARGET}) ---"; cat "${BASELINE_JSON}"; echo
    echo "--- hoisted (${HOISTED_TARGET}) ---";            cat "${HOISTED_JSON}";  echo
  fi
  echo
  echo "=== compare done ==="
else
  run_one "${TARGET_BASE}${VARIANT_SUFFIX}"
  echo
  echo "=== done ==="
fi
