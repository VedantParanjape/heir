#!/bin/bash
# Build (bazel) + run one benchmark. The suite+kernel form derives the
# bazel target as //biscotti-bench/benchmarks/<suite>:bench_<kernel>.
# Everything after `--` is forwarded to the benchmark binary
# (google_benchmark flags).
#
# Usage:
#   ./scripts/run.sh <suite> <kernel> [-- <benchmark args>]
#   ./scripts/run.sh --compare <suite> <kernel> [-- <benchmark args>]
#
# --compare runs both the hoisted target (bench_<kernel>) and the
# baseline target (bench_<kernel>_baseline) back-to-back. Both must
# have already been emitted via `build.sh <suite> <kernel>` and
# `build.sh --baseline <suite> <kernel>` respectively.
#
# Examples:
#   ./scripts/run.sh mm mm4x4
#   ./scripts/run.sh mm mm4x4 -- --benchmark_repetitions=5
#   ./scripts/run.sh --compare mm mm4x4 -- --benchmark_repetitions=5
#
# Optional env vars:
#   HEIR_ROOT     - HEIR checkout root (default: <script dir>/../..)
#   BAZEL_MODE    - compilation mode passed to bazel (default: opt)
#   LOG_FILE      - if set, tee output here as well as stdout
#                   (single-target mode only)
#   COMPARE_PY    - path to google_benchmark's tools/compare.py.
#                   If found (via env or bazel's external cache),
#                   --compare invokes it on the two JSON outputs for a
#                   side-by-side stat diff. Fallback: print both runs
#                   sequentially.

set -euo pipefail

COMPARE=0
if [[ "${1:-}" == "--compare" ]]; then
  COMPARE=1
  shift
fi

if [[ $# -lt 2 ]]; then
  echo "usage: $0 [--compare] <suite> <kernel> [-- <benchmark args>]" >&2
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
BAZEL_MODE="${BAZEL_MODE:-dbg}"

TARGET_BASE="//biscotti-bench/benchmarks/${SUITE}:bench_${KERNEL}"

cd "$HEIR_ROOT"

run_one() {
  local target="$1"
  echo
  echo "=== bazel build (-c ${BAZEL_MODE}) ${target} ==="
  bazel build "-c" "${BAZEL_MODE}" "${target}"
  echo
  echo "=== running ${target} ==="
  if [[ -n "${LOG_FILE:-}" && $COMPARE -eq 0 ]]; then
    bazel run "-c" "${BAZEL_MODE}" "${target}" -- "${BENCH_ARGS[@]}" | tee "${LOG_FILE}"
  else
    bazel run "-c" "${BAZEL_MODE}" "${target}" -- "${BENCH_ARGS[@]}"
  fi
}

if [[ $COMPARE -eq 1 ]]; then
  # Locate google_benchmark's compare.py. Env override first, then the
  # bazel external-fetch cache. If neither, fall back to sequential
  # console output.
  if [[ -z "${COMPARE_PY:-}" ]]; then
    COMPARE_PY="$(find "${HOME}/.cache/bazel" -path '*google_benchmark*/tools/compare.py' -print -quit 2>/dev/null || true)"
  fi

  BASELINE_TARGET="${TARGET_BASE}_baseline"
  HOISTED_TARGET="${TARGET_BASE}"
  BASELINE_JSON=$(mktemp --suffix=.json)
  HOISTED_JSON=$(mktemp --suffix=.json)
  # shellcheck disable=SC2064
  trap "rm -f '${BASELINE_JSON}' '${HOISTED_JSON}'" EXIT

  echo
  echo "=== bazel build (-c ${BAZEL_MODE}) ${BASELINE_TARGET} ==="
  bazel build "-c" "${BAZEL_MODE}" "${BASELINE_TARGET}"
  echo "=== bazel build (-c ${BAZEL_MODE}) ${HOISTED_TARGET} ==="
  bazel build "-c" "${BAZEL_MODE}" "${HOISTED_TARGET}"

  # If the caller passed --benchmark_repetitions but not
  # --benchmark_report_aggregates_only, auto-add the latter so compare.py
  # gets a clean mean/median/stddev-only table.
  EXTRA_ARGS=()
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

  echo
  echo "=== running ${BASELINE_TARGET} → ${BASELINE_JSON} ==="
  bazel run "-c" "${BAZEL_MODE}" "${BASELINE_TARGET}" -- \
      --benchmark_format=json --benchmark_out="${BASELINE_JSON}" \
      "${BENCH_ARGS[@]}" "${EXTRA_ARGS[@]}"

  echo
  echo "=== running ${HOISTED_TARGET} → ${HOISTED_JSON} ==="
  bazel run "-c" "${BAZEL_MODE}" "${HOISTED_TARGET}" -- \
      --benchmark_format=json --benchmark_out="${HOISTED_JSON}" \
      "${BENCH_ARGS[@]}" "${EXTRA_ARGS[@]}"

  echo
  if [[ -n "${COMPARE_PY:-}" && -f "$COMPARE_PY" ]]; then
    echo "=== compare (${COMPARE_PY}) ==="
    python3 "$COMPARE_PY" benchmarks "$BASELINE_JSON" "$HOISTED_JSON"
  else
    echo "=== compare.py not found — dumping raw JSON outputs ==="
    echo "--- baseline (${BASELINE_TARGET}) ---"
    cat "$BASELINE_JSON"
    echo
    echo "--- hoisted (${HOISTED_TARGET}) ---"
    cat "$HOISTED_JSON"
  fi
  echo
  echo "=== compare done ==="
else
  run_one "${TARGET_BASE}"
  echo
  echo "=== done ==="
fi
