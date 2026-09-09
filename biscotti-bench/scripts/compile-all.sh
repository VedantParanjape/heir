#!/bin/bash
# Master compile driver. Builds every configured (suite, kernel) through the
# three configs -- hoisted / biscotti-baseline / vanilla-baseline -- by calling
# build.sh. Continues past failures and prints a pass/fail summary at the end,
# so you can start it and walk away.
#
# REQUIRES bash 4+ (associative arrays). Run it as `./compile_all.sh` or
# `bash compile_all.sh`, NOT `sh compile_all.sh`.
#
# ---- Run it detached so an SSH drop doesn't kill it ----
#   screen -S compile          # start a screen, run ./compile_all.sh, detach C-a d
#   screen -r compile          # reattach later
# or, no screen:
#   nohup ./compile_all.sh > compile_all.out 2>&1 &
#   tail -f compile_all.out
#
# ---- Configure what to compile ----
# Edit the KERNELS map below. Size classes are encoded in the kernel names
# (mm4x4, mm6x6_3x3, dot_96, argmax_16, ...); list the ones you want per suite.
# Edit VARIANTS to change which configs are built.
#
# ---- Env overrides ----
#   BUILD_SH=/path/to/build.sh    (default: sibling build.sh)
#   BAZEL_MODE=opt|dbg            (default: opt; forwarded to build.sh)
#   COYOTE_BISCOTTI_DIR / COYOTE_VANILLA_DIR  (needed by build.sh; export first)

set -uo pipefail   # deliberately NOT -e: one bad kernel must not abort the run

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SH="${BUILD_SH:-${SCRIPT_DIR}/build.sh}"
HEIR_ROOT="${HEIR_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BAZEL_MODE="${BAZEL_MODE:-opt}"
export BAZEL_MODE   # build.sh reads this
TIMEOUT="${TIMEOUT:-3600}"   # per-compile wall cap in seconds (default 60 min).
                             # Monolithic baselines (full unroll) can blow up in
                             # Coyote; the cap bounds them and records a TIMEOUT.

# ============================================================================
# EDIT ME: kernels (size classes) to compile per suite.
# ============================================================================
declare -A KERNELS=(
  # Big hoisted kernels that need >30 min — 1 h cap. Baselines excluded (their
  # monolithic Coyote is hopeless even at 1 h).
  [conv]="conv_34"
  [mm]="mm16x16"
  # [argmax]="argmax_5 argmax_6 argmax_7 argmax_8"
  # [dot]="dot_48 dot_96 dot_192 dot_384"
)

# Variants to build per kernel. "" = hoisted (default build.sh); others are
# build.sh flags. DEFAULT_VARIANTS applies unless the suite is overridden in
# SUITE_VARIANTS (use "@" there for the hoisted/empty variant).
DEFAULT_VARIANTS=("" "--baseline" "--baseline-vanilla")
# Per-suite variant overrides (use "@" for the hoisted/empty variant). Empty =
# every suite uses DEFAULT_VARIANTS.
declare -A SUITE_VARIANTS=(
  [conv]="@"   # conv_34: hoisted only
  [mm]="@"     # mm16x16: hoisted only (monolithic baselines hopeless even at 1 h)
)

label_for() {
  case "$1" in
    "")                   echo "hoisted" ;;
    "--baseline")         echo "biscotti-baseline" ;;
    "--baseline-vanilla") echo "vanilla-baseline" ;;
    *)                    echo "$1" ;;
  esac
}

if [[ ! -x "$BUILD_SH" && ! -f "$BUILD_SH" ]]; then
  echo "error: build.sh not found at $BUILD_SH (set BUILD_SH=...)" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
MASTER_LOG="${SCRIPT_DIR}/compile_all-${STAMP}.log"
declare -a RESULTS=()

echo "master log: $MASTER_LOG"
echo "build.sh:   $BUILD_SH"
echo "bazel mode: $BAZEL_MODE"
echo

# Pre-build the heir tools once in release mode, so the first kernel doesn't
# eat a long tool build mid-log and any toolchain error surfaces immediately.
echo "=== pre-building heir tools (-c ${BAZEL_MODE}) ==="
if ! ( cd "$HEIR_ROOT" && bazel build -c "$BAZEL_MODE" //tools:heir-opt //tools:heir-translate ) 2>&1 | tee -a "$MASTER_LOG"; then
  echo "error: heir tools failed to build; aborting before benchmarks." >&2
  exit 1
fi
echo

for suite in "${!KERNELS[@]}"; do
  # Baseline variants that have already timed out in this suite. Kernels are
  # listed in ascending size and monolithic Coyote cost only grows, so once a
  # baseline blows up at some size, skip that variant for all larger kernels
  # instead of burning the full cap on each. (Hoisted is never skipped.)
  unset timedOutVariant; declare -A timedOutVariant
  if [[ -n "${SUITE_VARIANTS[$suite]:-}" ]]; then
    read -ra suiteVariants <<< "${SUITE_VARIANTS[$suite]}"
  else
    suiteVariants=("${DEFAULT_VARIANTS[@]}")
  fi
  for kernel in ${KERNELS[$suite]}; do
    for variant in "${suiteVariants[@]}"; do
      [[ "$variant" == "@" ]] && variant=""   # "@" sentinel = hoisted (empty)
      lbl="$(label_for "$variant")"
      tag="${suite}/${kernel} [${lbl}]"
      if [[ -n "$variant" && -n "${timedOutVariant[$variant]:-}" ]]; then
        echo "    SKIP:    ${tag}  (${lbl} already timed out at a smaller ${suite} size)"
        RESULTS+=("SKIP    ${tag}")
        continue
      fi
      echo "==================================================================="
      echo ">>> building ${tag}  (cap ${TIMEOUT}s)"
      echo "==================================================================="
      # $variant is intentionally unquoted so "" expands to no argument.
      # -k 30: if it ignores TERM, follow up with KILL 30s later.
      rc=0
      timeout -k 30 "${TIMEOUT}" "$BUILD_SH" $variant "$suite" "$kernel" >>"$MASTER_LOG" 2>&1 || rc=$?
      if [[ $rc -eq 0 ]]; then
        echo "    OK:      ${tag}"
        RESULTS+=("OK      ${tag}")
      elif [[ $rc -eq 124 || $rc -eq 137 ]]; then
        # Timed out. bazel run detaches heir-opt/coyote, so reap orphans by name.
        pkill -f "run_coyote_from_circuit.py" 2>/dev/null || true
        pkill -f "tools/heir-opt.*${kernel}" 2>/dev/null || true
        echo "    TIMEOUT: ${tag}  (>${TIMEOUT}s — likely monolithic Coyote blowup)"
        RESULTS+=("TIMEOUT ${tag}")
        # Larger kernels of this baseline variant will only be worse — skip them.
        [[ -n "$variant" ]] && timedOutVariant["$variant"]=1
      else
        echo "    FAIL:    ${tag}  (rc=${rc}; see $MASTER_LOG)"
        RESULTS+=("FAIL    ${tag}")
      fi
    done
  done
done

echo
echo "================= SUMMARY ================="
printf '%s\n' "${RESULTS[@]}"
echo "-------------------------------------------"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL' || true)
timeouts=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^TIMEOUT' || true)
skipped=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^SKIP' || true)
echo "${#RESULTS[@]} builds, ${fails} failed, ${timeouts} timed out, ${skipped} skipped. Full log: $MASTER_LOG"
echo "==========================================="
exit $(( fails > 0 ? 1 : 0 ))
