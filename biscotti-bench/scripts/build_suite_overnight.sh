#!/bin/bash
# Overnight suite build (COMPILE ONLY via build.sh).
#
#   - Working set only (see KERNELS below), all three variants each.
#   - 1 hour (3600s) timeout per build.
#   - Per-(suite,variant) lane: process small->large; if a kernel TIMES OUT,
#     skip all LARGER kernels in THAT SAME lane only. Lanes (hoisted /
#     baseline / baseline_vanilla) are independent -- a baseline timeout does
#     NOT skip vanilla or hoisted.
#   - det HOISTED uses --threshold=8 (baselines are unrolled = threshold -1).
#   - mm16x16 is intentionally excluded (all variants).
#
# Launch (background + tee):
#   nohup ./scripts/build_suite_overnight.sh > suite-overnight.out 2>&1 &

set -u
HEIR=/local/scratch/a/paranjav/biscotti/heir
BENCH="$HEIR/biscotti-bench"
export COYOTE_BISCOTTI_DIR="$HEIR/lib/Transforms/CoyoteVectorizer/coyote"
export COYOTE_VANILLA_DIR="$HEIR/lib/Transforms/CoyoteVectorizer/coyote-vanilla"
# Write build/ and output/ artifacts to a dedicated tree so the overnight run
# never clobbers the known-good benchmarks/<suite>/{build,output}. Source is
# still read from the original benchmarks/<suite>/src. (build.sh honors this.)
export BENCH_ARTIFACT_ROOT="$BENCH/suite-build-artifacts"
mkdir -p "$BENCH_ARTIFACT_ROOT"
TIMEOUT=3600

LOGDIR="$BENCH/suite-build-logs-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"
: > "$SUMMARY"

log() { echo "$(date +%H:%M:%S)  $*" | tee -a "$SUMMARY"; }

# suite -> kernels, ordered small -> large (drives the skip logic)
declare -A KERNELS=(
  [argmax]="argmax_7 argmax_8 argmax_9"
  [conv]="conv_4 conv_6 conv_10 conv_18"
  [det]="det_3 det_4 det_5"
  [dot]="dot_48 dot_96 dot_192 dot_384"
  [mm]="mm4x4 mm6x6_3x3 mm8x8"
)
# variant -> build.sh flag
declare -A VARFLAG=( [hoisted]="" [baseline]="--baseline" [baseline_vanilla]="--baseline-vanilla" )

START=$(date +%s)
log "=== suite overnight build START ==="
log "logs:      $LOGDIR"
log "artifacts: $BENCH_ARTIFACT_ROOT  (build/ + output/ redirected here)"

for suite in argmax conv det dot mm; do
  for variant in hoisted baseline baseline_vanilla; do
    flag="${VARFLAG[$variant]}"
    skip_rest=0                       # per-(suite,variant) high-water mark
    for k in ${KERNELS[$suite]}; do
      tag="${suite}/${k}/${variant}"

      if [ "$skip_rest" -eq 1 ]; then
        log "SKIP     $tag  (larger than a timed-out kernel in this lane)"
        continue
      fi

      # det hoisted pins threshold=8; baselines are unrolled (threshold -1) so
      # no extra flag there.
      thr=""
      if [ "$suite" = "det" ] && [ "$variant" = "hoisted" ]; then thr="--threshold=8"; fi

      jlog="$LOGDIR/${suite}_${k}_${variant}.log"
      log "START    $tag"
      t0=$(date +%s)
      timeout "$TIMEOUT" "$BENCH/scripts/build.sh" $flag $thr "$suite" "$k" > "$jlog" 2>&1
      rc=$?
      t1=$(date +%s); dt=$((t1 - t0))

      if [ "$rc" -eq 124 ]; then
        log "TIMEOUT  $tag  (${dt}s >1hr) -> skipping larger kernels in this lane"
        skip_rest=1
      elif [ "$rc" -ne 0 ]; then
        log "FAIL     $tag  rc=$rc  (${dt}s)  [tail: $jlog]"
      else
        log "OK       $tag  (${dt}s)"
      fi
    done
  done
done

END=$(date +%s)
log "=== DONE  total=$(( (END-START)/60 )) min ==="
echo
echo "Summary written to: $SUMMARY"
