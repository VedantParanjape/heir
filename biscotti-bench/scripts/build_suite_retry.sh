#!/bin/bash
# Retry the builds that did NOT pass in the most recent suite-overnight run.
#
# Reads the latest suite-build-logs-*/summary.txt, collects every
# FAIL / TIMEOUT / SKIP (suite, kernel, variant), and rebuilds just those via
# build.sh -- same knobs as build_suite_overnight.sh (1 hr cap, all three
# variants as they appear, det hoisted @ threshold=8, artifacts redirected to
# suite-build-artifacts/). Fresh log dir per retry.
#
# Point it at a specific prior summary with:  SRC_SUMMARY=/path/to/summary.txt
#
# Launch:
#   nohup ./scripts/build_suite_retry.sh > suite-retry.out 2>&1 &

set -u
HEIR=/local/scratch/a/paranjav/biscotti/heir
BENCH="$HEIR/biscotti-bench"
export COYOTE_BISCOTTI_DIR="$HEIR/lib/Transforms/CoyoteVectorizer/coyote"
export COYOTE_VANILLA_DIR="$HEIR/lib/Transforms/CoyoteVectorizer/coyote-vanilla"
export BENCH_ARTIFACT_ROOT="$BENCH/suite-build-artifacts"
TIMEOUT=3600
# DRY_RUN=1 -> only print the jobs + the exact build.sh commands, build nothing.
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == "1" ]] || mkdir -p "$BENCH_ARTIFACT_ROOT"

# Which prior summary to read the failures from (default: most recent).
SRC_SUMMARY="${SRC_SUMMARY:-$(ls -dt "$BENCH"/suite-build-logs-*/summary.txt 2>/dev/null | head -1)}"
if [[ -z "${SRC_SUMMARY:-}" || ! -f "$SRC_SUMMARY" ]]; then
  echo "error: no prior summary.txt found (set SRC_SUMMARY=...)" >&2
  exit 1
fi

LOGDIR="$BENCH/suite-retry-logs-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"
: > "$SUMMARY"
log() { echo "$(date +%H:%M:%S)  $*" | tee -a "$SUMMARY"; }

declare -A VARFLAG=( [hoisted]="" [baseline]="--baseline" [baseline_vanilla]="--baseline-vanilla" )

log "=== suite RETRY start ==="
log "source summary: $SRC_SUMMARY"
log "logs:           $LOGDIR"
log "artifacts:      $BENCH_ARTIFACT_ROOT"

# Collect the non-OK jobs from the source summary as "suite/kernel/variant".
mapfile -t JOBS < <(grep -E "^[0-9:]+  (FAIL|TIMEOUT|SKIP) " "$SRC_SUMMARY" \
                    | sed -E 's#^[0-9:]+  (FAIL|TIMEOUT|SKIP) +##; s# .*##' | sort -u)

if [[ ${#JOBS[@]} -eq 0 ]]; then log "nothing to retry (no FAIL/TIMEOUT/SKIP)"; exit 0; fi
log "retrying ${#JOBS[@]} job(s): ${JOBS[*]}"

START=$(date +%s)
for job in "${JOBS[@]}"; do
  IFS=/ read -r suite kernel variant <<< "$job"
  flag="${VARFLAG[$variant]:-}"
  # det hoisted pins threshold=8; baselines are unrolled (threshold -1).
  thr=""
  if [[ "$suite" == "det" && "$variant" == "hoisted" ]]; then thr="--threshold=8"; fi

  jlog="$LOGDIR/${suite}_${kernel}_${variant}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "WOULD BUILD  $job  ->  timeout ${TIMEOUT} build.sh ${flag} ${thr} ${suite} ${kernel}"
    continue
  fi
  log "START    $job"
  t0=$(date +%s)
  timeout "$TIMEOUT" "$BENCH/scripts/build.sh" $flag $thr "$suite" "$kernel" > "$jlog" 2>&1
  rc=$?
  t1=$(date +%s); dt=$((t1 - t0))
  if [[ $rc -eq 124 ]]; then log "TIMEOUT  $job  (${dt}s >1hr)"
  elif [[ $rc -ne 0 ]]; then log "FAIL     $job  rc=$rc  (${dt}s)  [tail: $jlog]"
  else log "OK       $job  (${dt}s)"; fi
done
log "=== DONE  total=$(( ($(date +%s)-START)/60 )) min ==="
echo; echo "Summary: $SUMMARY"
