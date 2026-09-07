#!/bin/bash
# Register the "_baseline_vanilla" cc_binary variant in benchmark BUILD files
# so run.sh --compare / --baseline-vanilla have a target to build.
#
# Each suite's BUILD generates one cc_binary per (kernel, variant) via a
#   for variant in ["", "_baseline"]
# list comprehension. This script appends "_baseline_vanilla" to that list.
# It is:
#   * format-agnostic — handles single-line and multi-line variant lists,
#   * idempotent       — re-running never adds the variant twice,
#   * scoped           — only touches the `for variant in [...]` list, not the
#                        kernel list above it or the "_baseline" mention in
#                        comments.
#
# NOTE: adding the target is not enough — you must also have emitted the
# vanilla output for each kernel you intend to build:
#     ./scripts/build.sh --baseline-vanilla <suite> <kernel>
# otherwise the target's srcs (output/<kernel>_baseline_vanilla/*) won't exist.
#
# Usage (run from the dir that holds benchmarks/, e.g. biscotti-bench/):
#   ./enable-vanilla-target.sh                 # all suites
#   ./enable-vanilla-target.sh benchmarks/mm/BUILD   # one suite
#
# Tip: run it under a clean git tree so you can `git diff` / revert.

set -euo pipefail

FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
  # Default: every suite BUILD under ./benchmarks/
  mapfile -t FILES < <(find benchmarks -maxdepth 2 -name BUILD)
fi

PERL_EXPR='s/(for variant in \[[^\]]*?"_baseline")(?![^\]]*"_baseline_vanilla")/$1, "_baseline_vanilla"/s'

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "skip (not a file): $f" >&2; continue; }
  before=$(grep -c '_baseline_vanilla' "$f" || true)
  perl -0777 -pi -e "$PERL_EXPR" "$f"
  after=$(grep -c '_baseline_vanilla' "$f" || true)
  if grep -q 'for variant in' "$f"; then
    if [[ "$before" == "$after" ]]; then
      echo "unchanged (already has vanilla, or no variant list): $f"
    else
      echo "patched: $f"
    fi
  fi
done

echo "done."
