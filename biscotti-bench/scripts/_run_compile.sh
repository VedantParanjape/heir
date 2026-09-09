#!/bin/bash
export COYOTE_BISCOTTI_DIR=/local/scratch/a/paranjav/biscotti/heir/lib/Transforms/CoyoteVectorizer/coyote
export COYOTE_VANILLA_DIR=/local/scratch/a/paranjav/biscotti/heir/lib/Transforms/CoyoteVectorizer/coyote-vanilla
# tee the driver's stdout (>>> building / OK / TIMEOUT / SUMMARY) so the tally
# survives the screen closing.
bash /local/scratch/a/paranjav/biscotti/heir/biscotti-bench/scripts/compile-all.sh 2>&1 \
  | tee /local/scratch/a/paranjav/biscotti/heir/biscotti-bench/scripts/compile-all-run.out
