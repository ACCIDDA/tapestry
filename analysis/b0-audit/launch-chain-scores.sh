#!/bin/bash
set -euo pipefail
# Already planned. Dependencies ensure saved source fits and their checks completed.
for stage in 00 01 02 03 03b 04 05 06 07 09; do
  dependency=()
  case "$stage" in
    00) dependency=(--dependency=afterok:2496122);;
    01) dependency=(--dependency=afterok:2496123);;
    02) dependency=(--dependency=afterok:2496124);;
    03|03b) dependency=(--dependency=afterok:2496125);;
    07) dependency=(--dependency=afterok:2495540);;
    08) dependency=(--dependency=afterok:2495541);;
    09) dependency=(--dependency=afterok:2495715);;
  esac
  LANES=3 sbatch --job-name="b1-b0-score-$stage" --array=0-0 \
    --nodelist=g1803jles01 --time=01:00:00 "${dependency[@]}" scripts/jlessler.sbatch "b1-b0-score-$stage"
done
