#!/bin/bash
set -euo pipefail
# All stages use the original finalized panel, same seed set and evaluation draws.
# Stage 04 (complete finalized history), 05 (B0 weeks), 06 (B0 draws) reuse
# completed identical settings only after scientific preflight verifies equivalence.
stages=("${@:-}")
if [ "$#" -eq 0 ]; then stages=(00 01 02 03 07 08 09); fi
for stage in "${stages[@]}"; do
  test ! -e "data/experiments/b1-b0-chain-$stage" || { echo "Existing stage $stage: use status, never re-plan" >&2; exit 1; }
done
for stage in "${stages[@]}"; do
  dataset=data/audits/b0/original-panel-deadline2025.npz
  if [[ "$stage" == 07 || "$stage" == 08 || "$stage" == 09 ]]; then dataset=data/audits/b0/original-panel-unified.npz; fi
  scenario='ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen'
  if [[ "$stage" == 00 || "$stage" == 01 ]]; then
    scenario+=',input_mode=finalized_available,supplied_final=1,mask_rate=0.5'
  elif [[ "$stage" == 02 ]]; then
    scenario+=',input_mode=finalized_available,supplied_final=1'
  elif [[ "$stage" == 03 ]]; then
    scenario+=',input_mode=finalized_available'
  fi
  .venv/bin/python -m tapestry.experiment.planner plan \
    -e "b1-b0-chain-$stage" -s "$scenario" --seeds 42 43 44 --device cuda \
    --dataset "$dataset" --eval-members 2048
  .venv/bin/python analysis/b0-audit/pin_chain.py "$stage"
done
