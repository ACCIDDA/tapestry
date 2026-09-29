#!/bin/bash
set -euo pipefail
# A two-factor comparison. Reuse the completed normalized/full-finalized baseline.
# Do not modify or re-plan any existing experiment.
for arm in split draws split-draws; do
    if [ -e "data/experiments/b0-fitting-$arm" ]; then
        echo "b0-fitting-$arm already exists: inspect status; do not re-plan." >&2
        exit 1
    fi
done
for arm in split draws split-draws; do
    .venv/bin/python -m tapestry.experiment.planner plan -e "b0-fitting-$arm" \
      --dataset data/audits/b0/original-panel-unified.npz --eval-members 2048 \
      --seeds 42 43 44 --device cuda -s \
      'ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen'
    .venv/bin/python analysis/b0-audit/pin_fitting_controls.py "$arm"
done
