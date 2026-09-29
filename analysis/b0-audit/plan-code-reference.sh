#!/bin/bash
set -euo pipefail
if compgen -G 'data/experiments/b0-training-code-reference/*/s*/attempt-*' >/dev/null; then
    echo 'Existing attempts: use status and resume the saved snapshot.' >&2
    exit 1
fi
.venv/bin/python -m tapestry.experiment.planner plan -e b0-training-code-reference \
 --dataset data/processed/build_b_finalized.npz --eval-members 2048 --seeds 42 43 --device cuda -s \
 'ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen'
.venv/bin/python analysis/b0-audit/pin_reproduction.py b0-training-code-reference
