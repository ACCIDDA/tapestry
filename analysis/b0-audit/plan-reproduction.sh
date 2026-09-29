#!/bin/bash
set -euo pipefail
if compgen -G 'data/experiments/b0-exact-reproduction/*/s*/attempt-*' >/dev/null; then
    echo 'Existing attempts: use status and resume the saved snapshot; do not re-plan b0-exact-reproduction.' >&2
    exit 1
fi
.venv/bin/python -m tapestry.experiment.planner plan -e b0-exact-reproduction \
 --dataset data/processed/build_b_finalized.npz --eval-members 2048 --seeds 42 43 44 --device cuda -s \
 'ed_transform=logit,epochs=100,patience=30,fit_partition=target' \
 'ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen'
.venv/bin/python analysis/b0-audit/pin_reproduction.py
