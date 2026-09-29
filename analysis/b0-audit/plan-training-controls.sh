#!/bin/bash
set -euo pipefail
if compgen -G 'data/experiments/b0-current-training-control/*/s*/attempt-*' >/dev/null; then
    echo 'Existing attempts: use status and resume the saved snapshot; do not re-plan b0-current-training-control.' >&2
    exit 1
fi
if compgen -G 'data/experiments/b0-current-training-normalized/*/s*/attempt-*' >/dev/null; then
    echo 'Existing attempts: use status and resume the saved snapshot; do not re-plan b0-current-training-normalized.' >&2
    exit 1
fi
.venv/bin/python analysis/b0-audit/convert_original_panel.py
.venv/bin/python -m tapestry.experiment.planner plan -e b0-current-training-control \
 --dataset data/audits/b0/original-panel-unified.npz --eval-members 2048 --seeds 42 43 44 --device cuda -s \
 'ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen'
.venv/bin/python -m tapestry.experiment.planner plan -e b0-current-training-normalized \
 --dataset data/audits/b0/original-panel-unified.npz --eval-members 2048 --seeds 42 43 44 --device cuda -s \
 'ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen' \
 'ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen,input_mode=finalized_available'
.venv/bin/python analysis/b0-audit/pin_normalization.py b0-current-training-normalized
