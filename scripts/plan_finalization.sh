#!/bin/bash
# One formulation, evaluated under recent rolling and forward-season protocols.
set -euo pipefail
experiment=${1:-reporting-triangle-v11-20261001}
base='task=finalize,input_mode=vintaged,lookback=12,epochs=200,finalization_weeks=8'
.venv/bin/python -m tapestry.experiment.planner plan -e "$experiment" \
    -s "$base" "$base,finalization_cv=season,evaluation_seasons=recent_two" --seeds 42 --device cpu
