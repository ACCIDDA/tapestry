#!/bin/bash
set -euo pipefail
experiment=${1:-seasonal-nowcast-v1-20261001}
base='task=finalize,input_mode=vintaged,lookback=12,finalization_weeks=8'
scenarios=()
stage=${2:-curves}
if [[ "$stage" == selected ]]; then
    base="$base,finalization_scope=targets,finalization_gap=proxy,finalization_statistic=median,finalization_quantize=1"
    models=('adaptive_chain')
elif [[ "$stage" == robust ]]; then
    base="$base,finalization_scope=targets,finalization_gap=proxy,finalization_statistic=median"
    models=('adaptive_chain' 'adaptive_chain,finalization_halflife=2,finalization_pool=1')
elif [[ "$stage" == tuning ]]; then
    base="$base,finalization_scope=targets,finalization_gap=proxy"
    models=('adaptive_chain' 'adaptive_chain,finalization_halflife=2,finalization_pool=1' 'adaptive_chain,finalization_halflife=4,finalization_pool=1' 'adaptive_chain,finalization_pool=0')
elif [[ "$stage" == gaps ]]; then
    models=('adaptive_chain,finalization_gap=seasonal' 'adaptive_chain,finalization_gap=trend')
else
    models=(seasonal adaptive adaptive_chain)
fi
for model in "${models[@]}"; do
    scenarios+=("$base,finalization_model=$model,finalization_cv=season,evaluation_seasons=recent_two")
    scenarios+=("$base,finalization_model=$model")
done
.venv/bin/python -m tapestry.experiment.planner plan -e "$experiment" -s "${scenarios[@]}" --seeds 42 --device cpu
