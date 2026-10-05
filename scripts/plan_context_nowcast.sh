#!/bin/bash
set -euo pipefail
experiment=${1:-context-nowcast-v1-20261001}
base='task=finalize,input_mode=vintaged,lookback=12,finalization_weeks=8,finalization_scope=targets,finalization_gap=proxy,finalization_statistic=median,finalization_quantize=1,finalization_cv=season,evaluation_seasons=recent_two'
scenarios=()
if [[ ${2:-kernel} == balanced ]]; then
    models=('adaptive_chain' 'context_residual,finalization_penalty=1000,finalization_features=age,finalization_gate=admissions,finalization_strength=0.25')
elif [[ ${2:-kernel} == point ]]; then
    models=('adaptive_chain' 'context_residual,finalization_penalty=1000,finalization_features=age,finalization_gate=admissions')
elif [[ ${2:-kernel} == strength ]]; then
    models=('adaptive_chain' 'context_residual,finalization_penalty=1000,finalization_features=age,finalization_gate=admissions' 'context_residual,finalization_penalty=1000,finalization_features=age,finalization_gate=admissions,finalization_strength=0.5' 'context_residual,finalization_penalty=1000,finalization_features=age,finalization_gate=admissions,finalization_strength=0.25')
elif [[ ${2:-kernel} == age ]]; then
    models=('adaptive_chain' 'context_residual,finalization_penalty=1000,finalization_features=age' 'context_residual,finalization_penalty=100,finalization_features=age' 'context_residual,finalization_penalty=1000,finalization_features=age_momentum' 'context_residual,finalization_penalty=1000,finalization_features=age,finalization_gate=admissions')
elif [[ ${2:-kernel} == momentum ]]; then
    models=('adaptive_chain' 'context_residual,finalization_penalty=1000,finalization_gate=causal' 'context_residual,finalization_penalty=1000,finalization_features=momentum' 'context_residual,finalization_penalty=1000,finalization_features=momentum,finalization_gate=causal' 'context_residual,finalization_penalty=1000,finalization_gate=admissions')
elif [[ ${2:-kernel} == trajectory ]]; then
    models=('adaptive_chain' 'context_residual,finalization_penalty=1000' 'context_residual,finalization_penalty=1000,finalization_growth_weight=4' 'context_residual,finalization_penalty=1000,finalization_growth_weight=4,finalization_residual_halflife=8')
elif [[ ${2:-kernel} == residual ]]; then
    models=('adaptive_chain' 'context_residual,finalization_penalty=10' 'context_residual,finalization_penalty=100' 'context_residual,finalization_penalty=1000')
else
    models=('adaptive_chain' 'conditional_chain' 'conditional_chain,finalization_growth=0.1' 'conditional_chain,finalization_growth=0.4' 'conditional')
fi
for setting in "${models[@]}"; do
    scenarios+=("$base,finalization_model=$setting")
done
.venv/bin/python -m tapestry.experiment.planner plan -e "$experiment" -s "${scenarios[@]}" --seeds 42 --device cpu
