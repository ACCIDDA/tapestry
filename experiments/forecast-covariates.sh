#!/usr/bin/env bash
# Fresh matched geographic and covariate models. Planning only; repository root.
set -euo pipefail
base='task=forecast,input_mode=finalized_available,lookback=12,encoder=mlp,fit_partition=pathogen,ed_transform=logit,width=64,latent=16,epochs=300,patience=30,supplied_final=1,mask_rate=0.5'
scenarios=()
for spatial in none pooled attention; do
  scenarios+=("$base,spatial=$spatial")
  for sources in inpatient+outpatient ilinet+clinical_lab+flusurv inpatient+outpatient+ww_wval_like+ilinet+clinical_lab+flusurv+kinsa ww_wval_like kinsa; do
    for representation in raw smooth summary shared; do
      scenarios+=("$base,spatial=$spatial,covariate_set=$sources,covariate_encoder=$representation")
    done
  done
done
.venv/bin/python -m tapestry.experiment.planner plan -e forecast-geography-v2 \
  -s "${scenarios[@]}" --seeds 42 43 44 --device cuda
