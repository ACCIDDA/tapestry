#!/bin/bash
set -euo pipefail
name=b0-normalization-audit
.venv/bin/python -m tapestry.experiment.planner plan -e "$name" --seeds 42 43 44 --device cuda --eval-members 256 -s \
'ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen,supplied_final=1,mask_rate=0.5,input_mode=finalized_available' \
'ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen,supplied_final=1,mask_rate=0.5,covariate_encoder=summary,covariate_set=inpatient+outpatient+ww_wval_like+kinsa+ilinet+clinical_lab+flusurv,input_mode=finalized_available'
.venv/bin/python analysis/b0-audit/pin_normalization.py "$name"
