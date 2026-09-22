#!/bin/bash
#SBATCH --job-name=B1-decisive-report
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=output/slurm/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TAPESTRY_TORCH_THREADS=2 PYTHONUNBUFFERED=1
selection=()
report=data/experiments/B1-decisive-report
if [ "$#" -gt 0 ]; then
    selection=(--seeds "$1")
    report="data/experiments/B1-seed$1-report"
fi
export PYTHONPATH="$PWD/$report/code/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "${NTFY:-1}" != "0" ]; then
    sbatch --job-name=B1-report-notify --dependency="afterany:$SLURM_JOB_ID" \
        scripts/notify.sbatch B1-onlymask-refit "$SLURM_JOB_ID"
fi
for experiment in B1-onlymask-refit B1-direct-finalflag B1-joint-aux025; do
    .venv/bin/python -m tapestry.models.manager status -e "$experiment" "${selection[@]}"
    .venv/bin/python -m tapestry.models.manager rank -e "$experiment" "${selection[@]}"
done
.venv/bin/python -m tapestry.models.manager decisive -e B1-onlymask-refit --device cuda --report-output "$report" "${selection[@]}"
