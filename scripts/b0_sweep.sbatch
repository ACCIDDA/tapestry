#!/bin/bash
#SBATCH --job-name=b0-sweep
#SBATCH --partition=a100-gpu,l40-gpu,jlessler
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=output/slurm/b0-sweep-%A_%a.log

# Shared Longleaf GPU partitions. One array task = one jobs.csv row: a scenario with all
# of its seeds, each fitting three season folds and writing totals.csv. Slurm caps array
# indices, so submit large experiments in chunks of 1,000 with OFFSET, from the repository root:
#   sbatch --array=0-999 --export=ALL,OFFSET=0 scripts/b0_sweep.sbatch EXPERIMENT
#   sbatch --array=0-999 --export=ALL,OFFSET=1000 scripts/b0_sweep.sbatch EXPERIMENT
# `manager status` prints these commands for the tasks that remain.
set -euo pipefail
experiment=${1:?usage: sbatch --array=TASKS --export=ALL,OFFSET=N scripts/b0_sweep.sbatch EXPERIMENT}
task=$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
hostname
echo "Experiment ${experiment}, jobs.csv task ${task}"
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
.venv/bin/python -m tapestry.models.manager run -e "$experiment" --task "$task" \
    --device cuda --keep-going "${@:2}"
