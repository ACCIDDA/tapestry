#!/bin/bash
#SBATCH --job-name=b0-array
#SBATCH --partition=a100-gpu,l40-gpu,volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=output/slurm/b0-array-%A_%a.log

# One array task = one jobs.csv row: a scenario and all of its seeds, in sequence.
# Submit from the repository root with the tasks printed by `manager status`:
#   sbatch --array=0-13%6 scripts/b0_array.sbatch EXPERIMENT [extra manager options]
set -euo pipefail
experiment=${1:?usage: sbatch --array=TASKS scripts/b0_array.sbatch EXPERIMENT}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
hostname
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
.venv/bin/python -m tapestry.models.manager run -e "$experiment" --task "$SLURM_ARRAY_TASK_ID" \
    --device cuda --keep-going "${@:2}"
