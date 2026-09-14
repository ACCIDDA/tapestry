#!/bin/bash
#SBATCH --job-name=b0-distributed
#SBATCH --partition=jlessler
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/proj/jlessler/projects/tapestry-all/tapestry
#SBATCH --output=/proj/jlessler/projects/tapestry-all/tapestry/output/slurm/b0-distributed-%A_%a.log

# Submit workers with --gres=gpu:1 --array=0-N%6. Collector uses this same
# script with no GPUs and --dependency=afterany:ARRAY_JOB_ID ... collect.
set -euo pipefail
source /usr/share/lmod/lmod/init/bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
hostname
if [[ "${1:-worker}" == collect ]]; then
    .venv/bin/python scripts/distribute_b0.py collect
else
    .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
    .venv/bin/python scripts/distribute_b0.py worker --task "$SLURM_ARRAY_TASK_ID"
fi
