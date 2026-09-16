#!/bin/bash
#SBATCH --job-name=B0.1
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/slurm/B0.1-%A_%a.log

# One whole GPU per array element, configurable fitting processes per GPU.
# All six GPUs draw from one shared queue; there are no static task slices.
set -euo pipefail
experiment=${1:-B0.1}
lanes=${LANES:-8}
threads=${TAPESTRY_TORCH_THREADS:-1}
if (( SLURM_CPUS_PER_TASK < threads * lanes )); then
    echo 'Allocate at least lanes * threads CPUs with --cpus-per-task.' >&2
    exit 1
fi
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" TAPESTRY_TORCH_THREADS="$threads" PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m tapestry.models.dispatch -e "$experiment" --lanes "$lanes"
