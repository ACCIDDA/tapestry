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

# One whole GPU per array element, six fitting processes per GPU.
# All six GPUs draw from one shared queue; there are no static task slices.
set -euo pipefail
experiment=${1:-B0.1}
lanes=${LANES:-6}
if (( SLURM_CPUS_PER_TASK < 2 * lanes )); then
    echo 'Allocate at least two CPUs per lane with --cpus-per-task.' >&2
    exit 1
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m tapestry.models.dispatch -e "$experiment" --lanes "$lanes"
