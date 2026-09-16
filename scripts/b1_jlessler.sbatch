#!/bin/bash
#SBATCH --job-name=B1
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/slurm/B1-%A_%a.log

# Same shared dispatcher as B0.1: one allocation per GPU, multiple lanes inside.
# Submit --array=0-3 on jles01; optionally --array=0-1 --nodelist=g1803jles02.
# Start with LANES=4; use LANES=6 for six processes per GPU.
# This range is user-selected; B1 throughput/memory have not yet been measured.
set -euo pipefail
experiment=${1:?usage: sbatch --array=0-3 scripts/b1_jlessler.sbatch EXPERIMENT}
shift
lanes=${LANES:-4}
threads=${TAPESTRY_TORCH_THREADS:-1}
if (( SLURM_CPUS_PER_TASK < lanes * threads )); then
    echo 'Allocate at least lanes * threads CPUs.' >&2
    exit 1
fi
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" TAPESTRY_TORCH_THREADS="$threads" PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
test -s "data/experiments/$experiment/code/src/tapestry/models/b1_experiment.py"
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
.venv/bin/python -m tapestry.models.dispatch -e "$experiment" --lanes "$lanes" --gpu-count "${GPUS:-4}" "$@"
