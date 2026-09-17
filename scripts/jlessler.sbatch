#!/bin/bash
#SBATCH --job-name=tapestry
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/slurm/%x-%A_%a.log

# One launcher for every experiment, B0 or B1: the dispatcher is model-agnostic
# and reads the model from experiment.json. One whole GPU per array element,
# configurable fitting processes per GPU. All six GPUs draw from one shared
# queue; there are no static task slices.
#
#   sbatch --job-name=B1-onlynowcast --array=0-3 scripts/jlessler.sbatch B1-onlynowcast
#   LANES=4 GPUS=6 sbatch --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch B1-onlymask
set -euo pipefail
experiment=${1:?usage: sbatch --array=0-3 scripts/jlessler.sbatch EXPERIMENT}
shift
lanes=${LANES:-8}
threads=${TAPESTRY_TORCH_THREADS:-1}
if (( SLURM_CPUS_PER_TASK < threads * lanes )); then
    echo 'Allocate at least lanes * threads CPUs with --cpus-per-task.' >&2
    exit 1
fi
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" TAPESTRY_TORCH_THREADS="$threads" PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
test -s "data/experiments/$experiment/code/src/tapestry/models/manager.py"
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
.venv/bin/python -m tapestry.models.dispatch -e "$experiment" --lanes "$lanes" ${GPUS:+--gpu-count "$GPUS"} "$@"
