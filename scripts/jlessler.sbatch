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

# One launcher for every experiment: the dispatcher is scenario-agnostic and reads
# the scenario strings and seeds from the experiment's jobs.csv (written by
# `planner plan`), running the code snapshot in data/experiments/NAME/code. One
# whole GPU per array element, LANES fitting processes per GPU (default 8, one
# CPU thread each). All GPUs draw from one shared queue; no static task slices.
# Extra arguments go to the dispatcher (e.g. --retry-failed, --seeds 42).
#
#   sbatch --job-name=NAME --array=0-3 scripts/jlessler.sbatch NAME
#   LANES=4 GPUS=6 sbatch --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch NAME
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
test -s "data/experiments/$experiment/code/src/tapestry/experiment/planner.py"
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'

# Notifications are on by default: the first array element queues a follow-up job
# that publishes a summary once every allocation of this array has ended, whether
# it finished, timed out or was cancelled. NTFY=0 turns it off; NTFY_URL retargets.
# Only element 0 submits, so an array does not queue one notifier per element.
if [ "${NTFY:-1}" != "0" ] && [ "${SLURM_ARRAY_TASK_ID:-0}" = "${SLURM_ARRAY_TASK_MIN:-0}" ]; then
    array=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
    sbatch --job-name="$experiment-notify" --dependency="afterany:$array" \
        --kill-on-invalid-dep=yes scripts/notify.sbatch "$experiment" "$array" \
        >/dev/null 2>&1 && echo "queued ntfy summary for array $array" || \
        echo 'warning: could not queue the ntfy summary; training continues' >&2
fi

.venv/bin/python -m tapestry.experiment.dispatch -e "$experiment" --lanes "$lanes" ${GPUS:+--gpu-count "$GPUS"} "$@"
