#!/bin/bash
#SBATCH --job-name=tapestry-cpu
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=output/slurm/%x-%j.log
# Short manager experiments on spare CPUs while patron GPUs are occupied.
set -euo pipefail
experiment=${1:?usage: sbatch scripts/cpu.sbatch EXPERIMENT}
workers=${FIT_WORKERS:-3}
threads=${TAPESTRY_TORCH_THREADS:-4}
if (( SLURM_CPUS_PER_TASK < workers * threads )); then
    echo 'Allocate at least workers * threads CPUs.' >&2
    exit 1
fi
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
export TAPESTRY_TORCH_THREADS="$threads" PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "${NTFY:-1}" != "0" ]; then
    sbatch --job-name="$experiment-notify" --dependency="afterany:$SLURM_JOB_ID" \
        scripts/notify.sbatch "$experiment" "$SLURM_JOB_ID"
fi
.venv/bin/python -m tapestry.models.manager run -e "$experiment" --device cpu --fit-workers "$workers" --keep-going
.venv/bin/python -m tapestry.models.manager rank -e "$experiment"
