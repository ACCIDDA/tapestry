#!/bin/bash
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm/%x-%j.log
# CPU nowcaster; same planner/scorer and pinned source as other experiments.
set -euo pipefail
experiment=${1:?experiment required}
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TAPESTRY_TORCH_THREADS=4 PYTHONUNBUFFERED=1
if [ "${NTFY:-1}" != "0" ]; then
    sbatch --job-name="$experiment-notify" --dependency="afterany:$SLURM_JOB_ID" \
        scripts/notify.sbatch "$experiment" "$SLURM_JOB_ID" || true
fi
.venv/bin/python -m tapestry.experiment.planner run -e "$experiment" --fit-workers 2
.venv/bin/python -m tapestry.experiment.planner rank -e "$experiment"
