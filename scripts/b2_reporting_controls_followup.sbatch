#!/bin/bash
#SBATCH --job-name=b2-reporting-controls-followup
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm/%x-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/data/experiments/b2-reporting-controls-20261001/code/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python -m tapestry.experiment.planner status -e b2-reporting-controls-20261001
.venv/bin/python -m tapestry.experiment.planner rank -e b2-reporting-controls-20261001 --no-plots
# Planning copies source plus scripts/notify.py; the isolated checkout has both.
# The inference-only snapshot is sufficient for rank, but cannot snapshot a new plan.
PYTHONPATH="$PWD/src" .venv/bin/python scripts/plan_b2_augmentation_controls.py replay
allocation=$(LANES=8 GPUS=2 sbatch --parsable --job-name=b2-reporting-control-replay-20261001 \
  --array=0-1 --time=04:00:00 --nodelist=g1803jles01 scripts/jlessler.sbatch b2-reporting-control-replay-20261001)
echo "Control replay allocation: $allocation"
sbatch --dependency="afterok:$allocation:3352500" --kill-on-invalid-dep=yes scripts/b2_reporting_controls_report.sbatch
