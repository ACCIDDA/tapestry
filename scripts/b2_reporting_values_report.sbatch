#!/bin/bash
#SBATCH --job-name=b2-reporting-values-report
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm/%x-%j.log
set -euo pipefail
experiment=b2-reporting-values-only-20261002
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python -m tapestry.experiment.planner status -e "$experiment"
.venv/bin/python -m tapestry.experiment.planner rank -e "$experiment" --no-plots
.venv/bin/python docs/experiments/b2-reporting-augmentation-20261001/values_only_report.py
