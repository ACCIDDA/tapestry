#!/bin/bash
#SBATCH --job-name=b2-weekend-report
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles02
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm/%x-%j.log
set -euo pipefail
experiment=${1:-b2-weekend-20261002}
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python -m tapestry.experiment.planner status -e "$experiment"
.venv/bin/python scripts/report_b2_weekend.py -e "$experiment"
