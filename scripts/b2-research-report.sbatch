#!/bin/bash
#SBATCH --job-name=b2-research-report
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm/%x-%j.log
set -euo pipefail
experiment=${1:-b2-direct-research-v1}
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
.venv/bin/python -m tapestry.experiment.planner status -e "$experiment"
.venv/bin/python -m tapestry.experiment.planner rank -e "$experiment" --allow-incomplete --no-plots
.venv/bin/python "data/experiments/$experiment/analysis/report.py" --experiment "$experiment"
