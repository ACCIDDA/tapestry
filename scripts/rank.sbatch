#!/bin/bash
#SBATCH --job-name=tapestry-rank
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%x-%j.log
set -euo pipefail
experiment=${1:?usage: sbatch scripts/rank.sbatch EXPERIMENT}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m tapestry.experiment.planner status -e "$experiment"
.venv/bin/python -m tapestry.experiment.planner rank -e "$experiment" "$@"
