#!/bin/bash
#SBATCH --job-name=geography-report
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/geography-report-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/data/experiments/forecast-geography-v2/code/src"
export MPLBACKEND=Agg OMP_NUM_THREADS=4
.venv/bin/python -m tapestry.experiment.planner rank -e forecast-geography-v2
.venv/bin/python analysis/forecast-geography-v2/analyze.py data/experiments/forecast-geography-v2/ranking-e53fd92c4f32
