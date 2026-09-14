#!/bin/bash
#SBATCH --job-name=b0-eval-parallel
#SBATCH --partition=jlessler
#SBATCH --nodelist=g1803jles01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --chdir=/proj/jlessler/projects/tapestry-all/tapestry
#SBATCH --output=/proj/jlessler/projects/tapestry-all/tapestry/output/slurm/b0-eval-parallel-%j.log

set -euo pipefail
source /usr/share/lmod/lmod/init/bash
module load r/4.5.0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
hostname
.venv/bin/python scripts/parallel_b0_evaluation.py
