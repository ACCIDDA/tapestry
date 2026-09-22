#!/bin/bash
#SBATCH --job-name=B2-postprocess
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=output/slurm/B2-postprocess-%j.log
set -euo pipefail
source /usr/share/lmod/lmod/init/bash
module load r/4.5.0
export PYTHONPATH="$PWD/data/experiments/B2-screen/code/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
hostname
ranking=$(.venv/bin/python -m tapestry.models.manager rank -e B2-screen |
    tee "output/slurm/B2-ranking-${SLURM_JOB_ID}.log" | tail -n 1)
.venv/bin/python scripts/plot_b2_screen.py --ranking "$ranking" --output docs/results/B2-screen
.venv/bin/python -m tapestry.models.manager compare -e B2-screen --workers 2
