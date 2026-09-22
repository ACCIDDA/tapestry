#!/bin/bash
#SBATCH --job-name=b0-compare
#SBATCH --partition=jlessler
#SBATCH --nodelist=g1803jles01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm/b0-compare-%j.log

# Score an experiment's completed runs with nine concurrent EpiBench cases.
# Submit from the repository root: sbatch scripts/b0_compare.sbatch EXPERIMENT [extra manager options]
set -euo pipefail
experiment=${1:?usage: sbatch scripts/b0_compare.sbatch EXPERIMENT}
source /usr/share/lmod/lmod/init/bash
module load r/4.5.0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
hostname
.venv/bin/python -m tapestry.models.manager compare -e "$experiment" --workers 9 "${@:2}"
