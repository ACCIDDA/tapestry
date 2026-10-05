#!/bin/bash
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --nodelist=g1803jles02
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%x-%j.log
set -euo pipefail
experiment=${1:?experiment required}
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m tapestry.experiment.planner status -e "$experiment"
.venv/bin/python -m tapestry.experiment.planner rank -e "$experiment" --no-plots
# Analysis uses the current corrected scorer; saved prediction artifacts stay fixed.
export PYTHONPATH="$PWD/src"
.venv/bin/python docs/experiments/context-nowcast-20261001/forecast_report.py -e "$experiment" --output "${2:-docs/experiments/context-nowcast-20261001}"
.venv/bin/python docs/experiments/context-nowcast-20261001/rescore_histories.py -e "$experiment" --output "${2:-docs/experiments/context-nowcast-20261001}"
.venv/bin/python docs/experiments/context-nowcast-20261001/tradeoff_report.py --output "${2:-docs/experiments/context-nowcast-20261001}"
if (( $# >= 4 )); then
    .venv/bin/python docs/experiments/context-nowcast-20261001/forecast_bootstrap.py -e "$experiment" \
        --output "$2" --strength "$3" --uncertainty "$4"
fi
