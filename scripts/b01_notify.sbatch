#!/bin/bash
#SBATCH --job-name=B0.1-notify
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=output/slurm/B0.1-notify-%j.log

# Use --dependency=afterany:ARRAY_A:ARRAY_B so even timeout/cancellation sends a summary.
set -euo pipefail
experiment=${1:?Experiment required}
shift
# Reuse the fixed manager code; the notifier is independent of training workers.
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python "data/experiments/$experiment/notifications/b01_notify.py" -e "$experiment" --jobs "$@"
