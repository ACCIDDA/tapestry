#!/bin/bash
#SBATCH --job-name=notify
#SBATCH --partition=jlessler
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=output/slurm/%x-%j.log

# Publish an experiment summary to ntfy once the watched allocations have ended.
# scripts/jlessler.sbatch queues this automatically with --dependency=afterany,
# so a timeout or cancellation still sends a summary. Run it by hand for any
# experiment with: sbatch scripts/notify.sbatch EXPERIMENT ARRAY_JOB_ID [...]
set -euo pipefail
experiment=${1:?usage: sbatch scripts/notify.sbatch EXPERIMENT ARRAY_JOB_ID [ARRAY_JOB_ID ...]}
shift
# Reuse the experiment's pinned snapshot; the notifier never depends on the
# working tree, so an edit mid-run cannot change what it reports.
export PYTHONPATH="$PWD/data/experiments/$experiment/code/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python "data/experiments/$experiment/notifications/b01_notify.py" \
    -e "$experiment" --jobs "$@" ${NTFY_URL:+--url "$NTFY_URL"}
