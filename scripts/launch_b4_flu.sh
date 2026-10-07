#!/bin/bash
# Run from the isolated Longleaf checkout after syncing code/data links.
set -euo pipefail
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
experiment=b4-flu-600-20261005
validation=b4-flu-check-20261005
if [[ -e data/experiments/$experiment/launch.json ]]; then
  echo "Launch already recorded; inspect status instead of duplicating." >&2
  exit 1
fi
mkdir -p output/slurm
.venv/bin/python scripts/plan_b4_flu.py --experiment "$validation" --smoke --plan
.venv/bin/python scripts/plan_b4_flu.py --experiment "$experiment" --plan
check=$(LANES=8 GPUS=1 sbatch --parsable --job-name="$validation" --array=0-0 \
  --cpus-per-task=8 --mem=80G --time=00:40:00 scripts/jlessler.sbatch "$validation")
check=${check%%;*}
l40=$(LANES=10 GPUS=6 sbatch --parsable --job-name="$experiment" --array=0-3 \
  --dependency="afterok:$check" --kill-on-invalid-dep=yes \
  --cpus-per-task=10 --mem=95G --time=12:00:00 scripts/jlessler.sbatch "$experiment")
l40=${l40%%;*}
h100=$(LANES=16 GPUS=6 sbatch --parsable --job-name=b4-flu-600-h100 --array=0-1 \
  --dependency="afterok:$check" --kill-on-invalid-dep=yes \
  --nodelist=g1803jles02 --cpus-per-task=16 --mem=180G --time=12:00:00 \
  scripts/jlessler.sbatch "$experiment")
h100=${h100%%;*}
rank=$(sbatch --parsable --partition=jlessler --nodelist=g1803jles01 --job-name=b4-flu-rank \
  --dependency="afterany:$l40:$h100" --kill-on-invalid-dep=yes \
  --cpus-per-task=2 --mem=24G --time=02:00:00 --output=output/slurm/b4-flu-rank-%j.log \
  --wrap='export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; .venv/bin/python -m tapestry.experiment.planner rank -e b4-flu-600-20261005 --allow-incomplete --no-plots')
rank=${rank%%;*}
.venv/bin/python - "$check" "$l40" "$h100" "$rank" <<'PY'
import sys,json
from pathlib import Path
from datetime import datetime,timezone
r=dict(validation=sys.argv[1],l40=sys.argv[2],h100=sys.argv[3],rank=sys.argv[4],submitted=datetime.now(timezone.utc).isoformat())
p=Path('data/experiments/b4-flu-600-20261005/launch.json');p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
PY
.venv/bin/python -m tapestry.experiment.planner status -e "$experiment"
