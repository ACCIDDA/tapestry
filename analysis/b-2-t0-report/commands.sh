#!/usr/bin/env bash
# Run from /proj/jlessler/projects/tapestry-all/tapestry on Longleaf.
# Reproduction only: do not re-plan an existing experiment while it runs.
# Plan all 309 scenarios from the checked-in design:
# .venv/bin/python - <<'PY'
# import json, subprocess
# scenarios = [r['scenario'] for r in json.load(open('analysis/b-2-t0/design.json'))['rows']]
# subprocess.run(['.venv/bin/python', '-m', 'tapestry.experiment.planner', 'plan',
#     '-e', 'b-2-t0', '-s', *scenarios, '--seeds', '42', '43', '44',
#     '--device', 'cuda', '--eval-members', '256'], check=True)
# PY
# Training launch (already completed):
# sbatch --job-name=b-2-t0 --array=0-3 scripts/jlessler.sbatch b-2-t0
# sbatch --job-name=b-2-t0-h100 --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch b-2-t0
# Monitor and rank:
# .venv/bin/python -m tapestry.experiment.planner status -e b-2-t0
# .venv/bin/python -m tapestry.experiment.planner rank -e b-2-t0
# Standard report with top-three fans/heatmaps (actual report job: 3061288):
sbatch --partition=jlessler --nodelist=g1803jles01 --cpus-per-task=4 --mem=32G \
  --time=01:00:00 --job-name=b-2-t0-report --output=analysis/b-2-t0-report/slurm-%j.log \
  --wrap='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python analysis/b-2-t0-report/build_report.py'
