cd /proj/jlessler/projects/tapestry-all/tapestry-b7-revisions-20261007
export PYTHONPATH=data/experiments/b7-folds-20261007/code-resume/src
.venv/bin/python scripts/compare_b6_b7.py plan
sbatch scripts/b7_compare.sbatch
.venv/bin/python scripts/compare_b6_b7.py status
.venv/bin/python scripts/compare_b6_b7.py rank
# Recompute score tables from completed saved forecasts only:
.venv/bin/python scripts/compare_b6_b7.py score
# Initial job: 4218799. Completed replay/score job: 4218875.
