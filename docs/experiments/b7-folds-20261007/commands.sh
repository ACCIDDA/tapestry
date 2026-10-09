# Longleaf checkout. These are the plan and launch commands used for this screen.
cd /proj/jlessler/projects/tapestry-all/tapestry-b7-revisions-20261007
export PYTHONPATH=src
.venv/bin/python scripts/plan_b7.py --experiment b7-folds-20261007 --seeds 44 45 --plan
LANES=4 GPUS=6 sbatch --job-name=b7-fast-h100 --array=0-1 --nodelist=g1803jles02 --cpus-per-task=4 --mem=160G --time=00:28:00 scripts/jlessler.sbatch b7-folds-20261007
LANES=4 GPUS=6 sbatch --job-name=b7-fast-l40 --array=0-3 --nodelist=g1803jles01 --cpus-per-task=4 --mem=100G --time=00:28:00 scripts/jlessler.sbatch b7-folds-20261007

# Inspection and ranking; these are safe to repeat.
.venv/bin/python -m tapestry.experiment.planner status -e b7-folds-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b7-folds-20261007 --allow-incomplete --no-plots
.venv/bin/python scripts/report_b7.py --experiment b7-folds-20261007

# The queued report after the original arrays:
# sbatch --dependency=afterany:4211524:4211525 scripts/b7_report.sbatch b7-folds-20261007
# Status prints the exact resubmission command for unfinished fits.
# Four optional fits were deliberately stopped for the time budget:
# A_blocks3_calendar_off (44,45), A_blocks3_correct_4_weeks (44,45).

# User-authorized 20-minute continuation of seven recorded partial attempts.
# Uses code-resume: same model and evaluation logic, with saved-fold reuse.
sbatch --array=0-1 scripts/b7_resume.sbatch b7-folds-20261007
sbatch --dependency=afterany:4217224 scripts/b7_report.sbatch b7-folds-20261007
.venv/bin/python -m tapestry.experiment.planner status -e b7-folds-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b7-folds-20261007 --allow-incomplete --no-plots
