# Manager commands and source snapshots

Run from the private remote project:

```bash
ssh longleaf
cd /proj/jlessler/projects/tapestry-all/tapestry-vintage-overnight-20261005
```

All three research experiments are complete; no resubmission is needed. Inspect
them with the ordinary manager. The `-a` is required for the hidden partition.

```bash
for vintage_experiment in vintage-overnight-round1-20261005 vintage-overnight-admissions-focused-20261005 vintage-overnight-c1-diluted-20261005; do
  PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e "$vintage_experiment"
  PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e "$vintage_experiment" --no-plots
done
squeue -a -j 3800158,3804709,3818337
```

The shared environment's editable installation points elsewhere, so manual commands
explicitly set `PYTHONPATH`. Each Slurm launch uses its experiment's pinned source.
The historical launch commands are retained in `index.md`; the fresh-name commands
below reproduce those configurations without replacing completed snapshots.

## First comparison: 78 runs, two folds each

The planner originally creates 90 seed runs. The narrowing helper retains only
seed 42 for full-strength empirical all-three and early-two/recent-finalized arms
on C1/C2/C3, leaving 78 runs. Those six conditions are single-seed pilots.

The independent `reproduction-source-v1` directory is checksum-identical to the
first-round code snapshot and includes its notifier. Do not use an experiment's
own snapshot as the source of a re-plan: planning replaces the destination snapshot.
The prepared directory avoids that self-replacement and provides the notifier path
required by the planner. Choose a fresh experiment name for another reproduction.

```bash
vintage_round1=vintage-overnight-round1-v1-reproduction
PYTHONPATH=reproduction-source-v1/src .venv/bin/python scripts/plan_vintage_overnight.py -e "$vintage_round1"
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_narrow.py -e "$vintage_round1"
LANES=4 GPUS=4 sbatch --job-name="$vintage_round1" --array=0-3 --time=06:00:00 scripts/jlessler.sbatch "$vintage_round1"
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e "$vintage_round1"
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e "$vintage_round1" --no-plots
```

Original array 3800158 completed all 78 runs. Idle owners 1/2/3 were released after
finishing their fits; owner 0 exited normally. Ordinary resumption of the original
experiment already preserves the narrowed jobs file.

## Admissions-focused follow-up: 36 runs, two folds each

The independent `reproduction-source-v2` matches the follow-up and dilution
training code, before the later descriptive metadata clarification.

```bash
vintage_admissions=vintage-overnight-admissions-v2-reproduction
PYTHONPATH=reproduction-source-v2/src .venv/bin/python scripts/plan_vintage_overnight.py -e "$vintage_admissions" --admissions
LANES=4 GPUS=4 sbatch --job-name="$vintage_admissions" --array=0-3 --time=02:30:00 scripts/jlessler.sbatch "$vintage_admissions"
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e "$vintage_admissions"
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e "$vintage_admissions" --no-plots
```

Original array 3804709 completed all 36 runs. It was submitted with a 135-minute
limit and an 08:25 EDT deadline on 5 October 2026; owners 2/3 were increased to
150 minutes while pending. One eight-minute C3 learned-admission seed 44 fit was
restarted onto a later-ending allocation, preserving its failed first attempt and
unchanged configuration. It completed on attempt 002. Idle owners 0/1/3 were
released; owner 2 exited normally. Fresh reproduction uses 150 minutes for all
owners and does not reuse the historical deadline.

## Final C1 dilution: three runs, two folds each

```bash
vintage_diluted=vintage-overnight-c1-diluted-v2-reproduction
PYTHONPATH=reproduction-source-v2/src .venv/bin/python scripts/plan_vintage_overnight.py -e "$vintage_diluted" --diluted
LANES=3 GPUS=1 sbatch --job-name="$vintage_diluted" --array=0 --time=00:45:00 scripts/jlessler.sbatch "$vintage_diluted"
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e "$vintage_diluted"
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e "$vintage_diluted" --no-plots
```

Original array 3818337 completed all three runs and exited normally. It had the
same historical 08:25 EDT scheduling deadline and a 45-minute limit.

## Reports and input audits for the completed original experiments

These commands summarize already-scored forecasts or inspect permitted inputs:

```bash
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_report.py -e vintage-overnight-round1-20261005
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_report.py -e vintage-overnight-admissions-focused-20261005 --reference-experiment vintage-overnight-round1-20261005
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_report.py -e vintage-overnight-c1-diluted-20261005 --reference-experiment vintage-overnight-round1-20261005
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_export.py
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_attribution.py
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_shared_targets.py
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_paired_plot.py
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_proxy_audit.py
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_metadata.py
```

The scheduler audit and successful seed counts are in `final-job-state.json`.
No research GPU allocation remained at 07:36 EDT. Independent reproduction-source
checksums are in `reproduction-source-checksums.json`. No reproduction experiment
was planned or launched while preparing those source copies. The earlier two-epoch
smoke job is excluded from the 117 research runs; an abandoned admissions queue was
cancelled before starting any fit. Scientific input/label invariants were checked
with `scripts/vintage_overnight_scientific_checks.py`; no broad test suite was added.
