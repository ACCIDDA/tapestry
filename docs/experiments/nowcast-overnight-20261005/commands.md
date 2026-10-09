# Manager commands for this study

Run from `/proj/jlessler/projects/tapestry-all/tapestry-nowcast-overnight-20261005`
on Longleaf. These commands reproduce the actual narrowed allocations. Do not
replan an experiment whose dispatcher is running: each experiment pins its source.
Use new experiment names for fresh reproductions, and update dependent source
paths in the planning scripts when moving the source experiments. The scripts
print their fully expanded common-planner command and write `design.csv`.

## GPU cohorts

The first cohort contains 18 seed-42 conditions, three-seed clean-input controls,
and the selected three-seed C3 full-error weak-joint condition: 26 runs total.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan.py -e nowcast-overnight-joint-pilot-20261005 --seeds 42
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-joint-pilot-20261005 --run-id mlp-target-scheduled_final-7ea1f60d868c --seeds 42 43 44
LANES=6 GPUS=2 sbatch --job-name=nowcast-joint-pilot --array=0-1 --nodelist=g1803jles02 --time=06:30:00 scripts/jlessler.sbatch nowcast-overnight-joint-pilot-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e nowcast-overnight-joint-pilot-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e nowcast-overnight-joint-pilot-20261005 --no-plots
```

The second cohort contains 13 seed-42 conditions and eight confirmation runs:
C1/C2 original-mask controls, C2 nonlinear two-stage correction and C3 half-error
weak joint each add seeds43/44, for 21 total runs.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_next.py -e nowcast-overnight-trajectory-pilot-20261005 --seeds 42 --backbones C1 C2 C3
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-trajectory-pilot-20261005 --run-id mlp-pathogen-scheduled_final-3e3263276271 --seeds 42 43 44
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-trajectory-pilot-20261005 --run-id mlp-target-scheduled_final-2b1c3145fb19 --seeds 42 43 44
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-trajectory-pilot-20261005 --run-id mlp-target-scheduled_final-f5b499484bba --seeds 42 43 44
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-trajectory-pilot-20261005 --run-id mlp-target-scheduled_final-1c2e24b26cfe --seeds 42 43 44
LANES=6 GPUS=2 sbatch --job-name=nowcast-trajectory-pilot --array=0-1 --nodelist=g1803jles02 --time=04:00:00 scripts/jlessler.sbatch nowcast-overnight-trajectory-pilot-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e nowcast-overnight-trajectory-pilot-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e nowcast-overnight-trajectory-pilot-20261005 --no-plots
```

The focused third cohort has C1 half-error joint weight.05 and C3 half-error joint
weight.01, each with three seeds, for six total runs.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_joint_refine.py -e nowcast-overnight-joint-refine-20261005 --seeds 42 43 44
LANES=6 GPUS=2 sbatch --job-name=nowcast-joint-refine --array=0-1 --nodelist=g1803jles02 --time=02:30:00 scripts/jlessler.sbatch nowcast-overnight-joint-refine-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e nowcast-overnight-joint-refine-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e nowcast-overnight-joint-refine-20261005 --allow-incomplete --no-plots
```

The fourth cohort isolates joint representation from synthetic reporting-error
augmentation: the same C1/C3 joint architectures use unchanged finalized training
inputs, auxiliary weight .05, and three seeds (six runs).

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_clean_joint.py -e nowcast-overnight-clean-joint-20261005 --seeds 42 43 44
LANES=6 GPUS=2 sbatch --job-name=nowcast-clean-joint --array=0-1 --nodelist=g1803jles02 --time=02:00:00 scripts/jlessler.sbatch nowcast-overnight-clean-joint-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e nowcast-overnight-clean-joint-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e nowcast-overnight-clean-joint-20261005 --allow-incomplete --no-plots
```

## Fixed-forecaster CPU cohorts

These require the completed saved models above. They never retrain the C1
forecaster. All corrected-input conditions have matched raw-input CPU controls.
The main nowcaster confirmation supplies saved per-seed estimators to later
conditions. Execute dependency order from top to bottom.

| Experiment name | Planning script and extra option | Runs |
|---|---|---:|
| `nowcast-overnight-c1-correction-confirm-20261005` | `nowcast_overnight_plan_replay.py --refit-c1-full` | 6 |
| `nowcast-overnight-pair-support-20261005` | `nowcast_overnight_plan_pair_support.py` | 15 |
| `nowcast-overnight-uncertainty-20261005` | `nowcast_overnight_plan_uncertainty.py` | 12 |
| `nowcast-overnight-residual-uncertainty-20261005` | `nowcast_overnight_plan_residual_uncertainty.py` | 12 |
| `nowcast-overnight-admissions-correction-20261005` | `nowcast_overnight_plan_admissions_correction.py` | 12 |
| `nowcast-overnight-covariate-ablation-v2-20261005` | `nowcast_overnight_plan_covariate_ablation.py` | 9 |
| `nowcast-overnight-mc-check-20261005` | `nowcast_overnight_plan_mc_check.py --eval-members 2048` | 12 |

For each row, replace `EXPERIMENT` and `PLANNING_SCRIPT` below with its exact values;
add the listed extra option to the plan command. For example, the uncertainty
cohort's complete command sequence is:

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_uncertainty.py -e nowcast-overnight-uncertainty-20261005 --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=4 sbatch --job-name=nowcast-uncertainty --array=0-3 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=32G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-uncertainty-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e nowcast-overnight-uncertainty-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e nowcast-overnight-uncertainty-20261005 --no-plots
```

General sequence (the protocol has literal commands for every historical launch):

```bash
PYTHONPATH=src .venv/bin/python scripts/PLANNING_SCRIPT -e EXPERIMENT --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=4 sbatch --job-name=EXPERIMENT --array=0-3 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=32G --time=01:00:00 scripts/jlessler.sbatch EXPERIMENT
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e EXPERIMENT
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e EXPERIMENT --no-plots
```

For resumption, add `--retry-failed` after the experiment name in the launch
command, retaining the CPU environment and node2 resource overrides. The generic
status command's suggested default GPU launch must not be used for these CPU
replays. `--allow-incomplete` gives provisional rankings with explicit seed counts.

## Scoring the predeclared component composition

This uses the same common scorer; `score` is the inference-free launch action.
Use both seasons. Add `--backbone C1` for the C1 transfer, or omit it for C3. Omit
`--half-errors` only to reproduce the initial full-error C3 composition.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py plan -e nowcast-overnight-joint-pilot-20261005 --half-errors --season 2025-2026
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py score -e nowcast-overnight-joint-pilot-20261005 --half-errors --season 2025-2026
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py status -e nowcast-overnight-joint-pilot-20261005 --half-errors --season 2025-2026
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py rank -e nowcast-overnight-joint-pilot-20261005 --half-errors --season 2025-2026
```

## Reports

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_report.py -e nowcast-overnight-joint-pilot-20261005 nowcast-overnight-trajectory-pilot-20261005 nowcast-overnight-joint-refine-20261005 nowcast-overnight-clean-joint-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_uncertainty_report.py -e nowcast-overnight-mc-check-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_wis_diagnostics.py
squeue -a -u "$USER" -o "%.18i %.30j %.9T %.10M %.25R"
```

The uncertainty report accepts each new CPU cohort name and writes separately
named tables/figures. The early `nowcast_overnight_forward_score.py -e EXPERIMENT`
reports only completed 2025–26 folds through the same scorer; it does not mark
unfinished two-fold manager runs complete. See [protocol](protocol.md) for the
historical failed launch attempts, narrowed allocation decisions and exact jobs.

Final factorial cohort: `nowcast-overnight-no-cov-admissions-20261005`, six runs,
planner `scripts/nowcast_overnight_plan_mc_complete.py --seeds 42 43 44`, CPU2048
inference on node2, array3818281. Exact launch/status/rank are in the06:03 protocol
entry. Regenerate the compact comparison export after reports:

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_principal_export.py
```

After generating the principal export, render the final CPU figure without re-scoring:

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_final_cpu_plot.py
```

Final status: all59 forecaster-training seed tasks and102 replay seed tasks completed; no nowcast job remains queued/running. Obsolete failed cohorts are retained as history, with completed replacement cohorts. See run-inventory.csv and results.md for deduplication.

## Additional strongest-backbone pipeline comparison

The07:13 protocol entry records the six C1 pipeline runs (full/half correction ×
seeds42/43/44). Both the nowcaster and forecaster have no covariates. This retrains
C1, unlike the CPU replay experiments. Add
`nowcast-overnight-c1-pipeline-20261005` to the main report's experiment list once
completed. Array3822546 runs on the two node2 H100s; its hard allocation end is
08:23:01 EDT. This supersedes the earlier06:18 statement that no jobs remained.

## Final reports after all cohorts complete

The07:13,07:29 and07:48 protocol entries contain exact plan/launch/status/rank
commands for the final C1 pipeline, same-corrector replay and H1002048 validation.
Their final arrays were3822546,3822803 and3823143. All finished. Historical
dependency IDs must be replaced with new array IDs when repeating the workflow.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_report.py -e nowcast-overnight-joint-pilot-20261005 nowcast-overnight-trajectory-pilot-20261005 nowcast-overnight-joint-refine-20261005 nowcast-overnight-clean-joint-20261005 nowcast-overnight-c1-pipeline-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirmed_plot.py
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_c1_pipeline_report.py
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_uncertainty_report.py -e nowcast-overnight-gpu-mixture-check-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_principal_export.py
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_final_cpu_plot.py
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_inventory.py
```

Final totals supersede earlier snapshots:65 training tasks and117 replay tasks,
all complete across both folds. The inventory distinguishes inference device and
draw count when deduplicating repeated recipes. No nowcast jobs remain queued.
