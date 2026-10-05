# Agent Instructions

This is a research project. Optimize for fast iterations. No need to test everything or run test. We run, we check, we evaluate. Compute is cheap.
Prefer simpler solutions, do not over-engineer or introduce unnecessary abstractions, frameworks, dependencies, or infrastructure.
Keep code simple and easy to change. 

## Plain-language explanations

Always explain in plain, concrete terms, including in research summaries and documentation.
Do not identify runs or experimental conditions with ambiguous shortcuts. A label
such as "finalized," "corrected," "control," or a run ID can describe many different
training and evaluation setups; it is not an adequate explanation by itself.
For each result or comparison, explicitly state which model was used, what data
and seasons it was trained on, what was changed in those training inputs, what
prediction labels it learned from, and what data and season it was evaluated on.
Distinguish retraining a model from giving an existing model different inputs.
Make clear what each treatment applies to: training inputs, prediction labels,
evaluation inputs, or the model itself. Never make the reader infer this from a name.
Use sentences such as "the model trained on X and evaluated on Y scored Z,
compared with ..." and explain the score and whether higher or lower is better.
Tables, legends, and summaries must also make these distinctions clear. Short
labels are acceptable only after an unambiguous definition beside the comparison;
do not rely on the reader remembering a run's meaning from earlier conversation.

Do not spend tokens inspecting images or browsing/viewing websites just to check results, except if ask. The user checks. Provide him with the graphs.
Do not take shortcut or hidden assumptions.  State every material assumption explicitly in your response and add them to the documentation.
Do not keep stuff around for fear of failure. A rewrote module -> the old one is discarded. even if that creates some problems, everything (calibrations, runs) will anyway be fully rerun with the latest version of the code. In writing and code, do keep track of the history behind a decision. Just describe the things. If something is important add it to the documentation's log.

When running on longleaf, we have two choices:
- regular GPUs partitions (you have one example)
- our patron nodes in a hidden partition named jlessler. We will run mostly on this, parallelizing runs in a single GPUs. we have
  - g1803jles01.ll.unc.edu:  512GB ram 56 physical CPU cores Quantity 4 of Nvidia L40, 48GB
  - g1803jles02.ll.unc.edu: 64 physical CPU cores 2Tb RAM 2x Nvidia H100, 96 GB

## Launching training and scoring jobs

Whenever you launch a job that trains or scores, **always give the user the
manager commands** alongside whatever you run yourself: the `plan`, the launch,
and the `status`/`rank` that follow. They are what the user runs to check on and
resume the work, so a launch reported without them is incomplete.

```bash
.venv/bin/python -m tapestry.experiment.planner plan -e NAME -s SCENARIO --seeds 42 43 --device cuda
sbatch --job-name=NAME --array=0-3 scripts/jlessler.sbatch NAME
.venv/bin/python -m tapestry.experiment.planner status -e NAME
.venv/bin/python -m tapestry.experiment.planner rank -e NAME
```

One planner and one scorer serve every scenario; `status` prints the exact
resubmission command for unfinished tasks. ntfy notifications are on by default:
`scripts/jlessler.sbatch` queues its own `afterany` summary job, so a timeout or
cancellation still reports. `NTFY=0` disables it, `NTFY_URL` retargets the topic.

## Testing scope

Keep tests only for plausible silent errors with material scientific consequences:
loss/score mathematics, scientific weighting, leakage, and data units/alignment.
Do not add tests for plotting, formatting, CLI strings, ordinary run failures,
artifact existence, or broad architecture execution. No coverage target and no
requirement to keep one test per component. Routine validation uses research runs
and inspection; frequently rewritten code does not need a general regression suite.
