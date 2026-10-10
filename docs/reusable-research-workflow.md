# Reusable research and submission workflow

How we went from data to the B7 influenza submission, and how to repeat it for another pathogen.
The shared dataset/problem pipeline described below is implemented. Scientific choices
for a new pathogen still have to be defined before running it.

"Tree search" here means the branching sequence of experiments (B0 → B7) that produced a submitted
recipe. It has nothing to do with the regression trees that correct preliminary reports. B7 is one
influenza recipe. The submitted Hub model is **ACCIDDA-Chromantis**; every submission is recorded in
the [submission log](submissions.md).

## The workflow in six steps

Each step names who does it. "You" means the human researcher. "LLM" means a Claude or Codex
assistant. "Code" means the planner, PyTorch, Slurm and the Python scorer. Numbers (losses, WIS)
always come from code, never from an LLM.

1. **Define the study (you, then LLM).** Before fitting anything, write `protocol.json`:
   - targets and their units;
   - which data are available on each Wednesday (for flu: sources at T-X, ED at T-0, Kinsa not revised);
   - how training inputs are made preliminary (for flu: the 2025–26 reporting-error process);
   - which values are the prediction labels;
   - fitting seasons and evaluation seasons, including evaluation weeks (should include October);
   - **the primary score, its tie-breakers, and which scores cannot veto a choice.**

   Any later change to this file starts a new comparison. Old results are not re-ranked under it.
2. **Screen (LLM proposes, you approve the batch and the time budget, code runs).** List candidates
   in `candidates.json`. Where possible, change one thing per candidate. Use two seeds per candidate
   and shared artificial reporting draws.
3. **Confirm (same split of work).** Rerun three to five candidates with five new seeds. Extra seeds
   measure how much fits vary between seeds. They do not remove bias from repeatedly choosing on the
   same seasons.
4. **Compare saved forecasts (code, with the LLM reporting).** Run the fitted models, without
   retraining, on the same archived Wednesday reports. Mix their forecasts and score everything on
   identical tasks. **Every comparison lists the exact candidate set and saved forecasts it used.**
   Assistants gave opposite recommendations on 7 October partly because they compared different sets.
5. **Release (you decide, code fits).** Record the choice and the reason in the study's decision list. Refit
   the chosen recipes on all completed seasons. B7 used ten seeds per recipe; that is a starting
   policy, not a rule. The release lists exact checkpoint paths and their hashes, not file-name
   patterns.
6. **Issue each week (code, then you).**
   - Refresh data, then run the release on the new data (no retraining) and export the Hub file.
   - Make the comparison graphs.
   - **Push a safe file from the current release before the deadline.**
   - Replace it after the deadline only on your explicit decision. On 8 October 2026, seven earlier
     late re-pushes to FluSight had all been merged.
   - Record every event in the [submission log](submissions.md): generated, submitted, merged,
     superseded, renamed. Keep the submitted CSV and any replaced CSV.

Weekly issuance does not mean weekly retraining. A new recipe, correction model or training set is
a new release.

## Who did what for B7 (5–8 October 2026)

| You decided | LLMs did |
|---|---|
| Assumptions: sources at T-X, ED at T-0, Kinsa not revised, 2025–26-like revisions, seasons interchangeable (no season is special) | Turned them into scenario settings; audited the code against them |
| Evaluation: per-Hub deadlines, 512 samples, natural and log scores, two seeds as default, include October, US and states reported separately (replacing the 80% states / 20% US mix), coverage is not a veto | Built the scorer and reports |
| Training objective: add the log-admission loss with weight 0.5 | Implemented it |
| Time budgets ("30 minutes", "20 more minutes", "done by 9") | Chose concrete candidates, pruned optional runs, resumed unfinished runs |
| Drop the FluSurv recipe; submit System2 on time; then "replace it" with B7 | Recommended System2 (Codex, 22:43), then B7 (Claude, then Codex, 23:58) |
| Model name, metadata flag, less code in the upload step | Wrote export, comparison graphs and git pushes |

The LLMs also ran some branches on their own initiative. One example is the peak/full-season model
study on 7 October, which you stopped at 22:05. The training audit was written by Claude (session
`dc5f13dc`). You gave the same audit prompt to Codex, then pasted Claude's findings into Codex with
your corrections at 01:05. Codex implemented the fixes. Keep such hand-offs in the decision list; a
pasted recommendation is not evidence that the person who pasted it originated it.

## Files

```text
experiments/<study>.json              candidates (name -> scenario string), seeds, protocol text
experiments/<study>-ensembles.json    saved-forecast groups to combine and rank (optional)
production/releases/<name>.json       recipes, checkpoint paths + SHA256, rule, input view, description
docs/experiments/<study>/index.md     generated report + a hand-written "Decisions" list
docs/submissions.md                   every issuance: data snapshot hash, release, CSV hash, PR/merge
production/submissions/               submitted CSVs; superseded/ keeps replaced ones
```

- **Decisions** are a short dated list in the study's report write-up. Each item states:
  - the decision;
  - the candidate set and evidence files used;
  - who proposed it and who decided it;
  - why a branch stopped: time, failure, or poor results.
- **No separate issuance record.** The submission log entry and the committed CSV are the record.
- **Dataset and problem files.** The concrete migration is defined in the
  [multi-pathogen pipeline](multipathogen-pipeline.md). A named dataset describes stored
  signals, units, geography, vintages, and covariates. A problem selects targets and inputs and fixes
  horizons, folds, truth, output type, and one primary evaluation. This replaces the earlier proposal
  for one pathogen file: a pathogen can have several scientifically different forecasting problems.
- **Git.** When committing, include `experiments/`, the release JSONs, the submission log and the
  submitted CSVs. Checkpoints, `output/` and Hub clones stay out. As of 8 October all of
  `production/` and `experiments/` are untracked (nothing committed yet, by the user's choice).
- **No new machinery.** No database, workflow framework or second scheduler.

## Code: one implementation per step (done 8 October 2026)

| Step | Module | Command |
|---|---|---|
| Data acquisition, dated panel | `chromantis.data`, `chromantis.dataset` | `python -m chromantis.data pull`, `python -m chromantis.dataset.build build` |
| Training (one route) | `experiment/fit.py`, `experiment/training.py` | `planner plan --problem ... --study ...`, `scripts/jlessler.sbatch`, `planner run` |
| Evaluate fits on other inputs | `experiment/fit.py` `replay` | `planner replay` |
| Internal evaluation | `evaluation/standard.py`, `ranking.py`, `report.py` | `planner rank` |
| Saved-forecast ensembles | `evaluation/ensembles.py` | `python -m chromantis.evaluation.ensembles` |
| Submission | `chromantis.production` + a release file | `python -m chromantis.production run` |

About 50 experiment-specific scripts and the training routes no submitted model used were
deleted; scenario options of the remaining route are all kept. Details, and the checks that the
new code reproduces the submitted B7 file and B7 training exactly:
[restructuring log](workflow.md#restructuring-log-8-october-2026). Target and input names,
units, horizons, folds, and evaluation now come from a required problem file; flu, COVID-19,
RSV, and joint-respiratory problem files use the same implementation.

## Another problem or pathogen

Reuse the procedure and the code; rerun the science. Do not carry over influenza's reporting
errors, Kinsa, ILI pretraining, season boundaries, recipes, or evaluation weights. Follow the
[multi-pathogen pipeline](multipathogen-pipeline.md): add or reuse a named dataset, write a problem
that fixes targets, inputs, horizons, folds and the primary score, check source revision histories,
then run steps 1–5 above. An unpublished forecast evaluated under that problem is the acceptance
test.

## Open gaps from B7

- **October was not evaluated.** You asked for October in the headline evaluation, but B7's selecting
  comparison started at the 23 November 2024 and 22 November 2025 reference dates. B7 was chosen
  without evidence on the early-season weeks it is now forecasting.
- **The selection score changed during the search:** first the 80/20 states/US mix, then all
  locations, then the log-admission loss, then "most likely to perform well this year." Step 1 now
  fixes the score before screening.
- **Hub metadata is out of date:** it says two recipes, five seeds and averaged quantiles. Updating
  it is deferred until the recipe settles (decision of 8 October 2026).
- **Package name:** the repository, documentation, Python package and commands use `chromantis`.

## History

- **8 October, first version (Codex).** A long proposal (about 3,600 words) with a development-history
  table, an actor field on every candidate, a separate `search.py`, and a full conversation evidence
  table.
- **8 October, rewrite (Claude).** At the user's request, the doc was cut to the workflow, roles,
  files and open gaps. Changes in this version:
  - corrected the Hub name;
  - corrected the audit authorship;
  - marked October as unmet;
  - added the on-time step, the candidate-set rule, target types and the git requirement;
  - removed `search.py` and the per-candidate actor fields.
- **8 October, package restructured (user decision: each step done once, unused code deleted).**
  The code changes proposed here were implemented; see the restructuring log.
- **8 October, second cut (user decision).** Replaced `decisions.jsonl` with a decision list on the
  study's report page and dropped the per-issuance JSON; the submission log plus the CSV is the
  record. The pathogen file is deferred until the second pathogen starts.
- **9 October, multi-pathogen plan (user decision).** Replaced the proposed pathogen file with
  separate dataset and problem contracts. Raw acquisition remains shared; dataset building creates
  named panels with covariates; each problem fixes targets, inputs, horizons, folds, and one primary
  evaluation; model recipes contain only training and model choices.
- **Evidence.** Claude sessions under `~/.claude/projects/-Users-chadi-Research-Tapestry/`
  (`eac542c9`, `bb6f8636`, `bd6c581f`, `6f220c65`, `2f2e91bb`, `dc5f13dc`, `1586eaad`, `73b3b686`,
  `fb4d71df`, `17ba82cf`) and Codex sessions under `~/.codex/sessions/2026/10/` (`5f6f69deae1c`,
  `afad9b37ac79`, `4d97e52b9c1b`, `d5ef40a4646d`, `f164be8480f1`), Eastern time.
