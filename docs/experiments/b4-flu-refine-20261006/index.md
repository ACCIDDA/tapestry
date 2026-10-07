# Focused flu refinement: contemporary ILI and correction window

The completed 600-configuration/two-seed sweep retained two different leaders: a joint flu MLP trained on real-tree-corrected synthetic histories led native score (0.779856), while a joint flu MLP trained with artificial errors plus finalized recent-history reconstruction led admissions log score (0.795342). Both use only flu pathogen inputs. Neither added COVID nor RSV consistently improved the 150 matched recipes. Seeds 42/43 and both seasonal folds were inspected. This refinement uses 24 configurations × seeds 42/43 = 48 runs, within the user's one-batch maximum of 60 configurations.

The design crosses two anchor recipes × four contemporary covariate choices (none, Kinsa, ILINet, Kinsa+ILINet) × three flu-admission correction windows (2,4,8 weeks). Each combination is independently retrained. ILINet uses the existing panel and publication lag; it is not the historical scaled ILI auxiliary task. Historical transfer is unchanged from each anchor (none), and no new scaling is introduced. Exact scenarios are in design.json and scenarios.txt. Original anchor cells are deliberately repeated as controls. Other architecture, optimizer, reporting-error and tree settings remain fixed within each anchor. Correction windows affect corrected training histories for the real-tree anchor and corrected evaluation inputs; the reconstruction anchor uses its existing artificial-error training unchanged. This experiment estimates the overall configured correction-window effect, not a pure evaluation-only intervention.

Both anchors learn finalized next-four-week flu admissions and flu ED labels. The forward fold trains 2022–23, 2023–24, 2024–25 and evaluates 2025–26; the retrospective fold trains 2022–23, 2023–24, 2025–26 and evaluates 2024–25. The real-tree anchor receives cross-fitted corrections of synthetic reports with trees trained on genuine report-to-mature pairs. The reconstruction anchor receives artificial reporting errors and an auxiliary loss on finalized recent flu values. Evaluation uses Wednesday/holiday reports, finalized fills for missing archives and reported ED. Each fit is evaluated with reported histories, tree-corrected flu admissions and a 50/50 forecast-distribution mixture. Seasons remain equally weighted for admissions. Retrospective ED has no frozen ensemble-relative support, so raw ED WIS in both seasons must accompany native/log admissions comparisons.

The completed main-run leaders' raw ED WIS (mean of two seeds) confirms a season tradeoff: the native leader has 2024–25 states/DC 0.006035 and US 0.005162 versus the log leader 0.005722 and 0.004964; in 2025–26 the native leader has 0.005070 and 0.004069 versus the log leader 0.005713 and 0.005056. Lower is better. Preserve both anchors rather than selecting solely on the composite.

Remote directory: `/proj/jlessler/projects/tapestry-all/tapestry-b4-flu-600-20261005`. The new plan snapshots the existing remote source; no model code was changed. Main sweep completed before refinement planning. Submission is guarded against duplicates by requiring the new experiment directory to be absent and against late launches by checking the 09:00 EDT epoch immediately before sbatch. Four L40 allocations with four workers each are used, following the main sweep's successful memory recovery.

```bash
cd /proj/jlessler/projects/tapestry-all/tapestry-b4-flu-600-20261005
export PYTHONPATH=src
.venv/bin/python -m tapestry.experiment.planner plan -e b4-flu-refine-20261006 \
  -s $(cat docs/experiments/b4-flu-refine-20261006/scenarios.txt) --seeds 42 43 --device cuda
LANES=4 GPUS=4 sbatch --job-name=b4-flu-refine-20261006 --array=0-3 \
  --cpus-per-task=8 --mem=95G --time=04:00:00 \
  scripts/jlessler.sbatch b4-flu-refine-20261006
.venv/bin/python -m tapestry.experiment.planner status -e b4-flu-refine-20261006
.venv/bin/python -m tapestry.experiment.planner rank -e b4-flu-refine-20261006 --no-plots
```

Do not replan or duplicate a submitted batch. Use status for recovery and monitor its job-id.txt plus the execution log below. No additional refinement batch is authorized. Jobs submitted before the cutoff may finish afterward.

## Submission

Slurm accepted array **4005335** before the 09:00 EDT cutoff. Submission returned successfully; scheduler/start verification follows. No second batch was submitted. Hourly monitoring continues for this refinement and final analysis.

Scheduler verification: all four allocations started on `g1803jles01` at **08:58:29 EDT**, before the cutoff. The plan contains exactly 24 configurations and 48 seed runs. Main training arrays ended successfully around 08:21:30–08:21:32 EDT; main ranking `3977479` completed successfully at 08:25:27 EDT. On initial refinement status all 48 runs were unfinished; subsequent checks should inspect dispatcher and logs for worker progress. No dependent refinement rank job has been submitted: the hourly monitor should run the common manager rank after completion.

## Training completed

All 48/48 runs completed successfully. Four allocations ended at 09:18:03–09:18:05 EDT on October 6. The comparison took about 20 minutes, faster than the initial 1–2 hour estimate. Common-manager final ranking started during the user's status check; interpretation remains pending.

Final ranking completed: `ranking-8717617b218d`. [Completed analysis](analysis/README.md) and CSV tables are saved. No new winner overall; retain both original covariate settings and two-week correction. All work for this experiment is complete.
