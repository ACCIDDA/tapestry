# Overnight nowcasting and forecasting study, 5 October 2026

The study retrains the three leading B2 architectures. C1 is a pathogen-specific MLP with distance-based spatial sharing and no covariates. C2 is a target-specific MLP with summarized inpatient, wastewater, Kinsa, ILINet, clinical laboratory and FluSurv inputs. C3 is a target-specific MLP with neighbor sharing and Kinsa inputs. None has a finality flag or random input dropout in the first batch. All learn unchanged finalized future labels at t+1 through t+4. Joint models also learn unchanged finalized recent labels at t−3 through t. Scores are WIS divided by matched Hub WIS; lower is better.

For evaluation in 2024–25, training trajectories and labels come from 2022–23, 2023–24 and 2025–26; empirical reporting errors come only from 2025–26. This is retrospective. For evaluation in 2025–26, training trajectories and labels come from 2022–23, 2023–24 and 2024–25; empirical reporting errors come only from 2024–25. These are development seasons, not untouched tests.

Evaluation uses archived Wednesday numerical reports under the B2 source-specific availability schedule. If that schedule assumes an observation is available but its archive is absent, the frozen final value is an explicit numerical proxy. Prediction labels always remain frozen final values. No archive proxy enters the empirical reporting-error library. Genuine missing final observations stay unavailable.

Direct forecasts use raw evaluation reports. Training inputs are either unchanged final histories or histories perturbed with permitted donor reporting errors at half or full strength. The seasonal error bootstrap preserves donor age and signal alignment; availability is unchanged. Models and label-normalization scales are refitted for every seasonal fold and seed.

## Current design and score definition

The initial cohort was narrowed to **26 runs**: all 18 seed-42 pilots, three
seeds for each of the three unchanged-input controls, and three seeds for C3
joint auxiliary weight 0.05. The trajectory cohort has **21 runs**: 13 seed-42 pilots plus eight confirmation runs (C1 and C2 mask .2, C2 nonlinear full correction, C3 half-error joint .05 each add seeds43/44). The
chronological entries below preserve why the original larger allocations were
changed; use the later 26-run/13-run reproduction commands rather than the
original all-seed launch when reproducing the actual cohort.

For each target, season and location, sum model WIS and divide by summed Hub
ensemble WIS on identical frozen tasks. State/DC ratios share 80% of the score
equally; US receives 20%. Admissions targets have weight 1 each and ED targets
weight 0.5 each, renormalized over the targets supported in that season. The
retrospective season scores only flu/COVID admissions; the forward season scores
all six targets. Report these seasons separately. The combined score gives each
season equal weight. Configuration summaries average the completed training
seeds, with their count and variability explicit; no missing seed is imputed.

## Joint-loss correction

Inspection of both the current workspace and the pinned weekend snapshot found that `objective_weights` was defined but not called. Training and validation instead called `loss_cell_weights` directly across all eight labels. Consequently, the weekend joint models ignored the advertised reconstruction weight, and adding four reconstruction labels altered the forecast objective's total weight. The old joint models remain valid tests of a pooled recent-and-future objective; their 0.25 and 1 configurations are not distinct weight treatments.

This study actually uses separately normalized future and reconstruction weights. The future objective has total weight 1 and the recent objective has total weight 0.05, 0.25 or 1 before component slicing. Epoch selection uses future prediction labels only, an intentional additional change: reconstruction is an auxiliary training objective, and the research goal is future forecasting. Thus new versus old joint models differ in both loss wiring and the early-stopping criterion.

## Initial batch and commands

Remote project: `/proj/jlessler/projects/tapestry-all/tapestry-nowcast-overnight-20261005`. Code was copied from the current uncommitted workspace into a private folder, not restored from Git HEAD. The shared virtual environment and immutable data are symlinked. Every planner command must set `PYTHONPATH=src` because the virtual environment's editable installation points to an older central checkout. Slurm sets the pinned experiment snapshot automatically.

The initial array 3800097 was restarted after four minutes because three controls were appended after its dispatchers had read the task list. No complete results were lost. The replacement reads all 18 configurations.

18 configurations: three architectures × unchanged final training inputs, half-strength reporting errors, full-strength reporting errors, and three repaired joint weights. Three training seeds 42, 43 and 44; two evaluated seasons each. Slurm array 3800217 uses both H100 GPUs on patron node 2 with six simultaneous fits per GPU.

```bash
cd /proj/jlessler/projects/tapestry-all/tapestry-nowcast-overnight-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan.py -e nowcast-overnight-joint-pilot-20261005 --seeds 42 43 44
# This calls the common planner plan -e NAME -s SCENARIOS --seeds 42 43 44 --device cuda.
# Its exact expanded command is recorded in output/nowcast-joint-plan.log.
LANES=6 GPUS=2 sbatch --job-name=nowcast-joint-pilot --array=0-1 --nodelist=g1803jles02 --time=06:30:00 scripts/jlessler.sbatch nowcast-overnight-joint-pilot-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-joint-pilot-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-joint-pilot-20261005 --allow-incomplete --no-plots
```

A focused numerical check verified that future-label weights sum to 1, recent-label weights sum to their requested 0.25 or 1, changing the latter leaves future weights unchanged, and the US retains 20% of future weight.

## Archive-coverage audit

An input-only audit confirms that the evaluated seasons differ strongly in actual archived coverage. For the latest event week across states/DC, 2024–25 evaluation uses final numerical proxies for 33–37% of admissions cells and approximately 88.5% of ED cells. In 2025–26 the corresponding fractions are 11.5–13.5% and 20.5–22.6%. Accordingly, unchanged-input fractions must not be interpreted as empirical zero revisions: they include proxies.

The existing protocol applies the learned correction even when the input value came from a final proxy; it does not give the nowcaster an oracle flag telling it those values are already final. No evaluation mask was changed after seeing these diagnostics. This limits the interpretation of retrospective versus forward-season performance and is particularly consequential for ED inputs.

`archived-target-input-audit.csv` reports each age, target, geography and season separately. Native absolute discrepancies and proxy fractions are descriptive input audits, not model scores. Its log-ratio descriptions use a floor of 5% of the largest current final value in that week's geography group, bounded below by 1 admission or 0.0001 ED proportion; that descriptive floor is not the training/bootstrap floor.

## Pilot replication was narrowed before results

At approximately 01:45 EDT, observed fit throughput indicated that three seeds for every exploratory recipe could consume more than four hours before the next research iteration. Under the dispatch queue lock, 28 unstarted non-control seed fits were removed from the first batch. No running fit or completed result was removed. All three clean-training controls retain seeds 42, 43 and 44. Other recipes retain seed 42; two already-attempted seed-43 runs were also preserved. The current first cohort therefore has 26 planned runs, not the original 54. The exact removed entries are recorded in the experiment's `deferred_replication.json`. Promising recipes will be confirmed with three seeds.

The planning wrapper now defaults to seed 42 for exploratory recipes and separately registers all three seeds for the clean-training controls. Use `--seeds 42` to reproduce this intended pilot allocation in a new experiment; the two preserved interrupted seed-43 attempts are historical exceptions recorded in the live jobs.csv. Do not re-plan a running experiment merely to regenerate its code snapshot.

The archive-coverage figures above cover all eligible seasonal forecast origins and their input cells. They are not restricted to frozen Hub scoring support and are not WIS-weighted percentages.

The restricted audit in `frozen-support-target-input-audit.csv` counts each target's distinct Hub-supported origin/location once, irrespective of how many horizons are scored, and is also not WIS-weighted. On that support, latest-input final-proxy fractions are 7.41% for 2024–25 flu admissions and 8.57% for 2024–25 COVID admissions. Only these two targets have frozen Hub scoring support in 2024–25. In 2025–26 all six targets are scored: admission inputs have 0% final proxies; state/DC ED proportions have 5.46% flu, 1.44% COVID and 4.93% RSV final proxies. The much larger all-season retrospective ED proxy fraction describes cross-signal input availability, not directly scored retrospective ED targets.

## Prepared next mechanisms

The next separate nowcaster uses small gradient-boosted trees to learn residual revisions from the reported eight-week trajectory, six-target multiscale levels, availability masks, annual timing, location indicators and permitted reported covariates. Each target has a 100-tree model with at most seven leaves per tree, minimum leaf size 100 and learning rate 0.05. It learns log(final/report) residual labels using the same native-unit floor as the phase regression. Final values are labels, never prediction features. Entire trajectory seasons are excluded from the corrector supplying that season's forecaster training inputs; the donor reporting-error distribution remains restricted to the permitted outer training fold. The final corrector sees all permitted training trajectory seasons. Half and full residual correction are planned. This uses scikit-learn already present in the cluster environment; the private model dependency declaration records it explicitly.

The next joint model can share its sampled t0 residual with all four future predictions. Each predictive member's inferred current-level correction is added to that member's future residuals in the decoder link space; the future head learns growth from the inferred level. This passes current-level uncertainty into the future prediction and lets forecast gradients update the t0 correction. It is compared against a plain repaired joint model using the same half-strength synthetic reporting errors and auxiliary loss weight 0.05. Both four-week reconstruction and current-week-only reconstruction are prepared. Half-strength errors follow the stronger forward-season signal in the previous C2/C3 joint experiments, while those previous experiments' ignored-weight configurations are treated as repeated realizations of the same specified objective, not causal weight treatments.

A separate regularization check restores B2's 0.2 training dropout probability to the model trained on unchanged final input histories. It does not alter evaluation availability or imply that missing reports occur at that rate. Evaluation still uses archived numerical values under the assumed B2 schedule.

These mechanisms are prepared, not yet claimed to improve forecasting. New mechanisms will be scored on the identical frozen Hub support. The 2025–26 forward-season score receives particular attention because apparent pooled improvements in old separate-nowcaster models were mostly retrospective.

Replaying the previously fitted C2 and C3 phase-plus-location ridge correctors, without fitting them again, also reveals uneven input correction in 2025–26. Over all eligible seasonal origins, latest state/DC flu admission MAE falls from 14.35 to about 12.73 and RSV admission MAE from 5.82 to about 5.36. COVID admission MAE rises from 8.48 to about 9.02; its mean signed error changes from −8.13 to approximately +2.05 admissions. All three ED-proportion MAEs increase. This is evidence against simply increasing the correction magnitude and supports testing less extrapolative trajectory models. The replay audit uses the original permitted-season fitted coefficients, unchanged archived evaluation inputs and final labels. It does not retrain a model or imply a forecasting improvement. Exact values, ages and geographies are in `saved-ridge-input-correction-audit.csv`.

## First forward-season pilot decision

At the first completed 2025–26 fold readout, C3 with full-strength synthetic training-input errors and repaired joint weight 0.05 scored 0.93504, versus 0.95205 for C3 trained on unchanged final inputs and evaluated on the same raw vintage inputs. C3 is the target-specific neighbor-sharing MLP with summarized Kinsa inputs. Both models trained on 2022–23, 2023–24 and 2024–25, using reporting errors only from 2024–25 when applicable. Both learned final t+1..t+4 labels; the joint model additionally learned final t−3..t0 labels. The score is WIS relative to matched Hub WIS, lower is better. This approximately 1.8% improvement is a seed-42 development result, not a confirmed finding.

The stronger C3 joint weight 0.25 was worse (1.03283), and all initially available C2 joint weights were worse than the C2 unchanged-input baseline. Therefore the next paired joint-anchor experiment uses weight 0.05 and focuses on C3; C2 remains useful for the separate trajectory-nowcaster comparison. Seeds 43 and 44 were restored for the C3 weight-0.05 recipe using `scripts/nowcast_overnight_confirm.py`, which extends an existing recipe under the dispatch lock without overwriting pinned code. This makes 28 planned initial-cohort runs.

The early readout links only completed 2025–26 fold files into explicitly single-season manifests and calls the shared frozen-support `score_run` and `rank` functions. The latter asserts identical target/season/location/horizon/task counts across models. It does not mark the pending 2024–25 fold or the full manager run as completed.

The nonlinear nowcaster uses squared error on log-revision labels, matching the original ridge regression's label scale. Its per-location/target floor is 5% of the maximum available **reported value in that input context**, with minimum 1 admission or 0.0001 ED proportion. This differs from the reporting-error generator's training-season peak floor. A learned log revision `d` produces `max(0, (reported + floor) * exp(strength * d) - floor)`, with `d` clipped to ±log(4), and ED proportions clipped above at 1. Thus “half correction” halves the log revision; it is not an arithmetic average of the full corrected and reported values. Tree leaf-value regularization is 10, minimum leaf size is 100 and all hyperparameters are fixed before evaluating this cohort. The tree regularization parameter is not mathematically the same penalty as ridge regression.

The second cohort is queued as array **3803457**, after the first array. It contains 12 seed-42 recipes: C2 has phase/location ridge, two nonlinear correction strengths and restored B2 training dropout; C3 has those four plus four paired joint variants. The C3 joint variants are plain half-strength errors, t0-anchored half-strength errors, t0-anchored full-strength errors, and t0-anchored half-strength errors with only t0 reconstruction. All use auxiliary weight 0.05. The first cohort's plain full-strength joint model supplies the matched full-strength comparison.

```bash
cd /proj/jlessler/projects/tapestry-all/tapestry-nowcast-overnight-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_next.py -e nowcast-overnight-trajectory-pilot-20261005 --seeds 42 --backbones C2 C3
LANES=6 GPUS=2 sbatch --job-name=nowcast-trajectory-pilot --array=0-1 --nodelist=g1803jles02 --time=04:00:00 --dependency=afterany:3800217 scripts/jlessler.sbatch nowcast-overnight-trajectory-pilot-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-trajectory-pilot-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-trajectory-pilot-20261005 --allow-incomplete --no-plots
```

A second queue narrowing removed the two not-yet-running interrupted C2 noisy-input seed-43 retries. The first cohort now has exactly 26 runs: 18 seed-42 pilots, six extra clean-control seeds, and two C3 weight-0.05 confirmation seeds. No active or completed fit was removed.

Both separate correctors change only the latest four weeks of the six target histories. Covariates can inform the nonlinear corrector but are not themselves corrected. Forecaster training retains the reporting-error generator's perturbed covariates; evaluation retains the archived covariate values and explicit final proxies under the B2 source schedule. The joint current-level anchor likewise concerns each target's own sampled t0 residual, not a correction to covariate series.

To reproduce the **actual 26-run first-cohort allocation** in a fresh experiment, use the pilot planner with seed 42, which automatically keeps three seeds for clean controls, then explicitly add the selected C3 weak-joint confirmation before launching. The confirmation helper works before or after dispatch initialization and never replaces the pinned snapshot. This does not reinstate any deferred weak-recipe repetition:

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan.py -e NEW_INITIAL_EXPERIMENT --seeds 42
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e NEW_INITIAL_EXPERIMENT --run-id mlp-target-scheduled_final-7ea1f60d868c --seeds 42 43 44
LANES=6 GPUS=2 sbatch --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch NEW_INITIAL_EXPERIMENT
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e NEW_INITIAL_EXPERIMENT
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e NEW_INITIAL_EXPERIMENT --allow-incomplete --no-plots
```

The named confirmation is the C3 target-specific neighbor/Kinsa MLP with full-strength local reporting-error augmentation, final recent-plus-future labels and reconstruction weight 0.05. The archived experiment's own `code/` directory remains the exact source of the already-run first cohort; the recipe allocation above describes reproducing the design with the current code.

### 02:49 EDT: predeclared target composition diagnostic

The seed-42 C3 weak-joint model improved RSV hospitalization and RSV ED forecasts,
while damaging COVID ED forecasts. Before reading its seed-43/44 outputs, we fixed
this composition rule: use the already-trained clean-input C3 components for flu
and COVID, and already-trained weak-joint C3 components for RSV. C3 has independent
models per target, so this substitutes complete target components; it does not
average predictions, retrain models, or change evaluation inputs. Both source
models train on 2022–23, 2023–24, and 2024–25 for this 2025–26 evaluation. The joint
source uses full-strength reporting errors drawn only from 2024–25 and finalized
recent/future labels, with auxiliary weight 0.05; the clean source uses unchanged
finalized training inputs and finalized future labels. Both receive raw archived
Wednesday evaluation values under B2 source availability/lags, including the
explicit final numerical proxies described above. The common frozen-support WIS
scorer checks identical evaluation tasks. Quantile-file composition asserts exact
alignment of dates, locations, truth, masks, and quantile levels.

This is development-based target selection from seed 42. Seeds 43 and 44 assess
training-seed replication on the same development season, not an untouched test.
Retrospective 2024–25 frozen support contains no RSV targets, so the composition
would equal the clean model there; the diagnostic applies the same fixed composition to both folds and reports them separately. Joint runs contain an extra season-boundary origin; composition selects the exact clean-model origin list before asserting dates, labels, masks, and locations. The scorer independently verifies identical frozen tasks.

From the private remote project directory:
```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py plan -e nowcast-overnight-joint-pilot-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py score -e nowcast-overnight-joint-pilot-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py status -e nowcast-overnight-joint-pilot-20261005
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py rank -e nowcast-overnight-joint-pilot-20261005
```
Repeat these four commands with `--season 2024-2025` for the retrospective fold (default is `2025-2026`).

### 02:55 EDT: fixed RSV composition repeats across training seeds

Forward 2025–26 WIS ratios for C3 seeds 42/43/44 are 0.952053/1.007169/1.040534
when trained on unchanged final inputs, 0.935041/0.972727/1.060809 for weak joint
training on full-strength synthetic reports, and 0.897810/0.923950/0.999570 for the
fixed RSV-joint/flu-COVID-clean composition. All models train on 2022–23, 2023–24,
and 2024–25; the synthetic reporting errors come only from 2024–25. Finalized
future labels remain unchanged; joint components additionally learn finalized
recent labels with auxiliary weight 0.05. All receive the same archived numerical
evaluation inputs and B2 source schedule described above. Lower WIS ratios are
better. The composed mean is 0.940443 versus 0.999919 for clean training, about
5.95% lower. The all-target joint mean is 0.989526, about 1.04% lower, and seed 44
worsens. RSV admissions improve for each seed; RSV ED improves for seeds 42/44
but worsens for seed 43. COVID ED worsens for all three joint-training seeds.

The composition rule was selected from seed 42 on the evaluation season, before
reading seeds 43/44. Repetition across seeds reduces concern about training
randomness but does not remove evaluation-season selection. This is a promising
research lead, not independent seasonal validation. The saved component CSVs and
graphs preserve the selection/replication distinction. Main uniform-model
comparisons remain intact.

### 03:05 EDT: preserve the strongest absolute B2 benchmark

The C1 clean-training forward scores for seeds 42/43/44 are
0.944978/0.924653/0.926103 (mean 0.931911), below the C3 selected composition mean
0.940443. The C3 composition improves its own architecture's control; it is not
the overall best result. C1 is the pathogen-specific MLP with distance spatial
sharing and no covariates, trained on unchanged finalized trajectories from
2022–23, 2023–24, and 2024–25 with finalized future labels and evaluated on the
same archived 2025–26 inputs/proxies as other models.

Before any second-cohort dispatcher began, added one C1 clean-training pilot
with the original B2 `mask_rate=0.2` regularization. This randomly masks training
inputs; it does not change archive availability or prediction labels. The
second cohort now has 13 seed-42 recipes: its previous 12 plus this C1 pilot.
Planner defaults now reproduce that cohort with `--backbones C1 C2 C3`.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_next.py -e nowcast-overnight-trajectory-pilot-20261005 --seeds 42 --backbones C1 C2 C3
# Already submitted as array 3803457; do not submit a duplicate:
LANES=6 GPUS=2 sbatch --job-name=nowcast-trajectory-pilot --dependency=afterany:3800217 --array=0-1 --nodelist=g1803jles02 --time=04:00:00 scripts/jlessler.sbatch nowcast-overnight-trajectory-pilot-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-trajectory-pilot-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-trajectory-pilot-20261005 --allow-incomplete --no-plots
```

At 03:15 EDT, with 23/26 initial runs complete, changed the pending second array
to a corresponding-element dependency. Each GPU can begin its next six-lane
allocation when its own previous allocation succeeds, without interrupting any
remaining retrospective confirmation:
```bash
scontrol update JobId=3803457 Dependency=aftercorr:3800217
```
For reproducing that scheduling choice, replace `--dependency=afterany:3800217`
with `--dependency=aftercorr:3800217` in the second-cohort launch above.

### 03:25 EDT: trajectory nowcaster input diagnostics

Both trajectory array elements began at approximately 03:19 EDT after all 26
initial runs completed successfully. The new nonlinear and anchored-joint paths
are fitting without runtime errors. Before forecasting finishes, the saved
nowcaster diagnostics show that the C3 nonlinear corrector reduces latest-week
state/DC MAE in 2025–26 from 14.35 to 10.94 flu admissions, 8.48 to 7.90 COVID
admissions, 5.82 to 5.43 RSV admissions, 0.000544 to 0.000462 flu ED proportion,
and 0.000203 to 0.000182 RSV ED proportion. COVID ED worsens from 0.000261 to
0.000301. Half correction gives COVID admissions MAE 7.19 and COVID ED 0.000271.
The ridge comparator worsens COVID admissions and all three ED input MAEs.

These are **all eligible seasonal input origins**, 2,652 latest-week state/DC
cells per target, including explicit numerical proxies. They are not restricted
to frozen Hub scoring tasks and are not WIS weighted. No forecast-improvement
claim follows from these input improvements. The nowcaster learned only from
2022–23, 2023–24 and 2024–25 synthetic reports, with errors drawn only from
2024–25 and finalized recent values as labels. Held-out 2025–26 finalized
values are used only to evaluate these diagnostics. The plot and CSV preserve
these scopes and separate native units for admissions and ED proportions.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_diagnostics.py -e nowcast-overnight-trajectory-pilot-20261005
```

### 03:50 EDT: focused confirmations and fixed-forecaster replay

Forward seed-42 C2 with original B2 20% random training-mask regularization scores
0.926447 versus 0.981359 without that regularization. C3 weak joint training
(auxiliary 0.05) with half-strength reporting errors scores 0.925835 versus
0.952053 for its unchanged-input control and 0.935041 for full-strength errors.
Both train on 2022–23, 2023–24 and 2024–25 with unchanged finalized future labels;
the joint model also learns finalized t−3..t0 labels and draws reporting errors
only from 2024–25. Evaluation remains raw archived 2025–26 values under the same
B2 source schedule/proxy policy. These remain seed-42 development leads. Restored
seeds 43/44 for these two existing recipes without replacing the live snapshot:
```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-trajectory-pilot-20261005 --run-id mlp-target-scheduled_final-2b1c3145fb19 --seeds 42 43 44
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-trajectory-pilot-20261005 --run-id mlp-target-scheduled_final-1c2e24b26cfe --seeds 42 43 44
# Array3803457 is already running. Resume only if needed:
LANES=6 GPUS=2 sbatch --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch nowcast-overnight-trajectory-pilot-20261005 --retry-failed
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-trajectory-pilot-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-trajectory-pilot-20261005 --allow-incomplete --no-plots
```
The trajectory cohort now has 17 runs: 13 pilots and four added confirmations.

The fixed-forecaster replay keeps C2/C3 forecaster weights trained on unchanged
final inputs and future labels from the three permitted training seasons. It
changes evaluation inputs using the saved ridge or nonlinear corrector, which
learned finalized recent labels from synthetic reports on the permitted training
seasons only. Each checkpoint path, hash, training-season list, and held-out fold
is recorded; label/date/availability alignment is asserted. No forecaster is
retrained. Both folds follow the fixed season/error-source definitions above.

CPU inference runs beside GPU training. Because CPU/GPU floating-point and Monte
Carlo draws differ, each architecture also has a matched raw-input **CPU replay**
control using the same checkpoint and seed. Corrected CPU forecasts should be
compared with that CPU control, not attributed directly to differences from the
old GPU forecast file. The first CPU array 3813407 failed because the dispatcher
forced CUDA. New pinned experiment v2 uses `DEVICE=cpu`; its array is3813953.
Some v2 attempts produce a valid forward fold then stop while the retrospective
nowcaster file is not yet available from the running trajectory cohort; retry
those attempts once the source files exist. No source model is refitted by replay.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_replay.py -e nowcast-overnight-fixed-replay-v2-20261005 --seeds 42
DEVICE=cpu LANES=1 GPUS=4 sbatch --job-name=nowcast-fixed-replay-v2 --array=0-3 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=24G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-fixed-replay-v2-20261005 --retry-failed
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-fixed-replay-v2-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-fixed-replay-v2-20261005 --allow-incomplete --no-plots
```

The stricter input diagnostic `frozen-support-nowcaster-correction-2025-2026.csv`
counts each target's own frozen-Hub origin/location once, with no WIS weighting.
Here the C3 full nonlinear corrector improves all six latest-week input MAEs,
including COVID ED from 0.000198 to 0.000187. Its all-season COVID ED worsening
therefore does not describe the same origin/location set as scored forecasts.
These remain input reconstruction diagnostics, not forecasting scores.

### 04:05 EDT: additional confirmations and shared-target diagnostic

Added seeds 43/44 for the C1 unchanged-final-input model with mask regularization
0.2 **before its pilot result**, to preserve a three-seed version of the strongest
absolute B2 benchmark. Also added seeds 43/44 for the C2 full nonlinear nowcaster
pipeline after its forward seed 42 score of 0.945398 versus 0.981359 for clean-input
C2 and 0.971709 for the ridge pipeline. C2 half nonlinear correction scored 0.973210.
This pipeline retrains C2 on season-cross-fitted corrected synthetic training
inputs and unchanged final future labels; it is distinct from fixed-forecaster
replay. The second cohort now has **21 runs**: 13 pilots plus eight confirmations
(C1 mask0.2, C2 mask0.2, C2 full nonlinear pipeline, C3 half-error weak joint).

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-trajectory-pilot-20261005 --run-id mlp-pathogen-scheduled_final-3e3263276271 --seeds 42 43 44
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_confirm.py -e nowcast-overnight-trajectory-pilot-20261005 --run-id mlp-target-scheduled_final-f5b499484bba --seeds 42 43 44
# Existing array 3803457 claims the added seeds; same status/rank/resume commands above.
```

The report additionally gives an equal average of flu and COVID admission WIS
ratios in both seasons. This is a secondary shared-target diagnostic; the
official score and frozen support remain unchanged. It removes the difference
in which target types contribute, not differences in calendar dates, training
seasons, error donors, or task support across years. Forward C3 three-seed means
on this diagnostic are 0.917457 for unchanged-input training and 0.926713 for
uniform full-error weak-joint training. The fixed RSV composition equals clean
training on this diagnostic by construction.

The nonlinear corrector is also being applied to saved C1 checkpoints. Its
Kinsa features belong to the separate nowcaster; the C1 forecaster still receives
no covariates and its weights remain unchanged. The replay code separately
constructs each model's inputs and asserts target-history alignment before
replacing only the forecast model's target values. This is an evaluation-input
change, not C1 retraining. Matched raw-input CPU replay remains the comparator.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_replay.py -e nowcast-overnight-c1-fixed-replay-20261005 --seeds 42 --c1-only
DEVICE=cpu LANES=1 GPUS=4 sbatch --job-name=nowcast-c1-fixed-replay --array=0-3 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=24G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-c1-fixed-replay-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-c1-fixed-replay-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-c1-fixed-replay-20261005 --allow-incomplete --no-plots
```

### 04:20 EDT: C1 confirmation and missing-donor-pair ablation

The three-seed C1 fixed-forecaster confirmation is essentially tied forward:
raw-input CPU replay averages 0.932828 and full nonlinear correction 0.932683.
The mean paired percentage change is −0.008%. The positive seed-42 pilot did not
establish a robust forward gain. Retrospective means remain 0.948477 versus
0.860002. The shared forward flu/COVID admission diagnostic worsens from 0.925638
to 0.938271. Each seed refitted the permitted-season nonlinear nowcaster while
reusing its own saved C1 forecaster; source-corrector parity for seed 42 passed.
The C1 forecaster uses distance spatial sharing, no covariates, unchanged final
training inputs and finalized future labels. The separate nonlinear corrector
uses reported trajectories, calendar, location and Kinsa features and learns
finalized recent-history labels. Both folds and donor seasons remain as defined
above. This does not justify calling the first C1 replay pilot a forward gain.

The empirical reporting-error library assigns a **zero residual fallback** when
a sampled donor lacks an archived report/final pair. Synthetic input values then
stay finalized at those cells; that is missing evidence, not an observed zero
revision. The old `target_evidence_fraction` metadata field measures final-label
and season support, not actual archived-pair coverage. New private snapshots
also record `archived_pair_fraction` explicitly. No active GPU snapshot changed.

The next nowcaster experiment compares a two-by-two design: include fallback
cells or supervise only cells with observed donor pairs; apply full or half of
the predicted log correction. Real observed zero revisions stay in training.
The filter changes only the nowcaster's supervised rows, leaving synthetic input
values, source availability, and finalized forecast labels unchanged. The donor
support flag is never an inference feature. The same three permitted trajectory
seasons and single permitted error season are used in each fold. The bounded
nonlinear nowcaster is refitted separately for seeds 42, 43 and 44; each saved
clean C1 forecast model stays fixed. Raw-input CPU replays provide matched
controls. The original all-cell/full-strength condition is repeated to check
that the data-handling extension did not change its numerical result.

This CPU-only array 3814780 has 15 runs: four corrected-input conditions × three
seeds, plus three raw-input controls. GPU confirmations continue independently.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_pair_support.py -e nowcast-overnight-pair-support-20261005 --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=6 sbatch --job-name=nowcast-pair-support --array=0-5 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=24G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-pair-support-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-pair-support-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-pair-support-20261005 --allow-incomplete --no-plots
```

### 04:42 EDT: donor-pair evidence and uncertainty propagation

All 15 fixed-C1 donor-support runs completed. All 12 raw/full-correction forecast
files repeated from the earlier confirmation are exactly identical, including
all quantiles, labels, masks and dates. Adding donor-pair metadata did not change
the original data or random draws. The C1 forecaster has pathogen-specific MLPs,
distance sharing, no covariates, and was trained on unchanged finalized inputs
and finalized future labels. Its weights remain fixed. The separate nonlinear
nowcaster uses reported eight-week trajectories, calendar/location/Kinsa, the
three permitted training seasons, and full-strength errors from the single
permitted donor season. It learns finalized recent-history labels. Evaluation
uses archived Wednesday values with B2 source availability/lags and the stated
final proxies. Both seasonal splits are exactly those defined at the top.

| Nowcaster supervision; applied correction | 2024–25 WIS ratio | 2025–26 WIS ratio | Forward mean paired change |
|---|---:|---:|---:|
| Raw-input CPU replay; no correction | .948477 | .932828 | 0% |
| All synthetic cells; full log correction | .860002 | .932683 | −.008% |
| All synthetic cells; half log correction | .895565 | .926574 | −.667% |
| Observed donor pairs only; full log correction | .859425 | .957600 | +2.668% |
| Observed donor pairs only; half log correction | .893465 | .932508 | −.028% |

Lower WIS is better. Every row has seeds 42, 43 and 44. All-cell half correction
improves forward by 1.540%, .138% and .322% respectively. Removing missing-pair
zero-error fallback cells from supervision does not improve forward results;
those cells may regularize the learned correction despite being missing revision
evidence. This is a development-season comparison, not independent validation.
Full correction reduces admission underprediction but raises overprediction and
dispersion: forward RSV-admission WIS improves (.84546 versus .88843), while
COVID-admission and COVID-ED WIS worsen. Input MAE gains alone are insufficient.

Next, CPU array 3814985 compares raw, half-corrected and fully corrected point
histories against an equal mixture of raw/full histories. All four conditions use
the same history-bank inference path, 256 predictive draws, and saved C1 models
and separately fitted nowcasters for seeds 42/43/44. No models are retrained.
Each mixture draw uses one coherent hypothesis across all weeks, targets and
locations, preserving the correction trajectory. This is a predefined sensitivity
distribution, **not an empirically calibrated posterior**. Full/half controls are
replayed on this path because changing latent draw layout could change Monte
Carlo output. Finalized prediction labels, covariates and availability stay fixed.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_uncertainty.py -e nowcast-overnight-uncertainty-20261005 --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=6 sbatch --job-name=nowcast-uncertainty --array=0-5 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=24G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-uncertainty-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-uncertainty-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-uncertainty-20261005 --allow-incomplete --no-plots
```

### 05:08 EDT: replicated mixture result and focused follow-up

The C1 equal raw/full-history mixture scored .923619/.921375/.927406 forward
for seeds 42/43/44, versus matched raw-input .940355/.924769/.933359. Its mean
is .924134 (mean paired change −.928%); all three improve. The shared flu/COVID
admissions diagnostic also improves (.918595 versus .925638). Retrospective
mixture mean .890064 improves on raw .948477, but full point correction remains
better retrospectively (.860002). The mixture is still a predefined sensitivity
distribution, not a fitted posterior. All model and split definitions above apply.

History-bank point controls are numerically equivalent at reported score precision,
but not bit-identical to ordinary inference: ED-quantile discrepancies are at most
5.96e−8, and a few rounded admission quantiles differ by one count. Scientific
alignment fields are identical. Matched bank-path controls are therefore retained.
Detailed differences are in `history-bank-numerical-parity.json`. Do not claim
exact parity for this inference-layout change. Exact parity did hold for the
separate earlier donor-support metadata extension.

The completed forward GPU confirmations show C2 nonlinear two-stage mean .970977
versus C2 clean .970696: no replicated gain. Restoring mask .2 gives C1 mean
.959286 versus .931911 without masking, and C2 .963593 versus .970696. C3 joint
with half-strength errors and auxiliary weight .05 scores .958501 versus clean
.999919. Its fixed RSV-component composition scores .898185/.937331/.945207,
mean .926908. That is narrowly below C1 clean's .931911 mean, but only seed42
beats the paired C1 value; the target rule was selected on this evaluation season.
It is not an independent-validation win. Retrospective support lacks RSV, so this
composition equals the clean C3 forecast there.

GPU array 3815061 follows the trajectory cohort on node2, with just two conditions
and three seeds: C1 joint half errors/auxiliary .05, and C3 joint half errors/
auxiliary .01. This tests the useful half-error treatment on the strongest clean
backbone and whether weaker reconstruction preserves C3 future forecasts. Both
use mask0, finalized future and recent labels, future-only early stopping, and
the same fixed seasonal splits. No live snapshot was changed.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_joint_refine.py -e nowcast-overnight-joint-refine-20261005 --seeds 42 43 44
LANES=6 GPUS=2 sbatch --job-name=nowcast-joint-refine --array=0-1 --dependency=aftercorr:3803457 --nodelist=g1803jles02 --time=02:30:00 scripts/jlessler.sbatch nowcast-overnight-joint-refine-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-joint-refine-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-joint-refine-20261005 --allow-incomplete --no-plots
```

CPU array 3816453 learns uncertainty from permitted synthetic training data. For
each trajectory season, its residual predictor was fitted without that season;
only recent labels belonging to that excluded season enter its residual bank.
The outer evaluation season never supplies labels or trajectories. Each draw
bootstraps a whole residual trajectory across four weeks, six targets and all
locations; donor seasons receive equal total probability. Cellwise weighted
median centering preserves the point center. Unsupported cells have zero noise.
The saved C1 forecaster stays fixed; the nowcaster is refitted per seed, with half
log correction applied. Residual scales .5 and 1 are predefined sensitivity
settings. Raw and half-point controls use the same history-bank inference. This
is an empirical distribution of synthetic training residuals, not a claim of
calibration to real evaluation-season revisions. Synthetic input perturbations
still come only from the one permitted reporting-error donor season in each fold.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_residual_uncertainty.py -e nowcast-overnight-residual-uncertainty-20261005 --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=6 sbatch --job-name=nowcast-residual-uncertainty --array=0-5 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=24G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-residual-uncertainty-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-residual-uncertainty-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-residual-uncertainty-20261005 --allow-incomplete --no-plots
```

The fixed half-error RSV composition is also scored in 2024–25 through the common
scorer. As expected from frozen target support, it is exactly the clean C3 score.
Its plan/launch/status/rank sequence is:

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py plan -e nowcast-overnight-joint-pilot-20261005 --half-errors --season 2024-2025
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py score -e nowcast-overnight-joint-pilot-20261005 --half-errors --season 2024-2025
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py status -e nowcast-overnight-joint-pilot-20261005 --half-errors --season 2024-2025
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py rank -e nowcast-overnight-joint-pilot-20261005 --half-errors --season 2024-2025
```

Use the same four commands with `--season 2025-2026` for the forward comparison.
Composition replaces whole RSV target distributions from the separately fitted
C3 joint model; it does not average forecasts or change flu/COVID predictions.

### 05:22 EDT: information and corrected-channel scope

The learned-residual uncertainty cohort completed all12 runs. Three-seed forward
means are .924974 for residual scale.5 and .925303 for scale1, versus .926574 for
half-point correction and .932828 for raw. All three seeds improve versus raw
at both scales. Scale.5 improves versus half-point for all three seeds, but one
increment is only .000008 WIS-ratio units. Retrospective means are .888814/.880579,
versus half-point .895565 and raw .948477. The raw/full mixture's .924134 forward
mean is still slightly lower; these are small development-season gains.

The target-level WIS decomposition shows persistent COVID-ED harm under every
correction variant. The raw/full mixture improves all three admission targets
on average, but COVID-ED WIS rises from1.23333 to1.24534. The new diagnostic figure
`fixed-c1-wis-components-2025-2026.png` separates underprediction, overprediction
and dispersion and annotates90% coverage. It averages the common scorer's
location-weighted target metrics across the three seeds.

CPU array3816746 tests **admissions-only evaluation-input correction**: ED histories
stay raw, while all six targets remain forecast/scored. It compares half-point,
raw/full mixture, and half-point plus residual scale.5 against raw. The underlying
nowcaster still learns all six recent targets from permitted synthetic training
data; only the applied correction/noise is restricted to admission channels.
This is an evaluation-informed scope diagnostic prompted by ED harm and sparse
ED donor archives, not an untouched validation result. Forecasters remain fixed.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_admissions_correction.py -e nowcast-overnight-admissions-correction-20261005 --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=6 sbatch --job-name=nowcast-admissions-correction --array=0-5 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=24G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-admissions-correction-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-admissions-correction-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-admissions-correction-20261005 --allow-incomplete --no-plots
```

**Kinsa enters through the nonlinear nowcaster even though C1's forecast network
has no covariates.** Corrected C1 therefore also adds information relative to raw
C1. CPU array3816790 isolates that factor: refit the nonlinear corrector with and
without Kinsa, retain the same full-correction/raw-history mixture, and reuse the
same saved C1 forecasters. Target training draws, trajectory/donor seasons and
finalized labels remain unchanged. Raw controls and both feature conditions each
have seeds42/43/44. No held-out labels enter the refit. The source corrector-parity
assertion is intentionally omitted for the no-Kinsa model because its feature
matrix differs; season and evaluation-alignment assertions remain active.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_covariate_ablation.py -e nowcast-overnight-covariate-ablation-20261005 --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=4 sbatch --job-name=nowcast-covariate-ablation --array=0-3 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=24G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-covariate-ablation-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-covariate-ablation-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-covariate-ablation-20261005 --allow-incomplete --no-plots
```

Repeated choices based on these development-season scores—including correction
strength, uncertainty mechanism, target scope and component composition—create
selection bias. Three new training seeds test optimization/augmentation variation;
they do not create new evaluation seasons or remove the selection bias.

The first no-Kinsa arm failed before training because the project represents no
covariates with an empty group string, not a group named `none`. This was corrected
in a fresh pinned experiment; successful Kinsa/raw outputs remain preserved in
the first cohort. This failure is not a model-performance result.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_covariate_ablation.py -e nowcast-overnight-covariate-ablation-v2-20261005 --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=4 sbatch --job-name=nowcast-covariate-ablation-v2 --array=0-3 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=24G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-covariate-ablation-v2-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-covariate-ablation-v2-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-covariate-ablation-v2-20261005 --allow-incomplete --no-plots
```

### 05:26 EDT: larger predictive sample check

The admissions-only correction cohort completed all12 runs. Forward means are
.922611 for raw/full mixture, .925135 for half-point correction and .924059 for
half-point plus residual scale.5, versus .932828 raw. Retrospective means are
.892404/.897323/.889451 respectively, versus .948477 raw. The mixture's shared
forward flu/COVID-admissions diagnostic is .922009 versus .925638 raw; correcting
all channels gave .918595 on that secondary diagnostic. Restricting corrections
therefore changes the tradeoff rather than uniformly improving every target.

CPU array3816995 repeats raw inference, all-channel mixture, admissions-only
mixture, and no-Kinsa all-channel mixture with **2048 predictive draws** instead
of256, across the same three trained seeds and both folds. Its purpose is to
check whether the roughly1% improvements survive reduced Monte Carlo error, not
to create a new training replication. The forecasters remain fixed; the no-Kinsa
nowcaster is refitted from the same permitted training data. It starts after the
corrected covariate-ablation cohort. All2048-draw effects are paired with the
2048-draw raw control, not the old256-draw score.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_mc_check.py -e nowcast-overnight-mc-check-20261005 --seeds 42 43 44 --eval-members 2048
DEVICE=cpu LANES=1 GPUS=4 sbatch --job-name=nowcast-mc-check --array=0-3 --dependency=afterany:3816930 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=32G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-mc-check-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-mc-check-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-mc-check-20261005 --allow-incomplete --no-plots
```

### 05:35 EDT: no-Kinsa confirmation and joint controls

The corrected no-Kinsa cohort completed9/9 runs. Its raw/full mixture scores
.890375 retrospectively and .923618 forward, versus .890064/.924134 with Kinsa
and .948477/.932828 raw. The gain therefore does not depend on adding Kinsa.
The no-covariate nowcaster still uses reported target trajectories, availability,
calendar and location features; it learns the same finalized recent labels from
the same permitted synthetic training trajectories. C1 weights remain fixed.

The focused joint follow-up's forward results are now complete. C1 half-error
joint auxiliary.05 scores .935705/1.026565/.931486, mean .964585 versus its clean
mean .931911. Its predeclared RSV-component composition scores
.939009/.967093/.939541, mean .948548, also worse. Thus C3's target-component gain
does not transfer to the stronger C1 backbone. C3 half-error joint auxiliary.01
scores .906280/1.061270/.922813, mean .963454, compared with .958501 at auxiliary.05.
Changing the loss weight did not reduce the substantial seed variation.

CPU Monte Carlo check3816995 is running. GPU array3817320 is queued after the
joint-refinement array: C1 and C3 joint architectures, auxiliary weight.05,
**unchanged finalized training inputs** (reporting probability0), three seeds
each. The new control learns finalized recent and future labels and uses
future-only early stopping, exactly as the noisy-input joint condition. Its
purpose is to isolate joint representation from synthetic-error augmentation;
comparing only direct-clean versus joint-noisy models changes both factors.
Evaluation remains raw archived Wednesday input with B2 availability and final
proxies. An error library is instantiated by the common code but no perturbation
is applied; reporting-error source is therefore marked none for this condition.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_clean_joint.py -e nowcast-overnight-clean-joint-20261005 --seeds 42 43 44
LANES=6 GPUS=2 sbatch --job-name=nowcast-clean-joint --array=0-1 --dependency=aftercorr:3815061 --nodelist=g1803jles02 --time=02:00:00 scripts/jlessler.sbatch nowcast-overnight-clean-joint-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-clean-joint-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-clean-joint-20261005 --allow-incomplete --no-plots
```

The C1 RSV composition uses this manager sequence for both seasons:

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py plan -e nowcast-overnight-joint-pilot-20261005 --backbone C1 --half-errors --season 2025-2026
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py score -e nowcast-overnight-joint-pilot-20261005 --backbone C1 --half-errors --season 2025-2026
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py status -e nowcast-overnight-joint-pilot-20261005 --backbone C1 --half-errors --season 2025-2026
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_compose.py rank -e nowcast-overnight-joint-pilot-20261005 --backbone C1 --half-errors --season 2025-2026
```

Repeat with `--season 2024-2025`. The retrospective composition remains exactly
clean C1 because frozen support has no RSV. No prediction labels are mixed or
changed, and the common scorer uses identical support for all components.

The no-Kinsa attribution audit compared every training episode and two synthetic
replicates per episode with augmentation seed42 in both folds: 155 episodes and310 draws per fold. Target
histories, finalized labels, target availability, dates and donor-pair support
were exactly equal with and without Kinsa. Only covariate features differed.
The saved audit is `covariate-ablation-target-draw-parity.json`. The target-level
WIS figure's three components also sum to total WIS within1.25e−14 across all144
plotted target/season/seed rows; its decomposition uses the common scorer.

For resource isolation, all training and inference allocations explicitly use
node2. The default notification wrapper had no node constraint; the two remaining
notification jobs3817356 and3817099 were also pinned to node2, and the private
notification script now requests node2 for future submissions.

## 06:03 EDT — final factorial cell and consolidation

Array3818281 runs six CPU evaluations on node2: three seeds of the saved C1 forecaster with raw histories and three with equal raw/full nonlinear-corrected admission histories. The nowcaster has no Kinsa or other covariates; ED histories remain raw. There are2,048 predictive draws per forecast. Training seasons, donor seasons, finalized labels and archive/proxy evaluation assumptions are unchanged. This completes the correction-scope × covariate diagnostic selected during evaluation-guided development.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_mc_complete.py -e nowcast-overnight-no-cov-admissions-20261005 --seeds 42 43 44
DEVICE=cpu LANES=1 GPUS=4 sbatch --job-name=nowcast-no-cov-admissions --array=0-3 --nodelist=g1803jles02 --gres=gpu:0 --cpus-per-task=2 --mem=32G --time=01:00:00 scripts/jlessler.sbatch nowcast-overnight-no-cov-admissions-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-no-cov-admissions-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-no-cov-admissions-20261005 --allow-incomplete --no-plots
```

No additional model conditions are planned. Consolidation will retain matched hardware and predictive-draw controls.

## 06:10 EDT — score contribution audit

CPU job3818442 decomposes the completed2,048-draw matched C1 scores by frozen Hub reference date. Whole-season location denominators and official geography/target weights remain fixed. Per-run sums must match the common scorer within1e−10. This is descriptive analysis, not a new scoring rule or model condition.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_time_contributions.py plan
sbatch --job-name=nowcast-time-audit --partition=jlessler --nodelist=g1803jles02 --cpus-per-task=2 --mem=16G --time=00:30:00 --output=logs/nowcast-time-audit-%j.out --wrap="cd /proj/jlessler/projects/tapestry-all/tapestry-nowcast-overnight-20261005 && PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_time_contributions.py score && PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_time_contributions.py rank"
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_time_contributions.py status
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_time_contributions.py rank
```

## 06:18 EDT — final factorial complete

Array3818281 completed6/6. No-covariate admission-only raw/full mixture: retrospective.889101 versus raw.944225; forward.918912 versus raw.929312, mean paired change−1.113905%. All three seeds improve. Shared forward flu/COVID-admission score.918324 versus.922424. Every saved array in the six repeated raw forecast files equals its prior2,048-draw control exactly. The final principal export contains12 GPU conditions and5 CPU conditions, all three seeds and both folds. All nowcast training/inference/audit allocations have finished. No additional conditions will be added simply to use remaining time.

## 07:13 EDT — missing strongest-backbone retrained pipeline

The user asked whether useful iteration continued through the requested window. The earlier completed C1 experiments changed evaluation inputs of saved forecasters, whereas nonlinear-pipeline retraining covered C2/C3. We therefore launched the missing C1 comparison: full versus half nonlinear correction, each with seeds42,43,44 and both seasonal folds. C1 and its nonlinear nowcaster have no covariates. Synthetic training errors are full strength; correction strength1 versus.5 applies to the learned log revision in both forecaster training and evaluation histories. Forecaster prediction labels remain finalized future values, and nowcaster labels remain finalized recent histories. Training trajectories and sole donor seasons follow the fixed protocol above. Forecaster training histories are corrected by trajectory-season cross-fitted nowcasters; held-out labels never fit the corrector. Evaluation uses archived Wednesday inputs, B2 availability/lags and the same explicit final proxies.

Array3822546 started on both node2 H100s at07:13:01; hard allocation end08:23:01. Three lanes per GPU run all six tasks concurrently. No pinned prior snapshots changed.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_c1_pipeline.py -e nowcast-overnight-c1-pipeline-20261005 --seeds 42 43 44
LANES=3 GPUS=2 sbatch --job-name=nowcast-c1-pipeline --array=0-1 --nodelist=g1803jles02 --time=01:10:00 --deadline=2026-10-05T08:25:00 scripts/jlessler.sbatch nowcast-overnight-c1-pipeline-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-c1-pipeline-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-c1-pipeline-20261005 --allow-incomplete --no-plots
```

## 07:29 EDT — same-corrector inference follow-up queued

The C1 half-correction pipeline's completed forward seeds42/43/44 score.954850/.995029/.950376 (mean.966752), versus unchanged-input-trained C1 mean.931911 under the same H100256 inference. All three are worse. Retrospective folds remain in progress.

To isolate changing evaluation histories from retraining, array3822803 is queued after successful completion of3822546. It reuses each pipeline's exact saved final nowcaster, supplying full or half corrected histories to the saved unchanged-input-trained C1; a repeated raw-input C1 control uses the same H100256 inference. No models are fitted. Nine tasks cover three recipes × three seeds, both folds. The15-minute allocation has deadline08:25 and will be cancelled if it cannot start meaningfully by08:10.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_c1_pipeline_replay.py -e nowcast-overnight-c1-pipeline-replay-20261005 --seeds 42 43 44
LANES=3 GPUS=2 sbatch --job-name=nowcast-c1-pipeline-replay --array=0-1 --nodelist=g1803jles02 --dependency=afterok:3822546 --kill-on-invalid-dep=yes --time=00:15:00 --deadline=2026-10-05T08:25:00 scripts/jlessler.sbatch nowcast-overnight-c1-pipeline-replay-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-c1-pipeline-replay-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-c1-pipeline-replay-20261005 --allow-incomplete --no-plots
```

The derived `scripts/nowcast_overnight_c1_pipeline_report.py` will score completed outputs through the common scorer and export the paired fixed-versus-retrained comparison. It does not change the official score or prediction files.

## 07:48 EDT — final H100 validation and completed inventory

All six C1 pipeline tasks completed, and point-replay array3822803 ran07:40:53–07:43. The exact saved nonlinear correctors give a forward mean.924929 for half correction of saved C1 inputs, versus.966752 when retraining C1 behind those correctors; full correction gives.929750 versus.948652. The matched raw H100256 score is.931911. All six raw forecast archives equal the original GPU controls exactly. Corrector source hashes and permitted training seasons were checked for all12 source files.

Array3823143 then ran the final no-covariate admission-only mixture versus raw at2,048 H100 draws, reusing the exact CPU-comparison checkpoints and fitting no models. It started07:43:56 and completed before07:47; its hard allocation end was08:03:56. Results: retrospective.888403 versus raw.943373; forward.918262 versus raw.928529, mean paired change−1.100077%. All three seeds improve. This confirms the modest CPU effect on H100 with its own matched raw control.

```bash
PYTHONPATH=src .venv/bin/python scripts/nowcast_overnight_plan_gpu_mixture.py -e nowcast-overnight-gpu-mixture-check-20261005 --seeds 42 43 44
LANES=3 GPUS=2 sbatch --job-name=nowcast-gpu-mixture --array=0-1 --nodelist=g1803jles02 --dependency=afterok:3822803 --kill-on-invalid-dep=yes --time=00:20:00 --deadline=2026-10-05T08:25:00 scripts/jlessler.sbatch nowcast-overnight-gpu-mixture-check-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner status -e nowcast-overnight-gpu-mixture-check-20261005
PYTHONPATH=src .venv/bin/python -m tapestry.experiment.planner rank -e nowcast-overnight-gpu-mixture-check-20261005 --allow-incomplete --no-plots
```

Final inventory:182 completed two-fold tasks,65 forecaster retrainings and117 replays. Exact recipe/seed/draw/device deduplication gives160 (65+95). Forty-two replays refit only a nowcaster;75 reuse correctors or raw inputs. Eleven obsolete failed replay tasks are excluded, and their replacements completed. Nine component-composition seed evaluations reuse fitted models separately. The principal export contains24 complete three-seed conditions with matched inference controls. No nowcast jobs remain queued or running.
