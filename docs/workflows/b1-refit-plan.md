# Handoff: align B1 selection and refitting with B0

Status: implementation plan, not an implemented fix or submitted experiment.

## Objective and evidence

Make `B1-fromB0` use B0's training procedure: choose the epoch count on inner
validation, then initialize a fresh model and fit all permitted training weeks
for exactly that count. Evaluate the refitted model on the outer held-out season.
Match validation context and origin selection as well as the final refit.

The September 17 run at commit `f0bdec6` scored 1.179307 for seed 42. B0's matching
configuration and seed scored 0.911210; its three-seed mean was 0.883377. Both use
the same frozen scoring reference. Their training procedures differ:

- `src/tapestry/models/season_cv.py`, `run`: B0 selects epochs on inner data,
  resets the component seed, and calls `fit` again on full training data.
- `src/tapestry/models/b1_run.py`, `fit_component` and `train`: B1 restores its
  best inner checkpoint and evaluates it directly. There is no full-data refit.
- `src/tapestry/models/b1_seasons.py`, `fold`: B1 currently masks hidden validation
  weeks from validation context as well as fitting context. B0 validation can
  observe earlier hidden weeks within the permitted training seasons.
- B1 currently accepts origins based on remaining labels, whereas B0 explicitly
  selects origins by calendar membership. Matching held-out week lists alone is
  insufficient to establish matching partitions.

The score gap does not yet isolate the effect of vintaged inputs. Treat the older
scores as historical results, not evidence that the refit change must improve WIS.

## Scope and fixed assumptions

Use the existing B1 dataset and its pinned final labels. It already carries
unchanged versions forward correctly. No rebuild is needed for this correction:

```
data/processed/build_b1_wednesday_calendar.npz
SHA-256: 451d004d7a9626f8b6cdc4815347ac2729d41ecf57d8607f714b06f89b27e2e7
```

Preserve source precedence, final-value filling, calendar, architecture, loss
weights, transformations, dropout recipes, and frozen evaluation support.
`B1-fromB0` remains the direct B0 architecture with mask rate zero: it consumes
availability but not the separate finality flag. Do not add that input as part
of the refit correction. Other B1 pipelines must retain their known-final bypass
and nowcast-loss rules if they use the shared training changes.

Reference finals remain retrospective conditioning information. Training-label
values and support still differ between B0 and B1. Correcting the training
procedure does not by itself produce an exact input-only experiment.

## Implementation

1. **Construct four explicit fold partitions.** Update `b1_seasons.py` and its
   callers together. Use the pinned B0 calendar and existing hidden-week pattern.
   Reconstruct each partition from the original episodes, rather than concatenating
   already-masked inner and validation episodes.

   | Partition | Permitted context | Supervised target dates | Origins for the direct four-week task |
   |---|---|---|---|
   | Inner fit | Training-season weeks excluding hidden weeks | Same inner weeks | Context-ending Saturday belongs to inner weeks |
   | Inner validation | All training-season weeks, including earlier hidden weeks | Hidden weeks only | Training origins whose four future targets include a hidden week |
   | Final refit | All training-season weeks, including hidden weeks | All training-season weeks | Context-ending Saturday belongs to training weeks |
   | Outer evaluation | Dataset context within the model calendar | Held-out season only | Context-ending Saturday belongs to held-out weeks |

   Map Wednesday episodes to B0 origins using `context_dates[-1]`, not the
   Wednesday's season. Compare actual origin and target-date sets against
   `season_cv.fold_data` and `validation_split`. Availability-based exclusions
   remain explicit; do not invent values to force real-data counts to match.
   Hidden or held-out context must zero values, availability, and finality flags
   wherever that partition excludes it. Every context date must still precede
   its forecast targets. For the two-stage task, account explicitly for its two
   recent targets and retain exclusion of visible supplied-final nowcast answers.

2. **Separate epoch selection from fixed-length fitting.** Keep one small fitting
   implementation rather than duplicating the optimization loop. With positive
   patience, select the best epoch for each independently fitted component on
   inner validation, using only inner-fit normalizers and loss scales. With
   `patience=0`, skip epoch selection and fit the full training partition for the
   configured epoch count; do not require validation to choose a checkpoint.

3. **Refit from scratch.** After selection, reset the component seed using the
   existing `seed + 10000 * component` rule. Create a fresh model and optimizer;
   do not continue training the selected checkpoint. Recompute input normalizers,
   logit centers, native Q95 loss scales, and scientific cell weights from the
   full refit partition. Preserve the existing unique-date handling and exclude
   held-out observations from all these statistics. Train exactly the component's
   selected epoch count, with no second early-stopping decision. Compute fixed
   weights for each permitted partition, retaining the existing dropout-dependent
   nowcast exclusions where applicable.

4. **Save and evaluate the refitted model.** `model.pt` and held-out forecasts must
   use refitted component weights and their refit normalizers. Retain compact,
   separate selection/refit histories in metadata: selected epoch, actual refit
   epochs, origin lists, permitted/hidden weeks, sample counts, seed, dataset hash,
   and a clear protocol marker such as `season_cv_refit_v1`. Selection checkpoints
   are optional diagnostics and must never be exported as final predictions.
   Update report/manifest consumers that currently assume one fitting phase.

5. **Use a fresh experiment.** Manager workers execute the source copy under the
   experiment's `code/`; updating the checkout does not change an existing plan.
   Completed attempts are also reused. Plan `B1-fromB0-refit` only after committing
   and syncing the implementation to Longleaf. Do not overwrite the completed
   `B1-fromB0` experiment or relabel its old results as refitted.

## Acceptance checks

Keep tests focused on scientific errors, not general execution coverage:

- A fully observed synthetic three-season fixture gives the same direct-task
  origin sets, hidden weeks, context masks, and target masks as B0 for all four
  partitions. Compare sets and cell masks, not just lengths.
- Perturbing held-out values cannot change inner/refit inputs, labels, scales,
  or weights. Perturbing hidden-week values cannot change inner fitting inputs
  or scales, but those values are restored for full refit. Validation may use
  earlier hidden context only, with labels still restricted to hidden weeks.
- Refit starts from the freshly seeded model, uses no selection optimizer state,
  and performs exactly the selected number of epochs per component. Its saved
  state and normalizers are those used for outer predictions. A lightweight
  controlled fixture can establish this without a GPU training suite.
- Availability and known-final masks retain their meanings; a filled recent final
  does not become a supervised visible nowcast answer. Unchanged source versions
  remain available. `patience=0` fits the full partition for the configured budget.
- The planned benchmark retains the same frozen scoring files, task keys,
  observations, quantile levels, target/geography weights, and season aggregation.
  Remaining real-data label or support differences are reported, not silently
  dropped or described as an input-only comparison.

Run the focused existing tests plus the necessary scientific checks above. Then
use the research run below to validate the implementation and assess performance.
Do not require matching B0 numerical scores: B1 still has different inputs and
some different training labels/support, and training is stochastic.

## Longleaf handoff and commands

The implementation owner should commit and sync code, verify the dataset hash,
and run manager preflight through `plan` before submission. The existing project
environment, population file, and frozen scoring reference are already installed.
Run from `/proj/jlessler/projects/tapestry-all/tapestry`:

```bash
b1_best=$(.venv/bin/python -c 'from tapestry.models.b1_scenarios import b0_top4; print(b0_top4("direct", 0.)[0].scenario_string)')
.venv/bin/python -m tapestry.models.manager plan -e B1-fromB0-refit \
  --suite B1-fromB0 --scenario "$b1_best" --seeds 42 43 44 \
  --dataset data/processed/build_b1_wednesday_calendar.npz \
  --eval-members 2048 --retrospective --device cuda
LANES=1 GPUS=1 sbatch --job-name=B1-fromB0-refit --array=0 scripts/jlessler.sbatch B1-fromB0-refit
.venv/bin/python -m tapestry.models.manager status -e B1-fromB0-refit
.venv/bin/python -m tapestry.models.manager rank -e B1-fromB0-refit
```

This selects only B0's rank-1 configuration, not all four recipes in the suite:
three seeds × three held-out seasons × six target components = 54 refitted
component models, plus 54 inner-selection fits when patience is positive. The
100-epoch selection cap, patience 30, and 2,048 evaluation draws remain unchanged.
Do not run this block against the current implementation; first complete the
changes and acceptance checks above. No job was launched when writing this plan.

## Results to return

Report each seed and paired B0 difference, then mean/SD across the three seeds;
include per-season/target WIS and calibration with the stated geography weighting.
Verify that every exported fold uses the refit protocol marker and new data hash.
Compare against the old B1 seed-42 score as a protocol change, not a pure vintage
effect. WIS over/underprediction shares are score components, not frequencies.

After this correction, a separate paired experiment can hold B1 labels, masks,
origins, and training procedure fixed while changing only recent inputs from
Wednesday versions to their pinned finals. A flag-aware direct model is another
separate experiment. Neither belongs in the refit patch.
