# Matched artificial-masking comparison

`forecast-no-mask-top32-v3` repeats the top 32 of 63 scenarios (ranked by original three-seed mean) and seeds 42–44 from
`forecast-geography-v2`, changing only `mask_rate=0.5` to `mask_rate=0`.
There are 96 new runs, three season folds per run. Wednesday availability,
finalized input values, source bundles, geography, representations, early stopping,
labels, score weights and evaluation members remain unchanged. Nowcasting stays
separate. The comparator's exact pinned code and notifier are copied after planning;
panel, population and frozen-support hashes and evaluation members must match.
All weights are trained afresh. The selection contains 17 independent-location,
9 attention and 6 pooled formulations. Selection uses the original masked-run
ranking and is therefore a selected-screen comparison, not an unbiased estimate
of masking effects across the full model space. The existing completed v2 runs are the controls.

This tests added artificial input dropout. It does not test restoring unpublished
inputs or explain every difference from B0. Early stopping can select different
epoch counts under the two fitting conditions; that is part of this comparison.
Use paired differences by scenario and seed, with season and geography breakdowns,
rather than comparing only the minimum score in each screen.

The preceding v2 allocation lasted **3:44:15**, September 24 21:31:05 through
September 25 01:15:20 Eastern, using four L40s and two H100s. Across completed seed
runs, median elapsed time was 52.6 minutes (range 21.3–86.7); each seed includes
all three folds and scoring. This is historical timing, not a guarantee for v3.

## Manager commands

From `/proj/jlessler/projects/tapestry-all/tapestry` on Longleaf:

```bash
# Plan only before starting; recipe rejects existing run attempts.
bash experiments/forecast-no-mask.sh
GPUS=6 sbatch --job-name=forecast-no-mask-top32-v3 --array=0-3 scripts/jlessler.sbatch forecast-no-mask-top32-v3
GPUS=6 sbatch --job-name=forecast-no-mask-top32-v3-h100 --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch forecast-no-mask-top32-v3
.venv/bin/python -m tapestry.experiment.planner status -e forecast-no-mask-top32-v3
.venv/bin/python -m tapestry.experiment.planner rank -e forecast-no-mask-top32-v3
```

## Reporting-availability clarification

A dedicated audit now checks the actual frozen scored target/location/issuance
support, deduplicating horizons. The latest Saturday's target is present on:

- wh flu admissions: 96.7% of scored cells.
- 2024–25 flu admissions: 92.6%; COVID admissions: 91.4%.
- 2025–26 admissions: 100% for all three pathogens.
- 2025–26 ED: 98.6% COVID, 94.6% flu, 95.1% RSV.

These are unweighted counts of the scored target's own input, not a claim that all
six histories are always complete. They exclude off-season/unscored issuances and
do not measure the training input gaps. A release recorded by Wednesday remains
available; there is no automatic withholding of the latest week. Archive absence
alone does not distinguish a real publication delay from incomplete acquisition.
The prior full-calendar missingness counts were not sufficient to explain the
B0/current performance difference.

Source: `analysis/forecast-geography-v2/scored_availability.py`; counts and cells in
`docs/results/forecast-geography-v2/scored_target_availability*.csv`.

The user reduced the scope to the best half while the full plan was being
submitted. Full-plan arrays 2460372/2460373 were cancelled and replaced by the
32-formulation experiment; their partial outputs are not used.

Reduced experiment submitted as arrays 2460419 (four L40s) and 2460420
(two H100s) on the patron nodes, partition jlessler.
