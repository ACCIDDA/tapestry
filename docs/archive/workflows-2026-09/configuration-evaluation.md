# Configuration exports, rankings, and projection fans

Sweep rankings use [total-WIS ratios](experiment-manager.md#ranking), computed
for every run. This page covers the full EpiBench pipeline with plots and fans,
intended for shortlisted configurations. Earlier published B0 reports were
withdrawn. See [Longleaf setup](../longleaf-setup.md) for the execution workflow.

Our challenges are **unversioned custom scoring configs**, with finalized,
non-vintaged evaluation truth. They are not registered as versioned
library challenges. Source/data hashes record reproducibility, not challenge versions.
Emily’s challenge ground truth is not used.

The default evaluation runs the complete installed EpiBenchmark command:
`python -m epibench score --config-path score.yaml`, once per target/season.
EpiBench loads and validates forecasts, loads truth, calls R scoringutils, computes
relative WIS, and writes `EpiBenchmark_scores.csv` and `summary.md`. All candidates
and the official ensemble are freshly scored together. The local R bridge is
used only by the [frozen hub support](hub-evaluation.md) workflow.

Install the research environment and R packages before registering an experiment;
keep dependencies fixed during execution. See [Longleaf setup](../longleaf-setup.md).

To score the collected B0 runs and publish the completed report:

```bash
sbatch scripts/b0_compare.sbatch b0
# After successful completion (the path is printed by compare):
.venv/bin/python scripts/publish_evaluation_docs.py --comparison data/experiments/b0/comparison-<hash>
.venv/bin/python -m mkdocs build --strict
```

The node launcher runs `manager compare --workers 9`. The underlying
`tapestry.evaluation.sweep` CLI supports other collections of saved runs via
`--runs`, `--frozen`, and `--output`; `--score-workers` defaults to two.
Completed EpiBench scores are reused when their input and scorer fingerprints
match. No scoring step retrains models.

Each `epibench/<target-season>/` directory contains the runnable `score.yaml`,
model CSV inputs, a local `hub/` snapshot, EpiBench output and log, and provenance.
The snapshot contains the exact frozen tasks and one frozen truth release; it
has no `.git` directory, so EpiBench cannot pull newer hub data. Hub schemas are
read from the pinned mirror commit. Tapestry checks complete support
before scoring and checks returned task keys, finite metrics and relative WIS
afterwards.

The publisher copies rankings and diagnostic figures and regenerates projection
fans from the saved EpiBench forecast inputs (72 SVG figures in total). Full
Hubverse exports stay in `data/`. CSV companions are supported because EpiBench's
submitted-model loader currently accepts CSV only.

## Quantile grid

Scoring and saved predictions use the hub's 23 levels: 0.01, 0.025, 0.05, 0.10,
…, 0.90, 0.95, 0.975, 0.99. The CV and prediction NPZ writers, Hubverse CSV/Parquet
exports, totals, and EpiBench inputs all use this grid. The exporter selects the
exact stored levels, with no interpolation or resampling.

WIS is recomputed for candidates and reference ensembles. WIS from the earlier
five-quantile reports is not numerically interchangeable with 23-quantile WIS.

[Emily’s configs](emily-configs.md) define vintage inputs and origin calendars.
They are separate from this evaluation, which uses the nine frozen evaluation
task sets and finalized-data B0 models.

## Identifiers

A configuration ID is its [scenario string](experiment-manager.md#scenario-strings),
for example `b0:h12:tr_4rt:ed_lin:geo1:dyn1:lw_first:enc_mlp:sp_none:hd_sh:dec_leg:nz_glob:z16:w64:ep50:pat0:bs8:m8:lr0.001`,
and `<scenario>:s42` identifies its seed-42 run. Seeds share a configuration ID.
Moving the run, switching execution device, or changing code or data does not
change it. Dataset and code hashes, the git commit, and evaluation draws are saved
as `provenance` in `configurations.json`; the manager warns when compared runs come
from different commits. `configurations.csv` lists each ID, seed, label, run path
(relative to the comparison folder), and scenario fields. An ID is a readable
configuration key, not an ordinal rank or performance label.

The published report uses `B0-<12 hex>-s42` hash IDs rather than scenario strings.

## Forecasts and support

`hubverse/<hub>/model-output/<model_id>/<reference_date>-<model_id>.parquet`
contains Hubverse long-form columns:

```text
reference_date,target,horizon,target_end_date,location,output_type,output_type_id,value
```

Only the 23 hub levels and available held-out target weeks are exported, including
origins with no corresponding ensemble submission. These are retrospective
forecast archives, not operational submissions. Files are partitioned by hub,
model, and reference date, with no truth columns mixed into submissions. Native
US and zero-padded state/DC FIPS are preserved. Hub horizons 0–3 map to internal
leads 1–4; reference date equals context end plus seven days. Admission rounding
from the saved forecasts is unchanged; ED forecasts remain proportions.
Malformed/crossing quantiles, duplicate units, invalid target dates, and missing
scoring tasks raise errors. Season-boundary targets outside their held-out fold
are excluded by the CV protocol.

All candidate runs and the official ensemble are freshly scored through the full
EpiBench config pipeline on the exact frozen `units.parquet` task sets. No arbitrary
missing-date allowance is used: each candidate must cover every scoring task.
Runs with narrower support stop explicitly rather than receiving an easier
ranking. The frozen hub commits, truth releases and unavailable target/seasons
are documented in the frozen comparison manifest.

## Rankings and plots

- `scores.parquet`: all per-unit scores, targets, seasons, and model IDs.
- `leaderboard.csv`: WIS, median AE, 50/95% coverage, bias and WIS components,
  split by target, season, geography (US or states/DC), and horizon; within-cell
  WIS ranks, counts, relative WIS and its valid denominator counts.
- `run_ranking.csv`: primary all-target scores and ranks for individual seeds,
  plus secondary influenza/admissions objectives.
- `configuration_ranking.csv`: objective means, sample SD and seed counts by
  configuration, ordered by `all_target_rank`, with middle-seed identifiers.
  One-seed SD is undefined, not zero.
- `plots/<target-season>/`: EpiBench WIS components, relative-WIS heatmap and
  reference-date time series, independently for US and states/DC; compatible
  score CSVs; projection fan SVGs for US and NC by default.
- `REPORT.md`: linked ranking summary and graph index.

Relative WIS in diagnostic plots uses the **official ensemble** as reference,
matching the B0 experiment objective, whereas the InfluPaint paper used
FluSight-baseline. The mean of individual WIS ratios is not the ratio
of mean WIS; both are named separately in the leaderboard. Zero reference WIS
produces an undefined per-task ratio and is counted explicitly.

The primary all-target objective gives each of the six targets equal weight,
then available season/geography cells equal weight within a target. Compute a
geometric mean WIS ratio for each seed, then an arithmetic mean across seeds to
rank configurations. The report, ranking downloads, and fans share this rule.

The secondary influenza objective is the geometric mean of mean-WIS/ensemble-mean-WIS
ratios, weighting each available season/geography cell equally. The admissions
objective first averages log ratios within each target and then equally weights
admission targets. These are admission-focused diagnostics; they do not select
the overall winner.
No absolute WIS is pooled across hospitalization and ED units. Configuration
scores average individual seed objectives and report their spread; three seeds and exploratory model selection do not establish a controlled
significance claim.

Projection fans connect the four horizons **from the same forecast origin**;
only the **top three configurations across all six targets** are shown,
plus the official ensemble in **light blue** and the best configuration for the
displayed target/season in **light red**. Rank configurations by the arithmetic
mean of their seed scores. Each seed score is the geometric mean WIS ratio,
weighting targets equally, then available season/geography cells within each
target equally. The target/season score equally weights US and states/DC.
All four horizons contribute through the leaderboard's `all` rows.

Fans use each configuration's **middle-performing seed**: the median seed by
overall score for the top three, and by target/season score for the season winner.
Identical representative runs appear once; if the same configuration has different
middle seeds under the two objectives, both are shown. Ties use seed number;
an even seed count uses the upper middle. The report includes the all-target
configuration ranking with mean seed scores, sample SD, and representative seeds.
This selection includes ED targets and matches the report's primary ranking.

Panel labels show variant names and seed numbers; `configurations.csv` lists each
variant's settings. Every fourth available origin is shown to reduce overlap; all origins
are exported and scored. Median and 50/95% intervals overlay frozen truth. US and
NC are a prespecified illustration, not selected for performance. Use
`--locations US 37 06 36` for other native hub location codes.

Finalized-data retrospective leave-one-season-out CV retains its information
advantage over operational submissions, including later-season training in the
first two folds. These rankings are exploratory, not untouched validation.

## What EpiBench still needs for a direct frozen benchmark

The interpretation of “full EpiBench” here is the complete **custom-config scoring
pipeline**, using the hub's 23 quantiles and retaining the official-ensemble reference,
and frozen dates. The four bundled library challenges cover influenza admissions
for three seasons and RSV admissions for 2025–26. They use the hub baseline as
reference and their own dates. Switching to those challenges
would change the experiment and still leave COVID and ED targets uncovered.

These are observations of the local checkout, not missing scoring metrics:

| Capability | Current EpiBench behavior | Adaptation here / useful upstream change |
|---|---|---|
| Exact evaluation task list | Config mode uses submitted-facet unions and warns about missing support; library mode requires a Cartesian grid | Accept an explicit task manifest and require identical support, including irregular season-boundary tasks |
| Frozen truth release | Scoring selects the latest `as_of` separately for each row | Accept a truth file or full-release cutoff policy; the snapshot contains only the chosen release |
| Pinned local hub | A local Git clone is automatically pulled | Add a no-update option and explicit commit pin; use a non-Git snapshot here |
| Challenge coverage | Four built-in scoring challenges; Emily supplies ten create configs covering more targets | Create configs supply vintage inputs, but still need scoring definitions to become library scorecards |
| Custom scorecard | Config mode writes scores and summary; scorecards are available only through library challenges | Allow a scorecard definition in custom configs, including grouped aggregation |
| Multiple targets/seasons and seed summaries | One target per scoring config, no B0 configuration/seed registry | Run nine configs; Tapestry aggregates the resulting EpiBench scores |
| Submitted Parquet forecasts | Submitted-model paths must be CSV files/directories | Export CSV; add Parquet support upstream for smaller inputs |
| Stable location codes | CSV truth loading and R/Python bridge reads infer numeric codes in state-only data | Explicit location schemas/string dtypes upstream; use Parquet truth and restore output FIPS here |
| Reproducibility manifest | Output does not include all code/data hashes and R package versions | Save these alongside each invocation |

Nothing above prevents using the full scoring command today with prepared local
inputs. No EpiBench source changes are required for this report. The B0 exploratory
objectives remain Tapestry aggregations, not an EpiBench library-challenge scorecard.
