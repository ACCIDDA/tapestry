# Configuration exports, rankings, and projection fans

Read the [completed report with inline figures and rankings](../results/b0-configuration-comparison.md).

This extends the existing frozen B0 comparison without model training. The user
confirmed that “B-10” meant B0. All 14 runs in the completed September 14 sweep
and the original September 13 CV run are included. Smoke tests under `tmp/` are
intentionally excluded.

The local source review covered `../influpaint/evaluation/README.md`,
`../influpaint/docs/workflows/evaluation.md`, its `plot_evaluation_results.py`,
and `../epibench/src/epibench/{scoring_bridge,build_plots,plot}.py`.
InfluPaint prepares saved quantiles, scores them through R scoringutils, then
plots components, relative WIS, and time series and produces rankings. EpiBench
uses the same engine. This implementation retains our existing external R
bridge (same EpiBench metrics, preserving string FIPS codes) and imports
EpiBench's actual `build_summary_figures` to produce the three diagnostic plots.
No new dependency or hub download is necessary in the existing environment.
EpiBench’s model-data CLI currently accepts CSV only; `--csv` writes CSV
companions alongside the compact Parquet archives. The delivered comparison
contains both. Its per-hub model directories can be passed directly as
EpiBench model-data paths. To add CSV companions to a Parquet-only export:

```bash
.venv/bin/python scripts/export_hubverse_csv.py data/evaluation/b0_configuration_comparison/hubverse
```

```bash
PYTHONPATH=src .venv/bin/python -m tapestry.evaluation.sweep \
  --runs data/experiments/b0_full_20260914/*_s4? \
         data/experiments/b0_season_cv_20260913 \
  --csv --output data/evaluation/b0_configuration_comparison
```

Use `--epibench /path/to/epibench` if the sibling checkout moves. The command
accepts any list of saved three-season CV runs. Add a future completed run to
`--runs` and use a new output directory to regenerate a comparison including it.
Resuming the same run set reuses scoring only when its quantiles, truth and R
script match the saved content fingerprint. Exporting or scoring never refits.

Refresh the documentation snapshot from completed results, then check the site:

```bash
.venv/bin/python scripts/publish_evaluation_docs.py
.venv/bin/python -m mkdocs build --strict
```

This copies the ranking tables and 72 SVG figures into `docs/assets/` and generates
the overview and nine target/season figure pages. The full forecast archive stays
in `data/`; documentation links use portable relative paths.

## Identifiers

`B0-<12 hexadecimal characters>` identifies a configuration, and
`B0-<12 hexadecimal characters>-s42` identifies its seed-42 realization.
`configurations.csv` maps these strings to the original directory names;
`configurations.json` stores the full identity and full SHA256 hash. Seeds share
a configuration ID. Moving the run or switching execution device does not change
it. Dataset content, model source hashes, population values, model/training/draw
settings, and any new configuration fields do change it. Source revision changes
are intentionally distinct, even for configurations with otherwise equal flags;
this distinguishes the original B0 from the later baseline implementation.
`--family B1` leaves room for another model family using the same saved schema.
An ID is a provenance key, not an ordinal rank or performance label.

## Forecasts and support

`hubverse/<hub>/model-output/<model_id>/<reference_date>-<model_id>.parquet`
contains Hubverse long-form columns:

```text
reference_date,target,horizon,target_end_date,location,output_type,output_type_id,value
```

All 23 saved quantiles and available held-out target weeks are retained, including
origins with no corresponding ensemble submission. These are retrospective
forecast archives, not operational submissions. Files are partitioned by hub,
model, and reference date, with no truth columns mixed into submissions. Native
US and zero-padded state/DC FIPS are preserved. Hub horizons 0–3 map to internal
leads 1–4; reference date equals context end plus seven days. Admission rounding
from the saved forecasts is unchanged; ED forecasts remain proportions.
Malformed/crossing quantiles, duplicate units, invalid target dates, and missing
scoring tasks raise errors. Season-boundary targets outside their held-out fold
remain excluded according to the existing CV protocol.

All candidate runs are freshly scored using R scoringutils on the exact frozen
`units.parquet` from the original comparison. The already-scored official
ensemble is reused, with its unit coverage and truth checked. No arbitrary
missing-date allowance is used: each candidate must cover every scoring task.
New runs with narrower support stop explicitly rather than receiving an easier
ranking. The frozen hub commits, truth releases and unavailable target/seasons
remain documented in the original comparison manifest.

## Rankings and plots

- `scores.parquet`: all per-unit scores, targets, seasons, and model IDs.
- `leaderboard.csv`: WIS, median AE, 50/95% coverage, bias and WIS components,
  split by target, season, geography (US or states/DC), and horizon; within-cell
  WIS ranks, counts, relative WIS and its valid denominator counts.
- `run_ranking.csv`: two exploratory objectives and ranks for individual seeds.
- `configuration_ranking.csv`: objective means, sample SD and seed counts by
  configuration. One-seed SD is undefined, not zero.
- `plots/<target-season>/`: EpiBench WIS components, relative-WIS heatmap and
  reference-date time series, independently for US and states/DC; compatible
  score CSVs; projection fan SVGs for US and NC by default.
- `REPORT.md`: linked ranking summary and graph index.

Relative WIS in diagnostic plots uses the **official ensemble** as reference,
matching the existing B0 experiment objective, whereas the original InfluPaint
paper used FluSight-baseline. The mean of individual WIS ratios is not the ratio
of mean WIS; both are named separately in the leaderboard. Zero reference WIS
produces an undefined per-task ratio and is counted explicitly.

The influenza objective is the geometric mean of mean-WIS/ensemble-mean-WIS
ratios, weighting each available season/geography cell equally. The admissions
objective first averages log ratios within each target and then equally weights
admission targets. These reproduce the existing sweep's selection objectives.
No absolute WIS is pooled across hospitalization and ED units. Configuration
scores average individual seed objectives and report their spread; unequal seed
counts and adaptive exploration prevent a controlled significance claim.

Projection fans connect the four horizons **from the same forecast origin**;
these are not horizon-specific quantile ribbons connected across different
origins. Every fourth available origin is shown to reduce overlap; all origins
are exported and scored. Median and 50/95% intervals overlay frozen truth. US and
NC are a prespecified illustration, not selected for performance. Use
`--locations US 37 06 36` for other native hub location codes.

Finalized-data retrospective leave-one-season-out CV retains its information
advantage over operational submissions, including later-season training in the
first two folds. These rankings are exploratory, not untouched validation.
