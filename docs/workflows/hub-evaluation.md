# Frozen hub support and historical comparison

The current report uses the [full EpiBench scoring workflow](configuration-evaluation.md)
and [configuration evaluation report](../results/b0-configuration-comparison.md).
The commands below document how the original frozen task set was established;
they document the historical 23-quantile result, not the current scoring entrypoint.
Current code and new exports use five quantiles; exact historical reproduction
requires its recorded source revision. The original frozen task/truth files remain unchanged.

This historical comparison evaluates already saved Build B quantiles, with no retraining. It implements
the same external `Rscript` -> `as_forecast_quantile()` -> `score()` pattern and
metrics as `../epibench/src/epibench/scoring_bridge.py`. It calls **R scoringutils**;
it does not substitute the earlier Python WIS calculation. The pipeline is in
`src/tapestry/evaluation/`.

## Run

From the repository root, with the existing environment:

```bash
PYTHONPATH=src .venv/bin/python -m tapestry.evaluation.compare \
  --run data/experiments/b0_season_cv_20260913 \
  --mirrors data/mirrors --cache data/evaluation/hub_cache \
  --output data/evaluation/b0_hub_comparison

PYTHONPATH=src .venv/bin/python -m tapestry.evaluation.pdf_report \
  --comparison data/evaluation/b0_hub_comparison \
  --output output/pdf/b0_hub_comparison.pdf --locations US NC
```

Use a new comparison output directory for a new completed run. The shared hub
cache is reused only when its pinned Git commit and requested reference dates
match. The comparator has `--model-name` and `--rscript` options. Any future model
can use this comparison by exporting the same seasonal `forecasts.npz` schema:
23 quantiles x origins x four horizons x six channels x locations, with labeled
context/target dates, locations, observed values, and masks. The six-channel
mapping is explicit in `hubs.py`; adding another hub/target means extending that
small registry, not rewriting scoring or plotting.

The PDF defaults to US and NC as a compact, prespecified illustration, showing
all four horizons and all three comparison models. Scoring uses **every eligible
location**, not just the plotted ones. `--locations US NC CA NY` chooses more
locations; `--locations all` renders all supported locations and can make a very
large PDF. Figures are vector graphics with only median, 50%, and 95% intervals.
Each location/season/target page has four horizon rows and three model columns,
with matching dates and y-axis limits. Missing weeks break lines.

Fresh environments need `pip install -e '.[evaluation]'` plus R packages
`scoringutils` and `purrr`. The initial run used scoringutils 2.1.1. Exact R/package
versions are saved per comparison. No SSH, new hub fetch, or model training is
required when using the local mirrors and saved forecasts.

## User-directed scoring support

The ensemble defines the evaluation support, as requested. A forecast unit is
`reference_date, target_end_date, location, horizon` within one target/season.
A unit must have a complete valid 23-quantile ensemble forecast, a held-out B0
forecast, and a finite observation in the pinned hub truth. We do not score dates
or locations simply because another model submitted them, and do not include a
target/season lacking comparable ensemble output. Pilot/off-season dates are
included only if the ensemble submitted matching units; this is explicitly not a
second hand-selected challenge calendar.

The time mapping is checked exactly: B0's lead 1-4 after context end maps to hub
horizon 0-3 with `reference_date = context_end + 7 days`. Target dates must remain
in the held-out season. Existing B0 admission quantile rounding is preserved;
hub quantiles are scored as submitted. ED values remain proportions, not percents.

Truth is the latest **full release per target** in the pinned canonical
`target-data/time-series.csv` (FluSight) or `.parquet` (COVID, RSV). Missing rows
are not filled from older releases. All models are scored against this same
truth, and differences from the earlier finalized CV labels are counted. The
export's original missing label does not forbid scoring a prediction if the
hub's frozen truth now provides an observation; such additional cells are counted.

Only the current CDC RSV hub is used, per the user's explicit instruction. There
is no older RSV-NET/catchment comparison. COVID means the current Forecast Hub,
not the Scenario Modeling Hub. Both hospitalization and ED channels are included
for COVID and RSV where the official ensemble supports them. Earlier unavailable
seasons are explicitly reported.

## Best-model selection and interpretation

"Best" is the lowest mean WIS among non-baseline submitted competitors covering
**100% of the exact ensemble-defined task set**. The official ensemble itself
is excluded from winning because it is plotted separately. Other submitted
ensemble models may compete. All available submitted models are scored on the
ensemble support they cover and listed with coverage; incomplete models are not
eligible to win by omitting difficult forecasts. If there is no complete eligible
competitor, submissions with at least 90% coverage are ranked on their identical
shared ensemble units. The three-model table and plots use that explicitly marked
subset, while full ensemble-supported scores remain available. If even that set
is empty, no winner is assigned. This fallback was introduced because the ED and
RSV archives have no fully complete individual competitor on the requested support.

The ranking pools available native locations including US, as does the headline
comparison table. US is not reconstructed by summing states. Because count-scale
US values can dominate averages, the CSV outputs also separate states/DC from US
and retain each location/horizon. No ranking is performed across targets with
different units. This "best complete competitor" need not equal the winner of an
official hub leaderboard, which may use relative WIS, other dates, or different
handling of incomplete submissions.

B0 uses finalized inputs and retrospective season cross-validation; hub models
were submitted with contemporaneous information. Evaluation of the first two
seasons trains on later seasons. Only the third fold is chronological, and its
inputs are still finalized. These comparisons describe performance under that
information advantage; they are not an operational leaderboard claim.

## Saved artifacts and validation

Each available target/season directory contains:

- `units.parquet`: the exact ensemble-derived scored task set and truth audit.
- `quantiles.parquet`: normalized wide forecasts for all scored models.
- `quantiles.csv.gz`: the actual long-form R scoringutils input, all 23 quantiles.
- `scores.csv`: per-unit R scores, including WIS, median AE, coverage 50/95,
  overprediction, underprediction, dispersion, and bias.
- `leaderboard.csv`: mean scores, task counts, coverage, and eligibility for best.
- `scores_by_horizon.csv` and `scores_by_location.csv`: subgroup diagnostics.
- `comparison_units.parquet`: identical display/ranking support, possibly smaller
  than the full scored set when the >=90% coverage fallback applies.
- `comparison.json`, `r_versions.txt`, `scoringutils.log`: selection and provenance.

The top-level `manifest.json` records pinned hub commits, full truth release dates,
source/code hashes, unavailable comparisons, and scoring rules.
`comparison_summary.csv` combines B0, best complete competitor, and ensemble.
Hub cache audits count invalid/incomplete/crossing quantile tasks, conflicting
observations, malformed dates, and unmaterialized LFS payloads. Invalid tasks are
excluded as whole tasks, not repaired by interpolation or rearranging quantiles.

Tests check extraction validation and the actual scoringutils result against an
independent pinball calculation. The PDF is rendered and inspected before delivery.

To regenerate only selection/summary tables from existing R scores (no refit or
rescoring):

```bash
PYTHONPATH=src .venv/bin/python -m tapestry.evaluation.summary data/evaluation/b0_hub_comparison
```

## Completed comparison

The first run scored nine supported target/season combinations with scoringutils
2.1.1. All locations including US are pooled below. Asterisked rows use the
explicit common-unit fallback for best-model comparison; full ensemble scores
remain in each case directory. Lower WIS is better.

| Target | Season | Best eligible competitor | B0 WIS | Best WIS | Ensemble WIS | Compared units |
|---|---|---|---:|---:|---:|---:|
| wk inc flu hosp | 2023-2024 | MIGHTE-Nsemble | 81.00 | 61.53 | 63.06 | 6,234 |
| wk inc flu hosp | 2024-2025 | FluSight-lop_norm | 168.36 | 165.78 | 177.57 | 5,616 |
| wk inc flu hosp | 2025-2026 | OHT_JHU-nbxd | 123.95 | 98.45 | 117.10 | 5,824 |
| wk inc flu prop ed visits | 2025-2026 | CMU-TimeSeries | 0.005174 | 0.005541 | 0.005792 | 4,660* |
| wk inc covid hosp | 2024-2025 | CEPH-Rtrend_covid | 48.31 | 54.89 | 42.21 | 6,968 |
| wk inc covid hosp | 2025-2026 | CMU-TimeSeries | 33.52 | 31.21 | 27.80 | 9,048 |
| wk inc covid prop ed visits | 2025-2026 | UGA_flucast-INFLAenza | 0.000577 | 0.000423 | 0.000463 | 5,122* |
| wk inc rsv hosp | 2025-2026 | UGA_flucast-INFLAenza | 35.96 | 40.80 | 29.01 | 5,824* |
| wk inc rsv prop ed visits | 2025-2026 | UGA_flucast-INFLAenza | 0.000769 | 0.000904 | 0.000824 | 5,350* |

Pooling states/DC with US, B0 beats the ensemble for influenza admissions in
2024-25 but not the other two influenza seasons. Separately, state/DC influenza
WIS is approximately 14%, 25%, and 17% lower than the ensemble across the three
seasons; native US is worse in all three and drives the weaker pooled results.
See [the geographic breakdown](../design/b0-next-experiments.md#diagnostic-correction-national-versus-state-performance). It is worse for COVID admissions in both available seasons and
RSV admissions in 2025-26. The displayed matched ED comparisons favor B0 for flu
and RSV, but favor the ensemble for COVID. These are finalized-CV comparisons
with an information advantage over prospective hub models, as explained above.

The PDF has 20 pages: two summary/method pages and 18 plot pages. It shows US and
NC, all four horizons, and only median/50%/95% forecast summaries.
