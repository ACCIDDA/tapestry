# Hub targets, submission formats, and scoring for 2026–27

Checked 5 October 2026. “This season” means 2026–27. The three hubs are CDC FluSight, CDC COVID-19 Forecast Hub, and CDC RSV Forecast Hub. “Forms” is interpreted as prediction representations and submission file formats. This is a review of published rules and evaluation code, not a model experiment; no training inputs, labels, model parameters, or evaluation inputs were changed.

## What is established about scoring

| Hub | Published scoring rules | Evidence about logarithms | Limit of the evidence |
| --- | --- | --- | --- |
| FluSight | WIS, its components, and interval coverage. At least 75% of available hospital-admission quantile forecasts, excluding horizon −1, are needed for official end-of-season inclusion. | The current submission guide says previous final rankings used log-transformed data. CDC's 2025–26 report explicitly scores naturally log-transformed admission counts. The Reich Lab dashboard configuration requests both native-scale and `log(value + 1)` scoring for admissions and ED proportions. | The dashboard still lists 2025–26 and older season evaluation sets. Its offset of 1 does not prove the offset in CDC's separate season report, nor a complete 2026–27 final-ranking protocol. |
| COVID-19 | The submission guide specifies WIS among the evaluation metrics. | The Reich Lab dashboard configuration requests both native-scale and `log(value + 1)` scoring for admissions and ED proportions. It also requests median absolute error and 50%/95% interval coverage, with relative metrics against CovidHub-baseline. | This is a rolling dashboard, not an explicit 2026–27 end-of-season ranking specification. |
| RSV | The submission guide specifies WIS among the evaluation metrics. | No explicit native-versus-log instruction or corresponding evaluation dashboard configuration was found in the reviewed public sources. | Unspecified does not mean native-scale. Do not claim either scale as confirmed. |

Sources: [FluSight evaluation rules][flu-output], [CDC 2025–26 evaluation][cdc-report], [FluSight dashboard configuration][flu-eval], [COVID submission rules][covid-output], [COVID dashboard configuration][covid-eval], [RSV submission rules][rsv-output].

WIS is lower-is-better and measures interval width together with penalties when observations fall outside prediction intervals. Applying a logarithm to observations and forecast quantiles **before** calculating WIS differs from taking a logarithm of an already calculated WIS, and from a logarithmic probability score for categorical forecasts.

For the dashboard configurations, `transform_defaults` sets `log_shift`, `offset: 1`, and `label: log`. Both targets inherit it. The inspected [dashboard evaluation implementation][eval-code] defaults to appending transformed scores alongside native-scale scores; its [transform resolver][transform-code] maps `log_shift` to `scoringutils::log_shift`. Thus the configuration means both scales, not that all displayed WIS is necessarily logarithmic. For ED, the input is a decimal proportion: the configured transformation is `log(1 + p)`, not `log(1 + 100*p)`.

The CDC **2025–26** hospital-admission report uses geometric aggregation of pairwise mean-WIS ratios on shared forecast tasks, normalized to FluSight-baseline. It excludes national and Puerto Rico forecasts from the main ranking and uses truth published July 1, 2026. These are historical rules, not assumptions about the coming season. The report does not specify the zero-handling offset. Its ED evaluation is described as forthcoming.

## Targets and prediction representations

| Hub | Exact target | Units submitted | Representation |
| --- | --- | --- | --- |
| FluSight | `wk inc flu hosp` | Weekly laboratory-confirmed hospital admission count; nonnegative integers required by the 2026–27 prose instructions | 23 quantiles; optionally 100 connected sample trajectories |
| FluSight | `wk inc flu prop ed visits` | ED visit proportion between 0 and 1 | 23 quantiles; optionally 100 connected sample trajectories |
| FluSight | `wk flu hosp rate change` | Probability of each admission-rate-change category | `pmf`: large decrease, decrease, stable, increase, large increase; probabilities sum to 1 |
| FluSight | `peak week inc flu hosp` | Probability of each candidate peak week | `pmf` over the Saturday dates in the current task configuration; probabilities sum to 1 |
| FluSight | `peak inc flu hosp` | Highest weekly admission count in the season; nonnegative integers | Quantiles |
| COVID-19 | `wk inc covid hosp` | Weekly laboratory-confirmed hospital admission count | 23 quantiles; optionally 200 sample trajectories |
| COVID-19 | `wk inc covid prop ed visits` | ED visit proportion between 0 and 1 | 23 quantiles; optionally 200 sample trajectories |
| RSV | `wk inc rsv hosp` | Weekly laboratory-confirmed hospital admission count | 23 quantiles; optionally 200 sample trajectories |
| RSV | `wk inc rsv prop ed visits` | ED visit proportion between 0 and 1 | 23 quantiles; optionally 200 sample trajectories |

Sources: each hub's [FluSight][flu-readme], [COVID][covid-readme], and [RSV][rsv-readme] overview, submission guide, and task configuration ([flu][flu-tasks], [COVID][covid-tasks], [RSV][rsv-tasks]).

FluSight calls all targets optional, with admissions the primary target. COVID and RSV overview instructions require admission quantiles for each submitted location/horizon; ED and samples are optional. Their machine-readable configurations are more permissive (`is_required: false`), so follow the prose requirement rather than interpreting successful validation as authorization for ED-only submission. FluSight's admission-quantile schema allows doubles despite its explicit new integer requirement; follow the integer instruction.

These are accepted submission targets. Acceptance does not establish inclusion in one official aggregate score. No complete 2026–27 scoring specification for FluSight peak, trend, and trajectory predictions was identified. Do not assume WIS applies to categorical probabilities, or that optional samples determine the main ranking. FluSight says tied peak weeks are not scored for that location.

Hospital labels are reported NHSN admissions; ED labels are NSSP disease-attributed visits divided by total ED visits. They are surveillance reports, not latent infection counts. Convert source percentages to proportions by dividing by 100. Submit native units even when the evaluator later transforms them.

## Weekly schedule and file format

All three hubs use Wednesday 11 PM US Eastern deadlines, with the following Saturday as `reference_date`. Weekly horizons are −1, 0, 1, 2, and 3, with `target_end_date = reference_date + 7 * horizon` days. FluSight explicitly excludes admission horizon −1 from summary evaluation. COVID's inspected dashboard evaluates horizons 0–3; the reviewed RSV instructions do not establish its final scoring-horizon exclusions.

FluSight's challenge runs October 7, 2026–May 26, 2027. RSV began September 23, 2026. COVID operates on a rolling schedule. National, state, DC, and Puerto Rico submissions are described, subject to target-data availability; accepted locations need not match locations included in rankings.

Submit CSV or Parquet under `model-output/team-model/`, named `YYYY-MM-DD-team-model.csv` or `.parquet`, using the Saturday reference date. The eight columns are:

```text
reference_date,target,horizon,target_end_date,location,output_type,output_type_id,value
```

Use zero-preserving FIPS strings and `US`. A quantile row uses `output_type=quantile` and its probability as `output_type_id`. Required weekly quantile levels are:

```text
0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
0.85, 0.90, 0.95, 0.975, 0.99
```

Sample IDs identify connected trajectories across horizons. FluSight specifically rejects independently drawing each horizon from marginal quantiles as a substitute for trajectory samples. Seasonal peak targets leave `horizon` and `target_end_date` blank/NA; their probability support or quantile level goes in `output_type_id`.

## Existing Chromantis work and implications

The [earlier FluSight review](flusight-2025-2026-model-review.md) and [Google comparison](../experiments/google-comparison-20261005/results.md) already identify the log-scale issue for the previous season. They do not establish a common current-season scoring protocol for all three hubs. Today's additional finding is the explicit offset of 1 in the two dashboard configurations, with both scales retained.

Chromantis's [current ranking](../workflow.md) scores native-unit predictions relative to the frozen Hub **ensemble**, gives US 20% and equally weighted states/DC 80%, weights admissions 1 and ED 0.5, and averages seasons equally. This is our research objective, not a documented official three-hub ranking. There is no evidence here for a shared official six-target aggregate with those weights.

For future protocol-matched scoring, keep native and log results explicitly separate; identify the target, evaluation dates, truth vintage, locations, eligible models, and comparator. An offset of 1 is now supported for reproducing the inspected dashboards, but remains unconfirmed for the CDC final FluSight report and for RSV. Rescoring saved quantiles changes evaluation only; it does not retrain a model or change its training labels. This review makes no scoring-code changes.

Decision log, 2026-10-05: distinguish official season reports from dashboard settings and submission eligibility; do not silently replace the existing research score or extend FluSight's historical rules to RSV/COVID. Remaining unknowns are the final 2026–27 ranking protocols, RSV transformation and aggregation, scoring of optional FluSight outputs, and end-of-season truth cutoffs.

[cdc-report]: https://www.cdc.gov/flu-forecasting/evaluation/2025-2026-report.html
[flu-readme]: https://github.com/cdcepi/FluSight-forecast-hub/blob/5b898bd2afafaeee153556558c08d0bc28a00955/README.md
[flu-output]: https://github.com/cdcepi/FluSight-forecast-hub/blob/5b898bd2afafaeee153556558c08d0bc28a00955/model-output/README.md
[flu-tasks]: https://github.com/cdcepi/FluSight-forecast-hub/blob/5b898bd2afafaeee153556558c08d0bc28a00955/hub-config/tasks.json
[covid-readme]: https://github.com/CDCgov/covid19-forecast-hub/blob/43581db5e1778d69ac136a7c7f2386223b8a9b07/README.md
[covid-output]: https://github.com/CDCgov/covid19-forecast-hub/blob/43581db5e1778d69ac136a7c7f2386223b8a9b07/model-output/README.md
[covid-tasks]: https://github.com/CDCgov/covid19-forecast-hub/blob/43581db5e1778d69ac136a7c7f2386223b8a9b07/hub-config/tasks.json
[rsv-readme]: https://github.com/CDCgov/rsv-forecast-hub/blob/c26d60093be995442f3fe2ab2f64a6fe813a1104/README.md
[rsv-output]: https://github.com/CDCgov/rsv-forecast-hub/blob/c26d60093be995442f3fe2ab2f64a6fe813a1104/model-output/README.md
[rsv-tasks]: https://github.com/CDCgov/rsv-forecast-hub/blob/c26d60093be995442f3fe2ab2f64a6fe813a1104/hub-config/tasks.json
[flu-eval]: https://github.com/reichlab/flusight-dashboard/blob/430681eef0c3b88f98c730089a5949d23595bfd7/predevals-config.yml
[covid-eval]: https://github.com/reichlab/covidhub-dashboard/blob/a3479b3274c3a74e890d6859dfb169554fa9e400/predevals-config.yml
[eval-code]: https://github.com/hubverse-org/hubPredEvalsData/blob/b80dca64c4f41623097dbd86edafc1caad6edce1/R/generate_eval_data.R
[transform-code]: https://github.com/hubverse-org/hubPredEvalsData/blob/b80dca64c4f41623097dbd86edafc1caad6edce1/R/utils-transform.R
