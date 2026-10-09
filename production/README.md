# Production: FluSight 2026-27 submissions

Code and records for the real-time weekly submissions to the CDC FluSight
Forecast Hub. Research code stays in `src/chromantis`; this folder holds only what
turns a trained model into a submitted file.

The [Chromantis submission log](../docs/submissions.md) covers all pathogens and model
recipes across successive submissions. Its first entry records the merged forecast for
10 October 2026: the three-recipe B7 model submitted as ACCIDDA-EpiLoom, replacing
the on-time eight-recipe model. It also identifies the outdated metadata and
fallback definition. Hub metadata changes are deferred until the recipe is settled.
The [reusable workflow proposal](../docs/reusable-research-workflow.md) traces the
Claude and Codex development conversations, human and LLM responsibilities, and
the planned package cleanup.

## Layout

- `releases/<name>.json` — a frozen model: its recipes, the exact fitted checkpoint of
  every seed with the SHA256 of its `model.pt`, the combination rule, the input view,
  the public model abbreviation and a description of training seasons, input
  treatments and labels. `releases/b7-20261007.json` is the merged B7 model;
  `releases/system2-20261007.json` the on-time System2 model it replaced.
  Checkpoints are copied from Longleaf to `data/production-fits/` (git-ignored); models
  are small MLPs and replay runs on a laptop CPU.
- `submissions/<reference date>/` — every file we submitted, under its Hub name, with its export record,
  interval PDFs and peer-comparison PDF; `superseded/` keeps replaced files the same way.
  The [submission log](../docs/submissions.md) records each issuance.
- `hubs/` (git-ignored) — two working clones, created 2026-10-07:
  - `hubs/FluSight-forecast-hub/` — `cdcepi/FluSight-forecast-hub`, read-only.
    Used for the current `hub-config/tasks.json`, `auxiliary-data/locations.csv`,
    `target-data/`, peer forecasts, and to validate files before opening a PR.
  - `hubs/FluSight-forecast-hub-fork/` — our fork `jcblemai/FluSight-forecast-hub`
    (`origin`), with `upstream` = `cdcepi`. PRs are opened against `cdcepi/FluSight-forecast-hub:main`.
  These are separate from `data/mirrors/hub_flusight_current.git`, the bare
  mirror the research data pipeline reads with Git publication times.
- `model-metadata/ACCIDDA-<model>.yml` — our metadata drafts.
- `output/` (git-ignored) — per-issuance forecasts, CSV, provenance JSON and plots.

The code is `src/chromantis/production/` (`python -m chromantis.production`).

## Weekly submission

```bash
# 1. refresh raw sources into an operational data root, then build its panel
#    (never overwrite the frozen research panel data/processed/panel.npz)
.venv/bin/python -m chromantis.dataset.build build --data-root data/operational-<date> --workers 2 \
    --output data/operational-<date>/processed/panel.npz
# 2. refresh merged peer submissions and reported history before plotting
git -C production/hubs/FluSight-forecast-hub pull --ff-only
# 3. forecast from the release, export, plot intervals and peer comparison (no fitting; nothing published)
.venv/bin/python -m chromantis.production run --release production/releases/b7-20261007.json \
    --dataset data/operational-<date>/processed/panel.npz --issuance <wednesday>
# 4. review production/output/<wednesday>/<release>/, then push a safe file before the
#    deadline (11 PM ET Wednesday) and open the PR
cd production/hubs/FluSight-forecast-hub-fork && git fetch upstream && git checkout -B submit-<reference> upstream/main
mkdir -p model-output/ACCIDDA-Chromantis && cp ../../output/<wednesday>/<release>/<reference>-ACCIDDA-Chromantis.csv model-output/ACCIDDA-Chromantis/
git add . && git commit -m "ACCIDDA-Chromantis <reference>" && git push -u origin submit-<reference>
gh pr create --repo cdcepi/FluSight-forecast-hub --head jcblemai:submit-<reference> --fill
```

Then record it (the CSV, its export record, interval PDFs and peer comparison, drawn from the
local Hub clone) and add the issuance to [docs/submissions.md](../docs/submissions.md):

```bash
.venv/bin/python -m chromantis.production record production/output/<wednesday>/<release>/<reference>-ACCIDDA-Chromantis.csv
```

A replacement after the deadline is the user's decision; record the replaced file with
`--superseded` (and `--as <Hub file name>` when the name differs).

`run` replays every released checkpoint (`forecast.py`: the saved network, input scales
and correction trees on the issuance's reported inputs; future labels must be absent;
each `model.pt` must match its released SHA256), then `export.py` (equal weight per
recipe and per seed, the release's rule, shared with research ensembles; checks the Hub
task list, quantile grid, units, ordering and horizon dates; caps ED at the Hub's 0.25),
then `intervals.py` (one PDF per target, one row per location, whole season with the
four previous seasons dashed and a zoom on the last 12 weeks, with the submitted
50/80/95% intervals and the Hub's latest target data) and `peers.py`. Horizons 0–3 only
(−1 is optional and unscored). `--compare-csv` overlays saved local candidates in the
peer comparison; `--no-plots` skips the figures.

`peers.py` writes `<CSV-stem>-comparison/` beside the exported file, with:

- `comparison.pdf`: one page per location, admissions/ED rows and full-season/zoom
  columns. Defaults are US, California, Texas, New York, North Carolina and Iowa.
- `comparison-<FIPS>.png`: the same pages as images, including `comparison-US.png`.
- `forecast-medians.csv`, `history.csv`, and `positions.csv`: plotted data and each
  local model's forecast-level percentile among available peers, not an accuracy rank.
- `manifest.json` and `README.md`: exact Hub commit, input hashes, model export notes,
  missing peers and the comparison assumptions.

The forecast date is read from the CSV. Merged same-week CSV and Parquet forecasts
are loaded from the local Hub's committed `HEAD`; the exact resolved commit is saved.
The comparison itself does not fetch or modify the Hub checkout. A run with no peers still
plots local forecasts and history and explicitly reports that peer bands are absent.
Missing location/target forecasts are never imputed. The default individually drawn
peers are Google_SAI-FluEns, OHT_JHU-nbxd, CMU-TimeSeries, UGA_flucast-INFLAenza,
UMass-flusion and NAU-vulPES when available; this is a fixed comparison list, not an
automatic assertion of current-season rank. The gray band and peer median use all
available Hub model medians equally, excluding the local candidates and exact copies.
History is the latest reported data at the pinned Hub commit, which can include
revisions; it is not a reconstruction of what the model saw. The gray band describes
disagreement among medians, not predictive uncertainty.

To make a new release: fit the selected recipes with `evaluation_seasons=production`
(study file + planner), copy the `eval_2026-2027` folders to `data/production-fits/`,
and write `releases/<name>.json` with each checkpoint's SHA256 and a description of the
training seasons, input treatments, labels and combination rule.

### Regenerate comparisons from existing CSVs

No training, inference or scoring is needed. The standalone command accepts one
or more local forecast files and can be rerun as peers arrive:

```bash
.venv/bin/python -m chromantis.production peers \
    production/submissions/2026-10-10/superseded/2026-10-10-ACCIDDA-EpiLoom.csv \
    output/b7/submission-20261007/2026-10-10-ACCIDDA-EpiLoomB7.csv \
    --labels 'EpiLoom on-time (8 recipes)' 'B7 (3 recipes)' \
    --locations US CA TX NY NC IA \
    --out production/output/2026-10-07/peer-comparison
```

Use `--peer-models MODEL ...` to choose individual peer curves and `--zoom-weeks 6`
to narrow the zoom. To include a downloaded pending peer submission, pass
`--peer-csv /path/YYYY-MM-DD-TEAM-MODEL.csv` (repeatable) to `peers`. Its original Hub filename identifies the peer model. Pending
PRs are not downloaded implicitly. If that same model already exists in the pinned
Hub commit, choose one version explicitly instead of counting it twice.

Two submission files can also be compared task by task (medians, 95% widths, ratios)
with `python -m chromantis.production diff SUBMITTED.csv NEW.csv OUTDIR --labels A B`, and
overlaid with `python -m chromantis.production intervals A.csv B.csv --labels A B`.

Decision log, 8 October 2026: make the peer comparison a standard final step of
submission generation; retain the existing uncertainty PDFs. Same day: the production
scripts (`make_submission.py`, `plot_submission.py`, `compare_submission.py`,
`scripts/forecast_b6.py`, `export_b6.py`, `compare_submissions.py`) moved into
`src/chromantis/production/`, and checkpoint globs (`models/*.json`) were replaced by
release files with checkpoint hashes. Re-exporting the 7 October forecasts through the
new code reproduced the submitted B7 CSV byte for byte.

## Hub rules that the submission code must respect (2026-27 README)

- Due 11 PM ET every Wednesday, 2026-10-07 through 2027-05-26. The file is named
  by the `reference_date` = Saturday ending the epiweek containing the due date.
- `target_end_date = reference_date + 7 * horizon`, horizons -1..3 (-1 is not
  scored but is encouraged).
- `wk inc flu hosp`: quantiles, integer values. `wk inc flu prop ed visits`:
  quantiles as proportions in [0, 1] (not percent). Optional `sample` output
  type needs exactly 100 temporally joined samples per task.
- Other optional targets: `wk flu hosp rate change` (pmf), `peak inc flu hosp`,
  `peak week inc flu hosp`.
- Model metadata goes in `model-metadata/<team>-<model>.yml`; our previous model
  there is `UNC_IDD-InfluPaint`.

## Weekly refresh

```bash
git -C production/hubs/FluSight-forecast-hub pull --ff-only
git -C production/hubs/FluSight-forecast-hub-fork fetch upstream
git -C production/hubs/FluSight-forecast-hub-fork checkout main
git -C production/hubs/FluSight-forecast-hub-fork merge --ff-only upstream/main
```

## Open decisions

### EpiLoom metadata prepared 2026-10-07

`model-metadata/ACCIDDA-EpiLoom.yml` is the metadata draft for the requested
EpiLoom name, also copied into the local Hub fork. It assumes the existing
`models/Tapestry.json` fallback (deleted 9 October 2026): two neural-network recipes, five seeds each,
trained on 2022-23 through 2025-26 to predict finalized flu admissions and ED
proportions, for prospective 2026-27 forecasts. One recipe uses Kinsa and
artificially degraded training histories corrected with historical report-to-final
trees; the other predicts quantiles with neighbor information exchange. Quantiles
are averaged equally across recipes and then seeds. Final model selection remains
open; revise the methods if that selection changes. Contributors, output license, version and designated-model status are retained
from the existing draft. Funding lists only the CDC Insight Net cooperative
agreement; the NIH reference was removed at the user’s request.
The debug forecast definition and output names have not been renamed.

The fork's `.github/workflows/validate-submission.yaml` skips forks via
`if: github.event.repository.fork != true`. Testing that workflow within the fork
requires removing that guard in a fork-only change; copying metadata alone does
not run GitHub CI. No workflow change or remote publication was made here.

- Model name: metadata prepared as `ACCIDDA-EpiLoom` (2026-10-07).
- Which trained model(s) and ensemble are submitted, and which targets.
- `Tapestry` is a debug name (2026-10-07). Its definition is the fallback:
  B4 production A + B. The Hub allows up to two designated models per team,
  so two distinct ACCIDDA models can both enter the FluSight ensemble.
