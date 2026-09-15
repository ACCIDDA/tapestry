# Features and tests

## Which data features are optional?

`model_data/finalized.py` builds the six-channel panel from only `cdc_nhsn_final`
and `cdc_nssp_trajectories`. `models/run.py` and `models/season_cv.py` load the
saved NPZ. Neither requires the explorer, Delphi, or historical raw snapshots
while fitting.

A canonical training artifact and raw snapshots serve different purposes.
The artifact gives every experiment the same tensor/masks; raw snapshots and
hashes explain where it came from and allow you to rebuild it. Publisher revision
archives can additionally support comparisons of preliminary versus later values.
A retrieval snapshot alone does not reconstruct revisions the publisher never saved.
The model is a finalized retrospective experiment even when raw revision archives
are present.

| Candidate | Observed use | Recommendation |
|---|---|---|
| All-source catalog and broad downloads | 25 catalog entries versus two sources used by this model; other sources support comparison and exploration | Research instructions default to the two needed sources. Retain adapters until those research questions are retired. Most catalog code is declarations. |
| Explorer schema discovery, filters, revision ledger and as-of queries | Useful for multi-source inspection; absent from the training path | Optional application. Skip building/running it for model-only work. Replacing it with a panel viewer would sacrifice source/revision comparison. |
| Snapshot checksums, staging, atomic publication | Prevent incomplete downloads becoming training inputs and record exact provenance | Keep. These protect reproducibility even with one canonical training dataset. |
| Delphi retries, resumable partitions and concurrency | Needed for large archive pulls; irrelevant to a two-CDC-source-only workflow | Keep while Delphi archives are supported; avoiding those downloads removes runtime cost without changing the adapter. |
| Source lineage and native geography filtering | Shared selection uses these to avoid false equivalence between measures, providers and geographic support | Keep. A state label on a county/site/HHS record does not make it a state observation. |
| PDF reports, EpiBench diagnostics and CSV companions | Evaluation presentation/interoperability rather than model fitting | Optional outputs. PDF is separate; separating EpiBench plotting from `sweep` scoring would be a useful simplification. |

These are source-based recommendations, not proof that no external notebook uses
an API.

## What the tests do

The suite keeps only tests that protect reported results. GitHub Actions
(`.github/workflows/tests.yml`) runs it on pushes and pull requests to `main`.

| File | What it catches |
|---|---|
| `test_model_data.py` | Window/target alignment, masks, observed zeros versus missing values, season boundaries, builder units and conflicting rows |
| `test_season_cv.py` | Held-out season values cannot change training inputs, labels or scales; WIS; five-quantile selection from saved archives |
| `test_b0.py` | Fair CRPS mathematics, missing-label exclusion, masked inputs not changing predictions, location ordering, population-transform inversion, checkpoint round trip, separate state/US head gradients |
| `test_hub_evaluation.py` | Invalid quantile tasks excluded, agreement with R `scoringutils`, equal scoring support for best-model selection, horizon/channel/FIPS export, ranking on identical tasks |
| `test_evaluation_sweep.py` | Seed averaging and target weighting in configuration ranking, stable configuration identities, Hubverse round trip, frozen-task and truth matching, equal-geography objective, end-to-end R/EpiBench sweep |
| `test_experiment_manager.py` | Short scenario strings round-trip and reject typos, CLI flags reproduce scenarios, one-factor suites, plan/run/resume/status with relative paths and commits |
| `test_epibench_pipeline.py` | EpiBench scoring with numeric FIPS and a zero reference, WIS values, refusal to reuse scores for changed inputs or missing tasks |

### Focused commands

Run from the repository root after `uv sync`.

```bash
# All tests; local fixtures, no publisher downloads.
uv run pytest -q

# Model/data changes: scientific correctness and leakage.
uv run pytest -q tests/test_model_data.py tests/test_b0.py tests/test_season_cv.py

# Scoring/export changes.
uv run pytest -q tests/test_hub_evaluation.py tests/test_evaluation_sweep.py tests/test_epibench_pipeline.py
```

Four integration tests invoke R `scoringutils` and/or EpiBench. They skip if
Rscript or EpiBench is absent; an installed Rscript without `scoringutils` and
`purrr` still fails, so run `Rscript scripts/setup_r.R`. Tests use temporary
fixtures and do not train the full research sweep.
