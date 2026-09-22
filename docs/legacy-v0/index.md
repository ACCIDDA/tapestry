# Legacy (v0): before the unified model

These pages are the project's history before the 2026-09-22 unified-model
restructure ([design](../design/restructure-2026-unified.md)). They describe
separate B0, B0.1, B1 and B2 models, their datasets (`finalized.npz`,
`vintaged.npz`, `build_b2.npz`) and modules that no longer exist
(`tapestry.models`, `tapestry.model_data`, the experiment manager and backends,
the EpiBench/R scorer). Commands on these pages do not run on the current code;
recover the code from git history (the restructure starts at commit `4a0b172`).
Their results were produced by that code and are kept as the record of what
was tried and why the current design looks the way it does. They are not
comparable one-to-one with results from the current `planner rank` score
(user decision 2026-09-22 in `icare.md`: one score, per-location WIS ratios).

The current workflow is [Training and prediction](../workflows/training.md).

Moved here on 2026-09-22 from `docs/design/`, `docs/data/`, `docs/results/` and
`docs/archive/workflows-2026-09/` (now `workflows/`), keeping their relative layout.

## Design

- [B0.1 architecture crosses](design/b0.1.md)
- [B1: masked-vintage model](design/b1.md)
- [B2 covariate experiment](design/b2.md)
- [B2 interpretation and next-season training](design/b2-next-season.md)
- [Historical surveillance transfer into B1](design/historical-curve-transfer.md)
- [Archived architecture candidates](design/old%20LLM%20things/architecture-candidates.md)

## Data

- [B0 definition: finalized six-channel pilot](data/build-b-finalized.md)
- [B2 model data](data/b2.md)
- [Forward benchmark data](data/forward-2025.md)
- [What is available when we forecast?](data/reporting-availability.md)
- [Local measure inventory, 2026-09-13](data/local-inventory-2026-09-13.md)

## Results

- B0: [B0.0 crosses](results/b0-crosses/index.md),
  [follow-up](results/b0-crosses/next-steps.md),
  [B0.1 crosses](results/b0-1-crosses/index.md),
  [pre-scaling-fix crosses](results/b0-crosses-legacy/index.md)
- B1: [overview](results/b1-conclusions.md), [first good results](results/first-good-b1.md),
  [screen and 300-epoch](results/b1-overnight/index.md),
  [reporting regimes](results/b1-reporting-regimes/index.md)
- B2: [covariate screen](results/B2-screen/index.md),
  [benchmark report](results/B2-screen/benchmark/REPORT.md),
  [Kinsa-only](results/B2-kinsa/index.md)
- Forward 2025–26: [benchmark](results/Forward-2025/index.md),
  [analysis](results/Forward-2025/analysis.md),
  [analysis tables](results/Forward-2025/analysis-tables.md),
  [check](results/Forward-2025-check/index.md)

## Workflows (pre-restructure commands)

See [the archive README](workflows/README.md) for the list.
