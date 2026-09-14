# Tapestry

Tapestry is a research pipeline for multi-disease epidemic forecasting. The
Python package and commands are named `tapestry`; the checkout folder remains `influpaintX`.

```text
Raw snapshots → shared selection → canonical weekly dataset → B0 → evaluation
                        └────────→ local explorer
```

The implemented model uses six channels: NHSN admissions and NSSP ED proportions
for influenza, COVID-19, and RSV, with explicit missingness masks. Training,
prediction, season cross-validation, hub scoring, and configuration comparison
are available. The broader vintage-aware architecture remains a research proposal;
the current model uses finalized retrospective data.

Start with [Getting started](getting-started.md) and the
[canonical dataset](data/build-b-finalized.md). Current commands are in
[training](workflows/training.md), [hub evaluation](workflows/hub-evaluation.md),
and [configuration comparison](workflows/configuration-evaluation.md).

For data inspection, use the [explorer](explorer/index.md). Raw immutable
snapshots retain exact inputs and provenance; the explorer's derived index can
be rebuilt. Training reads the saved canonical dataset independently.

[Completed experiments](results/b0-full-experiments.md) record observations.
The [configuration evaluation report](results/b0-configuration-comparison.md)
includes rankings and projection fans for all 15 runs.
[Design proposals](design/solution-b-plan.md) describe possible extensions.
The [code and test review](maintenance.md) distinguishes current needs from
optional features.
