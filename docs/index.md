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
[training](workflows/training.md) and [full EpiBench evaluation](workflows/configuration-evaluation.md).
The [frozen hub support](workflows/hub-evaluation.md) page records the original
comparison protocol.

For data inspection, use the [explorer](explorer/index.md). Raw immutable
snapshots retain exact inputs and provenance; the explorer's derived index can
be rebuilt. Training reads the saved canonical dataset independently.

The [canonical B0 report](results/b0-configuration-comparison.md) covers
14 variants and 42 runs. The best aggregate model, `conv_h12`, has 10.8% lower
influenza WIS and 8.1% lower three-admission WIS relative to ensemble parity
on the balanced selection objectives. The report includes seed variability,
matched controls, coverage, and projection fans.
[Design proposals](design/solution-b-plan.md) describe possible extensions.
The [code and test review](maintenance.md) distinguishes current needs from
optional features.
