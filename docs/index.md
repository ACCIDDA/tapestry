# Tapestry

Tapestry is a research pipeline for multi-disease epidemic forecasting. The
Python package and commands are named `tapestry`.

```text
Raw snapshots → shared selection → canonical weekly dataset → B0 → evaluation
                        └────────→ local explorer
```

The model uses six channels: NHSN admissions and NSSP ED proportions
for influenza, COVID-19, and RSV, with explicit missingness masks. Training,
prediction, season cross-validation, hub scoring, and configuration comparison
are available. The broader vintage-aware architecture remains a research proposal;
the current model uses finalized retrospective data.

Start with [Getting started](getting-started.md) and the
[canonical dataset](data/build-b-finalized.md). Current commands are in
[training](workflows/training.md) and [full EpiBench evaluation](workflows/configuration-evaluation.md).
The [frozen hub support](workflows/hub-evaluation.md) page documents how the
frozen evaluation task set is built.

For data inspection, use the [explorer](explorer/index.md). Raw immutable
snapshots retain exact inputs and provenance; the explorer's derived index can
be rebuilt. Training reads the saved canonical dataset independently.

[B0.0](results/b0-crosses/index.md) reports the completed post-scaling crosses
experiment. [B0.1](design/b0.1.md) specifies reference crosses comparing spatial and
cross-target attention, parameter sharing, temporal representations, and
uncertainty decoders, using the adopted season-first location-relative objective.
B0.1 has not been launched.
[Architecture](design/architecture.md) describes the proposed extensions.
[Features and tests](maintenance.md) distinguishes required from optional
features.
