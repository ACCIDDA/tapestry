# Filters and interpretation

The explorer groups choices as logical source → signal, with a checkbox for each
available release/provider/geography variant under each signal. NHSN and NSSP each combine their source
products into one group; each hub retains its own group and target definitions.
See [Shared selection](../data/selection.md) for the measure and file allowlists.
Searching covers providers, dataset titles, source paths, column names, and
complete series labels. Search and filters can restrict the available variants.
Delphi variants are highlighted in the selector and can be selected alongside
CDC variants from the same signal.

## Signal filters

| Filter | Meaning |
|---|---|
| Cadence | Catalog `temporal_resolution`: daily, weekly, monthly, or sample-based |
| Revision history | Whether the catalog marks the source as historically versioned |
| Spatial support | Native state data or an explicitly labeled national context |
| Data currency | Whether the last event date falls within its cadence-specific window |

Currency thresholds are 14 days for daily, 35 for weekly, 75 for monthly, and
45 for sample-based data. “Current” describes the indexed event dates; it is not
a guarantee that a publisher will continue maintaining a product.

Filters are evaluated by the local API before pagination. The API reports
variant counts; the UI additionally reports the distinct signals among returned
variants. The UI requests all matching variants. Native-state support is the
default and Reset restores it. Choose Any support or National broadcast for
national-only demographic and RSV-NET products.

## Series interpretation

Low-cardinality fields such as pathogen, subtype, model, target, unit, and
quantile define separate dimension variants. Publisher `season`, `respseason`,
and `season_week` fields are metadata and do not artificially split a covariate
across seasons.

Repeated raw rows for one state, series, event date, and latest vintage are shown
as an unweighted mean. Tooltips expose the contributing row count. County, HSA, HHS, catchment, and wastewater-site rows are excluded by the shared
post-intake filter before indexing.

## Scaling

**Divide each series by its max** scales each selected series independently for
visual comparison. It does not alter the index or raw data, and it is not the
normalization contract for model training.
