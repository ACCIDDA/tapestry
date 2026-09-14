# Data repository

The `tapestry.data` package is the durable acquisition layer. It separates
source-specific retrieval from a shared snapshot repository:

```text
tapestry.data
├── catalog.py          Dataset specifications and groups
├── cli.py              Reproducible command-line entry point
├── geography.py        Native state/national geography
├── http.py             Retrying and streaming HTTP client
├── models.py           Dataset and manifest value objects
├── repository.py       Atomic immutable snapshot writer
├── tables.py           Shared streaming raw-table readers
├── selection.py        Post-intake selection and canonical signal groups
└── sources/
    ├── socrata.py      CDC pagination and consistency checks
    ├── delphi.py       Delphi V5 via epidatpy + raw CSV
    └── hubverse.py     Read-only Git history and exports
```

## Core guarantees

1. Publisher-native values are retained; acquisition does not impute missing
   values or coerce suppression markers to zero.
2. Each committed payload has a SHA-256 record in its manifest.
3. `latest.json` advances only after a complete snapshot commits.
4. Version semantics and native geographic resolutions live in the catalog.
5. Shared selection admits native state and national observations; regional,
   county, catchment, and site observations are not relabeled as state data.

## Public boundary

Applications normally use `RawDataRepository`, `DatasetSpec`, and `CATALOG`:

```python
from tapestry.data import CATALOG, RawDataRepository

repository = RawDataRepository("data")
repository.initialize(CATALOG)
manifest = repository.latest("cdc_nhsn_final")
repository.verify_snapshot(manifest)
```

Fetcher classes are kept under `tapestry.data.sources`; downstream modeling
code should consume repository snapshots rather than call publishers directly.

## Shared consumer selection

The explorer and downstream analysis share [post-intake selection](selection.md):
25 raw datasets become 15 logical source groups, with explicit outcome allowlists.
Use `SelectedData` for native selected records; raw acquisition remains unchanged.
