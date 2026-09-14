# Storage and provenance

## On-disk layout

```text
data/
├── catalog.json
├── mirrors/
│   └── hub_flusight_current.git/
├── raw/
│   └── cdc_nhsn_final/
│       ├── latest.json
│       └── snapshots/<retrieval-id>/
│           ├── data.ndjson.gz
│           ├── metadata.json
│           └── manifest.json
├── .staging/
└── .explorer/
    ├── series.sqlite3
    └── revisions.parquet
```

`raw/` and `mirrors/` are durable source material. `.explorer/` is disposable
and can always be reconstructed. `series.sqlite3` contains searchable series
metadata plus the latest resolved point cache; `revisions.parquet` contains the
complete revision ledger used for as-of reads. `.staging/` contains in-progress work and may
also contain validated Delphi partitions retained after an interrupted pull.

## Atomic snapshots

A fetcher obtains a `SnapshotWriter`, streams its files into an isolated staging
directory, records row counts and metadata, and commits only after every file is
complete. Commit writes the manifest and atomically advances `latest.json`.

If a pull fails, an incomplete snapshot is not visible through `latest()`. Large
Delphi V5 pulls can deliberately preserve gzip CRC-valid partitions and reuse
them with `--resume-from`.

## Manifest contents

Each snapshot manifest records:

- dataset key and immutable retrieval identifier;
- retrieval time, source URL, and revision mode;
- whether the source exposes historical versions;
- each payload path, byte count, row count, and SHA-256 digest;
- fetcher-specific provenance such as source metadata or a Hub Git commit.

Verify the latest snapshot with:

```bash
python scripts/pull_covariates.py --data-root data verify cdc_nhsn_final
```

## Data lifecycle

Raw snapshots are append-only. Cleaning the disposable explorer index does not
remove source data. Before deleting staging or dated backup directories, inspect
them for resumable partitions or recovery material.
