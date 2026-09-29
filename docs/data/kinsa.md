# Kinsa cough, cold and flu signal (PopHIVE)

`pophive_kinsa_ili` is the national daily share of Kinsa smart-thermometer users
who report cough, cold or flu symptoms. Tapestry takes it from the
[PopHIVE Ingest](https://github.com/PopHIVE/Ingest/tree/main/data/kinsa_ili/standard)
repository, not from Kinsa. The [PopHIVE site](https://pophive.github.io/Ingest/#kinsa-ili)
describes the same table.

| | |
|---|---|
| Catalog key / group | `pophive_kinsa_ili` / `pophive` |
| Upstream | Kinsa Insights API, `/signal`, `COUGH_COLD_FLU`, `percent_ill`, national |
| Table | `data/kinsa_ili/standard/data.csv.gz`: `geography`, `time`, `kinsa_cough_cold_flu` |
| Geography | **National only** (`geography = 00`, indexed as `nation` / `US`) |
| Cadence and range | Daily, 2019-01-01 to 2026-09-20 (2,820 days, no gaps at the 2026-09-21 pull) |
| Units | Percent of active Kinsa users; observed range 0.644 to 4.821, median 1.285 |
| Explorer group | `kinsa`, filed under acute respiratory illness, not influenza |

## Access and terms

The upstream Kinsa API is confidential: it needs a Kinsa login, which PopHIVE's
scheduled `ingest.R` holds. Tapestry never calls it and needs no Kinsa
credentials. PopHIVE's `measure_info.json` says the data may be re-used with
attribution and gives a suggested citation crediting Kinsa Insights data
obtained through the PopHIVE platform; use it in any output that shows this
series. The repository declares no software license, so that notice is the only
stated term. Coverage follows where Kinsa devices are owned, which skews toward
households with young children.

## Vintages come from Git

PopHIVE commits the table on every update, so the commit that first contains a
row is when the value entered PopHIVE. The fetcher lists the 133 commits that
touched the file (GitHub API) and downloads the file at each one from
raw.githubusercontent.com; the full Ingest repository is over 10 GB, so it is
not mirrored. The pull takes about 15 seconds. `GITHUB_TOKEN` is optional; without it
GitHub allows 60 API requests per hour and a pull uses two.

The history has three regimes:

| Commit date (UTC) | Rows first published | Reference dates |
|---|---:|---|
| 2026-04-06 | 2,647 | 2019-01-01 to 2026-03-31 (a backfill) |
| 2026-05-04 | 33 | through 2026-05-03 (a catch-up after an ingestion fix) |
| 2026-05-27 onward | 1 to 4 per commit | one day earlier than the commit (124 of 124 single-row commits) |

- **No value has ever been revised** in 133 commits. The pipeline only appends
  dates after the newest cached one. Every row therefore has exactly one report
  time and the archive contains no revision events.
- **Before 2026-04-06 nothing existed in PopHIVE.** A Wednesday origin earlier
  than that has no Kinsa data, and the real-time archive is about five months
  old. It is a lower bound on when Kinsa could have supplied a value. No release
  lag is assumed or filled in.
- The first commit holds one junk row whose time is the string `NA`. It is
  skipped and counted in the manifest, and PopHIVE removed it on 2026-05-04.

## Acquisition

```bash
python scripts/pull_covariates.py --data-root data pull pophive_kinsa_ili
```

The snapshot holds `archive.csv.gz` and `commits.json`. The archive has one row
per value change, in the columns `report_time`, `geo_type`, `geo_value`,
`reference_time` and `kinsa_cough_cold_flu`; a vanished row would be an explicit
empty value. `commits.json` records every commit's SHA, time, message, file
SHA-256 and row count. The fetcher is `tapestry.data.sources.pophive` and reads
any PopHIVE `standard/data.csv.gz` with `geography`, `time` and measure columns.
It accepts national (`00`) and two-digit state FIPS geographies and rejects
anything finer.

## Explorer

The series appears with the other national sources, with vintages from
2026-04-06 and `as_of` views resolved from the commit times. As with other
national series, the explorer draws the national line in a state view and labels
it as a national parent broadcast. That is a display convention; the model
datasets never broadcast it.
