# Local explorer API

The explorer binds to `127.0.0.1` by default and exposes no write endpoints.

## Endpoints

### `GET /api/health`

Returns server status and the active SQLite index path.

### `GET /api/catalog`

Returns build metadata, indexed states, dataset counts, aggregation notes, and
warnings for skipped sources. `selection` includes `policy_version`,
`raw_catalog_datasets`, `logical_source_groups`, `indexed_source_groups`,
`indexed_signal_choices`, `indexed_variants`, grouping membership, and missing
downloads. Counts distinguish canonical signal choices from stored variants.
Deliberately excluded file paths remain in the warnings audit with
`status=excluded`; errors and unavailable canonical files have separate statuses.

### `GET /api/series`

Series IDs belong to one index build and can change after a rebuild.

Lists state-compatible selected variants. Each item includes `source_group`,
`source_group_title`, `signal_key`, `signal_title`, `variant_label`,
`variant_rank`, and raw provenance. Group by `signal_key` to inspect variants;
multiple variant IDs may be sent together in data/version requests.
`measure_id` identifies the CDC field, Delphi signal, or Hub target rather than
the long-form storage field. `column_name`, `column_description`, and
`column_metadata_dataset` expose the saved CDC publisher metadata used for the
label. `origin_column` identifies the grouping measure and `provider_kind`
(`cdc`, `delphi`, `hub`) controls provider colors. Current Hub targets join their
originating NHSN/NSSP columns while preserving acquisition and unit variants.
Search includes CDC display names and canonical signal identifiers.
`total` counts matching variants; `signal_count` counts distinct signals in the
returned page. Parameters:

| Parameter | Values |
|---|---|
| `state` | Required state name, abbreviation, supported FIPS form, or `US` |
| `q` | Free-text search |
| `cadence` | `daily`, `weekly`, `monthly`, or `sample` |
| `vintage` | `versioned` or `unversioned` |
| `support` | `native_state` or `national` |
| `freshness` | `current` or `lagging` |
| `limit` / `offset` | Pagination; `limit=0` returns all columns, other limits are capped at 1,000 |

Example:

```text
/api/series?state=NY&cadence=weekly&vintage=versioned&support=native_state
```

`US` is included in the catalog's location list only when native national points
exist. For `state=US`, series, data, and version queries read published US records
only; state observations are never summed or substituted. `support=native_state`
returns no US series; use `national` or omit the filter.

### `GET /api/data`

Returns plot-ready points for at most 100 series IDs.

```text
/api/data?state=NY&series=12,19&scale=true
```

Each point is `[event_date, value, contributing_rows]`. When scaling is enabled,
the response includes the original maximum and divisor, computed within the displayed version.

Add `as_of=YYYY-MM-DD` to select an information cutoff (UTC end of day); omit it
or use `latest` for the latest view. Versioned rows use publisher release dates.
Full `as_of` snapshots select a single release, while report-time archives select
the latest eligible revision per event date. Unversioned sources only cut event
dates at the selected date. Missing historical values remain missing.

### `GET /api/versions`

`/api/versions?state=NY&series=12,19` returns sorted dates for arrow navigation.
Dates come from the selected series’ recorded releases,
or event dates for unversioned data.

Series metadata includes `parent_dataset`, `parent_column`, `parent_transform`,
and `lineage_status`. Unmapped columns have a null parent column.
