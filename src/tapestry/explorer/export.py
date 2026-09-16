"""Export a compact, static copy of the explorer index for GitHub Pages.

The published explorer has no server. This module writes everything the browser
needs into one directory:

* ``catalog.json`` — the ``/api/catalog`` overview plus export metadata.
* ``series.json`` — ``/api/series`` metadata per series id, shared by all locations.
* ``locations.json`` — per location, ``[id, date_min, date_max, point_count,
  max_samples_per_point]`` for each series with data there; the browser rebuilds
  the unfiltered series list from both files and applies search and filters itself.
* ``ranges.json`` — ``"<series_id>|<state>" -> [row_start, row_end)`` into that
  series' revision file.
* ``revisions/<series_id>.parquet`` — one Snappy-compressed change log per series
  (float32 values, delta-encoded integers), sorted by state, event day, and release
  day. The browser downloads a series file whole: GitHub Pages gzips responses and
  applies byte ranges to the compressed stream, so ranged Parquet reads fail there.

Every row is one observation value that became visible at ``release`` (days
since 1970-01-01; null for sources without release dates). A null ``value``
means the observation was suppressed or absent from that release. The browser
resolves an as-of date by taking, per event, the latest row released on or
before that date — one rule for every source.

Thinning, so the file stays small enough to commit:

* Releases are collapsed to the last one per Wednesday week: a release counts
  toward the first Wednesday on or after its date.
* Rows that repeat the previous kept value for the same observation are dropped.
* Delphi claims (daily releases) keep Wednesday snapshots only while
  ``release week <= event + CLAIMS_REVISION_DAYS``, plus each observation's
  latest value.
* Full-snapshot Hub target files are converted from whole snapshots into this
  change log, including removals, so resolution matches the local server.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .index import ExplorerIndex, _print_progress

CLAIMS_REVISION_DAYS = 56
ROW_GROUP_SIZE = 16_000
WEDNESDAY = 2  # Monday = 0


def _days(values: Any) -> np.ndarray:
    """ISO date prefixes (``YYYY-MM-DD...``) to int days since epoch; empty -> -1."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if not isinstance(values, (pa.Array, pa.ChunkedArray)):
        values = pa.array(list(values), pa.string())
    prefix = pc.utf8_slice_codeunits(pc.fill_null(values, ""), 0, 10)
    present = pc.not_equal(prefix, "")
    parsed = pc.strptime(pc.if_else(present, prefix, "1970-01-01"), format="%Y-%m-%d", unit="s")
    days = pc.divide(pc.cast(parsed, pa.int64()), 86_400)
    return np.where(present.to_numpy(zero_copy_only=False), days.to_numpy(zero_copy_only=False), -1).astype(np.int64)


def _wednesday_bucket(release: np.ndarray) -> np.ndarray:
    weekday = (release + 3) % 7  # 1970-01-01 was a Thursday
    return np.where(release < 0, -1, release + (WEDNESDAY - weekday) % 7)


def _thin(state: np.ndarray, event: np.ndarray, release: np.ndarray, value: np.ndarray,
          samples: np.ndarray, *, claims: bool) -> np.ndarray:
    """Indices to keep from rows sorted by state, event, then release order."""
    n = len(state)
    if not n:
        return np.zeros(0, dtype=np.int64)
    same_obs_next = np.zeros(n, dtype=bool)
    same_obs_next[:-1] = (state[1:] == state[:-1]) & (event[1:] == event[:-1])
    last_of_obs = ~same_obs_next
    bucket = _wednesday_bucket(release)
    # Last release in each Wednesday week (unversioned rows share bucket -1).
    same_bucket_next = np.zeros(n, dtype=bool)
    same_bucket_next[:-1] = same_obs_next[:-1] & (bucket[1:] == bucket[:-1])
    keep = ~same_bucket_next
    if claims:
        keep &= (bucket <= event + CLAIMS_REVISION_DAYS) | last_of_obs
    kept = np.flatnonzero(keep)
    # Drop kept rows that repeat the previous kept value of the same observation.
    k_state, k_event, k_value, k_samples = state[kept], event[kept], value[kept], samples[kept]
    same_obs_prev = np.zeros(len(kept), dtype=bool)
    same_obs_prev[1:] = (k_state[1:] == k_state[:-1]) & (k_event[1:] == k_event[:-1])
    both_missing = np.isnan(k_value[1:]) & np.isnan(k_value[:-1])
    equal = np.zeros(len(kept), dtype=bool)
    equal[1:] = (both_missing | (k_value[1:] == k_value[:-1])) & (k_samples[1:] == k_samples[:-1])
    return kept[~(same_obs_prev & equal)]


class StaticExport:
    def __init__(self, index: ExplorerIndex, destination: Path,
                 progress: Callable[[str], None] = _print_progress):
        self.index = index
        self.destination = Path(destination)
        self.progress = progress

    def run(self) -> dict[str, Any]:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        if not self.index.is_current():
            raise RuntimeError("The explorer index is missing or stale; run `index` first")
        staging = self.destination.with_name(self.destination.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        (staging / "revisions").mkdir(parents=True)

        catalog = self.index._catalog()
        ledger = ds.dataset(self.index.revision_ledger_path, format="parquet")
        schema = pa.schema([
            ("series_id", pa.int32()), ("state", pa.string()), ("event", pa.int32()),
            ("release", pa.int32()), ("value", pa.float32()), ("n", pa.int32()),
        ])
        delta = "DELTA_BINARY_PACKED"
        parquet_options = dict(compression="snappy", use_dictionary=["state"], row_group_size=ROW_GROUP_SIZE,
                               column_encoding={"series_id": delta, "event": delta, "release": delta, "n": delta})
        ranges: dict[str, list[int]] = {}
        offset = 0
        source_rows = 0
        with closing(self.index.connect()) as connection:
            series = connection.execute(
                "SELECT id, dataset_key, source_path, full_snapshots FROM series ORDER BY id"
            ).fetchall()
            for position, row in enumerate(series, 1):
                sid, key, path, full = int(row["id"]), row["dataset_key"], row["source_path"], bool(row["full_snapshots"])
                table = ledger.to_table(
                    columns=["state", "event_date", "release_time", "value", "samples"],
                    filter=ds.field("series_id") == sid,
                )
                source_rows += table.num_rows
                # Re-aggregate streaming chunks per (state, event, release), as the server does.
                table = table.group_by(["state", "event_date", "release_time"]).aggregate(
                    [("value", "sum"), ("samples", "sum")]
                ).sort_by([("state", "ascending"), ("event_date", "ascending"), ("release_time", "ascending")])
                if full:
                    columns = self._snapshot_changes(connection, key, path, table)
                else:
                    columns = self._revision_columns(table, claims=key.startswith("delphi_claims_"))
                count = len(columns["state"])
                if count:
                    states = columns["state"]
                    starts = np.flatnonzero(np.r_[True, states[1:] != states[:-1]])
                    for start, end in zip(starts, np.r_[starts[1:], count]):
                        ranges[f"{sid}|{states[start]}"] = [int(start), int(end)]
                    batch = pa.table({
                        "series_id": pa.array(np.full(count, sid, dtype=np.int32)),
                        "state": pa.array(states, pa.string()),
                        "event": pa.array(columns["event"].astype(np.int32)),
                        "release": pa.array(columns["release"].astype(np.int32), mask=columns["release"] < 0),
                        "value": pa.array(columns["value"].astype(np.float32), mask=np.isnan(columns["value"])),
                        "n": pa.array(columns["samples"].astype(np.int32)),
                    }, schema=schema)
                    pq.write_table(batch, staging / "revisions" / f"{sid}.parquet", **parquet_options)
                    offset += count
                if position % 25 == 0 or position == len(series):
                    self.progress(f"exported {position}/{len(series)} series · {offset:,} rows kept of {source_rows:,}")

        overview = self.index.overview()
        overview["meta"] = {**overview["meta"], "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "export_rows": str(offset), "export_source_rows": str(source_rows),
                            "claims_revision_days": str(CLAIMS_REVISION_DAYS)}
        self._write_json(staging / "catalog.json", overview)
        self._write_json(staging / "ranges.json", ranges)
        coverage_fields = ("date_min", "date_max", "point_count", "max_samples_per_point")
        metadata: dict[str, Any] = {}
        locations: dict[str, list[list[Any]]] = {}
        for location in overview["states"]:
            listing = self.index.list_series(location["code"], limit=0)
            locations[location["code"]] = [[item["id"], *(item[field] for field in coverage_fields)]
                                           for item in listing["items"]]
            for item in listing["items"]:
                # Freshness is recomputed in the browser from date_max.
                metadata.setdefault(str(item["id"]), {key: value for key, value in item.items()
                                                      if key not in {*coverage_fields, "freshness"}})
        self._write_json(staging / "series.json", metadata)
        self._write_json(staging / "locations.json", locations)

        shutil.rmtree(self.destination, ignore_errors=True)
        staging.rename(self.destination)
        size = sum(path.stat().st_size for path in self.destination.rglob("*") if path.is_file())
        summary = {"destination": str(self.destination), "rows": offset, "source_rows": source_rows,
                   "bytes": size, "locations": len(overview["states"])}
        self.progress(f"wrote {size / 1e6:.1f} MB to {self.destination}")
        return summary

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @staticmethod
    def _mean(table) -> tuple[np.ndarray, np.ndarray]:
        total = table["value_sum"].to_numpy(zero_copy_only=False).astype(float)
        samples = np.nan_to_num(table["samples_sum"].to_numpy(zero_copy_only=False).astype(float)).astype(np.int64)
        value = np.where(samples > 0, np.nan_to_num(total) / np.maximum(samples, 1), np.nan)
        return value, samples

    def _revision_columns(self, table, *, claims: bool) -> dict[str, np.ndarray]:
        state = table["state"].to_numpy(zero_copy_only=False).astype(object)
        event = _days(table["event_date"])
        release = _days(table["release_time"])
        value, samples = self._mean(table)
        keep = _thin(state, event, release, value, samples, claims=claims)
        return {"state": state[keep], "event": event[keep], "release": release[keep],
                "value": value[keep], "samples": samples[keep]}

    def _snapshot_changes(self, connection: sqlite3.Connection, key: str, path: str, table) -> dict[str, np.ndarray]:
        """Convert whole-file snapshots into per-observation changes, including removals."""
        vintages = [row[0] for row in connection.execute(
            "SELECT vintage FROM releases WHERE dataset_key=? AND source_path=? ORDER BY vintage", (key, path))]
        empty = {name: np.zeros(0, dtype=dtype) for name, dtype in
                 (("state", object), ("event", np.int64), ("release", np.int64), ("value", float), ("samples", np.int64))}
        if not vintages or not table.num_rows:
            return empty
        vintage_index = {vintage: i for i, vintage in enumerate(vintages)}
        states = table["state"].to_numpy(zero_copy_only=False).astype(object)
        events = _days(table["event_date"])
        release_strings = table["release_time"].to_pylist()
        v_idx = np.array([vintage_index.get(r or "", -1) for r in release_strings])
        value, samples = self._mean(table)
        valid = v_idx >= 0
        keys = np.char.add(states[valid].astype(str), np.char.add("|", events[valid].astype(str)))
        obs_keys, obs_idx = np.unique(keys, return_inverse=True)
        grid = np.full((len(obs_keys), len(vintages)), np.nan)
        grid_n = np.zeros((len(obs_keys), len(vintages)), dtype=np.int64)
        grid[obs_idx, v_idx[valid]] = value[valid]
        grid_n[obs_idx, v_idx[valid]] = samples[valid]
        previous = np.concatenate([np.full((len(obs_keys), 1), np.nan), grid[:, :-1]], axis=1)
        previous_n = np.concatenate([np.zeros((len(obs_keys), 1), dtype=np.int64), grid_n[:, :-1]], axis=1)
        changed = ~((np.isnan(grid) & np.isnan(previous)) | ((grid == previous) & (grid_n == previous_n)))
        obs, vint = np.nonzero(changed)
        if not len(obs):
            return empty
        split = np.array([k.split("|") for k in obs_keys[obs]], dtype=object)
        release_days = _days(np.array(vintages, dtype=object))
        state = split[:, 0].astype(object)
        event = split[:, 1].astype(np.int64)
        release = release_days[vint]
        change_value = grid[obs, vint]
        change_n = grid_n[obs, vint]
        order = np.lexsort((vint, event, state.astype(str)))
        state, event, release, change_value, change_n = (state[order], event[order], release[order],
                                                         change_value[order], change_n[order])
        keep = _thin(state, event, release, change_value, change_n, claims=False)
        return {"state": state[keep], "event": event[keep], "release": release[keep],
                "value": change_value[keep], "samples": change_n[keep]}


def export_static(index: ExplorerIndex, destination: str | Path,
                  progress: Callable[[str], None] = _print_progress) -> dict[str, Any]:
    return StaticExport(index, Path(destination), progress).run()
