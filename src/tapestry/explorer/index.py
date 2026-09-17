"""Build and query the disposable surveillance index and revision ledger.

The explorer intentionally keeps the acquisition boundary read-only. It builds
a disposable SQLite metadata/latest-point index and a Parquet revision ledger
under ``data/.explorer`` from the latest immutable raw snapshot of each
downloaded dataset. HTTP serving lives in ``server.py``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from contextlib import closing, contextmanager, ExitStack, suppress
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..data.geography import STATE_NAMES, normalize_state, observation_geography
from ..data.lineage import series_lineage
from ..data.tables import Artifact, TableSource
from ..data.selection import SelectedData, POLICY_VERSION, describe, measure_columns


INDEX_SCHEMA_VERSION = 14
# Schema profiling only needs a representative prefix. The indexer makes a
# second full pass and discovers numeric columns that appear later.
PROFILE_ROW_LIMIT = 25_000
GENERIC_MISSING = {"", "na", "n/a", "nan", "null", "none", ".", "*", "-"}


STATE_COLUMN_NAMES = (
    "jurisdiction", "state", "state_code", "state_abbr", "state_abbreviation",
    "state_territory", "geography", "geo_value", "location", "location_code",
    "location_name", "fips", "area", "region",
)
DATE_COLUMN_NAMES = (
    "date", "week_end", "weekendingdate", "week_ending_date", "mmwrweek_end",
    "target_end_date", "target_date", "event_date", "time_value", "epiweek", "week",
    "reference_date", "reference_time", "forecast_date", "issue_date", "report_date", "season_week",
)
VINTAGE_COLUMN_NAMES = (
    "report_time", "issue", "issued", "as_of", "vintage", "posted", "release_date",
    "forecast_date", "reference_date",
)
FORCED_DIMENSION_NAMES = {
    "age", "age_group", "ahead", "category", "confidence", "geo_type", "horizon",
    "demographics_type", "demographics_values", "fill_method", "insurance", "level", "measure", "metric",
    "model", "model_id", "output_type", "output_type_id", "pathogen", "pathogen_target",
    "pcr_gene_target_agg", "pcr_target", "pcr_target_units", "quantile", "race",
    "race_ethnicity", "scenario", "scenario_id", "sex", "signal",
    "subtype", "target", "target_variable", "trend_source", "unit", "variable", "variant",
    "virus", "visit_type",
}
UNHELPFUL_DIMENSION_NAMES = {
    "created_at", "updated_at", "row_id", ":id", "id", "url", "notes", "description",
}
NON_MEASURE_COLUMN_NAMES = {"season", "respseason", "season_week"}
FRESHNESS_DAYS = {"daily": 14, "weekly": 35, "monthly": 75, "sample": 45}


def _print_progress(message: str) -> None:
    """Flush CLI progress even when stdout is connected to a pipe."""

    print(message, flush=True)



def _number(value: Any, missing: set[str] | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if text.lower() in (missing or GENERIC_MISSING):
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _date_string(value: Any) -> str | None:
    """Coerce common ISO/date/epiweeks into a browser-sortable ISO date."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in GENERIC_MISSING:
        return None
    iso_text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_text).date().isoformat()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        pass
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        try:
            return datetime.strptime(digits, "%Y%m%d").date().isoformat()
        except ValueError:
            pass
    if len(digits) == 6 and 1900 <= int(digits[:4]) <= 2200:
        year, week = int(digits[:4]), int(digits[4:])
        if 1 <= week <= 53:
            # MMWR and ISO weeks differ around New Year, but this conversion keeps
            # weekly observations ordered and consistently seven days apart.
            try:
                return date.fromisocalendar(year, week, 1).isoformat()
            except ValueError:
                base = date(year, 1, 4)
                return (base - timedelta(days=base.isoweekday() - 1) + timedelta(weeks=week - 1)).isoformat()
    return None


def _vintage_string(value: Any) -> str | None:
    """Keep intraday V5 report times when choosing the latest revision."""
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return _date_string(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp.isoformat()


def _column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_lower = {item.lower(): item for item in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _is_identifier_column(name: str) -> bool:
    lower = name.lower()
    return (
        lower in {"id", ":id", "fips", "geoid", "row_id", "location_id", "site_id"}
        or lower.endswith(("_fips", "_geoid", "_code", "_id"))
    )


@dataclass
class ColumnProfile:
    nonmissing: int = 0
    numeric: int = 0
    state_values: int = 0
    unique: set[str] = field(default_factory=set)
    unique_overflow: bool = False

    def observe(
        self,
        value: Any,
        missing: set[str],
        unique_limit: int = 64,
        *,
        check_state: bool = False,
    ) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text.lower() in missing:
            return
        self.nonmissing += 1
        if _number(value, missing) is not None:
            self.numeric += 1
        if check_state and normalize_state(value) is not None:
            self.state_values += 1
        if not self.unique_overflow:
            self.unique.add(text)
            if len(self.unique) > unique_limit:
                self.unique.clear()
                self.unique_overflow = True


class _BuildError(RuntimeError):
    """A failure that must discard the whole unpublished SQLite/Parquet pair."""


class _RevisionLedgerWriter:
    """Stream normalized revision batches to one typed Parquet file."""

    def __init__(self, destination: Path, *, build_id: str = "", batch_rows: int = 100_000):
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as parquet  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "Building the revision ledger requires PyArrow; install "
                "'tapestry[explorer]' before building the explorer index"
            ) from error
        self.pa = pa
        self.batch_rows = batch_rows
        self.schema = pa.schema([
            ("series_id", pa.int64()),
            ("state", pa.string()),
            ("event_date", pa.string()),
            ("release_time", pa.string()),
            ("value", pa.float64()),
            ("samples", pa.int64()),
            ("source_snapshot", pa.string()),
        ], metadata={b"build_id": build_id.encode()})
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.writer = parquet.ParquetWriter(
            destination,
            self.schema,
            compression="zstd",
            use_dictionary=["state", "release_time", "source_snapshot"],
        )
        self.rows_written = 0
        self._snapshots: dict[int, str] = {}

    def append(
        self,
        connection: sqlite3.Connection,
        rows: Sequence[tuple[int, str, str, str, float | None, int]],
    ) -> None:
        if not rows:
            return
        missing = {int(row[0]) for row in rows if int(row[0]) not in self._snapshots}
        if missing:
            placeholders = ",".join("?" for _ in missing)
            self._snapshots.update(
                {
                    int(series_id): str(snapshot)
                    for series_id, snapshot in connection.execute(
                        f"SELECT id, snapshot_id FROM series WHERE id IN ({placeholders})",
                        sorted(missing),
                    )
                }
            )
        table = self.pa.Table.from_arrays(
            [
                self.pa.array([row[0] for row in rows], type=self.pa.int64()),
                self.pa.array([row[1] for row in rows], type=self.pa.string()),
                self.pa.array([row[2] for row in rows], type=self.pa.string()),
                self.pa.array([row[3] for row in rows], type=self.pa.string()),
                self.pa.array([row[4] for row in rows], type=self.pa.float64()),
                self.pa.array([row[5] for row in rows], type=self.pa.int64()),
                self.pa.array(
                    [self._snapshots[int(row[0])] for row in rows],
                    type=self.pa.string(),
                ),
            ],
            schema=self.schema,
        )
        # Cluster each bounded batch for compression and narrower row-group statistics.
        # Global sorting would require staging the entire revision archive.
        table = table.sort_by([("series_id", "ascending"), ("state", "ascending"),
                              ("event_date", "ascending"), ("release_time", "ascending")])
        try:
            self.writer.write_table(table, row_group_size=min(self.batch_rows, 25_000))
        except Exception as error:
            raise _BuildError(f"Failed to write revision ledger: {error}") from error
        self.rows_written += len(rows)

    def close(self) -> None:
        self.writer.close()


class ExplorerIndex(SelectedData):
    """Build and query a disposable long-form index of plottable observations."""

    def __init__(
        self,
        data_root: str | Path,
        index_path: str | Path | None = None,
        ledger_path: str | Path | None = None,
        *,
        batch_rows: int = 100_000,
        cache_mb: int = 256,
        strict: bool = False,
    ):
        super().__init__(data_root)
        self.index_path = (
            Path(index_path).expanduser().resolve()
            if index_path is not None
            else self.data_root / ".explorer" / "series.sqlite3"
        )
        # Revisions are an append-shaped analytical table. Keep them in a
        # separate columnar file so SQLite only has to answer metadata and
        # latest-point searches. A caller using a custom SQLite path can also
        # choose an explicit sibling ledger path.
        self.revision_ledger_path = (
            Path(ledger_path).expanduser().resolve()
            if ledger_path is not None
            else self.index_path.parent / "revisions.parquet"
        )
        self._write_lock = threading.Lock()
        if batch_rows < 1 or cache_mb < 1:
            raise ValueError("batch_rows and cache_mb must be positive")
        if self.index_path == self.revision_ledger_path:
            raise ValueError("SQLite and Parquet paths must be different")
        self.batch_rows = batch_rows
        self.cache_mb = cache_mb
        self.strict = strict

    @contextmanager
    def _build_lock(self):
        """Reject competing CLI builders, including ones sharing a ledger."""
        with self._write_lock, ExitStack() as stack:
            for path in sorted({self.index_path, self.revision_ledger_path}):
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(path.with_name(path.name + ".lock").open("a"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise _BuildError(f"Another builder is using {path}") from error
            yield

    def source_fingerprint(self) -> str:
        digest = hashlib.sha256(POLICY_VERSION.encode())
        catalog_path = self.data_root / "catalog.json"
        digest.update(catalog_path.read_bytes())
        for artifact in self.artifacts():
            stat = artifact.path.stat()
            digest.update(
                f"{artifact.dataset['key']}\0{artifact.snapshot_id}\0{artifact.relative_path}\0"
                f"{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
        return digest.hexdigest()

    def is_current(self) -> bool:
        if not self.index_path.is_file():
            return False
        try:
            with closing(self.connect()) as connection:
                values = dict(connection.execute("SELECT key, value FROM meta"))
            return (
                int(values.get("schema_version", -1)) == INDEX_SCHEMA_VERSION
                and values.get("source_fingerprint") == self.source_fingerprint()
                and self.revision_ledger_path.is_file()
                and not self._ledger_errors(values, self.revision_ledger_path)
            )
        except (OSError, sqlite3.Error, ValueError):
            return False

    def ensure(
        self,
        *,
        rebuild: bool = False,
        progress: Callable[[str], None] = _print_progress,
    ) -> None:
        if rebuild:
            self.build(progress=progress)
        elif self.is_current():
            if self.strict:
                with closing(self.connect()) as connection:
                    if connection.execute("SELECT 1 FROM sources WHERE status='error' LIMIT 1").fetchone():
                        raise _BuildError("Existing index has source errors; strict mode refused reuse")
            progress(f"Explorer index is current: {self.index_path}")
            return
        else:
            self.build(progress=progress)

    def build(self, *, progress: Callable[[str], None] = _print_progress) -> None:
        """Build privately, validate, then replace the two derived files.

        Stop the server before rebuilding: two file renames are not atomic as
        a pair. Matching build IDs detect interrupted publication on next use.
        """

        with self._build_lock():
            started = time.monotonic()
            build_id = uuid.uuid4().hex
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.index_path.with_name(f".{self.index_path.name}.{os.getpid()}.tmp")
            ledger_temporary = self.revision_ledger_path.with_name(
                f".{self.revision_ledger_path.name}.{os.getpid()}.tmp"
            )
            if temporary.exists():
                temporary.unlink()
            if ledger_temporary.exists():
                ledger_temporary.unlink()
            self.audit.clear()
            fingerprint = self.source_fingerprint()
            artifacts = self.artifacts()
            connection = sqlite3.connect(temporary)
            ledger_writer: _RevisionLedgerWriter | None = None
            try:
                # The index is disposable and being built in a private
                # temporary file. Exclusive locking and a larger cache reduce
                # overhead during the bulk load.
                connection.execute("PRAGMA locking_mode=EXCLUSIVE")
                connection.execute(f"PRAGMA cache_size=-{self.cache_mb * 1024}")
                self._create_schema(connection)
                ledger_writer = _RevisionLedgerWriter(
                    ledger_temporary, build_id=build_id, batch_rows=self.batch_rows
                )
                connection.execute("INSERT INTO meta VALUES (?, ?)", ("build_id", build_id))
                connection.execute("INSERT INTO meta VALUES (?, ?)", ("schema_version", str(INDEX_SCHEMA_VERSION)))
                connection.execute("INSERT INTO meta VALUES (?, ?)", ("source_fingerprint", fingerprint))
                connection.execute("INSERT INTO meta VALUES (?, ?)", ("built_at", datetime.now().astimezone().isoformat()))
                connection.execute("INSERT INTO meta VALUES (?, ?)", ("artifact_count", str(len(artifacts))))
                for number, artifact in enumerate(artifacts, 1):
                    artifact_started = time.monotonic()
                    previous_rows = ledger_writer.rows_written
                    progress(f"[{number}/{len(artifacts)}] {artifact.dataset['key']}: {artifact.relative_path}")
                    self._index_artifact(connection, artifact, progress, ledger_writer)
                    progress(f"    {ledger_writer.rows_written - previous_rows:,} revision rows; "
                             f"{time.monotonic() - artifact_started:.1f}s")
                    connection.commit()
                # Maintaining secondary indexes for every revision insert is
                # much slower than creating them once after the bulk load.
                self._create_indexes(connection)
                catalog = self._catalog()
                for sid, key, column, path, dims in connection.execute(
                    "SELECT id,dataset_key,value_column,source_path,dimensions FROM series"
                ).fetchall():
                    description = describe(key, column, path, json.loads(dims), catalog=catalog)
                    connection.execute("UPDATE series SET label=label || ? WHERE id=?", (
                        " · " + description["signal_title"] + " · " + description["signal_key"], sid))
                for item in self.audit:
                    connection.execute(
                        "INSERT INTO sources (dataset_key,snapshot_id,source_path,status,message) VALUES (?,?,?,?,?)",
                        (item["dataset_key"], "", item["source_path"], item["status"], item["message"]))
                connection.execute("INSERT INTO meta VALUES (?, ?)", ("selection_policy", POLICY_VERSION))
                assert ledger_writer is not None
                revision_count = ledger_writer.rows_written
                ledger_writer.close()
                ledger_writer = None
                connection.execute(
                    "INSERT INTO meta VALUES (?, ?)",
                    ("revision_ledger", self.revision_ledger_path.name),
                )
                connection.execute(
                    "INSERT INTO meta VALUES (?, ?)",
                    ("revision_row_count", str(revision_count)),
                )
                connection.execute("ANALYZE")
                self._check_build(connection, ledger_temporary, fingerprint)
                connection.executemany("INSERT INTO meta VALUES (?, ?)", [
                    ("build_seconds", f"{time.monotonic() - started:.3f}"),
                    ("batch_rows", str(self.batch_rows)),
                    ("cache_mb", str(self.cache_mb)),
                ])
                connection.commit()
            except BaseException:
                if ledger_writer is not None:
                    # A failed Parquet close must not prevent cleanup of the
                    # private files or mask the original indexing failure.
                    with suppress(Exception):
                        ledger_writer.close()
                connection.close()
                if temporary.exists():
                    temporary.unlink()
                if ledger_temporary.exists():
                    ledger_temporary.unlink()
                raise
            else:
                connection.close()
                self.revision_ledger_path.parent.mkdir(parents=True, exist_ok=True)
                ledger_temporary.replace(self.revision_ledger_path)
                temporary.replace(self.index_path)
                progress(f"Explorer index ready: {self.index_path} "
                         f"({revision_count:,} revision rows in {time.monotonic() - started:.1f}s)")

    def _check_build(self, connection, ledger_path: Path, fingerprint: str) -> None:
        if self.source_fingerprint() != fingerprint:
            raise _BuildError("Raw inventory changed during the build; run index again")
        errors = self._storage_errors(connection, ledger_path)
        if errors:
            raise _BuildError("; ".join(errors))
        if self.strict and connection.execute(
            "SELECT COUNT(*) FROM sources WHERE status='error'"
        ).fetchone()[0]:
            raise _BuildError("Strict build rejected source errors")

    @staticmethod
    def _ledger_errors(meta: Mapping[str, str], path: Path) -> list[str]:
        """Check the footer and pair identity without scanning 77M+ rows."""
        import pyarrow.parquet as parquet

        errors = []
        try:
            with parquet.ParquetFile(path) as ledger:
                if ledger.metadata.num_rows != int(meta.get("revision_row_count", -1)):
                    errors.append("Parquet revision count differs from SQLite metadata")
                build_id = (ledger.schema_arrow.metadata or {}).get(b"build_id", b"").decode()
                if build_id != meta.get("build_id", "") or (
                    int(meta.get("schema_version", 0)) >= 12 and not build_id
                ):
                    errors.append("SQLite and Parquet build IDs do not match; rebuild the index")
                expected = {"series_id": "int64", "state": "string", "event_date": "string",
                            "release_time": "string", "value": "double", "samples": "int64",
                            "source_snapshot": "string"}
                if {f.name: str(f.type) for f in ledger.schema_arrow} != expected:
                    errors.append("Unexpected Parquet revision schema")
        except (OSError, ValueError) as error:
            errors.append(f"Cannot read revision ledger: {error}")
        return errors

    @classmethod
    def _storage_errors(cls, connection, ledger_path: Path) -> list[str]:
        errors = [str(row[0]) for row in connection.execute("PRAGMA quick_check") if row[0] != "ok"]
        if connection.execute("PRAGMA foreign_key_check").fetchone():
            errors.append("SQLite contains orphan point rows")
        meta = dict(connection.execute("SELECT key, value FROM meta"))
        return errors + cls._ledger_errors(meta, ledger_path)

    def status(self, *, validate: bool = False) -> dict[str, Any]:
        """Inspect saved build health without rebuilding or opening a browser.

        Validation checks SQLite and the Parquet footer, not every raw value.
        Source errors are reported separately from storage integrity failures.
        """
        result: dict[str, Any] = {
            "index_path": str(self.index_path),
            "ledger_path": str(self.revision_ledger_path),
            "current": self.is_current(),
            "errors": [],
        }
        for key, path in (("sqlite_bytes", self.index_path), ("parquet_bytes", self.revision_ledger_path)):
            result[key] = path.stat().st_size if path.is_file() else 0
        try:
            with closing(self.connect()) as connection:
                result["metadata"] = dict(connection.execute("SELECT key, value FROM meta"))
                result["sources"] = dict(connection.execute("SELECT status, COUNT(*) FROM sources GROUP BY status"))
                result["source_errors"] = [dict(row) for row in connection.execute(
                    "SELECT dataset_key, source_path, message FROM sources WHERE status='error'"
                )]
                result["errors"] = (self._storage_errors(connection, self.revision_ledger_path)
                                    if validate else self._ledger_errors(result["metadata"], self.revision_ledger_path))
        except (OSError, sqlite3.Error, ValueError) as error:
            result["errors"].append(str(error))
        return result

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            -- Savepoint rollback requires a journal even for a disposable DB.
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY,
                dataset_key TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                rows_seen INTEGER NOT NULL DEFAULT 0,
                state_column TEXT,
                date_column TEXT,
                numeric_columns TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                message TEXT
            );
            CREATE TABLE series (
                id INTEGER PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                dataset_key TEXT NOT NULL,
                dataset_title TEXT NOT NULL,
                provider TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                value_column TEXT NOT NULL,
                state_column TEXT NOT NULL,
                date_column TEXT NOT NULL,
                dimensions TEXT NOT NULL,
                label TEXT NOT NULL,
                full_snapshots INTEGER NOT NULL DEFAULT 0,
                vintage_column TEXT
            );
            CREATE TABLE points (
                series_id INTEGER NOT NULL REFERENCES series(id),
                state TEXT NOT NULL,
                date TEXT NOT NULL,
                value REAL NOT NULL,
                samples INTEGER NOT NULL DEFAULT 1,
                vintage TEXT,
                PRIMARY KEY(series_id, state, date)
            ) WITHOUT ROWID;
            CREATE TABLE releases (
                dataset_key TEXT NOT NULL,
                source_path TEXT NOT NULL,
                vintage TEXT NOT NULL,
                PRIMARY KEY(dataset_key, source_path, vintage)
            ) WITHOUT ROWID;
            """
        )

    @staticmethod
    def _create_indexes(connection: sqlite3.Connection) -> None:
        """Create query indexes once, after all rows have been inserted."""

        connection.executescript(
            """
            CREATE INDEX points_state_series_date ON points(state, series_id, date);
            CREATE INDEX series_dataset_column ON series(dataset_key, value_column);
            """
        )

    def _index_artifact(
        self,
        connection: sqlite3.Connection,
        artifact: Artifact,
        progress: Callable[[str], None],
        ledger_writer: _RevisionLedgerWriter,
    ) -> None:
        try:
            found = False
            for source in self._table_sources(artifact):
                found = True
                progress(f"  indexing {source.source_path}")
                written_before = ledger_writer.rows_written if ledger_writer else 0
                connection.execute("SAVEPOINT table_index")
                try:
                    self._index_table(connection, source, progress, ledger_writer)
                except Exception as error:
                    # Parquet is append-only: once a table has emitted rows,
                    # its SQLite savepoint alone cannot undo the partial table.
                    if (isinstance(error, _BuildError) or self.strict or
                            (ledger_writer and ledger_writer.rows_written != written_before)):
                        raise _BuildError(f"Failed indexing {source.source_path}: {error}") from error
                    # Hub archives can mix CSV and Parquet (and can contain one
                    # malformed submission). Preserve useful siblings and expose
                    # the exact skipped member in Index details.
                    connection.execute("ROLLBACK TO table_index")
                    connection.execute("RELEASE table_index")
                    if ledger_writer is not None:
                        # Rolled-back series IDs can be reused by a sibling.
                        ledger_writer._snapshots.clear()
                    self._record_source(
                        connection,
                        source,
                        status="error",
                        message=f"{type(error).__name__}: {error}",
                    )
                    progress(f"  skipped {source.source_path}: {type(error).__name__}: {error}")
                else:
                    connection.execute("RELEASE table_index")
            if not found:
                self._record_source(connection, artifact, status="skipped", message="No supported tables")
        except _BuildError:
            raise
        except Exception as error:  # one unusual source must not make the explorer unusable
            if self.strict:
                raise _BuildError(f"Failed indexing {artifact.relative_path}: {error}") from error
            self._record_source(connection, artifact, status="error", message=f"{type(error).__name__}: {error}")
            progress(f"  skipped: {type(error).__name__}: {error}")

    def _record_source(
        self,
        connection: sqlite3.Connection,
        artifact: Artifact | TableSource,
        *,
        status: str,
        message: str | None = None,
        rows_seen: int = 0,
        state_column: str | None = None,
        date_column: str | None = None,
        numeric_columns: Sequence[str] = (),
    ) -> None:
        connection.execute(
            """INSERT INTO sources
               (dataset_key, snapshot_id, source_path, rows_seen, state_column, date_column,
                numeric_columns, status, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(artifact.dataset["key"]) if isinstance(artifact, Artifact) else artifact.dataset_key,
                artifact.snapshot_id,
                artifact.relative_path if isinstance(artifact, Artifact) else artifact.source_path,
                rows_seen,
                state_column,
                date_column,
                json.dumps(list(numeric_columns)),
                status,
                message,
            ),
        )

    def _index_table(
        self,
        connection: sqlite3.Connection,
        source: TableSource,
        progress: Callable[[str], None],
        ledger_writer: _RevisionLedgerWriter,
    ) -> None:
        geo_path_match = re.search(
            r"(?:^|/)geo_type=([^/]+)(?:/|$)", source.source_path, re.IGNORECASE
        )
        path_geo_type = geo_path_match.group(1).lower() if geo_path_match else None
        missing = GENERIC_MISSING | {item.lower() for item in source.missing_markers}
        profiles: dict[str, ColumnProfile] = defaultdict(ColumnProfile)
        rows_seen = 0
        for row in source.iter_rows():
            rows_seen += 1
            for key, value in row.items():
                name = str(key)
                profiles[name].observe(
                    value,
                    missing,
                    check_state=name.lower() in STATE_COLUMN_NAMES,
                )
            if rows_seen >= PROFILE_ROW_LIMIT:
                break
        columns = list(profiles)
        if not rows_seen or not columns:
            self._record_source(connection, source, status="empty", rows_seen=rows_seen)
            return

        state_column = self._state_column(profiles)
        date_column = self._date_column(columns, source.event_date_column)
        vintage_column = _column(columns, (source.vintage_column,) if source.vintage_column else ())
        if vintage_column is None:
            vintage_column = _column(columns, VINTAGE_COLUMN_NAMES)
        geo_level_column = _column(
            columns, ("geo_type", "geography_type", "geographic_level", "level")
        )
        geo_value_column = state_column or _column(
            columns, ("geo_value", "geography", "region", "location", "level")
        )
        resolutions = {value.lower() for value in source.geographic_resolutions}
        can_map_parent = bool(resolutions.intersection({"nation", "national"}))

        numeric_columns = [
            key for key, profile in profiles.items()
            if profile.numeric > 0
            and profile.numeric / max(profile.nonmissing, 1) >= 0.60
            and key not in {state_column, geo_value_column, date_column, vintage_column}
            and key.lower() not in FORCED_DIMENSION_NAMES
            and key.lower() not in NON_MEASURE_COLUMN_NAMES
            and not _is_identifier_column(key)
        ]
        numeric_columns = list(measure_columns(source.dataset_key, numeric_columns, source.source_path))
        if (not state_column and not can_map_parent) or not date_column or not numeric_columns:
            reasons = []
            if not state_column:
                reasons.append("no state column")
            if not date_column:
                reasons.append("no date column")
            if not numeric_columns:
                reasons.append("no numeric measure columns")
            if not state_column and not can_map_parent:
                reasons.append("no state or broadcastable parent geography")
            self._record_source(
                connection, source, status="skipped", message=", ".join(reasons),
                rows_seen=rows_seen, state_column=state_column, date_column=date_column,
                numeric_columns=numeric_columns,
            )
            return

        dimensions = self._dimension_columns(
            profiles,
            excluded={
                state_column, geo_value_column, geo_level_column, date_column, vintage_column,
                *numeric_columns,
            },
        )

        series_ids: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        numeric_column_set = set(numeric_columns)
        non_measure_columns = {
            state_column, geo_value_column, geo_level_column, date_column, vintage_column
        }
        raw_value_count = 0
        buffered_writes = 0
        point_buffer: dict[tuple[int, str, str, str], tuple[float, int]] = {}
        releases: set[str] = set(source.release_times)
        mapped_rows = 0
        scanned_rows = 0
        for row in source.iter_rows():
            scanned_rows += 1
            if scanned_rows % 1_000_000 == 0:
                progress(f"    {scanned_rows:,} rows scanned")
            # A publisher can add a measure late in a long revision archive. The
            # capped profiling pass keeps indexing practical, while this check
            # still discovers numeric columns whose first value occurs later.
            for key, raw_value in row.items():
                if (
                    key not in numeric_column_set
                    and key not in non_measure_columns
                    and key.lower() not in FORCED_DIMENSION_NAMES
                    and key.lower() not in NON_MEASURE_COLUMN_NAMES
                    and not _is_identifier_column(key)
                    and measure_columns(source.dataset_key, (key,), source.source_path)
                    and _number(raw_value, missing) is not None
                ):
                    numeric_columns.append(key)
                    numeric_column_set.add(key)
            day = _date_string(row.get(date_column))
            if day is None:
                continue
            geography = observation_geography(row, source.geographic_resolutions, path_geo_type)
            targets = ((("US", "national parent broadcast"),)
                       if geography == "US" else ((geography, None),) if geography else ())
            if not targets:
                continue
            mapped_rows += 1
            base_dimensions = tuple(
                (key, str(row[key]).strip())
                for key in dimensions
                if row.get(key) is not None and str(row[key]).strip().lower() not in missing
            )
            vintage = (_vintage_string(row.get(vintage_column)) if vintage_column else None) or source.fallback_vintage
            if vintage:
                releases.add(vintage)
            for state, support in targets:
                dimension_values = base_dimensions
                if support:
                    dimension_values += (("spatial_support", support),)
                for value_column in numeric_columns:
                    value = _number(row.get(value_column), missing)
                    if value is None and not vintage:
                        continue
                    cache_key = (value_column, dimension_values)
                    series_id = series_ids.get(cache_key)
                    if series_id is None:
                        series_id = self._series_id(
                            connection,
                            replace(source, vintage_column=vintage_column),
                            value_column,
                            state_column or geo_value_column or "geography",
                            date_column,
                            dimension_values,
                        )
                        series_ids[cache_key] = series_id
                    point_key = (series_id, state, day, vintage or "")
                    current = point_buffer.get(point_key, (0.0, 0))
                    if source.dataset_key.startswith("hub_") and current[1] and value is not None:
                        if current[0] / current[1] != value:
                            raise ValueError("Conflicting hub truth values for the same location, target, event and vintage")
                        continue
                    point_buffer[point_key] = (current[0] + (value or 0.0), current[1] + int(value is not None))
                    raw_value_count += 1
                    if len(point_buffer) >= self.batch_rows:
                        buffered_writes += self._flush_points(
                            connection, point_buffer, ledger_writer
                        )
                        point_buffer.clear()
        if mapped_rows == 0:
            self._record_source(
                connection,
                source,
                status="skipped",
                message="no state or national rows",
                rows_seen=rows_seen,
                state_column=state_column,
                date_column=date_column,
                numeric_columns=numeric_columns,
            )
            return
        buffered_writes += self._flush_points(connection, point_buffer, ledger_writer)
        connection.executemany("INSERT OR IGNORE INTO releases VALUES (?, ?, ?)",
                               ((source.dataset_key, source.source_path, vintage) for vintage in releases))
        series_id_values = list(series_ids.values())
        if series_id_values:
            placeholders = ",".join("?" for _ in series_id_values)
            connection.execute(
                f"""UPDATE series SET label = label || ' · unweighted state-date mean'
                    WHERE id IN ({placeholders})
                      AND EXISTS (
                        SELECT 1 FROM points
                        WHERE points.series_id=series.id AND points.samples > 1
                      )""",
                series_id_values,
            )
        self._record_source(
            connection,
            source,
            status="indexed",
            message=f"{raw_value_count} raw values collapsed into {buffered_writes} buffered state-date writes",
            rows_seen=scanned_rows, state_column=state_column, date_column=date_column,
            numeric_columns=numeric_columns,
        )
        progress(
            f"    indexed {mapped_rows:,} mapped rows into "
            f"{buffered_writes:,} state-date writes"
        )

    @staticmethod
    def _flush_points(
        connection: sqlite3.Connection,
        points: Mapping[tuple[int, str, str, str], tuple[float, int]],
        ledger_writer: _RevisionLedgerWriter,
    ) -> int:
        if not points:
            return 0
        ledger_writer.append(connection, [
            (*key, total if samples else None, samples)
            for key, (total, samples) in points.items()
        ])
        # One SQLite upsert per latest key in this batch, instead of one
        # per historical revision. Equal vintages retain sum/count means.
        latest = {}
        for (sid, state, day, vintage), (total, samples) in points.items():
            if not samples:
                continue
            key = (sid, state, day)
            previous = latest.get(key)
            if previous is None or (vintage or "") > (previous[2] or ""):
                latest[key] = (total, samples, vintage)
            elif vintage == previous[2]:
                latest[key] = (total + previous[0], samples + previous[1], vintage)
        connection.executemany(
            """INSERT INTO points(series_id, state, date, value, samples, vintage)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(series_id, state, date) DO UPDATE SET
                 value = CASE WHEN points.vintage IS excluded.vintage THEN points.value + excluded.value
                              WHEN excluded.vintage > COALESCE(points.vintage, '') THEN excluded.value ELSE points.value END,
                 samples = CASE WHEN points.vintage IS excluded.vintage THEN points.samples + excluded.samples
                                WHEN excluded.vintage > COALESCE(points.vintage, '') THEN excluded.samples ELSE points.samples END,
                 vintage = MAX(COALESCE(points.vintage, ''), COALESCE(excluded.vintage, ''))
               WHERE COALESCE(excluded.vintage, '') >= COALESCE(points.vintage, '')""",
            ((*key, *value) for key, value in sorted(latest.items())),
        )
        return len(points)

    @staticmethod
    def _state_column(profiles: Mapping[str, ColumnProfile]) -> str | None:
        candidates = []
        preferred = {name: index for index, name in enumerate(STATE_COLUMN_NAMES)}
        for key, profile in profiles.items():
            if profile.nonmissing == 0 or profile.state_values == 0:
                continue
            ratio = profile.state_values / profile.nonmissing
            name_rank = preferred.get(key.lower(), len(preferred) + 1)
            if ratio >= 0.20:
                candidates.append((name_rank, -ratio, key))
        return min(candidates)[2] if candidates else None

    @staticmethod
    def _date_column(columns: Sequence[str], catalog_column: str | None) -> str | None:
        if catalog_column:
            found = _column(columns, (catalog_column,))
            if found:
                return found
        return _column(columns, DATE_COLUMN_NAMES)

    @staticmethod
    def _dimension_columns(
        profiles: Mapping[str, ColumnProfile], *, excluded: set[str | None]
    ) -> list[str]:
        result = []
        for key, profile in profiles.items():
            lower = key.lower()
            if (
                key in excluded
                or lower in UNHELPFUL_DIMENSION_NAMES
                or lower in NON_MEASURE_COLUMN_NAMES
            ):
                continue
            if lower in FORCED_DIMENSION_NAMES:
                result.append(key)
        return sorted(result)

    @staticmethod
    def _series_id(
        connection: sqlite3.Connection,
        source: TableSource,
        value_column: str,
        state_column: str,
        date_column: str,
        dimensions: tuple[tuple[str, str], ...],
    ) -> int:
        dimensions_json = json.dumps(dict(dimensions), sort_keys=True, separators=(",", ":"))
        identity = json.dumps(
            [source.dataset_key, source.snapshot_id, source.source_path, value_column, dimensions],
            ensure_ascii=False, separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        path_signal = ""
        for part in source.source_path.split("/"):
            if part.startswith("signal="):
                path_signal = part.split("=", 1)[1]
                break
        measure = path_signal if value_column.lower() == "value" and path_signal else value_column
        qualifiers = " · ".join(f"{key}={value}" for key, value in dimensions)
        label = f"{source.dataset_key} · {measure}"
        if qualifiers:
            label += f" · {qualifiers}"
        if "!/" in source.source_path:
            # Hub archives often retain many dated truth snapshots with the same
            # columns. Keep their otherwise-identical choices distinguishable.
            label += f" · file={Path(source.source_path).name}"
        connection.execute(
            """INSERT OR IGNORE INTO series
               (fingerprint, dataset_key, dataset_title, provider, snapshot_id, source_path,
                value_column, state_column, date_column, dimensions, label, full_snapshots, vintage_column)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fingerprint, source.dataset_key, source.dataset_title, source.provider,
                source.snapshot_id, source.source_path, value_column, state_column,
                date_column, dimensions_json, label, int(source.full_snapshots), source.vintage_column,
            ),
        )
        row = connection.execute("SELECT id FROM series WHERE fingerprint = ?", (fingerprint,)).fetchone()
        assert row is not None
        return int(row[0])

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{self.index_path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _revision_rows(self, series_id: int, state: str) -> list[dict[str, Any]]:
        """Read one series/state slice from the columnar revision ledger."""

        if not self.revision_ledger_path.is_file():
            raise FileNotFoundError(
                f"Revision ledger is missing: {self.revision_ledger_path}; rebuild the index"
            )
        try:
            import pyarrow.dataset as dataset  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "Reading the revision ledger requires PyArrow; install "
                "'tapestry[explorer]'"
            ) from error
        source = dataset.dataset(self.revision_ledger_path, format="parquet")
        with closing(self.connect()) as connection:
            meta = dict(connection.execute("SELECT key, value FROM meta"))
        ledger_build_id = (source.schema.metadata or {}).get(b"build_id", b"").decode()
        if ledger_build_id != meta.get("build_id", ""):
            raise _BuildError("SQLite and Parquet build IDs do not match; rebuild the index")
        table = source.to_table(
            columns=["series_id", "state", "event_date", "release_time", "value", "samples"],
            filter=(dataset.field("series_id") == int(series_id))
            & dataset.field("state").isin([state, "US"]),
        )
        # A streaming build can emit the same logical revision in separate
        # Parquet row groups. Re-aggregate those chunks here before resolving
        # the latest release, preserving the old SQLite semantics.
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in table.to_pylist():
            key = (row["event_date"], row["release_time"])
            current = grouped.get(key)
            if current is None:
                current = dict(row)
                current["value"] = None
                current["samples"] = 0
                grouped[key] = current
            samples = int(row["samples"] or 0)
            if samples:
                current["value"] = (current["value"] or 0.0) + float(row["value"] or 0.0)
                current["samples"] += samples
        return list(grouped.values())

    def overview(self) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            meta = dict(connection.execute("SELECT key, value FROM meta"))
            counts = dict(connection.execute(
                "SELECT state, COUNT(DISTINCT series_id) FROM points GROUP BY state"
            ))
            national_count = counts.pop("US", 0)
            states = [{"code": code, "name": STATE_NAMES.get(code, code),
                       "series_count": counts.get(code, 0) + national_count}
                      for code in (STATE_NAMES if national_count else counts)]
            states.sort(key=lambda item: item["name"])
            if national_count:
                states.insert(0, {"code": "US", "name": "United States (US)",
                                  "series_count": national_count})
            datasets = [dict(row) for row in connection.execute(
                """SELECT s.dataset_key, MAX(s.dataset_title) AS title, MAX(s.provider) AS provider,
                          COUNT(DISTINCT s.id) AS series_count,
                          COUNT(p.value) AS point_count
                   FROM series s LEFT JOIN points p ON p.series_id=s.id
                   GROUP BY s.dataset_key ORDER BY s.dataset_key"""
            )]
            canonical = set()
            source_groups = set()
            for row in connection.execute("SELECT dataset_key,value_column,source_path,dimensions FROM series"):
                description = describe(row[0], row[1], row[2], json.loads(row[3]))
                canonical.add(description["signal_key"])
                source_groups.add(description["source_group"])
            selection = self.summary()
            selection.update(indexed_variants=sum(d["series_count"] for d in datasets),
                             indexed_signal_choices=len(canonical),
                             indexed_source_groups=len(source_groups))
            warnings = [dict(row) for row in connection.execute(
                """SELECT dataset_key, source_path, status, message FROM sources
                   WHERE status != 'indexed' ORDER BY dataset_key, source_path"""
            )]
        return {
            "meta": meta,
            "selection": selection,
            "states": states,
            "datasets": datasets,
            "warnings": warnings,
            "aggregation_note": (
                "Repeated raw rows for one state, series, and event date are shown as an "
                "unweighted mean after choosing the latest vintage. National context is stored once."
            ),
        }

    @staticmethod
    def _location_code(location: str) -> str:
        # US is an explorer location, not a state in the shared ingestion normalizer.
        if location.strip().upper() in {"US", "UNITED STATES", "UNITED STATES (US)"}:
            return "US"
        code = normalize_state(location)
        if code is None:
            raise ValueError(f"Unknown location: {location!r}")
        return code

    def list_series(
        self,
        state: str,
        *,
        query: str = "",
        cadence: str = "",
        vintage: str = "",
        support: str = "",
        freshness: str = "",
        limit: int = 250,
        offset: int = 0,
    ) -> dict[str, Any]:
        code = self._location_code(state)
        limit = -1 if int(limit) == 0 else max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        cadence = cadence.strip().lower()
        vintage = vintage.strip().lower()
        support = support.strip().lower()
        freshness = freshness.strip().lower()
        if vintage not in {"", "versioned", "unversioned"}:
            raise ValueError(f"Unknown vintage filter: {vintage!r}")
        if support not in {"", "native_state", "national"}:
            raise ValueError(f"Unknown support filter: {support!r}")
        if freshness not in {"", "current", "lagging"}:
            raise ValueError(f"Unknown freshness filter: {freshness!r}")

        catalog = self._catalog()
        filtered_keys = set(catalog)
        metadata_filtering = bool(cadence or vintage)
        if cadence:
            filtered_keys = {
                key for key in filtered_keys
                if str(catalog[key].get("temporal_resolution", "")).lower() == cadence
            }
        if vintage:
            expected = vintage == "versioned"
            filtered_keys = {
                key for key in filtered_keys if bool(catalog[key].get("versioned")) is expected
            }

        needle = f"%{query.strip().lower()}%"
        where = "p.state IN (?, 'US')"
        params: list[Any] = [code]
        if query.strip():
            where += (
                " AND (LOWER(s.label) LIKE ? OR LOWER(s.dataset_title) LIKE ?"
                " OR LOWER(s.provider) LIKE ? OR LOWER(s.source_path) LIKE ?"
                " OR LOWER(s.value_column) LIKE ?)"
            )
            params.extend([needle] * 5)
        if metadata_filtering:
            if filtered_keys:
                placeholders = ",".join("?" for _ in filtered_keys)
                where += f" AND s.dataset_key IN ({placeholders})"
                params.extend(sorted(filtered_keys))
            else:
                where += " AND 0"
        if support == "native_state":
            where += " AND s.dimensions NOT LIKE '%\"spatial_support\":%'"
        elif support == "national":
            where += " AND s.dimensions LIKE '%\"spatial_support\":\"national parent broadcast\"%'"

        having = ""
        having_params: list[Any] = []
        if freshness:
            keys_for_cutoffs = sorted(filtered_keys if metadata_filtering else catalog)
            cases = []
            today = date.today()
            for key in keys_for_cutoffs:
                cadence_name = str(catalog[key].get("temporal_resolution", "")).lower()
                cutoff = today - timedelta(days=FRESHNESS_DAYS.get(cadence_name, 35))
                cases.append("WHEN ? THEN ?")
                having_params.extend([key, cutoff.isoformat()])
            default_cutoff = (today - timedelta(days=35)).isoformat()
            cutoff_expression = (
                f"CASE s.dataset_key {' '.join(cases)} ELSE ? END"
                if cases else "?"
            )
            having_params.append(default_cutoff)
            operator = ">=" if freshness == "current" else "<"
            having = f" HAVING MAX(p.date) {operator} {cutoff_expression}"
        with closing(self.connect()) as connection:
            total = connection.execute(
                f"""SELECT COUNT(*) FROM (
                       SELECT s.id FROM series s JOIN points p ON p.series_id=s.id
                       WHERE {where} GROUP BY s.id{having})""", [*params, *having_params],
            ).fetchone()[0]
            rows = connection.execute(
                f"""SELECT s.id, s.label, s.dataset_key, s.dataset_title, s.provider,
                            s.source_path, s.value_column, s.dimensions, MIN(p.date) AS date_min,
                            MAX(p.date) AS date_max, COUNT(*) AS point_count,
                            MAX(p.samples) AS max_samples_per_point
                     FROM series s JOIN points p ON p.series_id=s.id
                     WHERE {where} GROUP BY s.id{having}
                     ORDER BY s.dataset_key, s.value_column, s.label
                     LIMIT ? OFFSET ?""",
                [*params, *having_params, limit, offset],
            )
            items = []
            for row in rows:
                item = dict(row)
                item["dimensions"] = json.loads(item["dimensions"])
                metadata = catalog.get(item["dataset_key"], {})
                temporal_resolution = str(metadata.get("temporal_resolution", "unknown"))
                cutoff_days = FRESHNESS_DAYS.get(temporal_resolution.lower(), 35)
                cutoff = date.today() - timedelta(days=cutoff_days)
                date_max = date.fromisoformat(item["date_max"])
                item.update({
                    "temporal_resolution": temporal_resolution,
                    "versioned": bool(metadata.get("versioned")),
                    "revision_mode": metadata.get("revision_mode"),
                    "vintage_semantics": metadata.get("vintage_semantics"),
                    "geographic_resolutions": metadata.get("geographic_resolutions", []),
                    "spatial_support": item["dimensions"].get(
                        "spatial_support", "native state"
                    ),
                    "freshness": "current" if date_max >= cutoff else "lagging",
                    "freshness_days": cutoff_days,
                })
                item.update(series_lineage(metadata, item["value_column"], item["source_path"], item["dimensions"]))
                item.update(describe(item["dataset_key"], item["value_column"], item["source_path"], item["dimensions"], catalog=catalog))
                items.append(item)
        return {
            "state": code,
            "signal_count": len({item["signal_key"] for item in items}),
            "query": query,
            "filters": {
                "cadence": cadence, "vintage": vintage, "support": support,
                "freshness": freshness,
            },
            "total": total,
            "offset": offset,
            "items": items,
        }

    @staticmethod
    def _cutoff(as_of: str | None) -> tuple[str | None, str | None]:
        if not as_of or as_of == "latest":
            return None, None
        try:
            day = date.fromisoformat(as_of).isoformat()
        except ValueError as error:
            raise ValueError("as_of must be an ISO date (YYYY-MM-DD) or latest") from error
        return day, day + "T23:59:59.999999"

    def versions(self, state: str, series_ids: Sequence[int]) -> dict[str, Any]:
        code = self._location_code(state)
        ids = list(dict.fromkeys(int(value) for value in series_ids))
        if len(ids) > 100:
            raise ValueError("At most 100 series can be requested at once")
        dates: set[str] = set()
        with closing(self.connect()) as connection:
            for sid in ids:
                item = connection.execute("SELECT * FROM series WHERE id=?", (sid,)).fetchone()
                if item is None:
                    continue
                dates.update(
                    (row["event_date"] if not row["release_time"] else row["release_time"])[:10]
                    for row in self._revision_rows(sid, code)
                )
        return {"state": code, "dates": sorted(dates)}

    def data(self, state: str, series_ids: Sequence[int], *, scale: bool = False,
             as_of: str | None = None) -> dict[str, Any]:
        code = self._location_code(state)
        day, cutoff = self._cutoff(as_of)
        ids = list(dict.fromkeys(int(value) for value in series_ids))
        if len(ids) > 100:
            raise ValueError("At most 100 series can be requested at once")
        catalog = self._catalog()
        output = []
        with closing(self.connect()) as connection:
            for sid in ids:
                row = connection.execute("SELECT * FROM series WHERE id=?", (sid,)).fetchone()
                if row is None:
                    continue
                item = dict(row)
                item["dimensions"] = json.loads(item["dimensions"])
                dataset = catalog.get(item["dataset_key"], {})
                item.update(series_lineage(dataset, item["value_column"], item["source_path"], item["dimensions"]))
                item.update(describe(item["dataset_key"], item["value_column"], item["source_path"], item["dimensions"], catalog=catalog))
                versioned = bool(dataset.get("versioned"))
                revision_rows = self._revision_rows(sid, code)
                resolved = None
                if day:
                    revision_rows = [row for row in revision_rows if row["event_date"] <= day]
                if versioned and cutoff:
                    revision_rows = [
                        row for row in revision_rows
                        if row["release_time"] and row["release_time"] <= cutoff
                    ]
                if item["full_snapshots"]:
                    resolved = connection.execute(
                        "SELECT MAX(vintage) FROM releases WHERE dataset_key=? AND source_path=? AND vintage<=?",
                        (item["dataset_key"], item["source_path"], cutoff or "9999"),
                    ).fetchone()[0]
                    revision_rows = [
                        row for row in revision_rows if row["release_time"] == (resolved or "unavailable")
                    ]

                # A null latest revision is meaningful: it suppresses an older
                # value. Resolve by release time before dropping rows with no
                # numeric samples.
                latest_by_date: dict[str, dict[str, Any]] = {}
                for revision in revision_rows:
                    event_date = revision["event_date"]
                    current = latest_by_date.get(event_date)
                    if current is None or revision["release_time"] > current["release_time"]:
                        latest_by_date[event_date] = revision
                rows = [
                    row for row in sorted(latest_by_date.values(), key=lambda value: value["event_date"])
                    if row["samples"] > 0
                ]
                points = [
                    [row["event_date"], row["value"] / row["samples"], row["samples"]]
                    for row in rows
                ]
                maximum = max((point[1] for point in points), default=None)
                divisor = maximum if scale and maximum not in {None, 0.0} else 1.0
                item.update(max=maximum, divisor=divisor, versioned=versioned,
                            as_of=day, resolved_vintage=resolved or max((r["release_time"] for r in rows), default=None),
                            version_behavior="Git commit snapshots" if item['source_path'] == 'git-history.ndjson.gz' else "publisher revisions" if versioned else "event-date cutoff",
                            aggregation="unweighted mean within the selected vintage",
                            points=[[d, v / divisor, n] for d, v, n in points])
                if day and versioned and not points:
                    item["version_note"] = "No archived values available by this date."
                elif item['source_path'] == 'git-history.ndjson.gz':
                    item['version_note'] = 'Complete target-file state at the latest eligible main-branch commit. Commit time is a publication proxy, not a provider release date.'
                elif day and versioned and not item["vintage_column"]:
                    item["version_note"] = "Availability is bounded by the saved snapshot/commit date; no row publication dates are recorded."
                output.append(item)
        return {"state": code, "state_name": "United States (US)" if code == "US" else STATE_NAMES[code], "scaled": scale, "as_of": day, "series": output}

