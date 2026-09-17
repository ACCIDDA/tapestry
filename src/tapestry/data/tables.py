"""Streaming raw-table access shared by selection and consumers.

Tables from tar exports must be consumed inside the iterator's current step;
its extracted temporary file is removed when the iterator advances.
"""
from __future__ import annotations
import csv
import gzip
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping
from .catalog import CATALOG
from .columns import saved_columns

SUPPORTED_TABLE_SUFFIXES = (".csv", ".csv.gz", ".ndjson", ".ndjson.gz", ".jsonl", ".jsonl.gz", ".parquet")

def _vintage_string(value):
    if value is None:
        return None
    timestamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp.isoformat()

@dataclass(frozen=True)
class TableSource:
    dataset_key: str
    dataset_title: str
    provider: str
    snapshot_id: str
    source_path: str
    event_date_column: str | None
    vintage_column: str | None
    geographic_resolutions: tuple[str, ...]
    missing_markers: tuple[str, ...]
    iter_rows: Callable[[], Iterator[dict[str, Any]]]
    revision_mode: str = "snapshot_only"
    fallback_vintage: str | None = None
    full_snapshots: bool = False
    release_times: tuple[str, ...] = ()
    path: Path | None = None


@dataclass(frozen=True)
class Artifact:
    dataset: Mapping[str, Any]
    snapshot_id: str
    snapshot_dir: Path
    relative_path: str
    path: Path


class RawTables:
    """Read the latest locally acquired snapshot of each catalog entry."""
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root).expanduser().resolve()

    def _catalog(self) -> dict[str, Mapping[str, Any]]:
        path = self.data_root / "catalog.json"
        if not path.is_file():
            raise FileNotFoundError(f"No catalog found at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = {str(item["key"]): dict(item) for item in payload.get("datasets", [])}
        # Enrich older on-disk catalogs without rewriting immutable acquisitions.
        for key, item in result.items():
            if key in CATALOG:
                item["parent_dataset"] = CATALOG[key].parent_dataset
                item["column_lineage"] = CATALOG[key].column_lineage
            if key.startswith("cdc_"):
                pointer = self.data_root / "raw" / key / "latest.json"
                if pointer.is_file():
                    snapshot = pointer.parent / "snapshots" / json.loads(pointer.read_text())["snapshot_id"]
                    column_path = snapshot / "columns.json"
                    if not column_path.is_file():
                        column_path = snapshot / "metadata.json"
                    if column_path.is_file():
                        item["column_metadata"] = saved_columns(column_path, column_path.stat().st_mtime_ns)
        return result

    def artifacts(self) -> list[Artifact]:
        catalog = self._catalog()
        artifacts: list[Artifact] = []
        for key, dataset in sorted(catalog.items()):
            pointer = self.data_root / "raw" / key / "latest.json"
            if not pointer.is_file():
                continue
            snapshot_id = str(json.loads(pointer.read_text(encoding="utf-8"))["snapshot_id"])
            snapshot_dir = self.data_root / "raw" / key / "snapshots" / snapshot_id
            manifest_path = snapshot_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for record in manifest.get("files", []):
                relative = str(record.get("path", ""))
                path = snapshot_dir / relative
                if path.is_file() and self._is_payload(relative):
                    artifacts.append(Artifact(dataset, snapshot_id, snapshot_dir, relative, path))
        return artifacts

    @staticmethod
    def _is_payload(name: str) -> bool:
        lower = name.lower()
        if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
            return True
        return any(lower.endswith(suffix) for suffix in SUPPORTED_TABLE_SUFFIXES)

    def _table_sources(self, artifact: Artifact) -> Iterator[TableSource]:
        lower = artifact.relative_path.lower()
        if lower.endswith((".tar.gz", ".tgz")):
            yield from self._tar_sources(artifact)
            return
        yield self._source_for_path(artifact, artifact.path, artifact.relative_path)

    def _tar_sources(self, artifact: Artifact) -> Iterator[TableSource]:
        with tarfile.open(artifact.path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not self._is_primary_hub_member(member.name):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                suffix = "".join(Path(member.name).suffixes) or ".table"
                temporary = tempfile.NamedTemporaryFile(
                    prefix="tapestry-table-", suffix=suffix, delete=False
                )
                temporary_path = Path(temporary.name)
                try:
                    with temporary:
                        shutil.copyfileobj(extracted, temporary)
                    source_name = f"{artifact.relative_path}!/{member.name}"
                    yield self._source_for_path(artifact, temporary_path, source_name, member.name)
                finally:
                    temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _is_tabular_member(name: str) -> bool:
        lower = name.lower()
        return any(lower.endswith(suffix) for suffix in SUPPORTED_TABLE_SUFFIXES)

    @classmethod
    def _is_primary_hub_member(cls, name: str) -> bool:
        """Index current/canonical Hub data, not bundled auxiliaries or old snapshots."""

        parts = PurePosixPath(name).parts
        if not parts or parts[0].lower() not in {"target-data", "data-truth"}:
            return False
        if (
            len(parts) > 1
            and parts[0].lower() == "target-data"
            and parts[1].lower() == "archive"
        ):
            return False
        return cls._is_tabular_member(name)

    def _source_for_path(
        self,
        artifact: Artifact,
        path: Path,
        source_name: str,
        format_name: str | None = None,
    ) -> TableSource:
        dataset = artifact.dataset
        name = (format_name or artifact.relative_path).lower()

        def rows() -> Iterator[dict[str, Any]]:
            yield from self._rows(path, name)

        manifest_path = artifact.snapshot_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        from .sources.hub_history import HISTORY_FILE, HISTORY_INDEX
        git_history = artifact.relative_path == HISTORY_FILE
        history = json.loads((artifact.snapshot_dir / HISTORY_INDEX).read_text()) if git_history else {}
        fallback = _vintage_string(manifest.get("source_state", {}).get("commit_time") or manifest.get("retrieved_at"))
        mode = str(dataset.get("revision_mode", "snapshot_only"))
        return TableSource(
            dataset_key=str(dataset["key"]),
            dataset_title=str(dataset.get("title", dataset["key"])),
            provider=str(dataset.get("provider", "")),
            snapshot_id=artifact.snapshot_id,
            source_path=source_name,
            event_date_column='date' if git_history else dataset.get("event_date_column"),
            vintage_column='_git_release' if git_history else dataset.get("vintage_column"),
            geographic_resolutions=tuple(
                str(item) for item in dataset.get("geographic_resolutions", ())
            ),
            missing_markers=tuple(str(item) for item in dataset.get("missing_value_markers", [])),
            iter_rows=rows,
            revision_mode='git_history' if git_history else mode,
            fallback_vintage=None if git_history else fallback if dataset.get("versioned") else None,
            full_snapshots=git_history or (mode == "as_of_column" and dataset.get("vintage_column") == "as_of"),
            release_times=tuple(_vintage_string(r['release_time']) for r in history.get('releases', [])),
            path=path,
        )

    @staticmethod
    def _rows(path: Path, format_name: str) -> Iterator[dict[str, Any]]:
        if format_name.endswith(".parquet"):
            try:
                import pyarrow.parquet as parquet  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError(
                    "Parquet table found; install the optional explorer dependency: "
                    "pip install 'tapestry[explorer]'"
                ) from error
            table = parquet.ParquetFile(path)
            for batch in table.iter_batches(batch_size=50_000):
                for row in batch.to_pylist():
                    yield {str(key): value for key, value in row.items()}
            return

        with path.open("rb") as probe:
            if probe.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Git LFS pointer: truth payload is unavailable; fetch it during intake")
        is_gzip = format_name.endswith(".gz")
        is_csv = format_name.endswith(".csv") or format_name.endswith(".csv.gz")
        opener: Callable[..., Any] = gzip.open if is_gzip else open
        with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as stream:
            if is_csv:
                csv.field_size_limit(16 * 1024 * 1024)
                for row in csv.DictReader(stream):
                    yield {str(key): value for key, value in row.items() if key is not None}
            else:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield {str(key): value for key, value in row.items()}
