"""Immutable raw snapshots with checksums and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import DatasetSpec, FileRecord, SnapshotManifest


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RawDataRepository:
    """Filesystem repository consumed by acquisition now and model datasets later."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def mirrors_dir(self) -> Path:
        return self.root / "mirrors"

    def initialize(self, catalog: dict[str, DatasetSpec]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.mirrors_dir.mkdir(exist_ok=True)
        (self.root / ".staging").mkdir(exist_ok=True)
        self._atomic_write_json(
            self.root / "catalog.json",
            {
                "schema_version": 1,
                "generated_at": _utc_now().isoformat(),
                "datasets": [catalog[key].to_dict() for key in sorted(catalog)],
            },
        )

    def begin_snapshot(self, spec: DatasetSpec) -> "SnapshotWriter":
        if not (self.root / "catalog.json").exists():
            raise FileNotFoundError(
                f"Repository {self.root} is not initialized; call initialize() first"
            )
        return SnapshotWriter(self, spec)

    def list_snapshots(self, dataset_key: str) -> tuple[SnapshotManifest, ...]:
        snapshots_dir = self.raw_dir / dataset_key / "snapshots"
        if not snapshots_dir.exists():
            return ()
        manifests = []
        for path in sorted(snapshots_dir.glob("*/manifest.json")):
            manifests.append(SnapshotManifest.from_dict(json.loads(path.read_text())))
        return tuple(manifests)

    def latest(self, dataset_key: str) -> SnapshotManifest:
        pointer = self.raw_dir / dataset_key / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError(f"No snapshots found for {dataset_key!r}")
        snapshot_id = json.loads(pointer.read_text())["snapshot_id"]
        manifest_path = (
            self.raw_dir / dataset_key / "snapshots" / snapshot_id / "manifest.json"
        )
        return SnapshotManifest.from_dict(json.loads(manifest_path.read_text()))

    def snapshot_path(self, manifest: SnapshotManifest) -> Path:
        return (
            self.raw_dir
            / manifest.dataset_key
            / "snapshots"
            / manifest.snapshot_id
        )

    def verify_snapshot(self, manifest: SnapshotManifest) -> None:
        """Raise when a payload file is missing or differs from its manifest."""

        snapshot = self.snapshot_path(manifest)
        failures = []
        expected_paths = {record.path for record in manifest.files}
        actual_paths = {
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        for unexpected in sorted(actual_paths - expected_paths):
            failures.append(f"unexpected: {unexpected}")
        for record in manifest.files:
            relative = Path(record.path)
            if relative.is_absolute() or ".." in relative.parts:
                failures.append(f"unsafe manifest path: {record.path}")
                continue
            path = snapshot / relative
            if not path.is_file():
                failures.append(f"missing: {record.path}")
                continue
            actual_bytes = path.stat().st_size
            if actual_bytes != record.bytes:
                failures.append(
                    f"size mismatch: {record.path} ({actual_bytes} != {record.bytes})"
                )
                continue
            actual_sha256 = _sha256(path)
            if actual_sha256 != record.sha256:
                failures.append(f"checksum mismatch: {record.path}")
        if failures:
            raise RuntimeError(
                f"Snapshot {manifest.dataset_key}/{manifest.snapshot_id} failed integrity "
                f"verification: {'; '.join(failures)}"
            )

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(_json_bytes(value))
        temporary.replace(path)


class SnapshotWriter(AbstractContextManager["SnapshotWriter"]):
    """Build a snapshot in staging and publish it only after all files succeed."""

    def __init__(self, repository: RawDataRepository, spec: DatasetSpec):
        self.repository = repository
        self.spec = spec
        now = _utc_now()
        self.retrieved_at = now.isoformat()
        self.snapshot_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
        self._staging = (
            repository.root
            / ".staging"
            / spec.key
            / f"{self.snapshot_id}-{uuid.uuid4().hex}"
        )
        self._staging.mkdir(parents=True)
        self._file_details: dict[str, tuple[int | None, str | None]] = {}
        self._committed = False
        self._preserve_on_failure = False

    def __enter__(self) -> "SnapshotWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        preserve = exc_type is not None and self._preserve_on_failure
        if not self._committed and not preserve and self._staging.exists():
            shutil.rmtree(self._staging)
        return False

    def preserve_on_failure(self) -> Path:
        """Keep this staging tree on an exceptional exit so a pull can resume."""

        self._preserve_on_failure = True
        return self._staging

    def path(
        self,
        relative_path: str,
        *,
        rows: int | None = None,
        media_type: str | None = None,
    ) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Snapshot path must be relative and contained: {relative_path}")
        result = self._staging / relative
        result.parent.mkdir(parents=True, exist_ok=True)
        self._file_details[relative.as_posix()] = (rows, media_type)
        return result

    def set_file_details(
        self,
        relative_path: str,
        *,
        rows: int | None = None,
        media_type: str | None = None,
    ) -> None:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Snapshot path must be relative and contained: {relative_path}")
        if not (self._staging / relative).is_file():
            raise FileNotFoundError(self._staging / relative)
        self._file_details[relative.as_posix()] = (rows, media_type)

    def write_json(self, relative_path: str, value: Any) -> Path:
        destination = self.path(relative_path, media_type="application/json")
        destination.write_bytes(_json_bytes(value))
        return destination

    def commit(
        self,
        *,
        selector: dict[str, Any],
        source_state: dict[str, Any] | None = None,
    ) -> SnapshotManifest:
        if self._committed:
            raise RuntimeError("Snapshot was already committed")

        files = []
        for path in sorted(item for item in self._staging.rglob("*") if item.is_file()):
            relative = path.relative_to(self._staging).as_posix()
            rows, media_type = self._file_details.get(relative, (None, None))
            files.append(
                FileRecord(
                    path=relative,
                    sha256=_sha256(path),
                    bytes=path.stat().st_size,
                    rows=rows,
                    media_type=media_type,
                )
            )

        if not files:
            raise RuntimeError(f"Refusing to commit an empty snapshot for {self.spec.key}")

        manifest = SnapshotManifest(
            schema_version=1,
            dataset_key=self.spec.key,
            snapshot_id=self.snapshot_id,
            retrieved_at=self.retrieved_at,
            source_url=self.spec.source_url,
            revision_mode=self.spec.revision_mode,
            versioned=self.spec.versioned,
            selector=selector,
            files=tuple(files),
            source_state=source_state or {},
        )
        self.write_json("manifest.json", manifest.to_dict())

        final = (
            self.repository.raw_dir
            / self.spec.key
            / "snapshots"
            / self.snapshot_id
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        self._staging.replace(final)
        self.repository._atomic_write_json(
            self.repository.raw_dir / self.spec.key / "latest.json",
            {"snapshot_id": self.snapshot_id},
        )
        self._committed = True
        return manifest
