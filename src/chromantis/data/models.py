"""Stable metadata objects shared by acquisition code and downstream datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RevisionMode(StrEnum):
    """How historical versions can be recovered from a source."""

    NONE = "none"
    SNAPSHOT_ONLY = "snapshot_only"
    INITIAL_RELEASE = "initial_release"
    AS_OF_COLUMN = "as_of_column"
    REPORT_TIME = "report_time"
    GIT_HISTORY = "git_history"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """A catalog entry for one independently retrievable raw dataset."""

    key: str
    title: str
    provider: str
    fetcher: str
    source_url: str
    description: str
    revision_mode: RevisionMode
    temporal_resolution: str
    geographic_resolutions: tuple[str, ...]
    measures: tuple[str, ...]
    parent_dataset: str = ""
    column_lineage: tuple[dict[str, Any], ...] = ()
    natural_key: tuple[str, ...] = ()
    event_date_column: str | None = None
    vintage_column: str | None = None
    snapshot_semantics: str = "full"
    vintage_semantics: str | None = None
    missing_value_markers: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    groups: tuple[str, ...] = ("all",)

    @property
    def versioned(self) -> bool:
        return self.revision_mode in {
            RevisionMode.INITIAL_RELEASE,
            RevisionMode.AS_OF_COLUMN,
            RevisionMode.REPORT_TIME,
            RevisionMode.GIT_HISTORY,
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["revision_mode"] = self.revision_mode.value
        result["versioned"] = self.versioned
        return result


@dataclass(frozen=True, slots=True)
class FileRecord:
    """Integrity and size information for one acquired file."""

    path: str
    sha256: str
    bytes: int
    rows: int | None = None
    media_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Machine-readable provenance for an immutable acquisition snapshot."""

    schema_version: int
    dataset_key: str
    snapshot_id: str
    retrieved_at: str
    source_url: str
    revision_mode: RevisionMode
    versioned: bool
    selector: dict[str, Any]
    files: tuple[FileRecord, ...]
    source_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["revision_mode"] = self.revision_mode.value
        result["files"] = [item.to_dict() for item in self.files]
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SnapshotManifest":
        return cls(
            schema_version=int(value["schema_version"]),
            dataset_key=str(value["dataset_key"]),
            snapshot_id=str(value["snapshot_id"]),
            retrieved_at=str(value["retrieved_at"]),
            source_url=str(value["source_url"]),
            revision_mode=RevisionMode(value["revision_mode"]),
            versioned=bool(value["versioned"]),
            selector=dict(value.get("selector", {})),
            files=tuple(FileRecord(**item) for item in value.get("files", [])),
            source_state=dict(value.get("source_state", {})),
        )
