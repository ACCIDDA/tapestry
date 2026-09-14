"""Raw, vintage-aware covariate acquisition and storage."""

from .catalog import CATALOG, get_spec
from .models import DatasetSpec, FileRecord, RevisionMode, SnapshotManifest
from .repository import RawDataRepository

__all__ = [
    "CATALOG",
    "DatasetSpec",
    "FileRecord",
    "RawDataRepository",
    "RevisionMode",
    "SnapshotManifest",
    "get_spec",
]
