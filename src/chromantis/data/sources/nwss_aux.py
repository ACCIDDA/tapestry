"""Acquire Delphi's full versioned NWSS auxiliary table as an immutable snapshot."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from contextlib import closing
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..http import HttpClient
from ..models import DatasetSpec
from ..repository import RawDataRepository


PAYLOAD = "aux_data.csv.gz"
REQUIRED_COLUMNS = frozenset({
    "report_time", "geo_value", "reference_time", "nwss_source",
    "sample_index", "pcr_target", "state_territory", "major_lab_method",
})


class _HashingReader:
    """Record the exact imported bytes while a raw or gzip stream consumes them."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        self.bytes += len(value)
        return value

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _request_url(spec: DatasetSpec) -> str:
    parts = urlsplit(spec.source_url)
    path = parts.path if parts.path.rstrip("/").endswith("aux_data") else parts.path.rstrip("/") + "/aux_data/"
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("source", str(spec.config["source"]))
    query.setdefault("limit", str(int(spec.config.get("limit", 100_000_000))))
    query.setdefault("format", "csv")
    return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), parts.fragment))


def _headers() -> dict[str, str]:
    key = os.environ.get("DELPHI_EPIDATA_KEY") or os.environ.get("DELPHI_API_KEY")
    return {"Accept-Encoding": "gzip", **({"token": key} if key else {})}


def _copy_csv_to_gzip(source, destination: Path) -> int:
    """Write deterministic gzip, validate its header, and count physical CSV rows."""
    first = source.read(1 << 16)
    if first.lstrip().startswith((b"{", b"[")):
        try:
            detail = json.loads((first + source.read()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = first[:500].decode("utf-8", errors="replace")
        raise RuntimeError(f"Delphi auxiliary endpoint returned an error: {detail}")
    if b"\n" not in first:
        raise RuntimeError("Delphi auxiliary CSV has no complete header row")
    header = first.split(b"\n", 1)[0].rstrip(b"\r").decode("utf-8-sig")
    columns = next(csv.reader([header]))
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise RuntimeError(f"Delphi NWSS auxiliary CSV lacks columns: {sorted(missing)}")

    line_count = first.count(b"\n")
    last = first[-1:] if first else b""
    with destination.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
        output.write(first)
        for chunk in iter(lambda: source.read(1 << 20), b""):
            output.write(chunk)
            line_count += chunk.count(b"\n")
            last = chunk[-1:]
    logical_lines = line_count + int(bool(last) and last != b"\n")
    return max(0, logical_lines - 1)


def _open_import(path: Path):
    raw = path.open("rb")
    hashing = _HashingReader(raw)
    magic = hashing.read(2)
    raw.seek(0)
    hashing.digest = hashlib.sha256()
    hashing.bytes = 0
    if magic == b"\x1f\x8b":
        return raw, hashing, gzip.GzipFile(fileobj=hashing, mode="rb")
    return raw, hashing, hashing


class DelphiV5AuxFetcher:
    """Stream the complete auxiliary response, or explicitly import cached bytes."""

    def __init__(self, client: HttpClient | None = None):
        self.client = client or HttpClient(timeout=1800.0)

    def fetch(
        self,
        repository: RawDataRepository,
        spec: DatasetSpec,
        *,
        import_file: str | Path | None = None,
    ):
        url = _request_url(spec)
        imported = Path(import_file).expanduser().resolve() if import_file else None
        if imported is not None and not imported.is_file():
            raise FileNotFoundError(f"NWSS auxiliary import does not exist: {imported}")

        with repository.begin_snapshot(spec) as snapshot:
            snapshot.preserve_on_failure()
            destination = snapshot.path(PAYLOAD, media_type="text/csv+gzip")
            request = dict(endpoint=url, source=str(spec.config["source"]),
                           limit=int(spec.config.get("limit", 100_000_000)), format="csv")
            snapshot.write_json("request.json", request)
            if imported is not None:
                raw, hashing, source = _open_import(imported)
                try:
                    rows = _copy_csv_to_gzip(source, destination)
                finally:
                    source.close()
                    if source is not hashing:
                        raw.close()
                acquisition = "explicit_local_import"
                import_record = dict(path=str(imported), sha256=hashing.digest.hexdigest(),
                                     bytes=hashing.bytes)
                snapshot.write_json("import.json", import_record)
                source_state = dict(source=str(spec.config["source"]), rows=rows,
                                    acquisition=acquisition, imported_file=import_record)
            else:
                with closing(self.client.open(url, headers=_headers())) as response:
                    encoding = response.headers.get("Content-Encoding", "").lower()
                    if encoding not in ("", "identity", "gzip"):
                        raise RuntimeError(f"Unsupported HTTP content encoding: {encoding}")
                    source = gzip.GzipFile(fileobj=response, mode="rb") if encoding == "gzip" else response
                    try:
                        rows = _copy_csv_to_gzip(source, destination)
                    finally:
                        if source is not response:
                            source.close()
                    source_state = dict(source=str(spec.config["source"]), rows=rows,
                        acquisition="download", final_url=response.geturl(),
                        etag=response.headers.get("ETag"), last_modified=response.headers.get("Last-Modified"),
                        content_length=response.headers.get("Content-Length"))
                acquisition = "download"
            snapshot.set_file_details(PAYLOAD, rows=rows, media_type="text/csv+gzip")
            return snapshot.commit(
                selector={**request, "acquisition": acquisition,
                          "import_file": str(imported) if imported is not None else None},
                source_state=source_state,
            )
