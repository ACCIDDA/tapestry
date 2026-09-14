"""Delphi V5 acquisition using epidatpy query construction and raw CSV storage."""

from __future__ import annotations

import gzip
import http.client
import json
import os
import shutil
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path
from urllib.error import URLError

from epidatpy import EpiDataContext
from requests import Session

from ..http import HttpClient, with_query
from ..models import DatasetSpec
from ..repository import RawDataRepository



def _safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _api_key() -> str | None:
    return os.environ.get("DELPHI_EPIDATA_KEY") or os.environ.get("DELPHI_API_KEY")


def _api_headers() -> dict[str, str]:
    key = _api_key()
    return {"token": key} if key else {}


def _source_metadata(source: str) -> dict:
    # The official client reads DELPHI_EPIDATA_KEY itself. A session header also
    # preserves this project's DELPHI_API_KEY alias without modifying os.environ.
    with Session() as session:
        session.headers.update(_api_headers())
        return EpiDataContext(session=session, use_cache=False).epidata_meta(source=source)


def _copy_csv_response_to_gzip_once(client: HttpClient, url: str, destination: Path) -> int:
    """Copy one cast-API response and return its physical CSV data-line count."""

    headers = {**_api_headers(), "Accept-Encoding": "gzip"}
    with closing(client.open(url, headers=headers)) as response:
        content_type = response.headers.get_content_type()
        content_encoding = response.headers.get("Content-Encoding", "").lower()
        if content_encoding not in {"", "identity", "gzip"}:
            raise RuntimeError(
                f"Delphi cast API returned unsupported content encoding {content_encoding!r}"
            )

        payload = (
            gzip.GzipFile(fileobj=response, mode="rb")
            if content_encoding == "gzip"
            else response
        )
        try:
            first = payload.read(1 << 16)
            if content_type == "application/json" or first.lstrip().startswith(b"{"):
                body = first + payload.read()
                try:
                    detail = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    detail = body[:500].decode("utf-8", errors="replace")
                raise RuntimeError(f"Delphi cast API returned an error: {detail}")

            line_count = first.count(b"\n")
            with destination.open("wb") as raw_stream, gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_stream, mtime=0
            ) as compressed:
                compressed.write(first)
                for chunk in iter(lambda: payload.read(1 << 20), b""):
                    line_count += chunk.count(b"\n")
                    compressed.write(chunk)
            return max(0, line_count - 1)
        finally:
            if payload is not response:
                payload.close()


def _copy_csv_response_to_gzip(
    client: HttpClient,
    url: str,
    destination: Path,
    *,
    stream_attempts: int = 5,
) -> int:
    """Retry a whole partition when a streamed HTTP response is interrupted."""

    if stream_attempts <= 0:
        raise ValueError("stream_attempts must be positive")
    transient = (
        http.client.IncompleteRead,
        http.client.RemoteDisconnected,
        ConnectionResetError,
        TimeoutError,
        URLError,
        EOFError,
        gzip.BadGzipFile,
        zlib.error,
    )
    for attempt in range(stream_attempts):
        try:
            return _copy_csv_response_to_gzip_once(client, url, destination)
        except transient:
            destination.unlink(missing_ok=True)
            if attempt + 1 == stream_attempts:
                raise
            time.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def _gzip_csv_rows(path: Path) -> int:
    """Validate a completed gzip member and count physical CSV data rows."""

    line_count = 0
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            line_count += chunk.count(b"\n")
    return max(0, line_count - 1)


class DelphiV5Fetcher:
    """Fetch one raw CSV per signal/geography from snapshot or archive views."""

    def __init__(self, client: HttpClient | None = None):
        self.client = client or HttpClient(timeout=600.0)

    def fetch(
        self,
        repository: RawDataRepository,
        spec: DatasetSpec,
        *,
        mode: str = "archive",
        snapshot_date: str | None = None,
        report_time: str | None = None,
        fill_method: str | None = None,
        signals: tuple[str, ...] | None = None,
        geo_types: tuple[str, ...] | None = None,
        workers: int = 4,
        resume_from: str | Path | None = None,
    ):
        if mode not in {"archive", "snapshot"}:
            raise ValueError("mode must be 'archive' or 'snapshot'")
        if mode == "archive" and snapshot_date is not None:
            raise ValueError("snapshot_date is only valid in snapshot mode")
        if mode == "snapshot" and report_time is not None:
            raise ValueError("report_time is only valid in archive mode")
        if workers <= 0:
            raise ValueError("workers must be positive")

        source = str(spec.config["source"])
        selected_signals = signals or tuple(str(item) for item in spec.config["signals"])
        selected_geo_types = geo_types or tuple(str(item) for item in spec.config["geo_types"])
        metadata = _source_metadata(source)
        source_meta = metadata.get(source)
        if not isinstance(source_meta, dict):
            raise ValueError(f"Delphi V5 metadata does not contain source {source!r}")
        for label, selected in (("signals", selected_signals), ("geo_types", selected_geo_types)):
            unknown = set(selected) - set(source_meta.get(label, ()))
            if unknown:
                raise ValueError(f"Unknown V5 {label} for {source}: {', '.join(sorted(unknown))}")
        # Delegate endpoint names and version-query validation to Delphi's client.
        # Its request_arguments() API lets us stream large CSV archives while
        # preserving publisher-native columns and representations.
        epidata = EpiDataContext(use_cache=False)
        query = {
            "source": source,
            "mode": mode,
            "signals": list(selected_signals),
            "geo_types": list(selected_geo_types),
            "snapshot_date": snapshot_date,
            "report_time": report_time,
            "fill_method": fill_method,
        }
        urls = {}
        for signal in selected_signals:
            for geo_type in selected_geo_types:
                kwargs = dict(source=source, signals=signal, geo_type=geo_type,
                              reference_time="*", geo_values="*", fill_method=fill_method)
                call = (
                    epidata.epidata_snapshot(**kwargs, snapshot_date=snapshot_date)
                    if mode == "snapshot"
                    else epidata.epidata_archive(**kwargs, report_time=report_time or "*")
                )
                url, params = call.request_arguments()
                urls[signal, geo_type] = with_query(url, {**params, "format": "csv"})
        resume_root = Path(resume_from).expanduser().resolve() if resume_from else None
        if resume_root is not None and not resume_root.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {resume_root}")

        if resume_root is not None:
            request_path = resume_root / "request.json"
            if not request_path.is_file() or json.loads(request_path.read_text()) != query:
                raise ValueError("Resume directory must contain request.json matching this exact V5 query")

        rows_by_file: dict[str, int] = {}
        resumed_files: list[str] = []
        with repository.begin_snapshot(spec) as snapshot:
            recovery_root = snapshot.preserve_on_failure()
            snapshot.write_json("metadata.json", metadata)
            snapshot.write_json("request.json", query)
            jobs: list[tuple[str, str, Path]] = []
            for signal in selected_signals:
                for geo_type in selected_geo_types:
                    relative = (
                        f"signal={_safe_component(signal)}/"
                        f"geo_type={_safe_component(geo_type)}/{mode}.csv.gz"
                    )
                    destination = snapshot.path(
                        relative, media_type="text/csv+gzip"
                    )
                    recovered = resume_root / relative if resume_root is not None else None
                    if recovered is not None and recovered.is_file():
                        try:
                            rows = _gzip_csv_rows(recovered)
                        except (EOFError, OSError, zlib.error):
                            pass
                        else:
                            shutil.copy2(recovered, destination)
                            snapshot.set_file_details(
                                relative, rows=rows, media_type="text/csv+gzip"
                            )
                            rows_by_file[relative] = rows
                            resumed_files.append(relative)
                            continue
                    url = urls[signal, geo_type]
                    jobs.append((relative, url, destination))

            executor = ThreadPoolExecutor(max_workers=workers)
            futures = {
                executor.submit(
                    _copy_csv_response_to_gzip, self.client, url, destination
                ): relative
                for relative, url, destination in jobs
            }
            try:
                for future in as_completed(futures):
                    relative = futures[future]
                    try:
                        rows = future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"Delphi partition {relative!r} failed after download retries. "
                            f"Completed partitions were retained at {recovery_root}; retry with "
                            f"--resume-from {recovery_root} (and use --workers 1 if interruptions "
                            "continue)."
                        ) from error
                    snapshot.set_file_details(
                        relative, rows=rows, media_type="text/csv+gzip"
                    )
                    rows_by_file[relative] = rows
            except BaseException:
                # Do not spend hours draining work that can no longer be
                # committed after one partition has failed.
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

            if not any(rows_by_file.values()):
                raise RuntimeError(f"Delphi source {source} returned no data")
            return snapshot.commit(
                selector={
                    "mode": mode,
                    "signals": list(selected_signals),
                    "geo_types": list(selected_geo_types),
                    "snapshot_date": snapshot_date,
                    "report_time": report_time,
                    "fill_method": fill_method,
                    "workers": workers,
                    "resume_from": str(resume_root) if resume_root else None,
                },
                source_state={
                    "source": source,
                    "rows": sum(rows_by_file.values()),
                    "resumed_files": resumed_files,
                    "metadata": metadata.get(source, {}),
                },
            )

