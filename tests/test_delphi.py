from __future__ import annotations

import gzip
import http.client
import io
import os
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from epidatpy import EpiDataContext

from tapestry.data.catalog import CATALOG
from tapestry.data.repository import RawDataRepository
from tapestry.data.sources.delphi import (
    DelphiV5Fetcher,
    _api_key,
    _copy_csv_response_to_gzip,
    _gzip_csv_rows,
)


class _CsvResponse(io.BytesIO):
    def __init__(self, body: bytes):
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Type"] = "text/csv"


class _Client:
    def __init__(self):
        self.urls: list[str] = []
        self.request_headers: list[dict[str, str]] = []

    def open(self, url: str, *, headers=None):
        self.urls.append(url)
        self.request_headers.append(headers or {})
        return _CsvResponse(
            b"signal,report_time,geo_type,geo_value,fill_method,reference_time,value\n"
            b"test,2026-01-02,state,ny,source,2026-01-01,1.5\n"
        )


class _InterruptedResponse(_CsvResponse):
    def __init__(self):
        super().__init__(b"")
        self.reads = 0

    def read(self, size=-1):
        self.reads += 1
        if self.reads == 1:
            return b"a,b\n"
        raise http.client.IncompleteRead(b"1,2")


class _RetryClient:
    def __init__(self):
        self.calls = 0

    def open(self, url: str, *, headers=None):
        self.calls += 1
        if self.calls == 1:
            return _InterruptedResponse()
        return _CsvResponse(b"a,b\n1,2\n")


class _GzipClient:
    def __init__(self):
        self.headers: dict[str, str] = {}

    def open(self, url: str, *, headers=None):
        self.headers = headers or {}
        response = _CsvResponse(gzip.compress(b"a,b\n1,2\n", mtime=0))
        response.headers["Content-Encoding"] = "gzip"
        return response


class _TruncatedGzipRetryClient:
    def __init__(self):
        self.calls = 0

    def open(self, url: str, *, headers=None):
        self.calls += 1
        body = gzip.compress(b"a,b\n1,2\n", mtime=0)
        if self.calls == 1:
            body = body[:-8]
        response = _CsvResponse(body)
        response.headers["Content-Encoding"] = "gzip"
        return response


class _PartlyFailingClient(_Client):
    def __init__(self):
        super().__init__()
        self.download_calls = 0

    def open(self, url: str, *, headers=None):
        self.urls.append(url)
        self.request_headers.append(headers or {})
        self.download_calls += 1
        if self.download_calls == 1:
            return _CsvResponse(b"a,b\n1,2\n")
        return _InterruptedResponse()


class DelphiTests(unittest.TestCase):
    def setUp(self):
        def metadata(source):
            spec = CATALOG[f"delphi_{source}"]
            return {source: {"signals": [*spec.config["signals"], "test", "first", "second"],
                             "geo_types": spec.config["geo_types"]}}
        self.metadata = patch.object(EpiDataContext, "epidata_meta", side_effect=metadata).start()
        self.addCleanup(patch.stopall)

    def test_api_key_alias_and_precedence(self):
        with patch.dict(os.environ, {"DELPHI_API_KEY": "alias"}, clear=True):
            self.assertEqual(_api_key(), "alias")
        with patch.dict(
            os.environ,
            {"DELPHI_API_KEY": "alias", "DELPHI_EPIDATA_KEY": "canonical"},
            clear=True,
        ):
            self.assertEqual(_api_key(), "canonical")

    def test_v5_uses_canonical_https_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = RawDataRepository(Path(temporary) / "data")
            repository.initialize(CATALOG)
            client = _Client()
            manifest = DelphiV5Fetcher(client).fetch(
                repository,
                CATALOG["delphi_nssp"],
                signals=("test",),
                geo_types=("state",),
                workers=1,
            )
            self.assertEqual(sum(item.rows or 0 for item in manifest.files), 1)
            self.assertTrue(all(url.startswith("https://") for url in client.urls))
            self.metadata.assert_called_once_with(source="nssp")
            self.assertTrue(any("/archive/?" in url for url in client.urls))
            self.assertTrue(any("format=csv" in url for url in client.urls))
            self.assertTrue(
                all(
                    headers.get("Accept-Encoding") == "gzip"
                    for headers in client.request_headers
                )
            )

    def test_gzip_row_count_validates_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(b"a,b\n1,2\n3,4\n")
            self.assertEqual(_gzip_csv_rows(path), 2)

    def test_streamed_partition_is_retried_from_the_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv.gz"
            client = _RetryClient()
            with patch("tapestry.data.sources.delphi.time.sleep"):
                rows = _copy_csv_response_to_gzip(client, "https://example.test/", path)
            self.assertEqual(rows, 1)
            self.assertEqual(client.calls, 2)
            self.assertEqual(_gzip_csv_rows(path), 1)

    def test_gzip_transfer_is_decompressed_and_recompressed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv.gz"
            client = _GzipClient()
            rows = _copy_csv_response_to_gzip(client, "https://example.test/", path)
            self.assertEqual(rows, 1)
            self.assertEqual(client.headers.get("Accept-Encoding"), "gzip")
            self.assertEqual(gzip.decompress(path.read_bytes()), b"a,b\n1,2\n")

    def test_truncated_gzip_transfer_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv.gz"
            client = _TruncatedGzipRetryClient()
            with patch("tapestry.data.sources.delphi.time.sleep"):
                rows = _copy_csv_response_to_gzip(client, "https://example.test/", path)
            self.assertEqual(rows, 1)
            self.assertEqual(client.calls, 2)
            self.assertEqual(_gzip_csv_rows(path), 1)

    def test_failed_v5_pull_retains_completed_partitions_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = RawDataRepository(Path(temporary) / "data")
            repository.initialize(CATALOG)
            client = _PartlyFailingClient()
            with patch("tapestry.data.sources.delphi.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "--resume-from") as raised:
                    DelphiV5Fetcher(client).fetch(
                        repository,
                        CATALOG["delphi_nssp"],
                        signals=("first", "second"),
                        geo_types=("state",),
                        workers=1,
                    )

            staging = tuple((repository.root / ".staging" / "delphi_nssp").iterdir())
            self.assertEqual(len(staging), 1)
            self.assertIn(str(staging[0]), str(raised.exception))
            completed = staging[0] / "signal=first/geo_type=state/archive.csv.gz"
            self.assertEqual(_gzip_csv_rows(completed), 1)

    def test_official_query_version_semantics(self):
        for mode, version in (("archive", {"report_time": "<2026-09-01"}),
                              ("snapshot", {"snapshot_date": "2026-09-01"})):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                repository = RawDataRepository(temporary)
                repository.initialize(CATALOG)
                client = _Client()
                manifest = DelphiV5Fetcher(client).fetch(
                    repository, CATALOG["delphi_nhsn"], mode=mode, **version,
                    signals=("confirmed_admissions_flu_ew",), geo_types=("state",), workers=1,
                )
                url = urlparse(client.urls[0])
                self.assertEqual(url.path, f"/epidata/v5/{mode}/")
                params = parse_qs(url.query)
                self.assertEqual(params["source"], ["nhsn"])
                self.assertNotIn("fill_method", params)
                wire_key = "report_time_query" if mode == "archive" else "snapshot_date"
                self.assertEqual(params[wire_key], [next(iter(version.values()))])
                for forbidden in ("issues", "epiweeks", "as_of", "time_values", "api_key"):
                    self.assertNotIn(forbidden, params)
                self.assertEqual(manifest.selector[next(iter(version))], next(iter(version.values())))
                repository.verify_snapshot(manifest)

    def test_fill_variant_is_only_sent_when_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = RawDataRepository(temporary)
            repository.initialize(CATALOG)
            client = _Client()
            manifest = DelphiV5Fetcher(client).fetch(
                repository, CATALOG["delphi_nhsn"], signals=("confirmed_admissions_flu_ew",),
                geo_types=("state",), fill_method="source", workers=1,
            )
            self.assertEqual(parse_qs(urlparse(client.urls[0]).query)["fill_method"], ["source"])
            self.assertEqual(manifest.selector["fill_method"], "source")

    def test_invalid_modes_and_metadata_do_not_download(self):
        for options in ({"mode": "archive", "snapshot_date": "2026-01-01"},
                        {"mode": "snapshot", "report_time": "2026-01-01"},
                        {"signals": ("removed_signal",)}, {"geo_types": ("invalid",)}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                repository = RawDataRepository(temporary)
                repository.initialize(CATALOG)
                client = _Client()
                with self.assertRaises(ValueError):
                    DelphiV5Fetcher(client).fetch(repository, CATALOG["delphi_nhsn"], **options)
                self.assertFalse(client.urls)
                self.assertFalse(repository.list_snapshots("delphi_nhsn"))

    def test_raw_source_specific_columns_and_timestamps_survive(self):
        bodies = {
            "delphi_nwss": (
                b"signal,report_time,geo_type,geo_value,fill_method,reference_time,nwss_source,sample_index,pcr_target,value\n"
                b"flu_avg_conc,2026-09-11 18:33:44,sewershed,001,source,2026-09-01,CDC_Verily,2,fluav,0\n"
            ),
        }
        for key, body in bodies.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                repository = RawDataRepository(temporary)
                repository.initialize(CATALOG)
                spec = CATALOG[key]
                client = _Client()
                with patch.object(client, "open", side_effect=lambda *a, **kw: _CsvResponse(body)):
                    manifest = DelphiV5Fetcher(client).fetch(
                        repository, spec, signals=(spec.config["signals"][0],),
                        geo_types=(spec.config["geo_types"][0],), workers=1,
                    )
                payload = next(f for f in manifest.files if f.path.endswith("csv.gz"))
                self.assertEqual(gzip.decompress((repository.snapshot_path(manifest) / payload.path).read_bytes()), body)

    def test_resume_reuses_only_matching_complete_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = RawDataRepository(temporary)
            repository.initialize(CATALOG)
            spec = CATALOG["delphi_nssp"]
            options = dict(signals=("first", "second"), geo_types=("state",), workers=1)
            with patch("tapestry.data.sources.delphi.time.sleep"):
                with self.assertRaises(RuntimeError):
                    DelphiV5Fetcher(_PartlyFailingClient()).fetch(repository, spec, **options)
            staging = next((repository.root / ".staging" / spec.key).iterdir())
            with self.assertRaisesRegex(ValueError, "matching"):
                DelphiV5Fetcher(_Client()).fetch(repository, spec, **options,
                                                report_time="2026-09-01", resume_from=staging)
            client = _Client()
            manifest = DelphiV5Fetcher(client).fetch(repository, spec, **options, resume_from=staging)
            self.assertEqual(len(client.urls), 1)
            self.assertEqual(manifest.source_state["resumed_files"],
                             ["signal=first/geo_type=state/archive.csv.gz"])
            repository.verify_snapshot(manifest)


if __name__ == "__main__":
    unittest.main()
