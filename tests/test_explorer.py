from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import tarfile
import tempfile
import threading
import unittest
import urllib.request
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from tapestry.explorer import ExplorerIndex, make_server, normalize_state
from tapestry.explorer.index import _BuildError


class ExplorerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        self.datasets = [
            {
                "key": "rows",
                "title": "Synthetic rows",
                "provider": "Test",
                "temporal_resolution": "weekly",
                "revision_mode": "snapshot_only",
                "versioned": False,
                "event_date_column": "week_end",
                "vintage_column": None,
                "geographic_resolutions": ["state", "hhs", "nation"],
                "missing_value_markers": ["*"],
            },
            {
                "key": "hub",
                "title": "Synthetic hub",
                "provider": "Test Hub",
                "temporal_resolution": "daily",
                "revision_mode": "as_of_column",
                "versioned": True,
                "event_date_column": "target_end_date",
                "vintage_column": "reference_date",
                "geographic_resolutions": ["state", "nation"],
                "missing_value_markers": [],
            },
            {
                "key": "catchment",
                "title": "Synthetic catchment",
                "provider": "Test",
                "temporal_resolution": "sample",
                "revision_mode": "report_time",
                "versioned": True,
                "event_date_column": "week_end",
                "vintage_column": None,
                "geographic_resolutions": ["catchment"],
                "missing_value_markers": [],
            },
        ]
        (self.data_root / "catalog.json").write_text(
            json.dumps({"schema_version": 1, "datasets": self.datasets}), encoding="utf-8"
        )

        ndjson_rows = [
            {"state_territory": "New York", "row_id": "36001", "week_end": "2025-01-04", "season": "2024-2025", "pathogen": "flu", "visits": "10", "rate": "1.5"},
            {"state_territory": "NY", "row_id": "36001", "week_end": "2025-01-11", "season": "2025-2026", "pathogen": "flu", "visits": "20", "rate": "3.0"},
            {"state_territory": "06", "row_id": "06001", "week_end": "2025-01-04", "pathogen": "flu", "visits": "5", "rate": "*"},
            {"state_territory": "CA", "row_id": "06013", "week_end": "2025-01-04", "pathogen": "flu", "visits": "15", "rate": "2.0"},
            {"state_territory": "CA", "row_id": "06001", "week_end": "2025-01-11", "pathogen": "flu", "visits": "15", "rate": "2.0"},
            {"state_territory": "hhs2", "week_end": "2025-01-18", "pathogen": "flu", "visits": "30", "rate": "4.0"},
            {"state_territory": "US", "week_end": "2025-01-25", "pathogen": "flu", "visits": "40", "rate": "5.0"},
        ]
        snapshot = self._snapshot("rows", "s1")
        payload = snapshot / "data.ndjson.gz"
        with gzip.open(payload, "wt", encoding="utf-8") as stream:
            for row in ndjson_rows:
                stream.write(json.dumps(row) + "\n")
        self._manifest("rows", "s1", [payload])

        hub_snapshot = self._snapshot("hub", "s2")
        csv_path = self.root / "forecast.csv"
        csv_path.write_text(
            "location,target_end_date,reference_date,model_id,quantile,value\n"
            "US36,2025-01-04,2024-12-20,model-a,0.5,7\n"
            "US36,2025-01-04,2024-12-27,model-a,0.5,9\n"
            "US36,2025-01-11,2025-01-03,model-a,0.5,12\n",
            encoding="utf-8",
        )
        archive = hub_snapshot / "hub-data.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            output.add(csv_path, arcname="target-data/model-a/forecast.csv")
            output.add(csv_path, arcname="target-data/archive/old-forecast.csv")
            output.add(csv_path, arcname="auxiliary-data/duplicate.csv")
        self._manifest("hub", "s2", [archive])

        catchment_snapshot = self._snapshot("catchment", "s3")
        catchment_payload = catchment_snapshot / "data.ndjson.gz"
        with gzip.open(catchment_payload, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"location": "CA", "week_end": "2025-01-04", "rate": 3.2}) + "\n")
        self._manifest("catchment", "s3", [catchment_payload])

        self.index = ExplorerIndex(self.data_root)
        self.index.build(progress=lambda _message: None)

    def test_v5_fill_variants_and_intraday_revisions_stay_distinct(self):
        self.datasets.append({
            "key": "v5", "title": "V5", "provider": "Delphi", "temporal_resolution": "daily",
            "revision_mode": "report_time", "versioned": True,
            "event_date_column": "reference_time", "vintage_column": "report_time",
            "geographic_resolutions": ["state"],
        })
        (self.data_root / "catalog.json").write_text(json.dumps({"datasets": self.datasets}))
        snapshot = self._snapshot("v5", "s4")
        payload = snapshot / "snapshot.csv.gz"
        with gzip.open(payload, "wt") as stream:
            stream.write(
                "signal,geo_type,geo_value,reference_time,report_time,fill_method,value\n"
                "cases,state,ny,2026-09-01,2026-09-11 18:00:00,zero,5\n"
                "cases,state,ny,2026-09-01,2026-09-11 09:00:00,zero,1\n"
                "cases,state,ny,2026-09-01,2026-09-11 18:00:00,source,8\n"
            )
        self._manifest("v5", "s4", [payload])
        self.index.build(progress=lambda _: None)
        with closing(self.index.connect()) as connection:
            rows = connection.execute(
                "SELECT s.dimensions, p.value, p.samples FROM series s JOIN points p "
                "ON s.id=p.series_id WHERE s.dataset_key='v5'"
            ).fetchall()
        self.assertEqual({json.loads(row[0])["fill_method"]: (row[1], row[2]) for row in rows},
                         {"zero": (5, 1), "source": (8, 1)})

    def tearDown(self):
        self.temporary.cleanup()

    def _snapshot(self, key: str, snapshot_id: str) -> Path:
        dataset = self.data_root / "raw" / key
        snapshot = dataset / "snapshots" / snapshot_id
        snapshot.mkdir(parents=True)
        (dataset / "latest.json").write_text(json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8")
        return snapshot

    def _manifest(self, key: str, snapshot_id: str, paths: list[Path]) -> None:
        snapshot = paths[0].parent
        records = []
        for path in paths:
            records.append(
                {
                    "path": path.relative_to(snapshot).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
        (snapshot / "manifest.json").write_text(
            json.dumps({"dataset_key": key, "snapshot_id": snapshot_id, "files": records}),
            encoding="utf-8",
        )

    def test_state_normalization(self):
        self.assertEqual(normalize_state("New York"), "NY")
        self.assertEqual(normalize_state("US06"), "CA")
        self.assertEqual(normalize_state(11), "DC")
        self.assertEqual(normalize_state(1.0), "AL")
        self.assertIsNone(normalize_state("US"))
        self.assertIsNone(normalize_state("01001"))

    def test_indexes_numeric_columns_and_tar_members(self):
        overview = self.index.overview()
        self.assertEqual(len(overview["states"]), 52)
        self.assertTrue({"CA", "NY", "NJ"}.issubset({item["code"] for item in overview["states"]}))
        self.assertEqual({item["dataset_key"] for item in overview["datasets"]}, {"hub", "rows"})

        listing = self.index.list_series("NY", limit=100)
        labels = [item["label"] for item in listing["items"]]
        self.assertTrue(any("visits" in label for label in labels))
        self.assertTrue(any("rate" in label for label in labels))
        self.assertTrue(any("hub · value" in label for label in labels))
        self.assertTrue(all(item["source_path"] for item in listing["items"]))
        self.assertFalse(any("county_fips" in label for label in labels))
        self.assertFalse(any("HHS parent broadcast" in label for label in labels))
        self.assertTrue(any("national parent broadcast" in label for label in labels))
        native_visits = [
            item for item in listing["items"]
            if "visits" in item["label"] and "broadcast" not in item["label"]
        ]
        self.assertEqual(len(native_visits), 1)
        self.assertNotIn("season=", native_visits[0]["label"])
        self.assertNotIn("catchment", {item["dataset_key"] for item in overview["datasets"]})

        visits_id = next(
            item["id"] for item in listing["items"]
            if "visits" in item["label"] and "broadcast" not in item["label"]
        )
        scaled = self.index.data("New York", [visits_id], scale=True)
        self.assertEqual(scaled["series"][0]["max"], 20.0)
        self.assertEqual(scaled["series"][0]["divisor"], 20.0)
        self.assertEqual(
            scaled["series"][0]["points"],
            [["2025-01-04", 0.5, 1], ["2025-01-11", 1.0, 1]],
        )

        hub_id = next(item["id"] for item in listing["items"] if "hub · value" in item["label"])
        hub_data = self.index.data("NY", [hub_id])
        # The later reference-date revision replaces the earlier value for one target date.
        self.assertEqual(hub_data["series"][0]["points"][0], ["2025-01-04", 9.0, 1])

        ca_listing = self.index.list_series("CA", query="visits")
        ca_visits_id = next(
            item["id"] for item in ca_listing["items"] if "broadcast" not in item["label"]
        )
        ca_data = self.index.data("CA", [ca_visits_id])
        self.assertEqual(ca_data["series"][0]["points"][0], ["2025-01-04", 10.0, 2])
        self.assertIn("unweighted state-date mean", ca_data["series"][0]["label"])

        with closing(self.index.connect()) as connection:
            hub_sources = {
                row[0]
                for row in connection.execute(
                    "SELECT source_path FROM sources WHERE dataset_key = 'hub'"
                )
            }
        self.assertEqual(
            hub_sources,
            {"hub-data.tar.gz!/target-data/model-a/forecast.csv"},
        )

    def test_search_and_http_endpoints(self):
        listing = self.index.list_series("NY", query="rate", limit=10)
        self.assertEqual(listing["total"], 2)
        self.assertTrue(all("rate" in item["label"] for item in listing["items"]))

        daily = self.index.list_series("NY", cadence="daily")
        self.assertEqual({item["dataset_key"] for item in daily["items"]}, {"hub"})
        unversioned = self.index.list_series("NY", vintage="unversioned")
        self.assertEqual({item["dataset_key"] for item in unversioned["items"]}, {"rows"})
        with self.assertRaises(ValueError):
            self.index.list_series("NY", support="hhs")
        self.assertTrue(all(item["freshness"] in {"current", "lagging"} for item in listing["items"]))

        server = make_server(self.index, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with urllib.request.urlopen(f"{base}/api/catalog") as response:
                payload = json.load(response)
            self.assertEqual(len(payload["states"]), 52)
            with urllib.request.urlopen(f"{base}/") as response:
                html = response.read().decode()
            self.assertIn("Surveillance data explorer", html)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_revision_history_is_in_parquet_and_sqlite_stays_compact(self):
        import pyarrow.parquet as parquet

        self.assertTrue(self.index.revision_ledger_path.is_file())
        schema = parquet.ParquetFile(self.index.revision_ledger_path).schema_arrow
        self.assertEqual(
            schema.names,
            [
                "series_id", "state", "event_date", "release_time", "value",
                "samples", "source_snapshot",
            ],
        )
        with closing(self.index.connect()) as connection:
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            metadata = dict(connection.execute("SELECT key, value FROM meta"))
        self.assertNotIn("revisions", tables)
        self.assertGreater(int(metadata["revision_row_count"]), 0)
        self.assertEqual(metadata["revision_ledger"], "revisions.parquet")

    def test_small_batches_preserve_points_and_revision_aggregation(self):
        def contents():
            with closing(self.index.connect()) as connection:
                points = [tuple(row) for row in connection.execute(
                    "SELECT * FROM points ORDER BY series_id,state,date"
                )]
                slices = connection.execute("SELECT DISTINCT series_id,state FROM points").fetchall()
            revisions = {tuple(key): sorted(self.index._revision_rows(*key),
                         key=lambda row: (row["event_date"], row["release_time"])) for key in slices}
            return points, revisions

        expected = contents()
        self.index.batch_rows = 2
        self.index.build(progress=lambda _: None)
        self.assertEqual(contents(), expected)
        self.assertEqual(self.index.status(validate=True)["errors"], [])

    def test_failure_after_parquet_write_preserves_previous_pair(self):
        before = (self.index.index_path.read_bytes(), self.index.revision_ledger_path.read_bytes())
        original = self.index._index_table

        def fail_after_write(*args):
            original(*args)
            raise ValueError("late malformed row")

        with patch.object(self.index, "_index_table", side_effect=fail_after_write):
            with self.assertRaisesRegex(_BuildError, "late malformed row"):
                self.index.build(progress=lambda _: None)
        self.assertEqual(before, (self.index.index_path.read_bytes(), self.index.revision_ledger_path.read_bytes()))
        self.assertFalse(list(self.index.index_path.parent.glob(".*.tmp")))

    def test_early_failure_rolls_back_sqlite_and_strict_mode_refuses_reuse(self):
        original = self.index._index_table

        def fail_first_source(connection, source, *args):
            if source.dataset_key == "rows":
                self.index._record_source(connection, source, status="indexed", message="must roll back")
                raise ValueError("malformed header")
            original(connection, source, *args)

        with patch.object(self.index, "_index_table", side_effect=fail_first_source):
            self.index.build(progress=lambda _: None)
        report = self.index.status(validate=True)
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["source_errors"]), 1)
        with closing(self.index.connect()) as connection:
            self.assertFalse(connection.execute("SELECT 1 FROM sources WHERE message='must roll back'").fetchone())
        self.index.strict = True
        with self.assertRaisesRegex(_BuildError, "strict mode refused reuse"):
            self.index.ensure(progress=lambda _: None)
        before = self.index.index_path.read_bytes()
        with patch.object(self.index, "_index_table", side_effect=fail_first_source):
            with self.assertRaisesRegex(_BuildError, "malformed header"):
                self.index.build(progress=lambda _: None)
        self.assertEqual(self.index.index_path.read_bytes(), before)

    def test_pair_mismatch_is_detected_on_reuse_validation_and_query(self):
        with sqlite3.connect(self.index.index_path) as connection:
            connection.execute("UPDATE meta SET value='wrong-build' WHERE key='build_id'")
        self.assertFalse(self.index.is_current())
        self.assertTrue(self.index.status(validate=True)["errors"])
        with self.assertRaisesRegex(_BuildError, "build IDs do not match"):
            self.index._revision_rows(1, "NY")

    def test_competing_builders_are_rejected_and_lock_is_released(self):
        other = ExplorerIndex(self.data_root)
        with self.index._build_lock():
            with self.assertRaisesRegex(_BuildError, "Another builder"):
                other.build(progress=lambda _: None)
        other.build(progress=lambda _: None)
        self.assertTrue(other.is_current())

    def test_custom_ledger_directory_is_created(self):
        other = ExplorerIndex(self.data_root, self.root / "custom" / "index.sqlite3",
                              self.root / "ledger" / "history.parquet")
        other.build(progress=lambda _: None)
        self.assertTrue(other.is_current())

    def test_raw_inventory_change_during_build_preserves_previous_pair(self):
        before = self.index.index_path.read_bytes()
        with patch.object(self.index, "source_fingerprint", side_effect=["before", "after"]):
            with self.assertRaisesRegex(_BuildError, "Raw inventory changed"):
                self.index.build(progress=lambda _: None)
        self.assertEqual(self.index.index_path.read_bytes(), before)

    def test_national_context_is_stored_once_and_available_to_each_state(self):
        item = next(i for i in self.index.list_series("NY", support="national")["items"]
                    if i["value_column"] == "visits")
        with closing(self.index.connect()) as connection:
            rows = connection.execute("SELECT state,value FROM points WHERE series_id=?", (item["id"],)).fetchall()
        self.assertEqual([tuple(row) for row in rows], [("US", 40)])
        self.assertEqual(self.index.data("NY", [item["id"]])["series"][0]["points"],
                         self.index.data("CA", [item["id"]])["series"][0]["points"])

    def test_us_location_uses_only_published_national_observations(self):
        location = next(item for item in self.index.overview()["states"] if item["code"] == "US")
        listing = self.index.list_series("United States", limit=0)
        self.assertEqual(location["series_count"], listing["total"])
        self.assertTrue(listing["items"])
        self.assertTrue(all(item["dimensions"]["spatial_support"] == "national parent broadcast"
                            for item in listing["items"]))
        national = next(item for item in listing["items"] if item["value_column"] == "visits")
        result = self.index.data("US", [national["id"]])
        self.assertEqual(result["state_name"], "United States (US)")
        self.assertEqual(result["series"][0]["points"], [["2025-01-25", 40, 1]])
        self.assertEqual(self.index.data("US", [national["id"]], as_of="2025-01-24")["series"][0]["points"], [])
        self.assertEqual(self.index.versions("US", [national["id"]])["dates"], ["2025-01-25"])
        native = next(item for item in self.index.list_series("NY", support="native_state")["items"]
                      if item["value_column"] == "visits")
        # US must not return or aggregate the native NY/CA points for a state-only ID.
        self.assertEqual(self.index.data("US", [native["id"]])["series"][0]["points"], [])
        self.assertEqual(self.index.versions("US", [native["id"]])["dates"], [])
        self.assertEqual(self.index.list_series("US", support="native_state")["items"], [])

    def test_outdated_disposable_index_is_rebuilt(self):
        with sqlite3.connect(self.index.index_path) as connection:
            connection.execute("UPDATE meta SET value='9' WHERE key='schema_version'")
        with patch.object(self.index, "build", wraps=self.index.build) as rebuild:
            self.index.ensure(progress=lambda _: None)
        rebuild.assert_called_once()
        self.assertTrue(self.index.is_current())


if __name__ == "__main__":
    unittest.main()
