"""Information-set tests: release dates, missingness, full snapshots, and scaling."""
from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from influpaintx.explorer import ExplorerIndex
from influpaintx.data.catalog import CATALOG
from influpaintx.data.lineage import series_lineage


class VersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.catalog = []
        self.index = ExplorerIndex(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, key, mode, rows, vintage="report_time"):
        dataset = {"key": key, "title": key, "provider": "Test", "revision_mode": mode,
                   "versioned": mode != "snapshot_only", "event_date_column": "date",
                   "vintage_column": vintage, "geographic_resolutions": ["state"]}
        self.catalog.append(dataset)
        snapshot = self.root / "raw" / key / "snapshots" / "s1"
        snapshot.mkdir(parents=True)
        (snapshot.parent.parent / "latest.json").write_text(json.dumps({"snapshot_id": "s1"}))
        payload = snapshot / "data.ndjson.gz"
        with gzip.open(payload, "wt") as stream:
            for row in rows:
                stream.write(json.dumps({"state": "NY", **row}) + "\n")
        (snapshot / "manifest.json").write_text(json.dumps({"files": [{"path": payload.name}]}))
        return dataset

    def build(self):
        (self.root / "catalog.json").write_text(json.dumps({"datasets": self.catalog}))
        self.index.build(progress=lambda _: None)
        return {row["dataset_key"]: row["id"] for row in self.index.list_series("NY", limit=0)["items"]}

    def points(self, sid, as_of=None, **kwargs):
        return self.index.data("NY", [sid], as_of=as_of, **kwargs)["series"][0]["points"]

    def test_report_time_cutoff_carries_unchanged_dates_and_preserves_zero(self):
        self.add("revised", "report_time", [
            {"date": "2025-01-04", "report_time": "2025-01-10T09:00:00Z", "value": 10},
            {"date": "2025-01-04", "report_time": "2025-01-10T21:00:00Z", "value": 20},
            {"date": "2025-01-04", "report_time": "2025-01-17", "value": 40},
            {"date": "2025-01-05", "report_time": "2025-01-10", "value": 0},
            {"date": "2025-01-11", "report_time": "2025-01-17", "value": 80},
        ])
        sid = self.build()["revised"]
        self.assertEqual(self.points(sid, "2025-01-09"), [])
        self.assertEqual(self.points(sid, "2025-01-10"), [["2025-01-04", 20, 1], ["2025-01-05", 0, 1]])
        self.assertEqual(self.points(sid, "2025-01-10", scale=True)[0][1], 1)
        self.assertEqual(self.points(sid)[0][1], 40)
        self.assertEqual(self.points(sid)[1][1], 0)
        self.assertEqual(self.index.versions("NY", [sid])["dates"], ["2025-01-10", "2025-01-17"])

    def test_national_publisher_revisions_are_available_without_state_aggregation(self):
        spec = self.add("national", "report_time", [
            {"state": "US", "date": "2025-01-04", "report_time": "2025-01-10", "value": 50},
            {"state": "US", "date": "2025-01-04", "report_time": "2025-01-17", "value": 70},
            {"state": "NY", "date": "2025-01-04", "report_time": "2025-01-10", "value": 999},
        ])
        spec["geographic_resolutions"] = ["state", "nation"]
        self.build()
        sid = self.index.list_series("US")["items"][0]["id"]
        self.assertEqual(self.index.data("US", [sid])["series"][0]["points"], [["2025-01-04", 70, 1]])
        self.assertEqual(self.index.data("US", [sid], as_of="2025-01-10")["series"][0]["points"],
                         [["2025-01-04", 50, 1]])
        self.assertEqual(self.index.versions("US", [sid])["dates"], ["2025-01-10", "2025-01-17"])

    def test_suppression_does_not_resurrect_an_older_value(self):
        self.add("suppression", "report_time", [
            {"date": "2025-01-04", "report_time": "2025-01-10", "value": 10},
            {"date": "2025-01-04", "report_time": "2025-01-17", "value": None},
        ])
        sid = self.build()["suppression"]
        self.assertEqual(self.points(sid, "2025-01-10"), [["2025-01-04", 10, 1]])
        self.assertEqual(self.points(sid, "2025-01-17"), [])
        self.assertEqual(self.points(sid), [])

    def test_full_snapshot_does_not_carry_omitted_rows_forward(self):
        self.add("snapshots", "as_of_column", [
            {"date": "2025-01-04", "as_of": "2025-01-10", "value": 10},
            {"date": "2025-01-05", "as_of": "2025-01-10", "value": 11},
            {"date": "2025-01-04", "as_of": "2025-01-17", "value": 15},
        ], vintage="as_of")
        sid = self.build()["snapshots"]
        self.assertEqual(len(self.points(sid, "2025-01-10")), 2)
        self.assertEqual(self.points(sid, "2025-01-17"), [["2025-01-04", 15, 1]])
        self.assertEqual(self.points(sid), [["2025-01-04", 15, 1]])
        self.assertEqual(self.points(sid, "2025-01-09"), [])

    def test_unversioned_only_cuts_event_dates(self):
        self.add("current", "snapshot_only", [
            {"date": "2025-01-04", "value": 20}, {"date": "2025-01-11", "value": 100},
        ], vintage=None)
        sid = self.build()["current"]
        self.assertEqual(self.points(sid, "2025-01-04"), [["2025-01-04", 20, 1]])
        self.assertEqual(self.points(sid, "2025-01-04", scale=True), [["2025-01-04", 1, 1]])
        self.assertEqual(self.index.versions("NY", [sid])["dates"], ["2025-01-04", "2025-01-11"])
        with self.assertRaises(ValueError):
            self.index.data("NY", [sid], as_of="2025-02-30")

    def test_all_columns_can_be_loaded_without_pagination(self):
        self.add("wide", "snapshot_only", [{"date": "2025-01-04", **{f"column{i}": i for i in range(1005)}}], vintage=None)
        self.build()
        listing = self.index.list_series("NY", limit=0)
        self.assertEqual(len(listing["items"]), 1005)
        self.assertEqual(listing["total"], 1005)

    def test_lineage_is_resolved_per_hub_target(self):
        spec = CATALOG["hub_flusight_current"].to_dict()
        hosp = series_lineage(spec, "value", "time-series.csv", {"target": "wk inc flu hosp"})
        ed = series_lineage(spec, "value", "target-ed-visits-prop.csv", {})
        self.assertEqual((hosp["parent_dataset"], hosp["parent_column"]), ("NHSN", "totalconfflunewadm"))
        self.assertEqual((ed["parent_dataset"], ed["parent_column"]), ("NSSP", "percent_visits_influenza"))
        self.assertTrue(all(s.parent_dataset for s in CATALOG.values()))

    def test_git_source_is_bounded_by_saved_snapshot_without_history_traversal(self):
        self.add("legacy", "git_history", [{"date": "2025-01-04", "value": 20}], vintage=None)
        manifest = self.root / "raw/legacy/snapshots/s1/manifest.json"
        content = json.loads(manifest.read_text())
        content["source_state"] = {"commit_time": "2025-01-17T12:00:00Z"}
        manifest.write_text(json.dumps(content))
        sid = self.build()["legacy"]
        self.assertEqual(self.points(sid, "2025-01-10"), [])
        self.assertEqual(self.points(sid, "2025-01-17"), [["2025-01-04", 20, 1]])
        self.assertEqual(self.index.versions("NY", [sid])["dates"], ["2025-01-17"])


if __name__ == "__main__":
    unittest.main()
