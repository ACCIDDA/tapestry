from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tapestry.data.catalog import CATALOG
from tapestry.data.repository import RawDataRepository


class RepositoryTests(unittest.TestCase):
    def test_commit_is_immutable_and_discoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = RawDataRepository(temporary)
            repository.initialize(CATALOG)
            spec = CATALOG["cdc_nhsn_final"]
            with repository.begin_snapshot(spec) as snapshot:
                snapshot.path("data.txt", rows=2, media_type="text/plain").write_text(
                    "a\nb\n", encoding="utf-8"
                )
                manifest = snapshot.commit(selector={"test": True})

            latest = repository.latest(spec.key)
            self.assertEqual(latest.snapshot_id, manifest.snapshot_id)
            self.assertEqual(latest.files[0].rows, 2)
            snapshot_path = repository.snapshot_path(latest)
            self.assertEqual((snapshot_path / "data.txt").read_text(), "a\nb\n")
            saved = json.loads((snapshot_path / "manifest.json").read_text())
            self.assertEqual(saved["dataset_key"], spec.key)
            self.assertFalse(saved["versioned"])
            repository.verify_snapshot(latest)

            (snapshot_path / "data.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                repository.verify_snapshot(latest)

            (snapshot_path / "data.txt").write_text("a\nb\n", encoding="utf-8")
            (snapshot_path / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unexpected: extra.txt"):
                repository.verify_snapshot(latest)

    def test_failed_snapshot_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = RawDataRepository(temporary)
            repository.initialize(CATALOG)
            spec = CATALOG["cdc_nhsn_final"]
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with repository.begin_snapshot(spec) as snapshot:
                    snapshot.path("partial.txt").write_text("partial")
                    raise RuntimeError("stop")
            self.assertEqual(repository.list_snapshots(spec.key), ())

    def test_snapshot_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = RawDataRepository(temporary)
            repository.initialize(CATALOG)
            with repository.begin_snapshot(CATALOG["cdc_nhsn_final"]) as snapshot:
                with self.assertRaises(ValueError):
                    snapshot.path("../escape")


if __name__ == "__main__":
    unittest.main()
