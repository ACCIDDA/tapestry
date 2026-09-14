from __future__ import annotations

import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from influpaintx.data.sources.hubverse import HubMirror


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=directory,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


class HubMirrorTests(unittest.TestCase):
    def test_read_only_history_access_and_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _git(source, "init", "-b", "main")
            _git(source, "config", "user.name", "Test")
            _git(source, "config", "user.email", "test@example.com")
            target_data = source / "target-data"
            target_data.mkdir()
            data_file = target_data / "time-series.csv"
            data_file.write_text("date,value\n2026-01-01,1\n", encoding="utf-8")
            _git(source, "add", "target-data/time-series.csv")
            _git(source, "commit", "-m", "first")
            first = _git(source, "rev-parse", "HEAD").strip()

            data_file.write_text("date,value\n2026-01-01,2\n", encoding="utf-8")
            _git(source, "commit", "-am", "second")

            mirror = HubMirror(root / "mirror.git")
            mirror.sync(str(source))
            self.assertEqual(
                mirror.read_file(first, "target-data/time-series.csv"),
                b"date,value\n2026-01-01,1\n",
            )
            self.assertEqual(
                mirror.commit_at("main", "2030-01-01"), mirror.resolve("main")
            )
            archive = root / "target-data.tar.gz"
            selected = mirror.archive(first, ("target-data", "missing"), archive)
            self.assertEqual(selected, ("target-data",))
            self.assertGreater(archive.stat().st_size, 0)
            with tarfile.open(archive, "r:gz") as saved:
                member = saved.extractfile("target-data/time-series.csv")
                self.assertIsNotNone(member)
                assert member is not None
                self.assertEqual(member.read(), b"date,value\n2026-01-01,1\n")


if __name__ == "__main__":
    unittest.main()
